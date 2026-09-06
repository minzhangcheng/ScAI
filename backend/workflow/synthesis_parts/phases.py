"""PI synthesis, persistence, and wait-gate phases."""
from __future__ import annotations


_PATHOLOGY_IMAGE_PROJECTION_LIMIT = 20


def _pathology_image_projection(detailed_results):
    """Bounded per-image state for the PI prompt, without the tile prose.

    Returns ``(projected, counts)``. The counts matter: truncating at a limit
    without saying so hid every observation past the cap, and the PI had no way
    to know the list it was reading was partial.
    """
    if not isinstance(detailed_results, list):
        return [], {}
    projected = []
    for entry in detailed_results[:_PATHOLOGY_IMAGE_PROJECTION_LIMIT]:
        if not isinstance(entry, dict):
            continue
        validation = entry.get("tile_validation") or {}
        projected.append({
            "image_id": entry.get("image_id"),
            "image_type": entry.get("image_type"),
            "status": entry.get("status"),
            "severity_rubric_name": entry.get("severity_rubric_name"),
            "severity_score": entry.get("severity_score"),
            "severity_max": entry.get("severity_max"),
            "severity_basis": str(entry.get("severity_basis") or "")[:300],
            "tiles_analyzed": validation.get("valid_tiles"),
            "tiles_expected": validation.get("expected_tiles"),
            "tile_coverage_status": validation.get("status"),
            "severity_observations": entry.get("severity_observations") or [],
        })
    total = len([e for e in detailed_results if isinstance(e, dict)])
    counts = {
        "total_images": total,
        "projected_images": len(projected),
        "omitted_images": max(0, total - len(projected)),
    }
    if counts["omitted_images"]:
        counts["note"] = (
            f"{counts['omitted_images']} further image(s) were analysed but are not "
            "listed here; their findings are in the findings list and their full "
            "per-image detail is retained on the round result."
        )
    return projected, counts


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )

async def synthesize_agent_results(self, state):
    analysis_state = state["analysis_state"]

    provider = getattr(self, "_settings_provider", None)
    pathology_enabled = True
    if provider is not None:
        model = provider.get_model("pathology")
        pathology_enabled = bool(model and model.strip())
    file_dicts = [f.model_dump() for f in analysis_state.data_files]
    file_dicts.extend(analysis_state.pending_uploaded_files)
    modality_warning = build_disabled_pathology_warning(file_dicts, pathology_enabled)
    if modality_warning:
        await self._send_message("modality_warning", modality_warning, AgentType.PI)

    skip = await self._require_core_agent(state, "pi", "pi_synthesis")
    if skip is not None:
        return skip

    logger.info(f"[{self.session_id}] PI synthesis for round {analysis_state.current_round}")

    await self._send_message("status", "PI is synthesizing analysis results...", AgentType.PI)

    # Extract findings from each agent, including error/status info for PI awareness
    agent_results = {}
    for agent_name, result_key in [
        ("biochemistry", "biochemistry_result"),
        ("pathology", "pathology_result"),
        ("bioinformatics", "bioinformatics_result"),
        ("literature", "literature_result"),
    ]:
        raw_result = state.get(result_key)
        result = raw_result if isinstance(raw_result, dict) else {}
        findings = result.get("findings", [])
        error = result.get("error")
        status = result.get("status")

        # Build result string with error/status context so PI knows what happened
        if not result:
            agent_results[agent_name] = json.dumps(
                {"status": "not_run", "findings": []}, ensure_ascii=False
            )
        elif error:
            agent_results[agent_name] = json.dumps(
                {"error": error, "findings": findings}, ensure_ascii=False
            )
            logger.warning(f"[{self.session_id}] {agent_name} had error during this round: {error}")
        elif status in ("skipped", "no_data", "waiting_data"):
            agent_results[agent_name] = json.dumps(
                {"status": status, "findings": findings}, ensure_ascii=False
            )
        elif agent_name == "bioinformatics":
            # The plan audit and the executor's own suggestions are included so
            # the PI synthesizes against what the round actually ran, not only
            # against the findings it chose to report. An unmet approved step is
            # the strongest signal that a conclusion should stay hedged.
            review = result.get("post_execution_review") or {}
            audit = result.get("plan_audit") or {}
            # Three states, not two. `None` means the binding status was never
            # recorded — a cached result, a restored snapshot, or a path that did
            # not carry it — and it must not read as verified. Treating unknown
            # as verified is what told the PI its findings were bound to
            # successful executions on every round, whatever actually happened.
            evidence_verified = result.get("evidence_verified")
            if evidence_verified is True:
                evidence_note = "Findings are linked to successful execution records."
            elif evidence_verified is False:
                evidence_note = (
                    "One or more findings are not linked to a current successful "
                    "execution receipt; do not upgrade those claims to validated, "
                    "confirmed, or dominant wording."
                )
            else:
                evidence_note = (
                    "Execution linkage was not recorded for these findings; do not "
                    "upgrade them to validated, confirmed, or dominant wording."
                )
            agent_results[agent_name] = json.dumps(
                {
                    "status": status or "completed",
                    "findings": findings,
                    "evidence_verified": evidence_verified,
                    "evidence_execution_ids": result.get("evidence_execution_ids", []),
                    "cohort_identity": result.get("cohort_identity") or (
                        (analysis_state.sandbox_state or {}).get(
                            "cohort_identity_v1", {}
                        )
                    ),
                    "cohort_transition": result.get("cohort_transition") or (
                        (analysis_state.sandbox_state or {}).get(
                            "cohort_transition_notice", {}
                        )
                    ),
                    "evidence_note": evidence_note,
                    "agent_suggestions": result.get("agent_suggestions", []),
                    # n_unverifiable belongs here even though nothing acts on
                    # it: without it the counts do not sum to n_steps, and a
                    # coverage report that silently omits what it did not
                    # check reads as a completeness claim the audit never made.
                    "plan_coverage": {
                        k: audit.get(k)
                        for k in ("n_steps", "n_completed", "n_skipped", "n_unmet",
                                  "n_unverifiable", "n_awaiting_pi_decision",
                                  "unmet_titles", "unmet_families")
                        if audit.get(k) is not None
                    } or None,
                    "supplementary_pass": {
                        "status": review.get("review_status"),
                        "verdict": (review.get("assessment") or {}).get("verdict"),
                        "remaining_gaps": (review.get("assessment") or {}).get("remaining_gaps", []),
                    } if review.get("review_status") in ("completed", "partial", "failed") else None,
                },
                ensure_ascii=False,
            )
        else:
            # Observations recorded by a modality are for the PI to weigh; passing
            # only `findings` recorded them and then threw them away one step
            # before the agent that was supposed to read them.
            payload = {"findings": findings}
            if status:
                payload["status"] = status
            for key in (
                "abnormal_markers",          # biochemistry: status + basis per measurement
                "provenance_notes",          # biochemistry: group/sample conflicts
                "severity_observations",     # pathology: how a grade fails to hang together
            ):
                value = result.get(key)
                if value:
                    payload[key] = value
            # Pathology's detailed_results carries every tile's prose. Forwarding it
            # whole re-sends the tile descriptions the findings already summarise and
            # crowds the PI's context; the structured per-image state is what the PI
            # actually weighs. The full detail stays on the result, unmodified.
            images, image_counts = _pathology_image_projection(
                result.get("detailed_results")
            )
            if images:
                payload["images"] = images
                payload["image_counts"] = image_counts
            agent_results[agent_name] = json.dumps(payload, ensure_ascii=False, default=str)

    # Cross-modal audit (Gap 0/1/2 fix) — innovation rounds only, non-critical.
    cms = None
    if analysis_state.analysis_phase == AnalysisPhase.INNOVATION:
        try:
            current_agent_raw = {
                "biochemistry": state.get("biochemistry_result") or {},
                "pathology": state.get("pathology_result") or {},
                "bioinformatics": state.get("bioinformatics_result") or {},
            }
            literature_context = self._build_literature_audit_context(state)
            biochemistry_raw = self._build_biochemistry_raw_context(state)
            cms = await self.pi_agent.build_cross_modal_synthesis(
                state=analysis_state,
                current_direction=analysis_state.current_round_direction,
                current_agent_raw=current_agent_raw,
                literature_context=literature_context,
                biochemistry_raw=biochemistry_raw,
            )
        except Exception as e:
            logger.warning(f"[{self.session_id}] cross-modal synthesis failed (non-critical): {e}")
            cms = None

    round_result = await self.pi_agent.synthesize_round(
        analysis_state,
        pathology_result=agent_results["pathology"],
        bioinformatics_result=agent_results["bioinformatics"],
        biochemistry_result=agent_results["biochemistry"],
        literature_result=agent_results["literature"],
        research_direction=analysis_state.current_round_direction,
        cross_modal_context=(cms.matrix_markdown if cms else ""),
    )
    if cms is not None:
        round_result.cross_modal_synthesis = cms
    round_result.pathology_coverage = extract_pathology_coverage(
        state.get("pathology_result")
    )

    logger.info(f"[{self.session_id}] Round {round_result.round_number} synthesis complete: new_findings={round_result.new_findings}, {len(round_result.next_steps)} next steps")

    # Tag round with current phase and direction
    round_result.phase = analysis_state.analysis_phase
    round_result.direction = analysis_state.current_round_direction or ""
    bioinformatics_result = state.get("bioinformatics_result") or {}

    return {
        "state": state, "analysis_state": analysis_state,
        "round_result": round_result, "bioinformatics_result": bioinformatics_result,
        "modality_warning": modality_warning,
    }

async def build_exploration_result(self, ctx):
    state = ctx["state"]
    analysis_state = ctx["analysis_state"]
    round_result = ctx["round_result"]
    bioinformatics_result = ctx["bioinformatics_result"]
    # Exploration phase: generate structured Report A and transition to innovation.
    # Skip both the Report A generation and the phase transition when this
    # round's bioinformatics analysis failed — otherwise we'd produce an
    # exploration report from empty findings and lock the user into
    # INNOVATION phase next round with no path to retry stage 1.
    if (
        analysis_state.analysis_phase == AnalysisPhase.EXPLORATION
        and self._exploration_round_succeeded(state)
    ):
        await self._send_message(
            "status",
            "Generating exploration report (Report A)...",
            AgentType.PI
        )
        report_generated = False
        try:
            exploration_report = await self._generate_evidence_gated_exploration_report(
                state, analysis_state, round_result
            )
            analysis_state.exploration_report = exploration_report
            round_result.exploration_report = exploration_report

            # Generate exploration report HTML
            try:
                from ...services.report import generate_exploration_report_html
                html_path = generate_exploration_report_html(
                    analysis_state, exploration_report, round_result
                )
                analysis_state.exploration_report_html_path = html_path
                logger.info(f"[{self.session_id}] Exploration report HTML saved to: {html_path}")
            except Exception as html_err:
                logger.error(f"[{self.session_id}] Exploration report HTML generation failed: {html_err}")

            html_url = f"/api/session/{self.session_id}/exploration-report/html" if analysis_state.exploration_report_html_path else None

            # Generate detailed report segment for exploration completion
            await self._build_and_save_detailed_segment(
                state=analysis_state,
                phase="exploration_complete",
                title="Exploration Phase Complete - Analysis Summary",
                graph_state=state,
                round_result=round_result,
            )
            detailed_report_url = f"/api/session/{self.session_id}/detailed-report/html" if analysis_state.detailed_report_html_path else None

            await self._send_message("exploration_report", {
                "report_a": exploration_report.model_dump(mode="json"),
                "round": round_result.round_number,
                "summary": round_result.pi_summary,
                "html_url": html_url,
                "detailed_report_url": detailed_report_url,
            }, AgentType.PI)
            report_generated = True
        except Exception as e:
            logger.exception(f"[{self.session_id}] Exploration report generation failed")
            await self._send_message(
                "error",
                f"Exploration report generation failed: {e}. Staying in the exploration phase — please fix the issue and retry generating the exploration report and direction candidates.",
                AgentType.PI
            )

        # Only advance to innovation when an exploration report was actually produced.
        if report_generated:
            analysis_state.analysis_phase = AnalysisPhase.INNOVATION
            logger.info(f"[{self.session_id}] Transitioning to innovation phase")
        else:
            logger.warning(
                f"[{self.session_id}] Exploration report not generated; staying in EXPLORATION phase for retry"
            )

    elif analysis_state.analysis_phase == AnalysisPhase.EXPLORATION:
        # Exploration did NOT succeed (bioinformatics stage1 failed/incomplete).
        # Most often this means the uploaded data has a problem. Surface an
        # explicit prompt so the user can fix/re-upload data and retry stage 1,
        # rather than silently advancing rounds toward max_rounds.
        bio = state.get("bioinformatics_result") or {}
        reason_code = classify_exploration_failure(bio)
        await self._send_message(
            "exploration_failed",
            {
                "round": round_result.round_number,
                "reason": reason_code,
            },
            AgentType.PI,
        )
        logger.warning(
            f"[{self.session_id}] Exploration failed (reason={reason_code}; "
            f"raw_status={bio.get('status')!r} raw_error={bio.get('error')!r}); "
            f"staying in EXPLORATION, prompting user to check data"
        )

    bioinformatics_result = state.get("bioinformatics_result") or {}

async def persist_round_result(self, ctx):
    state = ctx["state"]
    analysis_state = ctx["analysis_state"]
    round_result = ctx["round_result"]
    bioinformatics_result = ctx["bioinformatics_result"]
    round_result.data_driven_mechanism_programs = bioinformatics_result.get("mechanism_programs", [])
    round_result.bioinformatics_finding_records = list(
        bioinformatics_result.get("finding_records") or []
    )
    wait_seconds = analysis_state.round_wait_seconds
    round_result.wait_policy = (
        "no_wait" if wait_seconds == 0
        else "manual" if wait_seconds < 0
        else "countdown"
    )
    round_result.bioinformatics_evidence_verified = bioinformatics_result.get("evidence_verified")
    round_result.evidence_gate_degraded = bool(
        bioinformatics_result.get("evidence_gate_degraded")
        or any(
            finding.get("forced_release")
            for finding in bioinformatics_result.get("evidence_gate_findings") or []
            if isinstance(finding, dict)
        )
    )
    round_result.evidence_execution_ids = list(
        bioinformatics_result.get("evidence_execution_ids") or []
    )
    round_result.evidence_gate_findings = list(
        bioinformatics_result.get("evidence_gate_findings") or []
    )
    for gate_finding in round_result.evidence_gate_findings:
        if not gate_finding.get("forced_release"):
            continue
        gate = gate_finding.get("gate") or "mandatory-evidence-gate"
        rule_id = gate_finding.get("rule_id") or "unregistered-rule"
        count = gate_finding.get("rejection_count", 0)
        limit = gate_finding.get("rejection_limit", 0)
        round_result.evidence_notes.append(
            f"Mandatory gate {gate} ({rule_id}) was force-released after "
            f"{count}/{limit} rejection attempts; affected findings are degraded."
        )
    if round_result.bioinformatics_evidence_verified is False:
        round_result.evidence_notes.append(
            "Bioinformatics findings were not fully bound to successful execution evidence; "
            "treat them as partial and unvalidated."
        )
    derive_round_completion_status(state, round_result)
    lineage_id = current_lineage_id(analysis_state)
    restored_from = restore_provenance_id(analysis_state)
    _upsert_round_result(
        analysis_state.round_results,
        round_result,
        lineage_id=lineage_id,
        branch_id=None if lineage_id == "main" else lineage_id,
        branch_label=analysis_state.restore_branch_label,
        restored_from_snapshot_id=restored_from,
    )
    if restored_from is not None:
        consume_restore_provenance(analysis_state)
    from ..direction_coverage import active_batch_ready_for_integration
    if (
        analysis_state.analysis_phase == AnalysisPhase.INNOVATION
        and active_batch_ready_for_integration(analysis_state)
    ):
        analysis_state.pending_integration = True
        logger.info(
            f"[{self.session_id}] Direction batch fully attempted "
            "-> pending_integration=True"
        )
    # Round is now canonical in round_results; clear the per-round cache
    # so a later snapshot-restored round can't replay stale agent outputs.
    analysis_state.current_round_cache = None



    # P1: innovation rounds also get an incremental detailed-report segment.
    # Exploration round is already covered at the exploration_complete call above.
    if _should_build_innovation_segment(analysis_state, round_result):
        try:
            await self._build_and_save_detailed_segment(
                analysis_state,
                f"innovation_round_{analysis_state.current_round}",
                f"Innovation Round {analysis_state.current_round} — Analysis Results",
                graph_state=state,
                round_result=round_result,
            )
        except Exception as e:
            logger.exception(
                f"[{self.session_id}] Innovation detailed segment generation failed"
            )

    # Save state after each round
    if self.state_callback:
        try:
            await self.state_callback(analysis_state)
        except Exception as e:
            logger.exception("Failed to save state after round")

    # Bound long-running checkpoint growth at a semantic round boundary.
    # Pruning includes associated writes and dangling parent links; VACUUM
    # remains close-time maintenance because it takes a stronger SQLite
    # lock and is unnecessary on every round.
    try:
        await self._compact_checkpoints(vacuum=False)
    except Exception:
        logger.exception(
            "[%s] Failed to compact checkpoints at round boundary",
            self.session_id,
        )


async def finalize_synthesis(self, ctx):
    state = ctx["state"]
    analysis_state = ctx["analysis_state"]
    round_result = ctx["round_result"]
    modality_warning = ctx["modality_warning"]
    # Extract data requests from PI
    try:
        data_requests = await self.pi_agent.identify_data_needs(
            analysis_state,
            round_result
        )
    except Exception as e:
        logger.warning(f"Failed to identify data needs: {e}")
        data_requests = []

    # Send round summary with data requests
    effective_max_rounds = analysis_state.max_rounds if analysis_state.max_rounds > 0 else self.config.max_rounds
    cross_modal_candidates = []
    if round_result.cross_modal_synthesis:
        for c in round_result.cross_modal_synthesis.candidates:
            cross_modal_candidates.append(c.to_summary_dict())

    wait_seconds = analysis_state.round_wait_seconds
    no_wait = wait_seconds == 0
    wait_policy = (
        "no_wait" if no_wait else "manual" if wait_seconds < 0 else "countdown"
    )
    round_summary = build_round_summary_payload(
        round_result, analysis_state, effective_max_rounds,
        data_requests=data_requests,
        cross_modal_candidates=cross_modal_candidates,
        status="completed" if no_wait else "waiting_input",
        wait_policy=wait_policy,
    )

    await self._send_message(
        "round_summary",
        round_summary,
        AgentType.PI
    )

    # Also send traditional result message for compatibility
    await self._send_message(
        "result",
        {
            "round": round_result.round_number,
            "summary": round_result.pi_summary,
            "next_steps": round_result.next_steps,
            "new_findings": round_result.new_findings
        },
        AgentType.PI
    )

    # Auto-snapshot before round wait HITL

    # Clear continue event BEFORE announcing waiting status to prevent race
    # condition where frontend signals between status message and event.wait()
    self._continue_event.clear()

    # No-wait mode must never advertise a transient user-input boundary.
    self._set_status(
        analysis_state,
        TaskStatus.RUNNING if no_wait else TaskStatus.WAITING_INPUT,
    )

    max_rounds = analysis_state.max_rounds if analysis_state.max_rounds > 0 else self.config.max_rounds
    is_last_round = analysis_state.current_round >= max_rounds
    if no_wait:
        await self._send_message(
            "status",
            f"Round {round_result.round_number} complete. No-wait mode is enabled; continuing automatically.",
            AgentType.PI,
        )
    elif is_last_round:
        await self._send_message(
            "status",
            f"Round {round_result.round_number} complete (final round). You can provide additional data to extend analysis, or skip to generate the final report.",
            AgentType.PI
        )
    else:
        await self._send_message(
            "status",
            f"Round {round_result.round_number} complete. Waiting for your input - you can provide additional data or instructions.",
            AgentType.PI
        )

    # Send state sync for round completion
    await self._send_state_sync(analysis_state, "round_complete")

    return {
        **state,
        "analysis_state": analysis_state,
        "round_result": round_result,
        "data_requests": data_requests
    }


async def run_pi_synthesis(self, state):
    ctx = await synthesize_agent_results(self, state)
    if "round_result" not in ctx:
        return ctx
    await build_exploration_result(self, ctx)
    await persist_round_result(self, ctx)
    return await finalize_synthesis(self, ctx)

async def _wait_for_user_input(self, state: WorkflowState) -> WorkflowState:
    """Block between rounds, waiting for user input or continue signal from frontend.

    Behavior depends on analysis_state.round_wait_seconds (the user-facing
    "skip time" setting, unified across all HITL gates):
      -1  = manual mode: wait indefinitely for user signal
       0  = no-wait mode: proceed immediately without waiting
      >0  = countdown mode: frontend shows countdown timer, backend waits
            `wait_seconds + 60s` then auto-skips (60s buffer for client lag)

    Auto-skip semantics for the post-exploration direction-selection gate:
    equivalent to "let PI decide" (sets user_chosen_direction = [PI_DECIDE]).
    """
    analysis_state = state["analysis_state"]

    # Integration gate: auto-skip / timeout now triggers PI_DECIDE_SIGNAL
    # (instead of final_report_requested), so PI gets a chance to select
    # directions from integration candidates. User explicit finalize (via
    # /finalize-from-integration) still sets final_report_requested and
    # is handled first.
    is_integration_gate = (
        state.get("semantic_gate") == GATE_INTEGRATION_SELECTION
        or at_integration_gate(analysis_state)
    )
    if is_integration_gate:
        wait_seconds = analysis_state.round_wait_seconds
        if wait_seconds == 0:
            analysis_state.user_chosen_direction = PI_DECIDE_SIGNAL
            analysis_state.pending_human_input = PI_DECIDE_SIGNAL
        else:
            signaled = await wait_with_skip_time(
                self._continue_event, wait_seconds, self.session_id, "integration_gate",
            )
            if not signaled:
                analysis_state.user_chosen_direction = PI_DECIDE_SIGNAL
                analysis_state.pending_human_input = PI_DECIDE_SIGNAL
            else:
                _consume_signal_then_clear(self._continue_event)

        # User explicit finalize (via /finalize-from-integration API) takes priority.
        if analysis_state.consume_final_report_request():
            self._set_status(analysis_state, TaskStatus.RUNNING)
            await self._send_state_sync(analysis_state, "integration_finalize")
            return {
                **state,
                "analysis_state": analysis_state,
                "should_continue": False,
                "semantic_gate": None,
            }

        # Robustness: a generic continue (e.g. /human-input with no direction,
        # or any signal that did not go through /choose-direction) leaves the
        # gate unresolved — phase still INTEGRATION with candidates populated.
        # Routing would then send the graph to initialize_round and start a
        # fresh round with no integration direction selected. Treat such a
        # continue as "let PI decide" (same default as auto-skip / no-wait) so
        # the gate always resolves. An explicit selection clears the candidates
        # (at_integration_gate becomes False) and is left untouched here.
        if analysis_state.user_chosen_direction != PI_DECIDE_SIGNAL and at_integration_gate(analysis_state):
            analysis_state.user_chosen_direction = PI_DECIDE_SIGNAL
            analysis_state.pending_human_input = PI_DECIDE_SIGNAL

        # Auto-skip set PI_DECIDE_SIGNAL, or user chose directions via API.
        # If PI_DECIDE_SIGNAL, switch to INNOVATION so pi_planning handles it.
        if analysis_state.user_chosen_direction == PI_DECIDE_SIGNAL:
            analysis_state.analysis_phase = AnalysisPhase.INNOVATION
        self._set_status(analysis_state, TaskStatus.RUNNING)
        await self._send_state_sync(analysis_state, "integration_continue")
        return {
            **state,
            "analysis_state": analysis_state,
            "should_continue": True,
            "semantic_gate": None,
        }

    # Note: _continue_event is cleared in _pi_synthesis before the waiting
    # status is announced, to prevent a race condition with signal_continue().

    wait_seconds = analysis_state.round_wait_seconds  # -1=manual, 0=no wait, >0=countdown
    is_direction_gate = (
        analysis_state.current_round == 1
        and analysis_state.analysis_phase == AnalysisPhase.INNOVATION
    )

    # Filter out empty/whitespace-only directions from queue
    if analysis_state.directions_queue:
        analysis_state.directions_queue = [
            d for d in analysis_state.directions_queue if d and d.get("direction", "").strip()
        ]

    if wait_seconds == 0:
        # No-wait mode: proceed immediately (with let-PI-decide for direction gate)
        logger.info(f"[{self.session_id}] No-wait mode, proceeding immediately (round {analysis_state.current_round})")
        if is_direction_gate and not analysis_state.user_chosen_direction:
            analysis_state.user_chosen_direction = PI_DECIDE_SIGNAL
            analysis_state.pending_human_input = PI_DECIDE_SIGNAL
        self._set_status(analysis_state, TaskStatus.RUNNING)
        await self._send_state_sync(analysis_state, "auto_continue")
    else:
        if analysis_state.status != TaskStatus.WAITING_INPUT:
            self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
            await self._send_state_sync(analysis_state, "waiting_input")

        # P0-6: Do NOT clear _continue_event here. It was already cleared in
        # _pi_synthesis before WAITING_INPUT was announced; clearing again
        # would discard signals delivered during LangGraph's checkpoint I/O
        # window between the two nodes, forcing the user to click twice.
        while True:
            signaled = await wait_with_skip_time(
                self._continue_event,
                wait_seconds,
                self.session_id,
                GATE_DIRECTION_SELECTION if is_direction_gate else GATE_ROUND_WAIT,
            )
            if not signaled:
                # Auto-skip on timeout
                if is_direction_gate and not analysis_state.user_chosen_direction:
                    analysis_state.user_chosen_direction = PI_DECIDE_SIGNAL
                    analysis_state.pending_human_input = PI_DECIDE_SIGNAL
                self._set_status(analysis_state, TaskStatus.RUNNING)
                await self._send_state_sync(analysis_state, "auto_continue")
                break
            _consume_signal_then_clear(self._continue_event)

            if self._supplementary_queue and self._supplementary_queue[0][1] == "direction":
                req, _target = self._supplementary_queue.popleft()
                _phase = "integration" if analysis_state.analysis_phase == AnalysisPhase.INTEGRATION else "exploration"
                _rec = get_or_open_direction_record(
                    analysis_state, round_number=analysis_state.current_round,
                    phase=_phase,
                    integration_cycle=analysis_state.integration_cycle if _phase == "integration" else None,
                )
                _rec.supplementary_requests.append(req)
                self._set_status(analysis_state, TaskStatus.RUNNING)
                await self._send_state_sync(analysis_state, "supplementary_running")

                await self._run_supplementary_analysis(analysis_state, req, "direction")

                try:
                    last_round = analysis_state.round_results[-1] if analysis_state.round_results else None
                    if last_round and analysis_state.exploration_report:
                        exploration_report = await self._generate_evidence_gated_exploration_report(
                            state, analysis_state, last_round
                        )
                        analysis_state.exploration_report = exploration_report

                        from ...services.report import generate_exploration_report_html
                        html_path = generate_exploration_report_html(
                            analysis_state, exploration_report, last_round
                        )
                        analysis_state.exploration_report_html_path = html_path
                        html_url = f"/api/session/{self.session_id}/exploration-report/html"
                        detailed_report_url = f"/api/session/{self.session_id}/detailed-report/html" if analysis_state.detailed_report_html_path else None

                        await self._send_message("exploration_report", {
                            "report_a": exploration_report.model_dump(mode="json"),
                            "round": last_round.round_number,
                            "summary": last_round.pi_summary,
                            "html_url": html_url,
                            "detailed_report_url": detailed_report_url,
                        }, AgentType.PI)
                except Exception as e:
                    logger.error(f"[{self.session_id}] Exploration report regeneration failed: {e}")
                    await self._send_message("error", f"Report regeneration failed: {e}", AgentType.PI)

                self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
                await self._send_state_sync(analysis_state, "waiting_input")
                continue
            elif self._supplementary_queue and self._supplementary_queue[0][1] == "round":
                req, _target = self._supplementary_queue.popleft()
                self._set_status(analysis_state, TaskStatus.RUNNING)
                await self._send_state_sync(analysis_state, "supplementary_running")

                await self._run_supplementary_analysis(analysis_state, req, "round")

                self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
                await self._send_state_sync(analysis_state, "waiting_input")
                continue
            else:
                self._set_status(analysis_state, TaskStatus.RUNNING)
                await self._send_state_sync(analysis_state, "continue")
                logger.info(f"[{self.session_id}] Continue signal received")
                break

    # If user provided real input (text or files) on the last round, extend max_rounds to allow one more
    has_real_text = (
        analysis_state.pending_human_input
        and analysis_state.pending_human_input.strip() not in ('', '[CONTINUE]', '[PI_DECIDE]')
    )
    has_uploaded_files = bool(analysis_state.pending_uploaded_files)
    max_rounds = analysis_state.max_rounds if analysis_state.max_rounds > 0 else self.config.max_rounds
    if (has_real_text or has_uploaded_files) and analysis_state.current_round >= max_rounds:
        # Don't extend if multi-direction mode has completed all directions
        # AND the user didn't provide real new input/files (only synthetic signals).
        # If the user explicitly typed instructions or uploaded files, honor their intent.
        innovation_rounds_done = count_innovation_rounds_done(analysis_state)
        multi_dir_completed = (
            len(analysis_state.user_chosen_directions) > 1
            and innovation_rounds_done >= len(analysis_state.user_chosen_directions)
        )
        if multi_dir_completed and not has_real_text:
            logger.info(f"[{self.session_id}] Multi-direction completed, not extending max_rounds")
        else:
            analysis_state.max_rounds = analysis_state.current_round + 1
            logger.info(f"[{self.session_id}] User input on last round (text={has_real_text}, files={has_uploaded_files}), extending max_rounds to {analysis_state.max_rounds}")

    return {**state, "analysis_state": analysis_state}
