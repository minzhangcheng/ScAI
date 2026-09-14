"""Executor checkpoint-message restore support."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RestoredExecutorState:
    messages: list
    findings: list
    figures: list
    step_count: int
    max_steps: int
    analysis_log: list
    messages_rebuilt: bool = False


def fresh_messages(static_system_prompt: str, dynamic_context_message: dict, instruction: str):
    return [
        {"role": "system", "content": static_system_prompt},
        dynamic_context_message,
        {"role": "user", "content": f"Analysis instruction: {instruction}"},
    ]


def restore_executor_state(
    agent,
    state,
    *,
    checkpoint: dict | None,
    checkpoint_valid: bool,
    static_system_prompt: str,
    dynamic_context_message: dict,
    initial_instruction: str,
    max_steps: int,
    bindings,
) -> RestoredExecutorState:
    logger = getattr(bindings, "logger")
    tool_calls_are_paired = getattr(bindings, "tool_calls_are_paired")
    AnalysisStep = getattr(bindings, "AnalysisStep")
    CodeExecution = getattr(bindings, "CodeExecution")
    build_approved_plan_context = getattr(bindings, "build_approved_plan_context")
    rebuilt = False
    if checkpoint and checkpoint_valid:
        messages = checkpoint.get("messages", [])
        valid_messages = (
            all(
                isinstance(message, dict)
                and "role" in message
                and ("content" in message or "tool_calls" in message)
                for message in messages
            )
            and tool_calls_are_paired(messages)
        ) if messages else False
        if len(messages) < 2 or not valid_messages:
            logger.warning(
                "[Bioinformatics] Checkpoint messages invalid (len=%d, valid=%s), "
                "using fresh messages",
                len(messages),
                valid_messages,
            )
            messages = fresh_messages(
                static_system_prompt,
                dynamic_context_message,
                initial_instruction,
            )
            rebuilt = True
        elif (
            len(messages) >= 2
            and messages[0].get("role") == "system"
            and messages[1].get("role") == "user"
            and isinstance(messages[1].get("content"), str)
            and messages[1]["content"].startswith("Analysis instruction:")
        ):
            logger.info(
                "[Bioinformatics] Migrating checkpoint to split prompt format for "
                "better caching"
            )
            messages[0] = {"role": "system", "content": static_system_prompt}
            messages.insert(1, dynamic_context_message)
        findings = checkpoint.get("findings", [])
        figures = checkpoint.get("figures", [])
        step_count = checkpoint.get("step_number", 0)
        max_steps = max(max_steps, checkpoint.get("total_steps", max_steps))
        agent.analysis_context = checkpoint.get("analysis_context", {})
        analysis_log = [
            AnalysisStep(**item) for item in checkpoint.get("analysis_log", [])
        ]
        restored_code_history = checkpoint.get("code_history", [])
        if restored_code_history:
            agent.code_history = [
                CodeExecution(**item) for item in restored_code_history
            ]
            logger.info(
                "[Bioinformatics] Restored %s code_history entries from checkpoint",
                len(agent.code_history),
            )
        logger.info(
            "[Bioinformatics] Restored state: step=%s, findings=%s, figures=%s, "
            "new max_steps=%s",
            step_count,
            len(findings),
            len(figures),
            max_steps,
        )
    else:
        messages = fresh_messages(
            static_system_prompt,
            dynamic_context_message,
            initial_instruction,
        )
        if state.current_plan and state.current_plan.status == "approved":
            _source_context = dict(
                (getattr(state, "sandbox_state", None) or {}).get(
                    "resolved_adata_source_context_v1"
                )
                or {}
            )
            messages.append({
                "role": "user",
                "content": build_approved_plan_context(
                    state.current_plan,
                    source_context=_source_context or None,
                ),
            })
        findings = []
        figures = []
        step_count = 0
        analysis_log = []
        agent.code_history = []
        agent.analysis_context = {}
        agent.last_result = None
    return RestoredExecutorState(
        messages=messages,
        findings=findings,
        figures=figures,
        step_count=step_count,
        max_steps=max_steps,
        analysis_log=analysis_log,
        messages_rebuilt=rebuilt,
    )
