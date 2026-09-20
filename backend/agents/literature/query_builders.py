"""Tiered PubMed query construction and term selection.

Moved verbatim out of backend/agents/literature_agent.py; composed into LiteratureAgent.
"""
from typing import Dict, Any, List, Optional, Tuple
import re
import asyncio
import logging


from ...utils.formatting import as_list
from ...utils.llm_client import UpstreamExhaustedError

logger = logging.getLogger(__name__)
from ...models.schemas import (
    AnalysisState,
    DirectionSearchQuery,
    DirectionSearchTask,
    DirectionSearchTerm,
)


SUPPORTED_DIRECTION_SEARCH_FIELDS = frozenset(
    {"all", "tiab", "title", "author", "mesh", "majr", "pt", "sh", "subh", "nm"}
)
_DIRECTION_TERM_CONTROL_RE = re.compile(
    r"(?:\bAND\b|\bOR\b|\bNOT\b|[()\[\]{}\"])", re.IGNORECASE
)


class DirectionSearchCompilationError(ValueError):
    """Technical validation failure for a PI-authored direction-search task."""


def _direction_term(value: Any) -> DirectionSearchTerm:
    try:
        term = value if isinstance(value, DirectionSearchTerm) else DirectionSearchTerm.model_validate(value)
    except Exception as exc:
        raise DirectionSearchCompilationError(f"invalid direction search term: {exc}") from exc
    text = term.text.strip()
    if not text:
        raise DirectionSearchCompilationError("direction search term is empty")
    if len(text) > 240:
        raise DirectionSearchCompilationError("direction search term exceeds 240 characters")
    if _DIRECTION_TERM_CONTROL_RE.search(text):
        raise DirectionSearchCompilationError(
            f"direction search term contains Boolean/control syntax: {text!r}"
        )
    if term.field not in SUPPORTED_DIRECTION_SEARCH_FIELDS:
        raise DirectionSearchCompilationError(f"unsupported PubMed field tag: {term.field!r}")
    return term


def _compile_direction_term(term: DirectionSearchTerm) -> str:
    return f'"{term.text.strip()}"[{term.field}]'


def compile_direction_search_task(
    task: DirectionSearchTask | Dict[str, Any],
    *,
    candidate_ids: Optional[set[str] | List[str]] = None,
) -> str:
    """Compile one PI-approved task without adding scientific concepts."""
    try:
        task = task if isinstance(task, DirectionSearchTask) else DirectionSearchTask.model_validate(task)
    except Exception as exc:
        raise DirectionSearchCompilationError(f"invalid direction search task: {exc}") from exc
    if candidate_ids is not None and task.direction_id not in set(candidate_ids):
        raise DirectionSearchCompilationError(
            f"task {task.task_id!r} references unknown candidate {task.direction_id!r}"
        )
    if not task.task_id.strip() or not task.direction_id.strip():
        raise DirectionSearchCompilationError("task and direction identifiers must be non-empty")
    if not task.concept_groups:
        raise DirectionSearchCompilationError(f"task {task.task_id!r} has no concept groups")

    group_ids: set[str] = set()
    clauses: list[str] = []
    for group in task.concept_groups:
        if group.group_id in group_ids:
            raise DirectionSearchCompilationError(f"duplicate concept group id: {group.group_id!r}")
        group_ids.add(group.group_id)
        terms = [_compile_direction_term(_direction_term(term)) for term in group.terms]
        if not terms:
            raise DirectionSearchCompilationError(f"concept group {group.group_id!r} is empty")
        clauses.append(terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")")

    if task.scope is not None:
        missing = set(task.scope.required_group_ids) - group_ids
        if missing:
            raise DirectionSearchCompilationError(
                f"task {task.task_id!r} scope references unknown groups: {sorted(missing)}"
            )

    exclusion_ids: set[str] = set()
    for exclusion in task.exclusions:
        if exclusion.exclusion_id in exclusion_ids:
            raise DirectionSearchCompilationError(
                f"duplicate exclusion group id: {exclusion.exclusion_id!r}"
            )
        exclusion_ids.add(exclusion.exclusion_id)
        terms = [_compile_direction_term(_direction_term(term)) for term in exclusion.terms]
        if not terms:
            raise DirectionSearchCompilationError(f"exclusion group {exclusion.exclusion_id!r} is empty")
        clauses.append("NOT " + (terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"))

    return " AND ".join(clauses)


def compile_direction_search_plan(
    tasks: List[DirectionSearchTask | Dict[str, Any]],
    *,
    candidate_ids: Optional[set[str] | List[str]] = None,
) -> Dict[str, str]:
    """Validate/compile a bounded batch and return task_id -> exact query."""
    if len(tasks) > 256:
        raise DirectionSearchCompilationError("a direction search batch may contain at most 256 tasks")
    seen: set[str] = set()
    compiled: Dict[str, str] = {}
    for raw_task in tasks:
        try:
            task = raw_task if isinstance(raw_task, DirectionSearchTask) else DirectionSearchTask.model_validate(raw_task)
        except Exception as exc:
            raise DirectionSearchCompilationError(f"invalid direction search task: {exc}") from exc
        if task.task_id in seen:
            raise DirectionSearchCompilationError(f"duplicate task id: {task.task_id!r}")
        seen.add(task.task_id)
        compiled[task.task_id] = compile_direction_search_task(task, candidate_ids=candidate_ids)
    return compiled


compile_pi_direction_search_task = compile_direction_search_task
compile_pi_direction_search_plan = compile_direction_search_plan

from .shared import (  # noqa: F401
    # Lives in shared.py because build_focused_query needs it too and shared is
    # imported by this module, not the other way round. Re-exported here so the
    # name keeps resolving where it was first introduced.
    normalize_search_terms,
    _split_free_text_label,
    _FALLBACK_EXCEPTIONS,
    _MAX_MAIN_RETRIES,
    _RETRY_DELAY_BASE,
    plausibility_terms_for_tissue,
    _gene_case_variants,
    _sanitize_synonyms,
    status_from_judge,
    build_focused_query,
)


class QueryBuildersMixin:
    """Tiered PubMed query construction and term selection."""

    def _build_active_challenge_query(self, terms: Dict[str, List[str]]) -> Optional[str]:
        core = (
            self._format_term_list(terms["genes"])
            or self._format_term_list(terms["pathways"])
            or self._format_term_list(terms["cells"])
        )
        if not core:
            return None
        challenge_terms = (
            "contradict*[tiab] OR negative[tiab] OR absence[tiab] OR "
            "downregulat*[tiab] OR alternative[tiab] OR contrary[tiab] OR "
            "not associated[tiab]"
        )
        return " AND ".join(
            [f"({core})", f"({challenge_terms})", *self._organism_tissue_anchor(terms)]
        )

    def _organism_tissue_anchor(self, terms: Dict[str, List[str]]) -> List[str]:
        """The organism and tissue blocks, for queries that should stay on-subject.

        The two challenge builders carried no anchor at all, and it cost more
        than it bought: measured over eight real candidates, the unanchored form
        returned 159 articles of which 11 were in the study's organism and
        tissue, while the anchored form returned 17 of which 17 were. A tenth of
        the triage budget and more on-topic material in absolute terms.
        """
        anchor: List[str] = []
        organism = terms["organism"][0]
        tissue = terms["tissue"][0]
        if organism:
            anchor.append(f'"{organism}"[all]')
        if tissue:
            anchor.append(f'"{tissue}"[tiab]')
        return anchor

    def _build_analogous_mechanism(self, terms: Dict[str, List[str]]) -> Optional[str]:
        parts = []
        pathway_part = self._format_term_list(terms["pathways"])
        if pathway_part:
            parts.append(pathway_part)
        if not pathway_part:
            gene_part = self._format_term_list(terms["genes"])
            if gene_part:
                parts.append(gene_part)
        if not parts:
            return None

        org = terms["organism"][0]
        if org:
            parts.append(f'"{org}"[all]')
        tis = terms["tissue"][0]
        if tis:
            parts.append(f'"{tis}"[tiab]')

        return " AND ".join(parts)

    def _build_cell_type_focused(self, terms: Dict[str, List[str]]) -> Optional[str]:
        cell_part = self._format_term_list(terms["cells"])
        if not cell_part:
            gene_part = self._format_term_list(terms["genes"])
            if not gene_part:
                return None
            cell_part = gene_part
        parts = [cell_part]
        tis = terms["tissue"][0]
        if tis:
            parts.append(f'"{tis}"[tiab]')
        return " AND ".join(parts)

    def _build_contradictory_expectedness(self, terms: Dict[str, List[str]]) -> Optional[str]:
        gene_part = self._format_term_list(terms["genes"])
        if not gene_part:
            pathway_part = self._format_term_list(terms["pathways"])
            if pathway_part:
                gene_part = pathway_part
        if not gene_part:
            return None

        return " AND ".join(
            [
                f"({gene_part})",
                "(adverse[tiab] OR negative[tiab] OR inhibition[tiab] OR antagonist[tiab] OR detrimental[tiab])",
                *self._organism_tissue_anchor(terms),
            ]
        )

    def _build_cross_species_or_tissue(self, terms: Dict[str, List[str]]) -> Optional[str]:
        # Core mechanism term: prefer genes, then pathways, then cell types.
        core = (self._format_term_list(terms["genes"])
                or self._format_term_list(terms["pathways"])
                or self._format_term_list(terms["cells"]))
        if not core:
            return None
        parts = [core]
        org = terms["organism"][0]
        if org:
            # Cross-species: same mechanism studied in other organisms.
            # Target tissue is intentionally omitted so cross-tissue evidence is not excluded.
            parts.append(f'NOT "{org}"[all]')
        return " AND ".join(parts) if len(parts) > 1 else parts[0]

    def _build_degraded_fallback(self, terms: Dict[str, List[str]]) -> str:
        parts = []
        org = terms["organism"][0]
        tis = terms["tissue"][0]
        if org:
            parts.append(f'"{org}"[all]')
        if tis:
            parts.append(f'"{tis}"[tiab]')
        combined = []
        for key in ("pathways", "cells", "methods", "genes"):
            for t in terms.get(key, []):
                if t:
                    combined.append(f'"{t}"[tiab]')
        if combined:
            parts.append("(" + " OR ".join(combined) + ")")
        if not parts:
            return ""
        return " AND ".join(parts)

    def _build_direct_prior_work(self, terms: Dict[str, List[str]], deep: bool) -> Optional[str]:
        # Prefer genes, then fall back to cells, then pathways.
        core = (
            self._format_term_list(terms["genes"])
            or self._format_term_list(terms["cells"])
            or self._format_term_list(terms["pathways"])
        )
        if not core:
            return None
        parts = [core]

        org = terms["organism"][0]
        tis = terms["tissue"][0]
        if org:
            parts.append(f'"{org}"[all]')
        if tis and not deep:
            parts.append(f'"{tis}"[tiab]')

        return " AND ".join(parts)

    def _build_gene_model_relaxed(self, terms: Dict[str, List[str]]) -> Optional[str]:
        gene_part = self._format_term_list(terms["genes"])
        if not gene_part:
            return None
        return gene_part

    def _build_methodological_precedent(self, terms: Dict[str, List[str]]) -> Optional[str]:
        method_part = self._format_term_list(terms["methods"])
        if not method_part:
            return None

        parts = [method_part]
        org = terms["organism"][0]
        if org:
            parts.append(f'"{org}"[all]')
        tis = terms["tissue"][0]
        if tis:
            parts.append(f'"{tis}"[tiab]')

        return " AND ".join(parts)

    def _build_pathway_focused(self, terms: Dict[str, List[str]]) -> Optional[str]:
        pathway_part = self._format_term_list(terms["pathways"])
        gene_part = self._format_term_list(terms["genes"])
        if not pathway_part and not gene_part:
            return None
        parts = []
        if pathway_part:
            parts.append(pathway_part)
        if gene_part:
            parts.append(gene_part)
        return " AND ".join(parts) if len(parts) > 1 else parts[0]

    def _build_prior_art_query(self, terms: Dict[str, List[str]]) -> Optional[str]:
        core_parts = []
        gene_part = self._format_term_list(terms["genes"])
        pathway_part = self._format_term_list(terms["pathways"])
        cell_part = self._format_term_list(terms["cells"])
        if gene_part:
            core_parts.append(gene_part)
        if pathway_part:
            core_parts.append(pathway_part)
        if cell_part:
            core_parts.append(cell_part)
        if not core_parts:
            return None
        parts = [" AND ".join(core_parts[:2])]
        org = terms["organism"][0]
        tis = terms["tissue"][0]
        if org:
            parts.append(f'"{org}"[all]')
        if tis:
            parts.append(f'"{tis}"[tiab]')
        return " AND ".join(parts)

    def _build_tier3_broad_query(self, terms: Dict[str, List[str]], challenge: bool) -> str:
        tokens: List[str] = []
        for key in ("genes", "pathways", "cells"):
            for term in terms.get(key, []):
                clean = str(term).strip()
                if clean and clean not in tokens:
                    tokens.append(clean)
        if not tokens:
            context = terms.get("context", [""])[0]
            tokens = [w for w in str(context).split()[:6] if w]
        query = " ".join(tokens[:8])
        if challenge:
            query = f"{query} alternative mechanism negative evidence contradiction contrary findings"
        return query.strip()

    def _candidate_terms(self, candidate: Dict[str, Any]) -> Dict[str, List[str]]:
        # organism and tissue stay single-element: every builder reads [0] and
        # ANDs it as a mandatory arm. Normalising can empty that arm, and an
        # omitted constraint is strictly better than one that matches nothing.
        organism = normalize_search_terms([candidate.get("organism") or ""])
        tissue = normalize_search_terms([candidate.get("tissue") or ""])
        return {
            "genes": normalize_search_terms(as_list(candidate.get("key_genes"))),
            "cells": normalize_search_terms(as_list(candidate.get("key_cell_types"))),
            "pathways": normalize_search_terms(as_list(candidate.get("key_pathways"))),
            "methods": normalize_search_terms(as_list(candidate.get("suggested_methods"))),
            "organism": [organism[0] if organism else ""],
            "tissue": [tissue[0] if tissue else ""],
            # Left verbatim: the tier-3 fallback splits it into bare words and
            # never quotes it, so it is not subject to the phrase problem above.
            "context": [str(candidate.get("experimental_context") or "").strip()],
        }

    def _direction_queries(self, candidate: Dict[str, Any], deep: bool = False) -> List[DirectionSearchQuery]:
        terms = self._candidate_terms(candidate)
        queries: List[DirectionSearchQuery] = []
        meaningful = self._meaningful_term_count(terms)

        standard_builders = [
            # Keep direct_prior_work because negative-validation axes rely on this category name.
            ("direct_prior_work", lambda t: self._build_direct_prior_work(t, deep=False)),
            ("analogous_mechanism", self._build_analogous_mechanism),
            ("challenge_contradiction", self._build_active_challenge_query),
            ("prior_art", self._build_prior_art_query),
            ("contradictory_expectedness", self._build_contradictory_expectedness),
            ("methodological_precedent", self._build_methodological_precedent),
        ]
        deep_builders = [
            ("gene_model_relaxed", self._build_gene_model_relaxed),
            ("cell_type_focused", self._build_cell_type_focused),
            ("pathway_focused", self._build_pathway_focused),
            ("cross_species_or_tissue", self._build_cross_species_or_tissue),
        ]

        builders = deep_builders if deep else standard_builders
        # The degraded fallback depends only on `terms`, so every builder would
        # produce the same query string. Compute it once and emit a single
        # degraded query to avoid running the same PubMed search N times.
        degraded_fallback = self._build_degraded_fallback(terms) if meaningful < 2 else None
        emitted_degraded = False
        for category, builder in builders:
            query_str = builder(terms)
            if not query_str:
                continue
            if meaningful < 2 and degraded_fallback:
                if not emitted_degraded:
                    queries.append(DirectionSearchQuery(
                        category=category,
                        query=degraded_fallback,
                        degraded=True,
                        notes=f"Degraded: fewer than 2 meaningful axes (had {meaningful})"
                    ))
                    emitted_degraded = True
                continue
            queries.append(DirectionSearchQuery(category=category, query=query_str))

        return queries

    def _format_term_list(self, terms: List[str], field: str = "tiab") -> str:
        quoted = [f'"{t}"[{field}]' for t in terms if t]
        if not quoted:
            return ""
        if len(quoted) == 1:
            return quoted[0]
        return "(" + " OR ".join(quoted) + ")"

    async def _generate_pubmed_query(
        self,
        topic: str,
        state: Optional[AnalysisState] = None,
        current_round_findings: Optional[List[str]] = None
    ) -> str:
        """Use LLM to convert a research topic into an optimized PubMed search query.

        Passes raw experimental context to the LLM and lets it identify species,
        tissue, gene names, and disease models rather than using hardcoded regex.
        """
        ctx = self._build_research_context(topic, state, current_round_findings)

        # Build optional context sections. These inform terminology selection;
        # they are not additional scientific requirements for the query.
        extra_sections = []
        if ctx["treatment_groups"]:
            extra_sections.append(
                f"Treatment groups (context only): {', '.join(ctx['treatment_groups'])}"
            )
        if ctx["biochemistry_indicators"]:
            extra_sections.append(
                "Biochemistry indicators (context only): "
                f"{', '.join(ctx['biochemistry_indicators'][:8])}"
            )
        if ctx["previous_findings"]:
            extra_sections.append(
                "Prior coverage (do not use as positive query filters):\n"
                f"{ctx['previous_findings'][:500]}"
            )
        if ctx["current_round_findings"]:
            extra_sections.append(
                "Current findings (context only):\n"
                f"{ctx['current_round_findings'][:300]}"
            )
        extra_block = "\n\n".join(extra_sections)

        prompt = f"""Convert the following research topic into an optimized PubMed search query.

This context is not part of the authoritative question. Use it to understand terminology,
but do not automatically convert it into PubMed filters.

Rules for PubMed query:
1. Use MeSH terms where appropriate (e.g., "Liver"[MeSH Terms])
2. Use Boolean operators (AND, OR, NOT)
3. Keep the query focused but not overly restrictive. Do NOT tune the query toward a
   target number of hits: a precise query that returns few results on a genuinely narrow
   topic is a correct outcome, and broadening it until some count is reached only buys
   tangentially related papers that later get cited as support
4. Every top-level AND concept must be explicitly required by the authoritative research question
5. Add species, tissue, perturbation, disease model, or treatment as mandatory filters only when
   the authoritative question explicitly asks for evidence matched on that dimension
6. Use gene aliases as OR alternatives for the same target. Do not turn multiple genes found only
   in the context into separate mandatory AND concepts
7. Use field tags like [Title/Abstract], [MeSH] when helpful
8. Prefer the terms that match the question. If a precise query returns nothing, first
   check the terminology (synonyms, MeSH wording, gene aliases) before broadening the
   scope — a genuinely novel topic with no directly relevant literature is a finding,
   and widening the query until something matches only replaces it with a different topic

Return ONLY the PubMed query string, nothing else.

Authoritative research question:
{topic}

Context for terminology and disambiguation only:
Study species: {ctx.get("species") or "Not specified."}
Experimental background:
{ctx["initial_instruction"][:500] if ctx["initial_instruction"] else "Not available."}

{extra_block}"""

        response, _ = await self._chat_with_fallback(prompt, temperature=0.2)
        # Clean up the response - remove quotes, markdown, etc.
        query = response.strip().strip('"').strip("'").strip("`")
        # Remove markdown code block markers
        query = re.sub(r'^```\w*\n?', '', query)
        query = re.sub(r'\n?```$', '', query)
        return query.strip()

    async def _revise_zero_hit_query(
        self,
        topic: str,
        query: str,
        state: Optional[AnalysisState] = None,
        current_round_findings: Optional[List[str]] = None,
    ) -> str:
        """Revise one true-zero query without changing its scientific question."""
        ctx = self._build_research_context(topic, state, current_round_findings)
        prompt = f"""Revise a PubMed query that completed successfully but returned zero results.

Return one concept-preserving PubMed query. Keep every concept explicitly required by the
authoritative question. You may correct MeSH wording, add synonyms or gene aliases as OR
alternatives, and remove species, tissue, model, treatment, or other constraints that came
only from context. Do not replace the target mechanism or broaden to a different topic.

Return ONLY the revised PubMed query string, nothing else.

Authoritative research question:
{topic}

Executed query:
{query}

Context for terminology and disambiguation only:
Study species: {ctx.get("species") or "Not specified."}
Experimental background: {ctx["initial_instruction"][:500] or "Not available."}
Current findings: {ctx["current_round_findings"][:300] or "Not available."}"""
        response, _ = await self._chat_with_fallback(prompt, temperature=0.2)
        revised = response.strip().strip('"').strip("'").strip("`")
        revised = re.sub(r'^```\w*\n?', '', revised)
        revised = re.sub(r'\n?```$', '', revised)
        return revised.strip()

    def _meaningful_term_count(self, terms: Dict[str, List[str]]) -> int:
        count = 0
        for key in ("genes", "cells", "pathways", "methods"):
            if any(t for t in terms.get(key, []) if t):
                count += 1
        return count

    def _minimum_attempt_queries(self, candidate: Dict[str, Any]) -> List[DirectionSearchQuery]:
        terms = self._candidate_terms(candidate)
        support = self._build_direct_prior_work(terms, deep=False)
        challenge = self._build_active_challenge_query(terms)
        queries: List[DirectionSearchQuery] = []
        if support:
            queries.append(DirectionSearchQuery(category="direct_prior_work", query=support))
        if challenge:
            queries.append(DirectionSearchQuery(category="challenge_contradiction", query=challenge))
        # Keep both support and challenge so the minimum-attempt path can still
        # surface contradictory evidence (parity with the tier3 fallback below,
        # which also runs both); the shared direction deadline bounds total cost.
        if queries:
            return queries[:2]
        broad_support = self._build_tier3_broad_query(terms, challenge=False)
        broad_challenge = self._build_tier3_broad_query(terms, challenge=True)
        fallback = []
        if broad_support:
            fallback.append(DirectionSearchQuery(category="tier3_support", query=broad_support, degraded=True))
        if broad_challenge:
            fallback.append(DirectionSearchQuery(category="tier3_challenge", query=broad_challenge, degraded=True))
        if fallback:
            return fallback
        return []

    async def _validate_and_degrade_query(
        self,
        query: str,
        topic: str,
        max_results: int = 20,
        *,
        state: Optional[AnalysisState] = None,
        current_round_findings: Optional[List[str]] = None,
        enable_semantic_revision: bool = False,
        attempt_log: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Try a query and perform at most one bounded second attempt.

        General literature searches receive one concept-preserving semantic
        revision after a true zero result. PI-guided searches retain their
        legacy syntax-only degradation. A technical failure is retried without
        changing the scientific query.

        Returns (final_query, articles) tuple.
        """
        attempt = 0
        max_attempts = 2
        last_error: Optional[Exception] = None
        last_search_failed = False
        attempts = attempt_log if attempt_log is not None else []
        next_attempt_kind = "initial"
        while attempt < max_attempts:
            kind = next_attempt_kind
            try:
                articles = await self._pubmed_search(query, max_results=max_results)
                last_search_failed = False
                attempts.append({
                    "kind": kind,
                    "query": query,
                    "result_count": len(articles),
                })
                if articles:
                    if attempt > 0:
                        logger.info(
                            f"[Literature] Query succeeded after {attempt} "
                            f"retry/revision(s): {query[:80]}"
                        )
                    return query, articles
            except Exception as e:
                last_error = e
                last_search_failed = True
                attempts.append({
                    "kind": kind,
                    "query": query,
                    "result_count": None,
                    "error": str(e)[:500],
                })
                logger.warning(f"[Literature] PubMed search error (attempt {attempt + 1}): {e}")

            # Degrade the query
            if attempt == 0:
                if enable_semantic_revision and not last_search_failed:
                    try:
                        degraded = await self._revise_zero_hit_query(
                            topic,
                            query,
                            state=state,
                            current_round_findings=current_round_findings,
                        )
                    except UpstreamExhaustedError:
                        raise
                    except Exception as e:
                        logger.warning(
                            f"[Literature] Zero-hit query revision unavailable: {e}"
                        )
                        break
                elif enable_semantic_revision:
                    # A technical failure is retried without changing the
                    # scientific query. It must not trigger semantic broadening.
                    degraded = query
                else:
                    # Legacy PI-guided path: strip field tags and quotes.
                    degraded = re.sub(
                        r'\[(?:MeSH(?: Terms)?|Title/Abstract|Title|Author|Journal)\]',
                        '',
                        query,
                        flags=re.IGNORECASE,
                    )
                    degraded = re.sub(r'"([^"]*)"', r'\1', degraded)
                    degraded = degraded.strip()
                same_query = (
                    re.sub(r'\s+', ' ', degraded).strip().lower()
                    == re.sub(r'\s+', ' ', query).strip().lower()
                )
                if degraded and not same_query:
                    query = degraded
                    next_attempt_kind = (
                        "zero_hit_revision"
                        if enable_semantic_revision else "syntax_degradation"
                    )
                    logger.info(f"[Literature] Revising zero-result query: {query[:80]}")
                    await asyncio.sleep(0.4)
                    attempt += 1
                    continue
                if enable_semantic_revision and last_search_failed:
                    next_attempt_kind = "technical_retry"
                    await asyncio.sleep(0.4)
                    attempt += 1
                    continue
            break

        if last_search_failed and last_error is not None:
            raise last_error
        return query, []
