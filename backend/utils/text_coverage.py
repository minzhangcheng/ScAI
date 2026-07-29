"""Does a piece of generated text address what was asked of it?

Used to check an analysis plan against the questions its research direction was
selected to resolve. The comparison is pure string work with no domain
vocabulary — no tissue, disease, organism, or mechanism — so it stays valid for
any study the platform is pointed at.

This module previously also scored how much of a plan's vocabulary survived a
revision, and gated a repair pass on the result. Measured against every real
revision on disk, that ratio does not separate a revision that dropped the
study's mechanism vocabulary from one that kept it, so the score and its
consumers were removed rather than left as an enforcement mechanism someone
would rewire. The underlying loss is prevented in the revision prompt instead
(see `backend/agents/bioinformatics/planning.py`), and the raw material for a
future calibration is still on disk in every session's `plan_dialogue_history`.
"""
from typing import Iterable, List, Set
import re

# Tokens shorter than this are noise unless they mix letters and digits, which
# is the shape of an identifier (a gene symbol, a clone name, an assay code).
MIN_TERM_LENGTH = 4
MIN_ALNUM_TERM_LENGTH = 3

# An item counts as covered when at least this fraction of its own salient
# terms appears in the produced text. Matching whole sentences would never
# fire; matching a single token always would.
DEFAULT_MIN_OVERLAP = 0.6

# English function words plus the scaffolding every plan carries. Deliberately
# generic: adding a domain word here would make the check domain-specific,
# which is the failure mode this module exists to catch.
_STOPWORDS: Set[str] = {
    "about", "above", "after", "again", "against", "all", "also", "and", "any",
    "are", "because", "been", "before", "being", "below", "between", "both",
    "but", "cannot", "could", "does", "doing", "done", "down", "during", "each",
    "either", "else", "enough", "even", "ever", "every", "few", "for", "from",
    "further", "had", "has", "have", "having", "here", "how", "however", "into",
    "its", "itself", "just", "least", "less", "like", "made", "make", "many",
    "may", "might", "more", "most", "much", "must", "neither", "nor", "not",
    "now", "off", "once", "only", "onto", "other", "others", "otherwise", "our",
    "out", "over", "own", "per", "rather", "same", "shall", "should", "since",
    "some", "such", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "through", "thus", "too", "under",
    "unless", "until", "upon", "use", "used", "using", "very", "was", "were",
    "what", "when", "where", "whether", "which", "while", "who", "whom", "why",
    "will", "with", "within", "without", "would", "you", "your",
    # Plan scaffolding, which appears regardless of subject matter.
    "step", "steps", "plan", "plans", "title", "input", "inputs", "output",
    "outputs", "method", "methods", "rationale", "expected", "depends",
    "revise", "revised", "revision", "round", "rounds", "task", "tasks",
    "feedback", "json", "list", "lists", "note", "notes", "none", "null",
    "true", "false", "text",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+-]*")
_HAS_LETTER_RE = re.compile(r"[a-z]")
_HAS_DIGIT_RE = re.compile(r"[0-9]")


def _is_salient(token: str) -> bool:
    if token in _STOPWORDS:
        return False
    if not _HAS_LETTER_RE.search(token):
        # Pure numbers, versions, and thresholds are parameters, not content.
        return False
    if _HAS_DIGIT_RE.search(token):
        # Identifier-shaped tokens stay even when short.
        return len(token) >= MIN_ALNUM_TERM_LENGTH
    return len(token) >= MIN_TERM_LENGTH


def extract_salient_terms(text: str) -> Set[str]:
    """Content-bearing lowercase tokens in ``text``.

    Dotted paths are split so ``scanpy.tl.score_genes`` contributes
    ``scanpy`` and ``score_genes`` rather than one unmatchable blob.
    """
    if not text:
        return set()
    lowered = str(text).lower().replace(".", " ").replace("/", " ")
    return {token for token in _TOKEN_RE.findall(lowered) if _is_salient(token)}


def uncovered_items(
    items: Iterable[str],
    produced_text: str,
    *,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> List[str]:
    """Items whose salient terms are largely absent from ``produced_text``."""
    produced = extract_salient_terms(produced_text)
    uncovered: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        terms = extract_salient_terms(text)
        if not terms:
            continue
        if len(terms & produced) / len(terms) < min_overlap:
            uncovered.append(text)
    return uncovered
