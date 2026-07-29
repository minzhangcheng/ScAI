"""Domain-neutral retention checks for iteratively revised analysis plans.

An incremental revision loop rewrites plan prose from a bounded prompt.  Every
round the PI's feedback accumulates while the plan itself is re-derived, so the
specific content a plan carries decays: the terms a model is least likely to
reinvent unprompted are exactly the ones the research objective supplied.  The
generic methodology vocabulary survives because the model regenerates it
anyway, which makes the loss invisible in a diff of step titles.

These helpers measure that loss with pure string operations.  They contain no
domain vocabulary — no tissue, disease, organism, mechanism, or pathway names —
and therefore stay valid for any study the platform is pointed at.  The filter
is self-selecting: a term that the revision kept is not reported, so generic
words drop out on their own and what remains is whatever the revision uniquely
lost.
"""
from collections import Counter
from typing import Iterable, List, Sequence, Set
import re

# Tokens shorter than this are noise unless they mix letters and digits, which
# is the shape of an identifier (a gene symbol, a clone name, an assay code).
MIN_TERM_LENGTH = 4
MIN_ALNUM_TERM_LENGTH = 3

# English function words plus the boilerplate that every plan prompt contains.
# Deliberately generic: adding a domain word here would make the check
# domain-specific, which is the failure mode this module exists to catch.
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
    # Prompt/plan scaffolding that appears regardless of subject matter.
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


def term_frequencies(text: str) -> Counter:
    """Salient-token frequencies, used to rank rarity."""
    if not text:
        return Counter()
    lowered = str(text).lower().replace(".", " ").replace("/", " ")
    return Counter(
        token for token in _TOKEN_RE.findall(lowered) if _is_salient(token)
    )


def missing_salient_terms(
    required_text: str,
    produced_text: str,
    *,
    limit: int = 40,
    priority_text: str = "",
) -> List[str]:
    """Terms present in ``required_text`` but absent from ``produced_text``.

    Ordering decides what survives the ``limit`` cut, so it ranks by how
    load-bearing a term is, not by how unusual it looks. Terms from
    ``priority_text`` — what was explicitly asked for — come first, then terms
    the source repeated most, since repetition is emphasis. Ranking rarity
    first instead surfaces one-off compound phrases from incidental prose and
    pushes the genuinely central vocabulary past the cut.
    """
    produced = extract_salient_terms(produced_text)
    priority = extract_salient_terms(priority_text)
    counts = term_frequencies(required_text)
    missing = [term for term in counts if term not in produced]
    missing.sort(key=lambda term: (0 if term in priority else 1, -counts[term], term))
    return missing[:limit] if limit and limit > 0 else missing


def retention_ratio(required_text: str, produced_text: str) -> float:
    """Fraction of the source's salient terms that survived. 1.0 when empty."""
    required = extract_salient_terms(required_text)
    if not required:
        return 1.0
    produced = extract_salient_terms(produced_text)
    return len(required & produced) / len(required)


def uncovered_items(items: Iterable[str], produced_text: str, *, min_overlap: float = 0.6) -> List[str]:
    """Items whose salient terms are largely absent from ``produced_text``.

    Used to report which explicitly stated requirements a plan does not
    address.  An item counts as covered when at least ``min_overlap`` of its
    own salient terms appear somewhere in the produced text — matching whole
    sentences would never fire, and matching a single token always would.
    """
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


def format_terms(terms: Sequence[str]) -> str:
    """Comma-joined term list for prompt injection."""
    return ", ".join(terms)
