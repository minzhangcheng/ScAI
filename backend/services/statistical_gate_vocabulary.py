"""Pure text and method vocabulary used by statistical gate enforcement."""

from __future__ import annotations

import re


#: Wording that asserts a mechanism is absent. Matching is deliberately narrow:
#: a false positive here blocks a legitimate finding, which is worse than
#: missing one, since the underpowered caveat is attached to the score either
#: way and travels into the report.
_NEGATIVE_CLAIM_PATTERNS = (
    "not significant", "no significant", "not enriched", "no enrichment",
    "not supported", "no evidence for", "no evidence of", "rules out",
    "ruled out", "excluded as", "is absent", "not detected", "argues against",
    "does not support", "no change", "unchanged",
)


def _texts(findings) -> list[str]:
    """Flatten a findings payload to plain strings."""
    out: list[str] = []
    if isinstance(findings, str):
        return [findings]
    for item in findings or []:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            out.extend(str(v) for v in item.values() if isinstance(v, str))
    return out


def asserts_absence(text: str) -> bool:
    """Whether a sentence claims a mechanism is absent rather than unmeasured."""
    lowered = str(text).lower()
    return any(pattern in lowered for pattern in _NEGATIVE_CLAIM_PATTERNS)


def _mentions(text: str, needle: str) -> bool:
    return needle.lower() in str(text).lower()


# ------------------------------------------------------------- segmentation
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9*•\-\'\"])')
_CLAUSE_SPLIT = re.compile(
    r'(?:'
    r'\s*[;:]\s*'
    r'|,\s+(?:and|but|with|while|whereas|although|though|which|reflecting|'
    r'supporting|suggesting|indicating|consistent|alongside|characteri[sz]ed|'
    r'accompanied|driven|together|particularly|including|led\s+by|so|therefore|hence|thus)\s+'
    r'|\s+(?:while|whereas|although)\s+'
    r')'
)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _sentences(text):
    parts = [p.strip() for p in _SENT_SPLIT.split(str(text)) if p and p.strip()]
    return parts or [str(text)]


def _clauses(sent):
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(sent) if p and p.strip()]
    return parts or [sent]


def _tokens(text):
    return set(_WORD_RE.findall(str(text).lower()))


def _has_word(text, words):
    return bool(_tokens(text) & set(words))


# --------------------------------------------------------- inference markers
_P_NUMERIC = re.compile(
    r'\b(?:p|p[-_ ]?val(?:ue)?s?|p[-_ ]?adj|padj|adj[-_. ]?p(?:[-_ ]?val(?:ue)?)?|'
    r'adjusted\s+p(?:[-_ ]?val(?:ue)?)?|q|q[-_ ]?val(?:ue)?|fdr|bh)\b'
    r'\s*(?:[-_ ]?adjusted)?\s*'
    r'(?:[<>=~≤≥≈]{1,2}|\b(?:below|under|above|less\s+than|'
    r'greater\s+than|at\s+most|at\s+least|of|was|were|is|are)\b'
    r'|(?<=\s))\s*~?\s*\d', re.I)
_P_APPROX_ZERO = re.compile(
    r'\bp\s*(?:[-_ ]?val(?:ue)?)?\s*(?:approx\.?|~|≈)?'
    r'\s*[<>=~≤≈]{0,2}\s*(?:approx\.?\s*)?0\b', re.I)
_NAMED_TEST = re.compile(
    r"\b(?:fisher'?s?(?:\s+exact)?|chi[-\s]?squared?|chi2|chisq|"
    r"mann[-\s]?whitney|mwu|wilcoxon|kruskal[-\s]?wallis|"
    r"kolmogorov[-\s]?smirnov|ks\s+test|t[-\s]?test|welch|"
    r"permanova|binomial\s+test|proportions?[-\s]test|permutation\s+test)\b", re.I)
_COUNT_TEST = re.compile(
    r"\b(?:fisher'?s?(?:\s+exact)?|chi[-\s]?squared?|chi2|chisq|"
    r"binomial\s+test|proportions?[-\s]test|g[-\s]test)\b", re.I)
_SIG_WORD = re.compile(r'\bsignifican(?:t|tly|ce)\b', re.I)
_NEG_SIG = re.compile(
    r'\b(?:no|not|non|never|without|lack|lacks|lacking|absent|nor|only|neither)\b'
    r'[^.;:]{0,32}\bsignifican', re.I)
_NEG_SIG_TRAILING = re.compile(r'\bsignifican\w*\b[^,.;:]{0,25}\b(?:not|no)\b', re.I)
_NOT_SIG = re.compile(r'\((?:ns|n\.s\.)\)|\bnon[-\s]?significant\b', re.I)


#: Inference stated without a p-value. Every one of these is the same claim --
#: "this difference is not sampling noise" -- and a gate keyed only on p/FDR/
#: "significant" is keyed on one dialect of it.
_ALT_INFERENCE = re.compile(
    r'(?:\b(?:credible|confidence|highest[-\s]density)\s+interval\b'
    r'|\bposterior\s+(?:probability|odds)\b'
    r'|\bbayes\s+factor\b'
    r'|\b\d{2,3}\s*%\s*(?:ci|credible|confidence)\b'
    r'|\bexcludes?\s+(?:zero|0|the\s+null)\b'
    r'|\breject(?:s|ed|ing)?\s+the\s+null\b'
    r'|\bbeyond\s+(?:sampling|chance)\s+(?:noise|variation)\b'
    r'|\bunlikely\s+to\s+be\s+(?:due\s+to|explained\s+by)\s+chance\b'
    r'|\bstatistically\s+(?:reliable|credible|robust)\b'
    r'|\b(?:fwer|family[-\s]wise\s+error)\b'
    r'|\bz\s*[=<>]\s*[-+]?[\d.]'
    r'|\bt\s*=\s*[-+]?[\d.]{2,})', re.I)


#: Same grammar as _P_NUMERIC, but capturing the comparator and the number so a
#: quoted value can be checked against the resolution the method could attain.
_P_NUMERIC_VALUE = re.compile(
    r'\b(?:p|p[-_ ]?val(?:ue)?s?|p[-_ ]?adj|padj|adj[-_. ]?p(?:[-_ ]?val(?:ue)?)?|'
    r'adjusted\s+p(?:[-_ ]?val(?:ue)?)?|q|q[-_ ]?val(?:ue)?|fdr|bh)\b'
    r'\s*(?:[-_ ]?adjusted)?\s*'
    r'(?:([<>=~≤≥≈]{1,2})|\b(below|under|less\s+than|at\s+most|above|'
    r'greater\s+than|at\s+least|of|was|were|is|are)\b'
    r'|(?<=\s))\s*~?\s*(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)', re.I)

#: Comparators that already state the value as a bound. A floor reported as an
#: inequality is the correct reporting, and must not be refused.
_BOUNDED_COMPARATORS = ("<", "<=", "≤", "below", "under", "less than", "at most")

#: Vocabulary specific to a resampling-based enrichment result. Used so the
#: floor check cannot fire on an analytic p-value, which has no floor: a
#: model-fit adjusted p-value may legitimately be arbitrarily small.
_RESAMPLED_RESULT = re.compile(
    r'\b(?:nes\b|gsea|pre-?rank(?:ed)?|enrichment\s+score|'
    r'permutation|resampl(?:e|ed|ing)|bootstrap|monte\s*carlo)', re.I)

#: Keys under which a resample count may have been declared on a score.
_RESAMPLE_PARAM_KEYS = (
    "n_resamples", "permutation_num", "n_perm", "nperm", "permutations",
    "n_permutations", "n_bootstrap", "n_boot",
)

#: Conventional error-rate cutoffs. A repeated value equal to one of these is a
#: threshold being restated, not a shared resolution floor.
_CONVENTIONAL_CUTOFFS = (0.05, 0.01, 0.001, 0.1, 0.25, 0.2)

#: Tolerance on the floor comparison, because a floor value is almost always
#: quoted rounded: 1/501 is written "0.002". The next attainable value up is
#: 2/501, roughly double, so this cannot reach a value that is genuinely above
#: the floor -- it only catches the floor wearing fewer digits.
_FLOOR_ROUNDING_TOLERANCE = 1.02


def _has_p(clause):
    return bool(_P_NUMERIC.search(clause) or _P_APPROX_ZERO.search(clause))


def _states_a_null(clause):
    return bool(_NEG_SIG.search(clause) or _NEG_SIG_TRAILING.search(clause)
                or _NOT_SIG.search(clause))


def _has_sig_word(clause):
    return bool(_SIG_WORD.search(clause)) and not _states_a_null(clause)


def _has_inference(clause):
    return (_has_p(clause) or bool(_NAMED_TEST.search(clause))
            or _has_sig_word(clause) or bool(_ALT_INFERENCE.search(clause)))


# Distinct from workflow.post_execution_review.METHOD_SIGNATURES on purpose, and
# not a drifted copy of it: that table asks whether a planned step ran and reads
# substrings of code, this one asks what a registered score licenses and reads
# whole tokens of one declared field. Agreement between them is not a goal --
# betabinom and quasibinomial belong in its composition family and must stay out
# of _DA below. Do not unify them.
FAMILY_REQUIRED_KINDS = ("ranked_enrichment", "over_representation")
_RANKED = frozenset({"prerank", "preranked", "gsea", "gseapreranked", "fgsea",
                     "camera", "roast", "fry"})
_ORA = frozenset({"ora", "enrich", "enrichr", "hypergeometric", "hypergeom",
                  "overrepresentation", "clusterprofiler", "enrichgo", "enrichkegg",
                  "goatools", "gost", "gprofiler", "panther", "david"})
_SSGSEA = frozenset({"ssgsea", "aucell", "ucell", "gsva", "singscore"})
_MODULE = frozenset({"score_genes", "addmodulescore", "scoregenes", "module_score"})
_ACTIVITY = frozenset({"ulm", "mlm", "viper", "dorothea", "scenic", "pyscenic",
                       "decoupler", "progeny"})
_DE = frozenset({"deseq2", "pydeseq2", "deseqstats", "edger", "limma", "voom",
                 "rank_genes_groups", "mannwhitneyu", "wilcoxon", "ttest"})
#: Matching is whole-token, so a package's code spelling and its prose spelling
#: are separate entries and neither implies the other: "miloR::testNhoods" does
#: not contain the token "milo". Deliberately absent: quasibinomial and glmmTMB,
#: which fit one population at a time and so are the design this gate refuses
#: rather than an escape from it; pertpy, whose composition entry points already
#: match through sccoda and milo while the bare name would also capture its
#: perturbation and enrichment tools; and scipy.stats.betabinom, a univariate
#: distribution that carries no joint model however it is spelled.
_DA = frozenset({"sccoda", "sccomp", "propeller", "scanpro", "milo", "milopy",
                 "milor", "dirichlet", "betabinomial"})


def infer_method_kind(method):
    """Classify a declared method string into one analysis family.

    Whole-token match over the declared `method`, first hit wins, "other" when
    nothing matches. `method` is free text the model writes -- register_score
    takes no enum -- so this classifies a self-report rather than verifying one.
    That is why the tables stay narrow: at check_declared_testing_family and
    check_ora_background a name this misses costs a missed check, but at
    check_composition_claims the kind is an exemption, so a name it matches too
    eagerly costs a refusal that should have stood.
    """
    raw = str(method or "").lower()
    tok = _tokens(raw)

    def has(names):
        return bool(tok & names) or any(n in raw for n in names if "_" in n)
    if has(_DA):
        return "differential_abundance"
    if has(_RANKED):
        return "ranked_enrichment"
    if has(_SSGSEA):
        return "single_sample_enrichment"
    if has(_MODULE):
        return "module_score"
    if has(_ORA):
        return "over_representation"
    if has(_ACTIVITY):
        return "regulatory_activity"
    if has(_DE):
        return "differential_expression"
    return "other"


def score_kind(score):
    return infer_method_kind(getattr(score, "method", ""))
