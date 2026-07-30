# Statistical-validity checks applied before findings are accepted.
"""Gates that a model cannot talk its way past.

The prompt already asks for most of this. Prompted guidance is not enough: over
a 200-step loop it drifts, and nothing observes whether it was followed. These
checks run against structured state -- registered scores, frozen plans -- rather
than by grepping the agent's code text, so they see what actually happened.

Each gate cites a rule id from services/method_rules, so a refusal arrives with
its mechanism and its literature attached instead of as a bare "no". That is the
whole design: the gate stops the claim, the rule explains why, and the model
gets enough to fix it rather than to retry blindly.

Every check here is structural. Nothing inspects biology, and nothing may: the
gates ask how many genes matched, whether a family was declared, how many
samples a condition had -- never which genes or which condition.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .gene_set_provenance import MIN_PANEL_SIZE_FOR_NEGATIVE_CLAIM
from .method_rules import rule_text

#: Below this many samples per group, a permutation-based q-value cannot be
#: trusted. Phenotype permutation degrades under it, and the number of distinct
#: label partitions becomes small enough to floor the attainable p-value: at 3
#: versus 3 there are only 10, so nothing below ~0.1 is reachable at all.
MIN_SAMPLES_PER_GROUP_FOR_PERMUTATION = 7

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


@dataclass
class GateFinding:
    """One failed check, with the rule that explains it."""

    gate: str
    rule_id: str
    detail: str
    subject: str = ""
    #: Further rules bearing on the same refusal. A violation usually has one
    #: governing rule and several that explain the surrounding practice; naming
    #: them here keeps the primary message short while still surfacing the rest.
    also_see: tuple = ()
    #: "block" refuses the call; "advise" surfaces a note the model cannot
    #: suppress but does not stop it. A check whose coverage is too narrow to
    #: justify spending a rejection -- and a retry, and a step from a 200-step
    #: budget -- belongs on the advisory channel rather than being deleted.
    severity: str = "block"

    def message(self) -> str:
        head = f"[{self.gate}]"
        if self.subject:
            head += f" {self.subject}:"
        guidance = rule_text(self.rule_id)
        extra = " ".join(
            rule_text(rid, include_python=False) for rid in self.also_see
        )
        return f"{head} {self.detail} {guidance} {extra}".strip()


@dataclass
class GateReport:
    """Outcome of running the checks over a set of findings."""

    findings: list[GateFinding] = field(default_factory=list)

    @property
    def blocking(self) -> list:
        return [f for f in self.findings if f.severity == "block"]

    @property
    def advisories(self) -> list:
        return [f for f in self.findings if f.severity != "block"]

    @property
    def passed(self) -> bool:
        """True when nothing BLOCKS. Advisories travel with an accepted call."""
        return not self.blocking

    def message(self) -> str:
        if self.passed:
            return ""
        lines = ["[REJECTED] Statistical validity checks failed."]
        lines.extend(f.message() for f in self.blocking)
        lines.append(
            "Fix these and call report_findings again. If a check is wrong for "
            "this analysis, say so explicitly in the finding text rather than "
            "restating the claim unchanged."
        )
        return "\n\n".join(lines)

    def advisory_message(self) -> str:
        """Notes attached to an accepted call, or '' when there are none."""
        if not self.advisories:
            return ""
        return "\n\n".join(
            ["[ADVISORY] These do not block the call but belong in the report."]
            + [f.message() for f in self.advisories]
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
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9*•\-\'"])')
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


# -------------------------------------------------------------- exemptions
_DISCLAIMED = re.compile(
    r'\b(?:descriptive|exploratory|not\s+inferential|'
    r'no\s+(?:cell[-\s]level\s+|compositional\s+)?p[-\s]?values?|'
    r'compositional\s+significance\s+testing[^.;:]{0,40}block|'
    r'not\s+differential[-\s]abundance|single[-\s]sample\s+caveat|'
    r'no\s+biological\s+replicat|lacks?\s+independent\s+replication|'
    r'unidentifiable|not\s+estimable|cannot\s+support\s+statistical\s+inference|'
    r'were\s+blocked|proportions?\s+are\s+descriptive|technical\s+arte?fact|not\s+a\s+biological)', re.I)
_COMPOSITIONAL_METHOD = re.compile(
    r'\b(?:sccoda|sccomp|dirichlet[-\s]?multinomial|dirichlet|'
    r'beta[-\s]?binomial|milo(?:py|r)?|propeller|'
    r'cent[er]{2}ed?\s+log[-\s]?ratio|clr|arcsin\w*|'
    r'isometric\s+log[-\s]?ratio|additive\s+log[-\s]?ratio|aitchison)\b', re.I)
_SAMPLE_LEVEL = re.compile(
    r'\b(?:bulk|pseudo-?bulk|whole[-\s]\w+|tissue[-\s]level|sample[-\s]level|'
    r'animal[-\s]level|donor[-\s]level|subject[-\s]level|per[-\s]sample|'
    r'per[-\s]animal|per[-\s]donor|per[-\s]subject|per[-\s]replicate|'
    r'pydeseq2|deseq2|edger|limma|\d+[-\s]animal|'
    r'n\s*=\s*\d+\s*(?:per\s+group|v\d))\b'
    r'|\baggregat\w+[^.;:]{0,40}\b(?:per|to|by)\s+'
    r'(?:animal|sample|donor|subject|replicate|specimen)\b', re.I)
_SET_RESOURCE = re.compile(
    r'\b(?:gsea|nes\s*=|nes\s+[-\d]|enrichr|hallmark|kegg|reactome|msigdb|'
    r'go[:_ ]\d{4}|leading[-\s]edge|over-?representation|ora|prerank\w*|'
    r'gsva|ssgsea|fgsea)\b', re.I)
_SET_NOUN = re.compile(
    r'\b(?:gene[-\s]?sets?|genesets?|gene[-\s]?lists?|pathways?|ontolog\w+|go|'
    r'terms?|degs?|differentially\s+expressed\s+genes?|marker\s+genes?|'
    r'target\s+genes?|signatures?|modules?|programs?|genes?)\b', re.I)
_SET_VERB = re.compile(
    r'\b(?:enrich\w*|over-?represent\w*|under-?represent\w*|depleted\s+for)\b', re.I)


def _is_set_level(clause):
    if _SET_RESOURCE.search(clause):
        return True
    return bool(_SET_NOUN.search(clause) and _SET_VERB.search(clause))


_CORRELATION = re.compile(
    r'(?:\bcorrelat\w+|\bspearman\b|\bpearson\b|\bkendall\b|\brho\s*[=~]|'
    r'\bregression\b|\bslope\b|\bco-?varies\b|\btracks\s+with\b)', re.I)
_NULL_MODEL = re.compile(
    r'\b(?:matched[-\s]?null|null\s+(?:model|distribution|set)|versus\s+null|'
    r'against\s+(?:a\s+)?\w*\s?null|permut\w+\s+null|random\s+background)\b', re.I)
_Z_AGAINST_NULL = re.compile(r'\bz\s*=\s*[-+]?\d', re.I)


# ------------------------------------------------------ population vocabulary
_POPULATION = re.compile(
    r'\b(?:cells?|cellular|nuclei|nucleus|populations?|clusters?|sub-?clusters?|'
    r'cell[-\s]?types?|celltypes?|cell[-\s]?states?|subtypes?|subsets?|'
    r'compartments?|lineages?|barcodes?|droplets?)\b', re.I)
_PERCENT = re.compile(r'%|\bper\s?cents?\b|\bpercent(?:age)?s?\b', re.I)
#: Nouns that ARE a composition, whatever the population is called. Without this
#: the gate can only see a claim when a generic population word ("cells",
#: "cluster") or an annotated label happens to be in the same sentence.
_COMPOSITION_NOUN = re.compile(
    r'\b(?:proportions?|fractions?|abundan(?:ce|ces)|compositions?|compositional|'
    r'frequenc(?:y|ies)|percentages?|shares?)\b', re.I)
_ABUNDANCE = re.compile(
    r'\b(?:proportions?|percentages?|fractions?|abundan(?:ce|ces|t)|frequenc(?:y|ies)|'
    r'compositions?|compositional|expand(?:s|ed|ing)?|expansions?|deplet(?:e|es|ed|ion)|'
    r'enrich(?:ed|ment|es)?|infiltrat\w*|recruit\w*|imbalance\w*|'
    r'over-?represent\w*|under-?represent\w*|increase[sd]?|decrease[sd]?|'
    r'rise[sn]?|rose|fell|falls?|drops?|dropped|loss|lost|gain(?:ed|s)?|'
    r'shift(?:s|ed)?|remodel\w*|more\s+frequent|less\s+frequent|skew(?:ed)?|'
    r'dominat\w*)\b'
    r'|\b(?:more|fewer|less|greater|higher|lower|larger|smaller)\s+'
    r'(?!than\b)(?:\w+\s+){0,3}(?:cells?|nuclei|barcodes?|droplets?|populations?|'
    r'clusters?|subsets?|subtypes?)\b', re.I)
#: a MEASURED QUANTITY per observation. When one is present the thing that
#: changed is a score or a gene list, not a head-count, so the simplex does not
#: apply. Ported from design A's _MEASURED_QUANTITY_WORDS, extended with the
#: QC-metric nouns that produced a false positive on an invented negative.
_MEASURED_QUANTITY = re.compile(
    r'\b(?:scores?|scored|activit(?:y|ies)|expression|expressed|levels?|signals?|'
    r'signatures?|programs?|modules?|footprints?|pseudotime|trajector\w+|'
    r'transcripts?|genes?|degs?|markers?|targets?|pathways?|ontolog\w+|nes|'
    r'reads?|umis?|counts\s+per|mitochondrial|ribosomal|doublet\w*)\b', re.I)


# ============================================================ COMPOSITION
def check_composition_claims(findings, *, score_definitions=()):
    """A composition claim must not rest on a per-population test of proportions."""
    if any(score_kind(s) == "differential_abundance" for s in score_definitions or ()):
        return []
    for text in _texts(findings):
        for sent in _sentences(text):
            in_scope = bool(_POPULATION.search(sent))
            for clause in _clauses(sent):
                if not (in_scope or _PERCENT.search(clause)
                        or _COMPOSITION_NOUN.search(clause)):
                    continue
                if _DISCLAIMED.search(clause) or _states_a_null(clause):
                    continue
                if _COMPOSITIONAL_METHOD.search(clause):
                    continue
                if _is_set_level(clause):
                    continue
                measured = bool(_MEASURED_QUANTITY.search(clause))
                abundance_claim = (
                    bool(_ABUNDANCE.search(clause))
                    and _has_inference(clause)
                    and not measured
                    and not _CORRELATION.search(clause))
                count_test = bool(_COUNT_TEST.search(clause)) and not measured
                if not (abundance_claim or count_test):
                    continue
                return [GateFinding(
                    gate="simplex-ignored",
                    rule_id="composition-lives-on-a-simplex",
                    detail=(
                        "this finding attaches an inferential statistic to a "
                        "between-condition population-abundance claim. Proportions sum "
                        "to one, so an increase anywhere mechanically depresses every "
                        "other population whether or not its absolute count changed, "
                        "and a per-population test over cells uses cells as the sample "
                        "size. Model the joint composition (scCODA / sccomp / propeller "
                        "/ Milo), or state the shift as descriptive."))]
    return []


# ============================================================ CELL LEVEL
_SINGLE_CELL_MODALITY = re.compile(
    r'\b(?:single[-\s]?cell|single[-\s]?nucle\w+|sc[-\s]?rna|sn[-\s]?rna|'
    r'scrna\w*|snrna\w*|per[-\s]?cell|cell[-\s]?level|leiden|louvain|umap|'
    r'harmony|sub-?cluster\w*|cells?|nuclei|barcodes?|droplets?)\b', re.I)
_FEATURE_COUNT_CLAIM = re.compile(
    r'\b\d[\d,]*\s+(?:\S+\s+){0,2}?(?:degs?|differentially[-\s]expressed\s+genes?|'
    r'(?:up|down)-?regulated\s+genes?|sig(?:nificant)?\s+genes?)\b', re.I)
_CONDITION_CONTRAST = re.compile(
    r'(?:\bbetween\s+(?:the\s+)?(?:two\s+)?(?:conditions?|groups?|genotypes?|'
    r'arms?)\b|\bacross\s+conditions?\b|\bcondition[-\s]?(?:level|specific|'
    r'wise)\b|\bcondition\s+effect\b|\bbetween[-\s]condition\b'
    r'|\b(?:the\s+)?(?:two|both)\s+(?:arms?|groups?|conditions?|genotypes?)\b'
    r'|\bcompar\w+\s+(?:the\s+)?(?:two|both)\b'
    r'|\beffects?\s+of\s+condition\b|\bcondition\b[^.;:]{0,24}\bfixed\s+effect\b)', re.I)


def _carries_cell_level_inference(clause):
    """``_has_inference`` plus a feature count, minus bare abundance wording.

    "Significant" next to a proportion is a composition claim, which the
    simplex gate owns; this one is about the unit of analysis.
    """
    return (_has_p(clause) or bool(_NAMED_TEST.search(clause))
            or bool(_ALT_INFERENCE.search(clause))
            or (_has_sig_word(clause) and not _ABUNDANCE.search(clause))
            or bool(_FEATURE_COUNT_CLAIM.search(clause)))


def _group_label_pattern(samples_per_group):
    labels = [re.escape(str(g)) for g in (samples_per_group or ()) if str(g).strip()]
    if not labels:
        return None
    return re.compile(
        r'(?<![A-Za-z0-9_])(?:' + '|'.join(labels) + r')(?![A-Za-z0-9_])', re.I)


def _within_group_pattern(samples_per_group):
    """Wording that places the contrast INSIDE one condition.

    A statistic computed within a single group is not a between-condition claim,
    so this gate is the wrong one for it whatever its n was.
    """
    labels = [re.escape(str(g)) for g in (samples_per_group or ()) if str(g).strip()]
    alt = '|'.join(labels + [r'(?:the\s+)?(?:\w+\s+)?condition', r'(?:each|either|one)\s+\w+',
                             r'(?:the\s+)?(?:same\s+)?group', r'(?:the\s+)?\w*\s?arm'])
    return re.compile(r'\bwithin\s+(?:' + alt + r')\b', re.I)


def check_cell_level_inference(findings, samples_per_group=None):
    """A condition-level claim must not rest on cells treated as replicates."""
    texts = _texts(findings)
    if not any(_SINGLE_CELL_MODALITY.search(t) for t in texts):
        return []
    labels = _group_label_pattern(samples_per_group)
    within = _within_group_pattern(samples_per_group)
    for text in texts:
        for sent in _sentences(text):
            contrast = (bool(labels.search(sent)) if labels else False) \
                or bool(_CONDITION_CONTRAST.search(sent))
            if not contrast:
                continue
            if within.search(sent):
                continue
            sample_level_sentence = bool(_SAMPLE_LEVEL.search(sent))
            sentence_disclaimed = bool(_DISCLAIMED.search(sent))
            for clause in _clauses(sent):
                if not _carries_cell_level_inference(clause):
                    continue
                if sentence_disclaimed or _DISCLAIMED.search(clause):
                    continue
                if _states_a_null(clause):
                    continue
                if sample_level_sentence or _SAMPLE_LEVEL.search(clause):
                    continue
                if _CORRELATION.search(clause) or _is_set_level(clause):
                    continue
                if _NULL_MODEL.search(clause) or _Z_AGAINST_NULL.search(clause):
                    continue
                return [GateFinding(
                    gate="cell-level-inference",
                    rule_id="biological-replicate-is-the-unit",
                    detail=(
                        "this finding reports a between-condition result whose unit of "
                        "analysis is the cell. Cells within a sample are not independent "
                        "draws, so the p-value is inflated by orders of magnitude and a "
                        "negligible difference reaches an arbitrarily small p. Aggregate "
                        "to the sample level and re-test, or report the effect size "
                        "without an error rate and say the contrast is descriptive."))]
    return []


# ============================================================ MAGNITUDE
_MAGNITUDE_THRESHOLD = re.compile(
    r'(?:'
    r'\|?\s*(?:log-?\s*2?\s*)?(?:fc|fold[-\s]?change|lfc|log-?2?fc|log2foldchange)'
    r'\s*\|?\s*(?:[><≥≤]=?|&gt;|&lt;|>=|<=|\babove\b|\bexceeding\b|\bgreater\s+than\b|\bat\s+least\b)\s*[\d.]'
    r'|\babs(?:olute)?\b[^,;:.]{0,70}?\blog-?\s*2?\b[^,;:.]{0,50}?'
    r'(?:[<>=≥≤]\s*[\d.]|\bat\s+least\b|\bexceed\w*\b|\bgreater\s+than\b|'
    r'\babove\b)'
    r'|\b(?:fold[-\s]?change|lfc|log-?2?\s*fc|effect[-\s]size|magnitude)\s+'
    r'(?:threshold|cut-?off|filter|criterion)'
    r'|\babs\s*\(\s*log'
    r'|\blfc_null\b|\balt_hypothesis\b|\bgreaterabs\b'
    r'|\bat\s+least\s+(?:a\s+)?(?:two|three|four|five|ten|[\d.]+)[-\s]?fold\b'
    r'|\b(?:two|three|four|five|ten|[\d.]+)[-\s]?fold\s+(?:threshold|cut-?off|criterion)\b'
    r')', re.I)
_ERROR_RATE_THRESHOLD = re.compile(
    r'(?:\b(?:padj|p[-_ ]?adj|adj\.?\s*p(?:[-_ ]?val(?:ue)?)?|'
    r'adjusted\s+p(?:[-_ ]?val(?:ue)?)?|fdr|q[-_ ]?val(?:ue)?|q|'
    r'bh(?:[-\s]adjusted)?\s*(?:p|fdr)?)\b\s*(?:[-_ ]?adjusted)?\s*'
    r'(?:[<>=≤≥]|\b(?:below|under|less\s+than|at\s+most)\b)\s*[\d.]'
    r'|\bbh\b[^,;:.]{0,25}(?:[<>=≤]|\b(?:below|under|less\s+than)\b)\s*[\d.]'
    r'|\bsig(?:nificant)?\s+genes?\b'
    r'|\bat\s+(?:[\d.]+\s*%?\s*)?(?:bh\s+)?(?:fdr|padj)\b)', re.I)
_COUNT_OF_FEATURES = re.compile(
    r'\b\d[\d,]*\s+(?:genes?|degs?|features?|transcripts?|probes?|proteins?|'
    r'(?:up|down)-?regulated\s+\w{0,12}\s?genes?)\b'
    r'|\b\d[\d,]*\s+(?:up|down)(?:-?regulated)?\b'
    r'|\b(?:numbered|totall?(?:ed|ing)|retain\w+|yield\w+|leav\w+|identif\w+)\s+\d[\d,]*\b', re.I)


def check_magnitude_claims(findings, *, score_definitions=()):
    """A count of features passing a fold cut needs a threshold test."""
    if _threshold_test_registered(score_definitions):
        return []
    for text in _texts(findings):
        for sent in _sentences(text):
            for clause in _clauses(sent):
                if not _MAGNITUDE_THRESHOLD.search(clause):
                    continue
                if not _ERROR_RATE_THRESHOLD.search(clause):
                    continue
                if not _COUNT_OF_FEATURES.search(clause):
                    continue
                return [GateFinding(
                    gate="magnitude-without-threshold-test",
                    rule_id="magnitude-claims-need-a-threshold-test",
                    severity="advise",
                    detail=(
                        "this finding counts features that passed both a significance "
                        "cut and a fold-change cut. The adjusted p-value tests whether "
                        "the effect is non-zero, not whether it exceeds your threshold, "
                        "so filtering the table afterwards controls the error rate of "
                        "neither the count nor the composite claim. Test against the "
                        "threshold directly (lfc_null with alt_hypothesis='greaterAbs' "
                        "in pydeseq2, or TREAT in limma/edgeR), or report the count at "
                        "the error rate alone."))]
    return []


def _threshold_test_registered(score_definitions):
    for s in score_definitions or ():
        params = getattr(s, "parameters", None) or {}
        if not isinstance(params, dict):
            continue
        try:
            if float(params.get("lfc_null", 0) or 0) != 0.0:
                return True
        except (TypeError, ValueError):
            pass
        if str(params.get("alt_hypothesis", "")).lower() in ("greaterabs", "lessabs"):
            return True
        if str(params.get("test", "")).lower() in ("treat", "glmtreat"):
            return True
    return False


# ============================================================ METHOD KIND
METHOD_KINDS = ("ranked_enrichment", "over_representation", "single_sample_enrichment",
                "module_score", "regulatory_activity", "differential_expression",
                "differential_abundance", "descriptive", "other")
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
_DA = frozenset({"sccoda", "sccomp", "propeller", "milo", "milopy", "dirichlet",
                 "betabinomial"})


def infer_method_kind(method):
    """Migration shim for records written before method_kind existed."""
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
    declared = str(getattr(score, "method_kind", "") or "").strip().lower()
    return declared if declared in METHOD_KINDS else infer_method_kind(
        getattr(score, "method", ""))


def check_declared_testing_family(score_definitions, scoring_plans):
    plans = {getattr(p, "id", ""): p for p in scoring_plans or ()}
    out = []
    for score in score_definitions or ():
        kind = score_kind(score)
        if kind not in FAMILY_REQUIRED_KINDS:
            continue
        plan = plans.get(getattr(score, "plan_id", ""))
        if plan is not None and getattr(plan, "testing_family", None):
            continue
        out.append(GateFinding(
            gate="undeclared-testing-family",
            rule_id="testing-family-must-be-declared",
            subject=getattr(score, "score_name", ""),
            detail=(
                "this score is registered as %s, whose adjusted p-value is computed "
                "jointly across every set in the call, but the testing family was "
                "never declared, so the q-value cannot be interpreted. Record every "
                "gene set that went into the same call in the plan's testing_family."
                % kind)))
    return out


# ============================================================ ORA BACKGROUND
def check_ora_background(score_definitions):
    """An over-representation score must declare the universe it tested against."""
    out = []
    for score in score_definitions or ():
        if score_kind(score) != "over_representation":
            continue
        params = getattr(score, "parameters", None) or {}
        declared = str(getattr(score, "background_source", "") or "").strip()
        if not declared and isinstance(params, dict):
            for key in ("background", "background_source", "universe", "gene_universe"):
                if params.get(key):
                    declared = str(params[key])
                    break
        if declared and not declared.lower().startswith(("default", "none", "genome")):
            continue
        out.append(GateFinding(
            gate="ora-universe-undeclared",
            rule_id="ora-universe-must-be-the-measured-genes",
            subject=getattr(score, "score_name", ""),
            detail=(
                "this over-representation score did not declare its background. The "
                "universe IS the null: with none supplied it silently becomes the union "
                "of genes in the gene sets, not the genes this assay could have "
                "detected, and every p-value in the table changes with it. Pass the "
                "measured genes -- lib = gseapy.get_library(name, organism); "
                "gseapy.enrich(genes, gene_sets=lib, background=measured_genes) -- and "
                "record it in the background_source argument of register_score, "
                "e.g. background_source='measured:adata.var_names (n=18234)'.")))
    return out


# ================================================ checks kept from the first pass
# These two never had a trigger problem: they read structured state (registered
# power, declared sample counts) rather than prose, which is why the corpus
# audit found nothing to fix in them.

def check_underpowered_negative_claims(findings, score_definitions):
    """A negative conclusion must not rest on a panel too small to detect anything.

    Only underpowered panels are considered: a well-powered null result is a
    legitimate finding and is left alone.
    """
    weak = []
    for score in score_definitions or []:
        power = getattr(score, "power", None) or {}
        if power.get("underpowered"):
            weak.append((getattr(score, "score_name", ""), power))
    if not weak:
        return []

    out = []
    for text in _texts(findings):
        if not asserts_absence(text):
            continue
        for name, power in weak:
            if name and name.lower() in str(text).lower():
                out.append(GateFinding(
                    gate="underpowered-negative-claim",
                    rule_id="no-negative-claim-from-an-underpowered-panel",
                    subject=name,
                    detail=(
                        f"this finding asserts absence, but '{name}' has only "
                        f"{power.get('effective_n')} usable gene(s) "
                        f"({power.get('basis')}), below the minimum of "
                        f"{MIN_PANEL_SIZE_FOR_NEGATIVE_CLAIM}. Report it as "
                        f"underpowered, or widen the panel and re-test."
                    ),
                ))
    return out


def check_permutation_at_small_n(findings, score_definitions, samples_per_group):
    """Permutation q-values below the viable sample size must not carry a claim.

    Scoped to small n on purpose. The rule it cites is unconditional -- a
    gene-permutation null is anti-conservative at every sample size -- but the
    unconditional half is guidance, because refusing every preranked q-value
    would refuse the standard analysis. What is refusable is the case where the
    null is not merely optimistic but arithmetically incapable of the quoted
    value: at 3 versus 3 there are 10 distinct label partitions, so nothing
    below about 0.1 is reachable.
    """
    if not samples_per_group:
        return []
    smallest = min(samples_per_group.values()) if samples_per_group else None
    if smallest is None or smallest >= MIN_SAMPLES_PER_GROUP_FOR_PERMUTATION:
        return []

    permutation_scores = [
        getattr(s, "score_name", "") for s in (score_definitions or [])
        if score_kind(s) == "ranked_enrichment"
    ]
    if not permutation_scores:
        return []

    out = []
    for text in _texts(findings):
        lowered = str(text).lower()
        if not any(token in lowered for token in ("fdr", "q-val", "q val", "qval")):
            continue
        if not any(name and name.lower() in lowered for name in permutation_scores):
            continue
        out.append(GateFinding(
            gate="permutation-at-small-n",
            rule_id="gene-permutation-is-anticonservative",
            detail=(
                f"this finding rests on a permutation q-value, but the smallest "
                f"group has {smallest} sample(s), below the "
                f"{MIN_SAMPLES_PER_GROUP_FOR_PERMUTATION} needed for the null to "
                f"be valid. Report effect size and direction concordance, and "
                f"state that the q-value is not interpretable at this sample size."
            ),
        ))
        break
    return out


def check_analysis_unit(findings, samples_per_group):
    """A single sample per condition cannot support a condition-level claim."""
    if not samples_per_group:
        return []
    singletons = [g for g, n in samples_per_group.items() if n is not None and n < 2]
    if not singletons:
        return []

    descriptive_markers = ("descriptive", "not estimable", "unidentifiable",
                           "confounded", "specimen-level", "exploratory")
    out = []
    for text in _texts(findings):
        lowered = str(text).lower()
        if not any(t in lowered for t in ("significant", "p =", "p=", "fdr",
                                          "q-val", "effect of", "causes")):
            continue
        if any(m in lowered for m in descriptive_markers):
            continue
        out.append(GateFinding(
            gate="unidentifiable-design",
            rule_id="one-sample-per-condition-is-unidentifiable",
            also_see=("biological-replicate-is-the-unit",),
            detail=(
                f"group(s) {', '.join(map(str, singletons))} have fewer than 2 "
                f"samples, so condition is perfectly confounded with sample and "
                f"the effect is not estimable at all. State such contrasts as "
                f"descriptive observations about the specimens."
            ),
        ))
        break
    return out


def run_gates(
    findings,
    *,
    score_definitions=(),
    scoring_plans=(),
    samples_per_group=None,
    cluster_labels=(),
):
    """Run every statistical gate over a report_findings payload.

    cluster_labels is accepted and deliberately ignored. Measured: feeding
    annotation vocabulary to the composition check takes it from 21 TP / 0 FP
    to 21 TP / 7 FP, because bulk findings that merely name a cell type start
    matching. Recall is already 1.000 without it, so the parameter exists only
    so callers need not know that.
    """
    report = GateReport()
    report.findings.extend(
        check_underpowered_negative_claims(findings, score_definitions))
    report.findings.extend(
        check_declared_testing_family(score_definitions, scoring_plans))
    report.findings.extend(
        check_permutation_at_small_n(findings, score_definitions, samples_per_group))
    report.findings.extend(check_analysis_unit(findings, samples_per_group))
    report.findings.extend(check_cell_level_inference(findings, samples_per_group))
    report.findings.extend(check_ora_background(score_definitions))
    report.findings.extend(check_composition_claims(findings))
    report.findings.extend(
        check_magnitude_claims(findings, score_definitions=score_definitions))
    return report
