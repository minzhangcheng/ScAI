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
from .statistical_gate_vocabulary import (  # noqa: F401
    FAMILY_REQUIRED_KINDS,
    _ACTIVITY,
    _ALT_INFERENCE,
    _BOUNDED_COMPARATORS,
    _CONVENTIONAL_CUTOFFS,
    _COUNT_TEST,
    _DA,
    _DE,
    _FLOOR_ROUNDING_TOLERANCE,
    _MODULE,
    _NAMED_TEST,
    _NEGATIVE_CLAIM_PATTERNS,
    _NEG_SIG,
    _NEG_SIG_TRAILING,
    _NOT_SIG,
    _ORA,
    _P_APPROX_ZERO,
    _P_NUMERIC,
    _P_NUMERIC_VALUE,
    _RANKED,
    _RESAMPLED_RESULT,
    _RESAMPLE_PARAM_KEYS,
    _SENT_SPLIT,
    _CLAUSE_SPLIT,
    _SIG_WORD,
    _SSGSEA,
    _WORD_RE,
    _clauses,
    _DISCLAIMED,
    _finding_entries,
    _has_inference,
    _has_undisclaimed_inference,
    _has_p,
    _has_sig_word,
    _has_word,
    _mentions,
    _sentences,
    _samples_for_finding,
    _states_a_null,
    _texts,
    _tokens,
    asserts_absence,
    infer_method_kind,
    score_kind,
)
#: Below this many samples per group, a permutation-based q-value cannot be
#: trusted. Phenotype permutation degrades under it, and the number of distinct
#: label partitions becomes small enough to floor the attainable p-value: at 3
#: versus 3 there are only 10, so nothing below ~0.1 is reachable at all.
MIN_SAMPLES_PER_GROUP_FOR_PERMUTATION = 7

#: Shared membership, as a fraction of the SMALLER panel, above which two panels
#: tested together are disclosed as one line of evidence rather than two. Set at
#: a majority: below it the panels still have independent content, above it the
#: smaller one's result is mostly the larger one's result.
PANEL_REDUNDANCY_CONTAINMENT = 0.5

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


# -------------------------------------------------------------- exemptions
_COMPOSITIONAL_METHOD = re.compile(
    r'\b(?:sccoda|sccomp|dirichlet[-\s]?multinomial|dirichlet|'
    r'beta[-\s]?binomial|milo(?:py|r)?|propeller|scanpro|'
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
    """A composition claim must not rest on a per-population test of proportions.

    A registered joint model downgrades the refusal to a note; it does not
    delete it. `method` is free text the model writes itself -- register_score
    declares it as a bare string with no enum -- so honouring it as a stand-down
    would hand the model an off-switch for a blocking gate, and the exemption is
    payload-wide, so one scCODA score would also cover a claim about a
    population that model never touched. Downgrading keeps both halves honest:
    an analysis that did the work is not made to argue with the gate, and the
    simplex caveat still rides along to the report either way.
    """
    modelled = any(score_kind(s) == "differential_abundance"
                   for s in score_definitions or ())
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
                shape = (
                    "this finding attaches an inferential statistic to a "
                    "between-condition population-abundance claim. Proportions sum "
                    "to one, so an increase anywhere mechanically depresses every "
                    "other population whether or not its absolute count changed, "
                    "and a per-population test over cells uses cells as the sample "
                    "size.")
                remedy = (
                    " A joint compositional model is registered for this "
                    "analysis, so this is a note rather than a refusal: say that "
                    "the claim is that model's output, not a per-population test "
                    "computed beside it."
                    if modelled else
                    " Model the joint composition (scCODA / sccomp / propeller "
                    "/ Milo), or state the shift as descriptive.")
                return [GateFinding(
                    gate="simplex-ignored",
                    rule_id="composition-lives-on-a-simplex",
                    severity="advise" if modelled else "block",
                    detail=shape + remedy)]
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
    numeric_sizes = [
        value for value in samples_per_group.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    smallest = min(numeric_sizes) if numeric_sizes else None
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


def _declared_resamples(score) -> Optional[int]:
    """The resample count a score declared, or None when it declared none."""
    params = getattr(score, "parameters", None) or {}
    if not isinstance(params, dict):
        return None
    for key in _RESAMPLE_PARAM_KEYS:
        if key not in params:
            continue
        try:
            value = int(params[key])
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _quoted_p_values(text):
    """(comparator, value) for every p/q-like number stated in a text."""
    out = []
    for match in _P_NUMERIC_VALUE.finditer(str(text)):
        symbol, word, number = match.group(1), match.group(2), match.group(3)
        comparator = (symbol or word or "").strip().lower()
        try:
            out.append((comparator, float(number)))
        except (TypeError, ValueError):
            continue
    return out


def _names_score(text, score_name):
    """Whether a text refers to a registered score by name.

    Score names are identifiers ("hallmark_gsea") and prose is prose, so compare
    on word tokens rather than the raw string: a finding writes "the hallmark
    GSEA panel" for a score registered as "hallmark_gsea". Every token of the
    name must appear, which makes a partial coincidence ("check" alone matching
    "quick_check") insufficient.
    """
    words = set(_WORD_RE.findall(str(score_name).lower()))
    return bool(words) and words <= _tokens(text)


def _resamples_for(text, declared):
    """The resample count a text is answerable to, or None.

    A text that names registered scores is held to those; one that names none
    cannot be attributed, and the *most permissive* count is used. The direction
    is deliberate. Holding an unattributable finding to the coarsest score that
    happens to be registered refuses a correct number -- a cheap 100-draw probe
    beside a 10,000-draw run would bound the run's own p-values -- and the only
    rewording that satisfies the gate overstates the uncertainty by the ratio of
    the two counts. Observation (1) already covers the sloppy case advisorily.
    """
    named = [n for name, n in declared.items()
             if name and n and _names_score(text, name)]
    pool = named or [n for n in declared.values() if n]
    return max(pool) if pool else None


def check_resampling_resolution(findings, score_definitions):
    """A resampling p-value quoted below the resolution its draws can attain.

    Three separate observations, deliberately at different severities.

    The refusable one is narrow by construction: it fires only when the model
    itself declared how many resamples it drew AND then quoted a value at or
    under the floor those draws imply, as an equality rather than a bound. That
    is a self-contradiction in the model's own record, not a judgement about the
    analysis -- and the fix is a rewording it can perform without recomputing
    anything.

    The other two only advise. Absence of a declared count is left advisory
    because the method kind is inferred by matching tokens in a free-text field,
    and a mis-tokenised method must not cost a rejection. A shared value across
    findings is advisory because a coincidence, however unlikely, is not an
    error.
    """
    texts = _texts(findings)
    if not texts:
        return []

    out = []
    ranked = [s for s in (score_definitions or ()) if score_kind(s) == "ranked_enrichment"]
    declared = {
        getattr(s, "score_name", ""): _declared_resamples(s) for s in ranked
    }
    counts = [n for n in declared.values() if n]

    # (1) A resampling result whose resolution is simply not on record.
    undeclared = [name for name, n in declared.items() if name and not n]
    if undeclared:
        out.append(GateFinding(
            gate="resample-count-undeclared",
            rule_id="resampling-p-has-a-resolution-floor",
            severity="advise",
            subject=", ".join(sorted(undeclared)[:4]),
            detail=(
                "this score is a resampling-based enrichment, but no resample count "
                "is on record for it, so the smallest p-value it could have produced "
                "is unknown and a reader cannot tell an estimate from a bound. "
                f"Re-register it with the count in parameters (any of: "
                f"{', '.join(_RESAMPLE_PARAM_KEYS[:4])})."),
        ))

    # (2) A floor value quoted as though it were an estimate.
    #
    # Scoped to the sentence rather than the whole finding, because the ordinary
    # shape of a finding reports a resampling result and an analytic one side by
    # side ("... pydeseq2 ... adjusted p = 1e-8. GSEA on the same contrast ...").
    # A model-fit p-value has no resolution floor, and holding one to the
    # enrichment's floor refuses a correct number.
    if counts:
        hit = None
        for text in texts:
            for sent in _sentences(text):
                if not _RESAMPLED_RESULT.search(sent):
                    continue
                resamples = _resamples_for(sent, declared)
                if not resamples:
                    continue
                floor = 1.0 / (resamples + 1)
                for comparator, value in _quoted_p_values(sent):
                    if comparator in _BOUNDED_COMPARATORS:
                        continue
                    if value <= 0 or value > floor * _FLOOR_ROUNDING_TOLERANCE:
                        continue
                    if any(abs(value - c) < 1e-12 for c in _CONVENTIONAL_CUTOFFS):
                        # A conventional cutoff restated is a threshold, not an
                        # estimate -- the same carve-out observation (3) makes.
                        # gseapy's default of 1000 draws puts the floor within
                        # rounding distance of 0.001, so without this the gate
                        # refuses the most standard sentence in the method.
                        continue
                    hit = (value, resamples, floor)
                    break
                if hit:
                    break
            if hit:
                break
        if hit:
            value, resamples, floor = hit
            out.append(GateFinding(
                gate="p-below-resampling-resolution",
                rule_id="resampling-p-has-a-resolution-floor",
                also_see=("gene-permutation-is-anticonservative",),
                detail=(
                    f"this finding states a resampling p-value of {value:g} as an "
                    f"estimate, but the smallest value {resamples} resamples can "
                    f"produce is {floor:.3g}. Every result at that value is tied "
                    f"with every other one there, so the number carries no ordering "
                    f"and no precision. State it as an inequality (p < "
                    f"{floor:.3g}), rank on the effect statistic instead, or "
                    f"re-run with more resamples."),
            ))

    # (3) One value shared across several findings is the floor showing through.
    repeated = {}
    for text in texts:
        for _, value in _quoted_p_values(text):
            if value <= 0 or any(abs(value - c) < 1e-12 for c in _CONVENTIONAL_CUTOFFS):
                continue
            repeated.setdefault(value, 0)
            repeated[value] += 1
    tied = sorted(v for v, n in repeated.items() if n >= 3)
    if tied:
        out.append(GateFinding(
            gate="adjusted-p-tied-across-results",
            rule_id="resampling-p-has-a-resolution-floor",
            severity="advise",
            detail=(
                f"the same p/q-value ({', '.join(f'{v:g}' for v in tied[:3])}) is quoted "
                f"for three or more separate results. Independent tests do not normally "
                f"agree to full precision; a shared value is usually a resolution floor, "
                f"which multiplicity adjustment preserves. Say which results are tied and "
                f"do not rank them against each other on that value."),
        ))
    return out


def check_panel_redundancy(score_definitions, scoring_plans):
    """Panels tested together whose membership is largely the same panel.

    Advisory, never blocking, and the distinction matters. Overlap is not an
    error: a broad set and a focused component of it are both legitimate tests,
    and partitions of one parent set are subsets of each other by construction.
    Refusing here would be the platform overruling a design decision it is not
    entitled to make. What the platform IS entitled to do is refuse to let the
    overlap go unstated, because that is what turns one observation into an
    apparent agreement between two.

    Containment rather than Jaccard: a small panel wholly inside a large one is
    entirely redundant with it, yet scores low on Jaccard precisely because the
    large panel has other members.
    """
    from .analysis_plan import panel_overlap

    panels = []
    for score in (score_definitions or ()):
        name = str(getattr(score, "score_name", "") or "")
        genes = list(getattr(score, "gene_list", None) or [])
        if name and len(genes) >= MIN_PANEL_SIZE_FOR_NEGATIVE_CLAIM:
            panels.append((name, genes))
    if len(panels) < 2:
        return []

    seen = set()
    pairs = []
    for i, (name_a, genes_a) in enumerate(panels):
        for name_b, genes_b in panels[i + 1:]:
            if name_a == name_b:
                continue
            key = tuple(sorted((name_a, name_b)))
            if key in seen:
                continue
            overlap = panel_overlap(genes_a, genes_b)
            if overlap["containment"] < PANEL_REDUNDANCY_CONTAINMENT:
                continue
            seen.add(key)
            pairs.append((key, overlap))
    if not pairs:
        return []

    pairs.sort(key=lambda item: -item[1]["containment"])
    lines = [
        f"{a} and {b} share {o['n_shared']} of {min(o['n_a'], o['n_b'])} members "
        f"(containment {o['containment']:.2f})"
        for (a, b), o in pairs[:4]
    ]
    return [GateFinding(
        gate="overlapping-panels",
        rule_id="overlapping-sets-are-not-independent-evidence",
        severity="advise",
        detail=(
            "these panels were tested in the same analysis and are largely the same "
            "panel: " + "; ".join(lines) + ". Their statistics are not independent, so "
            "adjusting across them as separate tests understates the error rate, and "
            "reporting both results side by side reads as convergence where there is "
            "one observation. Name one as the test and the other as a sensitivity "
            "analysis, or show that the members they do not share carry the result."),
    )]


def check_analysis_unit(findings, samples_per_group):
    """A single sample per condition cannot support a condition-level claim.

    Advisory, not a refusal. ``run_gates`` selects the counts for the finding's
    modality; a direct caller is responsible for doing the same.
    """
    if not samples_per_group:
        return []
    singletons = [(g, n) for g, n in samples_per_group.items()
                  if n is not None and n < 2]
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
        counts = ", ".join(f"{g}={n}" for g, n in singletons)
        out.append(GateFinding(
            gate="unidentifiable-design",
            rule_id="one-sample-per-condition-is-unidentifiable",
            severity="advise",
            also_see=("biological-replicate-is-the-unit",),
            detail=(
                f"the replicate counts for this finding's analysis modality are "
                f"{counts}. With one independent unit in a condition, condition "
                f"is confounded with specimen and a condition effect is not "
                f"estimable -- state that contrast as a descriptive observation "
                f"about the specimens."
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
    samples_per_group_by_modality=None,
    cluster_labels=(),
):
    """Run every statistical gate over a report_findings payload.

    cluster_labels is accepted and deliberately ignored. Measured: feeding
    annotation vocabulary to the composition check takes it from 21 TP / 0 FP
    to 21 TP / 7 FP, because bulk findings that merely name a cell type start
    matching. Recall is already 1.000 without it, so the parameter exists only
    so callers need not know that. Those counts were measured while the
    composition check was still being called without score_definitions, so they
    describe its inputs rather than the operating point that ships now: a
    registered joint model downgrades that check to an advisory.
    """
    report = GateReport()
    report.findings.extend(
        check_underpowered_negative_claims(findings, score_definitions))
    report.findings.extend(
        check_declared_testing_family(score_definitions, scoring_plans))
    if samples_per_group_by_modality:
        for text, finding_modality in _finding_entries(findings):
            selected_modality, modality_samples = _samples_for_finding(
                finding_modality, samples_per_group_by_modality,
            )
            if selected_modality == "cross_modal":
                designs = list(samples_per_group_by_modality.values())
                lacks_replication = any(
                    not counts or min(counts.values()) < 2 for counts in designs
                )
                if lacks_replication and _has_undisclaimed_inference(str(text)):
                    report.findings.append(GateFinding(
                        gate="cross-modal-inference-unreplicated",
                        rule_id="biological-replicate-is-the-unit",
                        detail=(
                            "cross-modal agreement cannot restore independent "
                            "replication missing from either contributing modality. "
                            "Report the concordance descriptively and do not call it "
                            "significant or independent validation."
                        ),
                    ))
                continue
            if modality_samples == {}:
                if _has_inference(str(text)) or "effect of" in str(text).casefold():
                    report.findings.append(GateFinding(
                        gate="sample-design-unresolved",
                        rule_id="biological-replicate-is-the-unit",
                        detail=(
                            "the finding identifies its sequencing modality, but "
                            "independent-unit counts could not be assigned to treatment "
                            "groups. Verify the sample metadata before reporting an "
                            "inferential condition claim."
                        ),
                    ))
                continue
            if modality_samples is None:
                if (len(samples_per_group_by_modality) > 1
                        and (_has_inference(str(text))
                             or "effect of" in str(text).casefold())):
                    report.findings.append(GateFinding(
                        gate="modality-unresolved",
                        rule_id="biological-replicate-is-the-unit",
                        detail=(
                            "this inferential finding does not identify which sequencing "
                            "modality supports it. The session's modalities have different "
                            "replicate counts, so name the source as Bulk RNA-seq, scRNA-seq, "
                            "or Cross-modal before reporting the claim."
                        ),
                    ))
                continue
            unreplicated_single_cell = (
                selected_modality == "single_cell" and bool(modality_samples)
                and min(modality_samples.values()) < 2
            )
            if unreplicated_single_cell and _has_undisclaimed_inference(str(text)):
                report.findings.append(GateFinding(
                    gate="unreplicated-single-cell-inference",
                    rule_id="biological-replicate-is-the-unit",
                    detail=(
                        "the single-cell arm has fewer than two independent biological "
                        "units in at least one condition. Neither cell-level testing nor "
                        "one-pseudobulk-per-condition can estimate a condition effect. "
                        "Report the single-cell contrast descriptively, or base "
                        "condition-level inference on a separately replicated modality."
                    ),
                ))
            report.findings.extend(check_permutation_at_small_n(
                [text], score_definitions, modality_samples,
            ))
            report.findings.extend(check_analysis_unit([text], modality_samples))
            if not unreplicated_single_cell:
                report.findings.extend(check_cell_level_inference(
                    [text], modality_samples,
                ))
    else:
        report.findings.extend(check_permutation_at_small_n(
            findings, score_definitions, samples_per_group,
        ))
    report.findings.extend(
        check_resampling_resolution(findings, score_definitions))
    report.findings.extend(
        check_panel_redundancy(score_definitions, scoring_plans))
    if not samples_per_group_by_modality:
        report.findings.extend(check_analysis_unit(findings, samples_per_group))
        report.findings.extend(check_cell_level_inference(findings, samples_per_group))
    report.findings.extend(check_ora_background(score_definitions))
    report.findings.extend(
        check_composition_claims(findings, score_definitions=score_definitions))
    report.findings.extend(
        check_magnitude_claims(findings, score_definitions=score_definitions))
    return report
