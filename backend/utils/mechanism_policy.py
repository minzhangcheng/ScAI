"""Domain-neutral evidence policy for biological mechanism discovery."""

# Recall first, caution second. The caution clause is the one a model reaches
# for under uncertainty, so leading with it yielded agents that retreated to a
# generic description of a mechanism their own results had already named.
MECHANISM_EVIDENCE_POLICY = (
    "When observed genes, pathways, enrichment results, measurements, morphology, "
    "or perturbation responses match an established biological mechanism, name it "
    "explicitly and grade the supporting and contradictory evidence; do not fall back "
    "on a generic description of something the data already identify. Treat mechanisms "
    "explicitly named in the research objective or current direction as prespecified "
    "hypotheses: evaluate each as supported, contradicted, or insufficiently supported "
    "without treating it as established by default, and never drop one silently — a "
    "mechanism you cannot resolve is reported as unresolved, not omitted. Begin with "
    "unbiased analyses and do not assume or privilege a named mechanism without "
    "supporting evidence. Distinguish pathway activity, susceptibility, or priming from "
    "causal mechanism execution. "
    # An analysis whose instruments cannot reach the execution level reported
    # "not demonstrated" round after round, which reads as a statement about the
    # biology when it is a statement about the assay.
    "State the strongest class of evidence the available modalities can reach, and "
    "separate a mechanism that is not demonstrated from one that no measurement in "
    "this study could demonstrate — the second is a limit of the assay, not a finding "
    "about the biology. When that limit is reached and independent modalities converge, "
    "rank the remaining candidates and name the most probable one as an explicit "
    "inference carrying its residual uncertainty, rather than reporting the question as "
    "unresolved. "
    # Both directions, and the escape hatch closed: "that observation is only
    # compensation" would otherwise explain away every inconvenient result.
    "Regulated systems answer a perturbation by inducing the opposing process, so an "
    "observation running counter to a hypothesised mechanism may be that mechanism's "
    "counter-regulation rather than its absence, and an induced compensatory program may "
    "be a response to a pressure rather than proof that the pressure prevailed. Weigh "
    "both directions together before concluding, state which reading the whole pattern "
    "favours, and name the measurement that would separate them. The counter-regulatory "
    "reading must itself be supported by the data — it is not available whenever an "
    "observation disagrees with the hypothesis. "
    # Depletion is data. It is deliberately not described in survival terms: a
    # population missing from the capture has an open set of causes, and this is
    # a general-purpose platform.
    "When a population is depleted or absent from the captured data, treat that absence "
    "as an observation to be explained rather than a gap to pass over. Reason about what "
    "the missing population would have shown, and hold its causes open: a change of state "
    "or identity, loss during dissociation or capture, movement out of the sampled "
    "compartment, and elimination are all candidates; say which the remaining data support "
    "and which they cannot separate. Measurements from the populations that were captured "
    "describe the states that were retained, not the states that are missing."
)
