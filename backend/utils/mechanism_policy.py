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
    "causal mechanism execution."
)
