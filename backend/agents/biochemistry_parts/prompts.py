"""Prompt bodies for the biochemistry agent.

Kept out of the agent module, which is at its size limit. These are pure text
builders with no dependency on the agent class.
"""
from __future__ import annotations


def indicator_analysis_prompt(results_block: str, instruction: str, design_note: str) -> str:
    return f"""Analyze the following biochemistry indicators in the context of the research objective.

## Biochemistry Results:
{results_block}

## Research Context / Analysis Objective:
{instruction}

{design_note}
**IMPORTANT**: Classify each indicator ONLY against a comparator that is actually present above:
1. The reference_range supplied with that indicator, if any
2. Otherwise, the values of the same indicator in a control/reference group, if group labels
   are present above
No third option exists. No literature is supplied to this step, so a reference interval
quoted from memory cannot be checked by anyone reading the report, and it is routinely wrong
for a particular strain, age, sex, fasting state, or assay platform. If a published range comes
to mind, you may mention it in `significance` as something to verify — but it does NOT set the
status, and `basis` must not claim it

What must NOT be used as a criterion: what the experiment was expected to show. Reasoning from
the expected direction of change to the conclusion that a value is high or low is circular, and
it contaminates every finding and mechanism derived from it. The research context tells you which
indicators MATTER; it never tells you which are abnormal.

When neither 1 nor 2 is available for a measurement, its status is "not_classifiable". That is
a correct and useful answer — report what would be needed to classify it. Do not guess a status
to avoid leaving the field empty.

Emit one entry per MEASUREMENT, not one per indicator name: the same indicator measured in
two groups or two replicates is two entries. Copy each measurement's bracketed id into
`measurement_id` exactly as shown — that id, not the name, is what ties your judgement back to
the right measurement, and the host restores the authoritative indicator, group and sample from
it. Collapsing two measurements into one entry loses which group a status belongs to.

Please analyze and provide:
1. The status of each measurement, with the basis you used
2. Correlations between indicators, ONLY where the number of independent samples makes a
   correlation meaningful; with few samples or one value per indicator, say that instead
   of describing a relationship the data cannot show
3. Possible pathophysiological mechanisms, where the measurements support them — return an empty
   list if they do not
4. Suggested supplementary experiments

Reply in JSON format:
{{
    "abnormal_markers": [
        {{"measurement_id": "<the [id] shown in brackets for this measurement above, verbatim>",
          "indicator": "name", "value": number,
          "group": "<the group shown with this measurement above, or null>",
          "sample_number": "<the sample/replicate shown above, or null>",
          "status": "high/low/normal/not_classifiable",
          "basis": "reference_range | control_group:<name> | none",
          "significance": "brief explanation, or what is needed to classify it"}},
        ...
    ],
    "findings": ["finding 1", "finding 2", ...],
    "mechanisms": ["possible mechanism 1", ...],
    "suggested_experiments": ["experiment 1", ...]
}}
"""
def file_extraction_prompt(filename: str, content: str) -> str:
    return f"""Analyze this biochemistry data file and extract ALL numerical values.

Filename: {filename}

Content:
{content}

For EACH measurement found, extract:
1. Indicator name exactly as the file writes it. Keep the source name; do not translate it into
   a clinical panel abbreviation or any other assumed naming scheme. This file may hold clinical
   chemistry, enzyme activities, metabolite or ion concentrations, or a custom assay readout
2. Numerical value
3. Unit, exactly as given
4. The group / condition label the file associates with the measurement, if present
5. The sample or replicate identifier, if present
6. The reference interval printed alongside the measurement, if present

Reply ONLY in JSON format:
{{
    "results": [
        {{"indicator": "<name as written in the file>", "value": 0.0, "unit": "<as written>",
          "group": "<group label or null>", "sample": "<sample id or null>",
          "reference_range": "<as printed, or null>"}},
        ...
    ],
    "notes": "Any additional observations"
}}

Extract EVERY measured numerical value. Group labels, sample identifiers and reference intervals
are DATA, not metadata to discard: without them the downstream analysis cannot compare groups or
judge whether a value is out of range, and it will fall back to guessing. Ignore only
presentation artifacts such as page numbers and report dates.
"""
