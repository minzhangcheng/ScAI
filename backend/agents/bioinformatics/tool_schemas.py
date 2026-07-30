# OpenAI tool schemas exposed to the bioinformatics agent loop.
# Moved verbatim out of backend/agents/bioinformatics_agent.py.
"""
Bioinformatics Agent - Handles snRNA-seq analysis with iterative code execution
Supports autonomous technical steps and PI consultation for biological decisions
With step-level checkpointing for recovery from interruptions
"""
import logging


from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY

# Configure logger
logger = logging.getLogger(__name__)


# Define the tools for bioinformatics agent
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Generate and execute Python code for single-cell analysis. Use for technical steps like data loading, QC, normalization, clustering, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute"
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of what this code does"
                    },
                    "step_type": {
                        "type": "string",
                        "enum": ["technical", "biological"],
                        "description": "Whether this is a technical step (can proceed autonomously) or requires biological interpretation"
                    },
                    "persistent_outputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "REQUIRED when this cell creates variables intended for later "
                            "cells or snapshot restore (e.g., adata, results, markers, "
                            "model). List every user-global name that subsequent cells will "
                            "read. Empty array is allowed only when the cell creates no "
                            "values for later reuse. Dynamic namespace reads depend on "
                            "this declaration."
                        ),
                        "default": [],
                    }
                },
                "required": ["code", "description", "step_type", "persistent_outputs"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consult_pi",
            "description": "Request guidance from PI Agent on biological decisions like: analysis direction, which genes/pathways to analyze, whether to do trajectory analysis, result interpretation. Do NOT use for cell type annotation - use annotate_clusters instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The biological question or decision that needs PI guidance"
                    },
                    "context": {
                        "type": "string",
                        "description": "Relevant context from current analysis (e.g., cluster markers, current findings)"
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Possible options or choices if applicable"
                    }
                },
                "required": ["question", "context"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_literature",
            "description": "Search literature for biological information like marker genes, cell type signatures, pathway information",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to search for in literature"
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_findings",
            "description": "Report key findings and conclusions. Call this ONLY when you have completed ALL assigned tasks. In the exploration phase, you MUST complete the full pipeline including annotate_clusters BEFORE calling this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of key findings"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Summary of the analysis results"
                    },
                    "suggestions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Suggestions for next steps or further analysis"
                    },
                    "evidence_execution_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "IDs printed by successful execute_code calls that directly "
                            "support the reported findings. Failed executions and prior "
                            "rounds must not be cited."
                        )
                    }
                },
                "required": ["findings", "summary", "evidence_execution_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_publication_figure",
            "description": "Generate a publication-quality figure with PNG thumbnail and high-res PDF. The figure will be automatically reviewed by the PI Agent. Use this for important figures that should be included in the final report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python plotting code using matplotlib, seaborn, or scanpy. Should produce a single figure."
                    },
                    "figure_name": {
                        "type": "string",
                        "description": "Name for the figure file (English, no spaces, e.g., 'umap_cell_types')"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what the figure shows (will be used for figure legend)"
                    },
                    "figure_type": {
                        "type": "string",
                        "enum": ["umap", "heatmap", "violin", "dotplot", "barplot", "trajectory", "volcano", "other"],
                        "description": "Type of figure being generated"
                    }
                },
                "required": ["code", "figure_name", "description", "figure_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "annotate_clusters",
            "description": "Annotate all cell clusters at once using literature-based evidence. Call this ONCE after clustering and rank_genes_groups. Returns cell type annotations with confidence levels and literature evidence. Much more efficient than per-cluster consult_pi calls. You do NOT need to pass cluster_markers — they will be auto-extracted from adata.uns['rank_genes_groups'] if omitted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_markers": {
                        "type": "object",
                        "description": "Optional. Dict mapping each cluster ID to its observed top marker genes. Example: {\"0\": [\"MarkerA\", \"MarkerB\"], \"1\": [\"MarkerC\", \"MarkerD\"]}. If omitted, markers will be auto-extracted from adata.uns['rank_genes_groups'].",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "tissue": {
                        "type": "string",
                        "description": "Tissue or organ type. Extract it from the research context or data metadata; do not infer it from built-in examples."
                    },
                    "organism": {
                        "type": "string",
                        "description": "Organism (e.g., 'mouse', 'human'). Extract from the research context or data metadata."
                    },
                    "cluster_key": {
                        "type": "string",
                        "description": "Optional. Name of the adata.obs column holding cluster assignments (e.g., 'leiden_broad'). If omitted, the backend will try 'leiden', 'louvain', 'cluster' and any column starting with 'leiden'/'louvain'."
                    },
                    "rank_key": {
                        "type": "string",
                        "description": "Optional. Name of the adata.uns key holding rank_genes_groups results (e.g., 'rank_genes_broad'). If omitted, defaults to 'rank_genes_groups'."
                    }
                },
                "required": ["tissue", "organism"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_gene_set",
            "description": (
                "Fetch a named gene set from a versioned external resource, for the "
                "organism under analysis. Use this INSTEAD of writing a gene panel from "
                "memory: a fetched set is citable, re-fetchable, and identical across "
                "runs, whereas a remembered one is none of those. Sandbox code cannot "
                "reach the network, so this tool does the fetch and returns the genes. "
                "The resource publishes native collections for two organisms; for any "
                "other organism the set is projected through orthology and the mapping "
                "rate is returned with it. Note that a curated set is not automatically "
                "the right panel: many are broad and mix genes that drive a process with "
                "genes that suppress it. Read the genes before scoring them, and use "
                "register_analysis_plan's partition operation when a set needs splitting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "set_id": {
                        "type": "string",
                        "description": "Exact identifier of the gene set within the collection."
                    },
                    "collection": {
                        "type": "string",
                        "description": (
                            "Collection identifier within the resource. These are "
                            "ORGANISM-SPECIFIC and not guessable -- call with "
                            "list_collections=true FIRST to see what this organism and "
                            "release actually publish, then pick one. Passing another "
                            "organism's identifier fails."
                        )
                    },
                    "version": {
                        "type": "string",
                        "description": (
                            "Release identifier of the resource, e.g. '2025.1'. Pinning "
                            "it is what makes the panel re-fetchable later. The bare "
                            "release and the organism-stamped form the resource "
                            "publishes are both accepted."
                        )
                    },
                    "partition_by_direction": {
                        "type": "boolean",
                        "description": (
                            "Split the fetched set into direction-coherent parts instead "
                            "of returning it whole. An ontology process term contains the "
                            "genes that drive the process AND the genes that restrain it, "
                            "so a single module score over it rises whether the process is "
                            "running or being held in check -- the number looks like "
                            "evidence while carrying almost none. This uses the ontology's "
                            "own regulation terms to separate them, returning up to four "
                            "panels: positive, negative, ambiguous (annotated to both "
                            "directions, so unable to support a directional claim) and "
                            "core (the genes that carry out the process rather than tuning "
                            "it). Each part is a panel in its own right with its own "
                            "provenance. Only the biological-process branch of the ontology "
                            "publishes the regulation terms this needs; other collections "
                            "return a clear refusal."
                        )
                    },
                    "list_collections": {
                        "type": "boolean",
                        "description": (
                            "Return the collection identifiers available for this "
                            "organism and release, instead of fetching a set. Start "
                            "here when you do not already know the identifier -- "
                            "guessing produces a 404, not a useful suggestion."
                        )
                    },
                    "organism": {
                        "type": "string",
                        "description": (
                            "Organism for the returned symbols. Defaults to the organism "
                            "declared for this session."
                        )
                    },
                    "list_sets": {
                        "type": "boolean",
                        "description": (
                            "Return matching set identifiers instead of genes. Use with "
                            "a partial set_id to discover what the collection contains."
                        )
                    }
                },
                "required": ["version"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "register_analysis_plan",
            "description": (
                "Freeze a scoring plan BEFORE computing it. Call this first for any "
                "gene-panel score, then compute, then call register_score citing the "
                "returned plan_id. Freezing the panel in advance is what separates a "
                "curation judgement from selecting the panel that gave the answer you "
                "wanted -- once a result is visible, any change to the panel is "
                "recorded as a departure from the plan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "score_name": {
                        "type": "string",
                        "description": "Name of the score this plan governs; must match the later register_score call."
                    },
                    "method": {
                        "type": "string",
                        "description": "Computation you intend to use (e.g. 'sc.tl.score_genes', 'gseapy.prerank')."
                    },
                    "hypothesis": {
                        "type": "string",
                        "description": "What this score is meant to test, in one sentence."
                    },
                    "falsifier": {
                        "type": "string",
                        "description": (
                            "What result would argue AGAINST the hypothesis. Stating this "
                            "before seeing the result is what makes the test falsifiable."
                        )
                    },
                    "panel_genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The exact gene panel you will score. This is what gets frozen."
                    },
                    "panel_source": {
                        "type": "string",
                        "description": "Provenance of the panel, same controlled vocabulary as register_score.gene_list_source."
                    },
                    "baseline_genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "The unedited set from the versioned resource, when the panel was "
                            "derived from one. Supplying it lets the report show curated and "
                            "original side by side rather than only the edited version."
                        )
                    },
                    "baseline_source": {
                        "type": "string",
                        "description": "Provenance of baseline_genes, same controlled vocabulary."
                    },
                    "edit_operations": {
                        "type": "array",
                        "description": (
                            "Typed edits applied to the baseline. Each is "
                            "{operation, genes, rationale}. Allowed operations: partition "
                            "(split a mixed-direction set into direction-coherent subsets), "
                            "union, intersect, restrict_to_measured, remove, add_bespoke. "
                            "remove and add_bespoke are high-risk and require a rationale of "
                            "at least 20 characters."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "operation": {"type": "string"},
                                "genes": {"type": "array", "items": {"type": "string"}},
                                "rationale": {"type": "string"}
                            },
                            "required": ["operation"]
                        }
                    },
                    "testing_family": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Names of ALL gene sets you intend to pass to the same enrichment "
                            "call / multiple-testing correction as this one. A GSEA q-value is "
                            "computed across every set in the call, so the same panel with the "
                            "same score can come out significant or not depending only on what "
                            "else was in the family. Declare it so the result can be read."
                        )
                    }
                },
                "required": ["score_name", "method", "panel_genes", "panel_source"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "register_score",
            "description": "Register how a score was computed for the final report methodology section. Call this IMMEDIATELY AFTER computing any score (gene signature scores, pathway enrichment, TF activity, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "score_name": {
                        "type": "string",
                        "description": "Name of the score as stored in adata (e.g., 'response_score', 'custom_signature')"
                    },
                    "method": {
                        "type": "string",
                        "description": "Computation method (e.g., 'sc.tl.score_genes', 'gseapy.ssgsea', 'dc.run_ulm')"
                    },
                    "description": {
                        "type": "string",
                        "description": "What this score measures biologically (1-2 sentences)"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Key parameters used (e.g., {\"ctrl_size\": 50, \"n_bins\": 25})"
                    },
                    "gene_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Gene names used for the score (if applicable, top 50)"
                    },
                    "gene_list_source": {
                        "type": "string",
                        "description": (
                            "REQUIRED. Where the gene panel came from, in one of four "
                            "machine-checkable forms. "
                            "(1) database:<resource>:<version>:<set_id> for a named set from a "
                            "versioned resource - the version must be a concrete release, not "
                            "'latest'. Prefer this whenever a curated set exists. "
                            "(2) derived:<method>[:<detail>] when computed from this dataset. "
                            "(3) literature:PMID:<id> or literature:DOI:<id>. "
                            "(4) bespoke:<justification of at least 20 characters> when the "
                            "panel was assembled by hand - allowed, but say why no versioned "
                            "set fits, because hand-assembled panels are reported as such and "
                            "weighed accordingly."
                        )
                    },
                    "plan_id": {
                        "type": "string",
                        "description": (
                            "REQUIRED when a gene panel is used: the plan_id returned by "
                            "register_analysis_plan before this score was computed. The panel "
                            "you register here is re-hashed against the frozen one; if it has "
                            "changed, the difference is recorded on the score."
                        )
                    },
                    "n_genes_matched": {
                        "type": "integer",
                        "description": (
                            "How many genes of gene_list are actually present in the analysed "
                            "matrix, e.g. len([g for g in gene_list if g in adata.var_names]). "
                            "This, not the nominal panel size, governs statistical power: a "
                            "panel whose genes are mostly absent from the data cannot support "
                            "a negative conclusion. Supply it whenever a gene panel is used."
                        )
                    },
                    "background_source": {
                        "type": "string",
                        "description": (
                            "REQUIRED for over-representation tests (enrichr, Fisher, "
                            "hypergeometric): the gene universe the test was run "
                            "against. The universe IS the null -- with none supplied it "
                            "silently becomes the union of genes in the gene sets rather "
                            "than the genes this assay could have detected, and every "
                            "p-value changes with it. State what you actually passed, "
                            "e.g. 'measured:adata.var_names (n=18234)' or "
                            "'measured:genes surviving DE filtering (n=9120)'. "
                            "'default', 'none' and 'genome' are not accepted."
                        )
                    }
                },
                "required": ["score_name", "method", "description", "gene_list_source"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_documentation",
            "description": "Look up API documentation for an installed Python function/class. Returns a concise summary with signature, key parameters, and usage example. Use when unsure about correct API usage before writing code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "function_path": {
                        "type": "string",
                        "description": "Full dotted path to the function or module (e.g., 'scanpy.tl.umap', 'decoupler.get_collectri', 'gseapy.enrichr')"
                    },
                    "method_question": {
                        "type": "string",
                        "description": (
                            "A methodological question rather than an API one -- how to "
                            "choose a test, what a statistic assumes, whether a claim is "
                            "supportable by a given design. Searches the project's rule "
                            "corpus, which carries the reasoning and the primary "
                            "literature. Ask here BEFORE choosing a statistical approach; "
                            "the corpus records where the usual shortcuts fail. May be "
                            "given with or instead of function_path."
                        )
                    },
                    "question": {
                        "type": "string",
                        "description": "Optional: specific question about the API (e.g., 'what parameters control batch correction?')"
                    }
                },
                "required": ["function_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_species_genes",
            "description": "Validate whether gene names in the loaded data match the expected species. Call after loading data, before clustering. Randomly samples gene names from the dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ~50 randomly sampled gene names from adata.var_names"
                    }
                },
                "required": ["gene_names"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "register_mechanism_programs",
            "description": (
                "Register data-driven mechanism programs after completing downstream "
                "analysis, BEFORE report_findings. A mechanism program is a recurring, "
                "cell-type-localized or cross-validated pattern supported by observed "
                "genes, pathways, module scores, enrichment terms, bulk/scRNA overlap, "
                "or cell-state changes. "
                + MECHANISM_EVIDENCE_POLICY
                + " If no "
                "supportable program exists, pass an empty programs list and explain "
                "why in no_program_reason."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "programs": {
                        "type": "array",
                        "description": "0-8 data-driven mechanism programs.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "Short descriptive label derived from observed data."},
                                "description": {"type": "string"},
                                "cell_types": {"type": "array", "items": {"type": "string"}},
                                "conditions_compared": {"type": "string"},
                                "triggering_observations": {"type": "array", "items": {"type": "string"}, "description": "Concrete observations that triggered this program (required, non-empty)."},
                                "supporting_genes": {"type": "array", "items": {"type": "string"}},
                                "supporting_pathways": {"type": "array", "items": {"type": "string"}},
                                "supporting_scores": {"type": "array", "items": {"type": "string"}},
                                "supporting_tables": {"type": "array", "items": {"type": "string"}},
                                "evidence_execution_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Successful current-run execute_code IDs that support this program.",
                                },
                                "bulk_support": {"type": "string", "enum": ["supported", "not_supported", "mixed", "not_available", "unknown"]},
                                "evidence_strength": {"type": "string", "enum": ["strong", "moderate", "weak"]},
                                "specificity": {"type": "string", "enum": ["high", "medium", "low"]},
                                "alternative_explanations": {"type": "array", "items": {"type": "string"}},
                                "missing_evidence": {"type": "array", "items": {"type": "string"}},
                                "interpretation": {"type": "string", "description": "Cautious interpretation; do not claim confirmation."},
                                "downstream_literature_queries": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["label", "triggering_observations", "evidence_execution_ids", "evidence_strength", "specificity", "interpretation"],
                        },
                    },
                    "no_program_reason": {"type": "string", "description": "Required when programs is empty: why no data-driven mechanism program was registered."},
                },
                "required": ["programs"],
            },
        },
    },
]

# Derived from TOOLS to keep the dispatch chain in
# run_autonomous_analysis in sync — see the else branch there for the
# orphan-tool-call defense.
KNOWN_TOOL_NAMES: frozenset[str] = frozenset(
    t["function"]["name"] for t in TOOLS
)
