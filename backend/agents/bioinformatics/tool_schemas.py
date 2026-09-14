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
                            "REQUIRED when this cell creates variables that later cells "
                            "will read (e.g., adata, results, markers, model). Variable "
                            "names only -- never a filename or a path. List every "
                            "user-global name subsequent cells will use. Entries that "
                            "are not valid Python identifiers are ignored. Empty array "
                            "is allowed only when the cell creates no values for later "
                            "reuse. Every name listed must exist when the cell finishes, "
                            "or the cell fails. The declaration is a record of what this "
                            "cell produced; it does not decide what is saved."
                        ),
                        "default": [],
                    },
                    "plan_step_ids": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "description": (
                            "Current-plan step IDs this cell may support. The host records "
                            "these as technical candidate links; the PI decides whether "
                            "the step is scientifically complete. Omit when uncertain."
                        ),
                        "default": [],
                    },
                    "execution_purpose": {
                        "type": "string",
                        "enum": [
                            "planned", "supplementary", "repair", "unspecified"
                        ],
                        "description": (
                            "Producer intent only: planned for approved work, repair for a "
                            "correction, supplementary for out-of-plan exploration, or "
                            "unspecified when the producer does not nominate an intent."
                        ),
                        "default": "unspecified",
                    },
                    "input_profile_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Host BulkInputProfile IDs directly read by this cell. "
                            "Use an empty list when the cell does not consume bulk input."
                        ),
                        "default": [],
                    }
                },
                "required": [
                    "code", "description", "step_type"
                ]
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
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "The finding or conclusion.",
                                },
                                "modality": {
                                    "type": "string",
                                    "enum": [
                                        "single_cell", "bulk_rna_seq",
                                        "cross_modal",
                                    ],
                                    "description": (
                                        "The analysis modality that directly supports "
                                        "this finding. Use cross_modal only for a "
                                        "concordance/integration result, not to borrow "
                                        "replication from another modality."
                                    ),
                                },
                                "evidence_execution_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "Candidate current-run execute_code IDs that may "
                                        "support this finding. The host validates objective "
                                        "identity and records missing or invalid links as "
                                        "missing current execution linkage without rejecting "
                                        "the report."
                                    ),
                                    "default": [],
                                },
                            },
                            "required": ["text"],
                        },
                        "description": (
                            "List of key findings with a structured evidence modality "
                            "so the correct replicate design is applied."
                        )
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
                            "Candidate IDs printed by successful execute_code calls. The "
                            "host keeps only objectively current IDs; uncertain links are "
                            "retained for PI review rather than blocking completion."
                        ),
                        "default": [],
                    }
                },
                "required": ["findings", "summary"]
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
                    "logical_figure_id": {
                        "type": "string",
                        "description": (
                            "Stable semantic identity for this figure across redraws. "
                            "Reuse the exact same value when revising or regenerating "
                            "the same scientific figure; change it only for a distinct "
                            "figure. Filename suffixes are not revision metadata."
                        )
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what the figure shows (will be used for figure legend)"
                    },
                    "figure_type": {
                        "type": "string",
                        "description": (
                            "Type of figure being generated. Use the exact figure_type "
                            "declared by the approved plan requirement (for example umap, "
                            "heatmap, violin, dotplot, barplot, trajectory, volcano, "
                            "network, venn, scatter, qc, or another accurate open-vocabulary "
                            "label)."
                        )
                    },
                    "plan_step_id": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "ID of the approved analysis-plan step whose visual "
                            "deliverable this figure satisfies. This is required when "
                            "fulfilling a planned figure requirement; omit it only for "
                            "a genuinely out-of-plan, user-requested supplementary "
                            "figure. Use the most directly related approved step for an "
                            "additional supporting figure."
                        )
                    },
                    "evidence_execution_id": {
                        "type": "string",
                        "description": (
                            "Candidate current-run execute_code receipt that produced "
                            "or directly supports the data shown. If omitted or invalid, "
                            "the figure can still be reviewed as supplementary but cannot "
                            "receive plan credit."
                        ),
                    }
                },
                "required": [
                    "code",
                    "figure_name",
                    "description",
                    "figure_type",
                ]
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
                "rate is returned with it. For a native collection, list_sets=true also "
                "binds the complete versioned GMT in the sandbox; pass that path variable "
                "directly as gene_sets= to gseapy.prerank or enrichr. For a projected "
                "organism the source-organism GMT is not bound; fetch the named set by "
                "set_id so the projection and mapping rate are recorded. Note that a curated set is not automatically "
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
                            "a partial set_id to discover what the collection contains. "
                            "For native collections, the full GMT is bound for direct "
                            "gene_sets= use; projected source-organism collections are "
                            "not bound in the sandbox."
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
                            "Where the gene panel came from, preferably in one of four "
                            "machine-checkable forms. "
                            "(1) database:<resource>:<version>:<set_id> for a named set from a "
                            "versioned resource - the version must be a concrete release, not "
                            "'latest'. Prefer this whenever a curated set exists. "
                            "(2) derived:<method>[:<detail>] when computed from this dataset. "
                            "(3) literature:PMID:<id> or literature:DOI:<id>. "
                            "(4) bespoke:<justification of at least 20 characters> when the "
                            "panel was assembled by hand - allowed, but say why no versioned "
                            "set fits, because hand-assembled panels are reported as such and "
                            "weighed accordingly. Missing or invalid provenance is recorded "
                            "as a non-blocking PI review observation."
                        )
                    },
                    "plan_id": {
                        "type": "string",
                        "description": (
                            "When a gene panel is used, cite the plan_id returned by "
                            "register_analysis_plan before this score was computed. The panel "
                            "you register here is re-hashed against the frozen one; if it has "
                            "changed, the difference is recorded on the score. A missing or "
                            "unknown plan is recorded for PI review and does not reject the call."
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
            "description": (
                "Look up API documentation for an installed Python function/class "
                "and/or local methodological guidance. Method questions always "
                "search statistical caveats and, when the experimental pilot is "
                "enabled, separately labelled method decision cards. Provide at "
                "least one of function_path or method_question."
            ),
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
                            "corpus and, when enabled, an independently labelled set of "
                            "non-binding decision comparisons. Ask here BEFORE choosing "
                            "a statistical approach. May be given with or instead of "
                            "function_path."
                        )
                    },
                    "question": {
                        "type": "string",
                        "description": "Optional: specific question about the API (e.g., 'what parameters control batch correction?')"
                    }
                }
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
            "name": "register_method_decision",
            "description": (
                "Record the PI/Bioinformatics scientific method choice for a bulk "
                "input before reporting its findings. The platform supplies input "
                "facts but does not choose the method or fallback."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input_profile_id": {"type": "string"},
                    "claim_level": {
                        "type": "string",
                        "enum": ["inferential", "descriptive", "not_feasible"],
                    },
                    "method": {"type": "string"},
                    "rationale": {"type": "string"},
                    "execution_id": {"type": "string"},
                },
                "required": [
                    "input_profile_id", "claim_level", "method", "rationale",
                    "execution_id",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "register_publication_table",
            "description": (
                "Register an existing workspace CSV as a publication table. The "
                "host records its digest and current identity; unregistered CSV files "
                "are not included in reports."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "csv_path": {
                        "type": "string",
                        "description": "Workspace-relative path to an existing CSV file.",
                    },
                    "interpretation": {"type": "string"},
                    "evidence_execution_id": {
                        "type": "string",
                        "description": (
                            "Candidate current execution that produced the table. An "
                            "invalid or missing value leaves the table without a valid "
                            "producer receipt."
                        ),
                    },
                    "plan_step_ids": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "description": (
                            "Candidate approved-plan steps supported by the table. The PI "
                            "decides whether those steps are scientifically complete."
                        ),
                        "default": [],
                    },
                },
                "required": ["name", "description", "csv_path"],
            },
        },
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
                                "interpretation": {
                                    "type": "string",
                                    "description": (
                                        "What the evidence supports, stated at the "
                                        "strongest class of evidence the available "
                                        "modalities reach. Where the analysis ranks the "
                                        "candidates, name the most probable one as an "
                                        "explicit inference carrying its residual "
                                        "uncertainty. Do not present an assay limit as a "
                                        "finding, and do not claim support the data do "
                                        "not carry."
                                    ),
                                },
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
    {
        "type": "function",
        "function": {
            "name": "list_adata_checkpoints",
            "description": (
                "Read-only lookup of the approved AnnData checkpoint catalog. "
                "Use it to retrieve bounded, path-free facts about current, "
                "Stage 1, prior-round and retained working-object choices. It "
                "performs no load and returns no filesystem path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Page number for bounded pagination.",
                    },
                    "page_size": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Entries per page.",
                    },
                    "checkpoint_id": {
                        "type": ["string", "null"],
                        "description": "Exact opaque checkpoint id filter.",
                    },
                    "live_binding_id": {
                        "type": ["string", "null"],
                        "description": "Exact opaque live-binding id filter.",
                    },
                },
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


# Deliberately NOT a member of TOOLS. The counts are measured once, before
# planning, and every consumer treats them as the authority over the user's
# declaration; a tool the main executor could call would let a mid-analysis turn
# rewrite the number the plan was built on.
REGISTER_SAMPLE_COUNTS_TOOL = {
    "type": "function",
    "function": {
        "name": "register_sample_counts",
        "description": (
            "Register how many biological replicates each (group, data type) "
            "pair actually holds, as read from the data. Call once, after "
            "inspecting every sequencing input. This call covers sequencing "
            "inputs only; pathology and biochemistry replicate counts are "
            "registered by the platform elsewhere and must not be reported "
            "here. Report what the data shows, "
            "not what the experimental design says it should show: the whole "
            "point of this pass is that the declaration may be wrong. Two "
            "files of different formats holding the same specimen are one "
            "replicate. One file holding several multiplexed specimens is "
            "several. When a count cannot be determined, register the record "
            "with detected_count 0 and method 'undetermined' rather than "
            "guessing -- an undetermined record is skipped, a wrong one is not."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "description": "One entry per (group, data type) pair inspected.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "group_name": {
                                "type": "string",
                                "description": (
                                    "Treatment group name exactly as given in the "
                                    "experimental design. A matrix spanning groups "
                                    "must produce one record per group when its columns "
                                    "or metadata verify membership, even if the uploaded "
                                    "file is ungrouped. Use an empty string only when "
                                    "membership cannot be established from the data."
                                ),
                            },
                            "data_type": {
                                "type": "string",
                                "description": (
                                    "Data type exactly as given in the file listing "
                                    "(e.g. 'scRNA-seq', 'Loom', 'bulk RNA-seq')."
                                ),
                            },
                            "detected_count": {
                                "type": "integer",
                                "minimum": 0,
                                "description": (
                                    "Biological replicates found. 0 means undetermined, "
                                    "never 'no samples'."
                                ),
                            },
                            "method": {
                                "type": "string",
                                "description": (
                                    "How the count was established: 'obs_column:<name>', "
                                    "'barcode_suffix', 'matrix_columns', or 'undetermined'."
                                ),
                            },
                            "evidence": {
                                "type": "string",
                                "description": (
                                    "One line naming what was read and what it showed, "
                                    "e.g. \"adata.obs['sample'] has 1 unique value: Con_1\". "
                                    "This is recorded on the session."
                                ),
                            },
                        },
                        "required": ["group_name", "data_type", "detected_count",
                                     "method", "evidence"],
                    },
                },
            },
            "required": ["records"],
        },
    },
}

_TOOL_BY_NAME = {t["function"]["name"]: t for t in TOOLS}

#: The two-tool set for the pre-planning profiling pass. It reuses the
#: executor's own execute_code schema so persistent_outputs keeps its meaning,
#: and adds nothing that could emit an analysis artifact.
PROFILING_TOOLS = [_TOOL_BY_NAME["execute_code"], REGISTER_SAMPLE_COUNTS_TOOL]

PROFILING_TOOL_NAMES: frozenset[str] = frozenset(
    t["function"]["name"] for t in PROFILING_TOOLS
)
