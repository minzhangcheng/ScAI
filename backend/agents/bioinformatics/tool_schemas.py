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
                        "description": "Source of the gene list (e.g., 'curated gene set', 'top 50 DEGs', 'PMID:12345678')"
                    }
                },
                "required": ["score_name", "method", "description"]
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
                        "description": "Full dotted path to the function or module (e.g., 'scanpy.tl.umap', 'decoupler.op.collectri', 'gseapy.enrichr')"
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
