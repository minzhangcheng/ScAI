"""Agent node methods: pathology, bioinformatics, biochemistry, literature."""
import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..models.schemas import AgentType, AnalysisPhase, AnalysisStep, MessageRole, TaskStatus
from ..utils.llm_client import UpstreamExhaustedError
from ..utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ..utils.expression_contract import record_round_start
from ..agents.bioinformatics_agent import _evidence_gate_record, count_saved_figures
from ..agents.bioinformatics.exploration_diagnostics import (
    ensure_exploration_diagnostic_figure,
)
from ..agents.bioinformatics.stage1_baseline_figures import (
    ensure_stage1_baseline_figures,
)
from ..services.figure_contract import audit_figure_contract
from .helpers import _get_task_text, _has_stage1_keyword, _normalize_tasks, cache_hit, cache_invalidate, cache_store, decide_bioinfo_post_execution, MAX_EXECUTION_REVISION_ATTEMPTS, prepend_species, safe_pi_brief, STAGE1_KEYWORDS, WorkflowSupersededError
from ..utils.blob_store import BLOB_REF_PREFIX
from .retry import with_retry

if TYPE_CHECKING:
    from .state import WorkflowState
else:
    WorkflowState = dict

# Re-exported so `backend.workflow.agents.<name>` keeps resolving
# for existing importers and for tests that patch these by name.
from .agent_nodes.shared import (  # noqa: F401
    asyncio,
    hashlib,
    json,
    logging,
    Any,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
    AgentType,
    AnalysisPhase,
    AnalysisStep,
    MessageRole,
    TaskStatus,
    UpstreamExhaustedError,
    MECHANISM_EVIDENCE_POLICY,
    count_saved_figures,
    _get_task_text,
    _has_stage1_keyword,
    _normalize_tasks,
    cache_hit,
    cache_invalidate,
    cache_store,
    decide_bioinfo_post_execution,
    prepend_species,
    safe_pi_brief,
    STAGE1_KEYWORDS,
    BLOB_REF_PREFIX,
    with_retry,
    _reset_pi_instructions_for_stage2,
    _stable_code_hash,
    _STAGE1_INSTRUCTION,
    _MECHANISM_CAPTURE_INSTRUCTION,
    _build_exploration_instruction,
    _build_innovation_instruction,
    _build_stage2_instruction,
    build_phase_instruction,
)
from .agent_nodes import (  # noqa: F401
    ModalityNodesMixin,
    LiteratureNodeMixin,
    ProfilingNodeMixin,
)


import sys
from .agent_nodes import bioinformatics_node

logger = logging.getLogger(__name__)


class AgentNodesMixin(
    ModalityNodesMixin,
    LiteratureNodeMixin,
    ProfilingNodeMixin,
):
    """Mixin providing agent-node execution methods. Composed into SingleCellAnalysisWorkflow."""


    
    @with_retry("bioinformatics_analysis")
    async def _bioinformatics_analysis(self, state: WorkflowState) -> WorkflowState:
        bioinformatics_node.bind_facade(sys.modules[__name__])
        return await bioinformatics_node.run_bioinformatics_analysis(self, state)
    


    
