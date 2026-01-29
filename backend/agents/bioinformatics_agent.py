"""
Bioinformatics Agent - Handles snRNA-seq analysis with iterative code execution
Supports autonomous technical steps and PI consultation for biological decisions
"""
from typing import Dict, Any, List, Optional, Callable, Awaitable
import json
import logging

from .base import BaseAgent

# Configure logger
logger = logging.getLogger(__name__)
from ..models.schemas import (
    AgentType, AnalysisState, MessageRole, CodeExecution
)
from ..utils.llm_client import OpenAICompatibleClient, ToolCall
from ..utils.sandbox import LocalJupyterSandbox, CodeExecutionResult


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
                    }
                },
                "required": ["code", "description", "step_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consult_pi",
            "description": "Request guidance from PI Agent on biological decisions like: cell type annotation, analysis direction, which genes/pathways to analyze, whether to do trajectory analysis, result interpretation",
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
            "description": "Report key findings and conclusions from the analysis. Use this when you have completed a major analysis phase.",
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
                    }
                },
                "required": ["findings", "summary"]
            }
        }
    }
]


class BioinformaticsAgent(BaseAgent):
    """
    Bioinformatics Agent with Autonomous Mode
    
    Can autonomously execute technical steps:
    - Data loading, QC, filtering
    - Normalization, HVG selection
    - Dimensionality reduction, clustering
    - Basic visualization
    
    Requires PI consultation for biological decisions:
    - Cell type annotation
    - Result interpretation
    - Analysis direction choices
    - Gene/pathway selection
    - Trajectory analysis decisions
    """
    
    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        model: str,
        system_prompt: str,
        sandbox: LocalJupyterSandbox,
        literature_callback: Optional[Callable[[str], Awaitable[str]]] = None,
        pi_callback: Optional[Callable[[str, str], Awaitable[str]]] = None,
        message_callback: Optional[Callable[[str, str], Awaitable[None]]] = None
    ):
        super().__init__(
            agent_type=AgentType.BIOINFORMATICS,
            llm_client=llm_client,
            model=model,
            system_prompt=system_prompt
        )
        self.sandbox = sandbox
        self.literature_callback = literature_callback
        self.pi_callback = pi_callback
        self.message_callback = message_callback
        
        # Track execution history
        self.code_history: List[CodeExecution] = []
        self.last_result: Optional[CodeExecutionResult] = None
        self.analysis_context: Dict[str, Any] = {}
        
    async def _notify(self, message_type: str, content: str):
        """Send notification via callback"""
        if self.message_callback:
            await self.message_callback(message_type, content)
    
    def _build_notebook_context(self) -> str:
        """Build context string from notebook history"""
        if not self.code_history:
            return "Notebook is empty. This is the first code cell."
        
        context_parts = ["=== Notebook History (last 10 cells) ===\n"]
        
        recent_history = self.code_history[-10:]
        start_idx = len(self.code_history) - len(recent_history)
        
        for i, execution in enumerate(recent_history):
            context_parts.append(f"--- Cell {start_idx + i + 1} ---")
            context_parts.append(f"```python\n{execution.code}\n```")
            
            if execution.output:
                output = execution.output[:800]
                if len(execution.output) > 800:
                    output += "\n... (truncated)"
                context_parts.append(f"Output:\n{output}")
            
            if execution.error:
                context_parts.append(f"Error: {execution.error}")
            
            if execution.figures:
                context_parts.append(f"[Generated {len(execution.figures)} figure(s)]")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _build_data_context(self, state: AnalysisState) -> str:
        """Build context about available data"""
        data_parts = ["=== Available Data ===\n"]
        
        has_data = False
        
        if state.control_seq_path:
            has_data = True
            data_parts.append(f"Control scRNA-seq: {state.control_seq_path}")
            data_parts.append(f"  Load: adata_control = sc.read_h5ad('{state.control_seq_path}')")
            
        if state.treatment_seq_path:
            has_data = True
            data_parts.append(f"Treatment scRNA-seq: {state.treatment_seq_path}")
            data_parts.append(f"  Load: adata_treatment = sc.read_h5ad('{state.treatment_seq_path}')")
            
        if state.control_loom_path:
            has_data = True
            data_parts.append(f"Control Loom (velocity): {state.control_loom_path}")
            
        if state.treatment_loom_path:
            has_data = True
            data_parts.append(f"Treatment Loom (velocity): {state.treatment_loom_path}")
        
        if not has_data:
            data_parts.append("⚠️ No sequencing data uploaded")
        
        return "\n".join(data_parts)
    
    def _build_analysis_context(self) -> str:
        """Build context about current analysis state"""
        if not self.analysis_context:
            return ""
        
        parts = ["=== Current Analysis State ==="]
        for key, value in self.analysis_context.items():
            if isinstance(value, list):
                parts.append(f"{key}: {', '.join(str(v) for v in value[:10])}")
            else:
                parts.append(f"{key}: {value}")
        return "\n".join(parts)

    async def run_autonomous_analysis(
        self,
        state: AnalysisState,
        initial_instruction: str,
        max_steps: int = 30
    ) -> Dict[str, Any]:
        """
        Run analysis with autonomous technical steps and PI consultation for biological decisions
        """
        logger.info(f"[Bioinformatics] Starting autonomous analysis with max_steps={max_steps}")
        # Start sandbox if needed
        if self.sandbox.km is None:
            await self._notify("status", "Starting analysis environment...")
            await self.sandbox.start()
            logger.info("[Bioinformatics] Sandbox started")
        
        # Build initial prompt
        system_prompt = f"""{self.system_prompt}

You are conducting single-cell RNA-seq analysis. You have access to tools:
1. execute_code - Run Python code (scanpy, anndata, numpy, pandas, matplotlib available)
2. consult_pi - Ask PI for biological decisions
3. search_literature - Search for marker genes, pathways, biological knowledge
4. report_findings - Report conclusions when analysis phase is complete

WORKFLOW GUIDELINES:
- Technical steps can be done autonomously: data loading, QC, normalization, PCA, clustering, UMAP, basic DE
- For BIOLOGICAL DECISIONS, you MUST use consult_pi:
  * Cell type annotation (what are these clusters?)
  * Which clusters show interesting changes?
  * What genes or pathways to analyze?
  * Whether to do trajectory/velocity analysis?
  * Interpretation of results
  * Which comparisons matter biologically?

{self._build_data_context(state)}

{self._build_notebook_context()}

{self._build_analysis_context()}

**CRITICAL: DATA PREPROCESSING (MUST DO BEFORE PCA)**
Before running sc.tl.pca() or any dimensionality reduction:
1. Check for NaN: print(f"NaN in X: {{np.isnan(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X).sum()}}")
2. If NaN exists, fix it:
   - sc.pp.filter_genes(adata, min_cells=3)
   - adata.X = np.nan_to_num(adata.X, nan=0.0) if dense
   - adata.X.data = np.nan_to_num(adata.X.data, nan=0.0) if sparse
3. NEVER skip this step - PCA will fail with NaN values!

**CRITICAL: REPRODUCIBILITY IS MANDATORY**
- ALWAYS use random_state=42 or seed=42 in EVERY function that accepts it
- Examples: sc.tl.umap(..., random_state=42), sc.tl.leiden(..., random_state=42), 
  sc.tl.pca(..., random_state=42), train_test_split(..., random_state=42)
- NEVER omit the random_state parameter in stochastic functions
- Use n_jobs=1 when determinism is critical
- Print key metrics and results
- Generate informative visualizations
- Use report_findings when completing an analysis phase
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analysis instruction: {initial_instruction}"}
        ]
        
        all_findings = []
        all_figures = []
        step_count = 0
        completed = False
        
        while step_count < max_steps and not completed:
            step_count += 1
            
            self._set_llm_context()
            
            try:
                response = await self.llm_client.chat_with_tools(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    temperature=0.3,
                    max_tokens=4096
                )
            except Exception as e:
                await self._notify("error", f"LLM call failed: {str(e)}")
                break
            
            if not response.has_tool_calls:
                if response.content:
                    await self._notify("response", response.content)
                    messages.append({"role": "assistant", "content": response.content})
                break
            
            for tool_call in response.tool_calls:
                tool_name = tool_call.name
                args = tool_call.arguments
                
                if tool_name == "execute_code":
                    code = args.get("code", "")
                    description = args.get("description", "")
                    step_type = args.get("step_type", "technical")
                    
                    await self._notify("status", f"[{step_type.upper()}] {description}")
                    await self._notify("code", code)
                    
                    result = await self.sandbox.execute(code)
                    
                    code_exec = CodeExecution(
                        code=code,
                        output=result.output,
                        error=result.error,
                        figures=result.figures,
                        execution_time=0.0
                    )
                    self.code_history.append(code_exec)
                    self.last_result = result
                    
                    if result.output:
                        await self._notify("result", result.output[:2000])
                    for fig in result.figures:
                        await self._notify("figure", fig)
                        all_figures.append(fig)
                    if result.error:
                        await self._notify("error", result.error)
                    
                    tool_result = f"Execution {'successful' if not result.error else 'failed'}.\n"
                    if result.output:
                        tool_result += f"Output:\n{result.output[:1500]}\n"
                    if result.error:
                        tool_result += f"Error: {result.error}\n"
                    if result.figures:
                        tool_result += f"Generated {len(result.figures)} figure(s).\n"
                    
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(args)}
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                    
                elif tool_name == "consult_pi":
                    question = args.get("question", "")
                    context = args.get("context", "")
                    options = args.get("options", [])
                    
                    await self._notify("status", f"Consulting PI: {question[:100]}...")
                    await self._notify("pi_question", json.dumps({
                        "question": question,
                        "context": context,
                        "options": options
                    }))
                    
                    pi_response = ""
                    if self.pi_callback:
                        pi_response = await self.pi_callback(question, context)
                        await self._notify("pi_response", pi_response)
                    else:
                        pi_response = "PI guidance not available. Proceed with standard practices."
                    
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(args)}
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"PI Response: {pi_response}"
                    })
                    
                elif tool_name == "search_literature":
                    topic = args.get("topic", "")
                    
                    await self._notify("status", f"Searching literature: {topic[:50]}...")
                    
                    lit_result = ""
                    if self.literature_callback:
                        lit_result = await self.literature_callback(topic)
                        await self._notify("literature", lit_result[:1000])
                    else:
                        lit_result = "Literature search not available."
                    
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(args)}
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Literature findings: {lit_result[:2000]}"
                    })
                    
                elif tool_name == "report_findings":
                    findings = args.get("findings", [])
                    summary = args.get("summary", "")
                    suggestions = args.get("suggestions", [])
                    
                    all_findings.extend(findings)
                    
                    await self._notify("findings", json.dumps({
                        "findings": findings,
                        "summary": summary,
                        "suggestions": suggestions
                    }))
                    
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(args)}
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Findings recorded. Continue or wait for instructions."
                    })
                    
                    completed = True
                    break

        status = "completed" if completed else "max_steps_reached"
        logger.info(f"[Bioinformatics] Analysis {status}: {step_count} steps executed, {len(all_findings)} findings, {len(all_figures)} figures")

        return {
            "status": status,
            "findings": all_findings,
            "figures": all_figures,
            "steps_executed": step_count,
            "code_history": self.code_history,
            "success": True
        }

    async def execute_step(self, state: AnalysisState, instruction: str) -> Dict[str, Any]:
        """Execute a single analysis step (legacy interface)"""
        return await self.run_autonomous_analysis(state, instruction, max_steps=5)
    
    async def process(self, state: AnalysisState, instruction: str, **kwargs) -> Dict[str, Any]:
        """Main entry point for workflow"""
        if not state.control_seq_path and not state.treatment_seq_path:
            return {
                "status": "waiting_input",
                "message": "Please upload snRNA-seq data file",
                "findings": [],
                "success": False
            }
        
        result = await self.run_autonomous_analysis(
            state, 
            instruction,
            max_steps=kwargs.get("max_steps", 30)
        )
        
        return {
            "status": result["status"],
            "findings": result["findings"],
            "figures": result["figures"],
            "code_execution": self.code_history[-1] if self.code_history else None,
            "success": result["success"],
            "steps_executed": result["steps_executed"]
        }
    
    def get_all_code(self) -> str:
        """Get all executed code"""
        return "\n\n".join([
            f"# Cell {i+1}\n{ce.code}" 
            for i, ce in enumerate(self.code_history)
        ])
    
    def get_execution_summary(self) -> str:
        """Get summary of executions"""
        if not self.code_history:
            return "No code executed yet."
        
        errors = sum(1 for ce in self.code_history if ce.error)
        figures = sum(len(ce.figures) for ce in self.code_history)
        
        return f"Cells: {len(self.code_history)}, Success: {len(self.code_history) - errors}, Errors: {errors}, Figures: {figures}"
    
    async def save_state(self):
        """Save current state"""
        if self.sandbox:
            await self.sandbox.save_session()
    
    async def restore_state(self):
        """Restore state"""
        await self.sandbox.start()
