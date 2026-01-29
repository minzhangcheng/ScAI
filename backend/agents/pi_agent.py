"""
PI Agent - Principal Investigator that coordinates all agents
"""
from typing import Dict, Any, List, Optional
import json
import re
import logging

from .base import BaseAgent

# Configure logger
logger = logging.getLogger(__name__)
from ..models.schemas import (
    AgentType, AnalysisState, MessageRole, RoundResult
)
from ..utils.llm_client import OpenAICompatibleClient


class PIAgent(BaseAgent):
    """
    Principal Investigator Agent
    
    Responsibilities:
    - Coordinate overall research strategy
    - Analyze and synthesize results from all agents
    - Resolve conflicts between agents
    - Make decisions on next steps
    - Determine when to stop analysis
    """
    
    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        model: str,
        system_prompt: str
    ):
        super().__init__(
            agent_type=AgentType.PI,
            llm_client=llm_client,
            model=model,
            system_prompt=system_prompt
        )
        
    async def process(
        self,
        state: AnalysisState,
        instruction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Process instruction and coordinate analysis"""
        
        # Get context from state
        context = self._build_context(state)
        
        # Build prompt
        prompt = f"""
{context}

Current instruction: {instruction}

Based on the above information:
1. Analyze results from all aspects
2. Identify key findings and potential issues
3. Arbitrate if there are conflicts
4. Plan next steps for analysis

Reply in JSON format with the following fields:
- summary: Summary of the current round
- findings: List of key findings
- conflicts: Conflicts found and solutions
- next_steps: Tasks for each Agent to execute next
- should_continue: Whether to continue analysis (true/false)
- reason: Reason for continuing/stopping
"""
        
        response = await self.chat(prompt, temperature=0.5)
        
        # Parse response
        result = self._parse_response(response)
        
        return result
    
    async def initial_instruction(
        self,
        state: AnalysisState,
        user_instruction: str
    ) -> Dict[str, Any]:
        """
        Generate initial instructions for all agents

        Args:
            state: Current analysis state
            user_instruction: User's initial instruction

        Returns:
            Dict with instructions for each agent
        """
        logger.info(f"[PI] Generating initial instructions for project: {state.name}")
        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {user_instruction}

You need to develop the first round analysis plan for the team. Please assign specific tasks for each direction:

1. Pathology analysis: What types of sections need to be analyzed? What should be the focus?
2. Bioinformatics analysis: Basic single-cell sequencing analysis workflow, cell types and pathways to focus on
3. Biochemistry analysis: What indicators need attention?
4. Literature research: What topics need to be researched?

Reply in JSON format as follows:
{{
    "overview": "Overview of overall research strategy",
    "pathology_tasks": ["task1", "task2", ...],
    "bioinformatics_tasks": ["task1", "task2", ...],
    "biochemistry_tasks": ["task1", "task2", ...],
    "literature_topics": ["topic1", "topic2", ...]
}}
"""
        
        response = await self.chat(prompt, temperature=0.3)
        
        # Parse JSON from response (safely handle None)
        if response:
            try:
                # Extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Return default structure if parsing fails
        return {
            "overview": response or "Failed to generate initial instructions",
            "pathology_tasks": ["Analyze tissue sections, assess overall morphological changes"],
            "bioinformatics_tasks": ["Execute standard scRNA-seq/snRNA-seq analysis workflow"],
            "biochemistry_tasks": ["Summarize existing biochemical indicators"],
            "literature_topics": ["Literature related to the research topic and phenotype"]
        }
    
    async def process_human_guidance(
        self,
        state: AnalysisState,
        human_input: str
    ) -> Dict[str, Any]:
        """
        Process human researcher's guidance and adjust analysis direction
        
        Args:
            state: Current analysis state
            human_input: Human researcher's guidance/instruction
            
        Returns:
            Dict with:
            - response: PI's response to human
            - should_stop: Whether to stop analysis
            - adjusted_instructions: New instructions for agents (if applicable)
        """
        # Build context from current state
        context = self._build_context(state)
        
        prompt = f"""
## Current Analysis Status
Project: {state.name}
Current Round: {state.current_round}
Research Objective: {state.initial_instruction}

{context}

## Human Researcher Guidance
The human researcher has provided the following guidance:
"{human_input}"

As the PI Agent, please:
1. Carefully interpret what the human researcher wants
2. Decide if they want to stop/conclude the analysis, or adjust the direction
3. If adjusting direction, provide new specific instructions for each agent

Common stop/conclude keywords: stop, terminate, end, finish, conclude, summarize, 停止, 终止, 结束, 总结

Respond in JSON format:
{{
    "interpretation": "Your interpretation of what the human wants",
    "response": "Your response to the human researcher (be helpful and professional)",
    "should_stop": true/false,
    "reason": "Why you decided to stop or continue",
    "adjusted_instructions": {{
        "pathology_tasks": ["task1", "task2"],
        "bioinformatics_tasks": ["task1", "task2"],
        "biochemistry_tasks": ["task1", "task2"],
        "literature_topics": ["topic1", "topic2"]
    }},
    "focus_areas": ["key area 1", "key area 2"],
    "deprioritize": ["area to skip or reduce focus"]
}}

If should_stop is true, adjusted_instructions can be empty.
If adjusting direction, provide specific and actionable tasks.
"""
        
        response = await self.chat(prompt, temperature=0.3)
        
        # Parse response
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "interpretation": data.get("interpretation", ""),
                    "response": data.get("response", "Acknowledged. Adjusting analysis direction."),
                    "should_stop": data.get("should_stop", False),
                    "reason": data.get("reason", ""),
                    "adjusted_instructions": data.get("adjusted_instructions"),
                    "focus_areas": data.get("focus_areas", []),
                    "deprioritize": data.get("deprioritize", [])
                }
            else:
                logger.warning(f"[PI] No JSON found in response, using fallback parsing. Response: {response[:200]}...")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[PI] Failed to parse JSON response: {e}. Response: {response[:200]}...")
        
        # Fallback: check for stop keywords manually
        human_lower = human_input.lower()
        stop_keywords = ['stop', 'terminate', 'end', 'finish', 'conclude', 'summarize',
                        '停止', '终止', '结束', '总结']
        
        if any(keyword in human_lower for keyword in stop_keywords):
            return {
                "interpretation": "Human requested to stop the analysis",
                "response": "Understood. I will conclude the current analysis and generate a final report.",
                "should_stop": True,
                "reason": "Human explicitly requested to stop/conclude",
                "adjusted_instructions": None
            }
        
        # Default: treat as guidance to continue with adjustments
        return {
            "interpretation": human_input,
            "response": f"Acknowledged. I will incorporate your guidance: {human_input}",
            "should_stop": False,
            "reason": "Continuing with adjusted focus based on human guidance",
            "adjusted_instructions": {
                "pathology_tasks": [f"Focus on: {human_input}"],
                "bioinformatics_tasks": [f"Prioritize analysis related to: {human_input}"],
                "biochemistry_tasks": [f"Review biochemistry aspects of: {human_input}"],
                "literature_topics": [f"Search literature on: {human_input}"]
            }
        }
    
    async def identify_data_needs(
        self,
        state: AnalysisState,
        round_result: RoundResult
    ) -> Dict[str, List[str]]:
        """
        Identify what additional data would help the analysis
        
        Args:
            state: Current analysis state
            round_result: Results from the current round
            
        Returns:
            Dict with lists of needed data by category
        """
        prompt = f"""
## Current Analysis State
Project: {state.name}
Round: {round_result.round_number}
Research Objective: {state.initial_instruction}

## Round Summary
{round_result.pi_summary}

## Key Findings
Pathology: {round_result.pathology_findings}
Bioinformatics: {round_result.bioinformatics_findings}
Biochemistry: {round_result.biochemistry_findings}

## Next Steps Planned
{round_result.next_steps}

Based on the current findings and planned next steps, what additional data would help deepen the analysis?

Please identify specific data needs in JSON format:
{{
    "pathology": [
        "Special staining to assess specific features (e.g., fibrosis, lipid, immune cells)",
        "Additional tissue sections from different regions",
        ...
    ],
    "biochemistry": [
        "Tissue/organ function markers",
        "Metabolic indicators relevant to the phenotype",
        "Oxidative stress or inflammatory markers",
        ...
    ],
    "other": [
        "Western blot for specific proteins",
        "qPCR validation of key genes",
        ...
    ]
}}

Only include data that would significantly help answer the research question.
If no additional data is needed, return empty lists.
"""
        
        response = await self.chat(prompt, temperature=0.3)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "pathology": data.get("pathology", []),
                    "biochemistry": data.get("biochemistry", []),
                    "other": data.get("other", [])
                }
        except (json.JSONDecodeError, KeyError):
            pass
        
        return {"pathology": [], "biochemistry": [], "other": []}
    
    async def process_user_input(
        self,
        state: AnalysisState,
        user_message: str,
        uploaded_files: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user input after a round completes.
        Classify uploaded files and determine next actions.
        
        Args:
            state: Current analysis state
            user_message: User's text message
            uploaded_files: List of uploaded file info (type, filename, id)
            
        Returns:
            Dict with:
            - action: 'continue', 'stop', 'analyze_new_data'
            - file_assignments: Which files go to which agents
            - instructions: Updated instructions for agents
            - response: Response to user
        """
        uploaded_files = uploaded_files or []
        logger.info(f"[PI] Processing user input: {user_message[:100] if user_message else 'no message'}... ({len(uploaded_files)} files)")

        # Build file description
        file_descriptions = []
        for f in uploaded_files:
            file_descriptions.append(f"{f.get('type', 'unknown')}: {f.get('filename', 'unknown')}")
        
        prompt = f"""
## User Input Processing

The user has provided input after Round {state.current_round} analysis.

User message: "{user_message}"

Uploaded files:
{chr(10).join(file_descriptions) if file_descriptions else "None"}

## Task
1. Determine what the user wants:
   - Continue to next round?
   - Stop and generate final report?
   - Analyze new data first?
   - Adjust analysis direction?

2. If files were uploaded, classify each file:
   - Images of H&E, Sirius Red, Oil Red O, etc. → pathology
   - Images of Western blots, gels → biochemistry
   - Text describing biochemistry results → biochemistry
   - Other data files → determine appropriate agent

3. Extract any specific instructions from user message

Respond in JSON format:
{{
    "action": "continue" | "stop" | "analyze_new_data",
    "response": "Your response to the user",
    "file_assignments": {{
        "pathology": [
            {{"filename": "xxx.png", "image_type": "Sirius Red", "reason": "Fibrosis assessment"}}
        ],
        "biochemistry": [
            {{"filename": "xxx.txt", "data_type": "metabolic markers", "reason": "Enzyme activity values"}}
        ]
    }},
    "biochemistry_values": {{
        "Marker1": {{"value": 120, "unit": "U/L"}},
        "Marker2": {{"value": 95, "unit": "U/L"}}
    }},
    "adjusted_instructions": {{
        "pathology_tasks": ["specific task"],
        "bioinformatics_tasks": ["specific task"],
        "biochemistry_tasks": ["specific task"],
        "literature_topics": ["specific topic"]
    }},
    "focus_areas": ["area to emphasize"],
    "reason": "Why this action was chosen"
}}

Note: Extract biochemistry values from user message if they mention specific numbers (e.g., "ALT is 120 U/L").
"""
        
        response = await self.chat(prompt, temperature=0.3)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "action": data.get("action", "continue"),
                    "response": data.get("response", "Acknowledged"),
                    "file_assignments": data.get("file_assignments", {}),
                    "biochemistry_values": data.get("biochemistry_values", {}),
                    "adjusted_instructions": data.get("adjusted_instructions"),
                    "focus_areas": data.get("focus_areas", []),
                    "reason": data.get("reason", "")
                }
        except (json.JSONDecodeError, KeyError) as e:
            pass
        
        # Fallback: check for stop keywords
        user_lower = user_message.lower()
        stop_keywords = ['stop', 'terminate', 'end', 'finish', 'conclude', 'report',
                        '停止', '终止', '结束', '总结', '报告']
        
        if any(keyword in user_lower for keyword in stop_keywords):
            return {
                "action": "stop",
                "response": "Understood. Generating final report.",
                "file_assignments": {},
                "biochemistry_values": {},
                "adjusted_instructions": None,
                "focus_areas": [],
                "reason": "User requested to stop"
            }
        
        # Default: continue
        return {
            "action": "continue" if not uploaded_files else "analyze_new_data",
            "response": "Acknowledged. Proceeding with analysis.",
            "file_assignments": {
                "pathology": [{"filename": f["filename"], "image_type": "user_upload"} 
                             for f in uploaded_files if f.get("type") == "image"]
            },
            "biochemistry_values": {},
            "adjusted_instructions": None,
            "focus_areas": [],
            "reason": "Continuing analysis"
        }
    
    async def synthesize_round(
        self,
        state: AnalysisState,
        pathology_result: Optional[str] = None,
        bioinformatics_result: Optional[str] = None,
        biochemistry_result: Optional[str] = None,
        literature_result: Optional[str] = None
    ) -> RoundResult:
        """
        Synthesize results from all agents for a round

        Args:
            state: Current analysis state
            *_result: Results from each agent

        Returns:
            RoundResult with synthesis
        """
        logger.info(f"[PI] Synthesizing round {state.current_round} results")
        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

This is the results summary for Round {state.current_round}. Please synthesize:

## Pathology Analysis Results
{pathology_result or "No new results"}

## Bioinformatics Analysis Results
{bioinformatics_result or "No new results"}

## Biochemistry Analysis Results
{biochemistry_result or "No new results"}

## Literature Research Results
{literature_result or "No new results"}

Please provide comprehensive analysis in JSON format:
{{
    "summary": "Summary of this round",
    "pathology_findings": ["pathology finding 1", ...],
    "bioinformatics_findings": ["bioinformatics finding 1", ...],
    "biochemistry_findings": ["biochemistry finding 1", ...],
    "literature_findings": ["literature finding 1", ...],
    "conflicts": [
        {{"description": "conflict description", "resolution": "solution", "agents": ["agent1", "agent2"]}}
    ],
    "next_pathology": ["next pathology task"],
    "next_bioinformatics": ["next bioinformatics task"],
    "next_biochemistry": ["next biochemistry task"],
    "next_literature": ["next literature topic"],
    "new_findings": true/false,
    "should_stop": true/false,
    "stop_reason": "reason for stopping (if applicable)"
}}
"""
        
        response = await self.chat(prompt, temperature=0.3)
        
        # Parse response
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                return RoundResult(
                    round_number=state.current_round,
                    pathology_findings=data.get("pathology_findings", []),
                    bioinformatics_findings=data.get("bioinformatics_findings", []),
                    biochemistry_findings=data.get("biochemistry_findings", []),
                    literature_findings=data.get("literature_findings", []),
                    pi_summary=data.get("summary", ""),
                    conflicts=data.get("conflicts", []),
                    next_steps=self._extract_next_steps(data),
                    new_findings=data.get("new_findings", True)
                )
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Return default if parsing fails
        return RoundResult(
            round_number=state.current_round,
            pi_summary=response,
            new_findings=True
        )
    
    async def generate_final_summary(self, state: AnalysisState) -> str:
        """
        Generate final summary of the entire analysis

        Args:
            state: Final analysis state

        Returns:
            Final summary text
        """
        logger.info(f"[PI] Generating final summary after {len(state.round_results)} rounds")
        # Build summary of all rounds
        rounds_summary = []
        for result in state.round_results:
            rounds_summary.append(f"""
### Round {result.round_number}
**Pathology findings**: {', '.join(result.pathology_findings) if result.pathology_findings else 'None'}
**Bioinformatics findings**: {', '.join(result.bioinformatics_findings) if result.bioinformatics_findings else 'None'}
**Biochemistry findings**: {', '.join(result.biochemistry_findings) if result.biochemistry_findings else 'None'}
**Literature findings**: {', '.join(result.literature_findings) if result.literature_findings else 'None'}
**PI Summary**: {result.pi_summary}
""")
        
        prompt = f"""
Based on all the analysis results below, write a complete analysis report:

## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## Results from All Rounds
{''.join(rounds_summary)}

Please write a well-structured analysis report including:
1. Research Overview
2. Key Findings
   - Pathological phenotype
   - Single-cell transcriptomic features
   - Biochemical indicator changes
3. Molecular Mechanism Analysis
4. Conclusions
5. Limitations and Future Directions

Please write in academic paper style.
"""
        
        return await self.chat(prompt, temperature=0.3)
    
    def _build_context(self, state: AnalysisState) -> str:
        """Build context string from state"""
        context_parts = []
        
        # Add project info
        context_parts.append(f"## Project Information")
        context_parts.append(f"Project Name: {state.name}")
        context_parts.append(f"Research Objective: {state.initial_instruction}")
        
        # Add current round info
        context_parts.append(f"\nCurrent Round: {state.current_round}")
        
        # Add previous round summaries
        if state.round_results:
            context_parts.append("\n## Historical Analysis Results")
            for result in state.round_results[-3:]:  # Last 3 rounds
                context_parts.append(f"\n### Round {result.round_number}\n{result.pi_summary}")
        
        return "\n".join(context_parts)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse PI response to extract structured data"""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        return {
            "summary": response,
            "findings": [],
            "conflicts": [],
            "next_steps": [],
            "should_continue": True,
            "reason": ""
        }
    
    def _extract_next_steps(self, data: Dict[str, Any]) -> List[str]:
        """Extract next steps from parsed data"""
        steps = []
        
        if "next_pathology" in data:
            for task in data["next_pathology"]:
                steps.append(f"[Pathology] {task}")
        
        if "next_bioinformatics" in data:
            for task in data["next_bioinformatics"]:
                steps.append(f"[Bioinformatics] {task}")
        
        if "next_biochemistry" in data:
            for task in data["next_biochemistry"]:
                steps.append(f"[Biochemistry] {task}")
        
        if "next_literature" in data:
            for topic in data["next_literature"]:
                steps.append(f"[Literature] {topic}")
        
        return steps
    
    # ============================================
    # Iterative Bioinformatics Analysis Control
    # ============================================
    
    async def plan_bioinformatics_workflow(
        self,
        state: AnalysisState
    ) -> List[str]:
        """
        Plan the bioinformatics analysis workflow based on research goals
        
        Returns:
            List of high-level analysis steps to perform
        """
        # Build data availability info
        data_info = []
        if state.control_seq_path:
            data_info.append(f"- Control group scRNA-seq: {state.control_seq_path}")
        if state.treatment_seq_path:
            data_info.append(f"- Treatment group scRNA-seq: {state.treatment_seq_path}")
        if state.control_loom_path:
            data_info.append(f"- Control group Loom (RNA velocity): {state.control_loom_path}")
        if state.treatment_loom_path:
            data_info.append(f"- Treatment group Loom (RNA velocity): {state.treatment_loom_path}")
        
        has_both_groups = state.control_seq_path and state.treatment_seq_path
        has_velocity = state.control_loom_path or state.treatment_loom_path
        
        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## Uploaded Data
{chr(10).join(data_info) if data_info else "No data uploaded"}

## Data Characteristics
- Control and treatment groups: {"Yes, differential analysis required" if has_both_groups else "Single group only"}
- RNA velocity data: {"Yes, cell trajectory analysis possible" if has_velocity else "No"}

Please plan the single-cell sequencing analysis workflow for this project. Return an ordered list of analysis steps.
Note: The first step must be loading data.

Format requirement: Return a JSON array, each element is a brief description of an analysis step.
Example: ["Load data", "Quality control", "Normalization", ...]

Return only the JSON array, no other content.
"""
        
        response = await self.chat(prompt, temperature=0.3)
        
        try:
            # Try to extract JSON array
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                steps = json.loads(json_match.group())
                if isinstance(steps, list):
                    return steps
        except json.JSONDecodeError:
            pass
        
        # Default workflow if parsing fails (adapted based on data availability)
        default_steps = [
            "Load data and view basic information",
            "Quality control: filter low-quality cells and genes",
            "Data normalization and log transformation",
            "Identify highly variable genes",
            "PCA dimensionality reduction",
            "Build neighbor graph and UMAP visualization",
            "Leiden clustering",
            "Identify marker genes for each cluster",
            "Cell type annotation",
        ]
        
        if has_both_groups:
            default_steps.extend([
                "Differential expression analysis (treatment vs control)",
                "Pathway enrichment analysis",
                "Cell proportion analysis"
            ])
        
        if has_velocity:
            default_steps.append("RNA velocity analysis")
        
        return default_steps
    
    async def generate_bioinformatics_instruction(
        self,
        state: AnalysisState,
        current_step: str,
        previous_results: List[Dict[str, Any]],
        step_index: int,
        total_steps: int
    ) -> str:
        """
        Generate specific instruction for bioinformatics agent
        
        Args:
            state: Current analysis state
            current_step: The current step description
            previous_results: Results from previous steps
            step_index: Current step number (0-indexed)
            total_steps: Total number of planned steps
            
        Returns:
            Specific instruction string for the bioinformatics agent
        """
        # Build context from previous results
        prev_context = ""
        if previous_results:
            prev_context = "\n## Previous Analysis Results\n"
            for i, result in enumerate(previous_results[-5:]):  # Last 5 results
                findings = result.get("findings", [])
                desc = result.get("description", "")
                if findings or desc:
                    prev_context += f"\nStep {i+1} ({desc}):\n"
                    for finding in findings:
                        prev_context += f"- {finding}\n"
        
        # Build data path info for first step
        data_paths_info = ""
        if step_index == 0 or "load" in current_step.lower() or "read" in current_step.lower():
            data_paths = []
            if state.control_seq_path:
                data_paths.append(f"Control group data path: {state.control_seq_path}")
            if state.treatment_seq_path:
                data_paths.append(f"Treatment group data path: {state.treatment_seq_path}")
            if state.control_loom_path:
                data_paths.append(f"Control group Loom path: {state.control_loom_path}")
            if state.treatment_loom_path:
                data_paths.append(f"Treatment group Loom path: {state.treatment_loom_path}")
            if data_paths:
                data_paths_info = "\n## Data File Paths\n" + "\n".join(data_paths) + "\n"
        
        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## Current Progress
Step {step_index + 1}/{total_steps}: {current_step}
{data_paths_info}
{prev_context}

Please generate specific analysis instructions for the Bioinformatics Agent to execute "{current_step}".

Instruction requirements:
1. Be specific and clear about what the Agent should do
2. If loading data, must use the exact file paths provided above
3. Include necessary parameter suggestions (thresholds, method choices, etc.)
4. Specify expected outputs (charts, numbers, etc.)
5. If performing differential analysis, specify comparison groups (treatment vs control)

Return the instruction content directly, no additional formatting needed.
"""
        
        instruction = await self.chat(prompt, temperature=0.3)
        return instruction.strip()
    
    async def evaluate_bioinformatics_result(
        self,
        state: AnalysisState,
        step_description: str,
        result: Dict[str, Any],
        remaining_steps: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate bioinformatics result and decide next action
        
        Args:
            state: Current analysis state
            step_description: What step was just completed
            result: Result from the bioinformatics agent
            remaining_steps: List of remaining planned steps
            
        Returns:
            Dict with:
            - continue_analysis: bool - whether to continue
            - next_step: str - next step to execute (if continue)
            - modify_plan: bool - whether to modify the remaining steps
            - new_steps: List[str] - new steps to insert (if modifying)
            - findings: str - PI's interpretation of the result
            - issues: List[str] - any issues or concerns
        """
        findings = result.get("findings", [])
        error = result.get("error", "")
        success = result.get("success", False)
        
        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## Completed Analysis Step
{step_description}

## Execution Result
Success: {"Yes" if success else "No"}
Error: {error or "None"}
Findings: {json.dumps(findings, ensure_ascii=False) if findings else "No specific findings"}

## Remaining Planned Steps
{json.dumps(remaining_steps, ensure_ascii=False) if remaining_steps else "No more planned steps"}

Please evaluate this step's result and decide the next action.

Return in JSON format:
{{
    "continue_analysis": true/false,
    "findings": "PI's interpretation of this step's results",
    "issues": ["issue 1", "issue 2"],
    "need_retry": true/false,
    "retry_instruction": "If retry needed, provide modified instruction",
    "insert_steps": ["If additional analysis steps needed, list here"],
    "skip_steps": ["If some planned steps are no longer needed, list here"]
}}
"""
        
        response = await self.chat(prompt, temperature=0.3)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "continue_analysis": data.get("continue_analysis", True),
                    "findings": data.get("findings", ""),
                    "issues": data.get("issues", []),
                    "need_retry": data.get("need_retry", False),
                    "retry_instruction": data.get("retry_instruction", ""),
                    "insert_steps": data.get("insert_steps", []),
                    "skip_steps": data.get("skip_steps", [])
                }
        except json.JSONDecodeError:
            pass
        
        # Default: continue if successful
        return {
            "continue_analysis": success and len(remaining_steps) > 0,
            "findings": response[:500],
            "issues": [] if success else [error],
            "need_retry": not success,
            "retry_instruction": "",
            "insert_steps": [],
            "skip_steps": []
        }
    
    async def summarize_bioinformatics_analysis(
        self,
        state: AnalysisState,
        all_results: List[Dict[str, Any]]
    ) -> str:
        """
        Summarize the complete bioinformatics analysis
        
        Args:
            state: Current analysis state
            all_results: All results from the bioinformatics analysis
            
        Returns:
            Summary string
        """
        # Collect all findings
        all_findings = []
        all_figures = 0
        
        for result in all_results:
            findings = result.get("findings", [])
            figures = result.get("figures", [])
            desc = result.get("description", "")
            
            if findings:
                all_findings.append(f"**{desc}**: " + "; ".join(findings))
            all_figures += len(figures)
        
        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## Bioinformatics Analysis Completed
Total {len(all_results)} analysis steps executed
Generated {all_figures} figures

## Findings from Each Step
{chr(10).join(all_findings) if all_findings else "No specific findings recorded"}

Please synthesize the above results and write a concise bioinformatics analysis summary (200-300 words).
Focus on:
1. Cell type composition and changes
2. Major differential expression patterns
3. Key findings related to research objectives
4. Questions requiring further verification or analysis

Reply in the same language as the research objective.
"""
        
        summary = await self.chat(prompt, temperature=0.3)
        return summary.strip()
