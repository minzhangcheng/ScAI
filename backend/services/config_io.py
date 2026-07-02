"""Configuration import/export core functions.

apply_config_to_session: Create a new AnalysisState from a ConfigV1 object.
export_session_to_config: Export an existing AnalysisState to ConfigV1 JSON.
"""
import uuid
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from backend.models.config_schema import ConfigV1, AgentOverride, FileEntry, GroupEntry, IndicatorEntry, AGENT_NAMES
from backend.models.schemas import AnalysisState, TaskStatus, TreatmentGroup, DataFile, BiochemistryIndicator


def build_agent_server_configs(config: ConfigV1) -> Dict[str, Dict[str, Any]]:
    """Build per-agent LLM server config dict (base_url, api_key, model, temperature)."""
    default_llm = config.api_config.default_llm
    result: Dict[str, Dict[str, Any]] = {}
    for name in AGENT_NAMES:
        ov = (config.agents or {}).get(name) or AgentOverride()
        result[name] = {
            "base_url": ov.llm_base_url or default_llm.llm_base_url,
            "api_key": ov.api_key or default_llm.api_key,
            "model": ov.model or default_llm.model,
            "temperature": float(ov.temperature if ov.temperature is not None else 0.7),
        }
    return result


def apply_config_to_session(config: ConfigV1, user_id: str = "") -> AnalysisState:
    """Create a new pending AnalysisState from a ConfigV1 configuration.

    Returns a session with all config fields populated but no files uploaded.
    Treatment groups get new UUIDs; files get placeholder entries.
    """
    # Map group keys -> UUIDs
    group_key_to_uuid: Dict[str, str] = {}
    treatment_groups: List[TreatmentGroup] = []
    for g in config.data.groups:
        gid = str(uuid.uuid4())
        group_key_to_uuid[g.key] = gid
        treatment_groups.append(TreatmentGroup(
            id=gid,
            name=g.name,
            description=g.description,
            sample_count=g.sample_count,
            order=g.order,
        ))

    # Create placeholder data files (user must upload actual data later)
    data_files: List[DataFile] = []
    for f in config.data.files:
        group_uuid = group_key_to_uuid.get(f.group) if f.group else None
        data_files.append(DataFile(
            id=str(uuid.uuid4()),
            filename=f.filename or Path(f.local_path).name,
            file_path="",  # Placeholder — user must upload
            section=f.section,
            data_type=f.data_type,
            group_id=group_uuid,
            sample_number=f.sample_number,
            description=f.description,
        ))

    # Create biochemistry indicators
    indicators: List[BiochemistryIndicator] = []
    for ind in config.data.indicators:
        group_uuid = group_key_to_uuid.get(ind.group) if ind.group else None
        indicators.append(BiochemistryIndicator(
            id=str(uuid.uuid4()),
            indicator=ind.indicator,
            value=ind.value,
            unit=ind.unit,
            reference_range=ind.reference_range,
            interpretation=ind.interpretation,
            group_id=group_uuid,
            sample_number=ind.sample_number,
        ))

    # Build literature config in the state
    lit = config.api_config.literature
    literature_config: Dict[str, Any] = {
        "mode": lit.mode,
        "deep_research_model": lit.deep_research_model or "",
        "deep_research_llm_base_url": lit.deep_research_llm_base_url or "",
        "deep_research_api_key": lit.deep_research_api_key or "",
        "google_api_key": lit.google_api_key or "",
        "pubmed_email": lit.pubmed_email or "",
        "max_search_rounds": lit.max_search_rounds or 10,
    }

    # Apply fallback server_config for ALL agents
    agent_server_configs = build_agent_server_configs(config)
    custom_models: Dict[str, str] = {}
    custom_prompts: Dict[str, str] = {}
    custom_temperatures: Dict[str, float] = {}
    agent_vision_settings: Dict[str, bool] = {}

    for name in AGENT_NAMES:
        ov = (config.agents or {}).get(name) or AgentOverride()
        if ov.model:
            custom_models[name] = ov.model
        if ov.custom_prompt:
            custom_prompts[name] = ov.custom_prompt
        if ov.temperature is not None:
            custom_temperatures[name] = ov.temperature
        if ov.vision is not None:
            agent_vision_settings[name] = ov.vision

    state = AnalysisState(
        id=str(uuid.uuid4()),
        user_id=user_id,
        name=config.analysis.name,
        initial_instruction=config.analysis.initial_instruction,
        round_wait_seconds=config.analysis.round_wait_seconds,
        status=TaskStatus.PENDING,
        treatment_groups=treatment_groups,
        data_files=data_files,
        biochemistry_indicators=indicators,
        custom_prompts=custom_prompts,
        custom_models=custom_models,
        custom_temperatures=custom_temperatures,
        agent_server_configs=agent_server_configs,
        literature_config=literature_config,
        agent_vision_settings=agent_vision_settings,
    )

    # Write config_snapshot after creation
    state.config_snapshot = config.model_dump(mode="json", exclude_none=False)

    return state


def export_session_to_config(state: AnalysisState) -> Dict[str, Any]:
    """Export an AnalysisState to a ConfigV1-compatible dict.

    - local_path is rewritten to uploads/<filename> (portable)
    - API keys are included as-is (user warned in frontend)
    - config_snapshot is excluded
    """
    # Build group UUID -> key mapping
    uuid_to_key: Dict[str, str] = {}

    groups: List[Dict[str, Any]] = []
    for g in state.treatment_groups:
        key = g.name.lower().replace(" ", "_").replace("-", "_")[:50]
        base_key = key
        counter = 1
        while key in uuid_to_key.values():
            key = f"{base_key}_{counter}"
            counter += 1
        uuid_to_key[g.id] = key
        groups.append({
            "key": key,
            "name": g.name,
            "description": g.description,
            "sample_count": g.sample_count,
            "order": g.order,
        })

    files: List[Dict[str, Any]] = []
    for f in state.data_files:
        entry: Dict[str, Any] = {
            "local_path": f"uploads/{f.filename}" if f.filename else "",
            "section": f.section,
            "data_type": f.data_type,
            "description": f.description or "",
        }
        if f.group_id and f.group_id in uuid_to_key:
            entry["group"] = uuid_to_key[f.group_id]
        if f.sample_number:
            entry["sample_number"] = f.sample_number
        if f.filename:
            entry["filename"] = f.filename
        files.append(entry)

    indicators: List[Dict[str, Any]] = []
    for ind in state.biochemistry_indicators:
        entry: Dict[str, Any] = {
            "indicator": ind.indicator,
            "value": ind.value,
            "unit": ind.unit,
        }
        if ind.reference_range:
            entry["reference_range"] = ind.reference_range
        if ind.interpretation:
            entry["interpretation"] = ind.interpretation
        if ind.group_id and ind.group_id in uuid_to_key:
            entry["group"] = uuid_to_key[ind.group_id]
        if ind.sample_number:
            entry["sample_number"] = ind.sample_number
        indicators.append(entry)

    # Agent overrides
    agents: Dict[str, Dict[str, Any]] = {}
    for agent_name in AGENT_NAMES:
        override: Dict[str, Any] = {}
        if agent_name in state.custom_models:
            override["model"] = state.custom_models[agent_name]
        if agent_name in state.agent_server_configs:
            cfg = state.agent_server_configs[agent_name]
            if "llm_base_url" in cfg or "base_url" in cfg:
                override["llm_base_url"] = cfg.get("llm_base_url") or cfg.get("base_url", "")
            if "api_key" in cfg:
                override["api_key"] = cfg["api_key"]
        if agent_name in state.custom_prompts:
            override["custom_prompt"] = state.custom_prompts[agent_name]
        if agent_name in state.custom_temperatures:
            override["temperature"] = state.custom_temperatures[agent_name]
        if agent_name in state.agent_vision_settings:
            override["vision"] = state.agent_vision_settings[agent_name]
        if override:
            agents[agent_name] = override

    # Infer default_llm from agent_server_configs using Counter
    pairs = [
        (c.get("base_url") or c.get("llm_base_url", ""), c.get("api_key", ""))
        for c in state.agent_server_configs.values()
        if c.get("base_url") or c.get("llm_base_url")
    ]
    global_llm_url = ""
    global_llm_key = ""
    global_proxy = None
    # Read proxy_url from config_snapshot (if available)
    if state.config_snapshot:
        api_cfg = state.config_snapshot.get("api_config", {})
        default_llm_snap = api_cfg.get("default_llm", {})
        if default_llm_snap.get("proxy_url"):
            global_proxy = default_llm_snap["proxy_url"]
    if pairs:
        (base, key), _ = Counter(pairs).most_common(1)[0]
        global_llm_url = base
        global_llm_key = key

    # Export literature_config nested fields
    lit = state.literature_config or {}
    literature_export = {
        "mode": lit.get("mode", "disabled") or "disabled",
        "deep_research_model": lit.get("deep_research_model") or None,
        "deep_research_llm_base_url": lit.get("deep_research_llm_base_url") or None,
        "deep_research_api_key": lit.get("deep_research_api_key") or None,
        "google_api_key": lit.get("google_api_key") or None,
        "pubmed_email": lit.get("pubmed_email") or None,
        "max_search_rounds": lit.get("max_search_rounds", 10),
    }

    # Build output
    config = {
        "schema_version": "1.0",
        "analysis": {
            "name": state.name,
            "initial_instruction": state.initial_instruction,
            "round_wait_seconds": state.round_wait_seconds,
        },
        "api_config": {
            "default_llm": {
                "llm_base_url": global_llm_url,
                "api_key": global_llm_key,
                "proxy_url": global_proxy,
            },
            "literature": literature_export,
        },
        "data": {
            "groups": groups,
            "files": files,
            "indicators": indicators,
        },
        "agents": agents,
    }

    # Validate against schema (raises if invalid)
    ConfigV1.model_validate(config)
    return config