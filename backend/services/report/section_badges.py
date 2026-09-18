"""Shared chapter labels for live, fallback and regenerated report HTML."""


def section_badge(section):
    """Label round-owned chapters consistently in the TOC and body."""
    section_id = section.get("id", "")
    if section_id == "exploration_findings":
        return "general", "General / Exploration"
    if section_id.startswith("direction_"):
        return "specific", "Specific / Innovation"
    analysis_type = section.get("analysis_type", "general")
    type_class = "general" if analysis_type == "general" else "specific"
    if section_id.startswith("integration_gate_") or section_id == "integrated_findings":
        return type_class, "Integration"
    return type_class, "General" if type_class == "general" else "Specific"
