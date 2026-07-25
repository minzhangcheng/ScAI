"""Report variants split out of variants.py."""
from .renderers import (  # noqa: F401
    _render_score_badge,
    _render_direction_evidence_section_from_lists,
    _render_direction_evidence_section,
    _render_direction_candidates_rows,
)
from .exploration import (  # noqa: F401
    generate_exploration_report_html,
)
from .detailed import (  # noqa: F401
    generate_detailed_report_segment_html,
    _deduplicate_detailed_figure_blocks,
    generate_combined_detailed_report_html,
)
from .integration import (  # noqa: F401
    _inline_session_text,
    _render_integration_section,
    _render_novelty_score,
    _render_evidence_strength,
    _render_integration_candidates_rows,
    _render_integration_evidence_section,
    generate_integration_report_html,
)
from .annotation import (  # noqa: F401
    generate_annotation_context_report_html,
)
