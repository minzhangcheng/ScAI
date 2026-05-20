"""Report variant generators - HTML report builders for different analysis phases."""
from backend.utils.image_utils import compress_image_to_base64
from .generator import DetailedReportGenerator
from .variants import (
    generate_exploration_report_html,
    generate_detailed_report_segment_html,
    generate_combined_detailed_report_html,
    generate_annotation_context_report_html,
)
from .regenerator import generate_html_report

__all__ = [
    "DetailedReportGenerator",
    "compress_image_to_base64",
    "generate_exploration_report_html",
    "generate_detailed_report_segment_html",
    "generate_combined_detailed_report_html",
    "generate_annotation_context_report_html",
    "generate_html_report",
]