#!/usr/bin/env python3
"""
Regenerate HTML report with embedded base64 images.
This script takes an existing report folder and regenerates the HTML file
with all figures embedded as base64 for standalone viewing.
"""
import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for `backend` package resolution
# when run directly: python backend/regenerate_report_html.py <dir>
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Regenerate HTML report with embedded base64 images."
    )
    parser.add_argument(
        "report_directory",
        type=Path,
        help="Path to the report directory containing report.json"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language for the report (default: en)"
    )
    args = parser.parse_args()

    if not args.report_directory.exists():
        logger.error("Error: Directory %s does not exist", args.report_directory)
        sys.exit(1)

    from backend.services.report.regenerator import generate_html_report
    generate_html_report(args.report_directory, language=args.language)