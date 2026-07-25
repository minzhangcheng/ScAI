"""Route handlers split out of reports.py."""
from .core import (  # noqa: F401
    _should_skip_in_report_zip,
    _branch_query,
    _report_cache_key,
    _report_cache_key_for_lineage,
    router,
    _report_generation_pending,
    _pending_lock,
    _try_begin_generation,
    clear_generation_pending,
    clear_detailed_report_cache,
    _sanitize_nan,
    _load_report_json,
    _next_archive_version,
    _create_report_archive,
    _enforce_archive_retention,
)
from .html_views import (  # noqa: F401
    get_exploration_report_html,
    get_annotation_report_html,
    get_integration_report_html,
    get_detailed_report_html,
    get_detailed_report,
    generate_detailed_report,
    _generate_report_task,
)
from .downloads import (  # noqa: F401
    download_all_figures,
    download_all_tables,
    download_reproducibility_bundle,
    download_full_report,
    list_report_archives,
    download_report_archive,
)
from .assets import (  # noqa: F401
    get_report_html,
    preview_report,
    get_report_figure,
    get_report_figure_alt,
    get_report_table,
    get_report_table_alt,
    _resolve_report_file,
    get_report_file,
)
from .publication import (  # noqa: F401
    get_graphic_abstract,
    get_publication_figures,
    get_publication_tables,
    _find_workspace_figure,
    download_publication_figure_pdf,
    download_publication_figure_png,
    get_integrated_report,
    get_code_history,
    get_report,
    approve_partial_report,
)
