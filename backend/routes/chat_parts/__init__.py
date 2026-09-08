"""Modules split out of chat.py."""
from .core import (  # noqa: F401
    logger,
    router,
    _build_chat_context,
)
from .figures import (  # noqa: F401
    _build_figure_setup_code,
    _generate_figures_in_chat,
    _collect_session_figures,
)
from .delegation import (  # noqa: F401
    _send_chat_ws,
    _try_parse_delegation,
    _build_synthesis_prompt,
    _execute_chat_agent,
    _run_chat_delegation,
)
from .endpoints import (  # noqa: F401
    chat_with_pi,
    cancel_chat,
    get_chat_history,
)
