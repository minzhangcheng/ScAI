"""Tests for _TPL_NOOP_INSTALL template constant in sandbox.py."""


class TestTplNoopInstall:
    """The _TPL_NOOP_INSTALL template must exist, contain the unwind+install
    boilerplate, and be safe to embed in f-strings (no bare {})."""

    def test_tpl_noop_install_exists(self):
        from backend.utils.sandbox import _TPL_NOOP_INSTALL
        assert _TPL_NOOP_INSTALL is not None
        assert isinstance(_TPL_NOOP_INSTALL, str)
        assert len(_TPL_NOOP_INSTALL) > 100

    def test_tpl_contains_unwind_logic(self):
        from backend.utils.sandbox import _TPL_NOOP_INSTALL
        assert "_unwrap_close_guard" in _TPL_NOOP_INSTALL
        assert "_unwrap_show_guard" in _TPL_NOOP_INSTALL
        assert "_scai_noop" in _TPL_NOOP_INSTALL

    def test_tpl_contains_baseline_defensive_restore(self):
        from backend.utils.sandbox import _TPL_NOOP_INSTALL
        assert "_original_plt_close" in _TPL_NOOP_INSTALL
        assert "_capturing_show" in _TPL_NOOP_INSTALL

    def test_tpl_contains_noop_install(self):
        from backend.utils.sandbox import _TPL_NOOP_INSTALL
        assert "_noop_close" in _TPL_NOOP_INSTALL
        assert "_noop_show" in _TPL_NOOP_INSTALL
        assert "plt.close = _noop_close" in _TPL_NOOP_INSTALL
        assert "plt.show = _noop_show" in _TPL_NOOP_INSTALL

    def test_tpl_saves_orig_refs(self):
        from backend.utils.sandbox import _TPL_NOOP_INSTALL
        assert "_orig_close = plt.close" in _TPL_NOOP_INSTALL
        assert "_orig_show = plt.show" in _TPL_NOOP_INSTALL

    def test_tpl_no_bare_braces(self):
        from backend.utils.sandbox import _TPL_NOOP_INSTALL
        assert "{" not in _TPL_NOOP_INSTALL
        assert "}" not in _TPL_NOOP_INSTALL


class TestBioinformaticsAgentConsumesTplNoopInstall:
    """bioinformatics_agent.py must import and use _TPL_NOOP_INSTALL."""

    def test_bioinformatics_agent_imports_tpl(self):
        from pathlib import Path
        src = Path("backend/agents/bioinformatics_agent.py").read_text()
        assert "_TPL_NOOP_INSTALL" in src, "bioinformatics_agent.py must import _TPL_NOOP_INSTALL"

    def test_bioinformatics_agent_no_inline_unwind(self):
        """The inline unwind+install boilerplate should be gone."""
        from pathlib import Path
        src = Path("backend/agents/bioinformatics_agent.py").read_text()
        # After refactor, the raw inline unwind code is gone;
        # only the {_TPL_NOOP_INSTALL} reference remains (inside f-strings).
        # The string literal "_unwrap_close_guard = 0" should NOT appear
        # as raw code in this file (it's in the template constant in sandbox.py).
        count = src.count("_unwrap_close_guard = 0")
        assert count == 0, f"Expected 0 inline unwind in bioinformatics_agent.py, got {count}"


class TestChatConsumesTplNoopInstall:
    """chat.py must import and use _TPL_NOOP_INSTALL."""

    def test_chat_imports_tpl(self):
        from pathlib import Path
        src = Path("backend/routes/chat.py").read_text()
        assert "_TPL_NOOP_INSTALL" in src, "chat.py must import _TPL_NOOP_INSTALL"

    def test_chat_no_inline_unwind(self):
        from pathlib import Path
        src = Path("backend/routes/chat.py").read_text()
        count = src.count("_unwrap_close_guard = 0")
        assert count == 0, f"Expected 0 inline unwind in chat.py, got {count}"


class TestBioinfoAgentNoForwardRef:
    """Optional['asyncio.Event'] should not use string forward ref since
    asyncio is already imported at module level."""

    def test_no_forward_ref_on_pause_event(self):
        from pathlib import Path
        src = Path("backend/agents/bioinformatics_agent.py").read_text()
        assert "Optional['asyncio.Event']" not in src, \
            "Unnecessary forward ref: asyncio is imported at module top"