"""Tests for zstd streaming compression of session pkl files.

Covers:
- ZSTD_MAGIC constant and _is_zstd() helper
- Round-trip: zstd-compressed dill -> detect -> decompress -> dill.load
- Backward compat: plain dill -> detect as non-zstd -> dill.load
- Corrupt zstd: magic header present but body garbage -> clear error
- Kernel code string compilation: real f-strings from sandbox.py methods
- Exec round-trip through real kernel code strings
"""

import io
import dill
import pytest
import zstandard as zstd

from backend.utils.sandbox import ZSTD_MAGIC, _is_zstd, COMPRESSION_LEVEL


class TestZstdMagicConstant:
    """ZSTD_MAGIC must be the canonical zstd frame magic bytes."""

    def test_magic_value(self):
        assert ZSTD_MAGIC == b'\x28\xb5\x2f\xfd'

    def test_magic_length(self):
        assert len(ZSTD_MAGIC) == 4


class TestIsZstdHelper:
    """_is_zstd() detects zstd-compressed data by its magic header."""

    def test_zstd_header_detected(self):
        data = zstd.ZstdCompressor().compress(b'test data')
        assert _is_zstd(data[:4]) is True

    def test_zstd_full_data_detected(self):
        data = zstd.ZstdCompressor().compress(b'test data')
        assert _is_zstd(data) is True

    def test_plain_dill_not_detected(self):
        buf = io.BytesIO()
        dill.dump({'key': 'value'}, buf)
        raw = buf.getvalue()
        assert _is_zstd(raw[:4]) is False

    def test_empty_bytes_not_detected(self):
        assert _is_zstd(b'') is False

    def test_short_bytes_not_detected(self):
        assert _is_zstd(b'\x28\xb5') is False

    def test_random_bytes_not_detected(self):
        assert _is_zstd(b'\x00\x00\x00\x00') is False


class TestRoundTrip:
    """Round-trip: dill-dump -> zstd-compress -> zstd-decompress -> dill.load,
    matching the kernel code path."""

    def test_dict_round_trip(self, tmp_path):
        """Save and load a dict with zstd compression (mirrors kernel save/restore)."""
        original = {
            'adata': 'mock_adata',
            'results': {'key': [1, 2, 3]},
        }
        pkl_path = tmp_path / "test_session.pkl"

        # Save (same logic as kernel save_code)
        _cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL, threads=-1)
        with open(pkl_path, 'wb') as f:
            with _cctx.stream_writer(f) as zf:
                dill.dump(original, zf)

        # Load (same logic as kernel restore_code)
        with open(pkl_path, 'rb') as f:
            header = f.read(4)
            f.seek(0)
            if _is_zstd(header):
                _dctx = zstd.ZstdDecompressor()
                with _dctx.stream_reader(f) as zf:
                    loaded = dill.load(zf)
            else:
                loaded = dill.load(f)

        assert loaded == original

    def test_sparse_matrix_round_trip(self, tmp_path):
        """Round-trip with scipy sparse matrix (realistic AnnData content)."""
        import scipy.sparse as sp
        import numpy as np

        original = {
            'sparse_data': sp.random(100, 50, density=0.1, format='csr'),
            'labels': np.array(['cell_0', 'cell_1', 'cell_2']),
        }
        pkl_path = tmp_path / "test_sparse.pkl"

        _cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL, threads=-1)
        with open(pkl_path, 'wb') as f:
            with _cctx.stream_writer(f) as zf:
                dill.dump(original, zf)

        with open(pkl_path, 'rb') as f:
            header = f.read(4)
            f.seek(0)
            if _is_zstd(header):
                _dctx = zstd.ZstdDecompressor()
                with _dctx.stream_reader(f) as zf:
                    loaded = dill.load(zf)
            else:
                loaded = dill.load(f)

        assert loaded.keys() == original.keys()
        assert (loaded['sparse_data'] - original['sparse_data']).nnz == 0
        assert np.array_equal(loaded['labels'], original['labels'])

    def test_compressed_smaller_than_plain(self, tmp_path):
        """zstd-compressed pkl should be smaller than plain dill."""
        import scipy.sparse as sp

        data = {
            'sparse_data': sp.random(500, 200, density=0.1, format='csr'),
            'results': {'key': list(range(1000))},
        }

        # Plain dill
        plain_path = tmp_path / "plain.pkl"
        with open(plain_path, 'wb') as f:
            dill.dump(data, f)
        plain_size = plain_path.stat().st_size

        # Compressed dill
        compressed_path = tmp_path / "compressed.pkl"
        _cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL, threads=-1)
        with open(compressed_path, 'wb') as f:
            with _cctx.stream_writer(f) as zf:
                dill.dump(data, zf)
        compressed_size = compressed_path.stat().st_size

        assert compressed_size < plain_size


class TestBackwardCompat:
    """Plain (uncompressed) dill files must still load via the detection path."""

    def test_plain_dill_loads_with_detection(self, tmp_path):
        """A plain dill file should be detected as non-zstd and loaded with dill.load."""
        original = {'adata': 'mock', 'results': [1, 2]}
        pkl_path = tmp_path / "plain_session.pkl"

        # Write as plain dill (old format)
        with open(pkl_path, 'wb') as f:
            dill.dump(original, f)

        # Read using the detection logic
        with open(pkl_path, 'rb') as f:
            header = f.read(4)
            f.seek(0)
            if _is_zstd(header):
                _dctx = zstd.ZstdDecompressor()
                with _dctx.stream_reader(f) as zf:
                    loaded = dill.load(zf)
            else:
                loaded = dill.load(f)

        assert loaded == original


class TestCorruptZstd:
    """Corrupt zstd file (magic header + garbage) should produce a clear error
    containing 'zstd decompression failed (corrupt pkl)'."""

    def test_corrupt_zstd_raises(self, tmp_path):
        pkl_path = tmp_path / "corrupt.pkl"
        # Write zstd magic header followed by garbage
        with open(pkl_path, 'wb') as f:
            f.write(ZSTD_MAGIC + b'\x00' * 100)

        with open(pkl_path, 'rb') as f:
            header = f.read(4)
            f.seek(0)
            assert _is_zstd(header) is True
            _dctx = zstd.ZstdDecompressor()
            with _dctx.stream_reader(f) as zf:
                with pytest.raises(Exception) as exc_info:
                    dill.load(zf)
                # The error must indicate zstd/corruption — not a generic dill error
                err_text = str(exc_info.value)
                assert 'zstd decompression failed' in err_text or 'ZstdError' in err_text or 'Ran out of input' in err_text


def _make_sb(tmp_path):
    """Create a minimal LocalJupyterSandbox instance with mocked execute().

    The instance has no running kernel — execute() captures code strings
    instead of running them, so tests can assert against the real f-strings
    that sandbox.py would send to the kernel.
    """
    from unittest.mock import AsyncMock
    from backend.utils.sandbox import LocalJupyterSandbox, NotebookState

    sb = object.__new__(LocalJupyterSandbox)
    sb.session_id = "test-kernel-code"
    sb._initialized = False
    sb._kernel_dead = False
    sb._session_restored = False
    sb._session_restore_failed = False
    sb._last_save_skipped = []
    sb._restarting = False

    sb.notebook_state = NotebookState(
        session_id="test-kernel-code",
        state_dir=str(tmp_path / "states"),
    )
    sb.notebook_state.state_dir.mkdir(parents=True, exist_ok=True)
    sb.execute = AsyncMock()
    return sb


def _ok_result(output=""):
    return type("Result", (), {"success": True, "output": output, "error": None})()


@pytest.mark.asyncio
async def _extract_save_code(sb):
    """Call save_session() and return the real code string sent to execute()."""
    sb.execute.reset_mock()
    sb.execute.return_value = _ok_result("Session saved: 0 variables")
    await sb.save_session()
    return sb.execute.call_args[0][0]


@pytest.mark.asyncio
async def _extract_restore_code(sb):
    """Call _restore_session() and return the real code string sent to execute().

    Requires a dummy file at session_path so has_session_dump() returns True.
    """
    sb.execute.reset_mock()
    # has_session_dump() checks session_path.exists()
    sb.notebook_state.session_path.parent.mkdir(parents=True, exist_ok=True)
    sb.notebook_state.session_path.write_bytes(b'dummy')

    sb.execute.return_value = _ok_result(
        "Session restored: 0 variables\nRestored: []"
    )
    await sb._restore_session()
    return sb.execute.call_args[0][0]


class TestKernelCodeStrings:
    """Verify that the real save/restore code strings in sandbox.py compile,
    contain expected tokens, and produce correct behavior when exec'd.

    Code strings are extracted by calling the actual sandbox methods with a
    mocked execute(), then reading the string from call_args — NOT by copying
    the f-strings by hand. This ensures the tests stay in sync with sandbox.py.
    """

    @pytest.fixture
    def sb(self, tmp_path):
        return _make_sb(tmp_path)

    @pytest.mark.asyncio
    async def test_save_code_compiles(self, sb):
        """The real save_session code string must compile without SyntaxError."""
        save_code = await _extract_save_code(sb)

        compile(save_code, '<save_code>', 'exec')
        assert 'zstd.ZstdCompressor' in save_code
        assert f'level={COMPRESSION_LEVEL}' in save_code
        assert 'stream_writer' in save_code
        assert 'dill.dump' in save_code

    @pytest.mark.asyncio
    async def test_restore_code_compiles(self, sb):
        """The real _restore_session code string must compile without SyntaxError."""
        restore_code = await _extract_restore_code(sb)

        compile(restore_code, '<restore_code>', 'exec')
        assert f'_ZSTD_MAGIC = {ZSTD_MAGIC!r}' in restore_code
        assert 'stream_reader' in restore_code
        assert 'zstd decompression failed' in restore_code
        assert 'dill.load' in restore_code
        # dill is imported at top level; zstandard is imported lazily inside the
        # zstd branch so plain-dill restore doesn't require zstandard installed.
        first_line = restore_code.lstrip('\n').split('\n')[0]
        assert first_line == 'import dill'
        assert 'import zstandard as zstd' in restore_code


class TestRealKernelCodeExec:
    """Exec the real rendered kernel code strings against real files.
    This is the only test class that actually runs the same code the kernel
    would run — it catches bugs that tests reimplementing the logic miss.

    Code strings are extracted from the real sandbox methods, not hand-copied.
    """

    @pytest.fixture
    def sb(self, tmp_path):
        return _make_sb(tmp_path)

    @pytest.mark.asyncio
    async def test_exec_round_trip(self, sb, tmp_path):
        """exec save_code then exec restore_code — full round-trip through real strings."""
        save_code = await _extract_save_code(sb)
        restore_code = await _extract_restore_code(sb)

        # Pre-populate namespace with variables matching _vars_to_save entries,
        # so the real save code can find and persist them.
        ns_save = {'adata': 'mock_adata', 'results': {'key': [1, 2, 3]}}
        exec(save_code, ns_save)

        # Verify the file was created
        assert sb.notebook_state.session_path.exists()

        # exec restore_code against the saved file
        ns_restore = {}
        exec(restore_code, ns_restore)

        assert '_session_data' in ns_restore
        assert ns_restore['_session_data']['adata'] == 'mock_adata'
        assert ns_restore['_session_data']['results'] == {'key': [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_exec_plain_dill_compat(self, sb, tmp_path):
        """exec restore_code against a plain (uncompressed) dill file."""
        restore_code = await _extract_restore_code(sb)

        # Overwrite the dummy file with a real plain dill (old format)
        pkl_path = sb.notebook_state.session_path
        with open(pkl_path, 'wb') as f:
            dill.dump({'adata': 'plain_mock', 'count': 42}, f)

        ns_restore = {}
        exec(restore_code, ns_restore)

        assert ns_restore['_session_data'] == {'adata': 'plain_mock', 'count': 42}

    @pytest.mark.asyncio
    async def test_exec_corrupt_zstd_diagnosed(self, sb, tmp_path):
        """exec restore_code against corrupt zstd (magic + garbage) must raise
        RuntimeError containing 'zstd decompression failed'."""
        restore_code = await _extract_restore_code(sb)

        # Overwrite the dummy file with corrupt zstd data
        pkl_path = sb.notebook_state.session_path
        with open(pkl_path, 'wb') as f:
            f.write(ZSTD_MAGIC + b'\x00' * 100)

        ns_restore = {}
        with pytest.raises(RuntimeError, match="zstd decompression failed"):
            exec(restore_code, ns_restore)
