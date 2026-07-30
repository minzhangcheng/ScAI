"""decoupler must stay out of the resolved dependency graph.

decoupler 1.6.0 calls ``np.arange(start=..., stop=..., step=..., dtype=...)``
inside an eagerly signature-compiled ``@nb.njit`` (method_gsva.py,
method_aucell.py). numba 0.61 rejects the positional-only ``start`` passed as a
keyword, and the explicit signature forces compilation at import time, so
``import decoupler`` itself raises TypingError -- taking down CollecTRI,
PROGENy, DoRothEA, OmniPath resource access and every TF-activity method.

decoupler 1.7.0+ fixed the call, but 1.8.0 still declares ``numba<0.61`` and
1.9.0+ requires ``numpy>=2``. Both would force pip to move pins this project
deliberately holds, so decoupler is installed with ``--no-deps`` from
requirements-nodeps.txt after the main graph resolves.

These tests pin that arrangement: if someone moves decoupler back into
requirements.txt, pip will silently downgrade numba and the runtime breaks in a
way that only shows up as a mid-analysis tool failure.
"""
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_REQUIREMENTS = _REPO / "requirements.txt"
_NODEPS = _REPO / "requirements-nodeps.txt"


def _requirement_lines(path: Path) -> list[str]:
    """Non-comment, non-blank lines of a requirements file."""
    if not path.exists():
        return []
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def test_nodeps_requirements_file_exists():
    assert _NODEPS.exists(), (
        "requirements-nodeps.txt is missing; decoupler would fall back into "
        "the resolved graph and drag numba below 0.61."
    )


def test_decoupler_not_in_main_requirements():
    """A bare decoupler pin here makes pip downgrade numba (or fail to resolve)."""
    offenders = [ln for ln in _requirement_lines(_REQUIREMENTS)
                 if ln.lower().startswith("decoupler")]
    assert not offenders, (
        f"decoupler must not be an installable requirement in requirements.txt "
        f"(found {offenders!r}); it belongs in requirements-nodeps.txt."
    )


def test_decoupler_pinned_in_nodeps_at_or_above_1_7():
    """1.6.x cannot import under numba 0.61; 1.9+ needs numpy 2."""
    lines = [ln for ln in _requirement_lines(_NODEPS)
             if ln.lower().startswith("decoupler")]
    assert len(lines) == 1, f"expected exactly one decoupler pin, got {lines!r}"
    spec = lines[0]
    assert "==" in spec, f"decoupler must be pinned exactly, got {spec!r}"
    version = spec.split("==", 1)[1].strip()
    parts = version.split(".")
    major, minor = int(parts[0]), int(parts[1])
    assert (major, minor) >= (1, 7), (
        f"decoupler {version} still contains the keyword-arange njit that "
        f"fails to import under numba 0.61"
    )
    assert major < 2 and (major, minor) < (1, 9), (
        f"decoupler {version} requires numpy>=2, which conflicts with the "
        f"pinned numpy 1.26.4"
    )


def test_numba_and_numpy_pins_still_match_the_rationale():
    """If these move, the requirements-nodeps.txt rationale must be re-checked."""
    lines = _requirement_lines(_REQUIREMENTS)
    assert "numba==0.61.2" in lines, (
        "numba pin changed; re-verify the decoupler --no-deps rationale in "
        "requirements-nodeps.txt"
    )
    assert "numpy==1.26.4" in lines, (
        "numpy pin changed; re-verify the decoupler --no-deps rationale in "
        "requirements-nodeps.txt"
    )


@pytest.mark.parametrize("script", ["start.sh", "docker-entrypoint.sh"])
def test_installer_scripts_apply_nodeps_requirements(script):
    """Both entry points must install the --no-deps set, or the fix is a no-op."""
    src = (_REPO / script).read_text()
    assert "requirements-nodeps.txt" in src, (
        f"{script} does not install requirements-nodeps.txt"
    )
    assert "--no-deps" in src, f"{script} is missing the --no-deps install step"


def test_prompts_do_not_reference_the_2x_decoupler_api():
    """``decoupler.op.*`` is the >=2.0 namespace; on 1.x it raises on lookup.

    ScAI shipped ``decoupler.op.collectri`` as *the* worked example for
    lookup_documentation, so the one example the model was given for querying a
    curated resource could never succeed.
    """
    backend = _REPO / "backend"
    offenders = []
    for path in backend.rglob("*.py"):
        if "tests" in path.parts:
            continue
        if "decoupler.op." in path.read_text():
            offenders.append(str(path.relative_to(_REPO)))
    assert not offenders, (
        f"decoupler.op.* is the 2.x API but 1.x is pinned; found in {offenders}"
    )
