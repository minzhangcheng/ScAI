"""`is_live_runtime_continuation_safe` reaches its part modules via the facade.

pyflakes reports this as two errors that look like a defect and are not: the
facade "imports but does not use" the helper, and the part module uses an
"undefined name". Both are the split-facade pattern -- `bind_facade` copies the
facade's namespace into the part at call time.

The test exists because the obvious tidy-up is wrong. Deleting the facade's
"unused" import to silence pyflakes removes the only thing that supplies the
name, and the resulting NameError fires only on the resume path that reads it.
"""

def test_facade_supplies_the_pairing_helper_to_its_parts():
    """pyflakes calls these undefined; confirm the facade really supplies them.

    `bind_facade` runs lazily inside each route wrapper, not at import, so the
    check is that the name is on the facade for it to copy -- and then that
    copying it actually binds it.
    """
    import sys

    import backend.routes.lifecycle as lifecycle
    import backend.routes.lifecycle_parts.run_resume as run_resume
    import backend.workflow.core as core
    import backend.workflow.core_parts.execution as core_execution

    for facade, part in ((lifecycle, run_resume), (core, core_execution)):
        assert hasattr(facade, "is_live_runtime_continuation_safe"), (
            f"{facade.__name__} does not import the helper, so bind_facade "
            f"cannot supply it to {part.__name__}"
        )

    run_resume.bind_facade(sys.modules[lifecycle.__name__])
    assert hasattr(run_resume, "is_live_runtime_continuation_safe")

    core_execution.bind_facade(sys.modules[core.__name__])
    assert hasattr(core_execution, "is_live_runtime_continuation_safe")
