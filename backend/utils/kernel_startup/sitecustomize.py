"""Apply ScAI kernel isolation before IPython imports user-facing libraries."""

from scai_kernel_isolation import activate_from_environment, install_process_guard


activate_from_environment()
install_process_guard()

# Cache the publication-finalization trust module while only the trusted
# startup directory is on the import path.  The notebook workspace is added to
# ``sys.path`` later, during kernel initialization; importing this module only
# after that point would let a pre-existing workspace file with the same name
# replace the bootstrap-owned capability store.
import scai_kernel_trust as _scai_kernel_trust  # noqa: E402,F401
import scai_pydeseq2_compat as _scai_pydeseq2_compat  # noqa: E402,F401
import scai_scanpy_compat as _scai_scanpy_compat  # noqa: E402,F401
import scai_arboreto_compat as _scai_arboreto_compat  # noqa: E402,F401
