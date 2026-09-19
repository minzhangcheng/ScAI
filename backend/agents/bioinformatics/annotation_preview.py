"""Read-only candidate-label UMAP for the annotation decision report."""
from __future__ import annotations

import base64
import json
import logging
import uuid

logger = logging.getLogger(__name__)
_MARKER = "__SCAI_ANNOTATION_PREVIEW__"


def candidate_annotation_preview_code(source, labels):
    """Use the annotate_clusters call's exact source, never infer another dataset."""
    function = f"_scai_annotation_preview_{uuid.uuid4().hex}"
    return f'''
def {function}():
    import json
    import io
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    source = json.loads({json.dumps(json.dumps(source))})
    labels = json.loads({json.dumps(json.dumps(labels))})
    def unavailable(reason):
        print({_MARKER!r} + json.dumps({{"status": "unavailable", "reason": reason}}))
    adata = globals().get(source["adata_var"])
    if adata is None or not hasattr(adata, "obs") or not hasattr(adata, "obsm"):
        return unavailable("The annotation source object is unavailable.")
    if source["cluster_key"] not in adata.obs or "X_umap" not in adata.obsm:
        return unavailable("The annotation source has no retained UMAP coordinates or clustering column.")
    coordinates = np.asarray(adata.obsm["X_umap"])
    if coordinates.ndim != 2 or coordinates.shape[0] != adata.n_obs or coordinates.shape[1] < 2:
        return unavailable("The retained UMAP coordinates do not match the annotation source.")
    clusters = adata.obs[source["cluster_key"]].astype(str).to_numpy()
    finite = np.isfinite(coordinates[:, :2]).all(axis=1)
    if not finite.any():
        return unavailable("No finite retained UMAP coordinates are available.")
    groups = list(dict.fromkeys(clusters[finite]))
    figure, axes = plt.subplots(figsize=(11, max(5, min(16, len(groups) * 0.27))))
    try:
        colors = plt.get_cmap("tab20", max(len(groups), 1))
        for index, cluster in enumerate(groups):
            mask = (clusters == cluster) & finite
            label = str(cluster) + ": " + labels.get(str(cluster), "Unmapped candidate")
            axes.scatter(coordinates[mask, 0], coordinates[mask, 1], s=3,
                         color=colors(index), label=label, rasterized=True)
        axes.set(xlabel="UMAP 1", ylabel="UMAP 2", title="Candidate annotations — awaiting confirmation")
        axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8,
                    markerscale=3, frameon=False)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        display(Image(data=buffer.getvalue()))
        print({_MARKER!r} + json.dumps({{"status": "ready", "omitted_nonfinite": int((~finite).sum())}}))
    finally:
        plt.close(figure)
try:
    {function}()
finally:
    del {function}
'''


async def render_candidate_annotation_preview(sandbox, source, annotations):
    """Return report-only image bytes; do not register a publication artifact."""
    unavailable = {"note": "Candidate annotation UMAP preview is unavailable.", "image": None}
    if (sandbox is None or not isinstance(source, dict)
            or not all(isinstance(source.get(key), str) and source[key]
                       for key in ("adata_var", "cluster_key"))):
        return unavailable
    labels = {item.cluster_id: item.cell_type for item in annotations}
    try:
        result = await sandbox.execute(
            candidate_annotation_preview_code(source, labels),
            skip_safety_check=True, mutates_namespace=False,
        )
        if getattr(result, "error", None):
            return unavailable
        metadata = None
        for line in (getattr(result, "output", "") or "").splitlines():
            if line.startswith(_MARKER):
                metadata = json.loads(line[len(_MARKER):])
        if not metadata or metadata.get("status") != "ready":
            return {**unavailable, "note": (metadata or {}).get("reason") or unavailable["note"]}
        figures = getattr(result, "figures", []) or []
        if len(figures) != 1:
            return unavailable
        payload = base64.b64decode(figures[0], validate=True)
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            return unavailable
        note = "Candidate labels on retained UMAP coordinates; awaiting user confirmation."
        if metadata.get("omitted_nonfinite"):
            note += f" {metadata['omitted_nonfinite']} cells had non-finite coordinates and are not shown."
        return {"image": "data:image/png;base64," + base64.b64encode(payload).decode(), "note": note}
    except Exception:
        logger.warning("Candidate annotation preview could not be rendered", exc_info=True)
        return unavailable
