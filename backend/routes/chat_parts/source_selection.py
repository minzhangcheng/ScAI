"""PI-owned AnnData source decision for Chat work on a restored kernel."""
from __future__ import annotations

from typing import Any, Dict

from backend.models.schemas import AdataSourceSelection
from backend.services.adata_checkpoints import build_prompt_catalog_text


class ChatSourceSelectionError(RuntimeError):
    """No technically usable PI source decision was available."""


async def select_chat_source(pi_agent, sandbox, state, request: str) -> Dict[str, Any]:
    """Give the PI current kernel facts, then validate its explicit selection."""
    async with sandbox._exec_lock:
        inventory = await sandbox.inspect_checkpoint_adata_locked()
    if not isinstance(inventory, dict) or inventory.get("error"):
        raise ChatSourceSelectionError("The live AnnData inventory is unavailable")
    candidates = {
        str(item.get("name")): item
        for item in inventory.get("candidates") or []
        if isinstance(item, dict) and item.get("name")
    }
    previous = dict(
        (getattr(state, "sandbox_state", None) or {}).get(
            "resolved_adata_source_context_v1"
        ) or {}
    )
    preferred = str(previous.get("active_adata_var") or "")
    visible = list(candidates.values())[:40]
    if preferred in candidates and preferred not in {
        item.get("name") for item in visible
    }:
        visible[-1:] = [candidates[preferred]]
    prompt_inventory = {
        "canonical_present": inventory.get("canonical_present"),
        "candidates": [
            {
                "name": item.get("name"),
                "n_obs": item.get("n_obs"),
                "n_vars": item.get("n_vars"),
                "obs_columns": (item.get("obs_columns") or [])[:25],
                "obsm": (item.get("obsm") or [])[:10],
                "layers": (item.get("layers") or [])[:10],
            }
            for item in visible
        ],
        "omitted_candidates": max(0, len(candidates) - len(visible)),
    }
    catalog = build_prompt_catalog_text(state, max_entries=50)
    correction = ""
    for _ in range(2):
        decision = await pi_agent.select_chat_adata_source(
            request, prompt_inventory, catalog, previous, correction,
        )
        try:
            if not isinstance(decision, dict) or not isinstance(
                decision.get("rationale"), str
            ) or not decision["rationale"].strip():
                raise ValueError("PI did not provide a source rationale")
            if decision.get("uses_adata") is False:
                if decision.get("adata_source") is not None or decision.get(
                    "active_adata_var"
                ) is not None:
                    raise ValueError(
                        "A non-AnnData task must not select an AnnData source"
                    )
                return {
                    "selection": AdataSourceSelection().model_dump(mode="json"),
                    "active_adata_var": None,
                    "uses_adata": False,
                    "load_registered_inputs": False,
                    "rationale": decision["rationale"][:1000],
                    "candidate_names": list(candidates)[:40],
                }
            if decision.get("adata_source") is None:
                raise ChatSourceSelectionError(
                    "PI could not identify an unambiguous AnnData source: "
                    + decision["rationale"][:500]
                )
            selection = AdataSourceSelection.model_validate(decision["adata_source"])
            active_var = decision.get("active_adata_var")
            if selection.mode == "current":
                if active_var is not None and (
                    not isinstance(active_var, str) or active_var not in candidates
                ):
                    raise ValueError("The selected live AnnData variable is absent")
                if active_var is None and decision.get(
                    "load_registered_inputs"
                ) is not True:
                    raise ValueError(
                        "Select one live AnnData or explicitly choose registered inputs"
                    )
            elif active_var is not None:
                raise ValueError("A prepared source must not specify a live variable")
            return {
                "selection": selection.model_dump(mode="json"),
                "active_adata_var": active_var,
                "uses_adata": True,
                "load_registered_inputs": decision.get("load_registered_inputs") is True,
                "rationale": decision["rationale"][:1000],
                "candidate_names": list(candidates)[:40],
            }
        except ChatSourceSelectionError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            correction = (
                "\nYour previous source decision was technically invalid: "
                f"{exc}. Reconsider the listed candidates and return corrected JSON."
            )
    raise ChatSourceSelectionError("PI did not return a valid AnnData source decision")
