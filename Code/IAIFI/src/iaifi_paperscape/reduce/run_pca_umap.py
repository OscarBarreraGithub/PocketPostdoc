"""Step E: PCA denoising + UMAP dimensionality reduction.

Usage:
    python -m iaifi_paperscape.reduce.run_pca_umap [--incremental]

Loads SPECTER2 embeddings, selects the IAIFI + kNN-selected background subset,
applies PCA-50 (L2-normalized) then UMAP-2D.  Supports incremental updates
(transform-only for new papers) and full monthly refits with Procrustes
alignment and drift diagnostics.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import joblib
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from iaifi_paperscape.reduce.procrustes import align_to_previous, classify_drift
from iaifi_paperscape.utils.io import (
    load_json,
    load_numpy,
    load_yaml_config,
    save_json,
    save_numpy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "umap.yaml"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "interim" / "embeddings.npy"
EMBEDDING_INDEX_PATH = PROJECT_ROOT / "data" / "interim" / "embedding_index.json"
KNN_SELECTED_PATH = PROJECT_ROOT / "data" / "interim" / "knn_selected_ids.json"
KNN_SELECTED_PREV_PATH = PROJECT_ROOT / "data" / "interim" / "knn_selected_ids_prev.json"
IAIFI_METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "iaifi_metadata.jsonl"

PCA_OUT_PATH = PROJECT_ROOT / "data" / "interim" / "pca_50.npy"
PCA_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "pca_model.pkl"
UMAP_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "umap_model.pkl"
COORDS_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "coords_2d.npy"
PREV_COORDS_PATH = PROJECT_ROOT / "data" / "processed" / "coords_2d_prev.npy"
PREV_SELECTED_IDS_PATH = PROJECT_ROOT / "data" / "processed" / "selected_ids_prev.json"
SELECTED_IDS_PATH = PROJECT_ROOT / "data" / "processed" / "selected_ids.json"
DRIFT_DIAG_PATH = PROJECT_ROOT / "data" / "processed" / "drift_diagnostics.json"
PROCRUSTES_META_PATH = PROJECT_ROOT / "data" / "processed" / "procrustes_meta.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_row_to_arxiv_id(path: Path) -> list[str]:
    """Load the embedding index and return a list mapping row -> arXiv ID."""
    payload = load_json(path)
    if isinstance(payload, list):
        return [str(x) for x in payload]
    if isinstance(payload, dict):
        if "row_to_arxiv_id" in payload and isinstance(payload["row_to_arxiv_id"], list):
            return [str(x) for x in payload["row_to_arxiv_id"]]
        try:
            return [str(payload[str(i)]) for i in range(len(payload))]
        except KeyError:
            pass
        try:
            sorted_items = sorted(payload.items(), key=lambda kv: int(kv[0]))
            return [str(v) for _, v in sorted_items]
        except ValueError as exc:
            raise RuntimeError(
                "embedding_index.json has non-integer row keys; cannot map rows to arXiv IDs."
            ) from exc
    raise RuntimeError("Unsupported embedding index format.")


def _build_selected_mask(
    row_to_arxiv_id: list[str],
    knn_selected_ids: list[str],
    iaifi_count: int,
) -> np.ndarray:
    """Build a boolean mask selecting IAIFI rows + kNN-selected background rows."""
    knn_set = set(knn_selected_ids)
    n_total = len(row_to_arxiv_id)
    mask = np.zeros(n_total, dtype=bool)
    # All IAIFI papers (first iaifi_count rows)
    mask[:iaifi_count] = True
    # kNN-selected background papers
    for i in range(iaifi_count, n_total):
        if row_to_arxiv_id[i] in knn_set:
            mask[i] = True
    return mask


def _build_iaifi_anchor_mask(selected_mask: np.ndarray, iaifi_count: int) -> np.ndarray:
    """Build a boolean mask over the *selected* subset marking IAIFI papers.

    Returns an array of length selected_mask.sum() where the first
    iaifi_count True entries correspond to IAIFI papers.
    """
    # Within the selected subset, the first iaifi_count slots are IAIFI
    # (since all IAIFI papers are selected and they occupy the first rows).
    n_selected = int(selected_mask.sum())
    anchor = np.zeros(n_selected, dtype=bool)
    anchor[:iaifi_count] = True
    return anchor


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Step E: PCA + UMAP")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Use existing PCA/UMAP models to transform new papers only.",
    )
    args = parser.parse_args()

    config = load_yaml_config(CONFIG_PATH)
    pca_cfg = config.get("pca", {})
    umap_cfg = config.get("umap", {})
    update_cfg = config.get("update", {})

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    embeddings = load_numpy(EMBEDDINGS_PATH)
    row_to_arxiv_id = _load_row_to_arxiv_id(EMBEDDING_INDEX_PATH)
    knn_selected_ids = load_json(KNN_SELECTED_PATH)

    # Count IAIFI papers (they occupy the first N rows of embeddings)
    from iaifi_paperscape.utils.io import load_jsonl
    iaifi_count = len(load_jsonl(IAIFI_METADATA_PATH))

    print(
        f"Embeddings shape: {embeddings.shape}, "
        f"IAIFI count: {iaifi_count}, "
        f"kNN selected: {len(knn_selected_ids)}"
    )

    # Select only IAIFI + kNN-selected background embeddings
    selected_mask = _build_selected_mask(row_to_arxiv_id, knn_selected_ids, iaifi_count)
    selected_embeddings = embeddings[selected_mask]
    print(f"Selected subset: {selected_embeddings.shape[0]} papers")

    # -----------------------------------------------------------------------
    # Incremental mode: transform new papers using saved models
    # -----------------------------------------------------------------------
    # Build selected IDs list for this run
    selected_ids = [
        row_to_arxiv_id[i]
        for i in range(len(row_to_arxiv_id))
        if selected_mask[i]
    ]

    if args.incremental:
        print("Incremental mode: using saved PCA/UMAP models to transform new papers.")
        pca_model = joblib.load(PCA_MODEL_PATH)
        umap_model = joblib.load(UMAP_MODEL_PATH)

        # Load previous coordinates and ID ordering
        prev_coords = load_numpy(COORDS_OUT_PATH) if COORDS_OUT_PATH.exists() else None
        prev_ids = load_json(SELECTED_IDS_PATH) if SELECTED_IDS_PATH.exists() else None

        if prev_coords is not None and prev_ids is not None:
            prev_id_set = set(prev_ids)
            prev_id_to_row = {aid: i for i, aid in enumerate(prev_ids)}

            # Identify which papers are new vs existing
            new_indices = []  # indices into selected_embeddings
            existing_rows = []  # (index in selected_embeddings, row in prev_coords)
            for i, aid in enumerate(selected_ids):
                if aid in prev_id_to_row:
                    existing_rows.append((i, prev_id_to_row[aid]))
                else:
                    new_indices.append(i)

            print(f"Existing papers: {len(existing_rows)}, new papers: {len(new_indices)}")

            # Start with existing coordinates
            coords_all = np.zeros((len(selected_ids), 2), dtype=np.float64)
            for sel_i, prev_i in existing_rows:
                coords_all[sel_i] = prev_coords[prev_i]

            # Only transform new papers
            if new_indices:
                new_embeddings = selected_embeddings[new_indices]
                pca_new = pca_model.transform(new_embeddings)
                pca_new = normalize(pca_new, norm="l2", axis=1)
                coords_new = umap_model.transform(pca_new)
                for j, sel_i in enumerate(new_indices):
                    coords_all[sel_i] = coords_new[j]

            # Also update PCA for full set (needed by clustering)
            pca_full = pca_model.transform(selected_embeddings)
            pca_full = normalize(pca_full, norm="l2", axis=1)
            save_numpy(pca_full, PCA_OUT_PATH)

            save_numpy(coords_all, COORDS_OUT_PATH)
            save_json(selected_ids, SELECTED_IDS_PATH)
            print(f"Incremental update complete: {len(selected_ids)} papers ({len(new_indices)} newly projected).")
        else:
            # No previous data; fall back to transforming everything
            pca_all = pca_model.transform(selected_embeddings)
            pca_all = normalize(pca_all, norm="l2", axis=1)
            save_numpy(pca_all, PCA_OUT_PATH)
            coords_all = umap_model.transform(pca_all)
            save_numpy(coords_all, COORDS_OUT_PATH)
            save_json(selected_ids, SELECTED_IDS_PATH)
            print(f"Incremental update complete (no previous data): {coords_all.shape[0]} papers projected.")
        return

    # -----------------------------------------------------------------------
    # Full refit
    # -----------------------------------------------------------------------

    # Save current kNN selection as _prev before this run overwrites it.
    # This enables the HIGH_DRIFT retry to use the previous month's selection.
    if KNN_SELECTED_PATH.exists():
        shutil.copy2(KNN_SELECTED_PATH, KNN_SELECTED_PREV_PATH)

    # 1. PCA to 50 dims (denoising + speed)
    n_components = int(pca_cfg.get("n_components", 50))
    pca = PCA(n_components=n_components, random_state=42)
    pca_embeddings = pca.fit_transform(selected_embeddings)
    retained_var = float(pca.explained_variance_ratio_.sum())
    print(f"PCA retained variance ratio sum: {retained_var:.4f}")

    # L2-normalize PCA vectors so Euclidean clustering is geometry-consistent
    # with cosine neighborhoods.
    pca_embeddings = normalize(pca_embeddings, norm="l2", axis=1)
    save_numpy(pca_embeddings, PCA_OUT_PATH)
    PCA_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, PCA_MODEL_PATH)
    print(f"Saved PCA model to {PCA_MODEL_PATH}")

    # 2. UMAP to 2D
    reducer = umap.UMAP(
        n_neighbors=int(umap_cfg.get("n_neighbors", 40)),
        min_dist=float(umap_cfg.get("min_dist", 0.08)),
        n_components=int(umap_cfg.get("n_components", 2)),
        metric=str(umap_cfg.get("metric", "cosine")),
        random_state=int(umap_cfg.get("random_state", 42)),
    )
    coords_2d = reducer.fit_transform(pca_embeddings)
    UMAP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reducer, UMAP_MODEL_PATH)
    print(f"Saved UMAP model to {UMAP_MODEL_PATH}")

    # -----------------------------------------------------------------------
    # Monthly refit: Procrustes alignment if previous coords exist
    # -----------------------------------------------------------------------
    monthly_cfg = update_cfg.get("monthly_refit", {})
    save_previous = bool(monthly_cfg.get("save_previous", True))

    if PREV_COORDS_PATH.exists() and PREV_SELECTED_IDS_PATH.exists():
        print("Previous coordinates found — running Procrustes alignment.")
        old_coords = load_numpy(PREV_COORDS_PATH)
        old_selected_ids = load_json(PREV_SELECTED_IDS_PATH)

        # Issue 1: Build anchor mask by IAIFI ID intersection, not by position.
        # This allows alignment even when total N differs between old and new runs.
        from iaifi_paperscape.utils.io import load_jsonl as _load_jsonl
        iaifi_records = _load_jsonl(IAIFI_METADATA_PATH)
        iaifi_id_set = {
            str(rec.get("arxiv_id") or rec.get("id", ""))
            for rec in iaifi_records
        }

        old_id_to_row = {aid: i for i, aid in enumerate(old_selected_ids)}
        # Find IAIFI IDs present in both old and new selected sets
        common_iaifi_ids = []
        new_anchor_rows = []
        old_anchor_rows = []
        for i, aid in enumerate(selected_ids):
            if aid in iaifi_id_set and aid in old_id_to_row:
                common_iaifi_ids.append(aid)
                new_anchor_rows.append(i)
                old_anchor_rows.append(old_id_to_row[aid])

        if len(common_iaifi_ids) < 3:
            print(
                f"Only {len(common_iaifi_ids)} common IAIFI anchor papers; "
                "skipping Procrustes alignment (need >= 3)."
            )
        else:
            # Build aligned old_coords and new_coords subsets for anchors
            new_anchor_coords = coords_2d[new_anchor_rows]
            old_anchor_coords = old_coords[old_anchor_rows]

            # Create anchor mask over new coords (True for anchor positions)
            anchor_mask = np.zeros(coords_2d.shape[0], dtype=bool)
            anchor_mask[new_anchor_rows] = True

            # Build a temporary combined array for align_to_previous:
            # old_coords reordered to match new layout at anchor positions
            old_coords_reindexed = np.zeros_like(coords_2d)
            for new_i, old_i in zip(new_anchor_rows, old_anchor_rows):
                old_coords_reindexed[new_i] = old_coords[old_i]

            aligned_coords, drift = align_to_previous(
                coords_2d, old_coords_reindexed, anchor_mask
            )
            band = classify_drift(drift)
            print(
                f"Procrustes drift: median={drift['median']:.4f}, "
                f"p95={drift['p95']:.4f}, max={drift['max']:.4f}, "
                f"reflected={drift['reflected']}, band={band}, "
                f"anchors={len(common_iaifi_ids)}"
            )

            # Issue 2: HIGH_DRIFT — automatically rerun with previous month's kNN selection
            if band == "HIGH":
                frozen_knn_path = KNN_SELECTED_PATH.with_suffix(".frozen.json")
                if not frozen_knn_path.exists():
                    if KNN_SELECTED_PREV_PATH.exists():
                        print(
                            "HIGH DRIFT detected. Loading PREVIOUS month's "
                            "knn_selected_ids and rerunning PCA+UMAP."
                        )
                        shutil.copy2(KNN_SELECTED_PREV_PATH, frozen_knn_path)
                    else:
                        print(
                            "WARNING: HIGH DRIFT detected but no previous kNN "
                            "selection found (knn_selected_ids_prev.json missing). "
                            "Falling back to current selection for retry."
                        )
                        shutil.copy2(KNN_SELECTED_PATH, frozen_knn_path)

                    # Reload previous knn_selected_ids and rebuild selection
                    prev_knn_ids = load_json(frozen_knn_path)
                    frozen_mask = _build_selected_mask(
                        row_to_arxiv_id, prev_knn_ids, iaifi_count
                    )
                    frozen_embeddings = embeddings[frozen_mask]
                    frozen_selected_ids = [
                        row_to_arxiv_id[i]
                        for i in range(len(row_to_arxiv_id))
                        if frozen_mask[i]
                    ]

                    # Refit PCA + UMAP with frozen selection
                    pca2 = PCA(n_components=n_components, random_state=42)
                    pca_emb2 = normalize(pca2.fit_transform(frozen_embeddings), norm="l2", axis=1)
                    reducer2 = umap.UMAP(
                        n_neighbors=int(umap_cfg.get("n_neighbors", 40)),
                        min_dist=float(umap_cfg.get("min_dist", 0.08)),
                        n_components=int(umap_cfg.get("n_components", 2)),
                        metric=str(umap_cfg.get("metric", "cosine")),
                        random_state=int(umap_cfg.get("random_state", 42)),
                    )
                    coords_2d_retry = reducer2.fit_transform(pca_emb2)

                    # Rebuild anchor mask for frozen selection
                    frozen_id_to_row = {aid: i for i, aid in enumerate(frozen_selected_ids)}
                    retry_anchor_rows_new = []
                    retry_anchor_rows_old = []
                    for aid in common_iaifi_ids:
                        if aid in frozen_id_to_row and aid in old_id_to_row:
                            retry_anchor_rows_new.append(frozen_id_to_row[aid])
                            retry_anchor_rows_old.append(old_id_to_row[aid])

                    retry_anchor_mask = np.zeros(coords_2d_retry.shape[0], dtype=bool)
                    retry_anchor_mask[retry_anchor_rows_new] = True
                    old_reindexed2 = np.zeros_like(coords_2d_retry)
                    for new_i, old_i in zip(retry_anchor_rows_new, retry_anchor_rows_old):
                        old_reindexed2[new_i] = old_coords[old_i]

                    aligned_retry, drift_retry = align_to_previous(
                        coords_2d_retry, old_reindexed2, retry_anchor_mask
                    )
                    band_retry = classify_drift(drift_retry)
                    print(
                        f"Retry drift: median={drift_retry['median']:.4f}, "
                        f"p95={drift_retry['p95']:.4f}, band={band_retry}"
                    )

                    if band_retry == "HIGH":
                        print(
                            "WARNING: HIGH DRIFT persists after frozen rerun. "
                            "Proceeding with layout-changed notice."
                        )
                        drift = drift_retry
                        drift["layout_changed"] = True
                        band = band_retry
                    else:
                        # Retry succeeded: use frozen results
                        coords_2d = aligned_retry
                        drift = drift_retry
                        band = band_retry
                        selected_ids = frozen_selected_ids
                        selected_embeddings = frozen_embeddings
                        pca_embeddings = pca_emb2
                        save_numpy(pca_emb2, PCA_OUT_PATH)
                        joblib.dump(pca2, PCA_MODEL_PATH)
                        joblib.dump(reducer2, UMAP_MODEL_PATH)
                        aligned_coords = coords_2d
                else:
                    print(
                        "HIGH DRIFT on second attempt (frozen kNN). "
                        "Publishing with visible layout-change notice."
                    )
                    drift["layout_changed"] = True

            coords_2d = aligned_coords

            # Issue 3: Write both drift_diagnostics.json AND procrustes_meta.json
            diag_payload = {"drift": drift, "band": band}
            save_json(diag_payload, DRIFT_DIAG_PATH)
            save_json(
                {"reflected": drift.get("reflected", False), "drift": drift, "band": band},
                PROCRUSTES_META_PATH,
            )
    elif PREV_COORDS_PATH.exists():
        print(
            "Previous coords exist but no selected_ids_prev.json; "
            "skipping Procrustes alignment."
        )

    # Save current coords and ID ordering as "previous" for next refit
    if save_previous:
        save_numpy(coords_2d, PREV_COORDS_PATH)
        save_json(selected_ids, PREV_SELECTED_IDS_PATH)

    save_numpy(coords_2d, COORDS_OUT_PATH)
    save_json(selected_ids, SELECTED_IDS_PATH)
    print(f"Saved 2D coordinates: {coords_2d.shape} to {COORDS_OUT_PATH}")


if __name__ == "__main__":
    main()
