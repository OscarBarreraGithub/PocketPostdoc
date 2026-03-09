"""Step G.2: assemble web/data/papers.json from all processed pipeline outputs.

Usage:
    python -m iaifi_paperscape.export.export_web_assets [--strict]

Loads coords, clusters, labels, enrichment, neighbors, metadata, and theme
overrides, then assembles the final JSON for the frontend.

Flags:
    --strict    Make missing/empty theme_overrides.yml a hard error (for
                production builds).  Without this flag a warning is printed
                but output is still generated.

Outputs:
    web/data/papers.json           — main payload (may exclude abstracts)
    web/data/abstracts.json        — split out if abstracts exceed 2 MB
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from iaifi_paperscape.utils.io import (
    load_json,
    load_jsonl,
    load_numpy,
    load_yaml_config,
    save_json,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Configs
UMAP_CONFIG_PATH = PROJECT_ROOT / "configs" / "umap.yaml"
EMBEDDING_CONFIG_PATH = PROJECT_ROOT / "configs" / "embedding.yaml"
THEME_OVERRIDES_PATH = PROJECT_ROOT / "configs" / "theme_overrides.yml"

# Data inputs
COORDS_PATH = PROJECT_ROOT / "data" / "processed" / "coords_2d.npy"
CLUSTERS_PATH = PROJECT_ROOT / "data" / "processed" / "clusters.json"
CLUSTER_LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "cluster_labels.json"
ENRICHMENT_PATH = PROJECT_ROOT / "data" / "processed" / "enrichment.json"
NEIGHBORS_PATH = PROJECT_ROOT / "data" / "processed" / "neighbors.json"
IAIFI_METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "iaifi_metadata.jsonl"
BACKGROUND_PATH = PROJECT_ROOT / "data" / "raw" / "background_candidates.jsonl"
EMBEDDING_INDEX_PATH = PROJECT_ROOT / "data" / "interim" / "embedding_index.json"
KNN_SELECTED_PATH = PROJECT_ROOT / "data" / "interim" / "knn_selected_ids.json"

# Procrustes output (optional)
PROCRUSTES_META_PATH = PROJECT_ROOT / "data" / "processed" / "procrustes_meta.json"

# Output
OUTPUT_DIR = PROJECT_ROOT / "web" / "data"
PAPERS_JSON_PATH = OUTPUT_DIR / "papers.json"
ABSTRACTS_JSON_PATH = OUTPUT_DIR / "abstracts.json"

# Size threshold for abstract splitting (bytes)
ABSTRACTS_SPLIT_THRESHOLD = 2 * 1024 * 1024  # 2 MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_row_to_arxiv_id(path: Path) -> list[str]:
    """Load the embedding index (handles multiple on-disk formats)."""
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
                "embedding_index.json has non-integer row keys."
            ) from exc
    raise RuntimeError("Unsupported embedding index format.")


def _check_theme_overrides(path: Path, *, strict: bool) -> dict[str, str]:
    """Load theme_overrides.yml.  Warn or error if empty/comments-only.

    Returns a dict mapping cluster_id (str) -> display label.
    """
    if not path.exists():
        msg = f"theme_overrides.yml not found at {path}"
        if strict:
            print(f"ERROR: {msg}")
            sys.exit(1)
        print(f"WARNING: {msg} — using auto-labels only")
        return {}

    raw = load_yaml_config(path)

    # load_yaml_config returns {} for empty/comment-only files
    if not raw:
        msg = (
            "theme_overrides.yml is empty or contains only comments. "
            "Human-curated labels are MANDATORY before deployment."
        )
        if strict:
            print(f"ERROR: {msg}")
            sys.exit(1)
        print(f"WARNING: {msg}")
        return {}

    # Normalize keys to strings
    return {str(k): str(v) for k, v in raw.items()}


def _build_metadata_lookup(
    iaifi_records: list[dict],
    background_records: list[dict],
    knn_selected_ids: set[str],
) -> dict[str, dict]:
    """Build {arxiv_id: metadata} for all selected papers."""
    lookup: dict[str, dict] = {}

    for rec in iaifi_records:
        aid = rec.get("arxiv_id") or rec.get("id", "")
        lookup[str(aid)] = rec

    for rec in background_records:
        aid = str(rec.get("arxiv_id") or rec.get("id", ""))
        if aid in knn_selected_ids:
            lookup[aid] = rec

    return lookup


def _format_authors(meta: dict) -> str:
    """Format authors as semicolon-separated string."""
    authors = meta.get("authors", [])
    if isinstance(authors, str):
        return authors
    if isinstance(authors, list):
        if len(authors) > 5:
            return "; ".join(str(a) for a in authors[:5]) + "; et al."
        return "; ".join(str(a) for a in authors)
    return ""


def _get_categories(meta: dict) -> list[str]:
    """Extract arXiv category list."""
    cats = meta.get("categories", meta.get("category", ""))
    if isinstance(cats, list):
        return [str(c) for c in cats]
    if isinstance(cats, str) and cats:
        return [c.strip() for c in cats.replace(",", " ").split()]
    return []


def _get_year(meta: dict) -> int | None:
    """Extract publication year."""
    yr = meta.get("year") or meta.get("published", "")
    if isinstance(yr, int):
        return yr
    if isinstance(yr, str) and len(yr) >= 4:
        try:
            return int(yr[:4])
        except ValueError:
            pass
    return None


def _get_theme(meta: dict) -> str:
    """Extract IAIFI theme tag if present."""
    return str(meta.get("iaifi_theme", meta.get("theme", "")))


# ---------------------------------------------------------------------------
# Core assembly
# ---------------------------------------------------------------------------

def assemble_payload(*, strict: bool = False) -> dict:
    """Load all pipeline outputs and assemble the papers.json payload."""

    # ------------------------------------------------------------------
    # Load all data sources
    # ------------------------------------------------------------------
    print("Loading pipeline outputs...")

    coords = load_numpy(COORDS_PATH)  # (N, 2)
    clusters_data = load_json(CLUSTERS_PATH)
    labels_array = np.array(clusters_data["labels"], dtype=int)
    # Build stability map from clusters list (run_hdbscan emits per-cluster
    # stability inside clusters_data["clusters"], not a top-level "stability" map).
    stability_scores: dict = clusters_data.get("stability", {})
    if not stability_scores and "clusters" in clusters_data:
        stability_scores = {
            str(c["id"]): c["stability"]
            for c in clusters_data["clusters"]
            if "stability" in c
        }

    cluster_labels = load_json(CLUSTER_LABELS_PATH)
    enrichment = load_json(ENRICHMENT_PATH)
    neighbors = load_json(NEIGHBORS_PATH)

    row_to_arxiv_id = _load_row_to_arxiv_id(EMBEDDING_INDEX_PATH)
    knn_selected_ids = set(load_json(KNN_SELECTED_PATH))

    iaifi_records = load_jsonl(IAIFI_METADATA_PATH)
    background_records = load_jsonl(BACKGROUND_PATH)

    umap_config = load_yaml_config(UMAP_CONFIG_PATH)
    embedding_config = load_yaml_config(EMBEDDING_CONFIG_PATH)

    theme_overrides = _check_theme_overrides(THEME_OVERRIDES_PATH, strict=strict)

    # Procrustes reflected flag (optional)
    reflected = False
    if PROCRUSTES_META_PATH.exists():
        try:
            proc_meta = load_json(PROCRUSTES_META_PATH)
            reflected = bool(proc_meta.get("reflected", False))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Build selected-paper ordering (same as other pipeline steps)
    # ------------------------------------------------------------------
    n_iaifi = len(iaifi_records)
    selected_indices: list[int] = []
    for i, aid in enumerate(row_to_arxiv_id):
        if i < n_iaifi or aid in knn_selected_ids:
            selected_indices.append(i)

    selected_ids = [row_to_arxiv_id[i] for i in selected_indices]
    iaifi_id_set = {
        str(rec.get("arxiv_id") or rec.get("id", ""))
        for rec in iaifi_records
    }

    n_background = len(selected_ids) - n_iaifi
    n_total = len(selected_ids)

    print(f"Papers: {n_total} (IAIFI: {n_iaifi}, background: {n_background})")

    # Validate dimensions
    if coords.shape[0] != n_total:
        print(
            f"WARNING: coords rows ({coords.shape[0]}) != selected papers ({n_total}). "
            "Using min."
        )

    # ------------------------------------------------------------------
    # Metadata lookup
    # ------------------------------------------------------------------
    meta_lookup = _build_metadata_lookup(
        iaifi_records, background_records, knn_selected_ids
    )

    # ------------------------------------------------------------------
    # Build papers array
    # ------------------------------------------------------------------
    papers: list[dict] = []
    abstracts_map: dict[str, str] = {}

    for idx, aid in enumerate(selected_ids):
        if idx >= coords.shape[0]:
            break

        meta = meta_lookup.get(aid, {})
        cluster_id = int(labels_array[idx]) if idx < len(labels_array) else -1

        # Neighbor list: only arxiv IDs (no similarity scores)
        nn_list: list[str] = []
        if aid in neighbors:
            nn_list = [entry["id"] for entry in neighbors[aid][:10]]

        paper = {
            "id": aid,
            "x": round(float(coords[idx, 0]), 4),
            "y": round(float(coords[idx, 1]), 4),
            "t": meta.get("title", ""),
            "a": _format_authors(meta),
            "yr": _get_year(meta),
            "cat": _get_categories(meta),
            "iaifi": aid in iaifi_id_set,
            "theme": _get_theme(meta),
            "cluster": cluster_id,
            "cit": meta.get("citation_count", meta.get("citations", None)),
            "nn": nn_list,
        }
        papers.append(paper)

        # Collect abstract
        abstract = meta.get("abstract", "")
        if abstract:
            abstracts_map[aid] = abstract

    # ------------------------------------------------------------------
    # Build clusters array
    # ------------------------------------------------------------------
    # Gather unique non-noise cluster IDs
    unique_clusters = sorted(set(int(c) for c in labels_array if c != -1))

    # Compute centroids from 2D coords
    clusters_out: list[dict] = []
    for cid in unique_clusters:
        mask = labels_array == cid
        member_indices = np.where(mask)[0]

        # Filter to valid coord range
        valid = member_indices[member_indices < coords.shape[0]]
        if len(valid) == 0:
            continue

        cx = round(float(coords[valid, 0].mean()), 4)
        cy = round(float(coords[valid, 1].mean()), 4)
        size = int(mask.sum())

        # Label: prefer theme_overrides, fall back to auto_label
        cid_str = str(cid)
        auto_label_info = cluster_labels.get(cid_str, {})
        if isinstance(auto_label_info, dict):
            auto_label = auto_label_info.get("label", auto_label_info.get("auto_label", f"Cluster {cid}"))
        elif isinstance(auto_label_info, str):
            auto_label = auto_label_info
        else:
            auto_label = f"Cluster {cid}"

        display_label = theme_overrides.get(cid_str, auto_label)

        # Stability
        stab = stability_scores.get(cid_str, stability_scores.get(cid, None))
        stability_val = float(stab) if stab is not None else None

        # Enrichment
        enrich_data = enrichment.get(cid_str, None)
        enrichment_out = None
        if enrich_data is not None:
            enrichment_out = {
                "log2": enrich_data.get("log2_enrichment", 0.0),
                "ci_lo": enrich_data.get("ci_95_lo"),
                "ci_hi": enrich_data.get("ci_95_hi"),
            }

        cluster_obj = {
            "id": cid,
            "label": display_label,
            "auto_label": auto_label,
            "cx": cx,
            "cy": cy,
            "size": size,
            "stability": stability_val,
            "enrichment": enrichment_out,
        }
        clusters_out.append(cluster_obj)

    # ------------------------------------------------------------------
    # Build meta
    # ------------------------------------------------------------------
    umap_params_cfg = umap_config.get("umap", {})
    pca_cfg = umap_config.get("pca", {})

    version_str = (
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        f"_specter2prox_n{n_total}"
    )

    meta = {
        "version": version_str,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embedding_model": "specter2_proximity",
        "reflected": reflected,
        "n_iaifi": n_iaifi,
        "n_background": n_background,
        "n_total": n_total,
        "umap_params": {
            "n_neighbors": umap_params_cfg.get("n_neighbors", 40),
            "min_dist": umap_params_cfg.get("min_dist", 0.08),
            "metric": umap_params_cfg.get("metric", "cosine"),
            "pca_dims": pca_cfg.get("n_components", 50),
        },
        "disclaimer": "Local neighborhoods are meaningful; global geometry is approximate.",
    }

    # ------------------------------------------------------------------
    # Assemble full payload
    # ------------------------------------------------------------------
    payload = {
        "meta": meta,
        "papers": papers,
        "clusters": clusters_out,
        "abstracts": abstracts_map,
    }

    return payload


# ---------------------------------------------------------------------------
# Abstract splitting (G.3)
# ---------------------------------------------------------------------------

def _maybe_split_abstracts(payload: dict) -> tuple[dict, dict | None]:
    """If the abstracts map exceeds 2 MB when serialized, split it out.

    Returns (main_payload, abstracts_payload_or_None).
    """
    abstracts = payload.get("abstracts", {})
    if not abstracts:
        return payload, None

    abstracts_json = json.dumps(abstracts, ensure_ascii=False)
    abstracts_size = len(abstracts_json.encode("utf-8"))
    print(f"Abstracts size: {abstracts_size / 1024 / 1024:.2f} MB")

    if abstracts_size > ABSTRACTS_SPLIT_THRESHOLD:
        print("Abstracts exceed 2 MB — splitting into separate file.")
        main_payload = {k: v for k, v in payload.items() if k != "abstracts"}
        main_payload["meta"]["abstracts_split"] = True
        return main_payload, {"abstracts": abstracts}
    else:
        payload["meta"]["abstracts_split"] = False
        return payload, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step G.2: export web assets (papers.json)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if theme_overrides.yml is missing or empty (production mode).",
    )
    args = parser.parse_args()

    print("Step G.2: assembling web assets...")

    payload = assemble_payload(strict=args.strict)

    # Split abstracts if needed (G.3)
    main_payload, abstracts_payload = _maybe_split_abstracts(payload)

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(main_payload, PAPERS_JSON_PATH, indent=2)
    print(f"Saved {PAPERS_JSON_PATH} ({PAPERS_JSON_PATH.stat().st_size / 1024:.0f} KB)")

    if abstracts_payload is not None:
        save_json(abstracts_payload, ABSTRACTS_JSON_PATH, indent=2)
        print(
            f"Saved {ABSTRACTS_JSON_PATH} "
            f"({ABSTRACTS_JSON_PATH.stat().st_size / 1024:.0f} KB)"
        )

    n_papers = len(main_payload["papers"])
    n_clusters = len(main_payload["clusters"])
    print(f"Export complete: {n_papers} papers, {n_clusters} clusters")


if __name__ == "__main__":
    main()
