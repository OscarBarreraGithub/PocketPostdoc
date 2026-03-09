"""Step C: select a background corpus via diversity-aware kNN union."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from iaifi_paperscape.utils.io import load_json, load_jsonl, load_numpy, load_yaml_config, save_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "corpus.yaml"
IAIFI_METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "iaifi_metadata.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "interim" / "embeddings.npy"
EMBEDDING_INDEX_PATH = PROJECT_ROOT / "data" / "interim" / "embedding_index.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "knn_selected_ids.json"
METRICS_PATH = PROJECT_ROOT / "data" / "interim" / "knn_selection_metrics.json"


def _load_row_to_arxiv_id(path: Path) -> list[str]:
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


def main() -> None:
    config = load_yaml_config(CONFIG_PATH)
    selection_cfg = config.get("selection", {})
    diversity_cfg = selection_cfg.get("diversity", {})
    hubness_cfg = selection_cfg.get("hubness_monitor", {})

    embeddings = load_numpy(EMBEDDINGS_PATH)
    row_to_arxiv_id = _load_row_to_arxiv_id(EMBEDDING_INDEX_PATH)
    iaifi_count = len(load_jsonl(IAIFI_METADATA_PATH))

    if embeddings.ndim != 2:
        raise RuntimeError(f"Expected 2D embeddings, got shape={embeddings.shape}.")
    if len(row_to_arxiv_id) != embeddings.shape[0]:
        raise RuntimeError(
            "Embedding/index length mismatch: "
            f"{embeddings.shape[0]} rows vs {len(row_to_arxiv_id)} IDs."
        )
    if iaifi_count <= 0:
        raise RuntimeError("No IAIFI seed papers found in iaifi_metadata.jsonl.")
    if iaifi_count >= embeddings.shape[0]:
        raise RuntimeError(
            "Embeddings must include both IAIFI and candidate papers; "
            f"got total_rows={embeddings.shape[0]}, iaifi_rows={iaifi_count}."
        )

    iaifi_embeddings = embeddings[:iaifi_count]
    candidate_embeddings = embeddings[iaifi_count:]
    candidate_ids = row_to_arxiv_id[iaifi_count:]

    print(
        f"Loaded embeddings: total={embeddings.shape[0]}, iaifi={iaifi_count}, "
        f"candidates={candidate_embeddings.shape[0]}"
    )

    knn_initial_k = int(selection_cfg.get("knn_initial_k", 100))
    knn_initial_k = min(knn_initial_k, candidate_embeddings.shape[0])
    if knn_initial_k <= 0:
        raise RuntimeError("No candidate embeddings available for kNN selection.")

    nn = NearestNeighbors(n_neighbors=knn_initial_k, metric="cosine", algorithm="auto")
    nn.fit(candidate_embeddings)

    _, indices = nn.kneighbors(iaifi_embeddings)

    # Intentionally computed on raw top-k before per-seed diversity filtering.
    support_count = Counter(indices.ravel().tolist())

    # Dot-product dedup assumes unit vectors.
    candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
    assert np.allclose(candidate_norms, 1.0, atol=1e-3), (
        "candidate_embeddings must be L2-normalized before diversity dedup"
    )

    selected_indices: set[int] = set()
    diversity_enabled = bool(diversity_cfg.get("enabled", True))
    per_seed_diverse_k = int(diversity_cfg.get("per_seed_diverse_k", 12))
    near_duplicate_cosine = float(diversity_cfg.get("near_duplicate_cosine", 0.985))

    if diversity_enabled:
        for row in indices:
            kept: list[int] = []
            for cand_idx in row.tolist():
                if len(kept) >= per_seed_diverse_k:
                    break
                if not kept:
                    kept.append(cand_idx)
                    continue
                max_pair_sim = (
                    candidate_embeddings[cand_idx] @ candidate_embeddings[kept].T
                ).max()
                if max_pair_sim < near_duplicate_cosine:
                    kept.append(cand_idx)
            selected_indices.update(kept)
    else:
        selected_indices.update(indices.ravel().tolist())

    rng = np.random.RandomState(42)
    bridge_ballast = int(selection_cfg.get("bridge_ballast", 1000))
    remaining = sorted(set(range(candidate_embeddings.shape[0])) - selected_indices)
    bridge: set[int] = set()
    if remaining and bridge_ballast > 0:
        bridge_size = min(bridge_ballast, len(remaining))
        bridge = set(rng.choice(np.array(remaining), size=bridge_size, replace=False).tolist())
        selected_indices |= bridge

    max_background = int(selection_cfg.get("max_background", 10000))
    anti_hub_weight = float(diversity_cfg.get("anti_hub_weight", 0.2))
    if len(selected_indices) > max_background:
        all_selected = np.array(sorted(selected_indices), dtype=np.int32)
        sim_to_iaifi = candidate_embeddings[all_selected] @ iaifi_embeddings.T
        max_sim = sim_to_iaifi.max(axis=1)
        anti_hub = np.array(
            [1.0 / np.sqrt(1.0 + support_count.get(int(i), 0)) for i in all_selected],
            dtype=np.float32,
        )
        score = (1.0 - anti_hub_weight) * max_sim + anti_hub_weight * anti_hub
        top_k_idx = np.argsort(score)[-max_background:]
        selected_indices = set(all_selected[top_k_idx].tolist())

    selected_array = np.array(sorted(selected_indices), dtype=np.int32)
    neighbor_share = np.array(
        [support_count.get(int(i), 0) / iaifi_embeddings.shape[0] for i in selected_array],
        dtype=np.float32,
    )
    p50 = float(np.percentile(neighbor_share, 50)) if neighbor_share.size else 0.0
    p95 = float(np.percentile(neighbor_share, 95)) if neighbor_share.size else 0.0
    max_share = float(neighbor_share.max()) if neighbor_share.size else 0.0
    print(f"Hubness share stats: p50={p50:.3f}, p95={p95:.3f}, max={max_share:.3f}")

    hubness_metrics = {
        "p50": round(p50, 4),
        "p95": round(p95, 4),
        "max": round(max_share, 4),
        "total_selected": len(selected_indices),
    }
    save_json(hubness_metrics, METRICS_PATH, indent=2)
    print(f"Saved hubness metrics to {METRICS_PATH}")

    warn_threshold = float(hubness_cfg.get("warn_if_neighbor_share_gt", 0.15))
    if max_share > warn_threshold:
        print(
            "WARNING: method-hub candidate detected "
            f"(max neighbor share > {warn_threshold:.2f})"
        )

    kept_bridge = len(selected_indices & bridge)
    print(
        f"Selected {len(selected_indices)} background papers "
        f"({len(selected_indices) - kept_bridge} via kNN, {kept_bridge} bridge ballast)"
    )

    selected_ids = [candidate_ids[i] for i in selected_array.tolist()]
    save_json(selected_ids, OUTPUT_PATH, indent=2)
    print(f"Wrote {len(selected_ids)} IDs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
