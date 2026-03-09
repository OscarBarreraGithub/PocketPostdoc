"""Step F.1: HDBSCAN clustering with stability gating + KMeans benchmark.

Usage:
    python -m iaifi_paperscape.cluster.run_hdbscan

Loads L2-normalized PCA-50 embeddings, runs HDBSCAN with stability gating,
saves cluster assignments, then runs an offline KMeans benchmark (F.2).
"""

from __future__ import annotations

import logging
from pathlib import Path

import hdbscan
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from iaifi_paperscape.utils.io import load_numpy, load_yaml_config, save_json

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "clustering.yaml"
PCA_PATH = PROJECT_ROOT / "data" / "interim" / "pca_50.npy"
CLUSTERS_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "clusters.json"
BENCHMARK_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "kmeans_benchmark.json"


# ---------------------------------------------------------------------------
# Label -> persistence mapping using HDBSCAN private internals
# (pinned: hdbscan==0.8.38)
# ---------------------------------------------------------------------------

def build_label_persistence_map(clusterer: hdbscan.HDBSCAN) -> dict[int, float]:
    """Explicit map from output cluster label -> persistence.

    Do NOT rely on ``enumerate(sorted(unique_labels))``.
    """
    clusterer.generate_prediction_data()
    selected_tree_ids = list(clusterer.condensed_tree_._select_clusters())
    persistence = np.asarray(clusterer.cluster_persistence_, dtype=float)

    if len(selected_tree_ids) != len(persistence):
        raise RuntimeError(
            "HDBSCAN selected-cluster count != persistence count; "
            "abort unsafe stability gating."
        )

    # Internal ordering from HDBSCAN condensed tree.
    persistence_by_tree_id = {
        int(tree_id): float(p)
        for tree_id, p in zip(selected_tree_ids, persistence)
    }

    # Internal map: tree cluster id -> emitted integer label.
    cluster_map = clusterer._prediction_data.cluster_map
    label_to_persistence: dict[int, float] = {}
    for tree_id, label_id in cluster_map.items():
        if label_id == -1:
            continue
        if int(tree_id) in persistence_by_tree_id:
            label_to_persistence[int(label_id)] = persistence_by_tree_id[int(tree_id)]

    return label_to_persistence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_yaml_config(CONFIG_PATH)
    hdbscan_cfg = config.get("hdbscan", {})
    stability_cfg = config.get("stability_gating", {})
    benchmark_cfg = config.get("kmeans_benchmark", {})

    # -----------------------------------------------------------------------
    # Load PCA-50 embeddings (L2-normalized)
    # -----------------------------------------------------------------------
    pca_embeddings = load_numpy(PCA_PATH)
    print(f"Loaded PCA embeddings: {pca_embeddings.shape}")

    # -----------------------------------------------------------------------
    # F.1: HDBSCAN clustering
    # -----------------------------------------------------------------------
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(hdbscan_cfg.get("min_cluster_size", 50)),
        min_samples=int(hdbscan_cfg.get("min_samples", 10)),
        metric=str(hdbscan_cfg.get("metric", "euclidean")),
        cluster_selection_method=str(hdbscan_cfg.get("cluster_selection_method", "eom")),
        gen_min_span_tree=True,
    )
    labels = clusterer.fit_predict(pca_embeddings)
    probabilities = clusterer.probabilities_

    label_to_persistence = build_label_persistence_map(clusterer)

    unique_before = sorted(set(labels) - {-1})
    print(f"HDBSCAN found {len(unique_before)} clusters before stability gating.")

    # -----------------------------------------------------------------------
    # Stability gating: demote weak clusters to noise (-1)
    # -----------------------------------------------------------------------
    min_size = int(stability_cfg.get("min_size", 50))
    min_stability = float(stability_cfg.get("min_stability", 0.5))

    demoted = []
    for cluster_id in unique_before:
        mask = labels == cluster_id
        persistence = label_to_persistence.get(int(cluster_id), 0.0)
        if mask.sum() < min_size or persistence < min_stability:
            labels[mask] = -1
            demoted.append(cluster_id)

    unique_after = sorted(set(labels) - {-1})
    print(
        f"After stability gating: {len(unique_after)} clusters "
        f"({len(demoted)} demoted to noise)."
    )
    print(f"Noise points: {int((labels == -1).sum())} / {len(labels)}")

    # -----------------------------------------------------------------------
    # Build output payload
    # -----------------------------------------------------------------------
    clusters_info = []
    for cluster_id in unique_after:
        mask = labels == cluster_id
        clusters_info.append({
            "id": int(cluster_id),
            "size": int(mask.sum()),
            "stability": float(label_to_persistence.get(int(cluster_id), 0.0)),
        })

    payload = {
        "n_papers": int(len(labels)),
        "n_clusters": len(unique_after),
        "n_noise": int((labels == -1).sum()),
        "demoted_clusters": [int(c) for c in demoted],
        "labels": [int(l) for l in labels],
        "probabilities": [float(p) for p in probabilities],
        "clusters": clusters_info,
    }
    save_json(payload, CLUSTERS_OUT_PATH)
    print(f"Saved cluster assignments to {CLUSTERS_OUT_PATH}")

    # -----------------------------------------------------------------------
    # F.2: KMeans benchmark (offline only)
    # -----------------------------------------------------------------------
    if not benchmark_cfg.get("enabled", True):
        print("KMeans benchmark disabled; skipping.")
        return

    k_range = benchmark_cfg.get("k_range", [10, 15, 20, 25, 30])
    sample_size = min(5000, pca_embeddings.shape[0])

    benchmark_results = []
    for k in k_range:
        km = MiniBatchKMeans(n_clusters=k, random_state=42)
        km_labels = km.fit_predict(pca_embeddings)
        sil = float(silhouette_score(pca_embeddings, km_labels, sample_size=sample_size))

        # DBCV is native to HDBSCAN; compute on the HDBSCAN clusterer's output
        # rather than on KMeans. We log it once outside the loop.
        benchmark_results.append({"k": k, "silhouette": sil})
        print(f"  KMeans k={k}: silhouette={sil:.4f}")

    # DBCV from HDBSCAN (density-based cluster validation)
    try:
        dbcv_score = float(clusterer.relative_validity_)
        print(f"  HDBSCAN DBCV (relative_validity_): {dbcv_score:.4f}")
    except AttributeError:
        dbcv_score = None
        print("  HDBSCAN DBCV not available (requires prediction data).")

    benchmark_payload = {
        "kmeans": benchmark_results,
        "hdbscan_dbcv": dbcv_score,
    }
    save_json(benchmark_payload, BENCHMARK_OUT_PATH)
    print(f"Saved KMeans benchmark to {BENCHMARK_OUT_PATH}")


if __name__ == "__main__":
    main()
