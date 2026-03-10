"""Step F.1: HDBSCAN clustering with stability gating + KMeans benchmark.

Usage:
    python -m iaifi_paperscape.cluster.run_hdbscan

Loads L2-normalized PCA-50 embeddings, runs HDBSCAN with stability gating,
saves cluster assignments, then runs an offline KMeans benchmark (F.2).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from iaifi_paperscape.utils.io import load_numpy, load_yaml_config, save_json

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "clustering.yaml"
PCA_PATH = PROJECT_ROOT / "data" / "interim" / "pca_50.npy"
CLUSTERS_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "clusters.json"
BENCHMARK_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "kmeans_benchmark.json"


# ---------------------------------------------------------------------------
# Label -> stability mapping using sklearn.cluster.HDBSCAN
# ---------------------------------------------------------------------------

def build_label_persistence_map(clusterer: HDBSCAN) -> dict[int, float]:
    """Map from output cluster label -> stability proxy.

    sklearn's HDBSCAN does not expose the private condensed-tree internals
    that the standalone ``hdbscan`` package had.  Instead we use the mean
    membership probability (``clusterer.probabilities_``) of points assigned
    to each cluster as a proxy for cluster stability.  Values range [0, 1].
    """
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_
    # sklearn HDBSCAN may emit -1 (noise), -2 (inf rows), -3 (nan rows);
    # exclude all negative labels so we only iterate real clusters.
    unique_labels = sorted(lbl for lbl in set(labels) if lbl >= 0)

    label_to_stability: dict[int, float] = {}
    for cid in unique_labels:
        mask = labels == cid
        label_to_stability[int(cid)] = float(np.mean(probabilities[mask]))

    return label_to_stability


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
    clusterer = HDBSCAN(
        min_cluster_size=int(hdbscan_cfg.get("min_cluster_size", 50)),
        min_samples=int(hdbscan_cfg.get("min_samples", 10)),
        metric=str(hdbscan_cfg.get("metric", "euclidean")),
        cluster_selection_method=str(hdbscan_cfg.get("cluster_selection_method", "eom")),
        store_centers="centroid",
    )
    labels = clusterer.fit_predict(pca_embeddings)
    probabilities = clusterer.probabilities_

    label_to_persistence = build_label_persistence_map(clusterer)

    # Exclude all negative labels (noise / inf / nan markers from sklearn)
    unique_before = sorted(lbl for lbl in set(labels) if lbl >= 0)
    print(f"HDBSCAN found {len(unique_before)} clusters before stability gating.")

    # -----------------------------------------------------------------------
    # Stability gating: demote weak clusters to noise (-1)
    # -----------------------------------------------------------------------
    min_size = int(stability_cfg.get("min_size", 50))
    # NOTE: This threshold is based on mean membership probability (range
    # [0, 1]), not the old cluster_persistence_ metric.  0.3 is a
    # conservative starting point; tune after inspecting the first run.
    min_stability = float(stability_cfg.get("min_stability", 0.3))

    demoted = []
    for cluster_id in unique_before:
        mask = labels == cluster_id
        persistence = label_to_persistence.get(int(cluster_id), 0.0)
        if mask.sum() < min_size or persistence < min_stability:
            labels[mask] = -1  # demote to noise
            demoted.append(cluster_id)

    unique_after = sorted(lbl for lbl in set(labels) if lbl >= 0)
    n_noise = int((labels < 0).sum())
    print(
        f"After stability gating: {len(unique_after)} clusters "
        f"({len(demoted)} demoted to noise)."
    )
    print(f"Noise points: {n_noise} / {len(labels)}")

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
        "n_noise": int((labels < 0).sum()),
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
