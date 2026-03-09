"""Step E.3: densMAP evaluation (test-only, NOT used in production pipeline).

Usage:
    PYTHONPATH=src python3 -m iaifi_paperscape.reduce.densmap_eval

Compares standard UMAP (already computed) vs densMAP on the same PCA-50 input.
Outputs trustworthiness/continuity metrics to stdout and saves densMAP coords
for visual comparison.

This script is for evaluation only.  The production pipeline uses standard UMAP
(run_pca_umap.py).  densMAP preserves local density information more faithfully
but may produce less visually appealing layouts.  Run this to decide whether
densMAP is worth switching to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import umap
from sklearn.manifold import trustworthiness

from iaifi_paperscape.utils.io import load_numpy, load_yaml_config, save_numpy

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "umap.yaml"
PCA_INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "pca_50.npy"
COORDS_UMAP_PATH = PROJECT_ROOT / "data" / "processed" / "coords_2d.npy"
COORDS_DENSMAP_PATH = PROJECT_ROOT / "data" / "processed" / "coords_2d_densmap.npy"


def _continuity(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 15) -> float:
    """Compute continuity: whether high-d neighbors remain close in low-d.

    Implemented as trustworthiness with high/low arguments swapped.
    """
    return float(trustworthiness(X_low, X_high, n_neighbors=n_neighbors))


def main() -> None:
    print("Step E.3: densMAP evaluation (test-only)")
    print("=" * 60)

    # Load config and data
    config = load_yaml_config(CONFIG_PATH)
    umap_cfg = config.get("umap", {})

    pca_50 = load_numpy(PCA_INPUT_PATH)
    print(f"PCA-50 input shape: {pca_50.shape}")

    # Load standard UMAP coords (already computed by run_pca_umap.py)
    coords_umap = load_numpy(COORDS_UMAP_PATH)
    print(f"Standard UMAP coords shape: {coords_umap.shape}")

    # Run densMAP with the same parameters
    print("\nFitting densMAP (this may take a few minutes)...")
    reducer = umap.UMAP(
        n_neighbors=int(umap_cfg.get("n_neighbors", 40)),
        min_dist=float(umap_cfg.get("min_dist", 0.08)),
        n_components=int(umap_cfg.get("n_components", 2)),
        metric=str(umap_cfg.get("metric", "cosine")),
        random_state=int(umap_cfg.get("random_state", 42)),
        densmap=True,
    )
    coords_densmap = reducer.fit_transform(pca_50)
    save_numpy(coords_densmap, COORDS_DENSMAP_PATH)
    print(f"Saved densMAP coords to {COORDS_DENSMAP_PATH}")

    # Evaluate both embeddings
    k = 15
    print(f"\nComputing trustworthiness and continuity (k={k})...")

    trust_umap = float(trustworthiness(pca_50, coords_umap, n_neighbors=k))
    trust_densmap = float(trustworthiness(pca_50, coords_densmap, n_neighbors=k))

    cont_umap = _continuity(pca_50, coords_umap, n_neighbors=k)
    cont_densmap = _continuity(pca_50, coords_densmap, n_neighbors=k)

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'Metric':<25} {'UMAP':>12} {'densMAP':>12} {'Winner':>10}")
    print("-" * 60)
    print(
        f"{'Trustworthiness':<25} {trust_umap:>12.6f} {trust_densmap:>12.6f} "
        f"{'densMAP' if trust_densmap > trust_umap else 'UMAP':>10}"
    )
    print(
        f"{'Continuity':<25} {cont_umap:>12.6f} {cont_densmap:>12.6f} "
        f"{'densMAP' if cont_densmap > cont_umap else 'UMAP':>10}"
    )
    print("=" * 60)

    print(
        "\nNote: trustworthiness measures whether low-d neighbors are true "
        "high-d neighbors (precision). Continuity measures whether high-d "
        "neighbors remain close in low-d (recall). Higher is better for both."
    )
    print(
        "densMAP additionally preserves local density. Inspect the saved "
        "coords visually to judge layout quality."
    )


if __name__ == "__main__":
    main()
