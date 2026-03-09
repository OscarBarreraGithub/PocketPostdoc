"""Step F.4: per-cluster IAIFI enrichment scores with Bayesian smoothing.

Usage:
    python -m iaifi_paperscape.cluster.enrichment

Computes log2 enrichment of IAIFI papers in each cluster relative to the
overall IAIFI fraction, with 95% confidence intervals from the beta
distribution. Clusters smaller than ``min_cluster_size`` are skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

from iaifi_paperscape.utils.io import (
    load_json,
    load_jsonl,
    load_yaml_config,
    save_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "clustering.yaml"
CLUSTERS_PATH = PROJECT_ROOT / "data" / "processed" / "clusters.json"
IAIFI_METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "iaifi_metadata.jsonl"
EMBEDDING_INDEX_PATH = PROJECT_ROOT / "data" / "interim" / "embedding_index.json"
KNN_SELECTED_PATH = PROJECT_ROOT / "data" / "interim" / "knn_selected_ids.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "enrichment.json"


def _load_row_to_arxiv_id(path: Path) -> list[str]:
    """Load the embedding index."""
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


# ---------------------------------------------------------------------------
# Core enrichment computation (matches PLAN.md F.4 exactly)
# ---------------------------------------------------------------------------

def compute_enrichment(
    labels: np.ndarray,
    is_iaifi: np.ndarray,
    alpha: float = 1,
    beta: float = 1,
    min_cluster_size: int = 30,
) -> dict[int, dict | None]:
    """Per-cluster enrichment = log2(pC / pG).

    Parameters
    ----------
    labels : ndarray of int
        Cluster assignment per paper (-1 = noise).
    is_iaifi : ndarray of bool
        True for IAIFI papers.
    alpha, beta : float
        Bayesian smoothing parameters (Beta prior).
    min_cluster_size : int
        Skip clusters smaller than this.

    Returns
    -------
    enrichment : dict mapping cluster_id -> enrichment dict or None
    """
    n_total = len(is_iaifi)
    n_iaifi_total = int(is_iaifi.sum())
    pG = (n_iaifi_total + alpha) / (n_total + alpha + beta)

    enrichment: dict[int, dict | None] = {}
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        mask = labels == cluster_id
        n_cluster = int(mask.sum())
        n_iaifi_cluster = int((is_iaifi & mask).sum())

        if n_cluster < min_cluster_size:
            enrichment[cluster_id] = None
            continue

        pC = (n_iaifi_cluster + alpha) / (n_cluster + alpha + beta)
        log_enrichment = float(np.log2(pC / pG))

        # Uncertainty: approximate 95% CI using beta distribution
        lo = float(
            beta_dist.ppf(
                0.025,
                n_iaifi_cluster + alpha,
                n_cluster - n_iaifi_cluster + beta,
            )
        )
        hi = float(
            beta_dist.ppf(
                0.975,
                n_iaifi_cluster + alpha,
                n_cluster - n_iaifi_cluster + beta,
            )
        )

        enrichment[cluster_id] = {
            "log2_enrichment": log_enrichment,
            "iaifi_fraction": float(pC),
            "cluster_size": n_cluster,
            "iaifi_in_cluster": n_iaifi_cluster,
            "ci_95_lo": float(np.log2(lo / pG)) if lo > 0 else None,
            "ci_95_hi": float(np.log2(hi / pG)),
        }

    return enrichment


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config = load_yaml_config(CONFIG_PATH)
    enrichment_cfg = config.get("enrichment", {})
    alpha = float(enrichment_cfg.get("smoothing_alpha", 1))
    beta_val = float(enrichment_cfg.get("smoothing_beta", 1))
    min_size = int(enrichment_cfg.get("min_cluster_size", 30))

    # -----------------------------------------------------------------------
    # Load cluster assignments
    # -----------------------------------------------------------------------
    clusters_payload = load_json(CLUSTERS_PATH)
    labels = np.array(clusters_payload["labels"], dtype=int)

    # -----------------------------------------------------------------------
    # Build is_iaifi boolean mask over the selected subset
    # -----------------------------------------------------------------------
    iaifi_metadata = load_jsonl(IAIFI_METADATA_PATH)
    iaifi_count = len(iaifi_metadata)
    row_to_arxiv_id = _load_row_to_arxiv_id(EMBEDDING_INDEX_PATH)
    knn_selected_ids = set(load_json(KNN_SELECTED_PATH))

    # Reconstruct the selected-subset ordering (IAIFI first, then kNN background)
    is_iaifi_list: list[bool] = []
    for i, aid in enumerate(row_to_arxiv_id):
        if i < iaifi_count:
            is_iaifi_list.append(True)
        elif aid in knn_selected_ids:
            is_iaifi_list.append(False)
    is_iaifi = np.array(is_iaifi_list, dtype=bool)

    if len(is_iaifi) != len(labels):
        print(
            f"WARNING: is_iaifi length ({len(is_iaifi)}) != labels length ({len(labels)}). "
            "Truncating to min."
        )
        n = min(len(is_iaifi), len(labels))
        is_iaifi = is_iaifi[:n]
        labels = labels[:n]

    print(
        f"Papers: {len(labels)}, IAIFI: {int(is_iaifi.sum())}, "
        f"Background: {int((~is_iaifi).sum())}"
    )

    # -----------------------------------------------------------------------
    # Compute enrichment
    # -----------------------------------------------------------------------
    enrichment = compute_enrichment(
        labels, is_iaifi, alpha=alpha, beta=beta_val, min_cluster_size=min_size
    )

    # Print summary
    for cid, info in sorted(enrichment.items()):
        if info is None:
            print(f"  Cluster {cid}: skipped (too small)")
        else:
            print(
                f"  Cluster {cid}: log2_enrichment={info['log2_enrichment']:.3f}, "
                f"iaifi={info['iaifi_in_cluster']}/{info['cluster_size']}, "
                f"CI=[{info['ci_95_lo']}, {info['ci_95_hi']:.3f}]"
            )

    # Convert int keys to strings for JSON serialization
    serializable = {
        str(k): v for k, v in enrichment.items()
    }
    save_json(serializable, OUTPUT_PATH)
    print(f"Saved enrichment scores to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
