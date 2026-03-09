"""Step G.1: precompute top-10 nearest neighbors per paper.

Usage:
    python -m iaifi_paperscape.export.compute_neighbors

Builds a cosine-similarity kNN graph on L2-normalized 768-d embeddings of
the selected paper subset (IAIFI + kNN-selected background).  For each
paper the 10 closest neighbors (excluding self) are stored with similarity
scores.  Hubness diagnostics are logged to stdout.

Output: data/processed/neighbors.json
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from iaifi_paperscape.utils.io import load_json, load_numpy, save_json

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "interim" / "embeddings.npy"
EMBEDDING_INDEX_PATH = PROJECT_ROOT / "data" / "interim" / "embedding_index.json"
KNN_SELECTED_PATH = PROJECT_ROOT / "data" / "interim" / "knn_selected_ids.json"
IAIFI_METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "iaifi_metadata.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "neighbors.json"


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


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def compute_neighbors(
    embeddings: np.ndarray,
    paper_ids: list[str],
    n_neighbors: int = 10,
) -> dict[str, list[dict]]:
    """Build kNN graph and return neighbor lists.

    Parameters
    ----------
    embeddings : ndarray, shape (N, 768)
        L2-normalized embeddings for selected papers.
    paper_ids : list[str]
        arXiv IDs aligned with embedding rows.
    n_neighbors : int
        Number of neighbors to keep (excluding self).

    Returns
    -------
    neighbors : dict mapping arxiv_id -> list of {id, similarity}
    """
    if embeddings.shape[0] != len(paper_ids):
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) != paper_ids length ({len(paper_ids)})"
        )

    # L2-normalize (should already be normalized, but be safe)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings_normed = embeddings / norms

    # Fit kNN (n_neighbors + 1 to account for self-match)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
    nn.fit(embeddings_normed)
    distances, indices = nn.kneighbors(embeddings_normed)

    # Build neighbor dict (skip self at index 0)
    neighbors: dict[str, list[dict]] = {}
    for i, pid in enumerate(paper_ids):
        nbrs = []
        for j in range(1, n_neighbors + 1):
            nbrs.append({
                "id": paper_ids[indices[i][j]],
                "similarity": round(float(1.0 - distances[i][j]), 6),
            })
        neighbors[pid] = nbrs

    return neighbors


def log_hubness_diagnostics(
    neighbors: dict[str, list[dict]],
    paper_ids: list[str],
) -> None:
    """Print hubness diagnostics: p50, p95, max incoming share."""
    incoming: Counter[str] = Counter()
    for nbrs in neighbors.values():
        for edge in nbrs:
            incoming[edge["id"]] += 1

    n = len(paper_ids)
    incoming_share = np.array([incoming.get(pid, 0) / n for pid in paper_ids])

    p50 = float(np.percentile(incoming_share, 50))
    p95 = float(np.percentile(incoming_share, 95))
    max_share = float(incoming_share.max())

    print(
        f"Final graph hubness: "
        f"p50={p50:.3f}, "
        f"p95={p95:.3f}, "
        f"max={max_share:.3f}"
    )

    # Identify the top hub paper
    hub_idx = int(np.argmax(incoming_share))
    hub_id = paper_ids[hub_idx]
    hub_count = incoming[hub_id]
    print(f"Top hub: {hub_id} (appears as neighbor of {hub_count} papers)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Step G.1: computing nearest neighbors...")

    # Load data
    all_embeddings = load_numpy(EMBEDDINGS_PATH)
    row_to_arxiv_id = _load_row_to_arxiv_id(EMBEDDING_INDEX_PATH)
    knn_selected_ids = set(load_json(KNN_SELECTED_PATH))

    print(f"Full embedding matrix: {all_embeddings.shape}")

    # Count IAIFI papers (they come first in the embedding index)
    from iaifi_paperscape.utils.io import load_jsonl
    iaifi_metadata = load_jsonl(IAIFI_METADATA_PATH)
    n_iaifi = len(iaifi_metadata)

    # Build selected subset: IAIFI papers (first n_iaifi rows) + kNN-selected background
    selected_indices: list[int] = []
    for i, aid in enumerate(row_to_arxiv_id):
        if i < n_iaifi or aid in knn_selected_ids:
            selected_indices.append(i)

    selected_indices_arr = np.array(selected_indices)
    selected_embeddings = all_embeddings[selected_indices_arr]
    selected_ids = [row_to_arxiv_id[i] for i in selected_indices]

    print(
        f"Selected papers: {len(selected_ids)} "
        f"(IAIFI: {n_iaifi}, background: {len(selected_ids) - n_iaifi})"
    )

    if selected_embeddings.shape[1] != 768:
        print(
            f"WARNING: expected 768-d embeddings, got {selected_embeddings.shape[1]}-d. "
            "Proceeding anyway."
        )

    # Compute neighbors
    neighbors = compute_neighbors(selected_embeddings, selected_ids)

    # Hubness diagnostics
    log_hubness_diagnostics(neighbors, selected_ids)

    # Save
    save_json(neighbors, OUTPUT_PATH)
    print(f"Saved neighbors for {len(neighbors)} papers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
