"""Step F.3: auto-generate cluster label candidates via cTF-IDF.

Usage:
    python -m iaifi_paperscape.cluster.label_clusters

For each non-noise cluster, computes class-based TF-IDF on concatenated
titles + abstracts, extracts top-3 n-gram candidates, and saves results
alongside HDBSCAN stability scores.

Human curation via ``configs/theme_overrides.yml`` is MANDATORY before
deployment. Auto-labels are development aids, not user-facing text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from iaifi_paperscape.utils.io import (
    load_json,
    load_jsonl,
    load_yaml_config,
    save_json,
)
from iaifi_paperscape.utils.text import SCIENCE_STOPWORDS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "clustering.yaml"
CLUSTERS_PATH = PROJECT_ROOT / "data" / "processed" / "clusters.json"
IAIFI_METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "iaifi_metadata.jsonl"
BACKGROUND_PATH = PROJECT_ROOT / "data" / "raw" / "background_candidates.jsonl"
KNN_SELECTED_PATH = PROJECT_ROOT / "data" / "interim" / "knn_selected_ids.json"
EMBEDDING_INDEX_PATH = PROJECT_ROOT / "data" / "interim" / "embedding_index.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cluster_labels.json"


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


def _build_paper_lookup(
    iaifi_metadata: list[dict],
    background_metadata: list[dict],
) -> dict[str, dict]:
    """Build arxiv_id -> paper metadata dict from both corpora."""
    lookup: dict[str, dict] = {}
    for p in iaifi_metadata:
        aid = str(p.get("arxiv_id", p.get("id", "")))
        if aid:
            lookup[aid] = p
    for p in background_metadata:
        aid = str(p.get("arxiv_id", p.get("id", "")))
        if aid:
            lookup[aid] = p
    return lookup


def _paper_text(paper: dict) -> str:
    """Return concatenated title + abstract for TF-IDF."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", paper.get("summary", ""))
    return f"{title} {abstract}"


def main() -> None:
    config = load_yaml_config(CONFIG_PATH)
    labels_cfg = config.get("labels", {})
    ngram_range = tuple(labels_cfg.get("ngram_range", [1, 3]))
    n_candidates = int(labels_cfg.get("n_candidates", 3))

    # -----------------------------------------------------------------------
    # Load cluster assignments
    # -----------------------------------------------------------------------
    clusters_payload = load_json(CLUSTERS_PATH)
    labels = np.array(clusters_payload["labels"], dtype=int)
    label_to_persistence = {
        int(c["id"]): float(c["stability"])
        for c in clusters_payload["clusters"]
    }

    # -----------------------------------------------------------------------
    # Load paper metadata and build selected-subset ordering
    # -----------------------------------------------------------------------
    iaifi_metadata = load_jsonl(IAIFI_METADATA_PATH)
    background_metadata = load_jsonl(BACKGROUND_PATH)
    paper_lookup = _build_paper_lookup(iaifi_metadata, background_metadata)

    row_to_arxiv_id = _load_row_to_arxiv_id(EMBEDDING_INDEX_PATH)
    knn_selected_ids = set(load_json(KNN_SELECTED_PATH))
    iaifi_count = len(iaifi_metadata)

    # Build ordered list of papers matching the selected subset
    papers: list[dict] = []
    for i, aid in enumerate(row_to_arxiv_id):
        if i < iaifi_count or aid in knn_selected_ids:
            papers.append(paper_lookup.get(aid, {"arxiv_id": aid}))

    if len(papers) != len(labels):
        print(
            f"WARNING: paper count ({len(papers)}) != label count ({len(labels)}). "
            "Using min of both."
        )
        n = min(len(papers), len(labels))
        papers = papers[:n]
        labels = labels[:n]

    # -----------------------------------------------------------------------
    # cTF-IDF per cluster
    # -----------------------------------------------------------------------
    unique_labels = sorted(set(labels) - {-1})
    cluster_labels: dict[str, dict] = {}

    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id
        cluster_texts = [_paper_text(papers[i]) for i in range(len(papers)) if cluster_mask[i]]
        rest_texts = [_paper_text(papers[i]) for i in range(len(papers)) if not cluster_mask[i]]

        if not cluster_texts:
            continue

        # cTF-IDF: fit on all documents, then compare cluster vs rest
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=10000,
            stop_words=list(SCIENCE_STOPWORDS),
        )

        all_texts = cluster_texts + rest_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Mean TF-IDF vector for cluster documents vs rest
        cluster_vec = tfidf_matrix[: len(cluster_texts)].mean(axis=0)
        rest_vec = tfidf_matrix[len(cluster_texts) :].mean(axis=0)

        # cTF-IDF score: cluster mean - rest mean (higher = more distinctive)
        diff = np.asarray(cluster_vec - rest_vec).flatten()
        feature_names = vectorizer.get_feature_names_out()
        top_indices = diff.argsort()[::-1][:n_candidates]
        top_ngrams = [str(feature_names[i]) for i in top_indices]

        cluster_labels[str(int(cluster_id))] = {
            "auto_label": top_ngrams[0] if top_ngrams else "",
            "label_candidates": top_ngrams,
            "size": int(cluster_mask.sum()),
            "stability": float(label_to_persistence.get(int(cluster_id), 0.0)),
        }

        print(
            f"Cluster {cluster_id}: size={cluster_mask.sum()}, "
            f"stability={label_to_persistence.get(int(cluster_id), 0.0):.3f}, "
            f"candidates={top_ngrams}"
        )

    save_json(cluster_labels, OUTPUT_PATH)
    print(f"Saved {len(cluster_labels)} cluster labels to {OUTPUT_PATH}")
    print(
        "REMINDER: Human curation via configs/theme_overrides.yml is MANDATORY "
        "before deployment."
    )


if __name__ == "__main__":
    main()
