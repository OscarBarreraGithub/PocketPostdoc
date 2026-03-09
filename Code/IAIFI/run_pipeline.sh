#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

echo "=== IAIFI Paperscape Pipeline ==="
echo "Step A: Collect IAIFI papers"
python3 -m iaifi_paperscape.collect.build_iaifi_seed

echo "Step B: Collect background candidates"
python3 -m iaifi_paperscape.collect.build_background_corpus

echo "Step D: Embed all papers"
python3 -m iaifi_paperscape.embed.specter2_embed

echo "Step C: Select background (kNN union)"
python3 -m iaifi_paperscape.collect.select_background

echo "Step E: PCA + UMAP"
python3 -m iaifi_paperscape.reduce.run_pca_umap

echo "Step F: Cluster + label"
python3 -m iaifi_paperscape.cluster.run_hdbscan
python3 -m iaifi_paperscape.cluster.label_clusters
python3 -m iaifi_paperscape.cluster.enrichment

echo "Step G: Export web assets"
python3 -m iaifi_paperscape.export.compute_neighbors
python3 -m iaifi_paperscape.export.export_web_assets

echo "=== Done. Output: web/data/papers.json ==="
