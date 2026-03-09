.PHONY: all collect embed select reduce cluster export site clean

PYTHON = PYTHONPATH=src python3
PKG = iaifi_paperscape

all: site

# Step A: Collect IAIFI papers
data/raw/iaifi_metadata.jsonl: data/external/iaifi_papers.csv
	$(PYTHON) -m $(PKG).collect.build_iaifi_seed

# Step B: Collect background candidates
data/raw/background_candidates.jsonl: configs/corpus.yaml
	$(PYTHON) -m $(PKG).collect.build_background_corpus

# Step D: Embed all papers (IAIFI + candidates)
data/interim/embeddings.npy: data/raw/iaifi_metadata.jsonl data/raw/background_candidates.jsonl
	$(PYTHON) -m $(PKG).embed.specter2_embed

# Step C: Select background via kNN (requires embeddings)
data/interim/knn_selected_ids.json: data/interim/embeddings.npy
	$(PYTHON) -m $(PKG).collect.select_background

# Step E: PCA + UMAP
data/processed/coords_2d.npy data/interim/pca_50.npy: data/interim/knn_selected_ids.json data/interim/embeddings.npy
	$(PYTHON) -m $(PKG).reduce.run_pca_umap

# Step F: Cluster + label + enrichment
data/processed/clusters.json: data/processed/coords_2d.npy data/interim/pca_50.npy
	$(PYTHON) -m $(PKG).cluster.run_hdbscan
	$(PYTHON) -m $(PKG).cluster.label_clusters
	$(PYTHON) -m $(PKG).cluster.enrichment

# Step G: Export
web/data/papers.json: data/processed/clusters.json data/processed/coords_2d.npy data/processed/cluster_labels.json data/processed/enrichment.json data/processed/neighbors.json
	$(PYTHON) -m $(PKG).export.compute_neighbors
	$(PYTHON) -m $(PKG).export.export_web_assets

collect: data/raw/iaifi_metadata.jsonl data/raw/background_candidates.jsonl
embed: data/interim/embeddings.npy
select: data/interim/knn_selected_ids.json
reduce: data/processed/coords_2d.npy
cluster: data/processed/clusters.json
export site: web/data/papers.json

clean:
	rm -rf data/interim/ data/processed/ data/models/ web/data/papers.json web/data/abstracts.json
