# IAIFI Paperscape

An interactive 2D scatterplot visualization of ~9,000 physics and machine learning
papers relevant to [IAIFI](https://iaifi.org/) (Institute for Artificial Intelligence
and Fundamental Interactions). 547 IAIFI papers are highlighted against ~8,000
contextual background papers drawn from arXiv.

Background papers are selected via diversity-aware kNN union: each IAIFI paper
nominates its nearest neighbors in embedding space, duplicates are removed, and a
bridge-ballast sample is added for cartographic continuity. Papers are embedded with
SPECTER2 (proximity adapter, 768-d, L2-normalized), reduced to 2D via PCA (50 dims)
followed by UMAP, and clustered with HDBSCAN. Clusters are auto-labeled with TF-IDF
and scored for IAIFI enrichment.

The result is a navigable map of IAIFI's research landscape: where the institute
concentrates its effort, which subfields overlap, and where gaps or opportunities lie.

## Quick Start

```bash
pip install -r requirements.txt

# Populate the seed file with real IAIFI paper arXiv IDs
# (columns: arxiv_id, title, authors, published, iaifi_theme)
vim data/external/iaifi_papers.csv

# Run the full pipeline
bash run_pipeline.sh   # or: make

# Serve the frontend
cd web && python3 -m http.server 8080
```

Open `http://localhost:8080` in a browser.

## Pipeline Overview

The pipeline runs in an unusual order -- embedding happens *before* kNN selection so
that background papers can be chosen based on semantic proximity to IAIFI seeds.

| Step | Script | What it does |
|------|--------|--------------|
| **A** | `collect.build_iaifi_seed` | Fetch IAIFI paper metadata from arXiv using IDs in `data/external/iaifi_papers.csv` |
| **B** | `collect.build_background_corpus` | Harvest ~30K background candidate papers across 12 arXiv categories (2018--present) |
| **D** | `embed.specter2_embed` | Embed all papers (IAIFI + candidates) with SPECTER2 proximity adapter (768-d, L2-normalized) |
| **C** | `collect.select_background` | Select ~8--10K background papers via diversity-aware kNN union + 1K bridge ballast |
| **E** | `reduce.run_pca_umap` | PCA to 50 dims, then UMAP to 2D (cosine metric, 40 neighbors, min_dist 0.08) |
| **F** | `cluster.run_hdbscan` / `label_clusters` / `enrichment` | HDBSCAN clustering, TF-IDF auto-labeling, and IAIFI enrichment scoring |
| **G** | `export.compute_neighbors` / `export_web_assets` | Compute top-10 nearest neighbors per paper and export `web/data/papers.json` |
| **H** | `web/` | Static frontend with regl-scatterplot |

All modules live under `src/iaifi_paperscape/` and are invoked as `python3 -m iaifi_paperscape.<module>`.

## Frontend Features

- Interactive scatter plot (zoom, pan) via regl-scatterplot
- Click any paper for title, authors, abstract, arXiv link, and top-10 nearest neighbors
- Search by title or author
- Filter by IAIFI theme (AI / Physics / Both)
- Color by theme, arXiv category, or cluster enrichment
- Year range slider
- Cluster labels displayed at medium zoom
- Optional citation-based point sizing

## Project Structure

```
.
├── configs/
│   ├── corpus.yaml           # arXiv categories, date range, kNN selection params
│   ├── embedding.yaml        # SPECTER2 model settings
│   ├── umap.yaml             # PCA + UMAP hyperparameters, Procrustes config
│   ├── clustering.yaml       # HDBSCAN, labeling, enrichment settings
│   └── theme_overrides.yml   # Human-curated cluster label overrides
├── data/
│   ├── external/             # Input: iaifi_papers.csv (seed arXiv IDs)
│   ├── raw/                  # arXiv API responses
│   ├── interim/              # Embeddings, PCA, kNN selection
│   ├── processed/            # 2D coords, clusters, neighbors
│   └── models/               # Cached model artifacts
├── src/iaifi_paperscape/
│   ├── collect/              # Steps A, B, C (arXiv fetch + kNN selection)
│   ├── embed/                # Step D (SPECTER2)
│   ├── reduce/               # Step E (PCA + UMAP + Procrustes)
│   ├── cluster/              # Step F (HDBSCAN + labels + enrichment)
│   ├── export/               # Step G (neighbors + JSON export)
│   └── utils/                # I/O helpers, text processing
├── web/
│   ├── index.html            # Frontend entry point
│   ├── style.css
│   ├── scatter.js            # regl-scatterplot integration
│   └── data/                 # Generated: papers.json
├── notebooks/                # Exploratory analysis
├── .github/workflows/
│   ├── monthly_refresh.yml   # Automated monthly pipeline rerun
│   └── deploy.yml            # GitHub Pages deployment
├── run_pipeline.sh           # Run all steps sequentially
├── Makefile                  # Dependency-aware build (make targets: all, collect, embed, select, reduce, cluster, export, clean)
└── requirements.txt
```

## Configuration

All tunable parameters live in `configs/`:

- **`corpus.yaml`** -- arXiv categories, date range, candidate cap (30K), kNN selection
  method, diversity filtering, and bridge-ballast size.
- **`embedding.yaml`** -- SPECTER2 model and adapter, batch size, device, L2
  normalization, and sanity-check settings.
- **`umap.yaml`** -- PCA components (50), UMAP hyperparameters (n_neighbors=40,
  min_dist=0.08, cosine metric), and monthly-refit Procrustes settings.
- **`clustering.yaml`** -- HDBSCAN (min_cluster_size=50), TF-IDF labeling, enrichment
  formula (`log(pC/pG)` with Bayesian smoothing), and noise treatment.
- **`theme_overrides.yml`** -- Human-curated cluster labels. Auto-generated labels are
  never shipped without review; this file is the override layer.

## Monthly Updates

A GitHub Action (`.github/workflows/monthly_refresh.yml`) reruns the full pipeline on
the 1st of each month at 06:00 UTC and deploys updated data to GitHub Pages.

**Procrustes alignment** keeps the map stable between monthly refits. IAIFI papers serve
as anchor points: after UMAP refitting, the new coordinates are aligned to the previous
month's layout via Procrustes rotation, so returning users see a consistent spatial
arrangement. The previous month's coordinates are always saved for comparison.

For incremental updates (a few new papers between monthly refits), UMAP's `.transform()`
method places new points without refitting the full model.

## Pre-Launch Checklist

- [ ] Populate `data/external/iaifi_papers.csv` with the full IAIFI paper list
- [ ] Run pipeline end-to-end: `bash run_pipeline.sh`
- [ ] Review auto-generated cluster labels in `data/processed/cluster_labels.json`
- [ ] Populate `configs/theme_overrides.yml` with curated labels for each cluster
- [ ] Run with `--strict` flag to verify no missing overrides
- [ ] Deploy to GitHub Pages
