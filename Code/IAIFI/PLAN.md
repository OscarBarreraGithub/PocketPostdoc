# IAIFI Paperscape v2 — Implementation Plan

**Version:** 2.0 (Final)
**Date:** 2026-03-09
**Status:** Approved by Claude + Codex cross-review (v2 update)

---

## Success Criteria

These four properties define "done." If any fails, the project has not shipped.

1. **Usable at zoomed-out view.** IAIFI concentrations are visible against the background without a grey snowstorm obscuring structure. Background points recede; IAIFI points pop.
2. **Actionable on click.** Clicking any paper yields a high-quality "read next" neighborhood: title, abstract, top-5 nearest neighbors (with "show more" to 10), and an arXiv link. The neighborhood must feel topically coherent to a domain expert.
3. **Stable across updates.** A monthly corpus refresh does not scramble the map. Returning users recognize the same landscape. Achieved via Procrustes alignment on IAIFI anchor points plus drift diagnostics (median + 95th percentile anchor displacement) with a defined high-drift policy.
4. **Defensible semantics.** Local distances (within a cluster or neighborhood) are meaningful. Global geometry (distance between clusters) is approximate and disclaimed in the UI. No claims about "the field" that the embedding cannot support.

---

## Failure Modes and Mitigations

| Failure Mode | Cause | Mitigation |
|---|---|---|
| Background misses IAIFI modes | Centroid-based selection averages over multi-modal distribution | kNN union from each IAIFI paper (solved in v1.5) |
| Giant grey point cloud | kNN-union over-samples dense modes and introduces redundant neighbors | Diversity-aware kNN union (per-seed dedup + anti-hub scoring) plus multiscale rendering (v1.5b) |
| UMAP drift between refreshes | Background refresh changes graph topology; alignment alone cannot fully prevent movement | Procrustes alignment + drift diagnostics (anchor displacement median/p95) + freeze/warn policy for high-drift refreshes |
| Fake clusters | HDBSCAN finds structure in noise | Stability gating (n >= 50 AND stability >= threshold) + human curation |
| Meaningless auto-labels | TF-IDF returns "quantum field gauge" | Human override via `configs/theme_overrides.yml`; never ship raw labels |
| Over-interpretation of 2D layout | Users read global distances as meaningful | UI disclaimer: "Local neighborhoods are meaningful; global geometry is approximate" |
| arXiv IP ban | Too-fast API requests | Enforced 3s delay, single connection, disk caching |
| Hub papers dominate kNN | Some papers are nearest neighbors of many IAIFI papers | Monitor hubness (p50/p95/max neighbor share); optional inclusion cap/diversity sampling if UX degrades |

---

## 0. Dependencies

### Python (pip-installable, no GPU required)

```
transformers>=4.30
adapters>=0.2
torch>=2.0
umap-learn>=0.5
hdbscan>=0.8
scikit-learn>=1.3
numpy>=1.24
pandas>=2.0
feedparser>=6.0
requests>=2.31
tqdm>=4.65
pyyaml>=6.0
scipy>=1.11
joblib>=1.3
```

Total install footprint: ~2GB (mostly PyTorch). All CPU-only.

### Frontend (no build step)

- `regl-scatterplot` via CDN (~50KB)
- Vanilla JS + HTML + CSS
- No React, no bundler, no npm

### System

- Python 3.10+
- make (for pipeline orchestration)
- git
- gzip (for asset compression)

---

## 1. Directory Structure

```
Code/IAIFI/
  PLAN.md                          # this file
  Makefile                         # orchestrates pipeline steps A-H
  requirements.txt                 # Python dependencies
  run_pipeline.sh                  # alternative: sequential script runner

  configs/
    corpus.yaml                    # candidate pool categories, date range, cap
    embedding.yaml                 # model name, batch size, adapter
    umap.yaml                      # PCA dims, UMAP params
    clustering.yaml                # HDBSCAN params, stability thresholds
    theme_overrides.yml            # human-curated cluster labels (MANDATORY)

  data/                            # gitignored EXCEPT data/external/
    external/
      iaifi_papers.csv             # checked in — canonical IAIFI paper list
    raw/
      arxiv_api/                   # cached Atom XML responses
      iaifi_metadata.jsonl         # IAIFI paper metadata from arXiv
      background_candidates.jsonl  # full 30K candidate pool
    interim/
      corpus.parquet               # deduplicated, normalized records
      embeddings.npy               # 768-d float32 vectors (N x 768)
      embedding_index.json         # row index -> arxiv_id mapping
      knn_selected_ids.json        # arxiv IDs selected by kNN union
      pca_50.npy                   # PCA-reduced embeddings (N x 50)
    processed/
      coords_2d.npy               # UMAP output (N x 2)
      clusters.json                # HDBSCAN labels + stability scores
      cluster_labels.json          # auto-generated label candidates
      enrichment.json              # per-cluster enrichment scores
      neighbors.json               # top-10 neighbors per paper
    models/
      pca_model.pkl                # fitted PCA for .transform()
      umap_model.pkl               # fitted UMAP for .transform()

  src/iaifi_paperscape/
    __init__.py
    collect/
      __init__.py
      arxiv_client.py              # rate-limited arXiv API wrapper
      build_iaifi_seed.py          # step A: fetch IAIFI paper metadata
      build_background_corpus.py   # step B: harvest candidate pool
      select_background.py         # step C: kNN union selection
    embed/
      __init__.py
      specter2_embed.py            # step D: SPECTER2 embedding
    reduce/
      __init__.py
      run_pca_umap.py              # step E: PCA + UMAP
      procrustes.py                # Procrustes alignment for monthly refit
    cluster/
      __init__.py
      run_hdbscan.py               # step F: HDBSCAN clustering
      label_clusters.py            # step F: TF-IDF labeling
      enrichment.py                # step F: IAIFI enrichment scores
    export/
      __init__.py
      export_web_assets.py         # step G: assemble papers.json
      compute_neighbors.py         # step G: precompute top-10 neighbors
    utils/
      __init__.py
      io.py                        # file I/O helpers
      text.py                      # text normalization, stopwords

  web/                             # standalone static site
    index.html
    style.css
    scatter.js                     # regl-scatterplot + interaction logic
    data/
      papers.json                  # the data payload (~3-4MB, ~500KB gzipped)

  notebooks/                       # exploration / debugging (gitignored)
    explore_embeddings.ipynb
```

---

## 2. Configuration Files

### `configs/corpus.yaml`

```yaml
iaifi:
  seed_file: data/external/iaifi_papers.csv
  # columns: arxiv_id, title, authors, published, iaifi_theme (optional)

background:
  start_date: "2018-01-01"   # captures modern ML-physics wave; IAIFI founded 2020
  end_date: null              # null = today
  date_field: submittedDate   # use submittedDate for corpus boundaries; see Step B note
  query_time_utc: "0000"      # arXiv API expects YYYYMMDDHHMM (GMT)
  candidate_cap: 30000        # max papers in candidate pool
  categories:
    - hep-th
    - hep-ph
    - cs.LG
    - cs.AI
    - stat.ML
    - astro-ph.CO
    - astro-ph.HE
    - astro-ph.GA
    - astro-ph.IM
    - gr-qc
    - nucl-th
    - cond-mat.stat-mech
  per_category_cap: 4000      # prevent any single category from dominating
  rate_limit_seconds: 3       # arXiv API rate limit
  max_results_per_page: 2000  # arXiv pagination limit

selection:
  method: knn_union            # kNN union from each IAIFI paper
  k: 20                        # top-k neighbors per IAIFI paper
  space: embedding_768d        # operate in full embedding space, NOT PCA-reduced
  diversity:
    enabled: true
    per_seed_diverse_k: 12     # keep up to 12 non-redundant neighbors per IAIFI seed
    near_duplicate_cosine: 0.985
    anti_hub_weight: 0.2       # score = 0.8*relevance + 0.2*anti_hub
  target_background: 8000      # target after kNN union + dedup
  max_background: 10000        # hard cap
  bridge_ballast: 1000         # uniform random sample for cartographic continuity
  hubness_monitor:
    warn_if_neighbor_share_gt: 0.15
    optional_hard_cap_share: null   # set, e.g., 0.30, only if method hubs hurt UX
```

### `configs/embedding.yaml`

```yaml
model:
  base: allenai/specter2_base
  adapter: allenai/specter2_proximity
  adapter_source: hf
input:
  format: "title + [SEP] + abstract"
  max_tokens: 512
output:
  dimensions: 768
  normalize: l2                # L2-normalize all output vectors
compute:
  batch_size: 32
  device: cpu                  # GPU optional, CPU sufficient for <30K
  cache_path: data/interim/embeddings.npy
quality:
  sanity_check_count: 20       # random IAIFI papers to spot-check
  sanity_check_k: 10           # check top-k neighbors for each
  monitor_hubness: true
  track_truncation_rate: true
  truncation_warn_fraction: 0.15
  metadata_prefix_experiment: false  # OFF by default; enable only for controlled A/B tests
```

### `configs/umap.yaml`

```yaml
pca:
  n_components: 50             # denoising + speed
  whiten: false

umap:
  n_neighbors: 40              # broader than default 15; captures sub-field structure
  min_dist: 0.08               # moderate: avoids over-clumping
  n_components: 2
  metric: cosine
  random_state: 42
  # Note: standard UMAP default. Test densMAP but do not default to it.
  # densmap: false

update:
  incremental:
    enabled: true
    method: transform           # UMAP .transform() for adding few new papers
  monthly_refit:
    enabled: true
    procrustes_anchors: iaifi   # use IAIFI papers as anchor points
    save_previous: true         # keep previous coords for comparison
```

### `configs/clustering.yaml`

```yaml
hdbscan:
  min_cluster_size: 50
  min_samples: 10
  metric: euclidean             # on L2-normalized PCA-50 embeddings
  cluster_selection_method: eom # excess of mass (default)

stability_gating:
  min_size: 50                  # clusters smaller than this get no label
  min_stability: 0.5            # HDBSCAN stability threshold
  explicit_label_persistence_map: true  # do not assume enumerate(sorted(labels))

noise_treatment:
  label: "Interdisciplinary / Uncategorized"
  style: muted                  # reduced opacity, no drop

kmeans_benchmark:               # offline comparison only, NOT production
  enabled: true
  k_range: [10, 15, 20, 25, 30]
  evaluation:
    - silhouette
    - dbcv
    - manual_inspection_count: 3  # inspect 3 random clusters

labels:
  method: tfidf                 # TF-IDF / cTF-IDF on titles + abstracts
  n_candidates: 3               # top n-gram candidates per cluster
  ngram_range: [1, 3]
  human_override: configs/theme_overrides.yml  # MANDATORY — never ship raw labels

enrichment:
  formula: "log(pC / pG)"       # pC = IAIFI fraction in cluster, pG = overall
  smoothing_alpha: 1             # Bayesian smoothing (beta distribution)
  smoothing_beta: 1
  min_cluster_size: 30           # don't display enrichment for tiny clusters
  report_uncertainty: true       # include confidence intervals
```

### `configs/theme_overrides.yml`

```yaml
# Human-curated cluster labels. MANDATORY before deployment.
# Format: cluster_id -> display label
# Auto-generated candidates are in data/processed/cluster_labels.json
#
# Example:
# 0: "Lattice QCD & Gauge Theory"
# 1: "Deep Generative Models"
# 2: "Gravitational Wave Detection"
# ...
#
# This file MUST be reviewed and populated before shipping.
```

---

## 3. Pipeline Steps

### Step A: Collect IAIFI Papers

**Script:** `src/iaifi_paperscape/collect/build_iaifi_seed.py`
**Input:** `data/external/iaifi_papers.csv` (checked in)
**Output:** `data/raw/iaifi_metadata.jsonl`
**Dev time:** 2 hours | **Run time:** ~5 minutes

1. Read the canonical CSV of IAIFI paper arXiv IDs.
2. Strip version suffixes (`2301.12345v2` -> `2301.12345`) for dedup; keep versioned ID as metadata.
3. Fetch full metadata from arXiv API (`id_list` parameter, batches of 50).
4. Enforce 3s rate limit between requests.
5. For each paper, extract and store:

```python
{
    "arxiv_id": "2301.12345",
    "arxiv_id_versioned": "2301.12345v2",
    "title": "...",
    "abstract": "...",
    "authors": ["Smith, A.", "Jones, B."],
    "primary_category": "hep-th",
    "categories": ["hep-th", "cs.LG"],
    "published": "2023-01-15",
    "updated": "2023-03-20",
    "pdf_url": "https://arxiv.org/pdf/2301.12345",
    "is_iaifi": true,
    "iaifi_theme": "AI"  # from CSV if available, else null
}
```

6. Cache to disk. Subsequent runs skip papers already fetched.

### Step B: Collect Background Candidate Pool

**Script:** `src/iaifi_paperscape/collect/build_background_corpus.py`
**Input:** `configs/corpus.yaml`
**Output:** `data/raw/background_candidates.jsonl`
**Dev time:** 2 hours | **Run time:** ~30-45 minutes (one-time; cached thereafter)

1. For each category in `configs/corpus.yaml`, query arXiv API using full GMT timestamps. arXiv date ranges must be `YYYYMMDDHHMM`, not bare years:
   ```python
   from datetime import datetime, timezone

   def to_arxiv_ts(date_str: str | None, fallback_hhmm: str = "0000") -> str:
       if date_str is None:
           dt = datetime.now(timezone.utc)
       else:
           dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
       # arXiv expects YYYYMMDDHHMM in GMT, e.g. 201801010000
       return dt.strftime("%Y%m%d") + fallback_hhmm

   start_ts = to_arxiv_ts(config["background"]["start_date"], config["background"]["query_time_utc"])
   end_ts = to_arxiv_ts(config["background"]["end_date"], config["background"]["query_time_utc"])
   date_field = config["background"].get("date_field", "submittedDate")
   # Example window: submittedDate:[201801010000+TO+202603090000]
   search_query = f"cat:{category}+AND+{date_field}:[{start_ts}+TO+{end_ts}]"
   url = (
       "http://export.arxiv.org/api/query?"
       f"search_query={search_query}&start={start}&max_results=2000"
   )
   ```
2. Paginate until exhausted or `per_category_cap` reached.
3. Enforce 3s delay between requests. Retry with exponential backoff on 5xx.
4. Deduplicate by canonical arXiv ID (cross-listed papers appear in multiple categories).
5. Exclude any paper already in the IAIFI set.
6. Cap total pool at 30K papers.
7. Persist raw Atom XML responses to `data/raw/arxiv_api/` for reproducibility.
8. Extract metadata to JSONL (same schema as Step A, with `is_iaifi: false`).
9. Refresh semantics:
   - Use `submittedDate` as the canonical corpus boundary field (stable interpretation: "when did this paper enter arXiv?").
   - Use `lastUpdatedDate` only for delta-harvest checks in refresh jobs (to re-fetch changed metadata), then re-apply the `submittedDate` boundary before final inclusion.

**Hubness monitoring note:** SPECTER2 proximity can surface "method hubs" (generic surveys / transformer-method papers) across many IAIFI seeds. During Step C, log neighbor-share metrics for candidate papers (`p50`, `p95`, `max`). If `max_share > 0.15`, emit a warning and inspect top hub papers manually. Optional hard cap is available in config but defaults to off.

### Step C: Select Background via kNN Union

**Script:** `src/iaifi_paperscape/collect/select_background.py`
**Input:** Embeddings from Step D (circular dependency -- see note), `configs/corpus.yaml`
**Output:** `data/interim/knn_selected_ids.json`
**Dev time:** 2 hours | **Run time:** ~2 minutes

**Note:** Steps B, C, and D have a dependency: kNN selection requires embeddings, but we embed the full candidate pool before selecting. The pipeline order is: A -> B -> D (embed all 30K) -> C (select 8-10K) -> E (reduce selected subset).

Algorithm:

```python
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors

# Load all embeddings (IAIFI + candidates)
iaifi_embeddings = ...   # shape (547, 768)
candidate_embeddings = ... # shape (30000, 768)
rng = np.random.RandomState(42)

# Build kNN index on candidates
k = 20
nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='auto')
nn.fit(candidate_embeddings)

# For each IAIFI paper, find top-k neighbors in candidate pool
distances, indices = nn.kneighbors(iaifi_embeddings)
similarities = 1.0 - distances

# Count how many IAIFI seeds pull each candidate (hubness signal)
support_count = Counter(indices.ravel().tolist())

# Assert unit vectors before using dot product as cosine in diversity dedup.
candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
assert np.allclose(candidate_norms, 1.0, atol=1e-3), (
    "candidate_embeddings must be L2-normalized before diversity dedup"
)

# Diversity-aware union:
# For each IAIFI seed, keep only non-near-duplicate neighbors to reduce dense-mode redundancy.
per_seed_diverse_k = 12
near_duplicate_cosine = 0.985
selected_indices = set()
for row in indices:
    kept = []
    for cand_idx in row.tolist():
        if len(kept) >= per_seed_diverse_k:
            break
        if not kept:
            kept.append(cand_idx)
            continue
        # Candidate embeddings are unit vectors; dot product == cosine similarity.
        max_pair_sim = (candidate_embeddings[cand_idx] @ candidate_embeddings[kept].T).max()
        if max_pair_sim < near_duplicate_cosine:
            kept.append(cand_idx)
    selected_indices.update(kept)

# Add bridge ballast: ~1K uniform random sample from candidates NOT already selected
remaining = set(range(len(candidate_embeddings))) - selected_indices
bridge = set(rng.choice(
    list(remaining), size=min(1000, len(remaining)), replace=False
))
selected_indices |= bridge

# Enforce max_background cap with anti-hub scoring
max_background = 10000  # from config: selection.max_background
if len(selected_indices) > max_background:
    all_selected = np.array(sorted(selected_indices))
    sim_to_iaifi = candidate_embeddings[all_selected] @ iaifi_embeddings.T  # (N_sel, 547)
    max_sim = sim_to_iaifi.max(axis=1)
    anti_hub = np.array([1.0 / np.sqrt(1.0 + support_count.get(int(i), 0)) for i in all_selected])
    score = 0.8 * max_sim + 0.2 * anti_hub
    top_k_idx = np.argsort(score)[-max_background:]
    selected_indices = set(all_selected[top_k_idx].tolist())

# Hubness diagnostics (required log line in pipeline output)
neighbor_share = np.array([
    support_count.get(int(i), 0) / iaifi_embeddings.shape[0] for i in selected_indices
])
print(
    "Hubness share stats: "
    f"p50={np.percentile(neighbor_share, 50):.3f}, "
    f"p95={np.percentile(neighbor_share, 95):.3f}, "
    f"max={neighbor_share.max():.3f}"
)
if neighbor_share.max() > 0.15:
    print("WARNING: method-hub candidate detected (max neighbor share > 0.15)")

# Result: typically 8-10K unique papers
print(f"Selected {len(selected_indices)} background papers "
      f"({len(selected_indices) - len(bridge)} via kNN, {len(bridge)} bridge ballast)")
```

**Rationale for kNN union over centroid:** Centroid-based selection assumes IAIFI's research is unimodal. It biases toward papers near the "average" IAIFI paper, which is meaningless for an institute spanning lattice QCD and deep learning theory. kNN union preserves multi-modal structure: a lattice QCD IAIFI paper pulls in lattice QCD background, a normalizing flows paper pulls in generative modeling background.

**v2 density guardrail:** v2 keeps the kNN-union core but adds per-seed near-duplicate removal and anti-hub cap scoring before the 10K cap. This reduces "grey snowstorm" redundancy from dense modes while preserving support for sparser IAIFI modes.

### Step D: Embed Papers

**Script:** `src/iaifi_paperscape/embed/specter2_embed.py`
**Input:** `data/raw/iaifi_metadata.jsonl`, `data/raw/background_candidates.jsonl`
**Output:** `data/interim/embeddings.npy`, `data/interim/embedding_index.json`
**Dev time:** 1.5 hours | **Run time:** ~25 minutes (30K papers on CPU)

```python
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# Load model
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.load_adapter("allenai/specter2_proximity", source="hf")
model.set_active_adapters("allenai/specter2_proximity")
model.eval()

# Embed
def embed_batch(papers, batch_size=32):
    embeddings = []
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i+batch_size]
        texts = [f"{p['title']}{tokenizer.sep_token}{p['abstract']}" for p in batch]
        inputs = tokenizer(texts, padding=True, truncation=True,
                          max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Use [CLS] token embedding
        batch_emb = outputs.last_hidden_state[:, 0, :]
        # L2 normalize
        batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
        embeddings.append(batch_emb.numpy())
    return np.vstack(embeddings)
```

**Quality checks (mandatory before proceeding):**

For 20 randomly selected IAIFI papers, print their top-10 nearest neighbors by cosine similarity. Manual inspection: do the neighbors look topically coherent? If a lattice QCD paper's nearest neighbors include random cosmology papers, something is wrong.

```python
# Quality check
from sklearn.metrics.pairwise import cosine_similarity
for idx in random_iaifi_indices[:20]:
    sims = cosine_similarity(embeddings[idx:idx+1], embeddings)[0]
    top10 = np.argsort(sims)[-11:-1][::-1]  # exclude self
    print(f"\n{papers[idx]['title']}")
    for j in top10:
        print(f"  {sims[j]:.3f}  {papers[j]['title']}")
```

**Monitoring guardrails (required):**

1. **Hubness audit:** Reuse Step C logs (`p50`, `p95`, `max` neighbor share). If `max > 0.15`, inspect the top 10 hub papers and confirm they are not overwhelming user-visible neighborhoods.
2. **Truncation audit (512-token limit):**
   ```python
   import numpy as np
   token_lengths = []
   for p in papers:
       text = f"{p['title']}{tokenizer.sep_token}{p['abstract']}"
       n_tokens = len(tokenizer(text, truncation=False)["input_ids"])
       token_lengths.append(n_tokens)
   truncation_rate = float(np.mean(np.array(token_lengths) > 512))
   print(f"Truncation rate: {truncation_rate:.3f}")
   if truncation_rate > 0.15:
       print("WARNING: high truncation rate; run neighbor coherence spot-check before shipping")
   ```
3. **Controlled metadata-prefix experiment (optional, OFF by default):** If coherence checks fail and truncation is high, run a side-by-side experiment with a minimal prefix such as `"[cat=hep-th]"` prepended to input text. Accept only if it improves neighbor coherence without obvious "clustering-by-label" artifacts.

### Step E: Dimensionality Reduction

**Script:** `src/iaifi_paperscape/reduce/run_pca_umap.py`
**Input:** `data/interim/embeddings.npy` (selected subset only), `configs/umap.yaml`
**Output:** `data/processed/coords_2d.npy`, `data/interim/pca_50.npy`, `data/models/pca_model.pkl`, `data/models/umap_model.pkl`
**Dev time:** 1.5 hours | **Run time:** ~2 minutes

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import umap
import joblib

# 1. Select only IAIFI + kNN-selected background embeddings
selected_embeddings = embeddings[selected_mask]  # shape ~(9000, 768)

# 2. PCA to 50 dims (denoising + speed)
pca = PCA(n_components=50, random_state=42)
pca_embeddings = pca.fit_transform(selected_embeddings)
# L2-normalize PCA vectors so Euclidean clustering is geometry-consistent with cosine neighborhoods.
pca_embeddings = normalize(pca_embeddings, norm="l2", axis=1)
np.save("data/interim/pca_50.npy", pca_embeddings)
joblib.dump(pca, "data/models/pca_model.pkl")

# 3. UMAP to 2D
reducer = umap.UMAP(
    n_neighbors=40,
    min_dist=0.08,
    n_components=2,
    metric="cosine",
    random_state=42,
)
coords_2d = reducer.fit_transform(pca_embeddings)
joblib.dump(reducer, "data/models/umap_model.pkl")

np.save("data/processed/coords_2d.npy", coords_2d)
```

**Parameter table:**

| Parameter | Value | Rationale | Tuning guidance |
|---|---|---|---|
| PCA n_components | 50 | Standard denoising pre-step; preserves >95% variance for SPECTER2 | Increase to 100 if clusters look fragmented; decrease to 30 if too slow |
| n_neighbors | 40 | Broader than default 15; preserves sub-field structure without over-smoothing | Lower (20-30) if clusters are too diffuse; higher (50-60) for more global structure |
| min_dist | 0.08 | Slightly tighter than default 0.1; shows cluster structure without over-clumping | Increase to 0.15 if points overlap too much; decrease to 0.03 for tighter clusters |
| metric | cosine | PCA-50 vectors are L2-normalized before UMAP/clustering; this keeps local geometry consistent with cosine neighborhoods | Do not change |
| random_state | 42 | Reproducibility | Do not change |

**densMAP note:** Standard UMAP is the default. densMAP (`densmap=True`) preserves local density information more faithfully but can produce less visually appealing layouts. Test it; if it improves the "is this cluster actually dense or just UMAP artifact?" question, use it. Do not default to it without testing.

**Geometry consistency note (v2):** Clustering remains Euclidean for HDBSCAN, but it now operates on L2-normalized PCA vectors. On unit vectors, Euclidean distance is monotone in cosine distance, so clustering and neighbor semantics are aligned.

### Step E.1: Procrustes Alignment (Monthly Refit Only)

**Script:** `src/iaifi_paperscape/reduce/procrustes.py`
**When:** Only during monthly full refit, not incremental updates
**Dev time:** 1.5 hours

```python
import numpy as np
from scipy.linalg import orthogonal_procrustes

def align_to_previous(new_coords, old_coords, anchor_mask):
    """
    Align new UMAP layout to previous layout using IAIFI papers as anchors.

    anchor_mask: boolean array, True for IAIFI papers present in both layouts
    """
    # Extract anchor points
    new_anchors = new_coords[anchor_mask]
    old_anchors = old_coords[anchor_mask]

    # Center both sets
    new_mu = new_anchors.mean(axis=0)
    old_mu = old_anchors.mean(axis=0)
    new_centered = new_anchors - new_mu
    old_centered = old_anchors - old_mu

    # Scale to unit Frobenius norm
    new_norm = np.linalg.norm(new_centered)
    old_norm = np.linalg.norm(old_centered)
    new_unit = new_centered / new_norm
    old_unit = old_centered / old_norm

    # Solve orthogonal Procrustes: new_unit @ R ~= old_unit
    R, _ = orthogonal_procrustes(new_unit, old_unit)
    scale = old_norm / new_norm

    # Apply transform to all points (not just anchors)
    aligned_coords = ((new_coords - new_mu) @ R) * scale + old_mu

    # Drift diagnostics on anchors (post-alignment)
    anchor_disp = np.linalg.norm(aligned_coords[anchor_mask] - old_coords[anchor_mask], axis=1)
    drift = {
        "median": float(np.median(anchor_disp)),
        "p95": float(np.percentile(anchor_disp, 95)),
        "max": float(np.max(anchor_disp)),
    }

    return aligned_coords, drift
```

**Drift policy (required for monthly refits):**

1. Compute anchor displacement diagnostics after Procrustes: median and 95th percentile.
2. Define status bands:
   - `LOW_DRIFT`: `median <= 0.03` and `p95 <= 0.08` -> publish normally.
   - `MEDIUM_DRIFT`: not `LOW_DRIFT` and not `HIGH_DRIFT` -> publish, but flag for review in refresh logs.
   - `HIGH_DRIFT`: `median > 0.05` or `p95 > 0.12` -> trigger mitigation.
3. Mitigation for `HIGH_DRIFT`:
   - Freeze previous month's background selection (`knn_selected_ids.json`) and rerun refit once.
   - If still high-drift, publish with a visible "layout changed more than usual" notice and record diagnostics in `meta`.
   - Do not silently overwrite prior layout without diagnostics.

**Two update regimes (complementary, not alternative):**

| Regime | When | Method | Layout impact |
|---|---|---|---|
| Incremental | Few new IAIFI papers, same background | `umap_model.transform(normalize(pca.transform(new_embeddings), norm="l2", axis=1))` | Zero drift for existing points |
| Monthly refit | Background corpus refresh, new kNN selection | Full PCA + UMAP refit, then Procrustes alignment + drift diagnostics | Minimal drift when low-drift; freeze/warn policy when high-drift |

### Step F: Clustering and Labeling

**Script:** `src/iaifi_paperscape/cluster/run_hdbscan.py`, `label_clusters.py`, `enrichment.py`
**Input:** `data/interim/pca_50.npy` (L2-normalized PCA-50), paper metadata
**Output:** `data/processed/clusters.json`, `cluster_labels.json`, `enrichment.json`
**Dev time:** 2.5 hours | **Run time:** ~1 minute

#### F.1: HDBSCAN Clustering

```python
import numpy as np
import hdbscan

def build_label_persistence_map(clusterer):
    """
    Explicit map from output cluster label -> persistence.
    Do NOT rely on enumerate(sorted(unique_labels)).
    """
    clusterer.generate_prediction_data()
    selected_tree_ids = list(clusterer.condensed_tree_._select_clusters())
    persistence = np.asarray(clusterer.cluster_persistence_, dtype=float)

    if len(selected_tree_ids) != len(persistence):
        raise RuntimeError(
            "HDBSCAN selected-cluster count != persistence count; abort unsafe stability gating."
        )

    # Internal ordering from HDBSCAN condensed tree.
    persistence_by_tree_id = {
        int(tree_id): float(p)
        for tree_id, p in zip(selected_tree_ids, persistence)
    }

    # Internal map: tree cluster id -> emitted integer label.
    cluster_map = clusterer._prediction_data.cluster_map
    label_to_persistence = {}
    for tree_id, label_id in cluster_map.items():
        if label_id == -1:
            continue
        if int(tree_id) in persistence_by_tree_id:
            label_to_persistence[int(label_id)] = persistence_by_tree_id[int(tree_id)]

    return label_to_persistence

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,
    min_samples=10,
    metric='euclidean',        # on L2-normalized PCA-50 embeddings
    cluster_selection_method='eom',
)
labels = clusterer.fit_predict(pca_embeddings)
probabilities = clusterer.probabilities_
label_to_persistence = build_label_persistence_map(clusterer)

# Stability gating: suppress labels for weak clusters
for cluster_id in sorted(set(labels) - {-1}):
    mask = labels == cluster_id
    persistence = label_to_persistence.get(int(cluster_id), 0.0)
    if mask.sum() < 50 or persistence < 0.5:
        labels[mask] = -1  # demote to noise
```

**Noise point treatment:** Papers assigned to cluster -1 are labeled "Interdisciplinary / Uncategorized" in the UI. They are displayed with muted styling (reduced opacity, grey tone) but NOT dropped. These are often the most interesting papers -- genuinely interdisciplinary work that doesn't fit neatly.

#### F.2: KMeans Benchmark (Offline Only)

```python
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# Sweep k values for comparison
results = []
for k in [10, 15, 20, 25, 30]:
    km = MiniBatchKMeans(n_clusters=k, random_state=42)
    km_labels = km.fit_predict(pca_embeddings)
    sil = silhouette_score(pca_embeddings, km_labels, sample_size=5000)
    results.append({"k": k, "silhouette": sil})

# Log results but DO NOT use for production clustering
```

**Evaluation protocol:**
1. Silhouette score (higher = better separation)
2. DBCV (density-based cluster validation, native to HDBSCAN)
3. Manual inspection: print 10 random paper titles from 3 random clusters. Do they cohere?

#### F.3: Cluster Labeling

```python
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# For each cluster, compute TF-IDF on titles + abstracts
# Persisted from Step F.1 output (cluster_id -> persistence)
with open("data/processed/clusters.json", "r", encoding="utf-8") as f:
    clusters_payload = json.load(f)
label_to_persistence = {
    int(c["id"]): float(c["stability"])
    for c in clusters_payload["clusters"]
}
unique_labels = sorted(set(labels) - {-1})
for cluster_id in unique_labels:
    cluster_texts = [
        f"{p['title']} {p['abstract']}"
        for p, l in zip(papers, labels) if l == cluster_id
    ]
    rest_texts = [
        f"{p['title']} {p['abstract']}"
        for p, l in zip(papers, labels) if l != cluster_id
    ]

    # cTF-IDF: class-based TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
        stop_words=SCIENCE_STOPWORDS,  # "show", "study", "result", "model", "paper", "propose", etc.
    )
    # ... compute top 3 n-gram candidates per cluster

    cluster_labels[cluster_id] = {
        "auto_label": top_ngrams[0],
        "label_candidates": top_ngrams[:3],
        "size": len(cluster_texts),
        "stability": float(label_to_persistence.get(int(cluster_id), 0.0)),
    }
```

**Human curation is MANDATORY.** The pipeline outputs `data/processed/cluster_labels.json` with auto-generated candidates. A human must review these and populate `configs/theme_overrides.yml` before deployment. Auto-labels are development aids, not user-facing text.

#### F.4: Enrichment Scores

```python
import numpy as np

def compute_enrichment(labels, is_iaifi, alpha=1, beta=1):
    """
    Per-cluster enrichment = log(pC / pG)
    where pC = IAIFI fraction in cluster, pG = overall IAIFI fraction.
    Bayesian smoothing with alpha, beta.
    """
    n_total = len(is_iaifi)
    n_iaifi_total = is_iaifi.sum()
    pG = (n_iaifi_total + alpha) / (n_total + alpha + beta)

    enrichment = {}
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        mask = labels == cluster_id
        n_cluster = mask.sum()
        n_iaifi_cluster = (is_iaifi & mask).sum()

        if n_cluster < 30:  # minimum cluster size for enrichment display
            enrichment[cluster_id] = None
            continue

        pC = (n_iaifi_cluster + alpha) / (n_cluster + alpha + beta)
        log_enrichment = np.log2(pC / pG)

        # Uncertainty: approximate 95% CI using beta distribution
        from scipy.stats import beta as beta_dist
        lo = beta_dist.ppf(0.025, n_iaifi_cluster + alpha, n_cluster - n_iaifi_cluster + beta)
        hi = beta_dist.ppf(0.975, n_iaifi_cluster + alpha, n_cluster - n_iaifi_cluster + beta)

        enrichment[cluster_id] = {
            "log2_enrichment": float(log_enrichment),
            "iaifi_fraction": float(pC),
            "cluster_size": int(n_cluster),
            "iaifi_in_cluster": int(n_iaifi_cluster),
            "ci_95_lo": float(np.log2(lo / pG)) if lo > 0 else None,
            "ci_95_hi": float(np.log2(hi / pG)),
        }
    return enrichment
```

**Enrichment is exposed as an optional color-by mode in the UI**, not the default view. When a cluster's 95% confidence interval crosses 0 (i.e., `ci_95_lo` is negative and `ci_95_hi` is positive, or either is null), the enrichment color should be suppressed and the cluster displayed in grey. Only clusters with a confidently nonzero enrichment score receive the diverging color scale.

### Step G: Export Web Assets

**Script:** `src/iaifi_paperscape/export/export_web_assets.py`, `compute_neighbors.py`
**Input:** All processed data
**Output:** `web/data/papers.json`
**Dev time:** 1.5 hours | **Run time:** ~30 seconds

#### G.1: Precompute Neighbor Lists

```python
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors

# Build kNN on L2-normalized 768-d embeddings (selected papers only)
nn = NearestNeighbors(n_neighbors=11, metric='cosine')  # 11 to exclude self
nn.fit(selected_embeddings)
distances, indices = nn.kneighbors(selected_embeddings)

# Store top 10 per paper (excluding self)
neighbors = {}
for i, paper_id in enumerate(paper_ids):
    neighbors[paper_id] = [
        {"id": paper_ids[indices[i][j]], "similarity": float(1 - distances[i][j])}
        for j in range(1, 11)  # skip self at index 0
    ]

# Hubness diagnostics on final neighbor graph (method-hub monitoring)
incoming = Counter()
for nbrs in neighbors.values():
    for edge in nbrs:
        incoming[edge["id"]] += 1

incoming_share = np.array([incoming[pid] / len(paper_ids) for pid in paper_ids])
print(
    "Final graph hubness: "
    f"p50={np.percentile(incoming_share, 50):.3f}, "
    f"p95={np.percentile(incoming_share, 95):.3f}, "
    f"max={incoming_share.max():.3f}"
)
```

**Top 10 neighbors per paper** (not 20). The UX shows 5 in the side panel with a "show more" button to reveal all 10. This keeps JSON size manageable (~1.5MB for neighbor data alone).

If hubness diagnostics show degraded UX (same generic methods papers dominating many panels), apply optional post-filtering: per-paper incoming cap plus diversity rerank (MMR-style) within each neighbor list. Keep this off by default and gate by manual review.

#### G.2: JSON Schema for `papers.json`

```json
{
  "meta": {
    "version": "2026-03-09_specter2prox_n9547",
    "generated": "2026-03-09T14:30:00Z",
    "embedding_model": "specter2_proximity",
    "n_iaifi": 547,
    "n_background": 8453,
    "n_total": 9000,
    "umap_params": {
      "n_neighbors": 40,
      "min_dist": 0.08,
      "metric": "cosine",
      "pca_dims": 50
    },
    "disclaimer": "Local neighborhoods are meaningful; global geometry is approximate."
  },

  "papers": [
    {
      "id": "2301.12345",
      "x": 1.234,
      "y": -0.567,
      "t": "Machine Learning for Lattice QCD",
      "a": "Smith, A.; Jones, B.; et al.",
      "yr": 2023,
      "cat": ["hep-th", "cs.LG"],
      "iaifi": true,
      "theme": "AI",
      "cluster": 3,
      "cit": 42,
      "nn": ["2301.11111", "2302.22222", "2303.33333", "2304.44444",
             "2305.55555", "2306.66666", "2307.77777", "2308.88888",
             "2309.99999", "2310.00000"]
    }
  ],

  "clusters": [
    {
      "id": 0,
      "label": "Lattice QCD & Gauge Theory",
      "auto_label": "lattice gauge field",
      "cx": 2.3,
      "cy": 1.1,
      "size": 487,
      "stability": 0.82,
      "enrichment": {
        "log2": 1.45,
        "ci_lo": 0.92,
        "ci_hi": 1.98
      }
    }
  ],

  "abstracts": {
    "2301.12345": "We present a novel approach to..."
  }
}
```

**Key design decisions:**
- Short keys (`t`, `a`, `yr`, `cat`, `nn`) to reduce JSON size
- `abstracts` as a separate top-level map (loaded lazily on click, or split into a second file if >2MB)
- `nn` stores only arXiv IDs (not similarity scores -- saves ~40% space; look up by index if needed)
- `cit` is null/omitted if citation data unavailable (graceful degradation)
- Cluster enrichment nested in cluster objects

**Expected file sizes:**
| Component | Uncompressed | Gzipped |
|---|---|---|
| papers array (~9K entries) | ~2.5 MB | ~400 KB |
| abstracts map | ~3 MB | ~600 KB |
| clusters + meta | ~5 KB | ~2 KB |
| **Total** | **~5.5 MB** | **~1 MB** |

**If abstracts push the file too large:** Split into `papers.json` (coords + metadata, ~400KB gzipped) and `abstracts.json` (loaded on demand when side panel opens). This keeps initial load fast.

### Step H: Build Frontend

**Location:** `web/`
**Dev time:** 5 hours | **Run time:** N/A (static files)

#### H.1: `web/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IAIFI Research Landscape</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header>
    <h1>IAIFI Research Landscape</h1>
    <div id="controls">
      <input type="text" id="search" placeholder="Search by title or author...">
      <div id="theme-filters">
        <button data-theme="all" class="active">All</button>
        <button data-theme="AI">AI</button>
        <button data-theme="Physics">Physics</button>
        <button data-theme="Both">Both</button>
      </div>
      <div id="color-by">
        <label>Color by:</label>
        <select id="color-mode">
          <option value="theme">IAIFI Theme</option>
          <option value="category">arXiv Category</option>
          <option value="enrichment">Cluster Enrichment</option>
        </select>
      </div>
      <div id="year-slider">
        <!-- min/max derived from meta in papers.json at load time -->
        <label>Year: <span id="year-range"></span></label>
        <input type="range" id="year-min">
        <input type="range" id="year-max">
      </div>
    </div>
    <div id="data-version"></div>
  </header>

  <main>
    <div id="scatter-container"></div>
    <div id="tooltip" class="hidden"></div>
    <div id="side-panel" class="hidden">
      <button id="close-panel">&times;</button>
      <h2 id="panel-title"></h2>
      <p id="panel-authors"></p>
      <p id="panel-meta"></p>
      <div id="panel-abstract"></div>
      <h3>Related Papers</h3>
      <ul id="panel-neighbors"></ul>
      <a id="panel-arxiv" href="#" target="_blank">View on arXiv &rarr;</a>
    </div>
  </main>

  <footer>
    <p>Local neighborhoods are meaningful; global geometry is approximate.
       <a href="https://iaifi.org">IAIFI</a></p>
  </footer>

  <script src="https://unpkg.com/regl-scatterplot@1.9.2/dist/regl-scatterplot.min.js"></script>
  <script src="scatter.js"></script>
</body>
</html>
```

#### H.2: Visual Design Spec

| Element | Style |
|---|---|
| Background papers | `#e0e0e0`, 30% opacity, 3px radius |
| IAIFI AI theme | `#2196F3` (blue), 100% opacity, 5px radius |
| IAIFI Physics theme | `#FF9800` (amber), 100% opacity, 5px radius |
| IAIFI Both theme | `#9C27B0` (purple), 100% opacity, 5px radius |
| Hover state | Brighten point + enlarge to 8px + show tooltip |
| Selected point | Bright ring + side panel opens |
| Cluster labels | Semi-transparent text at centroid, visible at medium zoom, hidden at extremes |
| Noise / Uncategorized | Muted grey, 40% opacity, 3px radius |

**Point sizing (optional citation scaling):**
- Default OFF. Toggle in UI.
- If enabled: `radius = clamp(3 + log2(citations + 1), 3, 12)` for IAIFI papers.
- Legend required when enabled.
- Background papers always 3px regardless.

#### H.3: Interaction Spec

| Action | Result |
|---|---|
| **Hover** | Tooltip appears near cursor: title, year, primary category |
| **Click** | Side panel slides in from right: title, authors, year, categories, abstract, top-5 nearest neighbors (+ "show 5 more" button), arXiv link |
| **Search** | Text input filters papers by title/author substring match. Non-matching points dim to 10% opacity. |
| **Theme filter** | Buttons toggle IAIFI theme visibility. Background always visible. |
| **Color-by** | Dropdown switches between: Theme (default), arXiv Category, Cluster Enrichment |
| **Year slider** | Dual-handle range slider. Min/max bounds derived from `meta` in papers.json at load time (not hardcoded). Papers outside range dim to 10% opacity. |
| **Zoom/pan** | Standard scroll-to-zoom, drag-to-pan via regl-scatterplot |

#### H.4: Astro Blog Integration

The standalone site in `web/` is the primary deployment target. Integration with the Astro blog is via iframe:

```html
<!-- In the Astro blog page -->
<iframe
  src="https://abelclaw.github.io/iaifi-paperscape/"
  width="100%"
  height="800px"
  style="border: none; border-radius: 8px;"
  loading="lazy"
  title="IAIFI Research Landscape"
></iframe>
```

An optional thin Astro wrapper component can add context text above/below the iframe. The iframe approach keeps the data pipeline and blog decoupled -- they have different update cadences.

---

## 4. Multiscale UI (Progressive Enhancement)

**This does NOT block launch.** Ship v1.5a first, add v1.5b as a fast follow.

### v1.5a (Launch)

All points rendered at all zoom levels. Background at 30% opacity, IAIFI at full opacity. This is sufficient for initial deployment with ~9K total points.

Important caveat: opacity reduces visual dominance but does not solve sampling-density occlusion. Track simple occlusion diagnostics (e.g., screen-space fill ratio at default overview, plus median nearest-neighbor screen distance). If overview readability degrades, prioritize v1.5b.

### v1.5b (Fast Follow)

Background points replaced by precomputed hexbin density heatmap at overview zoom. Crossfade to individual points when zoomed in past a threshold. This is the backstop when opacity-only rendering is not enough.

**Implementation:**
1. Precompute hexbin density in Python (not live KDE in the browser):
   ```python
   from matplotlib.colors import Normalize
   import numpy as np

   # Hexbin the background points
   # Export as array of {cx, cy, count, opacity} for the frontend
   ```
2. Frontend renders hexbin layer on a secondary canvas beneath the scatter layer.
3. Opacity crossfade: hexbin fades out as zoom increases, points fade in.
4. **IAIFI points are ALWAYS visible at all zoom levels** -- they never collapse into hexbins.

**Dev time for v1.5b:** 4-6 hours additional frontend work.

---

## 5. Optional Citation Integration

**Status:** Non-blocking, feature-flagged. Everything works without it.

**Source:** Semantic Scholar API (`api.semanticscholar.org/graph/v1/paper/ArXiv:{id}`)
- Rate limit: 100 requests / 5 minutes (unauthenticated)
- For 9K papers: ~7.5 hours. Use authenticated key if available (1000 req/min).
- Cache responses to disk.

**If available:**
- Point sizing by citation count (default-off toggle, clamped range 3-12px, with legend)
- Side panel shows citation count
- Future v2: citation ego network overlay (limit edges to prevent visual overload)

**If unavailable:**
- Point sizing uniform (background 3px, IAIFI 5px)
- Semantic neighbors only (already precomputed from embeddings)
- Everything works fine

---

## 6. Deployment

### GitHub Pages (Primary)

1. Build pipeline produces `web/data/papers.json`
2. Push `web/` directory to `gh-pages` branch
3. URL: `abelclaw.github.io/iaifi-paperscape/`

### GitHub Action (Monthly Refresh)

```yaml
name: Monthly Paperscape Refresh
on:
  schedule:
    - cron: '0 6 1 * *'    # First of each month, 6 AM UTC
  workflow_dispatch:         # Manual trigger

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: bash run_pipeline.sh
      - run: |
          echo "DATA_VERSION=$(date +%Y-%m-%d)_$(echo ${{ hashFiles('web/data/papers.json') }} | head -c 8)_n$(jq '.meta.n_total' web/data/papers.json)" >> $GITHUB_ENV
      - uses: peaceiris/actions-gh-pages@v4
        with:
          publish_dir: ./web
```

### Data Version in UI

Displayed in the header: `"Data: 2026-03-09 | specter2_proximity | 9,000 papers"`

Composed from `meta.version` in `papers.json`, which encodes: date + model + dataset size.

---

## 7. Pipeline Orchestration

### `Makefile`

```makefile
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
web/data/papers.json: data/processed/clusters.json data/processed/coords_2d.npy
	$(PYTHON) -m $(PKG).export.compute_neighbors
	$(PYTHON) -m $(PKG).export.export_web_assets

collect: data/raw/iaifi_metadata.jsonl data/raw/background_candidates.jsonl
embed: data/interim/embeddings.npy
select: data/interim/knn_selected_ids.json
reduce: data/processed/coords_2d.npy
cluster: data/processed/clusters.json
export site: web/data/papers.json

clean:
	rm -rf data/interim/ data/processed/ data/models/ web/data/papers.json
```

### `run_pipeline.sh` (Alternative)

```bash
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
```

---

## 8. Time Estimates

| Step | Script(s) | Dev Time | Run Time |
|---|---|---|---|
| A: Collect IAIFI papers | build_iaifi_seed.py | 2 hours | 5 min |
| B: Collect background | build_background_corpus.py | 2 hours | 30-45 min (one-time) |
| C: kNN selection | select_background.py | 2 hours | 2 min |
| D: Embed | specter2_embed.py | 1.5 hours | 25 min (30K on CPU) |
| E: PCA + UMAP | run_pca_umap.py | 1.5 hours | 2 min |
| E.1: Procrustes | procrustes.py | 1.5 hours | seconds |
| F: Cluster + label + enrich | run_hdbscan.py, label_clusters.py, enrichment.py | 2.5 hours | 1 min |
| G: Export + neighbors | export_web_assets.py, compute_neighbors.py | 1.5 hours | 30 sec |
| H: Frontend | web/* | 5 hours | — |
| Polish + deploy | GitHub Pages + Action | 2 hours | — |
| **Total v1.5a (launch)** | | **~22 hours** | **~35 min** |
| v1.5b: Multiscale UI | (fast follow) | +5 hours | — |
| Optional: Citations | enrich_citations.py | +2 hours | 1-8 hours (API) |

**Pipeline re-run time (cached, monthly refresh):** ~10 minutes (skip collection, re-embed only new papers, full UMAP refit + Procrustes).

---

## 9. v2 Enhancements (Deferred)

These are explicitly out of scope for the v2 launch scope. Listed here for future reference.

| Enhancement | Description | Effort |
|---|---|---|
| Lasso select + export | Select region, export paper list as CSV/BibTeX | 4 hours |
| Semantic search | Text box -> SPECTER2 embedding -> highlight nearest papers | 6 hours |
| Citation ego network | On click, draw edges to cited/citing papers | 8 hours |
| LLM cluster narratives | 2-sentence cluster descriptions from representative abstracts | 4 hours |
| 3D mode | UMAP to 3D + three.js | 8 hours |
| Comparison mode | Overlay another institute's papers as different color layer | 6 hours |
| Live KDE density | Replace precomputed hexbin with live kernel density rendering | 10 hours |
| Temporal animation | Slider to animate papers appearing by publication year | 6 hours |

---

## 10. References

- arXiv API user manual: https://info.arxiv.org/help/api/user-manual.html
- arXiv API terms/rate limits: https://info.arxiv.org/help/api/tou.html
- arXiv category taxonomy: https://arxiv.org/category_taxonomy
- SPECTER2 model card: https://huggingface.co/allenai/specter2
- SPECTER2 proximity adapter: https://huggingface.co/allenai/specter2_proximity
- UMAP documentation: https://umap-learn.readthedocs.io/en/latest/
- UMAP .transform(): https://umap-learn.readthedocs.io/en/latest/transform.html
- HDBSCAN documentation: https://hdbscan.readthedocs.io/en/latest/
- HDBSCAN API reference: https://hdbscan.readthedocs.io/en/latest/api.html
- regl-scatterplot: https://github.com/flekschas/regl-scatterplot
- Procrustes analysis (scipy): https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html

---

## 11. Changes from v1.5

1. Bumped plan version header and metadata from v1.5 to v2.
2. Fixed arXiv date-range query format in Step B to `YYYYMMDDHHMM` (GMT), with explicit example `submittedDate:[201801010000+TO+202603090000]`.
3. Clarified refresh semantics in Step B: `submittedDate` for corpus boundaries, `lastUpdatedDate` only for delta-harvest checks before reapplying boundary rules.
4. Added diversity-aware kNN-union mitigation in Step C (per-seed near-duplicate removal + anti-hub scoring) to reduce dense-mode redundancy and "grey snowstorm" behavior.
5. Added hubness diagnostics and monitoring thresholds in Step B, Step C, and Step G; documented optional caps/diversity reranking only if UX degrades.
6. Fixed geometry mismatch by L2-normalizing PCA-50 vectors before UMAP/clustering (Step E), with explicit rationale linking Euclidean-on-unit-vectors to cosine semantics.
7. Replaced fragile HDBSCAN stability gating pseudocode with an explicit label->persistence map built from HDBSCAN internal cluster ordering (Step F.1), and propagated stable lookup into labeling (Step F.3).
8. Expanded Procrustes section with explicit transform logic, anchor drift diagnostics (median/p95/max), and a high-drift policy (freeze background selection rerun, then warn if still high drift).
9. Added embedding-quality monitoring for 512-token truncation and a controlled metadata-prefix experiment policy (off by default, with anti-label-leakage caution).
10. Updated multiscale UI section to explicitly note opacity-vs-occlusion limits in v1.5a and position v1.5b hexbin as the required backstop when overview readability drops.
