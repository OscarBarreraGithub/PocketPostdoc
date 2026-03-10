"""Step D: embed IAIFI and background papers with SPECTER2 proximity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from adapters import AutoAdapterModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer

from iaifi_paperscape.utils.io import load_jsonl, load_yaml_config, save_json, save_numpy


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "embedding.yaml"
IAIFI_METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "iaifi_metadata.jsonl"
BACKGROUND_PATH = PROJECT_ROOT / "data" / "raw" / "background_candidates.jsonl"
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "interim" / "embeddings.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "interim" / "embedding_index.json"


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _paper_text(paper: dict[str, Any], sep_token: str) -> str:
    title = str(paper.get("title", "") or "").strip()
    abstract = str(paper.get("abstract", "") or "").strip()
    return f"{title}{sep_token}{abstract}"


def load_specter2_model(
    config: dict[str, Any],
) -> tuple[Any, AutoAdapterModel]:
    """Load SPECTER2 base model and the proximity adapter."""
    model_cfg = config.get("model", {})
    base_model = str(model_cfg.get("base", "allenai/specter2_base"))
    adapter_name = str(model_cfg.get("adapter", "allenai/specter2_proximity"))
    adapter_source = str(model_cfg.get("adapter_source", "hf"))

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoAdapterModel.from_pretrained(base_model)
    loaded_name = model.load_adapter(adapter_name, source=adapter_source)
    model.set_active_adapters(loaded_name)
    model.eval()

    return tokenizer, model


def embed_batch(
    papers: list[dict[str, Any]],
    tokenizer: Any,
    model: AutoAdapterModel,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed papers with title + [SEP] + abstract using [CLS] + L2 normalization."""
    if not papers:
        hidden_size = int(getattr(model.config, "hidden_size", 768))
        return np.zeros((0, hidden_size), dtype=np.float32)

    device = next(model.parameters()).device
    all_embeddings: list[np.ndarray] = []

    for start in tqdm(
        range(0, len(papers), batch_size),
        desc="Embedding papers",
        unit="batch",
    ):
        batch = papers[start : start + batch_size]
        texts = [_paper_text(p, tokenizer.sep_token) for p in batch]
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        batch_emb = outputs.last_hidden_state[:, 0, :]
        batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
        all_embeddings.append(batch_emb.cpu().numpy().astype(np.float32, copy=False))

    return np.vstack(all_embeddings).astype(np.float32, copy=False)


def _compute_truncation_rate(
    papers: list[dict[str, Any]],
    tokenizer: Any,
    max_tokens: int,
) -> float:
    if not papers:
        return 0.0

    token_lengths: list[int] = []
    for paper in tqdm(papers, desc="Auditing truncation", unit="paper"):
        text = _paper_text(paper, tokenizer.sep_token)
        n_tokens = len(tokenizer(text, truncation=False)["input_ids"])
        token_lengths.append(n_tokens)

    return float(np.mean(np.array(token_lengths, dtype=np.int32) > max_tokens))


def _run_quality_checks(
    papers: list[dict[str, Any]],
    embeddings: np.ndarray,
    n_iaifi: int,
    sanity_count: int,
    sanity_k: int,
) -> None:
    if n_iaifi <= 0:
        print("Skipping IAIFI neighbor quality check: no IAIFI papers found.")
        return
    if embeddings.shape[0] <= 1:
        print("Skipping IAIFI neighbor quality check: not enough embeddings.")
        return

    rng = np.random.RandomState(42)
    sample_count = min(sanity_count, n_iaifi)
    sample_indices = rng.choice(np.arange(n_iaifi), size=sample_count, replace=False)

    print(
        f"Quality check: {sample_count} random IAIFI seeds, top-{sanity_k} neighbors each."
    )
    for idx in sample_indices:
        sims = cosine_similarity(embeddings[idx : idx + 1], embeddings)[0]
        ranked = np.argsort(sims)[::-1]
        neighbor_idxs = [j for j in ranked if j != idx][:sanity_k]

        title = str(papers[idx].get("title", ""))
        aid = str(papers[idx].get("arxiv_id", ""))
        print(f"\nSeed [{aid}] {title}")
        for nbr_idx in neighbor_idxs:
            nbr_title = str(papers[nbr_idx].get("title", ""))
            nbr_aid = str(papers[nbr_idx].get("arxiv_id", ""))
            print(f"  {sims[nbr_idx]:.3f}  [{nbr_aid}] {nbr_title}")


def main() -> None:
    config = load_yaml_config(CONFIG_PATH)

    iaifi_papers = load_jsonl(IAIFI_METADATA_PATH)
    background_papers = load_jsonl(BACKGROUND_PATH)
    papers = iaifi_papers + background_papers
    n_iaifi = len(iaifi_papers)

    if not papers:
        raise RuntimeError("No papers found to embed.")

    compute_cfg = config.get("compute", {})
    output_cfg = config.get("output", {})
    input_cfg = config.get("input", {})
    quality_cfg = config.get("quality", {})

    batch_size = int(compute_cfg.get("batch_size", 32))
    max_tokens = int(input_cfg.get("max_tokens", 512))
    embedding_dims = int(output_cfg.get("dimensions", 768))
    embeddings_path = _resolve_path(compute_cfg.get("cache_path", DEFAULT_EMBEDDINGS_PATH))

    tokenizer, model = load_specter2_model(config)

    device_name = str(compute_cfg.get("device", "cpu")).strip().lower()
    if device_name == "cuda" and not torch.cuda.is_available():
        print("WARNING: compute.device=cuda but CUDA is unavailable; falling back to CPU.")
        device_name = "cpu"
    model.to(torch.device(device_name))

    print(
        f"Embedding {len(papers)} papers ({n_iaifi} IAIFI + {len(background_papers)} background) "
        f"with batch_size={batch_size} on device={device_name}."
    )
    embeddings = embed_batch(
        papers=papers,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
    )

    if embeddings.ndim != 2 or embeddings.shape[1] != embedding_dims:
        raise RuntimeError(
            f"Unexpected embedding shape {embeddings.shape}; expected (N, {embedding_dims})."
        )

    # --- Quality checks run BEFORE saving ---
    quality_ok = True

    try:
        _run_quality_checks(
            papers=papers,
            embeddings=embeddings,
            n_iaifi=n_iaifi,
            sanity_count=int(quality_cfg.get("sanity_check_count", 20)),
            sanity_k=int(quality_cfg.get("sanity_check_k", 10)),
        )
    except Exception as exc:
        quality_ok = False
        print(f"WARNING: neighbor coherence check failed: {exc}")

    if bool(quality_cfg.get("track_truncation_rate", True)):
        try:
            truncation_rate = _compute_truncation_rate(
                papers=papers,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            )
            print(f"Truncation rate: {truncation_rate:.3f}")
            warn_fraction = float(quality_cfg.get("truncation_warn_fraction", 0.15))
            if truncation_rate > warn_fraction:
                quality_ok = False
                print(
                    "WARNING: high truncation rate; "
                    "run neighbor coherence spot-check before shipping"
                )
        except Exception as exc:
            quality_ok = False
            print(f"WARNING: truncation audit failed: {exc}")

    if not quality_ok:
        print(
            "\n" + "=" * 60
            + "\n  WARNING: quality checks flagged issues (see above)."
            + "\n  Embeddings will still be saved, but review before shipping."
            + "\n" + "=" * 60
        )

    # --- Save embeddings (always, even if quality checks flagged issues) ---
    save_numpy(embeddings.astype(np.float32, copy=False), embeddings_path)
    row_to_arxiv_id = {str(i): str(p.get("arxiv_id", "")) for i, p in enumerate(papers)}
    save_json(row_to_arxiv_id, INDEX_PATH, indent=2)
    print(f"Saved embeddings: {embeddings.shape} -> {embeddings_path}")
    print(f"Saved embedding index ({len(row_to_arxiv_id)} rows) -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
