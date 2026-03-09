"""Step A: build the IAIFI seed metadata JSONL from a canonical CSV list."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from iaifi_paperscape.collect.arxiv_client import build_session, fetch_by_ids
from iaifi_paperscape.utils.io import load_jsonl, load_yaml_config, save_jsonl
from iaifi_paperscape.utils.text import canonical_arxiv_id


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "corpus.yaml"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "iaifi_metadata.jsonl"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "arxiv_api" / "iaifi_seed"


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_seed_csv(seed_file: Path) -> tuple[list[str], dict[str, str | None]]:
    rows: list[dict[str, str]] = []
    with seed_file.open("r", encoding="utf-8", newline="") as f:
        filtered_lines = [
            line for line in f if line.strip() and not line.lstrip().startswith("#")
        ]
    reader = csv.DictReader(filtered_lines)
    for row in reader:
        rows.append({k: (v or "").strip() for k, v in row.items()})

    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    theme_by_id: dict[str, str | None] = {}
    for row in rows:
        aid = canonical_arxiv_id(row.get("arxiv_id", ""))
        if not aid:
            continue
        if aid not in seen_ids:
            seen_ids.add(aid)
            ordered_ids.append(aid)
        theme = row.get("iaifi_theme", "").strip() or None
        if aid not in theme_by_id:
            theme_by_id[aid] = theme
    return ordered_ids, theme_by_id


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = load_jsonl(path)
    existing_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        aid = canonical_arxiv_id(str(record.get("arxiv_id", "")))
        if aid:
            existing_by_id[aid] = record
    return existing_by_id


def main() -> None:
    config = load_yaml_config(CONFIG_PATH)
    iaifi_cfg = config.get("iaifi", {})
    bg_cfg = config.get("background", {})

    seed_file = _resolve_path(iaifi_cfg.get("seed_file", "data/external/iaifi_papers.csv"))
    output_path = DEFAULT_OUTPUT_PATH
    cache_dir = DEFAULT_CACHE_DIR

    ordered_ids, theme_by_id = _load_seed_csv(seed_file)
    existing_by_id = _load_existing(output_path)
    missing_ids = [aid for aid in ordered_ids if aid not in existing_by_id]

    fetched_by_id: dict[str, dict[str, Any]] = {}
    if missing_ids:
        session = build_session(bg_cfg)
        try:
            fetched = fetch_by_ids(
                missing_ids,
                session=session,
                cache_dir=cache_dir,
                rate_limit=float(bg_cfg.get("rate_limit_seconds", 3)),
                bg_cfg=bg_cfg,
            )
        finally:
            session.close()
        fetched_by_id = {
            canonical_arxiv_id(str(record.get("arxiv_id", ""))): record for record in fetched
        }

    output_records: list[dict[str, Any]] = []
    missing_after_fetch: list[str] = []
    for aid in ordered_ids:
        record = dict(fetched_by_id.get(aid) or existing_by_id.get(aid) or {})
        if not record:
            missing_after_fetch.append(aid)
            continue
        record["arxiv_id"] = aid
        record["is_iaifi"] = True
        record["iaifi_theme"] = theme_by_id.get(aid)
        output_records.append(record)

    save_jsonl(output_records, output_path)

    print(f"Loaded {len(ordered_ids)} IAIFI IDs from {seed_file}")
    print(f"Skipped {len(ordered_ids) - len(missing_ids)} already-cached IDs")
    print(f"Fetched {len(fetched_by_id)} IDs from arXiv")
    print(f"Wrote {len(output_records)} records to {output_path}")
    if missing_after_fetch:
        print(
            "WARNING: Missing metadata for IDs: "
            + ", ".join(sorted(missing_after_fetch))
        )


if __name__ == "__main__":
    main()

