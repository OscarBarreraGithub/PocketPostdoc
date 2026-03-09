"""Step B: build the background candidate corpus from arXiv category queries."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from iaifi_paperscape.collect.arxiv_client import (
    build_session,
    query_category,
    to_arxiv_ts,
)
from iaifi_paperscape.utils.io import (
    load_json,
    load_jsonl,
    load_yaml_config,
    save_json,
    save_jsonl,
)
from iaifi_paperscape.utils.text import canonical_arxiv_id


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "corpus.yaml"
IAIFI_METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "iaifi_metadata.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "background_candidates.jsonl"
META_PATH = PROJECT_ROOT / "data" / "raw" / "background_candidates.meta.json"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "arxiv_api"


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _in_submitted_window(record: dict[str, Any], start_date: date, end_date: date) -> bool:
    submitted = _parse_date(str(record.get("published", "")))
    if submitted is None:
        return False
    return start_date <= submitted <= end_date


def _load_iaifi_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for record in load_jsonl(path):
        aid = canonical_arxiv_id(str(record.get("arxiv_id", "")))
        if aid:
            ids.add(aid)
    return ids


def _parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized).astimezone(timezone.utc)
    except ValueError:
        return None


def _load_previous_run_ts(meta_path: Path, output_path: Path) -> datetime | None:
    if meta_path.exists():
        meta = load_json(meta_path)
        previous = _parse_iso_utc(meta.get("run_completed_at"))
        if previous is not None:
            return previous
    if output_path.exists():
        return datetime.fromtimestamp(output_path.stat().st_mtime, tz=timezone.utc)
    return None


def main() -> None:
    config = load_yaml_config(CONFIG_PATH)
    bg_cfg = config.get("background", {})

    categories = list(bg_cfg.get("categories", []))
    candidate_cap = int(bg_cfg.get("candidate_cap", 30000))
    per_category_cap = int(bg_cfg.get("per_category_cap", 4000))
    max_results_per_page = int(bg_cfg.get("max_results_per_page", 2000))
    date_field = str(bg_cfg.get("date_field", "submittedDate"))
    query_time_utc = str(bg_cfg.get("query_time_utc", "0000"))

    start_ts = to_arxiv_ts(bg_cfg.get("start_date"), query_time_utc)
    end_ts = to_arxiv_ts(bg_cfg.get("end_date"), query_time_utc)

    start_date = _parse_date(bg_cfg.get("start_date")) or date(1900, 1, 1)
    end_date = _parse_date(bg_cfg.get("end_date")) or datetime.now(timezone.utc).date()

    iaifi_ids = _load_iaifi_ids(IAIFI_METADATA_PATH)
    previous_run_dt = _load_previous_run_ts(META_PATH, OUTPUT_PATH)

    seen_ids: set[str] = set()
    first_seen_category: dict[str, str] = {}
    ordered_ids: list[str] = []
    selected_by_id: dict[str, dict[str, Any]] = {}

    session = build_session(bg_cfg)
    try:
        cap_reached = False
        for category in tqdm(categories, desc="Categories", unit="cat"):
            search_query = f"cat:{category} AND {date_field}:[{start_ts} TO {end_ts}]"
            params = {
                "start": 0,
                "max_results": max_results_per_page,
                "sortBy": bg_cfg.get("sort_by", "submittedDate"),
                "sortOrder": bg_cfg.get("sort_order", "descending"),
                "__per_category_cap": per_category_cap,
            }

            for record in query_category(
                category=category,
                search_query=search_query,
                params=params,
                session=session,
                bg_cfg=bg_cfg,
                cache_dir=CACHE_DIR,
            ):
                aid = canonical_arxiv_id(str(record.get("arxiv_id", "")))
                if not aid:
                    continue
                if aid in seen_ids:
                    continue

                seen_ids.add(aid)
                first_seen_category[aid] = category

                if aid in iaifi_ids:
                    continue
                record["arxiv_id"] = aid
                if not _in_submitted_window(record, start_date, end_date):
                    continue

                record["is_iaifi"] = False
                record["iaifi_theme"] = None
                record["first_seen_category"] = category
                selected_by_id[aid] = record
                ordered_ids.append(aid)

                if len(ordered_ids) >= candidate_cap:
                    cap_reached = True
                    break

            print(
                f"[base] category={category} selected={len(ordered_ids)} "
                f"seen={len(seen_ids)} cap={candidate_cap}"
            )
            if cap_reached:
                break

        refreshed = 0
        if previous_run_dt is not None and selected_by_id:
            refresh_start_ts = previous_run_dt.strftime("%Y%m%d%H%M")
            refresh_end_ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
            refresh_cap = max(candidate_cap, per_category_cap)
            for category in categories:
                refresh_query = (
                    f"cat:{category} AND lastUpdatedDate:[{refresh_start_ts} TO {refresh_end_ts}]"
                )
                refresh_params = {
                    "start": 0,
                    "max_results": max_results_per_page,
                    "sortBy": bg_cfg.get("sort_by", "submittedDate"),
                    "sortOrder": bg_cfg.get("sort_order", "descending"),
                    "__per_category_cap": refresh_cap,
                    "__force_refresh": True,
                }

                for record in query_category(
                    category=category,
                    search_query=refresh_query,
                    params=refresh_params,
                    session=session,
                    bg_cfg=bg_cfg,
                    cache_dir=CACHE_DIR,
                ):
                    aid = canonical_arxiv_id(str(record.get("arxiv_id", "")))
                    if aid not in selected_by_id:
                        continue
                    if aid in iaifi_ids:
                        continue
                    if not _in_submitted_window(record, start_date, end_date):
                        continue

                    record["arxiv_id"] = aid
                    record["is_iaifi"] = False
                    record["iaifi_theme"] = None
                    record["first_seen_category"] = first_seen_category.get(
                        aid, selected_by_id[aid].get("first_seen_category", category)
                    )
                    selected_by_id[aid] = record
                    refreshed += 1

            print(f"[refresh] previous_run={previous_run_dt.isoformat()} refreshed={refreshed}")
    finally:
        session.close()

    # Inclusion predicate remains submittedDate in [start_date, end_date].
    final_ids = [
        aid
        for aid in ordered_ids
        if aid in selected_by_id
        and _in_submitted_window(selected_by_id[aid], start_date, end_date)
    ]
    final_ids = final_ids[:candidate_cap]
    final_records = [selected_by_id[aid] for aid in final_ids]

    save_jsonl(final_records, OUTPUT_PATH)
    completed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    save_json(
        {
            "run_completed_at": completed_at,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "candidate_count": len(final_records),
            "candidate_cap": candidate_cap,
            "date_field": date_field,
            "categories": categories,
        },
        META_PATH,
        indent=2,
    )

    print(f"Wrote {len(final_records)} records to {OUTPUT_PATH}")
    print(f"Wrote run metadata to {META_PATH}")


if __name__ == "__main__":
    main()
