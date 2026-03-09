"""Fetch citation counts from Semantic Scholar for arXiv papers.

Usage:
    python -m iaifi_paperscape.collect.fetch_citations

Queries the Semantic Scholar API for each paper's citation count.
Caches results to data/processed/citations.json so re-runs skip
already-fetched papers.

Rate limits:
    - Unauthenticated: 100 requests per 5 minutes
    - Authenticated (S2_API_KEY env var): ~1000 requests per minute

Output: data/processed/citations.json  (mapping arxiv_id -> citation_count | null)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from iaifi_paperscape.utils.io import load_json, load_jsonl, save_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IAIFI_METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "iaifi_metadata.jsonl"
BG_METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "background_metadata.jsonl"
CITATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "citations.json"

S2_API_BASE = "https://api.semanticscholar.org/graph/v1/paper"

# Rate-limit parameters
UNAUTH_RATE_LIMIT = 100   # requests per 5 minutes
UNAUTH_WINDOW_S = 5 * 60  # 5 minutes in seconds
AUTH_RATE_LIMIT = 1000     # requests per minute (with API key)
AUTH_WINDOW_S = 60         # 1 minute in seconds


def _load_cache(path: Path) -> dict[str, int | None]:
    """Load existing citation cache from disk, or return empty dict."""
    if path.exists():
        try:
            return load_json(path)
        except (json.JSONDecodeError, Exception):
            return {}
    return {}


def _save_cache(cache: dict[str, int | None], path: Path) -> None:
    """Persist citation cache to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(cache, path)


def _collect_arxiv_ids() -> list[str]:
    """Collect all arXiv IDs from IAIFI and background metadata."""
    ids: list[str] = []
    for meta_path in (IAIFI_METADATA_PATH, BG_METADATA_PATH):
        if meta_path.exists():
            records = load_jsonl(meta_path)
            for rec in records:
                aid = str(rec.get("arxiv_id") or rec.get("id", ""))
                if aid:
                    ids.append(aid)
    return ids


def _fetch_citation_count(
    arxiv_id: str,
    session: requests.Session,
    api_key: str | None = None,
) -> int | None:
    """Query Semantic Scholar for a single paper's citation count.

    Returns the citation count as int, or None if the paper is not found
    or an error occurs.
    """
    url = f"{S2_API_BASE}/ArXiv:{arxiv_id}?fields=citationCount"
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        resp = session.get(url, headers=headers, timeout=30)
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            # Rate limited — caller should back off
            raise RateLimitError("Rate limited by Semantic Scholar API")
        resp.raise_for_status()
        data = resp.json()
        count = data.get("citationCount")
        return int(count) if count is not None else None
    except RateLimitError:
        raise
    except requests.RequestException as exc:
        print(f"  WARNING: request failed for {arxiv_id}: {exc}")
        return None


class RateLimitError(Exception):
    """Raised when the API returns a 429 status."""
    pass


def main() -> None:
    api_key = os.environ.get("S2_API_KEY", "").strip() or None
    if api_key:
        print("Using authenticated Semantic Scholar API (S2_API_KEY set).")
        rate_limit = AUTH_RATE_LIMIT
        window_s = AUTH_WINDOW_S
    else:
        print("Using unauthenticated Semantic Scholar API (100 req / 5 min).")
        rate_limit = UNAUTH_RATE_LIMIT
        window_s = UNAUTH_WINDOW_S

    # Load all arXiv IDs
    all_ids = _collect_arxiv_ids()
    if not all_ids:
        print("No arXiv IDs found in metadata files. Nothing to fetch.")
        return
    print(f"Total arXiv IDs found: {len(all_ids)}")

    # Load existing cache and determine what to fetch
    cache = _load_cache(CITATIONS_PATH)
    to_fetch = [aid for aid in all_ids if aid not in cache]
    print(f"Already cached: {len(cache)}, to fetch: {len(to_fetch)}")

    if not to_fetch:
        print("All papers already cached. Nothing to do.")
        return

    session = requests.Session()
    request_times: list[float] = []
    fetched = 0
    save_interval = 50  # save cache every N fetches

    for i, arxiv_id in enumerate(to_fetch):
        # Rate limiting: maintain a sliding window
        now = time.monotonic()
        # Remove timestamps outside the current window
        request_times = [t for t in request_times if now - t < window_s]

        if len(request_times) >= rate_limit:
            # Wait until the oldest request in the window expires
            wait_time = window_s - (now - request_times[0]) + 0.5
            if wait_time > 0:
                print(f"  Rate limit reached. Sleeping {wait_time:.1f}s ...")
                time.sleep(wait_time)

        # Fetch
        try:
            count = _fetch_citation_count(arxiv_id, session, api_key)
        except RateLimitError:
            # Back off and retry once
            print("  429 received. Backing off 60s ...")
            time.sleep(60)
            try:
                count = _fetch_citation_count(arxiv_id, session, api_key)
            except RateLimitError:
                print(f"  Still rate-limited for {arxiv_id}. Setting to null.")
                count = None

        request_times.append(time.monotonic())
        cache[arxiv_id] = count
        fetched += 1

        if count is not None:
            status = f"cit={count}"
        else:
            status = "not found"
        print(f"  [{i + 1}/{len(to_fetch)}] {arxiv_id}: {status}")

        # Periodic save
        if fetched % save_interval == 0:
            _save_cache(cache, CITATIONS_PATH)

    # Final save
    _save_cache(cache, CITATIONS_PATH)
    print(f"Done. Fetched {fetched} papers. Total cached: {len(cache)}.")
    print(f"Saved to {CITATIONS_PATH}")


if __name__ == "__main__":
    main()
