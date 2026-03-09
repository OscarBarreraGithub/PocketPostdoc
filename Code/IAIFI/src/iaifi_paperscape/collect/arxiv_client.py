"""arXiv API client helpers for IAIFI collection steps."""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tqdm import tqdm

from iaifi_paperscape.utils.text import canonical_arxiv_id


ARXIV_API_URL = "http://export.arxiv.org/api/query"
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", re.IGNORECASE)
_ARXIV_PREFIX_RE = re.compile(r"^arxiv:\s*", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def to_arxiv_ts(date_str: str | None, fallback_hhmm: str = "0000") -> str:
    """Convert YYYY-MM-DD (or None for now UTC) to arXiv timestamp."""
    if date_str is None:
        dt = datetime.now(timezone.utc)
    else:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return dt.strftime("%Y%m%d") + fallback_hhmm


def build_session(bg_cfg: dict[str, Any]) -> requests.Session:
    """Build a requests session with retrying HTTP adapters."""
    retries = Retry(
        total=bg_cfg.get("max_retries", 5),
        connect=bg_cfg.get("max_retries", 5),
        read=bg_cfg.get("max_retries", 5),
        status=bg_cfg.get("max_retries", 5),
        backoff_factor=0.5,
        status_forcelist=tuple(bg_cfg.get("retry_statuses", [429, 500, 502, 503, 504])),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": bg_cfg["user_agent"]})
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_with_network_retry(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    bg_cfg: dict[str, Any],
) -> requests.Response:
    """Retry GET requests on timeout/connection errors with exponential backoff."""
    max_tries = bg_cfg.get("max_retries", 5)
    base = bg_cfg.get("network_retry_base_seconds", 1.0)
    timeout = bg_cfg.get("timeout_seconds", 30)
    for attempt in range(max_tries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == max_tries - 1:
                raise
            time.sleep(base * (2**attempt))
    raise RuntimeError("Unreachable network retry state")


def _collapse_ws(value: str | None) -> str:
    return _WHITESPACE_RE.sub(" ", (value or "")).strip()


def _to_date(value: str | None) -> str | None:
    if not value:
        return None
    return value[:10]


def _extract_versioned_id(raw_value: str | None) -> str:
    value = (raw_value or "").strip().split("?", 1)[0]
    value = _ARXIV_PREFIX_RE.sub("", value)
    match = _ARXIV_URL_RE.search(value)
    if match:
        value = match.group(1)
    if value.lower().endswith(".pdf"):
        value = value[:-4]
    return value.strip()


def _extract_pdf_url(entry: Any, arxiv_id: str) -> str:
    links = entry.get("links", []) if hasattr(entry, "get") else []
    for link in links:
        href = (link.get("href") or "").strip()
        if not href:
            continue
        if (
            link.get("title") == "pdf"
            or link.get("type") == "application/pdf"
            or "/pdf/" in href
        ):
            if href.endswith(".pdf"):
                href = href[:-4]
            return href
    return f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else ""


def parse_entry(entry: Any) -> dict[str, Any]:
    """Parse a feedparser Atom entry into canonical metadata fields."""
    entry_id = _extract_versioned_id(entry.get("id") if hasattr(entry, "get") else "")
    arxiv_id = canonical_arxiv_id(entry_id)
    arxiv_id_versioned = entry_id or arxiv_id

    authors: list[str] = []
    for author in entry.get("authors", []) if hasattr(entry, "get") else []:
        name = _collapse_ws(author.get("name", "") if isinstance(author, dict) else str(author))
        if name:
            authors.append(name)

    categories: list[str] = []
    seen_categories: set[str] = set()
    for tag in entry.get("tags", []) if hasattr(entry, "get") else []:
        term = _collapse_ws(tag.get("term", "") if isinstance(tag, dict) else str(tag))
        if term and term not in seen_categories:
            seen_categories.add(term)
            categories.append(term)

    primary_category: str | None = None
    primary_obj = entry.get("arxiv_primary_category") if hasattr(entry, "get") else None
    if isinstance(primary_obj, dict):
        primary_category = _collapse_ws(primary_obj.get("term"))
    elif primary_obj is not None:
        primary_category = _collapse_ws(getattr(primary_obj, "term", ""))
    if not primary_category and categories:
        primary_category = categories[0]

    return {
        "arxiv_id": arxiv_id,
        "arxiv_id_versioned": arxiv_id_versioned,
        "title": _collapse_ws(entry.get("title") if hasattr(entry, "get") else ""),
        "abstract": _collapse_ws(entry.get("summary") if hasattr(entry, "get") else ""),
        "authors": authors,
        "primary_category": primary_category,
        "categories": categories,
        "published": _to_date(entry.get("published") if hasattr(entry, "get") else None),
        "updated": _to_date(entry.get("updated") if hasattr(entry, "get") else None),
        "pdf_url": _extract_pdf_url(entry, arxiv_id),
    }


def _stable_ids(arxiv_ids: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_id in arxiv_ids:
        aid = canonical_arxiv_id(raw_id)
        if aid and aid not in seen:
            seen.add(aid)
            ordered.append(aid)
    return ordered


def _id_batch_cache_path(cache_root: Path, batch_ids: list[str], batch_index: int) -> Path:
    joined = ",".join(batch_ids)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]
    return cache_root / f"id_batch_{batch_index:04d}_{digest}.xml"


def fetch_by_ids(
    arxiv_ids: Iterable[str],
    session: requests.Session,
    cache_dir: str | Path,
    rate_limit: float = 3,
    bg_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fetch arXiv metadata in id_list batches of 50 and cache raw XML."""
    if bg_cfg is None:
        bg_cfg = {"max_retries": 5, "network_retry_base_seconds": 1.0, "timeout_seconds": 30}

    ordered_ids = _stable_ids(arxiv_ids)
    if not ordered_ids:
        return []

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    parsed_by_id: dict[str, dict[str, Any]] = {}
    batch_range = range(0, len(ordered_ids), 50)
    for batch_start in tqdm(batch_range, desc="Fetching ID batches", unit="batch"):
        batch_ids = ordered_ids[batch_start : batch_start + 50]
        batch_index = batch_start // 50
        cache_path = _id_batch_cache_path(cache_root, batch_ids, batch_index)

        if cache_path.exists():
            raw_xml = cache_path.read_text(encoding="utf-8")
        else:
            params = {
                "id_list": ",".join(batch_ids),
                "start": 0,
                "max_results": len(batch_ids),
            }
            resp = get_with_network_retry(session, ARXIV_API_URL, params, bg_cfg)
            raw_xml = resp.text
            cache_path.write_text(raw_xml, encoding="utf-8")
            if batch_start + 50 < len(ordered_ids):
                time.sleep(rate_limit)

        parsed = feedparser.parse(raw_xml)
        for entry in getattr(parsed, "entries", []):
            payload = parse_entry(entry)
            aid = payload.get("arxiv_id", "")
            if aid:
                parsed_by_id[aid] = payload

    return [parsed_by_id[aid] for aid in ordered_ids if aid in parsed_by_id]


def _category_cache_path(
    cache_root: Path,
    category: str,
    search_query: str,
    start: int,
    max_results: int,
    sort_by: str,
    sort_order: str,
) -> Path:
    stable = "|".join([category, search_query, sort_by, sort_order])
    digest = hashlib.sha1(stable.encode("utf-8")).hexdigest()[:16]
    category_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", category)
    return cache_root / (
        f"cat_{category_slug}_{digest}_start{start:06d}_max{max_results}.xml"
    )


def query_category(
    category: str,
    search_query: str,
    params: dict[str, Any],
    session: requests.Session,
    bg_cfg: dict[str, Any],
    cache_dir: str | Path,
) -> Iterator[dict[str, Any]]:
    """Query one category with paginated arXiv API calls."""
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    start = int(params.get("start", 0))
    max_results = int(params.get("max_results", bg_cfg.get("max_results_per_page", 2000)))
    per_category_cap = int(params.get("__per_category_cap", bg_cfg.get("per_category_cap", 4000)))
    rate_limit_seconds = float(bg_cfg.get("rate_limit_seconds", 3))
    n_emitted = 0
    pbar = tqdm(total=per_category_cap, desc=f"Paginating {category}", unit="paper", leave=False)

    while n_emitted < per_category_cap:
        page_params = dict(params)
        force_refresh = bool(page_params.pop("__force_refresh", False))
        page_params.pop("__per_category_cap", None)
        page_params["search_query"] = search_query
        page_params["start"] = start
        page_params["max_results"] = max_results
        page_params.setdefault("sortBy", bg_cfg.get("sort_by", "submittedDate"))
        page_params.setdefault("sortOrder", bg_cfg.get("sort_order", "descending"))

        cache_path = _category_cache_path(
            cache_root=cache_root,
            category=category,
            search_query=search_query,
            start=start,
            max_results=max_results,
            sort_by=str(page_params.get("sortBy", "")),
            sort_order=str(page_params.get("sortOrder", "")),
        )

        if cache_path.exists() and not force_refresh:
            raw_xml = cache_path.read_text(encoding="utf-8")
        else:
            resp = get_with_network_retry(session, ARXIV_API_URL, page_params, bg_cfg)
            raw_xml = resp.text
            cache_path.write_text(raw_xml, encoding="utf-8")

        parsed = feedparser.parse(raw_xml)
        entries = list(getattr(parsed, "entries", []))
        if not entries:
            break

        for entry in entries:
            yield parse_entry(entry)
            n_emitted += 1
            pbar.update(1)
            if n_emitted >= per_category_cap:
                break

        if n_emitted >= per_category_cap or len(entries) < max_results:
            break

        start += max_results
        # arXiv API guideline: wait at least a few seconds between requests.
        time.sleep(rate_limit_seconds)

    pbar.close()

