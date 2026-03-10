#!/usr/bin/env python3
"""Scrape all IAIFI papers from the 4 category pages and output a CSV.

Usage:
    python3 scripts/scrape_iaifi_papers.py

Output:
    data/external/iaifi_papers.csv

Dependencies: only Python 3 standard library + beautifulsoup4.
If bs4 is missing, the script will attempt to install it automatically.
"""

import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from html.parser import HTMLParser
from urllib.request import urlopen, Request
from urllib.error import URLError

# Try to import BeautifulSoup; install if missing
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("beautifulsoup4 not found, attempting to install...")
    # Bootstrap pip if needed, then install bs4
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        from urllib.request import urlopen as _urlopen
        with open("/tmp/_get_pip.py", "wb") as _f:
            _f.write(_urlopen("https://bootstrap.pypa.io/get-pip.py").read())
        subprocess.check_call(
            [sys.executable, "/tmp/_get_pip.py", "--break-system-packages", "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--break-system-packages",
         "-q", "beautifulsoup4"],
    )
    from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CATEGORY_PAGES = {
    "AI": "https://iaifi.org/papers-ai.html",
    "Physics_theory": "https://iaifi.org/papers-theory.html",
    "Physics_experiment": "https://iaifi.org/papers-experiment.html",
    "Physics_astro": "https://iaifi.org/papers-astro.html",
}

# Map internal category keys to the theme label used in output
THEME_MAP = {
    "AI": "AI",
    "Physics_theory": "Physics",
    "Physics_experiment": "Physics",
    "Physics_astro": "Physics",
}

ARXIV_LINK_RE = re.compile(r"https://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_PATH = os.path.join(PROJECT_DIR, "data", "external", "iaifi_papers.csv")

# ---------------------------------------------------------------------------
# HTTP helper (stdlib only — no requests needed)
# ---------------------------------------------------------------------------

def fetch_page(url: str, timeout: int = 30) -> str:
    """Fetch a URL and return the response body as a string."""
    req = Request(url, headers={"User-Agent": "IAIFI-Scraper/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Extraction helpers (mirrors old scraper HTML structure)
# ---------------------------------------------------------------------------


def _extract_title(p_tag, arxiv_id: str) -> str:
    """Extract paper title from <strong><em> inside the <p> tag."""
    strong = p_tag.find("strong")
    if strong:
        em = strong.find("em")
        if em:
            return em.get_text(strip=True)
        return strong.get_text(strip=True)

    # Fallback: first non-empty text line
    lines = [l.strip() for l in p_tag.get_text().split("\n") if l.strip()]
    if lines:
        return lines[0]

    return f"Untitled ({arxiv_id})"


def _extract_authors(p_tag, arxiv_id: str) -> str:
    """Extract author list as a semicolon-separated string.

    Authors sit on the second text line of the <p> block (between the title
    and the [arXiv | code] bracket line).
    """
    lines = [l.strip() for l in p_tag.get_text().split("\n") if l.strip()]

    if len(lines) >= 2:
        author_line = lines[1]
        # If the line starts with '[' it is the links line, not authors
        if author_line.startswith("["):
            return ""
        # Clean trailing bracket artifacts that might be on the same line
        author_line = re.sub(r"\s*\[.*", "", author_line)
        # Normalise separators: replace commas with semicolons for CSV safety
        authors = [a.strip() for a in author_line.split(",") if a.strip()]
        return "; ".join(authors)

    return ""


def _arxiv_id_to_date(arxiv_id: str) -> str:
    """Derive an approximate publication date from the arXiv ID prefix.

    arXiv IDs are formatted as YYMM.NNNNN.  We map that to YYYY-MM-01.
    """
    match = re.match(r"(\d{2})(\d{2})\.", arxiv_id)
    if not match:
        return ""
    yy, mm = int(match.group(1)), int(match.group(2))
    yyyy = 2000 + yy
    try:
        return datetime(yyyy, mm, 1).strftime("%Y-%m-%d")
    except ValueError:
        return ""


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------


def scrape_page(url: str, category: str) -> list[dict]:
    """Scrape a single IAIFI category page and return paper dicts."""
    print(f"  Fetching {category}: {url} ...")
    html = fetch_page(url)

    soup = BeautifulSoup(html, "html.parser")
    papers = []

    arxiv_links = soup.find_all("a", href=ARXIV_LINK_RE)
    seen_on_page: set[str] = set()

    for link in arxiv_links:
        href = link.get("href", "")
        id_match = ARXIV_LINK_RE.search(href)
        if not id_match:
            continue

        arxiv_id = re.sub(r"v\d+$", "", id_match.group(1))  # strip version

        if arxiv_id in seen_on_page:
            continue
        seen_on_page.add(arxiv_id)

        p_tag = link.find_parent("p")
        if not p_tag:
            continue

        title = _extract_title(p_tag, arxiv_id)
        authors = _extract_authors(p_tag, arxiv_id)
        published = _arxiv_id_to_date(arxiv_id)

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "published": published,
            "iaifi_theme": THEME_MAP[category],
        })

    print(f"    -> {len(papers)} papers found")
    return papers


def scrape_all():
    """Scrape all 4 pages, deduplicate, and assign 'Both' theme where needed."""
    # Track themes per arxiv_id for the "Both" logic
    id_to_themes: dict[str, set[str]] = {}
    id_to_paper: dict[str, dict] = {}

    for category, url in CATEGORY_PAGES.items():
        try:
            papers = scrape_page(url, category)
        except Exception as exc:
            print(f"    ERROR scraping {category}: {exc}", file=sys.stderr)
            continue

        for p in papers:
            aid = p["arxiv_id"]
            theme = p["iaifi_theme"]

            if aid not in id_to_paper:
                id_to_paper[aid] = p
                id_to_themes[aid] = {theme}
            else:
                id_to_themes[aid].add(theme)

    # Resolve "Both" for papers appearing on AI *and* Physics pages
    duplicates_found = 0
    for aid, themes in id_to_themes.items():
        if len(themes) > 1:
            id_to_paper[aid]["iaifi_theme"] = "Both"
            duplicates_found += 1

    # Preserve insertion order (first-seen ordering)
    all_papers = list(id_to_paper.values())
    return all_papers, duplicates_found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Scraping IAIFI papers from 4 category pages...\n")

    all_papers, duplicates_found = scrape_all()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Write CSV
    fieldnames = ["arxiv_id", "title", "authors", "published", "iaifi_theme"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_papers)

    # Summary
    theme_counts: dict[str, int] = {}
    for p in all_papers:
        theme_counts[p["iaifi_theme"]] = theme_counts.get(p["iaifi_theme"], 0) + 1

    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total unique papers: {len(all_papers)}")
    print(f"Cross-listed (Both): {duplicates_found}")
    print(f"Per-theme counts:")
    for theme in sorted(theme_counts):
        print(f"  {theme}: {theme_counts[theme]}")
    print(f"\nOutput written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
