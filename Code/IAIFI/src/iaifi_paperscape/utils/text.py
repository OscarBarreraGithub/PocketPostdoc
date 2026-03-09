"""Text normalization helpers and domain stopwords."""

from __future__ import annotations

import re
import unicodedata


SCIENCE_STOPWORDS = [
    "a",
    "an",
    "and",
    "approach",
    "also",
    "analysis",
    "based",
    "can",
    "demonstrate",
    "describe",
    "develop",
    "effect",
    "first",
    "find",
    "for",
    "from",
    "however",
    "in",
    "introduce",
    "investigate",
    "is",
    "learning",
    "method",
    "methods",
    "model",
    "models",
    "new",
    "novel",
    "of",
    "on",
    "one",
    "our",
    "paper",
    "present",
    "problem",
    "propose",
    "proposed",
    "provide",
    "results",
    "result",
    "show",
    "shows",
    "significant",
    "study",
    "system",
    "systems",
    "that",
    "the",
    "their",
    "this",
    "through",
    "two",
    "use",
    "used",
    "using",
    "we",
    "well",
    "with",
    "work",
    "works",
    "data",
]


_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", re.IGNORECASE)
_ARXIV_PREFIX_RE = re.compile(r"^arxiv:\s*", re.IGNORECASE)
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def canonical_arxiv_id(arxiv_id: str) -> str:
    """Return canonical arXiv id by stripping URL/prefix and version suffix."""
    value = (arxiv_id or "").strip()
    if not value:
        return ""

    value = value.split("?", 1)[0].strip()
    value = _ARXIV_PREFIX_RE.sub("", value)

    url_match = _ARXIV_URL_RE.search(value)
    if url_match:
        value = url_match.group(1)

    if value.lower().endswith(".pdf"):
        value = value[:-4]

    value = _ARXIV_VERSION_RE.sub("", value)
    return value.strip()


def normalize_text(text: str) -> str:
    """Apply lightweight normalization for text processing."""
    value = unicodedata.normalize("NFKC", text or "").lower()
    value = _NON_ALNUM_RE.sub(" ", value)
    value = _WHITESPACE_RE.sub(" ", value)
    return value.strip()
