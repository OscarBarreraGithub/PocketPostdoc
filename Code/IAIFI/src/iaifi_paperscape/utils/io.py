"""I/O helpers for JSON, JSONL, NumPy, and YAML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


PathLike = str | Path


def load_jsonl(path: PathLike) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of JSON objects."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(records: Iterable[dict[str, Any]], path: PathLike) -> None:
    """Write records to a JSONL file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def load_json(path: PathLike) -> Any:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: Any, path: PathLike, *, indent: int = 2) -> None:
    """Write payload to a JSON file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def load_numpy(path: PathLike) -> np.ndarray:
    """Load a NumPy array from disk."""
    return np.load(Path(path), allow_pickle=False)


def save_numpy(array: np.ndarray, path: PathLike) -> None:
    """Save a NumPy array to disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, array)


def load_yaml_config(path: PathLike) -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if loaded is not None else {}
