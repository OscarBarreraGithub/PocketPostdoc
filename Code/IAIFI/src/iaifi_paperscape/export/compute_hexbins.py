"""H.12 backend: precompute hexbin density grid for v2.1b multiscale UI.

Usage:
    PYTHONPATH=src python3 -m iaifi_paperscape.export.compute_hexbins [--gridsize 30]

Loads 2D coordinates, identifies background points (non-IAIFI), and computes
a hexagonal binning of their density.  Exports web/data/hexbins.json as an
array of {cx, cy, count, opacity} records for the frontend hexbin layer.

This is a fast-follow feature (v2.1b), not blocking launch.  The frontend
renders hexbins on a secondary canvas beneath the scatter layer, with opacity
crossfade as zoom increases.  IAIFI points are ALWAYS visible at all zoom
levels and never collapse into hexbins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from iaifi_paperscape.utils.io import load_jsonl, load_numpy

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COORDS_PATH = PROJECT_ROOT / "data" / "processed" / "coords_2d.npy"
IAIFI_METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "iaifi_metadata.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "web" / "data" / "hexbins.json"


def compute_hexbins(
    coords: np.ndarray,
    n_iaifi: int,
    gridsize: int = 30,
) -> list[dict]:
    """Compute hexbin density for background points only.

    Parameters
    ----------
    coords : ndarray, shape (N, 2)
        2D coordinates for all papers (IAIFI first, then background).
    n_iaifi : int
        Number of IAIFI papers (first n_iaifi rows are IAIFI).
    gridsize : int
        Number of hexagons along the x-axis.

    Returns
    -------
    hexbins : list of dict
        Each dict has keys: cx, cy, count, opacity.
        Opacity is proportional to count, normalized to [0, 1].
    """
    # Background only -- IAIFI points are always rendered individually
    bg_coords = coords[n_iaifi:]

    if len(bg_coords) == 0:
        print("WARNING: no background points found; hexbin output will be empty.")
        return []

    fig, ax = plt.subplots()
    hb = ax.hexbin(
        bg_coords[:, 0],
        bg_coords[:, 1],
        gridsize=gridsize,
        mincnt=1,
    )

    offsets = hb.get_offsets()
    counts = hb.get_array()
    plt.close(fig)

    max_count = float(counts.max()) if len(counts) > 0 else 1.0
    if max_count == 0:
        max_count = 1.0

    hexbins = []
    for (cx, cy), count in zip(offsets, counts):
        count_int = int(count)
        if count_int < 1:
            continue
        hexbins.append({
            "cx": round(float(cx), 4),
            "cy": round(float(cy), 4),
            "count": count_int,
            "opacity": round(float(count / max_count), 4),
        })

    return hexbins


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H.12: precompute hexbin density for v2.1b multiscale UI"
    )
    parser.add_argument(
        "--gridsize",
        type=int,
        default=30,
        help="Number of hexagons along the x-axis (default: 30)",
    )
    args = parser.parse_args()

    print("H.12: computing hexbin density grid...")

    coords = load_numpy(COORDS_PATH)
    print(f"Loaded coords: {coords.shape}")

    iaifi_metadata = load_jsonl(IAIFI_METADATA_PATH)
    n_iaifi = len(iaifi_metadata)
    n_bg = coords.shape[0] - n_iaifi
    print(f"IAIFI papers: {n_iaifi}, background: {n_bg}")

    hexbins = compute_hexbins(coords, n_iaifi, gridsize=args.gridsize)

    # Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(hexbins, f, separators=(",", ":"))

    print(f"Saved {len(hexbins)} hexbin cells to {OUTPUT_PATH}")
    total_bg = int(sum(h["count"] for h in hexbins))
    print(f"Total background points covered: {total_bg}")


if __name__ == "__main__":
    main()
