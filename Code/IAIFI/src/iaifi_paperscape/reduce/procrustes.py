"""Step E.1: Procrustes alignment for monthly UMAP refits.

Aligns a new UMAP layout to the previous layout using IAIFI papers as
anchors, then computes drift diagnostics.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes


# ---------------------------------------------------------------------------
# Drift bands (v2.1 drift policy)
# ---------------------------------------------------------------------------
_LOW_DRIFT_MEDIAN = 0.03
_LOW_DRIFT_P95 = 0.08
_HIGH_DRIFT_MEDIAN = 0.05
_HIGH_DRIFT_P95 = 0.12


def classify_drift(drift: dict) -> str:
    """Return 'LOW', 'MEDIUM', or 'HIGH' drift band for a drift dict."""
    median = drift["median"]
    p95 = drift["p95"]
    if median > _HIGH_DRIFT_MEDIAN or p95 > _HIGH_DRIFT_P95:
        return "HIGH"
    if median <= _LOW_DRIFT_MEDIAN and p95 <= _LOW_DRIFT_P95:
        return "LOW"
    return "MEDIUM"


def align_to_previous(
    new_coords: np.ndarray,
    old_coords: np.ndarray,
    anchor_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Align new UMAP layout to previous layout using IAIFI papers as anchors.

    Parameters
    ----------
    new_coords : ndarray, shape (N, 2)
        New UMAP coordinates for all papers.
    old_coords : ndarray, shape (N, 2)
        Previous UMAP coordinates for all papers.
    anchor_mask : ndarray, shape (N,), dtype bool
        True for IAIFI papers present in both layouts.

    Returns
    -------
    aligned_coords : ndarray, shape (N, 2)
        Transformed new coordinates aligned to old layout.
    drift : dict
        Drift diagnostics: median, p95, max anchor displacement, reflected flag.
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
    # Reflection policy: allow det(R) < 0 (mirror). In 2D maps this preserves
    # neighborhoods and is acceptable; we do not force a det(R)=+1 rotation-only
    # transform.
    reflected = bool(np.linalg.det(R) < 0)
    scale = old_norm / new_norm

    # Apply transform to all points (not just anchors)
    aligned_coords = ((new_coords - new_mu) @ R) * scale + old_mu

    # Drift diagnostics on anchors (post-alignment)
    anchor_disp = np.linalg.norm(
        aligned_coords[anchor_mask] - old_coords[anchor_mask], axis=1
    )
    drift = {
        "median": float(np.median(anchor_disp)),
        "p95": float(np.percentile(anchor_disp, 95)),
        "max": float(np.max(anchor_disp)),
        "reflected": reflected,
    }

    return aligned_coords, drift
