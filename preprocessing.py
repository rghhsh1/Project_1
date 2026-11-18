"""Spectral preprocessing for the shrimp storage-time prediction pipeline.
 
The pipeline consists of three steps, applied in this order:

    1. Savitzky-Golay smoothing   (noise suppression)
    2. Multiplicative scatter correction, MSC   (scatter / path-length correction)
    3. Global maximum scaling     (fixed unit conversion)

Input / output convention
-------------------------
All functions accept a 2-D array with shape ``(n_spectral_points, n_samples)``,
i.e. one spectrum per column, and return an array of the same shape.

Leakage note
------------
Step 3 and the MSC reference spectrum are dataset-level quantities. When the
data are preprocessed in a single call, these quantities are estimated from all
samples, including those later used for validation and testing. The functions
below therefore accept explicit ``reference`` / ``global_max`` arguments:

    stats = fit_preprocessing(x_train)                 # training set only
    x_train = preprocess_pipeline(x_train, **stats)
    x_test  = preprocess_pipeline(x_test,  **stats)

Calling ``preprocess_pipeline(x)`` without these arguments reproduces the
behaviour used for the results reported in the manuscript.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

#: Default Savitzky-Golay parameters used throughout this study.
SG_WINDOW = 11
SG_POLY = 2


def _as_float_2d(spectra: np.ndarray) -> np.ndarray:
    """Return ``spectra`` as a float64 2-D array, validating the shape."""
    arr = np.asarray(spectra, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"spectra must be 2-D (n_spectral_points, n_samples), got shape {arr.shape}"
        )
    if arr.shape[0] < 3 or arr.shape[1] < 1:
        raise ValueError(f"spectra must have at least 3 rows, got shape {arr.shape}")
    return arr


def savitzky_golay_smoothing(
    spectra: np.ndarray,
    window_length: int = SG_WINDOW,
    polyorder: int = SG_POLY,
) -> np.ndarray:
    """Smooth every spectrum with a Savitzky-Golay filter along the spectral axis."""
    x = _as_float_2d(spectra)
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}")
    if window_length > x.shape[0]:
        raise ValueError(
            f"window_length ({window_length}) exceeds the number of spectral points "
            f"({x.shape[0]})"
        )
    if polyorder >= window_length:
        raise ValueError(
            f"polyorder ({polyorder}) must be smaller than window_length ({window_length})"
        )
    return savgol_filter(x, window_length=window_length, polyorder=polyorder, axis=0)


def msc(spectra: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
    """Multiplicative scatter correction.

    Each spectrum is regressed on ``reference`` with a straight line, and the
    corrected spectrum is ``(x - intercept) / slope``.

    Parameters
    ----------
    spectra
        Array of shape ``(n_spectral_points, n_samples)``.
    reference
        1-D reference spectrum of length ``n_spectral_points``. If ``None``, the
        mean spectrum of ``spectra`` is used. Pass a reference estimated on the
        training set to avoid using test data.
    """
    x = _as_float_2d(spectra)
    if reference is None:
        ref = x.mean(axis=1)
    else:
        ref = np.asarray(reference, dtype=np.float64).ravel()
        if ref.shape[0] != x.shape[0]:
            raise ValueError(
                f"reference has length {ref.shape[0]}, expected {x.shape[0]}"
            )

    corrected = np.empty_like(x)
    for i in range(x.shape[1]):
        column = x[:, i]
        slope, intercept = np.polyfit(ref, column, 1)
        if not np.isfinite(slope) or abs(slope) < 1e-12:
            raise ValueError(
                f"degenerate MSC fit at column {i}: slope={slope!r}. "
                "The spectrum is flat or unrelated to the reference."
            )
        corrected[:, i] = (column - intercept) / slope
    return corrected


def global_max_scale(
    spectra: np.ndarray, global_max: Optional[float] = None
) -> np.ndarray:
    """Divide all intensities by a single positive constant.

    ``global_max`` should be estimated on the training set when a leakage-free
    protocol is required.
    """
    x = _as_float_2d(spectra)
    scale = float(np.max(x)) if global_max is None else float(global_max)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"global_max must be a positive finite number, got {scale!r}")
    return x / scale


def fit_preprocessing(
    spectra: np.ndarray,
    window_length: int = SG_WINDOW,
    polyorder: int = SG_POLY,
) -> dict:
    """Estimate the dataset-level preprocessing statistics on ``spectra``.

    Use this on the training set only, then pass the result to
    :func:`preprocess_pipeline` for the training, validation and test sets.

    Returns a dict with ``reference`` and ``global_max``, which can be expanded
    directly into :func:`preprocess_pipeline` via ``**stats``.
    """
    smoothed = savitzky_golay_smoothing(spectra, window_length, polyorder)
    reference = smoothed.mean(axis=1)
    msc_output = msc(smoothed, reference=reference)
    return {"reference": reference, "global_max": float(np.max(msc_output))}


def preprocess_pipeline(
    spectra: np.ndarray,
    reference: Optional[np.ndarray] = None,
    global_max: Optional[float] = None,
    window_length: int = SG_WINDOW,
    polyorder: int = SG_POLY,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing, MSC and global maximum scaling.

    With ``reference=None`` and ``global_max=None`` both quantities are
    estimated from ``spectra`` itself, which reproduces the preprocessing used
    for the results reported in the manuscript. Pass the output of
    :func:`fit_preprocessing` to reuse training-set statistics.
    """
    x = savitzky_golay_smoothing(spectra, window_length, polyorder)
    x = msc(x, reference=reference)
    return global_max_scale(x, global_max=global_max)
