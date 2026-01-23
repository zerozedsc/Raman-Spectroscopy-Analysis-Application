"""Additional Visualization Methods for Analysis Page.

Currently includes:
- Derivative spectra overlay (Savitzky–Golay 1st/2nd derivative)

Kept in pages/analysis_page_utils so it can be dispatched by AnalysisThread.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from scipy.signal import savgol_filter

from .exploratory import interpolate_to_common_wavenumbers_with_groups


def _odd_window_length(n: int) -> int:
    n = int(n)
    if n < 3:
        n = 3
    if n % 2 == 0:
        n += 1
    return n


def create_derivative_spectra_plot(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Overlay mean spectra and their derivatives.

    Uses Savitzky–Golay differentiation.
    """

    if progress_callback:
        progress_callback(10)

    deriv_order = int(params.get("deriv_order", 1))
    window_length = _odd_window_length(params.get("window_length", 15))
    polyorder = int(params.get("polyorder", 3))
    show_original = bool(params.get("show_original", True))

    deriv_order = 1 if deriv_order not in (1, 2) else deriv_order
    polyorder = max(2, min(polyorder, window_length - 1))

    group_labels_map = params.get("_group_labels", None)
    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data, group_labels_map=group_labels_map, method="linear"
    )

    # Compute mean per label (dataset/group)
    unique = sorted(set(labels))
    means = {}
    for lab in unique:
        m = np.asarray([l == lab for l in labels], dtype=bool)
        if not np.any(m):
            continue
        means[lab] = np.mean(X[m, :], axis=0)

    if progress_callback:
        progress_callback(50)

    fig, ax = plt.subplots(figsize=(11, 6))

    for lab, y in means.items():
        if show_original:
            ax.plot(wavenumbers, y, linewidth=1.0, alpha=0.45, label=f"{lab} (orig)")

        # Savitzky-Golay derivative; delta uses mean spacing
        dx = float(np.mean(np.abs(np.diff(wavenumbers)))) if wavenumbers.size > 1 else 1.0
        dy = savgol_filter(y, window_length=window_length, polyorder=polyorder, deriv=deriv_order, delta=dx)
        ax.plot(wavenumbers, dy, linewidth=1.4, label=f"{lab} (d{deriv_order})")

    ax.set_title(f"Derivative Spectra Overlay (order={deriv_order})")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity / derivative")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    ax.invert_xaxis()

    summary = f"Derivative spectra computed (order={deriv_order}, window={window_length}, polyorder={polyorder})."

    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": None,
        "summary_text": summary,
        "detailed_summary": "",
        "raw_results": {},
    }
