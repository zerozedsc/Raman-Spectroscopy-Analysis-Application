"""Lightweight regression smoke tests.

Run with: uv run python smoke_tests.py

This is intentionally minimal and fast:
- Exercises analysis plot generators with mismatched wavenumber grids (grouped-mode failure mode)
- Exercises 3D figure rendering through MatplotlibWidget.update_plot (3D waterfall / peak scatter)

If this script exits with code 0, the core plotting pathways are at least importable and non-crashing.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def _make_synthetic_spectrum(wavenumbers: np.ndarray, peaks: list[tuple[float, float, float]]) -> np.ndarray:
    """Sum of Gaussians + small noise."""
    y = np.zeros_like(wavenumbers, dtype=float)
    for mu, sigma, amp in peaks:
        y += amp * np.exp(-0.5 * ((wavenumbers - mu) / sigma) ** 2)
    y += 0.02 * np.random.default_rng(0).standard_normal(size=wavenumbers.shape)
    return y


def main() -> int:
    # Import here so the script fails fast if imports break.
    from PySide6.QtWidgets import QApplication

    from components.widgets.matplotlib_widget import MatplotlibWidget
    from functions.visualization.analysis_plots import create_peak_scatter, create_waterfall_plot

    # Create a Qt app (required for MatplotlibWidget). No window is shown.
    app = QApplication.instance() or QApplication([])

    wn_a = np.linspace(400, 1800, 500)
    wn_b = np.linspace(450, 1750, 420)

    y_a1 = _make_synthetic_spectrum(wn_a, [(1000, 20, 1.0), (1650, 30, 0.7), (1400, 25, 0.4)])
    y_a2 = _make_synthetic_spectrum(wn_a, [(1005, 18, 1.1), (1645, 35, 0.65), (1410, 22, 0.35)])

    y_b1 = _make_synthetic_spectrum(wn_b, [(1002, 20, 0.9), (1652, 32, 0.75), (1390, 28, 0.45)])
    y_b2 = _make_synthetic_spectrum(wn_b, [(995,  25, 1.05), (1660, 28, 0.6), (1420, 26, 0.4)])

    df_a = pd.DataFrame({"s1": y_a1, "s2": y_a2}, index=wn_a)
    df_b = pd.DataFrame({"s1": y_b1, "s2": y_b2}, index=wn_b)

    # Grouped-mode style input (multi-dataset) for peak scatter
    dataset_data_multi = {"A": df_a, "B": df_b}

    # Single-dataset input for waterfall (registry expects single selection)
    dataset_data_single = {"A": df_a}

    # --- 2D peak scatter (resampling + masking) ---
    out_2d = create_peak_scatter(
        dataset_data_multi,
        {
            "peak_1_position": 1000,
            "peak_2_position": 1650,
            "tolerance": 15,
            "use_3d": False,
            "show_statistics": True,
            "show_legend": True,
        },
    )
    assert out_2d.get("primary_figure") is not None

    # --- 3D peak scatter (resampling + 3D axes) ---
    out_3d = create_peak_scatter(
        dataset_data_multi,
        {
            "peak_1_position": 1000,
            "peak_2_position": 1650,
            "peak_3_position": 1400,
            "tolerance": 20,
            "use_3d": True,
            "show_statistics": False,
            "show_legend": True,
        },
    )
    assert out_3d.get("primary_figure") is not None

    # --- 3D waterfall (axes projection + Poly3DCollection) ---
    out_wf_3d = create_waterfall_plot(
        dataset_data_single,
        {
            "use_3d": True,
            "offset_scale": 1.0,
            "max_spectra": 10,
            "colormap": "viridis",
            "show_grid": True,
            "reverse_order": False,
        },
    )
    assert out_wf_3d.get("primary_figure") is not None

    # --- Widget rendering smoke: copy plots into embedded figure without crashing ---
    w = MatplotlibWidget()
    w.update_plot(out_2d["primary_figure"])
    w.update_plot(out_3d["primary_figure"])
    w.update_plot(out_wf_3d["primary_figure"])

    # Close any remaining figures (update_plot closes the source figure internally).
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass

    # Avoid lint unused warnings; keep app alive for a tick.
    app.processEvents()

    print("✓ Smoke tests passed: 2D/3D peak scatter + 3D waterfall + widget update_plot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
