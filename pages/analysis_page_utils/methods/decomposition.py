"""Spectral Unmixing / Decomposition Methods

Implements decomposition methods useful for mixed Raman spectra:
- MCR-ALS (non-negative ALS using NNLS)
- NMF (sklearn)
- ICA (FastICA)

All methods support multi-dataset input by concatenating spectra after
interpolating to a common wavenumber grid.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from scipy.optimize import nnls

from sklearn.decomposition import FastICA, NMF
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .exploratory import interpolate_to_common_wavenumbers_with_groups


def _apply_scaling(X: np.ndarray, scaling_type: str) -> np.ndarray:
    scaling_type = str(scaling_type or "None")
    if scaling_type == "StandardScaler":
        return StandardScaler().fit_transform(X)
    if scaling_type == "MinMaxScaler":
        return MinMaxScaler().fit_transform(X)
    return X


def _make_nonnegative(X: np.ndarray, *, mode: str = "shift") -> tuple[np.ndarray, float]:
    """Ensure X is non-negative.

    Args:
        mode:
            - "shift": subtract global minimum so min becomes 0
            - "clip": clip negatives to 0

    Returns:
        X_nn, shift
    """
    X = np.asarray(X, dtype=float)
    if mode == "clip":
        return np.clip(X, 0.0, None), 0.0

    mn = float(np.nanmin(X)) if X.size else 0.0
    if not np.isfinite(mn):
        mn = 0.0
    shift = -mn if mn < 0 else 0.0
    return X + shift, shift


def perform_mcr_als_analysis(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """MCR-ALS using a simple NNLS alternating least squares loop.

    This is designed to be dependency-light (no external pyMCR requirement).
    """

    if progress_callback:
        progress_callback(10)

    n_components = int(params.get("n_components", 3))
    max_iter = int(params.get("max_iter", 50))
    tol = float(params.get("tol", 1e-4))
    scaling_type = params.get("scaling", "None")
    nonneg_mode = str(params.get("nonneg_mode", "shift"))

    group_labels_map = params.get("_group_labels", None)
    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data, group_labels_map=group_labels_map, method="linear"
    )

    X = _apply_scaling(X, scaling_type)
    X_nn, shift = _make_nonnegative(X, mode=nonneg_mode)

    n_samples, n_features = X_nn.shape
    n_components = max(2, min(n_components, n_samples, n_features))

    if progress_callback:
        progress_callback(25)

    # Initialize component spectra using NMF for stability
    nmf_init = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=0,
        max_iter=500,
    )
    C0 = nmf_init.fit_transform(X_nn)  # (n_samples, k)
    S = nmf_init.components_.T  # (n_features, k)

    errors: list[float] = []
    prev_err = None

    # ALS loop
    for it in range(max_iter):
        # Step 1: solve concentrations C with NNLS per sample
        C = np.zeros((n_samples, n_components), dtype=float)
        for i in range(n_samples):
            # Solve S c ~= x
            c_i, _ = nnls(S, X_nn[i, :])
            C[i, :] = c_i

        # Step 2: solve spectra S with NNLS per feature
        S_new = np.zeros((n_features, n_components), dtype=float)
        for j in range(n_features):
            s_j, _ = nnls(C, X_nn[:, j])
            S_new[j, :] = s_j
        S = S_new

        X_hat = C @ S.T
        err = float(np.linalg.norm(X_nn - X_hat) / (np.linalg.norm(X_nn) + 1e-12))
        errors.append(err)

        if prev_err is not None:
            if abs(prev_err - err) < tol:
                break
        prev_err = err

        if progress_callback:
            progress_callback(25 + int(65 * (it + 1) / max_iter))

    if progress_callback:
        progress_callback(90)

    # Component spectra plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for k in range(n_components):
        ax1.plot(wavenumbers, S[:, k], linewidth=1.2, label=f"Comp {k+1}")
    ax1.set_title("MCR-ALS: Component Spectra")
    ax1.set_xlabel("Wavenumber (cm⁻¹)")
    ax1.set_ylabel("Intensity (a.u.)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")
    ax1.invert_xaxis()

    # Concentration heatmap
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    im = ax2.imshow(C, aspect="auto", interpolation="nearest")
    ax2.set_title("MCR-ALS: Component Abundances (per spectrum)")
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Spectrum")
    ax2.set_xticks(np.arange(n_components))
    ax2.set_xticklabels([f"C{k+1}" for k in range(n_components)])
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # Data table
    df = pd.DataFrame({"label": [str(v) for v in labels]})
    for k in range(n_components):
        df[f"C{k+1}"] = C[:, k]

    summary = (
        f"MCR-ALS completed. Components: {n_components}. Iterations: {len(errors)}.\n"
        f"Final relative error: {errors[-1]:.4f}. Non-negativity: {nonneg_mode} (shift={shift:.3g})."
    )

    return {
        "primary_figure": fig1,
        "secondary_figure": fig2,
        "data_table": df,
        "summary_text": summary,
        "detailed_summary": f"Errors: {errors[:10]}{' ...' if len(errors) > 10 else ''}",
        "raw_results": {
            "C": C,
            "S": S,
            "errors": errors,
            "shift": shift,
            # Standardized component-viewer payload (used by Results UI)
            "wavenumbers": np.asarray(wavenumbers, dtype=float),
            "component_spectra": np.asarray(S, dtype=float),  # (n_features, k)
            "component_prefix": "Comp",
            "component_y_label": "Intensity (a.u.)",
            "labels": [str(v) for v in labels],
        },
    }


def perform_nmf_analysis(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    if progress_callback:
        progress_callback(10)

    n_components = int(params.get("n_components", 3))
    max_iter = int(params.get("max_iter", 500))
    nonneg_mode = str(params.get("nonneg_mode", "shift"))

    group_labels_map = params.get("_group_labels", None)
    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data, group_labels_map=group_labels_map, method="linear"
    )

    X_nn, shift = _make_nonnegative(X, mode=nonneg_mode)

    n_components = max(2, min(n_components, X_nn.shape[0], X_nn.shape[1]))

    if progress_callback:
        progress_callback(35)

    nmf = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=0,
        max_iter=max_iter,
    )
    W = nmf.fit_transform(X_nn)  # (n_samples, k)
    H = nmf.components_  # (k, n_features)

    if progress_callback:
        progress_callback(80)

    # Component spectra plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for k in range(n_components):
        ax1.plot(wavenumbers, H[k, :], linewidth=1.2, label=f"Comp {k+1}")
    ax1.set_title("NMF: Component Spectra")
    ax1.set_xlabel("Wavenumber (cm⁻¹)")
    ax1.set_ylabel("Intensity (a.u.)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")
    ax1.invert_xaxis()

    # Mixing coefficients heatmap
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    im = ax2.imshow(W, aspect="auto", interpolation="nearest")
    ax2.set_title("NMF: Mixing Coefficients (per spectrum)")
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Spectrum")
    ax2.set_xticks(np.arange(n_components))
    ax2.set_xticklabels([f"C{k+1}" for k in range(n_components)])
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    df = pd.DataFrame({"label": [str(v) for v in labels]})
    for k in range(n_components):
        df[f"C{k+1}"] = W[:, k]

    summary = (
        f"NMF completed. Components: {n_components}. Iterations: {getattr(nmf, 'n_iter_', 'n/a')}.\n"
        f"Reconstruction error: {float(getattr(nmf, 'reconstruction_err_', float('nan'))):.4f}. Non-negativity: {nonneg_mode} (shift={shift:.3g})."
    )

    return {
        "primary_figure": fig1,
        "secondary_figure": fig2,
        "data_table": df,
        "summary_text": summary,
        "detailed_summary": "",
        "raw_results": {
            "W": W,
            "H": H,
            "shift": shift,
            "nmf_model": nmf,
            # Standardized component-viewer payload (used by Results UI)
            "wavenumbers": np.asarray(wavenumbers, dtype=float),
            "component_spectra": np.asarray(H, dtype=float).T,  # (n_features, k)
            "component_prefix": "Comp",
            "component_y_label": "Intensity (a.u.)",
            "labels": [str(v) for v in labels],
        },
    }


def perform_ica_analysis(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    if progress_callback:
        progress_callback(10)

    n_components = int(params.get("n_components", 3))
    scaling_type = params.get("scaling", "StandardScaler")
    max_iter = int(params.get("max_iter", 500))

    group_labels_map = params.get("_group_labels", None)
    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data, group_labels_map=group_labels_map, method="linear"
    )

    X_scaled = _apply_scaling(X, scaling_type)

    n_components = max(2, min(n_components, X_scaled.shape[0], X_scaled.shape[1]))

    if progress_callback:
        progress_callback(35)

    ica = FastICA(
        n_components=n_components,
        whiten="unit-variance",
        random_state=0,
        max_iter=max_iter,
        tol=1e-4,
    )

    S = ica.fit_transform(X_scaled)  # (n_samples, k)
    A = ica.mixing_  # (n_features, k)

    if progress_callback:
        progress_callback(80)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for k in range(n_components):
        ax1.plot(wavenumbers, A[:, k], linewidth=1.1, label=f"IC{k+1}")
    ax1.set_title("ICA: Independent Component Spectra (mixing columns)")
    ax1.set_xlabel("Wavenumber (cm⁻¹)")
    ax1.set_ylabel("Relative intensity")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")
    ax1.invert_xaxis()

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    im = ax2.imshow(S, aspect="auto", interpolation="nearest")
    ax2.set_title("ICA: Source Activations (per spectrum)")
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Spectrum")
    ax2.set_xticks(np.arange(n_components))
    ax2.set_xticklabels([f"IC{k+1}" for k in range(n_components)])
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    df = pd.DataFrame({"label": [str(v) for v in labels]})
    for k in range(n_components):
        df[f"IC{k+1}"] = S[:, k]

    summary = f"ICA completed. Components: {n_components}. Iterations: {getattr(ica, 'n_iter_', 'n/a')}."

    return {
        "primary_figure": fig1,
        "secondary_figure": fig2,
        "data_table": df,
        "summary_text": summary,
        "detailed_summary": "",
        "raw_results": {
            "S": S,
            "A": A,
            "ica_model": ica,
            # Standardized component-viewer payload (used by Results UI)
            "wavenumbers": np.asarray(wavenumbers, dtype=float),
            "component_spectra": np.asarray(A, dtype=float),  # (n_features, k)
            "component_prefix": "IC",
            "component_y_label": "Relative intensity",
            "labels": [str(v) for v in labels],
        },
    }
