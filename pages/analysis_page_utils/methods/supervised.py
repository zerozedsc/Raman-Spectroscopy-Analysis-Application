"""Supervised / Chemometrics Analysis Methods

This module implements supervised exploratory methods commonly used in
biomedical Raman spectroscopy:
- PLS-DA (PLSRegression + discriminant wrapper)
- LDA / PCA-LDA

All functions follow the AnalysisThread calling convention:
    fn(dataset_data: Dict[str, pd.DataFrame], params: Dict[str, Any], progress_callback=None) -> Dict[str, Any]

Notes:
- Uses a non-GUI Matplotlib backend for thread safety.
- Uses the same wavenumber interpolation helper as PCA to support mixed grids.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from configs.configs import create_logs

from .exploratory import interpolate_to_common_wavenumbers_with_groups


def _encode_labels(labels: list[str]) -> tuple[np.ndarray, list[str]]:
    classes = sorted(set(map(str, labels)))
    y = np.asarray([classes.index(str(v)) for v in labels], dtype=int)
    return y, classes


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    Y = np.zeros((y.size, n_classes), dtype=float)
    Y[np.arange(y.size), y] = 1.0
    return Y


def _apply_scaling(X: np.ndarray, scaling_type: str) -> Tuple[np.ndarray, object | None]:
    scaling_type = str(scaling_type or "StandardScaler")
    if scaling_type == "StandardScaler":
        scaler = StandardScaler()
        return scaler.fit_transform(X), scaler
    if scaling_type == "MinMaxScaler":
        scaler = MinMaxScaler()
        return scaler.fit_transform(X), scaler
    return X, None


def _vip_scores(pls: PLSRegression) -> np.ndarray:
    """Compute VIP scores for a fitted PLSRegression.

    References:
    - Common chemometrics VIP formulation using X weights and Y loadings.

    Returns:
        vip: shape (n_features,)
    """
    # Shapes:
    # W: (n_features, n_components)
    # T: (n_samples, n_components)
    # Q: (n_targets, n_components)
    W = np.asarray(pls.x_weights_, dtype=float)
    T = np.asarray(pls.x_scores_, dtype=float)
    Q = np.asarray(pls.y_loadings_, dtype=float)

    p, a = W.shape
    if p == 0 or a == 0:
        return np.zeros((p,), dtype=float)

    # SSY explained per component (approx): sum(t_k^2) * sum(q_k^2)
    ssy = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    total_ssy = float(np.sum(ssy))
    if not np.isfinite(total_ssy) or total_ssy <= 0:
        return np.zeros((p,), dtype=float)

    vip = np.zeros((p,), dtype=float)
    for j in range(p):
        wj = W[j, :]
        vip[j] = np.sqrt(p * float(np.sum(ssy * (wj ** 2))) / total_ssy)

    return vip


def perform_pls_da_analysis(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """PLS-DA (supervised latent variable projection) for group comparison."""

    if progress_callback:
        progress_callback(10)

    n_components = int(params.get("n_components", 2))
    scaling_type = params.get("scaling", "StandardScaler")
    cv_folds = int(params.get("cv_folds", 5))
    show_vip = bool(params.get("show_vip", True))
    show_loadings = bool(params.get("show_loadings", True))

    group_labels_map = params.get("_group_labels", None)

    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data, group_labels_map=group_labels_map, method="linear"
    )

    if len(set(labels)) < 2:
        raise ValueError("PLS-DA requires at least 2 groups/classes")

    if progress_callback:
        progress_callback(25)

    X_scaled, _scaler = _apply_scaling(X, scaling_type)

    y, classes = _encode_labels(labels)
    Y = _one_hot(y, len(classes))

    # Bound components to data rank
    max_comps = max(1, min(int(n_components), X_scaled.shape[0] - 1, X_scaled.shape[1]))
    n_components = max_comps

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, Y)

    if progress_callback:
        progress_callback(55)

    # Scores in LV space
    scores = np.asarray(pls.x_scores_, dtype=float)

    # CV accuracy using logistic regression on LV scores (manual CV)
    cv_folds = max(2, min(cv_folds, int(np.min(np.bincount(y))) if y.size else 2))
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    from sklearn.linear_model import LogisticRegression

    y_pred_cv = np.full_like(y, fill_value=-1)
    accs: list[float] = []

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        Y_tr = _one_hot(y_tr, len(classes))

        pls_cv = PLSRegression(n_components=n_components)
        pls_cv.fit(X_tr, Y_tr)

        T_tr = pls_cv.transform(X_tr)
        T_te = pls_cv.transform(X_te)

        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(T_tr, y_tr)

        pred = clf.predict(T_te)
        y_pred_cv[test_idx] = pred
        accs.append(float(np.mean(pred == y_te)))

    cv_acc_mean = float(np.mean(accs)) if accs else float("nan")

    cm = confusion_matrix(y, y_pred_cv, labels=list(range(len(classes))))

    if progress_callback:
        progress_callback(75)

    # --- Figure 1: LV1 vs LV2 scores ---
    fig1, ax1 = plt.subplots(figsize=(9, 7))
    if scores.shape[1] >= 2:
        for i, cls in enumerate(classes):
            m = y == i
            ax1.scatter(
                scores[m, 0],
                scores[m, 1],
                s=70,
                alpha=0.8,
                label=str(cls),
                edgecolors="white",
                linewidths=0.8,
            )
        ax1.set_xlabel("LV1")
        ax1.set_ylabel("LV2")
        ax1.set_title("PLS-DA Scores (LV1 vs LV2)")
    else:
        for i, cls in enumerate(classes):
            m = y == i
            ax1.hist(scores[m, 0], bins=20, alpha=0.6, label=str(cls))
        ax1.set_xlabel("LV1")
        ax1.set_ylabel("Count")
        ax1.set_title("PLS-DA Scores (LV1)")

    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    # --- Figure 2: VIP and/or Loadings ---
    fig2 = None
    if show_vip or show_loadings:
        rows = int(bool(show_vip)) + int(bool(show_loadings))
        fig2, axes = plt.subplots(rows, 1, figsize=(10, 4.5 * rows), sharex=True)
        if rows == 1:
            axes = [axes]

        ax_idx = 0
        if show_vip:
            vip = _vip_scores(pls)
            ax = axes[ax_idx]
            ax.plot(wavenumbers, vip, color="purple", linewidth=1.2)
            ax.set_ylabel("VIP")
            ax.set_title("VIP Scores (Variable Importance in Projection)")
            ax.grid(True, alpha=0.25)
            ax.invert_xaxis()
            ax_idx += 1
        else:
            vip = None

        if show_loadings:
            # Use x_weights_ as a stable spectral importance proxy
            w = np.asarray(pls.x_weights_, dtype=float)
            ax = axes[ax_idx]
            max_plot = min(3, w.shape[1])
            for k in range(max_plot):
                ax.plot(wavenumbers, w[:, k], linewidth=1.0, label=f"LV{k+1}")
            ax.set_ylabel("X Weights")
            ax.set_title("PLS X Weights (per component)")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            ax.invert_xaxis()

        axes[-1].set_xlabel("Wavenumber (cm⁻¹)")

    # --- Data table ---
    cols = {"label": [str(v) for v in labels]}
    for k in range(scores.shape[1]):
        cols[f"LV{k+1}"] = scores[:, k]
    df_scores = pd.DataFrame(cols)

    # Summary
    summary = (
        f"PLS-DA completed. Components: {n_components}.\n"
        f"Classes: {', '.join(map(str, classes))}.\n"
        f"CV accuracy (logistic on LV scores, {cv_folds}-fold): {cv_acc_mean:.3f}"
    )

    create_logs(__name__, __file__, summary, status="info")

    return {
        "primary_figure": fig1,
        "secondary_figure": fig2,
        "data_table": df_scores,
        "summary_text": summary,
        "detailed_summary": f"Confusion matrix (CV):\n{cm}",
        "raw_results": {
            "classes": classes,
            "cv_accuracy_mean": cv_acc_mean,
            "confusion_matrix": cm,
            "pls_model": pls,
            # Standardized component-viewer payload (used by Results UI)
            "wavenumbers": np.asarray(wavenumbers, dtype=float),
            "component_spectra": np.asarray(getattr(pls, "x_weights_", None), dtype=float)
            if getattr(pls, "x_weights_", None) is not None
            else None,
            "component_prefix": "LV",
            "component_y_label": "X Weights",
            "vip": vip,
            "x_scores": scores,
            "labels": [str(v) for v in labels],
        },
    }


def perform_lda_analysis(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """LDA or PCA→LDA pipeline.

    - If `use_pca_first` is enabled, LDA is fit on PCA scores.
    - The plot shows LD1 vs LD2 when available.
    """

    if progress_callback:
        progress_callback(10)

    scaling_type = params.get("scaling", "StandardScaler")
    use_pca_first = bool(params.get("use_pca_first", True))
    pca_components = int(params.get("pca_components", 10))
    solver = str(params.get("solver", "svd"))
    shrinkage = params.get("shrinkage", None)

    group_labels_map = params.get("_group_labels", None)

    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data, group_labels_map=group_labels_map, method="linear"
    )

    if len(set(labels)) < 2:
        raise ValueError("LDA requires at least 2 groups/classes")

    X_scaled, _scaler = _apply_scaling(X, scaling_type)

    y, classes = _encode_labels(labels)

    if progress_callback:
        progress_callback(35)

    Z = X_scaled
    pca_model = None
    if use_pca_first:
        pca_components = max(2, min(pca_components, Z.shape[0] - 1, Z.shape[1]))
        pca_model = PCA(n_components=pca_components, random_state=0)
        Z = pca_model.fit_transform(Z)

    # Handle shrinkage only for lsqr/eigen solvers
    lda_kwargs: dict[str, Any] = {"solver": solver}
    if solver in {"lsqr", "eigen"}:
        if shrinkage in {"auto", None}:
            lda_kwargs["shrinkage"] = shrinkage
        else:
            try:
                lda_kwargs["shrinkage"] = float(shrinkage)
            except Exception:
                lda_kwargs["shrinkage"] = None

    lda = LinearDiscriminantAnalysis(**lda_kwargs)
    Z_lda = lda.fit_transform(Z, y)

    if progress_callback:
        progress_callback(65)

    # CV confusion matrix (on the same feature space)
    cv_folds = int(params.get("cv_folds", 5))
    cv_folds = max(2, min(cv_folds, int(np.min(np.bincount(y))) if y.size else 2))
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    y_pred_cv = np.full_like(y, fill_value=-1)
    for train_idx, test_idx in skf.split(Z, y):
        lda_cv = LinearDiscriminantAnalysis(**lda_kwargs)
        lda_cv.fit(Z[train_idx], y[train_idx])
        y_pred_cv[test_idx] = lda_cv.predict(Z[test_idx])

    cm = confusion_matrix(y, y_pred_cv, labels=list(range(len(classes))))
    cv_acc = float(np.mean(y_pred_cv == y))

    # --- Plot ---
    fig1, ax1 = plt.subplots(figsize=(9, 7))

    if Z_lda.ndim == 2 and Z_lda.shape[1] >= 2:
        for i, cls in enumerate(classes):
            m = y == i
            ax1.scatter(
                Z_lda[m, 0],
                Z_lda[m, 1],
                s=70,
                alpha=0.85,
                label=str(cls),
                edgecolors="white",
                linewidths=0.8,
            )
        ax1.set_xlabel("LD1")
        ax1.set_ylabel("LD2")
        ax1.set_title("LDA Scores (LD1 vs LD2)")
    else:
        # Binary case often yields 1D
        for i, cls in enumerate(classes):
            m = y == i
            ax1.hist(np.asarray(Z_lda).reshape(-1)[m], bins=20, alpha=0.6, label=str(cls))
        ax1.set_xlabel("LD1")
        ax1.set_ylabel("Count")
        ax1.set_title("LDA Scores (LD1)")

    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    # --- Coefficients plot (if available) ---
    fig2 = None
    try:
        if hasattr(lda, "coef_") and lda.coef_ is not None:
            coef = np.asarray(lda.coef_, dtype=float)
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            if use_pca_first:
                ax2.plot(np.arange(coef.shape[1]) + 1, coef[0], linewidth=1.0)
                ax2.set_xlabel("PC index")
                ax2.set_ylabel("LDA coef")
                ax2.set_title("LDA Coefficients in PCA space")
            else:
                ax2.plot(wavenumbers, coef[0], linewidth=1.0)
                ax2.set_xlabel("Wavenumber (cm⁻¹)")
                ax2.set_ylabel("LDA coef")
                ax2.set_title("LDA Coefficients vs Wavenumber")
                ax2.invert_xaxis()
            ax2.grid(True, alpha=0.25)
    except Exception:
        pass

    # --- Table ---
    cols = {"label": [str(v) for v in labels]}
    Z_flat = np.asarray(Z_lda)
    if Z_flat.ndim == 1:
        Z_flat = Z_flat.reshape(-1, 1)
    for k in range(Z_flat.shape[1]):
        cols[f"LD{k+1}"] = Z_flat[:, k]
    df_scores = pd.DataFrame(cols)

    summary = (
        f"LDA completed ({'PCA→LDA' if use_pca_first else 'LDA'}).\n"
        f"Classes: {', '.join(map(str, classes))}.\n"
        f"CV accuracy ({cv_folds}-fold): {cv_acc:.3f}"
    )

    return {
        "primary_figure": fig1,
        "secondary_figure": fig2,
        "data_table": df_scores,
        "summary_text": summary,
        "detailed_summary": f"Confusion matrix (CV):\n{cm}",
        "raw_results": {
            "classes": classes,
            "cv_accuracy": cv_acc,
            "confusion_matrix": cm,
            "lda_model": lda,
            "pca_model": pca_model,
        },
    }
