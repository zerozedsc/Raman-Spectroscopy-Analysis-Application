"""Quality Control / Validation Methods

Includes:
- Robust outlier detection (MinCovDet Mahalanobis, EllipticEnvelope, IsolationForest)
- Silhouette analysis (k-means validation)

These are intended for Analysis Page exploratory QC.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import warnings

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from sklearn.covariance import EllipticEnvelope, LedoitWolf, MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .exploratory import interpolate_to_common_wavenumbers_with_groups


def _apply_scaling(X: np.ndarray, scaling_type: str) -> np.ndarray:
    scaling_type = str(scaling_type or "StandardScaler")
    if scaling_type == "StandardScaler":
        return StandardScaler().fit_transform(X)
    if scaling_type == "MinMaxScaler":
        return MinMaxScaler().fit_transform(X)
    return X


def perform_outlier_detection(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Outlier detection on spectra with PCA scatter visualization."""

    if progress_callback:
        progress_callback(10)

    method = str(params.get("method", "mahalanobis_mcd"))
    contamination = float(params.get("contamination", 0.05))
    scaling_type = params.get("scaling", "StandardScaler")
    pca_components = int(params.get("pca_components", 5))

    # Performance/stability knobs (safe defaults for high-dimensional spectra)
    detector_pca_components = int(params.get("detector_pca_components", 20))
    mcd_support_fraction = params.get("mcd_support_fraction", 0.75)
    iso_n_estimators = int(params.get("iso_n_estimators", 100))

    group_labels_map = params.get("_group_labels", None)
    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data, group_labels_map=group_labels_map, method="linear"
    )

    X_scaled = _apply_scaling(X, scaling_type)

    if progress_callback:
        progress_callback(35)

    n_samples, n_features = X_scaled.shape
    if n_samples < 3:
        raise ValueError("Outlier detection requires at least 3 spectra")

    # PCA-reduced feature space for detectors (much faster + avoids rank issues)
    # Always reduce in typical Raman shape (many features, fewer samples).
    det_pca = None
    X_det = X_scaled
    try:
        max_det_comp = max(2, min(detector_pca_components, n_samples - 1, n_features))
        # If user sets detector_pca_components <= 0, treat as 'auto'.
        if detector_pca_components <= 0:
            max_det_comp = max(2, min(20, n_samples - 1, n_features))

        if max_det_comp < n_features:
            if progress_callback:
                progress_callback(40)
            det_pca = PCA(n_components=max_det_comp, random_state=0)
            X_det = det_pca.fit_transform(X_scaled)
    except Exception:
        # If PCA reduction fails for any reason, fall back to raw scaled features.
        det_pca = None
        X_det = X_scaled

    if progress_callback:
        progress_callback(45)

    # Fit detector
    scores = None
    is_outlier = None
    warn_msgs: list[str] = []
    used_method = method
    threshold = None

    if method == "mahalanobis_mcd":
        # NOTE: MinCovDet can be very slow / unstable in high-dimensional space.
        # We run it on PCA scores (X_det) and fall back to a shrinkage covariance
        # if it fails.
        try:
            support_fraction = float(mcd_support_fraction)
            support_fraction = float(np.clip(support_fraction, 0.5, 1.0))

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mcd = MinCovDet(support_fraction=support_fraction, random_state=42).fit(X_det)
            warn_msgs = [str(wi.message) for wi in w]

            # squared mahalanobis distances
            d2 = mcd.mahalanobis(X_det)
            scores = d2
            # threshold by percentile to match contamination
            threshold = float(np.quantile(d2, 1.0 - contamination))
            is_outlier = d2 >= threshold
            model = mcd
        except Exception as e:
            # Fast, stable fallback (still Mahalanobis-like)
            used_method = "mahalanobis_shrinkage"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                lw = LedoitWolf().fit(X_det)
            warn_msgs = [str(wi.message) for wi in w]

            centered = X_det - lw.location_
            precision = lw.precision_
            d2 = np.einsum("ij,jk,ik->i", centered, precision, centered)
            scores = d2
            threshold = float(np.quantile(d2, 1.0 - contamination))
            is_outlier = d2 >= threshold
            model = lw
    elif method == "elliptic_envelope":
        # EllipticEnvelope uses a robust covariance estimator internally; use
        # PCA-reduced space to avoid full-rank warnings and speed issues.
        support_fraction = None
        try:
            support_fraction = float(mcd_support_fraction)
            support_fraction = float(np.clip(support_fraction, 0.5, 1.0))
        except Exception:
            support_fraction = None

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ee = EllipticEnvelope(
                contamination=contamination,
                random_state=42,
                support_fraction=support_fraction,
            )
            pred = ee.fit_predict(X_det)  # -1 outlier, 1 inlier
        warn_msgs = [str(wi.message) for wi in w]
        is_outlier = pred < 0
        # decision_function: higher is more normal
        scores = -ee.decision_function(X_det)
        threshold = float(np.quantile(np.asarray(scores, dtype=float), 1.0 - contamination))
        model = ee
    else:
        iso = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=iso_n_estimators,
        )
        pred = iso.fit_predict(X_det)
        is_outlier = pred < 0
        scores = -iso.score_samples(X_det)
        threshold = float(np.quantile(np.asarray(scores, dtype=float), 1.0 - contamination))
        model = iso

    if progress_callback:
        progress_callback(60)

    # PCA for plotting
    pca_components = max(2, min(pca_components, X_scaled.shape[0] - 1, X_scaled.shape[1]))
    pca = PCA(n_components=pca_components, random_state=0)
    Z = pca.fit_transform(X_scaled)

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    in_m = ~is_outlier
    out_m = is_outlier

    ax1.scatter(Z[in_m, 0], Z[in_m, 1], s=55, alpha=0.75, label="Inlier", edgecolors="white", linewidths=0.6)
    ax1.scatter(Z[out_m, 0], Z[out_m, 1], s=70, alpha=0.9, label="Outlier", color="red", edgecolors="white", linewidths=0.6)
    ax1.set_title("Outlier Detection (PCA space)")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    # Secondary figure: scores + mean spectra (inliers vs outliers)
    fig2, (ax2, ax3) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        gridspec_kw={"height_ratios": [1.0, 1.2], "hspace": 0.35},
    )

    s_arr = np.asarray(scores, dtype=float)
    ax2.plot(np.arange(len(s_arr)), s_arr, linewidth=1.0, color="#111111", alpha=0.9)
    ax2.scatter(np.where(out_m)[0], s_arr[out_m], color="#dc3545", s=30, label="Outlier")
    if threshold is not None and np.isfinite(threshold):
        ax2.axhline(float(threshold), color="#dc3545", linestyle="--", linewidth=1.0, alpha=0.6, label="Threshold")
    ax2.set_title("Outlier Scores", fontweight="bold")
    ax2.set_xlabel("Spectrum index")
    ax2.set_ylabel("Score (higher = more outlier-like)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=9)

    # Mean spectra comparison
    try:
        Xin = np.asarray(X, dtype=float)[in_m, :]
        Xout = np.asarray(X, dtype=float)[out_m, :]
        wn = np.asarray(wavenumbers, dtype=float).reshape(-1)

        if Xin.size:
            mu_in = np.nanmean(Xin, axis=0)
            sd_in = np.nanstd(Xin, axis=0)
            ax3.plot(wn, mu_in, color="#1f77b4", linewidth=1.8, label=f"Inlier mean (n={int(np.sum(in_m))})")
            ax3.fill_between(wn, mu_in - sd_in, mu_in + sd_in, color="#1f77b4", alpha=0.18)

        if Xout.size:
            mu_out = np.nanmean(Xout, axis=0)
            sd_out = np.nanstd(Xout, axis=0)
            ax3.plot(wn, mu_out, color="#dc3545", linewidth=1.8, label=f"Outlier mean (n={int(np.sum(out_m))})")
            ax3.fill_between(wn, mu_out - sd_out, mu_out + sd_out, color="#dc3545", alpha=0.18)

        ax3.set_title("Mean Spectra: Inliers vs Outliers", fontweight="bold")
        ax3.set_xlabel("Wavenumber (cm⁻¹)")
        ax3.set_ylabel("Intensity")
        ax3.grid(True, alpha=0.25)
        ax3.legend(loc="best", fontsize=9)
    except Exception:
        ax3.text(
            0.5,
            0.5,
            "Mean spectra comparison unavailable",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.axis("off")

    # Dataset extraction for easier filtering in the table
    label_strs = [str(v) for v in labels]
    datasets = [s.rsplit("_", 1)[0] if "_" in s else s for s in label_strs]

    df = pd.DataFrame(
        {
            "dataset": datasets,
            "label": label_strs,
            "outlier": is_outlier.astype(bool),
            "score": s_arr,
            "pc1": Z[:, 0],
            "pc2": Z[:, 1],
        }
    )

    n_out = int(np.sum(is_outlier))
    n_in = int(np.sum(~is_outlier))

    # Extreme outliers (top-N by score)
    extreme_n = int(params.get("extreme_n", 10))
    extreme_n = int(np.clip(extreme_n, 1, max(1, len(label_strs))))
    extreme_idx = list(np.argsort(-s_arr)[:extreme_n])
    extreme_items = [
        {
            "rank": int(i + 1),
            "index": int(idx),
            "label": label_strs[idx],
            "score": float(s_arr[idx]),
            "outlier": bool(is_outlier[idx]),
        }
        for i, idx in enumerate(extreme_idx)
    ]

    # Best-effort silhouette diagnostic: does removing outliers increase clusterability?
    sil_full = None
    sil_inliers = None
    try:
        if n_samples >= 4:
            km = KMeans(n_clusters=2, random_state=0, n_init=10)
            lab_full = km.fit_predict(Z[:, :2])
            sil_full = float(silhouette_score(Z[:, :2], lab_full))
        if n_in >= 4:
            km2 = KMeans(n_clusters=2, random_state=0, n_init=10)
            lab_in = km2.fit_predict(Z[in_m, :2])
            sil_inliers = float(silhouette_score(Z[in_m, :2], lab_in))
    except Exception:
        sil_full = None
        sil_inliers = None

    summary = (
        f"Outlier detection completed. Method: {used_method}. "
        f"Outliers: {n_out}/{len(labels)} (contamination={contamination})."
    )
    if threshold is not None and np.isfinite(threshold):
        summary += f" Threshold: {float(threshold):.4g}."

    detailed_summary = ""
    if warn_msgs:
        # Keep it compact: show only the first few unique warnings
        uniq = []
        for m in warn_msgs:
            if m not in uniq:
                uniq.append(m)
        shown = uniq[:3]
        detailed_summary += "Warnings during fitting (first 3):\n"
        for m in shown:
            detailed_summary += f"- {m}\n"

    # Add quantitative stats
    try:
        detailed_summary += "\nScore statistics:\n"
        detailed_summary += f"- Inliers: n={n_in}\n"
        detailed_summary += f"- Outliers: n={n_out}\n"
        detailed_summary += f"- Score min/median/max: {float(np.nanmin(s_arr)):.4g} / {float(np.nanmedian(s_arr)):.4g} / {float(np.nanmax(s_arr)):.4g}\n"
        if threshold is not None and np.isfinite(threshold):
            detailed_summary += f"- Threshold (quantile 1-contamination): {float(threshold):.4g}\n"
        if sil_full is not None:
            detailed_summary += f"\nSilhouette (k=2 on PCA scatter): full={sil_full:.3f}"
            if sil_inliers is not None:
                detailed_summary += f", inliers-only={sil_inliers:.3f}"
            detailed_summary += "\n"

        detailed_summary += "\nExtreme scores (top):\n"
        for it in extreme_items[:10]:
            flag = "OUT" if it["outlier"] else "IN"
            detailed_summary += f"- #{it['rank']} idx={it['index']} score={it['score']:.4g} [{flag}] {it['label']}\n"
    except Exception:
        pass

    return {
        "primary_figure": fig1,
        "secondary_figure": fig2,
        "data_table": df,
        "summary_text": summary,
        "detailed_summary": detailed_summary,
        "raw_results": {
            "detector": model,
            "pca_model": pca,
            "detector_pca_model": det_pca,
            "fit_warnings": warn_msgs,
            "used_method": used_method,
            "threshold": threshold,
            "scores": s_arr,
            "labels": label_strs,
            "outlier_mask": is_outlier.astype(bool),
            "extreme": extreme_items,
            "silhouette_full": sil_full,
            "silhouette_inliers": sil_inliers,
        },
    }


def perform_silhouette_analysis(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Compute silhouette score over k range with k-means."""

    if progress_callback:
        progress_callback(10)

    k_min = int(params.get("k_min", 2))
    k_max = int(params.get("k_max", 10))
    scaling_type = params.get("scaling", "StandardScaler")
    pca_components = int(params.get("pca_components", 10))

    group_labels_map = params.get("_group_labels", None)
    _wn, X, _labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data, group_labels_map=group_labels_map, method="linear"
    )

    X_scaled = _apply_scaling(X, scaling_type)

    # Optional PCA to denoise
    if pca_components and pca_components > 0:
        pca_components = max(2, min(pca_components, X_scaled.shape[0] - 1, X_scaled.shape[1]))
        Z = PCA(n_components=pca_components, random_state=0).fit_transform(X_scaled)
    else:
        Z = X_scaled

    if progress_callback:
        progress_callback(35)

    k_min = max(2, k_min)
    k_max = max(k_min, k_max)
    k_max = min(k_max, Z.shape[0] - 1) if Z.shape[0] > 2 else k_min

    ks: list[int] = []
    scores: list[float] = []

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels_k = km.fit_predict(Z)
        # silhouette needs at least 2 clusters and less than n_samples
        s = float(silhouette_score(Z, labels_k))
        ks.append(k)
        scores.append(s)

    if progress_callback:
        progress_callback(80)

    best_idx = int(np.argmax(scores)) if scores else 0
    best_k = ks[best_idx] if ks else k_min
    best_s = scores[best_idx] if scores else float("nan")

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ks, scores, marker="o", linewidth=1.5)
    ax1.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"Best k={best_k}")
    ax1.set_title("Silhouette Score vs Number of Clusters")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Silhouette score")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    df = pd.DataFrame({"k": ks, "silhouette": scores})

    summary = f"Silhouette analysis completed. Best k={best_k} (score={best_s:.3f})."

    return {
        "primary_figure": fig1,
        "secondary_figure": None,
        "data_table": df,
        "summary_text": summary,
        "detailed_summary": "",
        "raw_results": {"best_k": best_k, "best_score": best_s},
    }
