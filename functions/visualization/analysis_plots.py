"""
Analysis Visualization Methods

This module implements various visualization methods for Raman spectral analysis,
including heatmaps, overlay plots, waterfall plots, correlation heatmaps, and scatter plots.

These functions are designed for use in the Analysis Page and can also be used
standalone for custom analysis workflows.

Created: November 21, 2025
Purpose: Consolidate analysis-specific visualizations in functions/visualization
         package for better code organization and reusability.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.signal import find_peaks


def _as_float_1d(a: Any) -> np.ndarray:
    """Convert an array-like to a 1D float numpy array."""
    out = np.asarray(a, dtype=float).reshape(-1)
    return out


def _ensure_ascending_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensure x is strictly ascending for interpolation, keeping y aligned."""
    if x.size == 0:
        return x, y
    if x[0] > x[-1]:
        x = x[::-1]
        y = y[::-1]

    # Remove duplicate x values (np.interp requires increasing x)
    x_unique, unique_idx = np.unique(x, return_index=True)
    y_unique = y[unique_idx]
    return x_unique, y_unique


def _compute_common_wavenumber_grid(
    dataset_data: Dict[str, pd.DataFrame],
    *,
    max_points: Optional[int] = None,
) -> np.ndarray:
    """Compute a common overlapping wavenumber grid across datasets.

    Uses the intersection of ranges to avoid extrapolation, and a representative
    step based on the median spacing across datasets.
    """
    if not dataset_data:
        return np.array([], dtype=float)

    w_arrays: list[np.ndarray] = []
    steps: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []

    for df in dataset_data.values():
        w = _as_float_1d(df.index.values)
        if w.size < 2:
            continue
        # Work in ascending order for robust range/step calculations
        if w[0] > w[-1]:
            w = w[::-1]
        w_arrays.append(w)
        mins.append(float(np.min(w)))
        maxs.append(float(np.max(w)))
        diffs = np.diff(w)
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size:
            steps.append(float(np.median(np.abs(diffs))))

    if not w_arrays:
        # Fallback: first dataset index
        first_df = next(iter(dataset_data.values()))
        return _as_float_1d(first_df.index.values)

    start = max(mins)
    end = min(maxs)
    if not np.isfinite(start) or not np.isfinite(end) or start >= end:
        # No overlap: fall back to first dataset index to avoid hard crash.
        first_df = next(iter(dataset_data.values()))
        return _as_float_1d(first_df.index.values)

    step = float(np.median(steps)) if steps else None
    if step is None or (not np.isfinite(step)) or step <= 0:
        # Conservative fallback: use first dataset's spacing if available
        w0 = w_arrays[0]
        diffs0 = np.diff(w0)
        step = float(np.median(np.abs(diffs0))) if diffs0.size else 1.0

    # Inclusive grid (make sure end is included if possible)
    n = int(np.floor((end - start) / step)) + 1
    if n < 2:
        # Very narrow overlap, just return two points
        grid = np.array([start, end], dtype=float)
    else:
        grid = start + step * np.arange(n, dtype=float)
        # Clip to end due to floating error
        grid = grid[grid <= end + 1e-9]

    if max_points is not None and max_points > 1 and grid.size > max_points:
        idx = np.linspace(0, grid.size - 1, max_points, dtype=int)
        grid = grid[idx]

    return grid


def _resample_dataframe_to_grid(df: pd.DataFrame, w_target: np.ndarray) -> pd.DataFrame:
    """Resample a (wavenumber-indexed) spectra DataFrame to a target grid."""
    w_target = _as_float_1d(w_target)
    if w_target.size == 0:
        return df

    w_src = _as_float_1d(df.index.values)
    out = {}
    for col in df.columns:
        y = _as_float_1d(df[col].values)
        x_u, y_u = _ensure_ascending_xy(w_src, y)
        if x_u.size < 2:
            # Degenerate axis; keep constant if possible
            out[col] = np.full_like(w_target, fill_value=float(y_u[0]) if y_u.size else np.nan)
            continue
        # Ensure target sits within source range to avoid extrapolation
        w_min, w_max = float(x_u[0]), float(x_u[-1])
        w_clipped = np.clip(w_target, w_min, w_max)
        out[col] = np.interp(w_clipped, x_u, y_u)

    resampled = pd.DataFrame(out, index=w_target)
    return resampled


def create_spectral_heatmap(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Create heatmap visualization of spectral data.

    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - cluster_rows: Cluster rows (spectra) (default True)
            - cluster_cols: Cluster columns (wavenumbers) (default False)
            - colormap: Colormap name (default 'viridis')
            - normalize: Normalize spectra (default True)
            - show_dendrograms: Show dendrograms (default True)
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with heatmap visualization
    """
    if progress_callback:
        progress_callback(10)

    # Get parameters
    cluster_rows = params.get("cluster_rows", True)
    cluster_cols = params.get("cluster_cols", False)
    colormap = params.get("colormap", "viridis")
    normalize = params.get("normalize", True)
    show_dendrograms = params.get("show_dendrograms", True)

    # Optional: cap the number of wavenumbers for performance
    max_wavenumbers = params.get("max_wavenumbers", None)

    # Combine all datasets
    all_spectra = []
    labels = []

    wavenumbers = _compute_common_wavenumber_grid(dataset_data, max_points=max_wavenumbers)

    for dataset_name, df in dataset_data.items():
        df_rs = _resample_dataframe_to_grid(df, wavenumbers)
        for col in df_rs.columns:
            all_spectra.append(df_rs[col].values)
            labels.append(f"{dataset_name}_{col}")

    # Create data matrix (rows=spectra, cols=wavenumbers)
    data_matrix = np.asarray(all_spectra, dtype=float)

    if progress_callback:
        progress_callback(30)

    # Normalize if requested
    if normalize:
        # P0-4 FIX: Add epsilon guard to prevent division by zero for flat spectra
        # Row-wise normalization (each spectrum)
        min_vals = data_matrix.min(axis=1, keepdims=True)
        max_vals = data_matrix.max(axis=1, keepdims=True)
        denom = max_vals - min_vals
        denom[denom < 1e-12] = 1.0  # Guard against division by zero
        data_matrix = (data_matrix - min_vals) / denom

    if progress_callback:
        progress_callback(50)

    # Perform clustering
    row_linkage = None
    col_linkage = None
    row_order = np.arange(data_matrix.shape[0])
    col_order = np.arange(data_matrix.shape[1])

    if cluster_rows:
        row_linkage = linkage(data_matrix, method="average", metric="euclidean")
        row_dendrogram = dendrogram(row_linkage, no_plot=True)
        row_order = row_dendrogram["leaves"]

    if cluster_cols:
        col_linkage = linkage(data_matrix.T, method="average", metric="euclidean")
        col_dendrogram = dendrogram(col_linkage, no_plot=True)
        col_order = col_dendrogram["leaves"]

    # Reorder data
    data_ordered = data_matrix[row_order, :][:, col_order]
    labels_ordered = [labels[i] for i in row_order]

    if progress_callback:
        progress_callback(70)

    # Create figure with optional dendrograms
    if show_dendrograms and (cluster_rows or cluster_cols):
        fig = plt.figure(figsize=(14, 10))

        # Create grid
        if cluster_rows and cluster_cols:
            gs = fig.add_gridspec(
                2,
                2,
                width_ratios=[0.2, 1],
                height_ratios=[0.2, 1],
                hspace=0.05,
                wspace=0.05,
            )
            ax_heatmap = fig.add_subplot(gs[1, 1])
            ax_dend_row = fig.add_subplot(gs[1, 0])
            ax_dend_col = fig.add_subplot(gs[0, 1])
        elif cluster_rows:
            gs = fig.add_gridspec(1, 2, width_ratios=[0.2, 1], wspace=0.05)
            ax_heatmap = fig.add_subplot(gs[0, 1])
            ax_dend_row = fig.add_subplot(gs[0, 0])
        elif cluster_cols:
            gs = fig.add_gridspec(2, 1, height_ratios=[0.2, 1], hspace=0.05)
            ax_heatmap = fig.add_subplot(gs[1, 0])
            ax_dend_col = fig.add_subplot(gs[0, 0])
        else:
            ax_heatmap = fig.add_subplot(111)

        # Plot dendrograms
        if cluster_rows and show_dendrograms:
            # ✅ USER FIX: Add meaningful labels (dataset names) and colored branches
            # Extract just dataset names (remove spectrum numbers) for cleaner labels
            dataset_only_labels = [label.rsplit('_', 1)[0] if '_' in label else label for label in labels_ordered]
            
            # Create dendrogram with colored branches based on clustering
            dend_result = dendrogram(
                row_linkage,
                ax=ax_dend_row,
                orientation="left",
                labels=dataset_only_labels,  # ✅ Show actual dataset names
                color_threshold=row_linkage[:, 2].max() * 0.7,  # Color clusters
                above_threshold_color='#808080',  # Gray for high-level branches
            )
            ax_dend_row.set_xticks([])
            # ✅ Keep y-axis labels visible so users can see dataset names
            ax_dend_row.tick_params(axis='y', labelsize=8)  # Smaller font for many labels
            ax_dend_row.spines["top"].set_visible(False)
            ax_dend_row.spines["right"].set_visible(False)
            ax_dend_row.spines["bottom"].set_visible(False)
            ax_dend_row.spines["left"].set_visible(False)

        if cluster_cols and show_dendrograms:
            dendrogram(
                col_linkage,
                ax=ax_dend_col,
                orientation="top",
                no_labels=True,
                color_threshold=0,
            )
            ax_dend_col.set_xticks([])
            ax_dend_col.set_yticks([])
            ax_dend_col.spines["top"].set_visible(False)
            ax_dend_col.spines["right"].set_visible(False)
            ax_dend_col.spines["bottom"].set_visible(False)
            ax_dend_col.spines["left"].set_visible(False)
    else:
        fig, ax_heatmap = plt.subplots(figsize=(12, 8))

    # Plot heatmap
    im = ax_heatmap.imshow(
        data_ordered, aspect="auto", cmap=colormap, interpolation="nearest"
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap)
    cbar.set_label("Normalized Intensity" if normalize else "Intensity", fontsize=12)

    # Labels
    ax_heatmap.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax_heatmap.set_ylabel("Spectrum Index", fontsize=12)
    # ✅ USER FIX: Add more descriptive title explaining what hierarchical clustering shows
    title = "Hierarchical Clustering of Raman Spectra"
    if cluster_rows:
        title += " (Rows: Spectra Similarity)"
    if cluster_cols:
        title += " + (Cols: Wavenumber Patterns)"
    ax_heatmap.set_title(title, fontsize=14, fontweight="bold", pad=15)

    if progress_callback:
        progress_callback(90)

    # Summary text
    summary = f"Spectral heatmap created from {len(dataset_data)} dataset(s).\n"
    summary += f"Matrix: {data_matrix.shape[0]} spectra × {data_matrix.shape[1]} wavenumbers.\n"
    if cluster_rows:
        summary += "✓ Rows (spectra) clustered by similarity.\n"
    if cluster_cols:
        summary += "✓ Columns (wavenumbers) clustered by pattern.\n"
    if normalize:
        summary += "✓ Data normalized (0-1 range per spectrum).\n"
    
    # ✅ USER FIX: Add explanation of what hierarchical clustering shows
    summary += "\n💡 Dendrogram Interpretation:\n"
    summary += "  • Similar colors = grouped by clustering algorithm\n"
    summary += "  • Branch height = dissimilarity measure\n"
    summary += "  • Labels show dataset names for identification\n"

    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": None,
        "summary_text": summary,
        "detailed_summary": f"Matrix shape: {data_matrix.shape}",
        "raw_results": {
            "data_matrix": data_ordered,
            "labels": labels_ordered,
            "row_order": row_order,
            "col_order": col_order,
            "wavenumbers": wavenumbers[col_order] if wavenumbers.size else wavenumbers,
        },
    }


def create_mean_spectra_overlay(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Create overlay plot of mean spectra from multiple datasets.

    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - show_std: Show standard deviation bands (default True)
            - show_individual: Show individual spectra (default False)
            - alpha_individual: Alpha for individual spectra (default 0.1)
            - normalize: Normalize spectra (default False)
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with overlay plot
    """
    if progress_callback:
        progress_callback(10)

    # Get parameters
    show_std = params.get("show_std", True)
    show_individual = params.get("show_individual", False)
    alpha_individual = params.get("alpha_individual", 0.1)
    normalize = params.get("normalize", False)

    max_wavenumbers = params.get("max_wavenumbers", None)
    wavenumbers = _compute_common_wavenumber_grid(dataset_data, max_points=max_wavenumbers)

    if progress_callback:
        progress_callback(30)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_data)))

    for i, (dataset_name, df) in enumerate(dataset_data.items()):
        df_rs = _resample_dataframe_to_grid(df, wavenumbers)
        data = df_rs.values

        # Normalize if requested
        if normalize:
            # P0-4 FIX: Add epsilon guard to prevent division by zero for flat spectra
            min_vals = data.min(axis=0, keepdims=True)
            max_vals = data.max(axis=0, keepdims=True)
            denom = max_vals - min_vals
            denom[denom < 1e-12] = 1.0  # Guard against division by zero
            data = (data - min_vals) / denom

        # Calculate mean and std
        mean_spectrum = data.mean(axis=1)
        std_spectrum = data.std(axis=1)

        # Plot mean
        ax.plot(
            wavenumbers, mean_spectrum, label=dataset_name, color=colors[i], linewidth=2
        )

        # Plot std bands
        if show_std:
            ax.fill_between(
                wavenumbers,
                mean_spectrum - std_spectrum,
                mean_spectrum + std_spectrum,
                alpha=0.2,
                color=colors[i],
            )

        # Plot individual spectra
        if show_individual:
            for j in range(data.shape[1]):
                ax.plot(
                    wavenumbers,
                    data[:, j],
                    color=colors[i],
                    alpha=alpha_individual,
                    linewidth=0.5,
                )

    if progress_callback:
        progress_callback(80)

    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax.set_ylabel("Intensity", fontsize=12)
    ax.set_title("Mean Spectra Overlay", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    if progress_callback:
        progress_callback(90)

    summary = f"Overlay plot created for {len(dataset_data)} dataset(s).\n"
    total_spectra = sum(df.shape[1] for df in dataset_data.values())
    summary += f"Total spectra: {total_spectra}"
    if normalize:
        summary += "\nData normalized."

    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": None,
        "summary_text": summary,
        "detailed_summary": f"Datasets: {', '.join(dataset_data.keys())}",
        "raw_results": {"wavenumbers": wavenumbers},
    }


def create_waterfall_plot(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Create waterfall plot of spectra with vertical offset (2D or 3D).

    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - use_3d: Use 3D visualization (default False)
            - offset_scale: Vertical offset scale (default 1.0)
            - max_spectra: Maximum number of spectra to plot (default 50)
            - colormap: Colormap for gradient (default 'viridis')
            - show_grid: Show grid lines (default True)
            - reverse_order: Reverse plotting order (default False)
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with waterfall plot
    """
    if progress_callback:
        progress_callback(10)

    # Get parameters
    use_3d = params.get("use_3d", False)
    offset_scale = params.get("offset_scale", 1.0)
    max_spectra = params.get("max_spectra", 50)
    colormap = params.get("colormap", "viridis")
    show_grid = params.get("show_grid", True)
    reverse_order = params.get("reverse_order", False)

    # Combine all datasets
    all_spectra = []
    labels = []

    for dataset_name, df in dataset_data.items():
        for col in df.columns:
            all_spectra.append(df[col].values)
            labels.append(f"{dataset_name}_{col}")

    # Limit number of spectra
    if len(all_spectra) > max_spectra:
        # Sample evenly
        indices = np.linspace(0, len(all_spectra) - 1, max_spectra, dtype=int)
        all_spectra = [all_spectra[i] for i in indices]
        labels = [labels[i] for i in indices]

    wavenumbers = dataset_data[list(dataset_data.keys())[0]].index.values

    if progress_callback:
        progress_callback(40)

    # Color gradient
    cmap = plt.cm.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, len(all_spectra)))

    if use_3d:
        # === 3D WATERFALL PLOT ===
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Normalize spectra for consistent height
        all_spectra_normalized = []
        global_min = min(np.min(s) for s in all_spectra)
        global_max = max(np.max(s) for s in all_spectra)
        range_val = global_max - global_min

        for spectrum in all_spectra:
            normalized = (
                (spectrum - global_min) / range_val if range_val > 0 else spectrum
            )
            all_spectra_normalized.append(normalized)

        # Plot each spectrum as a 3D line with filled polygon
        plot_order = range(len(all_spectra_normalized))
        if reverse_order:
            plot_order = list(reversed(plot_order))

        for i in plot_order:
            spectrum = all_spectra_normalized[i]
            y_pos = i * offset_scale

            # Plot the spectrum line
            ax.plot(
                wavenumbers,
                [y_pos] * len(wavenumbers),
                spectrum,
                color=colors[i],
                linewidth=1.0,
                alpha=0.9,
            )

            # Add filled polygon underneath for visibility
            verts = [(wavenumbers[0], y_pos, 0)]
            for j, (wn, z) in enumerate(zip(wavenumbers, spectrum)):
                verts.append((wn, y_pos, z))
            verts.append((wavenumbers[-1], y_pos, 0))

            poly = Poly3DCollection(
                [verts], alpha=0.3, facecolor=colors[i], edgecolor="none"
            )
            ax.add_collection3d(poly)

        # Axis labels and styling
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_ylabel("Spectrum Index", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_zlabel(
            "Intensity (normalized)", fontsize=12, fontweight="bold", labelpad=10
        )
        ax.set_title("3D Waterfall Plot", fontsize=14, fontweight="bold")

        # Invert x-axis (Raman convention)
        ax.set_xlim(wavenumbers.max(), wavenumbers.min())
        ax.set_ylim(0, len(all_spectra_normalized) * offset_scale)
        ax.set_zlim(0, 1.1)

        # Adjust viewing angle for better visibility
        ax.view_init(elev=25, azim=45)

        if show_grid:
            ax.grid(True, alpha=0.3)
    else:
        # === 2D WATERFALL PLOT (Original) ===
        fig, ax = plt.subplots(figsize=(12, 10))

        # Calculate offset
        max_intensity = max(np.max(spec) for spec in all_spectra)
        offset = max_intensity * offset_scale

        # Plot spectra
        plot_order = range(len(all_spectra))
        if reverse_order:
            plot_order = reversed(plot_order)

        for i in plot_order:
            spectrum = all_spectra[i]
            y_offset = i * offset
            ax.plot(
                wavenumbers,
                spectrum + y_offset,
                color=colors[i],
                linewidth=1.0,
                alpha=0.8,
            )

            # Optional: fill under curve for better visibility
            ax.fill_between(
                wavenumbers, y_offset, spectrum + y_offset, color=colors[i], alpha=0.15
            )

        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
        ax.set_ylabel("Intensity (offset)", fontsize=12)
        ax.set_title("Waterfall Plot", fontsize=14, fontweight="bold")

        if show_grid:
            ax.grid(True, alpha=0.3, axis="x")

        ax.invert_xaxis()
        ax.set_yticks([])  # Remove y-ticks (offsets are arbitrary)

    if progress_callback:
        progress_callback(90)

    summary = f"Waterfall plot created with {len(all_spectra)} spectra.\n"
    summary += f"Mode: {'3D' if use_3d else '2D'}\n"
    summary += f"Offset scale: {offset_scale}\n"
    summary += f"Colormap: {colormap}"

    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": None,
        "summary_text": summary,
        "detailed_summary": f"Total spectra plotted: {len(all_spectra)}",
        "raw_results": {"n_spectra": len(all_spectra), "use_3d": use_3d},
        "loadings_figure": None  # Waterfall plot does not produce loadings
    }


def create_correlation_heatmap(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Create correlation heatmap between spectral regions.

    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - method: Correlation method ('pearson', 'spearman') (default 'pearson')
            - colormap: Colormap (default 'coolwarm')
            - cluster: Apply clustering (default True)
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with correlation heatmap
    """
    if progress_callback:
        progress_callback(10)

    # Get parameters
    method = params.get("method", "pearson")
    colormap = params.get("colormap", "coolwarm")
    cluster = params.get("cluster", True)
    max_wavenumbers = params.get("max_wavenumbers", None)

    # P1-4 FIX: Handle multi-dataset input properly (and differing wavenumber grids)
    multi_dataset_warning = ""
    wavenumbers = _compute_common_wavenumber_grid(dataset_data, max_points=max_wavenumbers)

    if len(dataset_data) > 1:
        # Resample each dataset to the common grid, then concatenate along columns (spectra)
        all_dfs = []
        for ds_name, ds_df in dataset_data.items():
            all_dfs.append(_resample_dataframe_to_grid(ds_df, wavenumbers))
        df = pd.concat(all_dfs, axis=1)
        dataset_name = f"{len(dataset_data)} datasets (concatenated)"
        multi_dataset_warning = f"\n⚠️ Note: {len(dataset_data)} datasets were concatenated for correlation analysis."
    else:
        dataset_name = list(dataset_data.keys())[0]
        df = _resample_dataframe_to_grid(dataset_data[dataset_name], wavenumbers)

    if progress_callback:
        progress_callback(30)

    # Calculate correlation between wavenumbers
    # Transpose so wavenumbers are rows
    corr_matrix = df.T.corr(method=method)

    if progress_callback:
        progress_callback(60)

    # Clustering
    if cluster:
        linkage_matrix = linkage(corr_matrix.values, method="average")
        dend = dendrogram(linkage_matrix, no_plot=True)
        order = dend["leaves"]
        corr_matrix_ordered = corr_matrix.iloc[order, order]
    else:
        corr_matrix_ordered = corr_matrix

    # ✅ FIX: Dynamic figure sizing based on matrix size (user requested tight fit)
    n_features = corr_matrix_ordered.shape[0]
    # Scale figure size with matrix size, but cap at reasonable limits
    fig_size = min(12, max(8, n_features * 0.05))  # Between 8 and 12 inches
    
    # Create figure with dynamic sizing
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    im = ax.imshow(
        corr_matrix_ordered.values, cmap=colormap, aspect="auto", vmin=-1, vmax=1
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation Coefficient", fontsize=12)

    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax.set_ylabel("Wavenumber (cm⁻¹)", fontsize=12)
    title_method = "Pearson" if str(method).lower() == "pearson" else "Spearman" if str(method).lower() == "spearman" else str(method)
    ax.set_title(
        f"Wavenumber–Wavenumber Correlation Map ({title_method})",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    # Show a readable subset of tick labels (full matrix can have hundreds of variables)
    try:
        wn = np.asarray(corr_matrix_ordered.index, dtype=float)
        n = int(wn.size)
        n_ticks = 10 if n >= 10 else n
        tick_idx = np.linspace(0, max(0, n - 1), num=max(2, n_ticks), dtype=int)
        tick_idx = np.unique(tick_idx)
        ax.set_xticks(tick_idx)
        ax.set_yticks(tick_idx)
        ax.set_xticklabels([f"{wn[i]:.0f}" for i in tick_idx], rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels([f"{wn[i]:.0f}" for i in tick_idx], fontsize=9)
    except Exception:
        pass

    # Give labels room (especially when embedded with a colorbar)
    try:
        fig.subplots_adjust(left=0.14, bottom=0.18, right=0.88, top=0.92)
    except Exception:
        pass

    if progress_callback:
        progress_callback(90)

    summary = f"Correlation heatmap created using {method} method.\n"
    summary += f"Dataset: {dataset_name}\n"
    summary += f"Wavenumber range: {df.shape[0]} points"
    summary += multi_dataset_warning

    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": corr_matrix_ordered,
        "summary_text": summary,
        "detailed_summary": f"Clustering: {cluster}",
        "raw_results": {
            "correlation_matrix": corr_matrix_ordered.values,
            "wavenumbers": np.asarray(corr_matrix_ordered.index, dtype=float),
        },
    }


def create_peak_scatter(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Create scatter plot of peak intensities across datasets with statistical annotations.

    Supports both 2D (2 peaks) and 3D (3 peaks) visualization modes with comprehensive
    statistics for research reference including mean, std, min, max, and CV%.

    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - peak_1_position: First peak wavenumber position (cm⁻¹)
            - peak_2_position: Second peak wavenumber position (cm⁻¹)
            - peak_3_position: Third peak wavenumber position (cm⁻¹)
            - tolerance: Tolerance for peak matching (default 10 cm⁻¹)
            - use_3d: Enable 3D scatter plot mode
            - show_statistics: Show statistical annotations
            - show_legend: Show legend on plot
            - colormap: Color scheme for datasets
            - marker_size: Size of scatter markers
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with peak scatter plot, statistics table, and detailed summary
    """
    if progress_callback:
        progress_callback(5)

    # Get parameters with defaults
    peak_1_pos = params.get("peak_1_position", 1000)
    peak_2_pos = params.get("peak_2_position", 1650)
    peak_3_pos = params.get("peak_3_position", 2900)
    tolerance = params.get("tolerance", 10)
    use_3d = params.get("use_3d", False)
    show_statistics = params.get("show_statistics", True)
    show_legend = params.get("show_legend", True)
    colormap_name = params.get("colormap", "tab10")
    marker_size = params.get("marker_size", 60)

    # Define peak positions based on mode
    if use_3d:
        peak_positions = [peak_1_pos, peak_2_pos, peak_3_pos]
    else:
        peak_positions = [peak_1_pos, peak_2_pos]

    if progress_callback:
        progress_callback(15)

    # IMPORTANT: datasets may have different wavenumber axes (grouped mode).
    # Resample onto a common overlapping grid to prevent boolean-index mismatch
    # and ensure peak extraction is comparable across datasets.
    wavenumbers = _compute_common_wavenumber_grid(dataset_data)
    dataset_data_rs: Dict[str, pd.DataFrame] = {
        name: _resample_dataframe_to_grid(df, wavenumbers)
        for name, df in dataset_data.items()
    }

    # Validate peak positions are within wavenumber range
    wn_min, wn_max = float(np.nanmin(wavenumbers)), float(np.nanmax(wavenumbers))
    valid_peaks = []
    warning_messages: list[str] = []

    for i, peak_pos in enumerate(peak_positions, 1):
        if peak_pos < wn_min or peak_pos > wn_max:
            warning_messages.append(
                f"Peak {i} ({peak_pos} cm⁻¹) is outside data range [{wn_min:.0f}, {wn_max:.0f}]"
            )
        else:
            valid_peaks.append(peak_pos)

    min_required_peaks = 3 if use_3d else 2
    if len(valid_peaks) < min_required_peaks:
        # Not enough valid peaks - return error info
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            f"Error: Not enough valid peaks\n\n{chr(10).join(warning_messages)}",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#ffdddd", edgecolor="red"),
        )
        ax.axis("off")

        return {
            "primary_figure": fig,
            "secondary_figure": None,
            "data_table": pd.DataFrame(),
            "summary_text": f"Error: {'; '.join(warning_messages)}",
            "detailed_summary": "Please adjust peak positions to be within the wavenumber range.",
            "raw_results": {"errors": warning_messages},
        }

    peak_positions = valid_peaks

    if progress_callback:
        progress_callback(25)

    # Extract peak intensities from all datasets
    peak_data = []

    for dataset_name, df in dataset_data_rs.items():
        for col in df.columns:
            spectrum = df[col].values

            intensities = {}
            for peak_pos in peak_positions:
                # Find closest wavenumber within tolerance
                idx = np.argmin(np.abs(wavenumbers - peak_pos))
                actual_wn = wavenumbers[idx]

                if np.abs(actual_wn - peak_pos) <= tolerance:
                    # Get local maximum within tolerance window for better peak detection
                    wn_mask = np.abs(wavenumbers - peak_pos) <= tolerance
                    # NOTE: spectrum is aligned to wavenumbers after resampling above
                    local_intensities = spectrum[wn_mask]
                    if len(local_intensities) > 0:
                        # Use maximum intensity in the tolerance window
                        max_idx = np.argmax(local_intensities)
                        peak_intensity = local_intensities[max_idx]
                        intensities[f"Peak_{peak_pos:.0f}"] = peak_intensity
                    else:
                        intensities[f"Peak_{peak_pos:.0f}"] = np.nan
                else:
                    intensities[f"Peak_{peak_pos:.0f}"] = np.nan

            peak_data.append(
                {"dataset": dataset_name, "spectrum_id": col, **intensities}
            )

    peak_df = pd.DataFrame(peak_data)

    # Remove rows with NaN values (spectra where peaks couldn't be found)
    valid_peak_cols = [f"Peak_{p:.0f}" for p in peak_positions]
    peak_df_clean = peak_df.dropna(subset=valid_peak_cols)

    if len(peak_df_clean) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "Error: No valid peak data found\n\nCheck peak positions and tolerance",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#ffdddd", edgecolor="red"),
        )
        ax.axis("off")

        return {
            "primary_figure": fig,
            "secondary_figure": None,
            "data_table": peak_df,
            "summary_text": "Error: No valid peak data found within tolerance",
            "detailed_summary": f"Tolerance: {tolerance} cm⁻¹. Try increasing tolerance or adjusting peak positions.",
            "raw_results": {"peak_positions": peak_positions, "tolerance": tolerance},
        }

    if progress_callback:
        progress_callback(50)

    # Get colormap
    try:
        cmap = plt.cm.get_cmap(colormap_name)
    except ValueError:
        cmap = plt.cm.tab10

    dataset_names = list(dataset_data_rs.keys())
    n_datasets = len(dataset_names)
    colors = [cmap(i / max(n_datasets - 1, 1)) for i in range(n_datasets)]

    # Create scatter plot
    if use_3d and len(peak_positions) >= 3:
        # 3D scatter plot
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        for j, dataset_name in enumerate(dataset_names):
            mask = peak_df_clean["dataset"] == dataset_name
            x = peak_df_clean.loc[mask, valid_peak_cols[0]].values
            y = peak_df_clean.loc[mask, valid_peak_cols[1]].values
            z = peak_df_clean.loc[mask, valid_peak_cols[2]].values

            ax.scatter(
                x,
                y,
                z,
                c=[colors[j]],
                s=marker_size,
                alpha=0.7,
                label=dataset_name,
                edgecolors="white",
                linewidth=0.5,
            )

        ax.set_xlabel(f"{peak_positions[0]:.0f} cm⁻¹", fontsize=11, labelpad=10)
        ax.set_ylabel(f"{peak_positions[1]:.0f} cm⁻¹", fontsize=11, labelpad=10)
        ax.set_zlabel(f"{peak_positions[2]:.0f} cm⁻¹", fontsize=11, labelpad=10)
        ax.set_title(
            "3D Peak Intensity Scatter Plot", fontsize=14, fontweight="bold", pad=20
        )

        # Sensible default camera angle (aligns with waterfall's readability)
        try:
            ax.view_init(elev=float(params.get("elev", 20)), azim=float(params.get("azim", 45)))
        except Exception:
            pass

        if show_legend:
            ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    else:
        # 2D scatter plot with optional statistics
        fig, ax = plt.subplots(figsize=(10, 8))

        for j, dataset_name in enumerate(dataset_names):
            mask = peak_df_clean["dataset"] == dataset_name
            x = peak_df_clean.loc[mask, valid_peak_cols[0]].values
            y = peak_df_clean.loc[mask, valid_peak_cols[1]].values

            ax.scatter(
                x,
                y,
                c=[colors[j]],
                s=marker_size,
                alpha=0.7,
                label=dataset_name,
                edgecolors="white",
                linewidth=0.5,
            )

            # Add mean point with error bars if statistics enabled
            if show_statistics and len(x) > 1:
                mean_x, mean_y = np.mean(x), np.mean(y)
                std_x, std_y = np.std(x), np.std(y)

                ax.errorbar(
                    mean_x,
                    mean_y,
                    xerr=std_x,
                    yerr=std_y,
                    c=colors[j],
                    marker="D",
                    markersize=12,
                    capsize=5,
                    capthick=2,
                    elinewidth=2,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    zorder=10,
                )

        ax.set_xlabel(f"Intensity at {peak_positions[0]:.0f} cm⁻¹", fontsize=12)
        ax.set_ylabel(f"Intensity at {peak_positions[1]:.0f} cm⁻¹", fontsize=12)
        ax.set_title("Peak Intensity Scatter Plot", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")

        if show_legend:
            ax.legend(loc="best", fontsize=10, framealpha=0.9)

    # Layout: tight_layout often fails (and spams warnings) for 3D axes.
    try:
        import warnings as _warnings

        if use_3d:
            fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.92)
        else:
            with _warnings.catch_warnings():
                _warnings.filterwarnings(
                    "ignore",
                    message=r"This figure includes Axes that are not compatible with tight_layout.*",
                    category=UserWarning,
                )
                fig.tight_layout(pad=1.0)
    except Exception:
        pass

    if progress_callback:
        progress_callback(75)

    # Calculate comprehensive statistics for each dataset and peak
    stats_data = []
    for dataset_name in dataset_names:
        mask = peak_df_clean["dataset"] == dataset_name
        dataset_subset = peak_df_clean[mask]
        n_spectra = len(dataset_subset)

        for peak_pos in peak_positions:
            col_name = f"Peak_{peak_pos:.0f}"
            values = dataset_subset[col_name].values

            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values) if len(values) > 1 else 0
                cv_percent = (std_val / mean_val * 100) if mean_val != 0 else 0

                stats_data.append(
                    {
                        "Dataset": dataset_name,
                        "Peak (cm⁻¹)": f"{peak_pos:.0f}",
                        "N": n_spectra,
                        "Mean": f"{mean_val:.2f}",
                        "Std": f"{std_val:.2f}",
                        "CV (%)": f"{cv_percent:.1f}",
                        "Min": f"{np.min(values):.2f}",
                        "Max": f"{np.max(values):.2f}",
                        "Range": f"{np.max(values) - np.min(values):.2f}",
                    }
                )

    stats_df = pd.DataFrame(stats_data)

    if progress_callback:
        progress_callback(90)

    # Generate summary text
    summary_lines = [
        f"Peak Intensity Scatter Plot ({'3D' if use_3d else '2D'} mode)",
        f"Peak positions: {', '.join([f'{p:.0f}' for p in peak_positions])} cm⁻¹",
        f"Tolerance: ±{tolerance} cm⁻¹",
        f"Total spectra analyzed: {len(peak_df_clean)}",
        f"Datasets: {len(dataset_names)}",
    ]

    if warning_messages:
        summary_lines.append(f"\n⚠️ Warnings: {'; '.join(warning_messages)}")

    # Detailed summary with statistics interpretation
    detailed_lines = ["Statistical Summary by Dataset and Peak:\n"]

    for dataset_name in dataset_names:
        detailed_lines.append(f"\n📊 {dataset_name}:")
        dataset_mask = stats_df["Dataset"] == dataset_name
        dataset_stats = stats_df[dataset_mask]

        for _, row in dataset_stats.iterrows():
            cv = float(row["CV (%)"])
            cv_interpretation = (
                "low variability"
                if cv < 10
                else "moderate variability" if cv < 25 else "high variability"
            )
            detailed_lines.append(
                f"  • {row['Peak (cm⁻¹)']} cm⁻¹: Mean={row['Mean']}, CV={row['CV (%)']}% ({cv_interpretation})"
            )

    detailed_lines.append(
        "\n💡 Research Note: CV < 10% indicates good reproducibility. "
        "The diamond markers (◆) show mean ± std for each dataset."
    )

    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": stats_df,
        "summary_text": "\n".join(summary_lines),
        "detailed_summary": "\n".join(detailed_lines),
        "raw_results": {
            "peak_positions": peak_positions,
            "tolerance": tolerance,
            "full_peak_data": peak_df_clean.to_dict(),
            "statistics": stats_data,
            "warnings": warning_messages,
        },
    }


__all__ = [
    "create_spectral_heatmap",
    "create_mean_spectra_overlay",
    "create_waterfall_plot",
    "create_correlation_heatmap",
    "create_peak_scatter",
]
