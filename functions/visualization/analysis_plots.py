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

    # Combine all datasets
    all_spectra = []
    labels = []

    for dataset_name, df in dataset_data.items():
        for col in df.columns:
            all_spectra.append(df[col].values)
            labels.append(f"{dataset_name}_{col}")

    # Create data matrix (rows=spectra, cols=wavenumbers)
    data_matrix = np.array(all_spectra)
    wavenumbers = dataset_data[list(dataset_data.keys())[0]].index.values

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
            dendrogram(
                row_linkage,
                ax=ax_dend_row,
                orientation="left",
                no_labels=True,
                color_threshold=0,
            )
            ax_dend_row.set_xticks([])
            ax_dend_row.set_yticks([])
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
    ax_heatmap.set_xlabel("Wavenumber Index", fontsize=12)
    ax_heatmap.set_ylabel("Spectrum Index", fontsize=12)
    ax_heatmap.set_title("Spectral Heatmap", fontsize=14, fontweight="bold")

    if progress_callback:
        progress_callback(90)

    summary = f"Heatmap created with {data_matrix.shape[0]} spectra.\n"
    if cluster_rows:
        summary += "Rows clustered. "
    if cluster_cols:
        summary += "Columns clustered. "
    if normalize:
        summary += "Data normalized."

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

    # Get wavenumbers from first dataset
    wavenumbers = dataset_data[list(dataset_data.keys())[0]].index.values

    if progress_callback:
        progress_callback(30)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_data)))

    for i, (dataset_name, df) in enumerate(dataset_data.items()):
        data = df.values

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
        "raw_results": {},
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
            - colormap: Colormap (default 'RdBu_r')
            - cluster: Apply clustering (default True)
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with correlation heatmap
    """
    if progress_callback:
        progress_callback(10)

    # Get parameters
    method = params.get("method", "pearson")
    colormap = params.get("colormap", "RdBu_r")
    cluster = params.get("cluster", True)

    # P1-4 FIX: Handle multi-dataset input properly
    multi_dataset_warning = ""
    if len(dataset_data) > 1:
        # Concatenate all datasets for correlation analysis
        all_dfs = []
        for ds_name, ds_df in dataset_data.items():
            all_dfs.append(ds_df)
        # Concatenate along columns (spectra)
        df = pd.concat(all_dfs, axis=1)
        dataset_name = f"{len(dataset_data)} datasets (concatenated)"
        multi_dataset_warning = f"\n⚠️ Note: {len(dataset_data)} datasets were concatenated for correlation analysis."
    else:
        dataset_name = list(dataset_data.keys())[0]
        df = dataset_data[dataset_name]

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

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 9))

    im = ax.imshow(
        corr_matrix_ordered.values, cmap=colormap, aspect="auto", vmin=-1, vmax=1
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation Coefficient", fontsize=12)

    ax.set_xlabel("Wavenumber Index", fontsize=12)
    ax.set_ylabel("Wavenumber Index", fontsize=12)
    ax.set_title("Wavenumber Correlation Heatmap", fontsize=14, fontweight="bold")

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
        "raw_results": {"correlation_matrix": corr_matrix_ordered.values},
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

    # Get wavenumbers from first dataset
    first_dataset_name = list(dataset_data.keys())[0]
    wavenumbers = dataset_data[first_dataset_name].index.values

    # Validate peak positions are within wavenumber range
    wn_min, wn_max = wavenumbers.min(), wavenumbers.max()
    valid_peaks = []
    warnings = []

    for i, peak_pos in enumerate(peak_positions, 1):
        if peak_pos < wn_min or peak_pos > wn_max:
            warnings.append(
                f"Peak {i} ({peak_pos} cm⁻¹) is outside data range [{wn_min:.0f}, {wn_max:.0f}]"
            )
        else:
            valid_peaks.append(peak_pos)

    if len(valid_peaks) < 2:
        # Not enough valid peaks - return error info
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            f"Error: Not enough valid peaks\n\n{chr(10).join(warnings)}",
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
            "summary_text": f"Error: {'; '.join(warnings)}",
            "detailed_summary": "Please adjust peak positions to be within the wavenumber range.",
            "raw_results": {"errors": warnings},
        }

    peak_positions = valid_peaks

    if progress_callback:
        progress_callback(25)

    # Extract peak intensities from all datasets
    peak_data = []

    for dataset_name, df in dataset_data.items():
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

    dataset_names = list(dataset_data.keys())
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

    plt.tight_layout()

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

    if warnings:
        summary_lines.append(f"\n⚠️ Warnings: {'; '.join(warnings)}")

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
            "warnings": warnings,
        },
    }


__all__ = [
    "create_spectral_heatmap",
    "create_mean_spectra_overlay",
    "create_waterfall_plot",
    "create_correlation_heatmap",
    "create_peak_scatter",
]
