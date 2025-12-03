"""
Visualization Methods

This module implements various visualization methods for Raman spectral data
including heatmaps, overlay plots, waterfall plots, and scatter plots.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.signal import find_peaks


def create_spectral_heatmap(dataset_data: Dict[str, pd.DataFrame],
                           params: Dict[str, Any],
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
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
        # Row-wise normalization (each spectrum)
        data_matrix = (data_matrix - data_matrix.min(axis=1, keepdims=True)) / \
                     (data_matrix.max(axis=1, keepdims=True) - data_matrix.min(axis=1, keepdims=True))
    
    if progress_callback:
        progress_callback(50)
    
    # Perform clustering
    row_linkage = None
    col_linkage = None
    row_order = np.arange(data_matrix.shape[0])
    col_order = np.arange(data_matrix.shape[1])
    
    if cluster_rows:
        row_linkage = linkage(data_matrix, method='average', metric='euclidean')
        row_dendrogram = dendrogram(row_linkage, no_plot=True)
        row_order = row_dendrogram['leaves']
    
    if cluster_cols:
        col_linkage = linkage(data_matrix.T, method='average', metric='euclidean')
        col_dendrogram = dendrogram(col_linkage, no_plot=True)
        col_order = col_dendrogram['leaves']
    
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
            gs = fig.add_gridspec(2, 2, width_ratios=[0.2, 1], height_ratios=[0.2, 1],
                                 hspace=0.05, wspace=0.05)
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
            dendrogram(row_linkage, ax=ax_dend_row, orientation='left',
                      no_labels=True, color_threshold=0)
            ax_dend_row.set_xticks([])
            ax_dend_row.set_yticks([])
            ax_dend_row.spines['top'].set_visible(False)
            ax_dend_row.spines['right'].set_visible(False)
            ax_dend_row.spines['bottom'].set_visible(False)
            ax_dend_row.spines['left'].set_visible(False)
        
        if cluster_cols and show_dendrograms:
            dendrogram(col_linkage, ax=ax_dend_col, orientation='top',
                      no_labels=True, color_threshold=0)
            ax_dend_col.set_xticks([])
            ax_dend_col.set_yticks([])
            ax_dend_col.spines['top'].set_visible(False)
            ax_dend_col.spines['right'].set_visible(False)
            ax_dend_col.spines['bottom'].set_visible(False)
            ax_dend_col.spines['left'].set_visible(False)
    else:
        fig, ax_heatmap = plt.subplots(figsize=(12, 8))
    
    # Plot heatmap
    im = ax_heatmap.imshow(data_ordered, aspect='auto', cmap=colormap,
                          interpolation='nearest')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap)
    cbar.set_label('Normalized Intensity' if normalize else 'Intensity',
                  fontsize=12)
    
    # Labels
    ax_heatmap.set_xlabel('Wavenumber Index', fontsize=12)
    ax_heatmap.set_ylabel('Spectrum Index', fontsize=12)
    ax_heatmap.set_title('Spectral Heatmap', fontsize=14, fontweight='bold')
    
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
            "col_order": col_order
        }
    }


def create_mean_spectra_overlay(dataset_data: Dict[str, pd.DataFrame],
                               params: Dict[str, Any],
                               progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
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
            data = (data - data.min(axis=0, keepdims=True)) / \
                  (data.max(axis=0, keepdims=True) - data.min(axis=0, keepdims=True))
        
        # Calculate mean and std
        mean_spectrum = data.mean(axis=1)
        std_spectrum = data.std(axis=1)
        
        # Plot mean
        ax.plot(wavenumbers, mean_spectrum, label=dataset_name,
               color=colors[i], linewidth=2)
        
        # Plot std bands
        if show_std:
            ax.fill_between(wavenumbers,
                           mean_spectrum - std_spectrum,
                           mean_spectrum + std_spectrum,
                           alpha=0.2, color=colors[i])
        
        # Plot individual spectra
        if show_individual:
            for j in range(data.shape[1]):
                ax.plot(wavenumbers, data[:, j],
                       color=colors[i], alpha=alpha_individual, linewidth=0.5)
    
    if progress_callback:
        progress_callback(80)
    
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title('Mean Spectra Overlay', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
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
        "raw_results": {}
    }


def create_waterfall_plot(dataset_data: Dict[str, pd.DataFrame],
                         params: Dict[str, Any],
                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Create waterfall plot of spectra with vertical offset.
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - offset_scale: Vertical offset scale (default 1.0)
            - max_spectra: Maximum number of spectra to plot (default 50)
            - colormap: Colormap for gradient (default 'viridis')
            - reverse_order: Reverse plotting order (default False)
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with waterfall plot
    """
    if progress_callback:
        progress_callback(10)
    
    # Get parameters
    offset_scale = params.get("offset_scale", 1.0)
    max_spectra = params.get("max_spectra", 50)
    colormap = params.get("colormap", "viridis")
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
        indices = np.linspace(0, len(all_spectra)-1, max_spectra, dtype=int)
        all_spectra = [all_spectra[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    wavenumbers = dataset_data[list(dataset_data.keys())[0]].index.values
    
    if progress_callback:
        progress_callback(40)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate offset
    max_intensity = max(np.max(spec) for spec in all_spectra)
    offset = max_intensity * offset_scale
    
    # Color gradient
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(all_spectra)))
    
    # Plot spectra
    plot_order = range(len(all_spectra))
    if reverse_order:
        plot_order = reversed(plot_order)
    
    for i in plot_order:
        spectrum = all_spectra[i]
        y_offset = i * offset
        ax.plot(wavenumbers, spectrum + y_offset,
               color=colors[i], linewidth=1.0, alpha=0.8)
    
    if progress_callback:
        progress_callback(80)
    
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Intensity (offset)', fontsize=12)
    ax.set_title('Waterfall Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_xaxis()
    
    # Remove y-ticks (offsets are arbitrary)
    ax.set_yticks([])
    
    if progress_callback:
        progress_callback(90)
    
    summary = f"Waterfall plot created with {len(all_spectra)} spectra.\n"
    summary += f"Offset scale: {offset_scale}"
    
    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": None,
        "summary_text": summary,
        "detailed_summary": f"Colormap: {colormap}",
        "raw_results": {}
    }


def create_correlation_heatmap(dataset_data: Dict[str, pd.DataFrame],
                               params: Dict[str, Any],
                               progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
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
    
    # Use first dataset
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
        linkage_matrix = linkage(corr_matrix.values, method='average')
        dend = dendrogram(linkage_matrix, no_plot=True)
        order = dend['leaves']
        corr_matrix_ordered = corr_matrix.iloc[order, order]
    else:
        corr_matrix_ordered = corr_matrix
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 9))
    
    im = ax.imshow(corr_matrix_ordered.values, cmap=colormap,
                  aspect='auto', vmin=-1, vmax=1)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=12)
    
    ax.set_xlabel('Wavenumber Index', fontsize=12)
    ax.set_ylabel('Wavenumber Index', fontsize=12)
    ax.set_title('Wavenumber Correlation Heatmap', fontsize=14, fontweight='bold')
    
    if progress_callback:
        progress_callback(90)
    
    summary = f"Correlation heatmap created using {method} method.\n"
    summary += f"Dataset: {dataset_name}\n"
    summary += f"Wavenumber range: {df.shape[0]} points"
    
    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": corr_matrix_ordered,
        "summary_text": summary,
        "detailed_summary": f"Clustering: {cluster}",
        "raw_results": {"correlation_matrix": corr_matrix_ordered.values}
    }


def create_peak_scatter(dataset_data: Dict[str, pd.DataFrame],
                       params: Dict[str, Any],
                       progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Create scatter plot of peak intensities across datasets.
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - peak_positions: List of wavenumber positions (default auto-detect)
            - tolerance: Tolerance for peak matching (default 5 cm-1)
            - prominence: Prominence threshold for auto-detection (default 0.05)
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with peak scatter plot
    """
    if progress_callback:
        progress_callback(10)
    
    # Get parameters
    peak_positions = params.get("peak_positions", None)
    tolerance = params.get("tolerance", 5)
    prominence = params.get("prominence", 0.05)
    
    # Get wavenumbers from first dataset
    wavenumbers = dataset_data[list(dataset_data.keys())[0]].index.values
    
    # Auto-detect peaks if not provided
    if peak_positions is None:
        # Use mean spectrum from first dataset for peak detection
        first_dataset = list(dataset_data.values())[0]
        mean_spectrum = first_dataset.mean(axis=1).values
        
        # Normalize
        norm_spectrum = (mean_spectrum - mean_spectrum.min()) / (mean_spectrum.max() - mean_spectrum.min())
        
        # Find peaks
        peaks, _ = find_peaks(norm_spectrum, prominence=prominence)
        peak_positions = wavenumbers[peaks]
        
        # Limit to top 10 peaks
        if len(peak_positions) > 10:
            # Get prominences and select top 10
            _, properties = find_peaks(norm_spectrum, prominence=prominence)
            top_indices = np.argsort(properties['prominences'])[-10:]
            peak_positions = wavenumbers[peaks[top_indices]]
    
    if progress_callback:
        progress_callback(40)
    
    # Extract peak intensities from all datasets
    peak_data = []
    
    for dataset_name, df in dataset_data.items():
        for col in df.columns:
            spectrum = df[col].values
            
            intensities = []
            for peak_pos in peak_positions:
                # Find closest wavenumber
                idx = np.argmin(np.abs(wavenumbers - peak_pos))
                if np.abs(wavenumbers[idx] - peak_pos) <= tolerance:
                    intensities.append(spectrum[idx])
                else:
                    intensities.append(np.nan)
            
            peak_data.append({
                'dataset': dataset_name,
                'spectrum': col,
                **{f'Peak_{peak_pos:.0f}': intensity
                   for peak_pos, intensity in zip(peak_positions, intensities)}
            })
    
    peak_df = pd.DataFrame(peak_data)
    
    if progress_callback:
        progress_callback(70)
    
    # Create scatter plot
    n_peaks = len(peak_positions)
    fig, axes = plt.subplots(1, min(n_peaks, 4), figsize=(16, 4))
    
    if n_peaks == 1:
        axes = [axes]
    elif n_peaks < 4:
        pass
    else:
        # Only show first 4 peaks
        peak_positions = peak_positions[:4]
    
    dataset_names = list(dataset_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_names)))
    
    for i, peak_pos in enumerate(peak_positions):
        ax = axes[i] if n_peaks > 1 else axes[0]
        
        col_name = f'Peak_{peak_pos:.0f}'
        
        for j, dataset_name in enumerate(dataset_names):
            mask = peak_df['dataset'] == dataset_name
            values = peak_df.loc[mask, col_name].values
            x = np.random.normal(j, 0.04, size=len(values))
            
            ax.scatter(x, values, c=[colors[j]], alpha=0.6, s=50, label=dataset_name if i == 0 else "")
        
        ax.set_xticks(range(len(dataset_names)))
        ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax.set_ylabel('Intensity', fontsize=10)
        ax.set_title(f'{peak_pos:.0f} cm⁻¹', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    if n_peaks > 1:
        axes[0].legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    if progress_callback:
        progress_callback(90)
    
    summary = f"Peak scatter plot created for {len(peak_positions)} peaks.\n"
    summary += f"Peak positions: {', '.join([f'{p:.0f}' for p in peak_positions])} cm⁻¹"
    
    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": peak_df,
        "summary_text": summary,
        "detailed_summary": f"Total spectra: {len(peak_df)}",
        "raw_results": {"peak_positions": peak_positions}
    }
