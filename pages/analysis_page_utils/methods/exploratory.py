"""
Exploratory Analysis Methods

This module implements exploratory data analysis methods like PCA, UMAP,
t-SNE, and clustering techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional, Tuple, List
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for thread safety
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy import stats
from scipy.interpolate import interp1d

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# =============================================================================
# HIGH-CONTRAST COLOR PALETTE for multi-group visualization
# =============================================================================
def get_high_contrast_colors(num_groups: int) -> List[str]:
    """
    Get high-contrast color palette for clear visual distinction.
    
    ✅ FIX: Use distinct colors like "red and blue, green and red, yellow and blue"
    for maximum visibility in t-SNE, PCA, UMAP visualizations.
    
    Args:
        num_groups: Number of groups/datasets to color
        
    Returns:
        List of hex color strings with high contrast
    """
    if num_groups == 2:
        # Maximum contrast: Blue and Red
        return ['#0066cc', '#ff4444']
    elif num_groups == 3:
        # High contrast: Blue, Red, Green
        return ['#0066cc', '#ff4444', '#00cc66']
    elif num_groups == 4:
        # Blue, Red, Green, Orange
        return ['#0066cc', '#ff4444', '#00cc66', '#ff9900']
    elif num_groups == 5:
        # Blue, Red, Green, Orange, Purple
        return ['#0066cc', '#ff4444', '#00cc66', '#ff9900', '#9933ff']
    elif num_groups == 6:
        # Blue, Red, Green, Orange, Purple, Cyan
        return ['#0066cc', '#ff4444', '#00cc66', '#ff9900', '#9933ff', '#00cccc']
    else:
        # For 7+ groups, use tab10 with good spacing
        colors = plt.cm.tab10(np.linspace(0, 0.9, num_groups))
        return [f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' for c in colors]


# =============================================================================
# WAVENUMBER INTERPOLATION for multi-dataset dimension mismatch fix
# =============================================================================
def interpolate_to_common_wavenumbers(
    dataset_data: Dict[str, pd.DataFrame],
    method: str = 'linear'
) -> Tuple[np.ndarray, List[np.ndarray], List[str], np.ndarray]:
    """
    Interpolate all datasets to a common wavenumber grid.
    
    ✅ FIX: Resolves "ValueError: all the input array dimensions except for the
    concatenation axis must match exactly, but along dimension 1, the array at
    index 0 has size 2000 and the array at index 1 has size 559"
    
    This function finds the common overlapping wavenumber range across all datasets
    and resamples each dataset to this common grid using scipy.interpolate.interp1d.
    
    For Raman spectroscopy:
    - Different instruments may have different wavenumber resolutions
    - Different measurement settings may result in different ranges
    - This interpolation ensures all spectra have the same dimension for analysis
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame} where each DataFrame
                     has wavenumbers as index and spectra as columns
        method: Interpolation method ('linear', 'cubic', 'nearest')
        
    Returns:
        Tuple containing:
            - common_wavenumbers: The shared wavenumber grid (1D array)
            - interpolated_spectra: List of interpolated spectra matrices
            - labels: List of dataset labels for each spectrum
            - X: Concatenated matrix ready for analysis (n_spectra, n_wavenumbers)
    """
    print("[DEBUG] interpolate_to_common_wavenumbers() called")
    
    # Step 1: Find common wavenumber range
    wn_mins = []
    wn_maxs = []
    wn_counts = []
    
    for dataset_name, df in dataset_data.items():
        wavenumbers = df.index.values.astype(float)
        wn_mins.append(np.min(wavenumbers))
        wn_maxs.append(np.max(wavenumbers))
        wn_counts.append(len(wavenumbers))
        print(f"[DEBUG] Dataset '{dataset_name}': {len(wavenumbers)} points, range [{np.min(wavenumbers):.1f}, {np.max(wavenumbers):.1f}]")
    
    # Common range is the intersection of all ranges
    common_min = max(wn_mins)
    common_max = min(wn_maxs)
    
    if common_min >= common_max:
        raise ValueError(
            f"No overlapping wavenumber range found between datasets. "
            f"Ranges: min={wn_mins}, max={wn_maxs}"
        )
    
    # Use the minimum number of points from all datasets within the common range
    # This prevents unnecessary upsampling and preserves data integrity
    avg_density = np.mean(wn_counts) / np.mean([mx - mn for mn, mx in zip(wn_mins, wn_maxs)])
    n_common_points = int(avg_density * (common_max - common_min))
    n_common_points = max(n_common_points, 50)  # At least 50 points
    n_common_points = min(n_common_points, max(wn_counts))  # Don't exceed original resolution
    
    common_wavenumbers = np.linspace(common_min, common_max, n_common_points)
    
    print(f"[DEBUG] Common wavenumber range: [{common_min:.1f}, {common_max:.1f}] with {n_common_points} points")
    
    # Step 2: Interpolate each dataset to common grid
    interpolated_spectra = []
    labels = []
    
    for dataset_name, df in dataset_data.items():
        original_wn = df.index.values.astype(float)
        spectra = df.values  # Shape: (n_wavenumbers, n_spectra)
        
        # Interpolate each spectrum
        interpolated = np.zeros((n_common_points, spectra.shape[1]))
        
        for col_idx in range(spectra.shape[1]):
            spectrum = spectra[:, col_idx]
            
            # Create interpolation function
            # Use fill_value='extrapolate' to handle edge cases, but we shouldn't need it
            # since we're using the common overlapping range
            interp_func = interp1d(
                original_wn, 
                spectrum, 
                kind=method, 
                fill_value='extrapolate',
                bounds_error=False
            )
            
            interpolated[:, col_idx] = interp_func(common_wavenumbers)
        
        # Transpose to get (n_spectra, n_wavenumbers) for vstack
        interpolated_spectra.append(interpolated.T)
        labels.extend([dataset_name] * spectra.shape[1])
        
        print(f"[DEBUG] Interpolated '{dataset_name}': {spectra.shape[1]} spectra, now {n_common_points} points each")
    
    # Step 3: Concatenate all interpolated spectra
    X = np.vstack(interpolated_spectra)
    
    print(f"[DEBUG] Final combined matrix shape: {X.shape}")
    
    return common_wavenumbers, interpolated_spectra, labels, X


def interpolate_to_common_wavenumbers_with_groups(
    dataset_data: Dict[str, pd.DataFrame],
    group_labels_map: Optional[Dict[str, str]] = None,
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Interpolate datasets with group label support.
    
    Same as interpolate_to_common_wavenumbers but handles group_labels_map
    for PCA, UMAP, t-SNE group coloring.
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        group_labels_map: Optional {dataset_name: group_label} mapping
        method: Interpolation method
        
    Returns:
        Tuple containing:
            - common_wavenumbers: The shared wavenumber grid (1D array)
            - X: Concatenated matrix ready for analysis (n_spectra, n_wavenumbers)
            - labels: List of group labels for each spectrum
    """
    common_wn, interp_spectra, raw_labels, X = interpolate_to_common_wavenumbers(
        dataset_data, method=method
    )
    
    # Apply group labels if provided
    if group_labels_map:
        labels = []
        idx = 0
        for dataset_name, df in dataset_data.items():
            n_spectra = df.shape[1]  # Number of columns = number of spectra
            if dataset_name in group_labels_map:
                labels.extend([group_labels_map[dataset_name]] * n_spectra)
            else:
                labels.extend([dataset_name] * n_spectra)
            idx += n_spectra
    else:
        labels = raw_labels
    
    return common_wn, X, labels


def add_confidence_ellipse(ax, x, y, n_std=1.96, facecolor='none', edgecolor='red', linestyle='--', linewidth=2, alpha=0.7, label=None):
    """
    Add a confidence ellipse to a matplotlib axis using DUAL-LAYER PATTERN.
    
    ✅ FIX #3 (P0): Dual-layer ellipses for better visibility
    Consensus from 6 AI analyses: α=0.08 fill + α=0.85 edge prevents dark overlaps
    
    For Raman spectroscopy Chemometrics, 95% confidence ellipses (n_std=1.96) are critical
    for proving statistical group separation in PCA plots.
    
    Args:
        ax: matplotlib axis object
        x, y: Data coordinates (numpy arrays)
        n_std: Number of standard deviations (1.96 for 95% CI)
        facecolor: Color for fill layer (will be made very transparent)
        edgecolor: Color for edge layer
        linestyle, linewidth, alpha: Edge styling
        label: Legend label for the ellipse
    
    Returns:
        Ellipse patch object (edge layer, for legend)
    """
    if x.size == 0 or y.size == 0:
        return None
    
    # Calculate covariance matrix
    cov = np.cov(x, y)
    
    # Calculate eigenvalues and eigenvectors (principal axes of ellipse)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort eigenvalues and eigenvectors (largest first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Calculate angle of rotation
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    
    # Width and height are "full" widths, not radii
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    
    # Mean position (center of ellipse)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # ✅ DUAL-LAYER PATTERN: Layer 1 - Very transparent fill (barely visible)
    color_to_use = edgecolor if facecolor == 'none' else facecolor
    ellipse_fill = Ellipse(
        xy=(mean_x, mean_y), 
        width=width, 
        height=height, 
        angle=angle,
        facecolor=color_to_use,
        edgecolor='none',  # No edge on fill layer
        alpha=0.08,  # ✅ Ultra-light fill (8% opacity)
        zorder=5
    )
    ax.add_patch(ellipse_fill)
    
    # ✅ DUAL-LAYER PATTERN: Layer 2 - Bold visible edge (strong boundary)
    ellipse_edge = Ellipse(
        xy=(mean_x, mean_y), 
        width=width, 
        height=height, 
        angle=angle,
        facecolor='none',  # No fill on edge layer
        edgecolor=edgecolor,
        linestyle=linestyle,
        linewidth=linewidth if linewidth else 2.5,  # ✅ Thicker edge
        alpha=0.85,  # ✅ Strong edge visibility
        label=label,  # Only the edge gets the label for legend
        zorder=15  # Above scatter points
    )
    ax.add_patch(ellipse_edge)
    
    print(f"[DEBUG] Dual-layer ellipse added: center=({mean_x:.2f}, {mean_y:.2f}), size={width:.2f}x{height:.2f}, fill α=0.08, edge α=0.85")
    return ellipse_edge  # Return edge ellipse for legend


def perform_pca_analysis(dataset_data: Dict[str, pd.DataFrame],
                        params: Dict[str, Any],
                        progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Perform Principal Component Analysis on spectral data with multi-dataset support.
    
    Critical Raman Spectroscopy Context:
    - For multi-dataset comparison, ALL datasets are concatenated into ONE matrix
    - PCA is performed on the combined matrix to find variance patterns across groups
    - This allows visualization of group separation in the same PC space
    - Score distributions show overlap/separation between groups (key for classification)
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
            - Wavenumbers as index, spectra as columns
            - Multiple datasets for group comparison (e.g., "Control" vs "Disease")
        params: Analysis parameters
            - n_components: Number of components (default 3)
            - scaling: Scaler type ('StandardScaler', 'MinMaxScaler', 'None')
            - show_loadings: Show PC loadings plot (spectral interpretation)
            - show_scree: Show scree plot (variance explained)
            - show_distributions: Show score distribution plots (group comparison)
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary containing:
            - primary_figure: Scores plot (PC1 vs PC2 scatter)
            - secondary_figure: Score distributions (PC1, PC2, PC3 histograms/KDE)
            - data_table: PC scores DataFrame with dataset labels
            - summary_text: Analysis summary
            - raw_results: Full PCA results (model, scores, loadings, variance)
    """
    if progress_callback:
        progress_callback(10)
    
    # Get parameters
    n_components = params.get("n_components", 3)
    scaling_type = params.get("scaling", "StandardScaler")
    show_ellipses = params.get("show_ellipses", True)  # Confidence ellipses (critical for Chemometrics)
    show_loadings = params.get("show_loadings", True)
    show_scree = params.get("show_scree", True)
    show_distributions = params.get("show_distributions", True)
    group_labels_map = params.get("_group_labels", None)  # {dataset_name: group_label}
    
    print(f"[DEBUG] PCA parameters: n_components={n_components}, show_ellipses={show_ellipses}")
    print(f"[DEBUG] show_loadings={show_loadings}, show_scree={show_scree}, show_distributions={show_distributions}")
    
    # ✅ FIX: Use interpolation to handle datasets with different wavenumber ranges
    # This resolves "ValueError: all input array dimensions except for concatenation axis must match"
    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data, 
        group_labels_map=group_labels_map,
        method='linear'
    )
    
    print(f"[DEBUG] Combined matrix after interpolation: {X.shape}")
    print(f"[DEBUG] Common wavenumber range: [{wavenumbers[0]:.1f}, {wavenumbers[-1]:.1f}]")
    
    if progress_callback:
        progress_callback(30)
    
    # Apply scaling (essential for comparing datasets with different intensities)
    if scaling_type == "StandardScaler":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaling_type == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    if progress_callback:
        progress_callback(50)
    
    # Perform PCA on COMBINED matrix (key for group comparison)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)  # Shape: (total_spectra, n_components)
    
    if progress_callback:
        progress_callback(70)
    
    # === FIGURE 1: PC1 vs PC2 scores scatter plot WITH confidence ellipses ===
    print("[DEBUG] Creating PCA scores plot")
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    unique_labels = sorted(set(labels))
    num_groups = len(unique_labels)
    print(f"[DEBUG] Number of groups/datasets: {num_groups}")
    print(f"[DEBUG] Group labels: {unique_labels}")
    
    # Use HIGH-CONTRAST color palette for clear distinction
    # For 2 datasets: blue (#1f77b4) and yellow/gold (#ffd700)
    # For 3+ datasets: use qualitative palettes with maximum contrast
    if num_groups == 2:
        # Maximum contrast for 2 groups: blue and yellow-gold
        colors = np.array([[0.12, 0.47, 0.71, 1.0],  # Blue
                          [1.0, 0.84, 0.0, 1.0]])    # Gold/Yellow
        print("[DEBUG] Using high-contrast 2-color palette: Blue and Gold")
    elif num_groups == 3:
        # High contrast for 3 groups: blue, red, green
        colors = np.array([[0.12, 0.47, 0.71, 1.0],  # Blue
                          [0.84, 0.15, 0.16, 1.0],   # Red
                          [0.17, 0.63, 0.17, 1.0]])  # Green
        print("[DEBUG] Using high-contrast 3-color palette: Blue, Red, Green")
    else:
        # For 4+ groups, use tab10 but with better spacing
        colors = plt.cm.tab10(np.linspace(0, 0.9, num_groups))
        print(f"[DEBUG] Using tab10 palette for {num_groups} groups")
    
    # Plot each dataset with distinct color
    for i, dataset_label in enumerate(unique_labels):
        mask = np.array([l == dataset_label for l in labels])
        num_points = np.sum(mask)
        print(f"[DEBUG] Group '{dataset_label}': {num_points} spectra")
        
        ax1.scatter(scores[mask, 0], scores[mask, 1],
                   c=[colors[i]], label=dataset_label,
                   alpha=0.7, s=100, edgecolors='white', linewidth=1.0)
        
        # Add 95% confidence ellipse (CRITICAL for Chemometrics) - controlled by parameter
        if show_ellipses and num_points >= 3:  # User-controlled + need at least 3 points
            print(f"[DEBUG] Adding 95% CI ellipse for '{dataset_label}' ({num_points} points, show_ellipses=True)")
            add_confidence_ellipse(
                ax1, 
                scores[mask, 0], 
                scores[mask, 1],
                n_std=1.96,  # 95% confidence interval
                edgecolor=colors[i],
                linestyle='--',
                linewidth=2,
                alpha=0.6,
                label=f'{dataset_label} 95% CI'
            )
        elif not show_ellipses:
            print(f"[DEBUG] Ellipses disabled by user (show_ellipses=False) for '{dataset_label}'")
        else:
            print(f"[DEBUG] Skipping ellipse for '{dataset_label}' (only {num_points} points, need ≥3)")
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                   fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                   fontsize=12, fontweight='bold')
    
    # ✅ FIX #6 (P1): Clear title and legend labels
    # Title changes based on whether ellipses are shown
    if show_ellipses:
        ax1.set_title('PCA Score Plot with 95% Confidence Ellipses', fontsize=14, fontweight='bold')
       # Add explanatory footnote for scientific clarity
        ax1.text(0.02, 0.02,
                "95% Confidence Ellipses calculated using Hotelling's T² (1.96σ)",
                transform=ax1.transAxes,
                fontsize=9, color='#555555', style='italic',
                verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         alpha=0.92, edgecolor='#cccccc', linewidth=0.5))
    else:
        ax1.set_title('PCA Score Plot', fontsize=14, fontweight='bold')
    
    # Larger legend with better visibility
    ax1.legend(loc='best', framealpha=0.95, fontsize=10, 
              edgecolor='#cccccc', fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    print("[DEBUG] PCA scores plot created successfully")
    
    # === FIGURE 3: Scree Plot (Variance Explained) ===
    fig_scree = None
    if show_scree:
        print("[DEBUG] Creating scree plot...")
        
        # ✅ FIX #7 (P1): Side-by-side layout (bar LEFT | cumulative RIGHT)
        from matplotlib.gridspec import GridSpec
        
        fig_scree = plt.figure(figsize=(14, 5.5))
        gs = GridSpec(1, 2, figure=fig_scree, wspace=0.25)
        
        pc_indices = np.arange(1, n_components + 1)
        explained_variance = pca.explained_variance_ratio_ * 100
        cumulative_variance = np.cumsum(explained_variance)
        
        # LEFT: Bar chart for individual variance
        ax_bar = fig_scree.add_subplot(gs[0, 0])
        bar_colors = ['#e74c3c' if var > 10 else '#4a90e2' for var in explained_variance]
        bars = ax_bar.bar(pc_indices, explained_variance, color=bar_colors, 
                          edgecolor='white', linewidth=1.5, alpha=0.85, width=0.65)
        
        for bar, var in zip(bars, explained_variance):
            ax_bar.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{var:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax_bar.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
        ax_bar.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
        ax_bar.set_title('Individual Variance per PC', fontsize=13, fontweight='bold')
        ax_bar.set_xticks(pc_indices)
        ax_bar.set_ylim(0, max(explained_variance) * 1.15)
        ax_bar.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # RIGHT: Cumulative variance line
        ax_cum = fig_scree.add_subplot(gs[0, 1])
        ax_cum.plot(pc_indices, cumulative_variance, marker='o', markersize=9, 
                   linewidth=2.8, color='#2ecc71', markeredgecolor='white',
                   markeredgewidth=1.5, alpha=0.95, label='Cumulative')
        ax_cum.axhline(y=80, color='#f39c12', linestyle='--', linewidth=2.5,
                      alpha=0.75, label='80% Threshold')
        ax_cum.axhline(y=95, color='#e74c3c', linestyle='--', linewidth=2.5,
                      alpha=0.75, label='95% Threshold')
        
        for i, cum in enumerate(cumulative_variance):
            if i < 5:
                ax_cum.text(i+1, cum + 2, f'{cum:.1f}%', ha='center', va='bottom',
                           fontsize=9, fontweight='bold', color='#2ecc71')
        
        ax_cum.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
        ax_cum.set_ylabel('Cumulative Variance (%)', fontsize=12, fontweight='bold')
        ax_cum.set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
        ax_cum.set_xticks(pc_indices)
        ax_cum.set_ylim(0, 105)
        ax_cum.grid(True, alpha=0.3, linestyle='--')
        ax_cum.legend(loc='lower right', fontsize=10, framealpha=0.9)
        
        fig_scree.tight_layout(pad=1.2)
        print("[DEBUG] Side-by-side scree plot created successfully")
    
    # === FIGURE 4: Biplot (Scores + Loadings Overlay) ===
    fig_biplot = None
    if show_loadings and n_components >= 2:
        print("[DEBUG] Creating biplot...")
        fig_biplot, ax_biplot = plt.subplots(figsize=(12, 10))
        
        # Plot scores (same as primary figure but without ellipses for clarity)
        for i, dataset_label in enumerate(unique_labels):
            mask = np.array([l == dataset_label for l in labels])
            ax_biplot.scatter(scores[mask, 0], scores[mask, 1],
                            c=[colors[i]], label=dataset_label, s=60, alpha=0.6,
                            edgecolors='white', linewidths=0.5)
        
        # Overlay loadings as arrows (scaled for visibility)
        loading_scale = np.max(np.abs(scores[:, :2])) * 0.8
        
        # Select top contributing wavenumbers (peaks in loadings)
        pc1_loadings = pca.components_[0]
        pc2_loadings = pca.components_[1]
        loading_magnitude = np.sqrt(pc1_loadings**2 + pc2_loadings**2)
        
        # Show top 15 most influential wavenumbers
        top_indices = np.argsort(loading_magnitude)[-15:]
        
        for idx in top_indices:
            ax_biplot.arrow(0, 0,
                           pc1_loadings[idx] * loading_scale,
                           pc2_loadings[idx] * loading_scale,
                           head_width=loading_scale*0.02, head_length=loading_scale*0.03,
                           fc='#d13438', ec='#8b0000', alpha=0.8, linewidth=0.8)
            
            # Label with wavenumber - thinner text, no box to be cleaner
            ax_biplot.text(pc1_loadings[idx] * loading_scale * 1.15,
                          pc2_loadings[idx] * loading_scale * 1.15,
                          f'{int(wavenumbers[idx])}',
                          fontsize=8, ha='center', va='center', color='#8b0000', fontweight='bold')
        
        ax_biplot.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                           fontsize=12, fontweight='bold')
        ax_biplot.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
                           fontsize=12, fontweight='bold')
        ax_biplot.set_title('PCA Biplot: Scores + Influential Wavenumbers', 
                          fontsize=14, fontweight='bold')
        ax_biplot.legend(loc='best', fontsize=11, framealpha=0.9)
        ax_biplot.grid(True, alpha=0.3)
        ax_biplot.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
        ax_biplot.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
        fig_biplot.tight_layout()
        print("[DEBUG] Biplot created successfully")
    
    # === FIGURE 5: Cumulative Variance Explained ===
    fig_cumvar = None
    if show_scree:
        print("[DEBUG] Creating cumulative variance plot...")
        fig_cumvar, ax_cumvar = plt.subplots(figsize=(10, 6))
        
        pc_indices = np.arange(1, n_components + 1)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_ * 100)
        
        # Area plot
        ax_cumvar.fill_between(pc_indices, cumulative_variance, alpha=0.4, color='#28a745')
        ax_cumvar.plot(pc_indices, cumulative_variance, color='#28a745', marker='o',
                      linewidth=3, markersize=10, markerfacecolor='white',
                      markeredgewidth=2, markeredgecolor='#28a745')
        
        # Add threshold lines
        ax_cumvar.axhline(y=80, color='#ffc107', linestyle='--', linewidth=2, 
                         label='80% Threshold', alpha=0.7)
        ax_cumvar.axhline(y=95, color='#dc3545', linestyle='--', linewidth=2,
                         label='95% Threshold', alpha=0.7)
        
        # Annotate values
        for i, cum_var in enumerate(cumulative_variance):
            ax_cumvar.text(i+1, cum_var + 2, f'{cum_var:.1f}%',
                          ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax_cumvar.set_xlabel('Number of Principal Components', fontsize=12, fontweight='bold')
        ax_cumvar.set_ylabel('Cumulative Variance Explained (%)', fontsize=12, fontweight='bold')
        ax_cumvar.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        ax_cumvar.set_xticks(pc_indices)
        ax_cumvar.set_ylim(0, 105)
        ax_cumvar.grid(True, alpha=0.3)
        ax_cumvar.legend(loc='lower right', fontsize=11)
        fig_cumvar.tight_layout()
        print("[DEBUG] Cumulative variance plot created successfully")
    
    if progress_callback:
        progress_callback(75)
    
    # === FIGURE 2: Loadings Plot (Spectral interpretation) - ENHANCED WITH SUBPLOTS ===
    print(f"[DEBUG] show_loadings parameter: {show_loadings}")
    fig_loadings = None
    if show_loadings:
        print("[DEBUG] Creating loadings figure with subplots...")
        
        # Get max_loadings_components parameter (default 3, max 5)
        max_loadings = params.get("max_loadings_components", 3)
        max_loadings = min(max_loadings, n_components, 5)  # Ensure within bounds
        
        print(f"[DEBUG] Creating {max_loadings} loading subplot(s)")
        
        # Create subplot grid (vertical stack for better readability)
        fig_loadings, axes = plt.subplots(max_loadings, 1, figsize=(12, 4 * max_loadings))
        
        # Handle single subplot case (axes won't be array)
        if max_loadings == 1:
            axes = [axes]
        
        # Color palette for components
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Plot each component in its own subplot
        for pc_idx in range(max_loadings):
            ax = axes[pc_idx]
            
            # Plot loadings for this component
            ax.plot(wavenumbers, pca.components_[pc_idx], 
                   linewidth=2, color=colors[pc_idx], label=f'PC{pc_idx+1}')
            
            # Explained variance for this component
            explained_var = pca.explained_variance_ratio_[pc_idx] * 100
            
            # Styling
            ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Loading Value', fontsize=11, fontweight='bold')
            ax.set_title(f'PC{pc_idx+1} Loadings (Explained Variance: {explained_var:.2f}%)', 
                        fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()  # Raman convention: high to low wavenumber
            # Remove x-axis tick labels (wavenumbers) as requested
            ax.set_xticklabels([])
            
            # ✅ FIX #8 (P1): Annotate top 5 peak positions for this component (increased from 3)
            # Consensus from 6 AI analyses: Top 5 peaks provide better spectral interpretation
            loadings = pca.components_[pc_idx]
            abs_loadings = np.abs(loadings)
            top_indices = np.argsort(abs_loadings)[-5:]  # Top 5 peaks (increased from 3)
            
            for peak_idx in top_indices:
                peak_wn = wavenumbers[peak_idx]
                peak_val = loadings[peak_idx]
                ax.plot(peak_wn, peak_val, 'o', color=colors[pc_idx], markersize=6, 
                       markeredgecolor='black', markeredgewidth=0.5)
                ax.annotate(f'{peak_wn:.0f}', 
                           xy=(peak_wn, peak_val), 
                           xytext=(0, 10 if peak_val > 0 else -15),
                           textcoords='offset points',
                           fontsize=8, fontweight='bold',
                           ha='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#cccccc'))
        
        fig_loadings.tight_layout()
        print(f"[DEBUG] Loadings figure created successfully with {max_loadings} subplots")
    else:
        print("[DEBUG] Loadings figure skipped (show_loadings=False)")
    
    if progress_callback:
        progress_callback(80)
    
    # === FIGURE 3: Score Distributions (CRITICAL for Raman classification) ===
    fig_distributions = None
    if show_distributions and len(unique_labels) > 1:
        # Get number of components to show (default 3, max 6)
        n_dist_comps = params.get("n_distribution_components", 3)
        n_pcs_to_plot = min(n_dist_comps, n_components)
        
        # Calculate grid dimensions (max 2 columns)
        n_cols = 2 if n_pcs_to_plot > 1 else 1
        n_rows = int(np.ceil(n_pcs_to_plot / n_cols))
        
        # Create figure with appropriate size
        fig_distributions, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        
        # Flatten axes for easy iteration if multiple
        if n_pcs_to_plot > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
            
        fig_distributions.suptitle('PC Score Distributions', fontsize=16, fontweight='bold')
        
        # Plot distributions for each PC
        for idx in range(n_pcs_to_plot):
            ax = axes_flat[idx]
            pc_idx = idx  # 0-based index
            
            # Plot histogram/KDE for each dataset
            for i, dataset_label in enumerate(unique_labels):
                mask = np.array([l == dataset_label for l in labels])
                pc_scores = scores[mask, pc_idx]
                
                # Calculate KDE (Kernel Density Estimation)
                try:
                    kde = stats.gaussian_kde(pc_scores)
                    x_range = np.linspace(pc_scores.min() - 1, pc_scores.max() + 1, 200)
                    kde_values = kde(x_range)
                    
                    # Plot KDE curve
                    ax.plot(x_range, kde_values, color=colors[i], linewidth=2.5,
                           label=dataset_label, alpha=0.9)
                    
                    # Fill under curve for visibility
                    ax.fill_between(x_range, kde_values, alpha=0.25, color=colors[i])
                except Exception:
                    # Fallback if KDE fails (e.g. singular matrix due to too few points)
                    pass
                
                # Add histogram for reference
                ax.hist(pc_scores, bins=20, density=True, alpha=0.15,
                       color=colors[i], edgecolor='white', linewidth=0.5)
            
            # Statistical test (Mann-Whitney U for 2 groups)
            if len(unique_labels) == 2:
                mask1 = np.array([l == unique_labels[0] for l in labels])
                mask2 = np.array([l == unique_labels[1] for l in labels])
                pc1_scores = scores[mask1, pc_idx]
                pc2_scores = scores[mask2, pc_idx]
                
                try:
                    # Mann-Whitney U test
                    statistic, p_value = stats.mannwhitneyu(pc1_scores, pc2_scores)
                    
                    # Calculate effect size (Cohen's d)
                    mean_diff = np.mean(pc1_scores) - np.mean(pc2_scores)
                    pooled_std = np.sqrt((np.std(pc1_scores)**2 + np.std(pc2_scores)**2) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    # Add statistical annotation
                    ax.text(0.05, 0.95, 
                           f'Mann–Whitney U\np={p_value:.2e}\nδ={cohens_d:.2f}',
                           transform=ax.transAxes, fontsize=10,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                except Exception:
                    pass
            
            # Formatting
            ax.set_xlabel(f'PC{pc_idx+1} Score', fontsize=12, fontweight='bold')
            ax.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax.set_title(f'PC{pc_idx+1} ({pca.explained_variance_ratio_[pc_idx]*100:.1f}%)',
                        fontsize=13, fontweight='bold')
            if idx == 0:  # Only show legend on first plot to save space
                ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            
        # Hide empty subplots if any
        # Ensure axes_flat is always a list/array even if single subplot
        if not isinstance(axes_flat, (list, np.ndarray)):
            axes_flat = [axes_flat]
            
        if n_pcs_to_plot < len(axes_flat):
            for idx in range(n_pcs_to_plot, len(axes_flat)):
                axes_flat[idx].axis('off')
                axes_flat[idx].set_visible(False) # Explicitly hide
        
        plt.tight_layout()
    
    if progress_callback:
        progress_callback(90)
    
    # Create data table with dataset labels
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    scores_df = pd.DataFrame(scores, columns=pc_columns)
    scores_df['Dataset'] = labels
    
    # === ENHANCED SUMMARY TEXT ===
    n_datasets = len(unique_labels)
    total_spectra = X.shape[0]
    total_variance = np.sum(pca.explained_variance_ratio_[:min(3, n_components)]) * 100
    
    # Header
    summary = f"╔═══════════════════════════════════════════════════════╗\n"
    summary += f"║       PCA ANALYSIS RESULTS - COMPREHENSIVE SUMMARY    ║\n"
    summary += f"╚═══════════════════════════════════════════════════════╝\n\n"
    
    # Basic Information
    summary += f"📊 ANALYSIS OVERVIEW\n"
    summary += f"{'─' * 55}\n"
    summary += f"  Total Datasets:    {n_datasets}\n"
    summary += f"  Total Spectra:     {total_spectra}\n"
    summary += f"  Components:        {n_components}\n"
    summary += f"  Scaling Method:    {scaling_type}\n"
    summary += f"  Datasets:          {', '.join(unique_labels)}\n\n"
    
    # Variance Explained
    summary += f"📈 VARIANCE EXPLAINED\n"
    summary += f"{'─' * 55}\n"
    
    # Show variance for each component (up to 10)
    max_components_summary = min(10, n_components)
    for i in range(max_components_summary):
        var_pct = pca.explained_variance_ratio_[i] * 100
        cumvar_pct = np.sum(pca.explained_variance_ratio_[:i+1]) * 100
        
        # Visual bar for variance
        bar_length = int(var_pct / 2)  # Scale to 50 chars max
        bar = '█' * bar_length + '░' * (50 - bar_length)
        
        summary += f"  PC{i+1:2d}:  {var_pct:5.2f}% │{bar}│ Cumulative: {cumvar_pct:5.2f}%\n"
    
    if n_components > 10:
        remaining_var = np.sum(pca.explained_variance_ratio_[10:]) * 100
        summary += f"  ...   {remaining_var:5.2f}% (remaining {n_components-10} components)\n"
    
    summary += f"\n  First 3 PCs:       {total_variance:.2f}% of total variance\n"
    cumvar_all = np.sum(pca.explained_variance_ratio_) * 100
    summary += f"  All {n_components} PCs:        {cumvar_all:.2f}% of total variance\n\n"
    
    # Top Spectral Features per Component
    summary += f"🔬 TOP SPECTRAL FEATURES (Peak Wavenumbers)\n"
    summary += f"{'─' * 55}\n"
    
    max_components_features = min(5, n_components)  # Show top 5 components
    for pc_idx in range(max_components_features):
        loadings = pca.components_[pc_idx]
        abs_loadings = np.abs(loadings)
        top_3_indices = np.argsort(abs_loadings)[-3:][::-1]  # Top 3 peaks, descending
        
        top_features = [f"{wavenumbers[idx]:.0f} cm⁻¹" for idx in top_3_indices]
        summary += f"  PC{pc_idx+1}:  {', '.join(top_features)}\n"
    
    summary += f"\n"
    
    # Group Separation Analysis (for binary comparison)
    if len(unique_labels) == 2:
        summary += f"📉 GROUP SEPARATION ANALYSIS\n"
        summary += f"{'─' * 55}\n"
        
        mask1 = np.array([l == unique_labels[0] for l in labels])
        mask2 = np.array([l == unique_labels[1] for l in labels])
        
        # PC1 separation
        pc1_mean1 = np.mean(scores[mask1, 0])
        pc1_mean2 = np.mean(scores[mask2, 0])
        pc1_separation = abs(pc1_mean1 - pc1_mean2)
        
        pc1_std1 = np.std(scores[mask1, 0])
        pc1_std2 = np.std(scores[mask2, 0])
        pc1_pooled_std = np.sqrt((pc1_std1**2 + pc1_std2**2) / 2)
        
        cohens_d = pc1_separation / pc1_pooled_std if pc1_pooled_std > 0 else 0
        
        summary += f"  Groups:            {unique_labels[0]} vs {unique_labels[1]}\n"
        summary += f"  PC1 Mean Diff:     {pc1_separation:.3f}\n"
        summary += f"  Cohen's d:         {cohens_d:.3f}\n"
        
        if cohens_d > 0.8:
            effect = "LARGE (Excellent separation)"
            indicator = "✓✓✓"
        elif cohens_d > 0.5:
            effect = "MEDIUM (Moderate separation)"
            indicator = "✓✓"
        elif cohens_d > 0.2:
            effect = "SMALL (Weak separation)"
            indicator = "✓"
        else:
            effect = "NEGLIGIBLE (Poor separation)"
            indicator = "✗"
        
        summary += f"  Effect Size:       {indicator} {effect}\n\n"
    
    # Interpretation Guide
    summary += f"💡 INTERPRETATION GUIDE\n"
    summary += f"{'─' * 55}\n"
    summary += f"  • Scores Plot:     Shows sample clustering in PC space\n"
    summary += f"  • Loadings Plot:   Shows spectral features driving each PC\n"
    summary += f"  • High loading:    Strong contribution to that component\n"
    summary += f"  • Clusters:        Biochemically similar samples group together\n"
    summary += f"  • Outliers:        Samples far from origin may be anomalous\n\n"
    
    if show_loadings:
        max_loadings = params.get("max_loadings_components", 3)
        max_loadings = min(max_loadings, n_components, 5)
        summary += f"  Loading plots generated for first {max_loadings} component(s)\n"
    
    summary += f"\n{'═' * 55}\n"
    
    print(f"[DEBUG] Enhanced summary generated ({len(summary)} characters)")
    # Statistical summary for multi-dataset comparison
    detailed_summary = f"Scaling: {scaling_type}\nTotal spectra: {X.shape[0]}\n"
    detailed_summary += f"Datasets: {n_datasets} groups\n"
    
    if len(unique_labels) == 2:
        # Add separation metrics for binary comparison
        mask1 = np.array([l == unique_labels[0] for l in labels])
        mask2 = np.array([l == unique_labels[1] for l in labels])
        
        # Calculate separation in PC1
        pc1_separation = abs(np.mean(scores[mask1, 0]) - np.mean(scores[mask2, 0]))
        pc1_pooled_std = np.sqrt((np.std(scores[mask1, 0])**2 + np.std(scores[mask2, 0])**2) / 2)
        separation_ratio = pc1_separation / pc1_pooled_std if pc1_pooled_std > 0 else 0
        
        detailed_summary += f"\nPC1 Separation:\n"
        detailed_summary += f"  Mean difference: {pc1_separation:.2f}\n"
        detailed_summary += f"  Cohen's d: {separation_ratio:.2f}\n"
        
        if separation_ratio > 0.8:
            detailed_summary += "  Interpretation: Large effect (good separation)"
        elif separation_ratio > 0.5:
            detailed_summary += "  Interpretation: Medium effect (moderate separation)"
        else:
            detailed_summary += "  Interpretation: Small effect (limited separation)"
    
    # Debug logging for returned figures
    print(f"[DEBUG] PCA return values:")
    print(f"[DEBUG]   primary_figure (scores): {fig1 is not None}")
    print(f"[DEBUG]   loadings_figure: {fig_loadings is not None}")
    print(f"[DEBUG]   distributions_figure: {fig_distributions is not None}")
    
    if fig_loadings is None:
        print(f"[DEBUG] WARNING: loadings_figure is None! show_loadings={show_loadings}")
    
    print(f"[DEBUG] PCA return values:")
    print(f"[DEBUG]   primary_figure (scores): {fig1 is not None}")
    print(f"[DEBUG]   scree_figure: {fig_scree is not None}")
    print(f"[DEBUG]   loadings_figure: {fig_loadings is not None}")
    print(f"[DEBUG]   biplot_figure: {fig_biplot is not None}")
    print(f"[DEBUG]   cumulative_variance_figure: {fig_cumvar is not None}")
    print(f"[DEBUG]   distributions_figure: {fig_distributions is not None}")
    
    return {
        "primary_figure": fig1,  # Scores plot with confidence ellipses (PC1 vs PC2)
        "scree_figure": fig_scree,  # Scree plot (variance explained per PC)
        "loadings_figure": fig_loadings,  # Loadings plot (spectral features)
        "biplot_figure": fig_biplot,  # Biplot (scores + loading arrows)
        "cumulative_variance_figure": fig_cumvar,  # Cumulative variance plot
        "distributions_figure": fig_distributions,  # Score distributions (separate tab)
        "secondary_figure": None,  # Deprecated - kept for compatibility
        "data_table": scores_df,
        "summary_text": summary,
        "detailed_summary": detailed_summary,
        "raw_results": {
            "pca_model": pca,
            "scores": scores,
            "loadings": pca.components_,
            "explained_variance": pca.explained_variance_ratio_,
            "labels": labels,
            "unique_labels": unique_labels
        }
    }



def perform_umap_analysis(dataset_data: Dict[str, pd.DataFrame],
                         params: Dict[str, Any],
                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Perform UMAP dimensionality reduction with multi-dataset support.
    
    Similar to PCA, supports group assignment for classification tasks.
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - n_neighbors: Number of neighbors (default 15)
            - min_dist: Minimum distance (default 0.1)
            - n_components: Number of components (default 2)
            - metric: Distance metric (default 'euclidean')
            - _group_labels: Optional {dataset_name: group_label} mapping
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with embedding plot and results
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is not installed. Install with: pip install umap-learn")
    
    if progress_callback:
        progress_callback(10)
    
    # Get parameters
    n_neighbors = params.get("n_neighbors", 15)
    min_dist = params.get("min_dist", 0.1)
    n_components = params.get("n_components", 2)
    metric = params.get("metric", "euclidean")
    group_labels_map = params.get("_group_labels", None)  # {dataset_name: group_label}
    
    print(f"[DEBUG] UMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
    print(f"[DEBUG] Group labels map: {group_labels_map}")
    
    # ✅ FIX: Use interpolation to handle datasets with different wavenumber ranges
    # This resolves "ValueError: all input array dimensions except for concatenation axis must match"
    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data, 
        group_labels_map=group_labels_map,
        method='linear'
    )
    
    print(f"[DEBUG] Combined matrix shape after interpolation: {X.shape}")
    print(f"[DEBUG] Unique labels: {sorted(set(labels))}")
    
    if progress_callback:
        progress_callback(30)
    
    # Perform UMAP
    print(f"[DEBUG] Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=42
    )
    embedding = reducer.fit_transform(X)
    
    if progress_callback:
        progress_callback(80)
    
    # ✅ FIX: Use high-contrast color palette for clear visual distinction
    fig, ax = plt.subplots(figsize=(12, 10))
    
    unique_labels = sorted(set(labels))
    num_groups = len(unique_labels)
    
    # Use unified high-contrast color palette
    colors = get_high_contrast_colors(num_groups)
    
    print(f"[DEBUG] Plotting {num_groups} groups with high-contrast colors")
    
    for i, dataset_label in enumerate(unique_labels):
        mask = np.array([l == dataset_label for l in labels])
        num_points = np.sum(mask)
        print(f"[DEBUG] Group '{dataset_label}': {num_points} spectra")
        
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                  color=colors[i], label=dataset_label,
                  alpha=0.7, s=100, edgecolors='white', linewidth=1.5)
    
    ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
    ax.set_title('UMAP Projection', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.95, fontsize=10, 
              edgecolor='#cccccc', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Create data table
    embedding_df = pd.DataFrame(
        embedding,
        columns=[f'UMAP{i+1}' for i in range(n_components)]
    )
    embedding_df['Dataset'] = labels
    
    summary = f"UMAP completed with {n_components} components.\n"
    summary += f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}"
    
    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": embedding_df,
        "summary_text": summary,
        "detailed_summary": f"Total spectra: {X.shape[0]}",
        "raw_results": {"embedding": embedding, "reducer": reducer},
        "loadings_figure": None  # UMAP does not produce loadings
    }


def perform_tsne_analysis(dataset_data: Dict[str, pd.DataFrame],
                         params: Dict[str, Any],
                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Perform t-SNE dimensionality reduction.
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - perplexity: Perplexity parameter (default 30)
            - learning_rate: Learning rate (default 200)
            - n_iter: Number of iterations (default 1000)
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with embedding plot and results
    """
    if progress_callback:
        progress_callback(10)
    
    # Get parameters
    perplexity = params.get("perplexity", 30)
    learning_rate = params.get("learning_rate", 200)
    n_iter = params.get("n_iter", 1000)
    group_labels_map = params.get("_group_labels", None)  # {dataset_name: group_label}
    
    print(f"[DEBUG] t-SNE parameters: perplexity={perplexity}, learning_rate={learning_rate}")
    print(f"[DEBUG] t-SNE n_iter={n_iter} (will use as max_iter for sklearn)")
    
    # ✅ FIX: Use interpolation to handle datasets with different wavenumber ranges
    # This resolves "ValueError: all input array dimensions except for concatenation axis must match"
    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data, 
        group_labels_map=group_labels_map,
        method='linear'
    )
    
    n_samples = X.shape[0]
    print(f"[DEBUG] Combined matrix shape after interpolation: {X.shape}")
    
    # CRITICAL FIX: Perplexity must be less than n_samples
    if perplexity >= n_samples:
        new_perplexity = max(1, n_samples - 1)
        print(f"[DEBUG] Adjusting perplexity from {perplexity} to {new_perplexity} (n_samples={n_samples})")
        perplexity = new_perplexity
    
    if progress_callback:
        progress_callback(30)
    
    # Perform t-SNE (sklearn uses max_iter, not n_iter)
    print(f"[DEBUG] Creating TSNE with max_iter={n_iter}")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,  # CRITICAL FIX: sklearn uses max_iter not n_iter
        random_state=42
    )
    embedding = tsne.fit_transform(X)
    
    if progress_callback:
        progress_callback(80)
    
    # ✅ FIX: Use high-contrast color palette for clear visual distinction
    # User requested: "distinct colours (eg: red and blue, green and red, yellow and blue)"
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = sorted(set(labels))
    num_groups = len(unique_labels)
    
    # Use unified high-contrast color palette (same as PCA and UMAP)
    colors = get_high_contrast_colors(num_groups)
    print(f"[DEBUG] t-SNE using high-contrast colors for {num_groups} groups: {colors}")
    
    for i, dataset_label in enumerate(unique_labels):
        mask = np.array([l == dataset_label for l in labels])
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                  color=colors[i], label=dataset_label,
                  alpha=0.7, s=100, edgecolors='white', linewidth=1.0)
    
    ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
    ax.set_title('t-SNE Projection', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.95, fontsize=10,
              edgecolor='#cccccc', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Create data table
    embedding_df = pd.DataFrame(embedding, columns=['tSNE1', 'tSNE2'])
    embedding_df['Dataset'] = labels
    
    summary = f"t-SNE completed with 2 components.\n"
    summary += f"Parameters: perplexity={perplexity}, learning_rate={learning_rate}, n_iter={n_iter}"
    
    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": embedding_df,
        "summary_text": summary,
        "detailed_summary": f"Total spectra: {X.shape[0]}",
        "detailed_summary": f"Total spectra: {X.shape[0]}",
        "raw_results": {"embedding": embedding},
        "loadings_figure": None  # t-SNE does not produce loadings
    }


def perform_hierarchical_clustering(dataset_data: Dict[str, pd.DataFrame],
                                   params: Dict[str, Any],
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Perform hierarchical clustering analysis.
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - linkage_method: Linkage method (default 'ward')
            - distance_metric: Distance metric (default 'euclidean')
            - n_clusters: Number of clusters to color (optional)
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with dendrogram and results
    """
    if progress_callback:
        progress_callback(10)
    
    # Get parameters
    linkage_method = params.get("linkage_method", "ward")
    distance_metric = params.get("distance_metric", "euclidean")
    n_clusters = params.get("n_clusters", None)
    
    # ✅ FIX: Use interpolation to handle datasets with different wavenumber ranges
    # This resolves "ValueError: all input array dimensions except for concatenation axis must match"
    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data, 
        group_labels_map=None,  # Hierarchical clustering uses dataset names as labels
        method='linear'
    )
    
    print(f"[DEBUG] Hierarchical clustering: Combined matrix shape after interpolation: {X.shape}")
    
    if progress_callback:
        progress_callback(40)
    
    # Perform hierarchical clustering
    if linkage_method == 'ward':
        Z = linkage(X, method='ward')
    else:
        distances = pdist(X, metric=distance_metric)
        Z = linkage(distances, method=linkage_method)
    
    if progress_callback:
        progress_callback(70)
    
    # Create dendrogram
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate color threshold (70% of max distance)
    max_d = np.max(Z[:, 2])
    color_threshold = 0.7 * max_d
    
    # Plot dendrogram with improved visualization
    dend = dendrogram(
        Z, 
        ax=ax, 
        labels=labels if params.get("show_labels", False) else None,
        leaf_font_size=8,
        color_threshold=color_threshold,
        above_threshold_color='#bcbcbc'  # Light gray for upper links
    )
    
    # Add threshold line
    ax.axhline(y=color_threshold, c='r', lw=1, linestyle='--', alpha=0.5, label='Color Threshold')
    
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance', fontsize=12, fontweight='bold')
    ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    if params.get("show_labels", False):
        plt.setp(ax.get_xticklabels(), rotation=90)
    
    plt.tight_layout()
    
    summary = f"Hierarchical clustering completed.\n"
    summary += f"Linkage: {linkage_method}, Distance metric: {distance_metric}\n"
    summary += f"Total spectra: {X.shape[0]}"
    
    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": None,
        "summary_text": summary,
        "detailed_summary": f"Linkage matrix shape: {Z.shape}",
        "raw_results": {"linkage_matrix": Z, "labels": labels},
        "loadings_figure": None  # Clustering does not produce loadings
    }


def perform_kmeans_clustering(dataset_data: Dict[str, pd.DataFrame],
                              params: Dict[str, Any],
                              progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Perform K-means clustering analysis.
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - n_clusters: Number of clusters (default 3)
            - max_iter: Maximum iterations (default 300)
            - n_init: Number of initializations (default 10)
            - show_pca: Show clusters in PCA space
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with cluster visualization and results
    """
    if progress_callback:
        progress_callback(10)
    
    # Get parameters
    n_clusters = params.get("n_clusters", 3)
    max_iter = params.get("max_iter", 300)
    n_init = params.get("n_init", 10)
    show_pca = params.get("show_pca", True)
    
    print(f"[DEBUG] K-Means parameters received:")
    print(f"[DEBUG]   n_clusters = {n_clusters} (type: {type(n_clusters).__name__})")
    print(f"[DEBUG]   n_init = {n_init} (type: {type(n_init).__name__})")
    print(f"[DEBUG]   max_iter = {max_iter} (type: {type(max_iter).__name__})")
    print(f"[DEBUG]   show_pca = {show_pca}")
    
    # ✅ FIX: Use interpolation to handle datasets with different wavenumber ranges
    # This resolves "ValueError: all input array dimensions except for concatenation axis must match"
    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data, 
        group_labels_map=None,  # K-Means uses dataset names as labels
        method='linear'
    )
    
    print(f"[DEBUG] K-Means: Combined matrix shape after interpolation: {X.shape}")
    
    if progress_callback:
        progress_callback(30)
    
    # Perform K-means clustering
    print(f"[DEBUG] Creating KMeans with n_clusters={n_clusters}, n_init={n_init}, max_iter={max_iter}")
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter,
                   n_init=n_init, random_state=42)
    print(f"[DEBUG] Fitting KMeans model...")
    cluster_labels = kmeans.fit_predict(X)
    print(f"[DEBUG] KMeans completed. Inertia: {kmeans.inertia_:.2f}")
    
    if progress_callback:
        progress_callback(60)
    
    # Create visualization
    if show_pca:
        # Project to PCA space for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        centers_pca = pca.transform(kmeans.cluster_centers_)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # ✅ FIX: Use high-contrast colors for cluster visualization
        colors = get_high_contrast_colors(n_clusters)
        for i in range(n_clusters):
            mask = cluster_labels == i
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[colors[i]], label=f'Cluster {i+1}',
                      alpha=0.7, s=50)
        
        # Plot centroids
        ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
                  c='red', marker='X', s=200, edgecolors='black',
                  linewidths=2, label='Centroids')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title('K-means Clustering (PCA Projection)', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    else:
        # Just show cluster assignments
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(cluster_labels)), cluster_labels)
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Cluster ID', fontsize=12)
        ax.set_title('K-means Cluster Assignments', fontsize=14, fontweight='bold')
    
    if progress_callback:
        progress_callback(90)
    
    # Create data table
    results_df = pd.DataFrame({
        'Dataset': labels,
        'Cluster': cluster_labels
    })
    
    # Calculate cluster statistics
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    
    summary = f"K-means clustering completed with {n_clusters} clusters.\n"
    for i in range(n_clusters):
        count = cluster_counts.get(i, 0)
        pct = count / len(cluster_labels) * 100
        summary += f"Cluster {i+1}: {count} spectra ({pct:.1f}%)\n"
    
    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": results_df,
        "summary_text": summary,
        "detailed_summary": f"Inertia: {kmeans.inertia_:.2f}\nIterations: {kmeans.n_iter_}",
        "raw_results": {
            "kmeans_model": kmeans,
            "cluster_labels": cluster_labels,
            "cluster_centers": kmeans.cluster_centers_
        },
        "loadings_figure": None  # Clustering does not produce loadings
    }


def create_spectrum_preview_figure(dataset_data: Dict[str, pd.DataFrame]) -> Figure:
    """
    Create a preview figure showing MEAN SPECTRA ONLY with vertical offset stacking.
    
    ✅ FIX #5 (P1): Mean-only display reduces visual clutter
    Consensus from 6 AI analyses: Show only mean, not all 74 individual spectra
    Implements RAMANMETRIX (2018) spectral stacking standard: 15% offset
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
    
    Returns:
        Matplotlib Figure object
    """
    # ✅ FIX: Correct unpacking of plt.subplots (returns tuple of fig, ax)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Colorblind-safe palette (Tableau 10)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    offset = 0
    max_intensity_overall = 0
    
    for idx, (dataset_name, df) in enumerate(dataset_data.items()):
        # Calculate statistics
        mean_spectrum = df.mean(axis=1).values  # Mean across columns (spectra)
        std_spectrum = df.std(axis=1).values
        wavenumbers = df.index.values
        n_spectra = df.shape[1]
        
        # Apply vertical offset for stacking
        mean_with_offset = mean_spectrum + offset
        
        # Plot MEAN line only (bold, prominent)
        ax.plot(
            wavenumbers, mean_with_offset,
            color=colors[idx % len(colors)],
            linewidth=2.8,  # Thick for visibility
            label=f'{dataset_name} (mean, n={n_spectra})',
            alpha=0.95,
            zorder=10 + idx
        )
        
        # Optional: Add VERY subtle ±0.5σ envelope (barely visible)
        ax.fill_between(
            wavenumbers,
            mean_with_offset - std_spectrum * 0.5,
            mean_with_offset + std_spectrum * 0.5,
            color=colors[idx % len(colors)],
            alpha=0.08,  # Barely visible (8% opacity)
            edgecolor='none',
            zorder=5 + idx
        )
        
        # Calculate next offset (15% above max intensity)
        max_intensity = (mean_with_offset + std_spectrum).max()
        offset = max_intensity * 1.15  # 15% spacing (RAMANMETRIX standard)
        max_intensity_overall = max(max_intensity_overall, max_intensity)
    
    # Styling
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Intensity (offset for clarity)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Spectra (Vertically Stacked)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.invert_xaxis()  # Raman convention: high → low wavenumber
    
    # Adjust y-limits
    ax.set_ylim(-max_intensity_overall * 0.05, offset + max_intensity_overall * 0.05)
    
    fig.tight_layout(pad=1.2)
    return fig
