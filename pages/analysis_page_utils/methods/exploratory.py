"""
Exploratory Analysis Methods

This module implements exploratory data analysis methods like PCA, UMAP,
t-SNE, and clustering techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional
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

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def add_confidence_ellipse(ax, x, y, n_std=1.96, facecolor='none', edgecolor='red', linestyle='--', linewidth=2, alpha=0.7, label=None):
    """
    Add a confidence ellipse to a matplotlib axis.
    
    For Raman spectroscopy Chemometrics, 95% confidence ellipses (n_std=1.96) are critical
    for proving statistical group separation in PCA plots.
    
    Args:
        ax: matplotlib axis object
        x, y: Data coordinates (numpy arrays)
        n_std: Number of standard deviations (1.96 for 95% CI)
        facecolor, edgecolor, linestyle, linewidth, alpha: matplotlib styling
        label: Legend label for the ellipse
    
    Returns:
        Ellipse patch object
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
    
    # Create ellipse
    ellipse = Ellipse(xy=(mean_x, mean_y), width=width, height=height, angle=angle,
                      facecolor=facecolor, edgecolor=edgecolor, linestyle=linestyle,
                      linewidth=linewidth, alpha=alpha, label=label)
    
    ax.add_patch(ellipse)
    print(f"[DEBUG] Ellipse added to axis at ({mean_x:.2f}, {mean_y:.2f}), size: {width:.2f}x{height:.2f}")
    return ellipse


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
    
    # CRITICAL: Concatenate ALL datasets into ONE matrix for group comparison
    all_spectra = []
    labels = []
    
    for dataset_name, df in dataset_data.items():
        spectra_matrix = df.values.T  # Shape: (n_spectra, n_wavenumbers)
        all_spectra.append(spectra_matrix)
        
        # Use group label if available, otherwise use dataset name
        if group_labels_map and dataset_name in group_labels_map:
            label = group_labels_map[dataset_name]
        else:
            label = dataset_name
        
        labels.extend([label] * spectra_matrix.shape[0])
    
    X = np.vstack(all_spectra)  # Combined matrix: (total_spectra, n_wavenumbers)
    wavenumbers = dataset_data[list(dataset_data.keys())[0]].index.values
    
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
    
    # Title changes based on whether ellipses are shown
    if show_ellipses:
        ax1.set_title('PCA Score Plot with 95% Confidence Ellipses', fontsize=14, fontweight='bold')
    else:
        ax1.set_title('PCA Score Plot', fontsize=14, fontweight='bold')
    
    # Larger legend with better visibility
    ax1.legend(loc='best', framealpha=0.95, fontsize=10, 
              edgecolor='#cccccc', fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    print("[DEBUG] PCA scores plot created successfully")
    
    if progress_callback:
        progress_callback(75)
    
    # === FIGURE 2: Loadings Plot (Spectral interpretation) ===
    print(f"[DEBUG] show_loadings parameter: {show_loadings}")
    fig_loadings = None
    if show_loadings:
        print("[DEBUG] Creating loadings figure...")
        fig_loadings, ax_loadings = plt.subplots(figsize=(12, 6))
        
        ax_loadings.plot(wavenumbers, pca.components_[0], label='PC1', linewidth=2, color='#1f77b4')
        ax_loadings.plot(wavenumbers, pca.components_[1], label='PC2', linewidth=2, color='#ff7f0e')
        if n_components >= 3:
            ax_loadings.plot(wavenumbers, pca.components_[2], label='PC3', linewidth=2, color='#2ca02c')
        
        ax_loadings.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12, fontweight='bold')
        ax_loadings.set_ylabel('Loading Value', fontsize=12, fontweight='bold')
        ax_loadings.set_title('PCA Loadings (Spectral Features)', fontsize=14, fontweight='bold')
        ax_loadings.legend(loc='best', fontsize=11)
        ax_loadings.grid(True, alpha=0.3)
        ax_loadings.invert_xaxis()  # Raman convention: high to low wavenumber
        ax_loadings.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        fig_loadings.tight_layout()
        print("[DEBUG] Loadings figure created successfully")
    else:
        print("[DEBUG] Loadings figure skipped (show_loadings=False)")
    
    if progress_callback:
        progress_callback(80)
    
    # === FIGURE 3: Score Distributions (CRITICAL for Raman classification) ===
    fig_distributions = None
    if show_distributions and len(unique_labels) > 1:
        # Create grid for PC1, PC2, PC3 distributions
        n_pcs_to_plot = min(3, n_components)
        fig_distributions, axes = plt.subplots(1, n_pcs_to_plot, figsize=(6*n_pcs_to_plot, 5))
        if n_pcs_to_plot == 1:
            axes = [axes]  # Make it iterable
        
        fig_distributions.suptitle('PC Score Distributions', fontsize=16, fontweight='bold')
        
        # Plot distributions for PC1, PC2, PC3
        for pc_idx in range(n_pcs_to_plot):
            ax = axes[pc_idx]
            
            # Plot histogram/KDE for each dataset
            for i, dataset_label in enumerate(unique_labels):
                mask = np.array([l == dataset_label for l in labels])
                pc_scores = scores[mask, pc_idx]
                
                # Calculate KDE (Kernel Density Estimation)
                kde = stats.gaussian_kde(pc_scores)
                x_range = np.linspace(pc_scores.min() - 1, pc_scores.max() + 1, 200)
                kde_values = kde(x_range)
                
                # Plot KDE curve
                ax.plot(x_range, kde_values, color=colors[i], linewidth=2.5,
                       label=dataset_label, alpha=0.9)
                
                # Fill under curve for visibility
                ax.fill_between(x_range, kde_values, alpha=0.25, color=colors[i])
                
                # Add histogram for reference
                ax.hist(pc_scores, bins=20, density=True, alpha=0.15,
                       color=colors[i], edgecolor='white', linewidth=0.5)
            
            # Statistical test (Mann-Whitney U for 2 groups)
            if len(unique_labels) == 2:
                mask1 = np.array([l == unique_labels[0] for l in labels])
                mask2 = np.array([l == unique_labels[1] for l in labels])
                pc1_scores = scores[mask1, pc_idx]
                pc2_scores = scores[mask2, pc_idx]
                
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
            
            # Formatting
            ax.set_xlabel(f'PC{pc_idx+1} Score', fontsize=12, fontweight='bold')
            ax.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax.set_title(f'PC{pc_idx+1} ({pca.explained_variance_ratio_[pc_idx]*100:.1f}%)',
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
    
    if progress_callback:
        progress_callback(90)
    
    # Create data table with dataset labels
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    scores_df = pd.DataFrame(scores, columns=pc_columns)
    scores_df['Dataset'] = labels
    
    # Summary text
    n_datasets = len(unique_labels)
    total_spectra = X.shape[0]
    total_variance = np.sum(pca.explained_variance_ratio_[:3]) * 100
    
    summary = f"PCA completed with {n_components} components on {n_datasets} dataset(s) ({total_spectra} spectra).\n"
    summary += f"First 3 PCs explain {total_variance:.1f}% of variance.\n"
    summary += f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, "
    summary += f"PC2: {pca.explained_variance_ratio_[1]*100:.1f}%"
    if n_components >= 3:
        summary += f", PC3: {pca.explained_variance_ratio_[2]*100:.1f}%"
    summary += f"\n\nDatasets: {', '.join(unique_labels)}"
    
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
    
    return {
        "primary_figure": fig1,  # Scores plot with confidence ellipses
        "loadings_figure": fig_loadings,  # Loadings plot (separate tab)
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
    
    # Combine all datasets (like PCA approach)
    all_spectra = []
    labels = []
    
    for dataset_name, df in dataset_data.items():
        spectra_matrix = df.values.T
        all_spectra.append(spectra_matrix)
        
        # Use group label if available, otherwise use dataset name
        if group_labels_map and dataset_name in group_labels_map:
            label = group_labels_map[dataset_name]
        else:
            label = dataset_name
        
        labels.extend([label] * spectra_matrix.shape[0])
    
    X = np.vstack(all_spectra)
    
    print(f"[DEBUG] Combined matrix shape: {X.shape}")
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
    
    # Create figure with high-contrast colors (like PCA)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    unique_labels = sorted(set(labels))
    num_groups = len(unique_labels)
    
    # Use same color scheme as PCA for consistency
    if num_groups == 2:
        colors = ['#0066cc', '#ffd700']  # Blue and Gold
    elif num_groups == 3:
        colors = ['#0066cc', '#cc0000', '#00cc66']  # Blue, Red, Green
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, num_groups))
    
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
        "raw_results": {"embedding": embedding, "reducer": reducer}
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
    
    print(f"[DEBUG] t-SNE parameters: perplexity={perplexity}, learning_rate={learning_rate}")
    print(f"[DEBUG] t-SNE n_iter={n_iter} (will use as max_iter for sklearn)")
    
    # Combine all datasets
    all_spectra = []
    labels = []
    
    for dataset_name, df in dataset_data.items():
        spectra_matrix = df.values.T
        all_spectra.append(spectra_matrix)
        labels.extend([dataset_name] * spectra_matrix.shape[0])
    
    X = np.vstack(all_spectra)
    
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, dataset_label in enumerate(unique_labels):
        mask = np.array([l == dataset_label for l in labels])
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                  c=[colors[i]], label=dataset_label,
                  alpha=0.7, s=50)
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE Projection', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
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
        "raw_results": {"embedding": embedding}
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
    
    # Combine all datasets
    all_spectra = []
    labels = []
    
    for dataset_name, df in dataset_data.items():
        spectra_matrix = df.values.T
        all_spectra.append(spectra_matrix)
        labels.extend([dataset_name] * spectra_matrix.shape[0])
    
    X = np.vstack(all_spectra)
    
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
    
    dendrogram(Z, ax=ax, labels=labels, leaf_font_size=8)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
    
    summary = f"Hierarchical clustering completed.\n"
    summary += f"Linkage: {linkage_method}, Distance metric: {distance_metric}\n"
    summary += f"Total spectra: {X.shape[0]}"
    
    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": None,
        "summary_text": summary,
        "detailed_summary": f"Linkage matrix shape: {Z.shape}",
        "raw_results": {"linkage_matrix": Z, "labels": labels}
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
    
    # Combine all datasets
    all_spectra = []
    labels = []
    
    for dataset_name, df in dataset_data.items():
        spectra_matrix = df.values.T
        all_spectra.append(spectra_matrix)
        labels.extend([dataset_name] * spectra_matrix.shape[0])
    
    X = np.vstack(all_spectra)
    
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
        
        # Plot clusters
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
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
        }
    }
