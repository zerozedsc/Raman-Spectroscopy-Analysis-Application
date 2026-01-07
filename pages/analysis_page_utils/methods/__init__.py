"""
Analysis Method Implementations

This module provides all analysis method implementations that
operate on Raman spectral datasets.
"""

from .exploratory import (
    perform_pca_analysis,
    perform_umap_analysis,
    perform_tsne_analysis,
    perform_hierarchical_clustering,
    perform_kmeans_clustering,
)

from .statistical import (
    perform_spectral_comparison,
    perform_peak_analysis,
    perform_correlation_analysis,
    perform_anova_test,
)

# Import visualization functions from functions.visualization package
from functions.visualization import (
    create_spectral_heatmap,
    create_mean_spectra_overlay,
    create_waterfall_plot,
    create_correlation_heatmap,
    create_peak_scatter,
)

__all__ = [
    # Exploratory
    "perform_pca_analysis",
    "perform_umap_analysis",
    "perform_tsne_analysis",
    "perform_hierarchical_clustering",
    "perform_kmeans_clustering",
    # Statistical
    "perform_spectral_comparison",
    "perform_peak_analysis",
    "perform_correlation_analysis",
    "perform_anova_test",
    # Visualization
    "create_spectral_heatmap",
    "create_mean_spectra_overlay",
    "create_waterfall_plot",
    "create_correlation_heatmap",
    "create_peak_scatter",
]
