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

from .supervised import (
    perform_pls_da_analysis,
)

from .decomposition import (
    perform_mcr_als_analysis,
    perform_nmf_analysis,
    perform_ica_analysis,
)

from .qc import (
    perform_outlier_detection,
)

from .visualization_extra import (
    create_derivative_spectra_plot,
)

from .statistical import (
    perform_spectral_comparison,
    perform_peak_analysis,
    perform_correlation_analysis,
    perform_anova_test,
    perform_pairwise_statistical_tests,
    perform_band_ratio_analysis,
)

# Import visualization functions from functions.visualization package
from functions.visualization import (
    create_spectral_heatmap,
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
    # Supervised / Chemometrics
    "perform_pls_da_analysis",
    # Decomposition
    "perform_mcr_als_analysis",
    "perform_nmf_analysis",
    "perform_ica_analysis",
    # QC
    "perform_outlier_detection",
    # Statistical
    "perform_spectral_comparison",
    "perform_peak_analysis",
    "perform_correlation_analysis",
    "perform_anova_test",
    "perform_pairwise_statistical_tests",
    "perform_band_ratio_analysis",
    # Visualization (extra)
    "create_derivative_spectra_plot",
    # Visualization
    "create_spectral_heatmap",
    "create_waterfall_plot",
    "create_correlation_heatmap",
    "create_peak_scatter",
]
