"""
Visualization package for Raman spectroscopy analysis.

This package provides comprehensive visualization capabilities for Raman spectral data,
including basic plotting, peak analysis, dimensionality reduction, ML explainability,
and figure management.

Main Components:
    - RamanVisualizer: Main visualization class with all methods
    - FigureManager: Figure management and export utilities
    - Standalone functions: Specialized plotting functions
    - Peak assignment: Raman peak database queries (functions.visualization.peak_assignment)
    - Basic plots: Simple visualization utilities (functions.visualization.basic_plots)
    - Model evaluation: ML model assessment plots (functions.visualization.model_evaluation)
    - ML visualization: PCA/t-SNE/UMAP plots (functions.visualization.ml_visualization) [Phase 2]
    - Explainability: SHAP model explanations (functions.visualization.explainability) [Phase 3]
    - Interactive inspection: Spectrum inspection with SHAP/LIME (functions.visualization.interactive_inspection) [Phase 4A+4B]
    - LIME analysis: LIME explainability for model predictions (functions.visualization.lime_analysis) [Phase 5]

Usage:
    >>> from functions.visualization import RamanVisualizer, FigureManager
    >>> viz = RamanVisualizer(df=dataframe)
    >>> viz.visualize_raman_spectra()
    
    >>> # Direct function imports (Phase 1, 2, 3, 4 & 5 refactoring - backward compatible)
    >>> from functions.visualization import get_peak_assignment, visualize_raman_spectra
    >>> from functions.visualization import pca2d  # Phase 2
    >>> from functions.visualization import shap_explain  # Phase 3
    >>> from functions.visualization import inspect_spectra  # Phase 4A
    >>> from functions.visualization import spectrum_with_highlights_spectrum  # Phase 4B
    >>> from functions.visualization import create_enhanced_table  # Phase 4B
    >>> from functions.visualization import lime_explain  # Phase 5
    
For backward compatibility, all imports from the original visualization.py
are available directly from this package.
"""
# Import everything from core for backward compatibility
from .core import (
    RamanVisualizer,
)

# Import FigureManager
from .figure_manager import FigureManager, add_figure_manager_to_raman_pipeline

# Import extracted modules (Phase 1 refactoring)
from .peak_assignment import (
    get_peak_assignment,
    get_multiple_peak_assignments,
    find_peaks_in_range,
    clear_cache as clear_peak_cache
)

from .basic_plots import (
    visualize_raman_spectra,
    visualize_processed_spectra,
    extract_raman_characteristics
)

from .model_evaluation import (
    confusion_matrix_heatmap,
)

# Import extracted modules (Phase 2 refactoring)
from .ml_visualization import (
    pca2d,
)

# Import extracted modules (Phase 3 refactoring)
from .explainability import (
    shap_explain,
)

# Import extracted modules (Phase 4A refactoring)
from .interactive_inspection import (
    inspect_spectra,
)

# Import extracted modules (Phase 4B refactoring)
from .interactive_inspection import (
    spectrum_with_highlights_spectrum,
    create_shap_plots,
    create_enhanced_table,
    plot_institution_distribution,
)

# Import extracted modules (Phase 5 refactoring)
from .lime_analysis import (
    lime_explain,
)

# Import analysis plot functions (November 2025 refactoring)
from .analysis_plots import (
    create_spectral_heatmap,
    create_mean_spectra_overlay,
    create_waterfall_plot,
    create_correlation_heatmap,
    create_peak_scatter,
)

# Define public API
__all__ = [
    # Main class
    'RamanVisualizer',
    
    # Figure management
    'FigureManager',
    'add_figure_manager_to_raman_pipeline',
    
    # Peak assignment functions (Phase 1)
    'get_peak_assignment',
    'get_multiple_peak_assignments',
    'find_peaks_in_range',
    'clear_peak_cache',
    
    # Basic plotting functions (Phase 1)
    'visualize_raman_spectra',
    'visualize_processed_spectra',
    'extract_raman_characteristics',
    
    # Model evaluation functions (Phase 1)
    'confusion_matrix_heatmap',
    
    # ML visualization functions (Phase 2)
    'pca2d',
    
    # Explainability functions (Phase 3)
    'shap_explain',
    
    # Interactive inspection functions (Phase 4A)
    'inspect_spectra',
    
    # Interactive inspection helper functions (Phase 4B)
    'spectrum_with_highlights_spectrum',
    'create_shap_plots',
    'create_enhanced_table',
    'plot_institution_distribution',
    
    # LIME explainability functions (Phase 5)
    'lime_explain',
    
    # Analysis plot functions (November 2025 refactoring)
    'create_spectral_heatmap',
    'create_mean_spectra_overlay',
    'create_waterfall_plot',
    'create_correlation_heatmap',
    'create_peak_scatter',
]

# Version info
__version__ = '2.1.0'
__author__ = 'Raman Analysis Team'
