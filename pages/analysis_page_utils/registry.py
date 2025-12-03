"""
Analysis Methods Registry

This module defines all available analysis methods with their configurations,
parameters, and visualization functions. Methods are organized by category:
- Exploratory Analysis
- Statistical Analysis
- Visualization Methods
"""

from typing import Dict, Any, Callable


# Analysis Methods Registry
ANALYSIS_METHODS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "exploratory": {
        "pca": {
            "name": "PCA (Principal Component Analysis)",
            "description": "Dimensionality reduction using PCA to identify variance patterns. Select multiple datasets to compare groups (e.g., Control vs Disease).",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "n_components": {
                    "type": "spinbox",
                    "default": 3,
                    "range": (2, 100),  # Removed arbitrary limit - users should be free to choose based on their data
                    "label": "Number of Components"
                },
                "scaling": {
                    "type": "combo",
                    "options": ["StandardScaler", "MinMaxScaler", "None"],
                    "default": "StandardScaler",
                    "label": "Scaling Method"
                },
                "show_ellipses": {
                    "type": "checkbox",
                    "default": False,
                    "label": "Show 95% Confidence Ellipses"
                },
                "show_loadings": {
                    "type": "checkbox",
                    "default": False,
                    "label": "Show Loading Plot"
                },
                "max_loadings_components": {
                    "type": "spinbox",
                    "default": 3,
                    "range": (1, 5),
                    "label": "Loading Components to Plot (max 5)"
                },
                "show_scree": {
                    "type": "checkbox",
                    "default": False,
                    "label": "Show Scree Plot"
                },
                "show_distributions": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Score Distributions"
                },
                "n_distribution_components": {
                    "type": "spinbox",
                    "default": 3,
                    "range": (1, 6),
                    "label": "Distribution Components (max 6)"
                }
            },
            "function": "perform_pca_analysis"
        },
        "umap": {
            "name": "UMAP (Uniform Manifold Approximation)",
            "description": "Non-linear dimensionality reduction preserving local and global structure. Select multiple datasets to compare groups.",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "n_neighbors": {
                    "type": "spinbox",
                    "default": 15,
                    "range": (5, 100),
                    "label": "Number of Neighbors"
                },
                "min_dist": {
                    "type": "double_spinbox",
                    "default": 0.1,
                    "range": (0.0, 1.0),
                    "step": 0.05,
                    "label": "Minimum Distance"
                },
                "n_components": {
                    "type": "spinbox",
                    "default": 2,
                    "range": (2, 3),
                    "label": "Number of Dimensions"
                },
                "metric": {
                    "type": "combo",
                    "options": ["euclidean", "cosine", "manhattan", "correlation"],
                    "default": "euclidean",
                    "label": "Distance Metric"
                }
            },
            "function": "perform_umap_analysis"
        },
        "tsne": {
            "name": "t-SNE (t-Distributed Stochastic Neighbor Embedding)",
            "description": "Non-linear dimensionality reduction for cluster visualization",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "perplexity": {
                    "type": "spinbox",
                    "default": 30,
                    "range": (5, 100),
                    "label": "Perplexity"
                },
                "learning_rate": {
                    "type": "double_spinbox",
                    "default": 200.0,
                    "range": (10.0, 1000.0),
                    "step": 10.0,
                    "label": "Learning Rate"
                },
                "n_iter": {
                    "type": "spinbox",
                    "default": 1000,
                    "range": (250, 5000),
                    "label": "Max Iterations"
                }
            },
            "function": "perform_tsne_analysis"
        },
        "hierarchical_clustering": {
            "name": "Hierarchical Clustering with Dendrogram",
            "description": "Hierarchical cluster analysis with dendrogram visualization",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "n_clusters": {
                    "type": "spinbox",
                    "default": 3,
                    "range": (2, 20),
                    "label": "Number of Clusters"
                },
                "linkage_method": {
                    "type": "combo",
                    "options": ["ward", "complete", "average", "single"],
                    "default": "ward",
                    "label": "Linkage Method"
                },
                "distance_metric": {
                    "type": "combo",
                    "options": ["euclidean", "cosine", "manhattan", "correlation"],
                    "default": "euclidean",
                    "label": "Distance Metric"
                },
                "show_labels": {
                    "type": "checkbox",
                    "default": False,
                    "label": "Show Sample Labels"
                }
            },
            "function": "perform_hierarchical_clustering"
        },
        "kmeans": {
            "name": "K-Means Clustering",
            "description": "Partitioning clustering algorithm",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "n_clusters": {
                    "type": "spinbox",
                    "default": 3,
                    "range": (2, 20),
                    "label": "Number of Clusters"
                },
                "n_init": {
                    "type": "spinbox",
                    "default": 10,
                    "range": (1, 50),
                    "label": "Number of Initializations"
                },
                "max_iter": {
                    "type": "spinbox",
                    "default": 300,
                    "range": (10, 1000),
                    "label": "Max Iterations"
                },
                "show_elbow": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Elbow Plot"
                }
            },
            "function": "perform_kmeans_clustering"
        }
    },
    "statistical": {
        "spectral_comparison": {
            "name": "Group Mean Spectral Comparison",
            "description": "Compare mean spectra across groups with statistical testing",
            "min_datasets": 2,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "confidence_level": {
                    "type": "double_spinbox",
                    "default": 0.95,
                    "range": (0.80, 0.99),
                    "step": 0.01,
                    "label": "Confidence Level"
                },
                "fdr_correction": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Apply FDR Correction"
                },
                "show_ci": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Confidence Intervals"
                },
                "highlight_significant": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Highlight Significant Regions"
                }
            },
            "function": "perform_spectral_comparison"
        },
        "peak_analysis": {
            "name": "Peak Detection and Analysis",
            "description": "Automated peak detection with statistical comparison",
            "min_datasets": 1,
            "max_datasets": 1,
            "dataset_selection_mode": "single",
            "params": {
                "prominence_threshold": {
                    "type": "double_spinbox",
                    "default": 0.1,
                    "range": (0.01, 1.0),
                    "step": 0.01,
                    "label": "Prominence Threshold"
                },
                "width_min": {
                    "type": "spinbox",
                    "default": 5,
                    "range": (1, 50),
                    "label": "Minimum Peak Width"
                },
                "top_n_peaks": {
                    "type": "spinbox",
                    "default": 20,
                    "range": (5, 100),
                    "label": "Top N Peaks to Display"
                },
                "show_assignments": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Biochemical Assignments"
                }
            },
            "function": "perform_peak_analysis"
        },
        "correlation_analysis": {
            "name": "Spectral Correlation Analysis",
            "description": "Analyze correlations between spectral regions",
            "min_datasets": 1,
            "max_datasets": 1,
            "dataset_selection_mode": "single",
            "params": {
                "method": {
                    "type": "combo",
                    "options": ["pearson", "spearman", "kendall"],
                    "default": "pearson",
                    "label": "Correlation Method"
                },
                "show_heatmap": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Correlation Heatmap"
                },
                "threshold": {
                    "type": "double_spinbox",
                    "default": 0.7,
                    "range": (0.0, 1.0),
                    "step": 0.05,
                    "label": "Correlation Threshold"
                }
            },
            "function": "perform_correlation_analysis"
        },
        "anova_test": {
            "name": "ANOVA Statistical Test",
            "description": "One-way ANOVA across multiple groups",
            "min_datasets": 3,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "alpha": {
                    "type": "double_spinbox",
                    "default": 0.05,
                    "range": (0.01, 0.1),
                    "step": 0.01,
                    "label": "Significance Level (α)"
                },
                "post_hoc": {
                    "type": "combo",
                    "options": ["tukey", "bonferroni", "none"],
                    "default": "tukey",
                    "label": "Post-hoc Test"
                },
                "show_boxplot": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Box Plot"
                }
            },
            "function": "perform_anova_test"
        }
    },
    "visualization": {
        "heatmap": {
            "name": "Spectral Heatmap with Clustering",
            "description": "2D heatmap visualization with hierarchical clustering",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "cluster_rows": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Cluster Rows (Samples)"
                },
                "cluster_cols": {
                    "type": "checkbox",
                    "default": False,
                    "label": "Cluster Columns (Wavenumbers)"
                },
                "colormap": {
                    "type": "combo",
                    "options": ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "RdYlBu"],
                    "default": "viridis",
                    "label": "Colormap"
                },
                "normalize": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Normalize Intensities"
                },
                "show_dendrograms": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Dendrograms"
                }
            },
            "function": "create_spectral_heatmap"
        },
        "mean_spectra_overlay": {
            "name": "Mean Spectra Overlay Plot",
            "description": "Overlay mean spectra from different groups/datasets",
            "min_datasets": 2,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "show_std": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Standard Deviation"
                },
                "show_ci": {
                    "type": "checkbox",
                    "default": False,
                    "label": "Show Confidence Intervals"
                },
                "alpha_fill": {
                    "type": "double_spinbox",
                    "default": 0.2,
                    "range": (0.0, 1.0),
                    "step": 0.05,
                    "label": "Fill Transparency"
                },
                "line_width": {
                    "type": "double_spinbox",
                    "default": 1.5,
                    "range": (0.5, 5.0),
                    "step": 0.5,
                    "label": "Line Width"
                }
            },
            "function": "create_mean_spectra_overlay"
        },
        "waterfall_plot": {
            "name": "Waterfall Plot",
            "description": "visualization of multiple spectra",
            "min_datasets": 1,
            "max_datasets": 1,
            "dataset_selection_mode": "single",
            "params": {
                "offset_scale": {
                    "type": "double_spinbox",
                    "default": 1.0,
                    "range": (0.1, 5.0),
                    "step": 0.1,
                    "label": "Offset Scale"
                },
                "max_spectra": {
                    "type": "spinbox",
                    "default": 50,
                    "range": (10, 200),
                    "label": "Maximum Spectra to Display"
                },
                "colormap": {
                    "type": "combo",
                    "options": ["viridis", "plasma", "coolwarm", "rainbow"],
                    "default": "viridis",
                    "label": "Colormap"
                }
            },
            "function": "create_waterfall_plot"
        },
        "correlation_heatmap": {
            "name": "Correlation Heatmap",
            "description": "Heatmap of pairwise spectral correlations",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "method": {
                    "type": "combo",
                    "options": ["pearson", "spearman"],
                    "default": "pearson",
                    "label": "Correlation Method"
                },
                "colormap": {
                    "type": "combo",
                    "options": ["coolwarm", "RdYlBu", "RdBu", "seismic"],
                    "default": "coolwarm",
                    "label": "Colormap"
                },
                "show_values": {
                    "type": "checkbox",
                    "default": False,
                    "label": "Show Correlation Values"
                },
                "cluster": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Cluster Samples"
                }
            },
            "function": "create_correlation_heatmap"
        },
        "peak_intensity_scatter": {
            "name": "Peak Intensity Scatter Plot",
            "description": "2D/3D scatter plot of peak intensities",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "peak_1_position": {
                    "type": "spinbox",
                    "default": 1000,
                    "range": (400, 4000),
                    "label": "Peak 1 Position (cm⁻¹)"
                },
                "peak_2_position": {
                    "type": "spinbox",
                    "default": 1650,
                    "range": (400, 4000),
                    "label": "Peak 2 Position (cm⁻¹)"
                },
                "use_3d": {
                    "type": "checkbox",
                    "default": False,
                    "label": "3D Scatter (3 peaks)"
                },
                "peak_3_position": {
                    "type": "spinbox",
                    "default": 2900,
                    "range": (400, 4000),
                    "label": "Peak 3 Position (cm⁻¹)"
                },
                "show_legend": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Legend"
                }
            },
            "function": "create_peak_scatter"
        }
    }
}


def get_method_info(category: str, method_key: str) -> Dict[str, Any]:
    """
    Get information about a specific analysis method.
    
    Args:
        category: Analysis category
        method_key: Unique method identifier
    
    Returns:
        Method information dictionary
    
    Raises:
        KeyError: If category or method not found
    """
    if category not in ANALYSIS_METHODS:
        raise KeyError(f"Category '{category}' not found in registry")
    
    if method_key not in ANALYSIS_METHODS[category]:
        raise KeyError(f"Method '{method_key}' not found in category '{category}'")
    
    return ANALYSIS_METHODS[category][method_key]


def get_all_categories() -> list:
    """Get list of all available analysis categories."""
    return list(ANALYSIS_METHODS.keys())


def get_methods_in_category(category: str) -> Dict[str, Dict[str, Any]]:
    """
    Get all methods in a specific category.
    
    Args:
        category: Analysis category
    
    Returns:
        Dictionary of methods in the category
    """
    if category not in ANALYSIS_METHODS:
        raise KeyError(f"Category '{category}' not found in registry")
    
    return ANALYSIS_METHODS[category]
