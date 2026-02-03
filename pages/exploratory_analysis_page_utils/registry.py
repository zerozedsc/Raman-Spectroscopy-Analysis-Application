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
                    "range": (
                        2,
                        100,
                    ),  # Removed arbitrary limit - users should be free to choose based on their data
                    "label": "Number of Components",
                },
                "scaling": {
                    "type": "combo",
                    "options": ["StandardScaler", "MinMaxScaler", "None"],
                    "default": "StandardScaler",
                    "label": "Scaling Method",
                },
                "show_ellipses": {
                    "type": "checkbox",
                    "default": False,
                    "label": "Show 95% Confidence Ellipses",
                },
                "show_loadings": {
                    "type": "checkbox",
                    "default": False,
                    "label": "Show Loading Plot",
                },
                "max_loadings_components": {
                    "type": "spinbox",
                    "default": 1,
                    "range": (1, 100),
                    "label": "Loading Components to Plot",
                },
                "show_scree": {
                    "type": "checkbox",
                    "default": False,
                    "label": "Show Scree Plot",
                },
                "show_distributions": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Score Distributions",
                },
                "n_distribution_components": {
                    "type": "spinbox",
                    "default": 1,
                    "range": (1, 100),
                    "label": "Distribution Components",
                },
                "enable_pca_lda": {
                    "type": "checkbox",
                    "default": False,
                    "label": "Enable PCA→LDA (decision boundary)",
                },
                "pca_lda_cv_folds": {
                    "type": "spinbox",
                    "default": 5,
                    "range": (2, 20),
                    "label": "PCA→LDA CV folds",
                },
            },
            "function": "perform_pca_analysis",
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
                    "label": "Number of Neighbors",
                },
                "min_dist": {
                    "type": "double_spinbox",
                    "default": 0.1,
                    "range": (0.0, 1.0),
                    "step": 0.05,
                    "label": "Minimum Distance",
                },
                "n_components": {
                    "type": "spinbox",
                    "default": 2,
                    "range": (2, 3),
                    "label": "Number of Dimensions",
                },
                "metric": {
                    "type": "combo",
                    "options": ["euclidean", "cosine", "manhattan", "correlation"],
                    "default": "euclidean",
                    "label": "Distance Metric",
                },
                "random_seed": {
                    "type": "spinbox",
                    "default": 42,
                    "range": (0, 2147483647),
                    "label": "Random Seed (for reproducibility)",
                },
            },
            "function": "perform_umap_analysis",
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
                    "label": "Perplexity",
                },
                "learning_rate": {
                    "type": "double_spinbox",
                    "default": 200.0,
                    "range": (10.0, 1000.0),
                    "step": 10.0,
                    "label": "Learning Rate",
                },
                "n_iter": {
                    "type": "spinbox",
                    "default": 1000,
                    "range": (250, 5000),
                    "label": "Max Iterations",
                },
                "random_seed": {
                    "type": "spinbox",
                    "default": 42,
                    "range": (0, 2147483647),
                    "label": "Random Seed (for reproducibility)",
                },
            },
            "function": "perform_tsne_analysis",
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
                    "label": "Number of Clusters",
                },
                "linkage_method": {
                    "type": "combo",
                    "options": ["ward", "complete", "average", "single"],
                    "default": "ward",
                    "label": "Linkage Method",
                },
                "distance_metric": {
                    "type": "combo",
                    "options": ["euclidean", "cosine", "manhattan", "correlation"],
                    "default": "euclidean",
                    "label": "Distance Metric",
                },
                "show_labels": {
                    "type": "checkbox",
                    "default": False,
                    "label": "Show Sample Labels",
                },
            },
            "function": "perform_hierarchical_clustering",
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
                    "label": "Number of Clusters",
                },
                "n_init": {
                    "type": "spinbox",
                    "default": 10,
                    "range": (1, 50),
                    "label": "Number of Initializations",
                },
                "max_iter": {
                    "type": "spinbox",
                    "default": 300,
                    "range": (10, 1000),
                    "label": "Max Iterations",
                },
                "show_elbow": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Elbow Plot",
                },
                "elbow_max_k": {
                    "type": "spinbox",
                    "default": 10,
                    "range": (3, 30),
                    "label": "Elbow Plot Max k",
                },
                "show_silhouette": {
                    "type": "checkbox",
                    "default": False,
                    "label": "Show Silhouette Validation",
                },
                "silhouette_k_min": {
                    "type": "spinbox",
                    "default": 2,
                    "range": (2, 50),
                    "label": "Silhouette Min k",
                },
                "silhouette_k_max": {
                    "type": "spinbox",
                    "default": 10,
                    "range": (2, 100),
                    "label": "Silhouette Max k",
                },
                "silhouette_use_pca": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Silhouette: Use PCA (faster)",
                },
                "silhouette_pca_components": {
                    "type": "spinbox",
                    "default": 10,
                    "range": (2, 100),
                    "label": "Silhouette PCA Components",
                },
                "random_seed": {
                    "type": "spinbox",
                    "default": 42,
                    "range": (0, 2147483647),
                    "label": "Random Seed (for reproducibility)",
                },
            },
            "function": "perform_kmeans_clustering",
        },

        "pls_da": {
            "name": "PLS-DA (Partial Least Squares Discriminant Analysis)",
            "description": "Supervised latent-variable projection maximizing class separation (chemometrics)",
            "min_datasets": 2,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "n_components": {
                    "type": "spinbox",
                    "default": 2,
                    "range": (2, 50),
                    "label": "Number of Components",
                },
                "scaling": {
                    "type": "combo",
                    "options": ["StandardScaler", "MinMaxScaler", "None"],
                    "default": "StandardScaler",
                    "label": "Scaling Method",
                },
                "cv_folds": {
                    "type": "spinbox",
                    "default": 5,
                    "range": (2, 20),
                    "label": "Cross-Validation Folds",
                },
                "show_vip": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show VIP Plot",
                },
                "show_loadings": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Component Weights",
                },
            },
            "function": "perform_pls_da_analysis",
        },


        "mcr_als": {
            "name": "MCR-ALS (Multivariate Curve Resolution)",
            "description": "Non-negative spectral unmixing into component spectra and abundances",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "n_components": {
                    "type": "spinbox",
                    "default": 3,
                    "range": (2, 20),
                    "label": "Number of Components",
                },
                "max_iter": {
                    "type": "spinbox",
                    "default": 50,
                    "range": (10, 500),
                    "label": "Max Iterations",
                },
                "tol": {
                    "type": "double_spinbox",
                    "default": 0.0001,
                    "range": (1e-6, 1e-2),
                    "step": 0.0001,
                    "label": "Convergence Tolerance",
                },
                "scaling": {
                    "type": "combo",
                    "options": ["None", "StandardScaler", "MinMaxScaler"],
                    "default": "None",
                    "label": "Scaling Method",
                },
                "nonneg_mode": {
                    "type": "combo",
                    "options": ["shift", "clip"],
                    "default": "shift",
                    "label": "Non-negativity Mode",
                },
            },
            "function": "perform_mcr_als_analysis",
        },

        "nmf": {
            "name": "NMF (Non-negative Matrix Factorization)",
            "description": "Non-negative decomposition into basis spectra and coefficients",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "n_components": {
                    "type": "spinbox",
                    "default": 3,
                    "range": (2, 20),
                    "label": "Number of Components",
                },
                "max_iter": {
                    "type": "spinbox",
                    "default": 500,
                    "range": (50, 3000),
                    "label": "Max Iterations",
                },
                "nonneg_mode": {
                    "type": "combo",
                    "options": ["shift", "clip"],
                    "default": "shift",
                    "label": "Non-negativity Mode",
                },
            },
            "function": "perform_nmf_analysis",
        },

        "ica": {
            "name": "ICA (Independent Component Analysis)",
            "description": "Blind source separation into statistically independent components",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "n_components": {
                    "type": "spinbox",
                    "default": 3,
                    "range": (2, 20),
                    "label": "Number of Components",
                },
                "scaling": {
                    "type": "combo",
                    "options": ["StandardScaler", "MinMaxScaler", "None"],
                    "default": "StandardScaler",
                    "label": "Scaling Method",
                },
                "max_iter": {
                    "type": "spinbox",
                    "default": 500,
                    "range": (100, 5000),
                    "label": "Max Iterations",
                },
            },
            "function": "perform_ica_analysis",
        },

        "outlier_detection": {
            "name": "Outlier Detection (QC)",
            "description": "Identify anomalous spectra using robust distance or isolation methods",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "method": {
                    "type": "combo",
                    "options": ["mahalanobis_mcd", "elliptic_envelope", "isolation_forest"],
                    "default": "isolation_forest",
                    "label": "Method",
                },
                "contamination": {
                    "type": "double_spinbox",
                    "default": 0.05,
                    "range": (0.01, 0.4),
                    "step": 0.01,
                    "label": "Contamination",
                },
                "scaling": {
                    "type": "combo",
                    "options": ["StandardScaler", "MinMaxScaler", "None"],
                    "default": "StandardScaler",
                    "label": "Scaling Method",
                },
                "detector_pca_components": {
                    "type": "spinbox",
                    "default": 20,
                    "range": (2, 100),
                    "label": "Detector PCA Components (speed/stability)",
                },
                "mcd_support_fraction": {
                    "type": "double_spinbox",
                    "default": 0.75,
                    "range": (0.5, 1.0),
                    "step": 0.05,
                    "label": "MCD Support Fraction",
                },
                "iso_n_estimators": {
                    "type": "spinbox",
                    "default": 100,
                    "range": (50, 500),
                    "label": "IsolationForest Estimators",
                },
                "pca_components": {
                    "type": "spinbox",
                    "default": 5,
                    "range": (2, 50),
                    "label": "PCA Components (for plot)",
                },
            },
            "function": "perform_outlier_detection",
        },

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
                    "label": "Confidence Level",
                },
                "fdr_correction": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Apply FDR Correction",
                },
                "show_ci": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Confidence Intervals",
                },
                "highlight_significant": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Highlight Significant Regions",
                },
            },
            "function": "perform_spectral_comparison",
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
                    "label": "Prominence Threshold",
                },
                "width_min": {
                    "type": "spinbox",
                    "default": 5,
                    "range": (1, 50),
                    "label": "Minimum Peak Width",
                },
                "top_n_peaks": {
                    "type": "spinbox",
                    "default": 20,
                    "range": (5, 100),
                    "label": "Top N Peaks to Display",
                },
                "show_assignments": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Biochemical Assignments",
                },
                "assignment_label_orientation": {
                    "type": "combo",
                    "options": ["horizontal", "vertical"],
                    "default": "horizontal",
                    "label": "Assignment Label Orientation",
                },
            },
            "function": "perform_peak_analysis",
        },
        "correlation_analysis": {
            "name": "Spectral Correlation Analysis",
            "description": "Analyze spectra–spectra similarity or wavenumber–wavenumber band co-variation",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "mode": {
                    "type": "combo",
                    "options": ["wavenumbers", "spectra"],
                    "default": "wavenumbers",
                    "label": "Correlation Mode",
                },
                "method": {
                    "type": "combo",
                    "options": ["pearson", "spearman", "kendall"],
                    "default": "pearson",
                    "label": "Correlation Method",
                },
                "max_wavenumbers": {
                    "type": "spinbox",
                    "default": 600,
                    "range": (100, 2000),
                    "label": "Max Wavenumbers (for map)",
                },
                "show_heatmap": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Correlation Heatmap",
                },
                "threshold": {
                    "type": "double_spinbox",
                    "default": 0.7,
                    "range": (0.0, 1.0),
                    "step": 0.05,
                    "label": "Correlation Threshold",
                },
            },
            "function": "perform_correlation_analysis"
        },

        "pairwise_tests": {
            "name": "Pairwise Statistical Tests",
            "description": "Wavenumber-wise significance testing between two groups",
            "min_datasets": 2,
            "max_datasets": 2,
            "dataset_selection_mode": "multi",
            # Pairwise tests must be a simple 2-dataset selection (grouped mode disabled)
            "allow_grouped_mode": False,
            "params": {
                "test_type": {
                    "type": "combo",
                    "options": ["t_test", "mann_whitney", "wilcoxon"],
                    "default": "t_test",
                    "label": "Test Type",
                },
                "alpha": {
                    "type": "double_spinbox",
                    "default": 0.05,
                    "range": (0.001, 0.2),
                    "step": 0.005,
                    "label": "Significance Level (α)",
                },
                "fdr_correction": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Apply FDR Correction",
                },
                "show_mean_overlay": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Mean Overlay",
                },
            },
            "function": "perform_pairwise_statistical_tests",
        },

        "band_ratio": {
            "name": "Band Ratio Analysis",
            "description": "Compute biochemical band ratios (e.g., protein/lipid) per spectrum",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            # Band ratio expects direct dataset selection (grouped mode disabled)
            "allow_grouped_mode": False,
            "params": {
                "band1_center": {
                    "type": "spinbox",
                    "default": 1650,
                    "range": (200, 4000),
                    "label": "Band 1 Center (cm⁻¹)",
                },
                "band1_width": {
                    "type": "spinbox",
                    "default": 20,
                    "range": (1, 200),
                    "label": "Band 1 Half-Width (cm⁻¹)",
                },
                "band2_center": {
                    "type": "spinbox",
                    "default": 1450,
                    "range": (200, 4000),
                    "label": "Band 2 Center (cm⁻¹)",
                },
                "band2_width": {
                    "type": "spinbox",
                    "default": 20,
                    "range": (1, 200),
                    "label": "Band 2 Half-Width (cm⁻¹)",
                },
                "measure": {
                    "type": "combo",
                    "options": ["area", "height"],
                    "default": "area",
                    "label": "Band Measure",
                },
            },
            "function": "perform_band_ratio_analysis",
        }
        ,
        "anova_test": {
            "name": "ANOVA (Wavenumber-wise)",
            "description": "One-way ANOVA across 3+ groups at each wavenumber (supports grouped mode and multiple-testing correction)",
            "min_datasets": 3,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "alpha": {
                    "type": "double_spinbox",
                    "default": 0.05,
                    "range": (0.001, 0.2),
                    "step": 0.005,
                    "label": "Significance Level (α)",
                },
                "p_adjust": {
                    "type": "combo",
                    "options": ["none", "fdr_bh", "bonferroni"],
                    "default": "fdr_bh",
                    "label": "Multiple-testing correction",
                },
                "post_hoc": {
                    "type": "combo",
                    "options": ["none", "tukey"],
                    "default": "none",
                    "label": "Post-hoc test (optional)",
                },
                "max_posthoc_wavenumbers": {
                    "type": "spinbox",
                    "default": 20,
                    "range": (0, 200),
                    "label": "Max post-hoc wavenumbers (0=off)",
                },
                "show_mean_overlay": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show mean spectra overlay",
                },
                "highlight_significant": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Highlight significant regions",
                },
            },
            "function": "perform_anova_test",
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
                "max_wavenumbers": {
                    "type": "spinbox",
                    "default": 1200,
                    "range": (200, 5000),
                    "label": "Max Wavenumbers (resample cap)",
                },
                "cluster_rows": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Cluster Rows (Samples)",
                },
                "cluster_cols": {
                    "type": "checkbox",
                    "default": False,
                    "label": "Cluster Columns (Wavenumbers)",
                },
                "colormap": {
                    "type": "combo",
                    "options": [
                        "viridis",
                        "plasma",
                        "inferno",
                        "magma",
                        "cividis",
                        "coolwarm",
                        "RdYlBu",
                    ],
                    "default": "viridis",
                    "label": "Colormap",
                },
                "normalize": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Normalize Intensities",
                },
                "show_dendrograms": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Dendrograms",
                },
            },
            "function": "create_spectral_heatmap",
        },
        # NOTE: Mean Spectra Overlay Plot removed to reduce overlap with
        # Pairwise Statistical Tests and Group Mean Spectral Comparison.
        "waterfall_plot": {
            "name": "Waterfall Plot",
            "description": "2D/3D visualization of multiple spectra with offset",
            "min_datasets": 1,
            "max_datasets": 1,
            "dataset_selection_mode": "single",
            "params": {
                "use_3d": {
                    "type": "checkbox",
                    "default": False,
                    "label": "3D Waterfall Plot",
                },
                "offset_scale": {
                    "type": "double_spinbox",
                    "default": 1.0,
                    "range": (0.1, 5.0),
                    "step": 0.1,
                    "label": "Offset Scale",
                },
                "max_spectra": {
                    "type": "spinbox",
                    "default": 50,
                    "range": (10, 200),
                    "label": "Maximum Spectra to Display",
                },
                "colormap": {
                    "type": "combo",
                    "options": [
                        "viridis",
                        "plasma",
                        "coolwarm",
                        "rainbow",
                        "jet",
                        "turbo",
                    ],
                    "default": "viridis",
                    "label": "Colormap",
                },
                "show_grid": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Grid Lines",
                },
            },
            "function": "create_waterfall_plot",
        },
        "correlation_heatmap": {
            "name": "Correlation Heatmap",
            "description": "Heatmap of pairwise spectral correlations",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "max_wavenumbers": {
                    "type": "spinbox",
                    "default": 600,
                    "range": (100, 2000),
                    "label": "Max Wavenumbers (resample cap)",
                },
                "method": {
                    "type": "combo",
                    "options": ["pearson", "spearman"],
                    "default": "pearson",
                    "label": "Correlation Method",
                },
                "colormap": {
                    "type": "combo",
                    "options": ["coolwarm", "RdYlBu", "RdBu", "seismic"],
                    "default": "coolwarm",
                    "label": "Colormap",
                },
                "show_values": {
                    "type": "checkbox",
                    "default": False,
                    "label": "Show Correlation Values",
                },
                "cluster": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Cluster Samples",
                },
            },
            "function": "create_correlation_heatmap",
        },
        "peak_intensity_scatter": {
            "name": "Peak Intensity Scatter Plot",
            "description": "2D/3D scatter plot of peak intensities with statistical annotations",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "peak_1_position": {
                    "type": "spinbox",
                    "default": 1000,
                    "range": (400, 4000),
                    "label": "Peak 1 Position (cm⁻¹)",
                },
                "peak_2_position": {
                    "type": "spinbox",
                    "default": 1650,
                    "range": (400, 4000),
                    "label": "Peak 2 Position (cm⁻¹)",
                },
                "peak_3_position": {
                    "type": "spinbox",
                    "default": 2900,
                    "range": (400, 4000),
                    "label": "Peak 3 Position (cm⁻¹)",
                },
                "tolerance": {
                    "type": "spinbox",
                    "default": 10,
                    "range": (1, 50),
                    "label": "Peak Tolerance (cm⁻¹)",
                },
                "use_3d": {
                    "type": "checkbox",
                    "default": False,
                    "label": "3D Scatter (3 peaks)",
                },
                "show_statistics": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Statistics",
                },
                "show_legend": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Legend",
                },
                "colormap": {
                    "type": "combo",
                    "default": "tab10",
                    "options": [
                        "tab10",
                        "Set1",
                        "Set2",
                        "Dark2",
                        "Paired",
                        "viridis",
                        "plasma",
                    ],
                    "label": "Color Scheme",
                },
                "marker_size": {
                    "type": "spinbox",
                    "default": 60,
                    "range": (20, 200),
                    "label": "Marker Size",
                },
            },
            "function": "create_peak_scatter",
        },

        "derivative_spectra": {
            "name": "Derivative Spectra (Savitzky–Golay)",
            "description": "Visualize 1st/2nd derivative spectra to enhance peak resolution",
            "min_datasets": 1,
            "max_datasets": None,
            "dataset_selection_mode": "multi",
            "params": {
                "deriv_order": {
                    "type": "combo",
                    "options": [1, 2],
                    "default": 1,
                    "label": "Derivative Order",
                },
                "window_length": {
                    "type": "spinbox",
                    "default": 15,
                    "range": (5, 301),
                    "label": "Window Length (odd)",
                },
                "polyorder": {
                    "type": "spinbox",
                    "default": 3,
                    "range": (2, 10),
                    "label": "Polynomial Order",
                },
                "show_original": {
                    "type": "checkbox",
                    "default": True,
                    "label": "Show Original (faint)",
                },
            },
            "function": "create_derivative_spectra_plot",
        },
    },
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
