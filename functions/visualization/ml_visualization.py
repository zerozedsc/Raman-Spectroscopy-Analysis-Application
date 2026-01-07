"""
ML Visualization Module for Raman Spectroscopy Analysis

This module provides dimensionality reduction visualization functions for machine learning
models trained on Raman spectroscopy data. It supports PCA, with future support for t-SNE
and UMAP.

Key Features:
    - PCA 2D visualization with decision boundaries
    - Multiple data input modes (DataFrame, numpy array, SpectralContainer)
    - Automatic data source detection from ML_PROPERTY
    - Pre-calculated decision boundary support
    - Centroid visualization and analysis

Dependencies:
    - numpy: Array operations
    - pandas: DataFrame handling
    - matplotlib: Plotting
    - sklearn: PCA computation
    - ramanspy: Spectral container handling
    - functions.configs: Logging utilities

Author: MUHAMMAD HELMI BIN ROZAIN
Date: October 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Tuple
from sklearn.decomposition import PCA

try:
    import ramanspy as rp
except ImportError:
    rp = None

from functions.configs import console_log


# ============================================================================
# HELPER FUNCTIONS (Private)
# ============================================================================


def _prepare_data_from_ml_property(
    ml_property, prefer_train: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Extract data from ML_PROPERTY instance.

    Args:
        ml_property: ML_PROPERTY instance containing training/test data
        prefer_train: Whether to prefer training data over test data

    Returns:
        Tuple of (X, y, common_axis, n_features)

    Raises:
        ValueError: If no valid data found in ML_PROPERTY
    """
    if ml_property is None:
        raise ValueError("ML_PROPERTY is None")

    # Try training data first if preferred
    if prefer_train and (
        ml_property.X_train is not None
        and ml_property.y_train is not None
        and ml_property.common_axis is not None
    ):

        console_log("🔄 Using training data from ML_PROPERTY...")
        X = ml_property.X_train
        y = ml_property.y_train
        common_axis = ml_property.common_axis
        n_features = ml_property.n_features_in

        console_log(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")

    # Fallback to test data
    elif (
        ml_property.X_test is not None
        and ml_property.y_test is not None
        and ml_property.common_axis is not None
    ):

        console_log("🔄 Using test data from ML_PROPERTY...")
        X = ml_property.X_test
        y = ml_property.y_test
        common_axis = ml_property.common_axis
        n_features = ml_property.n_features_in

        console_log(f"📊 Test data: {X.shape[0]} samples, {X.shape[1]} features")

    else:
        raise ValueError(
            "No valid data found in ML_PROPERTY. "
            "Ensure X_train/X_test, y_train/y_test, and common_axis are populated."
        )

    return X, y, common_axis, n_features


def _prepare_data_from_dataframe(
    df: Union[pd.DataFrame, np.ndarray], labels: Union[str, List, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features and labels from DataFrame or numpy array.

    Args:
        df: DataFrame or numpy array containing spectral data
        labels: Either column name (str) or array of labels

    Returns:
        Tuple of (X, y) as numpy arrays

    Raises:
        ValueError: If dimensions don't match or invalid input
    """
    console_log(f"Using DataFrame/Array mode with shape: {df.shape}")

    # Handle pandas DataFrame
    if isinstance(df, pd.DataFrame):
        console_log("Input is pandas DataFrame")

        if isinstance(labels, str):
            # Labels is a column name
            if labels in df.columns:
                y = df[labels].values
                X = df.drop(columns=[labels]).values
            else:
                raise ValueError(f"Label column '{labels}' not found in DataFrame")
        else:
            # Labels is an array
            y = np.array(labels)
            X = df.values

    # Handle numpy array
    elif isinstance(df, np.ndarray):
        console_log("Input is numpy array")

        if labels is None:
            raise ValueError("Labels must be provided when using numpy array input")

        X = df
        y = np.array(labels)

    else:
        raise ValueError(
            f"Unsupported data type: {type(df)}. Expected DataFrame or numpy array."
        )

    # Verify dimensions match
    if X.shape[0] != len(y):
        raise ValueError(
            f"Shape mismatch: Data has {X.shape[0]} rows but {len(y)} labels provided"
        )

    return X, y


def _prepare_data_from_containers(
    containers: List,
    labels: List[str],
    common_axis: np.ndarray,
    current_n_features: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and interpolate data from SpectralContainer objects.

    Args:
        containers: List of SpectralContainer objects
        labels: List of labels for each spectrum
        common_axis: Common wavenumber axis for interpolation
        current_n_features: Expected number of features after interpolation

    Returns:
        Tuple of (X, y) as numpy arrays

    Raises:
        ValueError: If no valid spectra found or missing parameters
    """
    console_log(f"Using SpectralContainer mode with {len(containers)} containers")

    if common_axis is None or current_n_features is None:
        raise ValueError(
            "common_axis and current_n_features must be provided for SpectralContainer input"
        )

    X_list = []
    y_list = []
    label_idx = 0

    for container_idx, s_container in enumerate(containers):
        if s_container.spectral_data is None or s_container.spectral_data.size == 0:
            continue

        for single_spectrum in s_container.spectral_data:
            if single_spectrum.ndim != 1:
                continue

            # Interpolate spectrum to common axis if needed
            if len(s_container.spectral_axis) != current_n_features:
                interp_spectrum = np.interp(
                    common_axis, s_container.spectral_axis, single_spectrum
                )
                X_list.append(interp_spectrum)
            else:
                X_list.append(single_spectrum)

            # Add corresponding label
            if label_idx < len(labels):
                y_list.append(labels[label_idx])
                label_idx += 1
            else:
                # Fallback based on container position
                if container_idx < len(containers) // 2:
                    y_list.append("class_0")
                else:
                    y_list.append("class_1")

    if not X_list:
        raise ValueError("No valid spectra found in containers")

    X = np.array(X_list)
    y = np.array(y_list)

    # Final dimension check
    if X.shape[0] != len(y):
        console_log(
            f"Warning: Shape mismatch X={X.shape[0]}, y={len(y)}. Using integer indices."
        )
        y = np.arange(X.shape[0])

    return X, y


def _compute_pca(
    X: np.ndarray, n_components: int = 2, sample_limit: Optional[int] = None
) -> Tuple[PCA, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA on spectral data.

    Args:
        X: Feature matrix (n_samples, n_features)
        n_components: Number of PCA components (default: 2)
        sample_limit: Optional limit on samples for plotting

    Returns:
        Tuple of (pca_model, X_pca_transformed, X_original, y_limited, indices_used)
    """
    console_log(f"Computing PCA with {n_components} components on {X.shape[0]} samples")

    # Store original data
    X_original = X.copy()

    # Fit PCA on full dataset
    pca = PCA(n_components=n_components)
    pca.fit(X_original)

    # Apply sample limiting if needed
    if sample_limit is not None and X.shape[0] > sample_limit:
        console_log(f"Limiting plot to {sample_limit} samples out of {X.shape[0]}")
        indices = np.random.choice(X.shape[0], sample_limit, replace=False)
    else:
        indices = np.arange(X.shape[0])

    # Transform data
    X_pca = pca.transform(X_original[indices])

    return pca, X_pca, X_original, indices


def _plot_pca_scatter(
    X_pca: np.ndarray,
    y: np.ndarray,
    explained_variance: np.ndarray,
    title: str = "PCA",
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = "tab10",
    add_centroids: bool = True,
    show_centroid_line: bool = True,
    legend: bool = True,
    legend_loc: str = "best",
) -> Tuple[plt.Figure, dict]:
    """
    Create PCA scatter plot with centroids and styling.

    Args:
        X_pca: Transformed PCA data (n_samples, 2)
        y: Labels for each sample
        explained_variance: Variance explained by each component
        title: Plot title
        figsize: Figure size (width, height)
        cmap: Colormap name
        add_centroids: Whether to plot class centroids
        show_centroid_line: Whether to draw line between centroids
        legend: Whether to show legend
        legend_loc: Legend location

    Returns:
        Tuple of (figure, metadata_dict)
    """
    fig = plt.figure(figsize=figsize)
    unique_labels = np.unique(y)

    # Create color mapping
    if len(unique_labels) == 2:
        # Binary classification: specific colors
        label_to_color = {
            unique_labels[0]: "#1f77b4",  # Blue for first class
            unique_labels[1]: "#ff7f0e",  # Orange for second class
        }
    else:
        # Multiclass: use colormap
        colors = plt.get_cmap(cmap, len(unique_labels))
        label_to_color = {label: colors(i) for i, label in enumerate(unique_labels)}

    # Plot data points
    for label in unique_labels:
        idxs = np.where(y == label)[0]
        plt.scatter(
            X_pca[idxs, 0],
            X_pca[idxs, 1],
            label=label,
            color=label_to_color[label],
            alpha=0.7,
            s=60,
            edgecolors="white",
            linewidth=0.5,
        )

    # Calculate and plot centroids
    centroids = []
    for label in unique_labels:
        idxs = np.where(y == label)[0]
        centroid = X_pca[idxs].mean(axis=0)
        centroids.append(centroid)

        if add_centroids:
            plt.scatter(
                *centroid,
                color=label_to_color[label],
                edgecolor="black",
                s=200,
                marker="X",
                zorder=5,
            )
            plt.text(
                centroid[0],
                centroid[1] - 0.5,
                f"{label} centroid",
                fontsize=10,
                weight="bold",
                ha="center",
            )

    # Draw line between centroids (binary classification)
    if show_centroid_line and len(centroids) == 2:
        plt.plot(
            [centroids[0][0], centroids[1][0]],
            [centroids[0][1], centroids[1][1]],
            "k--",
            lw=2,
            label="Centroid Line",
            alpha=0.8,
        )

    # Formatting
    title_with_variance = (
        f"{title}\n"
        f"(PC1: {explained_variance[0]:.1%}, PC2: {explained_variance[1]:.1%} variance explained)"
    )

    plt.title(title_with_variance, fontsize=12, fontweight="bold")
    plt.xlabel(f"PC1 ({explained_variance[0]:.1%} variance)", fontsize=11)
    plt.ylabel(f"PC2 ({explained_variance[1]:.1%} variance)", fontsize=11)

    if legend:
        plt.legend(loc=legend_loc, frameon=True, fancybox=True, shadow=True)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Prepare metadata
    metadata = {
        "centroids": centroids,
        "unique_labels": unique_labels,
        "label_to_color": label_to_color,
        "explained_variance": explained_variance,
    }

    return fig, metadata


def _add_decision_boundary(
    boundary_data: dict, decision_boundary_alpha: float = 0.3
) -> bool:
    """
    Add pre-calculated decision boundary to current plot.

    Args:
        boundary_data: Dictionary containing boundary mesh and predictions
        decision_boundary_alpha: Transparency of the boundary contour

    Returns:
        bool: True if boundary was successfully plotted, False otherwise
    """
    try:
        # Validate boundary data
        required_keys = ["xx", "yy", "Z", "pca"]
        if not all(key in boundary_data for key in required_keys):
            console_log("⚠️ Boundary data incomplete, missing required keys")
            return False

        xx = boundary_data["xx"]
        yy = boundary_data["yy"]
        Z = boundary_data["Z"]

        # Plot decision boundary
        plt.contourf(
            xx,
            yy,
            Z,
            levels=50,
            alpha=decision_boundary_alpha,
            cmap="RdYlBu_r",
            vmin=0,
            vmax=1,
        )

        # Plot decision boundary line
        plt.contour(
            xx,
            yy,
            Z,
            levels=[0.5],
            colors="black",
            linestyles="--",
            linewidths=2,
            alpha=0.8,
        )

        # Add colorbar
        cbar = plt.colorbar(label="Prediction Probability", shrink=0.8)
        cbar.ax.tick_params(labelsize=10)

        console_log("✅ Decision boundary plotted successfully!")

        # Log boundary info
        if "explained_variance_ratio" in boundary_data:
            exp_var = boundary_data["explained_variance_ratio"]
            console_log(
                f"📊 Boundary PCA variance: PC1={exp_var[0]:.1%}, PC2={exp_var[1]:.1%}"
            )

        return True

    except Exception as e:
        console_log(f"❌ Error plotting decision boundary: {e}")
        return False


# ============================================================================
# PUBLIC API
# ============================================================================


def pca2d(
    df: Union[pd.DataFrame, np.ndarray] = None,
    containers: List = None,
    labels: Union[str, List, np.ndarray] = None,
    common_axis: np.ndarray = None,
    current_n_features: int = None,
    ml_property=None,
    title: str = "PCA",
    figsize: Tuple[int, int] = (12, 6),
    sample_limit: int = 100,
    cmap: str = "tab10",
    add_centroids: bool = True,
    show_centroid_line: bool = True,
    legend: bool = True,
    legend_loc: str = "best",
    show_decision_boundary: bool = False,
    decision_boundary_alpha: float = 0.3,
    use_precalculated_boundary: bool = True,
) -> plt:
    """
    Perform PCA on spectral data and create 2D visualization.

    This function supports multiple input modes:
    1. Auto-detect from ml_property (if provided and df/containers are None)
    2. DataFrame/numpy array with labels
    3. SpectralContainer list with labels

    Args:
        df: DataFrame or numpy array containing spectral data (optional)
        containers: List of SpectralContainer objects (optional)
        labels: Labels for samples (str for DataFrame column, list/array for explicit labels)
        common_axis: Common wavenumber axis for interpolation (needed for containers)
        current_n_features: Number of features after interpolation (needed for containers)
        ml_property: ML_PROPERTY instance for auto-detection and boundary data
        title: Plot title
        figsize: Figure size (width, height)
        sample_limit: Maximum samples to plot (for performance)
        cmap: Matplotlib colormap name
        add_centroids: Whether to plot class centroids
        show_centroid_line: Whether to draw line between centroids (binary only)
        legend: Whether to show legend
        legend_loc: Legend location
        show_decision_boundary: Whether to plot decision boundary
        decision_boundary_alpha: Transparency of decision boundary
        use_precalculated_boundary: Whether to use pre-calculated boundary from ml_property

    Returns:
        matplotlib.pyplot: Plot object for further customization

    Raises:
        ValueError: If insufficient data provided or invalid input

    Examples:
        >>> # Auto-detect from ML_PROPERTY
        >>> pca2d(ml_property=my_ml_property)

        >>> # Explicit DataFrame
        >>> pca2d(df=my_dataframe, labels='class_column')

        >>> # Numpy array with explicit labels
        >>> pca2d(df=X_array, labels=y_array)

        >>> # With decision boundary
        >>> pca2d(ml_property=ml_prop, show_decision_boundary=True)

    Notes:
        - Decision boundary requires pre-calculation via predict() with calculate_pca_boundary=True
        - Function fits PCA on full dataset but may plot subset for performance
        - For SpectralContainer mode, all parameters (containers, labels, common_axis,
          current_n_features) must be provided
    """
    # ========================================================================
    # STEP 1: DATA PREPARATION - AUTO-DETECT OR EXPLICIT
    # ========================================================================

    # Auto-detect from ML_PROPERTY
    if df is None and containers is None:
        if ml_property is not None:
            X, y, common_axis, current_n_features = _prepare_data_from_ml_property(
                ml_property
            )
        else:
            raise ValueError(
                "No data provided. Either supply df/containers or provide ml_property with data."
            )

    # Explicit DataFrame/numpy array
    elif df is not None:
        X, y = _prepare_data_from_dataframe(df, labels)

    # Explicit SpectralContainer list
    elif containers is not None:
        if labels is None:
            raise ValueError("Labels must be provided for SpectralContainer input")
        X, y = _prepare_data_from_containers(
            containers, labels, common_axis, current_n_features
        )

    else:
        raise ValueError("Either df, containers, or ml_property must be provided")

    # ========================================================================
    # STEP 2: PCA COMPUTATION
    # ========================================================================

    console_log(
        f"Generating PCA plot for {X.shape[0]} spectra with "
        f"{len(np.unique(y))} unique classes: {np.unique(y)}"
    )

    # Store original data for PCA fitting
    X_original = X.copy()
    y_original = y.copy()

    # Compute PCA and apply sample limiting
    pca, X_pca, X_full, indices = _compute_pca(
        X, n_components=2, sample_limit=sample_limit
    )
    y_plot = y[indices]

    explained_var = pca.explained_variance_ratio_

    # ========================================================================
    # STEP 3: DECISION BOUNDARY (if requested and available)
    # ========================================================================

    boundary_plotted = False
    boundary_pca = None

    if show_decision_boundary and ml_property is not None:
        if (
            use_precalculated_boundary
            and hasattr(ml_property, "pca_boundary_data")
            and ml_property.pca_boundary_data is not None
        ):

            console_log("🚀 Using pre-calculated PCA decision boundary...")

            boundary_data = ml_property.pca_boundary_data
            boundary_plotted = _add_decision_boundary(
                boundary_data, decision_boundary_alpha
            )

            if boundary_plotted and "pca" in boundary_data:
                # Use boundary PCA for consistency
                boundary_pca = boundary_data["pca"]
                X_pca = boundary_pca.transform(X_original[indices])

                if "explained_variance_ratio" in boundary_data:
                    explained_var = boundary_data["explained_variance_ratio"]
        else:
            if show_decision_boundary:
                console_log("⚠️ No pre-calculated boundary data found in ml_property")
                console_log(
                    "💡 Tip: Run predict() with calculate_pca_boundary=True to generate it"
                )

    # ========================================================================
    # STEP 4: SCATTER PLOT
    # ========================================================================

    fig, metadata = _plot_pca_scatter(
        X_pca=X_pca,
        y=y_plot,
        explained_variance=explained_var,
        title=title,
        figsize=figsize,
        cmap=cmap,
        add_centroids=add_centroids,
        show_centroid_line=show_centroid_line,
        legend=legend,
        legend_loc=legend_loc,
    )

    plt.show()

    # ========================================================================
    # STEP 5: SUMMARY LOGGING
    # ========================================================================

    console_log(f"\n📊 PCA Summary:")
    console_log(f"- Total variance explained by PC1 + PC2: {sum(explained_var):.1%}")
    console_log(f"- Number of samples plotted: {len(X_pca)}")
    console_log(f"- Original feature dimensions: {X_original.shape[1]}")

    if show_decision_boundary:
        if boundary_plotted:
            console_log("- Decision boundary: ✅ Using pre-calculated data")
        else:
            console_log("- Decision boundary: ❌ Not available")
            console_log(
                "  💡 Run predict() with calculate_pca_boundary=True to generate it"
            )

    return plt


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "pca2d",
]
