"""
Model Evaluation and Statistical Visualization Module

This module provides functions for evaluating machine learning models and
visualizing statistical distributions of spectral data.

Functions:
    confusion_matrix_heatmap: Plot confusion matrix with per-class accuracy
    plot_institution_distribution: t-SNE visualization of spectral data distribution

Author: MUHAMMAD HELMI BIN ROZAIN
Created: 2025-10-01 (Extracted from core.py during Phase 1 refactoring)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ramanspy as rp
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from typing import List, Tuple, Dict, Union
from functions.configs import console_log


def confusion_matrix_heatmap(
    y_true: list,
    y_pred: list,
    class_labels: list,
    title: str = "Confusion Matrix",
    figsize: tuple = (10, 8),
    cmap: str = "Blues",
    normalize: bool = True,
    show_counts: bool = True,
    fmt: str = None,
    show_heatmap: bool = True,
) -> Tuple[Dict[str, float], sns.heatmap]:
    """
    Plot a confusion matrix as a heatmap with per-class accuracy.

    Parameters:
    -----------
    y_true : list
        True labels
    y_pred : list
        Predicted labels
    class_labels : list
        List of class labels
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height) in inches
    cmap : str
        Matplotlib colormap for the heatmap
    normalize : bool
        Whether to normalize the confusion matrix (default: True)
    show_counts : bool
        Whether to show raw counts in each cell (default: True)
    fmt : str
        Format for annotations (default: '.2f' for normalized, 'd' for counts)
    show_heatmap : bool
        Whether to show the heatmap (default: True)

    Returns:
    --------
    per_class_accuracy : dict
        Dictionary mapping class labels to their prediction accuracy (recall)
    ax : seaborn heatmap axis or None
        The heatmap axis object if show_heatmap=True, else None

    Examples:
    ---------
    >>> y_true = ['benign', 'cancer', 'benign', 'cancer', 'benign']
    >>> y_pred = ['benign', 'cancer', 'cancer', 'cancer', 'benign']
    >>> class_labels = ['benign', 'cancer']
    >>> per_class_acc, ax = confusion_matrix_heatmap(y_true, y_pred, class_labels)
    >>> print(f"Benign accuracy: {per_class_acc['benign']:.1f}%")
    """
    # Check input lengths
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length. "
            f"Got {len(y_true)} and {len(y_pred)}."
        )

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    # Calculate per-class prediction accuracy (recall)
    per_class_accuracy = {}
    for idx, label in enumerate(class_labels):
        total = cm[idx, :].sum()
        correct = cm[idx, idx]
        acc = (correct / total) * 100 if total > 0 else 0
        per_class_accuracy[label] = acc

    # Normalize if requested
    if normalize and fmt is None:
        with np.errstate(all="ignore"):
            cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        if fmt is None:
            fmt = ".2f"
    elif fmt is None:
        cm_display = cm
        fmt = "d"
    else:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = fmt

    # Prepare annotation labels
    if show_counts and normalize:
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm_display[i, j]:.2f}\n({cm[i, j]})"
    elif show_counts:
        annot = cm
    else:
        annot = cm_display

    ax = None

    if show_heatmap:
        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            cm_display,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar=False,
            square=True,
            linewidths=0.5,
        )
        plt.title(title)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.show()

    return per_class_accuracy, ax


def plot_institution_distribution(
    spectral_containers: List[rp.SpectralContainer],
    container_labels: List[str],
    container_names: List[str] = None,
    sample_limit: int = 500,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    figsize: tuple = (12, 8),
    alpha: float = 0.6,
    point_size: int = 50,
    title: str = "Institution Distribution in Feature Space",
    show_legend: bool = True,
    save_plot: bool = False,
    save_path: str = None,
    color_palette: str = "tab10",
    interpolate_to_common_axis: bool = True,
    common_axis: np.ndarray = None,
    show_class_info: bool = True,
    class_labels: List[str] = None,
) -> dict:
    """
    Plot t-SNE visualization of spectral data distribution across different containers/institutions.

    This function helps visualize how spectra from different institutions or datasets
    are distributed in a 2D feature space, useful for identifying batch effects or
    data quality issues.

    Parameters:
    -----------
    spectral_containers : List[rp.SpectralContainer]
        List of SpectralContainer objects to compare
    container_labels : List[str]
        Labels for each container (e.g., ['CHUM_benign', 'CHUM_cancer', 'UHN_benign', ...])
    container_names : List[str], optional
        Institution/group names for each container. If None, extracted from container_labels
    sample_limit : int
        Maximum number of samples to use for t-SNE (for performance)
    perplexity : int
        t-SNE perplexity parameter (typically between 5 and 50)
    n_iter : int
        Number of iterations for t-SNE optimization
    random_state : int
        Random seed for reproducibility
    figsize : tuple
        Figure size (width, height)
    alpha : float
        Point transparency (0-1)
    point_size : int
        Size of scatter plot points
    title : str
        Plot title
    show_legend : bool
        Whether to show legend
    save_plot : bool
        Whether to save the plot
    save_path : str
        Path to save the plot (if save_plot=True)
    color_palette : str
        Matplotlib colormap name
    interpolate_to_common_axis : bool
        Whether to interpolate all spectra to a common axis
    common_axis : np.ndarray, optional
        Common axis for interpolation. If None, uses the first container's axis
    show_class_info : bool
        Whether to show class information in the legend
    class_labels : List[str], optional
        Class labels (e.g., ['benign', 'cancer']) for extracting class info

    Returns:
    --------
    dict : Results dictionary containing:
        - success: bool
        - embedded_data: np.ndarray (n_samples, 2)
        - institutions: np.ndarray (institution names)
        - labels: np.ndarray (container labels)
        - classes: np.ndarray (class labels if available)
        - container_info: list of dicts
        - unique_institutions: list
        - tsne_params: dict
        - data_info: dict

    Examples:
    ---------
    >>> from ramanspy import SpectralContainer
    >>> # Assuming you have containers loaded
    >>> results = plot_institution_distribution(
    ...     spectral_containers=[benign_container, cancer_container],
    ...     container_labels=['Benign', 'Cancer'],
    ...     sample_limit=200
    ... )
    >>> if results['success']:
    ...     print(f"Analyzed {results['data_info']['total_spectra']} spectra")
    """
    try:
        console_log("🔬 Starting spectral distribution analysis...")

        # Validate inputs
        if len(spectral_containers) != len(container_labels):
            raise ValueError(
                f"Number of containers ({len(spectral_containers)}) must match "
                f"number of labels ({len(container_labels)})"
            )

        # Extract institution names if not provided
        if container_names is None:
            container_names = []
            for label in container_labels:
                # Extract institution name (everything before first underscore)
                inst_name = label.split("_")[0] if "_" in label else label
                container_names.append(inst_name)

        # Get unique institutions
        unique_institutions = list(set(container_names))
        console_log(
            f"📊 Found {len(unique_institutions)} unique institutions: {unique_institutions}"
        )

        # Determine common axis for interpolation
        if interpolate_to_common_axis:
            if common_axis is None:
                # Use the axis from the container with the most features
                max_features = 0
                best_axis = None
                for container in spectral_containers:
                    if len(container.spectral_axis) > max_features:
                        max_features = len(container.spectral_axis)
                        best_axis = container.spectral_axis
                common_axis = best_axis
                console_log(f"🔧 Using common axis with {len(common_axis)} features")
            else:
                console_log(
                    f"🔧 Using provided common axis with {len(common_axis)} features"
                )

        # Collect all spectral data
        all_data = []
        all_labels = []
        all_institutions = []
        all_classes = []
        container_info = []

        for i, (container, label, inst_name) in enumerate(
            zip(spectral_containers, container_labels, container_names)
        ):
            if container.spectral_data is None or len(container.spectral_data) == 0:
                console_log(
                    f"⚠️  Warning: Container {i} ({label}) has no spectral data, skipping..."
                )
                continue

            # Extract class information if available
            class_info = "unknown"
            if class_labels:
                for class_label in class_labels:
                    if class_label.lower() in label.lower():
                        class_info = class_label
                        break

            console_log(
                f"📈 Processing container {i+1}/{len(spectral_containers)}: "
                f"{label} ({len(container.spectral_data)} spectra)"
            )

            # Process each spectrum in the container
            for spectrum_idx, spectrum in enumerate(container.spectral_data):
                try:
                    # Interpolate to common axis if needed
                    if interpolate_to_common_axis and len(
                        container.spectral_axis
                    ) != len(common_axis):
                        interpolated_spectrum = np.interp(
                            common_axis, container.spectral_axis, spectrum
                        )
                        all_data.append(interpolated_spectrum)
                    else:
                        all_data.append(spectrum)

                    all_labels.append(label)
                    all_institutions.append(inst_name)
                    all_classes.append(class_info)
                    container_info.append(
                        {
                            "container_idx": i,
                            "spectrum_idx": spectrum_idx,
                            "label": label,
                            "institution": inst_name,
                            "class": class_info,
                        }
                    )

                except Exception as e:
                    console_log(
                        f"⚠️  Warning: Error processing spectrum {spectrum_idx} "
                        f"in container {i}: {e}"
                    )
                    continue

        if not all_data:
            raise ValueError("No valid spectral data found in any container")

        # Convert to numpy arrays
        all_data = np.array(all_data)
        all_labels = np.array(all_labels)
        all_institutions = np.array(all_institutions)
        all_classes = np.array(all_classes)

        console_log(
            f"📊 Total collected data: {all_data.shape[0]} spectra "
            f"with {all_data.shape[1]} features"
        )

        # Sample data if too large
        if len(all_data) > sample_limit:
            console_log(
                f"🎲 Sampling {sample_limit} spectra from {len(all_data)} "
                f"total for t-SNE performance"
            )
            indices = np.random.choice(len(all_data), sample_limit, replace=False)
            all_data = all_data[indices]
            all_labels = all_labels[indices]
            all_institutions = all_institutions[indices]
            all_classes = all_classes[indices]
            container_info = [container_info[i] for i in indices]

        # Perform t-SNE
        console_log(
            f"🧮 Running t-SNE with perplexity={perplexity}, n_iter={n_iter}..."
        )
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
            verbose=1,
        )
        embedded = tsne.fit_transform(all_data)

        # Create the plot
        plt.figure(figsize=figsize)

        # Get colors
        cmap = plt.get_cmap(color_palette)
        colors = [
            cmap(i / len(unique_institutions)) for i in range(len(unique_institutions))
        ]
        institution_colors = {
            inst: colors[i] for i, inst in enumerate(unique_institutions)
        }

        # Plot by institution
        for inst in unique_institutions:
            mask = all_institutions == inst

            # Count classes for this institution if class info is available
            class_counts = {}
            if show_class_info and class_labels:
                for class_label in class_labels:
                    class_count = np.sum(
                        (all_institutions == inst) & (all_classes == class_label)
                    )
                    if class_count > 0:
                        class_counts[class_label] = class_count

            # Create legend label
            if class_counts:
                class_info_str = ", ".join(
                    [f"{cls}:{cnt}" for cls, cnt in class_counts.items()]
                )
                legend_label = f"{inst} ({class_info_str})"
            else:
                legend_label = f"{inst} (n={np.sum(mask)})"

            plt.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                label=legend_label,
                alpha=alpha,
                s=point_size,
                color=institution_colors[inst],
            )

        # Customize plot
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("t-SNE Component 1", fontsize=12)
        plt.ylabel("t-SNE Component 2", fontsize=12)

        if show_legend:
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot if requested
        if save_plot:
            if save_path is None:
                save_path = f"institution_distribution_tsne_{random_state}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            console_log(f"💾 Plot saved to: {save_path}")

        plt.show()

        # Calculate and display statistics
        console_log("\n📈 Distribution Statistics:")
        console_log("=" * 50)
        for inst in unique_institutions:
            mask = all_institutions == inst
            count = np.sum(mask)
            percentage = (count / len(all_institutions)) * 100
            console_log(f"{inst}: {count} spectra ({percentage:.1f}%)")

            if show_class_info and class_labels:
                for class_label in class_labels:
                    class_count = np.sum(
                        (all_institutions == inst) & (all_classes == class_label)
                    )
                    if class_count > 0:
                        class_percentage = (class_count / count) * 100
                        console_log(
                            f"  └─ {class_label}: {class_count} ({class_percentage:.1f}%)"
                        )

        # Return results
        results = {
            "success": True,
            "embedded_data": embedded,
            "institutions": all_institutions,
            "labels": all_labels,
            "classes": all_classes,
            "container_info": container_info,
            "unique_institutions": unique_institutions,
            "tsne_params": {
                "perplexity": perplexity,
                "n_iter": n_iter,
                "random_state": random_state,
            },
            "data_info": {
                "total_spectra": len(all_data),
                "n_features": all_data.shape[1],
                "sampled": len(all_data) < len(container_info),
            },
        }

        console_log(f"\n✅ Analysis completed successfully!")
        return results

    except Exception as e:
        console_log(f"❌ Error in plot_institution_distribution: {e}")
        import traceback

        traceback.print_exc()

        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
