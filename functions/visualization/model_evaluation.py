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
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from typing import List, Tuple, Dict, Union, Optional
from functions.configs import console_log


def _display_label(label: str, label_display_map: Optional[Dict[str, str]]) -> str:
    """Return a display-safe label for plots.

    We keep model/metric computation keyed on the *original* labels, but allow plots
    to use an alternate display label (e.g., forced English) to avoid CJK glyph issues
    in embedded Matplotlib widgets.
    """
    if not label_display_map:
        return str(label)
    return str(label_display_map.get(str(label), str(label)))


def create_confusion_matrix_figure(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = True,
    figsize: tuple = (6, 5),
    cmap: str = "Blues",
    label_display_map: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Create a confusion matrix heatmap figure for embedding in Qt.

    Unlike confusion_matrix_heatmap(), this function does not call plt.show().
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    total = float(np.sum(cm))
    acc = float(np.trace(cm) / total) if total > 0 else 0.0
    if normalize:
        with np.errstate(all="ignore"):
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_display)
        annot = np.empty_like(cm_display, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm_display[i, j]:.2f}\n({cm[i, j]})"
    else:
        cm_display = cm
        annot = cm

    # NOTE:
    # We intentionally avoid seaborn.heatmap() here.
    # In the Qt app we render via MatplotlibWidget which historically copies
    # Axes elements across Figures. Seaborn uses QuadMesh which can be fragile
    # to copy. Using imshow + text annotations is robust.

    fig = plt.Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Use stable color scaling so the heatmap is comparable across runs.
    if normalize:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, float(np.max(cm)) if np.max(cm) > 0 else 1.0

    im = ax.imshow(cm_display, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")

    # Ticks/labels (allow overriding display labels)
    display_labels = [_display_label(lab, label_display_map) for lab in class_labels]
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(display_labels)
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    # Add more context in the title (matches typical "reference" confusion matrix layouts)
    norm_suffix = " (normalized)" if normalize else ""
    ax.set_title(f"{title}{norm_suffix}\nAccuracy: {acc:.2%}")
    # Keep x tick labels readable
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    try:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    except Exception:
        pass

    # Annotate cells
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            txt = annot[i, j]
            # Contrast-aware text color
            try:
                v = float(cm_display[i, j])
            except Exception:
                v = 0.0
            color = "white" if v >= 0.5 else "black"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )

    fig.tight_layout()
    return fig


def create_roc_curve_figure(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_labels: List[str],
    title: str = "ROC Curve",
    figsize: tuple = (6, 5),
    label_display_map: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Create ROC curve figure.

    For binary: y_score can be shape (n_samples,) or (n_samples, 2).
    For multiclass: y_score must be shape (n_samples, n_classes) and a one-vs-rest
    macro-average curve will be plotted.
    """
    fig = plt.Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Map labels to integer indices
    label_to_idx = {lab: i for i, lab in enumerate(class_labels)}
    y_int = np.asarray([label_to_idx.get(str(v), -1) for v in y_true], dtype=int)
    valid = y_int >= 0
    if not np.all(valid):
        y_int = y_int[valid]
        y_score = np.asarray(y_score)[valid]

    y_score = np.asarray(y_score)
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        pos_score = y_score[:, 1]
    else:
        pos_score = y_score

    if len(class_labels) == 2 and pos_score.ndim == 1:
        fpr, tpr, _ = roc_curve(y_int, pos_score, pos_label=1)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
    else:
        # One-vs-rest macro average
        # Build one-hot true labels
        n_classes = len(class_labels)
        Y = np.zeros((y_int.shape[0], n_classes), dtype=int)
        Y[np.arange(y_int.shape[0]), y_int] = 1
        if y_score.ndim != 2 or y_score.shape[1] != n_classes:
            raise ValueError("For multiclass ROC, y_score must be (n_samples, n_classes)")

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(Y[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            disp = _display_label(class_labels[i], label_display_map)
            ax.plot(fpr[i], tpr[i], linewidth=1.5, label=f"{disp} (AUC={roc_auc[i]:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def create_prediction_distribution_figure(
    *,
    y_true: np.ndarray | None,
    y_score: np.ndarray,
    class_labels: List[str],
    title: str = "Prediction Distribution",
    figsize: tuple = (6, 5),
    bins: int = 20,
    label_display_map: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Create a histogram of predicted probabilities.

    - Binary case: uses probability of positive class.
      If y_true is provided, distributions are separated by true class.

    - Multiclass case: plots a histogram for each class probability column.
      This avoids Matplotlib's "one color per dataset" error and supports >2 groups.
    """
    fig = plt.Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    y_score = np.asarray(y_score)

    # Binary: pick positive class column
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        score = y_score[:, 1]
        if y_true is None or len(class_labels) != 2:
            ax.hist(score, bins=bins, alpha=0.85, color="#1f77b4")
        else:
            # Split by true label
            label_to_idx = {lab: i for i, lab in enumerate(class_labels)}
            y_int = np.asarray([label_to_idx.get(str(v), -1) for v in y_true], dtype=int)
            mask0 = y_int == 0
            mask1 = y_int == 1
            ax.hist(
                score[mask0],
                bins=bins,
                alpha=0.65,
                label=_display_label(class_labels[0], label_display_map),
                color="#1f77b4",
            )
            ax.hist(
                score[mask1],
                bins=bins,
                alpha=0.65,
                label=_display_label(class_labels[1], label_display_map),
                color="#ff7f0e",
            )
            ax.legend(loc="best")

        ax.set_xlabel("Predicted probability")

    # Multiclass: plot each class probability distribution
    elif y_score.ndim == 2 and y_score.shape[1] >= 3:
        n_classes = int(y_score.shape[1])
        labels = [_display_label(lab, label_display_map) for lab in list(class_labels)]
        if len(labels) != n_classes:
            labels = [f"class_{i}" for i in range(n_classes)]
        cmap = plt.get_cmap("tab10")
        for i in range(n_classes):
            color = cmap(i % 10)
            ax.hist(
                y_score[:, i],
                bins=bins,
                alpha=0.45,
                label=str(labels[i]),
                color=color,
            )
        ax.legend(loc="best")
        ax.set_xlabel("Predicted probability (per class)")

    # Fallback (e.g., 1D scores)
    else:
        score = y_score
        ax.hist(score, bins=bins, alpha=0.85, color="#1f77b4")
        ax.set_xlabel("Score")

    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def create_feature_importance_figure(
    *,
    feature_importances: np.ndarray,
    wavenumbers: np.ndarray | None = None,
    title: str = "Feature Importance",
    figsize: tuple = (7, 4),
    top_k: int | None = None,
) -> plt.Figure:
    """Plot feature importance as a spectrum-like line plot.

    If wavenumbers are provided and match feature_importances length, they are used
    as the x-axis.

    If top_k is provided, will display only the top_k most important features as a stem plot.
    """
    fi = np.asarray(feature_importances, dtype=float)
    fig = plt.Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if wavenumbers is not None:
        x = np.asarray(wavenumbers, dtype=float)
        if x.shape[0] != fi.shape[0]:
            x = None
    else:
        x = None

    if top_k is not None and top_k > 0:
        idx = np.argsort(fi)[::-1][: int(top_k)]
        idx_sorted = np.sort(idx)
        x_plot = x[idx_sorted] if x is not None else idx_sorted
        ax.stem(x_plot, fi[idx_sorted], basefmt=" ", linefmt="#28a745", markerfmt="o")
        ax.set_xlabel("Wavenumber (cm⁻¹)" if x is not None else "Feature index")
    else:
        x_plot = x if x is not None else np.arange(fi.shape[0])
        ax.plot(x_plot, fi, color="#28a745", linewidth=1.5)
        ax.set_xlabel("Wavenumber (cm⁻¹)" if x is not None else "Feature index")

    ax.set_ylabel("Importance")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


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
