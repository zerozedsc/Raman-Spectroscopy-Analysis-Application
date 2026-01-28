"""
Exploratory Analysis Methods

This module implements exploratory data analysis methods like PCA, UMAP,
t-SNE, and clustering techniques.
"""

import numpy as np
import pandas as pd
import seaborn as sns  # ✅ Phase 4: Publication-quality statistical visualizations
from typing import Dict, Any, Callable, Optional, Tuple, List
import matplotlib
import warnings

from configs.configs import create_logs

matplotlib.use("Agg")  # Use non-GUI backend for thread safety
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from scipy import stats
from scipy.interpolate import interp1d

try:
    import umap.umap_ as umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def _log_debug(message: str, *args) -> None:
    """Lightweight debug logger that routes through create_logs.

    Supports old-style logging format strings ("%s") to minimize churn.
    """

    if args:
        try:
            message = message % args
        except Exception:
            message = f"{message} {args!r}"
    create_logs(__name__, __file__, message, status="debug")


def _safe_tight_layout(fig: Figure, *, pad: float = 1.2, rect=None) -> None:
    """Apply tight_layout without spamming warnings.

    Some figures (legends, colorbars, 3D axes) are not compatible with tight_layout
    and Matplotlib emits a UserWarning to stderr. For interactive GUI usage, repeated
    redraws can flood logs/terminal.
    """

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"This figure includes Axes that are not compatible with tight_layout.*",
                category=UserWarning,
            )
            if rect is not None:
                fig.tight_layout(pad=pad, rect=rect)
            else:
                fig.tight_layout(pad=pad)
    except Exception:
        # Best-effort fallback
        try:
            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.92)
        except Exception:
            pass


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
        return ["#0066cc", "#ff4444"]
    elif num_groups == 3:
        # High contrast: Blue, Red, Green
        return ["#0066cc", "#ff4444", "#00cc66"]
    elif num_groups == 4:
        # Blue, Red, Green, Orange
        return ["#0066cc", "#ff4444", "#00cc66", "#ff9900"]
    elif num_groups == 5:
        # Blue, Red, Green, Orange, Purple
        return ["#0066cc", "#ff4444", "#00cc66", "#ff9900", "#9933ff"]
    elif num_groups == 6:
        # Blue, Red, Green, Orange, Purple, Cyan
        return ["#0066cc", "#ff4444", "#00cc66", "#ff9900", "#9933ff", "#00cccc"]
    else:
        # For 7+ groups, use tab10 with good spacing
        colors = plt.cm.tab10(np.linspace(0, 0.9, num_groups))
        return [
            f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
            for c in colors
        ]


# =============================================================================
# WAVENUMBER INTERPOLATION for multi-dataset dimension mismatch fix
# =============================================================================
def interpolate_to_common_wavenumbers(
    dataset_data: Dict[str, pd.DataFrame], method: str = "linear"
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
    _log_debug("interpolate_to_common_wavenumbers() called")

    # Step 1: Find common wavenumber range
    wn_mins = []
    wn_maxs = []
    wn_counts = []

    for dataset_name, df in dataset_data.items():
        wavenumbers = df.index.values.astype(float)
        wn_mins.append(np.min(wavenumbers))
        wn_maxs.append(np.max(wavenumbers))
        wn_counts.append(len(wavenumbers))
        _log_debug(
            f"Dataset '{dataset_name}': {len(wavenumbers)} points, range [{np.min(wavenumbers):.1f}, {np.max(wavenumbers):.1f}]"
        )

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
    avg_density = np.mean(wn_counts) / np.mean(
        [mx - mn for mn, mx in zip(wn_mins, wn_maxs)]
    )
    n_common_points = int(avg_density * (common_max - common_min))
    n_common_points = max(n_common_points, 50)  # At least 50 points
    n_common_points = min(
        n_common_points, max(wn_counts)
    )  # Don't exceed original resolution

    common_wavenumbers = np.linspace(common_min, common_max, n_common_points)

    _log_debug(
        f"Common wavenumber range: [{common_min:.1f}, {common_max:.1f}] with {n_common_points} points"
    )

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
                fill_value="extrapolate",
                bounds_error=False,
            )

            interpolated[:, col_idx] = interp_func(common_wavenumbers)

        # Transpose to get (n_spectra, n_wavenumbers) for vstack
        interpolated_spectra.append(interpolated.T)
        labels.extend([dataset_name] * spectra.shape[1])

        _log_debug(
            f"Interpolated '{dataset_name}': {spectra.shape[1]} spectra, now {n_common_points} points each"
        )

    # Step 3: Concatenate all interpolated spectra
    X = np.vstack(interpolated_spectra)

    _log_debug(f"Final combined matrix shape: {X.shape}")

    return common_wavenumbers, interpolated_spectra, labels, X


def interpolate_to_common_wavenumbers_with_groups(
    dataset_data: Dict[str, pd.DataFrame],
    group_labels_map: Optional[Dict[str, str]] = None,
    method: str = "linear",
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


def add_confidence_ellipse(
    ax,
    x,
    y,
    n_std=1.96,
    facecolor="none",
    edgecolor="red",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=None,
):
    """
    Add a confidence ellipse to a matplotlib axis using DUAL-LAYER PATTERN.


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
    color_to_use = edgecolor if facecolor == "none" else facecolor
    ellipse_fill = Ellipse(
        xy=(mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        facecolor=color_to_use,
        edgecolor="none",  # No edge on fill layer
        alpha=0.04,  # ✅ Ultra-light fill (4% opacity) - reduced for less dark overlay
        zorder=5,
    )
    ax.add_patch(ellipse_fill)

    # ✅ DUAL-LAYER PATTERN: Layer 2 - Bold visible edge (strong boundary)
    ellipse_edge = Ellipse(
        xy=(mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",  # No fill on edge layer
        edgecolor=edgecolor,
        linestyle=linestyle,
        linewidth=linewidth if linewidth else 2.5,  # ✅ Thicker edge
        alpha=0.04,
        label=label,  # Only the edge gets the label for legend
        zorder=15,  # Above scatter points
    )
    ax.add_patch(ellipse_edge)

    _log_debug(
        f"Dual-layer ellipse added: center=({mean_x:.2f}, {mean_y:.2f}), size={width:.2f}x{height:.2f}, fill α=0.04, edge α=0.85"
    )
    return ellipse_edge  # Return edge ellipse for legend


def perform_pca_analysis(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
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
    enable_pca_lda = params.get("enable_pca_lda", False)
    pca_lda_cv_folds = params.get("pca_lda_cv_folds", 5)
    show_ellipses = params.get(
        "show_ellipses", True
    )  # Confidence ellipses (critical for Chemometrics)
    show_loadings = params.get("show_loadings", True)
    show_scree = params.get("show_scree", True)
    show_distributions = params.get("show_distributions", True)
    group_labels_map = params.get("_group_labels", None)  # {dataset_name: group_label}

    _log_debug(
        f"PCA parameters: n_components={n_components}, show_ellipses={show_ellipses}"
    )
    _log_debug(
        f"show_loadings={show_loadings}, show_scree={show_scree}, show_distributions={show_distributions}"
    )

    # UX / safety: distributions rely on the same PCA outputs used by loadings.
    # Users reported that enabling distributions alone can lead to failures; ensure loadings is on.
    if show_distributions and not show_loadings:
        _log_debug("Enabling show_loadings automatically because show_distributions=True")
        show_loadings = True

    # ✅ FIX: Use interpolation to handle datasets with different wavenumber ranges
    # This resolves "ValueError: all input array dimensions except for concatenation axis must match"
    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data, group_labels_map=group_labels_map, method="linear"
    )

    _log_debug(f"Combined matrix after interpolation: {X.shape}")
    _log_debug(
        f"Common wavenumber range: [{wavenumbers[0]:.1f}, {wavenumbers[-1]:.1f}]"
    )

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
    _log_debug("Creating PCA scores plot")
    fig1, ax1 = plt.subplots(figsize=(10, 8))

    unique_labels = sorted(set(labels))
    num_groups = len(unique_labels)
    _log_debug(f"Number of groups/datasets: {num_groups}")
    _log_debug(f"Group labels: {unique_labels}")

    # Use HIGH-CONTRAST color palette for clear distinction
    # For 2 datasets: blue (#1f77b4) and yellow/gold (#ffd700)
    # For 3+ datasets: use qualitative palettes with maximum contrast
    if num_groups == 2:
        # Maximum contrast for 2 groups: blue and yellow-gold
        colors = np.array(
            [[0.12, 0.47, 0.71, 1.0], [1.0, 0.84, 0.0, 1.0]]  # Blue
        )  # Gold/Yellow
        _log_debug("Using high-contrast 2-color palette: Blue and Gold")
    elif num_groups == 3:
        # High contrast for 3 groups: blue, red, green
        colors = np.array(
            [
                [0.12, 0.47, 0.71, 1.0],  # Blue
                [0.84, 0.15, 0.16, 1.0],  # Red
                [0.17, 0.63, 0.17, 1.0],
            ]
        )  # Green
        _log_debug("Using high-contrast 3-color palette: Blue, Red, Green")
    else:
        # For 4+ groups, use tab10 but with better spacing
        colors = plt.cm.tab10(np.linspace(0, 0.9, num_groups))
        _log_debug(f"Using tab10 palette for {num_groups} groups")

    # Plot each dataset with distinct color
    for i, dataset_label in enumerate(unique_labels):
        mask = np.array([l == dataset_label for l in labels])
        num_points = np.sum(mask)
        _log_debug(f"Group '{dataset_label}': {num_points} spectra")

        ax1.scatter(
            scores[mask, 0],
            scores[mask, 1],
            c=[colors[i]],
            label=dataset_label,
            alpha=0.7,
            s=100,
            edgecolors="white",
            linewidth=1.0,
        )

        # Add 95% confidence ellipse (CRITICAL for Chemometrics) - controlled by parameter
        if (
            show_ellipses and num_points >= 3
        ):  # User-controlled + need at least 3 points
            _log_debug(
                f"Adding 95% CI ellipse for '{dataset_label}' ({num_points} points, show_ellipses=True)"
            )
            add_confidence_ellipse(
                ax1,
                scores[mask, 0],
                scores[mask, 1],
                n_std=1.96,  # 95% confidence interval
                edgecolor=colors[i],
                linestyle="--",
                linewidth=2,
                alpha=0.6,
                label=f"{dataset_label} 95% CI",
            )
        elif not show_ellipses:
            _log_debug(
                f"Ellipses disabled by user (show_ellipses=False) for '{dataset_label}'"
            )
        else:
            _log_debug(
                f"Skipping ellipse for '{dataset_label}' (only {num_points} points, need ≥3)"
            )

    # Optional: PCA→LDA overlay (decision boundary in PC1–PC2)
    pca_lda_info = None
    if enable_pca_lda:
        try:
            from matplotlib.colors import ListedColormap
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.model_selection import StratifiedKFold, cross_val_predict
            from sklearn.metrics import confusion_matrix, accuracy_score

            if scores.shape[1] < 2:
                raise ValueError(
                    "PCA→LDA requires at least 2 PCA components to plot the decision boundary."
                )

            X_2d = scores[:, :2]
            label_to_int = {lab: idx for idx, lab in enumerate(unique_labels)}
            y_int = np.array([label_to_int[l] for l in labels], dtype=int)

            # Fit a simple LDA on the 2D PC scores
            lda = LinearDiscriminantAnalysis()

            # CV metrics (best-effort: reduce folds if a class has too few samples)
            min_class_count = int(np.min(np.bincount(y_int))) if len(y_int) else 0
            n_splits = int(max(2, min(pca_lda_cv_folds, min_class_count)))
            cv_accuracy = None
            cv_cm = None
            if n_splits >= 2 and len(unique_labels) >= 2:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                y_pred = cross_val_predict(lda, X_2d, y_int, cv=cv)
                cv_accuracy = float(accuracy_score(y_int, y_pred))
                cv_cm = confusion_matrix(y_int, y_pred)

            lda.fit(X_2d, y_int)

            # Decision regions
            x_min, x_max = float(np.min(X_2d[:, 0]) - 1.0), float(np.max(X_2d[:, 0]) + 1.0)
            y_min, y_max = float(np.min(X_2d[:, 1]) - 1.0), float(np.max(X_2d[:, 1]) + 1.0)
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 250),
                np.linspace(y_min, y_max, 250),
            )
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z = lda.predict(grid).reshape(xx.shape)

            # Use the same palette used for points (truncate if needed)
            base_colors = [tuple(c[:3]) for c in colors[: len(unique_labels)]]
            cmap = ListedColormap(base_colors)
            # Make the overlay visible (users reported "no change"), but keep it subtle.
            # Use a low zorder so points/ellipses stay on top.
            ax1.contourf(xx, yy, Z, cmap=cmap, alpha=0.18, antialiased=True, zorder=0)

            # Boundary lines
            if len(unique_labels) == 2:
                ax1.contour(xx, yy, Z, levels=[0.5], colors=["#333333"], linewidths=1.8, zorder=1)
            else:
                ax1.contour(xx, yy, Z, colors=["#333333"], linewidths=1.0, alpha=0.7, zorder=1)

            pca_lda_info = {
                "enabled": True,
                "cv_folds_used": n_splits,
                "cv_accuracy": cv_accuracy,
                "cv_confusion_matrix": cv_cm,
            }

            _log_debug(
                f"PCA→LDA overlay enabled (folds_used={n_splits}, cv_accuracy={cv_accuracy})"
            )
        except Exception as e:
            # Never fail the PCA analysis just because LDA overlay failed
            _log_debug(f"PCA→LDA overlay failed: {e}")
            pca_lda_info = {"enabled": False, "error": str(e)}

    ax1.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )

    # ✅ FIX #6 (P1): Clear title and legend labels
    # Title changes based on whether ellipses are shown
    if show_ellipses:
        ax1.set_title(
            "PCA Score Plot with 95% Confidence Ellipses",
            fontsize=14,
            fontweight="bold",
        )
        # Add explanatory footnote for scientific clarity
        ax1.text(
            0.02,
            0.02,
            "95% Confidence Ellipses calculated using Hotelling's T² (1.96σ)",
            transform=ax1.transAxes,
            fontsize=9,
            color="#555555",
            style="italic",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.92,
                edgecolor="#cccccc",
                linewidth=0.5,
            ),
        )
    else:
        ax1.set_title("PCA Score Plot", fontsize=14, fontweight="bold")

    # Larger legend with better visibility
    ax1.legend(
        loc="best",
        framealpha=0.95,
        fontsize=10,
        edgecolor="#cccccc",
        fancybox=True,
        shadow=True,
    )
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    ax1.axvline(x=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

    _log_debug("PCA scores plot created successfully")

    # === FIGURE 3: Scree Plot (Variance Explained) ===
    fig_scree = None
    if show_scree:
        _log_debug("Creating scree plot...")

        # ✅ FIX #7 (P1): Side-by-side layout (bar LEFT | cumulative RIGHT)
        from matplotlib.gridspec import GridSpec

        fig_scree = plt.figure(figsize=(14, 5.5))
        gs = GridSpec(1, 2, figure=fig_scree, wspace=0.25)

        pc_indices = np.arange(1, n_components + 1)
        explained_variance = pca.explained_variance_ratio_ * 100
        cumulative_variance = np.cumsum(explained_variance)

        # LEFT: Bar chart for individual variance
        ax_bar = fig_scree.add_subplot(gs[0, 0])
        bar_colors = [
            "#e74c3c" if var > 10 else "#4a90e2" for var in explained_variance
        ]
        bars = ax_bar.bar(
            pc_indices,
            explained_variance,
            color=bar_colors,
            edgecolor="white",
            linewidth=1.5,
            alpha=0.85,
            width=0.65,
        )

        for bar, var in zip(bars, explained_variance):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{var:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax_bar.set_xlabel("Principal Component", fontsize=12, fontweight="bold")
        ax_bar.set_ylabel("Variance Explained (%)", fontsize=12, fontweight="bold")
        ax_bar.set_title("Individual Variance per PC", fontsize=13, fontweight="bold")
        ax_bar.set_xticks(pc_indices)
        ax_bar.set_ylim(0, max(explained_variance) * 1.15)
        ax_bar.grid(True, alpha=0.3, axis="y", linestyle="--")

        # RIGHT: Cumulative variance line
        ax_cum = fig_scree.add_subplot(gs[0, 1])
        ax_cum.plot(
            pc_indices,
            cumulative_variance,
            marker="o",
            markersize=9,
            linewidth=2.8,
            color="#2ecc71",
            markeredgecolor="white",
            markeredgewidth=1.5,
            alpha=0.95,
            label="Cumulative",
        )
        ax_cum.axhline(
            y=80,
            color="#f39c12",
            linestyle="--",
            linewidth=2.5,
            alpha=0.75,
            label="80% Threshold",
        )
        ax_cum.axhline(
            y=95,
            color="#e74c3c",
            linestyle="--",
            linewidth=2.5,
            alpha=0.75,
            label="95% Threshold",
        )

        for i, cum in enumerate(cumulative_variance):
            if i < 5:
                ax_cum.text(
                    i + 1,
                    cum + 2,
                    f"{cum:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#2ecc71",
                )

        ax_cum.set_xlabel("Principal Component", fontsize=12, fontweight="bold")
        ax_cum.set_ylabel("Cumulative Variance (%)", fontsize=12, fontweight="bold")
        ax_cum.set_title(
            "Cumulative Variance Explained", fontsize=13, fontweight="bold"
        )
        ax_cum.set_xticks(pc_indices)
        ax_cum.set_ylim(0, 105)
        ax_cum.grid(True, alpha=0.3, linestyle="--")
        ax_cum.legend(loc="lower right", fontsize=10, framealpha=0.9)

        fig_scree.tight_layout(pad=1.2)
        _log_debug("Side-by-side scree plot created successfully")

    # === FIGURE 4: Biplot (Scores + Loadings Overlay) ===
    fig_biplot = None
    if show_loadings and n_components >= 2:
        _log_debug("Creating biplot...")
        fig_biplot, ax_biplot = plt.subplots(figsize=(12, 10))

        # Plot scores (same as primary figure but without ellipses for clarity)
        for i, dataset_label in enumerate(unique_labels):
            mask = np.array([l == dataset_label for l in labels])
            ax_biplot.scatter(
                scores[mask, 0],
                scores[mask, 1],
                c=[colors[i]],
                label=dataset_label,
                s=60,
                alpha=0.6,
                edgecolors="white",
                linewidths=0.5,
            )

        # Overlay loadings as arrows (scaled for visibility)
        loading_scale = np.max(np.abs(scores[:, :2])) * 0.8

        # Select top contributing wavenumbers (peaks in loadings)
        pc1_loadings = pca.components_[0]
        pc2_loadings = pca.components_[1]
        loading_magnitude = np.sqrt(pc1_loadings**2 + pc2_loadings**2)

        # Show top 15 most influential wavenumbers
        top_indices = np.argsort(loading_magnitude)[-15:]

        for idx in top_indices:
            ax_biplot.arrow(
                0,
                0,
                pc1_loadings[idx] * loading_scale,
                pc2_loadings[idx] * loading_scale,
                head_width=loading_scale * 0.02,
                head_length=loading_scale * 0.03,
                fc="#d13438",
                ec="#8b0000",
                alpha=0.8,
                linewidth=0.8,
            )

            # Label with wavenumber - thinner text, no box to be cleaner
            ax_biplot.text(
                pc1_loadings[idx] * loading_scale * 1.15,
                pc2_loadings[idx] * loading_scale * 1.15,
                f"{int(wavenumbers[idx])}",
                fontsize=8,
                ha="center",
                va="center",
                color="#8b0000",
                fontweight="bold",
            )

        ax_biplot.set_xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
            fontsize=12,
            fontweight="bold",
        )
        ax_biplot.set_ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
            fontsize=12,
            fontweight="bold",
        )
        ax_biplot.set_title(
            "PCA Biplot: Scores + Influential Wavenumbers",
            fontsize=14,
            fontweight="bold",
        )
        ax_biplot.legend(loc="best", fontsize=11, framealpha=0.9)
        ax_biplot.grid(True, alpha=0.3)
        ax_biplot.axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.3)
        ax_biplot.axvline(x=0, color="k", linestyle="--", linewidth=0.5, alpha=0.3)
        fig_biplot.tight_layout()
        _log_debug("Biplot created successfully")

    # === FIGURE 5: Cumulative Variance Explained ===
    fig_cumvar = None
    if show_scree:
        _log_debug("Creating cumulative variance plot...")
        fig_cumvar, ax_cumvar = plt.subplots(figsize=(10, 6))

        pc_indices = np.arange(1, n_components + 1)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_ * 100)

        # Area plot
        ax_cumvar.fill_between(
            pc_indices, cumulative_variance, alpha=0.4, color="#28a745"
        )
        ax_cumvar.plot(
            pc_indices,
            cumulative_variance,
            color="#28a745",
            marker="o",
            linewidth=3,
            markersize=10,
            markerfacecolor="white",
            markeredgewidth=2,
            markeredgecolor="#28a745",
        )

        # Add threshold lines
        ax_cumvar.axhline(
            y=80,
            color="#ffc107",
            linestyle="--",
            linewidth=2,
            label="80% Threshold",
            alpha=0.7,
        )
        ax_cumvar.axhline(
            y=95,
            color="#dc3545",
            linestyle="--",
            linewidth=2,
            label="95% Threshold",
            alpha=0.7,
        )

        # Annotate values
        for i, cum_var in enumerate(cumulative_variance):
            ax_cumvar.text(
                i + 1,
                cum_var + 2,
                f"{cum_var:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax_cumvar.set_xlabel(
            "Number of Principal Components", fontsize=12, fontweight="bold"
        )
        ax_cumvar.set_ylabel(
            "Cumulative Variance Explained (%)", fontsize=12, fontweight="bold"
        )
        ax_cumvar.set_title(
            "Cumulative Variance Explained", fontsize=14, fontweight="bold"
        )
        ax_cumvar.set_xticks(pc_indices)
        ax_cumvar.set_ylim(0, 105)
        ax_cumvar.grid(True, alpha=0.3)
        ax_cumvar.legend(loc="lower right", fontsize=11)
        fig_cumvar.tight_layout()
        _log_debug("Cumulative variance plot created successfully")

    if progress_callback:
        progress_callback(75)

    # === FIGURE 2: Loadings Plot (Spectral interpretation) - ENHANCED WITH SUBPLOTS ===
    _log_debug(f"show_loadings parameter: {show_loadings}")
    fig_loadings = None
    if show_loadings:
        _log_debug("Creating loadings figure with dynamic subplot layout...")
        
        # Get max_loadings_components parameter (default 2)
        # Respect the requested count up to computed n_components.
        max_loadings = params.get("max_loadings_components", 2)
        max_loadings = min(max_loadings, n_components)  # Ensure within bounds

        _log_debug(f"Creating {max_loadings} loading subplot(s)")
        
        # ✅ Dynamic layout calculation (max 2 columns, auto rows)
        n_cols = 2 if max_loadings > 1 else 1
        n_rows = int(np.ceil(max_loadings / n_cols))
        
        # ✅ Dynamic title size based on number of plots
        base_title_size = 14
        if max_loadings <= 2:
            title_size = base_title_size
        elif max_loadings <= 4:
            title_size = base_title_size - 2  # 12pt
        else:
            title_size = base_title_size - 4  # 10pt
        
        # ✅ Scale figure height dynamically to prevent overlap
        fig_height = max(6, 2.5 * n_rows)  # At least 2.5 inches per row
        
        # Create subplot grid with constrained layout for better spacing
        fig_loadings, axes = plt.subplots(n_rows, n_cols, figsize=(12, fig_height),
                                          constrained_layout=True)
        
        # Handle single subplot case (axes won't be array)
        if max_loadings == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Dynamic color palette using matplotlib's tab20 colormap
        cmap = plt.get_cmap('tab20', max(max_loadings, 10))
        colors = [cmap(i) for i in range(max_loadings)]
        
        # Plot each component in its own subplot
        for pc_idx in range(max_loadings):
            ax = axes[pc_idx]
            
            # ✅ FIX: Ensure wavenumbers and loadings have same length
            loadings = pca.components_[pc_idx]
            min_len = min(len(wavenumbers), len(loadings))
            wn_truncated = wavenumbers[:min_len]
            loadings_truncated = loadings[:min_len]
            
            # Plot loadings for this component
            ax.plot(wn_truncated, loadings_truncated, 
                   linewidth=2, color=colors[pc_idx], label=f'PC{pc_idx+1}')
            
            # Explained variance for this component
            explained_var = pca.explained_variance_ratio_[pc_idx] * 100
            
            # ✅ Calculate if this subplot is in the last row
            current_row = pc_idx // n_cols
            is_last_row = (current_row == n_rows - 1)
            
            # Styling
            ax.set_ylabel('Loading Value', fontsize=11, fontweight='bold')
            ax.set_title(f'PC{pc_idx+1} Loadings ({explained_var:.2f}%)', 
                        fontsize=title_size, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            # Display wavenumber increasing left → right (avoid Raman-style inverted axis).
            
            # ✅ Only show x-axis labels and ticks on last row
            if is_last_row:
                ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11, fontweight='bold')
            else:
                ax.set_xlabel('')  # Clear x-axis label
                ax.tick_params(axis='x', labelbottom=False)  # Hide x tick labels
                ax.set_xticklabels([])  # Force remove all x tick labels
            
            # ✅ Annotate top 5 peak positions for this component (use truncated arrays)
            abs_loadings = np.abs(loadings_truncated)
            top_indices = np.argsort(abs_loadings)[-5:]  # Top 5 peaks

            # Avoid overlapping labels by assigning discrete vertical levels.
            # (This also makes dragging less necessary for typical cases.)
            peak_idx_sorted = sorted(list(top_indices), key=lambda i: float(wn_truncated[i]))
            wn_span = float(np.nanmax(wn_truncated) - np.nanmin(wn_truncated)) if len(wn_truncated) else 1.0
            min_sep_wn = max(15.0, wn_span / 40.0)
            used_levels: Dict[int, List[float]] = {lvl: [] for lvl in range(6)}

            label_offsets: Dict[int, int] = {}
            for peak_idx in peak_idx_sorted:
                x = float(wn_truncated[peak_idx])
                chosen_level = None
                for lvl in range(6):
                    if all(abs(x - prev_x) >= min_sep_wn for prev_x in used_levels[lvl]):
                        chosen_level = lvl
                        used_levels[lvl].append(x)
                        break
                if chosen_level is None:
                    chosen_level = 5
                    used_levels[chosen_level].append(x)

                # Offset direction depends on sign of loading (keeps labels on the 'outside')
                sign = 1 if float(loadings_truncated[peak_idx]) >= 0 else -1
                label_offsets[int(peak_idx)] = sign * (12 + chosen_level * 10)

            for peak_idx in peak_idx_sorted:
                peak_wn = wn_truncated[peak_idx]
                peak_val = loadings_truncated[peak_idx]
                ax.plot(peak_wn, peak_val, 'o', color=colors[pc_idx], markersize=6, 
                       markeredgecolor='black', markeredgewidth=0.5)
                ax.annotate(f'{peak_wn:.0f}', 
                           xy=(peak_wn, peak_val), 
                           xytext=(0, int(label_offsets.get(int(peak_idx), 10 if peak_val > 0 else -15))),
                           textcoords='offset points',
                           fontsize=8, fontweight='bold',
                           ha='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#cccccc'))
        
        # Hide unused subplots if grid has extras
        for pc_idx in range(max_loadings, len(axes)):
            axes[pc_idx].set_visible(False)
        
        # No need for tight_layout when using constrained_layout
        _log_debug(
            f"Loadings figure created successfully with {max_loadings} subplots in {n_rows}x{n_cols} grid (constrained layout)"
        )
    else:
        _log_debug("Loadings figure skipped (show_loadings=False)")

    if progress_callback:
        progress_callback(80)

    # === FIGURE 3: Score Distributions (CRITICAL for Raman classification) ===
    fig_distributions = None
    if show_distributions and len(unique_labels) > 1:
        # Get number of components to show (default 2)
        n_dist_comps = params.get("n_distribution_components", 2)
        n_pcs_to_plot = min(n_dist_comps, n_components)
        
        # ✅ Dynamic layout calculation (max 2 columns, auto rows)
        n_cols = 2 if n_pcs_to_plot > 1 else 1
        n_rows = int(np.ceil(n_pcs_to_plot / n_cols))
        
        # ✅ Dynamic title size based on number of plots
        base_title_size = 14
        if n_pcs_to_plot <= 2:
            title_size = base_title_size
            main_title_size = 16
        elif n_pcs_to_plot <= 4:
            title_size = base_title_size - 2  # 12pt
            main_title_size = 14
        else:
            title_size = base_title_size - 3  # 11pt
            main_title_size = 13
        
        # ✅ Scale figure height dynamically
        fig_height = max(6, 2.5 * n_rows)
        
        # ✅ USER FIX: Use sharex/sharey for cleaner multi-plot layout (no redundant labels)
        # Create figure with constrained layout and shared axes
        fig_distributions, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6*n_cols, fig_height),
            sharex=True,  # Share x-axis across all subplots
            sharey=True,  # Share y-axis across all subplots
            constrained_layout=True
        )
        
        # Flatten axes for easy iteration if multiple
        if n_pcs_to_plot > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
            
        # ✅ USER FIX: Use simpler suptitle without redundant "PC Score Distributions"
        # Each subplot will have its own PC label (e.g., "PC1 45.3%")
        fig_distributions.suptitle('Principal Component Distributions', fontsize=main_title_size, fontweight='bold')
        
        # Plot distributions for each PC
        for idx in range(n_pcs_to_plot):
            ax = axes_flat[idx]
            pc_idx = idx  # 0-based index
            
            # ✅ Calculate if this subplot is in the last row
            current_row = idx // n_cols
            is_last_row = (current_row == n_rows - 1)
            
            # ✅ Phase 4: Use seaborn for publication-quality KDE plots
            for i, dataset_label in enumerate(unique_labels):
                mask = np.array([l == dataset_label for l in labels])
                pc_scores = scores[mask, pc_idx]

                # Robust color selection (avoid IndexError if palette shorter than groups)
                # colors may be a numpy array; never use it in boolean context (ambiguous truth value).
                color = colors[i % len(colors)] if colors is not None and len(colors) > 0 else None

                # Seaborn kdeplot with automatic bandwidth selection (better than manual KDE)
                try:
                    sns.kdeplot(
                        x=pc_scores,
                        ax=ax,
                        color=color,
                        linewidth=2.5,
                        label=dataset_label,
                        fill=True,
                        alpha=0.25,
                        bw_adjust=0.75,  # Smoother curves for Raman data
                        common_norm=False,  # Normalize each distribution independently
                    )
                    
                    # ✅ Add rug plot to show actual data points (publication standard)
                    sns.rugplot(
                        x=pc_scores,
                        ax=ax,
                        color=color,
                        alpha=0.5,
                        height=0.05,
                        linewidth=1.0,
                    )
                except Exception as e:
                    _log_debug(f"Seaborn KDE failed for {dataset_label}, PC{pc_idx+1}: {e}")
                    # Fallback to histogram if seaborn fails
                    ax.hist(
                        pc_scores,
                        bins=20,
                        density=True,
                        alpha=0.3,
                        color=color,
                        edgecolor="white",
                        linewidth=0.5,
                        label=dataset_label,
                    )

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
                    pooled_std = np.sqrt(
                        (np.std(pc1_scores) ** 2 + np.std(pc2_scores) ** 2) / 2
                    )
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

                    # Add statistical annotation
                    ax.text(
                        0.05,
                        0.95,
                        f"Mann–Whitney U\np={p_value:.2e}\nδ={cohens_d:.2f}",
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
                    )
                except Exception:
                    pass
            
            # ✅ USER FIX: Remove individual x/y axis labels (user: "it looks not proper and ugly")
            # Only use subplot titles to show which PC is displayed
            # The shared x-axis label "PC Score" and shared y-axis label "Density" will be added later
            
            # Subplot title shows PC number and variance %
            ax.set_title(f'PC{pc_idx+1} ({pca.explained_variance_ratio_[pc_idx]*100:.1f}%)',
                        fontsize=title_size, fontweight='bold')
            
            # ✅ Remove individual axis labels (sharex/sharey handles this automatically)
            # matplotlib will show labels only on edge plots when sharex/sharey is used
            
            # Show legend only on first plot to save space
            if idx == 0:
                ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
            ax.grid(True, alpha=0.3, axis="y")
            ax.axvline(x=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
        
        # ✅ USER FIX: Add shared axis labels as figure-level labels (cleaner for multi-plot)
        # This gives a publication-quality look with minimal repetition
        fig_distributions.supxlabel('PC Score', fontsize=13, fontweight='bold')
        fig_distributions.supylabel('Density', fontsize=13, fontweight='bold')

        # Hide empty subplots if any
        # Ensure axes_flat is always a list/array even if single subplot
        if not isinstance(axes_flat, (list, np.ndarray)):
            axes_flat = [axes_flat]

        if n_pcs_to_plot < len(axes_flat):
            for idx in range(n_pcs_to_plot, len(axes_flat)):
                axes_flat[idx].axis('off')
                axes_flat[idx].set_visible(False) # Explicitly hide
        
        # No need for tight_layout when using constrained_layout
    
    if progress_callback:
        progress_callback(90)

    # Create data table with dataset labels
    pc_columns = [f"PC{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, columns=pc_columns)
    scores_df["Dataset"] = labels

    # === ENHANCED SUMMARY TEXT ===
    n_datasets = len(unique_labels)
    total_spectra = X.shape[0]
    total_variance = np.sum(pca.explained_variance_ratio_[: min(3, n_components)]) * 100

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
        cumvar_pct = np.sum(pca.explained_variance_ratio_[: i + 1]) * 100

        # Visual bar for variance
        bar_length = int(var_pct / 2)  # Scale to 50 chars max
        bar = "█" * bar_length + "░" * (50 - bar_length)

        summary += (
            f"  PC{i+1:2d}:  {var_pct:5.2f}% │{bar}│ Cumulative: {cumvar_pct:5.2f}%\n"
        )

    if n_components > 10:
        remaining_var = np.sum(pca.explained_variance_ratio_[10:]) * 100
        summary += (
            f"  ...   {remaining_var:5.2f}% (remaining {n_components-10} components)\n"
        )

    summary += f"\n  First 3 PCs:       {total_variance:.2f}% of total variance\n"
    cumvar_all = np.sum(pca.explained_variance_ratio_) * 100
    summary += (
        f"  All {n_components} PCs:        {cumvar_all:.2f}% of total variance\n\n"
    )

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
        max_loadings = params.get("max_loadings_components", 2)
        max_loadings = min(max_loadings, n_components)
        summary += f"  Loading plots generated for first {max_loadings} component(s)\n"

    if enable_pca_lda:
        summary += "\n🧠 PCA→LDA (optional overlay)\n"
        summary += f"{'─' * 55}\n"
        if pca_lda_info and pca_lda_info.get("enabled"):
            folds_used = pca_lda_info.get("cv_folds_used")
            acc = pca_lda_info.get("cv_accuracy")
            summary += f"  Enabled:          Yes\n"
            if folds_used:
                summary += f"  CV folds used:    {folds_used}\n"
            if acc is not None:
                summary += f"  CV accuracy:      {acc*100:.1f}%\n"
        else:
            summary += "  Enabled:          Yes (but overlay failed)\n"
            if pca_lda_info and pca_lda_info.get("error"):
                summary += f"  Error:            {pca_lda_info['error']}\n"

    summary += f"\n{'═' * 55}\n"

    _log_debug(f"Enhanced summary generated ({len(summary)} characters)")
    # Statistical summary for multi-dataset comparison
    detailed_summary = f"Scaling: {scaling_type}\nTotal spectra: {X.shape[0]}\n"
    detailed_summary += f"Datasets: {n_datasets} groups\n"

    if len(unique_labels) == 2:
        # Add separation metrics for binary comparison
        mask1 = np.array([l == unique_labels[0] for l in labels])
        mask2 = np.array([l == unique_labels[1] for l in labels])

        # Calculate separation in PC1
        pc1_separation = abs(np.mean(scores[mask1, 0]) - np.mean(scores[mask2, 0]))
        pc1_pooled_std = np.sqrt(
            (np.std(scores[mask1, 0]) ** 2 + np.std(scores[mask2, 0]) ** 2) / 2
        )
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
    _log_debug("PCA return values:")
    _log_debug(f"  primary_figure (scores): {fig1 is not None}")
    _log_debug(f"  loadings_figure: {fig_loadings is not None}")
    _log_debug(f"  distributions_figure: {fig_distributions is not None}")

    if fig_loadings is None:
        _log_debug(
            f"WARNING: loadings_figure is None! show_loadings={show_loadings}"
        )

    _log_debug("PCA return values:")
    _log_debug(f"  primary_figure (scores): {fig1 is not None}")
    _log_debug(f"  scree_figure: {fig_scree is not None}")
    _log_debug(f"  loadings_figure: {fig_loadings is not None}")
    _log_debug(f"  biplot_figure: {fig_biplot is not None}")
    _log_debug(f"  cumulative_variance_figure: {fig_cumvar is not None}")
    _log_debug(f"  distributions_figure: {fig_distributions is not None}")

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
            "wavenumbers": wavenumbers,
            "labels": labels,
            "unique_labels": unique_labels,
            "colors": colors,  # Add colors for distributions update
            # Explicit, stable label→color mapping (prevents mismatches in dynamic UI)
            "label_to_color": {
                str(lab): tuple(float(c) for c in np.asarray(colors[idx]).ravel()[:4])
                for idx, lab in enumerate(list(unique_labels))
                if colors is not None and idx < len(colors)
            },
            "pca_lda": pca_lda_info,
        }
    }


def perform_umap_analysis(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
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

    _log_debug(
        f"UMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}"
    )
    _log_debug(f"Group labels map: {group_labels_map}")

    # ✅ FIX: Use interpolation to handle datasets with different wavenumber ranges
    # This resolves "ValueError: all input array dimensions except for concatenation axis must match"
    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data, group_labels_map=group_labels_map, method="linear"
    )

    _log_debug(f"Combined matrix shape after interpolation: {X.shape}")
    _log_debug(f"Unique labels: {sorted(set(labels))}")

    if progress_callback:
        progress_callback(30)

    # Perform UMAP
    # P1-2: Use configurable random seed for reproducibility
    random_seed = params.get("random_seed", 42)
    _log_debug(
        f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}, random_seed={random_seed}"
    )
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_seed,
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

    _log_debug(f"Plotting {num_groups} groups with high-contrast colors")

    for i, dataset_label in enumerate(unique_labels):
        mask = np.array([l == dataset_label for l in labels])
        num_points = np.sum(mask)
        _log_debug(f"Group '{dataset_label}': {num_points} spectra")

        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            color=colors[i],
            label=dataset_label,
            alpha=0.7,
            s=100,
            edgecolors="white",
            linewidth=1.5,
        )

    ax.set_xlabel("UMAP 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=12, fontweight="bold")
    ax.set_title("UMAP Projection", fontsize=14, fontweight="bold")
    ax.legend(
        loc="best",
        framealpha=0.95,
        fontsize=10,
        edgecolor="#cccccc",
        fancybox=True,
        shadow=True,
    )
    ax.grid(True, alpha=0.3)

    # Create data table
    embedding_df = pd.DataFrame(
        embedding, columns=[f"UMAP{i+1}" for i in range(n_components)]
    )
    embedding_df["Dataset"] = labels

    summary = f"UMAP completed with {n_components} components.\n"
    summary += (
        f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}"
    )

    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": embedding_df,
        "summary_text": summary,
        "detailed_summary": f"Total spectra: {X.shape[0]}",
        "raw_results": {"embedding": embedding, "reducer": reducer},
        "loadings_figure": None,  # UMAP does not produce loadings
    }


def perform_tsne_analysis(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
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

    _log_debug(
        f"t-SNE parameters: perplexity={perplexity}, learning_rate={learning_rate}"
    )
    _log_debug(f"t-SNE n_iter={n_iter} (will use as max_iter for sklearn)")

    # ✅ FIX: Use interpolation to handle datasets with different wavenumber ranges
    # This resolves "ValueError: all input array dimensions except for concatenation axis must match"
    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data, group_labels_map=group_labels_map, method="linear"
    )

    n_samples = X.shape[0]
    _log_debug(f"Combined matrix shape after interpolation: {X.shape}")

    # CRITICAL FIX: Perplexity must be less than n_samples
    if perplexity >= n_samples:
        new_perplexity = max(1, n_samples - 1)
        _log_debug(
            f"Adjusting perplexity from {perplexity} to {new_perplexity} (n_samples={n_samples})"
        )
        perplexity = new_perplexity

    if progress_callback:
        progress_callback(30)

    # Perform t-SNE (sklearn uses max_iter, not n_iter)
    # P1-2: Use configurable random seed for reproducibility
    random_seed = params.get("random_seed", 42)
    _log_debug(f"Creating TSNE with max_iter={n_iter}, random_seed={random_seed}")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,  # CRITICAL FIX: sklearn uses max_iter not n_iter
        random_state=random_seed,
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
    _log_debug(f"t-SNE using high-contrast colors for {num_groups} groups: {colors}")

    for i, dataset_label in enumerate(unique_labels):
        mask = np.array([l == dataset_label for l in labels])
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            color=colors[i],
            label=dataset_label,
            alpha=0.7,
            s=100,
            edgecolors="white",
            linewidth=1.0,
        )

    ax.set_xlabel("t-SNE 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("t-SNE 2", fontsize=12, fontweight="bold")
    ax.set_title("t-SNE Projection", fontsize=14, fontweight="bold")
    ax.legend(
        loc="best",
        framealpha=0.95,
        fontsize=10,
        edgecolor="#cccccc",
        fancybox=True,
        shadow=True,
    )
    ax.grid(True, alpha=0.3)

    # Create data table
    embedding_df = pd.DataFrame(embedding, columns=["tSNE1", "tSNE2"])
    embedding_df["Dataset"] = labels

    summary = f"t-SNE completed with 2 components.\n"
    summary += f"Parameters: perplexity={perplexity}, learning_rate={learning_rate}, n_iter={n_iter}"

    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": embedding_df,
        "summary_text": summary,
        "detailed_summary": f"Total spectra: {X.shape[0]}",
        "raw_results": {"embedding": embedding},
        "loadings_figure": None,  # t-SNE does not produce loadings
    }


def perform_hierarchical_clustering(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
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

    show_labels = bool(params.get("show_labels", False))

    # Build per-spectrum labels (dataset_name + column) for dendrogram display.
    # The interpolation helper returns group labels (often dataset names repeated),
    # which makes dendrogram tick labels unreadable/unhelpful.
    def _truncate_label(s: str, max_len: int = 42) -> str:
        s = str(s)
        if len(s) <= max_len:
            return s
        return s[: max_len - 1] + "…"

    sample_labels: List[str] = []
    if show_labels:
        for dataset_name, df in dataset_data.items():
            try:
                cols = list(getattr(df, "columns", []))
            except Exception:
                cols = []

            if cols:
                sample_labels.extend([
                    _truncate_label(f"{dataset_name}:{col}") for col in cols
                ])
            else:
                # Fallback: label by index if columns are missing
                try:
                    n_spectra = int(df.shape[1])
                except Exception:
                    n_spectra = 0
                sample_labels.extend([
                    _truncate_label(f"{dataset_name}:{i}") for i in range(n_spectra)
                ])

    # ✅ FIX: Use interpolation to handle datasets with different wavenumber ranges
    # This resolves "ValueError: all input array dimensions except for concatenation axis must match"
    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data,
        group_labels_map=None,  # Hierarchical clustering uses dataset names as labels
        method="linear",
    )

    _log_debug(
        "Hierarchical clustering: Combined matrix shape after interpolation: %s",
        X.shape,
    )

    if progress_callback:
        progress_callback(40)

    # Perform hierarchical clustering
    if linkage_method == "ward":
        Z = linkage(X, method="ward")
    else:
        distances = pdist(X, metric=distance_metric)
        Z = linkage(distances, method=linkage_method)

    if progress_callback:
        progress_callback(70)

    # Create dendrogram (+ optional spectral heatmap) with robust readability.
    n_samples = int(X.shape[0]) if hasattr(X, "shape") else 0

    # When N is large, a full dendrogram becomes visually meaningless (cramped).
    # Truncate aggressively to keep the plot interpretable.
    max_leaf_labels = int(params.get("max_leaf_labels", 60))
    max_full_dendrogram_leaves = int(params.get("max_full_dendrogram_leaves", 160))
    max_heatmap_samples = int(params.get("max_heatmap_samples", 140))

    should_truncate = bool(n_samples > max_full_dendrogram_leaves)

    # Heatmap is useful for interpretation, but only when the row count is reasonable.
    show_heatmap = bool((not should_truncate) and n_samples > 1 and n_samples <= max_heatmap_samples)

    # For dendrogram+heatmap alignment, prefer top orientation.
    orientation = "top"

    # Dynamic sizing tuned for GUI embedding.
    # - W grows gently with sample count
    # - H reserves room for rotated labels + heatmap
    fig_w = float(max(12, min(0.18 * (n_samples if not should_truncate else max_leaf_labels), 24)))
    fig_h = float(9.5 if show_heatmap else 8.0)

    if show_labels and (not should_truncate):
        fig_h = max(fig_h, 10.5)

    if show_heatmap:
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(
            2,
            1,
            height_ratios=[3.2, 1.3],
            hspace=0.06,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_hm = fig.add_subplot(gs[1, 0])
    else:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax_hm = None

    # Determine dendrogram color threshold
    max_d = float(np.max(Z[:, 2])) if Z.size else 0.0
    color_threshold = 0.7 * max_d

    # If user requested a specific number of clusters, choose the merge distance
    # that yields (approximately) that many clusters.
    try:
        if n_clusters is not None:
            k = int(n_clusters)
            if k >= 2 and Z.shape[0] >= (k - 1):
                # In linkage matrix, the (k-1)th from the end merge distance is a
                # common heuristic threshold for k clusters.
                color_threshold = float(Z[-(k - 1), 2])
    except Exception:
        # Fall back to 70% max distance
        color_threshold = 0.7 * max_d

    # Plot dendrogram with improved visualization
    dend_kwargs: Dict[str, Any] = {
        "ax": ax,
        "orientation": orientation,
        "leaf_font_size": 8,
        "color_threshold": color_threshold,
        "above_threshold_color": "#bcbcbc",  # Light gray for upper links
    }

    if should_truncate:
        dend_kwargs.update(
            {
                "truncate_mode": "lastp",
                "p": int(max(10, min(max_leaf_labels, max_full_dendrogram_leaves))),
                "show_leaf_counts": True,
                "no_labels": True,
            }
        )
    else:
        # Only show meaningful labels when explicitly requested AND manageable.
        if show_labels and sample_labels and len(sample_labels) == n_samples and n_samples <= max_leaf_labels:
            dend_kwargs["labels"] = sample_labels
        else:
            dend_kwargs["no_labels"] = True

    dend = dendrogram(Z, **dend_kwargs)

    # Threshold line (top orientation)
    ax.axhline(
        y=color_threshold,
        c="r",
        lw=1.0,
        linestyle="--",
        alpha=0.55,
        label="Color Threshold",
        zorder=10,
    )

    ax.set_xlabel("Sample", fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)

    title = "Hierarchical Clustering Dendrogram"
    if should_truncate:
        title += f" (showing last {max_leaf_labels} leaves)"
    ax.set_title(title, fontsize=14, fontweight="bold")

    if show_labels and (not should_truncate) and n_samples <= max_leaf_labels:
        plt.setp(ax.get_xticklabels(), rotation=90)
    
    # Compute cluster assignments and add clear legends.
    # We provide two independent visual guides:
    # 1) Dataset/Group legend + colored leaf strip (stable mapping, independent of SciPy branch colors)
    # 2) Cluster legend (colors reflect cluster assignment at the chosen threshold)
    # NOTE: Coloring leaf labels provides a stable legend mapping that doesn't depend
    # on SciPy's internal dendrogram branch coloring.
    cluster_ids = None
    try:
        if n_clusters is not None:
            cluster_ids = fcluster(Z, t=int(n_clusters), criterion="maxclust")
        else:
            cluster_ids = fcluster(Z, t=color_threshold, criterion="distance")
    except Exception:
        cluster_ids = None

    # --- Dataset/Group color strip + legend (recommended for interpretability) ---
    dataset_legend = None
    try:
        leaves = dend.get("leaves", []) or []
        if (not should_truncate) and leaves and labels and len(labels) == n_samples:
            ordered_groups = [str(labels[i]) for i in leaves]
            unique_groups = list(dict.fromkeys(ordered_groups))  # preserve first-seen order

            # Keep the legend readable; still show the color strip even if many groups.
            if len(unique_groups) <= 6:
                palette = get_high_contrast_colors(len(unique_groups))
            else:
                # tab20 provides more distinct colors when there are many datasets
                pal = plt.cm.tab20(np.linspace(0, 1, len(unique_groups)))
                palette = [
                    f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}" for c in pal
                ]

            group_to_color = {g: palette[i] for i, g in enumerate(unique_groups)}

            # SciPy dendrogram x positions are 5, 15, 25, ... (step 10)
            xs = np.arange(len(leaves)) * 10 + 5
            y0 = -0.02 * max_d if max_d > 0 else -0.02
            ax.scatter(
                xs,
                np.full_like(xs, y0, dtype=float),
                c=[group_to_color[g] for g in ordered_groups],
                s=18,
                marker="s",
                linewidths=0,
                zorder=20,
            )
            try:
                lo, hi = ax.get_ylim()
                ax.set_ylim(bottom=min(lo, y0 * 4.0), top=hi)
            except Exception:
                pass

            # Legend (cap at 12 entries to avoid cramping the GUI)
            from matplotlib.patches import Patch

            legend_groups = unique_groups[:12]
            ds_handles = [
                Patch(facecolor=group_to_color[g], edgecolor="#333333", label=str(g))
                for g in legend_groups
            ]
            if len(unique_groups) > len(legend_groups):
                ds_handles.append(Patch(facecolor="#ffffff", edgecolor="#ffffff", label=f"… +{len(unique_groups) - len(legend_groups)} more"))

            dataset_legend = ax.legend(
                handles=ds_handles,
                title="Dataset/Group",
                loc="upper left",
                fontsize=8,
                title_fontsize=9,
                framealpha=0.92,
            )
            ax.add_artist(dataset_legend)
    except Exception:
        dataset_legend = None

    # --- Cluster legend (only when clusters are computed and labels are visible) ---
    if cluster_ids is not None and show_labels and (not should_truncate):
        # Map clusters (1..k) to a high-contrast palette
        unique_clusters = sorted(set(int(c) for c in cluster_ids))
        colors = get_high_contrast_colors(len(unique_clusters))
        cluster_to_color = {c: colors[i] for i, c in enumerate(unique_clusters)}

        # dend['leaves'] are indices into the original observation order
        leaves = dend.get("leaves", [])
        ordered_cluster_ids = [int(cluster_ids[i]) for i in leaves]

        # Color the tick labels by their assigned cluster
        # (Only works when labels are actually shown.)
        try:
            tick_labels = ax.get_xmajorticklabels()
            for tick_label, cid in zip(tick_labels, ordered_cluster_ids):
                tick_label.set_color(cluster_to_color.get(cid, "#000000"))
        except Exception:
            pass

        # Cluster legend
        from matplotlib.patches import Patch

        legend_handles = [
            Patch(facecolor=cluster_to_color[c], edgecolor="#333333", label=f"Cluster {c}")
            for c in unique_clusters
        ]
        ax.legend(
            handles=legend_handles + [ax.lines[-1]],
            loc="upper right",
            fontsize=9,
            framealpha=0.92,
        )

    # --- Optional spectral heatmap aligned to dendrogram leaves ---
    leaves = dend.get("leaves", []) or []
    ordered_cluster_ids = None
    if cluster_ids is not None and leaves:
        try:
            ordered_cluster_ids = [int(cluster_ids[i]) for i in leaves]
        except Exception:
            ordered_cluster_ids = None

    if ax_hm is not None and leaves:
        try:
            X_ordered = np.asarray(X, dtype=float)[leaves, :]
            w = np.asarray(wavenumbers, dtype=float).reshape(-1)

            # Row-wise normalization improves interpretability for heatmaps
            # (shows pattern differences, not absolute intensity scaling).
            row_min = np.nanmin(X_ordered, axis=1, keepdims=True)
            row_max = np.nanmax(X_ordered, axis=1, keepdims=True)
            denom = row_max - row_min
            denom[~np.isfinite(denom)] = 1.0
            denom[denom < 1e-12] = 1.0
            X_norm = (X_ordered - row_min) / denom

            im = ax_hm.imshow(
                X_norm,
                aspect="auto",
                interpolation="nearest",
                cmap=params.get("heatmap_cmap", "coolwarm"),
            )

            # Set meaningful wavenumber ticks (imshow uses pixel indices by default).
            def _set_wavenumber_ticks(_ax, _w, max_ticks: int = 8):
                try:
                    _w = np.asarray(_w, dtype=float).reshape(-1)
                    n_cols = int(_w.shape[0])
                    if n_cols <= 0:
                        return
                    tick_idx = np.linspace(0, n_cols - 1, min(max_ticks, n_cols), dtype=int)
                    tick_lbl = [f"{_w[i]:.0f}" for i in tick_idx]
                    _ax.set_xticks(tick_idx)
                    _ax.set_xticklabels(tick_lbl)
                except Exception:
                    pass

            _set_wavenumber_ticks(ax_hm, w, max_ticks=8)
            ax_hm.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11, fontweight="bold")
            ax_hm.set_ylabel("", fontsize=10)
            ax_hm.set_yticks([])
            ax_hm.grid(False)

            # Draw cluster boundaries for quick visual segmentation.
            if ordered_cluster_ids is not None and len(ordered_cluster_ids) == X_norm.shape[0]:
                try:
                    breaks = []
                    for i in range(1, len(ordered_cluster_ids)):
                        if ordered_cluster_ids[i] != ordered_cluster_ids[i - 1]:
                            breaks.append(i)
                    for b in breaks:
                        ax_hm.axhline(b - 0.5, color="#111111", linewidth=0.6, alpha=0.35)
                except Exception:
                    pass

            # Minimal colorbar (kept small to avoid cramping GUI).
            try:
                cbar = fig.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.01)
                cbar.set_label("Normalized intensity (0–1)", fontsize=9)
                cbar.ax.tick_params(labelsize=8)
            except Exception:
                pass
        except Exception as e:
            create_logs(__name__, __file__, f"Failed to render dendrogram heatmap: {e}", status="debug")

    # Provide layout hints for the embedded Matplotlib copy widget.
    # (The widget may re-run tight_layout on the copied figure.)
    try:
        # Reserve bottom space when labels are shown, and when a heatmap is present.
        bottom = 0.22 if (show_labels and (not should_truncate) and n_samples <= max_leaf_labels) else 0.12
        if show_heatmap:
            bottom = max(bottom, 0.14)
        fig._tight_layout_rect = [0.06, bottom, 0.98, 0.95]
        fig._tight_layout_pad = 1.2
    except Exception:
        pass

    try:
        fig.tight_layout(pad=getattr(fig, "_tight_layout_pad", 1.2), rect=getattr(fig, "_tight_layout_rect", None))
    except Exception:
        try:
            plt.tight_layout()
        except Exception:
            pass
    
    # Dataset counts (stable, independent of dendrogram leaf order)
    dataset_ranges = []
    current_idx = 0
    for dataset_name, df in dataset_data.items():
        n_spectra = df.shape[1]
        dataset_ranges.append(
            {
                "name": dataset_name,
                "start": current_idx,
                "end": current_idx + n_spectra - 1,
                "count": n_spectra,
            }
        )
        current_idx += n_spectra

    summary = f"Hierarchical clustering completed.\\n"
    summary += f"Linkage: {linkage_method}, Distance metric: {distance_metric}\\n"
    if n_clusters is not None:
        summary += f"Requested clusters: {int(n_clusters)} (color threshold = {color_threshold:.3g})\\n"
    if should_truncate:
        summary += f"Displayed dendrogram leaves: {max_leaf_labels} (truncated for readability)\n"
    if show_heatmap:
        summary += "✓ Spectral heatmap shown (rows reordered by dendrogram leaves)\n"
    summary += f"Total spectra: {X.shape[0]}\\n\\n"
    summary += "Dataset Information:\\n"
    for ds_info in dataset_ranges:
        summary += f"  {ds_info['name']}: samples {ds_info['start']}-{ds_info['end']} (n={ds_info['count']})\\n"

    # Quantitative cluster quality metrics (best-effort).
    metrics_lines: List[str] = []
    metrics_payload: Dict[str, Any] = {}
    try:
        if cluster_ids is not None:
            cids = np.asarray(cluster_ids, dtype=int)
            uniq = sorted(set(int(v) for v in cids))
            if len(uniq) >= 2 and len(uniq) < n_samples:
                try:
                    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

                    sil_metric = "euclidean" if linkage_method == "ward" else distance_metric
                    sil = float(silhouette_score(X, cids, metric=sil_metric))
                    db = float(davies_bouldin_score(X, cids))
                    ch = float(calinski_harabasz_score(X, cids))
                    metrics_payload.update({"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch})
                    metrics_lines.append(f"Silhouette: {sil:.3f} (higher=better)")
                    metrics_lines.append(f"Davies–Bouldin: {db:.3f} (lower=better)")
                    metrics_lines.append(f"Calinski–Harabasz: {ch:.1f} (higher=better)")
                except Exception:
                    pass

        try:
            from scipy.cluster.hierarchy import cophenet

            coph_corr, _ = cophenet(Z, pdist(X))
            if np.isfinite(coph_corr):
                metrics_payload["cophenetic_correlation"] = float(coph_corr)
                metrics_lines.append(f"Cophenetic correlation: {float(coph_corr):.3f} (closer to 1=better) ")
        except Exception:
            pass
    except Exception:
        pass

    detailed_summary = f"Linkage matrix shape: {Z.shape}\n"
    if metrics_lines:
        detailed_summary += "\nCluster Quality Metrics:\n" + "\n".join(f"- {s}" for s in metrics_lines) + "\n"

    # Build a secondary plot: mean spectra per cluster (interpretable validation)
    fig_means = None
    try:
        if cluster_ids is not None:
            cids = np.asarray(cluster_ids, dtype=int)
            uniq = sorted(set(int(v) for v in cids))
            if len(uniq) >= 2 and len(uniq) <= 12:
                fig_means, axm = plt.subplots(figsize=(12, 5))
                palette = get_high_contrast_colors(len(uniq))
                for i, cid in enumerate(uniq):
                    mask = cids == cid
                    if not np.any(mask):
                        continue
                    mean_spec = np.nanmean(np.asarray(X, dtype=float)[mask, :], axis=0)
                    axm.plot(
                        np.asarray(wavenumbers, dtype=float),
                        mean_spec,
                        linewidth=1.8,
                        color=palette[i],
                        label=f"Cluster {cid} (n={int(np.sum(mask))})",
                        alpha=0.9,
                    )
                axm.set_title("Cluster Mean Spectra", fontsize=13, fontweight="bold")
                axm.set_xlabel("Wavenumber (cm⁻¹)")
                axm.set_ylabel("Intensity")
                axm.grid(True, alpha=0.25)
                axm.legend(loc="best", fontsize=9)
                try:
                    fig_means._tight_layout_rect = [0.08, 0.12, 0.98, 0.95]
                    fig_means._tight_layout_pad = 1.2
                except Exception:
                    pass
                try:
                    fig_means.tight_layout(pad=1.2)
                except Exception:
                    pass
    except Exception:
        fig_means = None
    
    return {
        "primary_figure": fig,
        "secondary_figure": fig_means,
        "data_table": None,
        "summary_text": summary,
        "detailed_summary": detailed_summary,
        "raw_results": {
            "linkage_matrix": Z,
            "labels": labels,
            "wavenumbers": np.asarray(wavenumbers, dtype=float),
            "cluster_ids": (np.asarray(cluster_ids, dtype=int) if cluster_ids is not None else None),
            "leaves": leaves,
            "cluster_metrics": metrics_payload,
        },
        "loadings_figure": None,  # Clustering does not produce loadings
    }


def perform_kmeans_clustering(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
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
    show_elbow = params.get("show_elbow", True)
    elbow_max_k = params.get("elbow_max_k", 10)
    show_silhouette = bool(params.get("show_silhouette", False))
    silhouette_k_min = int(params.get("silhouette_k_min", 2))
    silhouette_k_max = int(params.get("silhouette_k_max", 10))
    silhouette_use_pca = bool(params.get("silhouette_use_pca", True))
    silhouette_pca_components = int(params.get("silhouette_pca_components", 10))

    _log_debug("K-Means parameters received:")
    _log_debug(f"  n_clusters = {n_clusters} (type: {type(n_clusters).__name__})")
    _log_debug(f"  n_init = {n_init} (type: {type(n_init).__name__})")
    _log_debug(f"  max_iter = {max_iter} (type: {type(max_iter).__name__})")
    _log_debug(f"  show_pca = {show_pca}")
    _log_debug(f"  show_elbow = {show_elbow}")
    _log_debug(f"  elbow_max_k = {elbow_max_k}")
    _log_debug(f"  show_silhouette = {show_silhouette}")
    _log_debug(f"  silhouette_k_min = {silhouette_k_min}")
    _log_debug(f"  silhouette_k_max = {silhouette_k_max}")
    _log_debug(f"  silhouette_use_pca = {silhouette_use_pca}")
    _log_debug(f"  silhouette_pca_components = {silhouette_pca_components}")

    # ✅ FIX: Use interpolation to handle datasets with different wavenumber ranges
    # This resolves "ValueError: all input array dimensions except for concatenation axis must match"
    wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data,
        group_labels_map=None,  # K-Means uses dataset names as labels
        method="linear",
    )

    _log_debug(f"K-Means: Combined matrix shape after interpolation: {X.shape}")

    if progress_callback:
        progress_callback(30)

    # Perform K-means clustering
    # P1-2: Use configurable random seed for reproducibility
    random_seed = params.get("random_seed", 42)
    _log_debug(
        f"Creating KMeans with n_clusters={n_clusters}, n_init={n_init}, max_iter={max_iter}, random_seed={random_seed}"
    )
    kmeans = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_seed,
    )
    _log_debug("Fitting KMeans model...")
    cluster_labels = kmeans.fit_predict(X)
    _log_debug(f"KMeans completed. Inertia: {kmeans.inertia_:.2f}")

    # Optional validation plots (secondary figure): Elbow + Silhouette
    secondary_fig = None
    inertia_by_k = None
    silhouette_by_k = None

    if show_elbow or show_silhouette:
        rows = int(bool(show_elbow)) + int(bool(show_silhouette))
        secondary_fig, axes = plt.subplots(rows, 1, figsize=(9, 4.8 * rows), sharex=False)
        if rows == 1:
            axes = [axes]

        ax_idx = 0

        if show_elbow:
            # Elbow plot: inertia for k=1..K
            n_samples = int(X.shape[0])
            try:
                max_k_allowed = max(1, min(int(elbow_max_k), max(1, n_samples - 1)))
            except Exception:
                max_k_allowed = max(1, min(10, max(1, n_samples - 1)))

            # Always include at least k=1 and k=2 when possible
            if max_k_allowed < 2 and n_samples >= 2:
                max_k_allowed = 2

            ks = list(range(1, max_k_allowed + 1))
            inertia_vals: list[float] = []
            for k in ks:
                km = KMeans(
                    n_clusters=k,
                    max_iter=max_iter,
                    n_init=n_init,
                    random_state=random_seed,
                )
                km.fit(X)
                inertia_vals.append(float(km.inertia_))

            inertia_by_k = {"k": ks, "inertia": inertia_vals}
            ax_elbow = axes[ax_idx]
            ax_elbow.plot(ks, inertia_vals, marker="o", linewidth=1.8)
            ax_elbow.set_xlabel("k (number of clusters)", fontsize=11, fontweight="bold")
            ax_elbow.set_ylabel("Inertia (within-cluster SSE)", fontsize=11, fontweight="bold")
            ax_elbow.set_title("K-Means Elbow Plot", fontsize=13, fontweight="bold")
            ax_elbow.grid(True, alpha=0.3)

            # Highlight the chosen k if it is within the computed range
            if n_clusters in ks:
                chosen_idx = ks.index(n_clusters)
                ax_elbow.scatter(
                    [n_clusters],
                    [inertia_vals[chosen_idx]],
                    s=80,
                    c="#d62728",
                    edgecolors="black",
                    zorder=5,
                    label=f"Chosen k={n_clusters}",
                )
                ax_elbow.axvline(n_clusters, color="#d62728", linestyle="--", alpha=0.6)
                ax_elbow.legend(loc="best", framealpha=0.9)

            ax_idx += 1

        if show_silhouette:
            # Silhouette score curve over k range (computed on PCA-reduced space by default)
            from sklearn.metrics import silhouette_score

            n_samples = int(X.shape[0])
            k_min = max(2, int(silhouette_k_min))
            k_max = max(k_min, int(silhouette_k_max))
            k_max = min(k_max, max(2, n_samples - 1))

            # Feature space for silhouette (PCA speeds up + reduces noise)
            Z = X
            if silhouette_use_pca:
                try:
                    n_feat = int(X.shape[1])
                    n_comp = max(2, min(int(silhouette_pca_components), n_feat, n_samples - 1))
                    Z = PCA(n_components=n_comp).fit_transform(X)
                except Exception:
                    Z = X

            ks = list(range(k_min, k_max + 1))
            sil_vals: list[float] = []
            for k in ks:
                # Silhouette requires: 2 <= k < n_samples
                if k < 2 or k >= n_samples:
                    sil_vals.append(float("nan"))
                    continue

                if k == int(n_clusters):
                    labels_k = cluster_labels
                else:
                    km = KMeans(
                        n_clusters=k,
                        max_iter=max_iter,
                        n_init=n_init,
                        random_state=random_seed,
                    )
                    labels_k = km.fit_predict(X)

                try:
                    sil = float(silhouette_score(Z, labels_k))
                except Exception:
                    sil = float("nan")
                sil_vals.append(sil)

            silhouette_by_k = {"k": ks, "silhouette": sil_vals}
            ax_sil = axes[ax_idx]
            ax_sil.plot(ks, sil_vals, marker="o", linewidth=1.8, color="#2ca02c")
            ax_sil.set_xlabel("k (number of clusters)", fontsize=11, fontweight="bold")
            ax_sil.set_ylabel("Silhouette score", fontsize=11, fontweight="bold")
            ax_sil.set_title("Silhouette Score vs k", fontsize=13, fontweight="bold")
            ax_sil.grid(True, alpha=0.3)

            if int(n_clusters) in ks:
                chosen_idx = ks.index(int(n_clusters))
                ax_sil.scatter(
                    [int(n_clusters)],
                    [sil_vals[chosen_idx]],
                    s=80,
                    c="#d62728",
                    edgecolors="black",
                    zorder=5,
                    label=f"Chosen k={n_clusters}",
                )
                ax_sil.axvline(int(n_clusters), color="#d62728", linestyle="--", alpha=0.6)
                ax_sil.legend(loc="best", framealpha=0.9)

        try:
            secondary_fig.tight_layout()
        except Exception:
            pass

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
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=[colors[i]],
                label=f"Cluster {i+1}",
                alpha=0.7,
                s=50,
            )

        # Plot centroids
        ax.scatter(
            centers_pca[:, 0],
            centers_pca[:, 1],
            c="red",
            marker="X",
            s=200,
            edgecolors="black",
            linewidths=2,
            label="Centroids",
        )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
        ax.set_title(
            "K-means Clustering (PCA Projection)", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    else:
        # Just show cluster assignments
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(cluster_labels)), cluster_labels)
        ax.set_xlabel("Sample Index", fontsize=12)
        ax.set_ylabel("Cluster ID", fontsize=12)
        ax.set_title("K-means Cluster Assignments", fontsize=14, fontweight="bold")

    if progress_callback:
        progress_callback(90)

    # Create data table
    results_df = pd.DataFrame({"Dataset": labels, "Cluster": cluster_labels})

    # Calculate cluster statistics
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()

    summary = f"K-means clustering completed with {n_clusters} clusters.\n"
    for i in range(n_clusters):
        count = cluster_counts.get(i, 0)
        pct = count / len(cluster_labels) * 100
        summary += f"Cluster {i+1}: {count} spectra ({pct:.1f}%)\n"

    return {
        "primary_figure": fig,
        "secondary_figure": secondary_fig,
        "data_table": results_df,
        "summary_text": summary,
        "detailed_summary": f"Inertia: {kmeans.inertia_:.2f}\nIterations: {kmeans.n_iter_}",
        "raw_results": {
            "kmeans_model": kmeans,
            "cluster_labels": cluster_labels,
            "cluster_centers": kmeans.cluster_centers_,
            "elbow": inertia_by_k,
            "silhouette": silhouette_by_k,
        },
        "loadings_figure": None,  # Clustering does not produce loadings
    }


def create_spectrum_preview_figure(
    dataset_data: Dict[str, pd.DataFrame], show_all: bool = False
) -> Figure:
    """
    Create a preview figure showing spectra stacked behind each other (overlaid).
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        show_all: If True, show all individual spectra; if False, show only mean spectra

    Returns:
        Matplotlib Figure object
    """
    # ✅ FIX: Correct unpacking of plt.subplots (returns tuple of fig, ax)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Dynamic color palette using matplotlib's tab20 colormap (handles any number of datasets)
    n_datasets = len(dataset_data)
    cmap = plt.get_cmap('tab20', max(n_datasets, 10))
    colors = [cmap(i) for i in range(n_datasets)]
    
    max_intensity_overall = 0

    for idx, (dataset_name, df) in enumerate(dataset_data.items()):
        wavenumbers = df.index.values
        n_spectra = df.shape[1]
        color = colors[idx % len(colors)]

        if show_all:
            # Show ALL individual spectra with low alpha (overlaid, not offset)
            for col_idx, col in enumerate(df.columns):
                spectrum = df[col].values
                alpha = 0.3 if n_spectra > 10 else 0.5
                ax.plot(
                    wavenumbers,
                    spectrum,
                    color=color,
                    linewidth=0.8,
                    alpha=alpha,
                    label=f"{dataset_name}" if col_idx == 0 else None,
                    zorder=5 + idx,
                )
                max_intensity_overall = max(max_intensity_overall, spectrum.max())
        else:
            # Show MEAN only (default behavior) - overlaid without offset
            mean_spectrum = df.mean(axis=1).values
            std_spectrum = df.std(axis=1).values
            
            # Plot MEAN line only (bold, prominent) - NO OFFSET
            ax.plot(
                wavenumbers, mean_spectrum,
                color=color,
                linewidth=2.8,
                label=f"{dataset_name} (mean, n={n_spectra})",
                alpha=0.95,
                zorder=10 + idx,
            )

            # Add VERY subtle ±0.5σ envelope (barely visible)
            ax.fill_between(
                wavenumbers,
                mean_spectrum - std_spectrum * 0.5,
                mean_spectrum + std_spectrum * 0.5,
                color=color,
                alpha=0.08,
                edgecolor="none",
                zorder=5 + idx,
            )
            
            # Track max intensity
            max_intensity_overall = max(max_intensity_overall, (mean_spectrum + std_spectrum).max())
    
    # Styling
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Intensity', fontsize=12, fontweight='bold')
    title = 'All Spectra' if show_all else 'Mean Spectra'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    # Display wavenumber increasing left → right.

    # Adjust y-limits
    ax.set_ylim(-max_intensity_overall * 0.05, max_intensity_overall * 1.05)
    
    _safe_tight_layout(fig, pad=1.2)
    return fig
