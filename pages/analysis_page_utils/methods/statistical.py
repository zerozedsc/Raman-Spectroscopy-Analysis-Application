"""
Statistical Analysis Methods

This module implements statistical analysis methods for Raman spectra including
spectral comparison, peak analysis, correlation analysis, and ANOVA.
"""

import traceback

from configs.configs import create_logs

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import normalize


def perform_spectral_comparison(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Perform statistical comparison of spectral datasets.

    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - confidence_level: Confidence level (default 0.95)
            - fdr_correction: Apply FDR correction (default True)
            - show_ci: Show confidence intervals (default True)
            - highlight_significant: Highlight significant regions (default True)
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with comparison plots and statistics
    """
    if progress_callback:
        progress_callback(10)

    # Get parameters
    confidence_level = params.get("confidence_level", 0.95)
    fdr_correction = params.get("fdr_correction", True)
    show_ci = params.get("show_ci", True)
    highlight_significant = params.get("highlight_significant", True)

    # Get datasets
    dataset_names = list(dataset_data.keys())
    if len(dataset_names) < 2:
        raise ValueError("At least 2 datasets required for comparison")
    
    # For multi-dataset comparison, plot all means
    # Use colormap for multiple datasets
    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_names)))
    
    # Extract first two datasets for statistical testing (t-test is pairwise)
    df1 = dataset_data[dataset_names[0]]
    df2 = dataset_data[dataset_names[1]]

    wavenumbers = df1.index.values

    if progress_callback:
        progress_callback(30)

    # Calculate means and standard errors
    mean1 = df1.mean(axis=1).values
    mean2 = df2.mean(axis=1).values
    sem1 = df1.sem(axis=1).values
    sem2 = df2.sem(axis=1).values

    # Calculate confidence intervals
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    ci1 = z_score * sem1
    ci2 = z_score * sem2

    if progress_callback:
        progress_callback(50)

    # Perform t-tests at each wavenumber
    p_values = []
    for i in range(len(wavenumbers)):
        spec1 = df1.iloc[i, :].values
        spec2 = df2.iloc[i, :].values
        _, p = stats.ttest_ind(spec1, spec2)
        p_values.append(p)

    p_values = np.array(p_values)

    # FDR correction if requested
    if fdr_correction:
        from statsmodels.stats.multitest import fdrcorrection

        _, p_corrected = fdrcorrection(p_values, alpha=1 - confidence_level)
        significant_mask = p_corrected < (1 - confidence_level)
    else:
        significant_mask = p_values < (1 - confidence_level)

    if progress_callback:
        progress_callback(70)

    # Create primary figure: Mean spectra with CI
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot ALL datasets (not just first 2)
    for idx, dataset_name in enumerate(dataset_names):
        df = dataset_data[dataset_name]
        mean = df.mean(axis=1).values
        sem = df.sem(axis=1).values
        ci = z_score * sem
        
        color = colors[idx]
        ax1.plot(wavenumbers, mean, label=dataset_name, linewidth=2, color=color)
        
        if show_ci:
            ax1.fill_between(wavenumbers, mean - ci, mean + ci,
                            alpha=0.2, color=color)
    
    if highlight_significant:
        # Highlight significant regions
        sig_regions = np.where(significant_mask)[0]
        if len(sig_regions) > 0:
            y_min, y_max = ax1.get_ylim()
            for idx in sig_regions:
                ax1.axvspan(
                    wavenumbers[idx] - 2,
                    wavenumbers[idx] + 2,
                    alpha=0.1,
                    color="yellow",
                )

    ax1.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax1.set_ylabel("Intensity", fontsize=12)
    ax1.set_title("Spectral Comparison", fontsize=14, fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # Create secondary figure: P-value plot
    fig2, ax2 = plt.subplots(figsize=(12, 4))

    ax2.plot(wavenumbers, -np.log10(p_values), linewidth=1.5, color="black")
    ax2.axhline(
        -np.log10(1 - confidence_level),
        color="red",
        linestyle="--",
        label=f"{confidence_level*100:.0f}% significance",
    )
    ax2.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax2.set_ylabel("-log₁₀(p-value)", fontsize=12)
    ax2.set_title("Statistical Significance", fontsize=14, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    if progress_callback:
        progress_callback(90)

    # Create data table
    results_df = pd.DataFrame(
        {
            "Wavenumber": wavenumbers,
            f"{dataset_names[0]}_mean": mean1,
            f"{dataset_names[0]}_sem": sem1,
            f"{dataset_names[1]}_mean": mean2,
            f"{dataset_names[1]}_sem": sem2,
            "p_value": p_values,
            "significant": significant_mask,
        }
    )

    n_significant = np.sum(significant_mask)
    pct_significant = n_significant / len(wavenumbers) * 100

    summary = (
        f"Spectral comparison between {dataset_names[0]} and {dataset_names[1]}.\n"
    )
    summary += f"Significant regions: {n_significant} ({pct_significant:.1f}%)\n"
    summary += f"Confidence level: {confidence_level*100:.0f}%"
    if fdr_correction:
        summary += " (FDR corrected)"

    return {
        "primary_figure": fig1,
        "secondary_figure": fig2,
        "data_table": results_df,
        "summary_text": summary,
        "detailed_summary": f"Dataset 1 samples: {df1.shape[1]}, Dataset 2 samples: {df2.shape[1]}",
        "raw_results": {
            "p_values": p_values,
            "significant_mask": significant_mask,
            "means": [mean1, mean2],
            "sems": [sem1, sem2]
        },
        "loadings_figure": None  # Spectral comparison does not produce loadings
    }


def perform_peak_analysis(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Perform peak detection and analysis on spectra.

    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - prominence_threshold: Peak prominence threshold (default 0.05)
            - width_min: Minimum peak width (default 5)
            - top_n_peaks: Number of top peaks to analyze (default 20)
            - show_assignments: Show peak assignments (default True)
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with peak analysis plots and table
    """
    import json
    import os

    if progress_callback:
        progress_callback(10)

    # Get parameters
    prominence_threshold = params.get("prominence_threshold", 0.05)
    width_min = params.get("width_min", 5)
    top_n_peaks = params.get("top_n_peaks", 20)
    show_assignments = params.get("show_assignments", True)

    # Load Raman peak assignments from JSON (only if enabled)
    raman_peaks_data = {}
    if show_assignments:
        try:
            peaks_json_path = os.path.join("assets", "data", "raman_peaks.json")
            if os.path.exists(peaks_json_path):
                with open(peaks_json_path, "r", encoding="utf-8") as f:
                    raman_peaks_data = json.load(f)
                create_logs(
                    __name__,
                    __file__,
                    f"Loaded {len(raman_peaks_data)} peak assignments from raman_peaks.json",
                    status="debug",
                )
            else:
                create_logs(
                    __name__,
                    __file__,
                    f"raman_peaks.json not found at {peaks_json_path}",
                    status="warning",
                )
        except Exception as e:
            create_logs(
                __name__,
                __file__,
                f"Error loading raman_peaks.json: {e}\n\n{traceback.format_exc()}",
                status="error",
            )

    # Use first dataset for mean spectrum
    dataset_name = list(dataset_data.keys())[0]
    df = dataset_data[dataset_name]

    wavenumbers = df.index.values
    mean_spectrum = df.mean(axis=1).values

    if progress_callback:
        progress_callback(40)

    # Normalize spectrum for peak detection
    spectrum_normalized = (mean_spectrum - mean_spectrum.min()) / (
        mean_spectrum.max() - mean_spectrum.min()
    )

    # Find peaks
    peaks, properties = find_peaks(
        spectrum_normalized, prominence=prominence_threshold, width=width_min
    )

    if progress_callback:
        progress_callback(70)

    # Get top N peaks by prominence
    prominences = properties["prominences"]
    sorted_indices = np.argsort(prominences)[::-1][:top_n_peaks]
    top_peaks = peaks[sorted_indices]
    top_prominences = prominences[sorted_indices]

    create_logs(
        __name__,
        __file__,
        f"Peak detection: Found {len(peaks)} total peaks, showing top {len(top_peaks)} peaks",
        status="debug",
    )
    create_logs(__name__, __file__, f"top_n_peaks parameter: {top_n_peaks}", status="debug")
    create_logs(__name__, __file__, f"Actual peaks shown: {len(top_peaks)}", status="debug")

    # Helper function to find closest peak assignment
    def find_peak_assignment(wavenumber: float, tolerance: float = 10.0) -> str:
        """
        Find the closest peak assignment from raman_peaks.json.

        Args:
            wavenumber: Detected peak wavenumber
            tolerance: Maximum distance to consider a match (default 10 cm⁻¹)

        Returns:
            Component assignment string or empty string if no match
        """
        if not raman_peaks_data:
            return ""

        closest_match = None
        closest_distance = float("inf")

        for peak_wn_str, peak_info in raman_peaks_data.items():
            try:
                ref_wavenumber = float(peak_wn_str)
                distance = abs(wavenumber - ref_wavenumber)

                if distance < closest_distance and distance <= tolerance:
                    closest_distance = distance
                    closest_match = peak_info.get("assignment", "")
            except (ValueError, TypeError):
                continue

        if closest_match:
            # Truncate long assignments
            if len(closest_match) > 40:
                closest_match = closest_match[:37] + "..."
            return closest_match

        return ""

    # Create primary figure: Spectrum with peaks
    fig1, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(
        wavenumbers, mean_spectrum, linewidth=1.5, color="blue", label="Mean spectrum"
    )
    ax1.plot(
        wavenumbers[top_peaks],
        mean_spectrum[top_peaks],
        "ro",
        markersize=10,
        label=f"Top {len(top_peaks)} peaks",
        zorder=5,
    )

    # Annotate peaks with wavenumber (and component assignments if enabled)
    create_logs(
        __name__,
        __file__,
        f"Adding peak annotations for {len(top_peaks)} peaks (show_assignments={show_assignments})",
        status="debug",
    )

    def _assign_label_levels(
        peak_indices: np.ndarray,
        wn: np.ndarray,
        max_levels: int = 6,
    ) -> Dict[int, Dict[str, Any]]:
        """Assign non-overlapping annotation offsets using discrete vertical 'levels'.

        Strategy (simple + robust):
        - Sort by wavenumber (x)
        - Greedily place each peak into the lowest available level such that
          no other label in that level is too close in x.
        - Use +y levels first; if saturated, use -y levels.

        Returns:
            Mapping peak_idx -> {"level": int, "xytext": (dx, dy)}
        """

        peak_indices_sorted = sorted(list(peak_indices), key=lambda idx: float(wn[idx]))

        wn_min = float(np.nanmin(wn))
        wn_max = float(np.nanmax(wn))
        wn_span = max(wn_max - wn_min, 1.0)
        # Dynamic minimum separation in wavenumbers; tuned for readability.
        min_sep_wn = max(25.0, wn_span / 30.0)

        base_offset = 14
        step_offset = 16

        pos_levels: Dict[int, List[float]] = {lvl: [] for lvl in range(max_levels)}
        neg_levels: Dict[int, List[float]] = {lvl: [] for lvl in range(max_levels)}

        out: Dict[int, Dict[str, Any]] = {}
        for peak_idx in peak_indices_sorted:
            x = float(wn[peak_idx])

            chosen_level = None
            chosen_sign = +1

            # Try positive levels first.
            for lvl in range(max_levels):
                if all(abs(x - prev_x) >= min_sep_wn for prev_x in pos_levels[lvl]):
                    chosen_level = lvl
                    chosen_sign = +1
                    pos_levels[lvl].append(x)
                    break

            # Then try negative levels.
            if chosen_level is None:
                for lvl in range(max_levels):
                    if all(abs(x - prev_x) >= min_sep_wn for prev_x in neg_levels[lvl]):
                        chosen_level = lvl
                        chosen_sign = -1
                        neg_levels[lvl].append(x)
                        break

            # Fallback: stack on the last positive level.
            if chosen_level is None:
                chosen_level = max_levels - 1
                chosen_sign = +1
                pos_levels[chosen_level].append(x)

            dy = chosen_sign * (base_offset + chosen_level * step_offset)
            out[int(peak_idx)] = {
                "level": int(chosen_level) * int(chosen_sign),
                "xytext": (0, int(dy)),
                "min_sep_wn": float(min_sep_wn),
            }

        return out

    label_layout = _assign_label_levels(top_peaks, wavenumbers)

    for i, peak_idx in enumerate(top_peaks):
        wavenumber = wavenumbers[peak_idx]
        intensity = mean_spectrum[peak_idx]

        # Find component assignment (only if enabled)
        assignment = find_peak_assignment(wavenumber) if show_assignments else ""

        # Build annotation text: wavenumber on first line, assignment on second line
        if show_assignments and assignment:
            annotation_text = f"{wavenumber:.0f} cm⁻¹\n{assignment}"
        else:
            annotation_text = f"{wavenumber:.0f} cm⁻¹"

        # Non-overlapping label placement using discrete levels (saved in raw_results)
        xytext = tuple(label_layout.get(int(peak_idx), {}).get("xytext", (0, 18)))

        # Use different colors for peaks with vs without assignments (only when enabled)
        if show_assignments and assignment:
            box_color = "lightyellow"
            edge_color = "orange"
        else:
            box_color = "lightgray"
            edge_color = "gray"

        ax1.annotate(
            annotation_text,
            xy=(wavenumber, intensity),
            xytext=xytext,
            textcoords="offset points",
            ha="center",
            fontsize=7 if assignment else 8,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=box_color,
                alpha=0.8,
                edgecolor=edge_color,
            ),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="arc3,rad=0", color="red", lw=1
            ),
            zorder=10,
        )

        if show_assignments and assignment:
            create_logs(
                __name__,
                __file__,
                f"Peak {i+1}/{len(top_peaks)}: {wavenumber:.0f} cm⁻¹ -> {assignment[:30]}",
                status="debug",
            )
        else:
            create_logs(
                __name__,
                __file__,
                (
                    f"Peak {i+1}/{len(top_peaks)}: {wavenumber:.0f} cm⁻¹ (no assignment found)"
                    if show_assignments
                    else f"Peak {i+1}/{len(top_peaks)}: {wavenumber:.0f} cm⁻¹"
                ),
                status="debug",
            )

    ax1.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax1.set_ylabel("Intensity", fontsize=12)
    ax1.set_title(
        "Peak Analysis with Component Assignments" if show_assignments else "Peak Analysis",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # Create secondary figure: Peak intensity distribution
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    peak_wavenumbers = wavenumbers[top_peaks]
    peak_intensities = mean_spectrum[top_peaks]

    bars = ax2.bar(range(len(top_peaks)), peak_intensities, color="steelblue")
    ax2.set_xticks(range(len(top_peaks)))
    ax2.set_xticklabels(
        [f"{wn:.0f}" for wn in peak_wavenumbers], rotation=45, ha="right"
    )
    ax2.set_xlabel("Peak Position (cm⁻¹)", fontsize=12)
    ax2.set_ylabel("Peak Intensity", fontsize=12)
    ax2.set_title("Peak Intensity Distribution", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    if progress_callback:
        progress_callback(90)

    # Create data table (component assignments optional)
    if show_assignments:
        peak_assignments = [find_peak_assignment(wn) for wn in wavenumbers[top_peaks]]
    else:
        peak_assignments = ["" for _ in wavenumbers[top_peaks]]

    results_df = pd.DataFrame(
        {
            "Peak_Position": wavenumbers[top_peaks],
            "Intensity": mean_spectrum[top_peaks],
            "Component_Assignment": peak_assignments,
            "Prominence": top_prominences,
            "Width": properties["widths"][sorted_indices],
        }
    )
    results_df = results_df.sort_values("Intensity", ascending=False)

    summary = f"Peak analysis completed on {dataset_name}.\n"
    summary += f"Found {len(peaks)} peaks total, showing top {len(top_peaks)}.\n"
    summary += f"Peak detection threshold: prominence = {prominence_threshold:.3f}\n"
    if show_assignments:
        assigned_count = sum(1 for a in peak_assignments if a)
        summary += (
            f"Component assignments: {assigned_count}/{len(top_peaks)} peaks matched\n"
        )

    return {
        "primary_figure": fig1,
        "secondary_figure": fig2,
        "data_table": results_df,
        "summary_text": summary,
        "detailed_summary": f"Mean of {df.shape[1]} spectra analyzed",
        "raw_results": {
            "all_peaks": peaks,
            "top_peaks": top_peaks,
            "properties": properties
            ,
            "peak_label_layout": label_layout,
        },
        "loadings_figure": None  # Peak detection does not produce loadings
    }


def perform_correlation_analysis(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
        Perform correlation analysis.

        Supports two primary Raman-appropriate orientations:
            - mode="spectra": spectrum–spectrum similarity matrix (each spectrum is a variable)
            - mode="wavenumbers": wavenumber–wavenumber co-variation map (each band is a variable)

    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - mode: 'spectra' or 'wavenumbers' (default 'wavenumbers')
            - method: Correlation method ('pearson', 'spearman', 'kendall') (default 'pearson')
            - show_heatmap: Show heatmap figure (default True)
            - max_wavenumbers: Downsample cap for wavenumber–wavenumber maps (default 600)
            - threshold: Absolute correlation threshold for reporting top pairs (default 0.7)
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with correlation matrix and heatmap
    """
    if progress_callback:
        progress_callback(10)

    # Get parameters
    mode = params.get("mode", "wavenumbers")
    method = params.get("method", "pearson")
    show_heatmap = params.get("show_heatmap", True)
    threshold = float(params.get("threshold", 0.7))
    max_wavenumbers = params.get("max_wavenumbers", 600)

    # Build a common X matrix with consistent wavenumber axis
    # X shape: (n_spectra, n_wavenumbers)
    from .exploratory import interpolate_to_common_wavenumbers_with_groups

    wavenumbers, X, spectrum_labels = interpolate_to_common_wavenumbers_with_groups(
        dataset_data,
        group_labels_map=None,
        method="linear",
    )

    if progress_callback:
        progress_callback(40)

    def _downsample_wavenumber_axis(
        wn: np.ndarray, x: np.ndarray, target_points: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Downsample/bucket wavenumbers to keep correlation maps tractable."""
        wn = np.asarray(wn, dtype=float)
        if wn.size <= target_points or target_points is None or target_points <= 1:
            return wn, x

        n = int(wn.size)
        bins = int(target_points)
        bin_size = int(np.ceil(n / bins))

        wn_binned = []
        x_binned = []
        for start in range(0, n, bin_size):
            end = min(n, start + bin_size)
            wn_binned.append(float(np.mean(wn[start:end])))
            x_binned.append(np.mean(x[:, start:end], axis=1))

        wn_out = np.asarray(wn_binned, dtype=float)
        x_out = np.stack(x_binned, axis=1) if x_binned else x[:, :0]
        return wn_out, x_out

    # Calculate correlation matrix
    if str(mode).lower() == "spectra":
        # spectrum–spectrum: variables are spectra
        spectra_df = pd.DataFrame(X.T, columns=spectrum_labels)
        corr_df = spectra_df.corr(method=method)
        axis_labels = spectrum_labels
        axis_type = "spectra"
    else:
        # wavenumber–wavenumber: variables are bands
        try:
            cap = int(max_wavenumbers)
        except Exception:
            cap = 600
        wn_ds, X_ds = _downsample_wavenumber_axis(wavenumbers, X, cap)

        wn_labels = [f"{v:.1f}" for v in wn_ds]
        wn_df = pd.DataFrame(X_ds, columns=wn_labels)
        corr_df = wn_df.corr(method=method)
        axis_labels = wn_ds
        axis_type = "wavenumbers"

    if progress_callback:
        progress_callback(70)

    fig = None
    if show_heatmap:
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(corr_df.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation Coefficient", fontsize=12)

        # Labels (avoid unreadable tick spam)
        n_ticks = int(corr_df.shape[0])
        show_ticks = n_ticks <= 50
        if show_ticks:
            ax.set_xticks(range(n_ticks))
            ax.set_yticks(range(n_ticks))
            if axis_type == "wavenumbers":
                tick_labels = [f"{float(v):.0f}" for v in axis_labels]
            else:
                tick_labels = list(axis_labels)
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
            ax.set_yticklabels(tick_labels, fontsize=7)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("(ticks hidden: too many variables)", fontsize=10)

        if axis_type == "wavenumbers":
            title = f"Wavenumber–Wavenumber Correlation Map ({method.capitalize()})"
        else:
            title = f"Spectrum–Spectrum Correlation Matrix ({method.capitalize()})"
        ax.set_title(title, fontsize=13, fontweight="bold")
        plt.tight_layout()

    if progress_callback:
        progress_callback(90)

    # Statistics + thresholded pairs table
    corr_vals = corr_df.values
    triu = np.triu_indices_from(corr_vals, k=1)
    upper = corr_vals[triu]
    upper = upper[np.isfinite(upper)]

    summary = f"Correlation analysis completed.\n"
    summary += f"Mode: {axis_type} | Method: {method}\n"

    pairs_df = None
    if upper.size:
        mean_corr = float(np.mean(upper))
        std_corr = float(np.std(upper))
        min_corr = float(np.min(upper))
        max_corr = float(np.max(upper))
        summary += f"Mean correlation: {mean_corr:.3f} ± {std_corr:.3f}\n"
        summary += f"Range: [{min_corr:.3f}, {max_corr:.3f}]\n"

        # Report top pairs above threshold (absolute)
        mask = np.abs(upper) >= threshold
        if np.any(mask):
            idx_i = np.asarray(triu[0])[mask]
            idx_j = np.asarray(triu[1])[mask]
            vals = upper[mask]

            # Sort by absolute correlation descending
            order = np.argsort(-np.abs(vals))
            idx_i = idx_i[order]
            idx_j = idx_j[order]
            vals = vals[order]

            top_n = int(min(200, vals.size))
            idx_i = idx_i[:top_n]
            idx_j = idx_j[:top_n]
            vals = vals[:top_n]

            if axis_type == "wavenumbers":
                a = [float(axis_labels[i]) for i in idx_i]
                b = [float(axis_labels[j]) for j in idx_j]
                pairs_df = pd.DataFrame(
                    {"Wavenumber_A": a, "Wavenumber_B": b, "Correlation": vals}
                )
            else:
                a = [str(spectrum_labels[i]) for i in idx_i]
                b = [str(spectrum_labels[j]) for j in idx_j]
                pairs_df = pd.DataFrame({"Spectrum_A": a, "Spectrum_B": b, "Correlation": vals})

            summary += f"Pairs with |corr| ≥ {threshold:.2f}: {int(mask.sum())} (showing top {top_n})\n"
        else:
            summary += f"Pairs with |corr| ≥ {threshold:.2f}: 0\n"
    else:
        summary += "Not enough variables to compute correlations.\n"

    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": pairs_df if pairs_df is not None else corr_df,
        "summary_text": summary,
        "detailed_summary": f"Total spectra: {int(X.shape[0])} | Wavenumbers: {int(X.shape[1])}",
        "raw_results": {
            "mode": axis_type,
            "method": method,
            "wavenumbers": np.asarray(wavenumbers, dtype=float),
            "spectrum_labels": spectrum_labels,
            "correlation_matrix": corr_df.values,
        },
        "loadings_figure": None  # Correlation analysis does not produce loadings,
    }


def perform_anova_test(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Perform ANOVA test across multiple datasets.

    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - alpha: Significance level (default 0.05)
            - post_hoc: Perform post-hoc tests (default True)
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with ANOVA results and plots
    """
    if progress_callback:
        progress_callback(10)

    # Get parameters
    alpha = params.get("alpha", 0.05)
    post_hoc = params.get("post_hoc", True)

    # Check number of groups
    if len(dataset_data) < 3:
        raise ValueError("At least 3 datasets required for ANOVA")

    # Get common wavenumbers
    dataset_names = list(dataset_data.keys())
    wavenumbers = dataset_data[dataset_names[0]].index.values

    if progress_callback:
        progress_callback(30)

    # Perform ANOVA at each wavenumber
    f_statistics = []
    p_values = []

    for i in range(len(wavenumbers)):
        groups = [dataset_data[name].iloc[i, :].values for name in dataset_names]
        f_stat, p_val = stats.f_oneway(*groups)
        f_statistics.append(f_stat)
        p_values.append(p_val)

    f_statistics = np.array(f_statistics)
    p_values = np.array(p_values)

    # Identify significant wavenumbers
    significant_mask = p_values < alpha

    if progress_callback:
        progress_callback(70)

    # Create primary figure: F-statistic plot
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(12, 8))

    ax1a.plot(wavenumbers, f_statistics, linewidth=1.5, color="blue")
    ax1a.set_ylabel("F-statistic", fontsize=12)
    ax1a.set_title("ANOVA Results", fontsize=14, fontweight="bold")
    ax1a.grid(True, alpha=0.3)
    ax1a.invert_xaxis()

    ax1b.plot(wavenumbers, -np.log10(p_values), linewidth=1.5, color="black")
    ax1b.axhline(-np.log10(alpha), color="red", linestyle="--", label=f"α = {alpha}")
    ax1b.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax1b.set_ylabel("-log₁₀(p-value)", fontsize=12)
    ax1b.legend(loc="best")
    ax1b.grid(True, alpha=0.3)
    ax1b.invert_xaxis()

    plt.tight_layout()

    # Create secondary figure: Mean spectra of all groups
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_names)))
    for i, name in enumerate(dataset_names):
        mean_spec = dataset_data[name].mean(axis=1).values
        ax2.plot(wavenumbers, mean_spec, label=name, color=colors[i], linewidth=1.5)

    # Highlight significant regions
    if np.any(significant_mask):
        y_min, y_max = ax2.get_ylim()
        sig_regions = np.where(significant_mask)[0]
        for idx in sig_regions:
            ax2.axvspan(
                wavenumbers[idx] - 2, wavenumbers[idx] + 2, alpha=0.1, color="yellow"
            )

    ax2.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax2.set_ylabel("Intensity", fontsize=12)
    ax2.set_title("Mean Spectra by Group", fontsize=14, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    if progress_callback:
        progress_callback(90)

    # Create data table
    results_df = pd.DataFrame(
        {
            "Wavenumber": wavenumbers,
            "F_statistic": f_statistics,
            "p_value": p_values,
            "Significant": significant_mask,
        }
    )

    n_significant = np.sum(significant_mask)
    pct_significant = n_significant / len(wavenumbers) * 100

    summary = f"ANOVA completed across {len(dataset_names)} groups.\n"
    summary += f"Significant wavenumbers: {n_significant} ({pct_significant:.1f}%)\n"
    summary += f"Significance level: α = {alpha}"

    return {
        "primary_figure": fig1,
        "secondary_figure": fig2,
        "data_table": results_df,
        "summary_text": summary,
        "detailed_summary": f"Groups: {', '.join(dataset_names)}",
        "raw_results": {
            "f_statistics": f_statistics,
            "p_values": p_values,
            "significant_mask": significant_mask,
            "dataset_names": dataset_names,
        },
    }


def perform_pairwise_statistical_tests(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Pairwise statistical tests at each wavenumber.

    Supports:
      - Independent t-test (Welch)
      - Mann–Whitney U (non-parametric)
      - Wilcoxon signed-rank (paired; requires equal sample counts)
    """

    if progress_callback:
        progress_callback(10)

    dataset_names = list(dataset_data.keys())
    if len(dataset_names) != 2:
        raise ValueError("Pairwise tests require exactly 2 datasets")

    test_type = str(params.get("test_type", "t_test"))
    alpha = float(params.get("alpha", 0.05))
    fdr_correction = bool(params.get("fdr_correction", True))
    show_mean_overlay = bool(params.get("show_mean_overlay", True))

    from .exploratory import interpolate_to_common_wavenumbers

    wavenumbers, _interp_spectra, _labels, X = interpolate_to_common_wavenumbers(
        dataset_data, method="linear"
    )

    # Recover per-dataset spectra arrays aligned to wavenumbers
    df_a = dataset_data[dataset_names[0]]
    df_b = dataset_data[dataset_names[1]]

    # Resample both to common grid
    # NOTE: reuse the interpolation helper logic by calling it with 2 datasets.
    # We already computed the common wavenumbers; now just interpolate columns.
    def _resample(df: pd.DataFrame) -> np.ndarray:
        wn_src = df.index.values.astype(float)
        out = []
        for col in df.columns:
            y = df[col].values.astype(float)
            if wn_src[0] > wn_src[-1]:
                x = wn_src[::-1]
                y = y[::-1]
            else:
                x = wn_src
            out.append(np.interp(wavenumbers, x, y))
        return np.asarray(out, dtype=float)  # (n_spectra, n_wn)

    A = _resample(df_a)
    B = _resample(df_b)

    if progress_callback:
        progress_callback(35)

    p_values = np.full((wavenumbers.size,), np.nan, dtype=float)

    for i in range(wavenumbers.size):
        a = A[:, i]
        b = B[:, i]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < 2 or b.size < 2:
            continue

        if test_type == "mann_whitney":
            # Two-sided MWU
            _u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        elif test_type == "wilcoxon":
            if a.size != b.size:
                raise ValueError(
                    "Wilcoxon signed-rank test requires paired samples with equal counts"
                )
            _w, p = stats.wilcoxon(a, b)
        else:
            # Welch's t-test
            _t, p = stats.ttest_ind(a, b, equal_var=False)
        p_values[i] = float(p)

    if fdr_correction:
        try:
            from statsmodels.stats.multitest import fdrcorrection

            valid = np.isfinite(p_values)
            q = np.full_like(p_values, np.nan)
            if np.any(valid):
                _rej, q_valid = fdrcorrection(p_values[valid], alpha=alpha)
                q[valid] = q_valid
            q_values = q
        except Exception:
            q_values = None
    else:
        q_values = None

    if progress_callback:
        progress_callback(70)

    # Plot 1: mean overlay
    fig1 = None
    if show_mean_overlay:
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        mean_a = np.nanmean(A, axis=0)
        mean_b = np.nanmean(B, axis=0)
        ax1.plot(wavenumbers, mean_a, linewidth=1.8, label=f"{dataset_names[0]} mean")
        ax1.plot(wavenumbers, mean_b, linewidth=1.8, label=f"{dataset_names[1]} mean")
        ax1.set_xlabel("Wavenumber (cm⁻¹)")
        ax1.set_ylabel("Intensity")
        ax1.set_title("Mean Spectra Overlay (Pairwise)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")
        ax1.invert_xaxis()

    # Plot 2: p-value curve
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    y_plot = p_values
    if q_values is not None:
        y_plot = q_values
        label = "-log10(q)"
        thr = alpha
    else:
        label = "-log10(p)"
        thr = alpha

    eps = 1e-300
    ax2.plot(wavenumbers, -np.log10(np.clip(y_plot, eps, 1.0)), linewidth=1.1, color="black")
    ax2.axhline(-np.log10(thr), color="red", linestyle="--", label=f"α={alpha}")
    ax2.set_xlabel("Wavenumber (cm⁻¹)")
    ax2.set_ylabel(label)
    ax2.set_title(f"Pairwise Significance ({test_type})")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    ax2.invert_xaxis()

    if progress_callback:
        progress_callback(90)

    out = {
        "Wavenumber": wavenumbers,
        "p_value": p_values,
    }
    if q_values is not None:
        out["q_value"] = q_values
        sig = (q_values < alpha) & np.isfinite(q_values)
    else:
        sig = (p_values < alpha) & np.isfinite(p_values)
    out["significant"] = sig
    results_df = pd.DataFrame(out)

    n_sig = int(np.sum(sig))
    summary = (
        f"Pairwise statistical tests completed: {dataset_names[0]} vs {dataset_names[1]}.\n"
        f"Test: {test_type}. α={alpha}. "
        + ("FDR correction enabled. " if fdr_correction else "")
        + f"Significant wavenumbers: {n_sig}/{int(wavenumbers.size)}"
    )

    return {
        "primary_figure": fig1,
        "secondary_figure": fig2,
        "data_table": results_df,
        "summary_text": summary,
        "detailed_summary": "",
        "raw_results": {
            "dataset_names": dataset_names,
            "test_type": test_type,
            "p_values": p_values,
            "q_values": q_values,
        },
    }


def perform_band_ratio_analysis(
    dataset_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Compute band/peak ratios per spectrum and compare distributions."""

    if progress_callback:
        progress_callback(10)

    band1_center = float(params.get("band1_center", 1650.0))
    band1_width = float(params.get("band1_width", 20.0))
    band2_center = float(params.get("band2_center", 1450.0))
    band2_width = float(params.get("band2_width", 20.0))
    measure = str(params.get("measure", "area"))  # area|height

    from .exploratory import interpolate_to_common_wavenumbers

    wavenumbers, _interp_spectra, raw_labels, X = interpolate_to_common_wavenumbers(
        dataset_data, method="linear"
    )
    w = np.asarray(wavenumbers, dtype=float)

    # Ensure ascending for integration
    if w.size > 1 and w[0] > w[-1]:
        w = w[::-1]
        X = X[:, ::-1]

    def _band_value(y: np.ndarray, center: float, width: float) -> float:
        lo = center - width
        hi = center + width
        m = (w >= lo) & (w <= hi)
        if not np.any(m):
            return float("nan")
        yy = y[m]
        ww = w[m]
        if measure == "height":
            return float(np.nanmax(yy) - np.nanmin(yy))
        # default: area
        return float(np.trapz(yy - float(np.nanmin(yy)), ww))

    if progress_callback:
        progress_callback(45)

    b1 = np.asarray([_band_value(X[i, :], band1_center, band1_width) for i in range(X.shape[0])], dtype=float)
    b2 = np.asarray([_band_value(X[i, :], band2_center, band2_width) for i in range(X.shape[0])], dtype=float)
    ratio = b1 / (b2 + 1e-12)

    # Plot ratio distributions by dataset label
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    labels = [str(v) for v in raw_labels]
    uniq = sorted(set(labels))
    data = [ratio[np.asarray([l == u for l in labels], dtype=bool)] for u in uniq]
    cmap = plt.get_cmap("tab20", max(len(uniq), 3))
    group_colors = [cmap(i) for i in range(len(uniq))]

    # NOTE: Matplotlib's boxplot API has changed across versions (labels vs tick_labels).
    # To avoid blank/missing labels, we set ticks + ticklabels explicitly.
    bplot = ax1.boxplot(
        data,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "#2c3e50", "linewidth": 1.6},
        meanprops={
            "marker": "^",
            "markerfacecolor": "#2c3e50",
            "markeredgecolor": "#2c3e50",
            "markersize": 7,
        },
        whiskerprops={"color": "#6c757d", "linewidth": 1.2},
        capprops={"color": "#6c757d", "linewidth": 1.2},
    )

    # Make boxes clearly visible: keep a light fill, but a strong outline.
    for patch, color in zip(bplot.get("boxes", []), group_colors):
        try:
            # Face: semi-transparent
            patch.set_facecolor((color[0], color[1], color[2], 0.22))
        except Exception:
            patch.set_facecolor(color)
            patch.set_alpha(0.22)
        # Edge: solid, readable
        patch.set_edgecolor("#2c3e50")
        patch.set_linewidth(1.4)

    # Ensure tick labels always show
    ax1.set_xticks(np.arange(1, len(uniq) + 1))

    # Dynamic tick label sizing for long dataset/group names
    n_groups = int(len(uniq))
    max_label_len = int(max((len(str(u)) for u in uniq), default=0))
    xtick_fontsize = 10
    if max_label_len >= 24 or n_groups >= 10:
        xtick_fontsize = 7
    elif max_label_len >= 16 or n_groups >= 7:
        xtick_fontsize = 8
    elif max_label_len >= 12 or n_groups >= 5:
        xtick_fontsize = 9

    xtick_rotation = 30 if max_label_len < 18 else 45
    ax1.set_xticklabels(uniq, fontsize=xtick_fontsize, rotation=xtick_rotation, ha="right")

    ratio_title = (
        f"Band ratio ({measure}): {band1_center:.0f}±{band1_width:.0f} / {band2_center:.0f}±{band2_width:.0f} cm⁻¹"
    )
    ax1.set_title("Band Ratio Distributions", fontsize=13, fontweight="bold")
    ax1.set_ylabel(ratio_title)
    ax1.grid(True, axis="y", alpha=0.3)

    # Move plot area slightly up to make room for long x-tick labels
    # (especially in embedded widgets where resize-tight_layout can clip labels)
    bottom = 0.18
    if xtick_rotation >= 45:
        bottom = 0.26
    if max_label_len >= 24:
        bottom = 0.30
    if xtick_fontsize <= 8:
        bottom = max(bottom, 0.24)
    try:
        fig1.subplots_adjust(bottom=bottom, top=0.92, left=0.10, right=0.98)
    except Exception:
        pass

    try:
        from matplotlib.patches import Patch

        handles = [Patch(facecolor=c, edgecolor=c, alpha=0.35, label=lbl) for lbl, c in zip(uniq, group_colors)]
        ax1.legend(handles=handles, title="Group", loc="best", frameon=True)
    except Exception:
        pass

    # Bar chart of mean ratios
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    means = [float(np.nanmean(d)) if len(d) else float("nan") for d in data]
    ax2.bar(np.arange(len(uniq)), means, color=group_colors, alpha=0.75)
    ax2.set_xticks(np.arange(len(uniq)))
    ax2.set_xticklabels(uniq, fontsize=xtick_fontsize, rotation=xtick_rotation, ha="right")
    ax2.set_ylabel("Mean band ratio")
    ax2.set_title("Mean Band Ratio per Group", fontsize=13, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    try:
        fig2.subplots_adjust(bottom=max(0.18, min(0.32, bottom - 0.02)), top=0.90, left=0.10, right=0.98)
    except Exception:
        pass

    try:
        from matplotlib.patches import Patch

        handles = [Patch(facecolor=c, edgecolor=c, alpha=0.75, label=lbl) for lbl, c in zip(uniq, group_colors)]
        ax2.legend(handles=handles, title="Group", loc="best", frameon=True)
    except Exception:
        pass

    df = pd.DataFrame(
        {
            "label": labels,
            "band1_value": b1,
            "band2_value": b2,
            "ratio": ratio,
        }
    )

    if progress_callback:
        progress_callback(90)

    summary = (
        f"Band ratio computed using {measure}. "
        f"Band1: {band1_center:.0f}±{band1_width:.0f}, Band2: {band2_center:.0f}±{band2_width:.0f}. "
        f"Spectra: {int(X.shape[0])}."
    )

    return {
        "primary_figure": fig1,
        "secondary_figure": fig2,
        "data_table": df,
        "summary_text": summary,
        "detailed_summary": "",
        "raw_results": {},
    }
