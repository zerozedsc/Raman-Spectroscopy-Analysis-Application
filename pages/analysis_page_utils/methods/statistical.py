"""
Statistical Analysis Methods

This module implements statistical analysis methods for Raman spectra including
spectral comparison, peak analysis, correlation analysis, and ANOVA.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import normalize


def perform_spectral_comparison(dataset_data: Dict[str, pd.DataFrame],
                                params: Dict[str, Any],
                                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
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
    
    # Extract first two datasets for comparison
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
        _, p_corrected = fdrcorrection(p_values, alpha=1-confidence_level)
        significant_mask = p_corrected < (1 - confidence_level)
    else:
        significant_mask = p_values < (1 - confidence_level)
    
    if progress_callback:
        progress_callback(70)
    
    # Create primary figure: Mean spectra with CI
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(wavenumbers, mean1, label=dataset_names[0], linewidth=2, color='blue')
    ax1.plot(wavenumbers, mean2, label=dataset_names[1], linewidth=2, color='red')
    
    if show_ci:
        ax1.fill_between(wavenumbers, mean1 - ci1, mean1 + ci1,
                        alpha=0.2, color='blue')
        ax1.fill_between(wavenumbers, mean2 - ci2, mean2 + ci2,
                        alpha=0.2, color='red')
    
    if highlight_significant:
        # Highlight significant regions
        sig_regions = np.where(significant_mask)[0]
        if len(sig_regions) > 0:
            y_min, y_max = ax1.get_ylim()
            for idx in sig_regions:
                ax1.axvspan(wavenumbers[idx]-2, wavenumbers[idx]+2,
                          alpha=0.1, color='yellow')
    
    ax1.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title('Spectral Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # Create secondary figure: P-value plot
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    
    ax2.plot(wavenumbers, -np.log10(p_values), linewidth=1.5, color='black')
    ax2.axhline(-np.log10(1-confidence_level), color='red', linestyle='--',
               label=f'{confidence_level*100:.0f}% significance')
    ax2.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('-log₁₀(p-value)', fontsize=12)
    ax2.set_title('Statistical Significance', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    if progress_callback:
        progress_callback(90)
    
    # Create data table
    results_df = pd.DataFrame({
        'Wavenumber': wavenumbers,
        f'{dataset_names[0]}_mean': mean1,
        f'{dataset_names[0]}_sem': sem1,
        f'{dataset_names[1]}_mean': mean2,
        f'{dataset_names[1]}_sem': sem2,
        'p_value': p_values,
        'significant': significant_mask
    })
    
    n_significant = np.sum(significant_mask)
    pct_significant = n_significant / len(wavenumbers) * 100
    
    summary = f"Spectral comparison between {dataset_names[0]} and {dataset_names[1]}.\n"
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
        }
    }


def perform_peak_analysis(dataset_data: Dict[str, pd.DataFrame],
                         params: Dict[str, Any],
                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
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
    
    # Load Raman peak assignments from JSON
    raman_peaks_data = {}
    try:
        peaks_json_path = os.path.join("assets", "data", "raman_peaks.json")
        if os.path.exists(peaks_json_path):
            with open(peaks_json_path, 'r', encoding='utf-8') as f:
                raman_peaks_data = json.load(f)
            print(f"[DEBUG] Loaded {len(raman_peaks_data)} peak assignments from raman_peaks.json")
        else:
            print(f"[DEBUG] WARNING: raman_peaks.json not found at {peaks_json_path}")
    except Exception as e:
        print(f"[DEBUG] ERROR loading raman_peaks.json: {e}")
    
    # Use first dataset for mean spectrum
    dataset_name = list(dataset_data.keys())[0]
    df = dataset_data[dataset_name]
    
    wavenumbers = df.index.values
    mean_spectrum = df.mean(axis=1).values
    
    if progress_callback:
        progress_callback(40)
    
    # Normalize spectrum for peak detection
    spectrum_normalized = (mean_spectrum - mean_spectrum.min()) / (mean_spectrum.max() - mean_spectrum.min())
    
    # Find peaks
    peaks, properties = find_peaks(
        spectrum_normalized,
        prominence=prominence_threshold,
        width=width_min
    )
    
    if progress_callback:
        progress_callback(70)
    
    # Get top N peaks by prominence
    prominences = properties['prominences']
    sorted_indices = np.argsort(prominences)[::-1][:top_n_peaks]
    top_peaks = peaks[sorted_indices]
    top_prominences = prominences[sorted_indices]
    
    print(f"[DEBUG] Peak detection: Found {len(peaks)} total peaks, showing top {len(top_peaks)} peaks")
    print(f"[DEBUG] top_n_peaks parameter: {top_n_peaks}")
    print(f"[DEBUG] Actual peaks shown: {len(top_peaks)}")
    
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
        closest_distance = float('inf')
        
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
    
    ax1.plot(wavenumbers, mean_spectrum, linewidth=1.5, color='blue', label='Mean spectrum')
    ax1.plot(wavenumbers[top_peaks], mean_spectrum[top_peaks],
            'ro', markersize=10, label=f'Top {len(top_peaks)} peaks', zorder=5)
    
    # Annotate ALL peaks with wavenumber AND component assignment labels
    print(f"[DEBUG] Adding wavenumber + component annotations for {len(top_peaks)} peaks")
    for i, peak_idx in enumerate(top_peaks):
        wavenumber = wavenumbers[peak_idx]
        intensity = mean_spectrum[peak_idx]
        
        # Find component assignment
        assignment = find_peak_assignment(wavenumber)
        
        # Build annotation text: wavenumber on first line, assignment on second line
        if show_assignments and assignment:
            annotation_text = f'{wavenumber:.0f} cm⁻¹\n{assignment}'
        else:
            annotation_text = f'{wavenumber:.0f} cm⁻¹'
        
        # Alternate label positions to avoid overlap
        y_offset = 15 if i % 2 == 0 else 30
        
        # Use different colors for peaks with vs without assignments
        box_color = 'lightyellow' if assignment else 'lightgray'
        edge_color = 'orange' if assignment else 'gray'
        
        ax1.annotate(annotation_text,
                    xy=(wavenumber, intensity),
                    xytext=(0, y_offset), 
                    textcoords='offset points',
                    ha='center', 
                    fontsize=7 if assignment else 8,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=box_color, alpha=0.8, edgecolor=edge_color),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=1),
                    zorder=10)
        
        if assignment:
            print(f"[DEBUG] Peak {i+1}/{len(top_peaks)}: {wavenumber:.0f} cm⁻¹ → {assignment[:30]}")
        else:
            print(f"[DEBUG] Peak {i+1}/{len(top_peaks)}: {wavenumber:.0f} cm⁻¹ (no assignment found)")
    
    ax1.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title('Peak Analysis with Component Assignments', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # Create secondary figure: Peak intensity distribution
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    peak_wavenumbers = wavenumbers[top_peaks]
    peak_intensities = mean_spectrum[top_peaks]
    
    bars = ax2.bar(range(len(top_peaks)), peak_intensities, color='steelblue')
    ax2.set_xticks(range(len(top_peaks)))
    ax2.set_xticklabels([f'{wn:.0f}' for wn in peak_wavenumbers], rotation=45, ha='right')
    ax2.set_xlabel('Peak Position (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('Peak Intensity', fontsize=12)
    ax2.set_title('Peak Intensity Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    if progress_callback:
        progress_callback(90)
    
    # Create data table with component assignments
    peak_assignments = [find_peak_assignment(wn) for wn in wavenumbers[top_peaks]]
    
    results_df = pd.DataFrame({
        'Peak_Position': wavenumbers[top_peaks],
        'Intensity': mean_spectrum[top_peaks],
        'Component_Assignment': peak_assignments,
        'Prominence': top_prominences,
        'Width': properties['widths'][sorted_indices]
    })
    results_df = results_df.sort_values('Intensity', ascending=False)
    
    summary = f"Peak analysis completed on {dataset_name}.\n"
    summary += f"Found {len(peaks)} peaks total, showing top {len(top_peaks)}.\n"
    summary += f"Peak detection threshold: prominence = {prominence_threshold:.3f}\n"
    if show_assignments:
        assigned_count = sum(1 for a in peak_assignments if a)
        summary += f"Component assignments: {assigned_count}/{len(top_peaks)} peaks matched\n"
    
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
        }
    }


def perform_correlation_analysis(dataset_data: Dict[str, pd.DataFrame],
                                 params: Dict[str, Any],
                                 progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Perform correlation analysis between spectra.
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - method: Correlation method ('pearson', 'spearman') (default 'pearson')
            - show_pvalues: Show p-value matrix (default False)
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with correlation matrix and heatmap
    """
    if progress_callback:
        progress_callback(10)
    
    # Get parameters
    method = params.get("method", "pearson")
    show_pvalues = params.get("show_pvalues", False)
    
    # Combine all datasets
    all_spectra = []
    labels = []
    
    for dataset_name, df in dataset_data.items():
        for col in df.columns:
            all_spectra.append(df[col].values)
            labels.append(f"{dataset_name}_{col}")
    
    # Create DataFrame
    spectra_df = pd.DataFrame(all_spectra, index=labels).T
    
    if progress_callback:
        progress_callback(40)
    
    # Calculate correlation matrix
    if method == "pearson":
        corr_matrix = spectra_df.corr(method='pearson')
    elif method == "spearman":
        corr_matrix = spectra_df.corr(method='spearman')
    else:
        corr_matrix = spectra_df.corr()
    
    if progress_callback:
        progress_callback(70)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto',
                   vmin=-1, vmax=1)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=12)
    
    # Labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    
    ax.set_title(f'Correlation Matrix ({method.capitalize()})',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if progress_callback:
        progress_callback(90)
    
    # Statistics
    upper_triangle = np.triu(corr_matrix.values, k=1)
    upper_triangle_flat = upper_triangle[upper_triangle != 0]
    
    mean_corr = np.mean(upper_triangle_flat)
    std_corr = np.std(upper_triangle_flat)
    min_corr = np.min(upper_triangle_flat)
    max_corr = np.max(upper_triangle_flat)
    
    summary = f"Correlation analysis completed using {method} method.\n"
    summary += f"Mean correlation: {mean_corr:.3f} ± {std_corr:.3f}\n"
    summary += f"Range: [{min_corr:.3f}, {max_corr:.3f}]"
    
    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": corr_matrix,
        "summary_text": summary,
        "detailed_summary": f"Total spectra: {len(labels)}",
        "raw_results": {
            "correlation_matrix": corr_matrix.values,
            "labels": labels,
            "method": method
        }
    }


def perform_anova_test(dataset_data: Dict[str, pd.DataFrame],
                      params: Dict[str, Any],
                      progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
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
    
    ax1a.plot(wavenumbers, f_statistics, linewidth=1.5, color='blue')
    ax1a.set_ylabel('F-statistic', fontsize=12)
    ax1a.set_title('ANOVA Results', fontsize=14, fontweight='bold')
    ax1a.grid(True, alpha=0.3)
    ax1a.invert_xaxis()
    
    ax1b.plot(wavenumbers, -np.log10(p_values), linewidth=1.5, color='black')
    ax1b.axhline(-np.log10(alpha), color='red', linestyle='--',
                label=f'α = {alpha}')
    ax1b.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax1b.set_ylabel('-log₁₀(p-value)', fontsize=12)
    ax1b.legend(loc='best')
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
            ax2.axvspan(wavenumbers[idx]-2, wavenumbers[idx]+2,
                       alpha=0.1, color='yellow')
    
    ax2.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('Intensity', fontsize=12)
    ax2.set_title('Mean Spectra by Group', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    if progress_callback:
        progress_callback(90)
    
    # Create data table
    results_df = pd.DataFrame({
        'Wavenumber': wavenumbers,
        'F_statistic': f_statistics,
        'p_value': p_values,
        'Significant': significant_mask
    })
    
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
            "dataset_names": dataset_names
        }
    }
