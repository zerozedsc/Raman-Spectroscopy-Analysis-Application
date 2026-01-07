"""
Basic Plotting Functions for Raman Spectroscopy

This module provides simple visualization functions for Raman spectral data,
including raw spectra plotting, processed spectra visualization, and peak extraction.

Functions:
    visualize_raman_spectra: Plot raw Raman spectra from DataFrame
    visualize_processed_spectra: Plot preprocessed spectral data
    extract_raman_characteristics: Extract peaks and calculate AUC

Author: MUHAMMAD HELMI BIN ROZAIN
Created: 2025-10-01 (Extracted from core.py during Phase 1 refactoring)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from numpy import trapz
from typing import Union, List, Tuple
from functions.configs import console_log


def visualize_raman_spectra(
    df: pd.DataFrame,
    wavenumber_colname: str = "wavenumber",
    title: str = "Raman Spectra",
    figsize: tuple = (12, 6),
    xlim: tuple = None,
    ylim: tuple = None,
    legend: bool = True,
    legend_loc: str = "best",
    sample_limit: int = 10,
) -> plt:
    """
    Visualize the Raman spectra data from a DataFrame.
    Clears any existing plots before creating a new one.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing Raman data with wavenumber column and intensity columns
    wavenumber_colname : str
        Name of the wavenumber column (default: "wavenumber")
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height) in inches
    xlim : tuple or None
        Optional x-axis limits (min, max)
    ylim : tuple or None
        Optional y-axis limits (min, max)
    legend : bool
        Whether to display the legend
    legend_loc : str
        Legend location
    sample_limit : int
        Maximum number of samples to plot (to avoid overcrowding)

    Returns:
    --------
    plt : matplotlib.pyplot
        The plot object for further customization

    Examples:
    ---------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'wavenumber': [400, 500, 600],
    ...     'sample1': [10, 20, 15],
    ...     'sample2': [12, 18, 16]
    ... })
    >>> plt = visualize_raman_spectra(df)
    >>> plt.show()
    """
    # Check if DataFrame is empty
    if df.empty:
        console_log("DataFrame is empty. No data to plot.")
        return None

    # Determine wavenumber axis
    if df.index.name == wavenumber_colname:
        wavenumbers = df.index.values
        intensity_columns = df.columns
    elif wavenumber_colname in df.columns:
        wavenumbers = df[wavenumber_colname].values
        intensity_columns = [col for col in df.columns if col != wavenumber_colname]
    else:
        console_log(f"Wavenumber column '{wavenumber_colname}' not found in DataFrame.")
        return None

    # Limit number of samples to plot if needed
    if len(intensity_columns) > sample_limit:
        console_log(
            f"Limiting plot to {sample_limit} samples out of {len(intensity_columns)}"
        )
        intensity_columns = intensity_columns[:sample_limit]

    # Clear any existing plots
    plt.clf()
    plt.close("all")
    fig = plt.figure(num=1, figsize=figsize, clear=True)

    # Plot each spectrum
    for col in intensity_columns:
        if df.index.name == wavenumber_colname:
            plt.plot(wavenumbers, df[col], label=col)
        else:
            plt.plot(wavenumbers, df[col].values, label=col)

    # Set plot attributes
    plt.title(title)
    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.grid(True, alpha=0.3)

    if legend:
        plt.legend(loc=legend_loc, fontsize="small")

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()
    return plt


def visualize_processed_spectra(
    spectral_data: np.ndarray,
    spectral_axis: np.ndarray,
    title: str = "Processed Raman Spectra",
    figsize: tuple = (12, 6),
    xlim: tuple = None,
    ylim: tuple = None,
    legend: bool = True,
    sample_names: List[str] = None,
    legend_loc: str = "best",
    sample_limit: int = 10,
    cmap: str = "viridis",
    add_mean: bool = False,
) -> plt:
    """
    Visualize processed spectral data from ramanspy or numpy arrays.

    Parameters:
    -----------
    spectral_data : np.ndarray
        Array of shape (n_samples, n_wavenumbers) containing processed spectra
    spectral_axis : np.ndarray
        Array containing wavenumber axis
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height) in inches
    xlim : tuple or None
        Optional x-axis limits (min, max)
    ylim : tuple or None
        Optional y-axis limits (min, max)
    legend : bool
        Whether to display the legend
    sample_names : list or None
        List of names for each spectrum (if None, will use Sample 1, Sample 2, etc.)
    legend_loc : str
        Legend location
    sample_limit : int
        Maximum number of samples to plot (to avoid overcrowding)
    cmap : str
        Matplotlib colormap for spectra if no sample names provided
    add_mean : bool
        Whether to add the mean spectrum in bold

    Returns:
    --------
    plt : matplotlib.pyplot
        The plot object for further customization

    Examples:
    ---------
    >>> import numpy as np
    >>> spectral_data = np.random.rand(5, 100)  # 5 samples, 100 wavenumbers
    >>> spectral_axis = np.linspace(400, 1800, 100)
    >>> plt = visualize_processed_spectra(spectral_data, spectral_axis)
    >>> plt.show()
    """
    # Check inputs
    if spectral_data.shape[1] != len(spectral_axis):
        # Transpose if needed (sometimes ramanspy returns (n_wavenumbers, n_samples))
        if spectral_data.shape[0] == len(spectral_axis):
            spectral_data = spectral_data.T
        else:
            raise ValueError(
                f"Spectral data shape {spectral_data.shape} doesn't match "
                f"spectral axis length {len(spectral_axis)}"
            )

    # Limit samples if needed
    n_samples = spectral_data.shape[0]
    if n_samples > sample_limit:
        console_log(f"Limiting plot to {sample_limit} samples out of {n_samples}")
        plot_samples = min(n_samples, sample_limit)
    else:
        plot_samples = n_samples

    # Prepare sample names
    if sample_names is None:
        sample_names = [f"Sample {i+1}" for i in range(plot_samples)]
    elif len(sample_names) < plot_samples:
        # Extend sample names if too few
        sample_names = list(sample_names) + [
            f"Sample {i+1}" for i in range(len(sample_names), plot_samples)
        ]

    # Clear any existing plots
    plt.clf()
    plt.close("all")
    fig = plt.figure(num=1, figsize=figsize, clear=True)

    # Plot each spectrum
    cmap_func = plt.get_cmap(cmap)
    for i in range(plot_samples):
        color = (
            cmap_func(i / max(1, plot_samples - 1)) if n_samples > 1 else cmap_func(0.5)
        )
        plt.plot(
            spectral_axis,
            spectral_data[i],
            label=sample_names[i],
            color=color,
            alpha=0.7,
        )

    # Add mean spectrum if requested
    if add_mean and n_samples > 1:
        mean_spectrum = np.mean(spectral_data[:plot_samples], axis=0)
        plt.plot(spectral_axis, mean_spectrum, "k-", linewidth=2, label="Mean Spectrum")

    # Set plot attributes
    plt.title(title)
    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Intensity (normalized)")
    plt.grid(True, alpha=0.3)

    if legend:
        if n_samples <= 15:  # Only show legend if not too many samples
            plt.legend(loc=legend_loc, fontsize="small")

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()
    return plt


def extract_raman_characteristics(
    x: np.ndarray, y: np.ndarray, sample_name: str = "Sample", show_plot: bool = False
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Extract Raman characteristics from a single spectrum.

    Identifies prominent peaks and calculates the area under the curve (AUC).

    Parameters:
    -----------
    x : np.ndarray
        Wavenumber array
    y : np.ndarray
        Intensity array
    sample_name : str
        Name of the sample for labeling
    show_plot : bool
        Whether to show the plot of the spectrum with detected peaks

    Returns:
    --------
    top_peaks : list of tuples
        List of (wavenumber, intensity) tuples for the top 5 peaks
    auc : float
        Area under the curve

    Examples:
    ---------
    >>> x = np.linspace(400, 1800, 100)
    >>> y = np.random.rand(100) + np.exp(-(x - 1000)**2 / 10000)
    >>> peaks, auc = extract_raman_characteristics(x, y, "TestSample")
    >>> print(f"Found {len(peaks)} peaks, AUC = {auc:.2f}")
    """
    # Find peaks using simple threshold (80th percentile)
    peaks, props = find_peaks(y, height=np.percentile(y, 80))
    peak_positions = x[peaks]
    peak_intensities = y[peaks]

    # Sort peaks by intensity (highest first)
    sorted_idx = np.argsort(peak_intensities)[::-1]
    top_peaks = [
        (float(peak_positions[i]), float(peak_intensities[i])) for i in sorted_idx[:5]
    ]

    # Calculate area under curve
    auc = float(trapz(y, x))

    # Plot if requested
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="Raman Spectrum")
        plt.scatter(
            peak_positions, peak_intensities, color="red", label="Detected Peaks"
        )

        for pos, inten in top_peaks:
            plt.text(pos, inten, f"{int(pos)}", fontsize=8, ha="center")

        plt.title(f"Raman Spectrum - {sample_name}")
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Intensity (a.u.)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Print characteristic summary
    console_log(f"----- {sample_name} Characteristics -----")
    console_log(f"Top Peaks:")
    for i, (pos, inten) in enumerate(top_peaks):
        console_log(f"  {i+1}. {pos:.1f} cm⁻¹  (Intensity: {inten:.1f})")
    console_log(f"Total Area Under Curve (AUC): {auc:.2f}")
    console_log(f"Approximate Noise Level (std): {np.std(y):.2f}")

    return top_peaks, auc
