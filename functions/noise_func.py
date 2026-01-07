import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from typing import Tuple


class RamanNoiseProcessor:
    """
    A class to process Raman spectral data by adding noise and detecting baseline regions.
    Attributes:
    -----------
    df : pd.DataFrame
        DataFrame containing Raman spectral data with wavenumber as index and intensity values as columns
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def add_gaussian_noise(self, noise_level=0.01) -> pd.DataFrame:
        """
        Add Gaussian noise to the Raman spectrum.
        Parameters:
        -----------
        noise_level : float
            Standard deviation of the Gaussian noise to be added
        Returns:
        --------
        DataFrame with added Gaussian noise
        """
        noisy_df = self.df.copy()
        # Add noise to all numeric columns
        for col in noisy_df.columns:
            if np.issubdtype(noisy_df[col].dtype, np.number):
                noise = np.random.normal(
                    loc=0.0, scale=noise_level, size=noisy_df[col].shape
                )
                noisy_df[col] = noisy_df[col] + noise
        return noisy_df

    def auto_detect_baseline_region(self, window_size=50) -> pd.DataFrame:
        """
        Automatically detect flat (low-variance) region in the Raman spectrum.
        Uses a sliding window over the wavenumber axis (index).

        Parameters:
        -----------
        window_size : int
            Size of the sliding window for baseline detection
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the detected baseline region
        """
        wavenumber = self.df.index.values
        spectra = self.df.values

        min_std = np.inf
        min_idx = 0

        # Scan through the spectrum using a moving window
        for i in range(0, len(wavenumber) - window_size):
            segment = spectra[i : i + window_size, :]
            segment_std = np.std(segment)
            if segment_std < min_std:
                min_std = segment_std
                min_idx = i

        # Extract the best baseline region
        start_wn = wavenumber[min_idx]
        end_wn = wavenumber[min_idx + window_size - 1]

        console_log(f"Auto-detected baseline region: {start_wn:.2f}–{end_wn:.2f} cm⁻¹")

        # Return the sliced baseline DataFrame
        baseline_df = self.df.loc[start_wn:end_wn]
        return baseline_df

    def baselineAndGaussianNoise(self, window_size=50) -> pd.DataFrame:
        """
        Detect baseline and add Gaussian noise to the DataFrame.
        Parameters:
        -----------
        window_size : int
            Size of the sliding window for baseline detection

        Returns:
        --------
        pd.DataFrame with Gaussian noise added

        """
        # Detect baseline region
        baseline_df = self.auto_detect_baseline_region(window_size=window_size)

        # Calculate noise standard deviation
        noise_std = baseline_df.std().mean()

        # Add Gaussian noise
        noisy_df = self.add_gaussian_noise(noise_level=noise_std)

        return noisy_df
