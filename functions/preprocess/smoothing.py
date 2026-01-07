"""
Smoothing Methods for Raman Spectra

This module contains dedicated smoothing methods for noise reduction in Raman spectra.
These methods preserve peak shapes while reducing random noise.
"""

import numpy as np

try:
    from scipy.signal import savgol_filter

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from ..utils import create_logs
except ImportError:
    # Fallback logging function
    def create_logs(log_id, source, message, status="info"):
        print(f"[{status.upper()}] {source}: {message}")


class SavitzkyGolaySmoothing:
    """
    Savitzky-Golay smoothing filter for noise reduction.

    This is a pure smoothing method (derivative order = 0) that uses polynomial
    fitting within a moving window to smooth spectra while preserving peak shapes.

    Recommended for Raman spectroscopy preprocessing pipelines as it provides
    good noise reduction without significant peak distortion.

    Attributes:
        window_length (int): Length of the filter window (must be odd)
        polyorder (int): Order of polynomial for fitting (must be < window_length)
    """

    def __init__(self, window_length: int = 7, polyorder: int = 3):
        """
        Initialize Savitzky-Golay smoothing filter.

        Args:
            window_length (int): Filter window length (must be odd, >= 3).
                                Default 7 is suitable for most Raman spectra.
            polyorder (int): Polynomial order for fitting (must be < window_length).
                            Default 3 provides good smoothing while preserving peaks.
        """
        if not isinstance(window_length, int) or window_length <= 0:
            raise ValueError("Window length must be a positive integer")
        if window_length % 2 == 0:
            window_length += 1  # Ensure odd window length
        if not isinstance(polyorder, int) or polyorder < 0:
            raise ValueError("Polynomial order must be a non-negative integer")
        if window_length <= polyorder:
            raise ValueError(
                f"Window length ({window_length}) must be greater than polynomial order ({polyorder})"
            )

        self.window_length = window_length
        self.polyorder = polyorder

    def __call__(self, data) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing to data.

        Handles both SpectralContainer (RamanSPy) and numpy array (sklearn pipelines).

        Args:
            data: SpectralContainer or numpy array (1D or 2D)

        Returns:
            Same type as input with smoothing applied
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for Savitzky-Golay smoothing")

        # Detect input type
        is_container = hasattr(data, "spectral_data")

        if is_container:
            # SpectralContainer input (RamanSPy workflow)
            spectra = data.spectral_data
            axis = data.spectral_axis
        else:
            # numpy array input (sklearn pipeline)
            spectra = data
            axis = None

        # Apply smoothing
        if spectra.ndim == 1:
            smoothed = self._smooth_spectrum(spectra)
        elif spectra.ndim == 2:
            smoothed = np.array(
                [self._smooth_spectrum(spectrum) for spectrum in spectra]
            )
        else:
            raise ValueError("Spectra must be 1D or 2D array")

        # Return in same format as input
        if is_container:
            import ramanspy as rp

            return rp.SpectralContainer(smoothed, axis)
        else:
            return smoothed

    def _smooth_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing to a single spectrum.

        Args:
            spectrum (np.ndarray): 1D spectrum to smooth

        Returns:
            np.ndarray: Smoothed spectrum
        """
        if spectrum is None or spectrum.size == 0:
            return spectrum

        if len(spectrum) < self.window_length:
            create_logs(
                "smoothing_warning",
                "SavitzkyGolaySmoothing",
                f"Spectrum length ({len(spectrum)}) < window_length ({self.window_length}). Returning original.",
                status="warning",
            )
            return spectrum

        try:
            return savgol_filter(
                spectrum,
                window_length=self.window_length,
                polyorder=self.polyorder,
                deriv=0,  # Pure smoothing, no derivative
            )
        except Exception as e:
            create_logs(
                "smoothing_error",
                "SavitzkyGolaySmoothing",
                f"Smoothing failed: {e}. Returning original spectrum.",
                status="error",
            )
            return spectrum

    def apply(self, spectra) -> "rp.SpectralContainer":
        """Apply Savitzky-Golay smoothing to ramanspy SpectralContainer."""
        return self.__call__(spectra)


class MeanCentering:
    """
    Mean centering normalization for Raman spectra.

    Subtracts the mean intensity from each spectrum, centering the data around zero.
    This is useful as a preprocessing step before PCA or other statistical analyses.

    Attributes:
        per_feature (bool): If True, center across samples (column-wise for 2D arrays).
                           If False, center each spectrum individually (default).
    """

    def __init__(self, per_feature: bool = False):
        """
        Initialize mean centering.

        Args:
            per_feature (bool): If True, compute mean across all spectra for each
                               wavenumber position (column-wise). If False, center
                               each spectrum by its own mean (row-wise). Default False.
        """
        self.per_feature = per_feature
        self._fitted_mean = None

    def fit(self, spectra: np.ndarray) -> "MeanCentering":
        """
        Fit the mean centering (compute mean for later transform).

        Args:
            spectra (np.ndarray): 2D array of spectra (n_samples, n_features)

        Returns:
            self: Returns self for method chaining
        """
        if self.per_feature:
            if spectra.ndim == 1:
                self._fitted_mean = np.mean(spectra)
            else:
                self._fitted_mean = np.mean(spectra, axis=0)
        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """
        Apply mean centering to spectra.

        Args:
            spectra (np.ndarray): 1D or 2D array of spectra

        Returns:
            np.ndarray: Mean-centered spectra
        """
        if self.per_feature:
            if self._fitted_mean is None:
                raise ValueError("Must fit before transform when per_feature=True")
            return spectra - self._fitted_mean
        else:
            # Center each spectrum by its own mean
            if spectra.ndim == 1:
                return spectra - np.mean(spectra)
            else:
                return spectra - np.mean(spectra, axis=1, keepdims=True)

    def fit_transform(self, spectra: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(spectra).transform(spectra)

    def __call__(self, data) -> np.ndarray:
        """
        Apply mean centering to data.

        Handles both SpectralContainer (RamanSPy) and numpy array.

        Args:
            data: SpectralContainer or numpy array

        Returns:
            Same type as input with mean centering applied
        """
        # Detect input type
        is_container = hasattr(data, "spectral_data")

        if is_container:
            spectra = data.spectral_data
            axis = data.spectral_axis
        else:
            spectra = data
            axis = None

        # Apply centering
        if self.per_feature and self._fitted_mean is None:
            centered = self.fit_transform(spectra)
        else:
            centered = self.transform(spectra)

        # Return in same format as input
        if is_container:
            import ramanspy as rp

            return rp.SpectralContainer(centered, axis)
        else:
            return centered

    def apply(self, spectra) -> "rp.SpectralContainer":
        """Apply mean centering to ramanspy SpectralContainer."""
        return self.__call__(spectra)
