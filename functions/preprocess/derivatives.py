"""
Derivative Processing Methods for Raman Spectra

This module contains methods for calculating derivatives of Raman spectra
for peak enhancement and baseline removal.
"""

import numpy as np

try:
    import ramanspy as rp
    from scipy.signal import savgol_filter

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class Derivative:
    """
    Spectral derivative calculation using Savitzky-Golay method.

    Derivatives are useful for:
    - Peak resolution enhancement
    - Baseline removal (1st derivative)
    - Peak detection and analysis
    - Removing overlapping backgrounds

    Attributes:
        order (int): Derivative order (1 for first derivative, 2 for second)
        window_length (int): Length of the filter window (must be odd)
        polyorder (int): Order of polynomial for fitting
    """

    def __init__(self, order: int = 1, window_length: int = 11, polyorder: int = 3):
        """
        Initialize derivative processor.

        Args:
            order (int): Derivative order (1 or 2)
            window_length (int): Filter window length (must be odd, >= polyorder + 1)
                                 Default 11 provides good noise reduction for Raman spectra.
            polyorder (int): Polynomial order for Savitzky-Golay filter.
                            Default 3 balances noise reduction and peak preservation.
        """
        if order not in [1, 2]:
            raise ValueError("Derivative order must be 1 or 2")
        if not isinstance(window_length, int) or window_length <= 0:
            raise ValueError("Window length must be a positive integer")
        if window_length % 2 == 0:
            window_length += 1  # Ensure odd window length
        if window_length <= polyorder:
            raise ValueError("Window length must be greater than polynomial order")

        self.order = order
        self.window_length = window_length
        self.polyorder = polyorder

    def __call__(self, spectra: np.ndarray) -> np.ndarray:
        """Apply derivative calculation to numpy array format."""
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for derivative calculations")

        if spectra.ndim == 1:
            return self._calculate_derivative(spectra)
        elif spectra.ndim == 2:
            return np.array(
                [self._calculate_derivative(spectrum) for spectrum in spectra]
            )
        else:
            raise ValueError("Spectra must be 1D or 2D array")

    def apply(self, spectra: "rp.SpectralContainer") -> "rp.SpectralContainer":
        """Apply derivative calculation to ramanspy SpectralContainer."""
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for derivative calculations")

        data = spectra.spectral_data

        if data.ndim == 1:
            derivative_data = self._calculate_derivative(data)
        else:
            derivative_data = np.array(
                [self._calculate_derivative(spectrum) for spectrum in data]
            )

        return rp.SpectralContainer(derivative_data, spectra.spectral_axis)

    def _calculate_derivative(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Calculate derivative of a single spectrum.

        Args:
            spectrum (np.ndarray): 1D spectrum

        Returns:
            np.ndarray: Derivative of the spectrum
        """
        if len(spectrum) < self.window_length:
            # If spectrum is too short, use simple numerical differentiation
            if self.order == 1:
                return np.gradient(spectrum)
            else:  # order == 2
                grad1 = np.gradient(spectrum)
                return np.gradient(grad1)

        try:
            # Use Savitzky-Golay filter for derivative calculation
            derivative = savgol_filter(
                spectrum,
                window_length=self.window_length,
                polyorder=self.polyorder,
                deriv=self.order,
            )
            return derivative
        except Exception as e:
            # Fallback to simple numerical differentiation
            print(
                f"Warning: Savitzky-Golay failed ({e}), using numerical differentiation"
            )
            if self.order == 1:
                return np.gradient(spectrum)
            else:  # order == 2
                grad1 = np.gradient(spectrum)
                return np.gradient(grad1)
