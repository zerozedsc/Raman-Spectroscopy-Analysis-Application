"""
Wavenumber and Intensity Calibration Methods

This module contains methods for calibrating the wavenumber axis and
intensity values of Raman spectra.
"""

import numpy as np
from typing import Dict, Optional

try:
    import ramanspy as rp

    RAMANSPY_AVAILABLE = True
except ImportError:
    RAMANSPY_AVAILABLE = False


class WavenumberCalibration:
    """
    Wavenumber axis calibration using reference peaks (e.g., Silicon at 520 cm⁻¹).

    This method corrects for systematic wavenumber shifts by comparing measured
    peak positions with known reference values and applying polynomial correction.

    Attributes:
        reference_peaks (dict): Dictionary of reference peak positions {name: wavenumber}
        poly_order (int): Order of polynomial for wavenumber correction
    """

    def __init__(self, reference_peaks: Dict[str, float] = None, poly_order: int = 3):
        """
        Initialize wavenumber calibration.

        Args:
            reference_peaks (dict): Reference peak positions {name: wavenumber}
                                  Default: {"Si": 520.5} (Silicon peak)
            poly_order (int): Order of polynomial for correction (default: 3)
        """
        if reference_peaks is None:
            self.reference_peaks = {"Si": 520.5}
        else:
            self.reference_peaks = reference_peaks

        if not isinstance(poly_order, int) or poly_order < 1:
            raise ValueError("Polynomial order must be a positive integer")

        self.poly_order = poly_order

    def calibrate(
        self, measured_peaks: Dict[str, float], wavenumbers: np.ndarray
    ) -> np.ndarray:
        """
        Calibrate wavenumber axis using measured vs reference peak positions.

        Args:
            measured_peaks (dict): Measured peak positions {name: wavenumber}
            wavenumbers (np.ndarray): Original wavenumber axis

        Returns:
            np.ndarray: Corrected wavenumber axis
        """
        # Extract common peaks between reference and measured
        common_peaks = set(self.reference_peaks.keys()) & set(measured_peaks.keys())

        if len(common_peaks) == 0:
            raise ValueError(
                "No common peaks found between reference and measured peaks"
            )

        if len(common_peaks) < self.poly_order + 1:
            # Reduce polynomial order if insufficient peaks
            self.poly_order = max(1, len(common_peaks) - 1)

        # Prepare data for polynomial fitting
        measured_values = np.array([measured_peaks[peak] for peak in common_peaks])
        reference_values = np.array(
            [self.reference_peaks[peak] for peak in common_peaks]
        )

        # Fit polynomial: reference = poly(measured)
        poly_coeffs = np.polyfit(measured_values, reference_values, self.poly_order)

        # Apply correction to entire wavenumber axis
        corrected_wavenumbers = np.polyval(poly_coeffs, wavenumbers)

        return corrected_wavenumbers

    def __call__(
        self, wavenumbers: np.ndarray, measured_peaks: Dict[str, float]
    ) -> np.ndarray:
        """Make class callable for pipeline compatibility."""
        return self.calibrate(measured_peaks, wavenumbers)


class IntensityCalibration:
    """
    A callable class to perform intensity calibration on Raman spectra.

    This process corrects for the instrument's wavelength-dependent response
    by using a reference standard with a known spectral emission profile.

    Attributes:
        reference (np.ndarray): The "ground truth" emission profile of the
                                calibration standard (e.g., a NIST lamp).
    """

    def __init__(self, reference: np.ndarray = None):
        """
        Initialize intensity calibration.

        Args:
            reference (np.ndarray): Reference spectrum for calibration
        """
        self.reference = reference

    def _calculate_correction_factor(self, measured_standard: np.ndarray):
        """Calculate the correction factor for intensity calibration."""
        if self.reference is None:
            raise ValueError(
                "Reference spectrum must be provided for intensity calibration"
            )

        if len(self.reference) != len(measured_standard):
            raise ValueError("Reference and measured spectra must have the same length")

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            correction_factor = self.reference / measured_standard
            # Replace inf and nan with 1 (no correction)
            correction_factor = np.where(
                np.isfinite(correction_factor), correction_factor, 1.0
            )

        return correction_factor

    def __call__(
        self, spectra: np.ndarray, measured_standard: np.ndarray
    ) -> np.ndarray:
        """Apply intensity calibration to numpy array format."""
        correction_factor = self._calculate_correction_factor(measured_standard)

        if spectra.ndim == 1:
            return spectra * correction_factor
        elif spectra.ndim == 2:
            # Apply correction to each spectrum
            return spectra * correction_factor[np.newaxis, :]
        else:
            raise ValueError("Spectra must be 1D or 2D array")

    def apply(
        self, spectra: "rp.SpectralContainer", measured_standard: np.ndarray
    ) -> "rp.SpectralContainer":
        """Apply intensity calibration to ramanspy SpectralContainer."""
        if not RAMANSPY_AVAILABLE:
            raise ImportError("ramanspy is required for SpectralContainer operations")

        correction_factor = self._calculate_correction_factor(measured_standard)

        # Apply correction to spectral data
        corrected_data = spectra.spectral_data * correction_factor

        return rp.SpectralContainer(corrected_data, spectra.spectral_axis)
