"""
Advanced Baseline Correction Methods for Raman Spectra

This module contains advanced baseline correction techniques including
digital high-pass filtering for fluorescence baseline removal.

Methods:
- Butterworth High-Pass: Smooth phase response baseline removal
"""

import numpy as np
from typing import Literal, Optional

try:
    import ramanspy as rp

    RAMANSPY_AVAILABLE = True
except ImportError:
    RAMANSPY_AVAILABLE = False

try:
    from scipy.signal import butter, filtfilt

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from ..utils import create_logs
except ImportError:
    # Fallback logging function
    def create_logs(log_id, source, message, status="info"):
        print(f"[{status.upper()}] {source}: {message}")


class ButterworthHighPass:
    """
    Digital Butterworth High-Pass Filter for Baseline Removal

    Removes low-frequency fluorescence baseline with minimal passband ripple
    and smooth phase response. Uses IIR filtering with zero-phase implementation.

    Mathematical Definition:
    - Continuous-time transfer function (high-pass):
      H(ω) = 1 / √(1 + (ωc/ω)^(2n))
      where n is order and ωc is cutoff frequency

    - Discrete implementation:
      1. Use bilinear transform: s = (2/T) * (1 - z⁻¹) / (1 + z⁻¹)
      2. Compute IIR filter coefficients (b, a) = butter(n, fc, btype='highpass')
      3. Apply zero-phase filtering: x' = filtfilt(b, a, x)

    Parameter Selection:
    - Cutoff frequency fc ∈ (0, 0.5) relative to sampling rate in wavenumber index
    - Select fc low enough to preserve peak envelopes
    - Validate by checking peak area preservation after filtering
    - Typical order n = 2-4 for smooth response

    Why this helps:
    - Removes fluorescence baseline without peak distortion
    - Smooth frequency response (no ripple like Chebyshev)
    - Zero-phase filtering preserves peak positions
    - Computationally efficient (IIR filter)

    References:
    - Signal Processing for Spectroscopy (2023)
    - Butterworth filter design for baseline correction
    - ScienceDirect: Digital filtering in chromatography
    """

    def __init__(
        self,
        cutoff_freq: float = 0.01,
        filter_order: int = 3,
        validate_peaks: bool = True,
    ):
        """
        Initialize Butterworth High-Pass Filter.

        Args:
            cutoff_freq: Normalized cutoff frequency (0 < fc < 0.5)
                        Relative to wavenumber sampling rate
                        Lower values = remove slower baseline variations
            filter_order: Filter order (1-6 typical, higher = steeper rolloff)
            validate_peaks: If True, warn if peak areas change significantly
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for Butterworth filtering")

        if not (0 < cutoff_freq < 0.5):
            raise ValueError("Cutoff frequency must be in range (0, 0.5)")

        if not (1 <= filter_order <= 10):
            raise ValueError("Filter order must be between 1 and 10")

        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order
        self.validate_peaks = validate_peaks

        # Design filter coefficients
        self.b, self.a = butter(
            self.filter_order, self.cutoff_freq, btype="highpass", analog=False
        )

        create_logs(
            "butterworth_init",
            "ButterworthHighPass",
            f"Initialized Butterworth high-pass: order={filter_order}, fc={cutoff_freq}",
            status="info",
        )

    def __call__(self, spectra: np.ndarray) -> np.ndarray:
        """Apply filtering to numpy array format."""
        if spectra.ndim == 1:
            return self._filter_spectrum(spectra)
        elif spectra.ndim == 2:
            return np.array([self._filter_spectrum(s) for s in spectra])
        else:
            raise ValueError("Spectra must be 1D or 2D array")

    def apply(self, spectra: "rp.SpectralContainer") -> "rp.SpectralContainer":
        """Apply filtering to ramanspy SpectralContainer."""
        if not RAMANSPY_AVAILABLE:
            raise ImportError("ramanspy required for SpectralContainer operations")

        data = spectra.spectral_data

        if data.ndim == 1:
            filtered_data = self._filter_spectrum(data)
        else:
            filtered_data = np.array([self._filter_spectrum(s) for s in data])

        return rp.SpectralContainer(filtered_data, spectra.spectral_axis)

    def _filter_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Apply Butterworth high-pass filter to single spectrum.

        Uses filtfilt for zero-phase filtering to preserve peak positions.
        """
        if len(spectrum) < 3 * max(len(self.a), len(self.b)):
            create_logs(
                "butterworth_warning",
                "ButterworthHighPass",
                f"Spectrum too short ({len(spectrum)} points) for filter. Using as-is.",
                status="warning",
            )
            return spectrum

        try:
            # Apply zero-phase filtering
            filtered = filtfilt(self.b, self.a, spectrum)

            # Validate peak preservation if requested
            if self.validate_peaks:
                self._validate_peak_preservation(spectrum, filtered)

            return filtered

        except Exception as e:
            create_logs(
                "butterworth_error",
                "ButterworthHighPass",
                f"Filtering failed: {e}. Returning original spectrum.",
                status="error",
            )
            return spectrum

    def _validate_peak_preservation(
        self, original: np.ndarray, filtered: np.ndarray, tolerance: float = 0.2
    ):
        """
        Validate that peak areas are preserved after filtering.

        Warns if peak areas change by more than tolerance (20% default).
        """
        # Simple validation: check if total absolute area is preserved
        original_area = np.sum(np.abs(original))
        filtered_area = np.sum(np.abs(filtered))

        if original_area > 0:
            area_change = abs(filtered_area - original_area) / original_area

            if area_change > tolerance:
                create_logs(
                    "peak_preservation_warning",
                    "ButterworthHighPass",
                    f"Peak area changed by {area_change*100:.1f}%. "
                    + f"Consider lowering cutoff frequency (current: {self.cutoff_freq})",
                    status="warning",
                )

    def get_frequency_response(self, n_points: int = 1000) -> tuple:
        """
        Get frequency response of the filter for visualization.

        Args:
            n_points: Number of frequency points

        Returns:
            (frequencies, magnitude_db): Frequency response
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for frequency response")

        from scipy.signal import freqz

        # Compute frequency response
        w, h = freqz(self.b, self.a, worN=n_points)

        # Convert to dB
        magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)

        # Normalize frequency to [0, 0.5]
        frequencies = w / (2 * np.pi)

        return frequencies, magnitude_db

    @classmethod
    def auto_cutoff(
        cls,
        spectrum: np.ndarray,
        wavenumbers: Optional[np.ndarray] = None,
        baseline_width_estimate: float = 100.0,
        filter_order: int = 3,
    ) -> "ButterworthHighPass":
        """
        Automatically select cutoff frequency based on estimated baseline width.

        Args:
            spectrum: Example spectrum for analysis
            wavenumbers: Wavenumber axis (if None, uses indices)
            baseline_width_estimate: Estimated width of baseline features (cm⁻¹)
            filter_order: Filter order to use

        Returns:
            Configured ButterworthHighPass instance
        """
        if wavenumbers is not None:
            # Estimate sampling rate in wavenumber space
            wn_step = np.median(np.diff(wavenumbers))

            # Convert baseline width to normalized frequency
            # fc = (wn_step / baseline_width) / 2
            cutoff_freq = (wn_step / baseline_width_estimate) / 2
        else:
            # Use default if no wavenumber axis
            cutoff_freq = 0.01

        # Clamp to valid range
        cutoff_freq = np.clip(cutoff_freq, 0.001, 0.4)

        create_logs(
            "butterworth_auto",
            "ButterworthHighPass",
            f"Auto-selected cutoff frequency: {cutoff_freq:.4f} "
            + f"(baseline width: {baseline_width_estimate} cm⁻¹)",
            status="info",
        )

        return cls(cutoff_freq=cutoff_freq, filter_order=filter_order)


# Optional: Import statement
from typing import Optional
