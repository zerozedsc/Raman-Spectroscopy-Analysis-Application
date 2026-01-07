"""
Feature Engineering Methods for Raman Spectra

This module contains advanced feature engineering techniques for creating
dimensionless, batch-invariant descriptors from Raman spectra, particularly
useful for medical applications (e.g., MGUS/MM classification).

Methods:
- Peak-Ratio Features: Dimensionless ratios robust to illumination and exposure
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Literal

try:
    import ramanspy as rp

    RAMANSPY_AVAILABLE = True
except ImportError:
    RAMANSPY_AVAILABLE = False

try:
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from ..utils import create_logs
except ImportError:
    # Fallback logging function
    def create_logs(log_id, source, message, status="info"):
        print(f"[{status.upper()}] {source}: {message}")


class PeakRatioFeatures:
    """
    Peak-Ratio Feature Engineering for Raman Spectra

    Creates dimensionless, batch-invariant descriptors that emphasize
    biochemical changes critical for disease separation (e.g., MGUS/MM).

    Purpose:
    - Create dimensionless features robust to global scaling
    - Cancel illumination and exposure time effects
    - Improve external generalization across instruments

    Peak Selection Guidelines:
    - Include literature-reported disease bands (e.g., 1149, 1527-1530 cm⁻¹ for MGUS/MM)
    - Add protein/DNA benchmarks (1004, 1090, 1250, 1445, 1660 cm⁻¹)
    - Select stable peaks with low across-train variance

    Intensity Extraction Methods:
    1. Local max: I(νₖ) = max_{ν ∈ [νₖ-Δ, νₖ+Δ]} x(ν)
    2. Local integral: I(νₖ) = ∫_{νₖ-Δ}^{νₖ+Δ} x(ν) dν
    3. Gaussian fit: I(ν) = A exp(-(ν-μ)²/(2σ²)), use fitted area

    Ratios:
    - Simple ratio: R_{a,b} = I(νₐ) / (I(νᵦ) + ε)
    - Log-ratio: ρ_{a,b} = log(I(νₐ)) - log(I(νᵦ)) (stabilizes variance)

    References:
    - Journal of Food Science (2022) - Peak ratio analysis
    - MGUS vs MM Raman biomarkers (2024)
    - Robust feature engineering for medical ML
    """

    def __init__(
        self,
        peak_positions: Optional[Dict[str, float]] = None,
        window_size: float = 10.0,
        extraction_method: Literal[
            "local_max", "local_integral", "gaussian_fit"
        ] = "local_max",
        ratio_mode: Literal["simple", "log", "both"] = "log",
        epsilon: float = 1e-10,
    ):
        """
        Initialize Peak-Ratio Feature Engineering.

        Args:
            peak_positions: Dictionary of peak names and their wavenumber positions
                          e.g., {'DNA_backbone': 1004, 'protein_amide_III': 1250}
                          If None, uses default MGUS/MM biomarker peaks
            window_size: Half-width of window around peak for extraction (cm⁻¹)
            extraction_method: Method for extracting peak intensity
            ratio_mode: Type of ratios to compute ('simple', 'log', or 'both')
            epsilon: Small constant to avoid division by zero
        """
        self.peak_positions = peak_positions or self._get_default_peaks()
        self.window_size = window_size
        self.extraction_method = extraction_method
        self.ratio_mode = ratio_mode
        self.epsilon = epsilon
        self.wavenumbers = None

        # Store feature names for ML pipelines
        self.feature_names_ = []

    def _get_default_peaks(self) -> Dict[str, float]:
        """
        Get default peak positions for MGUS/MM classification.

        Based on literature-reported biomarkers and protein/DNA reference peaks.
        """
        return {
            "DNA_backbone": 1004.0,  # DNA reference
            "DNA_PO2_symm": 1090.0,  # DNA/RNA phosphate
            "MGUS_biomarker": 1149.0,  # Critical MGUS marker
            "protein_amide_III": 1250.0,  # Protein structure
            "lipid_CH2": 1445.0,  # Lipid content
            "MM_biomarker_1": 1527.0,  # MM marker (lower)
            "MM_biomarker_2": 1530.0,  # MM marker (upper)
            "protein_amide_I": 1660.0,  # Protein secondary structure
        }

    def fit(self, spectra: np.ndarray, wavenumbers: np.ndarray) -> "PeakRatioFeatures":
        """
        Fit peak ratio extractor (store wavenumber axis).

        Args:
            spectra: 2D array (n_samples, n_features) of training spectra
            wavenumbers: 1D array of wavenumber values

        Returns:
            self: Fitted transformer
        """
        self.wavenumbers = wavenumbers

        # Generate feature names
        peak_names = list(self.peak_positions.keys())
        self.feature_names_ = []

        for i, peak_a in enumerate(peak_names):
            for peak_b in peak_names[i + 1 :]:
                if self.ratio_mode in ["simple", "both"]:
                    self.feature_names_.append(f"ratio_{peak_a}/{peak_b}")
                if self.ratio_mode in ["log", "both"]:
                    self.feature_names_.append(f"log_ratio_{peak_a}/{peak_b}")

        create_logs(
            "peak_ratio_fit",
            "PeakRatioFeatures",
            f"Fitted with {len(peak_names)} peaks, generating {len(self.feature_names_)} features",
            status="info",
        )

        return self

    def transform(
        self, spectra: np.ndarray, wavenumbers: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract peak ratio features from spectra.

        Args:
            spectra: 1D or 2D array of spectra
            wavenumbers: Optional wavenumber axis (uses fitted if not provided)

        Returns:
            2D array (n_samples, n_features) of peak ratio features
        """
        if wavenumbers is not None:
            wn = wavenumbers
        elif self.wavenumbers is not None:
            wn = self.wavenumbers
        else:
            raise ValueError("Wavenumber axis not provided and not fitted")

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)

        n_samples = spectra.shape[0]
        n_features = len(self.feature_names_)
        features = np.zeros((n_samples, n_features))

        for i, spectrum in enumerate(spectra):
            features[i] = self._extract_ratios(spectrum, wn)

        return features

    def fit_transform(self, spectra: np.ndarray, wavenumbers: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(spectra, wavenumbers).transform(spectra, wavenumbers)

    def _extract_ratios(
        self, spectrum: np.ndarray, wavenumbers: np.ndarray
    ) -> np.ndarray:
        """
        Extract all pairwise peak ratios for a single spectrum.
        """
        # Extract intensities for all peaks
        peak_intensities = {}
        for peak_name, peak_position in self.peak_positions.items():
            intensity = self._extract_peak_intensity(
                spectrum, wavenumbers, peak_position
            )
            peak_intensities[peak_name] = intensity

        # Compute pairwise ratios
        ratios = []
        peak_names = list(self.peak_positions.keys())

        for i, peak_a in enumerate(peak_names):
            for peak_b in peak_names[i + 1 :]:
                I_a = peak_intensities[peak_a]
                I_b = peak_intensities[peak_b]

                if self.ratio_mode in ["simple", "both"]:
                    # Simple ratio with epsilon for stability
                    ratio = I_a / (I_b + self.epsilon)
                    ratios.append(ratio)

                if self.ratio_mode in ["log", "both"]:
                    # Log-ratio (symmetric and variance-stabilizing)
                    log_ratio = np.log(I_a + self.epsilon) - np.log(I_b + self.epsilon)
                    ratios.append(log_ratio)

        return np.array(ratios)

    def _extract_peak_intensity(
        self, spectrum: np.ndarray, wavenumbers: np.ndarray, peak_position: float
    ) -> float:
        """
        Extract intensity at specific peak position using selected method.
        """
        # Find window around peak
        mask = (wavenumbers >= peak_position - self.window_size) & (
            wavenumbers <= peak_position + self.window_size
        )

        if not np.any(mask):
            create_logs(
                "peak_extraction_warning",
                "PeakRatioFeatures",
                f"No data points found near peak at {peak_position} cm⁻¹",
                status="warning",
            )
            return 0.0

        window_wn = wavenumbers[mask]
        window_spectrum = spectrum[mask]

        if self.extraction_method == "local_max":
            return self._local_max(window_spectrum)

        elif self.extraction_method == "local_integral":
            return self._local_integral(window_wn, window_spectrum)

        elif self.extraction_method == "gaussian_fit":
            return self._gaussian_fit(window_wn, window_spectrum, peak_position)

        else:
            raise ValueError(f"Unknown extraction method: {self.extraction_method}")

    def _local_max(self, window_spectrum: np.ndarray) -> float:
        """Extract local maximum intensity in window."""
        return np.max(window_spectrum)

    def _local_integral(
        self, window_wn: np.ndarray, window_spectrum: np.ndarray
    ) -> float:
        """
        Integrate intensity over window using trapezoidal rule.
        """
        return np.trapz(window_spectrum, window_wn)

    def _gaussian_fit(
        self, window_wn: np.ndarray, window_spectrum: np.ndarray, peak_position: float
    ) -> float:
        """
        Fit Gaussian peak and return area (A * σ * √(2π)).

        Gaussian model: I(ν) = A * exp(-(ν - μ)² / (2σ²))
        """
        if not SCIPY_AVAILABLE:
            # Fallback to local max
            return self._local_max(window_spectrum)

        def gaussian(x, amplitude, mean, std):
            return amplitude * np.exp(-((x - mean) ** 2) / (2 * std**2))

        # Initial guess
        p0 = [
            np.max(window_spectrum),  # amplitude
            peak_position,  # mean
            self.window_size / 3,  # std (assume ~3σ covers window)
        ]

        try:
            # Fit Gaussian
            popt, _ = curve_fit(
                gaussian,
                window_wn,
                window_spectrum,
                p0=p0,
                bounds=(
                    [0, window_wn[0], 0.1],
                    [np.inf, window_wn[-1], self.window_size],
                ),
            )

            amplitude, mean, std = popt

            # Return area: A * σ * √(2π)
            area = amplitude * std * np.sqrt(2 * np.pi)
            return area

        except Exception as e:
            create_logs(
                "gaussian_fit_warning",
                "PeakRatioFeatures",
                f"Gaussian fit failed: {e}. Using local max.",
                status="warning",
            )
            return self._local_max(window_spectrum)

    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        return self.feature_names_

    def __call__(self, spectra: np.ndarray, wavenumbers: np.ndarray) -> np.ndarray:
        """Make callable for pipeline compatibility."""
        if self.wavenumbers is None:
            return self.fit_transform(spectra, wavenumbers)
        else:
            return self.transform(spectra, wavenumbers)

    def apply(self, data) -> np.ndarray:
        """
        Apply peak ratio feature extraction to RamanSPy SpectralContainer.

        Args:
            data: SpectralContainer with spectral_data and spectral_axis

        Returns:
            np.ndarray: Peak ratio features (n_samples, n_features)
        """
        # Detect input type
        is_container = hasattr(data, "spectral_data")

        if is_container:
            spectra = data.spectral_data
            wavenumbers = data.spectral_axis
        else:
            # Assume numpy array with wavenumber axis stored
            spectra = data
            wavenumbers = self.wavenumbers
            if wavenumbers is None:
                raise ValueError(
                    "Wavenumber axis not provided. Use fit() first or pass SpectralContainer."
                )

        return self.__call__(spectra, wavenumbers)
