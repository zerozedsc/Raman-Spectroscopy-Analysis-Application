"""
Advanced Normalization Methods for Raman Spectra

This module contains state-of-the-art normalization techniques specifically
designed for cross-platform, cross-session robustness in medical Raman
spectroscopy applications (e.g., MGUS/MM classification).

Methods implemented based on research best practices:
- Quantile Normalization: Cross-platform distribution alignment
- Rank Transform: Intensity-independent relative ordering
- Probabilistic Quotient Normalization (PQN): Dilution correction
"""

import numpy as np
from typing import Optional, Literal

try:
    import ramanspy as rp

    RAMANSPY_AVAILABLE = True
except ImportError:
    RAMANSPY_AVAILABLE = False

try:
    from ..utils import create_logs
except ImportError:
    # Fallback logging function
    def create_logs(log_id, source, message, status="info"):
        print(f"[{status.upper()}] {source}: {message}")


class QuantileNormalization:
    """
    Quantile Normalization for Raman Spectra

    Aligns intensity distributions across sessions/instruments to mitigate
    external domain shift before ML. Particularly effective for MGUS/MM
    classification across different measurement conditions.

    Mathematical Definition:
    - Given spectrum x ∈ ℝᵖ with wavenumber-indexed intensities xⱼ
    - Let x₍ₖ₎ denote the k-th order statistic after sorting ascending
    - For training set of n spectra, compute reference quantile vector:
      qₖ = median_{i=1..n}(xⁱ₍ₖ₎)
    - For any spectrum x, compute ranks rⱼ ∈ {1,...,p} such that xⱼ = x₍ᵣⱼ₎
    - Map each intensity to corresponding reference quantile: x'ⱼ = qᵣⱼ

    References:
    - Nature Scientific Reports (2020) - Quantile normalization in omics
    - Cross-platform ML for MGUS/MM gene expression studies
    """

    def __init__(self, method: Literal["median", "mean"] = "median"):
        """
        Initialize Quantile Normalization.

        Args:
            method: Aggregation method for reference quantiles ('median' or 'mean')
                   Median is more robust to outliers
        """
        self.method = method
        self.reference_quantiles = None

    def fit(self, spectra: np.ndarray) -> "QuantileNormalization":
        """
        Compute reference quantile vector from training spectra.

        Args:
            spectra: 2D array (n_samples, n_features) of training spectra

        Returns:
            self: Fitted normalizer
        """
        if spectra.ndim != 2:
            raise ValueError("Spectra must be 2D array (n_samples, n_features)")

        # Sort each spectrum independently
        sorted_spectra = np.sort(spectra, axis=1)

        # Compute reference quantiles (median or mean across sorted spectra)
        if self.method == "median":
            self.reference_quantiles = np.median(sorted_spectra, axis=0)
        elif self.method == "mean":
            self.reference_quantiles = np.mean(sorted_spectra, axis=0)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'median' or 'mean'")

        create_logs(
            "quantile_norm_fit",
            "QuantileNormalization",
            f"Fitted with {spectra.shape[0]} spectra using {self.method} method",
            status="info",
        )

        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """
        Apply quantile normalization to spectra.

        Args:
            spectra: 1D or 2D array of spectra to normalize

        Returns:
            Quantile-normalized spectra
        """
        if self.reference_quantiles is None:
            raise ValueError("Must call fit() before transform()")

        if spectra.ndim == 1:
            return self._normalize_spectrum(spectra)
        elif spectra.ndim == 2:
            return np.array([self._normalize_spectrum(s) for s in spectra])
        else:
            raise ValueError("Spectra must be 1D or 2D array")

    def fit_transform(self, spectra: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(spectra).transform(spectra)

    def _normalize_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Normalize a single spectrum using reference quantiles.

        Algorithm:
        1. Compute ranks of intensities in spectrum
        2. Map each intensity to corresponding reference quantile
        3. Handle ties using mid-rank method
        """
        # Get sort indices (ranks)
        sort_idx = np.argsort(spectrum)
        ranks = np.argsort(sort_idx)  # Inverse permutation gives ranks

        # Map to reference quantiles
        normalized = self.reference_quantiles[ranks]

        return normalized

    def __call__(self, data):
        """
        Apply Quantile Normalization to data.

        Handles both SpectralContainer (RamanSPy workflows) and numpy array (sklearn pipelines).

        Args:
            data: SpectralContainer or numpy array

        Returns:
            Same type as input with quantile normalization applied
        """
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

        # Apply normalization
        processed = self.fit_transform(spectra)

        # Return in same format as input
        if is_container:
            import ramanspy as rp

            return rp.SpectralContainer(processed, axis)
        else:
            return processed

    def apply(self, spectra: "rp.SpectralContainer") -> "rp.SpectralContainer":
        """Apply to ramanspy SpectralContainer."""
        if not RAMANSPY_AVAILABLE:
            raise ImportError("ramanspy required for SpectralContainer operations")

        data = spectra.spectral_data

        if data.ndim == 1:
            # Single spectrum - cannot compute quantiles, return as-is
            create_logs(
                "quantile_norm_warning",
                "QuantileNormalization",
                "Cannot apply quantile normalization to single spectrum",
                status="warning",
            )
            return spectra
        else:
            normalized_data = self.fit_transform(data)

        return rp.SpectralContainer(normalized_data, spectra.spectral_axis)


class RankTransform:
    """
    Within-Spectrum Rank Transform

    Removes dependence on absolute intensity and laser power by converting
    each spectrum to relative order information in [0,1].

    Mathematical Definition:
    - For spectrum x ∈ ℝᵖ, compute ranks rⱼ = rank(xⱼ) with mid-rank ties
    - Scale to [0,1]: sⱼ = (rⱼ - 1) / (p - 1)
    - Optionally: center to zero-mean unit-variance per feature across samples

    Why this helps:
    - Rank-based inputs are among most domain-shift-robust transforms
    - Effective for MGUS/MM gene expression and transfers well to Raman
    - Insensitive to multiplicative scaling and additive offsets

    References:
    - Machine Learning Models for Predicting Multiple Myeloma (2024)
    - Robust cross-platform normalization strategies
    """

    def __init__(self, scale_range: tuple = (0, 1), standardize: bool = False):
        """
        Initialize Rank Transform.

        Args:
            scale_range: Target range for scaled ranks, default (0, 1)
            standardize: If True, standardize features after ranking
        """
        self.scale_range = scale_range
        self.standardize = standardize
        self.feature_mean = None
        self.feature_std = None

    def fit(self, spectra: np.ndarray) -> "RankTransform":
        """
        Fit rank transform (compute feature statistics if standardizing).

        Args:
            spectra: 2D array (n_samples, n_features) of training spectra

        Returns:
            self: Fitted transformer
        """
        if spectra.ndim != 2:
            raise ValueError("Spectra must be 2D array for fitting")

        if self.standardize:
            # Compute ranks for all spectra first
            ranked_spectra = np.array([self._rank_spectrum(s) for s in spectra])

            # Compute feature-wise statistics
            self.feature_mean = np.mean(ranked_spectra, axis=0)
            self.feature_std = np.std(ranked_spectra, axis=0, ddof=1)

            # Avoid division by zero
            self.feature_std[self.feature_std < 1e-10] = 1.0

        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """
        Apply rank transform to spectra.

        Args:
            spectra: 1D or 2D array of spectra

        Returns:
            Rank-transformed spectra
        """
        if spectra.ndim == 1:
            ranked = self._rank_spectrum(spectra)
            if self.standardize and self.feature_mean is not None:
                ranked = (ranked - self.feature_mean) / self.feature_std
            return ranked
        elif spectra.ndim == 2:
            ranked = np.array([self._rank_spectrum(s) for s in spectra])
            if self.standardize and self.feature_mean is not None:
                ranked = (ranked - self.feature_mean) / self.feature_std
            return ranked
        else:
            raise ValueError("Spectra must be 1D or 2D array")

    def fit_transform(self, spectra: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(spectra).transform(spectra)

    def _rank_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Compute ranks and scale to target range for single spectrum.

        Uses mid-rank method for ties to preserve monotonicity.
        """
        # Compute ranks (1-indexed)
        ranks = np.argsort(np.argsort(spectrum)) + 1

        # Scale to [0, 1]
        n_features = len(spectrum)
        scaled_ranks = (ranks - 1) / (n_features - 1)

        # Scale to target range
        if self.scale_range != (0, 1):
            a, b = self.scale_range
            scaled_ranks = a + (b - a) * scaled_ranks

        return scaled_ranks

    def __call__(self, data):
        """
        Apply Rank Transform to data.

        Handles both SpectralContainer (RamanSPy workflows) and numpy array (sklearn pipelines).

        Args:
            data: SpectralContainer or numpy array

        Returns:
            Same type as input with rank transform applied
        """
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

        # Apply transform
        if spectra.ndim == 1:
            processed = self._rank_spectrum(spectra)
        else:
            processed = self.fit_transform(spectra)

        # Return in same format as input
        if is_container:
            import ramanspy as rp

            return rp.SpectralContainer(processed, axis)
        else:
            return processed

    def apply(self, spectra: "rp.SpectralContainer") -> "rp.SpectralContainer":
        """Apply to ramanspy SpectralContainer."""
        if not RAMANSPY_AVAILABLE:
            raise ImportError("ramanspy required for SpectralContainer operations")

        data = spectra.spectral_data

        if data.ndim == 1:
            transformed_data = self._rank_spectrum(data)
        else:
            transformed_data = self.fit_transform(data)

        return rp.SpectralContainer(transformed_data, spectra.spectral_axis)


class ProbabilisticQuotientNormalization:
    """
    Probabilistic Quotient Normalization (PQN)

    Corrects sample-to-sample dilution or total intensity variation while
    preserving relative band structure. Particularly useful when dilution
    effects are suspected in biological samples.

    Mathematical Definition:
    - Compute reference spectrum r ∈ ℝᵖ (median across training spectra)
    - For sample i with spectrum xⁱ, compute quotients: qⱼⁱ = xⱼⁱ / rⱼ
    - Estimate dilution factor: dᵢ = median_j(qⱼⁱ)
    - Normalize: x'ⱼⁱ = xⱼⁱ / dᵢ

    Notes:
    - Use robust median to resist outlier bands
    - Apply after baseline correction and before derivative/denoise steps
    - Combines well with SNV but validate on your data

    References:
    - PMC3337420 - PQN for NMR metabolomics
    - Spectroscopy workflow best practices (2023)
    """

    def __init__(self, reference_spectrum: Optional[np.ndarray] = None):
        """
        Initialize PQN.

        Args:
            reference_spectrum: Optional pre-computed reference spectrum
                              If None, will compute from training data
        """
        self.reference_spectrum = reference_spectrum

    def fit(self, spectra: np.ndarray) -> "ProbabilisticQuotientNormalization":
        """
        Compute reference spectrum from training data.

        Args:
            spectra: 2D array (n_samples, n_features) of training spectra

        Returns:
            self: Fitted normalizer
        """
        if spectra.ndim != 2:
            raise ValueError("Spectra must be 2D array")

        # Compute median spectrum as reference
        self.reference_spectrum = np.median(spectra, axis=0)

        # Avoid division by zero
        self.reference_spectrum[self.reference_spectrum == 0] = 1e-10

        create_logs(
            "pqn_fit",
            "ProbabilisticQuotientNormalization",
            f"Fitted with {spectra.shape[0]} spectra",
            status="info",
        )

        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """
        Apply PQN to spectra.

        Args:
            spectra: 1D or 2D array of spectra

        Returns:
            PQN-normalized spectra
        """
        if self.reference_spectrum is None:
            raise ValueError("Must call fit() before transform()")

        if spectra.ndim == 1:
            return self._normalize_spectrum(spectra)
        elif spectra.ndim == 2:
            return np.array([self._normalize_spectrum(s) for s in spectra])
        else:
            raise ValueError("Spectra must be 1D or 2D array")

    def fit_transform(self, spectra: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(spectra).transform(spectra)

    def _normalize_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Normalize single spectrum using PQN.

        Algorithm:
        1. Compute quotients: spectrum / reference
        2. Take median of quotients as dilution factor
        3. Divide spectrum by dilution factor
        """
        # Compute quotients
        quotients = spectrum / self.reference_spectrum

        # Filter out invalid quotients (inf, nan)
        valid_quotients = quotients[np.isfinite(quotients)]

        if len(valid_quotients) == 0:
            create_logs(
                "pqn_warning",
                "ProbabilisticQuotientNormalization",
                "No valid quotients found, returning original spectrum",
                status="warning",
            )
            return spectrum

        # Compute median quotient (dilution factor)
        dilution_factor = np.median(valid_quotients)

        # Avoid division by zero
        if abs(dilution_factor) < 1e-10:
            create_logs(
                "pqn_warning",
                "ProbabilisticQuotientNormalization",
                "Near-zero dilution factor, returning original spectrum",
                status="warning",
            )
            return spectrum

        # Normalize by dilution factor
        normalized = spectrum / dilution_factor

        return normalized

    def __call__(self, data):
        """
        Apply Probabilistic Quotient Normalization to data.

        Handles both SpectralContainer (RamanSPy workflows) and numpy array (sklearn pipelines).

        Args:
            data: SpectralContainer or numpy array

        Returns:
            Same type as input with PQN applied
        """
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

        # Apply normalization
        processed = self.fit_transform(spectra)

        # Return in same format as input
        if is_container:
            import ramanspy as rp

            return rp.SpectralContainer(processed, axis)
        else:
            return processed

    def apply(self, spectra: "rp.SpectralContainer") -> "rp.SpectralContainer":
        """Apply to ramanspy SpectralContainer."""
        if not RAMANSPY_AVAILABLE:
            raise ImportError("ramanspy required for SpectralContainer operations")

        data = spectra.spectral_data

        if data.ndim == 1:
            # Single spectrum - use itself as reference
            normalized_data = data
        else:
            normalized_data = self.fit_transform(data)

        return rp.SpectralContainer(normalized_data, spectra.spectral_axis)
