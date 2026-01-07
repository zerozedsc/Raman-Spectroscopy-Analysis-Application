from utils import *
from typing import Any, Callable, Dict, List, Optional, Tuple
from configs.configs import *

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    from scipy import ndimage
    from scipy.ndimage import grey_opening

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    create_logs(
        "torch_import_error",
        "RamanPipeline",
        "PyTorch is not available. Some advanced preprocessing steps will be skipped.",
        status="warning",
    )

import numpy as np
import pandas as pd
import ramanspy as rp
import matplotlib.pyplot as plt
import pickle as pkl
import os
import traceback


# SPIKE REMOVAL
class Gaussian:
    """
    A callable class to remove cosmic ray spikes from Raman spectra using a
    Gaussian filter-based approach.

    This method works by smoothing the spectrum with a Gaussian filter and
    identifying points that deviate significantly from the smoothed version.
    The deviation is measured against the noise level, estimated using the
    Median Absolute Deviation (MAD).

    Attributes:
        kernel (int): The standard deviation (sigma) for the Gaussian kernel.
                      This controls the degree of smoothing. Must be an odd integer.
        threshold (float): The number of standard deviations a point must be
                           from the smoothed spectrum to be considered a spike.
    """

    def __init__(self, kernel: int = 5, threshold: float = 3.0):
        """
        Initializes the Gaussian despike processor.

        Args:
            kernel (int): The size of the Gaussian kernel. It's used as the
                          standard deviation (sigma) for the filter.
                          A larger kernel results in more smoothing.
                          Defaults to 5.
            threshold (float): The modified Z-score threshold for spike detection.
                               Defaults to 3.0.
        """
        if not isinstance(kernel, int) or kernel <= 0:
            raise ValueError("Kernel must be a positive integer.")
        if not isinstance(threshold, (float, int)) or threshold <= 0:
            raise ValueError("Threshold must be a positive number.")

        self.kernel = kernel
        self.threshold = threshold

    def __call__(self, spectra: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian despiking to a 2D numpy array of spectra.
        This makes the class compatible with scikit-learn pipelines.

        Args:
            spectra (np.ndarray): A 2D numpy array where each row is a spectrum.

        Returns:
            np.ndarray: The despiked spectra.
        """
        if spectra.ndim != 2:
            raise ValueError(
                "Input for pipeline must be a 2D numpy array (n_samples, n_features)."
            )

        # Apply the despiking function along each row (each spectrum)
        return np.apply_along_axis(self._despike_spectrum, 1, spectra)

    def apply(self, spectra: rp.SpectralContainer) -> rp.SpectralContainer:
        """
        Apply Gaussian despiking to a ramanspy SpectralContainer.

        Args:
            spectra (SpectralContainer): The ramanspy container holding the spectral data.

        Returns:
            SpectralContainer: A new container with the despiked data.
        """
        data = spectra.spectral_data

        # Handle both single (1D) and multiple (2D) spectra
        if data.ndim == 1:
            despiked_data = self._despike_spectrum(data)
        else:
            despiked_data = np.array([self._despike_spectrum(s) for s in data])

        return SpectralContainer(despiked_data, spectra.spectral_axis)

    def _despike_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Core logic to remove spikes from a single 1D spectrum.

        Args:
            spectrum (np.ndarray): A 1D numpy array representing a single spectrum.

        Returns:
            np.ndarray: The despiked spectrum.
        """
        if spectrum is None or spectrum.size == 0:
            create_logs(
                "despike_empty_spectrum",
                "RamanPipeline",
                "Received an empty or None spectrum for despiking.",
                status="warning",
            )
            return spectrum

        # Create a copy to avoid modifying the original data
        despiked = np.copy(spectrum)

        # 1. Smooth the spectrum using a Gaussian filter
        # The `sigma` of the filter is controlled by our `kernel` parameter.
        smoothed_spectrum = gaussian_filter1d(despiked, sigma=self.kernel)

        # 2. Calculate the difference (residual)
        residual = despiked - smoothed_spectrum

        # 3. Estimate noise using Median Absolute Deviation (MAD) for robustness
        # The factor 0.6745 makes MAD an unbiased estimator for the standard deviation
        # for normally distributed data.
        mad = np.median(np.abs(residual - np.median(residual)))
        if mad < 1e-9:  # Avoid division by zero for flat residuals
            return despiked  # No spikes to correct

        # 4. Calculate the modified Z-score for each point
        # This score tells us how many standard deviations away each point is.
        modified_z_score = 0.6745 * residual / mad

        # 5. Identify spike locations
        # Spikes are points where the absolute Z-score exceeds the threshold.
        spike_indices = np.where(np.abs(modified_z_score) > self.threshold)[0]

        # 6. Replace spikes with the value from the smoothed spectrum
        # This is a simple and effective way to correct the spike.
        for i in spike_indices:
            # Boundary checks to avoid index errors
            if i > 0 and i < len(despiked) - 1:
                # Replace spike with the average of its neighbors for smoother results
                despiked[i] = (despiked[i - 1] + despiked[i + 1]) / 2
            elif i == 0 and len(despiked) > 1:
                despiked[i] = despiked[i + 1]  # Use next point if spike is at the start
            elif i == len(despiked) - 1 and len(despiked) > 1:
                despiked[i] = despiked[
                    i - 1
                ]  # Use previous point if spike is at the end

        return despiked


# NORMALIZATION METHODS
class SNV:
    """Standard Normal Variate (SNV) normalization for Raman spectra."""

    def __call__(self, spectra):
        """Apply SNV to numpy array format (for sklearn pipelines)."""
        # spectra: 2D numpy array (n_samples, n_features)
        return np.apply_along_axis(self.snv_normalisation, 1, spectra)

    def apply(self, spectra):
        """Apply SNV to ramanspy SpectralContainer format."""
        data = spectra.spectral_data
        # Handle both 1D and 2D data
        if data.ndim == 1:
            # Single spectrum
            snv_data = self.snv_normalisation(data)
        else:
            # Multiple spectra
            snv_data = np.array([self.snv_normalisation(spectrum) for spectrum in data])

        return rp.SpectralContainer(snv_data, spectra.spectral_axis)

    def snv_normalisation(self, spectrum):
        """
        Apply Standard Normal Variate normalization to a single spectrum.

        Args:
            spectrum: 1D numpy array representing a single spectrum

        Returns:
            Normalized spectrum using SNV: (spectrum - mean) / std
        """
        spectrum = np.asarray(spectrum)
        mean_val = np.mean(spectrum)
        std_val = np.std(spectrum, ddof=1)  # Use sample standard deviation

        # Enhanced error handling
        if std_val == 0:
            create_logs(
                "snv_zero_std",
                "RamanPipeline",
                "Standard deviation is zero. Returning original spectrum.",
                status="warning",
            )
            return spectrum - mean_val
        elif np.isnan(std_val) or np.isinf(std_val):
            create_logs(
                "snv_invalid_std",
                "RamanPipeline",
                "Standard deviation is NaN or Inf. Returning original spectrum.",
                status="warning",
            )
            return spectrum
        elif std_val < 1e-10:  # Very small std
            create_logs(
                "snv_small_std",
                "RamanPipeline",
                "Standard deviation is too small. Returning original spectrum.",
                status="warning",
            )
            return spectrum - mean_val

        return (spectrum - mean_val) / std_val


class MovingAverage:
    """Moving Average smoothing for Raman spectra."""

    def __init__(self, window_length=15):
        self.window_length = window_length

    def __call__(self, spectra):
        return self.apply(spectra)

    def apply(self, spectra):
        # spectra: rp.SpectralContainer
        smoothed_data = []
        for spectrum in spectra.spectral_data:
            smoothed = np.convolve(
                spectrum, np.ones(self.window_length) / self.window_length, mode="same"
            )
            smoothed_data.append(smoothed)
        return spectra.__class__(np.array(smoothed_data), spectra.spectral_axis)


# BASELINE CORRECTION METHODS
class BaselineCorrection:
    """
    A comprehensive baseline correction class for Raman spectroscopy.

    Provides easy access to all available baseline correction methods with
    pre-configured parameter sets for different use cases.
    """

    def __init__(self, region: Optional[Tuple[int, int]] = None):
        """
        Initialize BaselineCorrection class.

        Parameters
        ----------
        region : Optional[Tuple[int, int]]
            Wavenumber region for polynomial methods that require it.
        """
        self.region = region
        self._methods_registry = self._build_methods_registry()
        self._presets = self._build_presets()

    def _build_methods_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build registry of all available baseline correction methods with their parameters."""
        return {
            # =================== LEAST SQUARES METHODS ===================
            "ASLS": {
                "class": rp.preprocessing.baseline.ASLS,
                "default_params": {
                    "lam": 1e6,
                    "p": 0.01,
                    "diff_order": 2,
                    "max_iter": 50,
                    "tol": 0.001,
                },
                "category": "least_squares",
                "description": "Asymmetric Least Squares - Good for biological samples",
                "suitable_for": ["biological", "fluorescence", "smooth_baseline"],
            },
            "IASLS": {
                "class": rp.preprocessing.baseline.IASLS,
                "default_params": {
                    "lam": 1e6,
                    "p": 0.01,
                    "lam_1": 1e-4,
                    "max_iter": 50,
                    "tol": 0.001,
                },
                "category": "least_squares",
                "description": "Improved Asymmetric Least Squares - Enhanced version of ASLS",
                "suitable_for": ["biological", "complex_baseline"],
            },
            "AIRPLS": {
                "class": rp.preprocessing.baseline.AIRPLS,
                "default_params": {
                    "lam": 1e6,
                    "diff_order": 2,
                    "max_iter": 50,
                    "tol": 0.001,
                },
                "category": "least_squares",
                "description": "Adaptive Iteratively Reweighted Penalized Least Squares - Very robust",
                "suitable_for": ["biological", "robust", "varying_baseline"],
            },
            "ARPLS": {
                "class": rp.preprocessing.baseline.ARPLS,
                "default_params": {
                    "lam": 1e5,
                    "diff_order": 2,
                    "max_iter": 50,
                    "tol": 0.001,
                },
                "category": "least_squares",
                "description": "Asymmetrically Reweighted Penalized Least Squares - Popular choice",
                "suitable_for": ["biological", "general_purpose", "fluorescence"],
            },
            "DRPLS": {
                "class": rp.preprocessing.baseline.DRPLS,
                "default_params": {
                    "lam": 1e5,
                    "eta": 0.5,
                    "max_iter": 50,
                    "tol": 0.001,
                },
                "category": "least_squares",
                "description": "Doubly Reweighted Penalized Least Squares - Advanced method",
                "suitable_for": ["complex_baseline", "advanced"],
            },
            "IARPLS": {
                "class": rp.preprocessing.baseline.IARPLS,
                "default_params": {
                    "lam": 1e5,
                    "diff_order": 2,
                    "max_iter": 50,
                    "tol": 0.001,
                },
                "category": "least_squares",
                "description": "Improved Asymmetrically Reweighted Penalized Least Squares",
                "suitable_for": ["biological", "robust", "improved_convergence"],
            },
            "ASPLS": {
                "class": rp.preprocessing.baseline.ASPLS,
                "default_params": {
                    "lam": 1e6,
                    "diff_order": 2,
                    "max_iter": 50,
                    "alpha": 0.95,
                },
                "category": "least_squares",
                "description": "Adaptive Smoothness Penalized Least Squares - Self-adaptive",
                "suitable_for": ["adaptive", "self_tuning", "biological"],
            },
            # =================== POLYNOMIAL METHODS ===================
            "Poly": {
                "class": rp.preprocessing.baseline.Poly,
                "default_params": {"poly_order": 2, "regions": None},
                "requires_region": True,
                "category": "polynomial",
                "description": "Simple polynomial fitting",
                "suitable_for": ["simple", "fast", "smooth_baseline"],
            },
            "ModPoly": {
                "class": rp.preprocessing.baseline.ModPoly,
                "default_params": {
                    "poly_order": 2,
                    "tol": 0.001,
                    "max_iter": 250,
                    "weights": None,
                    "use_original": False,
                    "mask_initial_peaks": False,
                },
                "category": "polynomial",
                "description": "Modified polynomial - Iteratively excludes peaks",
                "suitable_for": ["peak_exclusion", "iterative"],
            },
            "PenalisedPoly": {
                "class": rp.preprocessing.baseline.PenalisedPoly,
                "default_params": {
                    "poly_order": 6,
                    "tol": 0.001,
                    "max_iter": 100,
                    "cost_function": "asymmetric_truncated_quadratic",
                },
                "category": "polynomial",
                "description": "Penalised polynomial fitting - Combines polynomial with penalties",
                "suitable_for": ["penalized", "robust_polynomial"],
            },
            "IModPoly": {
                "class": rp.preprocessing.baseline.IModPoly(),
                "default_params": {"poly_order": 2, "tol": 0.001, "max_iter": 200},
                "category": "polynomial",
                "description": "Improved Modified polynomial - Enhanced peak detection",
                "suitable_for": ["improved_peak_detection", "iterative"],
            },
            # =================== SPECIALIZED METHODS ===================
            "Goldindec": {
                "class": rp.preprocessing.baseline.Goldindec,
                "default_params": {"poly_order": 4, "tol": 0.001, "max_iter": 100},
                "category": "statistical",
                "description": "Goldindec algorithm - Statistical approach",
                "suitable_for": ["statistical", "robust"],
            },
            "IRSQR": {
                "class": rp.preprocessing.baseline.IRSQR,
                "default_params": {
                    "lam": 1e5,
                    "quantile": 0.05,
                    "max_iter": 50,
                    "tol": 0.001,
                },
                "category": "quantile",
                "description": "Iterative Reweighted Spline Quantile Regression - Quantile-based",
                "suitable_for": ["quantile_based", "robust", "outlier_resistant"],
            },
            "CornerCutting": {
                "class": rp.preprocessing.baseline.CornerCutting,
                "default_params": {"max_iter": 100},
                "category": "geometric",
                "description": "Corner Cutting algorithm - Geometric approach",
                "suitable_for": ["geometric", "fast", "simple"],
            },
            "FABC": {
                "class": rp.preprocessing.baseline.FABC,
                "default_params": {
                    "lam": 1e6,
                    "scale": 1.0,
                    "num_std": 3.0,
                    "max_iter": 50,
                },
                "category": "automatic",
                "description": "Fully Automatic Baseline Correction - Requires minimal parameters",
                "suitable_for": ["automatic", "minimal_tuning", "general_purpose"],
            },
        }

    def _build_presets(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Build preset configurations for different use cases."""
        return {
            "peak_preserving": {
                "Poly_gentle": {"method": "Poly", "params": {"poly_order": 2}},
                "ASLS_gentle": {
                    "method": "ASLS",
                    "params": {"lam": 1e5, "p": 0.05, "max_iter": 30, "tol": 0.01},
                },
                "ARPLS_gentle": {
                    "method": "ARPLS",
                    "params": {"lam": 1e4, "max_iter": 30, "tol": 0.01},
                },
                "IRSQR_gentle": {
                    "method": "IRSQR",
                    "params": {"lam": 1e4, "quantile": 0.05, "max_iter": 30},
                },
                "FABC_gentle": {
                    "method": "FABC",
                    "params": {
                        "lam": 1e5,
                        "scale": 0.5,
                        "num_std": 2.0,
                        "max_iter": 30,
                    },
                },
            },
            "biological_optimized": {
                "AIRPLS_bio": {
                    "method": "AIRPLS",
                    "params": {"lam": 1e6, "max_iter": 50, "tol": 0.001},
                },
                "ARPLS_bio": {
                    "method": "ARPLS",
                    "params": {"lam": 1e5, "max_iter": 50, "tol": 0.001},
                },
                "ASPLS_bio": {
                    "method": "ASPLS",
                    "params": {"lam": 1e6, "alpha": 0.95, "max_iter": 50},
                },
                "ModPoly_bio": {
                    "method": "ModPoly",
                    "params": {"poly_order": 4, "tol": 0.01, "max_iter": 50},
                },
                "FABC_bio": {
                    "method": "FABC",
                    "params": {"lam": 1e6, "scale": 1.0, "num_std": 3.0},
                },
            },
            "fast_processing": {
                "Poly_fast": {"method": "Poly", "params": {"poly_order": 2}},
                "CornerCutting_fast": {
                    "method": "CornerCutting",
                    "params": {"max_iter": 50},
                },
                "ASLS_fast": {
                    "method": "ASLS",
                    "params": {"lam": 1e5, "max_iter": 20, "tol": 0.01},
                },
            },
            "robust_methods": {
                "AIRPLS_robust": {
                    "method": "AIRPLS",
                    "params": {"lam": 1e6, "max_iter": 100, "tol": 0.0001},
                },
                "IARPLS_robust": {
                    "method": "IARPLS",
                    "params": {"lam": 1e5, "max_iter": 100, "tol": 0.0001},
                },
                "IRSQR_robust": {
                    "method": "IRSQR",
                    "params": {"lam": 1e5, "quantile": 0.05, "max_iter": 100},
                },
            },
        }

    def get_preset_method(self, preset_category: str, method_name: str) -> Any:
        """
        Get a method from a preset configuration.

        Parameters
        ----------
        preset_category : str
            Category of preset (e.g., 'peak_preserving', 'biological_optimized').
        method_name : str
            Specific method name within the preset.

        Returns
        -------
        Baseline correction method instance.
        """
        if preset_category not in self._presets:
            available = list(self._presets.keys())
            raise ValueError(
                f"Preset '{preset_category}' not available. Available presets: {available}"
            )

        if method_name not in self._presets[preset_category]:
            available = list(self._presets[preset_category].keys())
            raise ValueError(
                f"Method '{method_name}' not in preset '{preset_category}'. Available: {available}"
            )

        preset_config = self._presets[preset_category][method_name]
        base_method = preset_config["method"]
        params = preset_config["params"]

        return self.get_method(base_method, custom_params=params)

    def get_preset_category(self, preset_category: str) -> Dict[str, Any]:
        """
        Get all methods from a preset category.

        Parameters
        ----------
        preset_category : str
            Category of preset.

        Returns
        -------
        Dict[str, Any]
            Dictionary of method instances.
        """
        if preset_category not in self._presets:
            available = list(self._presets.keys())
            raise ValueError(
                f"Preset '{preset_category}' not available. Available presets: {available}"
            )

        methods = {}
        for method_name in self._presets[preset_category]:
            methods[method_name] = self.get_preset_method(preset_category, method_name)

        return methods

    def list_methods(
        self, category: Optional[str] = None, suitable_for: Optional[str] = None
    ) -> List[str]:
        """
        List available methods, optionally filtered by category or suitability.

        Parameters
        ----------
        category : Optional[str]
            Filter by method category.
        suitable_for : Optional[str]
            Filter by suitability.

        Returns
        -------
        List[str]
            List of method names.
        """
        methods = []
        for name, info in self._methods_registry.items():
            if category and info["category"] != category:
                continue
            if suitable_for and suitable_for not in info["suitable_for"]:
                continue
            methods.append(name)
        return methods

    def list_presets(self) -> List[str]:
        """List available preset categories."""
        return list(self._presets.keys())

    def get_method_info(self, method_name: str) -> Dict[str, Any]:
        """Get detailed information about a method."""
        if method_name not in self._methods_registry:
            raise ValueError(f"Method '{method_name}' not found.")
        return self._methods_registry[method_name].copy()

    def recommend_methods(self, use_case: str) -> List[str]:
        """
        Recommend methods based on use case.

        Parameters
        ----------
        use_case : str
            Use case description (biological, peak_perserving, fast, robust, fluorescence, automatic, simple).

        Returns
        -------
        List[str]
            Recommended method names.
        """
        recommendations = {
            "biological": ["ARPLS", "AIRPLS", "ASPLS", "ASLS"],
            "peak_preserving": ["IRSQR", "FABC", "ASLS"],
            "fast": ["Poly", "CornerCutting", "ASLS"],
            "robust": ["AIRPLS", "IARPLS", "IRSQR"],
            "fluorescence": ["ARPLS", "AIRPLS", "ASLS"],
            "automatic": ["FABC", "ASPLS"],
            "simple": ["Poly", "CornerCutting"],
        }

        return recommendations.get(use_case.lower(), [])

    def get_method(
        self,
        method_name: str,
        custom_params: Optional[Dict[str, Any]] = None,
        preset: Optional[str] = None,
    ) -> Any:
        """
        Get a baseline correction method with specified parameters.

        Parameters
        ----------
        method_name : str
            Name of the baseline correction method.
        custom_params : Optional[Dict[str, Any]]
            Custom parameters to override defaults.
        preset : Optional[str]
            Use a preset configuration (overrides custom_params).

        Returns
        -------
        Baseline correction method instance.
        """
        if preset:
            return self.get_preset_method(preset, method_name)

        if method_name not in self._methods_registry:
            available = list(self._methods_registry.keys())
            raise ValueError(
                f"Method '{method_name}' not available. Available methods: {available}"
            )

        method_info = self._methods_registry[method_name]
        method_class = method_info["class"]
        params = method_info["default_params"].copy()

        # Handle region requirement for polynomial methods
        if method_info.get("requires_region", False):
            if self.region is None:
                raise ValueError(f"Method '{method_name}' requires a region to be set.")
            params["regions"] = [self.region]

        # Override with custom parameters (with validation)
        if custom_params:
            # Filter and validate custom parameters
            validated_params = self._validate_and_filter_params(
                method_name, method_class, custom_params
            )
            params.update(validated_params)

        try:
            return method_class(**params)
        except TypeError as e:
            create_logs(
                "baseline_method_error",
                "RamanPipeline",
                f"Error initializing method '{method_name}': {e} \n Falling back to default parameters.",
                status="error",
            )

            # Fallback to default parameters only
            fallback_params = method_info["default_params"].copy()
            if method_info.get("requires_region", False):
                fallback_params["regions"] = [self.region]

            return method_class(**fallback_params)

    def _validate_and_filter_params(
        self, method_name: str, method_class: Any, custom_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and filter parameters for a specific baseline method.

        Parameters
        ----------
        method_name : str
            Name of the baseline method.
        method_class : Any
            The baseline method class.
        custom_params : Dict[str, Any]
            Custom parameters to validate.

        Returns
        -------
        Dict[str, Any]
            Filtered and validated parameters.
        """
        # Get valid parameters for this method
        valid_params = self._get_valid_parameters(method_name, method_class)

        # Filter out invalid parameters
        validated_params = {}
        invalid_params = []

        for param_name, param_value in custom_params.items():
            if param_name in valid_params:
                validated_params[param_name] = param_value
            else:
                invalid_params.append(param_name)

        # Warn about invalid parameters
        if invalid_params:
            create_logs(
                "invalid_baseline_params",
                "RamanPipeline",
                f"Invalid parameters for method '{method_name}': {invalid_params}. \nValid parameters are: {valid_params}.",
                status="warning",
            )

        return validated_params

    def _get_valid_parameters(self, method_name: str, method_class: Any) -> set:
        """
        Get valid parameters for a baseline method.

        Parameters
        ----------
        method_name : str
            Name of the baseline method.
        method_class : Any
            The baseline method class.

        Returns
        -------
        set
            Set of valid parameter names.
        """
        # Method-specific parameter mappings
        method_params = {
            # Least Squares Methods
            "ASLS": {"lam", "p", "diff_order", "max_iter", "tol"},
            "IASLS": {"lam", "p", "lam_1", "max_iter", "tol", "diff_order"},
            "AIRPLS": {"lam", "diff_order", "max_iter", "tol"},
            "ARPLS": {"lam", "diff_order", "max_iter", "tol"},
            "DRPLS": {"lam", "eta", "max_iter", "tol", "diff_order"},
            "IARPLS": {"lam", "diff_order", "max_iter", "tol"},
            "ASPLS": {"lam", "diff_order", "max_iter", "alpha", "tol"},
            # Polynomial Methods
            "Poly": {"poly_order", "regions"},
            "ModPoly": {"poly_order", "tol", "max_iter"},
            "PenalisedPoly": {
                "poly_order",
                "tol",
                "max_iter",
                "cost_function",
                "threshold",
                "alpha_factor",
                "weights",
            },
            "IModPoly": {"poly_order", "tol", "max_iter"},
            # Specialized Methods
            "Goldindec": {"poly_order", "tol", "max_iter"},
            "IRSQR": {"lam", "quantile", "max_iter", "tol", "diff_order"},
            "CornerCutting": {"max_iter"},
            "FABC": {"lam", "scale", "num_std", "max_iter", "diff_order"},
        }

        # Return method-specific parameters if available
        if method_name in method_params:
            return method_params[method_name]

        # Fallback: try to inspect the method's __init__ signature
        try:
            import inspect

            signature = inspect.signature(method_class.__init__)
            # Get parameter names excluding 'self'
            valid_params = set(signature.parameters.keys()) - {"self"}
            return valid_params
        except Exception as e:
            create_logs(
                "baseline_param_inspection_error",
                "RamanPipeline",
                f"Error inspecting parameters for method '{method_name}': {e}. \nReturning common baseline parameters.",
                status="warning",
            )

            # Last resort: return common baseline parameters
            return {
                "lam",
                "p",
                "poly_order",
                "tol",
                "max_iter",
                "diff_order",
                "alpha",
                "eta",
                "quantile",
                "scale",
                "num_std",
                "cost_function",
                "threshold",
                "alpha_factor",
                "weights",
                "regions",
            }

    def get_method_with_auto_fallback(
        self, method_name: str, custom_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Get a baseline method with automatic fallback to similar methods if the requested one fails.

        Parameters
        ----------
        method_name : str
            Name of the baseline correction method.
        custom_params : Optional[Dict[str, Any]]
            Custom parameters to override defaults.

        Returns
        -------
        Baseline correction method instance.
        """
        # Primary attempt
        try:
            return self.get_method(method_name, custom_params)
        except (TypeError, ValueError) as primary_error:
            create_logs(
                "baseline_method_error",
                "RamanPipeline",
                f"Primary method '{method_name}' failed: {primary_error}. \nAttempting fallbacks...",
                status="error",
            )

            # Get similar methods as fallbacks
            fallback_methods = self._get_fallback_methods(method_name)

            for fallback_method in fallback_methods:
                try:
                    create_logs(
                        "baseline_method_fallback",
                        "RamanPipeline",
                        f"Trying fallback method '{fallback_method}' for '{method_name}'...",
                        status="info",
                    )

                    return self.get_method(fallback_method, custom_params)
                except (TypeError, ValueError) as fallback_error:
                    create_logs(
                        "baseline_method_fallback_error",
                        "RamanPipeline",
                        f"Fallback method '{fallback_method}' failed: {fallback_error}. \nTrying next fallback...",
                        status="error",
                    )
                    continue

            # Ultimate fallback: ARPLS with minimal parameters
            create_logs(
                "baseline_method_fallback",
                "RamanPipeline",
                f"All fallbacks failed. Using ARPLS with default parameters.",
                status="error",
            )
            return self.get_method("ARPLS", custom_params=None)

    def _get_fallback_methods(self, method_name: str) -> List[str]:
        """
        Get fallback methods for a given baseline method.

        Parameters
        ----------
        method_name : str
            Name of the original method.

        Returns
        -------
        List[str]
            List of fallback method names in order of preference.
        """
        fallback_mapping = {
            # Polynomial methods fallbacks
            "PenalisedPoly": ["ModPoly", "IModPoly", "ARPLS", "AIRPLS"],
            "ModPoly": ["PenalisedPoly", "IModPoly", "ARPLS"],
            "IModPoly": ["ModPoly", "PenalisedPoly", "ARPLS"],
            "Poly": ["ModPoly", "ARPLS", "ASLS"],
            # Least squares methods fallbacks
            "ASLS": ["ARPLS", "AIRPLS", "ASPLS"],
            "ARPLS": ["AIRPLS", "ASLS", "ASPLS"],
            "AIRPLS": ["ARPLS", "IARPLS", "ASLS"],
            "ASPLS": ["ARPLS", "AIRPLS", "ASLS"],
            "IASLS": ["ASLS", "ARPLS", "AIRPLS"],
            "DRPLS": ["ARPLS", "AIRPLS", "ASLS"],
            "IARPLS": ["AIRPLS", "ARPLS", "ASLS"],
            # Specialized methods fallbacks
            "IRSQR": ["ARPLS", "AIRPLS", "ASLS"],
            "FABC": ["ARPLS", "AIRPLS", "ModPoly"],
            "CornerCutting": ["ARPLS", "ModPoly", "ASLS"],
            "Goldindec": ["ARPLS", "ModPoly", "ASLS"],
        }

        return fallback_mapping.get(method_name, ["ARPLS", "AIRPLS", "ASLS"])


class MultiScaleConv1D:
    """
    Multi-scale 1D Convolutional baseline correction for Raman spectra.

    This method uses multiple convolutional filters with different kernel sizes
    to capture baseline features at different scales, then subtracts the
    reconstructed baseline from the original spectrum.
    """

    def __init__(
        self, kernel_sizes=[5, 11, 21, 41], weights=None, mode="reflect", iterations=1
    ):
        """
        Initialize MultiScaleConv1D baseline correction.

        Parameters
        ----------
        kernel_sizes : list of int
            Different kernel sizes for multi-scale analysis.
        weights : list of float, optional
            Weights for combining different scales. If None, uses equal weights.
        mode : str
            Padding mode for convolution ('reflect', 'constant', 'nearest', 'wrap').
        iterations : int
            Number of iterations to apply the correction.
        """
        self.kernel_sizes = kernel_sizes
        self.weights = weights if weights is not None else [1.0] * len(kernel_sizes)
        self.mode = mode
        self.iterations = iterations

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    def __call__(self, spectra):
        """Apply baseline correction to numpy array format (for sklearn pipelines)."""
        if spectra.ndim == 1:
            return self._correct_spectrum(spectra)
        else:
            return np.array([self._correct_spectrum(spectrum) for spectrum in spectra])

    def apply(self, spectra):
        """Apply baseline correction to ramanspy SpectralContainer format."""
        data = spectra.spectral_data

        if data.ndim == 1:
            # Single spectrum
            corrected_data = self._correct_spectrum(data)
        else:
            # Multiple spectra
            corrected_data = np.array(
                [self._correct_spectrum(spectrum) for spectrum in data]
            )

        return rp.SpectralContainer(corrected_data, spectra.spectral_axis)

    def _correct_spectrum(self, spectrum):
        """
        Apply multi-scale convolutional baseline correction to a single spectrum.

        Parameters
        ----------
        spectrum : np.ndarray
            1D array representing a single spectrum.

        Returns
        -------
        np.ndarray
            Baseline-corrected spectrum.
        """
        spectrum = np.asarray(spectrum)
        corrected = spectrum.copy()

        for _ in range(self.iterations):
            baseline = self._estimate_baseline(corrected)
            corrected = spectrum - baseline

            # Ensure no negative values that are too extreme
            corrected = np.maximum(corrected, -np.abs(spectrum).max() * 0.1)

        return corrected

    def _estimate_baseline(self, spectrum):
        """
        Estimate baseline using multi-scale convolution.

        Parameters
        ----------
        spectrum : np.ndarray
            Input spectrum.

        Returns
        -------
        np.ndarray
            Estimated baseline.
        """
        baselines = []

        for kernel_size in self.kernel_sizes:
            # Create a simple averaging kernel
            kernel = np.ones(kernel_size) / kernel_size

            # Apply convolution with padding
            baseline = self._convolve_with_padding(spectrum, kernel)
            baselines.append(baseline)

        # Combine baselines with weights
        combined_baseline = np.zeros_like(spectrum)
        for baseline, weight in zip(baselines, self.weights):
            combined_baseline += weight * baseline

        return combined_baseline

    def _convolve_with_padding(self, signal, kernel):
        """
        Apply convolution with appropriate padding.

        Parameters
        ----------
        signal : np.ndarray
            Input signal.
        kernel : np.ndarray
            Convolution kernel.

        Returns
        -------
        np.ndarray
            Convolved signal with same length as input.
        """
        # Calculate padding
        pad_width = len(kernel) // 2

        # Pad the signal
        if self.mode == "reflect":
            padded_signal = np.pad(signal, pad_width, mode="reflect")
        elif self.mode == "constant":
            padded_signal = np.pad(
                signal, pad_width, mode="constant", constant_values=0
            )
        elif self.mode == "nearest":
            padded_signal = np.pad(signal, pad_width, mode="edge")
        elif self.mode == "wrap":
            padded_signal = np.pad(signal, pad_width, mode="wrap")
        else:
            raise ValueError(f"Unknown padding mode: {self.mode}")

        # Apply convolution
        convolved = np.convolve(padded_signal, kernel, mode="valid")

        return convolved


if TORCH_AVAILABLE:

    class Transformer1DBaseline:
        """
        Transformer-based baseline correction for Raman spectra using PyTorch.

        This method uses a real transformer architecture to learn and predict
        baseline patterns in Raman spectra.
        """

        def __init__(
            self,
            d_model=64,
            nhead=8,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.1,
            window_size=128,
            overlap=0.5,
            learning_rate=1e-3,
            epochs=50,
            device=None,
        ):
            """
            Initialize Transformer1DBaseline correction.

            Parameters
            ----------
            d_model : int
                Dimension of the model (embedding size).
            nhead : int
                Number of attention heads.
            num_layers : int
                Number of transformer encoder layers.
            dim_feedforward : int
                Dimension of feedforward network.
            dropout : float
                Dropout rate.
            window_size : int
                Size of the sliding window for processing.
            overlap : float
                Overlap between windows (0.0 to 1.0).
            learning_rate : float
                Learning rate for training.
            epochs : int
                Number of training epochs.
            device : str, optional
                Device to use ('cuda', 'cpu', or None for auto-detection).
            """
            if not TORCH_AVAILABLE:
                raise ImportError(
                    "PyTorch is not available. Please install it to use Transformer1DBaseline."
                )

            self.d_model = d_model
            self.nhead = nhead
            self.num_layers = num_layers
            self.dim_feedforward = dim_feedforward
            self.dropout = dropout
            self.window_size = window_size
            self.overlap = overlap
            self.learning_rate = learning_rate
            self.epochs = epochs

            # Set device
            if device is None:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self.device = torch.device(device)

            # Initialize model
            self.model = None

        def __call__(self, spectra):
            """Apply baseline correction to numpy array format (for sklearn pipelines)."""
            if spectra.ndim == 1:
                return self._correct_spectrum(spectra)
            else:
                return np.array(
                    [self._correct_spectrum(spectrum) for spectrum in spectra]
                )

        def apply(self, spectra):
            """Apply baseline correction to ramanspy SpectralContainer format."""
            data = spectra.spectral_data

            if data.ndim == 1:
                corrected_data = self._correct_spectrum(data)
            else:
                corrected_data = np.array(
                    [self._correct_spectrum(spectrum) for spectrum in data]
                )

            return rp.SpectralContainer(corrected_data, spectra.spectral_axis)

        def _correct_spectrum(self, spectrum):
            """
            Apply transformer-based baseline correction to a single spectrum.

            Parameters
            ----------
            spectrum : np.ndarray
                1D array representing a single spectrum.

            Returns
            -------
            np.ndarray
                Baseline-corrected spectrum.
            """
            spectrum = np.asarray(spectrum)

            # Initialize and train model if not already done
            if self.model is None:
                self._initialize_model(len(spectrum))
                self._train_model(spectrum)

            # Predict baseline
            baseline = self._predict_baseline(spectrum)

            # Return corrected spectrum
            corrected = spectrum - baseline
            return corrected

        def _initialize_model(self, spectrum_length):
            """Initialize the transformer model."""
            self.spectrum_length = spectrum_length
            self.model = BaselineTransformer(
                spectrum_length=spectrum_length,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                window_size=self.window_size,
            ).to(self.device)

        def _train_model(self, spectrum):
            """
            Train the transformer model on the input spectrum.

            This uses self-supervised learning where the model learns to reconstruct
            the smoothed version of the spectrum (pseudo-baseline).
            """
            # Create training data (smoothed version as target baseline)
            baseline_target = self._create_pseudo_baseline(spectrum)

            # Convert to tensors
            input_tensor = torch.FloatTensor(spectrum).unsqueeze(0).to(self.device)
            target_tensor = (
                torch.FloatTensor(baseline_target).unsqueeze(0).to(self.device)
            )

            # Setup optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()

            # Training loop
            self.model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()

                # Forward pass
                predicted_baseline = self.model(input_tensor)
                loss = criterion(predicted_baseline, target_tensor)

                # Backward pass
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:
                    create_logs(
                        "transformer_training_progress",
                        "RamanPipelineTransformerTraining",
                        f"Epoch {epoch}/{self.epochs}, Loss: {loss.item():.4f}",
                        status="info",
                    )

        def _predict_baseline(self, spectrum):
            """Predict baseline using trained model."""
            self.model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(spectrum).unsqueeze(0).to(self.device)
                baseline = self.model(input_tensor)
                return baseline.cpu().numpy().squeeze()

        def _create_pseudo_baseline(self, spectrum):
            """
            Create pseudo-baseline for training using multiple smoothing techniques.
            """
            # Method 1: Heavy smoothing with large kernel
            kernel_size = max(21, len(spectrum) // 20)
            if kernel_size % 2 == 0:
                kernel_size += 1

            baseline1 = np.convolve(
                spectrum, np.ones(kernel_size) / kernel_size, mode="same"
            )

            # Method 2: Percentile-based smoothing
            from scipy import ndimage

            baseline2 = ndimage.percentile_filter(
                spectrum, percentile=10, size=kernel_size
            )

            # Method 3: Morphological opening (removes peaks)
            from scipy.ndimage import grey_opening

            baseline3 = grey_opening(spectrum, size=kernel_size // 3)

            # Combine baselines
            baseline = (baseline1 + baseline2 + baseline3) / 3

            return baseline

    class BaselineTransformer(nn.Module):
        """
        Transformer model for baseline prediction.
        """

        def __init__(
            self,
            spectrum_length,
            d_model=64,
            nhead=8,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.1,
            window_size=128,
        ):
            super(BaselineTransformer, self).__init__()

            self.spectrum_length = spectrum_length
            self.d_model = d_model
            self.window_size = window_size

            # Input projection
            self.input_projection = nn.Linear(1, d_model)

            # Positional encoding with dynamic max_len
            max_len = max(spectrum_length + 1000, 10000)  # Add buffer for safety
            self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=max_len)

            # Transformer encoder
            encoder_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

            # Output projection
            self.output_projection = nn.Linear(d_model, 1)

            # Smoothing layer with adaptive kernel size
            kernel_size = min(21, spectrum_length // 10)  # Adaptive kernel size
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(3, kernel_size)  # Ensure minimum size

            padding = kernel_size // 2
            self.smooth_conv = nn.Conv1d(
                1, 1, kernel_size=kernel_size, padding=padding, bias=False
            )
            # Initialize smoothing kernel
            with torch.no_grad():
                self.smooth_conv.weight.fill_(1.0 / kernel_size)

        def forward(self, x):
            """
            Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Input spectrum of shape (batch_size, spectrum_length)

            Returns
            -------
            torch.Tensor
                Predicted baseline of shape (batch_size, spectrum_length)
            """
            # Reshape for transformer: (batch_size, seq_len, 1)
            x = x.unsqueeze(-1)

            # Project to model dimension
            x = self.input_projection(x)  # (batch_size, seq_len, d_model)

            # Add positional encoding
            x = self.pos_encoding(x)

            # Apply transformer
            x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

            # Project back to single dimension
            x = self.output_projection(x)  # (batch_size, seq_len, 1)

            # Squeeze last dimension
            x = x.squeeze(-1)  # (batch_size, seq_len)

            # Apply smoothing
            x = x.unsqueeze(1)  # (batch_size, 1, seq_len)
            x = self.smooth_conv(x)  # (batch_size, 1, seq_len)
            x = x.squeeze(1)  # (batch_size, seq_len)

            return x

    class PositionalEncoding(nn.Module):
        """
        Positional encoding for transformer.
        """

        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            # Keep pe as (max_len, d_model) - don't transpose
            pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

            self.register_buffer("pe", pe)

        def forward(self, x):
            # x shape: (batch_size, seq_len, d_model)
            seq_len = x.size(1)

            # Handle case where sequence is longer than max_len
            if seq_len > self.pe.size(1):
                # Extend positional encoding if needed
                self._extend_pe(seq_len, x.device)

            # Add positional encoding: (batch_size, seq_len, d_model)
            x = x + self.pe[:, :seq_len, :]
            return self.dropout(x)

        def _extend_pe(self, new_max_len, device):
            """Extend positional encoding if sequence is longer than max_len."""
            old_max_len = self.pe.size(1)
            d_model = self.pe.size(2)

            # Create extended positional encoding
            pe_extended = torch.zeros(new_max_len, d_model, device=device)
            position = torch.arange(
                0, new_max_len, dtype=torch.float, device=device
            ).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float, device=device)
                * (-np.log(10000.0) / d_model)
            )

            pe_extended[:, 0::2] = torch.sin(position * div_term)
            pe_extended[:, 1::2] = torch.cos(position * div_term)
            # Shape: (1, new_max_len, d_model)
            pe_extended = pe_extended.unsqueeze(0)

            # Update the registered buffer
            self.register_buffer("pe", pe_extended)

    class LightweightTransformer1D:
        """
        Lightweight transformer for faster processing with pre-trained patterns.
        """

        def __init__(
            self, pattern_library_size=5, d_model=32, nhead=4, num_layers=2, epochs=20
        ):
            """
            Initialize lightweight transformer.

            Parameters
            ----------
            pattern_library_size : int
                Number of baseline patterns to learn.
            d_model : int
                Model dimension (smaller for speed).
            nhead : int
                Number of attention heads.
            num_layers : int
                Number of transformer layers.
            epochs : int
                Training epochs.
            """
            self.pattern_library_size = pattern_library_size
            self.d_model = d_model
            self.nhead = nhead
            self.num_layers = num_layers
            self.epochs = epochs

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = None
            self.pattern_library = []

        def __call__(self, spectra):
            """Apply baseline correction to numpy array format."""
            if spectra.ndim == 1:
                return self._correct_spectrum(spectra)
            else:
                return np.array(
                    [self._correct_spectrum(spectrum) for spectrum in spectra]
                )

        def apply(self, spectra):
            """Apply baseline correction to ramanspy SpectralContainer format."""
            data = spectra.spectral_data

            if data.ndim == 1:
                corrected_data = self._correct_spectrum(data)
            else:
                corrected_data = np.array(
                    [self._correct_spectrum(spectrum) for spectrum in data]
                )

            return rp.SpectralContainer(corrected_data, spectra.spectral_axis)

        def _correct_spectrum(self, spectrum):
            """Apply lightweight transformer baseline correction."""
            spectrum = np.asarray(spectrum)

            # Quick pattern matching first
            if len(self.pattern_library) > 0:
                baseline = self._pattern_match_baseline(spectrum)
            else:
                # Fallback to simple smoothing
                baseline = self._simple_baseline(spectrum)

            return spectrum - baseline

        def _pattern_match_baseline(self, spectrum):
            """Match spectrum to learned patterns."""
            # Simple implementation - can be enhanced
            similarities = []
            for pattern in self.pattern_library:
                if len(pattern) == len(spectrum):
                    # Compute similarity (correlation)
                    corr = np.corrcoef(spectrum, pattern)[0, 1]
                    similarities.append(corr if not np.isnan(corr) else 0)
                else:
                    similarities.append(0)

            if similarities:
                best_match_idx = np.argmax(similarities)
                return self.pattern_library[best_match_idx] * 0.8  # Scale down
            else:
                return self._simple_baseline(spectrum)

        def _simple_baseline(self, spectrum):
            """Simple baseline estimation."""
            kernel_size = max(15, len(spectrum) // 30)
            if kernel_size % 2 == 0:
                kernel_size += 1

            baseline = np.convolve(
                spectrum, np.ones(kernel_size) / kernel_size, mode="same"
            )
            return baseline

        def learn_patterns(self, spectra_list):
            """Learn baseline patterns from a list of spectra."""
            for spectrum in spectra_list[: self.pattern_library_size]:
                baseline = self._simple_baseline(spectrum)
                self.pattern_library.append(baseline)


# INTENSITY CALIBRATION
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
        Initializes the IntensityCalibration processor.

        Args:
            reference (np.ndarray, optional): A 1D numpy array representing the
                known, true spectrum of the calibration standard. Defaults to None.
        """
        if reference is not None and not isinstance(reference, np.ndarray):
            raise TypeError("Reference must be a 1D numpy array.")

        self.reference = reference
        self.correction_factor = None

    def _calculate_correction_factor(self, measured_standard: np.ndarray):
        """
        Calculates the correction factor based on the measured standard.

        The factor is the ratio of the true reference spectrum to the spectrum
        measured by the instrument.

        Args:
            measured_standard (np.ndarray): The spectrum of the standard as
                                            measured by the instrument.
        """
        if self.reference is None:
            raise ValueError(
                "A reference spectrum must be set before calculating the correction factor."
            )

        if self.reference.shape != measured_standard.shape:
            raise ValueError(
                "Reference and measured standard spectra must have the same shape."
            )

        # Create a copy to avoid division by zero warnings on the original data
        measured_safe = np.copy(measured_standard)

        # Prevent division by zero or by very small numbers
        measured_safe[measured_safe < 1e-9] = 1e-9

        self.correction_factor = self.reference / measured_safe
        create_logs(
            "intensity_calibration",
            "RamanPipeline",
            f"Calculated correction factor: {self.correction_factor}",
            status="info",
        )

    def __call__(
        self, spectra: np.ndarray, measured_standard: np.ndarray
    ) -> np.ndarray:
        """
        Apply intensity calibration to a 2D numpy array of spectra.
        This makes the class compatible with scikit-learn pipelines.

        Args:
            spectra (np.ndarray): A 2D numpy array (n_samples, n_features) to be calibrated.
            measured_standard (np.ndarray): The measured spectrum of the standard.

        Returns:
            np.ndarray: The calibrated spectra.
        """
        if self.correction_factor is None:
            self._calculate_correction_factor(measured_standard)

        if spectra.ndim != 2:
            raise ValueError("Input spectra for pipeline must be a 2D numpy array.")

        # Apply correction by element-wise multiplication (broadcasting)
        return spectra * self.correction_factor

    def apply(
        self, spectra: rp.SpectralContainer, measured_standard: np.ndarray
    ) -> rp.SpectralContainer:
        """
        Apply intensity calibration to a ramanspy SpectralContainer.

        Args:
            spectra (SpectralContainer): The container with data to be calibrated.
            measured_standard (np.ndarray): The measured spectrum of the standard.

        Returns:
            SpectralContainer: A new container with the calibrated data.
        """
        if self.correction_factor is None:
            self._calculate_correction_factor(measured_standard)

        calibrated_data = spectra.spectral_data * self.correction_factor

        return SpectralContainer(calibrated_data, spectra.spectral_axis)


# WAVENUMBER CALIBRATION
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
            reference_peaks: Known peak positions for calibration standards
            poly_order: Order of polynomial correction (default: 3)
        """
        if reference_peaks is None:
            # Default to Silicon reference peak
            self.reference_peaks = {"Si": 520.5}
        else:
            self.reference_peaks = reference_peaks
        self.poly_order = poly_order
        self.correction_coeffs = None

        create_logs(
            "wavenumber_calibration",
            "WavenumberCalibration",
            f"Initialized with reference peaks: {self.reference_peaks}",
            status="info",
        )

    def calibrate(
        self, measured_peaks: Dict[str, float], wavenumbers: np.ndarray
    ) -> np.ndarray:
        """
        Calibrate wavenumber axis based on measured vs reference peak positions.

        Args:
            measured_peaks: Dictionary of measured peak positions {name: wavenumber}
            wavenumbers: Original wavenumber axis to be corrected

        Returns:
            Corrected wavenumber axis
        """
        if not measured_peaks:
            create_logs(
                "wavenumber_calibration",
                "WavenumberCalibration",
                "No measured peaks provided, returning original wavenumbers",
                status="warning",
            )
            return wavenumbers

        # Extract matching peaks
        common_peaks = set(measured_peaks.keys()).intersection(
            set(self.reference_peaks.keys())
        )
        if not common_peaks:
            create_logs(
                "wavenumber_calibration",
                "WavenumberCalibration",
                "No common peaks found between measured and reference",
                status="error",
            )
            return wavenumbers

        measured_vals = [measured_peaks[peak] for peak in common_peaks]
        reference_vals = [self.reference_peaks[peak] for peak in common_peaks]

        # Fit polynomial correction
        self.correction_coeffs = np.polyfit(
            measured_vals, reference_vals, self.poly_order
        )

        # Apply correction to full wavenumber axis
        corrected_wavenumbers = np.polyval(self.correction_coeffs, wavenumbers)

        create_logs(
            "wavenumber_calibration",
            "WavenumberCalibration",
            f"Applied correction using {len(common_peaks)} peaks: {list(common_peaks)}",
            status="info",
        )

        return corrected_wavenumbers

    def __call__(
        self, wavenumbers: np.ndarray, measured_peaks: Dict[str, float]
    ) -> np.ndarray:
        """Make the class callable for pipeline integration."""
        return self.calibrate(measured_peaks, wavenumbers)


# MSC NORMALIZATION
class MSC:
    """
    Multiplicative Scatter Correction (MSC) for Raman spectra.

    MSC corrects for multiplicative scattering effects by fitting each spectrum
    to a reference spectrum (usually the mean) using linear regression, then
    correcting for the slope and offset.

    This is particularly useful for biological samples where scattering effects
    can vary significantly between measurements.
    """

    def __init__(self):
        """Initialize MSC normalization."""
        self.mean_spectrum = None
        self.is_fitted = False

        create_logs("msc_init", "MSC", "Initialized MSC normalization", status="info")

    def fit(self, spectra: np.ndarray) -> "MSC":
        """
        Fit MSC to training spectra by calculating mean reference spectrum.

        Args:
            spectra: 2D array of spectra (n_samples, n_wavenumbers)

        Returns:
            Self for method chaining
        """
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)

        self.mean_spectrum = np.mean(spectra, axis=0)
        self.is_fitted = True

        create_logs(
            "msc_fit",
            "MSC",
            f"Fitted MSC with {spectra.shape[0]} spectra",
            status="info",
        )

        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """
        Apply MSC correction to spectra.

        Args:
            spectra: 2D array of spectra to correct

        Returns:
            MSC-corrected spectra
        """
        if not self.is_fitted:
            raise ValueError("MSC must be fitted before transform. Call fit() first.")

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
            single_spectrum = True
        else:
            single_spectrum = False

        corrected = []
        for spectrum in spectra:
            # Linear regression: spectrum = a + b * mean_spectrum
            coeffs = np.polyfit(self.mean_spectrum, spectrum, 1)
            slope, intercept = coeffs[0], coeffs[1]

            # MSC correction: (spectrum - intercept) / slope
            corrected_spectrum = (spectrum - intercept) / slope
            corrected.append(corrected_spectrum)

        result = np.array(corrected)

        if single_spectrum:
            result = result.squeeze()

        create_logs(
            "msc_transform",
            "MSC",
            f"Applied MSC correction to {spectra.shape[0]} spectra",
            status="info",
        )

        return result

    def fit_transform(self, spectra: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(spectra).transform(spectra)

    def __call__(self, spectra: np.ndarray) -> np.ndarray:
        """Make the class callable for pipeline integration."""
        if not self.is_fitted:
            return self.fit_transform(spectra)
        else:
            return self.transform(spectra)

    def apply(self, spectra: rp.SpectralContainer) -> rp.SpectralContainer:
        """Apply MSC to ramanspy SpectralContainer."""
        corrected_data = self(spectra.spectral_data)
        return rp.SpectralContainer(corrected_data, spectra.spectral_axis)


# DERIVATIVE PREPROCESSING
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

    def __init__(self, order: int = 1, window_length: int = 5, polyorder: int = 2):
        """
        Initialize derivative processor.

        Args:
            order: Derivative order (1 or 2)
            window_length: Length of filter window (must be odd and >= polyorder + 1)
            polyorder: Order of polynomial for Savitzky-Golay filter
        """
        if order not in [1, 2]:
            raise ValueError("Derivative order must be 1 or 2")

        if window_length % 2 == 0:
            window_length += 1  # Ensure odd window length

        if window_length <= polyorder:
            polyorder = window_length - 1

        self.order = order
        self.window_length = window_length
        self.polyorder = polyorder

        create_logs(
            "derivative_init",
            "Derivative",
            f"Initialized {order}-order derivative with window={window_length}, poly={polyorder}",
            status="info",
        )

    def __call__(self, spectra: np.ndarray) -> np.ndarray:
        """
        Calculate derivatives of spectra.

        Args:
            spectra: 1D or 2D array of spectra

        Returns:
            Derivative spectra
        """
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            raise ImportError("scipy is required for derivative calculation")

        if spectra.ndim == 1:
            result = savgol_filter(
                spectra, self.window_length, self.polyorder, deriv=self.order
            )
        else:
            result = savgol_filter(
                spectra, self.window_length, self.polyorder, deriv=self.order, axis=-1
            )

        create_logs(
            "derivative_calc",
            "Derivative",
            f"Calculated {self.order}-order derivative for {spectra.shape} data",
            status="info",
        )

        return result

    def apply(self, spectra: rp.SpectralContainer) -> rp.SpectralContainer:
        """Apply derivative to ramanspy SpectralContainer."""
        derivative_data = self(spectra.spectral_data)
        return rp.SpectralContainer(derivative_data, spectra.spectral_axis)


# ADVANCED COSMIC RAY REMOVAL
class MedianDespike:
    """
    Median filter-based cosmic ray removal for Raman spectra.

    This method uses median filtering to identify and remove cosmic ray spikes.
    It's particularly effective for narrow, high-intensity spikes that are
    characteristic of cosmic ray events.

    Attributes:
        kernel_size (int): Size of median filter kernel
        threshold (float): Threshold for spike detection (in MAD units)
    """

    def __init__(self, kernel_size: int = 5, threshold: float = 3.0):
        """
        Initialize median-based despiker.

        Args:
            kernel_size: Size of median filter kernel (should be odd)
            threshold: Detection threshold in MAD (Median Absolute Deviation) units
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size

        self.kernel_size = kernel_size
        self.threshold = threshold

        create_logs(
            "median_despike_init",
            "MedianDespike",
            f"Initialized with kernel_size={kernel_size}, threshold={threshold}",
            status="info",
        )

    def __call__(self, spectra: np.ndarray) -> np.ndarray:
        """
        Remove cosmic ray spikes from spectra.

        Args:
            spectra: 1D or 2D array of spectra

        Returns:
            Despiked spectra
        """
        if spectra.ndim == 1:
            return self._despike_spectrum(spectra)
        else:
            return np.array([self._despike_spectrum(spectrum) for spectrum in spectra])

    def _despike_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Remove spikes from a single spectrum."""
        try:
            from scipy.signal import medfilt
        except ImportError:
            raise ImportError("scipy is required for median filtering")

        # Apply median filter
        filtered = medfilt(spectrum, kernel_size=self.kernel_size)

        # Calculate residuals
        residual = spectrum - filtered

        # Calculate MAD (Median Absolute Deviation)
        mad = np.median(np.abs(residual - np.median(residual)))

        # Identify spikes using MAD-based threshold
        # Factor 1.4826 relates MAD to standard deviation for normal distribution
        threshold_value = self.threshold * 1.4826 * mad
        spike_mask = np.abs(residual) > threshold_value

        # Replace spikes with median-filtered values
        corrected = spectrum.copy()
        corrected[spike_mask] = filtered[spike_mask]

        n_spikes = np.sum(spike_mask)
        create_logs(
            "median_despike",
            "MedianDespike",
            f"Removed {n_spikes} spikes from spectrum",
            status="info",
        )

        return corrected

    def apply(self, spectra: rp.SpectralContainer) -> rp.SpectralContainer:
        """Apply median despiking to ramanspy SpectralContainer."""
        despiked_data = self(spectra.spectral_data)
        return rp.SpectralContainer(despiked_data, spectra.spectral_axis)


# RamanPipeline class for preprocessing Raman spectral data
class RamanPipeline:
    """
    A class to handle the preprocessing of Raman spectral data.

    Attributes:
    -----------
    region : tuple
        The range of wavenumbers to consider for analysis.
    """

    def __init__(self, region: tuple[int, int] = (1050, 1700)):
        """
        Initializes the RamanPipeline class.
        """
        self.region = region

    def pipeline_hirschsprung_multi(
        self,
        hirsch_dfs: List[pd.DataFrame],
        normal_dfs: List[pd.DataFrame],
        wavenumber_rowname: str = "wavenumber",
        region: Tuple[int, int] = (1050, 1700),
    ) -> Tuple[np.ndarray, list, np.ndarray, pd.DataFrame]:
        """
        Preprocessing pipeline for multiple Hirshsprung disease and normal Raman DataFrames.

        Parameters:
        ----------
        hirsch_dfs : List[pd.DataFrame]
            List of DataFrames containing Hirshsprung disease Raman spectra.
        normal_dfs : List[pd.DataFrame]
            List of DataFrames containing normal Raman spectra.
        region : Tuple[int, int], optional
            The range of wavenumbers to consider for analysis (default is (1050, 1700)).

        Returns:
        -------
        processed_data : np.ndarray
            The processed spectral data.
        labels : list
            List of labels indicating the source of each spectrum (Hirshsprung or normal).
        wavenumbers : np.ndarray
            The wavenumbers corresponding to the spectral data.
        merged_df : pd.DataFrame
            The merged DataFrame containing all spectra, with wavenumbers as the first column.

        """

        # Handle empty input cases
        if not hirsch_dfs and not normal_dfs:
            raise ValueError(
                "Both hirsch_dfs and normal_dfs are empty. At least one must be provided."
            )
        elif hirsch_dfs:
            wavenumbers = hirsch_dfs[0]["wavenumber"].values
        else:
            wavenumbers = normal_dfs[0]["wavenumber"].values

        # Concatenate all hirsch and normal DataFrames (drop wavenumber, keep only intensity columns)
        all_hirsch = (
            [df.drop(wavenumber_rowname, axis=1) for df in hirsch_dfs]
            if hirsch_dfs
            else []
        )
        all_normal = (
            [df.drop(wavenumber_rowname, axis=1) for df in normal_dfs]
            if normal_dfs
            else []
        )
        merged_df = pd.concat(all_hirsch + all_normal, axis=1)

        intensities = merged_df.values.T  # shape: (n_samples, n_wavenumbers)

        # Labels
        labels = []
        if hirsch_dfs:
            labels += ["hirsch"] * sum(df.shape[1] - 1 for df in hirsch_dfs)
        if normal_dfs:
            labels += ["normal"] * sum(df.shape[1] - 1 for df in normal_dfs)

        # Preprocessing pipeline
        pipeline = rp.preprocessing.Pipeline(
            [
                rp.preprocessing.misc.Cropper(region=region),
                rp.preprocessing.despike.WhitakerHayes(),
                rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
                rp.preprocessing.baseline.ASPLS(lam=1e5, tol=0.01),
                rp.preprocessing.normalise.Vector(),
            ]
        )
        spectra = rp.SpectralContainer(intensities, wavenumbers)
        data = pipeline.apply(spectra)

        # Return processed data, labels, and wavenumbers for further analysis
        return data.spectral_data, labels, data.spectral_axis, merged_df

    def preprocess(
        self,
        dfs: List[pd.DataFrame],
        label: str,
        wavenumber_col: str = "wavenumber",
        intensity_cols: Optional[List[str]] = None,
        region: Tuple[int, int] = (1050, 1700),
        preprocessing_steps: Optional[List[Callable]] = None,
        visualize_steps: bool = False,
        show_parameters_in_title: bool = False,
        max_plot_visualize_steps: int = 10,
        save_pkl: bool = False,
        save_pkl_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Dynamic preprocessing pipeline for generic Raman spectral DataFrames.

        Parameters
        ----------
        dfs : List[pd.DataFrame]
            List of DataFrames containing Raman spectra.
        label : str
            Label for the spectra (e.g., 'hirsch' or 'normal').
        wavenumber_col : str
            Column name for wavenumber axis.
        intensity_cols : Optional[List[str]]
            List of columns for intensity data. If None, use all except wavenumber_col and label_col.
        region : Tuple[int, int]
            Wavenumber region to crop.
        preprocessing_steps : Optional[List[Callable]]
            List of ramanspy preprocessing steps. If None, use default.
        visualize_steps : bool
            If True, visualize each preprocessing step.
        show_parameters_in_title : bool
            If True, show parameters in the plot title.
        max_plot_visualize_steps : int
            Maximum number of spectra to visualize at each step.

        Returns
        -------
        dict:
            Dictionary containing processed spectra, labels, raw DataFrame, and preprocessing info.
            {
                'processed': SpectralContainer,
                'labels': List[str],
                'raw': pd.DataFrame,
                'plot_data': dict,
                'preprocessing_info': dict
            }
        """
        ONDEBUG = False

        if type(dfs) is not list:
            dfs = [dfs]

        # Merge all DataFrames
        merged_df = pd.concat(dfs, axis=1 if dfs[0].index.name == wavenumber_col else 0)

        # Check if index is wavenumber
        if merged_df.index.name == wavenumber_col:
            wavenumbers = merged_df.index.values
            if intensity_cols is None:
                intensity_cols = merged_df.columns.tolist()
            intensities = merged_df[intensity_cols].values.T
        elif wavenumber_col in merged_df.columns:
            wavenumbers = merged_df[wavenumber_col].values
            if intensity_cols is None:
                exclude = {wavenumber_col}
                intensity_cols = [
                    col for col in merged_df.columns if col not in exclude
                ]
            intensities = merged_df[intensity_cols].values
        else:
            raise ValueError(
                f"Wavenumber column '{wavenumber_col}' not found in DataFrame index or columns."
            )

        # Labels: assign the provided label to all spectra
        n_spectra = intensities.shape[0] if intensities.ndim == 2 else 1
        labels = [label] * n_spectra if label is not None else []

        # Default preprocessing pipeline
        if preprocessing_steps is None:
            preprocessing_steps = [
                rp.preprocessing.misc.Cropper(region=region),
                rp.preprocessing.despike.WhitakerHayes(),
                rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
                rp.preprocessing.baseline.ASPLS(lam=1e5, tol=0.01),
                rp.preprocessing.normalise.Vector(),
            ]

        # Initialize preprocessing info dictionary
        preprocessing_info = {
            "pipeline_config": {
                "total_steps": len(preprocessing_steps),
                "wavenumber_col": wavenumber_col,
                "region": region,
                "n_spectra": n_spectra,
                "original_wavenumber_range": (wavenumbers.min(), wavenumbers.max()),
                "original_data_shape": intensities.shape,
            },
            "steps": [],
            "parameters_used": {},
            "step_order": [],
            "execution_info": {
                "execution_time": None,
                "memory_usage": None,
                "errors": [],
            },
        }

        # Record start time
        import time

        start_time = time.time()

        # Apply steps one-by-one (if visualization enabled)
        spectra = rp.SpectralContainer(intensities, wavenumbers)

        # Store initial spectra info
        preprocessing_info["pipeline_config"][
            "initial_spectral_axis_shape"
        ] = spectra.spectral_axis.shape
        preprocessing_info["pipeline_config"][
            "initial_spectral_data_shape"
        ] = spectra.spectral_data.shape

        plot_data = {}
        if visualize_steps:
            # Create combined figure with all steps
            fig, axes = plt.subplots(
                len(preprocessing_steps) + 1,
                1,
                figsize=(12, 3 * (len(preprocessing_steps) + 1)),
                sharex=True,
            )

            # Create individual figure for raw spectra (save but don't show)
            raw_fig, raw_ax = plt.subplots(1, 1, figsize=(12, 6))
            for spectrum in spectra.spectral_data[:max_plot_visualize_steps]:
                raw_ax.plot(spectra.spectral_axis, spectrum, alpha=0.6)
            raw_ax.set_title("Raw Spectra")
            raw_ax.set_xlabel("Wavenumber (cm⁻¹)")
            raw_ax.set_ylabel("Intensity")
            plt.close(raw_fig)  # Close individual figure to avoid showing it

            # Store raw spectra data
            plot_data["raw"] = {
                "figure": raw_fig,
                "title": "Raw Spectra",
                "step_index": 0,
            }

            # Plot raw spectra on combined axes
            axes[0].set_title("Raw Spectra")
            for spectrum in spectra.spectral_data[:max_plot_visualize_steps]:
                axes[0].plot(spectra.spectral_axis, spectrum, alpha=0.6)

            # Process each preprocessing step
            for i, step in enumerate(preprocessing_steps):
                step_start_time = time.time()

                # Store pre-step info
                pre_step_shape = spectra.spectral_data.shape
                pre_step_axis_shape = spectra.spectral_axis.shape

                try:
                    spectra = step.apply(spectra)

                    # Store post-step info
                    post_step_shape = spectra.spectral_data.shape
                    post_step_axis_shape = spectra.spectral_axis.shape
                    step_execution_time = time.time() - step_start_time

                    # Extract step information
                    step_info = {
                        "step_index": i + 1,
                        "step_name": step.__class__.__name__,
                        "step_module": step.__class__.__module__,
                        "parameters": self._extract_step_parameters(step),
                        "execution_time": step_execution_time,
                        "data_transformation": {
                            "input_shape": pre_step_shape,
                            "output_shape": post_step_shape,
                            "input_axis_shape": pre_step_axis_shape,
                            "output_axis_shape": post_step_axis_shape,
                            "shape_changed": pre_step_shape != post_step_shape,
                            "axis_changed": pre_step_axis_shape != post_step_axis_shape,
                        },
                    }

                    # Add specific parameter interpretations based on step type
                    if hasattr(step, "__class__"):
                        step_info["step_category"] = self._categorize_step(step)
                        step_info["parameter_description"] = self._describe_parameters(
                            step
                        )

                    preprocessing_info["steps"].append(step_info)
                    preprocessing_info["step_order"].append(step.__class__.__name__)
                    preprocessing_info["parameters_used"][
                        f"step_{i+1}_{step.__class__.__name__}"
                    ] = step_info["parameters"]

                except Exception as e:
                    error_info = {
                        "step_index": i + 1,
                        "step_name": step.__class__.__name__,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    preprocessing_info["execution_info"]["errors"].append(error_info)
                    create_logs(
                        "preprocess_error",
                        "RamanPipeline",
                        f"Error in step {i+1} ({step.__class__.__name__}): {e}",
                        status="error",
                    )
                    continue

                # Enhanced title with parameters
                if show_parameters_in_title:
                    title = self._create_enhanced_title(step)
                else:
                    title = f"After {step.__class__.__name__}"

                # Create individual figure for this step (save but don't show)
                step_fig, step_ax = plt.subplots(1, 1, figsize=(12, 6))
                for spectrum in spectra.spectral_data[:max_plot_visualize_steps]:
                    step_ax.plot(spectra.spectral_axis, spectrum, alpha=0.6)
                step_ax.set_title(title)
                step_ax.set_xlabel("Wavenumber (cm⁻¹)")
                step_ax.set_ylabel("Intensity")

                # Add parameter info if available
                if show_parameters_in_title:
                    param_text = self._format_parameters_for_plot(step)
                    if param_text:
                        step_ax.text(
                            0.02,
                            0.98,
                            param_text,
                            transform=step_ax.transAxes,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                            fontsize=9,
                        )

                # Close individual figure to avoid showing it
                plt.close(step_fig)

                # Store individual step figure
                step_key = f"step_{i+1}_{step.__class__.__name__}"
                plot_data[step_key] = {
                    "figure": step_fig,
                    "title": title,
                    "step_index": i + 1,
                    "step_name": step.__class__.__name__,
                    "parameters": step_info["parameters"],
                }

                # Plot on combined axes (this will be shown)
                axes[i + 1].set_title(title)
                for spectrum in spectra.spectral_data[:max_plot_visualize_steps]:
                    axes[i + 1].plot(spectra.spectral_axis, spectrum, alpha=0.6)

            # Debug information (if enabled)
            if show_parameters_in_title and ONDEBUG:
                create_logs(
                    "preprocess_debug",
                    "RamanPipeline",
                    f"Debugging preprocessing steps and parameters.\nStep {i}: {step.__class__.__name__}\nDIR:{[attr for attr in dir(step) if not attr.startswith('__')]}",
                    status="info",
                )
                if hasattr(step, "__dict__"):
                    create_logs(
                        "preprocess_debug",
                        "RamanPipeline",
                        f"Step {i} attributes: {step.__dict__}",
                        status="info",
                    )
                if hasattr(step, "_parameters"):
                    create_logs(
                        "preprocess_debug",
                        "RamanPipeline",
                        f"Step {i} parameters: {step._parameters}",
                        status="info",
                    )

            # Format the combined figure
            for ax in axes:
                ax.set_ylabel("Intensity")
            axes[-1].set_xlabel("Wavenumber (cm⁻¹)")
            plt.tight_layout()

            # Store the combined figure
            plot_data["combined_figure"] = fig

            # Show only the combined figure
            plt.show()

        else:
            # Even without visualization, process steps and collect info
            for i, step in enumerate(preprocessing_steps):
                step_start_time = time.time()

                # Store pre-step info
                pre_step_shape = spectra.spectral_data.shape
                pre_step_axis_shape = spectra.spectral_axis.shape

                try:
                    spectra = step.apply(spectra)

                    # Store post-step info
                    post_step_shape = spectra.spectral_data.shape
                    post_step_axis_shape = spectra.spectral_axis.shape
                    step_execution_time = time.time() - step_start_time

                    # Extract step information
                    step_info = {
                        "step_index": i + 1,
                        "step_name": step.__class__.__name__,
                        "step_module": step.__class__.__module__,
                        "parameters": self._extract_step_parameters(step),
                        "execution_time": step_execution_time,
                        "data_transformation": {
                            "input_shape": pre_step_shape,
                            "output_shape": post_step_shape,
                            "input_axis_shape": pre_step_axis_shape,
                            "output_axis_shape": post_step_axis_shape,
                            "shape_changed": pre_step_shape != post_step_shape,
                            "axis_changed": pre_step_axis_shape != post_step_axis_shape,
                        },
                    }

                    # Add specific parameter interpretations
                    if hasattr(step, "__class__"):
                        step_info["step_category"] = self._categorize_step(step)
                        step_info["parameter_description"] = self._describe_parameters(
                            step
                        )

                    preprocessing_info["steps"].append(step_info)
                    preprocessing_info["step_order"].append(step.__class__.__name__)
                    preprocessing_info["parameters_used"][
                        f"step_{i+1}_{step.__class__.__name__}"
                    ] = step_info["parameters"]

                except Exception as e:
                    error_info = {
                        "step_index": i + 1,
                        "step_name": step.__class__.__name__,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    preprocessing_info["execution_info"]["errors"].append(error_info)
                    create_logs(
                        "preprocess_error",
                        "RamanPipeline",
                        f"Error in step {i+1} ({step.__class__.__name__}): {e}",
                        status="error",
                    )
                    continue

                # Store only the final step figure if not visualizing all steps
                if i == len(preprocessing_steps) - 1:
                    title = f"Final - After {step.__class__.__name__}"

                    # Create figure for final step (save but don't show)
                    final_fig, final_ax = plt.subplots(1, 1, figsize=(12, 6))
                    for spectrum in spectra.spectral_data[:max_plot_visualize_steps]:
                        final_ax.plot(spectra.spectral_axis, spectrum, alpha=0.6)
                    final_ax.set_title(title)
                    final_ax.set_xlabel("Wavenumber (cm⁻¹)")
                    final_ax.set_ylabel("Intensity")
                    plt.close(final_fig)  # Close to avoid showing

                    step_key = f"final_{step.__class__.__name__}"
                    plot_data[step_key] = {
                        "figure": final_fig,
                        "title": title,
                        "step_index": i + 1,
                        "step_name": step.__class__.__name__,
                        "parameters": step_info["parameters"],
                    }

        # Complete preprocessing info
        total_execution_time = time.time() - start_time
        preprocessing_info["execution_info"]["execution_time"] = total_execution_time
        preprocessing_info["pipeline_config"][
            "final_spectral_axis_shape"
        ] = spectra.spectral_axis.shape
        preprocessing_info["pipeline_config"][
            "final_spectral_data_shape"
        ] = spectra.spectral_data.shape
        preprocessing_info["pipeline_config"]["final_wavenumber_range"] = (
            spectra.spectral_axis.min(),
            spectra.spectral_axis.max(),
        )

        # Add summary statistics
        preprocessing_info["summary"] = {
            "total_steps_executed": len(
                [s for s in preprocessing_info["steps"] if "error" not in s]
            ),
            "total_errors": len(preprocessing_info["execution_info"]["errors"]),
            "total_execution_time": total_execution_time,
            "average_step_time": (
                total_execution_time / len(preprocessing_steps)
                if preprocessing_steps
                else 0
            ),
            "data_reduction_factor": {
                "spectral_points": (
                    preprocessing_info["pipeline_config"][
                        "initial_spectral_axis_shape"
                    ][0]
                    / preprocessing_info["pipeline_config"][
                        "final_spectral_axis_shape"
                    ][0]
                    if preprocessing_info["pipeline_config"][
                        "final_spectral_axis_shape"
                    ][0]
                    > 0
                    else 1
                ),
                "wavenumber_range_reduction": (
                    (
                        preprocessing_info["pipeline_config"][
                            "original_wavenumber_range"
                        ][1]
                        - preprocessing_info["pipeline_config"][
                            "original_wavenumber_range"
                        ][0]
                    )
                    / (
                        preprocessing_info["pipeline_config"]["final_wavenumber_range"][
                            1
                        ]
                        - preprocessing_info["pipeline_config"][
                            "final_wavenumber_range"
                        ][0]
                    )
                    if (
                        preprocessing_info["pipeline_config"]["final_wavenumber_range"][
                            1
                        ]
                        - preprocessing_info["pipeline_config"][
                            "final_wavenumber_range"
                        ][0]
                    )
                    > 0
                    else 1
                ),
            },
        }

        data = {
            "processed": spectra,
            "labels": labels,
            "raw": merged_df,
            "plot_data": plot_data,
            "preprocessing_info": preprocessing_info,
        }

        if save_pkl:
            try:
                save_dir = os.path.join(CURRENT_DIR, "data", "preprocessed_data")
                os.makedirs(save_dir, exist_ok=True)
                save_pkl_name = (
                    save_pkl_name if save_pkl_name else f"{label}_preprocessed.pkl"
                )
                pkl_path = os.path.join(save_dir, save_pkl_name)
                pkl_path += ".pkl" if not pkl_path.endswith(".pkl") else ""
                with open(pkl_path, "wb") as f:
                    pkl.dump(data, f)
            except Exception as e:
                create_logs(
                    "preprocess_error",
                    "RamanPipeline",
                    f"Error saving preprocessed data: {e}\n{traceback.format_exc()}",
                    status="error",
                )
                raise e

        return data

    def _categorize_step(self, step) -> str:
        """
        Categorize the preprocessing step.

        Parameters
        ----------
        step : preprocessing step object
            The preprocessing step to categorize.

        Returns
        -------
        str
            Category of the step.
        """
        step_name = step.__class__.__name__
        module_name = step.__class__.__module__

        # Category mapping based on class name and module
        if "baseline" in module_name or step_name in [
            "ASLS",
            "AIRPLS",
            "ARPLS",
            "ModPoly",
            "IModPoly",
            "ASPLS",
            "PenalisedPoly",
        ]:
            return "baseline_correction"
        elif "denoise" in module_name or step_name in ["SavGol", "MovingAverage"]:
            return "denoising"
        elif "despike" in module_name or step_name in ["WhitakerHayes"]:
            return "despiking"
        elif (
            "normalise" in module_name
            or "normalize" in module_name
            or step_name in ["Vector", "SNV"]
        ):
            return "normalization"
        elif "misc" in module_name or step_name in ["Cropper"]:
            return "preprocessing"
        elif step_name in ["MultiScaleConv1D", "Transformer1DBaseline"]:
            return "advanced_baseline"
        else:
            return "other"

    def _describe_parameters(self, step) -> Dict[str, str]:
        """
        Provide human-readable descriptions of parameters.

        Parameters
        ----------
        step : preprocessing step object
            The preprocessing step.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping parameter names to descriptions.
        """
        parameters = self._extract_step_parameters(step)
        descriptions = {}

        # Common parameter descriptions
        param_descriptions = {
            "lam": "Smoothness parameter (higher = smoother baseline)",
            "p": "Asymmetry parameter (0-1, lower = more asymmetric)",
            "poly_order": "Polynomial order (higher = more flexible)",
            "tol": "Convergence tolerance (lower = more precise)",
            "max_iter": "Maximum number of iterations",
            "window_length": "Window size for smoothing",
            "polyorder": "Polynomial order for fitting",
            "region": "Wavenumber range for processing",
            "diff_order": "Order of difference matrix",
            "alpha": "Learning rate or weighting factor",
            "quantile": "Quantile for robust estimation",
            "scale": "Scaling factor",
            "num_std": "Number of standard deviations",
            "eta": "Regularization parameter",
            "cost_function": "Cost function for optimization",
            "threshold": "Threshold value for processing",
            "alpha_factor": "Alpha weighting factor",
            "weights": "Weighting scheme",
            "deriv": "Derivative order",
            "mode": "Processing mode",
            "cval": "Constant value for padding",
        }

        for param_name, param_value in parameters.items():
            if param_name in param_descriptions:
                descriptions[param_name] = (
                    f"{param_descriptions[param_name]} (value: {param_value})"
                )
            else:
                descriptions[param_name] = f"Parameter value: {param_value}"

        return descriptions

    def _create_enhanced_title(self, step) -> str:
        """
        Create enhanced title with parameter information for preprocessing steps.

        Parameters
        ----------
        step : preprocessing step object
            The preprocessing step to create title for.

        Returns
        -------
        str
            Enhanced title with parameter information.
        """
        step_name = step.__class__.__name__

        # Common parameters to display for different step types
        param_info = []

        try:
            # Check if step has kwargs dictionary (ramanspy native steps)
            if hasattr(step, "kwargs") and isinstance(step.kwargs, dict):
                kwargs_dict = step.kwargs

                # Map common parameters to their display formats
                common_params = {
                    "region": lambda x: (
                        f"region=({x[0]}-{x[1]})"
                        if isinstance(x, (tuple, list)) and len(x) == 2
                        else f"region={x}"
                    ),
                    "window_length": lambda x: f"window={x}",
                    "polyorder": lambda x: f"poly={x}",
                    "lam": lambda x: f"λ={x:.0e}",
                    "p": lambda x: f"p={x}",
                    "poly_order": lambda x: f"order={x}",
                    "tol": lambda x: f"tol={x}",
                    "max_iter": lambda x: f"iter={x}",
                    "filter_win_size": lambda x: f"win={x}",
                    "kernel_size": lambda x: f"kernel={x}",
                    "threshold": lambda x: f"thresh={x}",
                    "cost_function": lambda x: (
                        f"cost={x}" if len(str(x)) < 10 else f"cost=custom"
                    ),
                    "deriv": lambda x: f"deriv={x}" if x != 0 else "",
                    "weights": lambda x: f"weighted" if x is not None else "",
                    "alpha_factor": lambda x: f"α={x}" if x != 0.99 else "",
                }

                # Extract parameters from kwargs
                for param, formatter in common_params.items():
                    if param in kwargs_dict:
                        value = kwargs_dict[param]
                        if value is not None:
                            result = formatter(value)
                            if result:  # Only add non-empty strings
                                param_info.append(result)

            # Access parameters from step._parameters dictionary as backup
            elif hasattr(step, "_parameters") and isinstance(step._parameters, dict):
                params_dict = step._parameters

                # Use the same parameter formatting as above
                common_params = {
                    "region": lambda x: (
                        f"region=({x[0]}-{x[1]})"
                        if isinstance(x, (tuple, list)) and len(x) == 2
                        else f"region={x}"
                    ),
                    "window_length": lambda x: f"window={x}",
                    "polyorder": lambda x: f"poly={x}",
                    "lam": lambda x: f"λ={x:.0e}",
                    "p": lambda x: f"p={x}",
                    "poly_order": lambda x: f"order={x}",
                    "tol": lambda x: f"tol={x}",
                    "max_iter": lambda x: f"iter={x}",
                    "filter_win_size": lambda x: f"win={x}",
                    "cost_function": lambda x: f"cost={x}",
                }

                for param, formatter in common_params.items():
                    if param in params_dict:
                        param_info.append(formatter(params_dict[param]))

            # Direct attribute access (fallback)
            else:
                # Same attribute checks as before
                if hasattr(step, "region") and step.region is not None:
                    if isinstance(step.region, (tuple, list)) and len(step.region) == 2:
                        param_info.append(f"region=({step.region[0]}-{step.region[1]})")
                    else:
                        param_info.append(f"region={step.region}")

                if hasattr(step, "window_length") and hasattr(step, "polyorder"):
                    param_info.append(f"window={step.window_length}")
                    param_info.append(f"poly={step.polyorder}")

                if hasattr(step, "lam"):
                    param_info.append(f"λ={step.lam:.0e}")

                if hasattr(step, "p"):
                    param_info.append(f"p={step.p}")

                if hasattr(step, "poly_order"):
                    param_info.append(f"order={step.poly_order}")

                if hasattr(step, "tol"):
                    param_info.append(f"tol={step.tol}")

                if hasattr(step, "max_iter"):
                    param_info.append(f"iter={step.max_iter}")

                if hasattr(step, "filter_win_size"):
                    param_info.append(f"win={step.filter_win_size}")

                if hasattr(step, "window_length") and not hasattr(step, "polyorder"):
                    param_info.append(f"window={step.window_length}")

                if step_name in ["PenalisedPoly", "ModPoly", "IModPoly"]:
                    if hasattr(step, "cost_function"):
                        param_info.append(f"cost={step.cost_function}")

            # Special case for specific classes
            if step_name == "SNV":
                param_info.append("norm=SNV")

            # Inspect the step's __dict__ as a last resort
            if not param_info and hasattr(step, "__dict__"):
                for key, value in step.__dict__.items():
                    if not key.startswith("_") and not callable(value):
                        if isinstance(value, (int, float, str, bool, tuple, list)):
                            if isinstance(value, float) and abs(value) > 1000:
                                param_info.append(f"{key}={value:.0e}")
                            else:
                                param_info.append(f"{key}={value}")

        except Exception as e:
            # If parameter extraction fails, just show the class name
            create_logs(
                "preprocess_error",
                "RamanPipeline",
                f"Error extracting parameters from {step_name}: {e}",
                status="error",
            )

        # Create title
        if param_info:
            # Filter out empty strings
            param_info = [p for p in param_info if p]
            params_str = ", ".join(param_info)
            title = f"After {step_name} ({params_str})"
        else:
            title = f"After {step_name}"

        # Limit title length to prevent overcrowding
        if len(title) > 80:
            title = title[:77] + "..."

        return title

    def _get_baseline_method_params(self, step) -> List[str]:
        """
        Extract parameters specifically for baseline correction methods.

        Parameters
        ----------
        step : baseline correction step
            The baseline correction step.

        Returns
        -------
        List[str]
            List of parameter strings.
        """
        params = []

        # Common baseline parameters
        baseline_params = {
            "lam": lambda x: f"λ={x:.0e}",
            "p": lambda x: f"p={x}",
            "poly_order": lambda x: f"order={x}",
            "tol": lambda x: f"tol={x}",
            "max_iter": lambda x: f"iter={x}",
            "diff_order": lambda x: f"diff={x}",
            "quantile": lambda x: f"q={x}",
            "alpha": lambda x: f"α={x}",
            "eta": lambda x: f"η={x}",
            "scale": lambda x: f"scale={x}",
            "num_std": lambda x: f"std={x}",
            "lam_1": lambda x: f"λ₁={x:.0e}",
        }

        for param_name, formatter in baseline_params.items():
            if hasattr(step, param_name):
                value = getattr(step, param_name)
                if value is not None:
                    params.append(formatter(value))

        return params

    def _extract_step_parameters(self, step) -> Dict[str, Any]:
        """
        Extract parameters from a preprocessing step for storage.

        Parameters
        ----------
        step : preprocessing step object
            The preprocessing step to extract parameters from.

        Returns
        -------
        Dict[str, Any]
            Dictionary of parameters and their values.
        """
        parameters = {}

        try:
            # Check if step has kwargs dictionary (ramanspy native steps)
            if hasattr(step, "kwargs") and isinstance(step.kwargs, dict):
                parameters.update(step.kwargs)

            # Check for _parameters attribute
            elif hasattr(step, "_parameters") and isinstance(step._parameters, dict):
                parameters.update(step._parameters)

            # Direct attribute access for common parameters
            else:
                common_attrs = [
                    "region",
                    "window_length",
                    "polyorder",
                    "lam",
                    "p",
                    "poly_order",
                    "tol",
                    "max_iter",
                    "filter_win_size",
                    "kernel_size",
                    "threshold",
                    "cost_function",
                    "alpha",
                    "alpha_factor",
                    "weights",
                    "deriv",
                    "delta",
                    "mode",
                    "cval",
                ]

                for attr in common_attrs:
                    if hasattr(step, attr):
                        parameters[attr] = getattr(step, attr)

            # Add step class name
            parameters["class_name"] = step.__class__.__name__

        except Exception as e:
            create_logs(
                "preprocess_error",
                "RamanPipeline",
                f"Error extracting parameters from {step.__class__.__name__}: {e}",
                status="error",
            )
            parameters["class_name"] = step.__class__.__name__
            parameters["extraction_error"] = str(e)

        return parameters

    def _format_parameters_for_plot(self, step):
        """
        Format parameters for display in plots.

        Parameters
        ----------
        step : preprocessing step object
            The preprocessing step.

        Returns
        -------
        str
            Formatted parameter string.
        """
        try:
            parameters = self._extract_step_parameters(step)
            param_lines = []

            # Skip certain keys for display
            skip_keys = {
                "class_name",
                "extraction_error",
                "method",
                "mode",
                "cval",
                "delta",
            }

            for key, value in parameters.items():
                if key in skip_keys:
                    continue

                if isinstance(value, float):
                    if abs(value) > 1000:
                        param_lines.append(f"{key}: {value:.0e}")
                    else:
                        param_lines.append(f"{key}: {value:.3f}")
                elif isinstance(value, (tuple, list)) and len(value) == 2:
                    param_lines.append(f"{key}: ({value[0]}-{value[1]})")
                else:
                    param_lines.append(f"{key}: {value}")

            # Limit to 6 lines for readability
            return "\n".join(param_lines[:6])

        except Exception as e:
            return f"Parameters: {step.__class__.__name__}"


class EnhancedRamanPipeline(RamanPipeline):
    """Enhanced RamanPipeline with progress tracking support."""

    def preprocess_with_progress(
        self,
        dfs: List[pd.DataFrame],
        label: str,
        preprocessing_steps: List[Callable],
        progress_callback: Callable[[int, str, int], bool] = None,
        wavenumber_col: str = "wavenumber",
        intensity_cols: Optional[List[str]] = None,
        region: Tuple[int, int] = (1050, 1700),
        visualize_steps: bool = False,
        save_pkl: bool = False,
        save_pkl_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Enhanced preprocessing pipeline with progress tracking.

        Parameters
        ----------
        dfs : List[pd.DataFrame]
            List of DataFrames containing Raman spectra.
        label : str
            Label for the spectra.
        preprocessing_steps : List[Callable]
            List of preprocessing steps to apply.
        progress_callback : Callable[[int, str, int], bool], optional
            Callback function for progress updates. Should return False to cancel.
            Parameters: (step_index, step_name, progress_percent)
        wavenumber_col : str
            Column name for wavenumber axis.
        intensity_cols : Optional[List[str]]
            List of columns for intensity data.
        region : Tuple[int, int]
            Wavenumber region to crop.
        visualize_steps : bool
            If True, visualize each preprocessing step.
        save_pkl : bool
            If True, save results to pickle file.
        save_pkl_name : Optional[str]
            Name for pickle file.

        Returns
        -------
        dict
            Dictionary containing processed spectra and metadata.
        """

        if type(dfs) is not list:
            dfs = [dfs]

        # Merge all DataFrames
        merged_df = pd.concat(dfs, axis=1 if dfs[0].index.name == wavenumber_col else 0)

        # Check if index is wavenumber
        if merged_df.index.name == wavenumber_col:
            wavenumbers = merged_df.index.values
            if intensity_cols is None:
                intensity_cols = merged_df.columns.tolist()
            intensities = merged_df[intensity_cols].values.T
        elif wavenumber_col in merged_df.columns:
            wavenumbers = merged_df[wavenumber_col].values
            if intensity_cols is None:
                exclude = {wavenumber_col}
                intensity_cols = [
                    col for col in merged_df.columns if col not in exclude
                ]
            intensities = merged_df[intensity_cols].values
        else:
            raise ValueError(
                f"Wavenumber column '{wavenumber_col}' not found in DataFrame index or columns."
            )

        # Labels: assign the provided label to all spectra
        n_spectra = intensities.shape[0] if intensities.ndim == 2 else 1
        labels = [label] * n_spectra if label is not None else []

        # Initialize preprocessing info dictionary
        preprocessing_info = {
            "pipeline_config": {
                "total_steps": len(preprocessing_steps),
                "wavenumber_col": wavenumber_col,
                "region": region,
                "n_spectra": n_spectra,
                "original_wavenumber_range": (wavenumbers.min(), wavenumbers.max()),
                "original_data_shape": intensities.shape,
            },
            "steps": [],
            "parameters_used": {},
            "step_order": [],
            "execution_info": {
                "execution_time": None,
                "memory_usage": None,
                "errors": [],
            },
        }

        # Record start time
        import time

        start_time = time.time()

        # Apply steps one-by-one with progress tracking
        spectra = rp.SpectralContainer(intensities, wavenumbers)

        # Store initial spectra info
        preprocessing_info["pipeline_config"][
            "initial_spectral_axis_shape"
        ] = spectra.spectral_axis.shape
        preprocessing_info["pipeline_config"][
            "initial_spectral_data_shape"
        ] = spectra.spectral_data.shape

        # Process each preprocessing step with progress tracking
        for i, step in enumerate(preprocessing_steps):
            step_start_time = time.time()
            step_name = step.__class__.__name__

            # Call progress callback at start of step
            if progress_callback:
                should_continue = progress_callback(i, step_name, 0)
                if not should_continue:
                    raise InterruptedError("Processing cancelled by user")

            # Store pre-step info
            pre_step_shape = spectra.spectral_data.shape
            pre_step_axis_shape = spectra.spectral_axis.shape

            try:
                # Apply the preprocessing step
                spectra = step.apply(spectra)

                # Store post-step info
                post_step_shape = spectra.spectral_data.shape
                post_step_axis_shape = spectra.spectral_axis.shape
                step_execution_time = time.time() - step_start_time

                # Extract step information
                step_info = {
                    "step_index": i + 1,
                    "step_name": step_name,
                    "step_module": step.__class__.__module__,
                    "parameters": self._extract_step_parameters(step),
                    "execution_time": step_execution_time,
                    "data_transformation": {
                        "input_shape": pre_step_shape,
                        "output_shape": post_step_shape,
                        "input_axis_shape": pre_step_axis_shape,
                        "output_axis_shape": post_step_axis_shape,
                        "shape_changed": pre_step_shape != post_step_shape,
                        "axis_changed": pre_step_axis_shape != post_step_axis_shape,
                    },
                }

                # Add specific parameter interpretations based on step type
                if hasattr(step, "__class__"):
                    step_info["step_category"] = self._categorize_step(step)
                    step_info["parameter_description"] = self._describe_parameters(step)

                preprocessing_info["steps"].append(step_info)
                preprocessing_info["step_order"].append(step_name)
                preprocessing_info["parameters_used"][f"step_{i+1}_{step_name}"] = (
                    step_info["parameters"]
                )

                # Call progress callback at end of step
                if progress_callback:
                    step_progress = int(((i + 1) / len(preprocessing_steps)) * 100)
                    should_continue = progress_callback(i, step_name, step_progress)
                    if not should_continue:
                        raise InterruptedError("Processing cancelled by user")

            except InterruptedError:
                # Re-raise cancellation
                raise
            except Exception as e:
                error_info = {
                    "step_index": i + 1,
                    "step_name": step_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                preprocessing_info["execution_info"]["errors"].append(error_info)
                create_logs(
                    "preprocess_error",
                    "EnhancedRamanPipeline",
                    f"Error in step {i+1} ({step_name}): {e}",
                    status="error",
                )
                raise  # Re-raise the error

        # Complete preprocessing info
        total_execution_time = time.time() - start_time
        preprocessing_info["execution_info"]["execution_time"] = total_execution_time
        preprocessing_info["pipeline_config"][
            "final_spectral_axis_shape"
        ] = spectra.spectral_axis.shape
        preprocessing_info["pipeline_config"][
            "final_spectral_data_shape"
        ] = spectra.spectral_data.shape
        preprocessing_info["pipeline_config"]["final_wavenumber_range"] = (
            spectra.spectral_axis.min(),
            spectra.spectral_axis.max(),
        )

        # Add summary statistics
        preprocessing_info["summary"] = {
            "total_steps_executed": len(
                [s for s in preprocessing_info["steps"] if "error" not in s]
            ),
            "total_errors": len(preprocessing_info["execution_info"]["errors"]),
            "total_execution_time": total_execution_time,
            "average_step_time": (
                total_execution_time / len(preprocessing_steps)
                if preprocessing_steps
                else 0
            ),
            "data_reduction_factor": {
                "spectral_points": (
                    preprocessing_info["pipeline_config"][
                        "initial_spectral_axis_shape"
                    ][0]
                    / preprocessing_info["pipeline_config"][
                        "final_spectral_axis_shape"
                    ][0]
                    if preprocessing_info["pipeline_config"][
                        "final_spectral_axis_shape"
                    ][0]
                    > 0
                    else 1
                ),
                "wavenumber_range_reduction": (
                    (
                        preprocessing_info["pipeline_config"][
                            "original_wavenumber_range"
                        ][1]
                        - preprocessing_info["pipeline_config"][
                            "original_wavenumber_range"
                        ][0]
                    )
                    / (
                        preprocessing_info["pipeline_config"]["final_wavenumber_range"][
                            1
                        ]
                        - preprocessing_info["pipeline_config"][
                            "final_wavenumber_range"
                        ][0]
                    )
                    if (
                        preprocessing_info["pipeline_config"]["final_wavenumber_range"][
                            1
                        ]
                        - preprocessing_info["pipeline_config"][
                            "final_wavenumber_range"
                        ][0]
                    )
                    > 0
                    else 1
                ),
            },
        }

        data = {
            "processed": spectra,
            "labels": labels,
            "raw": merged_df,
            "plot_data": {},
            "preprocessing_info": preprocessing_info,
        }

        if save_pkl:
            try:
                save_dir = os.path.join(CURRENT_DIR, "data", "preprocessed_data")
                os.makedirs(save_dir, exist_ok=True)
                save_pkl_name = (
                    save_pkl_name if save_pkl_name else f"{label}_preprocessed.pkl"
                )
                pkl_path = os.path.join(save_dir, save_pkl_name)
                pkl_path += ".pkl" if not pkl_path.endswith(".pkl") else ""
                with open(pkl_path, "wb") as f:
                    pkl.dump(data, f)
            except Exception as e:
                create_logs(
                    "preprocess_error",
                    "EnhancedRamanPipeline",
                    f"Error saving preprocessed data: {e}\n{traceback.format_exc()}",
                    status="error",
                )
                raise e

        return data


# Add preprocessing step registry for dynamic pipeline building
class PreprocessingStepRegistry:
    """Registry for managing all available preprocessing steps with their parameters."""

    def __init__(self):
        self._steps = self._build_step_registry()

    def _build_step_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive registry of preprocessing steps organized by category."""
        return {
            "miscellaneous": {
                "Cropper": {
                    "class": rp.preprocessing.misc.Cropper,
                    "default_params": {"region": (800, 1800)},
                    "param_info": {
                        "region": {
                            "type": "tuple",
                            "range": [(400, 4000)],
                            "description": "Wavenumber range to extract (start, end)",
                        }
                    },
                    "description": "Crop the intensity values and the shift axis associated with the band range(s) specified",
                },
                "BackgroundSubtractor": {
                    "class": rp.preprocessing.misc.BackgroundSubtractor,
                    "default_params": {"background": None},
                    "param_info": {
                        "background": {
                            "type": "optional",
                            "description": "Fixed reference background to subtract",
                        }
                    },
                    "description": "Subtract a fixed reference background",
                },
            },
            "denoising": {
                "SavGol": {
                    "class": rp.preprocessing.denoise.SavGol,
                    "default_params": {"window_length": 7, "polyorder": 3},
                    "param_info": {
                        "window_length": {
                            "type": "int",
                            "range": [3, 99],
                            "step": 2,
                            "description": "Window length (must be odd)",
                        },
                        "polyorder": {
                            "type": "int",
                            "range": [1, 10],
                            "description": "Polynomial order",
                        },
                    },
                    "description": "Denoising based on Savitzky-Golay filtering",
                },
                "Whittaker": {
                    "class": rp.preprocessing.denoise.Whittaker,
                    "default_params": {"lam": 1e5, "d": 2},
                    "param_info": {
                        "lam": {
                            "type": "scientific",
                            "range": [1e2, 1e12],
                            "description": "Smoothing parameter",
                        },
                        "d": {
                            "type": "int",
                            "range": [1, 4],
                            "description": "Order of differences",
                        },
                    },
                    "description": "Denoising based on Discrete Penalised Least Squares (Whittaker−Henderson smoothing)",
                },
                "Kernel": {
                    "class": rp.preprocessing.denoise.Kernel,
                    "default_params": {"kernel_type": "uniform", "kernel_size": 7},
                    "param_info": {
                        "kernel_type": {
                            "type": "choice",
                            "choices": ["uniform", "gaussian", "triangular"],
                            "description": "Type of kernel",
                        },
                        "kernel_size": {
                            "type": "int",
                            "range": [3, 21],
                            "step": 2,
                            "description": "Size of kernel (must be odd)",
                        },
                    },
                    "description": "Denoising based on kernel/window smoothers",
                },
                "Gaussian": {
                    "class": rp.preprocessing.denoise.Gaussian,
                    "default_params": {"sigma": 1.0, "order": 0},
                    "param_info": {
                        "sigma": {
                            "type": "float",
                            "range": [0.1, 5.0],
                            "step": 0.1,
                            "description": "Standard deviation for Gaussian kernel",
                        },
                        "order": {
                            "type": "int",
                            "range": [0, 3],
                            "description": "Order of the filter (0 = Gaussian, 1 = first derivative, etc.)",
                        },
                    },
                    "description": "Denoising based on a Gaussian filter",
                },
                "MovingAverage": {
                    "class": MovingAverage,
                    "default_params": {"window_length": 15},
                    "param_info": {
                        "window_length": {
                            "type": "int",
                            "range": [3, 51],
                            "step": 2,
                            "description": "Window length for moving average",
                        }
                    },
                    "description": "Simple moving average smoothing",
                },
            },
            "cosmic_ray_removal": {
                "WhitakerHayes": {
                    "class": rp.preprocessing.despike.WhitakerHayes,
                    "default_params": {"kernel_size": 7, "threshold": 8},
                    "param_info": {
                        "kernel_size": {
                            "type": "int",
                            "range": [3, 15],
                            "step": 2,
                            "description": "Kernel size for filtering",
                        },
                        "threshold": {
                            "type": "float",
                            "range": [1, 20],
                            "step": 0.1,
                            "description": "Threshold for spike detection",
                        },
                    },
                    "description": "Cosmic rays removal based on modified z-scores filtering",
                },
                "Gaussian": {
                    "class": Gaussian,
                    "default_params": {"kernel": 5, "threshold": 3.0},
                    "param_info": {
                        "kernel": {
                            "type": "int",
                            "range": [3, 15],
                            "step": 2,
                            "description": "Gaussian kernel size (standard deviation)",
                        },
                        "threshold": {
                            "type": "float",
                            "range": [1.0, 10.0],
                            "step": 0.1,
                            "description": "MAD-based threshold for spike detection",
                        },
                    },
                    "description": "Cosmic ray removal using Gaussian filter and MAD-based detection",
                },
                "MedianDespike": {
                    "class": MedianDespike,
                    "default_params": {"kernel_size": 5, "threshold": 3.0},
                    "param_info": {
                        "kernel_size": {
                            "type": "int",
                            "range": [3, 15],
                            "step": 2,
                            "description": "Median filter kernel size",
                        },
                        "threshold": {
                            "type": "float",
                            "range": [1.0, 10.0],
                            "step": 0.1,
                            "description": "MAD-based threshold for spike detection",
                        },
                    },
                    "description": "Cosmic ray removal using median filtering",
                },
            },
            "baseline_correction": {
                # Least squares methods
                "ASLS": {
                    "class": rp.preprocessing.baseline.ASLS,
                    "default_params": {
                        "lam": 1e6,
                        "p": 0.01,
                        "diff_order": 2,
                        "max_iter": 50,
                        "tol": 1e-6,
                    },
                    "param_info": {
                        "lam": {
                            "type": "scientific",
                            "range": [1e2, 1e12],
                            "description": "Smoothing parameter",
                        },
                        "p": {
                            "type": "float",
                            "range": [0.001, 0.999],
                            "step": 0.001,
                            "description": "Asymmetry parameter",
                        },
                        "diff_order": {
                            "type": "int",
                            "range": [1, 3],
                            "description": "Difference order",
                        },
                        "max_iter": {
                            "type": "int",
                            "range": [1, 1000],
                            "description": "Maximum iterations",
                        },
                        "tol": {
                            "type": "scientific",
                            "range": [1e-9, 1e-3],
                            "description": "Convergence tolerance",
                        },
                    },
                    "description": "Baseline correction based on Asymmetric Least Squares (AsLS)",
                },
                "IASLS": {
                    "class": rp.preprocessing.baseline.IASLS,
                    "default_params": {
                        "lam": 1e6,
                        "p": 0.01,
                        "lam_1": 1e-4,
                        "max_iter": 50,
                        "tol": 1e-6,
                    },
                    "param_info": {
                        "lam": {
                            "type": "scientific",
                            "range": [1e2, 1e12],
                            "description": "Smoothing parameter",
                        },
                        "p": {
                            "type": "float",
                            "range": [0.001, 0.999],
                            "step": 0.001,
                            "description": "Asymmetry parameter",
                        },
                        "lam_1": {
                            "type": "scientific",
                            "range": [1e-6, 1e-2],
                            "description": "Secondary smoothing parameter",
                        },
                        "max_iter": {
                            "type": "int",
                            "range": [1, 1000],
                            "description": "Maximum iterations",
                        },
                        "tol": {
                            "type": "scientific",
                            "range": [1e-9, 1e-3],
                            "description": "Convergence tolerance",
                        },
                    },
                    "description": "Baseline correction based on Improved Asymmetric Least Squares (IAsLS)",
                },
                "AIRPLS": {
                    "class": rp.preprocessing.baseline.AIRPLS,
                    "default_params": {
                        "lam": 1e6,
                        "diff_order": 2,
                        "max_iter": 50,
                        "tol": 1e-6,
                    },
                    "param_info": {
                        "lam": {
                            "type": "scientific",
                            "range": [1e2, 1e12],
                            "description": "Smoothing parameter",
                        },
                        "diff_order": {
                            "type": "int",
                            "range": [1, 3],
                            "description": "Difference order",
                        },
                        "max_iter": {
                            "type": "int",
                            "range": [1, 1000],
                            "description": "Maximum iterations",
                        },
                        "tol": {
                            "type": "scientific",
                            "range": [1e-9, 1e-3],
                            "description": "Convergence tolerance",
                        },
                    },
                    "description": "Baseline correction based on Adaptive Iteratively Reweighted Penalized Least Squares (airPLS)",
                },
                "ARPLS": {
                    "class": rp.preprocessing.baseline.ARPLS,
                    "default_params": {
                        "lam": 1e5,
                        "diff_order": 2,
                        "max_iter": 50,
                        "tol": 1e-6,
                    },
                    "param_info": {
                        "lam": {
                            "type": "scientific",
                            "range": [1e2, 1e12],
                            "description": "Smoothing parameter",
                        },
                        "diff_order": {
                            "type": "int",
                            "range": [1, 3],
                            "description": "Difference order",
                        },
                        "max_iter": {
                            "type": "int",
                            "range": [1, 1000],
                            "description": "Maximum iterations",
                        },
                        "tol": {
                            "type": "scientific",
                            "range": [1e-9, 1e-3],
                            "description": "Convergence tolerance",
                        },
                    },
                    "description": "Baseline correction based on Asymmetrically Reweighted Penalized Least Squares (arPLS)",
                },
                "DRPLS": {
                    "class": rp.preprocessing.baseline.DRPLS,
                    "default_params": {
                        "lam": 1e5,
                        "eta": 0.5,
                        "max_iter": 50,
                        "tol": 1e-6,
                    },
                    "param_info": {
                        "lam": {
                            "type": "scientific",
                            "range": [1e2, 1e12],
                            "description": "Smoothing parameter",
                        },
                        "eta": {
                            "type": "float",
                            "range": [0.1, 0.9],
                            "step": 0.1,
                            "description": "Reweighting parameter",
                        },
                        "max_iter": {
                            "type": "int",
                            "range": [1, 1000],
                            "description": "Maximum iterations",
                        },
                        "tol": {
                            "type": "scientific",
                            "range": [1e-9, 1e-3],
                            "description": "Convergence tolerance",
                        },
                    },
                    "description": "Baseline correction based on Doubly Reweighted Penalized Least Squares (drPLS)",
                },
                "IARPLS": {
                    "class": rp.preprocessing.baseline.IARPLS,
                    "default_params": {
                        "lam": 1e5,
                        "diff_order": 2,
                        "max_iter": 50,
                        "tol": 1e-6,
                    },
                    "param_info": {
                        "lam": {
                            "type": "scientific",
                            "range": [1e2, 1e12],
                            "description": "Smoothing parameter",
                        },
                        "diff_order": {
                            "type": "int",
                            "range": [1, 3],
                            "description": "Difference order",
                        },
                        "max_iter": {
                            "type": "int",
                            "range": [1, 1000],
                            "description": "Maximum iterations",
                        },
                        "tol": {
                            "type": "scientific",
                            "range": [1e-9, 1e-3],
                            "description": "Convergence tolerance",
                        },
                    },
                    "description": "Baseline correction based on Improved Asymmetrically Reweighted Penalized Least Squares (IarPLS)",
                },
                "ASPLS": {
                    "class": rp.preprocessing.baseline.ASPLS,
                    "default_params": {
                        "lam": 1e6,
                        "diff_order": 2,
                        "max_iter": 50,
                        "alpha": 0.95,
                    },
                    "param_info": {
                        "lam": {
                            "type": "scientific",
                            "range": [1e2, 1e12],
                            "description": "Smoothing parameter",
                        },
                        "diff_order": {
                            "type": "int",
                            "range": [1, 3],
                            "description": "Difference order",
                        },
                        "max_iter": {
                            "type": "int",
                            "range": [1, 1000],
                            "description": "Maximum iterations",
                        },
                        "alpha": {
                            "type": "float",
                            "range": [0.5, 0.99],
                            "step": 0.01,
                            "description": "Adaptive parameter",
                        },
                    },
                    "description": "Baseline correction based on Adaptive Smoothness Penalized Least Squares (asPLS)",
                },
                # Polynomial fitting methods
                "Poly": {
                    "class": rp.preprocessing.baseline.Poly,
                    "default_params": {"poly_order": 3, "regions": None},
                    "param_info": {
                        "poly_order": {
                            "type": "int",
                            "range": [1, 10],
                            "description": "Polynomial order",
                        },
                        "regions": {
                            "type": "optional",
                            "description": "Regions for polynomial fitting (optional)",
                        },
                    },
                    "description": "Baseline correction based on polynomial fitting",
                },
                "ModPoly": {
                    "class": rp.preprocessing.baseline.ModPoly,
                    "default_params": {"poly_order": 3, "tol": 0.001, "max_iter": 250},
                    "param_info": {
                        "poly_order": {
                            "type": "int",
                            "range": [1, 10],
                            "description": "Polynomial order",
                        },
                        "tol": {
                            "type": "float",
                            "range": [0.0001, 0.1],
                            "step": 0.0001,
                            "description": "Tolerance for convergence",
                        },
                        "max_iter": {
                            "type": "int",
                            "range": [1, 500],
                            "description": "Maximum iterations",
                        },
                    },
                    "description": "Baseline correction based on modified polynomial fitting",
                },
                "PenalisedPoly": {
                    "class": rp.preprocessing.baseline.PenalisedPoly,
                    "default_params": {"poly_order": 6, "tol": 0.001, "max_iter": 100},
                    "param_info": {
                        "poly_order": {
                            "type": "int",
                            "range": [1, 15],
                            "description": "Polynomial order",
                        },
                        "tol": {
                            "type": "float",
                            "range": [0.0001, 0.1],
                            "step": 0.0001,
                            "description": "Tolerance for convergence",
                        },
                        "max_iter": {
                            "type": "int",
                            "range": [1, 500],
                            "description": "Maximum iterations",
                        },
                    },
                    "description": "Baseline correction based on penalised polynomial fitting",
                },
                "IModPoly": {
                    "class": rp.preprocessing.baseline.IModPoly,
                    "default_params": {"poly_order": 3, "tol": 0.001, "max_iter": 200},
                    "param_info": {
                        "poly_order": {
                            "type": "int",
                            "range": [1, 10],
                            "description": "Polynomial order",
                        },
                        "tol": {
                            "type": "float",
                            "range": [0.0001, 0.1],
                            "step": 0.0001,
                            "description": "Tolerance for convergence",
                        },
                        "max_iter": {
                            "type": "int",
                            "range": [1, 500],
                            "description": "Maximum iterations",
                        },
                    },
                    "description": "Baseline correction based on improved modified polynomial fitting",
                },
                # Other methods
                "Goldindec": {
                    "class": rp.preprocessing.baseline.Goldindec,
                    "default_params": {"poly_order": 4, "tol": 0.001, "max_iter": 100},
                    "param_info": {
                        "poly_order": {
                            "type": "int",
                            "range": [1, 10],
                            "description": "Polynomial order",
                        },
                        "tol": {
                            "type": "float",
                            "range": [0.0001, 0.1],
                            "step": 0.0001,
                            "description": "Tolerance for convergence",
                        },
                        "max_iter": {
                            "type": "int",
                            "range": [1, 500],
                            "description": "Maximum iterations",
                        },
                    },
                    "description": "Baseline correction based on Goldindec",
                },
                "IRSQR": {
                    "class": rp.preprocessing.baseline.IRSQR,
                    "default_params": {
                        "lam": 1e5,
                        "quantile": 0.05,
                        "max_iter": 50,
                        "tol": 1e-6,
                    },
                    "param_info": {
                        "lam": {
                            "type": "scientific",
                            "range": [1e2, 1e12],
                            "description": "Smoothing parameter",
                        },
                        "quantile": {
                            "type": "float",
                            "range": [0.01, 0.2],
                            "step": 0.01,
                            "description": "Quantile for regression",
                        },
                        "max_iter": {
                            "type": "int",
                            "range": [1, 500],
                            "description": "Maximum iterations",
                        },
                        "tol": {
                            "type": "scientific",
                            "range": [1e-9, 1e-3],
                            "description": "Convergence tolerance",
                        },
                    },
                    "description": "Baseline correction based on Iterative Reweighted Spline Quantile Regression (IRSQR)",
                },
                "CornerCutting": {
                    "class": rp.preprocessing.baseline.CornerCutting,
                    "default_params": {"max_iter": 100},
                    "param_info": {
                        "max_iter": {
                            "type": "int",
                            "range": [1, 500],
                            "description": "Maximum iterations",
                        }
                    },
                    "description": "Baseline correction based on Corner Cutting",
                },
                "FABC": {
                    "class": rp.preprocessing.baseline.FABC,
                    "default_params": {
                        "lam": 1e6,
                        "scale": 1.0,
                        "num_std": 3.0,
                        "max_iter": 50,
                    },
                    "param_info": {
                        "lam": {
                            "type": "scientific",
                            "range": [1e2, 1e12],
                            "description": "Smoothing parameter",
                        },
                        "scale": {
                            "type": "float",
                            "range": [0.1, 5.0],
                            "step": 0.1,
                            "description": "Scale parameter",
                        },
                        "num_std": {
                            "type": "float",
                            "range": [1.0, 10.0],
                            "step": 0.5,
                            "description": "Number of standard deviations",
                        },
                        "max_iter": {
                            "type": "int",
                            "range": [1, 500],
                            "description": "Maximum iterations",
                        },
                    },
                    "description": "Baseline correction based on Fully automatic baseline correction (FABC)",
                },
                # Custom advanced methods
                "MultiScaleConv1D": {
                    "class": MultiScaleConv1D,
                    "default_params": {
                        "kernel_sizes": [5, 11, 21, 41],
                        "weights": None,
                        "mode": "reflect",
                        "iterations": 1,
                    },
                    "param_info": {
                        "kernel_sizes": {
                            "type": "list",
                            "description": "List of kernel sizes for multi-scale convolution",
                        },
                        "weights": {
                            "type": "optional",
                            "description": "Weights for combining different scales (optional)",
                        },
                        "mode": {
                            "type": "choice",
                            "choices": [
                                "reflect",
                                "constant",
                                "nearest",
                                "mirror",
                                "wrap",
                            ],
                            "description": "Boundary condition for convolution",
                        },
                        "iterations": {
                            "type": "int",
                            "range": [1, 10],
                            "description": "Number of correction iterations",
                        },
                    },
                    "description": "Multi-scale convolutional baseline correction",
                },
            },
            "calibration": {
                "WavenumberCalibration": {
                    "class": WavenumberCalibration,
                    "default_params": {
                        "reference_peaks": {"Si": 520.5},
                        "poly_order": 3,
                    },
                    "param_info": {
                        "reference_peaks": {
                            "type": "dict",
                            "description": "Reference peak positions {name: wavenumber}",
                        },
                        "poly_order": {
                            "type": "int",
                            "range": [1, 5],
                            "description": "Polynomial order for wavenumber correction",
                        },
                    },
                    "description": "Wavenumber axis calibration using reference peaks",
                },
                "IntensityCalibration": {
                    "class": IntensityCalibration,
                    "default_params": {"reference": None},
                    "param_info": {
                        "reference": {
                            "type": "optional",
                            "description": "Reference standard spectrum for calibration",
                        }
                    },
                    "description": "Intensity calibration using reference standards",
                },
            },
            "derivatives": {
                "Derivative": {
                    "class": Derivative,
                    "default_params": {"order": 1, "window_length": 5, "polyorder": 2},
                    "param_info": {
                        "order": {
                            "type": "choice",
                            "choices": [1, 2],
                            "description": "Derivative order (1st or 2nd)",
                        },
                        "window_length": {
                            "type": "int",
                            "range": [3, 21],
                            "step": 2,
                            "description": "Savitzky-Golay window length",
                        },
                        "polyorder": {
                            "type": "int",
                            "range": [1, 6],
                            "description": "Polynomial order for fitting",
                        },
                    },
                    "description": "Spectral derivatives using Savitzky-Golay method",
                }
            },
            "normalisation": {
                "Vector": {
                    "class": rp.preprocessing.normalise.Vector,
                    "default_params": {"pixelwise": False},
                    "param_info": {
                        "pixelwise": {
                            "type": "bool",
                            "description": "Apply normalisation pixelwise (True) or spectrumwise (False)",
                        }
                    },
                    "description": "Vector normalisation",
                },
                "MinMax": {
                    "class": rp.preprocessing.normalise.MinMax,
                    "default_params": {"pixelwise": False, "a": 0, "b": 1},
                    "param_info": {
                        "pixelwise": {
                            "type": "bool",
                            "description": "Apply normalisation pixelwise (True) or spectrumwise (False)",
                        },
                        "a": {
                            "type": "float",
                            "range": [-10.0, 10.0],
                            "description": "Minimum value for scaling",
                        },
                        "b": {
                            "type": "float",
                            "range": [-10.0, 10.0],
                            "description": "Maximum value for scaling",
                        },
                    },
                    "description": "Min-max normalisation",
                },
                "MaxIntensity": {
                    "class": rp.preprocessing.normalise.MaxIntensity,
                    "default_params": {"pixelwise": False},
                    "param_info": {
                        "pixelwise": {
                            "type": "bool",
                            "description": "Apply normalisation pixelwise (True) or spectrumwise (False)",
                        }
                    },
                    "description": "Max intensity normalisation",
                },
                "AUC": {
                    "class": rp.preprocessing.normalise.AUC,
                    "default_params": {"pixelwise": False},
                    "param_info": {
                        "pixelwise": {
                            "type": "bool",
                            "description": "Apply normalisation pixelwise (True) or spectrumwise (False)",
                        }
                    },
                    "description": "Area under the curve normalisation",
                },
                "SNV": {
                    "class": SNV,
                    "default_params": {},
                    "param_info": {},
                    "description": "Standard Normal Variate normalisation",
                },
                "MSC": {
                    "class": MSC,
                    "default_params": {},
                    "param_info": {},
                    "description": "Multiplicative Scatter Correction for scattering effects",
                },
            },
        }

    def get_categories(self) -> List[str]:
        """Get all available preprocessing categories."""
        return list(self._steps.keys())

    def get_methods_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get all methods in a specific category."""
        return self._steps.get(category, {})

    def get_method_info(self, category: str, method: str) -> Dict[str, Any]:
        """Get information about a specific method."""
        return self._steps.get(category, {}).get(method, {})

    def create_method_instance(
        self, category: str, method: str, params: Dict[str, Any] = None
    ) -> Any:
        """Create an instance of a preprocessing method with given parameters."""
        method_info = self.get_method_info(category, method)
        if not method_info:
            raise ValueError(f"Method {method} not found in category {category}")

        method_class = method_info["class"]
        final_params = method_info["default_params"].copy()
        if params:
            final_params.update(params)

        try:
            return method_class(**final_params)
        except Exception as e:
            create_logs(
                "PreprocessingStepRegistry",
                "method_creation",
                f"Error creating {method}: {e}",
                status="error",
            )
            raise

    def get_all_methods(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get all methods organized by category."""
        return self._steps


# Create global registry instance
PREPROCESSING_REGISTRY = PreprocessingStepRegistry()
