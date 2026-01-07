"""
Baseline Correction Methods for Raman Spectra

This module contains comprehensive baseline correction methods including
least squares, polynomial, and advanced deep learning approaches.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import ramanspy as rp
    from scipy import ndimage
    from scipy.ndimage import grey_opening

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.nn import TransformerEncoder, TransformerEncoderLayer

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ..utils import create_logs
except ImportError:
    # Fallback logging function
    def create_logs(log_id, source, message, status="info"):
        print(f"[{status.upper()}] {source}: {message}")


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
        try:
            # Import all ramanspy baseline methods
            from ramanspy.preprocessing.baseline import (
                ASLS,
                ARPLS,
                AIRPLS,
                ASPLS,
                IASLS,
                DRPLS,
                IARPLS,
                Poly,
                ModPoly,
                PenalisedPoly,
                IModPoly,
                Goldindec,
                IRSQR,
                CornerCutting,
                FABC,
            )

            registry = {
                # Asymmetric Least Squares methods
                "ASLS": {
                    "class": ASLS,
                    "category": "least_squares",
                    "description": "Asymmetric Least Squares baseline correction",
                    "default_params": {"lam": 1e6, "p": 0.01},
                    "suitable_for": ["general", "biological", "fluorescence"],
                    "computational_cost": "medium",
                },
                "ARPLS": {
                    "class": ARPLS,
                    "category": "least_squares",
                    "description": "Asymptotically Reweighted Penalized Least Squares",
                    "default_params": {"lam": 1e6},
                    "suitable_for": ["biological", "fluorescence", "robust"],
                    "computational_cost": "medium",
                },
                "AIRPLS": {
                    "class": AIRPLS,
                    "category": "least_squares",
                    "description": "Adaptive Iteratively Reweighted Penalized Least Squares",
                    "default_params": {"lam": 1e6},
                    "suitable_for": ["biological", "automatic", "robust"],
                    "computational_cost": "medium",
                },
                "ASPLS": {
                    "class": ASPLS,
                    "category": "least_squares",
                    "description": "Asymmetric Smoothing Penalized Least Squares",
                    "default_params": {"lam": 1e6, "alpha": 0.5},
                    "suitable_for": ["biological", "peak_preserving"],
                    "computational_cost": "medium",
                },
                "IASLS": {
                    "class": IASLS,
                    "category": "least_squares",
                    "description": "Improved Asymmetric Least Squares",
                    "default_params": {"lam": 1e6, "p": 0.01, "lam_1": 1e4},
                    "suitable_for": ["biological", "fluorescence"],
                    "computational_cost": "high",
                },
                "DRPLS": {
                    "class": DRPLS,
                    "category": "least_squares",
                    "description": "Doubly Reweighted Penalized Least Squares",
                    "default_params": {"lam": 1e6, "eta": 0.5},
                    "suitable_for": ["robust", "biological"],
                    "computational_cost": "high",
                },
                "IARPLS": {
                    "class": IARPLS,
                    "category": "least_squares",
                    "description": "Improved Asymptotically Reweighted Penalized Least Squares",
                    "default_params": {"lam": 1e6},
                    "suitable_for": ["robust", "biological"],
                    "computational_cost": "high",
                },
                # Polynomial methods
                "Poly": {
                    "class": Poly,
                    "category": "polynomial",
                    "description": "Polynomial baseline fitting",
                    "default_params": {"poly_order": 3},
                    "suitable_for": ["simple", "fast"],
                    "computational_cost": "low",
                    "requires_region": True,
                },
                "ModPoly": {
                    "class": ModPoly,
                    "category": "polynomial",
                    "description": "Modified Polynomial baseline correction",
                    "default_params": {"poly_order": 3},
                    "suitable_for": ["simple", "fast"],
                    "computational_cost": "low",
                },
                "PenalisedPoly": {
                    "class": PenalisedPoly,
                    "category": "polynomial",
                    "description": "Penalized Polynomial baseline correction",
                    "default_params": {"poly_order": 3},
                    "suitable_for": ["robust", "biological"],
                    "computational_cost": "medium",
                },
                "IModPoly": {
                    "class": IModPoly,
                    "category": "polynomial",
                    "description": "Improved Modified Polynomial",
                    "default_params": {"poly_order": 3},
                    "suitable_for": ["simple", "robust"],
                    "computational_cost": "medium",
                },
                # Specialized methods
                "Goldindec": {
                    "class": Goldindec,
                    "category": "specialized",
                    "description": "Goldindec baseline correction",
                    "default_params": {"poly_order": 3},
                    "suitable_for": ["specialized"],
                    "computational_cost": "medium",
                },
                "IRSQR": {
                    "class": IRSQR,
                    "category": "specialized",
                    "description": "Iterative Reweighted Quantile Regression",
                    "default_params": {"lam": 1e6, "quantile": 0.05},
                    "suitable_for": ["robust", "peak_preserving"],
                    "computational_cost": "high",
                },
                "CornerCutting": {
                    "class": CornerCutting,
                    "category": "specialized",
                    "description": "Corner Cutting baseline correction",
                    "default_params": {},
                    "suitable_for": ["fast", "simple"],
                    "computational_cost": "low",
                },
                "FABC": {
                    "class": FABC,
                    "category": "specialized",
                    "description": "Fully Automated Baseline Correction",
                    "default_params": {"lam": 1e6},
                    "suitable_for": ["automatic", "biological"],
                    "computational_cost": "medium",
                },
            }

        except ImportError as e:
            create_logs(
                "baseline_import_error",
                "BaselineCorrection",
                f"Could not import ramanspy baseline methods: {e}. Using fallback registry.",
                status="warning",
            )
            # Fallback registry with basic methods
            registry = {
                "MultiScaleConv1D": {
                    "class": MultiScaleConv1D,
                    "category": "convolutional",
                    "description": "Multi-scale convolutional baseline correction",
                    "default_params": {"kernel_sizes": [5, 11, 21, 41]},
                    "suitable_for": ["general", "biological"],
                    "computational_cost": "medium",
                }
            }

        return registry

    def _build_presets(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Build preset configurations for different use cases."""
        return {
            "biological": {
                "ARPLS": {"lam": 1e6},
                "AIRPLS": {"lam": 1e6},
                "ASPLS": {"lam": 1e6, "alpha": 0.5},
            },
            "fluorescence": {
                "ARPLS": {"lam": 1e7},
                "ASLS": {"lam": 1e7, "p": 0.001},
                "IASLS": {"lam": 1e7, "p": 0.001, "lam_1": 1e5},
            },
            "fast": {
                "ASLS": {"lam": 1e5, "p": 0.01},
                "ModPoly": {"poly_order": 2},
                "CornerCutting": {},
            },
            "robust": {
                "AIRPLS": {"lam": 1e6},
                "IARPLS": {"lam": 1e6},
                "IRSQR": {"lam": 1e6, "quantile": 0.05},
            },
        }

    def get_method(
        self, method_name: str, custom_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Get a baseline correction method with specified parameters."""
        if method_name not in self._methods_registry:
            available = list(self._methods_registry.keys())
            raise ValueError(
                f"Method '{method_name}' not available. Available: {available}"
            )

        method_info = self._methods_registry[method_name]
        method_class = method_info["class"]
        params = method_info["default_params"].copy()

        # Handle region requirement for polynomial methods
        if method_info.get("requires_region", False):
            if self.region is None:
                raise ValueError(f"Method '{method_name}' requires a region to be set.")
            params["regions"] = [self.region]

        # Override with custom parameters
        if custom_params:
            params.update(custom_params)

        try:
            return method_class(**params)
        except Exception as e:
            create_logs(
                "baseline_method_error",
                "BaselineCorrection",
                f"Error initializing {method_name}: {e}. Using defaults.",
                status="error",
            )
            return method_class(**method_info["default_params"])

    def list_methods(self) -> List[str]:
        """List all available baseline correction methods."""
        return list(self._methods_registry.keys())

    def get_method_info(self, method_name: str) -> Dict[str, Any]:
        """Get information about a specific method."""
        if method_name not in self._methods_registry:
            raise ValueError(f"Method '{method_name}' not found.")
        return self._methods_registry[method_name].copy()


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
        if not hasattr(spectra, "spectral_data"):
            # Handle numpy arrays
            return self(spectra)

        data = spectra.spectral_data

        if data.ndim == 1:
            corrected_data = self._correct_spectrum(data)
        else:
            corrected_data = np.array(
                [self._correct_spectrum(spectrum) for spectrum in data]
            )

        return rp.SpectralContainer(corrected_data, spectra.spectral_axis)

    def _correct_spectrum(self, spectrum):
        """Apply multi-scale convolutional baseline correction to a single spectrum."""
        spectrum = np.asarray(spectrum)
        corrected = spectrum.copy()

        for _ in range(self.iterations):
            baseline = self._estimate_baseline(corrected)
            corrected = spectrum - baseline

            # Ensure no extremely negative values
            corrected = np.maximum(corrected, -np.abs(spectrum).max() * 0.1)

        return corrected

    def _estimate_baseline(self, spectrum):
        """Estimate baseline using multi-scale convolution."""
        baselines = []

        for kernel_size in self.kernel_sizes:
            # Create a simple averaging kernel
            kernel = np.ones(kernel_size) / kernel_size
            baseline = self._convolve_with_padding(spectrum, kernel)
            baselines.append(baseline)

        # Combine baselines with weights
        combined_baseline = np.zeros_like(spectrum)
        for baseline, weight in zip(baselines, self.weights):
            combined_baseline += weight * baseline

        return combined_baseline

    def _convolve_with_padding(self, signal, kernel):
        """Apply convolution with appropriate padding."""
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


# PyTorch-based baseline methods (only available if PyTorch is installed)
if TORCH_AVAILABLE:

    class Transformer1DBaseline:
        """
        Transformer-based baseline correction for Raman spectra using PyTorch.

        This method uses a transformer architecture to learn and predict
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
            """Initialize Transformer1DBaseline correction."""
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

            self.model = None

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
            """Apply transformer-based baseline correction to a single spectrum."""
            spectrum = np.asarray(spectrum)

            # Initialize and train model if not already done
            if self.model is None:
                self._initialize_model(len(spectrum))
                self._train_model(spectrum)

            # Predict baseline
            baseline = self._predict_baseline(spectrum)
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
            """Train the transformer model on the input spectrum."""
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
                predicted_baseline = self.model(input_tensor)
                loss = criterion(predicted_baseline, target_tensor)
                loss.backward()
                optimizer.step()

        def _predict_baseline(self, spectrum):
            """Predict baseline using trained model."""
            self.model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(spectrum).unsqueeze(0).to(self.device)
                baseline = self.model(input_tensor)
                return baseline.cpu().numpy().squeeze()

        def _create_pseudo_baseline(self, spectrum):
            """Create pseudo-baseline for training using multiple smoothing techniques."""
            # Heavy smoothing with large kernel
            kernel_size = max(21, len(spectrum) // 20)
            if kernel_size % 2 == 0:
                kernel_size += 1

            baseline1 = np.convolve(
                spectrum, np.ones(kernel_size) / kernel_size, mode="same"
            )

            if SCIPY_AVAILABLE:
                # Percentile-based smoothing
                baseline2 = ndimage.percentile_filter(
                    spectrum, percentile=10, size=kernel_size
                )
                # Morphological opening
                baseline3 = grey_opening(spectrum, size=kernel_size // 3)
                baseline = (baseline1 + baseline2 + baseline3) / 3
            else:
                baseline = baseline1

            return baseline

    class BaselineTransformer(nn.Module):
        """Transformer model for baseline prediction."""

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

            # Positional encoding
            max_len = max(spectrum_length + 1000, 10000)
            self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=max_len)

            # Transformer encoder
            encoder_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Output projection
            self.output_projection = nn.Linear(d_model, 1)

        def forward(self, x):
            # x shape: (batch_size, sequence_length)
            batch_size, seq_len = x.shape

            # Reshape for input projection
            x = x.unsqueeze(-1)  # (batch_size, sequence_length, 1)
            x = self.input_projection(x)  # (batch_size, sequence_length, d_model)

            # Add positional encoding
            x = self.pos_encoding(x)

            # Apply transformer
            x = self.transformer(x)

            # Output projection
            x = self.output_projection(x)  # (batch_size, sequence_length, 1)
            x = x.squeeze(-1)  # (batch_size, sequence_length)

            return x

    class PositionalEncoding(nn.Module):
        """Positional encoding for transformer."""

        def __init__(self, d_model, dropout=0.1, max_len=10000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer("pe", pe)

        def forward(self, x):
            x = x + self.pe[: x.size(1), :].transpose(0, 1)
            return self.dropout(x)

    class LightweightTransformer1D:
        """Lightweight transformer-based baseline correction."""

        def __init__(self, d_model=32, nhead=4, num_layers=2, epochs=20):
            """Initialize lightweight transformer."""
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is required for LightweightTransformer1D")

            self.d_model = d_model
            self.nhead = nhead
            self.num_layers = num_layers
            self.epochs = epochs
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = None

        def __call__(self, spectra):
            """Apply lightweight transformer baseline correction."""
            if spectra.ndim == 1:
                return self._correct_spectrum(spectra)
            else:
                return np.array(
                    [self._correct_spectrum(spectrum) for spectrum in spectra]
                )

        def _correct_spectrum(self, spectrum):
            """Correct a single spectrum using lightweight transformer."""
            # Simple implementation - use heavy smoothing as baseline
            kernel_size = max(15, len(spectrum) // 30)
            if kernel_size % 2 == 0:
                kernel_size += 1

            baseline = np.convolve(
                spectrum, np.ones(kernel_size) / kernel_size, mode="same"
            )
            return spectrum - baseline

else:
    # Placeholder classes when PyTorch is not available
    class Transformer1DBaseline:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for Transformer1DBaseline")

    class LightweightTransformer1D:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LightweightTransformer1D")
