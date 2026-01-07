"""
Normalization Methods for Raman Spectra

This module contains various normalization techniques for preprocessing
Raman spectroscopy data.
"""

import numpy as np
try:
    import ramanspy as rp
    RAMANSPY_AVAILABLE = True
except ImportError:
    RAMANSPY_AVAILABLE = False

try:
    from ..utils import create_logs
except ImportError:
    # Fallback logging function
    def create_logs(log_id, source, message, status='info'):
        print(f"[{status.upper()}] {source}: {message}")


class SNV:
    """Standard Normal Variate (SNV) normalization for Raman spectra."""

    def __call__(self, spectra):
        """Apply SNV to numpy array format (for sklearn pipelines)."""
        # spectra: 2D numpy array (n_samples, n_features)
        return np.apply_along_axis(self.snv_normalisation, 1, spectra)

    def apply(self, spectra):
        """Apply SNV to both numpy arrays and ramanspy SpectralContainer format."""
        # Handle numpy arrays directly
        if isinstance(spectra, np.ndarray):
            if spectra.ndim == 1:
                return self.snv_normalisation(spectra)
            else:
                return np.apply_along_axis(self.snv_normalisation, 1, spectra)
        
        # Handle ramanspy SpectralContainer format
        if not RAMANSPY_AVAILABLE:
            raise ImportError("ramanspy is required for SpectralContainer operations")
            
        data = spectra.spectral_data
        # Handle both 1D and 2D data
        if data.ndim == 1:
            snv_data = self.snv_normalisation(data)
        else:
            snv_data = np.apply_along_axis(self.snv_normalisation, 1, data)
        
        # Return new SpectralContainer with processed data
        return rp.SpectralContainer(snv_data, spectra.spectral_axis)

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
            create_logs("snv_warning", "SNV", 
                       "Standard deviation is zero. Returning zero-centered spectrum.", 
                       status='warning')
            return spectrum - mean_val
        elif np.isnan(std_val) or np.isinf(std_val):
            create_logs("snv_error", "SNV", 
                       "Invalid standard deviation (NaN or Inf). Returning original spectrum.", 
                       status='error')
            return spectrum
        elif std_val < 1e-10:
            create_logs("snv_warning", "SNV", 
                       "Very small standard deviation. Results may be unstable.", 
                       status='warning')

        return (spectrum - mean_val) / std_val


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
        """Initialize MSC processor."""
        self.reference_spectrum = None
    
    def fit(self, spectra: np.ndarray) -> 'MSC':
        """
        Fit MSC by calculating the mean reference spectrum.
        
        Args:
            spectra (np.ndarray): 2D array of spectra (n_samples, n_features)
            
        Returns:
            self: Returns self for method chaining
        """
        if spectra.ndim != 2:
            raise ValueError("Spectra must be a 2D array")
            
        # Calculate mean spectrum as reference
        self.reference_spectrum = np.mean(spectra, axis=0)
        return self
    
    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """
        Apply MSC transformation to spectra.
        
        Args:
            spectra (np.ndarray): 2D array of spectra to transform
            
        Returns:
            np.ndarray: MSC-corrected spectra
        """
        if self.reference_spectrum is None:
            raise ValueError("MSC must be fitted before transformation")
            
        if spectra.ndim == 1:
            # Single spectrum
            return self._correct_spectrum(spectra)
        elif spectra.ndim == 2:
            # Multiple spectra
            corrected_spectra = np.zeros_like(spectra)
            for i, spectrum in enumerate(spectra):
                corrected_spectra[i] = self._correct_spectrum(spectrum)
            return corrected_spectra
        else:
            raise ValueError("Spectra must be 1D or 2D array")
    
    def fit_transform(self, spectra: np.ndarray) -> np.ndarray:
        """Fit MSC and transform spectra in one step."""
        return self.fit(spectra).transform(spectra)
    
    def __call__(self, data):
        """
        Apply MSC to data.
        
        Handles both SpectralContainer (RamanSPy workflows) and numpy array (sklearn pipelines).
        
        Args:
            data: SpectralContainer or 2D numpy array
            
        Returns:
            Same type as input with MSC applied
        """
        # Detect input type
        is_container = hasattr(data, 'spectral_data')
        
        if is_container:
            # SpectralContainer input (RamanSPy workflow)
            spectra = data.spectral_data
            axis = data.spectral_axis
        else:
            # numpy array input (sklearn pipeline)
            spectra = data
            axis = None
        
        # Apply MSC
        processed = self.fit_transform(spectra)
        
        # Return in same format as input
        if is_container:
            import ramanspy as rp
            return rp.SpectralContainer(processed, axis)
        else:
            return processed
    
    def apply(self, spectra: 'rp.SpectralContainer') -> 'rp.SpectralContainer':
        """Apply MSC to ramanspy SpectralContainer."""
        if not RAMANSPY_AVAILABLE:
            raise ImportError("ramanspy is required for SpectralContainer operations")
            
        data = spectra.spectral_data
        
        if data.ndim == 1:
            # Single spectrum - use itself as reference
            corrected_data = data  # No correction possible for single spectrum
        else:
            # Multiple spectra
            corrected_data = self.fit_transform(data)
            
        return rp.SpectralContainer(corrected_data, spectra.spectral_axis)
    
    def _correct_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Correct a single spectrum using MSC.
        
        Args:
            spectrum (np.ndarray): 1D spectrum to correct
            
        Returns:
            np.ndarray: MSC-corrected spectrum
        """
        if len(spectrum) != len(self.reference_spectrum):
            raise ValueError("Spectrum and reference must have the same length")
        
        # Linear regression: spectrum = a * reference + b
        # Solve using least squares
        A = np.vstack([self.reference_spectrum, np.ones(len(self.reference_spectrum))]).T
        try:
            a, b = np.linalg.lstsq(A, spectrum, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Fallback: return original spectrum if regression fails
            create_logs("msc_warning", "MSC", 
                       "Linear regression failed. Returning original spectrum.", 
                       status='warning')
            return spectrum
        
        # Correct spectrum: (spectrum - b) / a
        if np.abs(a) < 1e-10:
            create_logs("msc_warning", "MSC", 
                       "Near-zero slope detected. Returning original spectrum.", 
                       status='warning')
            return spectrum
            
        corrected_spectrum = (spectrum - b) / a
        return corrected_spectrum


class Vector:
    """Vector normalization for Raman spectra."""
    
    def __init__(self, norm='l2', pixelwise=False):
        """
        Initialize vector normalization.
        
        Args:
            norm (str): Type of normalization ('l1', 'l2', 'max')
            pixelwise (bool): Whether to normalize each pixel independently
        """
        if norm not in ['l1', 'l2', 'max']:
            raise ValueError("Norm must be 'l1', 'l2', or 'max'")
        self.norm = norm
        self.pixelwise = pixelwise
    
    def __call__(self, spectra):
        """Apply vector normalization."""
        # Handle SpectralContainer objects
        if hasattr(spectra, 'spectral_data'):
            # SpectralContainer format
            normalized_data = []
            for spectrum in spectra.spectral_data:
                normalized_data.append(self._normalize_spectrum(spectrum))
            return spectra.__class__(np.array(normalized_data), spectra.spectral_axis)
        
        # Handle numpy arrays
        if isinstance(spectra, np.ndarray):
            if spectra.ndim == 1:
                return self._normalize_spectrum(spectra)
            elif spectra.ndim == 2:
                return np.array([self._normalize_spectrum(spectrum) for spectrum in spectra])
            else:
                raise ValueError("Spectra must be 1D or 2D array")
        
        raise TypeError("Spectra must be numpy array or SpectralContainer")
    
    def _normalize_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Normalize a single spectrum."""
        if self.norm == 'l1':
            norm_value = np.sum(np.abs(spectrum))
        elif self.norm == 'l2':
            norm_value = np.sqrt(np.sum(spectrum**2))
        elif self.norm == 'max':
            norm_value = np.max(np.abs(spectrum))
        
        if norm_value == 0 or np.isnan(norm_value):
            return spectrum
        
        return spectrum / norm_value
    
    def apply(self, spectra):
        """Apply vector normalization - compatibility method for preview system."""
        return self.__call__(spectra)


class MinMax:
    """Min-Max normalization for Raman spectra."""
    
    def __init__(self, feature_range=(0, 1)):
        """
        Initialize MinMax normalization.
        
        Args:
            feature_range (tuple): Target range for normalization
        """
        self.feature_range = feature_range
        self.data_min = None
        self.data_max = None
    
    def fit(self, spectra: np.ndarray):
        """Fit MinMax scaler to data."""
        if spectra.ndim == 1:
            self.data_min = np.min(spectra)
            self.data_max = np.max(spectra)
        else:
            self.data_min = np.min(spectra, axis=0)
            self.data_max = np.max(spectra, axis=0)
        return self
    
    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        if self.data_min is None or self.data_max is None:
            raise ValueError("MinMax scaler must be fitted before transformation")
        
        # Avoid division by zero
        data_range = self.data_max - self.data_min
        data_range = np.where(data_range == 0, 1, data_range)
        
        # Scale to [0, 1]
        scaled = (spectra - self.data_min) / data_range
        
        # Scale to feature_range
        min_val, max_val = self.feature_range
        return scaled * (max_val - min_val) + min_val
    
    def fit_transform(self, spectra: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(spectra).transform(spectra)
    
    def __call__(self, spectra: np.ndarray) -> np.ndarray:
        """Make class callable for pipeline compatibility."""
        return self.fit_transform(spectra)


class AUC:
    """Area Under Curve normalization for Raman spectra."""
    
    def __call__(self, spectra: np.ndarray) -> np.ndarray:
        """Apply AUC normalization."""
        if spectra.ndim == 1:
            return self._normalize_spectrum(spectra)
        elif spectra.ndim == 2:
            return np.array([self._normalize_spectrum(spectrum) for spectrum in spectra])
        else:
            raise ValueError("Spectra must be 1D or 2D array")
    
    def _normalize_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Normalize a single spectrum by its area."""
        area = np.trapz(np.abs(spectrum))
        if area == 0 or np.isnan(area):
            return spectrum
        return spectrum / area


class MovingAverage:
    """Moving Average smoothing for Raman spectra."""

    def __init__(self, window_length=15):
        """
        Initialize Moving Average smoother.
        
        Args:
            window_length (int): Size of the moving average window
        """
        if not isinstance(window_length, int) or window_length <= 0:
            raise ValueError("Window length must be a positive integer")
        if window_length % 2 == 0:
            window_length += 1  # Ensure odd window length
        self.window_length = window_length

    def __call__(self, spectra):
        """Apply moving average smoothing."""
        return self.apply(spectra)

    def apply(self, spectra):
        """Apply moving average to ramanspy SpectralContainer."""
        if hasattr(spectra, 'spectral_data'):
            # SpectralContainer format
            if not RAMANSPY_AVAILABLE:
                raise ImportError("ramanspy is required for SpectralContainer operations")
                
            smoothed_data = []
            for spectrum in spectra.spectral_data:
                smoothed_data.append(self._smooth_spectrum(spectrum))
            return spectra.__class__(np.array(smoothed_data), spectra.spectral_axis)
        else:
            # NumPy array format
            if spectra.ndim == 1:
                return self._smooth_spectrum(spectra)
            else:
                return np.array([self._smooth_spectrum(spectrum) for spectrum in spectra])
    
    def _smooth_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply moving average to a single spectrum."""
        if len(spectrum) < self.window_length:
            return spectrum
        
        # Simple moving average using convolution
        kernel = np.ones(self.window_length) / self.window_length
        
        # Pad the spectrum to handle edges
        pad_width = self.window_length // 2
        padded_spectrum = np.pad(spectrum, pad_width, mode='edge')
        
        # Apply convolution
        smoothed = np.convolve(padded_spectrum, kernel, mode='valid')
        
        return smoothed
