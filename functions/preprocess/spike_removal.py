"""
Cosmic Ray and Spike Removal Methods

This module contains methods for removing cosmic ray spikes and other
artifacts from Raman spectra.
"""

import numpy as np

try:
    import ramanspy as rp
    from scipy.ndimage import gaussian_filter1d, median_filter
    from scipy.stats import median_abs_deviation as median_absolute_deviation

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from configs.configs import create_logs


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
            raise ValueError("Kernel size must be a positive integer")
        if not isinstance(threshold, (float, int)) or threshold <= 0:
            raise ValueError("Threshold must be a positive number")

        self.kernel = kernel
        self.threshold = threshold

    def __call__(self, data):
        """
        Apply Gaussian despiking to data.

        Handles both SpectralContainer (RamanSPy workflows) and numpy array (sklearn pipelines).

        Args:
            data: SpectralContainer or 2D numpy array where each row is a spectrum

        Returns:
            Same type as input with spikes removed
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

        # Validate array
        if spectra.ndim != 2:
            raise ValueError("Input spectra must be a 2D array")

        # Apply the despiking function along each row (each spectrum)
        processed = np.apply_along_axis(self._despike_spectrum, 1, spectra)

        # Return in same format as input
        if is_container:
            import ramanspy as rp

            return rp.SpectralContainer(processed, axis)
        else:
            return processed

    def apply(self, spectra: "rp.SpectralContainer") -> "rp.SpectralContainer":
        """
        Apply Gaussian despiking to a ramanspy SpectralContainer.

        Args:
            spectra (SpectralContainer): The ramanspy container holding the spectral data.

        Returns:
            SpectralContainer: A new container with the despiked data.
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "SciPy and ramanspy are required for SpectralContainer operations"
            )

        data = spectra.spectral_data

        # Handle both single (1D) and multiple (2D) spectra
        if data.ndim == 1:
            despiked_data = self._despike_spectrum(data)
        else:
            despiked_data = np.apply_along_axis(self._despike_spectrum, 1, data)

        return rp.SpectralContainer(despiked_data, spectra.spectral_axis)

    def _despike_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Core logic to remove spikes from a single 1D spectrum.

        Args:
            spectrum (np.ndarray): A 1D numpy array representing a single spectrum.

        Returns:
            np.ndarray: The despiked spectrum.
        """
        if spectrum is None or spectrum.size == 0:
            return spectrum

        # Create a copy to avoid modifying the original data
        despiked = np.copy(spectrum)

        if not SCIPY_AVAILABLE:
            # Simple fallback without scipy
            return despiked

        # 1. Smooth the spectrum using a Gaussian filter
        # The `sigma` of the filter is controlled by our `kernel` parameter.
        smoothed_spectrum = gaussian_filter1d(despiked, sigma=self.kernel)

        # 2. Calculate the difference (residual)
        residual = despiked - smoothed_spectrum

        # 3. Estimate noise using Median Absolute Deviation (MAD) for robustness
        # The factor 0.6745 makes MAD an unbiased estimator for the standard deviation
        # for normally distributed data.
        mad = np.median(np.abs(residual - np.median(residual)))
        if mad < 1e-9:
            return despiked

        # 4. Calculate the modified Z-score for each point
        # This score tells us how many standard deviations away each point is.
        modified_z_score = 0.6745 * residual / mad

        # 5. Identify spike locations
        # Spikes are points where the absolute Z-score exceeds the threshold.
        spike_indices = np.where(np.abs(modified_z_score) > self.threshold)[0]

        # 6. Replace spikes with the value from the smoothed spectrum
        # This is a simple and effective way to correct the spike.
        for i in spike_indices:
            despiked[i] = smoothed_spectrum[i]

        return despiked


class MedianDespike:
    """
    Median filter-based cosmic ray removal for Raman spectra.

    This method uses median filtering to identify and remove cosmic ray spikes.
    It's particularly effective for narrow, high-intensity spikes that are
    characteristic of cosmic ray events.

    Attributes:
        kernel_size (int): Size of median filter kernel (must be odd)
        threshold (float): Threshold for spike detection (in MAD units)
        max_iter (int): Maximum iterations for iterative spike removal
    """

    def __init__(self, kernel_size: int = 7, threshold: float = 3.5, max_iter: int = 3):
        """
        Initialize MedianDespike processor.

        Args:
            kernel_size (int): Size of median filter kernel (default: 7, must be odd)
            threshold (float): Threshold for spike detection in MAD units (default: 3.5)
            max_iter (int): Number of iterations for spike removal (default: 3).
                           Multiple iterations help remove clustered cosmic rays.
        """
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError("Kernel size must be a positive integer")
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
        if not isinstance(threshold, (float, int)) or threshold <= 0:
            raise ValueError("Threshold must be a positive number")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")

        self.kernel_size = kernel_size
        self.threshold = threshold
        self.max_iter = max_iter

    def __call__(self, data):
        """
        Apply median despiking to data.

        Handles both SpectralContainer (RamanSPy workflows) and numpy array (sklearn pipelines).

        Args:
            data: SpectralContainer or 2D numpy array (n_spectra, n_wavenumbers)

        Returns:
            Same type as input with spikes removed
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

        # Validate array
        if spectra.ndim != 2:
            raise ValueError("Input spectra must be a 2D array")

        # Process array
        processed = np.apply_along_axis(self._despike_spectrum, 1, spectra)

        # Return in same format as input
        if is_container:
            import ramanspy as rp

            return rp.SpectralContainer(processed, axis)
        else:
            return processed

    def _despike_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Remove spikes from a single spectrum using iterative median filtering.

        Args:
            spectrum (np.ndarray): 1D spectrum to despike

        Returns:
            np.ndarray: Despiked spectrum
        """
        if spectrum is None or spectrum.size == 0:
            return spectrum

        if not SCIPY_AVAILABLE:
            # Simple fallback without scipy
            return spectrum.copy()

        # Create a copy to avoid modifying the original
        despiked = spectrum.copy()

        # Iterative spike removal for clustered cosmic rays
        for iteration in range(self.max_iter):
            # Apply median filter
            filtered = median_filter(despiked, size=self.kernel_size)

            # Calculate residual
            residual = despiked - filtered

            # Calculate MAD-based threshold
            mad = np.median(np.abs(residual - np.median(residual)))
            if mad < 1e-9:
                break  # No more spikes to remove

            # Identify spikes
            modified_z_score = 0.6745 * residual / mad
            spike_indices = np.where(np.abs(modified_z_score) > self.threshold)[0]

            if len(spike_indices) == 0:
                break  # No more spikes found

            # Replace spikes with filtered values
            despiked[spike_indices] = filtered[spike_indices]

        return despiked

    def apply(self, spectra: "rp.SpectralContainer") -> "rp.SpectralContainer":
        """Apply median despiking to ramanspy SpectralContainer."""
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "SciPy and ramanspy are required for SpectralContainer operations"
            )

        data = spectra.spectral_data

        # Handle both 1D and 2D data
        if data.ndim == 1:
            despiked_data = self._despike_spectrum(data)
        else:
            despiked_data = np.apply_along_axis(self._despike_spectrum, 1, data)

        return rp.SpectralContainer(despiked_data, spectra.spectral_axis)
