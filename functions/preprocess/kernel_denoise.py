"""
Kernel-based Denoising Methods

This module provides kernel/window smoothing methods for Raman spectra denoising.
It includes a wrapper for ramanspy's Kernel class that fixes numpy API issues.
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
    def create_logs(log_id, source, message, status="info"):
        print(f"[{status.upper()}] {source}: {message}")


class Kernel:
    """
    Wrapper for ramanspy's Kernel denoising that fixes numpy.uniform bug.

    This class wraps rp.preprocessing.denoise.Kernel and applies a monkey-patch
    to fix the incorrect numpy.uniform usage (should be numpy.random.uniform).

    Attributes:
        kernel_type (str): Type of kernel - 'uniform', 'gaussian', or 'triangular'
        kernel_size (int): Size of the kernel window (must be odd)
    """

    def __init__(self, kernel_type: str = "uniform", kernel_size: int = 7):
        """
        Initialize the Kernel denoiser with fixed numpy API.

        Args:
            kernel_type (str): Type of kernel ('uniform', 'gaussian', 'triangular')
            kernel_size (int): Size of kernel window (must be odd)
        """
        if not RAMANSPY_AVAILABLE:
            raise ImportError("ramanspy is required for Kernel denoising")

        self.kernel_type = kernel_type
        self.kernel_size = kernel_size

        # Apply monkey-patch to fix numpy.uniform bug in ramanspy
        self._patch_numpy_uniform()

        # Create the actual ramanspy Kernel instance
        self._kernel = rp.preprocessing.denoise.Kernel(
            kernel_type=kernel_type, kernel_size=kernel_size
        )

    def _patch_numpy_uniform(self):
        """
        Monkey-patch numpy.uniform to numpy.random.uniform if needed.

        This fixes a bug in ramanspy where it incorrectly calls numpy.uniform
        instead of numpy.random.uniform.
        """
        if not hasattr(np, "uniform"):
            # Add the missing attribute by redirecting to the correct location
            np.uniform = np.random.uniform
            create_logs(
                "kernel_denoise",
                "Kernel._patch_numpy_uniform",
                "Applied numpy.uniform -> numpy.random.uniform patch",
                "info",
            )

    def __call__(self, spectrum):
        """
        Apply kernel denoising to the spectrum.

        Args:
            spectrum: Raman spectrum to denoise (ramanspy Spectrum object or array)

        Returns:
            Denoised spectrum
        """
        return self._kernel(spectrum)

    def __repr__(self):
        return (
            f"Kernel(kernel_type='{self.kernel_type}', kernel_size={self.kernel_size})"
        )
