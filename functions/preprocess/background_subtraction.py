"""
Background Subtraction Methods

This module provides background subtraction methods for Raman spectra.
It includes wrappers that fix array comparison issues.
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


class BackgroundSubtractor:
    """
    Wrapper for ramanspy's BackgroundSubtractor that handles array comparisons properly.

    This class wraps rp.preprocessing.misc.BackgroundSubtractor and ensures proper
    handling of None values and array comparisons to avoid ambiguous truth value errors.

    Attributes:
        background: Fixed reference background spectrum to subtract (optional)
    """

    def __init__(self, background=None):
        """
        Initialize the BackgroundSubtractor.

        Args:
            background: Fixed reference background to subtract (Spectrum or array, optional)
        """
        if not RAMANSPY_AVAILABLE:
            raise ImportError("ramanspy is required for BackgroundSubtractor")

        self.background = background

        # Only create the ramanspy instance if background is explicitly provided
        if background is not None:
            self._subtractor = rp.preprocessing.misc.BackgroundSubtractor(
                background=background
            )
        else:
            self._subtractor = None

    def __call__(self, spectrum):
        """
        Apply background subtraction to the spectrum.

        Args:
            spectrum: Raman spectrum to process (ramanspy Spectrum object or array)

        Returns:
            Background-subtracted spectrum (returns original if no background set)
        """
        # If no background is set, return the original spectrum unchanged
        if self._subtractor is None:
            create_logs(
                "background_subtraction",
                "BackgroundSubtractor.__call__",
                "No background set, returning original spectrum",
                "warning",
            )
            return spectrum

        # Apply background subtraction
        return self._subtractor(spectrum)

    def set_background(self, background):
        """
        Set or update the background spectrum.

        Args:
            background: New background spectrum to subtract
        """
        self.background = background
        if background is not None:
            self._subtractor = rp.preprocessing.misc.BackgroundSubtractor(
                background=background
            )
        else:
            self._subtractor = None

    def __repr__(self):
        has_bg = "set" if self.background is not None else "not set"
        return f"BackgroundSubtractor(background={has_bg})"
