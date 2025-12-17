"""
Custom FABC Implementation - Bypass RamanSPy Bug

This module provides a fixed implementation of FABC (Fully Automatic Baseline Correction)
that works around the bug in ramanspy's baseline wrapper.

Bug Details:
- ramanspy/preprocessing/baseline.py incorrectly passes x_data to np.apply_along_axis()
- numpy.apply_along_axis() does NOT accept x_data as a keyword argument
- This causes TypeError when using ramanspy's FABC wrapper

Solution:
- Call pybaselines.whittaker.fabc directly
- Implement proper SpectralContainer handling
- Compatible with preprocessing registry
"""

import numpy as np
from typing import Optional, Union
import warnings

try:
    import ramanspy as rp
    RAMANSPY_AVAILABLE = True
except ImportError:
    RAMANSPY_AVAILABLE = False
    
try:
    from pybaselines import api
    # FABC is in pybaselines.api module, not whittaker
    PYBASELINES_AVAILABLE = True
    
    # Try to get fabc from api
    try:
        pybaselines_fabc = api.Baseline().fabc
    except AttributeError:
        # Fallback: try getting it from the whittaker module if available
        try:
            from pybaselines import whittaker
            pybaselines_fabc = whittaker.fabc
        except (ImportError, AttributeError):
            PYBASELINES_AVAILABLE = False
            pybaselines_fabc = None
            
except ImportError:
    PYBASELINES_AVAILABLE = False
    pybaselines_fabc = None


class FABCFixed:
    """
    Fixed FABC implementation that bypasses ramanspy's broken wrapper.
    
    Fully Automatic Baseline Correction (FABC) performs automatic baseline
    correction using smoothness and derivative constraints.
    
    Parameters
    ----------
    lam : float, optional
        Smoothing parameter. Larger values produce smoother baselines.
        Default: 1e6
    scale : float, optional
        Scale parameter for noise estimation. If None, automatically estimated.
        Default: None
    num_std : float, optional
        Number of standard deviations for thresholding.
        Default: 3.0
    diff_order : int, optional
        Order of differential matrix. Must be >= 1.
        Default: 2
    min_length : int, optional
        Minimum length for smoothing operations.
        Default: 2
    weights : ndarray, optional
        Optional weight array. If None, uses automatic weighting.
        Default: None
    weights_as_mask : bool, optional
        If True, treats weights as a binary mask.
        Default: False
        
    Notes
    -----
    This is a wrapper around pybaselines.whittaker.fabc that:
    1. Bypasses ramanspy's broken baseline wrapper
    2. Properly handles SpectralContainer objects
    3. Maintains compatibility with preprocessing registry
    
    References
    ----------
    Liu et al., "Fully automatic baseline correction of Raman spectra",
    Journal of Raman Spectroscopy, 2017.
    """
    
    def __init__(self, 
                 lam: float = 1e6,
                 scale: Optional[float] = None,
                 num_std: float = 3.0,
                 diff_order: int = 2,
                 min_length: int = 2,
                 weights: Optional[np.ndarray] = None,
                 weights_as_mask: bool = False):
        """Initialize FABC with parameters."""
        # Check if pybaselines is available (api module)
        try:
            from pybaselines import api
        except ImportError:
            raise ImportError(
                "pybaselines is required for FABC. "
                "Install with: pip install pybaselines"
            )
        
        # CRITICAL: Type conversions for parameters that MUST be specific types
        self.lam = float(lam)  # Ensure float
        self.scale = None if scale is None else float(scale)  # Ensure float or None
        self.num_std = float(num_std)  # Ensure float
        self.diff_order = int(diff_order)  # MUST be int, not float!
        self.min_length = int(min_length)  # MUST be int, not float!
        self.weights = weights  # Can be None or ndarray
        self.weights_as_mask = bool(weights_as_mask)  # Ensure bool
        
        # Create pybaselines Baseline fitter instance
        self._baseline_fitter = None
        
    def _get_baseline_fitter(self, x_data: np.ndarray):
        """Get or create baseline fitter with x_data."""
        # Create new fitter for each call (pybaselines needs x_data at initialization)
        from pybaselines import api
        return api.Baseline(x_data=x_data)
        
    def _process_spectrum(self, spectrum: np.ndarray, x_data: np.ndarray) -> np.ndarray:
        """
        Process a single spectrum to remove baseline.
        
        Parameters
        ----------
        spectrum : ndarray
            1D array containing spectral intensity data
        x_data : ndarray
            1D array containing x-axis values (wavenumbers)
            
        Returns
        -------
        corrected : ndarray
            Baseline-corrected spectrum
        """
        try:
            # Get baseline fitter
            fitter = self._get_baseline_fitter(x_data)
            
            # Call FABC - returns (baseline, params) tuple
            baseline, params = fitter.fabc(
                data=spectrum,
                lam=self.lam,
                scale=self.scale,
                num_std=self.num_std,
                diff_order=self.diff_order,
                min_length=self.min_length,
                weights=self.weights,
                weights_as_mask=self.weights_as_mask
            )
            
            # Return baseline-corrected spectrum
            return spectrum - baseline
            
        except Exception as e:
            warnings.warn(
                f"FABC baseline correction failed: {e}. Returning original spectrum.",
                RuntimeWarning
            )
            return spectrum
    
    def __call__(self, 
                 data: Union[np.ndarray, 'rp.SpectralContainer'],
                 spectral_axis: Optional[np.ndarray] = None) -> Union[np.ndarray, 'rp.SpectralContainer']:
        """
        Apply FABC baseline correction to data.
        
        Handles both SpectralContainer (RamanSPy workflows) and numpy array (sklearn pipelines).
        
        Parameters
        ----------
        data : ndarray or SpectralContainer
            Input data to process
        spectral_axis : ndarray, optional
            Spectral axis data (wavenumbers). Required for numpy arrays.
            Automatically extracted from SpectralContainer.
            
        Returns
        -------
        corrected : same type as input
            Baseline-corrected data in same format as input
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
            axis = spectral_axis
            
            if axis is None:
                raise ValueError(
                    "spectral_axis must be provided when using numpy array input"
                )
        
        # Validate input
        if spectra.ndim == 1:
            # Single spectrum - reshape to 2D
            spectra = spectra.reshape(1, -1)
            was_1d = True
        elif spectra.ndim == 2:
            was_1d = False
        else:
            raise ValueError(f"Input data must be 1D or 2D, got {spectra.ndim}D")
        
        if len(axis) != spectra.shape[1]:
            raise ValueError(
                f"Spectral axis length ({len(axis)}) must match "
                f"number of points in spectrum ({spectra.shape[1]})"
            )
        
        # Process each spectrum
        corrected_spectra = []
        for spectrum in spectra:
            corrected = self._process_spectrum(spectrum, axis)
            corrected_spectra.append(corrected)
        
        corrected_spectra = np.array(corrected_spectra)
        
        # Reshape back if input was 1D
        if was_1d:
            corrected_spectra = corrected_spectra.reshape(-1)
        
        # Return in same format as input
        if is_container:
            if not RAMANSPY_AVAILABLE:
                raise ImportError("ramanspy is required for SpectralContainer operations")
            return rp.SpectralContainer(corrected_spectra, axis)
        else:
            return corrected_spectra
    
    def apply(self, spectra: 'rp.SpectralContainer') -> 'rp.SpectralContainer':
        """
        Apply FABC to ramanspy SpectralContainer.
        
        This method provides compatibility with ramanspy's preprocessing API.
        
        Parameters
        ----------
        spectra : SpectralContainer
            Input spectral data
            
        Returns
        -------
        corrected : SpectralContainer
            Baseline-corrected spectral data
        """
        if not RAMANSPY_AVAILABLE:
            raise ImportError("ramanspy is required for SpectralContainer operations")
        
        return self.__call__(spectra)
    
    def __repr__(self):
        """String representation."""
        return (
            f"FABCFixed(lam={self.lam}, scale={self.scale}, num_std={self.num_std}, "
            f"diff_order={self.diff_order}, min_length={self.min_length})"
        )


# Export for easy import
__all__ = ['FABCFixed']
