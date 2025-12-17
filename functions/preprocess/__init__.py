"""
Enhanced Raman Preprocessing Package

A comprehensive preprocessing package for Raman spectroscopy data,
specifically designed for medical applications and disease detection.

This package provides:
- Cosmic ray and spike removal
- Wavenumber and intensity calibration  
- Baseline correction methods (including advanced Butterworth filtering)
- Derivative processing
- Normalization techniques (including advanced cross-platform methods)
- Feature engineering (peak-ratio features)
- Deep learning-based preprocessing (CDAE)
- Dynamic pipeline building
- Registry system for UI integration
"""

# Import all preprocessing classes and functions
from .spike_removal import Gaussian, MedianDespike
from .calibration import WavenumberCalibration, IntensityCalibration
from .normalization import SNV, MSC, MovingAverage
from .baseline import (
    BaselineCorrection, MultiScaleConv1D, Transformer1DBaseline,
    LightweightTransformer1D
)
from .derivatives import Derivative
from .pipeline import RamanPipeline, EnhancedRamanPipeline
from .registry import PreprocessingStepRegistry
from .smoothing import SavitzkyGolaySmoothing, MeanCentering

# Import advanced methods
from .advanced_normalization import (
    QuantileNormalization, RankTransform, ProbabilisticQuotientNormalization
)
from .feature_engineering import PeakRatioFeatures
from .advanced_baseline import ButterworthHighPass

# Try to import deep learning module (requires PyTorch)
try:
    from .deep_learning import ConvolutionalAutoencoder
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# Create global registry instance
PREPROCESSING_REGISTRY = PreprocessingStepRegistry()

# Export all main classes and the registry
__all__ = [
    # Spike removal
    'Gaussian', 'MedianDespike',
    
    # Calibration
    'WavenumberCalibration', 'IntensityCalibration',
    
    # Smoothing
    'SavitzkyGolaySmoothing', 'MeanCentering',
    
    # Normalization (basic)
    'SNV', 'MSC', 'MovingAverage',
    
    # Normalization (advanced)
    'QuantileNormalization', 'RankTransform', 'ProbabilisticQuotientNormalization',
    
    # Baseline correction
    'BaselineCorrection', 'MultiScaleConv1D', 'Transformer1DBaseline',
    'LightweightTransformer1D', 'ButterworthHighPass',
    
    # Derivatives
    'Derivative',
    
    # Feature Engineering
    'PeakRatioFeatures',
    
    # Pipeline
    'RamanPipeline', 'EnhancedRamanPipeline',
    
    # Registry
    'PreprocessingStepRegistry', 'PREPROCESSING_REGISTRY',
    
    # Deep learning (if available)
    'DL_AVAILABLE'
]

# Add deep learning to exports if available
if DL_AVAILABLE:
    __all__.append('ConvolutionalAutoencoder')

# Package metadata
__version__ = "1.1.0"
__author__ = "MUHAMMAD HELMI BIN ROZAIN"
__description__ = "Medical-grade Raman spectroscopy preprocessing package with advanced cross-platform methods"
