"""
Analysis Page Utilities

This module provides utility classes and functions for the Analysis Page,
including analysis method registry, threading, and result management.
"""

from .registry import ANALYSIS_METHODS
from .result import AnalysisResult
from .thread import AnalysisThread
from .widgets import create_parameter_widgets

__all__ = [
    "ANALYSIS_METHODS",
    "AnalysisResult",
    "AnalysisThread",
    "create_parameter_widgets",
]
