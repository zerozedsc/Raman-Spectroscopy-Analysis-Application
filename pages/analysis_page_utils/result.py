"""
Analysis Result Container

This module defines the AnalysisResult class for storing and managing
analysis results with all associated data and visualizations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
from matplotlib.figure import Figure


@dataclass
class AnalysisResult:
    """
    Container for analysis results with comprehensive metadata.

    Attributes:
        category: Analysis category (exploratory, statistical, visualization)
        method_key: Unique method identifier
        method_name: Human-readable method name
        params: Parameters used for analysis
        dataset_names: List of dataset names analyzed
        n_spectra: Total number of spectra analyzed
        execution_time: Time taken to run analysis (seconds)
        summary_text: Brief summary for quick stats display
        detailed_summary: Detailed summary for results panel
        primary_figure: Main visualization figure
        secondary_figure: Secondary visualization figure (optional)
        data_table: Numerical results as DataFrame (optional)
        data_table: Numerical results as DataFrame (optional)
        raw_results: Raw analysis output for further processing
        dataset_data: Original dataset data for spectrum visualization (optional)
    """

    category: str
    method_key: str
    method_name: str
    params: Dict[str, Any]
    dataset_names: List[str]
    n_spectra: int
    execution_time: float
    summary_text: str
    detailed_summary: str
    primary_figure: Optional[Figure] = None
    secondary_figure: Optional[Figure] = None
    data_table: Optional[pd.DataFrame] = None
    raw_results: Dict[str, Any] = field(default_factory=dict)
    dataset_data: Optional[Dict[str, pd.DataFrame]] = None

    def __post_init__(self):
        """Validate result data after initialization."""
        if not self.category:
            raise ValueError("Category cannot be empty")
        if not self.method_key:
            raise ValueError("Method key cannot be empty")
        if self.n_spectra < 1:
            raise ValueError("Number of spectra must be positive")
