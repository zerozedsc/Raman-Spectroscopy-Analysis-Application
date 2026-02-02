"""
Analysis Thread Worker

This module provides a QThread worker for running analysis methods
in the background to keep the UI responsive.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import time
import traceback
from typing import Dict, Any
import pandas as pd

from PySide6.QtCore import QThread, Signal

from .result import AnalysisResult
from .registry import get_method_info
from configs.configs import create_logs

# Import analysis method implementations
from .methods import (
    perform_pca_analysis,
    perform_umap_analysis,
    perform_tsne_analysis,
    perform_hierarchical_clustering,
    perform_kmeans_clustering,
    perform_pls_da_analysis,
    perform_mcr_als_analysis,
    perform_nmf_analysis,
    perform_ica_analysis,
    perform_outlier_detection,
    perform_spectral_comparison,
    perform_peak_analysis,
    perform_correlation_analysis,
    perform_anova_test,
    perform_pairwise_statistical_tests,
    perform_band_ratio_analysis,
    create_derivative_spectra_plot,
    create_spectral_heatmap,
    create_waterfall_plot,
    create_correlation_heatmap,
    create_peak_scatter,
)


class _AnalysisCancelledError(Exception):
    """Internal exception used to abort analysis when cancellation is requested."""

    pass


class AnalysisThread(QThread):
    """Worker thread for running analysis methods in background."""

    progress = Signal(int)
    finished = Signal(AnalysisResult)
    error = Signal(str, str)  # error_message, traceback_str
    cancelled = Signal()

    def __init__(
        self,
        category: str,
        method_key: str,
        params: Dict[str, Any],
        dataset_data: Dict[str, pd.DataFrame],
    ):
        super().__init__()
        self.category = category
        self.method_key = method_key
        self.params = params
        self.dataset_data = dataset_data
        self._is_cancelled = False

    def run(self):
        """Execute the analysis method."""
        try:
            start_time = time.time()

            if self.is_interruption_requested():
                raise _AnalysisCancelledError()

            # Get method info
            method_info = get_method_info(self.category, self.method_key)
            function_name = method_info["function"]

            # Map function names to actual functions
            function_map = {
                "perform_pca_analysis": perform_pca_analysis,
                "perform_umap_analysis": perform_umap_analysis,
                "perform_tsne_analysis": perform_tsne_analysis,
                "perform_hierarchical_clustering": perform_hierarchical_clustering,
                "perform_kmeans_clustering": perform_kmeans_clustering,
                "perform_pls_da_analysis": perform_pls_da_analysis,
                "perform_mcr_als_analysis": perform_mcr_als_analysis,
                "perform_nmf_analysis": perform_nmf_analysis,
                "perform_ica_analysis": perform_ica_analysis,
                "perform_outlier_detection": perform_outlier_detection,
                "perform_spectral_comparison": perform_spectral_comparison,
                "perform_peak_analysis": perform_peak_analysis,
                "perform_correlation_analysis": perform_correlation_analysis,
                "perform_anova_test": perform_anova_test,
                "perform_pairwise_statistical_tests": perform_pairwise_statistical_tests,
                "perform_band_ratio_analysis": perform_band_ratio_analysis,
                "create_derivative_spectra_plot": create_derivative_spectra_plot,
                "create_spectral_heatmap": create_spectral_heatmap,
                "create_waterfall_plot": create_waterfall_plot,
                "create_correlation_heatmap": create_correlation_heatmap,
                "create_peak_scatter": create_peak_scatter,
            }

            if function_name not in function_map:
                raise ValueError(f"Analysis function '{function_name}' not found")

            analysis_function = function_map[function_name]

            # Update progress
            self.progress.emit(10)

            if self.is_interruption_requested():
                raise _AnalysisCancelledError()

            # Prepare data
            dataset_names = list(self.dataset_data.keys())
            n_spectra = sum(df.shape[1] for df in self.dataset_data.values())

            self.progress.emit(20)

            if self.is_interruption_requested():
                raise _AnalysisCancelledError()

            # Run analysis
            create_logs(
                "AnalysisThread",
                "run_analysis",
                f"Running {method_info['name']} with {n_spectra} spectra",
                status="info",
            )

            def _guarded_progress(p: int):
                if self.is_interruption_requested():
                    raise _AnalysisCancelledError()
                self._update_progress(p)

            # Execute the analysis function
            result = analysis_function(
                dataset_data=self.dataset_data,
                params=self.params,
                progress_callback=_guarded_progress,
            )

            if self.is_interruption_requested():
                raise _AnalysisCancelledError()

            execution_time = time.time() - start_time

            # Create AnalysisResult object
            raw_results = result.get("raw_results", {})

            # Store additional figures in raw_results for PCA multi-tab visualization
            if "scree_figure" in result:
                raw_results["scree_figure"] = result["scree_figure"]
            if "loadings_figure" in result:
                raw_results["loadings_figure"] = result["loadings_figure"]
            if "biplot_figure" in result:
                raw_results["biplot_figure"] = result["biplot_figure"]
            if "cumulative_variance_figure" in result:
                raw_results["cumulative_variance_figure"] = result[
                    "cumulative_variance_figure"
                ]
            if "distributions_figure" in result:
                raw_results["distributions_figure"] = result["distributions_figure"]

            analysis_result = AnalysisResult(
                category=self.category,
                method_key=self.method_key,
                method_name=method_info["name"],
                params=self.params,
                dataset_names=dataset_names,
                n_spectra=n_spectra,
                execution_time=execution_time,
                summary_text=result.get("summary_text", "Analysis completed"),
                detailed_summary=result.get("detailed_summary", ""),
                primary_figure=result.get("primary_figure"),
                secondary_figure=result.get("secondary_figure"),
                data_table=result.get("data_table"),
                raw_results=raw_results,
                dataset_data=self.dataset_data,
            )

            self.progress.emit(100)
            self.finished.emit(analysis_result)

            create_logs(
                "AnalysisThread",
                "run_analysis",
                f"Analysis completed in {execution_time:.2f}s",
                status="info",
            )

        except _AnalysisCancelledError:
            create_logs(
                "AnalysisThread",
                "run_analysis",
                "Analysis cancelled by user",
                status="info",
            )
            self.cancelled.emit()

        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"Analysis failed: {str(e)}\n{tb_str}"
            create_logs("AnalysisThread", "run_analysis", error_msg, status="error")
            self.error.emit(str(e), tb_str)

    def _update_progress(self, progress: int):
        """
        Update progress callback for analysis functions.

        Args:
            progress: Progress value (0-100)
        """
        # Map analysis progress (20-90) to thread progress
        thread_progress = 20 + int((progress / 100) * 70)
        self.progress.emit(thread_progress)

    def cancel(self):
        """
        Cancel the running analysis using cooperative interruption.

        This method replaces the unsafe terminate() approach with Qt's
        recommended cooperative interruption pattern to prevent:
        - Data corruption during NumPy/SciPy operations
        - Memory leaks from interrupted Matplotlib rendering
        - Crashes from inconsistent thread state
        """
        # IMPORTANT: Do not block the UI thread here. Cancellation is best-effort and
        # must be handled cooperatively by analysis functions via progress callbacks.
        self._is_cancelled = True
        self.requestInterruption()  # Set Qt interruption flag

    def is_interruption_requested(self) -> bool:
        """
        Check if cancellation was requested.

        Analysis methods should call this periodically and abort cleanly
        if True is returned.

        Returns:
            True if cancellation was requested, False otherwise
        """
        return self._is_cancelled or self.isInterruptionRequested()
