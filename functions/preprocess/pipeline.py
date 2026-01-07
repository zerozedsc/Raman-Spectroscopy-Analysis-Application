"""
Raman Preprocessing Pipeline Module

This module contains the main pipeline classes for processing Raman spectra,
including progress tracking and comprehensive metadata collection.
"""

import os
import time
import pickle as pkl
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import ramanspy as rp
    import matplotlib.pyplot as plt

    RAMANSPY_AVAILABLE = True
except ImportError:
    RAMANSPY_AVAILABLE = False

try:
    from ..utils import create_logs, CURRENT_DIR
except ImportError:
    # Fallback logging function and current directory
    def create_logs(log_id, source, message, status="info"):
        print(f"[{status.upper()}] {source}: {message}")

    CURRENT_DIR = os.getcwd()


class RamanPipeline:
    """
    A class to handle the preprocessing of Raman spectral data.

    Attributes:
    -----------
    region : tuple
        The range of wavenumbers to consider for analysis.
    """

    def __init__(self, region: tuple[int, int] = (1050, 1700)):
        """
        Initialize the RamanPipeline.

        Parameters
        ----------
        region : tuple[int, int]
            Wavenumber region for analysis (default: 1050-1700 cm⁻¹)
        """
        self.region = region

    def pipeline_hirschsprung_multi(
        self,
        hirsch_dfs: List[pd.DataFrame],
        normal_dfs: List[pd.DataFrame],
        wavenumber_rowname: str = "wavenumber",
        region: Tuple[int, int] = (1050, 1700),
    ) -> Tuple[np.ndarray, list, np.ndarray, pd.DataFrame]:
        """
        Preprocessing pipeline specifically for Hirschsprung disease multi-sample analysis.

        Parameters
        ----------
        hirsch_dfs : List[pd.DataFrame]
            List of DataFrames containing Hirschsprung spectra.
        normal_dfs : List[pd.DataFrame]
            List of DataFrames containing normal spectra.
        wavenumber_rowname : str
            Column name for wavenumber values.
        region : Tuple[int, int]
            Wavenumber region to analyze.

        Returns
        -------
        Tuple containing:
            processed_data : np.ndarray
                The preprocessed spectral data.
            labels : list
                The labels for each spectrum.
            wavenumbers : np.ndarray
                The wavenumbers corresponding to the spectral data.
            merged_df : pd.DataFrame
                The merged DataFrame containing all spectra.
        """
        if not RAMANSPY_AVAILABLE:
            raise ImportError("ramanspy is required for pipeline processing")

        # Handle empty input cases
        if not hirsch_dfs and not normal_dfs:
            raise ValueError(
                "Both hirsch_dfs and normal_dfs are empty. At least one must be provided."
            )
        elif hirsch_dfs:
            wavenumbers = hirsch_dfs[0]["wavenumber"].values
        else:
            wavenumbers = normal_dfs[0]["wavenumber"].values

        # Concatenate all hirsch and normal DataFrames (drop wavenumber, keep only intensity columns)
        all_hirsch = (
            [df.drop(wavenumber_rowname, axis=1) for df in hirsch_dfs]
            if hirsch_dfs
            else []
        )
        all_normal = (
            [df.drop(wavenumber_rowname, axis=1) for df in normal_dfs]
            if normal_dfs
            else []
        )
        merged_df = pd.concat(all_hirsch + all_normal, axis=1)

        intensities = merged_df.values.T  # shape: (n_samples, n_wavenumbers)

        # Labels
        labels = []
        if hirsch_dfs:
            labels += ["hirsch"] * sum(df.shape[1] - 1 for df in hirsch_dfs)
        if normal_dfs:
            labels += ["normal"] * sum(df.shape[1] - 1 for df in normal_dfs)

        # Preprocessing pipeline
        pipeline = rp.preprocessing.Pipeline(
            [
                rp.preprocessing.misc.Cropper(region=region),
                rp.preprocessing.despike.WhitakerHayes(),
                rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
                rp.preprocessing.baseline.ASPLS(lam=1e5, tol=0.01),
                rp.preprocessing.normalise.Vector(),
            ]
        )
        spectra = rp.SpectralContainer(intensities, wavenumbers)
        data = pipeline.apply(spectra)

        return data.spectral_data, labels, data.spectral_axis, merged_df

    def preprocess(
        self,
        dfs: List[pd.DataFrame],
        label: str,
        wavenumber_col: str = "wavenumber",
        intensity_cols: Optional[List[str]] = None,
        region: Tuple[int, int] = (1050, 1700),
        preprocessing_steps: Optional[List[Callable]] = None,
        visualize_steps: bool = False,
        show_parameters_in_title: bool = False,
        max_plot_visualize_steps: int = 10,
        save_pkl: bool = False,
        save_pkl_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Dynamic preprocessing pipeline for generic Raman spectral DataFrames.

        Parameters
        ----------
        dfs : List[pd.DataFrame]
            List of DataFrames containing Raman spectra.
        label : str
            Label for the spectra (e.g., 'hirsch' or 'normal').
        wavenumber_col : str
            Column name for wavenumber axis.
        intensity_cols : Optional[List[str]]
            List of columns for intensity data. If None, use all except wavenumber_col.
        region : Tuple[int, int]
            Wavenumber region to crop.
        preprocessing_steps : Optional[List[Callable]]
            List of preprocessing steps. If None, use default.
        visualize_steps : bool
            If True, visualize each preprocessing step.
        show_parameters_in_title : bool
            If True, show parameters in plot titles.
        max_plot_visualize_steps : int
            Maximum number of spectra to visualize at each step.
        save_pkl : bool
            If True, save results to pickle file.
        save_pkl_name : Optional[str]
            Name for pickle file.

        Returns
        -------
        dict:
            Dictionary containing processed spectra, labels, raw DataFrame, and preprocessing info.
        """
        if not RAMANSPY_AVAILABLE:
            raise ImportError("ramanspy and matplotlib are required for preprocessing")

        if type(dfs) is not list:
            dfs = [dfs]

        # Merge all DataFrames
        merged_df = pd.concat(dfs, axis=1 if dfs[0].index.name == wavenumber_col else 0)

        # Check if index is wavenumber
        if merged_df.index.name == wavenumber_col:
            wavenumbers = merged_df.index.values
            if intensity_cols is None:
                intensity_cols = merged_df.columns.tolist()
            intensities = merged_df[intensity_cols].values.T
        elif wavenumber_col in merged_df.columns:
            wavenumbers = merged_df[wavenumber_col].values
            if intensity_cols is None:
                exclude = {wavenumber_col}
                intensity_cols = [
                    col for col in merged_df.columns if col not in exclude
                ]
            intensities = merged_df[intensity_cols].values
        else:
            raise ValueError(f"Wavenumber column '{wavenumber_col}' not found.")

        # Labels: assign the provided label to all spectra
        n_spectra = intensities.shape[0] if intensities.ndim == 2 else 1
        labels = [label] * n_spectra if label is not None else []

        # Default preprocessing pipeline
        if preprocessing_steps is None:
            preprocessing_steps = [
                rp.preprocessing.misc.Cropper(region=region),
                rp.preprocessing.despike.WhitakerHayes(),
                rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
                rp.preprocessing.baseline.ASPLS(lam=1e5, tol=0.01),
                rp.preprocessing.normalise.Vector(),
            ]

        # Initialize preprocessing info dictionary
        preprocessing_info = {
            "pipeline_config": {
                "total_steps": len(preprocessing_steps),
                "wavenumber_col": wavenumber_col,
                "region": region,
                "n_spectra": n_spectra,
                "original_wavenumber_range": (wavenumbers.min(), wavenumbers.max()),
                "original_data_shape": intensities.shape,
            },
            "steps": [],
            "parameters_used": {},
            "step_order": [],
            "execution_info": {
                "execution_time": None,
                "memory_usage": None,
                "errors": [],
            },
        }

        # Record start time
        start_time = time.time()

        # Apply steps one-by-one
        spectra = rp.SpectralContainer(intensities, wavenumbers)

        # Store initial spectra info
        preprocessing_info["pipeline_config"][
            "initial_spectral_axis_shape"
        ] = spectra.spectral_axis.shape
        preprocessing_info["pipeline_config"][
            "initial_spectral_data_shape"
        ] = spectra.spectral_data.shape

        plot_data = {}

        if visualize_steps:
            # Create combined figure with all steps
            fig, axes = plt.subplots(
                len(preprocessing_steps) + 1,
                1,
                figsize=(12, 3 * (len(preprocessing_steps) + 1)),
                sharex=True,
            )

            # Plot raw spectra
            axes[0].set_title("Raw Spectra")
            for spectrum in spectra.spectral_data[:max_plot_visualize_steps]:
                axes[0].plot(spectra.spectral_axis, spectrum, alpha=0.6)

        # Process each preprocessing step
        for i, step in enumerate(preprocessing_steps):
            step_start_time = time.time()

            # Store pre-step info
            pre_step_shape = spectra.spectral_data.shape
            pre_step_axis_shape = spectra.spectral_axis.shape

            try:
                spectra = step.apply(spectra)

                # Store post-step info
                post_step_shape = spectra.spectral_data.shape
                post_step_axis_shape = spectra.spectral_axis.shape
                step_execution_time = time.time() - step_start_time

                # Extract step information
                step_info = {
                    "step_index": i + 1,
                    "step_name": step.__class__.__name__,
                    "step_module": step.__class__.__module__,
                    "parameters": self._extract_step_parameters(step),
                    "execution_time": step_execution_time,
                    "data_transformation": {
                        "input_shape": pre_step_shape,
                        "output_shape": post_step_shape,
                        "input_axis_shape": pre_step_axis_shape,
                        "output_axis_shape": post_step_axis_shape,
                        "shape_changed": pre_step_shape != post_step_shape,
                        "axis_changed": pre_step_axis_shape != post_step_axis_shape,
                    },
                }

                # Add specific parameter interpretations based on step type
                if hasattr(step, "__class__"):
                    step_info["step_category"] = self._categorize_step(step)
                    step_info["parameter_description"] = self._describe_parameters(step)

                preprocessing_info["steps"].append(step_info)
                preprocessing_info["step_order"].append(step.__class__.__name__)
                preprocessing_info["parameters_used"][
                    f"step_{i+1}_{step.__class__.__name__}"
                ] = step_info["parameters"]

                # Visualization
                if visualize_steps:
                    # Enhanced title with parameters
                    if show_parameters_in_title:
                        title = self._create_enhanced_title(step)
                    else:
                        title = f"After {step.__class__.__name__}"

                    # Plot on combined axes
                    axes[i + 1].set_title(title)
                    for spectrum in spectra.spectral_data[:max_plot_visualize_steps]:
                        axes[i + 1].plot(spectra.spectral_axis, spectrum, alpha=0.6)

            except Exception as e:
                error_info = {
                    "step_index": i + 1,
                    "step_name": step.__class__.__name__,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                preprocessing_info["execution_info"]["errors"].append(error_info)
                create_logs(
                    "preprocess_error",
                    "RamanPipeline",
                    f"Error in step {i+1} ({step.__class__.__name__}): {e}",
                    status="error",
                )
                continue

        if visualize_steps:
            # Format the combined figure
            for ax in axes:
                ax.set_ylabel("Intensity")
            axes[-1].set_xlabel("Wavenumber (cm⁻¹)")
            plt.tight_layout()
            plot_data["combined_figure"] = fig
            plt.show()

        # Complete preprocessing info
        total_execution_time = time.time() - start_time
        preprocessing_info["execution_info"]["execution_time"] = total_execution_time
        preprocessing_info["pipeline_config"][
            "final_spectral_axis_shape"
        ] = spectra.spectral_axis.shape
        preprocessing_info["pipeline_config"][
            "final_spectral_data_shape"
        ] = spectra.spectral_data.shape
        preprocessing_info["pipeline_config"]["final_wavenumber_range"] = (
            spectra.spectral_axis.min(),
            spectra.spectral_axis.max(),
        )

        # Add summary statistics
        preprocessing_info["summary"] = {
            "total_steps_executed": len(
                [s for s in preprocessing_info["steps"] if "error" not in s]
            ),
            "total_errors": len(preprocessing_info["execution_info"]["errors"]),
            "total_execution_time": total_execution_time,
            "average_step_time": (
                total_execution_time / len(preprocessing_steps)
                if preprocessing_steps
                else 0
            ),
        }

        data = {
            "processed": spectra,
            "labels": labels,
            "raw": merged_df,
            "plot_data": plot_data,
            "preprocessing_info": preprocessing_info,
        }

        # Save to pickle if requested
        if save_pkl:
            try:
                save_dir = os.path.join(CURRENT_DIR, "data", "preprocessed_data")
                os.makedirs(save_dir, exist_ok=True)
                save_pkl_name = (
                    save_pkl_name if save_pkl_name else f"{label}_preprocessed.pkl"
                )
                pkl_path = os.path.join(save_dir, save_pkl_name)
                pkl_path += ".pkl" if not pkl_path.endswith(".pkl") else ""
                with open(pkl_path, "wb") as f:
                    pkl.dump(data, f)
            except Exception as e:
                create_logs(
                    "preprocess_error",
                    "RamanPipeline",
                    f"Error saving preprocessed data: {e}",
                    status="error",
                )
                raise e

        return data

    def _categorize_step(self, step) -> str:
        """Categorize the preprocessing step."""
        step_name = step.__class__.__name__
        module_name = step.__class__.__module__

        # Category mapping based on class name and module
        if "baseline" in module_name or step_name in [
            "ASLS",
            "AIRPLS",
            "ARPLS",
            "ModPoly",
            "IModPoly",
            "ASPLS",
        ]:
            return "baseline_correction"
        elif "denoise" in module_name or step_name in ["SavGol", "MovingAverage"]:
            return "denoising"
        elif "despike" in module_name or step_name in [
            "WhitakerHayes",
            "Gaussian",
            "MedianDespike",
        ]:
            return "despiking"
        elif (
            "normalise" in module_name
            or "normalize" in module_name
            or step_name in ["Vector", "SNV", "MSC"]
        ):
            return "normalization"
        elif "misc" in module_name or step_name in ["Cropper"]:
            return "preprocessing"
        elif step_name in ["MultiScaleConv1D", "Transformer1DBaseline"]:
            return "advanced_baseline"
        elif step_name in ["WavenumberCalibration", "IntensityCalibration"]:
            return "calibration"
        elif step_name in ["Derivative"]:
            return "derivatives"
        else:
            return "other"

    def _describe_parameters(self, step) -> Dict[str, str]:
        """Provide human-readable descriptions of parameters."""
        parameters = self._extract_step_parameters(step)
        descriptions = {}

        # Common parameter descriptions
        param_descriptions = {
            "lam": "Smoothness parameter (higher = smoother baseline)",
            "p": "Asymmetry parameter (0-1, lower = more asymmetric)",
            "poly_order": "Polynomial order (higher = more flexible)",
            "tol": "Convergence tolerance (lower = more precise)",
            "max_iter": "Maximum number of iterations",
            "window_length": "Window size for smoothing",
            "polyorder": "Polynomial order for fitting",
            "region": "Wavenumber range for processing",
            "diff_order": "Order of difference matrix",
            "alpha": "Learning rate or weighting factor",
            "quantile": "Quantile for robust estimation",
            "scale": "Scaling factor",
            "num_std": "Number of standard deviations",
            "eta": "Regularization parameter",
            "threshold": "Threshold value for processing",
            "kernel_size": "Kernel size for filtering",
            "order": "Derivative order",
            "reference_peaks": "Reference peak positions for calibration",
        }

        for param_name, param_value in parameters.items():
            if param_name in param_descriptions:
                descriptions[param_name] = (
                    f"{param_descriptions[param_name]} (value: {param_value})"
                )
            else:
                descriptions[param_name] = f"Parameter value: {param_value}"

        return descriptions

    def _create_enhanced_title(self, step) -> str:
        """Create enhanced title with parameter information."""
        step_name = step.__class__.__name__
        param_info = []

        try:
            # Extract key parameters for display
            if hasattr(step, "region") and step.region is not None:
                if isinstance(step.region, (tuple, list)) and len(step.region) == 2:
                    param_info.append(f"region=({step.region[0]}-{step.region[1]})")

            if hasattr(step, "window_length") and hasattr(step, "polyorder"):
                param_info.append(f"window={step.window_length}, poly={step.polyorder}")

            if hasattr(step, "lam"):
                param_info.append(f"λ={step.lam:.0e}")

            if hasattr(step, "p"):
                param_info.append(f"p={step.p}")

            if hasattr(step, "kernel_size"):
                param_info.append(f"kernel={step.kernel_size}")

            if hasattr(step, "threshold"):
                param_info.append(f"threshold={step.threshold}")

            if hasattr(step, "order"):
                param_info.append(f"order={step.order}")

        except Exception:
            pass

        # Create title
        if param_info:
            params_str = ", ".join(param_info)
            title = f"After {step_name} ({params_str})"
        else:
            title = f"After {step_name}"

        # Limit title length
        if len(title) > 80:
            title = title[:77] + "..."

        return title

    def _extract_step_parameters(self, step) -> Dict[str, Any]:
        """Extract parameters from a preprocessing step for storage."""
        parameters = {}

        try:
            # Check if step has kwargs dictionary
            if hasattr(step, "kwargs") and isinstance(step.kwargs, dict):
                parameters.update(step.kwargs)

            # Check for _parameters attribute
            elif hasattr(step, "_parameters") and isinstance(step._parameters, dict):
                parameters.update(step._parameters)

            # Direct attribute access for common parameters
            else:
                common_attrs = [
                    "region",
                    "window_length",
                    "polyorder",
                    "lam",
                    "p",
                    "poly_order",
                    "tol",
                    "max_iter",
                    "kernel_size",
                    "threshold",
                    "order",
                    "reference_peaks",
                    "alpha",
                    "mode",
                ]

                for attr in common_attrs:
                    if hasattr(step, attr):
                        parameters[attr] = getattr(step, attr)

            # Add step class name
            parameters["class_name"] = step.__class__.__name__

        except Exception as e:
            create_logs(
                "preprocess_error",
                "RamanPipeline",
                f"Error extracting parameters from {step.__class__.__name__}: {e}",
                status="error",
            )
            parameters["class_name"] = step.__class__.__name__
            parameters["extraction_error"] = str(e)

        return parameters


class EnhancedRamanPipeline(RamanPipeline):
    """Enhanced RamanPipeline with progress tracking support."""

    def preprocess_with_progress(
        self,
        dfs: List[pd.DataFrame],
        label: str,
        preprocessing_steps: List[Callable],
        progress_callback: Callable[[int, str, int], bool] = None,
        wavenumber_col: str = "wavenumber",
        intensity_cols: Optional[List[str]] = None,
        region: Tuple[int, int] = (1050, 1700),
        visualize_steps: bool = False,
        save_pkl: bool = False,
        save_pkl_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Enhanced preprocessing pipeline with progress tracking.

        Parameters
        ----------
        dfs : List[pd.DataFrame]
            List of DataFrames containing Raman spectra.
        label : str
            Label for the spectra.
        preprocessing_steps : List[Callable]
            List of preprocessing steps to apply.
        progress_callback : Callable[[int, str, int], bool], optional
            Callback function for progress updates. Should return False to cancel.
            Parameters: (step_index, step_name, progress_percent)
        wavenumber_col : str
            Column name for wavenumber axis.
        intensity_cols : Optional[List[str]]
            List of columns for intensity data.
        region : Tuple[int, int]
            Wavenumber region to crop.
        visualize_steps : bool
            If True, visualize each preprocessing step.
        save_pkl : bool
            If True, save results to pickle file.
        save_pkl_name : Optional[str]
            Name for pickle file.

        Returns
        -------
        dict
            Dictionary containing processed spectra and metadata.
        """
        if not RAMANSPY_AVAILABLE:
            raise ImportError("ramanspy is required for preprocessing with progress")

        total_steps = len(preprocessing_steps)

        # Process setup
        if progress_callback:
            should_continue = progress_callback(0, "Initializing...", 0)
            if not should_continue:
                return {"cancelled": True}

        # Use the parent class method but with progress tracking
        # This is a simplified version - in practice you'd modify the main preprocess method
        # to include progress callbacks at each step

        result = self.preprocess(
            dfs=dfs,
            label=label,
            wavenumber_col=wavenumber_col,
            intensity_cols=intensity_cols,
            region=region,
            preprocessing_steps=preprocessing_steps,
            visualize_steps=visualize_steps,
            save_pkl=save_pkl,
            save_pkl_name=save_pkl_name,
        )

        # Signal completion
        if progress_callback:
            progress_callback(total_steps, "Complete", 100)

        return result
