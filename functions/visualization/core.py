import os
from typing import Dict, Any, List, Tuple, Optional
from functions.ML import RamanML, MLModel
from functions.configs import *
from numpy import trapz

# Import extracted modules (Phase 1, 2, 3, 4 & 5 refactoring)
from . import peak_assignment
from . import basic_plots
from . import model_evaluation
from . import ml_visualization
from . import explainability
from . import interactive_inspection
from . import lime_analysis

from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import seaborn as sns
import ramanspy as rp
import matplotlib.pyplot as plt
import shap
import time
import ramanspy as rp


class RamanVisualizer:
    """
    A class to visualize Raman spectra data.

    Attributes:
    -----------
    df : pandas DataFrame
        DataFrame containing Raman data with wavenumber column and intensity columns
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        spectral_container: List[rp.SpectralContainer] = None,
        labels: List[str] = None,
        common_axis: np.ndarray = None,
        n_features: int = None,
        ML_PROPERTY: Union[RamanML, MLModel] = None,
    ):
        """ """

        self.ML_PROPERTY = ML_PROPERTY
        self.df = df
        self.spectral_container = spectral_container
        self.labels = labels
        if self.ML_PROPERTY is not None:
            self.common_axis = self.ML_PROPERTY.common_axis
            self.n_features = self.ML_PROPERTY.n_features_in
        else:
            self.common_axis = common_axis
            self.n_features = n_features

    def get_peak_assignment(
        self,
        peak: Union[int, float, str],
        pick_near: bool = False,
        tolerance: int = 5,
        json_file_path: str = "assets/data/raman_peaks.json",
    ) -> dict:
        """
        Get the assignment meaning of a Raman peak based on the raman_peaks.json database.

        **NOTE**: This method now delegates to the peak_assignment module.

        Parameters:
        -----------
        peak : Union[int, float, str]
            The wavenumber peak to look up. If float, will be rounded to int.
        pick_near : bool, optional
            If True, will search for the nearest peak within tolerance if exact match not found.
            Default is False.
        tolerance : int, optional
            Maximum distance (in cm⁻¹) to search for nearby peaks when pick_near=True.
            Default is 5.
        json_file_path : str, optional
            Path to the raman_peaks.json file. Default is "assets/data/raman_peaks.json".

        Returns:
        --------
        dict
            Dictionary containing peak assignment information:
            - If found: {"peak": peak, "assignment": assignment, "reference_number": ref_num}
            - If not found and pick_near=False: {"assignment": "Not Found"}
            - If not found and pick_near=True: nearest match or "Not Found"
        """
        # Delegate to extracted module
        return peak_assignment.get_peak_assignment(
            peak, pick_near, tolerance, json_file_path
        )

    def get_multiple_peak_assignments(
        self,
        peaks: List[Union[int, float, str]],
        pick_near: bool = False,
        tolerance: int = 5,
        json_file_path: str = "assets/data/raman_peaks.json",
    ) -> List[dict]:
        """
        Get assignments for multiple peaks at once.

        **NOTE**: This method now delegates to the peak_assignment module.

        Parameters:
        -----------
        peaks : List[Union[int, float, str]]
            List of wavenumber peaks to look up.
        pick_near : bool, optional
            If True, will search for the nearest peak within tolerance if exact match not found.
        tolerance : int, optional
            Maximum distance (in cm⁻¹) to search for nearby peaks when pick_near=True.
        json_file_path : str, optional
            Path to the raman_peaks.json file.

        Returns:
        --------
        List[dict]
            List of dictionaries containing peak assignment information for each input peak.
        """
        # Delegate to extracted module
        return peak_assignment.get_multiple_peak_assignments(
            peaks, pick_near, tolerance, json_file_path
        )

    def find_peaks_in_range(
        self,
        min_wavenumber: Union[int, float],
        max_wavenumber: Union[int, float],
        json_file_path: str = "assets/data/raman_peaks.json",
    ) -> List[dict]:
        """
        Find all peaks within a specified wavenumber range.

        **NOTE**: This method now delegates to the peak_assignment module.

        Parameters:
        -----------
        min_wavenumber : Union[int, float]
            Minimum wavenumber of the range.
        max_wavenumber : Union[int, float]
            Maximum wavenumber of the range.
        json_file_path : str, optional
            Path to the raman_peaks.json file.

        Returns:
        --------
        List[dict]
            List of all peaks within the specified range.
        """
        # Delegate to extracted module
        return peak_assignment.find_peaks_in_range(
            min_wavenumber, max_wavenumber, json_file_path
        )

    def visualize_raman_spectra(
        self,
        wavenumber_colname: str = "wavenumber",
        title="Raman Spectra",
        figsize=(12, 6),
        xlim=None,
        ylim=None,
        legend=True,
        legend_loc="best",
        sample_limit=10,
    ) -> plt:
        """
        Visualize the Raman spectra data.

        **NOTE**: This method now delegates to the basic_plots module.

        Parameters:
        -----------
        wavenumber_colname : str
            Name of the wavenumber column
        title : str
            Plot title
        figsize : tuple
            Figure size (width, height) in inches
        xlim : tuple or None
            Optional x-axis limits (min, max)
        ylim : tuple or None
            Optional y-axis limits (min, max)
        legend : bool
            Whether to display the legend
        legend_loc : str
            Legend location
        sample_limit : int
            Maximum number of samples to plot (to avoid overcrowding)

        Returns:
        --------
        plt : matplotlib.pyplot
            The plot object for further customization
        """
        # Delegate to extracted module
        return basic_plots.visualize_raman_spectra(
            df=self.df,
            wavenumber_colname=wavenumber_colname,
            title=title,
            figsize=figsize,
            xlim=xlim,
            ylim=ylim,
            legend=legend,
            legend_loc=legend_loc,
            sample_limit=sample_limit,
        )

    def visualize_processed_spectra(
        self,
        spectral_data,
        spectral_axis,
        title="Processed Raman Spectra",
        figsize=(12, 6),
        xlim=None,
        ylim=None,
        legend=True,
        sample_names=None,
        legend_loc="best",
        sample_limit=10,
        cmap="viridis",
        add_mean=False,
    ) -> plt:
        """
        Visualize processed spectral data from ramanspy.

        **NOTE**: This method now delegates to the basic_plots module.

        Parameters:
        -----------
        spectral_data : np.ndarray
            Array of shape (n_samples, n_wavenumbers) containing processed spectra
        spectral_axis : np.ndarray
            Array containing wavenumber axis
        title : str
            Plot title
        figsize : tuple
            Figure size (width, height) in inches
        xlim : tuple or None
            Optional x-axis limits (min, max)
        ylim : tuple or None
            Optional y-axis limits (min, max)
        legend : bool
            Whether to display the legend
        sample_names : list or None
            List of names for each spectrum (if None, will use Sample 1, Sample 2, etc.)
        legend_loc : str
            Legend location
        sample_limit : int
            Maximum number of samples to plot (to avoid overcrowding)
        cmap : str
            Matplotlib colormap for spectra if no sample names provided
        add_mean : bool
            Whether to add the mean spectrum in bold

        Returns:
        --------
        plt : matplotlib.pyplot
            The plot object for further customization
        """
        # Delegate to extracted module
        return basic_plots.visualize_processed_spectra(
            spectral_data=spectral_data,
            spectral_axis=spectral_axis,
            title=title,
            figsize=figsize,
            xlim=xlim,
            ylim=ylim,
            legend=legend,
            sample_names=sample_names,
            legend_loc=legend_loc,
            sample_limit=sample_limit,
            cmap=cmap,
            add_mean=add_mean,
        )

    def extract_raman_characteristics(
        x: np.ndarray,
        y: np.ndarray,
        sample_name: str = "Sample",
        show_plot: bool = False,
    ) -> tuple[list[tuple], float]:
        """
        Extract Raman characteristics from the spectrum.

        **NOTE**: This method now delegates to the basic_plots module.

        Parameters:
        ----------
        x : np.ndarray
            Wavenumber array
        y : np.ndarray
            Intensity array
        sample_name : str
            Name of the sample for labeling
        show_plot : bool
            Whether to show the plot of the spectrum
        Returns:
        -------
        top_peaks : list[tuple]
            List of tuples containing peak positions and intensities
        auc : float
            Area under the curve

        """
        # Delegate to extracted module
        return basic_plots.extract_raman_characteristics(x, y, sample_name, show_plot)

    def pca2d(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        containers: List[rp.SpectralContainer] = None,
        labels: List[str] = None,
        common_axis: np.ndarray = None,
        current_n_features: int = None,
        title: str = "PCA",
        figsize: tuple = (12, 6),
        sample_limit: int = 100,
        cmap: str = "tab10",
        add_centroids: bool = True,
        show_centroid_line: bool = True,
        legend: bool = True,
        legend_loc: str = "best",
        show_decision_boundary: bool = False,
        decision_boundary_alpha: float = 0.3,
        # Parameters for using pre-calculated boundary data
        use_precalculated_boundary: bool = True,
    ) -> plt:
        """
        Perform PCA on DataFrame/numpy array or SpectralContainer objects and plot the first two PCs.
        Now automatically uses data from ML_PROPERTY when parameters are not provided.

        Args:
            df (pd.DataFrame or np.ndarray, optional):
                DataFrame or numpy array containing spectral data. If None, uses data from ML_PROPERTY.
            containers (List[rp.SpectralContainer], optional):
                List of SpectralContainer objects containing spectral data. If None, uses data from ML_PROPERTY.
            labels (List[str], optional):
                List of labels corresponding to the data points. If None, uses data from ML_PROPERTY.
            common_axis (np.ndarray, optional):
                Common spectral axis for interpolation. If None, uses data from ML_PROPERTY.
            current_n_features (int, optional):
                Number of features in the current dataset. If None, uses data from ML_PROPERTY.
            title (str):
                Title for the PCA plot.
            figsize (tuple):
                Size of the figure (width, height).
            sample_limit (int):
                Maximum number of samples to plot (to avoid overcrowding).
            cmap (str):
                Colormap for the plot.
            add_centroids (bool):
                Whether to add centroids for each class in the plot.
            show_centroid_line (bool):
                Whether to draw a line between centroids of the two classes.
            legend (bool):
                Whether to show the legend in the plot.
            legend_loc (str):
                Location of the legend in the plot.
            show_decision_boundary (bool):
                Whether to show the decision boundary using pre-calculated data from ML_PROPERTY.
            decision_boundary_alpha (float):
                Transparency of the decision boundary contour.
            use_precalculated_boundary (bool):
                Whether to use pre-calculated boundary data from ML_PROPERTY if available.

        Returns:
            plt: The matplotlib plot object.

        Notes:
            - If no parameters are provided, the function will automatically use training data from ML_PROPERTY
            - Decision boundary data must be pre-calculated using predict() with calculate_pca_boundary=True
            - No on-the-fly boundary calculation is performed to avoid redundant predictions
        """
        # Delegate to extracted module, passing ML_PROPERTY for auto-detection
        return ml_visualization.pca2d(
            df=df,
            containers=containers,
            labels=labels,
            common_axis=common_axis,
            current_n_features=current_n_features,
            ml_property=self.ML_PROPERTY if hasattr(self, "ML_PROPERTY") else None,
            title=title,
            figsize=figsize,
            sample_limit=sample_limit,
            cmap=cmap,
            add_centroids=add_centroids,
            show_centroid_line=show_centroid_line,
            legend=legend,
            legend_loc=legend_loc,
            show_decision_boundary=show_decision_boundary,
            decision_boundary_alpha=decision_boundary_alpha,
            use_precalculated_boundary=use_precalculated_boundary,
        )

    def confusion_matrix_heatmap(
        self,
        y_true: list,
        y_pred: list,
        class_labels: list,
        title: str = "Confusion Matrix",
        figsize: tuple = (10, 8),
        cmap: str = "Blues",
        normalize: bool = True,
        show_counts: bool = True,
        fmt: str = None,
        show_heatmap: bool = True,
    ) -> tuple[dict, sns.heatmap]:
        """
        Plot a confusion matrix as a heatmap.

        **NOTE**: This method now delegates to the model_evaluation module.

        Parameters:
            y_true (list):
                True labels
            y_pred (list):
                Predicted labels
            class_labels (list):
                List of class labels
            title (str):
                Plot title
            figsize (tuple):
                Figure size (width, height) in inches
            cmap (str):
                Matplotlib colormap for the heatmap
            normalize (bool):
                Whether to normalize the confusion matrix (default: True)
            show_counts (bool):
                Whether to show raw counts in each cell (default: True)
            fmt (str):
                Format for annotations (default: '.2f' for normalized, 'd' for counts)
            show_heatmap (bool):
                Whether to show the heatmap (default: True)

        Returns:
            plt (matplotlib.pyplot):
                The plot object for further customization
        """
        # Delegate to extracted module
        return model_evaluation.confusion_matrix_heatmap(
            y_true=y_true,
            y_pred=y_pred,
            class_labels=class_labels,
            title=title,
            figsize=figsize,
            cmap=cmap,
            normalize=normalize,
            show_counts=show_counts,
            fmt=fmt,
            show_heatmap=show_heatmap,
        )

    def shap_explain(
        self,
        ML_PROPERTY: Union[RamanML, MLModel] = None,
        nsamples: int = 50,
        show_plots: bool = True,
        max_display: int = 15,
        wavenumber_axis: np.ndarray = None,
        # Performance optimization parameters
        max_background_samples: int = 50,
        max_test_samples: int = 20,
        reduce_features: bool = True,
        max_features: int = 300,
        use_kmeans_sampling: bool = False,
        fast_mode: bool = False,
        # NEW: Use base estimator for CalibratedClassifierCV
        use_base_estimator: bool = True,
        # NEW: Force KernelExplainer for all models
        force_kernel_explainer: bool = False,
        shap_output_mode: str = "auto",  # NEW: "auto", "full", "sparse"
        # NEW: Increase samples for KernelExplainer
        kernel_nsamples_multiplier: float = 2.0,
    ) -> dict:
        """
        Generate SHAP explanations using stored training/test data from previous model training.
        REFACTORED: Now uses nested functions for better organization and tracking.

        Args:
            ML_PROPERTY (ML_PROPERTY): ML_PROPERTY instance with trained model
            nsamples (int): Number of samples for SHAP KernelExplainer. Lower = faster.
            show_plots (bool): Whether to generate and show SHAP plots.
            max_display (int): Maximum number of features to display in plots.
            wavenumber_axis (np.ndarray, optional): Wavenumber axis for feature names.
            max_background_samples (int): Maximum background samples for SHAP explainer.
            max_test_samples (int): Maximum test samples to explain.
            reduce_features (bool): Whether to use feature reduction for SVC models.
            max_features (int): Maximum features after reduction (only for SVC).
            use_kmeans_sampling (bool): Use K-means clustering for background sample selection.
            fast_mode (bool): Enable ultra-fast mode with minimal samples.
            use_base_estimator (bool): For CalibratedClassifierCV, use base estimator for SHAP
            force_kernel_explainer (bool): Force KernelExplainer for all models, even SVC.
            shap_output_mode (str): "auto", "full", or "sparse" for SHAP output mode.
            kernel_nsamples_multiplier (float): Multiplier for KernelExplainer samples.

        Returns:
            dict: Comprehensive SHAP analysis results including performance metrics.

        **NOTE**: This method now delegates to the explainability module.
        """

        # Delegate to extracted module, passing ML_PROPERTY for data extraction
        # Use instance ML_PROPERTY if none provided
        if ML_PROPERTY is None:
            ML_PROPERTY = self.ML_PROPERTY

        return explainability.shap_explain(
            ml_property=ML_PROPERTY,
            nsamples=nsamples,
            show_plots=show_plots,
            max_display=max_display,
            wavenumber_axis=wavenumber_axis,
            max_background_samples=max_background_samples,
            max_test_samples=max_test_samples,
            reduce_features=reduce_features,
            max_features=max_features,
            use_kmeans_sampling=use_kmeans_sampling,
            fast_mode=fast_mode,
            use_base_estimator=use_base_estimator,
            force_kernel_explainer=force_kernel_explainer,
            shap_output_mode=shap_output_mode,
            kernel_nsamples_multiplier=kernel_nsamples_multiplier,
        )

    def lime_explain(
        self,
        ML_PROPERTY: Union[RamanML, MLModel] = None,
        test_spectra: list[rp.SpectralContainer] = None,
        true_labels: list[str] = None,
        num_features: int = 20,  # Number of features to include in the explanation
        show_plots: bool = True,
        num_samples: int = 1000,  # Number of samples to generate for LIME explanations
        # Sample selection parameters
        max_test_samples: int = 5,  # Maximum number of test samples to explain
        # Specific indices to explain (overrides max_test_samples)
        sample_indices: List[int] = None,
        seed: int = 42,  # Random seed for reproducibility
        # Visualization parameters
        figsize: tuple = (10, 6),
        positive_label: str = "disease",
        negative_label: str = "normal",
        # Performance parameters
        # Use feature selection for faster computation
        use_feature_selection: bool = True,
        max_features: int = 300,  # Maximum features to use if feature selection enabled
        use_kmeans_sampling: bool = False,  # Use K-means for background sampling
    ) -> dict:
        """
        Generate LIME explanations for the model predictions with unified prediction function.

        Delegator to lime_analysis module (Phase 5 refactoring).

        Args:
            ML_PROPERTY (ML_PROPERTY): ML_PROPERTY instance with trained model
            test_spectra (list[rp.SpectralContainer]): Test spectra to explain. If None, uses stored test data.
            true_labels (list[str]): True labels for test spectra. If None, uses stored test labels.
            num_features (int): Number of features to include in the explanation
            show_plots (bool): Whether to show plots
            num_samples (int): Number of samples to generate for LIME explanations
            max_test_samples (int): Maximum number of test samples to explain
            sample_indices (List[int]): Specific indices to explain (overrides max_test_samples)
            seed (int): Random seed for reproducibility
            figsize (tuple): Figure size for plots
            positive_label (str): Label for positive class (default: "disease")
            negative_label (str): Label for negative class (default: "normal")
            use_feature_selection (bool): Use feature selection for faster computation
            max_features (int): Maximum features to use if feature selection enabled
            use_kmeans_sampling (bool): Use K-means for background sampling

        Returns:
            dict: Dictionary containing LIME explanations and visualizations
        """
        # Delegator to lime_analysis module (Phase 5 refactoring)
        return lime_analysis.lime_explain(
            self,
            ML_PROPERTY,
            test_spectra,
            true_labels,
            num_features,
            show_plots,
            num_samples,
            max_test_samples,
            sample_indices,
            seed,
            figsize,
            positive_label,
            negative_label,
            use_feature_selection,
            max_features,
            use_kmeans_sampling,
        )

    def _visualize_lime_explanation(
        self,
        explanation,
        class_names,
        feature_names,
        true_label,
        predicted_label,
        confidence,
        positive_label="cancer",
        negative_label="benign",
        figsize=(12, 8),
        sample_index=0,
        wavenumber_axis=None,
    ):
        """
        Visualize a LIME explanation with customized styling for Raman spectroscopy.

        Delegator to lime_analysis module (Phase 5 refactoring).

        Args:
            explanation: The LIME explanation object
            class_names: List of class names
            feature_names: List of feature names
            true_label: The true label of the sample
            predicted_label: The predicted label
            confidence: The prediction confidence
            positive_label: Label for positive class (default: "cancer")
            negative_label: Label for negative class (default: "benign")
            figsize: Figure size for the plot
            sample_index: Index of the sample in the test set
            wavenumber_axis: The wavenumber axis (if available)
        """
        # Delegator to lime_analysis module (Phase 5 refactoring)
        return lime_analysis._visualize_lime_explanation(
            self,
            explanation,
            class_names,
            feature_names,
            true_label,
            predicted_label,
            confidence,
            positive_label,
            negative_label,
            figsize,
            sample_index,
            wavenumber_axis,
        )

    def _visualize_lime_comparison(
        self,
        top_features,
        class_names,
        positive_label="cancer",
        negative_label="benign",
        figsize=(14, 10),
    ):
        """
        Visualize comparison of feature importance between classes.

        Delegator to lime_analysis module (Phase 5 refactoring).

        Args:
            top_features: Dictionary with class names as keys and lists of (feature, weight) tuples as values
            class_names: List of class names
            positive_label: Name for positive label
            negative_label: Name for negative label
            figsize: Figure size for plot
        """
        # Delegator to lime_analysis module (Phase 5 refactoring)
        return lime_analysis._visualize_lime_comparison(
            self, top_features, class_names, positive_label, negative_label, figsize
        )

    def inspect_spectra(
        self,
        ML_PROPERTY: Union[RamanML, MLModel] = None,
        test_spectra: list[rp.SpectralContainer] = None,
        true_labels: list[str] = None,
        n_samples: int = 5,
        sample_indices: List[int] = None,
        seed: int = None,
        class_to_inspect: str = None,
        show_plots: bool = True,
        max_display: int = 10,
        nsamples: int = 100,
        fast_mode: bool = False,
        show_shap_plots: bool = False,
        show_lime_plots: bool = False,
        positive_label: str = "cancer",
        negative_label: str = "benign",
        lime_num_features: int = 20,
        lime_num_samples: int = 1000,
        lime_figsize: tuple = (10, 6),
        lime_seed: int = 42,
        lime_use_feature_selection: bool = True,
        lime_max_features: int = 300,
        lime_use_kmeans_sampling: bool = False,
        **kwargs,
    ) -> dict:
        """
        Randomly select spectra and show detailed SHAP and/or LIME explanations for why they were
        classified as positive_label or negative_label. Compatible with both RamanML and MLModel classes.

        Parameters:
        -----------
        ML_PROPERTY : Union[RamanML, MLModel], optional
            ML_PROPERTY instance with trained model. If None, uses stored ML_PROPERTY.
        test_spectra : list, optional
            List of SpectralContainer objects. If None, uses stored test data.
        true_labels : list, optional
            True labels for the spectra. If None, uses stored test labels.
        n_samples : int
            Number of random spectra to inspect
        sample_indices : list, optional
            Specific indices of spectra to inspect. If provided, overrides n_samples.
        seed : int, optional
            Random seed for reproducibility
        class_to_inspect : str, optional
            If provided, only inspect spectra from this class ("benign" or "cancer")
        show_plots : bool
            Whether to show general plots (spectrum with highlights, SHAP plots, tables)
        max_display : int
            Maximum number of features to display in SHAP plots
        nsamples : int
            Number of samples for SHAP computation
        fast_mode : bool
            Enable fast mode for SHAP computation
        show_shap_plots : bool
            Whether to show SHAP plots for each inspected spectrum
        show_lime_plots : bool
            Whether to show LIME plots for each inspected spectrum (independent of show_plots)
        positive_label : str
            Label for positive class (default: "cancer")
        negative_label : str
            Label for negative class (default: "benign")
        lime_num_features : int
            Number of features to include in LIME explanation
        lime_num_samples : int
            Number of samples to generate for LIME explanations
        lime_figsize : tuple
            Figure size for LIME plots
        lime_seed : int
            Random seed for LIME explanations
        lime_use_feature_selection : bool
            Whether to use feature selection for LIME explanations
        lime_max_features : int
            Maximum number of features to use in LIME explanations
        lime_use_kmeans_sampling : bool
            Whether to use KMeans sampling for LIME explanations

        Returns:
        --------
        dict
            Dictionary of inspection results including both SHAP and LIME explanations
        """
        # Delegator to interactive_inspection module (Phase 4A refactoring)
        # Use self.ML_PROPERTY as fallback if ML_PROPERTY is None
        if ML_PROPERTY is None:
            ML_PROPERTY = self.ML_PROPERTY

        return interactive_inspection.inspect_spectra(
            ml_property=ML_PROPERTY,
            test_spectra=test_spectra,
            true_labels=true_labels,
            n_samples=n_samples,
            sample_indices=sample_indices,
            seed=seed,
            class_to_inspect=class_to_inspect,
            show_plots=show_plots,
            max_display=max_display,
            nsamples=nsamples,
            fast_mode=fast_mode,
            show_shap_plots=show_shap_plots,
            show_lime_plots=show_lime_plots,
            positive_label=positive_label,
            negative_label=negative_label,
            lime_num_features=lime_num_features,
            lime_num_samples=lime_num_samples,
            lime_figsize=lime_figsize,
            lime_seed=lime_seed,
            lime_use_feature_selection=lime_use_feature_selection,
            lime_max_features=lime_max_features,
            lime_use_kmeans_sampling=lime_use_kmeans_sampling,
            visualizer_instance=self,
            **kwargs,
        )

    def plot_container_distribution(
        self,
        spectral_containers: List[rp.SpectralContainer],
        container_labels: List[str],
        **kwargs,
    ) -> dict:
        """
        Method version for RamanVisualizer class.

        Usage:
        ------
        visualizer = RamanVisualizer()
        results = visualizer.plot_container_distribution(
            spectral_containers=[chumDF_benign["processed"], chumDF_cancer["processed"], ...],
            container_labels=['CHUM_benign', 'CHUM_cancer', ...],
            sample_limit=1000
        )
        """
        return plot_institution_distribution(
            spectral_containers, container_labels, **kwargs
        )


def spectrum_with_highlights_spectrum(
    spectrum,
    common_axis_subset,
    top_positive,
    top_negative,
    positive_label,
    negative_label,
    spec_info,
    pred_label,
    confidence: Union[int, float] = 0,
    spectrum_idx: int = -1,
) -> plt:
    """
    Original spectrum with highlighted regions

    Args:
        spectrum (np.ndarray): The Raman spectrum data.
        common_axis_subset (np.ndarray): The common wavenumber axis subset.
        top_positive (list): List of tuples with top positive contributors (wavenumber, SHAP value).
        top_negative (list): List of tuples with top negative contributors (wavenumber, SHAP value).
        positive_label (str): Label for positive contributors.
        negative_label (str): Label for negative contributors.
        spec_info (dict): Dictionary containing spectrum information like container index, spectrum index, and true label.
        pred_label (str): Predicted label for the spectrum.
        confidence (Union[int, float]): Confidence score for the prediction.
        spectrum_idx (int): Index of the spectrum in the dataset.

    Returns:
        plt: The matplotlib figure object containing the plot.
    """
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(
        common_axis_subset,
        spectrum[: len(common_axis_subset)],
        "k-",
        linewidth=2,
        label="Spectrum",
    )

    # Highlight top positive and negative regions
    # Ensure top_positive and top_negative are not empty before accessing [0][0]
    if top_positive:
        for wn, sv in top_positive:
            plt.axvline(
                x=wn,
                color="red",
                alpha=0.6,
                linestyle="--",
                linewidth=1.5,
                label=(
                    f"{positive_label} Contributors" if wn == top_positive[0][0] else ""
                ),
            )
    if top_negative:
        for wn, sv in top_negative:
            plt.axvline(
                x=wn,
                color="blue",
                alpha=0.6,
                linestyle="--",
                linewidth=1.5,
                label=(
                    f"{negative_label} Contributors" if wn == top_negative[0][0] else ""
                ),
            )

    # Enhanced title with spectrum index information
    # Ensure confidence is scalar before formatting
    confidence_scalar = confidence
    if hasattr(confidence_scalar, "item"):
        confidence_scalar = confidence_scalar.item()
    elif (
        isinstance(confidence_scalar, (np.ndarray, list))
        and len(confidence_scalar) == 1
    ):
        confidence_scalar = confidence_scalar[0]
        if hasattr(confidence_scalar, "item"):
            confidence_scalar = confidence_scalar.item()

    try:
        confidence_float = float(confidence_scalar)
    except (TypeError, ValueError):
        console_log(
            f"Warning: Could not convert confidence '{confidence_scalar}' to float. Using 0.0."
        )
        confidence_float = 0.0

    title_text = (
        f"Spectrum {spectrum_idx+1} [Container: {spec_info['container_idx']}, "
        f"Index: {spec_info['spectrum_idx']}] - "
        f"True: {spec_info['true_label']}, Predicted: {pred_label} "
        f"(Confidence: {confidence_float:.3f})"
    )  # Use confidence_float

    plt.title(title_text, fontsize=12, fontweight="bold")
    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig1


def create_shap_plots(
    spectrum_idx,
    spec_info,
    shap_values,
    common_axis_subset,
    top_positive,
    top_negative,
    positive_label,
    negative_label,
) -> plt:
    """
    Helper function to create SHAP plots.

    Args:
        spectrum_idx (int): Index of the spectrum in the dataset.
        spec_info (dict): Dictionary containing spectrum information like container index, spectrum index, and true label.
        shap_values (np.ndarray): SHAP values for the spectrum.
        common_axis_subset (np.ndarray): The common wavenumber axis subset.
        top_positive (list): List of tuples with top positive contributors (wavenumber, SHAP value).
        top_negative (list): List of tuples with top negative contributors (wavenumber, SHAP value).
        positive_label (str): Label for positive contributors.
        negative_label (str): Label for negative contributors.

    Returns:
        plt: The matplotlib figure object containing the SHAP plot.

    """
    # Delegator to interactive_inspection module (Phase 4B refactoring)
    return interactive_inspection.create_shap_plots(
        spectrum_idx,
        spec_info,
        shap_values,
        common_axis_subset,
        top_positive,
        top_negative,
        positive_label,
        negative_label,
    )


def create_enhanced_table(
    self: RamanVisualizer,
    spectrum_idx,
    spec_info,
    pred_label,
    shap_result,
    shap_values,
    top_positive,
    top_negative,
    positive_label,
    negative_label,
    confidence: Union[int, float] = 0,
) -> plt:
    """
    Enhanced Top contributors table with peak assignments

    Args:
        self (RamanVisualizer): Instance of the RamanVisualizer class.
        spectrum_idx (int): Index of the spectrum in the dataset.
        spec_info (dict): Dictionary containing spectrum information like container index, spectrum index, and true label.
        pred_label (str): Predicted label for the spectrum.
        shap_result (dict): SHAP result dictionary containing expected value and other metadata.
        shap_values (np.ndarray): SHAP values for the spectrum.
        top_positive (list): List of tuples with top positive contributors (wavenumber, SHAP value).
        top_negative (list): List of tuples with top negative contributors (wavenumber, SHAP value).
        positive_label (str): Label for positive contributors.
        negative_label (str): Label for negative contributors.
        confidence (Union[int, float]): Confidence score for the prediction.

    Returns:
        plt: The matplotlib figure object containing the enhanced table.

    """
    # Delegator to interactive_inspection module (Phase 4B refactoring)
    return interactive_inspection.create_enhanced_table(
        self,
        spectrum_idx,
        spec_info,
        pred_label,
        shap_result,
        shap_values,
        top_positive,
        top_negative,
        positive_label,
        negative_label,
        confidence,
    )


def plot_institution_distribution(
    spectral_containers: List[rp.SpectralContainer],
    container_labels: List[str],
    container_names: List[str] = None,
    sample_limit: int = 500,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    figsize: tuple = (12, 8),
    alpha: float = 0.6,
    point_size: int = 50,
    title: str = "Institution Distribution in Feature Space",
    show_legend: bool = True,
    save_plot: bool = False,
    save_path: str = None,
    color_palette: str = "tab10",
    interpolate_to_common_axis: bool = True,
    common_axis: np.ndarray = None,
    show_class_info: bool = True,
    class_labels: List[str] = None,
) -> dict:
    """
    Plot t-SNE visualization of spectral data distribution across different containers/institutions.

    Delegator to interactive_inspection module (Phase 4B refactoring).

    Parameters:
    -----------
    spectral_containers : List[rp.SpectralContainer]
        List of SpectralContainer objects to compare
    container_labels : List[str]
        Labels for each container (e.g., ['CHUM_benign', 'CHUM_cancer', 'UHN_benign', ...])
    container_names : List[str], optional
        Institution/group names for each container. If None, extracted from container_labels
    sample_limit : int
        Maximum number of samples to use for t-SNE (for performance)
    perplexity : int
        t-SNE perplexity parameter
    n_iter : int
        Number of iterations for t-SNE
    random_state : int
        Random seed for reproducibility
    figsize : tuple
        Figure size (width, height)
    alpha : float
        Point transparency
    point_size : int
        Size of scatter plot points
    title : str
        Plot title
    show_legend : bool
        Whether to show legend
    save_plot : bool
        Whether to save the plot
    save_path : str
        Path to save the plot (if save_plot=True)
    color_palette : str
        Matplotlib colormap name
    interpolate_to_common_axis : bool
        Whether to interpolate all spectra to a common axis
    common_axis : np.ndarray, optional
        Common axis for interpolation. If None, uses the first container's axis
    show_class_info : bool
        Whether to show class information in the legend
    class_labels : List[str], optional
        Class labels (e.g., ['benign', 'cancer']) for extracting class info

    Returns:
    --------
    dict : Results dictionary containing embedded data, labels, and metadata
    """
    # Delegator to interactive_inspection module (Phase 4B refactoring)
    return interactive_inspection.plot_institution_distribution(
        spectral_containers=spectral_containers,
        container_labels=container_labels,
        container_names=container_names,
        sample_limit=sample_limit,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        figsize=figsize,
        alpha=alpha,
        point_size=point_size,
        title=title,
        show_legend=show_legend,
        save_plot=save_plot,
        save_path=save_path,
        color_palette=color_palette,
        interpolate_to_common_axis=interpolate_to_common_axis,
        common_axis=common_axis,
        show_class_info=show_class_info,
        class_labels=class_labels,
    )
