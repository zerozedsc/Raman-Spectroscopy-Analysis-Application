"""
Interactive Spectrum Inspection Module

This module provides interactive inspection capabilities for Raman spectra with integrated
SHAP and LIME explanations. It's designed to work with both RamanML and MLModel classes,
providing detailed visual explanations of ML model predictions.

Key Features:
-------------
- Interactive spectrum inspection with visual explanations
- SHAP (SHapley Additive exPlanations) integration for model interpretability
- LIME (Local Interpretable Model-agnostic Explanations) support
- Automatic model type detection and handling
- Support for calibrated classifiers
- Intelligent background data selection for SHAP
- Peak highlighting and contribution analysis
- Enhanced prediction tables with confidence scores

Main Function:
--------------
inspect_spectra()
    Comprehensive inspection of spectra with SHAP/LIME explanations

Created: Phase 4A of visualization module refactoring
Purpose: Extract interactive inspection functionality from monolithic core.py
Pattern: Hybrid extraction (standalone function with module-level helpers)
"""

import numpy as np
from typing import List, Union, Optional, Dict, Any, Tuple
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.manifold import TSNE
from scipy.special import expit
import matplotlib.pyplot as plt
import traceback

# Try to import RamanPy types
try:
    import ramanspy as rp
except ImportError:
    rp = None

# Try to import project-specific types
try:
    from functions.ML import RamanML, MLModel
except ImportError:
    # Create fallback type hints
    RamanML = Any
    MLModel = Any

# Import visualization utilities
from functions._utils_ import console_log


def inspect_spectra(
    ml_property: Optional[Union[RamanML, MLModel]] = None,  # type: ignore
    test_spectra: Optional[list] = None,
    true_labels: Optional[list] = None,
    n_samples: int = 5,
    sample_indices: Optional[List[int]] = None,
    seed: Optional[int] = None,
    class_to_inspect: Optional[str] = None,
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
    visualizer_instance: Any = None,
    **kwargs,
) -> dict:
    """
    Randomly select spectra and show detailed SHAP and/or LIME explanations for why they were
    classified as positive_label or negative_label. Compatible with both RamanML and MLModel classes.

    This function provides comprehensive inspection of Raman spectra with integrated explainability
    through SHAP and LIME techniques. It handles various data sources, automatically detects model
    types, and generates visual explanations for model predictions.

    Parameters
    ----------
    ml_property : Union[RamanML, MLModel], optional
        ML property instance with trained model. If None, should be provided via visualizer_instance.
    test_spectra : list, optional
        List of SpectralContainer objects. If None, uses stored test data.
    true_labels : list, optional
        True labels for the spectra. If None, uses stored test labels.
    n_samples : int, default=5
        Number of random spectra to inspect.
    sample_indices : list, optional
        Specific indices of spectra to inspect. If provided, overrides n_samples.
    seed : int, optional
        Random seed for reproducibility.
    class_to_inspect : str, optional
        If provided, only inspect spectra from this class ("benign" or "cancer").
    show_plots : bool, default=True
        Whether to show general plots (spectrum with highlights, SHAP plots, tables).
    max_display : int, default=10
        Maximum number of features to display in SHAP plots.
    nsamples : int, default=100
        Number of samples for SHAP computation.
    fast_mode : bool, default=False
        Enable fast mode for SHAP computation.
    show_shap_plots : bool, default=False
        Whether to show SHAP plots for each inspected spectrum.
    show_lime_plots : bool, default=False
        Whether to show LIME plots for each inspected spectrum (independent of show_plots).
    positive_label : str, default="cancer"
        Label for positive class.
    negative_label : str, default="benign"
        Label for negative class.
    lime_num_features : int, default=20
        Number of features to include in LIME explanation.
    lime_num_samples : int, default=1000
        Number of samples to generate for LIME explanations.
    lime_figsize : tuple, default=(10, 6)
        Figure size for LIME plots.
    lime_seed : int, default=42
        Random seed for LIME explanations.
    lime_use_feature_selection : bool, default=True
        Whether to use feature selection for LIME explanations.
    lime_max_features : int, default=300
        Maximum number of features to use in LIME explanations.
    lime_use_kmeans_sampling : bool, default=False
        Whether to use KMeans sampling for LIME explanations.
    visualizer_instance : Any, optional
        RamanVisualizer instance for accessing helper methods (shap_explain, lime_explain).

    Returns
    -------
    dict
        Dictionary of inspection results with the following structure:
        {
            'results': list of dict
                List of per-spectrum results containing:
                - spectrum_info: container/spectrum indices and metadata
                - prediction: label, confidence, correctness
                - shap_explanation: top contributors and values
                - lime_explanation: LIME explanation data (if requested)
            'summary': dict
                Summary statistics including:
                - total_spectra_analyzed
                - correct_predictions
                - accuracy
                - shap/lime explanation counts
            'lime_full_results': dict (if show_lime_plots=True)
                Complete LIME results
            'ml_type_used': str
                Type of ML property used ('RamanML' or 'MLModel')
        }

    Raises
    ------
    ValueError
        If no ML property is available, or if data sources are insufficient.

    Examples
    --------
    >>> # Basic usage with RamanML
    >>> viz = RamanVisualizer()
    >>> results = viz.inspect_spectra(
    ...     n_samples=5,
    ...     show_shap_plots=True,
    ...     positive_label="cancer",
    ...     negative_label="benign"
    ... )

    >>> # Inspect specific spectra with LIME
    >>> results = viz.inspect_spectra(
    ...     sample_indices=[0, 5, 10],
    ...     show_lime_plots=True,
    ...     lime_num_features=30
    ... )

    >>> # Filter by class
    >>> results = viz.inspect_spectra(
    ...     class_to_inspect="cancer",
    ...     n_samples=10,
    ...     seed=42
    ... )

    Notes
    -----
    - Automatically handles both RamanML and MLModel instances
    - Supports calibrated classifiers with automatic base estimator extraction
    - Intelligently selects background data for SHAP explanations:
        1. Uses existing training data if available (RamanML)
        2. Falls back to test data subset (both RamanML and MLModel)
        3. Creates synthetic background from provided spectra if needed
    - SHAP and LIME explanations can be generated independently
    - For large datasets, consider using sample_indices to inspect specific spectra
    """
    # Helper functions defined at module level for clarity

    def _validate_and_prepare_ml_property():
        """Validate and prepare ML_PROPERTY instance."""
        nonlocal ml_property

        if ml_property is None:
            raise ValueError(
                "ML_PROPERTY instance is required for spectrum inspection."
            )

        # Detect ML_PROPERTY type and get the actual sklearn model
        ml_type = type(ml_property).__name__

        if ml_type == "RamanML":
            actual_model = getattr(ml_property, "_model", None)
            if actual_model is None:
                raise ValueError("No trained model found in RamanML instance.")
        elif ml_type == "MLModel":
            actual_model = getattr(ml_property, "sklearn_model", None)
            if actual_model is None:
                raise ValueError("No sklearn model found in MLModel instance.")
        else:
            raise ValueError(f"Unsupported ML_PROPERTY type: {ml_type}")

        console_log(f"Using {ml_type} with model type: {type(actual_model).__name__}")

        return actual_model, ml_type

    def _prepare_test_data():
        """Prepare test data from various sources with priority logic."""

        # Priority logic for data source selection
        # 1. Use provided test_spectra and true_labels if both given
        # 2. Use stored X_test and y_test if available
        # 3. Fall back to X_train and y_train only for RamanML

        if test_spectra is not None and true_labels is not None:
            console_log("📊 Using provided test_spectra and true_labels")

            # Flatten test spectra into a list of individual spectra with indices
            all_spectra = []
            label_idx = 0

            for container_idx, container in enumerate(test_spectra):
                for spectrum_idx, spectrum in enumerate(container.spectral_data):
                    if label_idx < len(true_labels):
                        all_spectra.append(
                            {
                                "container_idx": container_idx,
                                "spectrum_idx": spectrum_idx,
                                "true_label": true_labels[label_idx],
                                "spectrum": spectrum,
                                "spectral_axis": container.spectral_axis,
                            }
                        )
                        label_idx += 1

            return all_spectra

        # Try to use stored test data first (works for both RamanML and MLModel)
        elif (
            hasattr(ml_property, "X_test")
            and ml_property.X_test is not None
            and hasattr(ml_property, "y_test")
            and ml_property.y_test is not None
            and hasattr(ml_property, "common_axis")
            and ml_property.common_axis is not None
        ):

            console_log("📊 Using stored test data (X_test, y_test)")

            # Convert stored test data back to spectral format for analysis
            X_test = ml_property.X_test
            y_test = ml_property.y_test
            common_axis = ml_property.common_axis

            # Create artificial spectral containers from stored data
            all_spectra = []
            for idx, (spectrum, label) in enumerate(zip(X_test, y_test)):
                all_spectra.append(
                    {
                        "container_idx": 0,
                        "spectrum_idx": idx,
                        "true_label": label,
                        "spectrum": spectrum,
                        "spectral_axis": common_axis,
                    }
                )

            return all_spectra

        # Fall back to training data only for RamanML
        elif (
            ml_type == "RamanML"
            and hasattr(ml_property, "X_train")
            and ml_property.X_train is not None
            and hasattr(ml_property, "y_train")
            and ml_property.y_train is not None
            and hasattr(ml_property, "common_axis")
            and ml_property.common_axis is not None
        ):

            console_log(
                "📊 Using stored training data (X_train, y_train) - RamanML fallback"
            )

            # Convert stored training data back to spectral format for analysis
            X_train = ml_property.X_train
            y_train = ml_property.y_train
            common_axis = ml_property.common_axis

            # Create artificial spectral containers from stored data
            all_spectra = []
            for idx, (spectrum, label) in enumerate(zip(X_train, y_train)):
                all_spectra.append(
                    {
                        "container_idx": 0,
                        "spectrum_idx": idx,
                        "true_label": label,
                        "spectrum": spectrum,
                        "spectral_axis": common_axis,
                    }
                )

            return all_spectra

        else:
            # No data available
            ml_type_str = (
                ml_type if "ml_type" in locals() else type(ml_property).__name__
            )

            if ml_type_str == "MLModel":
                error_msg = (
                    "No data available for inspection. MLModel requires either:\n"
                    "1. Provide test_spectra and true_labels explicitly, OR\n"
                    "2. Call predict() first to populate X_test and y_test"
                )
            else:
                error_msg = (
                    "No data available for inspection. Please either:\n"
                    "1. Provide test_spectra and true_labels explicitly, OR\n"
                    "2. Train a model first to populate training/test data"
                )

            raise ValueError(error_msg)

    def _filter_and_select_spectra(all_spectra):
        """Filter by class and select random spectra."""
        # Filter by class if requested
        if class_to_inspect:
            filtered_spectra = [
                s for s in all_spectra if s["true_label"] == class_to_inspect
            ]
            if not filtered_spectra:
                console_log(f"No spectra found with class {class_to_inspect}")
                return {
                    "results": [],
                    "summary": f"No spectra found for class: {class_to_inspect}",
                }
            all_spectra = filtered_spectra

        # Select random spectra
        selected_spectra = []
        if sample_indices is None:
            if n_samples <= 0 or n_samples > len(all_spectra):
                raise ValueError(
                    f"n_samples must be between 1 and {len(all_spectra)}. Provided: {n_samples}"
                )
            rng = np.random.RandomState(seed) if seed is not None else np.random
            selected_indices = rng.choice(
                len(all_spectra), min(n_samples, len(all_spectra)), replace=False
            )
            selected_spectra = [all_spectra[idx] for idx in selected_indices]
        else:
            for idx in sample_indices:
                if 0 <= idx < len(all_spectra):
                    selected_spectra.append(all_spectra[idx])
                else:
                    console_log(f"Warning: Index {idx} out of range, skipping.")

        return selected_spectra

    def _get_unified_prediction_function(model):
        """Create a unified prediction function that handles all model types including CalibratedClassifierCV."""
        is_calibrated = isinstance(model, CalibratedClassifierCV)

        if is_calibrated:
            # For CalibratedClassifierCV, always use predict_proba (it's calibrated)
            def predict_fn(X):
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                return model.predict_proba(X)

            return predict_fn
        else:
            # For non-calibrated models, use the original logic
            is_svc = isinstance(model, SVC)
            is_rf = hasattr(model, "n_estimators")  # RandomForestClassifier check

            def predict_fn(X):
                if X.ndim == 1:
                    X = X.reshape(1, -1)

                if is_rf or (
                    hasattr(model, "predict_proba")
                    and hasattr(model, "probability")
                    and model.probability
                ):
                    return model.predict_proba(X)
                elif is_svc:
                    # For SVC without probability, use decision_function
                    decision = model.decision_function(X)
                    if decision.ndim == 1:
                        prob_pos = expit(decision)
                        return np.column_stack([1 - prob_pos, prob_pos])
                    else:
                        exp_scores = np.exp(
                            decision - np.max(decision, axis=1, keepdims=True)
                        )
                        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                else:
                    # Fallback for other model types
                    if hasattr(model, "predict_proba"):
                        return model.predict_proba(X)
                    else:
                        predictions = model.predict(X)
                        n_classes = len(np.unique(predictions))
                        dummy_probs = np.eye(n_classes)[predictions]
                        return dummy_probs

            return predict_fn

    def _get_model_class_names(model):
        """Get class names from the model."""
        if hasattr(model, "classes_"):
            return model.classes_
        elif isinstance(model, CalibratedClassifierCV):
            # For CalibratedClassifierCV, get classes from base estimator
            if hasattr(model, "estimators_") and len(model.estimators_) > 0:
                return model.estimators_[0].classes_
            elif hasattr(model, "base_estimator") and hasattr(
                model.base_estimator, "classes_"
            ):
                return model.base_estimator.classes_
            else:
                return np.array(["class_0", "class_1"])  # Fallback
        else:
            return np.array(["class_0", "class_1"])  # Fallback

    def _predict_single_spectrum(spectrum, spec_info, model, unified_predict):
        """Make prediction for a single spectrum."""
        # Interpolate to common axis if needed
        if len(spec_info["spectral_axis"]) != len(ml_property.common_axis):
            spectrum = np.interp(
                ml_property.common_axis, spec_info["spectral_axis"], spectrum
            )

        try:
            # Get prediction probabilities
            pred_proba = unified_predict(spectrum.reshape(1, -1))[0]

            # Get class names from model
            class_names = _get_model_class_names(model)

            # Get predicted class and confidence
            pred_idx = np.argmax(pred_proba)
            pred_label = class_names[pred_idx]
            confidence = pred_proba[pred_idx]

            return pred_label, confidence, spectrum

        except Exception as e:
            console_log(f"Error in prediction: {e}")
            return None, None, spectrum

    def _generate_lime_explanations(selected_spectra):
        """Generate LIME explanations if requested."""
        lime_results = None
        if show_lime_plots and len(selected_spectra) > 0 and visualizer_instance:
            console_log("\nGenerating LIME explanations...")
            try:
                # Extract sample indices for LIME
                lime_sample_indices = [
                    spec_info["spectrum_idx"] for spec_info in selected_spectra
                ]

                lime_results = visualizer_instance.lime_explain(
                    test_spectra=test_spectra,
                    true_labels=true_labels,
                    sample_indices=lime_sample_indices,
                    num_features=lime_num_features,
                    show_plots=show_lime_plots,
                    num_samples=lime_num_samples,
                    figsize=lime_figsize,
                    positive_label=positive_label,
                    negative_label=negative_label,
                    use_feature_selection=lime_use_feature_selection,
                    max_features=lime_max_features,
                    seed=lime_seed,
                    use_kmeans_sampling=lime_use_kmeans_sampling,
                )

                if lime_results["success"]:
                    console_log(
                        f"LIME explanations generated successfully for {lime_results['num_samples']} samples"
                    )
                else:
                    console_log(
                        f"LIME explanation failed: {lime_results.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                console_log(f"Error generating LIME explanations: {e}")
                lime_results = {"success": False, "error": str(e)}

        return lime_results

    def _get_shap_explanation(spectrum, spec_info, spectrum_idx):
        """Generate SHAP explanation for a single spectrum with updated background data logic."""
        if not visualizer_instance:
            console_log(
                "Warning: No visualizer instance available for SHAP explanations"
            )
            return None

        try:
            # Create a temporary single-spectrum dataset
            single_spectrum_data = np.array([spectrum]).reshape(1, -1)

            # Store original data
            original_X_test = getattr(ml_property, "X_test", None)
            original_y_test = getattr(ml_property, "y_test", None)
            original_X_train = getattr(ml_property, "X_train", None)
            original_y_train = getattr(ml_property, "y_train", None)

            # Set up data for SHAP
            ml_property.X_test = single_spectrum_data
            ml_property.y_test = np.array([spec_info["true_label"]])

            # Smart background data selection
            background_data_source = None

            # Priority 1: Use existing training data if available (RamanML)
            if original_X_train is not None and len(original_X_train) > 0:
                ml_property.X_train = original_X_train
                ml_property.y_train = original_y_train
                background_data_source = "existing_training_data"
                console_log(
                    f"Using existing training data with {len(original_X_train)} samples for SHAP background"
                )

            # Priority 2: Use existing test data if available (both RamanML and MLModel)
            elif original_X_test is not None and len(original_X_test) > 1:
                # Use other test samples as background (exclude current sample)
                background_indices = [
                    i for i in range(len(original_X_test)) if i != spectrum_idx
                ]
                if len(background_indices) >= 10:
                    # Use subset for efficiency
                    selected_bg_indices = np.random.choice(
                        background_indices,
                        min(50, len(background_indices)),
                        replace=False,
                    )
                    ml_property.X_train = original_X_test[selected_bg_indices]
                    if original_y_test is not None:
                        ml_property.y_train = original_y_test[selected_bg_indices]
                    else:
                        # Create dummy labels
                        available_labels = ["benign", "cancer"]
                        if (
                            hasattr(ml_property, "metadata")
                            and "labels" in ml_property.metadata
                        ):
                            available_labels = ml_property.metadata["labels"]
                        ml_property.y_train = np.random.choice(
                            available_labels, len(selected_bg_indices)
                        )

                    background_data_source = "test_data_subset"
                    console_log(
                        f"Using test data subset with {len(ml_property.X_train)} samples for SHAP background"
                    )
                else:
                    raise ValueError(
                        "Insufficient test data for background (need at least 10 samples)"
                    )

            # Priority 3: Create synthetic background from provided test_spectra
            elif test_spectra is not None:
                console_log("Creating background data from provided test_spectra...")

                all_background_data = []
                for container in test_spectra:
                    for spec in container.spectral_data:
                        if len(spec) == len(ml_property.common_axis):
                            all_background_data.append(spec)
                        else:
                            # Interpolate if needed
                            interp_spec = np.interp(
                                ml_property.common_axis, container.spectral_axis, spec
                            )
                            all_background_data.append(interp_spec)

                        # Limit background size for efficiency
                        if len(all_background_data) >= 100:
                            break

                    if len(all_background_data) >= 100:
                        break

                if len(all_background_data) < 10:
                    # Create synthetic background if still not enough
                    console_log(
                        "Warning: Very few background samples. Creating synthetic data."
                    )
                    n_synthetic = max(20, 50 - len(all_background_data))
                    for _ in range(n_synthetic):
                        synthetic_spec = spectrum + np.random.normal(
                            0, spectrum.std() * 0.1, len(spectrum)
                        )
                        all_background_data.append(synthetic_spec)

                ml_property.X_train = np.array(all_background_data)

                # Create corresponding labels
                available_labels = ["benign", "cancer"]
                if (
                    hasattr(ml_property, "metadata")
                    and "labels" in ml_property.metadata
                ):
                    available_labels = ml_property.metadata["labels"]
                ml_property.y_train = np.random.choice(
                    available_labels, len(all_background_data)
                )

                background_data_source = "synthetic_from_test_spectra"
                console_log(
                    f"Created background dataset with {len(ml_property.X_train)} samples"
                )

            else:
                raise ValueError("No background data available for SHAP explanation")

            # Get SHAP explanation
            shap_result = visualizer_instance.shap_explain(
                show_plots=show_shap_plots,
                max_display=max_display,
                nsamples=nsamples,
                fast_mode=fast_mode,
                max_test_samples=1,
                max_background_samples=min(50, len(ml_property.X_train)),
                reduce_features=False,
                max_features=single_spectrum_data.shape[1],
                use_base_estimator=kwargs.get("use_base_estimator", True),
                force_kernel_explainer=kwargs.get("force_kernel_explainer", False),
                shap_output_mode=kwargs.get("shap_output_mode", "auto"),
                kernel_nsamples_multiplier=kwargs.get("kernel_nsamples_multiplier", 1),
            )

            # Add background data source info to result
            if shap_result and shap_result.get("success"):
                shap_result["background_data_source"] = background_data_source

            # Restore original data
            if original_X_test is not None:
                ml_property.X_test = original_X_test
            else:
                ml_property.X_test = None

            if original_y_test is not None:
                ml_property.y_test = original_y_test
            else:
                ml_property.y_test = None

            if original_X_train is not None:
                ml_property.X_train = original_X_train
            else:
                ml_property.X_train = None

            if original_y_train is not None:
                ml_property.y_train = original_y_train
            else:
                ml_property.y_train = None

            return shap_result

        except Exception as e:
            console_log(f"Error in SHAP explanation for spectrum {spectrum_idx+1}: {e}")

            # Restore original data even on error
            if "original_X_test" in locals() and original_X_test is not None:
                ml_property.X_test = original_X_test
            if "original_y_test" in locals() and original_y_test is not None:
                ml_property.y_test = original_y_test
            if "original_X_train" in locals() and original_X_train is not None:
                ml_property.X_train = original_X_train
            if "original_y_train" in locals() and original_y_train is not None:
                ml_property.y_train = original_y_train

            return None

    def _process_shap_values(shap_result, model):
        """Process SHAP values and extract top contributors."""
        if not shap_result or not shap_result.get("success"):
            return None, None, None

        shap_values_raw = shap_result["shap_values"]

        # Extract SHAP values based on model type
        if hasattr(model, "_model") and isinstance(model._model, SVC):
            if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
                sv_positive = shap_values_raw[1]
                if isinstance(sv_positive, np.ndarray):
                    if sv_positive.ndim == 1:
                        shap_values = sv_positive
                    elif sv_positive.ndim == 2 and sv_positive.shape[0] == 1:
                        shap_values = sv_positive[0]
                    else:
                        shap_values = (
                            sv_positive[0] if sv_positive.shape[0] > 0 else sv_positive
                        )
                else:
                    shap_values = np.asarray(sv_positive)
            else:
                if isinstance(shap_values_raw, np.ndarray):
                    shap_values = shap_values_raw.flatten()
                elif hasattr(shap_values_raw, "values"):
                    shap_values = np.asarray(shap_values_raw.values).flatten()
                else:
                    shap_values = np.asarray(
                        shap_values_raw[0]
                        if isinstance(shap_values_raw, list)
                        and len(shap_values_raw) > 0
                        else shap_values_raw
                    )
        else:
            # Standard extraction for other model types
            if isinstance(shap_values_raw, list) and len(shap_values_raw) > 0:
                if len(shap_values_raw) == 2:
                    if isinstance(shap_values_raw[1], np.ndarray):
                        if shap_values_raw[1].ndim > 1:
                            shap_values = shap_values_raw[1][0]
                        else:
                            shap_values = shap_values_raw[1]
                    else:
                        shap_values = shap_values_raw[1]
                else:
                    if isinstance(shap_values_raw[0], np.ndarray):
                        if shap_values_raw[0].ndim > 1:
                            shap_values = shap_values_raw[0][0]
                        else:
                            shap_values = shap_values_raw[0]
                    else:
                        shap_values = shap_values_raw[0]
            else:
                if hasattr(shap_values_raw, "values"):
                    shap_values_array = shap_values_raw.values
                    if shap_values_array.ndim == 3:
                        if shap_values_array.shape[2] == 2:
                            shap_values = shap_values_array[0, :, 1]
                        else:
                            shap_values = shap_values_array[0, :, 0]
                    elif shap_values_array.ndim == 2:
                        shap_values = shap_values_array[0]
                    else:
                        shap_values = shap_values_array
                elif hasattr(shap_values_raw, "__getitem__"):
                    shap_values = shap_values_raw[0]
                else:
                    shap_values = shap_values_raw

        # Ensure shap_values is proper format
        if hasattr(shap_values, "ndim") and shap_values.ndim > 1:
            shap_values = shap_values.flatten()

        if not isinstance(shap_values, np.ndarray):
            shap_values = np.asarray(shap_values)

        shap_values = shap_values.astype(float)

        # Process common axis
        if hasattr(ml_property, "common_axis") and ml_property.common_axis is not None:
            try:
                if isinstance(ml_property.common_axis, (np.ndarray, list, tuple)):
                    common_axis_length = len(ml_property.common_axis)
                    common_axis_subset = np.asarray(
                        ml_property.common_axis, dtype=float
                    )
                else:
                    common_axis_length = len(shap_values)
                    common_axis_subset = np.arange(len(shap_values), dtype=float)
            except (TypeError, ValueError) as e:
                common_axis_length = len(shap_values)
                common_axis_subset = np.arange(len(shap_values), dtype=float)
        else:
            common_axis_length = len(shap_values)
            common_axis_subset = np.arange(len(shap_values), dtype=float)

        if not isinstance(common_axis_subset, np.ndarray):
            common_axis_subset = np.asarray(common_axis_subset, dtype=float)
        else:
            common_axis_subset = common_axis_subset.astype(float)

        if len(shap_values) != common_axis_length:
            min_length = min(len(shap_values), common_axis_length)
            shap_values = shap_values[:min_length]
            if len(common_axis_subset) >= min_length:
                common_axis_subset = common_axis_subset[:min_length]
            else:
                common_axis_subset = np.arange(min_length, dtype=float)

        return shap_values, common_axis_subset, shap_result

    def _extract_top_contributors(shap_values, common_axis_subset):
        """Extract top positive and negative contributors."""
        sorted_idx = np.argsort(shap_values)
        top_negative_idx = sorted_idx[:5]
        top_positive_idx = sorted_idx[-5:][::-1]

        # Safe extraction function
        def safe_extract_to_float(arr_input, idx_input):
            try:
                current_idx = int(
                    idx_input.item() if hasattr(idx_input, "item") else idx_input
                )
                val = arr_input[current_idx]
                if hasattr(val, "item"):
                    return float(val.item())
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, np.ndarray):
                    if val.size == 1:
                        return float(val.flat[0])
                    else:
                        return float(val.flat[0])
                if isinstance(val, list):
                    if len(val) == 1 and isinstance(val[0], (int, float)):
                        return float(val[0])
                    elif len(val) > 0 and isinstance(val[0], (int, float)):
                        return float(val[0])
                    else:
                        return 0.0
                return float(val)
            except Exception:
                return 0.0

        try:
            top_negative = [
                (
                    safe_extract_to_float(common_axis_subset, idx),
                    safe_extract_to_float(shap_values, idx),
                )
                for idx in top_negative_idx
            ]

            top_positive = [
                (
                    safe_extract_to_float(common_axis_subset, idx),
                    safe_extract_to_float(shap_values, idx),
                )
                for idx in top_positive_idx
            ]
        except Exception as e:
            console_log(f"Error with safe extraction method: {e}")
            top_positive = [(0.0, 0.0)] * 5
            top_negative = [(0.0, 0.0)] * 5

        return top_positive, top_negative

    def _add_lime_explanation_data(spec_result, lime_results, spec_info):
        """Add LIME explanation data if available."""
        if lime_results and lime_results["success"]:
            # Find corresponding LIME explanation for this spectrum
            for lime_exp in lime_results["explanation_objects"]:
                if lime_exp["sample_idx"] == spec_info["spectrum_idx"]:
                    spec_result["lime_explanation"] = {
                        "sample_idx": lime_exp["sample_idx"],
                        "true_label": lime_exp["true_label"],
                        "predicted_label": lime_exp["predicted_label"],
                        "confidence": lime_exp["confidence"],
                        "explanation_available": True,
                    }
                    break
        return spec_result

    def _create_summary_statistics(results):
        """Create summary statistics for the inspection results."""
        if results:
            correct_predictions = sum(
                1 for r in results if r["prediction"]["is_correct"]
            )
            accuracy = correct_predictions / len(results)

            summary = {
                "total_spectra_analyzed": len(results),
                "correct_predictions": correct_predictions,
                "accuracy": accuracy,
                "class_filter": class_to_inspect,
                "random_seed": seed,
                "shap_explanations_generated": sum(
                    1 for r in results if r["shap_explanation"] is not None
                ),
                "lime_explanations_generated": sum(
                    1 for r in results if r["lime_explanation"] is not None
                ),
                "lime_results_summary": None,
            }
        else:
            summary = {
                "total_spectra_analyzed": 0,
                "error": "No spectra could be analyzed successfully",
            }

        return summary

    # === MAIN EXECUTION ===
    try:
        # 1. Validate and prepare ML_PROPERTY
        actual_model, ml_type = _validate_and_prepare_ml_property()

        # 2. Prepare test data
        all_spectra = _prepare_test_data()

        # 3. Filter and select spectra
        selected_spectra = _filter_and_select_spectra(all_spectra)
        if isinstance(selected_spectra, dict) and "results" in selected_spectra:
            return selected_spectra  # Early return for error case

        console_log(f"Inspecting {len(selected_spectra)} spectra...")

        # 4. Generate LIME explanations if requested
        lime_results = _generate_lime_explanations(selected_spectra)

        # 5. Get unified prediction function
        unified_predict = _get_unified_prediction_function(actual_model)

        results = []

        # 6. Process each spectrum
        for spectrum_idx, spec_info in enumerate(selected_spectra):
            console_log(
                f"\nAnalyzing spectrum {spectrum_idx+1}/{len(selected_spectra)}..."
            )

            spectrum = spec_info["spectrum"]

            # 7. Make prediction
            pred_label, confidence, processed_spectrum = _predict_single_spectrum(
                spectrum, spec_info, actual_model, unified_predict
            )

            if pred_label is None:
                continue

            # 8. Initialize spectrum result
            spec_result = {
                "spectrum_info": spec_info,
                "prediction": {
                    "label": pred_label,
                    "confidence": float(confidence) if confidence is not None else None,
                    "is_correct": pred_label == spec_info["true_label"],
                },
                "shap_explanation": None,
                "lime_explanation": None,
            }

            # 9. Get SHAP explanation if requested
            if show_plots:
                shap_result = _get_shap_explanation(
                    processed_spectrum, spec_info, spectrum_idx
                )

                if shap_result:
                    # Process SHAP values
                    shap_values, common_axis_subset, shap_result_processed = (
                        _process_shap_values(shap_result, ml_property)
                    )

                    if shap_values is not None:
                        # Extract top contributors
                        top_positive, top_negative = _extract_top_contributors(
                            shap_values, common_axis_subset
                        )

                        # Generate visualizations
                        spectrum_with_highlights_spectrum(
                            processed_spectrum,
                            common_axis_subset,
                            top_positive,
                            top_negative,
                            positive_label,
                            negative_label,
                            spec_info,
                            pred_label,
                            confidence,
                            spectrum_idx,
                        )

                        create_shap_plots(
                            spectrum_idx,
                            spec_info,
                            shap_values,
                            common_axis_subset,
                            top_positive,
                            top_negative,
                            positive_label,
                            negative_label,
                        )

                        create_enhanced_table(
                            visualizer_instance,
                            spectrum_idx,
                            spec_info,
                            pred_label,
                            shap_result_processed,
                            shap_values,
                            top_positive,
                            top_negative,
                            positive_label,
                            negative_label,
                            confidence,
                        )

                        # Store SHAP results
                        spec_result["shap_explanation"] = {
                            "top_positive_contributors": top_positive,
                            "top_negative_contributors": top_negative,
                            "shap_values": (
                                shap_values.tolist()
                                if hasattr(shap_values, "tolist")
                                else list(shap_values)
                            ),
                        }

            # 10. Add LIME explanation data if available
            spec_result = _add_lime_explanation_data(
                spec_result, lime_results, spec_info
            )

            results.append(spec_result)

            # 11. Log summary for this spectrum
            console_log(
                f"Spectrum {spectrum_idx+1} Summary [Container: {spec_info['container_idx']}, Index: {spec_info['spectrum_idx']}]:"
            )
            console_log(f"  True Label: {spec_info['true_label']}")
            console_log(f"  Predicted: {pred_label} (Confidence: {confidence:.3f})")
            console_log(
                f"  Correct: {'Yes' if pred_label == spec_info['true_label'] else 'No'}"
            )

            if spec_result["shap_explanation"]:
                top_pos = spec_result["shap_explanation"]["top_positive_contributors"][
                    0
                ]
                top_neg = spec_result["shap_explanation"]["top_negative_contributors"][
                    0
                ]
                console_log(
                    f"  Top {positive_label} contributor: {top_pos[0]:.1f} cm⁻¹ (SHAP: {top_pos[1]:.4f})"
                )
                console_log(
                    f"  Top {negative_label} contributor: {top_neg[0]:.1f} cm⁻¹ (SHAP: {top_neg[1]:.4f})"
                )

            if spec_result["lime_explanation"]:
                console_log(f"  LIME explanation: Available")

        # 12. Create summary statistics
        summary = _create_summary_statistics(results)
        if lime_results:
            summary["lime_results_summary"] = lime_results

        return {
            "results": results,
            "summary": summary,
            "lime_full_results": lime_results,
            "ml_type_used": ml_type,
        }

    except Exception as e:
        console_log(f"Error in inspect_spectra: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# ========================================================================
# Helper visualization functions (Phase 4B)
# These functions are used by inspect_spectra and other methods
# ========================================================================


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
    )

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
    fig2 = plt.figure(figsize=(12, 6))
    colors = ["red" if sv > 0 else "blue" for sv in shap_values]

    # Plot against wavenumbers instead of feature indices
    plt.bar(common_axis_subset, shap_values, color=colors, alpha=0.6, width=1.0)

    # Add color legend
    red_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="red", alpha=0.6, label=f"{positive_label} (Positive)"
    )
    blue_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="blue", alpha=0.6, label=f"{negative_label} (Negative)"
    )
    plt.legend(handles=[red_patch, blue_patch])

    plt.title(
        f"SHAP Values - Spectrum {spectrum_idx+1} [Container: {spec_info['container_idx']}, Index: {spec_info['spectrum_idx']}]",
        fontsize=12,
        fontweight="bold",
    )
    plt.ylabel("SHAP Value")
    plt.xlabel("Raman Shift (cm⁻¹)")

    # Set y-axis limits based on SHAP values range
    shap_min = np.min(shap_values)
    shap_max = np.max(shap_values)
    shap_range = shap_max - shap_min
    padding = shap_range * 0.05  # 5% padding
    plt.ylim(shap_min - padding, shap_max + padding)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig2


def create_enhanced_table(
    visualizer_instance: Any,
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
        visualizer_instance (Any): Instance of the RamanVisualizer class for accessing peak assignment methods.
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
    positive_label = positive_label.upper()
    negative_label = negative_label.upper()

    fig3 = plt.figure(figsize=(14, 12))
    plt.axis("off")

    # Calculate prediction mathematics
    total_shap_sum = np.sum(shap_values)
    expected_val = shap_result.get("expected_value", 0)
    if isinstance(expected_val, (list, np.ndarray)):
        expected_val = float(expected_val[0]) if len(expected_val) > 0 else 0.0
    else:
        expected_val = float(expected_val)

    prediction_score = expected_val + total_shap_sum

    # Calculate contribution percentages
    positive_sum = sum(sv for _, sv in top_positive)
    negative_sum = sum(sv for _, sv in top_negative)
    total_abs_contribution = abs(positive_sum) + abs(negative_sum)

    # Get peak assignments for top contributors
    def get_peak_info(wavenumber):
        """Helper function to get peak assignment with fallback"""
        try:
            assignment_info = visualizer_instance.get_peak_assignment(
                wavenumber, pick_near=True, tolerance=10
            )
            if assignment_info.get("assignment") == "Not Found":
                # Try with larger tolerance
                assignment_info = visualizer_instance.get_peak_assignment(
                    wavenumber, pick_near=True, tolerance=20
                )
            return assignment_info
        except Exception as e:
            console_log(
                f"Error in get_peak_info for {wavenumber}: {e} \n {traceback.format_exc()}"
            )
            return {"assignment": str(e), "peak": wavenumber}

    # Create enhanced table data with 4 columns
    table_data = []

    # Header section with mathematical explanation
    table_data.append(["MATHEMATICAL PREDICTION BREAKDOWN", "", "", ""])
    table_data.append(["Base Model Expectation", f"{expected_val:.4f}", "", ""])
    table_data.append(["Total SHAP Contribution", f"{total_shap_sum:.4f}", "", ""])
    table_data.append(["Final Prediction Score", f"{prediction_score:.4f}", "", ""])
    table_data.append(["Predicted Class", pred_label, f"({confidence:.3f})", ""])
    table_data.append(["", "", "", ""])  # Empty row

    # positive_label contributors section with peak assignments
    table_data.append(
        [
            f"Top {positive_label} Contributors",
            "Wavenumber",
            "SHAP Value",
            "Peak Assignment",
        ]
    )

    for idx, (wn, sv) in enumerate(top_positive):
        percentage = (
            (abs(sv) / total_abs_contribution * 100)
            if total_abs_contribution > 0
            else 0
        )

        # Get peak assignment
        peak_info = get_peak_info(wn)
        assignment = peak_info.get("assignment", "Unknown")

        # Truncate assignment if too long
        if len(assignment) > 50:
            assignment = assignment[:47] + "..."

        # Add distance info if found nearby peak
        if "distance" in peak_info and peak_info["distance"] > 0:
            wn_display = f"{wn:.1f} cm⁻¹ (~{peak_info['peak']} cm⁻¹)"
        else:
            wn_display = f"{wn:.1f} cm⁻¹"

        table_data.append(
            [f"#{idx+1}", wn_display, f"{sv:.4f} ({percentage:.1f}%)", assignment]
        )

    # Summary row for positive contributors
    table_data.append(
        [
            f"{positive_label} Total:",
            f"Net Push: +{positive_sum:.4f}",
            f"({abs(positive_sum)/total_abs_contribution*100:.1f}%)",
            "",
        ]
    )
    table_data.append(["", "", "", ""])  # Empty row

    # negative_label contributors section with peak assignments
    table_data.append(
        [
            f"Top {negative_label} Contributors",
            "Wavenumber",
            "SHAP Value",
            "Peak Assignment",
        ]
    )

    for idx, (wn, sv) in enumerate(top_negative):
        percentage = (
            (abs(sv) / total_abs_contribution * 100)
            if total_abs_contribution > 0
            else 0
        )

        # Get peak assignment
        peak_info = get_peak_info(wn)
        assignment = peak_info.get("assignment", "Unknown")

        # Truncate assignment if too long
        if len(assignment) > 50:
            assignment = assignment[:47] + "..."

        # Add distance info if found nearby peak
        if "distance" in peak_info and peak_info["distance"] > 0:
            wn_display = f"{wn:.1f} cm⁻¹ (~{peak_info['peak']} cm⁻¹)"
        else:
            wn_display = f"{wn:.1f} cm⁻¹"

        table_data.append(
            [f"#{idx+1}", wn_display, f"{sv:.4f} ({percentage:.1f}%)", assignment]
        )

    # Summary row for negative contributors
    table_data.append(
        [
            f"{negative_label} Total:",
            f"Net Push: {negative_sum:.4f}",
            f"({abs(negative_sum)/total_abs_contribution*100:.1f}%)",
            "",
        ]
    )
    table_data.append(["", "", "", ""])  # Empty row

    # Enhanced decision logic with peak context
    net_direction = positive_label if total_shap_sum > 0 else negative_label
    confidence_explanation = (
        "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
    )

    # Get the most influential peak
    most_influential_wn = (
        top_positive[0][0]
        if abs(top_positive[0][1]) > abs(top_negative[0][1])
        else top_negative[0][0]
    )
    most_influential_sv = (
        top_positive[0][1]
        if abs(top_positive[0][1]) > abs(top_negative[0][1])
        else top_negative[0][1]
    )
    most_influential_info = get_peak_info(most_influential_wn)
    most_influential_assignment = most_influential_info.get("assignment", "Unknown")

    # Truncate if too long
    if len(most_influential_assignment) > 60:
        most_influential_assignment = most_influential_assignment[:57] + "..."

    table_data.append(["DECISION LOGIC & PEAK ANALYSIS", "", "", ""])
    table_data.append(
        ["Net SHAP Direction", f"→ {net_direction}", f"(Δ={total_shap_sum:.4f})", ""]
    )
    table_data.append(
        ["Confidence Level", confidence_explanation, f"({confidence:.3f})", ""]
    )
    table_data.append(
        [
            "Key Decision Factor",
            f"{most_influential_wn:.1f} cm⁻¹",
            f"SHAP: {most_influential_sv:.4f}",
            most_influential_assignment,
        ]
    )

    # Add biological interpretation if available
    biological_context = []
    for wn, sv in top_positive + top_negative:
        peak_info = get_peak_info(wn)
        assignment = peak_info.get("assignment", "")
        if assignment and assignment not in ["Not Found", "Unknown", "Error in lookup"]:
            # Look for biological keywords
            bio_keywords = [
                "protein",
                "lipid",
                "nucleic",
                "collagen",
                "DNA",
                "RNA",
                "amino",
                "carbohydrate",
                "phosphate",
                "membrane",
            ]
            if any(keyword.lower() in assignment.lower() for keyword in bio_keywords):
                biological_context.append(
                    (
                        wn,
                        assignment[:40] + "..." if len(assignment) > 40 else assignment,
                    )
                )

    if biological_context:
        table_data.append(["", "", "", ""])  # Empty row
        table_data.append(["BIOLOGICAL CONTEXT", "", "", ""])
        # Show top 3
        for wn, bio_info in biological_context[:3]:
            table_data.append(["Biomolecular Signal", f"{wn:.1f} cm⁻¹", "", bio_info])

    # Create table with adjusted column widths for 4 columns
    table = plt.table(
        cellText=table_data,
        cellLoc="left",
        loc="center",
        colWidths=[0.25, 0.20, 0.20, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Enhanced styling with fixed coloring for positive (red) and negative (blue)
    for table_row_idx, row in enumerate(table_data):
        for j in range(4):
            cell = table[(table_row_idx, j)]

            # Header sections
            if (
                "BREAKDOWN" in str(row[0])
                or "DECISION LOGIC" in str(row[0])
                or "BIOLOGICAL CONTEXT" in str(row[0])
            ):
                cell.set_facecolor("#E8E8E8")
                cell.set_text_props(weight="bold", color="black")
                cell.set_height(0.08)

            # Positive label header
            elif f"Top {positive_label}" in str(row[0]):
                cell.set_facecolor("#FFE6E6")
                cell.set_text_props(weight="bold", color="#8B0000")

            # Negative label header
            elif f"Top {negative_label}" in str(row[0]):
                cell.set_facecolor("#E6F3FF")
                cell.set_text_props(weight="bold", color="#000080")

            # Individual positive contributor rows
            elif (
                str(row[0]).startswith("#")
                and table_row_idx > 7
                and table_row_idx < len(table_data)
                and "Top " + positive_label
                in table_data[table_row_idx - int(row[0][1])][0]
            ):
                cell.set_facecolor("#FFF0F0")

            # Individual negative contributor rows
            elif str(row[0]).startswith("#") and table_row_idx > 15:
                cell.set_facecolor("#F0F8FF")

            # Total rows with special emphasis
            elif f"{positive_label} Total:" in str(row[0]):
                cell.set_facecolor("#FFE6E6")
                cell.set_text_props(weight="bold", color="#8B0000")

            elif f"{negative_label} Total:" in str(row[0]):
                cell.set_facecolor("#E6F3FF")
                cell.set_text_props(weight="bold", color="#000080")

            # Net direction row
            elif "Net SHAP Direction" in str(row[0]):
                color = "#FFE6E6" if net_direction == positive_label else "#E6F3FF"
                text_color = "#8B0000" if net_direction == positive_label else "#000080"
                cell.set_facecolor(color)
                if j == 1 or j == 2:
                    cell.set_text_props(weight="bold", color=text_color)

            # Key decision factor
            elif "Key Decision Factor" in str(row[0]):
                color = "#FFF0F0" if most_influential_sv > 0 else "#F0F8FF"
                text_color = "#8B0000" if most_influential_sv > 0 else "#000080"
                cell.set_facecolor(color)
                if j == 2:
                    cell.set_text_props(color=text_color)

            # Peak assignment column styling
            if j == 3 and str(row[j]) and str(row[j]) not in ["", "Peak Assignment"]:
                cell.set_text_props(style="italic", size=8)

            # Mathematical values
            if j in [1, 2] and any(
                char in str(row[j]) for char in ["±", "(", ":", "Δ"]
            ):
                cell.set_text_props(family="monospace", size=8)

    plt.suptitle(
        f"ENHANCED PREDICTION ANALYSIS WITH PEAK ASSIGNMENTS\nSpectrum {spectrum_idx+1} [Container: {spec_info['container_idx']}, Index: {spec_info['spectrum_idx']}]",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return fig3


def plot_institution_distribution(
    spectral_containers: List,
    container_labels: List[str],
    container_names: Optional[List[str]] = None,
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
    save_path: Optional[str] = None,
    color_palette: str = "tab10",
    interpolate_to_common_axis: bool = True,
    common_axis: Optional[np.ndarray] = None,
    show_class_info: bool = True,
    class_labels: Optional[List[str]] = None,
) -> dict:
    """
    Plot t-SNE visualization of spectral data distribution across different containers/institutions.

    Parameters
    ----------
    spectral_containers : List
        List of SpectralContainer objects to compare
    container_labels : List[str]
        Labels for each container (e.g., ['CHUM_benign', 'CHUM_cancer', 'UHN_benign', ...])
    container_names : List[str], optional
        Institution/group names for each container. If None, extracted from container_labels
    sample_limit : int, default=500
        Maximum number of samples to use for t-SNE (for performance)
    perplexity : int, default=30
        t-SNE perplexity parameter
    n_iter : int, default=1000
        Number of iterations for t-SNE
    random_state : int, default=42
        Random seed for reproducibility
    figsize : tuple, default=(12, 8)
        Figure size (width, height)
    alpha : float, default=0.6
        Point transparency
    point_size : int, default=50
        Size of scatter plot points
    title : str, default="Institution Distribution in Feature Space"
        Plot title
    show_legend : bool, default=True
        Whether to show legend
    save_plot : bool, default=False
        Whether to save the plot
    save_path : str, optional
        Path to save the plot (if save_plot=True)
    color_palette : str, default='tab10'
        Matplotlib colormap name
    interpolate_to_common_axis : bool, default=True
        Whether to interpolate all spectra to a common axis
    common_axis : np.ndarray, optional
        Common axis for interpolation. If None, uses the first container's axis
    show_class_info : bool, default=True
        Whether to show class information in the legend
    class_labels : List[str], optional
        Class labels (e.g., ['benign', 'cancer']) for extracting class info

    Returns
    -------
    dict
        Results dictionary containing embedded data, labels, and metadata
    """
    try:
        console_log("🔬 Starting spectral distribution analysis...")

        # Validate inputs
        if len(spectral_containers) != len(container_labels):
            raise ValueError(
                f"Number of containers ({len(spectral_containers)}) must match number of labels ({len(container_labels)})"
            )

        # Extract institution names if not provided
        if container_names is None:
            container_names = []
            for label in container_labels:
                inst_name = label.split("_")[0] if "_" in label else label
                container_names.append(inst_name)

        # Get unique institutions
        unique_institutions = list(set(container_names))
        console_log(
            f"📊 Found {len(unique_institutions)} unique institutions: {unique_institutions}"
        )

        # Determine common axis for interpolation
        if interpolate_to_common_axis:
            if common_axis is None:
                # Use the axis from the container with the most features
                max_features = 0
                best_axis = None
                for container in spectral_containers:
                    if len(container.spectral_axis) > max_features:
                        max_features = len(container.spectral_axis)
                        best_axis = container.spectral_axis
                common_axis = best_axis
                console_log(f"🔧 Using common axis with {len(common_axis)} features")
            else:
                console_log(
                    f"🔧 Using provided common axis with {len(common_axis)} features"
                )

        # Collect all spectral data
        all_data = []
        all_labels = []
        all_institutions = []
        all_classes = []
        container_info = []

        for i, (container, label, inst_name) in enumerate(
            zip(spectral_containers, container_labels, container_names)
        ):
            if container.spectral_data is None or len(container.spectral_data) == 0:
                console_log(
                    f"⚠️  Warning: Container {i} ({label}) has no spectral data, skipping..."
                )
                continue

            # Extract class information if available
            class_info = "unknown"
            if class_labels:
                for class_label in class_labels:
                    if class_label.lower() in label.lower():
                        class_info = class_label
                        break

            console_log(
                f"📈 Processing container {i+1}/{len(spectral_containers)}: {label} ({len(container.spectral_data)} spectra)"
            )

            # Process each spectrum in the container
            for spectrum_idx, spectrum in enumerate(container.spectral_data):
                try:
                    # Interpolate to common axis if needed
                    if interpolate_to_common_axis and len(
                        container.spectral_axis
                    ) != len(common_axis):
                        interpolated_spectrum = np.interp(
                            common_axis, container.spectral_axis, spectrum
                        )
                        all_data.append(interpolated_spectrum)
                    else:
                        all_data.append(spectrum)

                    all_labels.append(label)
                    all_institutions.append(inst_name)
                    all_classes.append(class_info)
                    container_info.append(
                        {
                            "container_idx": i,
                            "spectrum_idx": spectrum_idx,
                            "label": label,
                            "institution": inst_name,
                            "class": class_info,
                        }
                    )

                except Exception as e:
                    console_log(
                        f"⚠️  Warning: Error processing spectrum {spectrum_idx} in container {i}: {e}"
                    )
                    continue

        if not all_data:
            raise ValueError("No valid spectral data found in any container")

        # Convert to numpy array
        all_data = np.array(all_data)
        all_labels = np.array(all_labels)
        all_institutions = np.array(all_institutions)
        all_classes = np.array(all_classes)

        console_log(
            f"📊 Total collected data: {all_data.shape[0]} spectra with {all_data.shape[1]} features"
        )

        # Sample data if too large
        if len(all_data) > sample_limit:
            console_log(
                f"🎲 Sampling {sample_limit} spectra from {len(all_data)} total for t-SNE performance"
            )
            indices = np.random.choice(len(all_data), sample_limit, replace=False)
            all_data = all_data[indices]
            all_labels = all_labels[indices]
            all_institutions = all_institutions[indices]
            all_classes = all_classes[indices]
            container_info = [container_info[i] for i in indices]

        # Perform t-SNE
        console_log(
            f"🧮 Running t-SNE with perplexity={perplexity}, n_iter={n_iter}..."
        )
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
            verbose=1,
        )
        embedded = tsne.fit_transform(all_data)

        # Create the plot
        plt.figure(figsize=figsize)

        # Get colors
        cmap = plt.get_cmap(color_palette)
        colors = [
            cmap(i / len(unique_institutions)) for i in range(len(unique_institutions))
        ]
        institution_colors = {
            inst: colors[i] for i, inst in enumerate(unique_institutions)
        }

        # Plot by institution
        for inst in unique_institutions:
            mask = all_institutions == inst

            # Count classes for this institution if class info is available
            class_counts = {}
            if show_class_info and class_labels:
                for class_label in class_labels:
                    class_count = np.sum(
                        (all_institutions == inst) & (all_classes == class_label)
                    )
                    if class_count > 0:
                        class_counts[class_label] = class_count

            # Create legend label
            if class_counts:
                class_info_str = ", ".join(
                    [f"{cls}:{cnt}" for cls, cnt in class_counts.items()]
                )
                legend_label = f"{inst} ({class_info_str})"
            else:
                legend_label = f"{inst} (n={np.sum(mask)})"

            plt.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                label=legend_label,
                alpha=alpha,
                s=point_size,
                color=institution_colors[inst],
            )

        # Customize plot
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("t-SNE Component 1", fontsize=12)
        plt.ylabel("t-SNE Component 2", fontsize=12)

        if show_legend:
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot if requested
        if save_plot:
            if save_path is None:
                save_path = f"institution_distribution_tsne_{random_state}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            console_log(f"💾 Plot saved to: {save_path}")

        plt.show()

        # Calculate and display statistics
        console_log("\n📈 Distribution Statistics:")
        console_log("=" * 50)
        for inst in unique_institutions:
            mask = all_institutions == inst
            count = np.sum(mask)
            percentage = (count / len(all_institutions)) * 100
            console_log(f"{inst}: {count} spectra ({percentage:.1f}%)")

            if show_class_info and class_labels:
                for class_label in class_labels:
                    class_count = np.sum(
                        (all_institutions == inst) & (all_classes == class_label)
                    )
                    if class_count > 0:
                        class_percentage = (class_count / count) * 100
                        console_log(
                            f"  └─ {class_label}: {class_count} ({class_percentage:.1f}%)"
                        )

        # Return results
        results = {
            "success": True,
            "embedded_data": embedded,
            "institutions": all_institutions,
            "labels": all_labels,
            "classes": all_classes,
            "container_info": container_info,
            "unique_institutions": unique_institutions,
            "tsne_params": {
                "perplexity": perplexity,
                "n_iter": n_iter,
                "random_state": random_state,
            },
            "data_info": {
                "total_spectra": len(all_data),
                "n_features": all_data.shape[1],
                "sampled": len(all_data) < len(container_info),
            },
        }

        console_log(f"\n✅ Analysis completed successfully!")
        return results

    except Exception as e:
        console_log(f"❌ Error in plot_institution_distribution: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
