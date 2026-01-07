"""
Model Explainability Module for Raman Spectroscopy ML Models

This module provides SHAP (SHapley Additive exPlanations) analysis for machine learning
models trained on Raman spectroscopy data. It handles various model types including:
- Random Forest classifiers (TreeExplainer)
- Support Vector Machines (KernelExplainer/LinearExplainer)
- Calibrated classifiers (with base estimator extraction)
- Generic sklearn models (KernelExplainer)

Key Features:
- Automatic model type detection and optimal SHAP explainer selection
- Performance optimization through feature selection and sample limiting
- Support for both binary and multiclass classification
- Comprehensive visualization (summary plots, feature importance)
- Flexible output modes: auto, full, sparse

Example Usage:
    ```python
    from functions.visualization import shap_explain

    # With ML_PROPERTY (auto-detection)
    results = shap_explain(
        ml_property=my_ml_property,
        nsamples=50,
        show_plots=True,
        max_display=15
    )

    # With performance optimization
    results = shap_explain(
        ml_property=my_ml_property,
        fast_mode=True,
        reduce_features=True,
        max_features=300
    )

    # Force specific explainer
    results = shap_explain(
        ml_property=my_ml_property,
        force_kernel_explainer=True,
        kernel_nsamples_multiplier=2.0
    )
    ```

Phase 3 Refactoring:
    Extracted from RamanVisualizer class to improve modularity and testability.
    Original method: 959 lines with 10 nested helper functions.
    New module: Standalone functions with comprehensive documentation.
"""

import time
import numpy as np
import shap
from typing import Union, List, Optional, Dict, Tuple, Any
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Import for type hints
try:
    from functions.ML import RamanML, MLModel
except ImportError:
    RamanML = Any
    MLModel = Any

# Import for logging
from functions.configs import console_log, create_logs


def shap_explain(
    ml_property: Optional[Union[RamanML, MLModel]] = None,
    nsamples: int = 50,
    show_plots: bool = True,
    max_display: int = 15,
    wavenumber_axis: Optional[np.ndarray] = None,
    # Performance optimization parameters
    max_background_samples: int = 50,
    max_test_samples: int = 20,
    reduce_features: bool = True,
    max_features: int = 300,
    use_kmeans_sampling: bool = False,
    fast_mode: bool = False,
    # Model-specific parameters
    use_base_estimator: bool = True,
    force_kernel_explainer: bool = False,
    shap_output_mode: str = "auto",
    kernel_nsamples_multiplier: float = 2.0,
) -> Dict[str, Any]:
    """
    Generate SHAP explanations for ML model predictions using stored training/test data.

    SHAP (SHapley Additive exPlanations) provides model-agnostic interpretability by
    computing feature contributions to individual predictions. This function automatically
    selects the optimal SHAP explainer based on model type and provides comprehensive
    feature importance analysis.

    Args:
        ml_property: ML_PROPERTY instance with trained model and data.
            Can be RamanML or MLModel type.
        nsamples: Number of samples for SHAP KernelExplainer.
            Lower values = faster computation but less accurate.
            Recommended: 50-100 for balanced speed/accuracy.
        show_plots: Whether to generate and display SHAP plots.
            Includes summary plot and feature importance plot.
        max_display: Maximum number of features to display in plots.
            Recommended: 15-20 for readability.
        wavenumber_axis: Wavenumber axis for feature names (cm⁻¹).
            If None, will try to extract from ml_property.common_axis.
        max_background_samples: Maximum background samples for SHAP explainer.
            Larger = more accurate but slower. Recommended: 50-100.
        max_test_samples: Maximum test samples to explain.
            Limits number of predictions to analyze. Recommended: 10-20.
        reduce_features: Whether to use feature selection for large feature sets.
            Reduces computation time for high-dimensional data.
        max_features: Maximum features after reduction (if reduce_features=True).
            Recommended: 200-300 for Raman spectra.
        use_kmeans_sampling: Use K-means clustering for background sample selection.
            Can improve representativeness but adds overhead.
        fast_mode: Enable ultra-fast mode with minimal samples.
            Sets aggressive limits on all parameters.
        use_base_estimator: For CalibratedClassifierCV, use base estimator for SHAP.
            Often faster and provides similar explanations.
        force_kernel_explainer: Force KernelExplainer for all models.
            Ensures consistency but may be slower than native explainers.
        shap_output_mode: SHAP output mode: "auto", "full", or "sparse".
            - "auto": Automatic selection based on model type
            - "full": Complete SHAP values (slower but comprehensive)
            - "sparse": Minimal output (faster)
        kernel_nsamples_multiplier: Multiplier for KernelExplainer samples.
            Increases nsamples when using full output mode.

    Returns:
        dict: Comprehensive SHAP analysis results containing:
            - success (bool): Whether analysis completed successfully
            - shap_values (list/array): SHAP values for each class
            - expected_value (list): Base values for each class
            - explainer: SHAP explainer object
            - feature_importance (array): Overall feature importance scores
            - feature_importance_ranking (array): Ranked feature indices
            - mean_shap_values (list): Mean absolute SHAP values per class
            - top_features (dict): Top contributing features per class
            - class_explanations (dict): Class-specific detailed explanations
            - n_classes (int): Number of classes
            - n_features (int): Number of features
            - background_size (int): Number of background samples used
            - test_size (int): Number of test samples explained
            - processing_time (float): Total computation time in seconds
            - wavenumber_axis (list): Feature names (wavenumbers)
            - model_type (str): Type of model analyzed
            - base_model_type (str): Base model type (for calibrated models)
            - is_calibrated (bool): Whether model is calibrated
            - shap_model_used (str): Model used for SHAP computation
            - optimization_settings (dict): Settings used for optimization
            - summary_plot (Figure): SHAP summary plot (if show_plots=True)
            - importance_plot (Figure): Feature importance plot (if show_plots=True)

    Raises:
        ValueError: If ml_property is None or has no trained model
        ValueError: If no test data available
        ValueError: If unable to create background data

    Example:
        >>> # Basic usage
        >>> results = shap_explain(ml_property=my_ml_property)
        >>> print(f"Top feature: {results['top_features']['class_0'][0]}")

        >>> # Fast mode for quick analysis
        >>> results = shap_explain(
        ...     ml_property=my_ml_property,
        ...     fast_mode=True,
        ...     show_plots=False
        ... )

        >>> # High-quality explanations
        >>> results = shap_explain(
        ...     ml_property=my_ml_property,
        ...     nsamples=200,
        ...     shap_output_mode="full",
        ...     max_background_samples=100
        ... )

    Notes:
        - TreeExplainer is used for Random Forest models (fastest)
        - LinearExplainer is used for linear SVC models
        - KernelExplainer is used as fallback (slowest but works for all models)
        - Feature selection significantly speeds up computation for high-dimensional data
        - Fast mode reduces accuracy but provides 5-10x speedup
    """

    # Main execution flow with nested helper functions
    try:
        start_time = time.time()
        console_log("🚀 Starting enhanced SHAP explanation analysis...")

        # Performance advice based on settings
        if shap_output_mode == "full" and not fast_mode:
            console_log(
                "💡 Performance Tip: You're using 'full' mode for complete SHAP values."
            )
            console_log(
                "    This may take longer but gives the most detailed explanations."
            )
            console_log(
                "    Consider using fast_mode=True or shap_output_mode='sparse' for speed."
            )
        elif force_kernel_explainer:
            console_log(
                "💡 Performance Tip: force_kernel_explainer=True ensures consistent results"
            )
            console_log("    but may be slower than native explainers.")

        # Step 1: Validate and prepare data
        X_train, y_train, X_test, y_test, model = _validate_and_prepare_data(
            ml_property
        )

        # Step 2: Enhanced model type detection
        model_info = _detect_model_type_enhanced(
            model, use_base_estimator, force_kernel_explainer, shap_output_mode
        )
        model_info["original_model"] = model

        # Step 3: Get wavenumber axis
        if wavenumber_axis is None:
            if (
                hasattr(ml_property, "common_axis")
                and ml_property.common_axis is not None
            ):
                wavenumber_axis = ml_property.common_axis
            else:
                wavenumber_axis = None

        # Step 4: Enhanced data optimization
        data_info = _optimize_data_for_performance(
            X_train,
            y_train,
            X_test,
            y_test,
            model_info,
            max_background_samples,
            max_test_samples,
            nsamples,
            max_features,
            fast_mode,
            reduce_features,
            use_kmeans_sampling,
            kernel_nsamples_multiplier,
        )

        # Update wavenumber axis if feature selection was applied
        if data_info["feature_selector"] is not None and wavenumber_axis is not None:
            wavenumber_axis = wavenumber_axis[
                data_info["feature_selector"].get_support()
            ]

        console_log(
            f"🔬 Computing SHAP values using strategy: {model_info['shap_strategy']}"
        )
        console_log(f"    Test samples: {data_info['test_data'].shape[0]}")
        console_log(f"    Background samples: {data_info['background_data'].shape[0]}")

        # Step 5: Create enhanced SHAP explainer
        shap_result = _create_enhanced_shap_explainer(
            model_info, data_info, nsamples, kernel_nsamples_multiplier
        )

        # Step 6: Process SHAP values
        processed_shap = _process_shap_values(shap_result, data_info)

        # Step 7: Extract top features
        feature_analysis = _extract_top_features(
            processed_shap, data_info, wavenumber_axis, max_display
        )

        # Step 8: Generate plots
        plots_dict = _generate_plots(
            processed_shap,
            feature_analysis,
            data_info,
            model_info,
            wavenumber_axis,
            show_plots,
            max_display,
            use_base_estimator,
        )

        # Step 9: Create enhanced final results
        processing_time = time.time() - start_time
        results = _create_final_results(
            shap_result,
            processed_shap,
            feature_analysis,
            data_info,
            model_info,
            plots_dict,
            processing_time,
            wavenumber_axis,
            X_train,
            max_background_samples,
            max_test_samples,
            nsamples,
            use_kmeans_sampling,
            fast_mode,
            use_base_estimator,
        )

        # Add enhanced metadata
        results["optimization_settings"].update(
            {
                "shap_strategy_used": model_info["shap_strategy"],
                "shap_output_mode": shap_output_mode,
                "force_kernel_explainer": force_kernel_explainer,
                "kernel_nsamples_multiplier": kernel_nsamples_multiplier,
                "nsamples_used": shap_result.get("nsamples_used", nsamples),
            }
        )

        console_log(
            f"✅ Enhanced SHAP explanation completed in {processing_time:.2f} seconds!"
        )
        console_log(f"    Strategy used: {model_info['shap_strategy']}")
        console_log(f"    Output mode: {shap_output_mode}")

        return results

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        console_log(f"❌ Error in enhanced SHAP explanation: {e}")

        create_logs(
            "shap_explain",
            "ML",
            f"Error in SHAP explanation: {e} \n {error_details}",
            status="error",
        )
        return {
            "success": False,
            "msg": "shap_explain_error",
            "detail": f"{e} \n {error_details}",
        }


def _validate_and_prepare_data(
    ml_property: Optional[Union[RamanML, MLModel]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Validate ML_PROPERTY and prepare data for SHAP explanation.

    Args:
        ml_property: ML_PROPERTY instance with trained model

    Returns:
        tuple: (X_train, y_train, X_test, y_test, model)

    Raises:
        ValueError: If ML_PROPERTY is None, has no model, or has no test data
    """
    if ml_property is None:
        raise ValueError("ML_PROPERTY instance is required for SHAP explanation.")

    # Check if we have a trained model
    model = None
    ml_type = type(ml_property).__name__

    if ml_type == "RamanML":
        model = getattr(ml_property, "_model", None)
    elif ml_type == "MLModel":
        model = getattr(ml_property, "sklearn_model", None)

    if model is None:
        raise ValueError("No trained model found in ML_PROPERTY instance.")

    # For MLModel, we might not have training data, so we'll work with what we have
    X_test = getattr(ml_property, "X_test", None)
    y_test = getattr(ml_property, "y_test", None)
    X_train = getattr(ml_property, "X_train", None)
    y_train = getattr(ml_property, "y_train", None)

    if X_test is None:
        raise ValueError("No test data found. X_test is required for SHAP explanation.")

    # If no training data, create background data from test data
    if X_train is None or len(X_train) == 0:
        console_log("No training data found. Using subset of test data as background.")
        n_background = min(50, max(10, len(X_test) // 2))
        if n_background == 0:
            n_background = 10

        if len(X_test) < n_background:
            console_log(
                f"Warning: Only {len(X_test)} test samples available, duplicating to create {n_background} background samples"
            )
            indices = np.random.choice(len(X_test), n_background, replace=True)
        else:
            indices = np.random.choice(len(X_test), n_background, replace=False)

        X_train = X_test[indices]
        ml_property.X_train = X_train

        if y_test is not None:
            y_train = y_test[indices]
            ml_property.y_train = y_train
        else:
            y_train = np.array(["unknown"] * len(X_train))
            ml_property.y_train = y_train

    if len(X_train) == 0:
        raise ValueError("Cannot create background data - no samples available")

    # Ensure y_train is available
    if y_train is None:
        console_log("No training labels found. Using test labels as training labels.")
        if y_test is not None:
            if hasattr(ml_property, "X_train") and len(ml_property.X_train) < len(
                X_test
            ):
                if len(ml_property.X_train) <= len(y_test):
                    y_train = y_test[: len(ml_property.X_train)]
                else:
                    repeats = len(ml_property.X_train) // len(y_test) + 1
                    y_train = np.tile(y_test, repeats)[: len(ml_property.X_train)]
            else:
                y_train = y_test
            ml_property.y_train = y_train
        else:
            available_labels = ["benign", "cancer"]
            if hasattr(ml_property, "metadata") and "labels" in ml_property.metadata:
                available_labels = ml_property.metadata["labels"]
            elif hasattr(model, "classes_"):
                available_labels = model.classes_.tolist()

            y_train = np.random.choice(available_labels, len(X_train))
            ml_property.y_train = y_train
            console_log(
                f"Created dummy training labels with classes: {available_labels}"
            )

    return X_train, y_train, X_test, y_test, model


def _detect_model_type_enhanced(
    model: Any,
    use_base_estimator: bool,
    force_kernel_explainer: bool,
    shap_output_mode: str,
) -> Dict[str, Any]:
    """
    Enhanced model type detection with optimal SHAP strategy selection.

    Args:
        model: Trained sklearn model
        use_base_estimator: Whether to use base estimator for calibrated models
        force_kernel_explainer: Force KernelExplainer for all models
        shap_output_mode: Output mode ("auto", "full", "sparse")

    Returns:
        dict: Model information including:
            - model_type: Type name of the model
            - is_svc: Whether model is SVC
            - is_rf: Whether model is Random Forest
            - is_calibrated: Whether model is calibrated
            - is_linear_svc: Whether model is linear SVC
            - base_model_type: Type of base estimator (for calibrated)
            - shap_model: Model to use for SHAP computation
            - base_estimator: Base estimator (for calibrated models)
            - shap_strategy: Selected SHAP strategy name
    """
    model_type = type(model).__name__
    is_svc = isinstance(model, SVC)
    is_rf = isinstance(model, RandomForestClassifier)
    is_calibrated = isinstance(model, CalibratedClassifierCV)

    # Handle CalibratedClassifierCV
    base_estimator = None
    shap_model = model
    shap_strategy = "auto"

    if is_calibrated:
        # Enhanced: Better base estimator extraction
        base_estimator = None

        # Method 1: Try estimators_ attribute (most common)
        if hasattr(model, "estimators_") and len(model.estimators_) > 0:
            try:
                base_estimator = model.estimators_[0]
            except (IndexError, AttributeError):
                pass  # Silently handle exception

        # Method 2: Try base_estimator attribute (deprecated but sometimes present)
        if base_estimator is None and hasattr(model, "base_estimator"):
            try:
                base_estimator = model.base_estimator
            except AttributeError:
                pass  # Silently handle exception

        # Method 3: Try to access calibrated_classifiers_ (newer sklearn versions)
        if base_estimator is None and hasattr(model, "calibrated_classifiers_"):
            try:
                if len(model.calibrated_classifiers_) > 0:
                    base_estimator = model.calibrated_classifiers_[0].base_estimator
            except (IndexError, AttributeError):
                pass  # Silently handle exception

        # Method 4: Inspect the model's internal structure
        if base_estimator is None:
            try:
                # Try common attribute names
                for attr_name in [
                    "base_estimator",
                    "estimator",
                    "estimators_",
                    "calibrated_classifiers_",
                ]:
                    if hasattr(model, attr_name):
                        attr_value = getattr(model, attr_name)

                        if (
                            attr_name == "estimators_"
                            and isinstance(attr_value, list)
                            and len(attr_value) > 0
                        ):
                            base_estimator = attr_value[0]
                            break
                        elif (
                            attr_name == "calibrated_classifiers_"
                            and hasattr(attr_value, "__len__")
                            and len(attr_value) > 0
                        ):
                            if hasattr(attr_value[0], "base_estimator"):
                                base_estimator = attr_value[0].base_estimator
                                break
                        elif (
                            attr_name in ["base_estimator", "estimator"]
                            and attr_value is not None
                        ):
                            base_estimator = attr_value
                            break

            except Exception:
                pass  # Silently handle exception

        if base_estimator is not None:
            base_is_svc = isinstance(base_estimator, SVC)
            base_is_rf = isinstance(base_estimator, RandomForestClassifier)
            base_is_linear_svc = (
                base_is_svc
                and hasattr(base_estimator, "kernel")
                and base_estimator.kernel == "linear"
            )
            base_model_type = type(base_estimator).__name__

            # Enhanced: Determine optimal SHAP strategy based on output mode
            if shap_output_mode == "full" or force_kernel_explainer:
                shap_strategy = "kernel_for_full_values"
                shap_model = (
                    model  # Use calibrated model for full probability explanations
                )
            elif shap_output_mode == "sparse" or (
                use_base_estimator and not force_kernel_explainer
            ):
                shap_strategy = "base_estimator_for_speed"
                shap_model = base_estimator
            else:  # auto mode
                if base_is_rf:
                    shap_strategy = "base_estimator_tree"
                    shap_model = base_estimator
                elif base_is_linear_svc:
                    shap_strategy = "base_estimator_linear"
                    shap_model = base_estimator
                else:
                    shap_strategy = "kernel_balanced"
                    shap_model = model

            # Update flags to reflect the chosen strategy
            if shap_strategy.startswith("base_estimator"):
                is_svc = base_is_svc
                is_rf = base_is_rf
                is_linear_svc = base_is_linear_svc
            else:
                is_svc = False
                is_rf = False
                is_linear_svc = False

        else:
            shap_strategy = "kernel_fallback"
            is_svc = False
            is_rf = False
            is_linear_svc = False
            base_model_type = "Unknown"
    else:
        # Non-calibrated models
        is_linear_svc = is_svc and hasattr(model, "kernel") and model.kernel == "linear"
        base_model_type = model_type

        if force_kernel_explainer:
            shap_strategy = "kernel_forced"
        elif shap_output_mode == "full":
            shap_strategy = "kernel_for_full_values"
        elif is_rf:
            shap_strategy = "tree_native"
        elif is_linear_svc:
            shap_strategy = "linear_native"
        else:
            shap_strategy = "kernel_auto"

    return {
        "model_type": model_type,
        "is_svc": is_svc,
        "is_rf": is_rf,
        "is_calibrated": is_calibrated,
        "is_linear_svc": is_linear_svc,
        "is_base_rf": is_calibrated
        and base_estimator is not None
        and isinstance(base_estimator, RandomForestClassifier),
        "base_model_type": base_model_type,
        "shap_model": shap_model,
        "base_estimator": base_estimator,
        "shap_strategy": shap_strategy,
    }


def _optimize_data_for_performance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_info: Dict[str, Any],
    max_background_samples: int,
    max_test_samples: int,
    nsamples: int,
    max_features: int,
    fast_mode: bool,
    reduce_features: bool,
    use_kmeans_sampling: bool,
    kernel_nsamples_multiplier: float,
) -> Dict[str, Any]:
    """
    Enhanced data optimization with strategy-aware settings.

    Args:
        X_train: Training feature data
        y_train: Training labels
        X_test: Test feature data
        y_test: Test labels
        model_info: Model information dict from _detect_model_type_enhanced
        max_background_samples: Maximum background samples
        max_test_samples: Maximum test samples
        nsamples: Number of samples for KernelExplainer
        max_features: Maximum features after reduction
        fast_mode: Whether fast mode is enabled
        reduce_features: Whether to use feature reduction
        use_kmeans_sampling: Whether to use K-means sampling
        kernel_nsamples_multiplier: Multiplier for kernel samples

    Returns:
        dict: Optimized data containing:
            - background_data: Optimized background samples
            - test_data: Optimized test samples
            - labels: Unique class labels
            - feature_selector: Feature selector object (or None)
    """
    background_data = X_train.copy()
    test_data = X_test.copy()
    labels = list(np.unique(y_train))

    # Strategy-specific optimization
    strategy = model_info["shap_strategy"]

    # Apply fast mode settings
    if fast_mode:
        max_background_samples = min(max_background_samples, 20)
        max_test_samples = min(max_test_samples, 10)
        nsamples = min(nsamples, 20)
        max_features = min(max_features, 200)
    elif strategy in ["kernel_for_full_values", "kernel_forced", "kernel_balanced"]:
        # For full SHAP values, use more samples but optimize background
        if not fast_mode:
            # Slightly more for quality
            max_background_samples = min(max_background_samples, 40)
            # Increase for better quality
            nsamples = int(nsamples * kernel_nsamples_multiplier)
    elif strategy.startswith("base_estimator") or strategy in [
        "tree_native",
        "linear_native",
    ]:
        # For base estimators, we can afford more samples
        max_background_samples = min(max_background_samples, 100)
        max_test_samples = min(max_test_samples, 50)

    # Feature selection based on strategy
    feature_selector = None
    if (
        strategy in ["kernel_for_full_values", "kernel_forced", "kernel_balanced"]
        and reduce_features
        and background_data.shape[1] > max_features
    ):
        # For kernel explainer with full values, be more conservative with feature reduction
        from sklearn.feature_selection import SelectKBest, f_classif

        selector = SelectKBest(f_classif, k=max_features)
        background_data = selector.fit_transform(background_data, y_train)
        test_data = selector.transform(test_data)
        feature_selector = selector

    elif (
        (model_info["is_svc"] or strategy.startswith("base_estimator"))
        and reduce_features
        and background_data.shape[1] > max_features
    ):
        # Standard feature reduction for other strategies
        from sklearn.feature_selection import SelectKBest, f_classif

        selector = SelectKBest(f_classif, k=max_features)
        background_data = selector.fit_transform(background_data, y_train)
        test_data = selector.transform(test_data)
        feature_selector = selector

    # Background sampling optimization
    if background_data.shape[0] > max_background_samples:
        if use_kmeans_sampling and strategy in [
            "kernel_for_full_values",
            "kernel_balanced",
        ]:
            from sklearn.cluster import KMeans

            kmeans = KMeans(
                n_clusters=max_background_samples, random_state=42, n_init=10
            )
            kmeans.fit(background_data)
            background_data = kmeans.cluster_centers_
        else:
            indices = np.random.choice(
                background_data.shape[0], max_background_samples, replace=False
            )
            background_data = background_data[indices]

    # Test data limitation
    if test_data.shape[0] > max_test_samples:
        test_data = test_data[:max_test_samples]

    return {
        "background_data": background_data,
        "test_data": test_data,
        "labels": labels,
        "feature_selector": feature_selector,
    }


def _create_enhanced_shap_explainer(
    model_info: Dict[str, Any],
    data_info: Dict[str, Any],
    nsamples: int,
    kernel_nsamples_multiplier: float,
) -> Dict[str, Any]:
    """
    Enhanced SHAP explainer creation with strategy-based approach.

    Args:
        model_info: Model information from _detect_model_type_enhanced
        data_info: Data information from _optimize_data_for_performance
        nsamples: Base number of samples for KernelExplainer
        kernel_nsamples_multiplier: Multiplier for kernel samples

    Returns:
        dict: SHAP explainer results containing:
            - explainer: SHAP explainer object
            - shap_values: Computed SHAP values
            - expected_value: Base values
            - strategy_used: Strategy name used
            - nsamples_used: Number of samples used
    """
    model = model_info["shap_model"]
    background_data = data_info["background_data"]
    test_data = data_info["test_data"]
    feature_selector = data_info["feature_selector"]
    strategy = model_info["shap_strategy"]

    # Adjusted nsamples for current strategy
    current_nsamples = nsamples
    if strategy in ["kernel_for_full_values", "kernel_forced"]:
        current_nsamples = int(nsamples * kernel_nsamples_multiplier)
    elif strategy.startswith("base_estimator"):
        # Faster for base estimators
        current_nsamples = max(nsamples // 2, 10)

    # Strategy-based explainer creation
    if strategy == "tree_native":
        if feature_selector is not None:

            class TreeFeatureWrapper:
                def __init__(self, model, selector):
                    self.model = model
                    self.selector = selector

                def predict_proba(self, X):
                    return self.model.predict_proba(self.selector.transform(X))

            wrapped_model = TreeFeatureWrapper(model, feature_selector)
            explainer = shap.Explainer(wrapped_model.predict_proba, background_data)
            shap_values = explainer(test_data)
            expected_value = explainer.expected_value
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(test_data)
            expected_value = explainer.expected_value

    elif strategy == "linear_native":
        try:
            if feature_selector is not None:
                wrapped_model = _create_enhanced_feature_wrapper(
                    model, feature_selector, "svc_enhanced"
                )
                explainer = shap.LinearExplainer(wrapped_model, background_data)
                shap_values = explainer.shap_values(test_data)
            else:
                if hasattr(model, "predict_proba") and model.probability:
                    explainer = shap.LinearExplainer(model, background_data)
                    shap_values = explainer.shap_values(test_data)
                else:
                    raise Exception(
                        "Linear SVC without probability - falling back to Kernel"
                    )

            expected_value = explainer.expected_value

        except Exception:
            return _create_kernel_fallback(
                model, background_data, test_data, feature_selector, current_nsamples
            )

    # Handle all kernel strategies
    elif strategy in [
        "kernel_forced",
        "kernel_balanced",
        "kernel_auto",
        "kernel_for_full_values",
        "base_estimator_for_speed",
        "base_estimator_tree",
        "base_estimator_linear",
        "kernel_fallback",
    ]:
        return _create_kernel_fallback(
            model, background_data, test_data, feature_selector, current_nsamples
        )

    else:
        # Default fallback
        return _create_kernel_fallback(
            model, background_data, test_data, feature_selector, current_nsamples
        )

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "expected_value": expected_value,
        "strategy_used": strategy,
        "nsamples_used": current_nsamples,
    }


def _create_kernel_fallback(
    model: Any,
    background_data: np.ndarray,
    test_data: np.ndarray,
    feature_selector: Optional[Any],
    current_nsamples: int,
) -> Dict[str, Any]:
    """
    Create KernelExplainer as fallback for all models.

    Args:
        model: Trained sklearn model
        background_data: Background samples for explainer
        test_data: Test samples to explain
        feature_selector: Feature selector object (or None)
        current_nsamples: Number of samples for KernelExplainer

    Returns:
        dict: SHAP explainer results
    """
    # Create prediction function
    if hasattr(model, "predict_proba"):
        predict_fn = model.predict_proba
    else:

        def predict_fn(X):
            if hasattr(model, "decision_function"):
                decision = model.decision_function(X)
                from scipy.special import expit

                if decision.ndim == 1:
                    prob_pos = expit(decision)
                    return np.column_stack([1 - prob_pos, prob_pos])
                else:
                    exp_scores = np.exp(
                        decision - np.max(decision, axis=1, keepdims=True)
                    )
                    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            else:
                # Last resort
                predictions = model.predict(X)
                n_classes = len(np.unique(predictions))
                return np.eye(n_classes)[predictions]

    if feature_selector is not None:

        class KernelFeatureWrapper:
            def __init__(self, model, selector, predict_fn):
                self.model = model
                self.selector = selector
                self.predict_fn = predict_fn

            def __call__(self, X):
                X_transformed = self.selector.transform(X)
                return self.predict_fn(X_transformed)

        wrapped_predict = KernelFeatureWrapper(model, feature_selector, predict_fn)
        explainer = shap.KernelExplainer(wrapped_predict, background_data)
        shap_values = explainer.shap_values(test_data, nsamples=current_nsamples)
    else:
        explainer = shap.KernelExplainer(predict_fn, background_data, link="identity")
        shap_values = explainer.shap_values(test_data, nsamples=current_nsamples)

    expected_value = explainer.expected_value

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "expected_value": expected_value,
        "strategy_used": "kernel_fallback",
        "nsamples_used": current_nsamples,
    }


def _create_enhanced_feature_wrapper(
    model: Any, feature_selector: Any, wrapper_type: str = "generic"
) -> Any:
    """
    Create a model wrapper that applies feature selection before prediction.

    Args:
        model: Trained sklearn model
        feature_selector: Feature selector object (e.g., SelectKBest)
        wrapper_type: Type of wrapper ('generic', 'svc_enhanced', 'tree')

    Returns:
        Wrapped model object with feature selection
    """

    class EnhancedFeatureWrapper:
        def __init__(self, model, selector):
            self.model = model
            self.selector = selector

        def predict(self, X):
            return self.model.predict(self.selector.transform(X))

        def predict_proba(self, X):
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(self.selector.transform(X))
            else:
                # Fallback for models without predict_proba
                decision = self.model.decision_function(self.selector.transform(X))
                from scipy.special import expit

                if decision.ndim == 1:
                    prob_pos = expit(decision)
                    return np.column_stack([1 - prob_pos, prob_pos])
                else:
                    exp_scores = np.exp(
                        decision - np.max(decision, axis=1, keepdims=True)
                    )
                    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        def decision_function(self, X):
            if hasattr(self.model, "decision_function"):
                return self.model.decision_function(self.selector.transform(X))
            else:
                raise AttributeError("Model does not have decision_function method")

    return EnhancedFeatureWrapper(model, feature_selector)


def _process_shap_values(
    shap_result: Dict[str, Any], data_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process and normalize SHAP values from different explainer types.

    Args:
        shap_result: SHAP explainer results from _create_enhanced_shap_explainer
        data_info: Data information from _optimize_data_for_performance

    Returns:
        dict: Processed SHAP values containing:
            - shap_values: Normalized SHAP values (list of arrays)
            - expected_value: Base values (list)
            - mean_shap_values: Mean absolute SHAP values per class
            - n_classes: Number of classes
    """
    shap_values = shap_result["shap_values"]
    expected_value = shap_result["expected_value"]
    labels = data_info["labels"]

    # Determine n_classes from SHAP values
    if isinstance(shap_values, list):
        n_classes = len(shap_values)
    elif hasattr(shap_values, "values"):
        shap_values_array = shap_values.values
        if shap_values_array.ndim == 3:
            n_classes = shap_values_array.shape[-1]
        else:
            n_classes = len(labels)
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            n_classes = shap_values.shape[-1]
        else:
            n_classes = len(labels)
    else:
        n_classes = len(labels)

    # Normalize SHAP values to consistent format
    if hasattr(shap_values, "values"):
        shap_values_array = shap_values.values
        if shap_values_array.ndim == 3:
            n_classes = shap_values_array.shape[-1]
            shap_values = [shap_values_array[:, :, i] for i in range(n_classes)]
        elif shap_values_array.ndim == 2:
            n_classes = 1
            shap_values = [shap_values_array]
        else:
            n_classes = 1
            shap_values = [shap_values_array.reshape(1, -1)]
    elif isinstance(shap_values, list):
        n_classes = len(shap_values)
        for i, sv in enumerate(shap_values):
            if isinstance(sv, np.ndarray) and sv.ndim == 1:
                shap_values[i] = sv.reshape(1, -1)
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            n_classes = shap_values.shape[-1]
            shap_values = [shap_values[:, :, i] for i in range(n_classes)]
        elif shap_values.ndim == 2:
            n_classes = 1
            shap_values = [shap_values]
        else:
            n_classes = 1
            shap_values = [shap_values.reshape(1, -1)]

    # Calculate mean absolute SHAP values for feature importance
    mean_shap_values = []
    for i, sv in enumerate(shap_values):
        if isinstance(sv, np.ndarray):
            mean_shap_values.append(np.mean(np.abs(sv), axis=0))
        else:
            mean_shap_values.append(np.abs(np.array(sv)).flatten())

    # Ensure expected_value is properly formatted
    if not isinstance(expected_value, (list, np.ndarray)):
        expected_value = [expected_value] * n_classes
    elif isinstance(expected_value, np.ndarray):
        if expected_value.ndim == 0:
            expected_value = [float(expected_value)] * n_classes
        elif len(expected_value) != n_classes:
            expected_value = [float(expected_value[0])] * n_classes
        else:
            expected_value = [float(ev) for ev in expected_value]

    return {
        "shap_values": shap_values,
        "expected_value": expected_value,
        "mean_shap_values": mean_shap_values,
        "n_classes": n_classes,
    }


def _extract_top_features(
    processed_shap: Dict[str, Any],
    data_info: Dict[str, Any],
    wavenumber_axis: Optional[np.ndarray],
    max_display: int,
) -> Dict[str, Any]:
    """
    Extract top contributing features for each class.

    Args:
        processed_shap: Processed SHAP values from _process_shap_values
        data_info: Data information from _optimize_data_for_performance
        wavenumber_axis: Wavenumber axis for feature names (or None)
        max_display: Maximum number of features to extract per class

    Returns:
        dict: Feature analysis containing:
            - top_features: Dict mapping class labels to top feature lists
            - overall_importance: Overall feature importance scores
            - feature_importance_indices: Ranked feature indices
    """
    shap_values = processed_shap["shap_values"]
    mean_shap_values = processed_shap["mean_shap_values"]
    n_classes = processed_shap["n_classes"]
    labels = data_info["labels"]

    # Calculate overall feature importance
    overall_importance = np.mean([msv for msv in mean_shap_values], axis=0)
    feature_importance_indices = np.argsort(overall_importance)[::-1]

    # Get top features for each class
    top_features = {}
    for class_idx in range(n_classes):
        class_shap = mean_shap_values[class_idx]

        if hasattr(class_shap, "ndim") and class_shap.ndim > 1:
            class_shap = class_shap.flatten()

        top_indices = np.argsort(class_shap)[-max_display:][::-1]

        class_features = []
        for rank, idx in enumerate(top_indices):
            try:
                idx_val = int(idx)
                importance_val = float(class_shap[idx_val])

                if wavenumber_axis is not None and idx_val < len(wavenumber_axis):
                    wavenumber_val = float(wavenumber_axis[idx_val])
                    feature_name = f"{wavenumber_val:.1f} cm⁻¹"
                else:
                    wavenumber_val = float(idx_val)
                    feature_name = f"Feature_{idx_val}"

                class_features.append(
                    {
                        "rank": int(rank + 1),
                        "feature_index": idx_val,
                        "wavenumber": wavenumber_val,
                        "feature_name": feature_name,
                        "importance": importance_val,
                    }
                )
            except (ValueError, TypeError, IndexError):
                continue

        label_name = (
            labels[class_idx] if class_idx < len(labels) else f"class_{class_idx}"
        )
        top_features[label_name] = class_features

    return {
        "top_features": top_features,
        "overall_importance": overall_importance,
        "feature_importance_indices": feature_importance_indices,
    }


def _generate_plots(
    processed_shap: Dict[str, Any],
    feature_analysis: Dict[str, Any],
    data_info: Dict[str, Any],
    model_info: Dict[str, Any],
    wavenumber_axis: Optional[np.ndarray],
    show_plots: bool,
    max_display: int,
    use_base_estimator: bool,
) -> Dict[str, Any]:
    """
    Generate SHAP plots if requested.

    Args:
        processed_shap: Processed SHAP values
        feature_analysis: Feature analysis results
        data_info: Data information
        model_info: Model information
        wavenumber_axis: Wavenumber axis for labels
        show_plots: Whether to show plots
        max_display: Maximum features to display
        use_base_estimator: Whether base estimator was used

    Returns:
        dict: Dictionary containing plot Figure objects (or empty dict)
    """
    if not show_plots:
        return {}

    plots_dict = {}
    test_data = data_info["test_data"]
    shap_values = processed_shap["shap_values"]
    n_classes = processed_shap["n_classes"]
    overall_importance = feature_analysis["overall_importance"]
    labels = data_info["labels"]

    try:
        import matplotlib.pyplot as plt

        # Create feature names
        feature_names = []
        if wavenumber_axis is not None:
            feature_names = [
                f"{float(wavenumber_axis[i]):.1f} cm⁻¹"
                for i in range(test_data.shape[1])
            ]
        else:
            feature_names = [f"Feature_{i}" for i in range(test_data.shape[1])]

        # Summary plot
        try:
            plt.figure(figsize=(12, 8))

            if n_classes == 1:
                shap.summary_plot(
                    shap_values[0],
                    test_data,
                    feature_names=feature_names,
                    max_display=max_display,
                    show=False,
                )
            else:
                if n_classes == 2:
                    if (
                        model_info["is_calibrated"]
                        or model_info["is_rf"]
                        or model_info.get("is_base_rf", False)
                    ):
                        shap.summary_plot(
                            shap_values[1],
                            test_data,
                            feature_names=feature_names,
                            max_display=max_display,
                            show=False,
                        )
                        plt.title(
                            f'SHAP Summary Plot - {model_info["model_type"]}\n(Showing {labels[1] if len(labels) > 1 else "Positive"} Class)',
                            fontsize=14,
                            fontweight="bold",
                        )
                    else:
                        shap.summary_plot(
                            shap_values,
                            test_data,
                            feature_names=feature_names,
                            max_display=max_display,
                            show=False,
                        )
                        plt.title(
                            f'SHAP Summary Plot - {model_info["model_type"]}',
                            fontsize=14,
                            fontweight="bold",
                        )
                else:
                    shap.summary_plot(
                        shap_values,
                        test_data,
                        feature_names=feature_names,
                        max_display=max_display,
                        show=False,
                    )
                    plt.title(
                        f'SHAP Summary Plot - {model_info["model_type"]}',
                        fontsize=14,
                        fontweight="bold",
                    )

            plt.tight_layout()
            plots_dict["summary_plot"] = plt.gcf()
            if show_plots:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            # Summary plot is best-effort (SHAP can fail for some model/data combos).
            console_log(f"[SHAP] Failed to create summary plot: {e}")

        # Feature importance plot
        try:
            top_indices = np.argsort(overall_importance)[-max_display:]
            top_importance = overall_importance[top_indices]

            if wavenumber_axis is not None:
                try:
                    feature_names_imp = [
                        f"{float(wavenumber_axis[i]):.1f} cm⁻¹" for i in top_indices
                    ]
                except (ValueError, TypeError, IndexError):
                    feature_names_imp = [f"Feature_{i}" for i in top_indices]
            else:
                feature_names_imp = [f"Feature_{i}" for i in top_indices]

            plt.figure(figsize=(10, 8))
            bars = plt.barh(
                range(len(top_importance)), top_importance, color="steelblue", alpha=0.7
            )
            plt.yticks(range(len(feature_names_imp)), feature_names_imp)
            plt.xlabel("Mean |SHAP value|", fontsize=12)
            plt.ylabel("Raman Shift (cm⁻¹)", fontsize=12)

            shap_model_name = f"{type(model_info['shap_model']).__name__}" + (
                " (Base)" if model_info["is_calibrated"] and use_base_estimator else ""
            )
            plt.title(
                f"Feature Importance (SHAP) - {shap_model_name}",
                fontsize=14,
                fontweight="bold",
            )

            for i, (bar, val) in enumerate(zip(bars, top_importance)):
                plt.text(
                    val + 0.001,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}",
                    ha="left",
                    va="center",
                    fontsize=9,
                )

            plt.grid(axis="x", alpha=0.3)
            plt.tight_layout()

            plots_dict["importance_plot"] = plt.gcf()
            if show_plots:
                plt.show()
            else:
                plt.close()

        except Exception:
            pass  # Silently handle exception

    except Exception as e:
        create_logs(
            "shap_plots", "ML", f"Error generating plots: {e}", status="warning"
        )
        plots_dict = {"plot_error": str(e)}

    return plots_dict


def _create_final_results(
    shap_result: Dict[str, Any],
    processed_shap: Dict[str, Any],
    feature_analysis: Dict[str, Any],
    data_info: Dict[str, Any],
    model_info: Dict[str, Any],
    plots_dict: Dict[str, Any],
    processing_time: float,
    wavenumber_axis: Optional[np.ndarray],
    X_train: np.ndarray,
    max_background_samples: int,
    max_test_samples: int,
    nsamples: int,
    use_kmeans_sampling: bool,
    fast_mode: bool,
    use_base_estimator: bool,
) -> Dict[str, Any]:
    """
    Create the final results dictionary with all SHAP analysis information.

    Args:
        shap_result: SHAP explainer results
        processed_shap: Processed SHAP values
        feature_analysis: Feature analysis results
        data_info: Data information
        model_info: Model information
        plots_dict: Generated plots
        processing_time: Total processing time
        wavenumber_axis: Wavenumber axis
        X_train: Original training data
        max_background_samples: Max background samples setting
        max_test_samples: Max test samples setting
        nsamples: Number of samples setting
        use_kmeans_sampling: Whether K-means was used
        fast_mode: Whether fast mode was used
        use_base_estimator: Whether base estimator was used

    Returns:
        dict: Comprehensive SHAP analysis results
    """
    # Prepare class-specific explanations
    class_explanations = {}
    shap_values = processed_shap["shap_values"]
    expected_value = processed_shap["expected_value"]
    mean_shap_values = processed_shap["mean_shap_values"]
    n_classes = processed_shap["n_classes"]
    top_features = feature_analysis["top_features"]
    labels = data_info["labels"]

    for i, label in enumerate(labels[:n_classes]):
        try:
            if i < len(shap_values) and i < len(mean_shap_values):
                class_explanations[label] = {
                    "shap_values": shap_values[i],
                    "expected_value": (
                        expected_value[i] if i < len(expected_value) else 0.0
                    ),
                    "mean_abs_shap": mean_shap_values[i],
                    "top_features": top_features.get(label, []),
                }
        except Exception:
            pass  # Silently handle exception

    return {
        "success": True,
        "msg": "shap_explain_success",
        "shap_values": shap_values,
        "expected_value": expected_value,
        "explainer": shap_result["explainer"],
        "feature_importance": feature_analysis["overall_importance"],
        "feature_importance_ranking": feature_analysis["feature_importance_indices"],
        "mean_shap_values": mean_shap_values,
        "top_features": top_features,
        "class_explanations": class_explanations,
        "n_classes": n_classes,
        "n_features": int(data_info["test_data"].shape[1]),
        "background_size": int(data_info["background_data"].shape[0]),
        "test_size": int(data_info["test_data"].shape[0]),
        "processing_time": float(processing_time),
        "wavenumber_axis": (
            wavenumber_axis.tolist() if wavenumber_axis is not None else None
        ),
        "model_type": model_info["model_type"],
        "base_model_type": (
            model_info["base_model_type"]
            if model_info["is_calibrated"]
            else model_info["model_type"]
        ),
        "is_calibrated": model_info["is_calibrated"],
        "shap_model_used": type(model_info["shap_model"]).__name__,
        "used_base_estimator": model_info["is_calibrated"] and use_base_estimator,
        "class_labels": labels,
        "optimization_settings": {
            "max_background_samples": max_background_samples,
            "max_test_samples": max_test_samples,
            "nsamples": nsamples,
            "feature_reduction_used": data_info["feature_selector"] is not None,
            "features_before_reduction": (
                X_train.shape[1]
                if data_info["feature_selector"] is not None
                else data_info["test_data"].shape[1]
            ),
            "features_after_reduction": data_info["test_data"].shape[1],
            "kmeans_sampling": use_kmeans_sampling,
            "fast_mode": fast_mode,
            "is_linear_svc": model_info["is_linear_svc"],
            "explainer_type": type(shap_result["explainer"]).__name__,
            "use_base_estimator": use_base_estimator,
        },
        **plots_dict,
    }
