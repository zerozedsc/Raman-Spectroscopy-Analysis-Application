"""
LIME Analysis Module for Raman Spectroscopy Explainability (Phase 5 Refactoring).

This module provides LIME (Local Interpretable Model-agnostic Explanations) analysis
for explaining individual predictions in Raman spectroscopy classification tasks.

LIME creates simple surrogate models that explain individual predictions by perturbing
the input and observing how predictions change, providing local interpretability for
complex ML models.

Main Functions:
    - lime_explain: Generate LIME explanations for model predictions
    - _visualize_lime_explanation: Visualize individual LIME explanations
    - _visualize_lime_comparison: Compare feature importance across classes

Usage:
    >>> from functions.visualization import RamanVisualizer
    >>> viz = RamanVisualizer()
    >>> results = viz.lime_explain(ML_PROPERTY=ml_model, test_spectra=test_data)

    >>> # Or use standalone function
    >>> from functions.visualization.lime_analysis import lime_explain
    >>> results = lime_explain(visualizer_instance, ML_PROPERTY=ml_model)

Dependencies:
    - lime: LIME library for model explanations
    - sklearn: Machine learning models (SVC, RandomForest, etc.)
    - numpy, matplotlib: Data manipulation and visualization
    - ramanspy: Spectral data containers

Author: MUHAMMAD HELMI BIN ROZAIN
Phase: 5 (Extracted from core.py)
"""

import re
import time
from typing import Dict, Any, List, Tuple, Optional, Union

import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
import ramanspy as rp
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import expit

from functions.ML import RamanML, MLModel
from functions.configs import console_log


def lime_explain(
    visualizer_instance,
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

    LIME (Local Interpretable Model-agnostic Explanations) creates a simple surrogate model
    that explains individual predictions by perturbing the input and seeing how predictions change.

    Args:
        visualizer_instance: RamanVisualizer instance (for self reference)
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
    try:
        start_time = time.time()
        ML_PROPERTY = (
            ML_PROPERTY if ML_PROPERTY is not None else visualizer_instance.ML_PROPERTY
        )

        if ML_PROPERTY is None:
            raise ValueError(
                "No ML_PROPERTY instance provided. Please provide a ML_PROPERTY instance with trained data."
            )

        # Handle test data input - either from parameters or stored data
        if test_spectra is not None and true_labels is not None:
            # Convert spectral containers to numpy arrays
            all_spectra = []
            for container_idx, container in enumerate(test_spectra):
                for spectrum_idx, spectrum in enumerate(container.spectral_data):
                    # Interpolate to common axis if needed
                    if len(container.spectral_axis) != len(ML_PROPERTY.common_axis):
                        interp_spectrum = np.interp(
                            ML_PROPERTY.common_axis, container.spectral_axis, spectrum
                        )
                        all_spectra.append(interp_spectrum)
                    else:
                        all_spectra.append(spectrum)

            X_test = np.array(all_spectra)
            y_test = np.array(true_labels[: len(all_spectra)])

            console_log(f"Using provided test data: {X_test.shape[0]} test samples")
        else:
            # Use stored test data
            if (
                ML_PROPERTY.X_train is None
                or ML_PROPERTY.y_train is None
                or ML_PROPERTY.X_test is None
                or ML_PROPERTY.y_test is None
            ):
                raise ValueError(
                    "No training/test data found. Please run train_svc() or train_rf() first or provide test_spectra and true_labels."
                )

            X_test = ML_PROPERTY.X_test
            y_test = ML_PROPERTY.y_test
            console_log(f"Using stored test data: {X_test.shape[0]} test samples")

        # Get training data (always from stored data)
        X_train = ML_PROPERTY.X_train
        y_train = ML_PROPERTY.y_train

        # Get model and data
        if hasattr(ML_PROPERTY, "_model") and ML_PROPERTY._model is not None:
            model = ML_PROPERTY._model
        else:
            raise ValueError(
                "No trained model found. Train a model first or provide one."
            )

        # === UNIFIED MODEL TYPE DETECTION (same as SHAP) ===
        is_svc = isinstance(model, SVC)
        is_rf = isinstance(model, RandomForestClassifier)
        is_linear_svc = is_svc and hasattr(model, "kernel") and model.kernel == "linear"

        console_log(f"Creating unified prediction function for {type(model).__name__}")
        console_log(
            f"Model type detection: SVC={is_svc}, RF={is_rf}, Linear SVC={is_linear_svc}"
        )

        # Get class names and wavenumber axis
        class_names = list(np.unique(y_train))
        if len(class_names) != 2:
            console_log(
                f"Warning: LIME visualization works best for binary classification. "
                f"Found {len(class_names)} classes: {class_names}"
            )

        wavenumber_axis = (
            ML_PROPERTY.common_axis if hasattr(ML_PROPERTY, "common_axis") else None
        )

        console_log(
            f"Original data shapes: X_train: {X_train.shape}, X_test: {X_test.shape}"
        )

        # Handle sample indices BEFORE feature selection
        if sample_indices is not None:
            # Use specified indices - check against test set size
            indices = sample_indices
            if any(idx >= len(X_test) for idx in indices):
                console_log(
                    f"Warning: Some indices exceed the test set size ({len(X_test)}). Filtering valid indices."
                )
                indices = [idx for idx in indices if idx < len(X_test)]

            if not indices:
                raise ValueError("No valid sample indices provided for the test set.")

            console_log(f"Selected sample indices from test set: {indices}")
        else:
            # Randomly select samples from test set
            np.random.seed(seed)
            n_samples = min(max_test_samples, len(X_test))
            indices = np.random.choice(len(X_test), n_samples, replace=False)
            console_log(f"Randomly selected {n_samples} samples from test set")

        # === FEATURE SELECTION WITH UNIFIED PREDICTION FUNCTION ===
        feature_selector = None
        if use_feature_selection and X_train.shape[1] > max_features:
            console_log(
                f"Applying feature selection to reduce from {X_train.shape[1]} to {max_features} features"
            )
            selector = SelectKBest(f_classif, k=max_features)
            X_train_reduced = selector.fit_transform(X_train, y_train)
            X_test_reduced = selector.transform(X_test)
            feature_selector = selector

            # Update wavenumber axis if provided
            if wavenumber_axis is not None:
                wavenumber_axis_reduced = wavenumber_axis[selector.get_support()]
            else:
                wavenumber_axis_reduced = np.arange(X_train_reduced.shape[1])

            # === UNIFIED PREDICTION FUNCTION WITH FEATURE SELECTION ===
            def unified_predict_fn(x_reduced):
                """
                Unified prediction function that handles CalibratedClassifierCV and feature selection.
                """
                # Ensure input is 2D
                if x_reduced.ndim == 1:
                    x_reduced = x_reduced.reshape(1, -1)

                # Handle feature selection if applied
                if feature_selector is not None:
                    if (
                        hasattr(model, "n_features_in_")
                        and model.n_features_in_ != x_reduced.shape[1]
                    ):
                        # Model was trained on original features, need to expand
                        original_shape = (x_reduced.shape[0], X_train.shape[1])
                        x_expanded = np.zeros(original_shape)
                        selected_indices_feat = selector.get_support(indices=True)
                        x_expanded[:, selected_indices_feat] = x_reduced
                        x_input = x_expanded
                    else:
                        x_input = x_reduced
                else:
                    x_input = x_reduced

                # FIXED: Handle CalibratedClassifierCV first
                if isinstance(model, CalibratedClassifierCV):
                    # CalibratedClassifierCV always has predict_proba and it's calibrated
                    return model.predict_proba(x_input)
                elif is_rf:
                    # RandomForest always has predict_proba
                    return model.predict_proba(x_input)
                elif is_svc:
                    # For SVC, check if probability is enabled
                    if hasattr(model, "predict_proba") and model.probability:
                        return model.predict_proba(x_input)
                    else:
                        # For SVC without probability, use decision_function (same as SHAP)
                        decision = model.decision_function(x_input)
                        if decision.ndim == 1:
                            # Binary classification
                            prob_pos = expit(decision)
                            return np.column_stack([1 - prob_pos, prob_pos])
                        else:
                            # Multiclass - use softmax
                            exp_scores = np.exp(
                                decision - np.max(decision, axis=1, keepdims=True)
                            )
                            return exp_scores / np.sum(
                                exp_scores, axis=1, keepdims=True
                            )
                else:
                    # For other model types, try predict_proba first
                    if hasattr(model, "predict_proba"):
                        return model.predict_proba(x_input)
                    else:
                        # Fallback for models without predict_proba
                        predictions = model.predict(x_input)
                        # Convert to dummy probabilities (not ideal but works)
                        n_classes = len(np.unique(predictions))
                        dummy_probs = np.eye(n_classes)[predictions]
                        return dummy_probs

            feature_names = [
                f"{wavenumber_axis_reduced[i]:.1f} cm⁻¹"
                for i in range(len(wavenumber_axis_reduced))
            ]

        else:
            # No feature selection
            X_train_reduced = X_train
            X_test_reduced = X_test

            # === UNIFIED PREDICTION FUNCTION WITHOUT FEATURE SELECTION ===
            def unified_predict_fn(x):
                """Unified prediction function without feature selection."""
                if x.ndim == 1:
                    x = x.reshape(1, -1)

                if hasattr(model, "predict_proba") and model.probability:
                    return model.predict_proba(x)
                else:
                    # Same logic as SHAP for SVC without probability
                    decision = model.decision_function(x)
                    if decision.ndim == 1:
                        prob_pos = expit(decision)
                        return np.column_stack([1 - prob_pos, prob_pos])
                    else:
                        exp_scores = np.exp(
                            decision - np.max(decision, axis=1, keepdims=True)
                        )
                        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            if wavenumber_axis is not None:
                feature_names = [
                    f"{wavenumber_axis[i]:.1f} cm⁻¹"
                    for i in range(len(wavenumber_axis))
                ]
            else:
                feature_names = [
                    f"Feature_{i}" for i in range(X_train_reduced.shape[1])
                ]

        console_log(
            f"After feature selection: X_train_reduced: {X_train_reduced.shape}, X_test_reduced: {X_test_reduced.shape}"
        )

        # Now use the unified_predict_fn for both prediction verification AND LIME
        console_log(f"Setting up LIME explainer with unified prediction function...")

        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train_reduced,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification",
            random_state=seed,
            discretize_continuous=False,  # Keep continuous for spectral data
        )

        # Select the samples from the reduced test set using the same indices
        selected_samples_reduced = X_test_reduced[indices]
        selected_labels = y_test[indices]

        console_log(
            f"Selected samples shape after feature selection: {selected_samples_reduced.shape}"
        )

        # Generate explanations
        console_log(f"Generating LIME explanations for {len(indices)} samples...")
        explanations = []

        for i, (original_idx, x_sample, y_true) in enumerate(
            zip(indices, selected_samples_reduced, selected_labels)
        ):
            console_log(
                f"\nExplaining sample {i+1}/{len(indices)} (original index {original_idx})..."
            )

            # === USE UNIFIED PREDICTION FUNCTION FOR CONSISTENCY ===
            y_pred_proba = unified_predict_fn(x_sample.reshape(1, -1))[0]
            y_pred_idx = np.argmax(y_pred_proba)

            # Get class names - handle CalibratedClassifierCV
            if isinstance(model, CalibratedClassifierCV):
                if hasattr(model, "estimators_") and len(model.estimators_) > 0:
                    model_class_names = model.estimators_[0].classes_
                elif hasattr(model, "base_estimator") and hasattr(
                    model.base_estimator, "classes_"
                ):
                    model_class_names = model.base_estimator.classes_
                else:
                    model_class_names = class_names  # Use the ones we defined earlier
            else:
                model_class_names = (
                    model.classes_ if hasattr(model, "classes_") else class_names
                )

            y_pred = model_class_names[y_pred_idx]
            confidence = y_pred_proba[y_pred_idx]

            console_log(
                f"True class: {y_true}, Predicted: {y_pred} (confidence: {confidence:.3f})"
            )

            # Generate explanation using the SAME unified function
            try:
                exp = explainer.explain_instance(
                    x_sample,
                    unified_predict_fn,  # <- Use the same function!
                    num_features=min(num_features, len(feature_names)),
                    num_samples=num_samples,
                    top_labels=len(class_names),
                )

                # Store explanation info
                exp_dict = {
                    "sample_idx": original_idx,  # Use original index
                    "true_label": y_true,
                    "predicted_label": y_pred,
                    "confidence": float(confidence),
                    "explanation": exp,
                }

                # Visualize if requested
                if show_plots:
                    _visualize_lime_explanation(
                        visualizer_instance,
                        exp,
                        class_names,
                        feature_names,
                        y_true,
                        y_pred,
                        confidence,
                        positive_label,
                        negative_label,
                        figsize=figsize,
                        sample_index=original_idx,
                        wavenumber_axis=(
                            wavenumber_axis_reduced
                            if feature_selector
                            else wavenumber_axis
                        ),
                    )

                explanations.append(exp_dict)

            except Exception as e:
                console_log(f"Error explaining instance {original_idx}: {str(e)}")
                continue

        # Create data arrays for wavenumbers with most impact
        top_features = {}
        for class_name in class_names:
            # Collect feature weights across all explanations for this class
            class_feature_weights = {}

            for exp_dict in explanations:
                exp = exp_dict["explanation"]

                # Get the feature list for this class
                try:
                    class_idx = class_names.index(class_name)
                    for feature_id, weight in exp.local_exp[class_idx]:
                        feature_name = feature_names[feature_id]

                        if feature_name not in class_feature_weights:
                            class_feature_weights[feature_name] = []

                        class_feature_weights[feature_name].append(weight)
                except (KeyError, IndexError, AttributeError, ValueError):
                    # This explanation might not have this class
                    continue

            # Calculate average weights
            avg_weights = {
                feature: np.mean(weights)
                for feature, weights in class_feature_weights.items()
            }

            # Sort features by absolute weight and take top ones
            sorted_features = sorted(
                avg_weights.items(), key=lambda x: abs(x[1]), reverse=True
            )[:num_features]

            top_features[class_name] = sorted_features

        # Create comparison visualization of top features across classes if binary
        if len(class_names) == 2 and show_plots:
            _visualize_lime_comparison(
                visualizer_instance,
                top_features,
                class_names,
                positive_label=positive_label,
                negative_label=negative_label,
                figsize=figsize,
            )

        return {
            "success": True,
            "explanation_objects": explanations,
            "num_samples": len(indices),
            "top_features_by_class": top_features,
            "class_names": class_names,
            "feature_names": feature_names,
            "feature_selection_used": feature_selector is not None,
            "processing_time": time.time() - start_time,
            "wavenumber_axis": (
                (
                    wavenumber_axis_reduced if feature_selector else wavenumber_axis
                ).tolist()
                if wavenumber_axis is not None
                else None
            ),
            "unified_prediction_function_used": True,  # New flag to indicate fix
        }

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        console_log(f"LIME explanation error: {str(e)}")
        console_log(error_details)

        return {"success": False, "error": str(e), "error_details": error_details}


def _visualize_lime_explanation(
    visualizer_instance,
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

    Args:
        visualizer_instance: RamanVisualizer instance (for self reference)
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
    # Plot info
    plt.figure(figsize=figsize)

    # Pick the class index to show (predicted class)
    if len(class_names) == 2:
        # For binary classification
        show_class_idx = class_names.index(predicted_label)
    else:
        # For multiclass, show the predicted class
        show_class_idx = (
            class_names.index(predicted_label) if predicted_label in class_names else 0
        )

    # Get explanation data
    try:
        # Extract explanation for the selected class
        exp_list = explanation.as_list(label=show_class_idx)

        # Sort by absolute impact
        exp_list = sorted(exp_list, key=lambda x: abs(float(x[1])), reverse=True)

        # Extract features and weights
        features = [x[0] for x in exp_list]
        weights = [float(x[1]) for x in exp_list]

        # Split into positive and negative impacts
        pos_features = []
        pos_weights = []
        neg_features = []
        neg_weights = []

        for f, w in zip(features, weights):
            if w > 0:
                pos_features.append(f)
                pos_weights.append(w)
            else:
                neg_features.append(f)
                neg_weights.append(abs(w))  # Use absolute for plotting

        # Formatting and colors
        pos_color = "#D62728"  # Red for positive impacts (cancer)
        neg_color = "#1F77B4"  # Blue for negative impacts (benign)

        # Create horizontal bar chart with separate bars
        plt.figure(figsize=figsize)

        # Calculate positions for bars
        num_features = max(len(pos_features), len(neg_features))
        bar_height = 0.4
        y_pos = np.arange(num_features) * 1.0

        # Ensure we have labels for all positions
        all_features = []
        for i in range(num_features):
            if i < len(pos_features) and i < len(neg_features):
                all_features.append(f"{pos_features[i]} / {neg_features[i]}")
            elif i < len(pos_features):
                all_features.append(pos_features[i])
            elif i < len(neg_features):
                all_features.append(neg_features[i])

        # Plot bars for positive impact (right side)
        if pos_weights:
            plt.barh(
                y_pos[: len(pos_weights)],
                pos_weights,
                height=bar_height,
                color=pos_color,
                alpha=0.7,
                label=f"Toward {positive_label.upper()}",
            )

        # Plot bars for negative impact (left side)
        if neg_weights:
            plt.barh(
                y_pos[: len(neg_weights)],
                [-w for w in neg_weights],
                height=bar_height,
                color=neg_color,
                alpha=0.7,
                label=f"Toward {negative_label.upper()}",
            )

        # Set up the plot
        plt.axvline(x=0, color="black", linestyle="-", alpha=0.5)
        plt.grid(alpha=0.3)

        # Custom y-ticks for wavenumbers
        try:
            # Try to extract numerical wavenumbers from feature names
            wavenumbers = []
            for feature in features[:num_features]:
                match = re.search(r"(\d+\.\d+)", feature)
                if match:
                    wavenumbers.append(f"{float(match.group(1)):.1f} cm⁻¹")
                else:
                    wavenumbers.append(feature)

            plt.yticks(y_pos[:num_features], wavenumbers)
        except (ValueError, TypeError):
            plt.yticks(
                y_pos[:num_features],
                [f.split(" ≤")[0].split(" >")[0] for f in all_features[:num_features]],
            )

        # Add labels
        plt.xlabel("Impact on Prediction")
        plt.ylabel("Raman Shift")

        # Title with accuracy information
        correct_str = "✓ Correct" if predicted_label == true_label else "✗ Incorrect"
        title = (
            f"LIME Explanation (Sample #{sample_index})\n"
            f"True: {true_label}, Predicted: {predicted_label} ({confidence:.3f}) - {correct_str}"
        )

        plt.title(title, fontweight="bold")
        plt.legend(loc="best")
        plt.tight_layout()

        # Add a table with key information
        table_data = [
            ["Prediction", predicted_label, f"Confidence: {confidence:.3f}"],
            ["True Label", true_label, f"Result: {correct_str}"],
            ["Class Probabilities", "", ""],
        ]

        # Add probabilities for each class
        class_probas = explanation.predict_proba.tolist()
        for i, (cls, proba) in enumerate(zip(class_names, class_probas)):
            formatted_proba = f"{proba:.3f}"
            highlight = "→ " if i == show_class_idx else ""
            table_data.append([f"{highlight}{cls}", formatted_proba, ""])

        # Compute table position
        table = plt.table(
            cellText=table_data,
            cellLoc="left",
            loc="bottom",
            bbox=[0.0, -0.32, 0.6, 0.20],  # Adjust as needed
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        # Color the table cells
        for i, row in enumerate(table_data):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:  # Prediction row
                    if j == 1:  # Class name
                        text_color = (
                            "#D62728"
                            if predicted_label == positive_label
                            else "#1F77B4"
                        )
                        cell.set_text_props(weight="bold", color=text_color)
                elif i == 1:  # True label row
                    if j == 1:  # Class name
                        text_color = (
                            "#D62728" if true_label == positive_label else "#1F77B4"
                        )
                        cell.set_text_props(weight="bold", color=text_color)
                elif i == 2:  # Header for probabilities
                    cell.set_facecolor("#F0F0F0")
                    cell.set_text_props(weight="bold")
                elif i > 2:  # Class probability rows
                    if f"{class_names[i-3]}" == positive_label:
                        cell.set_text_props(color="#D62728")
                    elif f"{class_names[i-3]}" == negative_label:
                        cell.set_text_props(color="#1F77B4")

        plt.subplots_adjust(bottom=0.25)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        console_log(f"Error in LIME visualization: {str(e)}")
        import traceback

        traceback.print_exc()


def _visualize_lime_comparison(
    visualizer_instance,
    top_features,
    class_names,
    positive_label="cancer",
    negative_label="benign",
    figsize=(14, 10),
):
    """
    Visualize comparison of feature importance between classes.

    Args:
        visualizer_instance: RamanVisualizer instance (for self reference)
        top_features: Dictionary with class names as keys and lists of (feature, weight) tuples as values
        class_names: List of class names
        positive_label: Name for positive label
        negative_label: Name for negative label
        figsize: Figure size for plot
    """
    if len(class_names) != 2:
        console_log("Feature comparison visualization requires exactly 2 classes.")
        return

    plt.figure(figsize=figsize)

    # Get the class names in correct order
    pos_class = positive_label
    neg_class = negative_label

    if pos_class not in top_features or neg_class not in top_features:
        # Try case-insensitive matching
        for cls in top_features.keys():
            if cls.lower() == pos_class.lower():
                pos_class = cls
            if cls.lower() == neg_class.lower():
                neg_class = cls

    # Organize the feature data
    pos_data = dict(top_features.get(pos_class, []))
    neg_data = dict(top_features.get(neg_class, []))

    # Get all unique features
    all_features = set(list(pos_data.keys()) + list(neg_data.keys()))

    # Extract wavenumbers for sorting
    feature_wavenumbers = {}
    for feature in all_features:
        # Extract wavenumber from feature name (format: "123.4 cm⁻¹")
        match = re.search(r"(\d+\.\d+)", feature)
        if match:
            feature_wavenumbers[feature] = float(match.group(1))
        else:
            # If we can't extract, use a default
            feature_wavenumbers[feature] = 0

    # Sort features by wavenumber
    sorted_features = sorted(all_features, key=lambda f: feature_wavenumbers[f])

    # Prepare data for plotting
    x = np.arange(len(sorted_features))
    bar_width = 0.35

    # Get feature importance values in the sorted order
    pos_values = [pos_data.get(f, 0) for f in sorted_features]
    neg_values = [neg_data.get(f, 0) for f in sorted_features]

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(
        x - bar_width / 2,
        pos_values,
        bar_width,
        label=f"{pos_class}",
        color="#D62728",
        alpha=0.7,
    )
    bars2 = ax.bar(
        x + bar_width / 2,
        neg_values,
        bar_width,
        label=f"{neg_class}",
        color="#1F77B4",
        alpha=0.7,
    )

    # Add some text for labels, title and axes ticks
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Average Feature Importance")
    ax.set_title("Comparison of Important Features by Class")
    ax.set_xticks(x)

    # Format x-tick labels to show just the wavenumber
    x_labels = []
    for feature in sorted_features:
        match = re.search(r"(\d+\.\d+)", feature)
        if match:
            x_labels.append(f"{float(match.group(1)):.1f}")
        else:
            x_labels.append(feature.split(" ")[0])

    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend()

    # Add data labels above bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.01:  # Only label significant values
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    # 3 points vertical offset for positive, -10 for negative
                    xytext=(0, 3 if height > 0 else -10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if height > 0 else "top",
                    fontsize=8,
                )

    autolabel(bars1)
    autolabel(bars2)

    # Add grid for readability
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    # Add "cm⁻¹" to the x-label
    ax.set_xlabel("Raman Shift (cm⁻¹)")

    # Adjust bottom margin to fit labels
    plt.subplots_adjust(bottom=0.15)

    plt.show()

    # Also create a table view of the data
    plt.figure(figsize=(10, len(sorted_features) * 0.4 + 3))
    plt.axis("off")

    table_data = []
    table_data.append(
        [
            "Raman Shift (cm⁻¹)",
            f"{pos_class} Importance",
            f"{neg_class} Importance",
            "Difference",
        ]
    )

    for i, feature in enumerate(sorted_features):
        pos_val = pos_data.get(feature, 0)
        neg_val = neg_data.get(feature, 0)
        diff = abs(pos_val) - abs(neg_val)

        row_data = [x_labels[i], f"{pos_val:.4f}", f"{neg_val:.4f}", f"{diff:.4f}"]
        table_data.append(row_data)

    table = plt.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.2, 0.26, 0.26, 0.26],
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color the table cells
    for i, row in enumerate(table_data):
        for j in range(4):
            cell = table[(i, j)]

            # Header row
            if i == 0:
                cell.set_facecolor("#F0F0F0")
                cell.set_text_props(weight="bold")
                if j == 1:
                    # Positive class (red)
                    cell.set_text_props(color="#D62728")
                elif j == 2:
                    # Negative class (blue)
                    cell.set_text_props(color="#1F77B4")
            # Data rows
            else:
                # Positive class column
                if j == 1 and float(row[j]) > 0:
                    cell.set_facecolor("#FFEDED")
                    cell.set_text_props(color="#D62728")
                # Negative class column
                elif j == 2 and float(row[j]) > 0:
                    cell.set_facecolor("#EDF6FF")
                    cell.set_text_props(color="#1F77B4")
                # Difference column
                elif j == 3:
                    try:
                        diff_val = float(row[j])
                        if diff_val > 0:  # Positive class more important
                            cell.set_facecolor("#FFEDED")
                            cell.set_text_props(color="#D62728")
                        elif diff_val < 0:  # Negative class more important
                            cell.set_facecolor("#EDF6FF")
                            cell.set_text_props(color="#1F77B4")
                    except (ValueError, TypeError):
                        pass

    plt.title("Feature Importance Comparison Table", fontweight="bold")
    plt.tight_layout()
    plt.show()
