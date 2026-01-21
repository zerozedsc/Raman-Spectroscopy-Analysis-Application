"""
Refactored ML Models for Raman Spectroscopy Classification

This module provides a unified, optimized architecture for machine learning models
used in Raman spectroscopy analysis, specifically for MGUS/MM classification.

Architecture:
- BaseRamanModel: Abstract base class with common functionality
- ProbabilityMixin: For models supporting probability predictions
- FeatureImportanceMixin: For models with feature importance
- Concrete model implementations extending base classes

Author: Refactored on 2025-10-15
"""

import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, log_loss, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import pickle
from notebook_utils.visualize import MLVisualize


# ============================================================================
# BASE CLASS AND MIXINS
# ============================================================================

class BaseRamanModel(ABC):
    """
    Abstract base class for all Raman spectroscopy classification models.
    
    Provides common functionality for:
    - Data validation and preparation
    - Label encoding/decoding
    - Evaluation metrics calculation
    - Confusion matrix visualization
    - Model serialization
    
    Subclasses must implement:
    - fit(): Train the model
    - _predict_raw(): Make raw predictions (implementation-specific)
    """
    
    def __init__(self, data_split: Dict[str, Any], model_name: str = "RamanModel"):
        """
        Initialize base model with common setup.
        
        Args:
            data_split: Dictionary containing X_train, X_test, y_train, y_test, unified_wavelengths
            model_name: Name identifier for the model
        """
        self.model_name = model_name
        self.data_split = data_split
        self.model = None  # Subclasses will set this
        
        # Validate required keys
        self._validate_data_split()
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        self.y_train_encoded = self.label_encoder.fit_transform(data_split['y_train'])
        self.y_test_encoded = self.label_encoder.transform(data_split['y_test'])
        
        # Store class information
        self.classes_ = self.label_encoder.classes_
        self.n_classes = len(self.classes_)
        
        print(f"=== {self.model_name} Initialized ===")
        print(f"Classes: {list(self.classes_)}")
        print(f"Encoded mapping: {dict(zip(self.classes_, range(self.n_classes)))}")
        print(f"Training samples: {len(data_split['X_train'])}")
        print(f"Test samples: {len(data_split['X_test'])}")
        print(f"Features: {data_split['X_train'].shape[1]}")
    
    def _validate_data_split(self) -> None:
        """Validate that data_split contains all required keys."""
        required_keys = ['X_train', 'X_test', 'y_train', 'y_test', 'unified_wavelengths']
        missing_keys = [key for key in required_keys if key not in self.data_split]
        if missing_keys:
            raise ValueError(f"Missing required keys in data_split: {missing_keys}")
    
    @abstractmethod
    def fit(self) -> None:
        """Train the model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Make raw predictions (implementation-specific).
        
        Args:
            X: Feature matrix
            
        Returns:
            Raw predictions (encoded labels or continuous values)
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions and return encoded labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted encoded labels
        """
        raw_predictions = self._predict_raw(X)
        
        # For regression-like models, round to nearest class
        if raw_predictions.dtype in [np.float32, np.float64]:
            predictions_rounded = np.round(raw_predictions).astype(int)
            predictions_clipped = np.clip(predictions_rounded, 0, self.n_classes - 1)
            return predictions_clipped
        
        return raw_predictions
    
    def predict_labels(self, X: np.ndarray) -> List[str]:
        """
        Make predictions and return original label strings.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of predicted label strings
        """
        predictions_encoded = self.predict(X)
        return list(self.label_encoder.inverse_transform(predictions_encoded))
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate model on test set with comprehensive metrics.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred_labels = self.predict_labels(self.data_split['X_test'])
        y_true_labels = self.data_split['y_test']
        
        # Classification metrics
        metrics = self._calculate_classification_metrics(y_true_labels, y_pred_labels)
        
        # Add model-specific metrics
        additional_metrics = self._get_additional_metrics()
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Print summary
        self._print_evaluation_summary(metrics, y_true_labels, y_pred_labels)
        
        return metrics
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, 
                                          y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate standard classification metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        return {
            'classification': {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted,
                'precision_per_class': dict(zip(self.classes_, precision_per_class)),
                'recall_per_class': dict(zip(self.classes_, recall_per_class)),
                'f1_per_class': dict(zip(self.classes_, f1_per_class))
            }
        }
    
    def _get_additional_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Hook for subclasses to add model-specific metrics.
        
        Returns:
            Dictionary of additional metrics or None
        """
        return None
    
    def _print_evaluation_summary(self, metrics: Dict[str, Any], 
                                  y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Print formatted evaluation summary."""
        print(f"\n=== {self.model_name} Evaluation Results ===")
        
        if 'classification' in metrics:
            cls_metrics = metrics['classification']
            print(f"\nAccuracy: {cls_metrics['accuracy']:.4f}")
            print(f"F1-Macro: {cls_metrics['f1_macro']:.4f}")
            print(f"F1-Weighted: {cls_metrics['f1_weighted']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.classes_, 
                                   zero_division=0))
    
    def plot_confusion_matrix(self, show_plot: bool = True) -> np.ndarray:
        """
        Compute and plot confusion matrix for test set.
        
        Args:
            show_plot: Whether to display the plot
            
        Returns:
            Confusion matrix as numpy array
        """
        y_pred_labels = self.predict_labels(self.data_split['X_test'])
        y_true = self.data_split['y_test']
        
        cm = confusion_matrix(y_true, y_pred_labels, labels=self.classes_)
        
        # Print text matrix
        self._print_confusion_matrix_text(cm, "Test Data")
        
        if show_plot:
            self._plot_confusion_matrix_figure(cm, y_true, y_pred_labels, "Test Data")
        
        return cm
    
    def _print_confusion_matrix_text(self, cm: np.ndarray, dataset_name: str) -> None:
        """Print confusion matrix in text format."""
        print(f"\nConfusion Matrix (Text) for {dataset_name}:")
        print("Predicted ->")
        labels = list(self.classes_) + ["Total"]
        print("Actual |", " | ".join(f"{label:>8}" for label in labels))
        print("-" * (10 + 10 * len(labels)))
        
        for i, true_label in enumerate(self.classes_):
            row = [f"{cm[i, j]:>8}" for j in range(len(self.classes_))]
            total = sum(cm[i, :])
            row.append(f"{total:>8}")
            print(f"{true_label:>6} | {' | '.join(row)}")
    
    def _plot_confusion_matrix_figure(self, cm: np.ndarray, y_true: np.ndarray, 
                                     y_pred: np.ndarray, dataset_name: str) -> None:
        """Create matplotlib figure for confusion matrix."""
        report = classification_report(y_true, y_pred, target_names=self.classes_, 
                                      zero_division=0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Calculate percentages
        total = cm.sum()
        annot = np.array([[f"{cm[i, j]}\n({cm[i, j]/total*100:.1f}%)"
                          for j in range(cm.shape[1])]
                          for i in range(cm.shape[0])])
        
        # Heatmap
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                   xticklabels=self.classes_,
                   yticklabels=self.classes_, ax=ax1)
        ax1.set_title(f'Confusion Matrix for {dataset_name}\n{self.model_name}')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Classification report
        ax2.text(0.1, 0.5, report, fontsize=10,
                verticalalignment='center', fontfamily='monospace')
        ax2.set_title(f'Classification Report for {dataset_name}')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def predict_new_data(self, X_new: np.ndarray, y_new: Optional[np.ndarray] = None,
                        show_confusion_matrix: bool = False) -> Dict[str, Any]:
        """
        Predict on new external data and optionally evaluate.
        
        Args:
            X_new: New feature matrix
            y_new: True labels (optional, for evaluation)
            show_confusion_matrix: Whether to show confusion matrix if y_new provided
            
        Returns:
            Dictionary with predictions and metrics (if y_new provided)
        """
        predictions_labels = self.predict_labels(X_new)
        predictions_encoded = self.predict(X_new)
        
        result = {
            'predictions_labels': predictions_labels,
            'predictions_encoded': predictions_encoded,
            'n_predictions': len(predictions_labels)
        }
        
        print(f"=== {self.model_name} Predictions on New Data ===")
        print(f"Number of samples: {len(predictions_labels)}")
        print(f"Predicted distribution: {dict(zip(*np.unique(predictions_labels, return_counts=True)))}")
        
        if y_new is not None:
            # Calculate standard classification metrics
            metrics = self._calculate_classification_metrics(y_new, predictions_labels)
            
            # Add model-specific metrics (e.g., log_loss for LogisticRegression)
            # Temporarily store original test data to call _get_additional_metrics
            original_test = (self.data_split['X_test'], self.data_split['y_test'], 
                           self.y_test_encoded)
            try:
                # Temporarily replace with new data for additional metrics calculation
                self.data_split['X_test'] = X_new
                self.data_split['y_test'] = y_new
                # Encode new labels
                y_new_encoded = self.label_encoder.transform(y_new)
                self.y_test_encoded = y_new_encoded
                
                additional_metrics = self._get_additional_metrics()
                if additional_metrics:
                    metrics.update(additional_metrics)
            finally:
                # Restore original test data
                self.data_split['X_test'], self.data_split['y_test'], self.y_test_encoded = original_test
            
            result['evaluation'] = metrics
            
            print("\n=== Evaluation on New Data ===")
            print(f"Accuracy: {metrics['classification']['accuracy']:.4f}")
            print(f"F1-Macro: {metrics['classification']['f1_macro']:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_new, predictions_labels, 
                                      target_names=self.classes_, zero_division=0))
            
            if show_confusion_matrix:
                cm = confusion_matrix(y_new, predictions_labels, labels=self.classes_)
                self._print_confusion_matrix_text(cm, "New Data")
                self._plot_confusion_matrix_figure(cm, y_new, predictions_labels, "New Data")
        
        return result
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to disk using pickle.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'data_split': self.data_split,
            'model_name': self.model_name,
            'model_class': self.__class__.__name__
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"{self.model_name} saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(model_data['data_split'])
        instance.model = model_data['model']
        instance.label_encoder = model_data['label_encoder']
        instance.model_name = model_data.get('model_name', instance.model_name)
        
        print(f"Model loaded from {filepath}")
        return instance


class ProbabilityMixin:
    """
    Mixin for models that support probability predictions.
    
    Provides:
    - predict_proba(): Get class probabilities
    - plot_probability_distributions(): Visualize prediction confidence
    """
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError(
                f"{self.model_name} does not support probability predictions"
            )
        
        return self.model.predict_proba(X)
    
    def plot_probability_distributions(self, X: Optional[np.ndarray] = None, 
                                      y: Optional[np.ndarray] = None) -> None:
        """
        Plot prediction probability distributions.
        
        Args:
            X: Feature matrix (defaults to test set)
            y: True labels (defaults to test labels)
        """
        if X is None:
            X = self.data_split['X_test']
            y = self.data_split['y_test']
        
        probas = self.predict_proba(X)
        max_probas = np.max(probas, axis=1)
        predictions = self.predict_labels(X)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram of max probabilities
        axes[0].hist(max_probas, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Maximum Probability')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{self.model_name}: Prediction Confidence Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot by true class
        data_by_class = {cls: max_probas[y == cls] for cls in self.classes_}
        axes[1].boxplot(data_by_class.values(), labels=data_by_class.keys())
        axes[1].set_xlabel('True Class')
        axes[1].set_ylabel('Maximum Probability')
        axes[1].set_title(f'{self.model_name}: Confidence by True Class')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nProbability Statistics:")
        print(f"  Mean confidence: {np.mean(max_probas):.3f}")
        print(f"  Median confidence: {np.median(max_probas):.3f}")
        print(f"  Min confidence: {np.min(max_probas):.3f}")
        print(f"  Max confidence: {np.max(max_probas):.3f}")


class FeatureImportanceMixin:
    """
    Mixin for models that provide feature importance.
    
    Provides:
    - get_feature_importance(): Extract feature importance scores
    - plot_feature_importance(): Visualize top features
    """
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, Any]:
        """
        Get feature importance scores.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature importance information
        """
        importance = self._extract_feature_importance()
        
        if importance is None:
            raise NotImplementedError(
                f"{self.model_name} does not support feature importance"
            )
        
        # Get wavelengths if available
        wavelengths = self.data_split.get('unified_wavelengths', 
                                         np.arange(len(importance)))
        
        # Sort by absolute importance
        sorted_indices = np.argsort(np.abs(importance))[::-1]
        top_indices = sorted_indices[:top_n]
        
        return {
            'importance': importance,
            'top_indices': top_indices,
            'top_wavelengths': wavelengths[top_indices],
            'top_values': importance[top_indices],
            'all_wavelengths': wavelengths
        }
    
    def _extract_feature_importance(self) -> Optional[np.ndarray]:
        """
        Extract feature importance from model.
        Must be implemented or overridden by subclass.
        
        Returns:
            Feature importance array or None
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use coefficients
            coef = self.model.coef_
            if len(coef.shape) == 2:
                # Multi-class: aggregate across classes
                return np.mean(np.abs(coef), axis=0)
            return coef
        return None
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """
        Plot top N most important features.
        
        Args:
            top_n: Number of top features to plot
        """
        importance_data = self.get_feature_importance(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.barh(range(top_n), importance_data['top_values'][::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f"{w:.1f} cm⁻¹" for w in importance_data['top_wavelengths'][::-1]])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{self.model_name}: Top {top_n} Most Important Features')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# CONCRETE MODEL IMPLEMENTATIONS
# ============================================================================

class LinearRegressionModel(BaseRamanModel):
    """
    Linear Regression model for Raman spectroscopy classification.
    
    Note: Treats classification as regression problem by encoding labels numerically.
    For proper classification, consider LogisticRegressionModel instead.
    """
    
    def __init__(self, data_split: Dict[str, Any], **kwargs):
        """
        Initialize Linear Regression model.
        
        Args:
            data_split: Data dictionary from data preparer
            **kwargs: Additional arguments for sklearn LinearRegression
        """
        super().__init__(data_split, model_name="Linear Regression")
        self.model = LinearRegression(**kwargs)
    
    def fit(self) -> None:
        """Fit the linear regression model."""
        print(f"\nFitting {self.model_name}...")
        self.model.fit(self.data_split['X_train'], self.y_train_encoded)
        print("Model fitted successfully.")
    
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Make raw predictions (continuous values)."""
        return self.model.predict(X)
    
    def _get_additional_metrics(self) -> Dict[str, Any]:
        """Add regression-specific metrics."""
        y_pred_encoded = self._predict_raw(self.data_split['X_test'])
        
        mse = mean_squared_error(self.y_test_encoded, y_pred_encoded)
        r2 = r2_score(self.y_test_encoded, y_pred_encoded)
        
        return {
            'regression': {
                'mean_squared_error': mse,
                'r2_score': r2,
                'note': 'Regression metrics for label encoding'
            }
        }


class LogisticRegressionModel(BaseRamanModel, ProbabilityMixin, FeatureImportanceMixin):
    """
    Logistic Regression model for Raman spectroscopy classification.
    
    Proper probabilistic classification with:
    - Class probability predictions
    - Feature importance via coefficients
    - ROC curve analysis
    - Confidence scores
    """
    
    def __init__(self, data_split: Dict[str, Any],
                 penalty: str = 'l2',
                 C: float = 1.0,
                 solver: str = 'lbfgs',
                 max_iter: int = 1000,
                 multi_class: str = 'auto',
                 class_weight: Optional[str] = None,
                 random_state: Optional[int] = 42,
                 scale_features: bool = True,
                 **kwargs):
        """
        Initialize Logistic Regression model.
        
        Args:
            data_split: Data dictionary
            penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
            C: Inverse regularization strength
            solver: Optimization algorithm
            max_iter: Maximum iterations
            multi_class: Multi-class strategy
            class_weight: Class balancing ('balanced' or None)
            random_state: Random seed
            scale_features: Whether to standardize features
            **kwargs: Additional LogisticRegression arguments
        """
        super().__init__(data_split, model_name="Logistic Regression")
        
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            class_weight=class_weight,
            random_state=random_state,
            **kwargs
        )
        
        print(f"Solver: {solver}, Penalty: {penalty}, C: {C}")
        print(f"Class weight: {class_weight}, Feature scaling: {scale_features}")
    
    def fit(self) -> None:
        """Fit the logistic regression model."""
        print(f"\nFitting {self.model_name}...")
        
        X_train = self.data_split['X_train'].copy()
        
        if self.scale_features:
            print("Applying feature standardization...")
            X_train = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_train, self.y_train_encoded)
        print("Model fitted successfully.")
        
        if hasattr(self.model, 'n_iter_'):
            print(f"Solver converged in {self.model.n_iter_[0]} iterations")
    
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Make raw predictions with optional scaling."""
        X_scaled = self.scaler.transform(X) if self.scale_features else X
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities with optional scaling."""
        X_scaled = self.scaler.transform(X) if self.scale_features else X
        return self.model.predict_proba(X_scaled)
    
    def _get_additional_metrics(self) -> Dict[str, Any]:
        """Add logistic regression specific metrics."""
        X_test = self.data_split['X_test']
        if self.scale_features:
            X_test = self.scaler.transform(X_test)
        
        y_pred_proba = self.model.predict_proba(X_test)
        logloss = log_loss(self.y_test_encoded, y_pred_proba)
        
        # AUC-ROC
        try:
            if self.n_classes == 2:
                auc_roc = roc_auc_score(self.y_test_encoded, y_pred_proba[:, 1])
            else:
                auc_roc = roc_auc_score(self.y_test_encoded, y_pred_proba, 
                                       multi_class='ovr', average='macro')
        except ValueError:
            auc_roc = None
        
        return {
            'logistic_regression': {
                'log_loss': logloss,
                'auc_roc': auc_roc,
                'n_iterations': getattr(self.model, 'n_iter_', 'N/A')
            }
        }
    
    def plot_roc_curves(self) -> None:
        """Plot ROC curves for each class."""
        from sklearn.preprocessing import label_binarize
        
        X_test = self.data_split['X_test']
        if self.scale_features:
            X_test = self.scaler.transform(X_test)
        
        y_pred_proba = self.model.predict_proba(X_test)
        y_test_bin = label_binarize(self.y_test_encoded, classes=range(self.n_classes))
        
        # Fix for binary classification: label_binarize returns 1D array for 2 classes
        # Need to reshape to 2D and add complementary probabilities
        if self.n_classes == 2:
            y_test_bin = np.column_stack([1 - y_test_bin, y_test_bin])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.classes_):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            auc = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
            ax.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{self.model_name}: ROC Curves')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class KNNModel(BaseRamanModel):
    """K-Nearest Neighbors classifier for Raman spectroscopy."""
    
    def __init__(self, data_split: Dict[str, Any], n_neighbors: int = 5, **kwargs):
        """
        Initialize KNN model.
        
        Args:
            data_split: Data dictionary
            n_neighbors: Number of neighbors to use
            **kwargs: Additional KNeighborsClassifier arguments
        """
        super().__init__(data_split, model_name="K-Nearest Neighbors")
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
        print(f"n_neighbors: {n_neighbors}")
    
    def fit(self) -> None:
        """Fit the KNN model."""
        print(f"\nFitting {self.model_name}...")
        self.model.fit(self.data_split['X_train'], self.y_train_encoded)
        print("Model fitted successfully.")
    
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)


class RandomForestModel(BaseRamanModel, FeatureImportanceMixin):
    """Random Forest classifier for Raman spectroscopy."""
    
    def __init__(self, data_split: Dict[str, Any], n_estimators: int = 100, **kwargs):
        """
        Initialize Random Forest model.
        
        Args:
            data_split: Data dictionary
            n_estimators: Number of trees
            **kwargs: Additional RandomForestClassifier arguments
        """
        super().__init__(data_split, model_name="Random Forest")
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(n_estimators=n_estimators, **kwargs)
        print(f"n_estimators: {n_estimators}")
    
    def fit(self) -> None:
        """Fit the Random Forest model."""
        print(f"\nFitting {self.model_name}...")
        self.model.fit(self.data_split['X_train'], self.y_train_encoded)
        print("Model fitted successfully.")
    
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)


class SVMModel(BaseRamanModel):
    """Support Vector Machine classifier for Raman spectroscopy."""
    
    def __init__(self, data_split: Dict[str, Any], C: float = 1.0, 
                 kernel: str = 'rbf', **kwargs):
        """
        Initialize SVM model.
        
        Args:
            data_split: Data dictionary
            C: Regularization parameter
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            **kwargs: Additional SVC arguments
        """
        super().__init__(data_split, model_name="Support Vector Machine")
        self.C = C
        self.kernel = kernel
        self.model = SVC(C=C, kernel=kernel, **kwargs)
        print(f"C: {C}, kernel: {kernel}")
    
    def fit(self) -> None:
        """Fit the SVM model."""
        print(f"\nFitting {self.model_name}...")
        self.model.fit(self.data_split['X_train'], self.y_train_encoded)
        print("Model fitted successfully.")
    
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)


class XGBoostModel(BaseRamanModel, ProbabilityMixin, FeatureImportanceMixin):
    """XGBoost classifier for Raman spectroscopy."""
    
    def __init__(self, data_split: Dict[str, Any],
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.3,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize XGBoost model.
        
        Args:
            data_split: Data dictionary
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            random_state: Random seed
            **kwargs: Additional XGBClassifier arguments
        """
        super().__init__(data_split, model_name="XGBoost")
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )
        
        print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, "
              f"learning_rate: {learning_rate}")
    
    def fit(self) -> None:
        """Fit the XGBoost model."""
        print(f"\nFitting {self.model_name}...")
        self.model.fit(self.data_split['X_train'], self.y_train_encoded)
        print("Model fitted successfully.")
    
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)



# ============================================================================
# EXPORT CONVENIENCE FUNCTION
# ============================================================================

def create_model(model_type: str, data_split: Dict[str, Any], **kwargs):
    """
    Factory function to create models by type.
    
    Args:
        model_type: Model type ('linear', 'logistic', 'knn', 'rf', 'svm', 'xgboost')
        data_split: Data dictionary
        **kwargs: Model-specific arguments
        
    Returns:
        Initialized model instance
    """
    models = {
        'linear': LinearRegressionModel,
        'logistic': LogisticRegressionModel,
        'knn': KNNModel,
        'rf': RandomForestModel,
        'randomforest': RandomForestModel,
        'svm': SVMModel,
        'xgboost': XGBoostModel,
        'xgb': XGBoostModel
    }
    
    model_class = models.get(model_type.lower())
    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(models.keys())}")
    
    return model_class(data_split, **kwargs)


if __name__ == "__main__":
    print("Refactored Raman ML Models")
    print("Available models:", ['LinearRegressionModel', 'LogisticRegressionModel',
                                'KNNModel', 'RandomForestModel', 'SVMModel', 
                                'XGBoostModel'])
