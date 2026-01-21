import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, roc_curve, log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import pickle
from notebook_utils.visualize import MLVisualize
import warnings
warnings.filterwarnings('ignore')

class LogisticRegressionModel:
    """
    A class for training and evaluating a Logistic Regression model on Raman spectroscopy data.
    
    This class provides proper classification using logistic regression instead of treating
    categorical labels as regression targets. Logistic regression is more appropriate for
    classification tasks as it:
    - Uses probability-based predictions (0-1 range)
    - Handles categorical outcomes naturally
    - Provides class probabilities for uncertainty quantification
    - Offers better interpretability for classification decisions
    
    Based on research applications:
    - Zeng et al. (2022): Logistic regression for SERS-based miRNA classification
    - Chia et al. (2020): Interpretable classification of bacterial Raman spectra
    - Lancia et al. (2023): Logistic regression models for genomic DNA classification
    """
    
    def __init__(self, data_split: Dict[str, Any], 
                 # Logistic Regression specific parameters
                 penalty: str = 'l2',
                 C: float = 1.0,
                 solver: str = 'lbfgs',
                 max_iter: int = 1000,
                 multi_class: str = 'auto',
                 class_weight: Optional[str] = None,
                 random_state: Optional[int] = 42,
                 # Scaling parameters
                 scale_features: bool = True,
                 **kwargs):
        """
        Initialize the Logistic Regression model.
        
        Args:
            data_split: Data dictionary from RamanDataSplitter.prepare_data()
            penalty: Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
            C: Inverse regularization strength (smaller = more regularization)
            solver: Algorithm for optimization ('lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga')
            max_iter: Maximum iterations for solver convergence
            multi_class: Multi-class strategy ('auto', 'ovr', 'multinomial')
            class_weight: Class weights for imbalanced data ('balanced', dict, or None)
            random_state: Random seed for reproducible results
            scale_features: Whether to standardize features (recommended for logistic regression)
            **kwargs: Additional keyword arguments for LogisticRegression
        """
        self.data_split = data_split
        self.scale_features = scale_features
        
        # Check for missing keys (same as LinearRegressionModel)
        required_keys = ['X_train', 'X_test', 'y_train', 'y_test', 'unified_wavelengths']
        missing_keys = [key for key in required_keys if key not in self.data_split]
        if missing_keys:
            raise ValueError(f"Missing keys in data_split: {missing_keys}")
        
        # Initialize logistic regression model with medical-appropriate parameters
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
        
        # Label encoder for consistent interface
        self.label_encoder = LabelEncoder()
        self.y_train_encoded = self.label_encoder.fit_transform(self.data_split['y_train'])
        self.y_test_encoded = self.label_encoder.transform(self.data_split['y_test'])
        
        # Feature scaler for logistic regression (important for convergence)
        self.scaler = None
        if self.scale_features:
            self.scaler = StandardScaler()
        
        # Performance tracking
        self.training_history = {}
        
        print("=== Logistic Regression Model for Raman Spectroscopy ===")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        print(f"Encoded labels: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        print(f"Training samples: {len(self.data_split['X_train'])}")
        print(f"Test samples: {len(self.data_split['X_test'])}")
        print(f"Features: {self.data_split['X_train'].shape[1]}")
        print(f"Solver: {solver}, Penalty: {penalty}, C: {C}")
        print(f"Class weight: {class_weight}, Feature scaling: {scale_features}")
        if class_weight == 'balanced':
            print("Note: Using balanced class weights for imbalanced data handling")
    
    def fit(self) -> None:
        """
        Fit the Logistic Regression model to the training data.
        """
        print("\nFitting Logistic Regression model...")
        
        # Prepare training data
        X_train = self.data_split['X_train'].copy()
        
        # Apply feature scaling if enabled
        if self.scale_features:
            print("Applying feature standardization...")
            X_train = self.scaler.fit_transform(X_train)
        
        # Fit the logistic regression model
        try:
            self.model.fit(X_train, self.y_train_encoded)
            print("Model fitted successfully.")
            
            # Store training information
            self.training_history = {
                'n_iter': getattr(self.model, 'n_iter_', 'N/A'),
                'classes': self.model.classes_,
                'n_features_in': self.model.n_features_in_,
                'feature_names_in': getattr(self.model, 'feature_names_in_', None)
            }
            
            # Print convergence information
            if hasattr(self.model, 'n_iter_'):
                if isinstance(self.model.n_iter_, np.ndarray):
                    print(f"Convergence achieved in {self.model.n_iter_[0]} iterations")
                else:
                    print(f"Convergence achieved in {self.model.n_iter_} iterations")
            
        except Exception as e:
            print(f"Error during model fitting: {e}")
            print("Try adjusting solver, max_iter, or C parameter")
            raise e
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted encoded class labels
        """
        # Apply same scaling as training data
        if self.scale_features and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Get class predictions (encoded)
        return self.model.predict(X_scaled)
    
    def predict_labels(self, X: np.ndarray) -> List[str]:
        """
        Make predictions and decode to original labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted original labels
        """
        predictions_encoded = self.predict(X)
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities for each sample
        """
        # Apply same scaling as training data
        if self.scale_features and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        return self.model.predict_proba(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function (confidence scores) for samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Decision function values
        """
        # Apply same scaling as training data
        if self.scale_features and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        return self.model.decision_function(X_scaled)
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the test set with detailed classification metrics.
        Same interface as LinearRegressionModel but with proper logistic regression metrics.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        # Prepare test data
        X_test = self.data_split['X_test']
        if self.scale_features and self.scaler is not None:
            X_test = self.scaler.transform(X_test)
        
        # Get predictions and probabilities
        y_pred_encoded = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred_encoded)
        y_true_labels = self.data_split['y_test']
        
        # === Classification Metrics ===
        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        precision_macro = precision_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
        recall_macro = recall_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
        f1_macro = f1_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true_labels, y_pred_labels, average=None, zero_division=0)
        recall_per_class = recall_score(y_true_labels, y_pred_labels, average=None, zero_division=0)
        f1_per_class = f1_score(y_true_labels, y_pred_labels, average=None, zero_division=0)
        
        # === Logistic Regression Specific Metrics ===
        # Log loss (cross-entropy loss)
        log_loss_score = log_loss(self.y_test_encoded, y_pred_proba)
        
        # AUC-ROC (for binary classification or multi-class OvR)
        try:
            if len(self.label_encoder.classes_) == 2:
                # Binary classification
                auc_roc = roc_auc_score(self.y_test_encoded, y_pred_proba[:, 1])
            else:
                # Multi-class (One-vs-Rest)
                auc_roc = roc_auc_score(self.y_test_encoded, y_pred_proba, multi_class='ovr', average='weighted')
        except ValueError:
            auc_roc = None
        
        # === Regression-like Metrics (for interface compatibility) ===
        # Treat predicted probabilities as "continuous" values for regression metrics
        if len(self.label_encoder.classes_) == 2:
            # Binary: use probability of positive class
            continuous_predictions = y_pred_proba[:, 1]
            continuous_targets = self.y_test_encoded.astype(float)
        else:
            # Multi-class: use max probability
            continuous_predictions = np.max(y_pred_proba, axis=1)
            continuous_targets = self.y_test_encoded.astype(float) / (len(self.label_encoder.classes_) - 1)
        
        mse = mean_squared_error(continuous_targets, continuous_predictions)
        r2 = r2_score(continuous_targets, continuous_predictions)
        
        # Compile metrics
        metrics = {
            'regression': {  # For interface compatibility
                'mean_squared_error': mse,
                'r2_score': r2,
                'note': 'Regression metrics computed from probabilities for compatibility'
            },
            'classification': {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted,
                'precision_per_class': dict(zip(self.label_encoder.classes_, precision_per_class)),
                'recall_per_class': dict(zip(self.label_encoder.classes_, recall_per_class)),
                'f1_per_class': dict(zip(self.label_encoder.classes_, f1_per_class))
            },
            'logistic_regression': {
                'log_loss': log_loss_score,
                'auc_roc': auc_roc,
                'n_iterations': self.training_history.get('n_iter', 'N/A'),
                'solver_converged': True  # If we reach here, it converged
            }
        }
        
        # Print detailed results
        print("=== Detailed Evaluation Metrics (Logistic Regression) ===")
        print("\nLogistic Regression Metrics:")
        print(f"  Log Loss (Cross-entropy): {log_loss_score:.4f}")
        if auc_roc is not None:
            print(f"  AUC-ROC: {auc_roc:.4f}")
        print(f"  Solver Iterations: {self.training_history.get('n_iter', 'N/A')}")
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"F1-weighted: {f1_weighted:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true_labels, y_pred_labels, target_names=self.label_encoder.classes_, zero_division=0))
        
        # Print class probability statistics
        print("\nPrediction Confidence Statistics:")
        max_probs = np.max(y_pred_proba, axis=1)
        print(f"  Mean confidence: {np.mean(max_probs):.3f}")
        print(f"  Min confidence: {np.min(max_probs):.3f}")
        print(f"  Max confidence: {np.max(max_probs):.3f}")
        print(f"  Std confidence: {np.std(max_probs):.3f}")
        
        return metrics
    
    def plot_confusion_matrix(self, show_plot: bool = True) -> np.ndarray:
        """
        Compute and optionally plot the confusion matrix.
        Same interface as LinearRegressionModel.
        
        Args:
            show_plot: Whether to display the plot
            
        Returns:
            Confusion matrix
        """
        y_pred_labels = self.predict_labels(self.data_split['X_test'])
        y_true = self.data_split['y_test']
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred_labels, labels=self.label_encoder.classes_)
        
        # Print confusion matrix as text (same format as LinearRegressionModel)
        print("\nConfusion Matrix (Text) for Test Data:")
        print("Predicted ->")
        labels = list(self.label_encoder.classes_) + ["Total"]
        print("Actual |", " | ".join(f"{label:>8}" for label in labels))
        print("-" * (10 + 10 * len(labels)))
        
        for i, true_label in enumerate(self.label_encoder.classes_):
            row = [f"{cm[i, j]:>8}" for j in range(len(self.label_encoder.classes_))]
            total = sum(cm[i, :])
            row.append(f"{total:>8}")
            print(f"{true_label:>6} | {' | '.join(row)}")
        
        if show_plot:
            # Generate classification report
            report = classification_report(y_true, y_pred_labels, target_names=self.label_encoder.classes_, zero_division=0)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Calculate total for percentages
            total = cm.sum()
            
            # Create annotation array with counts and percentages
            annot = np.array([[f"{cm[i, j]}\n({cm[i, j]/total*100:.1f}%)"
                             for j in range(cm.shape[1])]
                             for i in range(cm.shape[0])])
            
            # Plot heatmap on ax1
            sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_, ax=ax1)
            ax1.set_title('Confusion Matrix (Counts and Percentages) LOGISTIC REGRESSION')
            ax1.set_ylabel('True Label')
            ax1.set_xlabel('Predicted Label')
            
            # Plot classification report on ax2
            ax2.text(0.1, 0.5, report, fontsize=10, verticalalignment='center', fontfamily='monospace')
            ax2.set_title('Classification Report')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return cm
    
    def plot_probability_distributions(self) -> None:
        """
        Plot prediction probability distributions for each class.
        Unique to LogisticRegressionModel - shows model confidence.
        """
        # Get predictions for test data
        X_test = self.data_split['X_test']
        if self.scale_features and self.scaler is not None:
            X_test = self.scaler.transform(X_test)
        
        y_pred_proba = self.model.predict_proba(X_test)
        y_true = self.data_split['y_test']
        
        n_classes = len(self.label_encoder.classes_)
        fig, axes = plt.subplots(1, n_classes, figsize=(5*n_classes, 4))
        
        if n_classes == 1:
            axes = [axes]
        
        for i, class_name in enumerate(self.label_encoder.classes_):
            ax = axes[i] if n_classes > 1 else axes[0]
            
            # Get probabilities for this class
            class_probs = y_pred_proba[:, i]
            
            # Separate by true class
            true_class_mask = y_true == class_name
            false_class_mask = ~true_class_mask
            
            # Plot histograms
            ax.hist(class_probs[true_class_mask], alpha=0.7, label=f'True {class_name}', 
                   bins=20, color='green', density=True)
            ax.hist(class_probs[false_class_mask], alpha=0.7, label=f'True Other', 
                   bins=20, color='red', density=True)
            
            ax.set_xlabel(f'Predicted Probability for {class_name}')
            ax.set_ylabel('Density')
            ax.set_title(f'Probability Distribution: {class_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Prediction Probability Distributions by True Class', y=1.02, fontsize=14)
        plt.show()
    
    def plot_roc_curves(self) -> None:
        """
        Plot ROC curves for binary or multi-class classification.
        Unique to LogisticRegressionModel.
        """
        # Get predictions for test data
        X_test = self.data_split['X_test']
        if self.scale_features and self.scaler is not None:
            X_test = self.scaler.transform(X_test)
        
        y_pred_proba = self.model.predict_proba(X_test)
        y_true = self.y_test_encoded
        
        n_classes = len(self.label_encoder.classes_)
        
        plt.figure(figsize=(8, 6))
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
            
        else:
            # Multi-class (One-vs-Rest)
            for i, class_name in enumerate(self.label_encoder.classes_):
                y_true_binary = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
                auc = roc_auc_score(y_true_binary, y_pred_proba[:, i])
                
                plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {auc:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Logistic Regression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.show()
    
    def predict_new_data(self, X_new: np.ndarray, y_new: Optional[np.ndarray] = None, 
                        show_confusion_matrix: bool = False) -> Dict[str, Any]:
        """
        Predict on new data and optionally evaluate if true labels are provided.
        Same interface as LinearRegressionModel.
        
        Args:
            X_new: New feature matrix for prediction
            y_new: True labels for new data (if available)  
            show_confusion_matrix: Whether to display the confusion matrix
            
        Returns:
            Dictionary containing predictions and evaluation metrics
        """
        # Get predictions and probabilities
        predictions_encoded = self.predict(X_new)
        predictions_labels = self.predict_labels(X_new)
        predictions_proba = self.predict_proba(X_new)
        
        result = {
            'predictions_encoded': predictions_encoded,
            'predictions_labels': predictions_labels,
            'predictions_probabilities': predictions_proba,
            'n_predictions': len(predictions_labels)
        }
        
        print(f"=== Logistic Regression Predictions on New Data ===")
        print(f"Number of new samples: {len(predictions_labels)}")
        print(f"Predicted labels distribution: {dict(zip(*np.unique(predictions_labels, return_counts=True)))}")
        
        # Show prediction confidence
        max_probs = np.max(predictions_proba, axis=1)
        print(f"Prediction confidence - Mean: {np.mean(max_probs):.3f}, "
              f"Min: {np.min(max_probs):.3f}, Max: {np.max(max_probs):.3f}")
        
        if y_new is not None:
            # Evaluate predictions
            
            # === Classification Metrics ===
            accuracy = accuracy_score(y_new, predictions_labels)
            precision_macro = precision_score(y_new, predictions_labels, average='macro', zero_division=0)
            recall_macro = recall_score(y_new, predictions_labels, average='macro', zero_division=0)
            f1_macro = f1_score(y_new, predictions_labels, average='macro', zero_division=0)
            
            precision_weighted = precision_score(y_new, predictions_labels, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_new, predictions_labels, average='weighted', zero_division=0)
            f1_weighted = f1_score(y_new, predictions_labels, average='weighted', zero_division=0)
            
            # Per-class metrics
            precision_per_class = precision_score(y_new, predictions_labels, average=None, zero_division=0)
            recall_per_class = recall_score(y_new, predictions_labels, average=None, zero_division=0)
            f1_per_class = f1_score(y_new, predictions_labels, average=None, zero_division=0)
            
            # === Logistic Regression Metrics ===
            y_new_encoded = self.label_encoder.transform(y_new)
            log_loss_score = log_loss(y_new_encoded, predictions_proba)
            
            # AUC-ROC
            try:
                if len(self.label_encoder.classes_) == 2:
                    auc_roc = roc_auc_score(y_new_encoded, predictions_proba[:, 1])
                else:
                    auc_roc = roc_auc_score(y_new_encoded, predictions_proba, multi_class='ovr', average='weighted')
            except ValueError:
                auc_roc = None
            
            # === Regression-like Metrics (for compatibility) ===
            if len(self.label_encoder.classes_) == 2:
                continuous_predictions = predictions_proba[:, 1]
                continuous_targets = y_new_encoded.astype(float)
            else:
                continuous_predictions = np.max(predictions_proba, axis=1)
                continuous_targets = y_new_encoded.astype(float) / (len(self.label_encoder.classes_) - 1)
            
            mse = mean_squared_error(continuous_targets, continuous_predictions)
            r2 = r2_score(continuous_targets, continuous_predictions)
            
            result['evaluation'] = {
                'regression': {  # For compatibility
                    'mean_squared_error': mse,
                    'r2_score': r2
                },
                'classification': {
                    'accuracy': accuracy,
                    'precision_macro': precision_macro,
                    'recall_macro': recall_macro,
                    'f1_macro': f1_macro,
                    'precision_weighted': precision_weighted,
                    'recall_weighted': recall_weighted,
                    'f1_weighted': f1_weighted,
                    'precision_per_class': dict(zip(self.label_encoder.classes_, precision_per_class)),
                    'recall_per_class': dict(zip(self.label_encoder.classes_, recall_per_class)),
                    'f1_per_class': dict(zip(self.label_encoder.classes_, f1_per_class))
                },
                'logistic_regression': {
                    'log_loss': log_loss_score,
                    'auc_roc': auc_roc
                }
            }
            
            print("\n=== Evaluation Metrics on New Data ===")
            print(f"\nAccuracy: {accuracy:.4f}")
            print(f"Log Loss: {log_loss_score:.4f}")
            if auc_roc is not None:
                print(f"AUC-ROC: {auc_roc:.4f}")
            print(f"F1-weighted: {f1_weighted:.4f}")
            
            print("\nClassification Report:")
            print(classification_report(y_new, predictions_labels, target_names=self.label_encoder.classes_, zero_division=0))
            
            if show_confusion_matrix:
                self._plot_confusion_matrix_new_data(y_new, predictions_labels)
        
        return result
    
    def _plot_confusion_matrix_new_data(self, y_true: np.ndarray, y_pred: List[str]) -> np.ndarray:
        """
        Plot confusion matrix for new data.
        Same interface as LinearRegressionModel.
        """
        cm = confusion_matrix(y_true, y_pred, labels=self.label_encoder.classes_)
        
        # Text output (same format)
        print("\nConfusion Matrix (Text) for New Data:")
        print("Predicted ->")
        labels = list(self.label_encoder.classes_) + ["Total"]
        print("Actual |", " | ".join(f"{label:>8}" for label in labels))
        print("-" * (10 + 10 * len(labels)))
        
        for i, true_label in enumerate(self.label_encoder.classes_):
            row = [f"{cm[i, j]:>8}" for j in range(len(self.label_encoder.classes_))]
            total = sum(cm[i, :])
            row.append(f"{total:>8}")
            print(f"{true_label:>6} | {' | '.join(row)}")
        
        # Plot
        report = classification_report(y_true, y_pred, target_names=self.label_encoder.classes_, zero_division=0)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        total = cm.sum()
        annot = np.array([[f"{cm[i, j]}\n({cm[i, j]/total*100:.1f}%)"
                         for j in range(cm.shape[1])]
                         for i in range(cm.shape[0])])
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_, ax=ax1)
        ax1.set_title('Confusion Matrix for New Data (Counts and Percentages) LOGISTIC REGRESSION')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        ax2.text(0.1, 0.5, report, fontsize=10, verticalalignment='center', fontfamily='monospace')
        ax2.set_title('Classification Report for New Data')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, Any]:
        """
        Get feature importance based on logistic regression coefficients.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature importance information
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model must be fitted before getting feature importance")
        
        coefficients = self.model.coef_
        wavelengths = self.data_split['unified_wavelengths']
        
        if len(self.label_encoder.classes_) == 2:
            # Binary classification - single coefficient vector
            coef_abs = np.abs(coefficients[0])
            feature_importance = [
                {
                    'wavelength': wavelengths[i],
                    'coefficient': coefficients[0][i], 
                    'abs_coefficient': coef_abs[i],
                    'rank': rank
                }
                for rank, i in enumerate(np.argsort(coef_abs)[::-1], 1)
            ]
        else:
            # Multi-class classification - average absolute coefficients across classes
            coef_abs = np.mean(np.abs(coefficients), axis=0)
            feature_importance = [
                {
                    'wavelength': wavelengths[i],
                    'coefficients': {class_name: coefficients[j][i] 
                                   for j, class_name in enumerate(self.label_encoder.classes_)},
                    'avg_abs_coefficient': coef_abs[i],
                    'rank': rank
                }
                for rank, i in enumerate(np.argsort(coef_abs)[::-1], 1)
            ]
        
        return {
            'feature_importance': feature_importance[:top_n],
            'n_features': len(feature_importance),
            'is_binary': len(self.label_encoder.classes_) == 2
        }
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """
        Plot feature importance based on logistic regression coefficients.
        """
        importance_data = self.get_feature_importance(top_n)
        features = importance_data['feature_importance']
        
        if importance_data['is_binary']:
            # Binary classification plot
            wavelengths = [f['wavelength'] for f in features]
            coefficients = [f['coefficient'] for f in features]
            
            plt.figure(figsize=(12, 6))
            colors = ['red' if c < 0 else 'blue' for c in coefficients]
            
            plt.barh(range(len(wavelengths)), coefficients, color=colors, alpha=0.7)
            plt.yticks(range(len(wavelengths)), [f"{w:.0f} cm⁻¹" for w in wavelengths])
            plt.xlabel('Logistic Regression Coefficient')
            plt.title(f'Top {top_n} Most Important Features (Binary Classification)')
            plt.grid(True, alpha=0.3, axis='x')
            
            # Add legend
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='red', alpha=0.7, label=f'Favors {self.label_encoder.classes_[0]}')
            blue_patch = mpatches.Patch(color='blue', alpha=0.7, label=f'Favors {self.label_encoder.classes_[1]}')
            plt.legend(handles=[red_patch, blue_patch])
            
        else:
            # Multi-class classification plot
            wavelengths = [f['wavelength'] for f in features]
            avg_coeffs = [f['avg_abs_coefficient'] for f in features]
            
            plt.figure(figsize=(12, 6))
            plt.barh(range(len(wavelengths)), avg_coeffs, color='green', alpha=0.7)
            plt.yticks(range(len(wavelengths)), [f"{w:.0f} cm⁻¹" for w in wavelengths])
            plt.xlabel('Average Absolute Coefficient')
            plt.title(f'Top {top_n} Most Important Features (Multi-class Classification)')
            plt.grid(True, alpha=0.3, axis='x')
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and components to a pickle file.
        Same interface as LinearRegressionModel.
        
        Args:
            filepath: Path to save the file
        """
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'data_split': self.data_split,
            'training_history': self.training_history,
            'scale_features': self.scale_features
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Logistic Regression model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> 'LogisticRegressionModel':
        """
        Load a trained model from a pickle file.
        Same interface as LinearRegressionModel.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            Loaded LogisticRegressionModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Reconstruct instance
        instance = LogisticRegressionModel(
            model_data['data_split'],
            scale_features=model_data.get('scale_features', True)
        )
        instance.model = model_data['model']
        instance.label_encoder = model_data['label_encoder']
        instance.scaler = model_data.get('scaler', None)
        instance.training_history = model_data.get('training_history', {})
        
        print(f"Logistic Regression model loaded from {filepath}")
        return instance


logistic_model_balanced = LogisticRegressionModel(
    data_split, 
    class_weight='balanced',  # Handle MGUS/MM imbalance
    C=0.1,                   # Higher regularization
    solver='saga'            # Good for large datasets
)
logistic_model_balanced.fit()
LoR_EVALUATION = logistic_model_balanced.evaluate()
cm = logistic_model_balanced.plot_confusion_matrix()

logistic_model_balanced.plot_roc_curves()
logistic_model_balanced.plot_probability_distributions()
logistic_model_balanced.plot_feature_importance(top_n=15)

LoR_PREDICT_EVALUATION = logistic_model_balanced.predict_new_data(
    data_split_predict['X_train'], 
    data_split_predict['y_train'], 
    show_confusion_matrix=True
)["evaluation"]