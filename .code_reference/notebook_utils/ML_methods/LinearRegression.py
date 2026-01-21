import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import pickle
from notebook_utils.visualize import MLVisualize

class LinearRegressionModel:
    """
    A class for training and evaluating a Linear Regression model on Raman spectroscopy data.
    
    Note: Since the target labels are categorical (e.g., 'MGUS', 'MM'), this model treats
    the problem as regression by encoding labels numerically. For classification tasks,
    consider using LogisticRegression instead.
    """

    def __init__(self, data_split: Dict[str, Any], **kwargs):
        """
        Initialize the Linear Regression model.
        
        Args:
            data_split (dict): Data dictionary from RamanDataSplitter.prepare_data()
            **kwargs: Additional keyword arguments for LinearRegression
        """
        self.data_split = data_split

        # check for missing keys
        required_keys = ['X_train', 'X_test', 'y_train', 'y_test', 'unified_wavelengths']
        missing_keys = [key for key in required_keys if key not in self.data_split]
        if missing_keys:
            raise ValueError(f"Missing keys in data_split: {missing_keys}")

        self.model = LinearRegression(**kwargs)
        self.label_encoder = LabelEncoder()
        
        # Encode labels numerically for regression
        self.y_train_encoded = self.label_encoder.fit_transform(self.data_split['y_train'])
        self.y_test_encoded = self.label_encoder.transform(self.data_split['y_test'])
        
        print(f"Encoded labels: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the test set with detailed classification metrics.
        
        Returns:
            Dict[str, Any]: Evaluation metrics including regression and classification metrics
        """
        y_pred_encoded = self.predict(self.data_split['X_test'])
        y_pred_labels = self.predict_labels(self.data_split['X_test'])
        y_true_labels = self.data_split['y_test']
        
        # Regression metrics
        mse = mean_squared_error(self.y_test_encoded, y_pred_encoded)
        r2 = r2_score(self.y_test_encoded, y_pred_encoded)
        
        # Classification metrics
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
        
        metrics = {
            'regression': {
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
            }
        }
        
        print("=== Detailed Evaluation Metrics ===")
        print("\nRegression Metrics:")
        print(f"  Mean Squared Error: {mse:.4f}")
        print(f"  R² Score: {r2:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true_labels, y_pred_labels, target_names=self.label_encoder.classes_, zero_division=0))
        
        return metrics
    
    def fit(self) -> None:
        """
        Fit the Linear Regression model to the training data.
        """
        print("Fitting Linear Regression model...")
        self.model.fit(self.data_split['X_train'], self.y_train_encoded)
        print("Model fitted successfully.")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predicted encoded labels
        """
        return self.model.predict(X)
    
    def predict_labels(self, X: np.ndarray) -> List[str]:
        """
        Make predictions and decode to original labels.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            List[str]: Predicted original labels
        """
        predictions_encoded = self.predict(X)
        # Round to nearest integer for label decoding
        predictions_rounded = np.round(predictions_encoded).astype(int)
        # Clip to valid label range
        predictions_rounded = np.clip(predictions_rounded, 0, len(self.label_encoder.classes_) - 1)
        return self.label_encoder.inverse_transform(predictions_rounded)
    
    def plot_confusion_matrix(self, show_plot: bool = True) -> np.ndarray:
        """
        Compute and optionally plot the confusion matrix using sklearn's official implementation.
        
        Args:
            show_plot (bool): Whether to display the plot
            
        Returns:
            np.ndarray: Confusion matrix
        """
        y_pred_labels = self.predict_labels(self.data_split['X_test'])
        y_true = self.data_split['y_test']
        
        # Compute confusion matrix with original labels using sklearn
        cm = confusion_matrix(y_true, y_pred_labels, labels=self.label_encoder.classes_)
        
        # Print confusion matrix as text
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
            ax1.set_title('Confusion Matrix (Counts and Percentages) LINEAR REGRESSION')
            ax1.set_ylabel('True Label')
            ax1.set_xlabel('Predicted Label')
            
            # Plot classification report on ax2
            ax2.text(0.1, 0.5, report, fontsize=10, verticalalignment='center', fontfamily='monospace')
            ax2.set_title('Classification Report')
            ax2.axis('off')  # Hide axes for text
            
            plt.tight_layout()
            plt.show()
        
        return cm
    
    def predict_new_data(self, X_new: np.ndarray, y_new: Optional[np.ndarray] = None, show_confusion_matrix: bool = False) -> Dict[str, Any]:
        """
        Predict on new data and optionally evaluate if true labels are provided.
        
        Args:
            X_new (np.ndarray): New feature matrix for prediction
            y_new (np.ndarray, optional): True labels for new data (if available)
            show_confusion_matrix (bool): Whether to display the confusion matrix if y_new is provided
            
        Returns:
            Dict[str, Any]: Dictionary containing predictions and evaluation metrics (if y_new provided)
        """
        predictions_encoded = self.predict(X_new)
        predictions_labels = self.predict_labels(X_new)
        
        result = {
            'predictions_encoded': predictions_encoded,
            'predictions_labels': predictions_labels,
            'n_predictions': len(predictions_labels)
        }
        
        print(f"=== Predictions on New Data ===")
        print(f"Number of new samples: {len(predictions_labels)}")
        print(f"Predicted labels distribution: {dict(zip(*np.unique(predictions_labels, return_counts=True)))}")
        
        if y_new is not None:
            # Evaluate predictions if true labels are provided
            mse = mean_squared_error(self.label_encoder.transform(y_new), predictions_encoded)
            r2 = r2_score(self.label_encoder.transform(y_new), predictions_encoded)
            
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
            
            result['evaluation'] = {
                'regression': {
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
                }
            }
            
            print("\n=== Evaluation Metrics on New Data ===")
            print("\nRegression Metrics:")
            print(f"  Mean Squared Error: {mse:.4f}")
            print(f"  R² Score: {r2:.4f}")
            
            print("\nClassification Report:")
            print(classification_report(y_new, predictions_labels, target_names=self.label_encoder.classes_, zero_division=0))
            
            if show_confusion_matrix:
                self._plot_confusion_matrix_new_data(y_new, predictions_labels)
        
        return result

    def _plot_confusion_matrix_new_data(self, y_true: np.ndarray, y_pred: List[str], show_plot: bool = True) -> np.ndarray:
        """
        Compute and optionally plot the confusion matrix for new data.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (List[str]): Predicted labels
            show_plot (bool): Whether to display the plot
            
        Returns:
            np.ndarray: Confusion matrix
        """
        # Compute confusion matrix with original labels using sklearn
        cm = confusion_matrix(y_true, y_pred, labels=self.label_encoder.classes_)
        
        # Print confusion matrix as text
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
        
        if show_plot:
            # Generate classification report
            report = classification_report(y_true, y_pred, target_names=self.label_encoder.classes_, zero_division=0)
            
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
            ax1.set_title('Confusion Matrix for New Data (Counts and Percentages) LINEAR REGRESSION')
            ax1.set_ylabel('True Label')
            ax1.set_xlabel('Predicted Label')
            
            # Plot classification report on ax2
            ax2.text(0.1, 0.5, report, fontsize=10, verticalalignment='center', fontfamily='monospace')
            ax2.set_title('Classification Report for New Data')
            ax2.axis('off')  # Hide axes for text
            
            plt.tight_layout()
            plt.show()
        
        return cm
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and encoder to a pickle file.
        
        Args:
            filepath (str): Path to save the file
        """
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'data_split': self.data_split
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> 'LinearRegressionModel':
        """
        Load a trained model from a pickle file.
        
        Args:
            filepath (str): Path to the saved model file
            
        Returns:
            LinearRegressionModel: Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = LinearRegressionModel(model_data['data_split'])
        instance.model = model_data['model']
        instance.label_encoder = model_data['label_encoder']
        return instance
    

## below is how this class is used in the notebook
# Train and evaluate the Linear Regression model
linear_regression_model_original = LinearRegressionModel(data_split)
linear_regression_model_original.fit()
# need to show confusion matrix here because evaluate() does not show it
cm = linear_regression_model_original.plot_confusion_matrix()
linear_regression_visual = MLVisualize(model=linear_regression_model_original.model, data_split=linear_regression_model_original.data_split, label_encoder=linear_regression_model_original.label_encoder)
# decision_boundary
_ = linear_regression_visual.plot_decision_boundary_2d(dim_reduction="pca")
# predict with external data
LR_PREDICT_EVALUATION = linear_regression_model_original.predict_new_data(data_split_predict['X_train'], data_split_predict['y_train'], show_confusion_matrix=True)["evaluation"]