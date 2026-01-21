from sklearn.feature_selection import mutual_info_classif
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

class MutualInformationFeatureSelector:
    """
    A class for feature selection using Mutual Information (MI) based on benefit-cost analysis.
    
    Benefit: MI between features and disease labels (high is good).
    Cost: MI between features and batch labels (high is bad).
    Combined score: benefit - cost.
    Selects top N features with highest combined scores.
    """
    
    def __init__(self, data_split: Dict[str, Any], n_features_to_select: int = 200, random_state: int = 42):
        """
        Initialize the feature selector.
        
        Args:
            data_split (dict): Data dictionary from RamanDataPreparer.prepare_data()
            n_features_to_select (int): Number of top features to select (default 200)
            random_state (int): Random seed for reproducibility in MI calculation
        """
        self.data_split = data_split
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state
        
        # Check for required keys
        required_keys = ['X_train', 'y_train', 'batch_train', 'unified_wavelengths']
        missing_keys = [key for key in required_keys if key not in self.data_split]
        if missing_keys:
            raise ValueError(f"Missing keys in data_split: {missing_keys}")
        
        self.n_features = self.data_split['X_train'].shape[1]
        self.selected_indices = None
        self.mi_disease = None
        self.mi_batch = None
        self.combined_scores = None
        
    def fit(self) -> None:
        """
        Fit the feature selector by calculating MI scores and selecting top features.
        """
        print("Calculating Mutual Information for disease labels...")
        self.mi_disease = mutual_info_classif(
            self.data_split['X_train'], 
            self.data_split['y_train'], 
            random_state=self.random_state
        )
        
        print("Calculating Mutual Information for batch labels...")
        self.mi_batch = mutual_info_classif(
            self.data_split['X_train'], 
            self.data_split['batch_train'], 
            random_state=self.random_state
        )
        
        # Calculate combined scores
        self.combined_scores = self.mi_disease - self.mi_batch
        
        # Select top N features
        self.selected_indices = np.argsort(self.combined_scores)[::-1][:self.n_features_to_select]
        
        print(f"Selected {self.n_features_to_select} features with highest benefit-cost scores.")
        print(f"Top 5 selected wavelengths: {self.data_split['unified_wavelengths'][self.selected_indices[:5]]}")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the feature matrix to include only selected features.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Transformed feature matrix with selected features
        """
        if self.selected_indices is None:
            raise ValueError("Feature selector must be fitted before transforming data.")
        return X[:, self.selected_indices]
    
    def get_selected_features(self) -> Dict[str, Any]:
        """
        Get information about selected features.
        
        Returns:
            Dict containing selected indices, wavelengths, and scores
        """
        if self.selected_indices is None:
            raise ValueError("Feature selector must be fitted before getting selected features.")
        
        return {
            'indices': self.selected_indices,
            'wavelengths': self.data_split['unified_wavelengths'][self.selected_indices],
            'mi_disease': self.mi_disease[self.selected_indices],
            'mi_batch': self.mi_batch[self.selected_indices],
            'combined_scores': self.combined_scores[self.selected_indices]
        }
    
    def plot_scores(self, top_n: int = 50, show_plot: bool = True) -> None:
        """
        Plot the top N combined scores.
        
        Args:
            top_n (int): Number of top features to plot
            show_plot (bool): Whether to display the plot
        """
        if self.combined_scores is None:
            raise ValueError("Feature selector must be fitted before plotting scores.")
        
        # Get top N indices
        top_indices = np.argsort(self.combined_scores)[::-1][:top_n]
        top_wavelengths = self.data_split['unified_wavelengths'][top_indices]
        top_scores = self.combined_scores[top_indices]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_scores)), top_scores, color='skyblue')
        plt.xlabel('Feature Rank')
        plt.ylabel('Combined Score (MI Disease - MI Batch)')
        plt.title(f'Top {top_n} Features by Benefit-Cost Score')
        plt.xticks(range(len(top_scores)), [f'{w:.0f}' for w in top_wavelengths], rotation=45)
        plt.tight_layout()
        
        if show_plot:
            plt.show()


# Fit the feature selector ONLY on the original training data (data_split)
feature_selector_ds = MutualInformationFeatureSelector(data_split, n_features_to_select=200, random_state=42)
feature_selector_ds.fit()

# Get selected features info (from the training-based selector)
print("Selected wavelengths for ds:", feature_selector_ds.get_selected_features()['wavelengths'][:10])

# Transform training and test data using the training-based selector
X_train_selected = feature_selector_ds.transform(data_split['X_train'])
X_test_selected = feature_selector_ds.transform(data_split['X_test'])

# Transform NEW data using the SAME selector (from training)
X_new_selected = feature_selector_ds.transform(data_split_predict['X_train'])  # Use ds, not dsp
y_new = data_split_predict['y_train']  # Ensure labels are mapped correctly (e.g., MMnew -> MM)

# Create the feature-selected data split for training
fs_data_split = {
    'X_train': X_train_selected,
    'y_train': data_split['y_train'],
    'batch_train': data_split['batch_train'],
    'X_test': X_test_selected,
    'y_test': data_split['y_test'],
    'batch_test': data_split['batch_test'],
    'unified_wavelengths': data_split['unified_wavelengths']
}

# Now, train your model on fs_data_split and predict on X_new_selected, y_new
# Example: model.predict_new_data(X_new_selected, y_new, show_confusion_matrix=True)