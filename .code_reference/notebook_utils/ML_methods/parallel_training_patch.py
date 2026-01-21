"""
Parallel Training Support for MultiNetworkEnsembleSystem

This module provides helper functions for parallel execution of stochastic training runs.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from typing import Tuple, Any


def train_single_run_sklearn(X_scaled: np.ndarray,
                             y: np.ndarray,
                             n_hidden: int,
                             max_iter: int,
                             random_state: int,
                             run_idx: int,
                             n_classes: int = 2) -> Tuple[np.ndarray, Any]:
    """
    Train a single stochastic run (for parallel execution) - sklearn backend.
    
    Parameters:
    -----------
    X_scaled : array, shape (n_samples, n_features)
        Scaled feature matrix
    y : array, shape (n_samples,)
        Target labels
    n_hidden : int
        Number of hidden neurons
    max_iter : int
        Maximum training iterations
    random_state : int
        Base random state
    run_idx : int
        Run index (added to random_state for uniqueness)
    n_classes : int
        Number of classes for output layer
    
    Returns:
    --------
    prob_matrix : array, shape (n_samples, n_classes)
        Probability predictions for all classes (LOO cross-validation)
    network : MLPClassifier
        Trained network on full dataset
    """
    n_samples = X_scaled.shape[0]
    prob_matrix = np.zeros((n_samples, n_classes))
    
    # Create network with different random seed
    network = MLPClassifier(
        hidden_layer_sizes=(n_hidden,),
        activation='logistic',
        solver='lbfgs',
        max_iter=max_iter,
        random_state=random_state + run_idx,
        alpha=0.0001,
        warm_start=False
    )
    
    # Leave-One-Out Cross-Validation
    for i in range(n_samples):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[i] = False
        
        X_train_loo = X_scaled[train_mask]
        y_train_loo = y[train_mask]
        X_test_loo = X_scaled[i:i+1]
        
        network.fit(X_train_loo, y_train_loo)
        probs = network.predict_proba(X_test_loo)[0, :]  # Full probability distribution
        prob_matrix[i, :] = probs
    
    # Train final network on full dataset
    network.fit(X_scaled, y)
    
    return prob_matrix, network
