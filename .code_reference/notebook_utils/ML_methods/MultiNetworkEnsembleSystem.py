"""
Multi-Network Ensemble System for Raman Spectroscopy Classification

Based on:
Kothari et al., Scientific Reports (2021): "Raman spectroscopy and artificial 
intelligence to predict the Bayesian probability of breast cancer"

This implementation provides a complete multi-network Bayesian ensemble system
with stochastic neural networks for probabilistic classification with uncertainty
quantification.

Architecture:
- Three neural networks (FPHW, FP, HW) trained with different feature sets
- Stochastic training with multiple random initializations
- Bayesian probability aggregation
- Variance analysis for uncertainty quantification (VRA, VER)
- Clinical decision support with boundary detection

Author: AI Assistant
Date: December 16, 2025
"""

import numpy as np
import pandas as pd
import time
import sys
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, roc_curve,
    classification_report, log_loss
)
from scipy.signal import savgol_filter
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
warnings.filterwarnings('ignore')

# tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# PyTorch imports (optional - will use sklearn if not available)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

# ==============================================================================
# PYTORCH NEURAL NETWORK (Optional Backend)
# ==============================================================================

if TORCH_AVAILABLE:
    class TorchMLP(nn.Module):
        """PyTorch MLP for Multi-Network Ensemble (GPU-accelerated)."""
        
        def __init__(self, input_size: int, hidden_size: int = 3, output_size: int = 2):
            super(TorchMLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.sigmoid = nn.Sigmoid()  # Logistic activation (matches sklearn)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.sigmoid(x)
            x = self.fc2(x)
            x = self.softmax(x)
            return x

# ==============================================================================
# MULTI-NETWORK ENSEMBLE SYSTEM CLASS
# ==============================================================================


class MultiNetworkEnsembleSystem:
    """
    Complete Multi-Network Ensemble System for Raman Spectral Classification.
    
    This system combines three neural networks trained on different feature sets
    (FPHW, FP, HW) with Bayesian probability aggregation and variance analysis
    for robust classification with uncertainty quantification.
    
    Key Features:
    - Stochastic neural network training (multiple random initializations)
    - Three-network architecture for multi-scale biochemical analysis
    - Bayesian probability estimation with credible intervals
    - Intra-network variance (VRA) and Inter-network variance (VER)
    - Boundary detection for clinical decision support
    - Optional k-means autonomous labeling
    
    Usage:
    >>> biomarkers = {...}  # Define biomarker bands
    >>> system = MultiNetworkEnsembleSystem(biomarker_bands=biomarkers)
    >>> system.fit(data_split, n_stochastic_runs=25)
    >>> results = system.evaluate()
    >>> predictions = system.predict_new_data(X_new)
    """
    
    def __init__(self, 
                 biomarker_bands: Dict[str, List[float]],
                 n_hidden: int = 3,
                 max_iter: int = 500,
                 random_state: int = 42,
                 use_kmeans_labeling: bool = False,
                 use_torch: bool = False,
                 device: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize the Multi-Network Ensemble System.
        
        Parameters:
        -----------
        biomarker_bands : dict
            Dictionary defining the biomarker bands for each network.
            Example:
            {
                'FP': [796, 828, 1048, 1300, 1437, 1654],  # Fingerprint
                'HW': [2853, 2896, 2937],                   # High-wavenumber
                'FPHW': [796, 828, 1048, 1300, 1437, 1654, 2853, 2896, 2937]
            }
        n_hidden : int, default=3
            Number of hidden layer neurons (fixed at 3 in original paper)
        max_iter : int, default=500
            Maximum iterations for neural network training
        random_state : int, default=42
            Random seed for reproducibility
        use_kmeans_labeling : bool, default=False
            If True, use k-means autonomous labeling instead of provided labels
        use_torch : bool, default=False
            If True, use PyTorch backend (GPU-accelerated if available)
            If False, use scikit-learn MLPClassifier (CPU-only)
        device : str or torch.device, optional
            PyTorch device ('cuda', 'mps', 'cpu', or torch.device object)
            Only used if use_torch=True. Auto-detected if not specified.
        verbose : bool, default=True
            Print progress information
        """
        self.biomarker_bands = biomarker_bands
        self.n_hidden = n_hidden
        self.max_iter = max_iter
        self.random_state = random_state
        self.use_kmeans_labeling = use_kmeans_labeling
        self.verbose = verbose
        
        # Backend configuration
        if use_torch and not TORCH_AVAILABLE:
            if self.verbose:
                print("⚠ PyTorch not available. Falling back to scikit-learn backend.")
            self.use_torch = False
            self.device = None
        else:
            self.use_torch = use_torch
            
            if self.use_torch:
                # Configure device
                if device is None:
                    # Auto-detect best device
                    if torch.cuda.is_available():
                        self.device = torch.device('cuda')
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        self.device = torch.device('mps')
                    else:
                        self.device = torch.device('cpu')
                elif isinstance(device, str):
                    self.device = torch.device(device)
                else:
                    self.device = device
                
                # Set random seed for PyTorch
                torch.manual_seed(self.random_state)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.random_state)
            else:
                self.device = None
        
        # Validate biomarker bands
        required_networks = ['FP', 'HW', 'FPHW']
        for net in required_networks:
            if net not in biomarker_bands:
                raise ValueError(f"Missing required network '{net}' in biomarker_bands")
        
        # Initialize components
        self.networks = {}  # Will store trained networks
        self.scalers = {}   # Feature scalers for each network
        self.label_encoders = {}  # Label encoders for PyTorch (string -> int)
        self.training_results = {}  # Store stochastic training results
        self.variance_results = {}  # Store VRA and VER results
        
        # Store data information
        self.data_split = None
        self.wavelengths = None
        self.feature_matrices = {}
        self.is_fitted = False
        
        if self.verbose:
            print("=" * 70)
            print("MULTI-NETWORK ENSEMBLE SYSTEM INITIALIZED")
            print("=" * 70)
            backend_info = f"PyTorch (device={self.device})" if self.use_torch else "scikit-learn (CPU)"
            print(f"Backend: {backend_info}")
            print(f"\nNetwork configurations:")
            print(f"  - NN_FPHW: {len(biomarker_bands['FPHW'])} features → {n_hidden} hidden → 2 output")
            print(f"  - NN_FP:   {len(biomarker_bands['FP'])} features → {n_hidden} hidden → 2 output")
            print(f"  - NN_HW:   {len(biomarker_bands['HW'])} features → {n_hidden} hidden → 2 output")
            print(f"Hidden layer activation: logistic (sigmoid)")
            print(f"Output layer activation: softmax")
            print(f"Solver: lbfgs (Limited-memory BFGS)")
            print(f"K-means autonomous labeling: {use_kmeans_labeling}")
    
    # ============================================================================
    # FEATURE EXTRACTION METHODS
    # ============================================================================
    
    def extract_band_features(self, 
                             X: np.ndarray, 
                             wavelengths: np.ndarray, 
                             bands: List[float]) -> np.ndarray:
        """
        Extract specific band intensities from full spectra.
        
        Parameters:
        -----------
        X : array, shape (n_samples, n_wavelengths)
            Full spectrum matrix
        wavelengths : array, shape (n_wavelengths,)
            Wavelength/wavenumber axis
        bands : list of float
            Target band positions to extract
        
        Returns:
        --------
        features : array, shape (n_samples, len(bands))
            Extracted band intensities
        """
        features = np.zeros((X.shape[0], len(bands)))
        
        for i, band in enumerate(bands):
            # Find nearest wavelength index
            idx = np.argmin(np.abs(wavelengths - band))
            features[:, i] = X[:, idx]
        
        return features
    
    def prepare_feature_matrices(self, 
                                 X: np.ndarray, 
                                 wavelengths: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Prepare all three feature matrices (FPHW, FP, HW) from full spectra.
        Validates wavelength ranges and filters out-of-range bands.
        
        Parameters:
        -----------
        X : array, shape (n_samples, n_wavelengths)
            Full spectrum matrix
        wavelengths : array, shape (n_wavelengths,)
            Wavelength axis
        
        Returns:
        --------
        feature_matrices : dict
            Dictionary with keys 'FPHW', 'FP', 'HW' containing extracted features
        """
        feature_matrices = {}
        
        # Get wavelength range
        wl_min, wl_max = wavelengths.min(), wavelengths.max()
        
        if self.verbose:
            print(f"\n[Wavelength Validation]")
            print(f"  Data range: {wl_min:.1f} - {wl_max:.1f} cm\u207b\u00b9")
        
        # Validate and filter bands for each network
        for network_name, bands in self.biomarker_bands.items():
            # Check which bands are within range
            valid_bands = [b for b in bands if wl_min <= b <= wl_max]
            invalid_bands = [b for b in bands if b < wl_min or b > wl_max]
            
            if invalid_bands:
                if self.verbose:
                    print(f"  ⚠ {network_name}: Excluding {len(invalid_bands)} out-of-range bands: {invalid_bands}")
            
            if len(valid_bands) == 0:
                if self.verbose:
                    print(f"  ❌ {network_name}: No valid bands! Skipping this network.")
                # Skip this network
                continue
            
            # Extract features using only valid bands
            features = self.extract_band_features(X, wavelengths, valid_bands)
            feature_matrices[network_name] = features
            
            if self.verbose:
                print(f"  ✅ {network_name}: {features.shape[1]} bands extracted (shape: {features.shape})")
        
        # Special handling: If HW is missing, remove FPHW too
        if 'HW' not in feature_matrices and 'FPHW' in feature_matrices:
            if self.verbose:
                print(f"  ⚠ Removing FPHW since HW bands are not available")
            del feature_matrices['FPHW']
        
        if not feature_matrices:
            raise ValueError("No valid feature matrices could be extracted! Check wavelength range.")
        
        return feature_matrices
    
    # ============================================================================
    # K-MEANS AUTONOMOUS LABELING
    # ============================================================================
    
    def kmeans_autonomous_labeling(self, 
                                   X_FPHW: np.ndarray,
                                   X_FP: np.ndarray,
                                   X_HW: np.ndarray,
                                   n_clusters: int = 2) -> Dict[str, np.ndarray]:
        """
        Perform k-means clustering for autonomous labeling.
        
        Critical: Orient clusters so that tumor=1, healthy=0 based on:
        - Higher FP features (DNA/RNA increase in tumor)
        - Lower HW features (lipid loss in tumor)
        
        Parameters:
        -----------
        X_FPHW, X_FP, X_HW : arrays
            Feature matrices for each network
        n_clusters : int, default=2
            Number of clusters (typically 2 for binary classification)
        
        Returns:
        --------
        labels : dict
            Dictionary with keys 'FPHW', 'FP', 'HW' containing cluster labels
        """
        labels = {}
        
        for network_name, X in [('FPHW', X_FPHW), ('FP', X_FP), ('HW', X_HW)]:
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, 
                          n_init=50, 
                          random_state=self.random_state)
            cluster_labels = kmeans.fit_predict(X)
            
            # Orient clusters: tumor should have higher FP, lower HW
            if network_name == 'FP':
                # Tumor cluster has higher mean FP features
                mean_0 = np.mean(X[cluster_labels == 0])
                mean_1 = np.mean(X[cluster_labels == 1])
                if mean_0 > mean_1:
                    # Swap: cluster 0 should be healthy (lower values)
                    cluster_labels = 1 - cluster_labels
            
            elif network_name == 'HW':
                # Tumor cluster has lower mean HW features (lipid loss)
                mean_0 = np.mean(X[cluster_labels == 0])
                mean_1 = np.mean(X[cluster_labels == 1])
                if mean_0 < mean_1:
                    # Swap: cluster 0 should be healthy (higher values)
                    cluster_labels = 1 - cluster_labels
            
            else:  # FPHW
                # Use combined logic: FP higher + HW lower = tumor
                fp_mean_0 = np.mean(X[cluster_labels == 0, :6])  # First 6 are FP
                fp_mean_1 = np.mean(X[cluster_labels == 1, :6])
                if fp_mean_0 > fp_mean_1:
                    cluster_labels = 1 - cluster_labels
            
            labels[network_name] = cluster_labels
            
            if self.verbose:
                unique, counts = np.unique(cluster_labels, return_counts=True)
                print(f"K-means {network_name}: {dict(zip(unique, counts))}")
        
        return labels
    
    # ============================================================================
    # STOCHASTIC NEURAL NETWORK TRAINING
    # ============================================================================
    
    @staticmethod
    def _train_single_run_sklearn(X_scaled: np.ndarray,
                                  y: np.ndarray,
                                  n_hidden: int,
                                  max_iter: int,
                                  random_state: int,
                                  run_idx: int,
                                  verbose: bool = False) -> Tuple[np.ndarray, Any]:
        """
        Train a single stochastic run (for parallel execution).
        
        Returns:
        --------
        prob_vector : array, shape (n_samples,)
            Probability predictions for this run
        network : MLPClassifier
            Trained network
        """
        from sklearn.neural_network import MLPClassifier
        
        n_samples = X_scaled.shape[0]
        prob_vector = np.zeros(n_samples)
        
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
            prob = network.predict_proba(X_test_loo)[0, 1]
            prob_vector[i] = prob
        
        # Train final network on full dataset
        network.fit(X_scaled, y)
        
        return prob_vector, network
    
    def train_single_network_stochastic(self,
                                       X: np.ndarray,
                                       y: np.ndarray,
                                       network_name: str,
                                       n_stochastic_runs: int = 25,
                                       n_jobs: int = 1) -> Dict[str, Any]:
        """
        Train a single neural network with multiple stochastic runs.
        
        Each run uses:
        - Different random weight initialization
        - Leave-One-Out Cross-Validation
        - Probability predictions stored
        
        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            Feature matrix
        y : array, shape (n_samples,)
            Target labels (0=healthy, 1=tumor)
        network_name : str
            Network identifier ('FPHW', 'FP', or 'HW')
        n_stochastic_runs : int
            Number of random initialization runs
        
        Returns:
        --------
        results : dict
            {
                'prob_matrix': array (n_samples, n_runs),
                'p_mean': array (n_samples,),
                'p_std': array (n_samples,),
                'vra': array (n_samples,),  # Intra-network variance
                'networks': list of trained MLPClassifier instances
            }
        """
        n_samples = X.shape[0]
        prob_matrix = np.zeros((n_samples, n_stochastic_runs))
        trained_networks = []
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[network_name] = scaler
        
        if self.verbose:
            print(f"\n  Training NN_{network_name}: {n_stochastic_runs} runs × {n_samples} LOO iterations = {n_samples * n_stochastic_runs:,} total fits")
            print(f"  Features: {X.shape[1]}, Backend: scikit-learn")
        
        start_time_total = time.time()
        
        # Determine parallelization strategy
        n_workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        use_parallel = (n_workers > 1 and n_stochastic_runs > 1)
        
        if use_parallel and self.verbose:
            print(f"  \ud83d\ude80 Parallel execution: {n_workers} workers")
        elif self.verbose:
            print(f"  \ud83d\udd04 Sequential execution")
        
        print()  # Blank line before progress
        
        # PARALLEL EXECUTION (CPU-bound sklearn training)
        if use_parallel:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit all runs
                futures = {
                    executor.submit(
                        self._train_single_run_sklearn,
                        X_scaled, y, self.n_hidden, self.max_iter,
                        self.random_state, run_idx, False
                    ): run_idx
                    for run_idx in range(n_stochastic_runs)
                }
                
                # Collect results with progress tracking
                if TQDM_AVAILABLE and self.verbose:
                    futures_iterator = tqdm(
                        as_completed(futures),
                        total=n_stochastic_runs,
                        desc=f"  [{network_name}] Runs",
                        unit="run",
                        leave=True,
                        ncols=100
                    )
                else:
                    futures_iterator = as_completed(futures)
                
                for future in futures_iterator:
                    run_idx = futures[future]
                    prob_vector, network = future.result()
                    prob_matrix[:, run_idx] = prob_vector
                    trained_networks.append(network)
        
        # SEQUENTIAL EXECUTION (original code)
        else:
            # Train each run with individual progress bars
            for run_idx in range(n_stochastic_runs):
                run_start = time.time()
                
                # Create network with different random seed
                network = MLPClassifier(
                    hidden_layer_sizes=(self.n_hidden,),
                    activation='logistic',  # Sigmoid for hidden layer
                    solver='lbfgs',
                    max_iter=self.max_iter,
                    random_state=self.random_state + run_idx,  # Different seed each run
                    alpha=0.0001,  # L2 regularization
                    warm_start=False  # Reset weights each run
                )
                
                # Leave-One-Out Cross-Validation with progress bar
                loo_start = time.time()
                
                # Create progress bar for LOO iterations within this run
                if TQDM_AVAILABLE and self.verbose:
                    loo_iterator = tqdm(
                        range(n_samples),
                        desc=f"  [{network_name}] Run {run_idx+1}/{n_stochastic_runs}",
                        unit="sample",
                        leave=True,
                        ncols=100,
                        position=0
                    )
                else:
                    loo_iterator = range(n_samples)
                
                for i in loo_iterator:
                    # Create training set excluding sample i
                    train_mask = np.ones(n_samples, dtype=bool)
                    train_mask[i] = False
                    
                    X_train_loo = X_scaled[train_mask]
                    y_train_loo = y[train_mask]
                    X_test_loo = X_scaled[i:i+1]
                    
                    # Train on LOO training set
                    network.fit(X_train_loo, y_train_loo)
                    
                    # Predict probability for held-out sample
                    prob = network.predict_proba(X_test_loo)[0, 1]  # P(tumor)
                    prob_matrix[i, run_idx] = prob
                
                loo_time = time.time() - loo_start
                
                # Train final network on full dataset for this run
                network.fit(X_scaled, y)
                trained_networks.append(network)
                
                run_time = time.time() - run_start
                
                # Calculate run statistics
                prob_run = prob_matrix[:, run_idx]
                run_mean = np.mean(prob_run)
                run_std = np.std(prob_run)
                
                # Print summary for this run
                if self.verbose:
                    print(f"    ✓ Run {run_idx+1} completed: {run_time:.1f}s | P̄={run_mean:.3f}±{run_std:.3f}\n")
        
        # Compute Bayesian statistics
        p_mean = np.mean(prob_matrix, axis=1)
        p_std = np.std(prob_matrix, axis=1)
        vra = np.var(prob_matrix, axis=1)  # Intra-network variance
        
        total_time = time.time() - start_time_total
        
        results = {
            'prob_matrix': prob_matrix,
            'p_mean': p_mean,
            'p_std': p_std,
            'vra': vra,
            'networks': trained_networks
        }
        
        if self.verbose:
            print(f"\n  ═══ NN_{network_name} Training Complete ═══")
            print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"  Avg time per run: {total_time/n_stochastic_runs:.1f}s")
            print(f"  Mean probability: {np.mean(p_mean):.3f} ± {np.mean(p_std):.3f}")
            print(f"  Mean VRA (intra-network): {np.mean(vra):.4f}")
            print(f"  VRA range: [{np.min(vra):.4f}, {np.max(vra):.4f}]")
            print(f"  Samples with low VRA (<0.5): {np.sum(vra < 0.5)} ({np.sum(vra < 0.5)/n_samples*100:.1f}%)")
            print(f"  Samples with high VRA (>1.5): {np.sum(vra > 1.5)} ({np.sum(vra > 1.5)/n_samples*100:.1f}%)\n")
        
        return results
    
    def train_single_network_stochastic_torch(self,
                                              X: np.ndarray,
                                              y: np.ndarray,
                                              network_name: str,
                                              n_stochastic_runs: int = 25,
                                              n_jobs: int = 1) -> Dict[str, Any]:
        """
        Train a single neural network with PyTorch backend (GPU-accelerated).
        
        Uses same LOO cross-validation approach as sklearn version but with PyTorch.
        
        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            Feature matrix
        y : array, shape (n_samples,)
            Target labels (can be strings or numeric)
        network_name : str
            Network identifier ('FPHW', 'FP', or 'HW')
        n_stochastic_runs : int
            Number of random initialization runs
        
        Returns:
        --------
        results : dict with same structure as sklearn version
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        trained_networks = []
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[network_name] = scaler
        
        # Encode labels if they are strings
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        self.label_encoders[network_name] = label_encoder
        
        # Detect number of classes dynamically
        n_classes = len(label_encoder.classes_)
        
        # Store full probability matrix: (n_samples, n_stochastic_runs, n_classes)
        # This allows any number of classes (2, 3, 4, etc.)
        prob_matrix = np.zeros((n_samples, n_stochastic_runs, n_classes))
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y_encoded).to(self.device)
        
        if self.verbose:
            print(f"\n  Training NN_{network_name}: {n_stochastic_runs} runs × {n_samples} LOO iterations = {n_samples * n_stochastic_runs:,} total fits")
            print(f"  Features: {X.shape[1]}, Classes: {n_classes} {list(label_encoder.classes_)}, Backend: PyTorch/{self.device}")
            
            # GPU memory safety check
            if n_jobs > 1 and self.device.type == 'cuda':
                print(f"  ⚠️  Warning: n_jobs={n_jobs} ignored for GPU training (using sequential n_jobs=1 to avoid OOM)")
                n_jobs = 1  # Force sequential for GPU
            elif n_jobs > 1:
                print(f"  ⚠️  Note: Parallel PyTorch training not yet implemented (using sequential)")
                n_jobs = 1
        
        # Force sequential for PyTorch (parallel implementation can be added later)
        n_jobs = 1
        print()  # Blank line
        
        start_time_total = time.time()
        
        # Train each run with individual progress bars
        for run_idx in range(n_stochastic_runs):
            run_start = time.time()
            
            # Create network with different random seed
            torch.manual_seed(self.random_state + run_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state + run_idx)
            
            network = TorchMLP(
                input_size=n_features,
                hidden_size=self.n_hidden,
                output_size=len(np.unique(y))
            ).to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.LBFGS(network.parameters(), max_iter=20)
            
            # Leave-One-Out Cross-Validation with progress bar
            loo_start = time.time()
            
            # Create progress bar for LOO iterations within this run
            if TQDM_AVAILABLE and self.verbose:
                loo_iterator = tqdm(
                    range(n_samples),
                    desc=f"  [{network_name}] Run {run_idx+1}/{n_stochastic_runs}",
                    unit="sample",
                    leave=True,
                    ncols=100,
                    position=0
                )
            else:
                loo_iterator = range(n_samples)
            
            for i in loo_iterator:
                # Create training set excluding sample i
                train_mask = torch.ones(n_samples, dtype=torch.bool)
                train_mask[i] = False
                
                X_train_loo = X_tensor[train_mask]
                y_train_loo = y_tensor[train_mask]
                X_test_loo = X_tensor[i:i+1]
                
                # Reset network weights for each LOO iteration
                for layer in network.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                
                # Training closure for LBFGS
                def closure():
                    optimizer.zero_grad()
                    output = network(X_train_loo)
                    loss = criterion(output, y_train_loo)
                    loss.backward()
                    return loss
                
                # Train on LOO training set
                optimizer.step(closure)
                
                # Predict probability for held-out sample (full distribution)
                with torch.no_grad():
                    output = network(X_test_loo)  # Shape: (1, n_classes)
                    # Store full probability distribution for all classes
                    probs = output[0, :].cpu().numpy()  # Shape: (n_classes,)
                    prob_matrix[i, run_idx, :] = probs
            
            loo_time = time.time() - loo_start
            
            # Train final network on full dataset for this run
            for layer in network.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            
            def closure_full():
                optimizer.zero_grad()
                output = network(X_tensor)
                loss = criterion(output, y_tensor)
                loss.backward()
                return loss
            
            optimizer.step(closure_full)
            trained_networks.append(network)
            
            run_time = time.time() - run_start
            
            # Calculate run statistics (use max probability per sample)
            max_probs_run = np.max(prob_matrix[:, run_idx, :], axis=1)
            run_mean = np.mean(max_probs_run)
            run_std = np.std(max_probs_run)
            
            # Print summary for this run
            if self.verbose:
                print(f"    ✓ Run {run_idx+1} completed: {run_time:.1f}s | P̄_max={run_mean:.3f}±{run_std:.3f}\n")
        
        # Compute Bayesian statistics (average probability distribution across runs)
        # p_mean: (n_samples, n_classes) - mean probability distribution
        p_mean = np.mean(prob_matrix, axis=1)
        
        # For uncertainty metrics, use max probability per sample
        max_probs_per_sample = np.max(prob_matrix, axis=2)  # (n_samples, n_runs)
        p_std = np.std(max_probs_per_sample, axis=1)  # (n_samples,)
        vra = np.var(max_probs_per_sample, axis=1)  # (n_samples,)
        
        total_time = time.time() - start_time_total
        
        results = {
            'prob_matrix': prob_matrix,
            'p_mean': p_mean,
            'p_std': p_std,
            'vra': vra,
            'networks': trained_networks
        }
        
        if self.verbose:
            # Compute final predictions for summary
            y_pred_encoded = np.argmax(p_mean, axis=1)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
            
            print(f"\n  ═══ NN_{network_name} Training Complete ═══")
            print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"  Avg time per run: {total_time/n_stochastic_runs:.1f}s")
            print(f"  Classes detected: {n_classes} {list(label_encoder.classes_)}")
            
            # Show prediction distribution
            unique, counts = np.unique(y_pred, return_counts=True)
            print(f"  Predicted distribution:")
            for cls, cnt in zip(unique, counts):
                print(f"    {cls}: {cnt} ({cnt/n_samples*100:.1f}%)")
            
            print(f"  Mean max probability: {np.mean(np.max(p_mean, axis=1)):.3f} ± {np.mean(p_std):.3f}")
            print(f"  Mean VRA (intra-network): {np.mean(vra):.4f}")
            print(f"  VRA range: [{np.min(vra):.4f}, {np.max(vra):.4f}]")
            print(f"  Samples with low VRA (<0.5): {np.sum(vra < 0.5)} ({np.sum(vra < 0.5)/n_samples*100:.1f}%)")
            print(f"  Samples with high VRA (>1.5): {np.sum(vra > 1.5)} ({np.sum(vra > 1.5)/n_samples*100:.1f}%)\n")
        
        return results
    
    def train_all_networks_stochastic(self,
                                     feature_matrices: Dict[str, np.ndarray],
                                     y: np.ndarray,
                                     n_stochastic_runs: int = 25,
                                     n_jobs: int = 1) -> Dict[str, Dict[str, Any]]:
        """
        Train all three networks (FPHW, FP, HW) with stochastic runs.
        
        Parameters:
        -----------
        feature_matrices : dict
            Dictionary with keys 'FPHW', 'FP', 'HW' containing feature matrices
        y : array
            Target labels
        n_stochastic_runs : int
            Number of stochastic runs per network
        n_jobs : int
            Number of parallel workers (1=sequential, -1=all cores)
        
        Returns:
        --------
        all_results : dict
            Dictionary with results for each network
        """
        all_results = {}
        
        # Filter to only networks that have valid feature matrices
        available_networks = [name for name in ['FP', 'HW', 'FPHW'] if name in feature_matrices]
        
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"STOCHASTIC TRAINING: {len(available_networks)} NETWORKS × {n_stochastic_runs} RUNS EACH")
            print("=" * 70)
            print(f"  Available networks: {available_networks}")
            if len(available_networks) < 3:
                excluded = [name for name in ['FP', 'HW', 'FPHW'] if name not in available_networks]
                print(f"  Excluded networks (no valid bands): {excluded}")
        
        # Dependency check: FP is required for all training
        if 'FP' not in available_networks:
            error_msg = "❌ CRITICAL: FP network has no valid bands! Cannot proceed with training."
            if self.verbose:
                print(f"\n{error_msg}")
            raise ValueError(error_msg)
        
        # Training order: FP → HW → FPHW (train simpler networks first)
        network_names = ['FP', 'HW', 'FPHW']
        
        for idx, network_name in enumerate(network_names, 1):
            # Skip if network not available (no valid bands)
            if network_name not in available_networks:
                if self.verbose:
                    print(f"\n[{idx}/3] NN_{network_name} skipped (no valid bands)")
                continue
            
            # Skip if already trained (resume from keyboard interrupt)
            if network_name in all_results:
                if self.verbose:
                    print(f"\n[{idx}/3] NN_{network_name} already trained (resuming...)")
                continue
            
            # Dependency check: FPHW requires HW to be available
            if network_name == 'FPHW' and 'HW' not in available_networks:
                if self.verbose:
                    print(f"\n[{idx}/3] NN_{network_name} skipped (depends on HW which is unavailable)")
                continue
            
            if self.verbose:
                print(f"\n[{idx}/3] Training NN_{network_name}...")
            
            X = feature_matrices[network_name]
            network_start = time.time()
            
            try:
                # Choose backend
                if self.use_torch:
                    results = self.train_single_network_stochastic_torch(
                        X, y, network_name, n_stochastic_runs, n_jobs=n_jobs
                    )
                else:
                    results = self.train_single_network_stochastic(
                        X, y, network_name, n_stochastic_runs, n_jobs=n_jobs
                    )
                
                all_results[network_name] = results
                
                network_time = time.time() - network_start
                if self.verbose:
                    print(f"  ✅ NN_{network_name} completed in {network_time:.1f}s ({network_time/60:.1f} min)")
            
            except KeyboardInterrupt:
                if self.verbose:
                    print(f"\n\n⚠ Training interrupted by user!")
                    print(f"  Completed networks: {list(all_results.keys())}")
                    remaining = [n for n in available_networks if n not in all_results]
                    print(f"  Remaining: {remaining}")
                    print(f"\n  💡 Partial results saved. Re-run fit() to resume training.")
                # Return partial results to allow resuming
                return all_results
        
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"ALL NETWORKS TRAINED SUCCESSFULLY ({len(all_results)}/{len(available_networks)})")
            print("=" * 70)
            print(f"  Trained networks: {list(all_results.keys())}")
        
        return all_results
    
    # ============================================================================
    # BAYESIAN AGGREGATION AND VARIANCE ANALYSIS
    # ============================================================================
    
    def compute_inter_network_variance(self, 
                                      all_results: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """
        Compute VER (Inter-Network Variance) - variance between available networks.
        
        VER measures disagreement between networks:
        - Low VER (<1.0): Networks agree → Confident classification
        - High VER (≥1.0): Networks disagree → Boundary tissue
        
        Note: If only one network is available, returns zeros (no inter-network variance).
        
        Parameters:
        -----------
        all_results : dict
            Results from available networks
        
        Returns:
        --------
        ver : array, shape (n_samples,)
            Inter-network variance for each sample
        """
        # Get only available networks
        available_networks = list(all_results.keys())
        
        if len(available_networks) < 2:
            # Cannot compute inter-network variance with <2 networks
            n_samples = len(all_results[available_networks[0]]['p_mean'])
            return np.zeros(n_samples)
        
        # Stack probabilities from all available networks
        probs = [all_results[net]['p_mean'] for net in available_networks]
        prob_stack = np.stack(probs, axis=1)
        ver = np.var(prob_stack, axis=1)
        
        return ver
    
    def aggregate_network_predictions(self,
                                     all_results: Dict[str, Dict[str, Any]],
                                     method: str = 'mean') -> np.ndarray:
        """
        Aggregate predictions from available networks into final probability.
        
        Parameters:
        -----------
        all_results : dict
            Results from available networks
        method : str, default='mean'
            Aggregation method: 'mean', 'max', 'weighted', or 'voting'
        
        Returns:
        --------
        p_final : array
            Final aggregated probability
        """
        # Get predictions from available networks only
        available_networks = list(all_results.keys())
        probs = [all_results[net]['p_mean'] for net in available_networks]
        
        if len(probs) == 1:
            # Only one network available, use it directly
            return probs[0]
        
        if method == 'mean':
            p_final = np.mean(probs, axis=0)
        elif method == 'max':
            # Conservative: take maximum probability (as in paper)
            p_final = np.maximum.reduce(probs)
        elif method == 'weighted':
            # Weight by inverse variance (more confident networks get higher weight)
            weights = []
            for net in available_networks:
                w = 1.0 / (np.mean(all_results[net]['vra']) + 1e-6)
                weights.append(w)
            w_total = sum(weights)
            
            p_final = sum(w * p for w, p in zip(weights, probs)) / w_total
        elif method == 'voting':
            # Majority voting with threshold=0.5
            votes = sum((p > 0.5).astype(int) for p in probs)
            threshold = len(probs) // 2 + 1  # Majority
            p_final = (votes >= threshold).astype(float)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return p_final
    
    def detect_boundary_spectra(self,
                               vra: np.ndarray,
                               ver: np.ndarray,
                               vra_threshold: float = 1.0,
                               ver_threshold: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Detect boundary/uncertain spectra based on variance thresholds.
        
        Classification regime:
        - High confidence: VRA < threshold AND VER < threshold
        - Boundary: VRA < threshold AND VER ≥ threshold (CRITICAL!)
        - Low confidence: VRA ≥ threshold AND VER < threshold
        - Ambiguous: VRA ≥ threshold AND VER ≥ threshold
        
        Parameters:
        -----------
        vra : array
            Intra-network variance (from each network)
        ver : array
            Inter-network variance
        vra_threshold : float
            Threshold for VRA (typically 1.0)
        ver_threshold : float
            Threshold for VER (typically 1.0)
        
        Returns:
        --------
        flags : dict
            Dictionary with boolean arrays for each regime
        """
        # Compute average VRA across networks if needed
        if len(vra.shape) > 1:
            vra_avg = np.mean(vra, axis=1)
        else:
            vra_avg = vra
        
        flags = {
            'high_confidence': (vra_avg < vra_threshold) & (ver < ver_threshold),
            'boundary': (vra_avg < vra_threshold) & (ver >= ver_threshold),
            'low_confidence': (vra_avg >= vra_threshold) & (ver < ver_threshold),
            'ambiguous': (vra_avg >= vra_threshold) & (ver >= ver_threshold)
        }
        
        return flags
    
    # ============================================================================
    # MAIN TRAINING METHOD (fit)
    # ============================================================================
    
    def fit(self, 
            data_split: Dict[str, Any],
            n_stochastic_runs: int = 25,
            n_jobs: Optional[int] = None) -> 'MultiNetworkEnsembleSystem':
        """
        Fit the complete Multi-Network Ensemble System.
        
        Parameters:
        -----------
        data_split : dict
            Data dictionary containing:
            - 'X_train': Full spectrum training data
            - 'y_train': Training labels
            - 'unified_wavelengths': Wavelength axis
        n_stochastic_runs : int, default=25
            Number of stochastic training runs per network
        n_jobs : int, optional
            Number of parallel jobs for stochastic training.
            - None (default): Auto-detect (1 for GPU/PyTorch, CPU cores for sklearn)
            - 1: Sequential training (safest, recommended for GPU)
            - -1: Use all CPU cores (only safe for sklearn backend)
            - N>1: Use N parallel workers
            
            💡 Smart Defaults:
            - PyTorch + CUDA: n_jobs=1 (sequential, avoids GPU memory issues)
            - sklearn CPU: n_jobs=-1 (parallel, maximum speedup)
            
            ⚠️ GPU Warning: Parallel execution with PyTorch/CUDA can cause
            out-of-memory errors. Use n_jobs=1 for GPU training.
        
        Returns:
        --------
        self : fitted system
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("FITTING MULTI-NETWORK ENSEMBLE SYSTEM")
            print("=" * 70)
        
        self.data_split = data_split
        X_train = data_split['X_train']
        y_train = data_split['y_train']
        self.wavelengths = data_split['unified_wavelengths']
        
        # Step 1: Extract features for all networks
        if self.verbose:
            print("\n[Step 1] Extracting biomarker features...")
        self.feature_matrices = self.prepare_feature_matrices(X_train, self.wavelengths)
        
        # Step 2: K-means autonomous labeling (if enabled)
        if self.use_kmeans_labeling:
            if self.verbose:
                print("\n[Step 2] Performing k-means autonomous labeling...")
            kmeans_labels = self.kmeans_autonomous_labeling(
                self.feature_matrices['FPHW'],
                self.feature_matrices['FP'],
                self.feature_matrices['HW']
            )
            # Use k-means labels from FPHW network
            y_train = kmeans_labels['FPHW']
            if self.verbose:
                unique, counts = np.unique(y_train, return_counts=True)
                print(f"K-means label distribution: {dict(zip(unique, counts))}")
        
        # Step 3: Train all networks with stochastic runs
        if self.verbose:
            print("\n[Step 3] Training stochastic neural networks...")
        
        # Auto-detect n_jobs if not specified
        if n_jobs is None:
            if self.use_torch and self.device.type == 'cuda':
                n_jobs = 1  # Sequential for GPU (avoid OOM)
                if self.verbose:
                    print("  💡 Auto-detected: n_jobs=1 (GPU training, sequential to avoid memory issues)")
            else:
                n_jobs = -1  # Parallel for CPU
                if self.verbose:
                    n_cores = multiprocessing.cpu_count()
                    print(f"  💡 Auto-detected: n_jobs=-1 (CPU training, using {n_cores} cores)")
        
        training_results = self.train_all_networks_stochastic(
            self.feature_matrices,
            y_train,
            n_stochastic_runs,
            n_jobs=n_jobs
        )
        
        # Check if training was interrupted (partial results)
        if not isinstance(training_results, dict) or len(training_results) < len(self.feature_matrices):
            if self.verbose:
                print(f"\n⚠ Training incomplete. Only {len(training_results) if isinstance(training_results, dict) else 0}/{len(self.feature_matrices)} networks trained.")
                print(f"  Re-run fit() to continue training remaining networks.")
            self.training_results = training_results if isinstance(training_results, dict) else {}
            self.is_fitted = False
            return self
        
        self.training_results = training_results
        
        # Check which networks were successfully trained
        available_networks = list(self.training_results.keys())
        if self.verbose and len(available_networks) < 3:
            print(f"\n⚠ Only {len(available_networks)} networks available: {available_networks}")
        
        # Step 4: Compute inter-network variance (only if we have multiple networks)
        if len(available_networks) >= 2:
            if self.verbose:
                print("\n[Step 4] Computing inter-network variance...")
            ver = self.compute_inter_network_variance(self.training_results)
        else:
            if self.verbose:
                print("\n⚠ Skipping inter-network variance (need at least 2 networks)")
            ver = np.zeros(len(y_train))
        
        # Step 5: Detect boundary spectra
        if self.verbose:
            print("\n[Step 5] Detecting boundary spectra...")
        
        # Get VRA from available networks
        vra_values = []
        for net_name in available_networks:
            vra_values.append(self.training_results[net_name]['vra'])
        
        vra_avg = np.mean(vra_values, axis=0) if vra_values else np.zeros(len(y_train))
        
        boundary_flags = self.detect_boundary_spectra(vra_avg, ver)
        
        # Store variance results
        self.variance_results = {
            'vra_avg': vra_avg,
            'ver': ver,
            'boundary_flags': boundary_flags
        }
        
        # Add individual VRA if available
        if 'FPHW' in self.training_results:
            self.variance_results['vra_FPHW'] = self.training_results['FPHW']['vra']
        if 'FP' in self.training_results:
            self.variance_results['vra_FP'] = self.training_results['FP']['vra']
        if 'HW' in self.training_results:
            self.variance_results['vra_HW'] = self.training_results['HW']['vra']
        
        if self.verbose:
            for regime, flags in boundary_flags.items():
                count = np.sum(flags)
                pct = count / len(flags) * 100
                print(f"  {regime}: {count} ({pct:.1f}%)")
        
        self.is_fitted = True
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("MULTI-NETWORK ENSEMBLE SYSTEM FITTED SUCCESSFULLY")
            print("=" * 70)
        
        return self
    
    # ============================================================================
    # EVALUATION METHODS
    # ============================================================================
    
    def evaluate(self, 
                X_test: Optional[np.ndarray] = None,
                y_test: Optional[np.ndarray] = None,
                aggregation_method: str = 'max') -> Dict[str, Any]:
        """
        Evaluate the system on test data.
        
        Parameters:
        -----------
        X_test : array, optional
            Test spectra (if None, uses data_split['X_test'])
        y_test : array, optional
            Test labels (if None, uses data_split['y_test'])
        aggregation_method : str
            Method to aggregate network predictions ('mean', 'max', 'weighted')
        
        Returns:
        --------
        metrics : dict
            Comprehensive evaluation metrics
        """
        if not self.is_fitted:
            raise RuntimeError("System must be fitted before evaluation. Call .fit() first.")
        
        # Use provided test data or default to data_split
        if X_test is None:
            X_test = self.data_split['X_test']
        if y_test is None:
            y_test = self.data_split['y_test']
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("EVALUATING MULTI-NETWORK ENSEMBLE SYSTEM")
            print("=" * 70)
        
        # Extract test features
        test_features = self.prepare_feature_matrices(X_test, self.wavelengths)
        
        # Get available networks (only those that were actually trained)
        available_networks = list(self.training_results.keys())
        
        if self.verbose:
            print(f"  Using {len(available_networks)} networks: {available_networks}")
        
        # Predict with each available network
        predictions = {}
        for network_name in available_networks:
            # Skip if network not in test features (validation check)
            if network_name not in test_features:
                if self.verbose:
                    print(f"  ⚠ Skipping {network_name}: not in test features")
                continue
            
            X_scaled = self.scalers[network_name].transform(test_features[network_name])
            
            # Average predictions across all stochastic runs
            probs = []
            
            if self.use_torch:
                # PyTorch prediction (multi-class)
                label_encoder = self.label_encoders[network_name]
                n_classes = len(label_encoder.classes_)
                
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                
                # For multi-class: get full probability matrix (n_samples, n_classes)
                all_probs = []
                for net in self.training_results[network_name]['networks']:
                    net.eval()
                    with torch.no_grad():
                        output = net(X_tensor)  # (n_samples, n_classes)
                        prob_matrix = output.cpu().numpy()  # Full probability distribution
                        all_probs.append(prob_matrix)
                
                # Average across stochastic runs: shape (n_samples, n_classes)
                p_mean_matrix = np.mean(all_probs, axis=0)
                predictions[network_name] = p_mean_matrix
                
            else:
                # sklearn prediction (multi-class)
                all_probs = []
                for net in self.training_results[network_name]['networks']:
                    prob_matrix = net.predict_proba(X_scaled)  # (n_samples, n_classes)
                    all_probs.append(prob_matrix)
                
                # Average across stochastic runs
                p_mean_matrix = np.mean(all_probs, axis=0)
                predictions[network_name] = p_mean_matrix
        
        # Aggregate predictions from available networks
        # Now predictions are probability matrices (n_samples, n_classes)
        probs_list = [predictions[net] for net in predictions.keys()]
        
        if len(probs_list) == 0:
            raise RuntimeError("No networks available for prediction!")
        elif len(probs_list) == 1:
            p_final_matrix = probs_list[0]
        else:
            if aggregation_method == 'max':
                # Max pooling across networks (element-wise max for each class)
                p_final_matrix = np.maximum.reduce(probs_list)
            elif aggregation_method == 'mean':
                # Average across networks
                p_final_matrix = np.mean(probs_list, axis=0)
            elif aggregation_method == 'weighted':
                # Weight by training VRA (inverse)
                weights = []
                for net_name in predictions.keys():
                    vra_key = f'vra_{net_name}'
                    if vra_key in self.variance_results:
                        w = 1.0 / (np.mean(self.variance_results[vra_key]) + 1e-6)
                    else:
                        w = 1.0  # Default weight if VRA not available
                    weights.append(w)
                w_total = sum(weights)
                p_final_matrix = sum(w * p for w, p in zip(weights, probs_list)) / w_total
        
        # Multi-class predictions: argmax across classes
        y_pred_encoded = np.argmax(p_final_matrix, axis=1)
        
        # DEBUG: Show probability distribution statistics
        if self.verbose:
            print("\n  📊 Probability Distribution Analysis:")
            for class_idx in range(p_final_matrix.shape[1]):
                class_probs = p_final_matrix[:, class_idx]
                print(f"    Class {class_idx}: min={class_probs.min():.4f}, max={class_probs.max():.4f}, mean={class_probs.mean():.4f}")
            
            # Show prediction distribution
            unique, counts = np.unique(y_pred_encoded, return_counts=True)
            print(f"\n  📊 Prediction Distribution:")
            for cls, count in zip(unique, counts):
                print(f"    Class {cls}: {count} predictions ({100*count/len(y_pred_encoded):.1f}%)")
        
        # Decode predictions back to original string labels
        # Use label encoder from first available network (all should have same classes)
        first_network = list(predictions.keys())[0]
        label_encoder = self.label_encoders[first_network]
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        # Compute metrics (multi-class with macro averaging)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)  # Sensitivity
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Compute confusion matrix first (needed for both specificity and verbose output)
        cm = confusion_matrix(y_test, y_pred)
        
        # Compute specificity (multi-class: average across classes)
        try:
            # For multi-class: specificity per class = TN / (TN + FP)
            specificities = []
            for i in range(len(cm)):
                tn_i = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                fp_i = np.sum(cm[:, i]) - cm[i, i]
                spec = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0.0
                specificities.append(spec)
            specificity = np.mean(specificities)
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Could not compute specificity: {e}")
            specificity = 0.0
        
        # For verbose output: extract binary metrics if 2x2 matrix, otherwise use totals
        if cm.shape == (2, 2):
            # Binary classification: can unpack to TN, FP, FN, TP
            tn, fp, fn, tp = cm.ravel()
        else:
            # Multi-class: compute overall metrics
            # Diagonal = correct predictions, off-diagonal = errors
            tp = np.sum(np.diag(cm))  # Total correct
            tn = 0  # Not applicable for multi-class
            fp = np.sum(cm) - tp  # Total errors
            fn = 0  # Not applicable for multi-class
        
        # AUC-ROC (multi-class: use one-vs-rest with micro averaging)
        try:
            # Encode y_test for multi-class metrics
            y_test_encoded = label_encoder.transform(y_test)
            # Check if probabilities are valid (no NaN, proper range)
            if np.any(np.isnan(p_final_matrix)):
                if self.verbose:
                    print(f"  ⚠ Warning: Probability matrix contains NaN values")
                auc_roc = None
            elif np.any(p_final_matrix < 0) or np.any(p_final_matrix > 1):
                if self.verbose:
                    print(f"  ⚠ Warning: Probabilities out of range [0, 1]")
                auc_roc = None
            else:
                auc_roc = roc_auc_score(y_test_encoded, p_final_matrix, 
                                       multi_class='ovr', average='macro')
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Could not compute AUC-ROC: {e}")
            auc_roc = None
        
        # Log loss (multi-class)
        try:
            # log_loss expects string labels and probability matrix
            # Check for valid probabilities first
            if np.any(np.isnan(p_final_matrix)) or np.any(p_final_matrix <= 0):
                if self.verbose:
                    print(f"  ⚠ Warning: Invalid probabilities for log loss computation")
                logloss = None
            else:
                logloss = log_loss(y_test, p_final_matrix, labels=label_encoder.classes_)
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Could not compute log loss: {e}")
            logloss = None
        
        # Build predictions dict with only available networks
        predictions_dict = {net: predictions[net] for net in predictions.keys()}
        predictions_dict['final'] = p_final_matrix  # (n_samples, n_classes)
        predictions_dict['y_pred'] = y_pred  # String labels
        predictions_dict['y_pred_encoded'] = y_pred_encoded  # Numeric labels
        predictions_dict['probabilities'] = p_final_matrix  # For visualization
        predictions_dict['network_probabilities'] = predictions  # Backward compatibility
        
        # Compute variance analysis (if multiple networks available)
        if len(predictions) >= 2:
            # Get VRA from training results
            vra_values = []
            for net in predictions.keys():
                vra_key = f'vra_{net}'
                if vra_key in self.variance_results:
                    vra_values.append(self.variance_results[vra_key])
            
            if vra_values:
                vra = np.mean(vra_values, axis=0)
            else:
                vra = np.zeros(len(y_test))
            
            # Compute VER: variance of max probabilities across networks
            # For each sample, get max probability from each network, then compute variance
            max_probs_per_network = []
            for net in predictions.keys():
                max_probs = np.max(predictions[net], axis=1)  # Max prob per sample
                max_probs_per_network.append(max_probs)
            
            ver = np.var(np.stack(max_probs_per_network, axis=1), axis=1)
            
            # Detect boundary spectra
            boundary_flags = self.detect_boundary_spectra(vra, ver)
        else:
            # Single network: no inter-network variance
            vra = np.zeros(len(y_test))
            ver = np.zeros(len(y_test))
            boundary_flags = {
                'high_confidence': np.ones(len(y_test), dtype=bool),
                'boundary': np.zeros(len(y_test), dtype=bool),
                'low_confidence': np.zeros(len(y_test), dtype=bool),
                'ambiguous': np.zeros(len(y_test), dtype=bool)
            }
        
        predictions_dict['boundary_flags'] = boundary_flags
        
        # Build initial metrics dict
        per_class_metrics = {}
        
        # Add per-class metrics if binary classification
        try:
            for class_label in np.unique(y_test):
                class_mask = (y_test == class_label)
                class_pred_mask = (y_pred == class_label)
                
                tp_class = np.sum(class_mask & class_pred_mask)
                tn_class = np.sum(~class_mask & ~class_pred_mask)
                fp_class = np.sum(~class_mask & class_pred_mask)
                fn_class = np.sum(class_mask & ~class_pred_mask)
                
                sensitivity_class = tp_class / (tp_class + fn_class) if (tp_class + fn_class) > 0 else 0.0
                specificity_class = tn_class / (tn_class + fp_class) if (tn_class + fp_class) > 0 else 0.0
                
                per_class_metrics[str(class_label)] = {
                    'sensitivity': sensitivity_class,
                    'specificity': specificity_class
                }
        except:
            pass
        
        metrics = {
            'accuracy': accuracy,
            'sensitivity': recall,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'log_loss': logloss,
            'predictions': predictions_dict,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'variance_analysis': {
                'vra': vra,
                'ver': ver,
                'mean_vra': np.mean(vra),
                'mean_ver': np.mean(ver)
            },
            'boundary_detection': {
                'high_confidence': np.sum(boundary_flags['high_confidence']),
                'boundary': np.sum(boundary_flags['boundary']),
                'low_confidence': np.sum(boundary_flags['low_confidence']),
                'ambiguous': np.sum(boundary_flags['ambiguous'])
            },
            'per_class': per_class_metrics
        }
        
        # Print results
        if self.verbose:
            # Show probability distribution analysis
            print(f"\n📊 Probability Distribution Analysis:")
            for class_idx in range(p_final_matrix.shape[1]):
                class_label = label_encoder.classes_[class_idx]
                class_probs = p_final_matrix[:, class_idx]
                print(f"  Class '{class_label}' (idx={class_idx}): min={class_probs.min():.4f}, max={class_probs.max():.4f}, mean={class_probs.mean():.4f}")
            
            # Show prediction distribution
            unique, counts = np.unique(y_pred, return_counts=True)
            print(f"\n📊 Prediction Distribution:")
            for cls, count in zip(unique, counts):
                print(f"  '{cls}': {count} predictions ({100*count/len(y_pred):.1f}%)")
            
            print(f"\n[Test Set Performance]")
            print(f"  Accuracy:    {accuracy:.4f}")
            print(f"  Sensitivity: {recall:.4f} (Recall/TPR)")
            print(f"  Specificity: {specificity:.4f} (TNR)")
            print(f"  Precision:   {precision:.4f} (PPV)")
            print(f"  F1 Score:    {f1:.4f}")
            if auc_roc is not None:
                print(f"  AUC-ROC:     {auc_roc:.4f}")
            else:
                print(f"  AUC-ROC:     N/A (could not compute)")
            if logloss is not None:
                print(f"  Log Loss:    {logloss:.4f}")
            else:
                print(f"  Log Loss:    N/A (could not compute)")
            
            print(f"\n[Confusion Matrix]")
            print(f"  TN={tn}, FP={fp}")
            print(f"  FN={fn}, TP={tp}")
        
        return metrics
    
    # ============================================================================
    # PREDICTION METHODS
    # ============================================================================
    
    def predict_new_data(self,
                        X_new: np.ndarray,
                        y_new: Optional[np.ndarray] = None,
                        return_uncertainty: bool = True,
                        show_confusion_matrix: bool = False) -> Dict[str, Any]:
        """
        Predict on new data with uncertainty quantification.
        
        Parameters:
        -----------
        X_new : array, shape (n_samples, n_wavelengths)
            New full spectra
        y_new : array, optional
            True labels (if available for evaluation)
        return_uncertainty : bool
            If True, compute VRA and VER for new predictions
        show_confusion_matrix : bool
            If True and y_new provided, display confusion matrix
        
        Returns:
        --------
        results : dict
            Predictions with probabilities and uncertainties
        """
        if not self.is_fitted:
            raise RuntimeError("System must be fitted before prediction. Call .fit() first.")
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("PREDICTING ON NEW DATA")
            print("=" * 70)
            print(f"Number of new samples: {X_new.shape[0]}")
        
        # Extract features
        new_features = self.prepare_feature_matrices(X_new, self.wavelengths)
        
        # Get available networks (only those that were actually trained)
        available_networks = list(self.training_results.keys())
        
        if self.verbose:
            print(f"  Using {len(available_networks)} networks: {available_networks}")
        
        # Predict with each available network
        predictions = {}
        uncertainties = {}
        
        for network_name in available_networks:
            # Skip if network not in new features (validation check)
            if network_name not in new_features:
                if self.verbose:
                    print(f"  ⚠ Skipping {network_name}: not in new features")
                continue
            
            X_scaled = self.scalers[network_name].transform(new_features[network_name])
            
            # Collect predictions from all stochastic networks
            probs_all_runs = []
            
            if self.use_torch:
                # PyTorch prediction (multi-class)
                label_encoder = self.label_encoders[network_name]
                n_classes = len(label_encoder.classes_)
                
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                for net in self.training_results[network_name]['networks']:
                    net.eval()
                    with torch.no_grad():
                        output = net(X_tensor)  # (n_samples, n_classes)
                        prob_matrix = output.cpu().numpy()  # Full probability distribution
                        probs_all_runs.append(prob_matrix)
            else:
                # sklearn prediction (multi-class)
                for net in self.training_results[network_name]['networks']:
                    prob_matrix = net.predict_proba(X_scaled)  # (n_samples, n_classes)
                    probs_all_runs.append(prob_matrix)
            
            # Shape: (n_runs, n_samples, n_classes)
            probs_all_runs = np.array(probs_all_runs)
            
            # Average across runs: (n_samples, n_classes)
            p_mean_matrix = np.mean(probs_all_runs, axis=0)
            
            # For uncertainty: compute variance of max probabilities across runs
            max_probs_per_run = np.max(probs_all_runs, axis=2)  # (n_runs, n_samples)
            p_std = np.std(max_probs_per_run, axis=0)
            vra = np.var(max_probs_per_run, axis=0)
            
            predictions[network_name] = p_mean_matrix
            uncertainties[network_name] = {
                'prob_matrix': probs_all_runs,
                'p_std': p_std,
                'vra': vra
            }
        
        # Aggregate predictions from available networks
        # Now predictions are probability matrices (n_samples, n_classes)
        probs_list = [predictions[net] for net in predictions.keys()]
        
        if len(probs_list) == 0:
            raise RuntimeError("No networks available for prediction!")
        elif len(probs_list) == 1:
            p_final_matrix = probs_list[0]
        else:
            p_final_matrix = np.maximum.reduce(probs_list)  # Element-wise max
        
        # Multi-class predictions: argmax across classes
        y_pred_encoded = np.argmax(p_final_matrix, axis=1)
        
        # Decode predictions back to original string labels
        first_network = list(predictions.keys())[0]
        label_encoder = self.label_encoders[first_network]
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        # Compute VER (inter-network variance) if multiple networks available
        if len(predictions) >= 2:
            # Variance of max probabilities across networks
            max_probs_per_network = [np.max(predictions[net], axis=1) for net in predictions.keys()]
            ver = np.var(np.stack(max_probs_per_network, axis=1), axis=1)
        else:
            ver = np.zeros(len(y_pred))  # No inter-network variance with 1 network
        
        # Detect boundary spectra
        vra_values = [uncertainties[net]['vra'] for net in uncertainties.keys()]
        vra_avg = np.mean(vra_values, axis=0)
        boundary_flags = self.detect_boundary_spectra(vra_avg, ver)
        
        # Compile results with only available networks
        results = {
            'predictions': {
                'final_matrix': p_final_matrix,  # (n_samples, n_classes)
                'y_pred': y_pred,  # String labels
                'y_pred_encoded': y_pred_encoded  # Numeric labels
            },
            'uncertainties': uncertainties,
            'ver': ver,
            'vra_avg': vra_avg,
            'boundary_flags': boundary_flags,
            'n_samples': X_new.shape[0]
        }
        
        # Print summary
        if self.verbose:
            max_probs = np.max(p_final_matrix, axis=1)
            print(f"\n[Prediction Summary]")
            print(f"  Mean final probability: {np.mean(max_probs):.3f} ± {np.std(max_probs):.3f}")
            unique, counts = np.unique(y_pred, return_counts=True)
            for label, count in zip(unique, counts):
                print(f"  {label}: {count} ({count/len(y_pred)*100:.1f}%)")
            print(f"\n[Uncertainty Analysis]")
            print(f"  Mean VRA: {np.mean(vra_avg):.4f}")
            print(f"  Mean VER: {np.mean(ver):.4f}")
            for regime, flags in boundary_flags.items():
                count = np.sum(flags)
                pct = count / len(flags) * 100
                print(f"  {regime}: {count} ({pct:.1f}%)")
        
        # Evaluate if labels provided
        if y_new is not None:
            accuracy = accuracy_score(y_new, y_pred)
            precision = precision_score(y_new, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_new, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_new, y_pred, average='macro', zero_division=0)
            
            # Multi-class specificity
            try:
                cm = confusion_matrix(y_new, y_pred)
                specificities = []
                for i in range(len(cm)):
                    tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                    fp = np.sum(cm[:, i]) - cm[i, i]
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    specificities.append(spec)
                specificity = np.mean(specificities)
            except:
                specificity = 0.0
            
            results['evaluation'] = {
                'accuracy': accuracy,
                'sensitivity': recall,
                'specificity': specificity,
                'precision': precision,
                'f1_score': f1,
                'confusion_matrix': confusion_matrix(y_new, y_pred)
            }
            
            if self.verbose:
                print(f"\n[Evaluation Metrics]")
                print(f"  Accuracy:    {accuracy:.4f}")
                print(f"  Sensitivity: {recall:.4f}")
                print(f"  Specificity: {specificity:.4f}")
                print(f"  Precision:   {precision:.4f}")
                print(f"  F1 Score:    {f1:.4f}")
            
            if show_confusion_matrix:
                self._plot_confusion_matrix_new_data(y_new, y_pred)
        
        return results
    
    # ============================================================================
    # VISUALIZATION METHODS
    # ============================================================================
    
    def plot_confusion_matrix(self, show_plot: bool = True) -> np.ndarray:
        """
        Plot confusion matrix for test set predictions.
        
        Parameters:
        -----------
        show_plot : bool
            Whether to display the plot
        
        Returns:
        --------
        cm : array
            Confusion matrix
        """
        if not self.is_fitted:
            raise RuntimeError("System must be fitted before plotting.")
        
        # Get test predictions
        eval_results = self.evaluate(aggregation_method='max')
        cm = eval_results['confusion_matrix']
        y_pred = eval_results['predictions']['y_pred']
        y_true = self.data_split['y_test']
        
        if show_plot:
            # Get actual class names
            class_names = np.unique(y_true)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names,
                       yticklabels=class_names,
                       ax=ax, cbar_kws={'label': 'Count'})
            
            ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
            ax.set_title('Confusion Matrix - Multi-Network Ensemble System\n(Test Set)',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
        
        return cm
    
    def _plot_confusion_matrix_new_data(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot confusion matrix for new data predictions."""
        cm = confusion_matrix(y_true, y_pred)
        
        # Get actual class names
        class_names = np.unique(y_true)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix - Multi-Network Ensemble System\n(New Data)',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_probability_distributions(self) -> None:
        """
        Plot probability distributions from stochastic runs for training data.
        Shows Bayesian probability distributions for sample spectra.
        """
        if not self.is_fitted:
            raise RuntimeError("System must be fitted before plotting.")
        
        # Get first available network (prefer FPHW, then FP, then HW)
        available_networks = list(self.training_results.keys())
        if not available_networks:
            print("⚠ No trained networks available for plotting.")
            return
        
        # Prefer FPHW, but use first available if FPHW not available
        network_name = 'FPHW' if 'FPHW' in available_networks else available_networks[0]
        
        # Get class names for label
        class_names = self.label_encoder.classes_
        positive_class = class_names[1] if len(class_names) > 1 else class_names[0]
        
        # Select 6 random samples to display
        n_samples = self.training_results[network_name]['prob_matrix'].shape[0]
        sample_indices = np.random.choice(n_samples, min(6, n_samples), replace=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, sample_idx in enumerate(sample_indices):
            # Get probabilities from selected network
            probs = self.training_results[network_name]['prob_matrix'][sample_idx, :]
            
            axes[idx].hist(probs, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
            
            mean_p = np.mean(probs)
            std_p = np.std(probs)
            
            axes[idx].axvline(mean_p, color='red', linewidth=2, label=f'Mean: {mean_p:.3f}')
            axes[idx].axvline(mean_p - 2*std_p, color='orange', linestyle='--', label='95% CI')
            axes[idx].axvline(mean_p + 2*std_p, color='orange', linestyle='--')
            
            axes[idx].set_xlabel(f'P({positive_class})', fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].set_title(f'Sample {sample_idx}\nVRA={std_p**2:.4f}', fontsize=10, fontweight='bold')
            axes[idx].legend(fontsize=8)
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Bayesian Probability Distributions from Stochastic Runs (NN_{network_name})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_variance_analysis(self) -> None:
        """
        Plot VRA vs VER scatter plot with boundary detection.
        """
        if not self.is_fitted:
            raise RuntimeError("System must be fitted before plotting.")
        
        vra_avg = self.variance_results['vra_avg']
        ver = self.variance_results['ver']
        flags = self.variance_results['boundary_flags']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot each regime with different colors
        colors = {
            'high_confidence': 'green',
            'boundary': 'red',
            'low_confidence': 'orange',
            'ambiguous': 'purple'
        }
        
        for regime, color in colors.items():
            mask = flags[regime]
            ax.scatter(ver[mask], vra_avg[mask], 
                      c=color, alpha=0.6, s=50, label=regime.replace('_', ' ').title())
        
        # Threshold lines
        ax.axvline(1.0, color='gray', linestyle='--', linewidth=2, label='VER threshold')
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=2, label='VRA threshold')
        
        ax.set_xlabel('VER (Inter-Network Variance)', fontsize=12, fontweight='bold')
        ax.set_ylabel('VRA (Intra-Network Variance)', fontsize=12, fontweight='bold')
        ax.set_title('Variance Analysis: VRA vs VER\n(Training Data)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_network_agreement(self) -> None:
        """
        Plot probability comparison between available networks.
        """
        if not self.is_fitted:
            raise RuntimeError("System must be fitted before plotting.")
        
        # Get available networks
        available_networks = list(self.training_results.keys())
        
        if len(available_networks) < 2:
            print("⚠ Network agreement plot requires at least 2 networks.")
            print(f"  Only {available_networks[0]} is available.")
            return
        
        # Get predictions from available networks
        network_probs = {net: self.training_results[net]['p_mean'] for net in available_networks}
        
        # Get class names for label
        class_names = self.label_encoder.classes_
        positive_class = class_names[1] if len(class_names) > 1 else class_names[0]
        
        # Create subplot grid
        n_comparisons = len(available_networks) * (len(available_networks) - 1) // 2
        if n_comparisons == 1:
            fig, axes = plt.subplots(1, 1, figsize=(6, 5))
            axes = [axes]
        else:
            ncols = min(3, n_comparisons)
            nrows = (n_comparisons + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
            axes = axes.flatten() if n_comparisons > 1 else [axes]
        
        # Plot all pairwise comparisons
        colors = ['blue', 'green', 'purple', 'orange', 'red', 'brown']
        idx = 0
        for i, net1 in enumerate(available_networks):
            for j, net2 in enumerate(available_networks):
                if j > i:  # Only upper triangle (avoid duplicates)
                    p1 = network_probs[net1]
                    p2 = network_probs[net2]
                    
                    axes[idx].scatter(p1, p2, alpha=0.5, s=30, c=colors[idx % len(colors)])
                    axes[idx].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect agreement')
                    axes[idx].set_xlabel(f'P({positive_class}) - NN_{net1}', fontsize=11)
                    axes[idx].set_ylabel(f'P({positive_class}) - NN_{net2}', fontsize=11)
                    axes[idx].set_title(f'NN_{net1} vs NN_{net2}', fontsize=12, fontweight='bold')
                    axes[idx].legend()
                    axes[idx].grid(alpha=0.3)
                    idx += 1
        
        # Hide unused subplots
        for i in range(idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Network Agreement Analysis - {len(available_networks)} Networks (Training Data)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    # ============================================================================
    # PERSISTENCE METHODS
    # ============================================================================
    
    def save_model(self, filepath: str) -> None:
        """
        Save the complete trained system to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save file (.pkl)
        """
        if not self.is_fitted:
            raise RuntimeError("System must be fitted before saving.")
        
        model_data = {
            'biomarker_bands': self.biomarker_bands,
            'n_hidden': self.n_hidden,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'use_kmeans_labeling': self.use_kmeans_labeling,
            'networks': self.networks,
            'scalers': self.scalers,
            'training_results': self.training_results,
            'variance_results': self.variance_results,
            'data_split': self.data_split,
            'wavelengths': self.wavelengths,
            'feature_matrices': self.feature_matrices
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        if self.verbose:
            print(f"✓ Multi-Network Ensemble System saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str, verbose: bool = True) -> 'MultiNetworkEnsembleSystem':
        """
        Load a trained system from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to saved model file
        verbose : bool
            Print loading information
        
        Returns:
        --------
        system : MultiNetworkEnsembleSystem
            Loaded system instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Reconstruct instance
        system = MultiNetworkEnsembleSystem(
            biomarker_bands=model_data['biomarker_bands'],
            n_hidden=model_data['n_hidden'],
            max_iter=model_data['max_iter'],
            random_state=model_data['random_state'],
            use_kmeans_labeling=model_data['use_kmeans_labeling'],
            verbose=verbose
        )
        
        system.networks = model_data['networks']
        system.scalers = model_data['scalers']
        system.training_results = model_data['training_results']
        system.variance_results = model_data['variance_results']
        system.data_split = model_data['data_split']
        system.wavelengths = model_data['wavelengths']
        system.feature_matrices = model_data['feature_matrices']
        system.is_fitted = True
        
        if verbose:
            print(f"✓ Multi-Network Ensemble System loaded from {filepath}")
        
        return system
