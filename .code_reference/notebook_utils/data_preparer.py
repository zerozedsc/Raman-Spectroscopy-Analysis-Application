import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Any, Optional, Union
import pickle
from scipy.interpolate import interp1d
import traceback
import warnings

class RamanDataPreparer:
    """
    Enhanced Raman spectroscopy data preparation class with batch harmonization 
    and CORAL domain adaptation for disease detection ML models.
    
    This class provides comprehensive functionality for:
    - Patient-level data splitting to prevent data leakage
    - Unified wavelength interpolation for consistent feature representation
    - Advanced batch effect harmonization using multiple methods
    - CORAL domain adaptation for improved generalization
    - Before/after visualization capabilities
    - External harmonization parameter saving/loading for new data prediction
    
    References:
    - Johnson et al. (2007): ComBat for batch effect correction
    - Fortin et al. (2017): Harmonization for neuroimaging data
    - Sun & Saenko (2016): CORAL for domain adaptation
    """
    
    def __init__(self, 
                 raman_data: Dict[str, Dict[str, List[Dict]]],
                 selected_types: List[str] = None,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 label_key: str = 'type',
                 batch_key: str = 'Hikkoshi',
                 wavelength_range: Optional[Tuple[int, int]] = None,
                 wavelength_step: int = 1,
                 apply_snv: bool = False,
                 # New harmonization parameters
                 enable_harmonization: bool = False,
                 use_batch: Optional[Dict[str, str]] = None,
                 harmonization_method: str = 'neurocombat',
                 harmonization_reference_strategy: str = 'within_type',
                 reference_batch: str = 'A',
                 apply_coral: bool = False,
                 coral_lambda: float = 1.0,
                 external_harmonization_params: Optional[Dict] = None,
                 **kwargs):
        """
        Initialize the enhanced Raman data preparer with harmonization capabilities.
        
        Args:
            raman_data: The full Raman data dictionary
            selected_types: List of data types to include (e.g., ['MGUS', 'MM', 'NL'])
            test_size: Proportion of patients for testing per batch (0-1). If 0, all data goes to training.
            random_state: Random seed for reproducible splits
            label_key: Key from metadata to use as labels ('type')
            batch_key: Key from metadata to use as batch labels ('Hikkoshi')
            wavelength_range: Min and max wavelength for unified grid (auto-detected if None)
            wavelength_step: Step size for wavelength grid (cm⁻¹)
            apply_snv: Whether to apply Standard Normal Variate preprocessing
            
            # Harmonization Parameters
            enable_harmonization: Enable batch effect harmonization
            use_batch: Batch reference dict, e.g., {'Hikkoshi': 'A'}. If None, auto-optimize
            harmonization_method: 'neurocombat', 'empirical_bayes', 'full_bayes'
            harmonization_reference_strategy: 'within_type', 'across_types', 'external'
            reference_batch: Reference batch identifier for harmonization
            apply_coral: Enable CORAL domain adaptation
            coral_lambda: CORAL regularization parameter
            external_harmonization_params: Pre-saved harmonization parameters for new data
            **kwargs: Additional parameters for harmonization methods
        """
        
        # === Core Data Parameters ===
        self.raman_data = raman_data
        self.selected_types = selected_types or list(raman_data.keys())
        self.test_size = test_size
        self.random_state = random_state
        self.label_key = label_key
        self.batch_key = batch_key
        self.wavelength_range = wavelength_range
        self.wavelength_step = wavelength_step
        self.apply_snv = apply_snv
        
        # === Harmonization Parameters ===
        self.enable_harmonization = enable_harmonization
        self.use_batch = use_batch or {batch_key: reference_batch}
        self.harmonization_method = harmonization_method
        self.harmonization_reference_strategy = harmonization_reference_strategy
        self.reference_batch = reference_batch
        self.apply_coral = apply_coral
        self.coral_lambda = coral_lambda
        self.external_harmonization_params = external_harmonization_params
        self.harmonization_kwargs = kwargs
        
        # === Internal State Variables ===
        self.harmonization_parameters = {}
        self.coral_transformation = None
        self.scaler_params = None
        
        # === Validation and Setup ===
        self._validate_parameters()
        self._setup_wavelength_grid()
        self._setup_patient_splits()
        self._print_initialization_summary()
    
    # ==========================================
    # === PARAMETER VALIDATION AND SETUP ===
    # ==========================================
    
    def _validate_parameters(self) -> None:
        """Validate input parameters and data consistency."""
        # Validate selected types
        invalid_types = [t for t in self.selected_types if t not in self.raman_data]
        if invalid_types:
            raise ValueError(f"Invalid types selected: {invalid_types}")
        
        # Validate harmonization method
        valid_methods = ['neurocombat', 'empirical_bayes', 'full_bayes']
        if self.harmonization_method not in valid_methods:
            raise ValueError(f"Invalid harmonization method. Choose from: {valid_methods}")
        
        # Validate reference strategy
        valid_strategies = ['within_type', 'across_types', 'external']
        if self.harmonization_reference_strategy not in valid_strategies:
            raise ValueError(f"Invalid reference strategy. Choose from: {valid_strategies}")
    
    def _setup_wavelength_grid(self) -> None:
        """Set up unified wavelength grid for interpolation."""
        if self.wavelength_range is None:
            self.wavelength_range = self._auto_detect_wavelength_range()
        
        self.unified_wavelengths = np.arange(
            self.wavelength_range[0], 
            self.wavelength_range[1] + self.wavelength_step, 
            self.wavelength_step
        )
    
    def _setup_patient_splits(self) -> None:
        """Set up patient-level data splits."""
        self.all_patients = self._collect_all_patients()
        self.train_patients, self.test_patients = self._split_patients()
    
    def _print_initialization_summary(self) -> None:
        """Print initialization summary."""
        print("=== Enhanced Raman Data Preparer Initialized ===")
        print(f"Total patients: {len(self.all_patients)}")
        print(f"Train patients: {len(self.train_patients)}")
        print(f"Test patients: {len(self.test_patients)}")
        print(f"Unified wavelength grid: {len(self.unified_wavelengths)} points "
              f"from {self.wavelength_range[0]} to {self.wavelength_range[1]} cm⁻¹")
        print(f"Batch key: '{self.batch_key}' | SNV: {'Enabled' if self.apply_snv else 'Disabled'}")
        print(f"Harmonization: {'Enabled' if self.enable_harmonization else 'Disabled'} "
              f"({self.harmonization_method})")
        print(f"CORAL: {'Enabled' if self.apply_coral else 'Disabled'}")
        if self.external_harmonization_params:
            print("External harmonization parameters loaded for new data prediction")
        print("=" * 50)
    
    # ==========================================
    # === CORE DATA PROCESSING METHODS ===
    # ==========================================
    
    def _auto_detect_wavelength_range(self) -> Tuple[int, int]:
        """Automatically detect wavelength range from all spectra."""
        all_wavelengths = []
        for data_type in self.selected_types:
            if data_type in self.raman_data:
                for patient_id, spectra in self.raman_data[data_type].items():
                    for spectrum_data in spectra:
                        df = spectrum_data['dataframe']
                        wavelengths = df['wavelength'].values
                        all_wavelengths.extend(wavelengths)
        
        if not all_wavelengths:
            raise ValueError("No wavelength data found in the provided Raman data.")
        
        min_wavelength = int(np.floor(np.min(all_wavelengths)))
        max_wavelength = int(np.ceil(np.max(all_wavelengths)))
        
        print(f"Auto-detected wavelength range: {min_wavelength} to {max_wavelength} cm⁻¹")
        return (min_wavelength, max_wavelength)
    
    def _collect_all_patients(self) -> List[Tuple[str, str, str]]:
        """Collect all unique patients with batch information."""
        patients = []
        for data_type in self.selected_types:
            if data_type in self.raman_data:
                for patient_id in self.raman_data[data_type].keys():
                    if self.raman_data[data_type][patient_id]:
                        batch = self.raman_data[data_type][patient_id][0]['metadata'].get(
                            self.batch_key, 'Unknown'
                        )
                        patients.append((data_type, patient_id, batch))
        return patients
    
    def _split_patients(self) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """Split patients with batch-aware stratification."""
        if self.test_size == 0:
            print("test_size=0: All data allocated to training (no test split).")
            return self.all_patients, []
        
        # Group patients by batch
        batch_groups = {}
        for patient in self.all_patients:
            batch = patient[2]
            if batch not in batch_groups:
                batch_groups[batch] = []
            batch_groups[batch].append(patient)
        
        train_patients = []
        test_patients = []
        
        for batch, patients in batch_groups.items():
            if len(patients) == 0:
                continue
            
            # Split patients within this batch
            try:
                batch_train, batch_test = train_test_split(
                    patients,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=[p[0] for p in patients]  # Stratify by data type within batch
                )
            except ValueError:
                # If stratification fails, split without stratification
                batch_train, batch_test = train_test_split(
                    patients,
                    test_size=self.test_size,
                    random_state=self.random_state
                )
            
            train_patients.extend(batch_train)
            test_patients.extend(batch_test)
            
            print(f"Batch '{batch}': {len(patients)} patients -> "
                  f"Train: {len(batch_train)}, Test: {len(batch_test)}")
        
        return train_patients, test_patients
    
    def _extract_spectra_from_patients(self, 
                                     patients: List[Tuple[str, str, str]]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Extract spectra data and labels from patients with unified wavelength interpolation."""
        spectra_list = []
        labels_list = []
        sample_info = []
        batch_labels = []
        
        for data_type, patient_id, _ in patients:
            if data_type in self.raman_data and patient_id in self.raman_data[data_type]:
                for spectrum_idx, spectrum_data in enumerate(self.raman_data[data_type][patient_id]):
                    df = spectrum_data['dataframe']
                    metadata = spectrum_data['metadata']
                    
                    # Sort by wavelength and interpolate
                    df_sorted = df.sort_values('wavelength')
                    wavelengths = df_sorted['wavelength'].values
                    intensities = df_sorted['intensity'].values
                    
                    # Interpolate to unified wavelength grid
                    unified_intensities = self._interpolate_spectrum(wavelengths, intensities)
                    
                    spectra_list.append(unified_intensities)
                    labels_list.append(metadata.get(self.label_key, 'Unknown'))
                    sample_info.append(f"{data_type}_{patient_id}_spectrum_{spectrum_idx}")
                    batch_labels.append(metadata.get(self.batch_key, 'Unknown'))
        
        X = np.array(spectra_list)
        y = np.array(labels_list)
        
        return X, y, sample_info, batch_labels
    
    def _interpolate_spectrum(self, wavelengths: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """Interpolate a single spectrum to the unified wavelength grid."""
        if len(wavelengths) > 1:
            interp_func = interp1d(
                wavelengths, intensities, kind='linear',
                bounds_error=False, fill_value=np.nan
            )
            unified_intensities = interp_func(self.unified_wavelengths)
            
            # Handle NaN values
            if np.any(np.isnan(unified_intensities)):
                valid_mask = ~np.isnan(unified_intensities)
                if np.any(valid_mask):
                    unified_intensities = np.interp(
                        self.unified_wavelengths,
                        self.unified_wavelengths[valid_mask],
                        unified_intensities[valid_mask]
                    )
                else:
                    unified_intensities = np.zeros_like(self.unified_wavelengths)
        else:
            # Single point case
            unified_intensities = np.full(
                len(self.unified_wavelengths), 
                intensities[0] if len(intensities) > 0 else 0
            )
        
        return unified_intensities
    
    # ==========================================
    # === HARMONIZATION METHODS ===
    # ==========================================
    
    def _apply_harmonization(self, 
                           X_train: np.ndarray, 
                           y_train: np.ndarray, 
                           batch_train: List[str],
                           X_test: Optional[np.ndarray] = None,
                           batch_test: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply batch effect harmonization using the specified method."""
        print(f"\n=== Applying {self.harmonization_method} Harmonization ===")
        
        # Prepare harmonization parameters
        harmonizer_params = {
            'method': self.harmonization_method,
            'reference_strategy': self.harmonization_reference_strategy,
            'reference_batch': self.reference_batch,
            'use_batch': self.use_batch,
            **self.harmonization_kwargs
        }
        
        if self.harmonization_method == 'neurocombat':
            X_train_harmonized, X_test_harmonized = self._apply_neurocombat_harmonization(
                X_train, y_train, batch_train, X_test, batch_test
            )
        elif self.harmonization_method == 'empirical_bayes':
            X_train_harmonized, X_test_harmonized = self._apply_empirical_bayes_harmonization(
                X_train, y_train, batch_train, X_test, batch_test
            )
        elif self.harmonization_method == 'full_bayes':
            X_train_harmonized, X_test_harmonized = self._apply_full_bayes_harmonization(
                X_train, y_train, batch_train, X_test, batch_test
            )
        else:
            raise ValueError(f"Unknown harmonization method: {self.harmonization_method}")
        
        # Store harmonization parameters for external use
        self.harmonization_parameters = {
            'method': self.harmonization_method,
            'parameters': harmonizer_params,
            'reference_batch': self.reference_batch,
            'batch_statistics': self._calculate_batch_statistics(X_train, batch_train)
        }
        
        print(f"Harmonization completed using {self.harmonization_method}")
        return X_train_harmonized, X_test_harmonized
    
    def _apply_neurocombat_harmonization(self,
                                       X_train: np.ndarray,
                                       y_train: np.ndarray,
                                       batch_train: List[str],
                                       X_test: Optional[np.ndarray] = None,
                                       batch_test: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply neuroCombat harmonization."""
        try:
            from neuroCombat import neuroCombat
            
            # Prepare data (features x samples for neuroCombat)
            data_train = X_train.T
            
            # Prepare covariates DataFrame
            covars_dict = {'batch': list(batch_train)}
            if y_train is not None:
                covars_dict['diagnosis'] = list(y_train)
            
            covars = pd.DataFrame(covars_dict)
            categorical_cols = ['diagnosis'] if y_train is not None else []
            
            print(f"neuroCombat input: data {data_train.shape}, covars {covars.shape}")
            print(f"Reference batch: {self.reference_batch}")
            
            # Apply neuroCombat
            result = neuroCombat(
                dat=data_train,
                covars=covars,
                batch_col='batch',
                categorical_cols=categorical_cols,
                ref_batch=self.reference_batch
            )
            
            X_train_harmonized = result["data"].T
            
            # Apply to test data if provided
            X_test_harmonized = None
            if X_test is not None and batch_test is not None:
                # For test data, use the same parameters from training
                data_combined = np.vstack([X_train, X_test]).T
                batch_combined = batch_train + batch_test
                
                covars_combined_dict = {'batch': batch_combined}
                if y_train is not None:
                    # Extend with test labels (can be dummy for harmonization)
                    y_test_dummy = ['Unknown'] * len(batch_test)
                    y_combined = list(y_train) + y_test_dummy
                    covars_combined_dict['diagnosis'] = y_combined
                
                covars_combined = pd.DataFrame(covars_combined_dict)
                
                result_combined = neuroCombat(
                    dat=data_combined,
                    covars=covars_combined,
                    batch_col='batch',
                    categorical_cols=categorical_cols,
                    ref_batch=self.reference_batch
                )
                
                harmonized_combined = result_combined["data"].T
                X_test_harmonized = harmonized_combined[len(X_train):]
            
            # Store neuroCombat parameters
            if 'estimates' in result:
                self.harmonization_parameters['neurocombat_estimates'] = result['estimates']
            
            return X_train_harmonized, X_test_harmonized
            
        except ImportError:
            print("neuroCombat not available. Install with: pip install neuroCombat")
            print("Falling back to empirical Bayes harmonization...")
            return self._apply_empirical_bayes_harmonization(
                X_train, y_train, batch_train, X_test, batch_test
            )
        except Exception as e:
            print(f"neuroCombat failed: {e}")
            print("Falling back to empirical Bayes harmonization...")
            return self._apply_empirical_bayes_harmonization(
                X_train, y_train, batch_train, X_test, batch_test
            )
    
    def _apply_empirical_bayes_harmonization(self,
                                           X_train: np.ndarray,
                                           y_train: np.ndarray,
                                           batch_train: List[str],
                                           X_test: Optional[np.ndarray] = None,
                                           batch_test: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply empirical Bayes harmonization (simplified ComBat approach)."""
        print("Applying Empirical Bayes harmonization...")
        
        unique_batches = np.unique(batch_train)
        n_features = X_train.shape[1]
        
        # Calculate batch statistics
        batch_means = {}
        batch_vars = {}
        
        for batch in unique_batches:
            batch_mask = np.array(batch_train) == batch
            batch_data = X_train[batch_mask]
            
            batch_means[batch] = np.mean(batch_data, axis=0)
            batch_vars[batch] = np.var(batch_data, axis=0)
        
        # Reference batch statistics
        ref_mean = batch_means.get(self.reference_batch, np.mean(X_train, axis=0))
        ref_var = batch_vars.get(self.reference_batch, np.var(X_train, axis=0))
        
        # Apply harmonization to training data
        X_train_harmonized = X_train.copy()
        for i, batch in enumerate(batch_train):
            if batch != self.reference_batch:
                # Mean centering and variance scaling
                batch_offset = batch_means[batch] - ref_mean
                batch_scale = np.sqrt(ref_var / (batch_vars[batch] + 1e-8))
                
                X_train_harmonized[i] = (X_train[i] - batch_offset) * batch_scale
        
        # Apply to test data if provided
        X_test_harmonized = None
        if X_test is not None and batch_test is not None:
            X_test_harmonized = X_test.copy()
            for i, batch in enumerate(batch_test):
                if batch in batch_means and batch != self.reference_batch:
                    batch_offset = batch_means[batch] - ref_mean
                    batch_scale = np.sqrt(ref_var / (batch_vars[batch] + 1e-8))
                    
                    X_test_harmonized[i] = (X_test[i] - batch_offset) * batch_scale
        
        # Store parameters
        self.harmonization_parameters['batch_means'] = batch_means
        self.harmonization_parameters['batch_vars'] = batch_vars
        self.harmonization_parameters['reference_mean'] = ref_mean
        self.harmonization_parameters['reference_var'] = ref_var
        
        return X_train_harmonized, X_test_harmonized
    
    def _apply_full_bayes_harmonization(self,
                                      X_train: np.ndarray,
                                      y_train: np.ndarray,
                                      batch_train: List[str],
                                      X_test: Optional[np.ndarray] = None,
                                      batch_test: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply full Bayes harmonization (placeholder for future implementation)."""
        print("Full Bayes harmonization not yet implemented. Using empirical Bayes...")
        return self._apply_empirical_bayes_harmonization(
            X_train, y_train, batch_train, X_test, batch_test
        )
    
    def _calculate_batch_statistics(self, X: np.ndarray, batch_labels: List[str]) -> Dict:
        """Calculate batch statistics for parameter storage."""
        batch_stats = {}
        unique_batches = np.unique(batch_labels)
        
        for batch in unique_batches:
            batch_mask = np.array(batch_labels) == batch
            batch_data = X[batch_mask]
            
            batch_stats[batch] = {
                'mean': np.mean(batch_data, axis=0),
                'std': np.std(batch_data, axis=0),
                'count': len(batch_data)
            }
        
        return batch_stats
    
    # ==========================================
    # === CORAL DOMAIN ADAPTATION ===
    # ==========================================
    
    def _apply_coral_adaptation(self, 
                              X_train: np.ndarray, 
                              X_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply CORAL (CORrelation ALignment) domain adaptation."""
        if not self.apply_coral:
            return X_train, X_test
        
        print(f"\n=== Applying CORAL Domain Adaptation (λ={self.coral_lambda}) ===")
        
        try:
            # Calculate source (training) covariance
            X_train_centered = X_train - np.mean(X_train, axis=0)
            C_s = np.cov(X_train_centered.T)
            
            if X_test is not None:
                # Calculate target (test) covariance
                X_test_centered = X_test - np.mean(X_test, axis=0)
                C_t = np.cov(X_test_centered.T)
                
                # CORAL transformation matrix
                # T = C_s^(-1/2) * C_t^(1/2)
                try:
                    # Use SVD for numerical stability
                    U_s, S_s, Vt_s = np.linalg.svd(C_s)
                    C_s_inv_sqrt = U_s @ np.diag(1.0 / np.sqrt(S_s + 1e-8)) @ Vt_s
                    
                    U_t, S_t, Vt_t = np.linalg.svd(C_t)
                    C_t_sqrt = U_t @ np.diag(np.sqrt(S_t + 1e-8)) @ Vt_t
                    
                    coral_transform = C_s_inv_sqrt @ C_t_sqrt
                    
                    # Apply CORAL transformation with regularization
                    I = np.eye(coral_transform.shape[0])
                    regularized_transform = (1 - self.coral_lambda) * I + self.coral_lambda * coral_transform
                    
                    # Transform training data
                    X_train_coral = X_train_centered @ regularized_transform.T
                    X_train_coral += np.mean(X_train, axis=0)  # Add back mean
                    
                    # Transform test data with the same transformation
                    X_test_coral = X_test_centered @ regularized_transform.T
                    X_test_coral += np.mean(X_test, axis=0)  # Add back mean
                    
                    # Store transformation for future use
                    self.coral_transformation = {
                        'transform_matrix': regularized_transform,
                        'source_mean': np.mean(X_train, axis=0),
                        'target_mean': np.mean(X_test, axis=0) if X_test is not None else None
                    }
                    
                    print(f"CORAL transformation applied successfully")
                    return X_train_coral, X_test_coral
                    
                except np.linalg.LinAlgError:
                    print("CORAL transformation failed due to singular matrix. Skipping CORAL.")
                    return X_train, X_test
            else:
                # No test data for CORAL
                print("No test data available for CORAL. Skipping CORAL adaptation.")
                return X_train, X_test
                
        except Exception as e:
            print(f"CORAL adaptation failed: {e}")
            return X_train, X_test
    
    # ==========================================
    # === PREPROCESSING METHODS ===
    # ==========================================
    
    def _apply_snv(self, X: np.ndarray) -> np.ndarray:
        """Apply Standard Normal Variate (SNV) preprocessing."""
        mean = np.mean(X, axis=1, keepdims=True)
        std_dev = np.std(X, axis=1, keepdims=True)
        # Avoid division by zero
        std_dev = np.where(std_dev < 1e-8, 1.0, std_dev)
        return (X - mean) / std_dev
    
    # ==========================================
    # === MAIN DATA PREPARATION METHOD ===
    # ==========================================
    
    def prepare_data(self) -> Dict[str, Any]:
        """
        Prepare train/test data splits with optional harmonization and CORAL adaptation.
        
        Returns:
            Dictionary containing processed data and metadata
        """
        print("\n=== Starting Data Preparation Pipeline ===")
        
        # Step 1: Extract raw spectra data
        print("Step 1: Extracting training data...")
        X_train, y_train, train_info, batch_train = self._extract_spectra_from_patients(self.train_patients)
        
        X_test, y_test, test_info, batch_test = None, None, [], []
        if len(self.test_patients) > 0:
            print("Step 1: Extracting test data...")
            X_test, y_test, test_info, batch_test = self._extract_spectra_from_patients(self.test_patients)
        
        # Store original data for comparison
        X_train_original = X_train.copy()
        X_test_original = X_test.copy() if X_test is not None else None
        
        # Step 2: Apply harmonization if enabled
        if self.enable_harmonization:
            print("Step 2: Applying batch harmonization...")
            X_train, X_test = self._apply_harmonization(
                X_train, y_train, batch_train, X_test, batch_test
            )
        
        # Step 3: Apply CORAL if enabled
        if self.apply_coral:
            print("Step 3: Applying CORAL domain adaptation...")
            X_train, X_test = self._apply_coral_adaptation(X_train, X_test)
        
        # Step 4: Apply SNV if enabled
        if self.apply_snv:
            print("Step 4: Applying SNV preprocessing...")
            X_train = self._apply_snv(X_train)
            if X_test is not None:
                X_test = self._apply_snv(X_test)
        
        # Step 5: Generate summary statistics
        self._print_data_summary(X_train, y_train, batch_train, X_test, y_test, batch_test)
        
        # Prepare return dictionary
        data_split = {
            # Processed data
            'X_train': X_train,
            'X_test': X_test if X_test is not None else np.array([]),
            'y_train': y_train,
            'y_test': y_test if y_test is not None else np.array([]),
            
            # Original data for comparison
            'X_train_original': X_train_original,
            'X_test_original': X_test_original,
            
            # Metadata
            'train_patients': self.train_patients,
            'test_patients': self.test_patients,
            'train_info': train_info,
            'test_info': test_info,
            'batch_train': batch_train,
            'batch_test': batch_test,
            'unified_wavelengths': self.unified_wavelengths,
            'n_features': X_train.shape[1],
            
            # Processing parameters
            'harmonization_parameters': self.harmonization_parameters,
            'coral_transformation': self.coral_transformation,
            'processing_config': self._get_processing_config()
        }
        
        print("=== Data Preparation Pipeline Completed ===\n")
        return data_split
    
    def _print_data_summary(self, X_train, y_train, batch_train, X_test, y_test, batch_test):
        """Print comprehensive data summary."""
        print(f"\n=== Data Summary ===")
        print(f"Training set: {X_train.shape[0]} spectra, {X_train.shape[1]} features")
        if X_test is not None:
            print(f"Test set: {X_test.shape[0]} spectra, {X_test.shape[1]} features")
        
        # Label distribution
        unique_train_labels, train_counts = np.unique(y_train, return_counts=True)
        print("\nTraining set label distribution:")
        for label, count in zip(unique_train_labels, train_counts):
            print(f"  {label}: {count}")
        
        if y_test is not None and len(y_test) > 0:
            unique_test_labels, test_counts = np.unique(y_test, return_counts=True)
            print("\nTest set label distribution:")
            for label, count in zip(unique_test_labels, test_counts):
                print(f"  {label}: {count}")
        
        # Batch distribution
        unique_train_batches, train_batch_counts = np.unique(batch_train, return_counts=True)
        print(f"\nTraining set batch distribution ({self.batch_key}):")
        for batch, count in zip(unique_train_batches, train_batch_counts):
            print(f"  {batch}: {count}")
        
        if batch_test and len(batch_test) > 0:
            unique_test_batches, test_batch_counts = np.unique(batch_test, return_counts=True)
            print(f"\nTest set batch distribution ({self.batch_key}):")
            for batch, count in zip(unique_test_batches, test_batch_counts):
                print(f"  {batch}: {count}")
    
    def _get_processing_config(self) -> Dict:
        """Get current processing configuration."""
        return {
            'enable_harmonization': self.enable_harmonization,
            'harmonization_method': self.harmonization_method,
            'apply_coral': self.apply_coral,
            'coral_lambda': self.coral_lambda,
            'apply_snv': self.apply_snv,
            'reference_batch': self.reference_batch,
            'use_batch': self.use_batch
        }
    
    # ==========================================
    # === EXTERNAL DATA PREDICTION METHODS ===
    # ==========================================
    
    def prepare_external_data(self, 
                            new_raman_data: List[Tuple[Dict, str]],
                            harmonization_params: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare external Raman data for prediction using stored harmonization parameters.
        
        Args:
            new_raman_data: List of (spectrum_data_dict, desired_label) tuples
            harmonization_params: Pre-computed harmonization parameters
            
        Returns:
            Tuple of (X_external, y_external, batch_labels)
        """
        print(f"\n=== Preparing External Data for Prediction ===")
        print(f"Processing {len(new_raman_data)} external spectra...")
        
        # Extract spectra from new data
        spectra_list = []
        labels_list = []
        batch_labels = []
        
        for spectrum_data, desired_label in new_raman_data:
            df = spectrum_data['dataframe']
            metadata = spectrum_data['metadata']
            
            # Interpolate to unified wavelength grid
            df_sorted = df.sort_values('wavelength')
            wavelengths = df_sorted['wavelength'].values
            intensities = df_sorted['intensity'].values
            
            unified_intensities = self._interpolate_spectrum(wavelengths, intensities)
            
            spectra_list.append(unified_intensities)
            labels_list.append(desired_label)
            batch_labels.append(metadata.get(self.batch_key, 'Unknown'))
        
        X_external = np.array(spectra_list)
        y_external = np.array(labels_list)
        
        # Apply stored harmonization parameters
        harmonization_params = harmonization_params or self.harmonization_parameters
        if self.enable_harmonization and harmonization_params:
            X_external = self._apply_external_harmonization(X_external, batch_labels, harmonization_params)
        
        # Apply CORAL if transformation is available
        if self.apply_coral and self.coral_transformation:
            X_external = self._apply_external_coral(X_external)
        
        # Apply SNV if enabled
        if self.apply_snv:
            X_external = self._apply_snv(X_external)
        
        print(f"External data preparation completed: {X_external.shape}")
        print(f"Label distribution: {dict(zip(*np.unique(y_external, return_counts=True)))}")
        print(f"Batch distribution: {dict(zip(*np.unique(batch_labels, return_counts=True)))}")
        
        return X_external, y_external, batch_labels
    
    def _apply_external_harmonization(self, 
                                    X_external: np.ndarray, 
                                    batch_labels: List[str],
                                    harmonization_params: Dict) -> np.ndarray:
        """Apply harmonization to external data using stored parameters."""
        print("Applying stored harmonization parameters to external data...")
        
        if harmonization_params.get('method') == 'empirical_bayes':
            batch_means = harmonization_params.get('batch_means', {})
            batch_vars = harmonization_params.get('batch_vars', {})
            ref_mean = harmonization_params.get('reference_mean')
            ref_var = harmonization_params.get('reference_var')
            
            if ref_mean is not None and ref_var is not None:
                X_harmonized = X_external.copy()
                
                for i, batch in enumerate(batch_labels):
                    if batch in batch_means and batch != self.reference_batch:
                        batch_offset = batch_means[batch] - ref_mean
                        batch_scale = np.sqrt(ref_var / (batch_vars[batch] + 1e-8))
                        X_harmonized[i] = (X_external[i] - batch_offset) * batch_scale
                
                return X_harmonized
        
        print("Could not apply harmonization to external data. Using original data.")
        return X_external
    
    def _apply_external_coral(self, X_external: np.ndarray) -> np.ndarray:
        """Apply stored CORAL transformation to external data."""
        print("Applying stored CORAL transformation to external data...")
        
        try:
            transform_matrix = self.coral_transformation['transform_matrix']
            source_mean = self.coral_transformation['source_mean']
            
            # Center and transform
            X_centered = X_external - source_mean
            X_transformed = X_centered @ transform_matrix.T
            X_transformed += source_mean
            
            return X_transformed
            
        except Exception as e:
            print(f"Could not apply CORAL transformation: {e}")
            return X_external
    
    # ==========================================
    # === VISUALIZATION AND ANALYSIS ===
    # ==========================================
    def create_before_after_comparison(self, data_split: Dict, **kwargs):
        """
        Create comprehensive before/after comparison plots using visualization utilities.
        
        Args:
            data_split: Dictionary containing processed data from prepare_data()
            **kwargs: Additional parameters for visualization
            
        Returns:
            Visualization results from RamanSpectrumAdvanceVisualize
        """
        try:
            # Import your visualization module
            from notebook_utils.visualize import RamanSpectrumAdvanceVisualize
            
            # Initialize visualizer with original raman_data
            visualizer = RamanSpectrumAdvanceVisualize(self.raman_data)
            
            # Extract parameters from kwargs (remove duplicates)
            comparison_type = kwargs.pop('comparison_type', 'both')  # Remove from kwargs to avoid duplicate
            figsize = kwargs.pop('figsize', (15, 10))              # Remove from kwargs to avoid duplicate
            save_path = kwargs.pop('save_path', None)               # Remove from kwargs to avoid duplicate
            
            print("Creating harmonization visualization...")
            
            # Plot before/after comparison
            fig1, axes1 = visualizer.plot_harmonization_comparison(
                data_split=data_split,
                comparison_type=comparison_type,
                figsize=figsize,
                save_path=save_path,
                **kwargs  # Now kwargs won't have duplicate parameters
            )
            
            # Plot quantitative metrics
            metrics_save_path = save_path.replace('.png', '_metrics.png') if save_path else None
            fig2, axes2 = visualizer.plot_batch_effect_metrics(
                data_split=data_split,
                figsize=(12, 8),
                save_path=metrics_save_path
            )
            
            print("Harmonization visualization completed successfully!")
            
            return {
                'comparison_figure': fig1,
                'comparison_axes': axes1,
                'metrics_figure': fig2,
                'metrics_axes': axes2,
                'visualizer': visualizer
            }
            
        except ImportError as e:
            print(f"Visualization module import failed: {e}")
            print("Please ensure visualize.py is in the correct location")
            return None
        except Exception as e:
            print(f"Visualization failed: {e}")
            print(traceback.format_exc())  # This will show the full traceback
            return None


    # ==========================================
    # === UTILITY METHODS ===
    # ==========================================
    
    def save_harmonization_parameters(self, filepath: str) -> None:
        """Save harmonization parameters for future use with external data."""
        params_to_save = {
            'harmonization_parameters': self.harmonization_parameters,
            'coral_transformation': self.coral_transformation,
            'processing_config': self._get_processing_config(),
            'wavelength_range': self.wavelength_range,
            'unified_wavelengths': self.unified_wavelengths
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(params_to_save, f)
        
        print(f"Harmonization parameters saved to {filepath}")
    
    @classmethod
    def load_harmonization_parameters(cls, filepath: str) -> Dict:
        """Load previously saved harmonization parameters."""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        print(f"Harmonization parameters loaded from {filepath}")
        return params
    
    def get_patient_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about patients and spectra distribution."""
        stats = {
            'total_patients': len(self.all_patients),
            'train_patients': len(self.train_patients),
            'test_patients': len(self.test_patients),
            'patients_per_type': {},
            'processing_enabled': {
                'harmonization': self.enable_harmonization,
                'coral': self.apply_coral,
                'snv': self.apply_snv
            }
        }
        
        for data_type in self.selected_types:
            type_patients = [p for p in self.all_patients if p[0] == data_type]
            total_spectra = 0
            
            for _, patient_id, _ in type_patients:
                if patient_id in self.raman_data.get(data_type, {}):
                    total_spectra += len(self.raman_data[data_type][patient_id])
            
            stats['patients_per_type'][data_type] = {
                'patients': len(type_patients),
                'spectra': total_spectra
            }
        
        return stats


if __name__ == "__main__":
    # BELOW ARE HOW I USE THE CLASS
    _ = RamanDataPreparer(
        raman_data=raman_data,
        selected_types=['MGUS', 'MM'],
        test_size=0.15,
        random_state=40,
        label_key='type',
        apply_snv=False,
        
        # New harmonization parameters
        enable_harmonization=False,
        use_batch={'Hikkoshi': 'A'},  # Reference to batch A
        harmonization_method='empirical_bayes',  # Options: 'neurocombat', 'empirical_bayes', 'full_bayes'
        harmonization_reference_strategy='within_type',  # 'within_type', 'across_types', 'external'
        apply_coral=False,
        coral_lambda=1.0
    )
    # please keep data_split as the variable name
    data_split = _.prepare_data()

    # Visualization for before/after harmonization
    _ = _.create_before_after_comparison(
        data_split=data_split,
        comparison_type='both',  # Show both PCA and UMAP
        figsize=(16, 12),
        n_neighbors=15,  # UMAP parameter
        min_dist=0.1     # UMAP parameter
    )

    # Prepare external data with same harmonization
    external_preparer = RamanDataPreparer(
        raman_data=raman_data,
        selected_types=['MGUSnew', 'MMnew'],
        test_size=0,
        random_state=42,
        label_key='type',
        batch_key='Hikkoshi',
        apply_snv=False,
        
        # Load harmonization parameters
        enable_harmonization=True,
        external_harmonization_params=RamanDataPreparer.load_harmonization_parameters('harmonization_params.pkl'),
        apply_coral=True
    )
    # please keep data_split_predict as the variable name
    data_split_predict = external_preparer.prepare_data()

    # Modify labels as needed
    data_split_predict['y_train'] = np.where(
        data_split_predict['y_train'] == 'MMnew', 'MM', 
        data_split_predict['y_train']
    )
    data_split_predict['y_train'] = np.where(
    data_split_predict['y_train'] == 'MGUSnew', 'MGUS', 
    data_split_predict['y_train']
)