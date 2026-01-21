import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.interpolate import interp1d
import traceback
import warnings

class CustomRamanHarmonizer:
    """
    Custom harmonization algorithms specifically designed for Raman spectroscopy data.
    
    Based on established research in batch effect correction adapted for spectroscopic data:
    - Johnson et al. (2007): Empirical Bayes methods for microarray data
    - Fortin et al. (2017): Harmonization across scanners and sites
    - Robust statistical methods for spectroscopy harmonization
    """
    
    def __init__(self, 
                 spectra: np.ndarray,
                 labels: np.ndarray,
                 batch_labels: List[str],
                 covariates: Optional[np.ndarray] = None,
                 reference_batch: str = 'A',
                 custom_method: str = 'empirical_bayes'):
        """
        Initialize custom harmonizer.
        
        Args:
            spectra: Raman spectra (samples x wavenumbers)
            labels: Disease labels (MGUS/MM)
            batch_labels: Batch identifiers (A/B)
            covariates: Biological covariates to preserve
            reference_batch: Reference batch for alignment
            custom_method: Harmonization algorithm to use
        """
        self.spectra = spectra
        self.labels = labels
        self.batch_labels = np.array(batch_labels)
        self.covariates = covariates if covariates is not None else labels
        self.reference_batch = reference_batch
        self.custom_method = custom_method
        
        print(f"Custom Raman Harmonizer initialized:")
        print(f"  - Method: {custom_method}")
        print(f"  - Data shape: {spectra.shape}")
        print(f"  - Reference batch: {reference_batch}")
    
    def harmonize(self) -> np.ndarray:
        """Apply the selected custom harmonization method."""
        
        method_map = {
            'empirical_bayes': self._empirical_bayes_harmonization,
            'quantile_matching': self._quantile_matching_harmonization,
            'zscore_alignment': self._zscore_alignment_harmonization,
            'robust_scaling': self._robust_scaling_harmonization,
            'spectral_warping': self._spectral_warping_harmonization
        }
        
        if self.custom_method not in method_map:
            print(f"Unknown method {self.custom_method}, using empirical_bayes")
            self.custom_method = 'empirical_bayes'
        
        harmonization_func = method_map[self.custom_method]
        
        try:
            harmonized_spectra = harmonization_func()
            print(f"Custom harmonization completed successfully using {self.custom_method}")
            return harmonized_spectra
        except Exception as e:
            print(f"Custom harmonization failed: {e}")
            print("Full traceback:")
            print(traceback.format_exc())
            print("Falling back to simple mean centering...")
            return self._simple_mean_centering()
    
    def _empirical_bayes_harmonization(self) -> np.ndarray:
        """
        Empirical Bayes harmonization inspired by ComBat methodology.
        
        Mathematical framework:
        Y_ijg = α_g + X*β_g + γ_ig + δ_ig*ε_ijg
        
        Reference: Johnson et al. (2007) "Adjusting batch effects in microarray 
        expression data using empirical Bayes methods"
        """
        print("  Applying Empirical Bayes harmonization...")
        
        n_samples, n_features = self.spectra.shape
        unique_batches = np.unique(self.batch_labels)
        n_batches = len(unique_batches)
        
        harmonized_data = np.zeros_like(self.spectra)
        
        # Create design matrix for biological covariates
        if self.covariates is not None:
            unique_labels = np.unique(self.labels)
            X_design = np.zeros((n_samples, len(unique_labels)))
            for i, label in enumerate(unique_labels):
                X_design[:, i] = (self.labels == label).astype(float)
        else:
            X_design = np.ones((n_samples, 1))
        
        # Create batch indicator matrix
        batch_matrix = np.zeros((n_samples, n_batches))
        for i, batch in enumerate(unique_batches):
            batch_matrix[:, i] = (self.batch_labels == batch).astype(float)
        
        # Combined design matrix
        full_design = np.hstack([X_design, batch_matrix])
        
        # Process each wavenumber
        for g in range(n_features):
            if g % 300 == 0:
                print(f"    Processing wavenumber {g+1}/{n_features}")
            
            y_g = self.spectra[:, g]
            
            try:
                # Least squares estimation
                coeffs = np.linalg.lstsq(full_design, y_g, rcond=None)[0]
                
                # Extract effects
                beta_g = coeffs[:X_design.shape[1]]
                batch_effects = coeffs[X_design.shape[1]:]
                
                # Empirical Bayes shrinkage
                batch_var = np.var(batch_effects)
                if batch_var > 1e-8:
                    shrinkage_factor = max(0, 1 - (n_batches - 3) / (n_batches * batch_var))
                    shrunken_batch_effects = shrinkage_factor * batch_effects
                else:
                    shrunken_batch_effects = batch_effects
                
                # Reconstruct harmonized data
                biological_signal = X_design @ beta_g
                
                # Find reference batch index
                ref_idx = np.where(unique_batches == self.reference_batch)[0][0]
                
                for i, batch in enumerate(unique_batches):
                    batch_mask = self.batch_labels == batch
                    if batch == self.reference_batch:
                        harmonized_data[batch_mask, g] = biological_signal[batch_mask]
                    else:
                        batch_effect_removal = shrunken_batch_effects[i] - shrunken_batch_effects[ref_idx]
                        harmonized_data[batch_mask, g] = y_g[batch_mask] - batch_effect_removal
                        
            except np.linalg.LinAlgError:
                # Fallback to simple mean centering
                harmonized_data[:, g] = self._simple_mean_centering_single_feature(y_g)
        
        return harmonized_data
    
    def _quantile_matching_harmonization(self) -> np.ndarray:
        """
        Quantile matching harmonization for spectroscopy.
        
        Reference: Fortin et al. (2017) "Harmonization of cortical thickness 
        measurements across scanners and sites"
        """
        print("  Applying Quantile Matching harmonization...")
        
        harmonized_data = self.spectra.copy()
        unique_batches = np.unique(self.batch_labels)
        
        # Define quantile points
        quantiles = np.linspace(0.01, 0.99, 99)
        
        # Get reference batch data
        ref_mask = self.batch_labels == self.reference_batch
        ref_data = self.spectra[ref_mask]
        
        for batch in unique_batches:
            if batch == self.reference_batch:
                continue
                
            batch_mask = self.batch_labels == batch
            batch_data = self.spectra[batch_mask]
            
            print(f"    Harmonizing batch {batch} to reference {self.reference_batch}")
            
            # Process each wavenumber
            for g in range(self.spectra.shape[1]):
                if g % 400 == 0:
                    print(f"      Processing wavenumber {g+1}/{self.spectra.shape[1]}")
                
                # Get quantiles
                ref_quantiles = np.percentile(ref_data[:, g], quantiles * 100)
                batch_quantiles = np.percentile(batch_data[:, g], quantiles * 100)
                
                # Handle edge cases
                if len(np.unique(batch_quantiles)) < 3:
                    # Simple shift for degenerate cases
                    shift = np.mean(ref_data[:, g]) - np.mean(batch_data[:, g])
                    harmonized_data[batch_mask, g] = batch_data[:, g] + shift
                else:
                    # Quantile matching via interpolation
                    try:
                        interp_func = interp1d(
                            batch_quantiles, ref_quantiles,
                            kind='linear', bounds_error=False,
                            fill_value='extrapolate'
                        )
                        harmonized_data[batch_mask, g] = interp_func(batch_data[:, g])
                    except Exception:
                        # Fallback to simple shift
                        shift = np.mean(ref_data[:, g]) - np.mean(batch_data[:, g])
                        harmonized_data[batch_mask, g] = batch_data[:, g] + shift
        
        return harmonized_data
    
    def _zscore_alignment_harmonization(self) -> np.ndarray:
        """Z-score alignment harmonization."""
        print("  Applying Z-score Alignment harmonization...")
        
        harmonized_data = np.zeros_like(self.spectra)
        unique_batches = np.unique(self.batch_labels)
        
        # Reference batch statistics
        ref_mask = self.batch_labels == self.reference_batch
        ref_data = self.spectra[ref_mask]
        ref_mean = np.mean(ref_data, axis=0)
        ref_std = np.std(ref_data, axis=0)
        ref_std[ref_std < 1e-8] = 1.0  # Avoid division by zero
        
        for batch in unique_batches:
            batch_mask = self.batch_labels == batch
            batch_data = self.spectra[batch_mask]
            
            if batch == self.reference_batch:
                harmonized_data[batch_mask] = batch_data
            else:
                print(f"    Harmonizing batch {batch}")
                
                # Batch statistics
                batch_mean = np.mean(batch_data, axis=0)
                batch_std = np.std(batch_data, axis=0)
                batch_std[batch_std < 1e-8] = 1.0
                
                # Z-score normalize then rescale
                z_scores = (batch_data - batch_mean) / batch_std
                harmonized_data[batch_mask] = z_scores * ref_std + ref_mean
        
        return harmonized_data
    
    def _robust_scaling_harmonization(self) -> np.ndarray:
        """Robust scaling using median and MAD."""
        print("  Applying Robust Scaling harmonization...")
        
        harmonized_data = np.zeros_like(self.spectra)
        unique_batches = np.unique(self.batch_labels)
        
        # Reference batch robust statistics
        ref_mask = self.batch_labels == self.reference_batch
        ref_data = self.spectra[ref_mask]
        ref_median = np.median(ref_data, axis=0)
        ref_mad = np.median(np.abs(ref_data - ref_median), axis=0)
        ref_mad[ref_mad < 1e-8] = 1.0  # Avoid division by zero
        
        for batch in unique_batches:
            batch_mask = self.batch_labels == batch
            batch_data = self.spectra[batch_mask]
            
            if batch == self.reference_batch:
                harmonized_data[batch_mask] = batch_data
            else:
                print(f"    Harmonizing batch {batch}")
                
                # Batch robust statistics
                batch_median = np.median(batch_data, axis=0)
                batch_mad = np.median(np.abs(batch_data - batch_median), axis=0)
                batch_mad[batch_mad < 1e-8] = 1.0
                
                # Robust normalize then rescale
                normalized = (batch_data - batch_median) / batch_mad
                harmonized_data[batch_mask] = normalized * ref_mad + ref_median
        
        return harmonized_data
    
    def _spectral_warping_harmonization(self) -> np.ndarray:
        """Spectral warping for instrument-specific corrections."""
        print("  Applying Spectral Warping harmonization...")
        
        harmonized_data = self.spectra.copy()
        unique_batches = np.unique(self.batch_labels)
        
        # Reference batch spectrum
        ref_mask = self.batch_labels == self.reference_batch
        ref_data = self.spectra[ref_mask]
        ref_mean_spectrum = np.mean(ref_data, axis=0)
        
        for batch in unique_batches:
            if batch == self.reference_batch:
                continue
                
            batch_mask = self.batch_labels == batch
            batch_data = self.spectra[batch_mask]
            batch_mean_spectrum = np.mean(batch_data, axis=0)
            
            print(f"    Harmonizing batch {batch}")
            
            # Baseline correction
            ref_baseline = np.percentile(ref_mean_spectrum, 10)
            batch_baseline = np.percentile(batch_mean_spectrum, 10)
            baseline_offset = batch_baseline - ref_baseline
            
            # Intensity scaling
            ref_intensity = np.percentile(ref_mean_spectrum, 90)
            batch_intensity = np.percentile(batch_mean_spectrum, 90)
            intensity_scale = ref_intensity / (batch_intensity + 1e-8)
            
            # Apply corrections
            harmonized_data[batch_mask] = (batch_data - baseline_offset) * intensity_scale
        
        return harmonized_data
    
    def _simple_mean_centering(self) -> np.ndarray:
        """Simple mean centering fallback method."""
        print("  Applying simple mean centering...")
        
        harmonized_data = self.spectra.copy()
        unique_batches = np.unique(self.batch_labels)
        
        batch_means = {}
        for batch in unique_batches:
            batch_mask = self.batch_labels == batch
            batch_means[batch] = np.mean(self.spectra[batch_mask], axis=0)
        
        ref_mean = batch_means.get(self.reference_batch, np.mean(self.spectra, axis=0))
        
        for i, batch in enumerate(self.batch_labels):
            offset = batch_means[batch] - ref_mean
            harmonized_data[i] = self.spectra[i] - offset
        
        return harmonized_data
    
    def _simple_mean_centering_single_feature(self, y: np.ndarray) -> np.ndarray:
        """Simple mean centering for a single feature."""
        harmonized = y.copy()
        unique_batches = np.unique(self.batch_labels)
        
        batch_means = {}
        for batch in unique_batches:
            batch_mask = self.batch_labels == batch
            batch_means[batch] = np.mean(y[batch_mask])
        
        ref_mean = batch_means.get(self.reference_batch, np.mean(y))
        
        for i, batch in enumerate(self.batch_labels):
            offset = batch_means[batch] - ref_mean
            harmonized[i] = y[i] - offset
        
        return harmonized


class RamanBatchHarmonizer:
    """
    Unified Raman batch effect harmonization class with advanced parameter extraction and reuse.
    
    Supports:
    - Multiple harmonization methods (neuroCombat, pycombat, combat, custom, etc.)
    - Flexible batch handling (simple lists or complex dictionaries)
    - Parameter extraction and reuse for new data
    - Comprehensive visualization and metrics
    - Backward compatibility with existing code
    """
    
    def __init__(self, 
                 spectra: np.ndarray, 
                 labels: np.ndarray, 
                 batch_labels: Union[List[str], Dict[str, List[str]]], 
                 covariates: Optional[np.ndarray] = None,
                 reference_batch: str = 'A',
                 metadata_batch_settings: Dict[str, str] = None,
                 method: str = 'custom_harmonizer',
                 custom_method: str = 'empirical_bayes'):
        """
        Initialize the unified batch harmonizer.
        
        Args:
            spectra: Raman spectra data (samples x features)
            labels: Disease labels (MGUS/MM)
            batch_labels: Batch identifiers - either:
                - List[str]: Simple batch labels ['A', 'B', 'A', ...]
                - Dict[str, List[str]]: Complex batch dict {'Hikkoshi': ['A', 'B', ...], 'Site': [...]}
            covariates: Biological covariates to preserve
            reference_batch: Reference batch identifier
            metadata_batch_settings: Batch reference settings {'batch_key': 'reference_value'}
            method: Harmonization method
                - 'neuroCombat': NeuroCombat harmonization
                - 'pycombat': pyCombat (CoAxLab) harmonization
                - 'combat': Combat (epigenelabs) harmonization
                - 'recombat': reComBat harmonization
                - 'inmoose': inmoose harmonization
                - 'custom_harmonizer': Custom Raman harmonization
                - 'simple': Simple mean centering
            custom_method: Custom algorithm selection
                - 'empirical_bayes': Empirical Bayes harmonization
                - 'quantile_matching': Quantile matching harmonization
                - 'zscore_alignment': Z-score alignment harmonization
                - 'robust_scaling': Robust scaling harmonization
                - 'spectral_warping': Spectral warping harmonization
        """
        self.spectra = spectra
        self.labels = labels
        self.covariates = covariates if covariates is not None else labels
        self.reference_batch = reference_batch
        self.method = method
        self.custom_method = custom_method
        
        # Handle flexible batch labels input
        self._process_batch_labels(batch_labels, metadata_batch_settings)
        
        # Store original data for comparison
        self.original_spectra = spectra.copy()
        self.harmonized_spectra = None
        
        # Advanced parameters for extraction and reuse
        self.batch_parameters = {}
        self.harmonization_metadata = {}
        self.method_specific_params = {}
        
        self._print_harmonizer_info()
    
    def _process_batch_labels(self, batch_labels, metadata_batch_settings):
        """Process and validate batch labels input."""
        
        if isinstance(batch_labels, list):
            # Simple batch labels - backward compatibility
            self.batch_labels = batch_labels
            self.batch_labels_dict = {'batch': batch_labels}
            self.primary_batch_key = 'batch'
            self.metadata_batch_settings = metadata_batch_settings or {'batch': self.reference_batch}
            
        elif isinstance(batch_labels, dict):
            # Advanced batch labels dictionary
            self.batch_labels_dict = batch_labels
            self.primary_batch_key = list(batch_labels.keys())[0]
            self.batch_labels = batch_labels[self.primary_batch_key]
            self.metadata_batch_settings = metadata_batch_settings or {self.primary_batch_key: self.reference_batch}
            
            # Update reference batch from settings if provided
            if metadata_batch_settings and self.primary_batch_key in metadata_batch_settings:
                self.reference_batch = metadata_batch_settings[self.primary_batch_key]
        else:
            raise ValueError("batch_labels must be either List[str] or Dict[str, List[str]]")
        
        # Validate reference batch exists
        unique_batches = np.unique(self.batch_labels)
        if self.reference_batch not in unique_batches:
            warnings.warn(f"Reference batch '{self.reference_batch}' not found. Using '{unique_batches[0]}'")
            self.reference_batch = unique_batches[0]
            if hasattr(self, 'metadata_batch_settings'):
                self.metadata_batch_settings[self.primary_batch_key] = self.reference_batch
    
    def harmonize(self) -> np.ndarray:
        """Apply batch harmonization using the specified method."""
        print(f"Applying harmonization method: {self.method}")
        
        if self.method == 'custom_harmonizer':
            self.harmonized_spectra = self._apply_custom_harmonizer()
        elif self.method == 'neuroCombat':
            self.harmonized_spectra = self._apply_neurocombat()
        elif self.method == 'pycombat':
            self.harmonized_spectra = self._apply_pycombat()
        elif self.method == 'combat':
            self.harmonized_spectra = self._apply_combat()
        elif self.method == 'recombat':
            self.harmonized_spectra = self._apply_recombat()
        elif self.method == 'simple':
            self.harmonized_spectra = self._apply_simple_alignment()
        elif self.method == 'inmoose':
            self.harmonized_spectra = self._apply_inmoose_combat()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Extract parameters after harmonization
        self._extract_batch_parameters()
        
        return self.harmonized_spectra
    
    ## HARMONIZATION METHODS ##
    def _apply_custom_harmonizer(self) -> np.ndarray:
        """Apply custom harmonization algorithms specifically designed for Raman spectroscopy."""
        print("Applying custom Raman harmonization...")
        
        try:
            # Try to import custom harmonizer
            custom_harmonizer = CustomRamanHarmonizer(
                spectra=self.spectra,
                labels=self.labels,
                batch_labels=self.batch_labels,
                covariates=self.covariates,
                reference_batch=self.reference_batch,
                custom_method=self.custom_method
            )
            
            harmonized_spectra = custom_harmonizer.harmonize()
            
            # Extract parameters if available
            if hasattr(custom_harmonizer, 'batch_parameters'):
                self.batch_parameters.update(custom_harmonizer.batch_parameters)
            
            print("Custom harmonization completed successfully.")
            return harmonized_spectra
            
        except ImportError:
            print("Custom harmonizer not available. Falling back to simple alignment...")
            return self._apply_simple_alignment()
        except Exception as e:
            print(f"Custom harmonization failed: {e}")
            print("Falling back to simple alignment...")
            return self._apply_simple_alignment()
    
    def _apply_neurocombat(self) -> np.ndarray:
        """Apply neuroCombat harmonization with parameter extraction."""
        try:
            from neuroCombat import neuroCombat
            
            print("Applying neuroCombat harmonization...")
            
            # Prepare data (features x samples for neuroCombat)
            data = self.spectra.T
            
            # Prepare covariates DataFrame - ensure consistent data types
            covars_dict = {'batch': list(self.batch_labels)}  # Convert to list to avoid Series issues
            if self.covariates is not None:
                if hasattr(self.covariates, 'tolist'):  # Handle numpy arrays
                    covariates_list = self.covariates.tolist()
                else:
                    covariates_list = list(self.covariates)  # Convert to list
                
                unique_covariates = np.unique(covariates_list)
                if len(unique_covariates) <= 10:  # Categorical
                    covars_dict['diagnosis'] = covariates_list
                else:  # Continuous
                    covars_dict['diagnosis'] = covariates_list
            
            # Ensure all values are the same length
            batch_len = len(covars_dict['batch'])
            if 'diagnosis' in covars_dict:
                diag_len = len(covars_dict['diagnosis'])
                if batch_len != diag_len:
                    print(f"Warning: Batch length ({batch_len}) != Diagnosis length ({diag_len})")
                    min_len = min(batch_len, diag_len)
                    covars_dict['batch'] = covars_dict['batch'][:min_len]
                    covars_dict['diagnosis'] = covars_dict['diagnosis'][:min_len]
                    # Also trim data to match
                    data = data[:, :min_len]
            
            covars = pd.DataFrame(covars_dict)
            categorical_cols = ['diagnosis'] if self.covariates is not None else []
            
            print(f"neuroCombat input shapes: data {data.shape}, covars {covars.shape}")
            print(f"Covariates columns: {list(covars.columns)}")
            print(f"Categorical columns: {categorical_cols}")
            print(f"Reference batch: {self.reference_batch}")
            
            # Apply neuroCombat
            result = neuroCombat(
                dat=data,
                covars=covars,
                batch_col='batch',
                categorical_cols=categorical_cols,
                ref_batch=self.reference_batch
            )
            
            # Extract and store neuroCombat-specific parameters
            if 'estimates' in result:
                self.method_specific_params['neuroCombat_estimates'] = result['estimates']
            if 'info' in result:
                self.method_specific_params['neuroCombat_info'] = result['info']
            
            print("neuroCombat harmonization completed successfully.")
            return result["data"].T
            
        except ImportError as e:
            print(f"neuroCombat not available: {e}")
            print("Install with: pip install neuroCombat")
            print("Falling back to simple alignment...")
            return self._apply_simple_alignment()
        except Exception as e:
            print(f"neuroCombat failed with error: {e}")
            print("Full traceback:")
            print(traceback.format_exc())
            print("Falling back to simple alignment...")
            return self._apply_simple_alignment()
    
    def _apply_pycombat(self) -> np.ndarray:
        """Apply pyCombat using class-based approach."""
        try:
            from pycombat import Combat
            
            print("Applying pyCombat (CoAxLab) harmonization...")
            
            # Prepare data matrices
            Y = pd.DataFrame(self.spectra)  # samples x features
            b = list(self.batch_labels)  # Ensure it's a list, not numpy array
            
            # Prepare design matrix X for effects of interest
            if hasattr(self.labels, 'tolist'):  # Handle numpy arrays
                labels_list = self.labels.tolist()
            else:
                labels_list = list(self.labels)
            
            diagnosis_series = pd.Series(labels_list)
            X = pd.get_dummies(diagnosis_series, drop_first=True)
            C = None
            
            print(f"pyCombat input shapes: Y {Y.shape}, batch {len(b)}")
            print(f"Effects of interest (X) shape: {X.shape}")
            print(f"Batch labels unique values: {np.unique(b)}")
            print(f"Diagnosis labels unique values: {np.unique(self.labels)}")
            
            # Apply pyCombat
            combat = Combat()
            combat.fit(Y=Y, b=b, X=X, C=C)
            Y_adjusted = combat.transform(Y=Y, b=b, X=X, C=C)
            
            # Store pyCombat-specific parameters
            if hasattr(combat, 'estimates_'):
                self.method_specific_params['pycombat_estimates'] = combat.estimates_
            
            print("pyCombat harmonization completed successfully.")
            return Y_adjusted.values
            
        except ImportError as e:
            print(f"pycombat not available: {e}")
            print("Trying alternative Combat implementations...")
            return self._apply_combat()
        except Exception as e:
            print(f"pyCombat failed: {e}")
            print("Full traceback:")
            print(traceback.format_exc())
            print("Trying alternative Combat implementations...")
            return self._apply_combat()
    
    def _apply_combat(self) -> np.ndarray:
        """Apply Combat from combat package."""
        try:
            from combat.pycombat import pycombat
            
            print("Applying Combat (epigenelabs) harmonization...")
            
            # Prepare data
            data_df = pd.DataFrame(self.spectra.T)
            batch_series = pd.Series(self.batch_labels, name='batch')
            
            # Prepare covariates
            mod_df = None
            if self.covariates is not None:
                unique_covariates = np.unique(self.covariates)
                if len(unique_covariates) <= 10:  # Categorical
                    covariate_dummies = pd.get_dummies(self.covariates, prefix='diagnosis')
                    mod_df = covariate_dummies
                else:  # Continuous
                    mod_df = pd.DataFrame({'diagnosis': self.covariates})
            
            print(f"Combat input shapes: data {data_df.shape}, batch {len(batch_series)}")
            
            # Try different API variations
            try:
                harmonized_df = pycombat(
                    data=data_df,
                    batch=batch_series,
                    mod=mod_df,
                    ref_batch=self.reference_batch
                )
            except TypeError as e:
                if "ref_batch" in str(e):
                    harmonized_df = pycombat(
                        data=data_df,
                        batch=batch_series,
                        mod=mod_df,
                        ref=self.reference_batch
                    )
                else:
                    harmonized_df = pycombat(data_df, batch_series, mod_df)
            
            print("Combat harmonization completed successfully.")
            return harmonized_df.T.values
            
        except ImportError as e:
            print(f"combat package not available: {e}")
            print("Trying inmoose implementation...")
            return self._apply_inmoose_combat()
        except Exception as e:
            print(f"Combat failed: {e}")
            print("Trying inmoose implementation...")
            return self._apply_inmoose_combat()
    
    def _apply_inmoose_combat(self) -> np.ndarray:
        """Apply Combat from inmoose package."""
        try:
            from inmoose.pycombat import pycombat_norm
            
            print("Applying Combat (inmoose) harmonization...")
            
            data_df = pd.DataFrame(self.spectra.T)
            batch_list = self.batch_labels
            
            print(f"Inmoose Combat input shapes: data {data_df.shape}, batch {len(batch_list)}")
            
            harmonized_df = pycombat_norm(data_df, batch_list)
            
            print("Inmoose Combat harmonization completed successfully.")
            return harmonized_df.T.values
            
        except ImportError as e:
            print(f"inmoose not available: {e}")
            print("Trying reComBat...")
            return self._apply_recombat()
        except Exception as e:
            print(f"Inmoose Combat failed: {e}")
            print("Trying reComBat...")
            return self._apply_recombat()
    
    def _apply_recombat(self) -> np.ndarray:
        """Apply reComBat harmonization."""
        try:
            from reComBat import reComBat
            
            print("Applying reComBat harmonization...")
            
            data_df = pd.DataFrame(self.spectra.T)
            batch_series = pd.Series(self.batch_labels, name='batch')
            
            # Prepare covariates
            covariates_df = None
            if self.covariates is not None:
                unique_covariates = np.unique(self.covariates)
                if len(unique_covariates) <= 10:  # Categorical
                    le = LabelEncoder()
                    encoded_covariates = le.fit_transform(self.covariates)
                    covariates_df = pd.DataFrame({'diagnosis': encoded_covariates})
                else:  # Continuous
                    covariates_df = pd.DataFrame({'diagnosis': self.covariates})
            
            print(f"reComBat input shapes: data {data_df.shape}, batch {len(batch_series)}")
            
            harmonized_df = reComBat(
                data=data_df,
                batch=batch_series,
                covariates=covariates_df
            )
            
            print("reComBat harmonization completed successfully.")
            return harmonized_df.T.values
            
        except ImportError as e:
            print(f"reComBat not available: {e}")
            print("Falling back to simple alignment...")
            return self._apply_simple_alignment()
        except Exception as e:
            print(f"reComBat failed: {e}")
            print("Falling back to simple alignment...")
            return self._apply_simple_alignment()
    
    def _apply_simple_alignment(self) -> np.ndarray:
        """Apply simple mean-centering batch alignment with parameter extraction."""
        print("Applying simple batch alignment (mean centering)...")
        
        try:
            unique_batches = np.unique(self.batch_labels)
            batch_means = {}
            batch_stds = {}
            
            # Calculate batch statistics
            for batch in unique_batches:
                batch_indices = np.array(self.batch_labels) == batch
                batch_data = self.spectra[batch_indices]
                batch_means[batch] = np.mean(batch_data, axis=0)
                batch_stds[batch] = np.std(batch_data, axis=0)
                print(f"  Batch {batch}: {np.sum(batch_indices)} samples")
            
            # Validate reference batch
            if self.reference_batch not in batch_means:
                print(f"Warning: Reference batch '{self.reference_batch}' not found. Using first batch.")
                self.reference_batch = unique_batches[0]
            
            ref_mean = batch_means[self.reference_batch]
            print(f"  Using batch {self.reference_batch} as reference")
            
            # Store parameters for reuse
            self.batch_parameters.update({
                'batch_means': batch_means,
                'batch_stds': batch_stds,
                'reference_batch': self.reference_batch,
                'reference_mean': ref_mean
            })
            
            # Apply alignment
            aligned_spectra = self.spectra.copy()
            for i, batch in enumerate(self.batch_labels):
                batch_offset = batch_means[batch] - ref_mean
                aligned_spectra[i] = self.spectra[i] - batch_offset
            
            print("Simple batch alignment completed successfully.")
            return aligned_spectra
            
        except Exception as e:
            print(f"Simple alignment failed with error: {e}")
            print("Full traceback:")
            print(traceback.format_exc())
            print("Returning original data without harmonization...")
            return self.original_spectra.copy()
    
    def _extract_batch_parameters(self):
        """Extract comprehensive batch parameters for reuse with enhanced type-specific handling."""
        if self.harmonized_spectra is None:
            return
        
        unique_batches = np.unique(self.batch_labels)
        unique_labels = np.unique(self.labels)
        
        # Calculate batch-specific statistics
        batch_stats = {}
        for batch in unique_batches:
            batch_mask = np.array(self.batch_labels) == batch
            
            if np.any(batch_mask):
                original_data = self.original_spectra[batch_mask]
                harmonized_data = self.harmonized_spectra[batch_mask]
                batch_labels_subset = np.array(self.labels)[batch_mask]
                
                # General batch statistics
                batch_stats[batch] = {
                    'original_mean': np.mean(original_data, axis=0),
                    'original_std': np.std(original_data, axis=0),
                    'original_median': np.median(original_data, axis=0),
                    'harmonized_mean': np.mean(harmonized_data, axis=0),
                    'harmonized_std': np.std(harmonized_data, axis=0),
                    'harmonized_median': np.median(harmonized_data, axis=0),
                    'n_samples': np.sum(batch_mask)
                }
                
                # Type-specific statistics within each batch
                type_specific_original = {}
                type_specific_harmonized = {}
                
                for label in unique_labels:
                    label_mask = batch_labels_subset == label
                    if np.any(label_mask):
                        original_type_data = original_data[label_mask]
                        harmonized_type_data = harmonized_data[label_mask]
                        
                        type_specific_original[label] = {
                            'mean': np.mean(original_type_data, axis=0),
                            'std': np.std(original_type_data, axis=0),
                            'n_samples': np.sum(label_mask)
                        }
                        
                        type_specific_harmonized[label] = {
                            'mean': np.mean(harmonized_type_data, axis=0),
                            'std': np.std(harmonized_type_data, axis=0),
                            'n_samples': np.sum(label_mask)
                        }
                
                batch_stats[batch]['type_specific_original'] = type_specific_original
                batch_stats[batch]['type_specific_harmonized'] = type_specific_harmonized
                
                # For within_type mode compatibility
                batch_stats[batch]['type_specific_means'] = {
                    label: stats['mean'] 
                    for label, stats in type_specific_harmonized.items()
                }
        
        self.batch_parameters['batch_statistics'] = batch_stats
        
        # Calculate reference transformation parameters
        ref_mask = np.array(self.batch_labels) == self.reference_batch
        if np.any(ref_mask):
            ref_original = self.original_spectra[ref_mask]
            ref_harmonized = self.harmonized_spectra[ref_mask]
            ref_labels = np.array(self.labels)[ref_mask]
            
            self.batch_parameters['reference_transformation'] = {
                'original_mean': np.mean(ref_original, axis=0),
                'original_std': np.std(ref_original, axis=0),
                'harmonized_mean': np.mean(ref_harmonized, axis=0),
                'harmonized_std': np.std(ref_harmonized, axis=0)
            }
            
            # Type-specific reference statistics
            ref_type_specific = {}
            for label in unique_labels:
                label_mask = ref_labels == label
                if np.any(label_mask):
                    ref_type_original = ref_original[label_mask]
                    ref_type_harmonized = ref_harmonized[label_mask]
                    
                    ref_type_specific[label] = {
                        'original_mean': np.mean(ref_type_original, axis=0),
                        'original_std': np.std(ref_type_original, axis=0),
                        'harmonized_mean': np.mean(ref_type_harmonized, axis=0),
                        'harmonized_std': np.std(ref_type_harmonized, axis=0),
                        'n_samples': np.sum(label_mask)
                    }
            
            self.batch_parameters['reference_type_specific'] = ref_type_specific
        
        # Store comprehensive metadata
        self.harmonization_metadata = {
            'method': self.method,
            'custom_method': self.custom_method if self.method == 'custom_harmonizer' else None,
            'reference_batch': self.reference_batch,
            'primary_batch_key': self.primary_batch_key,
            'batch_keys': list(self.batch_labels_dict.keys()) if hasattr(self, 'batch_labels_dict') else ['batch'],
            'metadata_batch_settings': getattr(self, 'metadata_batch_settings', {}),
            'n_samples': len(self.spectra),
            'n_features': self.spectra.shape[1],
            'unique_batches': unique_batches.tolist(),
            'unique_labels': unique_labels.tolist(),
            'batch_counts': dict(zip(*np.unique(self.batch_labels, return_counts=True))),
            'label_counts': dict(zip(*np.unique(self.labels, return_counts=True))),
            'batch_label_distribution': {
                batch: dict(zip(*np.unique(np.array(self.labels)[np.array(self.batch_labels) == batch], return_counts=True)))
                for batch in unique_batches
            }
        }
    
    def get_harmonization_parameters(self) -> Dict[str, Any]:
        """Get complete harmonization parameters for reuse."""
        return {
            'batch_parameters': self.batch_parameters,
            'harmonization_metadata': self.harmonization_metadata,
            'method_specific_params': self.method_specific_params
        }
    
    def save_harmonization_parameters(self, filepath: str):
        """Save harmonization parameters to file."""
        import pickle
        
        try:
            params = self.get_harmonization_parameters()
            with open(filepath, 'wb') as f:
                pickle.dump(params, f)
            print(f"Harmonization parameters saved to {filepath}")
        except Exception as e:
            print(f"Failed to save harmonization parameters: {e}")
    
    def load_harmonization_parameters(self, filepath: str) -> Dict[str, Any]:
        """Load harmonization parameters from file."""
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                params = pickle.load(f)
            print(f"Harmonization parameters loaded from {filepath}")
            return params
        except Exception as e:
            print(f"Failed to load harmonization parameters: {e}")
            return {}
    
    def apply_harmonization_to_new_data(self, 
                                       new_spectra: np.ndarray,
                                       new_batch_labels: Union[List[str], Dict[str, List[str]]],
                                       new_labels: Optional[np.ndarray] = None,
                                       harmonization_params: Dict[str, Any] = None,
                                       data_type_reference_mode: str = 'mixed_reference') -> np.ndarray:
        """
        Apply learned harmonization parameters to new data with enhanced robustness.
        
        Args:
            new_spectra: New spectra to harmonize
            new_batch_labels: Batch labels for new data (same format as training)
            new_labels: Disease labels for new data (optional, for within_type mode)
            harmonization_params: Previously saved harmonization parameters
            data_type_reference_mode: Reference mode ('within_type', 'mixed_reference', etc.)
            
        Returns:
            Harmonized new spectra
        """
        
        params = harmonization_params or self.get_harmonization_parameters()
        
        if 'batch_parameters' not in params:
            warnings.warn("No batch parameters available. Returning original data.")
            return new_spectra
        
        print(f"Applying harmonization to new data with mode: {data_type_reference_mode}")
        
        # Handle batch labels format dynamically
        if isinstance(new_batch_labels, list):
            batch_labels_to_use = new_batch_labels
        elif isinstance(new_batch_labels, dict):
            primary_key = params['harmonization_metadata'].get('primary_batch_key', 'batch')
            batch_labels_to_use = new_batch_labels.get(primary_key, list(new_batch_labels.values())[0])
        else:
            warnings.warn("Invalid batch labels format. Returning original data.")
            return new_spectra
        
        # Get harmonization method from parameters
        method = params['harmonization_metadata'].get('method', 'simple')
        
        print(f"New data batch distribution: {dict(zip(*np.unique(batch_labels_to_use, return_counts=True)))}")
        print(f"Using trained harmonizer method: {method}")
        
        # Apply harmonization based on method and mode
        if method == 'neuroCombat' and 'neuroCombat_estimates' in params.get('method_specific_params', {}):
            return self._apply_neurocombat_to_new_data_enhanced(
                new_spectra, batch_labels_to_use, new_labels, params, data_type_reference_mode
            )
        elif method == 'pycombat' and 'pycombat_estimates' in params.get('method_specific_params', {}):
            return self._apply_pycombat_to_new_data_enhanced(
                new_spectra, batch_labels_to_use, new_labels, params, data_type_reference_mode
            )
        elif 'batch_means' in params['batch_parameters']:
            return self._apply_simple_alignment_to_new_data_enhanced(
                new_spectra, batch_labels_to_use, new_labels, params, data_type_reference_mode
            )
        else:
            print("No applicable harmonization method found for new data. Returning original.")
            return new_spectra
    
    def _apply_simple_alignment_to_new_data_enhanced(self, 
                                                   new_spectra: np.ndarray,
                                                   new_batch_labels: List[str],
                                                   new_labels: Optional[np.ndarray],
                                                   params: Dict[str, Any],
                                                   data_type_reference_mode: str) -> np.ndarray:
        """Apply simple alignment parameters to new data with enhanced mode handling."""
        
        batch_params = params['batch_parameters']
        batch_means = batch_params.get('batch_means', {})
        reference_batch = batch_params.get('reference_batch')
        
        if not batch_means or not reference_batch:
            warnings.warn("Incomplete batch parameters. Returning original data.")
            return new_spectra
        
        aligned_spectra = new_spectra.copy()
        
        print(f"Applying simple alignment to new data (reference: {reference_batch})")
        print(f"Data type reference mode: {data_type_reference_mode}")
        
        if data_type_reference_mode == 'within_type' and new_labels is not None:
            # Handle within_type mode: use type-specific reference if available
            batch_stats = batch_params.get('batch_statistics', {})
            
            for i, (batch, label) in enumerate(zip(new_batch_labels, new_labels)):
                # Try to find type-specific reference within the same batch
                type_specific_ref = None
                
                # Look for reference batch with same disease type
                if reference_batch in batch_stats:
                    ref_data = batch_stats[reference_batch]
                    if 'type_specific_means' in ref_data:
                        type_specific_ref = ref_data['type_specific_means'].get(label)
                
                # Fallback to batch-specific offset
                if batch in batch_means and reference_batch in batch_means:
                    if type_specific_ref is not None:
                        # Use type-specific reference
                        offset = batch_means[batch] - type_specific_ref
                    else:
                        # Use general batch offset
                        offset = batch_means[batch] - batch_means[reference_batch]
                    
                    aligned_spectra[i] = new_spectra[i] - offset
                else:
                    print(f"Warning: Batch '{batch}' not found in training parameters. No adjustment applied.")
        
        else:
            # Standard mixed reference mode
            ref_mean = batch_means[reference_batch]
            
            for i, batch in enumerate(new_batch_labels):
                if batch in batch_means:
                    offset = batch_means[batch] - ref_mean
                    aligned_spectra[i] = new_spectra[i] - offset
                else:
                    print(f"Warning: Batch '{batch}' not found in training parameters. No adjustment applied.")
        
        return aligned_spectra
    
    def _apply_neurocombat_to_new_data_enhanced(self, 
                                              new_spectra: np.ndarray,
                                              new_batch_labels: List[str],
                                              new_labels: Optional[np.ndarray],
                                              params: Dict[str, Any],
                                              data_type_reference_mode: str) -> np.ndarray:
        """Apply neuroCombat parameters to new data with enhanced handling."""
        
        try:
            from neuroCombat import neuroCombat
            
            print("Applying neuroCombat harmonization to new data...")
            print(f"Data type reference mode: {data_type_reference_mode}")
            
            # Get neuroCombat estimates from training
            estimates = params['method_specific_params'].get('neuroCombat_estimates')
            if estimates is None:
                print("neuroCombat estimates not available. Using simple fallback.")
                return self._apply_simple_alignment_to_new_data_enhanced(
                    new_spectra, new_batch_labels, new_labels, params, data_type_reference_mode
                )
            
            # Validate data dimensions first
            print(f"New data shape: {new_spectra.shape}")
            print(f"Number of new batch labels: {len(new_batch_labels)}")
            print(f"Number of new disease labels: {len(new_labels) if new_labels is not None else 'None'}")
            
            # Check if dimensions match
            if new_spectra.shape[0] != len(new_batch_labels):
                print(f"Error: Samples ({new_spectra.shape[0]}) != batch labels ({len(new_batch_labels)})")
                return self._apply_simple_alignment_to_new_data_enhanced(
                    new_spectra, new_batch_labels, new_labels, params, data_type_reference_mode
                )
            
            if new_labels is not None and new_spectra.shape[0] != len(new_labels):
                print(f"Error: Samples ({new_spectra.shape[0]}) != disease labels ({len(new_labels)})")
                return self._apply_simple_alignment_to_new_data_enhanced(
                    new_spectra, new_batch_labels, new_labels, params, data_type_reference_mode
                )
            
            # For neuroCombat new data, we need to be careful about applying pre-computed parameters
            # Currently neuroCombat doesn't have built-in new data transformation with saved parameters
            # So we'll use a more robust fallback approach
            
            print("neuroCombat new data parameter application is complex - using enhanced fallback.")
            print("This provides similar harmonization effects using training statistics.")
            
            return self._apply_enhanced_statistical_harmonization(
                new_spectra, new_batch_labels, new_labels, params, data_type_reference_mode
            )
            
        except Exception as e:
            print(f"neuroCombat new data application failed: {e}")
            print("Full traceback:")
            print(traceback.format_exc())
            print("Using enhanced fallback based on training statistics.")
            
            return self._apply_enhanced_statistical_harmonization(
                new_spectra, new_batch_labels, new_labels, params, data_type_reference_mode
            )
    
    def _apply_pycombat_to_new_data_enhanced(self, 
                                           new_spectra: np.ndarray,
                                           new_batch_labels: List[str],
                                           new_labels: Optional[np.ndarray],
                                           params: Dict[str, Any],
                                           data_type_reference_mode: str) -> np.ndarray:
        """Apply pyCombat parameters to new data with enhanced handling."""
        
        try:
            from pycombat import Combat
            
            print("Applying pyCombat harmonization to new data...")
            
            # Get pyCombat estimates from training
            estimates = params['method_specific_params'].get('pycombat_estimates')
            if estimates is None:
                print("pyCombat estimates not available. Using simple fallback.")
                return self._apply_simple_alignment_to_new_data_enhanced(
                    new_spectra, new_batch_labels, new_labels, params, data_type_reference_mode
                )
            
            # Prepare data matrices
            Y = pd.DataFrame(new_spectra)  # samples x features
            b = list(new_batch_labels)  # Ensure it's a list
            
            # Prepare design matrix X for effects of interest
            X = None
            if new_labels is not None:
                if hasattr(new_labels, 'tolist'):  # Handle numpy arrays
                    labels_list = new_labels.tolist()
                else:
                    labels_list = list(new_labels)
                
                diagnosis_series = pd.Series(labels_list)
                X = pd.get_dummies(diagnosis_series, drop_first=True)
            
            print(f"pyCombat new data input shapes: Y {Y.shape}, batch {len(b)}")
            
            # Create Combat instance and try to use pre-fitted parameters
            # Note: This is a simplified approach - actual pyCombat parameter reuse
            # might require more sophisticated handling
            combat = Combat()
            
            # For now, refit and transform (could be improved with parameter reuse)
            combat.fit(Y=Y, b=b, X=X, C=None)
            Y_adjusted = combat.transform(Y=Y, b=b, X=X, C=None)
            
            print("pyCombat new data harmonization completed successfully.")
            return Y_adjusted.values
            
        except Exception as e:
            print(f"pyCombat new data application failed: {e}")
            print("Full traceback:")
            print(traceback.format_exc())
            print("Using simple fallback based on reference batch statistics.")
            
            return self._apply_simple_alignment_to_new_data_enhanced(
                new_spectra, new_batch_labels, new_labels, params, data_type_reference_mode
            )
    
    def _apply_enhanced_statistical_harmonization(self, 
                                                new_spectra: np.ndarray,
                                                new_batch_labels: List[str],
                                                new_labels: Optional[np.ndarray],
                                                params: Dict[str, Any],
                                                data_type_reference_mode: str) -> np.ndarray:
        """
        Apply enhanced statistical harmonization using training parameters with proper reference mode handling.
        """
        
        print(f"Applying enhanced statistical harmonization with mode: {data_type_reference_mode}")
        
        batch_params = params['batch_parameters']
        harmonization_metadata = params['harmonization_metadata']
        
        # Get training reference information
        reference_batch = harmonization_metadata.get('reference_batch')
        batch_statistics = batch_params.get('batch_statistics', {})
        
        if not batch_statistics or not reference_batch:
            print("Warning: Incomplete training statistics. Using simple mean centering.")
            return self._apply_simple_mean_centering(new_spectra, new_batch_labels)
        
        aligned_spectra = new_spectra.copy()
        
        if data_type_reference_mode == 'within_type' and new_labels is not None:
            print("Applying within_type reference mode for new data...")
            
            # For within_type mode, harmonize each disease type separately using training stats
            for i, (batch, label) in enumerate(zip(new_batch_labels, new_labels)):
                if batch in batch_statistics and reference_batch in batch_statistics:
                    
                    # Get type-specific statistics if available
                    batch_stats = batch_statistics[batch]
                    ref_stats = batch_statistics[reference_batch]
                    
                    # Try to use type-specific harmonization
                    if ('type_specific_harmonized' in batch_stats and 
                        'type_specific_harmonized' in ref_stats and
                        label in batch_stats['type_specific_harmonized'] and
                        label in ref_stats['type_specific_harmonized']):
                        
                        # Use type-specific reference
                        batch_type_mean = batch_stats['type_specific_harmonized'][label]['mean']
                        ref_type_mean = ref_stats['type_specific_harmonized'][label]['mean']
                        offset = batch_type_mean - ref_type_mean
                        
                        print(f"   Using type-specific harmonization for {label} in batch {batch}")
                        
                    else:
                        # Fall back to general batch harmonization
                        batch_mean = batch_stats['harmonized_mean']
                        ref_mean = ref_stats['harmonized_mean']
                        offset = batch_mean - ref_mean
                        
                        print(f"   Using general batch harmonization for {label} in batch {batch}")
                    
                    aligned_spectra[i] = new_spectra[i] - offset
                    
                else:
                    print(f"   Warning: No training statistics for batch {batch}. No adjustment.")
        
        elif data_type_reference_mode == 'mixed_reference':
            print("Applying mixed_reference mode for new data...")
            
            # For mixed reference, use overall batch harmonization from training
            ref_stats = batch_statistics.get(reference_batch)
            if ref_stats is None:
                print(f"Warning: No reference batch {reference_batch} statistics available.")
                return new_spectra
            
            ref_mean = ref_stats['harmonized_mean']
            
            for i, batch in enumerate(new_batch_labels):
                if batch in batch_statistics:
                    batch_stats = batch_statistics[batch]
                    batch_mean = batch_stats['harmonized_mean']
                    offset = batch_mean - ref_mean
                    aligned_spectra[i] = new_spectra[i] - offset
                else:
                    print(f"   Warning: No training statistics for batch {batch}. No adjustment.")
        
        else:
            print(f"Unknown reference mode: {data_type_reference_mode}. Using mixed_reference.")
            return self._apply_enhanced_statistical_harmonization(
                new_spectra, new_batch_labels, new_labels, params, 'mixed_reference'
            )
        
        print("Enhanced statistical harmonization completed.")
        return aligned_spectra
    
    def _apply_simple_mean_centering(self, spectra: np.ndarray, batch_labels: List[str]) -> np.ndarray:
        """Simple fallback: mean centering per batch."""
        
        print("Applying simple mean centering fallback...")
        aligned_spectra = spectra.copy()
        
        unique_batches = list(set(batch_labels))
        if len(unique_batches) <= 1:
            print("Only one batch found. No adjustment needed.")
            return spectra
        
        # Use first batch as reference
        ref_batch = unique_batches[0]
        ref_mask = np.array(batch_labels) == ref_batch
        ref_mean = np.mean(spectra[ref_mask], axis=0)
        
        for batch in unique_batches:
            batch_mask = np.array(batch_labels) == batch
            if np.any(batch_mask):
                batch_mean = np.mean(spectra[batch_mask], axis=0)
                offset = batch_mean - ref_mean
                aligned_spectra[batch_mask] = spectra[batch_mask] - offset
        
        return aligned_spectra
    
    def plot_before_after_pca(self, n_components: int = 2) -> None:
        """Plot PCA before and after harmonization for comparison."""
        try:
            from sklearn.decomposition import PCA
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Before harmonization
            pca_before = PCA(n_components=n_components)
            X_pca_before = pca_before.fit_transform(self.original_spectra)
            
            batch_colors = {'A': '#1f77b4', 'B': '#ff7f0e'}  # Blue, Orange
            
            for batch in np.unique(self.batch_labels):
                mask = np.array(self.batch_labels) == batch
                color = batch_colors.get(batch, f'C{hash(batch) % 10}')
                axes[0].scatter(X_pca_before[mask, 0], X_pca_before[mask, 1], 
                              c=color, label=f'Batch {batch}', alpha=0.7, s=30)
            
            axes[0].set_title('Before Harmonization')
            axes[0].set_xlabel(f'PC1 ({pca_before.explained_variance_ratio_[0]:.2%} variance)')
            axes[0].set_ylabel(f'PC2 ({pca_before.explained_variance_ratio_[1]:.2%} variance)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # After harmonization
            if self.harmonized_spectra is not None:
                pca_after = PCA(n_components=n_components)
                X_pca_after = pca_after.fit_transform(self.harmonized_spectra)
                
                for batch in np.unique(self.batch_labels):
                    mask = np.array(self.batch_labels) == batch
                    color = batch_colors.get(batch, f'C{hash(batch) % 10}')
                    axes[1].scatter(X_pca_after[mask, 0], X_pca_after[mask, 1], 
                                  c=color, label=f'Batch {batch}', alpha=0.7, s=30)
                
                axes[1].set_title('After Harmonization')
                axes[1].set_xlabel(f'PC1 ({pca_after.explained_variance_ratio_[0]:.2%} variance)')
                axes[1].set_ylabel(f'PC2 ({pca_after.explained_variance_ratio_[1]:.2%} variance)')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, 'Harmonization not applied', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('After Harmonization')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"PCA plotting failed: {e}")
            print(traceback.format_exc())
    
    def plot_before_after_umap(self, n_neighbors: int = 35, min_dist: float = 0.05) -> None:
        """Plot UMAP before and after harmonization - colored by disease type with varied batch colors."""
        try:
            import umap
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Before harmonization
            reducer_before = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
            X_umap_before = reducer_before.fit_transform(self.original_spectra)
            
            # Generate dynamic colors for disease-batch combinations
            unique_diseases = np.unique(self.labels)
            unique_batches = np.unique(self.batch_labels)
            
            # Create a colormap with enough distinct colors
            import matplotlib.cm as cm
            colors = cm.Set1(np.linspace(0, 1, len(unique_diseases) * len(unique_batches)))
            
            # Disease base colors with batch variations
            disease_base_colors = {'MGUS': 'red', 'MM': 'blue', 'NL': 'green'}
            batch_markers = {'A': 'o', 'B': 's', 'C': '^', 'D': 'P', 'E': 'X'}
            
            color_idx = 0
            for disease in unique_diseases:
                for batch in unique_batches:
                    mask = (np.array(self.labels) == disease) & (np.array(self.batch_labels) == batch)
                    if np.any(mask):
                        # Use base color for disease, but vary saturation/brightness for batches
                        base_color = disease_base_colors.get(disease, colors[color_idx])
                        
                        # Create batch-specific color variations
                        if isinstance(base_color, str):
                            if disease == 'MGUS':
                                batch_color = {'A': '#FF0000', 'B': '#CC0000', 'C': '#990000'}.get(batch, '#FF6666')
                            elif disease == 'MM':
                                batch_color = {'A': '#0000FF', 'B': '#0000CC', 'C': '#000099'}.get(batch, '#6666FF')
                            elif disease == 'NL':
                                batch_color = {'A': '#00FF00', 'B': '#00CC00', 'C': '#009900'}.get(batch, '#66FF66')
                            else:
                                batch_color = colors[color_idx]
                        else:
                            batch_color = colors[color_idx]
                        
                        marker = batch_markers.get(batch, 'o')
                        
                        axes[0].scatter(X_umap_before[mask, 0], X_umap_before[mask, 1],
                                      c=batch_color, marker=marker,
                                      label=f'{disease}-Batch{batch}', alpha=0.7, s=30)
                        color_idx += 1
            
            axes[0].set_title('Before Harmonization')
            axes[0].set_xlabel('UMAP1')
            axes[0].set_ylabel('UMAP2')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].grid(True, alpha=0.3)
            
            # After harmonization
            if self.harmonized_spectra is not None:
                reducer_after = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
                X_umap_after = reducer_after.fit_transform(self.harmonized_spectra)
                
                color_idx = 0
                for disease in unique_diseases:
                    for batch in unique_batches:
                        mask = (np.array(self.labels) == disease) & (np.array(self.batch_labels) == batch)
                        if np.any(mask):
                            # Use same color scheme as before
                            base_color = disease_base_colors.get(disease, colors[color_idx])
                            
                            if isinstance(base_color, str):
                                if disease == 'MGUS':
                                    batch_color = {'A': '#FF0000', 'B': '#CC0000', 'C': '#990000'}.get(batch, '#FF6666')
                                elif disease == 'MM':
                                    batch_color = {'A': '#0000FF', 'B': '#0000CC', 'C': '#000099'}.get(batch, '#6666FF')
                                elif disease == 'NL':
                                    batch_color = {'A': '#00FF00', 'B': '#00CC00', 'C': '#009900'}.get(batch, '#66FF66')
                                else:
                                    batch_color = colors[color_idx]
                            else:
                                batch_color = colors[color_idx]
                            
                            marker = batch_markers.get(batch, 'o')
                            
                            axes[1].scatter(X_umap_after[mask, 0], X_umap_after[mask, 1],
                                          c=batch_color, marker=marker,
                                          label=f'{disease}-Batch{batch}', alpha=0.7, s=30)
                            color_idx += 1
                
                axes[1].set_title('After Harmonization')
                axes[1].set_xlabel('UMAP1')
                axes[1].set_ylabel('UMAP2')
                axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, 'Harmonization not applied', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('After Harmonization')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("UMAP not available. Install with: pip install umap-learn")
        except Exception as e:
            print(f"UMAP plotting failed: {e}")
            print(traceback.format_exc())
    
    def calculate_batch_mixing_metrics(self) -> Dict[str, float]:
        """Calculate quantitative metrics for batch mixing."""
        if self.harmonized_spectra is None:
            warnings.warn("Must run harmonize() first")
            return {}
        
        try:
            from sklearn.neighbors import NearestNeighbors
            from sklearn.metrics import silhouette_score
            
            metrics = {}
            
            # Calculate kBET (k-nearest neighbor Batch Effect Test)
            k = min(50, len(self.harmonized_spectra) // 4)
            
            if k < 5:
                print("Warning: Too few samples for reliable batch mixing metrics")
                k = max(2, len(self.harmonized_spectra) // 10)
            
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.harmonized_spectra)
            distances, indices = nbrs.kneighbors(self.harmonized_spectra)
            
            batch_mixing_scores = []
            for i in range(len(self.harmonized_spectra)):
                neighbor_indices = indices[i, 1:]  # Exclude self
                neighbor_batches = [self.batch_labels[idx] for idx in neighbor_indices]
                
                # Calculate proportion of neighbors from same batch
                same_batch_count = sum(1 for batch in neighbor_batches if batch == self.batch_labels[i])
                mixing_score = 1 - (same_batch_count / k)  # Higher = better mixing
                batch_mixing_scores.append(mixing_score)
            
            metrics['batch_mixing_score'] = np.mean(batch_mixing_scores)
            
            # Calculate silhouette score for disease separation
            le = LabelEncoder()
            encoded_labels = le.fit_transform(self.labels)
            
            metrics['disease_separation_before'] = silhouette_score(self.original_spectra, encoded_labels)
            metrics['disease_separation_after'] = silhouette_score(self.harmonized_spectra, encoded_labels)
            
            print("Batch mixing metrics:")
            print(f"  - Batch mixing score: {metrics['batch_mixing_score']:.3f} (higher is better, 0.5=random)")
            print(f"  - Disease separation before: {metrics['disease_separation_before']:.3f}")
            print(f"  - Disease separation after: {metrics['disease_separation_after']:.3f}")
            
            if metrics['disease_separation_after'] < metrics['disease_separation_before'] * 0.8:
                print("  ⚠️  Warning: Disease separation decreased significantly after harmonization")
            
            return metrics
            
        except Exception as e:
            print(f"Metrics calculation failed: {e}")
            print(traceback.format_exc())
            return {}
    
    def create_classifier_compatible_split(self, 
                                          harmonized_spectra: np.ndarray,
                                          labels: np.ndarray,
                                          batch_labels_dict: Dict[str, List[str]],
                                          unified_wavelengths: Optional[np.ndarray] = None,
                                          split_type: str = 'predict') -> Dict[str, Any]:
        """
        Create a data split dictionary compatible with classifier models.
        
        Args:
            harmonized_spectra: Harmonized spectral data
            labels: Disease labels
            batch_labels_dict: Dictionary of batch labels
            unified_wavelengths: Wavelength information
            split_type: Type of split ('train', 'test', 'predict')
            
        Returns:
            Dictionary with standardized keys for classifier compatibility
        """
        
        # Determine key prefix based on split type
        if split_type == 'predict':
            prefix = 'train'  # For external prediction, use 'train' keys
        else:
            prefix = split_type
        
        data_split = {
            f'X_{prefix}': harmonized_spectra,
            f'y_{prefix}': labels,
            f'batch_{prefix}_dict': batch_labels_dict,
        }
        
        # Add unified wavelengths if available
        if unified_wavelengths is not None:
            data_split['unified_wavelengths'] = unified_wavelengths
        
        # Add harmonization metadata
        data_split['harmonization_parameters'] = self.get_harmonization_parameters()
        
        print(f"Created classifier-compatible data split for '{split_type}' with keys: {list(data_split.keys())}")
        print(f"Data shape: {harmonized_spectra.shape}")
        print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        return data_split
    
    def get_harmonized_data(self) -> np.ndarray:
        """Get the harmonized spectra."""
        if self.harmonized_spectra is None:
            raise ValueError("Must run harmonize() first")
        return self.harmonized_spectra.copy()
    
    def save_harmonized_data(self, filepath: str) -> None:
        """Save harmonized data and parameters to file."""
        if self.harmonized_spectra is None:
            raise ValueError("Must run harmonize() first")
        
        try:
            data_to_save = {
                'harmonized_spectra': self.harmonized_spectra,
                'original_spectra': self.original_spectra,
                'labels': self.labels,
                'batch_labels': self.batch_labels,
                'batch_labels_dict': getattr(self, 'batch_labels_dict', {}),
                'reference_batch': self.reference_batch,
                'method': self.method,
                'custom_method': self.custom_method,
                'harmonization_parameters': self.get_harmonization_parameters()
            }
            
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
            
            print(f"Harmonized data and parameters saved to {filepath}")
            
        except Exception as e:
            print(f"Failed to save harmonized data: {e}")
            print(traceback.format_exc())
    
    def validate_new_data_compatibility(self, 
                                       new_batch_labels: Union[List[str], Dict[str, List[str]]],
                                       harmonization_params: Dict[str, Any] = None) -> bool:
        """
        Validate if new data is compatible with existing harmonization parameters.
        
        Args:
            new_batch_labels: Batch labels for new data
            harmonization_params: Harmonization parameters to validate against
            
        Returns:
            True if compatible, False otherwise
        """
        
        params = harmonization_params or self.get_harmonization_parameters()
        
        if 'harmonization_metadata' not in params:
            print("Warning: No harmonization metadata available for validation.")
            return False
        
        metadata = params['harmonization_metadata']
        
        # Handle batch labels format
        if isinstance(new_batch_labels, list):
            new_batches = set(new_batch_labels)
        elif isinstance(new_batch_labels, dict):
            primary_key = metadata.get('primary_batch_key', 'batch')
            new_batches = set(new_batch_labels.get(primary_key, []))
        else:
            print("Error: Invalid batch labels format.")
            return False
        
        # Check if new batches are compatible with training batches
        training_batches = set(metadata.get('unique_batches', []))
        
        if not new_batches.issubset(training_batches):
            unknown_batches = new_batches - training_batches
            print(f"Warning: Unknown batches found: {unknown_batches}")
            print(f"Training batches: {training_batches}")
            print("Harmonization may be less effective for unknown batches.")
            return False
        
        print("New data is compatible with existing harmonization parameters.")
        print(f"New batches: {new_batches}")
        print(f"Training batches: {training_batches}")
        
        return True
    
    def get_harmonization_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the harmonization process."""
        
        if self.harmonized_spectra is None:
            return {"status": "Not harmonized"}
        
        # Calculate some basic metrics
        original_var = np.var(self.original_spectra, axis=0).mean()
        harmonized_var = np.var(self.harmonized_spectra, axis=0).mean()
        
        # Calculate batch separation before/after
        batch_separation_before = self._calculate_batch_separation(self.original_spectra)
        batch_separation_after = self._calculate_batch_separation(self.harmonized_spectra)
        
        summary = {
            "status": "Harmonized",
            "method": self.method,
            "custom_method": self.custom_method if self.method == 'custom_harmonizer' else None,
            "reference_batch": self.reference_batch,
            "data_shape": self.harmonized_spectra.shape,
            "n_batches": len(np.unique(self.batch_labels)),
            "n_disease_types": len(np.unique(self.labels)),
            "batch_distribution": dict(zip(*np.unique(self.batch_labels, return_counts=True))),
            "disease_distribution": dict(zip(*np.unique(self.labels, return_counts=True))),
            "variance_reduction": {
                "original_mean_variance": original_var,
                "harmonized_mean_variance": harmonized_var,
                "variance_change_ratio": harmonized_var / original_var if original_var > 0 else 0
            },
            "batch_separation": {
                "before_harmonization": batch_separation_before,
                "after_harmonization": batch_separation_after,
                "separation_change": batch_separation_after - batch_separation_before
            }
        }
        
        return summary
    
    def _calculate_batch_separation(self, spectra: np.ndarray) -> float:
        """Calculate a simple metric for batch separation."""
        try:
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import LabelEncoder
            
            le = LabelEncoder()
            encoded_batches = le.fit_transform(self.batch_labels)
            
            # Handle case where all samples are from the same batch
            if len(np.unique(encoded_batches)) < 2:
                return 0.0
            
            score = silhouette_score(spectra, encoded_batches)
            return score
            
        except Exception as e:
            print(f"Batch separation calculation failed: {e}")
            return 0.0
    
    def print_harmonization_summary(self):
        """Print a detailed summary of the harmonization process."""
        
        summary = self.get_harmonization_summary()
        
        print("=" * 60)
        print("HARMONIZATION SUMMARY")
        print("=" * 60)
        print(f"Status: {summary['status']}")
        
        if summary['status'] == 'Harmonized':
            print(f"Method: {summary['method']}")
            if summary['custom_method']:
                print(f"Custom Method: {summary['custom_method']}")
            print(f"Reference Batch: {summary['reference_batch']}")
            print(f"Data Shape: {summary['data_shape']}")
            print(f"Number of Batches: {summary['n_batches']}")
            print(f"Number of Disease Types: {summary['n_disease_types']}")
            
            print(f"\nBatch Distribution:")
            for batch, count in summary['batch_distribution'].items():
                print(f"  Batch {batch}: {count} samples")
            
            print(f"\nDisease Distribution:")
            for disease, count in summary['disease_distribution'].items():
                print(f"  {disease}: {count} samples")
            
            var_info = summary['variance_reduction']
            print(f"\nVariance Analysis:")
            print(f"  Original mean variance: {var_info['original_mean_variance']:.6f}")
            print(f"  Harmonized mean variance: {var_info['harmonized_mean_variance']:.6f}")
            print(f"  Variance change ratio: {var_info['variance_change_ratio']:.3f}")
            
            batch_sep = summary['batch_separation']
            print(f"\nBatch Separation (Silhouette Score):")
            print(f"  Before harmonization: {batch_sep['before_harmonization']:.3f}")
            print(f"  After harmonization: {batch_sep['after_harmonization']:.3f}")
            print(f"  Change: {batch_sep['separation_change']:.3f}")
            
            if batch_sep['separation_change'] < -0.1:
                print("  ✓ Good: Batch separation reduced significantly")
            elif batch_sep['separation_change'] > 0.1:
                print("  ⚠️  Warning: Batch separation increased")
            else:
                print("  → Modest change in batch separation")
        
        print("=" * 60)
    
    def _print_harmonizer_info(self):
        """Print harmonizer initialization info."""
        print(f"Unified Raman Batch Harmonizer initialized:")
        print(f"  - Data shape: {self.spectra.shape}")
        print(f"  - Method: {self.method}")
        if self.method == 'custom_harmonizer':
            print(f"  - Custom method: {self.custom_method}")
        print(f"  - Primary batch key: {self.primary_batch_key}")
        print(f"  - Reference batch: {self.reference_batch}")
        print(f"  - Batch distribution: {dict(zip(*np.unique(self.batch_labels, return_counts=True)))}")
        print(f"  - Disease distribution: {dict(zip(*np.unique(self.labels, return_counts=True)))}")
        if hasattr(self, 'metadata_batch_settings'):
            print(f"  - Batch settings: {self.metadata_batch_settings}")


