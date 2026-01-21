import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import pickle
import pandas as pd
from notebook_utils.visualize import MLVisualize

class BiomarkerEnhancedLinearRegressionModel:
    """
    Biomarker-Enhanced Linear Regression Model for MGUS/MM Raman Spectroscopy Classification.
    
    Based on clinical research findings:
    - Yonezawa et al. (2024): Primary DNB markers at 1149 cm⁻¹ and 1527-1530 cm⁻¹
    - Russo et al. (2020): Disease progression biomarkers
    - Clinical validation on 834 Normal, 711 MGUS, and 970 MM spectra
    
    This model extracts clinically-validated biomarker features from Raman spectra
    for improved external generalization and clinical interpretability.
    """
    
    def __init__(self, data_split: Dict[str, Any], 
                 use_ridge: bool = True, 
                 alpha: float = 1.0,
                 biomarker_only: bool = False,
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize the Biomarker-Enhanced Linear Regression model.
        
        Args:
            data_split: Data dictionary from RamanDataSplitter.prepare_data()
            use_ridge: Use Ridge regression for better stability
            alpha: Ridge regularization parameter
            biomarker_only: Use only biomarker features (no full spectrum)
            verbose: Print detailed information
            **kwargs: Additional keyword arguments for LinearRegression/Ridge
        """
        self.data_split = data_split
        self.use_ridge = use_ridge
        self.alpha = alpha
        self.biomarker_only = biomarker_only
        self.verbose = verbose
        
        # Check for missing keys (same as original)
        required_keys = ['X_train', 'X_test', 'y_train', 'y_test', 'unified_wavelengths']
        missing_keys = [key for key in required_keys if key not in self.data_split]
        if missing_keys:
            raise ValueError(f"Missing keys in data_split: {missing_keys}")
        
        # Initialize model (Ridge for better stability with biomarkers)
        if use_ridge:
            self.model = Ridge(alpha=alpha, **kwargs)
            model_type = f"Ridge(α={alpha})"
        else:
            self.model = LinearRegression(**kwargs)
            model_type = "LinearRegression"
        
        # Label encoding (same as original)
        self.label_encoder = LabelEncoder()
        self.y_train_encoded = self.label_encoder.fit_transform(self.data_split['y_train'])
        self.y_test_encoded = self.label_encoder.transform(self.data_split['y_test'])
        
        # Biomarker extraction components
        self.biomarker_features_train = None
        self.biomarker_features_test = None
        self.biomarker_feature_names = None
        self.feature_importance_analysis = None
        
        if self.verbose:
            print("=== Biomarker-Enhanced Linear Regression Model ===")
            print(f"Model type: {model_type}")
            print(f"Biomarker-only mode: {biomarker_only}")
            print(f"Training samples: {len(self.data_split['X_train'])}")
            print(f"Test samples: {len(self.data_split['X_test'])}")
            print(f"Original features: {self.data_split['X_train'].shape[1]}")
            print(f"Encoded labels: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        # Extract biomarker features during initialization
        self._extract_biomarker_features()
    
    def _extract_biomarker_features(self) -> None:
        """
        Extract clinically-validated MGUS/MM biomarker features from Raman spectra.
        
        Based on literature review of MGUS/MM Raman spectroscopy biomarkers:
        - Primary DNB markers: 1149, 1527-1530 cm⁻¹ (Yonezawa et al., 2024)
        - Nucleic acid markers: 726, 781, 786, 1078, 1190, 1415 cm⁻¹
        - Protein markers: 1004, 1221, 1655 cm⁻¹
        - Lipid/metabolic markers: 1285, 1440 cm⁻¹
        """
        wavelengths = self.data_split['unified_wavelengths']
        
        # Define clinically-validated biomarker wavelengths
        biomarker_bands = {
            # Primary DNB (Dynamical Network Biomarker) markers - Yonezawa et al. 2024
            'DNB_primary_1149': 1149,      # Primary MGUS/MM discriminator
            'DNB_primary_1528': 1528,      # Primary MGUS/MM discriminator (1527-1530 range)
            
            # Nucleic acid markers (DNA/RNA changes in cancer)
            'nucleic_acid_726': 726,       # Adenine, thymine
            'nucleic_acid_781': 781,       # DNA/RNA backbone
            'nucleic_acid_786': 786,       # DNA/RNA backbone  
            'nucleic_backbone_1078': 1078, # PO₄⁻ symmetric stretch
            'nucleic_backbone_1190': 1190, # DNA/RNA backbone vibrations
            'nucleic_backbone_1415': 1415, # DNA/RNA backbone
            
            # Protein structure markers
            'phenylalanine_1004': 1004,    # Phe ring breathing (protein stress)
            'protein_amide_III_1221': 1221, # Protein structure changes
            'protein_amide_I_1655': 1655,  # Protein α-helix/β-sheet
            
            # Lipid and metabolic markers
            'phospholipid_1285': 1285,     # Phospholipid/cholesterol
            'lipid_CH2_1440': 1440,       # CH₂ deformation (membrane changes)
            
            # Background/reference bands
            'background_low_600': 600,     # Background reference
            'background_high_1800': 1800   # Background reference
        }
        
        if self.verbose:
            print(f"\n=== Extracting {len(biomarker_bands)} Biomarker Features ===")
        
        # Extract band intensities for training data
        biomarker_intensities_train = {}
        biomarker_intensities_test = {}
        
        for band_name, target_wavenumber in biomarker_bands.items():
            # Find closest wavelength index
            idx = np.argmin(np.abs(wavelengths - target_wavenumber))
            actual_wavenumber = wavelengths[idx]
            
            # Extract intensities
            biomarker_intensities_train[band_name] = self.data_split['X_train'][:, idx]
            biomarker_intensities_test[band_name] = self.data_split['X_test'][:, idx]
            
            if self.verbose and abs(actual_wavenumber - target_wavenumber) > 5:
                print(f"Warning: {band_name} target={target_wavenumber} cm⁻¹, actual={actual_wavenumber:.1f} cm⁻¹")
        
        # Create ratio features (batch-invariant and clinically meaningful)
        ratio_features_train = {
            # Primary DNB ratio (most important clinical marker)
            'DNB_ratio_1149_1528': (biomarker_intensities_train['DNB_primary_1149'] / 
                                   (biomarker_intensities_train['DNB_primary_1528'] + 1e-8)),
            
            # Nucleic acid to protein ratios (cell proliferation markers)
            'nucleic_protein_ratio': (np.mean([biomarker_intensities_train['nucleic_acid_726'], 
                                              biomarker_intensities_train['nucleic_acid_781']], axis=0) / 
                                    (biomarker_intensities_train['protein_amide_III_1221'] + 1e-8)),
            
            # Lipid to protein ratio (metabolic changes)
            'lipid_protein_ratio': (biomarker_intensities_train['lipid_CH2_1440'] / 
                                  (biomarker_intensities_train['protein_amide_I_1655'] + 1e-8)),
            
            # Metabolic stress ratio
            'metabolic_stress_ratio': (biomarker_intensities_train['phenylalanine_1004'] / 
                                     (biomarker_intensities_train['phospholipid_1285'] + 1e-8)),
            
            # Signal-to-background ratios (data quality markers)
            'signal_background_low': (biomarker_intensities_train['DNB_primary_1149'] / 
                                    (biomarker_intensities_train['background_low_600'] + 1e-8)),
            'signal_background_high': (biomarker_intensities_train['DNB_primary_1528'] / 
                                     (biomarker_intensities_train['background_high_1800'] + 1e-8)),
            
            # Advanced clinical ratios
            'MGUS_progression_marker': ((biomarker_intensities_train['nucleic_backbone_1078'] + 
                                       biomarker_intensities_train['nucleic_backbone_1190']) / 
                                      (biomarker_intensities_train['protein_amide_I_1655'] + 1e-8)),
            
            'MM_severity_marker': (biomarker_intensities_train['nucleic_acid_786'] / 
                                 (biomarker_intensities_train['DNB_primary_1149'] + 1e-8))
        }
        
        # Create corresponding test ratios
        ratio_features_test = {
            'DNB_ratio_1149_1528': (biomarker_intensities_test['DNB_primary_1149'] / 
                                   (biomarker_intensities_test['DNB_primary_1528'] + 1e-8)),
            'nucleic_protein_ratio': (np.mean([biomarker_intensities_test['nucleic_acid_726'], 
                                              biomarker_intensities_test['nucleic_acid_781']], axis=0) / 
                                    (biomarker_intensities_test['protein_amide_III_1221'] + 1e-8)),
            'lipid_protein_ratio': (biomarker_intensities_test['lipid_CH2_1440'] / 
                                  (biomarker_intensities_test['protein_amide_I_1655'] + 1e-8)),
            'metabolic_stress_ratio': (biomarker_intensities_test['phenylalanine_1004'] / 
                                     (biomarker_intensities_test['phospholipid_1285'] + 1e-8)),
            'signal_background_low': (biomarker_intensities_test['DNB_primary_1149'] / 
                                    (biomarker_intensities_test['background_low_600'] + 1e-8)),
            'signal_background_high': (biomarker_intensities_test['DNB_primary_1528'] / 
                                     (biomarker_intensities_test['background_high_1800'] + 1e-8)),
            'MGUS_progression_marker': ((biomarker_intensities_test['nucleic_backbone_1078'] + 
                                       biomarker_intensities_test['nucleic_backbone_1190']) / 
                                      (biomarker_intensities_test['protein_amide_I_1655'] + 1e-8)),
            'MM_severity_marker': (biomarker_intensities_test['nucleic_acid_786'] / 
                                 (biomarker_intensities_test['DNB_primary_1149'] + 1e-8))
        }
        
        # Combine absolute intensities and ratios
        all_features_train = {**biomarker_intensities_train, **ratio_features_train}
        all_features_test = {**biomarker_intensities_test, **ratio_features_test}
        
        # Convert to arrays
        self.biomarker_features_train = np.column_stack(list(all_features_train.values()))
        self.biomarker_features_test = np.column_stack(list(all_features_test.values()))
        self.biomarker_feature_names = list(all_features_train.keys())
        
        # Optional: Combine with original features
        if not self.biomarker_only:
            self.biomarker_features_train = np.column_stack([
                self.data_split['X_train'], 
                self.biomarker_features_train
            ])
            self.biomarker_features_test = np.column_stack([
                self.data_split['X_test'], 
                self.biomarker_features_test
            ])
            
            # Add original feature names
            original_feature_names = [f"wn_{wn:.1f}" for wn in wavelengths]
            self.biomarker_feature_names = original_feature_names + self.biomarker_feature_names
        
        if self.verbose:
            print(f"Biomarker feature extraction completed:")
            print(f"  Training features: {self.biomarker_features_train.shape}")
            print(f"  Test features: {self.biomarker_features_test.shape}")
            print(f"  Total biomarker features: {len(biomarker_bands) + len(ratio_features_train)}")
            
            # Show key biomarker statistics
            print(f"\n=== Key Biomarker Statistics ===")
            for class_name in self.label_encoder.classes_:
                class_mask = self.data_split['y_train'] == class_name
                dnb_1149_mean = np.mean(biomarker_intensities_train['DNB_primary_1149'][class_mask])
                dnb_1528_mean = np.mean(biomarker_intensities_train['DNB_primary_1528'][class_mask])
                dnb_ratio_mean = np.mean(ratio_features_train['DNB_ratio_1149_1528'][class_mask])
                
                print(f"{class_name}: DNB_1149={dnb_1149_mean:.3f}, DNB_1528={dnb_1528_mean:.3f}, Ratio={dnb_ratio_mean:.3f}")

    def _extract_biomarker_features_from_external(self, X_external: np.ndarray) -> np.ndarray:
        """
        Extract biomarker features from external data using same methodology.
        
        Args:
            X_external: External spectral data
            
        Returns:
            Extracted biomarker features
        """
        wavelengths = self.data_split['unified_wavelengths']
        
        # Same biomarker bands as in initialization
        biomarker_bands = {
            'DNB_primary_1149': 1149, 'DNB_primary_1528': 1528,
            'nucleic_acid_726': 726, 'nucleic_acid_781': 781, 'nucleic_acid_786': 786,
            'nucleic_backbone_1078': 1078, 'nucleic_backbone_1190': 1190, 'nucleic_backbone_1415': 1415,
            'phenylalanine_1004': 1004, 'protein_amide_III_1221': 1221, 'protein_amide_I_1655': 1655,
            'phospholipid_1285': 1285, 'lipid_CH2_1440': 1440,
            'background_low_600': 600, 'background_high_1800': 1800
        }
        
        # Extract intensities
        biomarker_intensities = {}
        for band_name, target_wavenumber in biomarker_bands.items():
            idx = np.argmin(np.abs(wavelengths - target_wavenumber))
            biomarker_intensities[band_name] = X_external[:, idx]
        
        # Create ratio features
        ratio_features = {
            'DNB_ratio_1149_1528': (biomarker_intensities['DNB_primary_1149'] / 
                                   (biomarker_intensities['DNB_primary_1528'] + 1e-8)),
            'nucleic_protein_ratio': (np.mean([biomarker_intensities['nucleic_acid_726'], 
                                              biomarker_intensities['nucleic_acid_781']], axis=0) / 
                                    (biomarker_intensities['protein_amide_III_1221'] + 1e-8)),
            'lipid_protein_ratio': (biomarker_intensities['lipid_CH2_1440'] / 
                                  (biomarker_intensities['protein_amide_I_1655'] + 1e-8)),
            'metabolic_stress_ratio': (biomarker_intensities['phenylalanine_1004'] / 
                                     (biomarker_intensities['phospholipid_1285'] + 1e-8)),
            'signal_background_low': (biomarker_intensities['DNB_primary_1149'] / 
                                    (biomarker_intensities['background_low_600'] + 1e-8)),
            'signal_background_high': (biomarker_intensities['DNB_primary_1528'] / 
                                     (biomarker_intensities['background_high_1800'] + 1e-8)),
            'MGUS_progression_marker': ((biomarker_intensities['nucleic_backbone_1078'] + 
                                       biomarker_intensities['nucleic_backbone_1190']) / 
                                      (biomarker_intensities['protein_amide_I_1655'] + 1e-8)),
            'MM_severity_marker': (biomarker_intensities['nucleic_acid_786'] / 
                                 (biomarker_intensities['DNB_primary_1149'] + 1e-8))
        }
        
        # Combine features
        all_features = {**biomarker_intensities, **ratio_features}
        biomarker_features_external = np.column_stack(list(all_features.values()))
        
        # Add original features if not biomarker-only mode
        if not self.biomarker_only:
            biomarker_features_external = np.column_stack([X_external, biomarker_features_external])
        
        return biomarker_features_external

    def fit(self) -> None:
        """
        Fit the Biomarker-Enhanced Linear Regression model to the training data.
        """
        if self.verbose:
            print("\n=== Fitting Biomarker-Enhanced Linear Regression Model ===")
        
        # Fit on biomarker-enhanced features
        self.model.fit(self.biomarker_features_train, self.y_train_encoded)
        
        # Analyze feature importance (for Ridge regression)
        if hasattr(self.model, 'coef_'):
            self._analyze_feature_importance()
        
        if self.verbose:
            print("Model fitted successfully on biomarker-enhanced features.")
    
    def _analyze_feature_importance(self) -> None:
        """Analyze feature importance for biomarker interpretation."""
        if not hasattr(self.model, 'coef_'):
            return
        
        coefficients = self.model.coef_
        feature_importance = np.abs(coefficients)
        
        # Create importance dataframe
        importance_data = []
        for i, (name, importance, coef) in enumerate(zip(self.biomarker_feature_names, 
                                                        feature_importance, coefficients)):
            # Determine clinical interpretation
            if 'DNB' in name:
                clinical_meaning = "Primary MGUS/MM biomarker"
            elif 'nucleic' in name:
                clinical_meaning = "DNA/RNA activity (proliferation)"  
            elif 'protein' in name:
                clinical_meaning = "Protein structure changes"
            elif 'lipid' in name or 'phospholipid' in name:
                clinical_meaning = "Membrane/metabolic changes"
            elif 'ratio' in name:
                clinical_meaning = "Batch-invariant clinical ratio"
            elif 'background' in name:
                clinical_meaning = "Data quality marker"
            else:
                clinical_meaning = "Spectral feature"
            
            importance_data.append({
                'feature': name,
                'coefficient': coef,
                'importance': importance,
                'clinical_meaning': clinical_meaning
            })
        
        # Sort by importance
        importance_data.sort(key=lambda x: x['importance'], reverse=True)
        self.feature_importance_analysis = importance_data
        
        if self.verbose:
            print(f"\n=== Top 10 Most Important Biomarkers ===")
            for i, data in enumerate(importance_data[:10]):
                print(f"{i+1:2d}. {data['feature'][:25]:25s} | "
                      f"Coef: {data['coefficient']:8.4f} | "
                      f"Imp: {data['importance']:7.4f} | "
                      f"{data['clinical_meaning']}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data using biomarker features.
        
        Args:
            X: Feature matrix (original spectral data)
            
        Returns:
            Predicted encoded labels
        """
        # Extract biomarker features from input
        biomarker_features = self._extract_biomarker_features_from_external(X)
        
        # Make predictions using biomarker-enhanced features
        return self.model.predict(biomarker_features)

    def predict_labels(self, X: np.ndarray) -> List[str]:
        """
        Make predictions and decode to original labels.
        
        Args:
            X: Feature matrix (original spectral data)
            
        Returns:
            Predicted original labels
        """
        predictions_encoded = self.predict(X)
        
        # Round to nearest integer for label decoding
        predictions_rounded = np.round(predictions_encoded).astype(int)
        
        # Clip to valid label range
        predictions_rounded = np.clip(predictions_rounded, 0, len(self.label_encoder.classes_) - 1)
        
        return self.label_encoder.inverse_transform(predictions_rounded)

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the test set with detailed classification metrics.
        Same interface as original LinearRegressionModel.
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
        
        if self.verbose:
            print("=== Detailed Evaluation Metrics (Biomarker-Enhanced) ===")
            print("\nRegression Metrics:")
            print(f"  Mean Squared Error: {mse:.4f}")
            print(f"  R² Score: {r2:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true_labels, y_pred_labels, target_names=self.label_encoder.classes_, zero_division=0))
        
        return metrics

    def predict_new_data(self, X_new: np.ndarray, y_new: Optional[np.ndarray] = None, 
                        show_confusion_matrix: bool = False) -> Dict[str, Any]:
        """
        Predict on new data and optionally evaluate. Same interface as original.
        
        Args:
            X_new: New feature matrix for prediction (original spectral data)
            y_new: True labels for new data (if available)
            show_confusion_matrix: Whether to display the confusion matrix
            
        Returns:
            Dictionary containing predictions and evaluation metrics
        """
        predictions_encoded = self.predict(X_new)
        predictions_labels = self.predict_labels(X_new)
        
        result = {
            'predictions_encoded': predictions_encoded,
            'predictions_labels': predictions_labels,
            'n_predictions': len(predictions_labels)
        }
        
        if self.verbose:
            print(f"=== Biomarker-Enhanced Predictions on New Data ===")
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
            
            if self.verbose:
                print("\n=== Evaluation Metrics on New Data ===")
                print("\nRegression Metrics:")
                print(f"  Mean Squared Error: {mse:.4f}")
                print(f"  R² Score: {r2:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_new, predictions_labels, target_names=self.label_encoder.classes_, zero_division=0))
                
                # Show biomarker analysis for external data
                self._analyze_external_biomarkers(X_new, y_new, predictions_labels)
            
            if show_confusion_matrix:
                self._plot_confusion_matrix_new_data(y_new, predictions_labels)
        
        return result

    def _analyze_external_biomarkers(self, X_new: np.ndarray, y_true: np.ndarray, y_pred: List[str]) -> None:
        """Analyze biomarker performance on external data for clinical interpretation."""
        if not self.verbose:
            return
        
        print("\n=== External Data Biomarker Analysis ===")
        
        # Extract biomarkers from external data
        external_biomarkers = self._extract_biomarker_features_from_external(X_new)
        
        # Analyze DNB ratio performance
        wavelengths = self.data_split['unified_wavelengths']
        idx_1149 = np.argmin(np.abs(wavelengths - 1149))
        idx_1528 = np.argmin(np.abs(wavelengths - 1528))
        
        dnb_1149 = X_new[:, idx_1149]
        dnb_1528 = X_new[:, idx_1528]
        dnb_ratio = dnb_1149 / (dnb_1528 + 1e-8)
        
        # Analyze by true class
        for class_name in self.label_encoder.classes_:
            true_mask = y_true == class_name
            if np.sum(true_mask) == 0:
                continue
                
            pred_mask = np.array(y_pred) == class_name
            correct_mask = true_mask & pred_mask
            
            # Statistics for this class
            true_count = np.sum(true_mask)
            correct_count = np.sum(correct_mask)
            accuracy_class = correct_count / true_count if true_count > 0 else 0
            
            # Biomarker statistics
            dnb_1149_mean = np.mean(dnb_1149[true_mask])
            dnb_1528_mean = np.mean(dnb_1528[true_mask])
            dnb_ratio_mean = np.mean(dnb_ratio[true_mask])
            
            print(f"{class_name}: Accuracy={accuracy_class:.3f} ({correct_count}/{true_count})")
            print(f"  DNB_1149: {dnb_1149_mean:.3f}, DNB_1528: {dnb_1528_mean:.3f}, Ratio: {dnb_ratio_mean:.3f}")

    def plot_confusion_matrix(self, show_plot: bool = True) -> np.ndarray:
        """
        Compute and optionally plot the confusion matrix. Same interface as original.
        """
        y_pred_labels = self.predict_labels(self.data_split['X_test'])
        y_true = self.data_split['y_test']
        
        # Compute confusion matrix with original labels using sklearn
        cm = confusion_matrix(y_true, y_pred_labels, labels=self.label_encoder.classes_)
        
        # Print confusion matrix as text (same as original)
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
            ax1.set_title('Confusion Matrix (Biomarker-Enhanced Linear Regression)')
            ax1.set_ylabel('True Label')
            ax1.set_xlabel('Predicted Label')
            
            # Plot classification report on ax2
            ax2.text(0.1, 0.5, report, fontsize=10, verticalalignment='center', fontfamily='monospace')
            ax2.set_title('Classification Report')
            ax2.axis('off')  # Hide axes for text
            
            plt.tight_layout()
            plt.show()
        
        return cm

    def _plot_confusion_matrix_new_data(self, y_true: np.ndarray, y_pred: List[str]) -> np.ndarray:
        """
        Compute and optionally plot the confusion matrix for new data. Same as original.
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
        ax1.set_title('Confusion Matrix for New Data (Biomarker-Enhanced)')
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
        """Save the trained model and all components to a pickle file."""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'data_split': self.data_split,
            'use_ridge': self.use_ridge,
            'alpha': self.alpha,
            'biomarker_only': self.biomarker_only,
            'biomarker_feature_names': self.biomarker_feature_names,
            'feature_importance_analysis': self.feature_importance_analysis
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Biomarker-Enhanced model saved to {filepath}")

    @staticmethod
    def load_model(filepath: str) -> 'BiomarkerEnhancedLinearRegressionModel':
        """Load a trained model from a pickle file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = BiomarkerEnhancedLinearRegressionModel(
            model_data['data_split'],
            use_ridge=model_data.get('use_ridge', True),
            alpha=model_data.get('alpha', 1.0),
            biomarker_only=model_data.get('biomarker_only', False),
            verbose=False
        )
        instance.model = model_data['model']
        instance.label_encoder = model_data['label_encoder']
        instance.biomarker_feature_names = model_data.get('biomarker_feature_names', [])
        instance.feature_importance_analysis = model_data.get('feature_importance_analysis', [])
        return instance

    def get_biomarker_report(self) -> Dict[str, Any]:
        """Get comprehensive biomarker analysis report for research documentation."""
        return {
            'model_type': 'Biomarker-Enhanced Linear Regression',
            'regularization': 'Ridge' if self.use_ridge else 'None',
            'alpha': self.alpha if self.use_ridge else 0,
            'biomarker_only_mode': self.biomarker_only,
            'n_biomarker_features': len(self.biomarker_feature_names) if self.biomarker_feature_names else 0,
            'feature_importance': self.feature_importance_analysis,
            'clinical_biomarkers': [
                'DNB_primary_1149: Primary MGUS/MM discriminator',
                'DNB_primary_1528: Primary MGUS/MM discriminator', 
                'DNB_ratio_1149_1528: Most important clinical ratio',
                'nucleic_protein_ratio: Cell proliferation marker',
                'MGUS_progression_marker: Disease progression indicator',
                'MM_severity_marker: Disease severity indicator'
            ]
        }


class MGUSAwareBiomarkerModel(BiomarkerEnhancedLinearRegressionModel):
    def __init__(self, data_split, mgus_strategy='conservative', **kwargs):
        super().__init__(data_split, **kwargs)
        self.mgus_strategy = mgus_strategy  # 'conservative', 'two_stage', 'stratified'
    
    def predict_new_data(self, X_new, y_new=None, **kwargs):
        """Enhanced prediction with MGUS-specific handling."""
        
        # Get base predictions
        result = super().predict_new_data(X_new, y_new, **kwargs)
        base_predictions = result['predictions_labels']
        
        # Apply MGUS-specific refinements
        if self.mgus_strategy == 'conservative':
            # Conservative: Classify uncertain MGUS as MM
            refined_predictions = self.apply_mgus_clinical_thresholds(
                self._extract_biomarker_features_from_external(X_new),
                base_predictions
            )
            
        elif self.mgus_strategy == 'two_stage':
            # Two-stage classification
            refined_predictions = self.predict_two_stage(X_new)
            
        else:
            refined_predictions = base_predictions
        
        # Update result
        result['predictions_labels'] = refined_predictions
        result['mgus_strategy'] = self.mgus_strategy
        
        # Recalculate metrics if ground truth available
        if y_new is not None:
            from sklearn.metrics import accuracy_score, classification_report
            
            new_accuracy = accuracy_score(y_new, refined_predictions)
            result['evaluation']['classification']['accuracy'] = new_accuracy
            
            if self.verbose:
                print(f"\n=== MGUS-Aware Classification Results ===")
                print(f"Strategy: {self.mgus_strategy}")
                print(f"Original Accuracy: {accuracy_score(y_new, base_predictions):.3f}")
                print(f"MGUS-Aware Accuracy: {new_accuracy:.3f}")
                print(f"Improvement: {new_accuracy - accuracy_score(y_new, base_predictions):+.3f}")
                print("\nDetailed Report:")
                print(classification_report(y_new, refined_predictions))
        
        return result

# Initialize Biomarker-Enhanced model (same interface as your original)
biomarker_lr_model = MGUSAwareBiomarkerModel(
    data_split,
    mgus_strategy='stratified',  # MGUS-specific strategy
    use_ridge=True,        # Use Ridge for better stability
    alpha=1.0,             # Regularization strength
    biomarker_only=False,  # Keep original + biomarker features
    verbose=True           # Show detailed biomarker analysis
)

# Train (same as your original)
biomarker_lr_model.fit()

# Evaluate (same interface, enhanced output)
BIOMARKER_LR_EVALUATION = biomarker_lr_model.evaluate()

# Plot confusion matrix (same as original)
cm = biomarker_lr_model.plot_confusion_matrix()

BIOMARKER_LR_PREDICT_EVALUATION = biomarker_lr_model.predict_new_data(
    data_split_predict['X_train'], 
    data_split_predict['y_train'], 
    show_confusion_matrix=True
)["evaluation"]

# Get biomarker analysis report (new feature)
biomarker_report = biomarker_lr_model.get_biomarker_report()
print(f"\nBiomarker-Enhanced External Accuracy: {BIOMARKER_LR_PREDICT_EVALUATION['classification']['accuracy']:.4f}")

