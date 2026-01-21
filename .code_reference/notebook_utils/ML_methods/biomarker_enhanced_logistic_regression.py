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
from typing import Dict, Any, List, Optional, Tuple
import pickle
import pandas as pd

class RamanBiomarkerDatabase:
    """
    External biomarker database for MGUS/MM classification.
    Easy to edit and extend for different research needs.
    """
    
    def __init__(self):
        self.biomarker_bands = self._define_biomarker_bands()
        self.ratio_definitions = self._define_ratio_features()
        self.clinical_interpretations = self._define_clinical_meanings()
        
    def _define_biomarker_bands(self) -> Dict[str, Dict]:
        """
        Define individual biomarker bands with metadata.
        Easy to modify wavenumbers and add new biomarkers.
        """
        return {
            # Primary DNB markers (Most Important)
            'DNB_primary_1149': {
                'target_wavenumber': 1149,
                'tolerance': 5,  # cm⁻¹
                'priority': 'HIGH',
                'assignment': 'Primary MGUS/MM discriminator',
                'reference': 'Yonezawa et al. (2024)'
            },
            'DNB_primary_1528': {
                'target_wavenumber': 1528,
                'tolerance': 3,
                'priority': 'HIGH', 
                'assignment': 'Primary MGUS/MM discriminator',
                'reference': 'Yonezawa et al. (2024)'
            },
            
            # Nucleic acid markers
            'adenine_726': {
                'target_wavenumber': 726,
                'tolerance': 5,
                'priority': 'MEDIUM',
                'assignment': 'Adenine (DNA/RNA)',
                'reference': 'Literature review'
            },
            'nucleic_backbone_781': {
                'target_wavenumber': 781,
                'tolerance': 5,
                'priority': 'MEDIUM',
                'assignment': 'DNA/RNA backbone',
                'reference': 'Literature review'
            },
            'nucleic_backbone_786': {
                'target_wavenumber': 786,
                'tolerance': 5,
                'priority': 'MEDIUM',
                'assignment': 'DNA/RNA backbone',
                'reference': 'Literature review'
            },
            'phosphate_1078': {
                'target_wavenumber': 1078,
                'tolerance': 5,
                'priority': 'MEDIUM',
                'assignment': 'PO₄⁻ symmetric stretch',
                'reference': 'Literature review'
            },
            'nucleic_backbone_1190': {
                'target_wavenumber': 1190,
                'tolerance': 5,
                'priority': 'MEDIUM',
                'assignment': 'DNA/RNA backbone vibrations',
                'reference': 'Literature review'
            },
            'nucleic_backbone_1415': {
                'target_wavenumber': 1415,
                'tolerance': 5,
                'priority': 'LOW',
                'assignment': 'DNA/RNA backbone',
                'reference': 'Literature review'
            },
            
            # Protein structure markers
            'phenylalanine_1004': {
                'target_wavenumber': 1004,
                'tolerance': 5,
                'priority': 'MEDIUM',
                'assignment': 'Phenylalanine ring breathing',
                'reference': 'Protein stress marker'
            },
            'amide_III_1221': {
                'target_wavenumber': 1221,
                'tolerance': 5,
                'priority': 'MEDIUM',
                'assignment': 'Amide III (protein structure)',
                'reference': 'Protein structural changes'
            },
            'amide_I_1655': {
                'target_wavenumber': 1655,
                'tolerance': 5,
                'priority': 'HIGH',
                'assignment': 'Amide I (α-helix/β-sheet)',
                'reference': 'Protein backbone'
            },
            
            # Lipid and metabolic markers
            'phospholipid_1285': {
                'target_wavenumber': 1285,
                'tolerance': 5,
                'priority': 'LOW',
                'assignment': 'Phospholipid/cholesterol',
                'reference': 'Membrane changes'
            },
            'lipid_CH2_1440': {
                'target_wavenumber': 1440,
                'tolerance': 5,
                'priority': 'MEDIUM',
                'assignment': 'CH₂ deformation',
                'reference': 'Membrane/lipid changes'
            },
            
            # Background/reference bands
            'background_low_600': {
                'target_wavenumber': 600,
                'tolerance': 10,
                'priority': 'LOW',
                'assignment': 'Background reference',
                'reference': 'Data quality marker'
            },
            'background_high_1800': {
                'target_wavenumber': 1800,
                'tolerance': 10,
                'priority': 'LOW',
                'assignment': 'Background reference', 
                'reference': 'Data quality marker'
            }
        }
    
    def _define_ratio_features(self) -> Dict[str, Dict]:
        """
        Define ratio features for batch-invariant classification.
        Easy to add new ratios and modify existing ones.
        """
        return {
            'DNB_ratio_primary': {
                'numerator': 'DNB_primary_1149',
                'denominator': 'DNB_primary_1528',
                'clinical_meaning': 'Primary MGUS/MM classification ratio',
                'priority': 'CRITICAL',
                'expected_direction': 'MGUS_higher'  # or 'MM_higher'
            },
            'nucleic_protein_ratio': {
                'numerator': ['adenine_726', 'nucleic_backbone_781'],  # Average multiple
                'denominator': 'amide_III_1221',
                'clinical_meaning': 'DNA/RNA proliferation vs protein structure',
                'priority': 'HIGH',
                'expected_direction': 'MM_higher'
            },
            'lipid_protein_ratio': {
                'numerator': 'lipid_CH2_1440',
                'denominator': 'amide_I_1655', 
                'clinical_meaning': 'Metabolic changes vs protein backbone',
                'priority': 'MEDIUM',
                'expected_direction': 'MM_higher'
            },
            'metabolic_stress_ratio': {
                'numerator': 'phenylalanine_1004',
                'denominator': 'phospholipid_1285',
                'clinical_meaning': 'Protein stress vs membrane stability',
                'priority': 'MEDIUM',
                'expected_direction': 'MM_higher'
            },
            'MGUS_progression_marker': {
                'numerator': ['phosphate_1078', 'nucleic_backbone_1190'],  # Sum multiple
                'denominator': 'amide_I_1655',
                'clinical_meaning': 'Nucleic acid activity progression marker',
                'priority': 'HIGH',
                'expected_direction': 'MM_higher'
            },
            'MM_severity_marker': {
                'numerator': 'nucleic_backbone_786',
                'denominator': 'DNB_primary_1149',
                'clinical_meaning': 'Disease severity indicator',
                'priority': 'HIGH',
                'expected_direction': 'MM_higher'
            },
            'signal_quality_low': {
                'numerator': 'DNB_primary_1149',
                'denominator': 'background_low_600',
                'clinical_meaning': 'Data quality - low frequency',
                'priority': 'LOW',
                'expected_direction': 'higher_better'
            },
            'signal_quality_high': {
                'numerator': 'DNB_primary_1528', 
                'denominator': 'background_high_1800',
                'clinical_meaning': 'Data quality - high frequency',
                'priority': 'LOW',
                'expected_direction': 'higher_better'
            }
        }
    
    def _define_clinical_meanings(self) -> Dict[str, str]:
        """Clinical interpretation of biomarker categories."""
        return {
            'DNB': 'Dynamical Network Biomarker - Primary disease discriminator',
            'nucleic': 'DNA/RNA markers - Cell proliferation and genetic activity',
            'protein': 'Protein structure markers - Cellular stress and function',
            'lipid': 'Lipid/metabolic markers - Membrane and energy metabolism',
            'ratio': 'Batch-invariant clinical ratios - Robust classification features',
            'background': 'Data quality markers - Technical validation',
            'progression': 'Disease progression markers - MGUS→MM transition',
            'severity': 'Disease severity markers - MM staging and prognosis'
        }
    
    def get_biomarker_subset(self, priority_levels: List[str] = ['HIGH', 'CRITICAL']) -> Dict[str, Dict]:
        """Get subset of biomarkers by priority level."""
        return {
            name: info for name, info in self.biomarker_bands.items() 
            if info['priority'] in priority_levels
        }
    
    def get_ratio_subset(self, priority_levels: List[str] = ['HIGH', 'CRITICAL']) -> Dict[str, Dict]:
        """Get subset of ratios by priority level.""" 
        return {
            name: info for name, info in self.ratio_definitions.items()
            if info['priority'] in priority_levels
        }
    
    def update_biomarker(self, name: str, updates: Dict):
        """Update existing biomarker or add new one."""
        if name in self.biomarker_bands:
            self.biomarker_bands[name].update(updates)
        else:
            self.biomarker_bands[name] = updates
            
    def update_ratio(self, name: str, updates: Dict):
        """Update existing ratio or add new one."""
        if name in self.ratio_definitions:
            self.ratio_definitions[name].update(updates)
        else:
            self.ratio_definitions[name] = updates

# Global instance for easy import
MGUS_MM_BIOMARKERS = RamanBiomarkerDatabase()

class BiomarkerEnhancedLogisticRegressionModel:
    """
    Biomarker-Enhanced Logistic Regression for MGUS/MM Raman Classification.
    
    Combines the statistical robustness of logistic regression with 
    clinically-validated biomarker features for improved generalization.
    
    Features:
    - External biomarker database (easy to edit)
    - Logistic regression with proper probability estimation
    - Biomarker feature extraction and validation
    - Clinical interpretation of coefficients
    - Same interface as your existing LogisticRegressionModel
    """
    
    def __init__(self, data_split: Dict[str, Any],
                 # Logistic Regression parameters
                 penalty: str = 'l2',
                 C: float = 1.0,
                 solver: str = 'lbfgs',
                 max_iter: int = 1000,
                 multi_class: str = 'auto',
                 class_weight: Optional[str] = None,
                 random_state: Optional[int] = 42,
                 # Biomarker parameters
                 biomarker_only: bool = False,
                 biomarker_priority: List[str] = ['CRITICAL', 'HIGH'],
                 scale_features: bool = True,
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize Biomarker-Enhanced Logistic Regression.
        
        Args:
            data_split: Data dictionary from RamanDataSplitter
            penalty: Regularization ('l1', 'l2', 'elasticnet', 'none')
            C: Inverse regularization strength
            solver: Optimization algorithm
            max_iter: Maximum iterations
            multi_class: Multi-class strategy
            class_weight: Class weights for imbalanced data
            random_state: Random seed
            biomarker_only: Use only biomarker features
            biomarker_priority: Priority levels to include ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
            scale_features: Standardize features
            verbose: Print detailed information
        """
        self.data_split = data_split
        self.biomarker_only = biomarker_only
        self.biomarker_priority = biomarker_priority
        self.scale_features = scale_features
        self.verbose = verbose
        
        # Check required keys
        required_keys = ['X_train', 'X_test', 'y_train', 'y_test', 'unified_wavelengths']
        missing_keys = [key for key in required_keys if key not in self.data_split]
        if missing_keys:
            raise ValueError(f"Missing keys in data_split: {missing_keys}")
        
        # Initialize logistic regression
        self.model = LogisticRegression(
            penalty=penalty, C=C, solver=solver, max_iter=max_iter,
            multi_class=multi_class, class_weight=class_weight,
            random_state=random_state, **kwargs
        )
        
        # Label encoding
        self.label_encoder = LabelEncoder()
        self.y_train_encoded = self.label_encoder.fit_transform(self.data_split['y_train'])
        self.y_test_encoded = self.label_encoder.transform(self.data_split['y_test'])
        
        # Feature scaling
        self.scaler = StandardScaler() if scale_features else None
        
        # Biomarker components
        self.biomarker_database = MGUS_MM_BIOMARKERS
        self.biomarker_features_train = None
        self.biomarker_features_test = None
        self.biomarker_feature_names = None
        self.biomarker_analysis = None
        
        if verbose:
            print("=== Biomarker-Enhanced Logistic Regression ===")
            print(f"Classes: {list(self.label_encoder.classes_)}")
            print(f"Model: LogisticRegression(C={C}, penalty={penalty}, solver={solver})")
            print(f"Biomarker priority: {biomarker_priority}")
            print(f"Biomarker-only mode: {biomarker_only}")
            print(f"Feature scaling: {scale_features}")
        
        # Extract biomarker features
        self._extract_biomarker_features()
    
    def _extract_biomarker_features(self) -> None:
        """Extract biomarker features using external database."""
        wavelengths = self.data_split['unified_wavelengths']
        
        # Get biomarkers based on priority
        selected_biomarkers = self.biomarker_database.get_biomarker_subset(self.biomarker_priority)
        selected_ratios = self.biomarker_database.get_ratio_subset(self.biomarker_priority)
        
        if self.verbose:
            print(f"\n=== Extracting Biomarker Features ===")
            print(f"Selected {len(selected_biomarkers)} individual biomarkers")
            print(f"Selected {len(selected_ratios)} ratio features")
        
        # Extract individual biomarker intensities
        biomarker_intensities_train = {}
        biomarker_intensities_test = {}
        
        for name, info in selected_biomarkers.items():
            target_wn = info['target_wavenumber']
            tolerance = info['tolerance']
            
            # Find closest wavelength
            idx = np.argmin(np.abs(wavelengths - target_wn))
            actual_wn = wavelengths[idx]
            
            if abs(actual_wn - target_wn) > tolerance:
                if self.verbose:
                    print(f"Warning: {name} target={target_wn}, actual={actual_wn:.1f} (>{tolerance} cm⁻¹)")
            
            biomarker_intensities_train[name] = self.data_split['X_train'][:, idx]
            biomarker_intensities_test[name] = self.data_split['X_test'][:, idx]
        
        # Create ratio features
        ratio_features_train = {}
        ratio_features_test = {}
        
        for ratio_name, ratio_info in selected_ratios.items():
            # Handle numerator (can be single biomarker or list)
            num_data = ratio_info['numerator']
            if isinstance(num_data, list):
                # Average multiple biomarkers
                num_train = np.mean([biomarker_intensities_train[b] for b in num_data], axis=0)
                num_test = np.mean([biomarker_intensities_test[b] for b in num_data], axis=0)
            else:
                num_train = biomarker_intensities_train[num_data]
                num_test = biomarker_intensities_test[num_data]
            
            # Handle denominator
            den_data = ratio_info['denominator']
            if isinstance(den_data, list):
                den_train = np.mean([biomarker_intensities_train[b] for b in den_data], axis=0)
                den_test = np.mean([biomarker_intensities_test[b] for b in den_data], axis=0)
            else:
                den_train = biomarker_intensities_train[den_data]
                den_test = biomarker_intensities_test[den_data]
            
            # Calculate ratios
            ratio_features_train[ratio_name] = num_train / (den_train + 1e-8)
            ratio_features_test[ratio_name] = num_test / (den_test + 1e-8)
        
        # Combine all features
        all_features_train = {**biomarker_intensities_train, **ratio_features_train}
        all_features_test = {**biomarker_intensities_test, **ratio_features_test}
        
        # Convert to arrays
        self.biomarker_features_train = np.column_stack(list(all_features_train.values()))
        self.biomarker_features_test = np.column_stack(list(all_features_test.values()))
        self.biomarker_feature_names = list(all_features_train.keys())
        
        # Combine with original features if not biomarker-only
        if not self.biomarker_only:
            self.biomarker_features_train = np.column_stack([
                self.data_split['X_train'],
                self.biomarker_features_train
            ])
            self.biomarker_features_test = np.column_stack([
                self.data_split['X_test'], 
                self.biomarker_features_test
            ])
            # Add wavelength names
            wn_names = [f"wn_{wn:.1f}" for wn in wavelengths]
            self.biomarker_feature_names = wn_names + self.biomarker_feature_names
        
        if self.verbose:
            print(f"Final feature matrix: {self.biomarker_features_train.shape}")
            print(f"Biomarker features: {len(all_features_train)}")
            if not self.biomarker_only:
                print(f"Original features: {len(wavelengths)}")
    
    def _extract_biomarker_features_from_external(self, X_external: np.ndarray) -> np.ndarray:
        """Extract biomarker features from external data."""
        wavelengths = self.data_split['unified_wavelengths']
        
        selected_biomarkers = self.biomarker_database.get_biomarker_subset(self.biomarker_priority)
        selected_ratios = self.biomarker_database.get_ratio_subset(self.biomarker_priority)
        
        # Extract individual biomarkers
        biomarker_intensities = {}
        for name, info in selected_biomarkers.items():
            idx = np.argmin(np.abs(wavelengths - info['target_wavenumber']))
            biomarker_intensities[name] = X_external[:, idx]
        
        # Create ratios
        ratio_features = {}
        for ratio_name, ratio_info in selected_ratios.items():
            # Numerator
            num_data = ratio_info['numerator']
            if isinstance(num_data, list):
                num_vals = np.mean([biomarker_intensities[b] for b in num_data], axis=0)
            else:
                num_vals = biomarker_intensities[num_data]
            
            # Denominator  
            den_data = ratio_info['denominator']
            if isinstance(den_data, list):
                den_vals = np.mean([biomarker_intensities[b] for b in den_data], axis=0)
            else:
                den_vals = biomarker_intensities[den_data]
            
            ratio_features[ratio_name] = num_vals / (den_vals + 1e-8)
        
        # Combine features
        all_features = {**biomarker_intensities, **ratio_features}
        biomarker_features_external = np.column_stack(list(all_features.values()))
        
        # Add original features if not biomarker-only
        if not self.biomarker_only:
            biomarker_features_external = np.column_stack([X_external, biomarker_features_external])
        
        return biomarker_features_external
    
    def fit(self) -> None:
        """Fit the biomarker-enhanced logistic regression model."""
        if self.verbose:
            print("\n=== Fitting Biomarker-Enhanced Logistic Regression ===")
        
        # Prepare features
        X_train = self.biomarker_features_train.copy()
        
        # Apply scaling
        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)
        
        # Fit model
        try:
            self.model.fit(X_train, self.y_train_encoded)
            if self.verbose:
                print("Model fitted successfully on biomarker-enhanced features")
                if hasattr(self.model, 'n_iter_'):
                    print(f"Convergence: {self.model.n_iter_} iterations")
        except Exception as e:
            print(f"Fitting error: {e}")
            raise e
            
        # Analyze biomarker coefficients
        self._analyze_biomarker_coefficients()
    
    def _analyze_biomarker_coefficients(self) -> None:
        """Analyze logistic regression coefficients for biomarker interpretation."""
        if not hasattr(self.model, 'coef_'):
            return
            
        coefficients = self.model.coef_[0] if len(self.label_encoder.classes_) == 2 else self.model.coef_
        
        # Create biomarker analysis
        biomarker_analysis = []
        
        for i, (name, coef) in enumerate(zip(self.biomarker_feature_names, coefficients)):
            # Get clinical meaning
            clinical_meaning = "Unknown"
            for category, meaning in self.biomarker_database.clinical_interpretations.items():
                if category.lower() in name.lower():
                    clinical_meaning = meaning
                    break
            
            # Determine effect direction for binary classification
            if len(self.label_encoder.classes_) == 2:
                if coef > 0:
                    effect = f"Favors {self.label_encoder.classes_[1]}"
                else:
                    effect = f"Favors {self.label_encoder.classes_[0]}"
            else:
                effect = "Multi-class coefficient"
            
            biomarker_analysis.append({
                'feature': name,
                'coefficient': coef,
                'abs_coefficient': abs(coef),
                'clinical_meaning': clinical_meaning,
                'effect_direction': effect
            })
        
        # Sort by importance
        biomarker_analysis.sort(key=lambda x: x['abs_coefficient'], reverse=True)
        self.biomarker_analysis = biomarker_analysis
        
        if self.verbose:
            print(f"\n=== Top 10 Biomarker Coefficients ===")
            for i, analysis in enumerate(biomarker_analysis[:10]):
                print(f"{i+1:2d}. {analysis['feature'][:20]:20s} | "
                      f"Coef: {analysis['coefficient']:8.4f} | "
                      f"{analysis['effect_direction']}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using biomarker features."""
        biomarker_features = self._extract_biomarker_features_from_external(X)
        if self.scaler is not None:
            biomarker_features = self.scaler.transform(biomarker_features)
        return self.model.predict(biomarker_features)
    
    def predict_labels(self, X: np.ndarray) -> List[str]:
        """Predict and decode to original labels."""
        predictions_encoded = self.predict(X)
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using biomarker features."""
        biomarker_features = self._extract_biomarker_features_from_external(X)
        if self.scaler is not None:
            biomarker_features = self.scaler.transform(biomarker_features)
        return self.model.predict_proba(biomarker_features)
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, Any]:
        """Get biomarker feature importance based on logistic regression coefficients."""
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model must be fitted before getting feature importance")
        
        coefficients = self.model.coef_
        
        if len(self.label_encoder.classes_) == 2:
            # Binary classification
            coef_abs = np.abs(coefficients[0])
            feature_importance = [
                {
                    'wavelength': self.biomarker_feature_names[i].replace('wn_', '').replace('_', ' '),
                    'coefficient': coefficients[0][i],
                    'abs_coefficient': coef_abs[i],
                    'rank': rank,
                    'clinical_meaning': self.biomarker_analysis[i]['clinical_meaning'] if self.biomarker_analysis else 'Unknown'
                }
                for rank, i in enumerate(np.argsort(coef_abs)[::-1], 1)
            ]
        else:
            # Multi-class classification
            coef_abs = np.mean(np.abs(coefficients), axis=0)
            feature_importance = [
                {
                    'wavelength': self.biomarker_feature_names[i].replace('wn_', '').replace('_', ' '),
                    'coefficients': {class_name: coefficients[j][i] 
                                   for j, class_name in enumerate(self.label_encoder.classes_)},
                    'avg_abs_coefficient': coef_abs[i],
                    'rank': rank,
                    'clinical_meaning': self.biomarker_analysis[i]['clinical_meaning'] if self.biomarker_analysis else 'Unknown'
                }
                for rank, i in enumerate(np.argsort(coef_abs)[::-1], 1)
            ]
        
        return {
            'feature_importance': feature_importance[:top_n],
            'n_features': len(feature_importance),
            'is_binary': len(self.label_encoder.classes_) == 2,
            'biomarker_enhanced': True
        }
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """Plot biomarker feature importance with clinical annotations."""
        importance_data = self.get_feature_importance(top_n)
        features = importance_data['feature_importance']
        
        if importance_data['is_binary']:
            # Binary classification plot
            wavelengths = [f['wavelength'] for f in features]
            coefficients = [f['coefficient'] for f in features]
            
            plt.figure(figsize=(14, 8))
            colors = ['red' if c < 0 else 'blue' for c in coefficients]
            
            bars = plt.barh(range(len(wavelengths)), coefficients, color=colors, alpha=0.7)
            plt.yticks(range(len(wavelengths)), [f"{w}" for w in wavelengths])
            plt.xlabel('Logistic Regression Coefficient')
            plt.title(f'Top {top_n} Most Important Biomarker Features (Binary Classification)')
            plt.grid(True, alpha=0.3, axis='x')
            
            # Add clinical meanings as text annotations
            for i, (bar, feature) in enumerate(zip(bars, features)):
                clinical = feature.get('clinical_meaning', '')[:30]
                plt.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                        clinical, ha='center', va='center', fontsize=8, alpha=0.8)
            
            # Legend
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='red', alpha=0.7, 
                                     label=f'Favors {self.label_encoder.classes_[0]}')
            blue_patch = mpatches.Patch(color='blue', alpha=0.7, 
                                      label=f'Favors {self.label_encoder.classes_[1]}')
            plt.legend(handles=[red_patch, blue_patch])
            
        else:
            # Multi-class plot
            wavelengths = [f['wavelength'] for f in features]
            avg_coeffs = [f['avg_abs_coefficient'] for f in features]
            
            plt.figure(figsize=(14, 8))
            bars = plt.barh(range(len(wavelengths)), avg_coeffs, color='green', alpha=0.7)
            plt.yticks(range(len(wavelengths)), [f"{w}" for w in wavelengths])
            plt.xlabel('Average Absolute Coefficient')
            plt.title(f'Top {top_n} Most Important Biomarker Features (Multi-class)')
            plt.grid(True, alpha=0.3, axis='x')
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    # Include all other methods from your original LogisticRegressionModel
    # (evaluate, plot_confusion_matrix, predict_new_data, etc.)
    # ... [Same implementations as your LogisticRegression.py]
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate model with same interface as original LogisticRegressionModel."""
        # Prepare test data with biomarker features
        X_test = self.biomarker_features_test
        if self.scaler is not None:
            X_test = self.scaler.transform(X_test)
        
        # Get predictions
        y_pred_encoded = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred_encoded)
        y_true_labels = self.data_split['y_test']
        
        # Calculate metrics (same as original)
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
        
        # Logistic regression specific metrics
        log_loss_score = log_loss(self.y_test_encoded, y_pred_proba)
        
        try:
            if len(self.label_encoder.classes_) == 2:
                auc_roc = roc_auc_score(self.y_test_encoded, y_pred_proba[:, 1])
            else:
                auc_roc = roc_auc_score(self.y_test_encoded, y_pred_proba, multi_class='ovr', average='weighted')
        except ValueError:
            auc_roc = None
        
        # Compatibility metrics
        if len(self.label_encoder.classes_) == 2:
            continuous_predictions = y_pred_proba[:, 1]
            continuous_targets = self.y_test_encoded.astype(float)
        else:
            continuous_predictions = np.max(y_pred_proba, axis=1)
            continuous_targets = self.y_test_encoded.astype(float) / (len(self.label_encoder.classes_) - 1)
            
        mse = mean_squared_error(continuous_targets, continuous_predictions)
        r2 = r2_score(continuous_targets, continuous_predictions)
        
        metrics = {
            'regression': {
                'mean_squared_error': mse,
                'r2_score': r2,
                'note': 'Compatibility metrics from probabilities'
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
            },
            'biomarker_enhanced': True
        }
        
        if self.verbose:
            print("=== Biomarker-Enhanced Logistic Regression Results ===")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Log Loss: {log_loss_score:.4f}")
            if auc_roc:
                print(f"AUC-ROC: {auc_roc:.4f}")
            print(f"F1-weighted: {f1_weighted:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true_labels, y_pred_labels, 
                                      target_names=self.label_encoder.classes_, zero_division=0))
        
        return metrics
    
    # Add other methods (plot_confusion_matrix, predict_new_data, save_model, etc.)
    # Same implementations as your original LogisticRegressionModel
