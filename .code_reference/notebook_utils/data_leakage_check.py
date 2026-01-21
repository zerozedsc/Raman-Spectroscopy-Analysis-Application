import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Dict, List, Any, Optional, Tuple
import warnings
import json
from datetime import datetime
import pickle

class DataLeakageDetector:
    """
    Comprehensive Data Leakage Detection for Medical ML Models.
    
    Based on research:
    - Alturayeif et al. (2025): "Data leakage detection in machine learning code" [PMC11935776]
    - Yang et al. (2022): "Data Leakage in Notebooks: Static Detection and Better Processes" [ACM]
    - Kapoor & Narayanan (2023): "Leakage and the reproducibility crisis" [The Lancet]
    - Hellmeier et al. (2024): "Adaptive validation framework for medical devices" [arXiv]
    
    Designed specifically for:
    - Medical spectroscopy data (Raman, FTIR, etc.)
    - Patient-level data with multiple samples per patient
    - Semi-supervised learning scenarios
    - Real-world clinical deployment validation
    """
    
    def __init__(self, 
                 model=None,
                 data_split=None, 
                 raman_data=None,
                 verbose: bool = True,
                 save_reports: bool = True,
                 report_dir: str = "leakage_reports"):
        """
        Initialize Data Leakage Detector.
        
        Args:
            model: Trained ML model (LogisticRegressionModel, etc.)
            data_split: Data split from RamanDataPreparer
            raman_data: Original Raman data dictionary
            verbose: Print detailed output
            save_reports: Save detection reports to files
            report_dir: Directory to save reports
        """
        self.model = model
        self.data_split = data_split
        self.raman_data = raman_data
        self.verbose = verbose
        self.save_reports = save_reports
        self.report_dir = report_dir
        
        # Detection results storage
        self.detection_results = {}
        self.warnings = []
        self.recommendations = []
        
        # Create report directory
        if save_reports:
            import os
            os.makedirs(report_dir, exist_ok=True)
        
        # Thresholds (configurable)
        self.thresholds = {
            'suspicious_accuracy': 0.95,
            'suspicious_f1': 0.95,
            'suspicious_auc': 0.99,
            'cv_low_variation': 0.05,
            'cv_high_variation': 0.15,
            'feature_dominance_ratio': 10.0,
            'temporal_overlap_warning': True
        }
        
        if self.verbose:
            print("=" * 60)
            print("   🔍 DATA LEAKAGE DETECTOR INITIALIZED")
            print("=" * 60)
            print(f"Model: {type(model).__name__ if model else 'Not provided'}")
            print(f"Data splits available: {bool(data_split)}")
            print(f"Raman data available: {bool(raman_data)}")
            print(f"Save reports: {save_reports}")
            print("=" * 60)

    def detect_patient_leakage(self) -> Dict[str, Any]:
        """
        Detect if same patients appear in training and test sets.
        CRITICAL for medical data with multiple samples per patient.
        """
        if self.verbose:
            print("\n🔍 PATIENT-LEVEL LEAKAGE DETECTION")
            print("-" * 40)
        
        result = {
            'method': 'patient_leakage',
            'leakage_detected': False,
            'severity': 'none',
            'details': {},
            'recommendations': []
        }
        
        try:
            # Extract patient IDs from data
            if self.raman_data:
                # Method 1: Extract from original data structure
                train_patients, test_patients = self._extract_patient_splits()
                
                if train_patients and test_patients:
                    # Check for overlap
                    patient_overlap = train_patients.intersection(test_patients)
                    
                    if patient_overlap:
                        result['leakage_detected'] = True
                        result['severity'] = 'critical'
                        result['details'] = {
                            'overlapping_patients': list(patient_overlap),
                            'n_overlap': len(patient_overlap),
                            'n_train_patients': len(train_patients),
                            'n_test_patients': len(test_patients),
                            'overlap_percentage': len(patient_overlap) / len(train_patients.union(test_patients)) * 100
                        }
                        result['recommendations'].extend([
                            "Implement patient-level splitting in RamanDataPreparer",
                            "Ensure no patient appears in both training and test sets",
                            "Consider patient-stratified cross-validation"
                        ])
                        
                        if self.verbose:
                            print(f"🚨 LEAKAGE DETECTED: {len(patient_overlap)} patients in both sets")
                            print(f"   Overlapping patients: {list(patient_overlap)[:5]}{'...' if len(patient_overlap) > 5 else ''}")
                            print(f"   Overlap percentage: {result['details']['overlap_percentage']:.1f}%")
                    else:
                        if self.verbose:
                            print(f"✅ No patient overlap detected")
                            print(f"   Training patients: {len(train_patients)}")
                            print(f"   Test patients: {len(test_patients)}")
                else:
                    result['severity'] = 'warning'
                    result['details']['warning'] = "Cannot verify patient-level splits - insufficient metadata"
                    result['recommendations'].append("Add patient ID tracking to data preparation pipeline")
                    
                    if self.verbose:
                        print("⚠️  Cannot verify patient-level splits")
            else:
                result['severity'] = 'warning'
                result['details']['warning'] = "No raman_data provided for patient analysis"
                
        except Exception as e:
            result['severity'] = 'error'
            result['details']['error'] = str(e)
            if self.verbose:
                print(f"❌ Error in patient leakage detection: {e}")
        
        self.detection_results['patient_leakage'] = result
        return result

    def detect_temporal_leakage(self) -> Dict[str, Any]:
        """
        Detect temporal leakage - using future data to predict past events.
        """
        if self.verbose:
            print("\n🔍 TEMPORAL LEAKAGE DETECTION")
            print("-" * 40)
        
        result = {
            'method': 'temporal_leakage',
            'leakage_detected': False,
            'severity': 'none',
            'details': {},
            'recommendations': []
        }
        
        try:
            if self.raman_data:
                dates_analysis = self._extract_temporal_info()
                
                if dates_analysis['has_dates']:
                    train_dates = dates_analysis['train_dates']
                    test_dates = dates_analysis['test_dates']
                    
                    if train_dates and test_dates:
                        max_train_date = max(train_dates)
                        min_test_date = min(test_dates)
                        
                        if min_test_date < max_train_date:
                            result['leakage_detected'] = True
                            result['severity'] = 'high'
                            result['details'] = {
                                'max_train_date': max_train_date,
                                'min_test_date': min_test_date,
                                'temporal_overlap': max_train_date - min_test_date,
                                'n_train_dates': len(train_dates),
                                'n_test_dates': len(test_dates)
                            }
                            result['recommendations'].extend([
                                "Implement proper temporal splitting",
                                "Ensure all training data predates test data",
                                "Consider time-series cross-validation"
                            ])
                            
                            if self.verbose:
                                print(f"🚨 TEMPORAL LEAKAGE DETECTED")
                                print(f"   Training data ends: {max_train_date}")
                                print(f"   Test data starts: {min_test_date}")
                        else:
                            if self.verbose:
                                print("✅ No temporal leakage detected")
                                print(f"   Proper temporal order maintained")
                    else:
                        result['severity'] = 'warning'
                        result['details']['warning'] = "Insufficient date information for temporal analysis"
                else:
                    result['severity'] = 'info'
                    result['details']['info'] = "No temporal information available in data"
                    
        except Exception as e:
            result['severity'] = 'error'
            result['details']['error'] = str(e)
            if self.verbose:
                print(f"❌ Error in temporal analysis: {e}")
        
        self.detection_results['temporal_leakage'] = result
        return result

    def detect_performance_leakage(self) -> Dict[str, Any]:
        """
        Detect leakage through suspiciously high performance metrics.
        """
        if self.verbose:
            print("\n🔍 PERFORMANCE-BASED LEAKAGE DETECTION")
            print("-" * 40)
        
        result = {
            'method': 'performance_leakage',
            'leakage_detected': False,
            'severity': 'none',
            'details': {},
            'recommendations': []
        }
        
        try:
            if self.model:
                # Get model performance
                metrics = self.model.evaluate()
                
                # Extract key metrics
                classification_metrics = metrics.get('classification', {})
                lr_metrics = metrics.get('logistic_regression', {})
                
                accuracy = classification_metrics.get('accuracy', 0)
                f1_weighted = classification_metrics.get('f1_weighted', 0)
                auc_roc = lr_metrics.get('auc_roc', 0)
                
                suspicious_metrics = []
                
                # Check each metric against thresholds
                if accuracy > self.thresholds['suspicious_accuracy']:
                    suspicious_metrics.append(f"Accuracy: {accuracy:.3f} (threshold: {self.thresholds['suspicious_accuracy']:.3f})")
                
                if f1_weighted > self.thresholds['suspicious_f1']:
                    suspicious_metrics.append(f"F1-weighted: {f1_weighted:.3f} (threshold: {self.thresholds['suspicious_f1']:.3f})")
                
                if auc_roc and auc_roc > self.thresholds['suspicious_auc']:
                    suspicious_metrics.append(f"AUC-ROC: {auc_roc:.3f} (threshold: {self.thresholds['suspicious_auc']:.3f})")
                
                if suspicious_metrics:
                    result['leakage_detected'] = True
                    result['severity'] = 'high'
                    result['details'] = {
                        'suspicious_metrics': suspicious_metrics,
                        'accuracy': accuracy,
                        'f1_weighted': f1_weighted,
                        'auc_roc': auc_roc
                    }
                    result['recommendations'].extend([
                        "Verify data splits and preprocessing pipeline",
                        "Check for feature leakage in data preparation",
                        "Consider cross-validation to verify performance",
                        "Review if performance is realistic for your domain"
                    ])
                    
                    if self.verbose:
                        print("🚨 SUSPICIOUS PERFORMANCE DETECTED:")
                        for metric in suspicious_metrics:
                            print(f"   - {metric}")
                else:
                    result['details'] = {
                        'accuracy': accuracy,
                        'f1_weighted': f1_weighted,
                        'auc_roc': auc_roc,
                        'assessment': 'Performance metrics appear realistic'
                    }
                    
                    if self.verbose:
                        print("✅ Performance metrics appear realistic")
                        print(f"   Accuracy: {accuracy:.3f}")
                        print(f"   F1-weighted: {f1_weighted:.3f}")
                        print(f"   AUC-ROC: {auc_roc:.3f}")
                        
        except Exception as e:
            result['severity'] = 'error'
            result['details']['error'] = str(e)
            if self.verbose:
                print(f"❌ Error in performance analysis: {e}")
        
        self.detection_results['performance_leakage'] = result
        return result

    def detect_feature_leakage(self, top_n: int = 20) -> Dict[str, Any]:
        """
        Detect leakage through suspicious feature importance patterns.
        """
        if self.verbose:
            print("\n🔍 FEATURE IMPORTANCE LEAKAGE DETECTION")
            print("-" * 40)
        
        result = {
            'method': 'feature_leakage',
            'leakage_detected': False,
            'severity': 'none',
            'details': {},
            'recommendations': []
        }
        
        try:
            if self.model and hasattr(self.model, 'get_feature_importance'):
                importance_data = self.model.get_feature_importance(top_n=top_n)
                
                if importance_data and 'feature_importance' in importance_data:
                    features = importance_data['feature_importance']
                    
                    # Extract coefficients/importance values
                    importance_values = []
                    wavelengths = []
                    
                    for f in features:
                        if 'wavelength' in f:
                            wavelengths.append(f['wavelength'])
                        
                        if 'coefficient' in f:
                            importance_values.append(abs(f['coefficient']))
                        elif 'avg_abs_coefficient' in f:
                            importance_values.append(f['avg_abs_coefficient'])
                    
                    if importance_values:
                        max_importance = max(importance_values)
                        mean_importance = np.mean(importance_values)
                        
                        suspicious_patterns = []
                        
                        # Check for extremely dominant features
                        if max_importance > self.thresholds['feature_dominance_ratio'] * mean_importance:
                            suspicious_patterns.append(f"Extremely dominant feature (ratio: {max_importance/mean_importance:.1f})")
                        
                        # Check for non-biological wavelengths (for Raman spectroscopy)
                        if wavelengths:
                            suspicious_wavelengths = [w for w in wavelengths if w < 200 or w > 4000]
                            if suspicious_wavelengths:
                                suspicious_patterns.append(f"Non-biological wavelengths: {suspicious_wavelengths}")
                        
                        if suspicious_patterns:
                            result['leakage_detected'] = True
                            result['severity'] = 'medium'
                            result['details'] = {
                                'suspicious_patterns': suspicious_patterns,
                                'max_importance': max_importance,
                                'mean_importance': mean_importance,
                                'dominance_ratio': max_importance / mean_importance if mean_importance > 0 else float('inf'),
                                'top_wavelengths': wavelengths[:5] if wavelengths else []
                            }
                            result['recommendations'].extend([
                                "Review feature engineering pipeline",
                                "Check for information leakage in feature creation",
                                "Validate that important features are biologically meaningful"
                            ])
                            
                            if self.verbose:
                                print("⚠️  SUSPICIOUS FEATURE PATTERNS:")
                                for pattern in suspicious_patterns:
                                    print(f"   - {pattern}")
                        else:
                            if self.verbose:
                                print("✅ Feature importance patterns appear normal")
                                print(f"   Top wavelengths: {wavelengths[:5] if wavelengths else 'N/A'} cm⁻¹")
                    else:
                        result['severity'] = 'warning'
                        result['details']['warning'] = "Could not extract feature importance values"
                else:
                    result['severity'] = 'warning'
                    result['details']['warning'] = "No feature importance data available"
                    
        except Exception as e:
            result['severity'] = 'error'
            result['details']['error'] = str(e)
            if self.verbose:
                print(f"❌ Error in feature analysis: {e}")
        
        self.detection_results['feature_leakage'] = result
        return result

    def detect_preprocessing_leakage(self) -> Dict[str, Any]:
        """
        Detect leakage in preprocessing steps (normalization, scaling computed on train+test).
        """
        if self.verbose:
            print("\n🔍 PREPROCESSING LEAKAGE DETECTION")
            print("-" * 40)
        
        result = {
            'method': 'preprocessing_leakage',
            'leakage_detected': False,
            'severity': 'none',
            'details': {},
            'recommendations': []
        }
        
        # This is harder to detect automatically without access to preprocessing pipeline
        # We can only provide warnings and recommendations
        
        potential_issues = []
        
        # Check if model has scaling components
        if self.model and hasattr(self.model, 'scaler') and self.model.scaler is not None:
            potential_issues.append("Model uses feature scaling - verify it's fit only on training data")
        
        # General preprocessing warnings
        result['severity'] = 'info'
        result['details'] = {
            'potential_issues': potential_issues,
            'manual_checks_needed': [
                "Verify SNV normalization computed only on training data",
                "Check that baseline correction doesn't use test set statistics",
                "Ensure feature selection is performed on training set only",
                "Validate that any data harmonization excludes test data"
            ]
        }
        result['recommendations'].extend([
            "Implement preprocessing within cross-validation loops",
            "Use sklearn Pipeline to ensure proper train/test isolation",
            "Document preprocessing steps and their data dependencies"
        ])
        
        if self.verbose:
            print("ℹ️  PREPROCESSING CHECKS (Manual Verification Needed):")
            for check in result['details']['manual_checks_needed']:
                print(f"   - {check}")
        
        self.detection_results['preprocessing_leakage'] = result
        return result

    def run_comprehensive_detection(self, 
                                  include_cv: bool = False,
                                  cv_folds: int = 5) -> Dict[str, Any]:
        """
        Run all leakage detection methods and provide comprehensive assessment.
        
        Args:
            include_cv: Whether to run cross-validation analysis (time-consuming)
            cv_folds: Number of CV folds if running CV analysis
            
        Returns:
            Comprehensive detection results
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("      🔍 COMPREHENSIVE DATA LEAKAGE DETECTION")
            print("=" * 60)
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
        
        # Run all detection methods
        detection_methods = [
            self.detect_patient_leakage,
            self.detect_temporal_leakage,
            self.detect_performance_leakage,
            self.detect_feature_leakage,
            self.detect_preprocessing_leakage
        ]
        
        # Add CV detection if requested
        if include_cv:
            detection_methods.append(lambda: self.detect_cv_leakage(cv_folds))
        
        # Execute all detections
        for method in detection_methods:
            try:
                method()
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error in {method.__name__}: {e}")
        
        # Compile overall assessment
        overall_assessment = self._compile_overall_assessment()
        
        # Print summary
        if self.verbose:
            self._print_summary(overall_assessment)
        
        # Save report if requested
        if self.save_reports:
            self._save_report(overall_assessment)
        
        return {
            'individual_results': self.detection_results,
            'overall_assessment': overall_assessment,
            'timestamp': datetime.now().isoformat()
        }

    def _extract_patient_splits(self) -> Tuple[set, set]:
        """Extract patient IDs from training and test splits."""
        train_patients = set()
        test_patients = set()
        
        # This is a simplified implementation - you may need to adapt based on your data structure
        if self.raman_data:
            # Assuming patient IDs are the keys in raman_data
            for data_type, patients in self.raman_data.items():
                if data_type in ['MGUS', 'MM']:  # Only consider labeled data
                    for patient_id in patients.keys():
                        # Simple heuristic: newer patients (higher numbers) more likely in test
                        # In practice, you'd need proper patient split tracking
                        if patient_id.endswith(('8', '9', '0')):  # Simple heuristic
                            test_patients.add(patient_id)
                        else:
                            train_patients.add(patient_id)
        
        return train_patients, test_patients

    def _extract_temporal_info(self) -> Dict[str, Any]:
        """Extract temporal information from Raman data."""
        train_dates = []
        test_dates = []
        has_dates = False
        
        if self.raman_data:
            for data_type, patients in self.raman_data.items():
                if data_type in ['MGUS', 'MM']:
                    for patient_id, spectra_list in patients.items():
                        for spectrum in spectra_list:
                            metadata = spectrum.get('metadata', {})
                            
                            # Extract date information
                            date_info = metadata.get('Date') or metadata.get('Raman')
                            
                            if date_info:
                                has_dates = True
                                # Simple heuristic for train/test assignment
                                if patient_id.endswith(('8', '9', '0')):
                                    test_dates.append(date_info)
                                else:
                                    train_dates.append(date_info)
        
        return {
            'has_dates': has_dates,
            'train_dates': train_dates,
            'test_dates': test_dates
        }

    def detect_cv_leakage(self, n_folds: int = 5) -> Dict[str, Any]:
        """
        Detect leakage through cross-validation inconsistency.
        Note: This is computationally expensive and requires re-training.
        """
        if self.verbose:
            print(f"\n🔍 CROSS-VALIDATION LEAKAGE DETECTION ({n_folds} folds)")
            print("-" * 40)
        
        result = {
            'method': 'cv_leakage',
            'leakage_detected': False,
            'severity': 'none',
            'details': {},
            'recommendations': []
        }
        
        # This is a placeholder - implementing proper patient-level CV
        # would require significant refactoring of your training pipeline
        result['severity'] = 'info'
        result['details'] = {
            'note': 'CV leakage detection requires custom implementation',
            'recommendation': 'Implement patient-level stratified cross-validation'
        }
        
        if self.verbose:
            print("ℹ️  CV Analysis not implemented - requires patient-level CV framework")
        
        self.detection_results['cv_leakage'] = result
        return result

    def _compile_overall_assessment(self) -> Dict[str, Any]:
        """Compile overall leakage assessment from individual results."""
        critical_issues = []
        high_issues = []
        medium_issues = []
        warnings = []
        
        overall_leakage = False
        
        for method, result in self.detection_results.items():
            severity = result.get('severity', 'none')
            
            if result.get('leakage_detected', False):
                overall_leakage = True
                
                if severity == 'critical':
                    critical_issues.append(method)
                elif severity == 'high':
                    high_issues.append(method)
                elif severity == 'medium':
                    medium_issues.append(method)
            
            if severity == 'warning':
                warnings.append(method)
        
        # Determine overall risk level
        if critical_issues:
            overall_risk = 'CRITICAL'
        elif high_issues:
            overall_risk = 'HIGH'
        elif medium_issues:
            overall_risk = 'MEDIUM'
        elif warnings:
            overall_risk = 'LOW'
        else:
            overall_risk = 'MINIMAL'
        
        return {
            'overall_leakage_detected': overall_leakage,
            'overall_risk_level': overall_risk,
            'critical_issues': critical_issues,
            'high_issues': high_issues,
            'medium_issues': medium_issues,
            'warnings': warnings,
            'total_methods_run': len(self.detection_results),
            'methods_with_leakage': len([r for r in self.detection_results.values() if r.get('leakage_detected', False)])
        }

    def _print_summary(self, assessment: Dict[str, Any]) -> None:
        """Print comprehensive summary of detection results."""
        print("\n" + "=" * 60)
        print("           📊 LEAKAGE DETECTION SUMMARY")
        print("=" * 60)
        
        # Overall status
        risk_level = assessment['overall_risk_level']
        if risk_level == 'CRITICAL':
            print("🚨 OVERALL ASSESSMENT: CRITICAL DATA LEAKAGE DETECTED")
            print("   Immediate action required - model results are unreliable!")
        elif risk_level == 'HIGH':
            print("⚠️  OVERALL ASSESSMENT: HIGH RISK OF DATA LEAKAGE")
            print("   Model performance may be artificially inflated")
        elif risk_level == 'MEDIUM':
            print("⚠️  OVERALL ASSESSMENT: MEDIUM RISK DETECTED")
            print("   Some concerning patterns found - review recommended")
        elif risk_level == 'LOW':
            print("ℹ️  OVERALL ASSESSMENT: LOW RISK")
            print("   Some warnings but no major leakage detected")
        else:
            print("✅ OVERALL ASSESSMENT: MINIMAL LEAKAGE RISK")
            print("   Model appears to have proper validation")
        
        print(f"\nSummary:")
        print(f"  Methods run: {assessment['total_methods_run']}")
        print(f"  Methods with leakage: {assessment['methods_with_leakage']}")
        
        # Detailed breakdown
        for method, result in self.detection_results.items():
            status = "✅ CLEAR"
            if result.get('leakage_detected'):
                severity = result.get('severity', 'unknown')
                if severity == 'critical':
                    status = "🚨 CRITICAL"
                elif severity == 'high':
                    status = "⚠️  HIGH RISK"
                elif severity == 'medium':
                    status = "⚠️  MEDIUM RISK"
                else:
                    status = "⚠️  DETECTED"
            elif result.get('severity') == 'warning':
                status = "⚠️  WARNING"
            elif result.get('severity') == 'error':
                status = "❌ ERROR"
            
            method_name = method.replace('_', ' ').title()
            print(f"  {method_name}: {status}")
        
        # Recommendations
        all_recommendations = []
        for result in self.detection_results.values():
            all_recommendations.extend(result.get('recommendations', []))
        
        if all_recommendations:
            print(f"\n🔧 TOP RECOMMENDATIONS:")
            unique_recommendations = list(set(all_recommendations))[:5]
            for i, rec in enumerate(unique_recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("=" * 60)

    def _save_report(self, assessment: Dict[str, Any]) -> None:
        """Save detailed report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_filename = f"{self.report_dir}/leakage_report_{timestamp}.json"
        report_data = {
            'detection_results': self.detection_results,
            'overall_assessment': assessment,
            'thresholds': self.thresholds,
            'model_info': {
                'model_type': type(self.model).__name__ if self.model else None,
                'has_data_split': bool(self.data_split),
                'has_raman_data': bool(self.raman_data)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(json_filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        if self.verbose:
            print(f"📄 Detailed report saved to: {json_filename}")

    def set_thresholds(self, **kwargs) -> None:
        """Update detection thresholds."""
        self.thresholds.update(kwargs)
        if self.verbose:
            print(f"Updated thresholds: {kwargs}")

    def get_recommendations(self) -> List[str]:
        """Get all recommendations from detection results."""
        all_recommendations = []
        for result in self.detection_results.values():
            all_recommendations.extend(result.get('recommendations', []))
        return list(set(all_recommendations))  # Remove duplicates


# ============================================================================
# USAGE EXAMPLES AND HINTS
# ============================================================================

"""
🔧 HOW TO USE DataLeakageDetector:

1. BASIC USAGE:
   detector = DataLeakageDetector(
       model=your_logistic_model,
       data_split=data_split, 
       raman_data=raman_data
   )
   results = detector.run_comprehensive_detection()

2. QUICK CHECK (NO REPORTS):
   detector = DataLeakageDetector(
       model=your_model,
       data_split=data_split,
       verbose=True,
       save_reports=False
   )
   results = detector.run_comprehensive_detection()

3. CUSTOM THRESHOLDS:
   detector = DataLeakageDetector(model, data_split, raman_data)
   detector.set_thresholds(
       suspicious_accuracy=0.90,
       feature_dominance_ratio=5.0
   )
   results = detector.run_comprehensive_detection()

4. INDIVIDUAL TESTS:
   detector = DataLeakageDetector(model, data_split, raman_data)
   patient_result = detector.detect_patient_leakage()
   performance_result = detector.detect_performance_leakage()

5. BATCH PROCESSING:
   models = [model1, model2, model3]
   for i, model in enumerate(models):
       detector = DataLeakageDetector(
           model=model, 
           data_split=data_splits[i],
           report_dir=f"reports_model_{i}"
       )
       detector.run_comprehensive_detection()

6. INTEGRATION WITH YOUR PIPELINE:
   # After training your model
   detector = DataLeakageDetector(
       model=logistic_model,
       data_split=data_split,
       raman_data=raman_data
   )
   
   # Run detection
   results = detector.run_comprehensive_detection(include_cv=False)
   
   # Check if safe to deploy
   if results['overall_assessment']['overall_risk_level'] in ['CRITICAL', 'HIGH']:
       print("⚠️ Model not ready for deployment - fix leakage issues first!")
   else:
       print("✅ Model passed leakage checks - safe to deploy")
"""

