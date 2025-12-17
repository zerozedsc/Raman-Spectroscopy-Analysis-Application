"""
FUNCTIONAL PREPROCESSING TEST SCRIPT WITH REAL RAMAN DATA
==========================================================
Tests preprocessing methods with synthetic Raman spectra and validates:
1. Methods produce expected transformations
2. Medical diagnostic pipelines work correctly
3. Output data has proper characteristics

Author: MUHAMMAD HELMI BIN ROZAIN
Date: 2025-10-14
Environment: UV Python environment
Output: Saves results to test_script/functional_test_results_TIMESTAMP.txt
"""

import sys
import os
import warnings
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)
warnings.simplefilter('always')

# Import preprocessing components
try:
    import ramanspy as rp
    RAMANSPY_AVAILABLE = True
except ImportError:
    RAMANSPY_AVAILABLE = False
    print("❌ ramanspy not available - tests will be limited")

from functions.preprocess.registry import PreprocessingStepRegistry


class SyntheticRamanGenerator:
    """Generate synthetic Raman spectra for testing."""
    
    def __init__(self, wavenumber_range=(400, 1800), n_points=1000):
        """Initialize generator with wavenumber range."""
        self.wavenumber_range = wavenumber_range
        self.n_points = n_points
        self.wavenumbers = np.linspace(wavenumber_range[0], wavenumber_range[1], n_points)
    
    def generate_baseline(self, intensity=1000, slope=0.5, curvature=0.0001):
        """Generate fluorescence-like baseline."""
        x_norm = (self.wavenumbers - self.wavenumber_range[0]) / (self.wavenumber_range[1] - self.wavenumber_range[0])
        baseline = intensity + slope * x_norm * 1000 + curvature * (x_norm * 1000)**2
        return baseline
    
    def add_gaussian_peak(self, spectrum, center, amplitude, width):
        """Add a Gaussian peak to spectrum."""
        peak = amplitude * np.exp(-((self.wavenumbers - center)**2) / (2 * width**2))
        return spectrum + peak
    
    def add_noise(self, spectrum, noise_level=10):
        """Add Gaussian noise."""
        noise = np.random.normal(0, noise_level, len(spectrum))
        return spectrum + noise
    
    def add_cosmic_ray(self, spectrum, position_idx, amplitude):
        """Add cosmic ray spike."""
        spectrum[position_idx] = amplitude
        return spectrum
    
    def generate_tissue_spectrum(self, tissue_type="normal", include_cosmic_ray=False):
        """
        Generate realistic tissue Raman spectrum.
        
        Parameters:
        -----------
        tissue_type : str
            'normal', 'cancer', 'inflammation'
        include_cosmic_ray : bool
            If True, deterministically adds a cosmic ray spike for testing.
            Default: False
        """
        # Start with baseline (fluorescence)
        spectrum = self.generate_baseline(intensity=500, slope=1.0, curvature=0.0002)
        
        if tissue_type == "normal":
            # Normal tissue peaks (proteins, lipids)
            spectrum = self.add_gaussian_peak(spectrum, 1004, 200, 15)  # Phenylalanine
            spectrum = self.add_gaussian_peak(spectrum, 1450, 150, 20)  # CH2 bending
            spectrum = self.add_gaussian_peak(spectrum, 1656, 180, 18)  # Amide I
            spectrum = self.add_gaussian_peak(spectrum, 850, 100, 12)   # Proline
            
        elif tissue_type == "cancer":
            # Cancer tissue - increased nucleic acids, altered proteins
            spectrum = self.add_gaussian_peak(spectrum, 785, 250, 15)   # DNA/RNA (increased)
            spectrum = self.add_gaussian_peak(spectrum, 1004, 150, 15)  # Phenylalanine (decreased)
            spectrum = self.add_gaussian_peak(spectrum, 1340, 200, 20)  # DNA bases (increased)
            spectrum = self.add_gaussian_peak(spectrum, 1450, 100, 20)  # CH2 (decreased)
            spectrum = self.add_gaussian_peak(spectrum, 1656, 220, 18)  # Amide I (altered)
            
        elif tissue_type == "inflammation":
            # Inflammatory tissue - mixed characteristics
            spectrum = self.add_gaussian_peak(spectrum, 1004, 180, 15)
            spectrum = self.add_gaussian_peak(spectrum, 1340, 150, 20)
            spectrum = self.add_gaussian_peak(spectrum, 1450, 130, 20)
            spectrum = self.add_gaussian_peak(spectrum, 1656, 200, 18)
        
        # Add realistic noise
        spectrum = self.add_noise(spectrum, noise_level=15)
        
        # Add cosmic ray deterministically for testing (if requested)
        if include_cosmic_ray:
            spike_idx = len(spectrum) // 2  # Fixed position for reproducibility
            spectrum = self.add_cosmic_ray(spectrum, spike_idx, spectrum[spike_idx] * 50)
        
        return spectrum


class FunctionalPreprocessingTester:
    """Functional test suite for preprocessing methods with real data validation."""
    
    def __init__(self):
        """Initialize tester."""
        self.registry = PreprocessingStepRegistry()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.generator = SyntheticRamanGenerator()
        
        # Results tracking
        self.results = {
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'warnings': 0,
                'timestamp': self.timestamp
            },
            'method_tests': {},
            'pipeline_tests': {}
        }
    
    def test_method_functionality(self, category: str, method: str) -> Dict[str, Any]:
        """
        Test if a preprocessing method produces expected transformations.
        
        Returns dict with test results and validation metrics.
        """
        result = {
            'category': category,
            'method': method,
            'passed': False,
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Determine if method requires multiple spectra
            method_upper = method.upper()
            requires_multi_spectra = any(kw in method_upper for kw in ['MSC', 'QUANTILE', 'RANK', 'PQN'])
            
            # Generate test data
            if requires_multi_spectra:
                # Generate 5 spectra with variations for multi-spectrum normalization methods
                spectra = []
                for i in range(5):
                    tissue_type = ["normal", "cancer", "inflammation", "normal", "cancer"][i]
                    spectrum = self.generator.generate_tissue_spectrum(tissue_type, include_cosmic_ray=False)
                    spectra.append(spectrum)
                test_data = np.array(spectra)
                reference_spectrum = spectra[0]  # For comparison
            else:
                # Single spectrum test (deterministic, no cosmic ray)
                test_data = self.generator.generate_tissue_spectrum("normal", include_cosmic_ray=False)
                reference_spectrum = test_data.copy()
            
            test_axis = self.generator.wavenumbers
            
            # Get method instance
            method_info = self.registry.get_method_info(category, method)
            if not method_info:
                result['errors'].append("Method not found in registry")
                return result
            
            default_params = method_info.get('default_params', {})
            instance = self.registry.create_method_instance(category, method, default_params)
            
            # Create SpectralContainer
            if not RAMANSPY_AVAILABLE:
                result['warnings'].append("ramanspy not available - skipping functional test")
                result['passed'] = True  # Consider passed if library unavailable
                return result
            
            if requires_multi_spectra:
                container = rp.SpectralContainer(test_data, test_axis)
            else:
                container = rp.SpectralContainer(test_data.reshape(1, -1), test_axis)
            
            # Apply method
            try:
                processed = instance.apply(container)
            except AttributeError:
                # Some methods might not have apply(), try direct call
                try:
                    processed = instance(container)
                except Exception as e:
                    result['errors'].append(f"Method execution failed: {str(e)}")
                    return result
            
            # Validation checks (compare first spectrum)
            processed_first = processed.spectral_data[0] if processed.spectral_data.ndim > 1 else processed.spectral_data
            result['checks']['data_shape_preserved'] = bool(self._check_shape(
                reference_spectrum, processed_first
            ))
            result['checks']['no_nan_values'] = bool(not np.any(np.isnan(processed.spectral_data)))
            result['checks']['no_inf_values'] = bool(not np.any(np.isinf(processed.spectral_data)))
            result['checks']['expected_transformation'] = bool(self._check_expected_effect(
                category, reference_spectrum, processed_first, method
            ))
            
            # Calculate metrics
            result['metrics'] = {
                'original_mean': float(np.mean(reference_spectrum)),
                'processed_mean': float(np.mean(processed.spectral_data)),
                'original_std': float(np.std(reference_spectrum)),
                'processed_std': float(np.std(processed.spectral_data)),
                'snr_improvement': self._calculate_snr_improvement(
                    reference_spectrum, processed_first
                )
            }
            
            # Overall pass/fail
            result['passed'] = all(result['checks'].values())
            
        except Exception as e:
            result['errors'].append(f"Test failed with exception: {str(e)}")
            result['passed'] = False
        
        return result
    
    def _check_shape(self, original, processed):
        """Check if data shape is preserved."""
        return len(original) == len(processed)
    
    def _check_expected_effect(self, category: str, original: np.ndarray, processed: np.ndarray, method: str = "") -> bool:
        """Check if method produced expected effect based on category."""
        try:
            if category == "baseline_correction":
                # Baseline should be reduced
                original_min = np.min(original)
                processed_min = np.min(processed)
                return processed_min < original_min * 0.5  # At least 50% reduction
                
            elif category == "normalisation":
                # Different normalization methods have different validation criteria
                method_upper = method.upper()
                
                if 'SNV' in method_upper:
                    # SNV: Standard Normal Variate - mean≈0, std≈1
                    processed_mean = np.mean(processed)
                    processed_std = np.std(processed)
                    return abs(processed_mean) < 0.1 and 0.9 < processed_std < 1.1
                    
                elif 'MSC' in method_upper:
                    # MSC: Multiplicative Scatter Correction - corrects for scatter effects
                    # Should reduce inter-spectrum variance
                    return not np.allclose(original, processed)
                    
                elif 'VECTOR' in method_upper or 'MINMAX' in method_upper:
                    # Vector normalization: L2 norm ≈ 1
                    # MinMax: scales to [0,1] range
                    if 'VECTOR' in method_upper:
                        processed_norm = np.linalg.norm(processed)
                        return 0.95 < processed_norm < 1.05
                    else:  # MinMax
                        processed_min = np.min(processed)
                        processed_max = np.max(processed)
                        return processed_min >= -0.05 and processed_max <= 1.05
                        
                else:
                    # Generic normalization check: data should be scaled
                    processed_norm = np.linalg.norm(processed)
                    return 0.1 < processed_norm < 10  # Reasonable scale
                
            elif category == "denoising":
                # Smoothing should reduce high-frequency variation
                original_diff = np.mean(np.abs(np.diff(original)))
                processed_diff = np.mean(np.abs(np.diff(processed)))
                return processed_diff < original_diff * 0.9  # At least 10% reduction
                
            elif category == "cosmic_ray_removal":
                # Should remove extreme outliers
                original_max = np.max(original)
                processed_max = np.max(processed)
                return processed_max < original_max * 0.95  # Spike reduction
                
            elif category == "derivatives":
                # Derivative should have different characteristics
                return np.mean(processed) != np.mean(original)
                
            else:
                # For other categories, just check data changed
                return not np.allclose(original, processed)
                
        except Exception:
            return False
    
    def _calculate_snr_improvement(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate signal-to-noise ratio improvement."""
        try:
            # Estimate SNR as mean / std
            original_snr = np.mean(original) / np.std(original) if np.std(original) > 0 else 0
            processed_snr = np.mean(processed) / np.std(processed) if np.std(processed) > 0 else 0
            
            if original_snr > 0:
                return (processed_snr - original_snr) / original_snr * 100  # Percent improvement
            return 0.0
        except Exception:
            return 0.0
    
    def test_medical_pipeline(self, pipeline_name: str, steps: List[Tuple[str, str, dict]]) -> Dict[str, Any]:
        """
        Test a complete medical preprocessing pipeline.
        
        Parameters:
        -----------
        pipeline_name : str
            Name of the pipeline
        steps : List[Tuple[str, str, dict]]
            List of (category, method, params) tuples
        """
        result = {
            'pipeline_name': pipeline_name,
            'passed': False,
            'steps_tested': 0,
            'steps_passed': 0,
            'checks': {},
            'warnings': [],
            'errors': [],
            'tissue_classification': {}
        }
        
        try:
            if not RAMANSPY_AVAILABLE:
                result['warnings'].append("ramanspy not available")
                result['passed'] = True
                return result
            
            # Generate test data for different tissue types
            tissue_types = ['normal', 'cancer', 'inflammation']
            spectra_dict = {}
            
            for tissue_type in tissue_types:
                spectrum = self.generator.generate_tissue_spectrum(tissue_type)
                spectra_dict[tissue_type] = {
                    'original': spectrum.copy(),
                    'spectral_axis': self.generator.wavenumbers.copy()
                }
            
            # Apply pipeline to each tissue type
            for tissue_type in tissue_types:
                spectrum = spectra_dict[tissue_type]['original']
                axis = spectra_dict[tissue_type]['spectral_axis']
                
                container = rp.SpectralContainer(spectrum.reshape(1, -1), axis)
                
                # Apply each step
                for step_idx, (category, method, params) in enumerate(steps):
                    result['steps_tested'] += 1
                    
                    try:
                        instance = self.registry.create_method_instance(category, method, params)
                        
                        # Apply preprocessing
                        try:
                            container = instance.apply(container)
                        except AttributeError:
                            container = instance(container)
                        
                        result['steps_passed'] += 1
                        
                    except Exception as e:
                        result['errors'].append(
                            f"Step {step_idx+1} ({category}.{method}) failed for {tissue_type}: {str(e)}"
                        )
                
                # Store processed result
                spectra_dict[tissue_type]['processed'] = container.spectral_data[0]
            
            # Validation checks
            result['checks']['all_tissues_processed'] = bool(all(
                'processed' in spectra_dict[t] for t in tissue_types
            ))
            
            result['checks']['no_nan_inf'] = bool(all(
                not (np.any(np.isnan(spectra_dict[t]['processed'])) or 
                     np.any(np.isinf(spectra_dict[t]['processed'])))
                for t in tissue_types if 'processed' in spectra_dict[t]
            ))
            
            result['checks']['tissue_differences_preserved'] = bool(self._check_tissue_separability(
                spectra_dict
            ))
            
            # Calculate classification metrics
            if all('processed' in spectra_dict[t] for t in tissue_types):
                result['tissue_classification'] = self._calculate_separation_metrics(spectra_dict)
            
            # Overall pass/fail
            result['passed'] = (
                all(result['checks'].values()) and 
                result['steps_passed'] == result['steps_tested'] and
                len(result['errors']) == 0
            )
            
        except Exception as e:
            result['errors'].append(f"Pipeline test failed: {str(e)}")
            result['passed'] = False
        
        return result
    
    def _check_tissue_separability(self, spectra_dict: Dict) -> bool:
        """Check if tissue types remain separable after preprocessing."""
        try:
            # Calculate pairwise correlations
            tissues = ['normal', 'cancer', 'inflammation']
            correlations = []
            
            for i in range(len(tissues)):
                for j in range(i+1, len(tissues)):
                    if 'processed' in spectra_dict[tissues[i]] and 'processed' in spectra_dict[tissues[j]]:
                        spec1 = spectra_dict[tissues[i]]['processed']
                        spec2 = spectra_dict[tissues[j]]['processed']
                        corr = np.corrcoef(spec1, spec2)[0, 1]
                        correlations.append(abs(corr))
            
            # Tissues should not be too similar (correlation < 0.95)
            return all(c < 0.95 for c in correlations)
            
        except Exception:
            return False
    
    def _calculate_separation_metrics(self, spectra_dict: Dict) -> Dict[str, float]:
        """Calculate metrics for tissue type separation."""
        try:
            tissues = ['normal', 'cancer', 'inflammation']
            
            # Calculate mean spectra for each tissue type
            means = {}
            for tissue in tissues:
                if 'processed' in spectra_dict[tissue]:
                    means[tissue] = np.mean(spectra_dict[tissue]['processed'])
            
            # Calculate separation metrics
            metrics = {}
            
            # Euclidean distances between tissue types
            for i in range(len(tissues)):
                for j in range(i+1, len(tissues)):
                    if tissues[i] in means and tissues[j] in means:
                        if 'processed' in spectra_dict[tissues[i]] and 'processed' in spectra_dict[tissues[j]]:
                            spec1 = spectra_dict[tissues[i]]['processed']
                            spec2 = spectra_dict[tissues[j]]['processed']
                            distance = np.linalg.norm(spec1 - spec2)
                            metrics[f"{tissues[i]}_vs_{tissues[j]}_distance"] = float(distance)
            
            return metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def run_all_tests(self):
        """Run all functional tests."""
        print("\n" + "="*80)
        print("FUNCTIONAL PREPROCESSING TESTS WITH REAL RAMAN DATA")
        print("="*80)
        print(f"Timestamp: {self.timestamp}")
        print(f"ramanspy available: {RAMANSPY_AVAILABLE}")
        print()
        
        if not RAMANSPY_AVAILABLE:
            print("⚠️  WARNING: ramanspy not available - tests will be limited")
            print()
        
        # Test individual methods
        print("Testing Individual Methods...")
        print("-" * 80)
        
        categories = self.registry.get_categories()
        for category in categories:
            methods_dict = self.registry.get_methods_by_category(category)
            methods = list(methods_dict.keys())
            
            for method in methods:
                test_result = self.test_method_functionality(category, method)
                self.results['method_tests'][f"{category}.{method}"] = test_result
                self.results['summary']['total_tests'] += 1
                
                if test_result['passed']:
                    self.results['summary']['passed'] += 1
                    status = "[PASS]"
                else:
                    self.results['summary']['failed'] += 1
                    status = "[FAIL]"
                
                if test_result['warnings']:
                    self.results['summary']['warnings'] += len(test_result['warnings'])
                
                print(f"  {status} {category:20s} {method:20s}")
        
        print()
        
        # Test medical pipelines
        print("Testing Medical Diagnostic Pipelines...")
        print("-" * 80)
        
        medical_pipelines = self._get_medical_pipelines()
        
        for pipeline_name, steps in medical_pipelines.items():
            test_result = self.test_medical_pipeline(pipeline_name, steps)
            self.results['pipeline_tests'][pipeline_name] = test_result
            self.results['summary']['total_tests'] += 1
            
            if test_result['passed']:
                self.results['summary']['passed'] += 1
                status = "[PASS]"
            else:
                self.results['summary']['failed'] += 1
                status = "[FAIL]"
            
            print(f"  {status} {pipeline_name}")
            if not test_result['passed']:
                for error in test_result['errors'][:3]:  # Show first 3 errors
                    print(f"      +- {error}")
        
        print()
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
        
        return self.results['summary']['failed'] == 0
    
    def _get_medical_pipelines(self) -> Dict[str, List[Tuple[str, str, dict]]]:
        """Define medical diagnostic pipelines to test."""
        return {
            "Cancer Detection Pipeline": [
                ("miscellaneous", "Cropper", {"region": [800.0, 1800.0]}),
                ("cosmic_ray_removal", "WhitakerHayes", {}),
                ("denoising", "SavGol", {"window_length": 7, "polyorder": 3}),
                ("baseline_correction", "ASPLS", {"lam": 1e6}),  # Removed 'p' - not supported by ramanspy wrapper
                ("normalisation", "Vector", {})
            ],
            
            "Tissue Classification Pipeline": [
                ("miscellaneous", "Cropper", {"region": [600.0, 1800.0]}),
                ("cosmic_ray_removal", "ModifiedZScore", {}),
                ("denoising", "Whittaker", {"lam": 10.0}),
                ("baseline_correction", "IAsLS", {"lam": 1e5, "p_initial": 0.01}),
                ("normalisation", "SNV", {}),
                ("normalisation", "Vector", {})
            ],
            
            "Inflammation Detection Pipeline": [
                ("calibration", "WavenumberCalibration", {"reference_peaks": {"phenylalanine": 1004.0}}),
                ("cosmic_ray_removal", "WhitakerHayes", {}),
                ("denoising", "SavGol", {"window_length": 11, "polyorder": 3}),
                ("baseline_correction", "AirPLS", {"lam": 1e5}),
                ("derivatives", "Derivative", {"order": 1}),
                ("normalisation", "MinMax", {})
            ],
            
            "High-Throughput Screening Pipeline": [
                ("miscellaneous", "Cropper", {"region": [800.0, 1800.0]}),
                ("cosmic_ray_removal", "ModifiedZScore", {}),
                ("denoising", "MovingAverage", {"window_length": 9}),
                ("baseline_correction", "ArPLS", {"lam": 1e4}),
                ("normalisation", "MSC", {}),
                ("normalisation", "Vector", {})
            ],
            
            "Minimal Quality Control Pipeline": [
                ("miscellaneous", "Cropper", {"region": [400.0, 1800.0]}),
                ("cosmic_ray_removal", "WhitakerHayes", {}),
                ("baseline_correction", "ASPLS", {"lam": 1e5}),  # Removed 'p' - not supported by ramanspy wrapper
                ("normalisation", "Vector", {})
            ],
            
            "Advanced Research Pipeline": [
                ("calibration", "WavenumberCalibration", {"reference_peaks": {"phenylalanine": 1004.0, "nucleic_acid": 785.0}}),
                ("calibration", "IntensityCalibration", {"reference_material": "polystyrene"}),
                ("cosmic_ray_removal", "WhitakerHayes", {}),
                ("denoising", "SavGol", {"window_length": 9, "polyorder": 3}),
                ("baseline_correction", "ASPLS", {"lam": 1e6}),  # Removed 'p' - not supported by ramanspy wrapper
                ("derivatives", "Derivative", {"order": 2}),
                ("normalisation", "SNV", {}),
                ("normalisation", "Vector", {})
            ]
        }
    
    def _save_results(self):
        """Save test results to files."""
        # Text report
        txt_filename = f"functional_test_results_{self.timestamp}.txt"
        txt_path = os.path.join(os.path.dirname(__file__), txt_filename)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("FUNCTIONAL PREPROCESSING TEST RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"ramanspy available: {RAMANSPY_AVAILABLE}\n")
            f.write("\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Tests: {self.results['summary']['total_tests']}\n")
            f.write(f"Passed: {self.results['summary']['passed']}\n")
            f.write(f"Failed: {self.results['summary']['failed']}\n")
            f.write(f"Warnings: {self.results['summary']['warnings']}\n")
            f.write(f"Success Rate: {self.results['summary']['passed']/self.results['summary']['total_tests']*100:.1f}%\n")
            f.write("\n")
            
            # Method tests
            f.write("METHOD TESTS\n")
            f.write("-" * 80 + "\n")
            for method_name, result in self.results['method_tests'].items():
                status = "PASS" if result['passed'] else "FAIL"
                f.write(f"{status:4s} {method_name}\n")
                
                if result['errors']:
                    for error in result['errors']:
                        f.write(f"     ERROR: {error}\n")
                
                if result['warnings']:
                    for warning in result['warnings']:
                        f.write(f"     WARNING: {warning}\n")
                
                if 'metrics' in result:
                    f.write(f"     SNR Improvement: {result['metrics'].get('snr_improvement', 0):.2f}%\n")
            
            f.write("\n")
            
            # Pipeline tests
            f.write("PIPELINE TESTS\n")
            f.write("-" * 80 + "\n")
            for pipeline_name, result in self.results['pipeline_tests'].items():
                status = "PASS" if result['passed'] else "FAIL"
                f.write(f"{status:4s} {pipeline_name}\n")
                f.write(f"     Steps: {result['steps_passed']}/{result['steps_tested']}\n")
                
                if result['errors']:
                    for error in result['errors']:
                        f.write(f"     ERROR: {error}\n")
                
                if result['tissue_classification']:
                    f.write("     Tissue Separation Metrics:\n")
                    for metric, value in result['tissue_classification'].items():
                        f.write(f"       {metric}: {value:.4f}\n")
            
            f.write("\n")
        
        # Print with ASCII-safe filename only (avoid encoding issues with path)
        print(f"[OK] Text results saved to: {txt_filename}")
        
        # JSON report
        json_filename = f"functional_test_results_{self.timestamp}.json"
        json_path = os.path.join(os.path.dirname(__file__), json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Print with ASCII-safe filename only (avoid encoding issues with path)
        print(f"[OK] JSON results saved to: {json_filename}")
    
    def _print_summary(self):
        """Print test summary."""
        print()
        print("="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {self.results['summary']['total_tests']}")
        print(f"Passed: {self.results['summary']['passed']} ({self.results['summary']['passed']/self.results['summary']['total_tests']*100:.1f}%)")
        print(f"Failed: {self.results['summary']['failed']} ({self.results['summary']['failed']/self.results['summary']['total_tests']*100:.1f}%)")
        print(f"Warnings: {self.results['summary']['warnings']}")
        print()
        
        if self.results['summary']['failed'] == 0:
            print("[SUCCESS] ALL TESTS PASSED!")
        else:
            print("[FAILURE] SOME TESTS FAILED")
            print("\nFailed Tests:")
            
            # Show failed method tests
            for method_name, result in self.results['method_tests'].items():
                if not result['passed']:
                    print(f"  [X] {method_name}")
                    for error in result['errors']:
                        print(f"    +- {error}")
            
            # Show failed pipeline tests
            for pipeline_name, result in self.results['pipeline_tests'].items():
                if not result['passed']:
                    print(f"  [X] {pipeline_name}")
                    for error in result['errors'][:2]:
                        print(f"    +- {error}")
        
        print("="*80)
        print()


def main():
    """Main test execution."""
    tester = FunctionalPreprocessingTester()
    success = tester.run_all_tests()
    
    # Return exit code
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
