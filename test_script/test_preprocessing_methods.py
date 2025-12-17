"""
Comprehensive Preprocessing Methods Test Script
Tests all preprocessing methods for warnings, errors, and parameter issues.
Run this script to validate preprocessing system integrity.
"""

import sys
import warnings
import logging
from typing import Dict, List, Tuple
import numpy as np

# Configure logging to capture all warnings
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(name)s: %(message)s'
)

# Capture warnings
warnings.simplefilter('always')

# Import preprocessing registry
from functions.preprocess.registry import PreprocessingStepRegistry

class PreprocessingTester:
    """Test all preprocessing methods comprehensively."""
    
    def __init__(self):
        self.registry = PreprocessingStepRegistry()
        self.results = {
            'passed': [],
            'warnings': [],
            'errors': []
        }
        
        # Create synthetic test data
        self.test_data = self._create_test_data()
    
    def _create_test_data(self):
        """Create synthetic Raman spectroscopy data for testing."""
        try:
            import ramanspy as rp
            # Create simple test spectrum
            wavenumbers = np.linspace(400, 4000, 1000)
            intensities = np.random.randn(1000) * 100 + 1000
            # Add some peaks
            intensities[400:450] += 5000
            intensities[700:750] += 3000
            return rp.Spectrum(intensities, wavenumbers)
        except Exception as e:
            print(f"[ERROR] Failed to create test data: {e}")
            return None
    
    def test_all_methods(self):
        """Test all preprocessing methods in registry."""
        print("="*80)
        print("PREPROCESSING METHODS COMPREHENSIVE TEST")
        print("="*80)
        print()
        
        all_methods = self.registry.get_all_methods()
        
        for category, methods in all_methods.items():
            print(f"\n{'='*80}")
            print(f"CATEGORY: {category.upper()}")
            print(f"{'='*80}")
            
            for method_name in methods.keys():
                self._test_method(category, method_name)
        
        self._print_summary()
    
    def _test_method(self, category: str, method_name: str):
        """Test a single preprocessing method."""
        print(f"\nTesting: {category} -> {method_name}")
        print("-" * 60)
        
        try:
            # Get method info
            info = self.registry.get_method_info(category, method_name)
            if not info:
                self.results['errors'].append(f"{category}/{method_name}: No method info found")
                print("[ERROR] No method info found")
                return
            
            # Check param_info exists
            param_info = info.get('param_info', {})
            default_params = info.get('default_params', {})
            
            print(f"  Default params: {default_params}")
            print(f"  Param info keys: {list(param_info.keys())}")
            
            # Check for parameter mismatches
            default_keys = set(default_params.keys())
            param_info_keys = set(param_info.keys())
            
            missing_in_info = default_keys - param_info_keys
            missing_in_defaults = param_info_keys - default_keys
            
            if missing_in_info:
                warning_msg = f"{category}/{method_name}: Default params not in param_info: {missing_in_info}"
                self.results['warnings'].append(warning_msg)
                print(f"  [WARNING] Default params missing in param_info: {missing_in_info}")
            
            if missing_in_defaults:
                warning_msg = f"{category}/{method_name}: Param_info not in default_params: {missing_in_defaults}"
                self.results['warnings'].append(warning_msg)
                print(f"  [WARNING] Param_info missing in default_params: {missing_in_defaults}")
            
            # Try to instantiate the method
            try:
                method_instance = self.registry.create_method_instance(
                    category, 
                    method_name, 
                    default_params
                )
                print(f"  [SUCCESS] Method instantiated successfully")
                self.results['passed'].append(f"{category}/{method_name}")
                
                # Try to apply to test data if available
                if self.test_data is not None and category not in ['miscellaneous', 'calibration']:
                    try:
                        result = method_instance.apply(self.test_data)
                        print(f"  [SUCCESS] Method applied to test data")
                    except Exception as apply_error:
                        print(f"  [WARNING] Method application failed: {apply_error}")
                        self.results['warnings'].append(f"{category}/{method_name}: Application failed - {apply_error}")
                        
            except Exception as inst_error:
                error_msg = f"{category}/{method_name}: Instantiation failed - {inst_error}"
                self.results['errors'].append(error_msg)
                print(f"  [ERROR] Instantiation failed: {inst_error}")
            
            # Validate parameter ranges
            for param_name, param_spec in param_info.items():
                param_type = param_spec.get('type', 'unknown')
                param_range = param_spec.get('range', None)
                
                if param_range:
                    if len(param_range) != 2:
                        warning_msg = f"{category}/{method_name}/{param_name}: Invalid range length {len(param_range)}"
                        self.results['warnings'].append(warning_msg)
                        print(f"  [WARNING] Invalid range for {param_name}: {param_range}")
                    elif param_range[0] >= param_range[1]:
                        warning_msg = f"{category}/{method_name}/{param_name}: Invalid range [min >= max]: {param_range}"
                        self.results['warnings'].append(warning_msg)
                        print(f"  [WARNING] Invalid range for {param_name}: {param_range}")
                
        except Exception as e:
            error_msg = f"{category}/{method_name}: Test failed - {e}"
            self.results['errors'].append(error_msg)
            print(f"[ERROR] Test failed: {e}")
    
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        print(f"\n[PASSED] {len(self.results['passed'])} methods")
        for method in self.results['passed']:
            print(f"  - {method}")
        
        print(f"\n[WARNINGS] {len(self.results['warnings'])} issues")
        for warning in self.results['warnings']:
            print(f"  - {warning}")
        
        print(f"\n[ERRORS] {len(self.results['errors'])} failures")
        for error in self.results['errors']:
            print(f"  - {error}")
        
        print("\n" + "="*80)
        total_tests = len(self.results['passed']) + len(self.results['errors'])
        if total_tests > 0:
            success_rate = (len(self.results['passed']) / total_tests) * 100
            print(f"Success Rate: {success_rate:.1f}% ({len(self.results['passed'])}/{total_tests})")
        print("="*80)


def main():
    """Main test execution."""
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE PREPROCESSING TEST")
    print("="*80)
    
    tester = PreprocessingTester()
    tester.test_all_methods()
    
    print("\n[INFO] Test completed. Check output above for details.")
    
    # Return exit code based on errors
    if tester.results['errors']:
        print(f"\n[FAIL] {len(tester.results['errors'])} errors found")
        return 1
    else:
        print("\n[PASS] All tests passed (warnings may exist)")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
