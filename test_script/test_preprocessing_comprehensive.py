"""
COMPREHENSIVE PREPROCESSING METHODS TEST SCRIPT
================================================
Tests ALL preprocessing methods for parameter issues, instantiation errors,
and validates the entire preprocessing registry system.

Author: MUHAMMAD HELMI BIN ROZAIN
Date: 2025-10-14
Environment: UV Python environment
Output: Saves results to test_script/test_results_TIMESTAMP.txt
"""

import sys
import os
import warnings
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)
warnings.simplefilter('always')

# Import preprocessing registry
from functions.preprocess.registry import PreprocessingStepRegistry

class ComprehensivePreprocessingTester:
    """Comprehensive test suite for all preprocessing methods."""
    
    def __init__(self):
        """Initialize tester with registry and result tracking."""
        self.registry = PreprocessingStepRegistry()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Results tracking
        self.results = {
            'summary': {
                'total_methods': 0,
                'passed': 0,
                'failed': 0,
                'warnings': 0,
                'timestamp': self.timestamp
            },
            'methods': {},
            'categories': {},
            'issues': {
                'critical': [],
                'warnings': [],
                'info': []
            }
        }
        
    def run_comprehensive_test(self):
        """Run complete test suite."""
        print("="*100)
        print(" COMPREHENSIVE PREPROCESSING METHODS TEST SUITE ".center(100, "="))
        print("="*100)
        print(f"\nTimestamp: {self.timestamp}")
        print(f"Python Environment: {sys.executable}")
        print(f"Working Directory: {os.getcwd()}\n")
        
        # Get all methods
        all_methods = self.registry.get_all_methods()
        
        # Count total methods
        total_count = sum(len(methods) for methods in all_methods.values())
        self.results['summary']['total_methods'] = total_count
        
        print(f"Found {total_count} preprocessing methods across {len(all_methods)} categories\n")
        print("="*100)
        
        # Test each category
        for category, methods in all_methods.items():
            self._test_category(category, methods)
        
        # Generate report
        self._generate_report()
        
    def _test_category(self, category: str, methods: Dict[str, Any]):
        """Test all methods in a category."""
        print(f"\n{'='*100}")
        print(f" CATEGORY: {category.upper()} ({len(methods)} methods) ".center(100, "="))
        print(f"{'='*100}\n")
        
        category_results = {
            'total': len(methods),
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'methods': {}
        }
        
        for method_name in sorted(methods.keys()):
            result = self._test_method(category, method_name)
            category_results['methods'][method_name] = result
            
            if result['status'] == 'PASSED':
                category_results['passed'] += 1
                self.results['summary']['passed'] += 1
            elif result['status'] == 'FAILED':
                category_results['failed'] += 1
                self.results['summary']['failed'] += 1
            
            if result['warnings']:
                category_results['warnings'] += result['warnings']
                self.results['summary']['warnings'] += result['warnings']
        
        self.results['categories'][category] = category_results
        
        # Print category summary
        print(f"\n{'-'*100}")
        print(f"Category Summary: {category}")
        print(f"  Passed: {category_results['passed']}/{category_results['total']}")
        print(f"  Failed: {category_results['failed']}/{category_results['total']}")
        print(f"  Warnings: {category_results['warnings']}")
        print(f"{'-'*100}")
        
    def _test_method(self, category: str, method_name: str) -> Dict[str, Any]:
        """Test a single preprocessing method comprehensively."""
        method_id = f"{category}/{method_name}"
        print(f"\n[TESTING] {method_id}")
        print(f"  {'-'*80}")
        
        result = {
            'status': 'PASSED',
            'warnings': 0,
            'errors': [],
            'checks': {
                'method_info_exists': False,
                'has_class': False,
                'has_param_info': False,
                'has_default_params': False,
                'params_match': False,
                'ranges_valid': False,
                'instantiation': False
            },
            'details': {}
        }
        
        try:
            # CHECK 1: Method info exists
            info = self.registry.get_method_info(category, method_name)
            if not info:
                result['status'] = 'FAILED'
                result['errors'].append("No method info found in registry")
                self._log_issue('critical', method_id, "Missing method info")
                print(f"  [FAILED] No method info found")
                return result
            
            result['checks']['method_info_exists'] = True
            print(f"  [✓] Method info exists")
            
            # CHECK 2: Has class definition
            if 'class' not in info:
                result['status'] = 'FAILED'
                result['errors'].append("No class definition")
                self._log_issue('critical', method_id, "Missing class definition")
                print(f"  [✗] No class definition")
                return result
            
            result['checks']['has_class'] = True
            print(f"  [✓] Has class: {info['class'].__name__}")
            
            # CHECK 3: Parameter info structure
            param_info = info.get('param_info', {})
            default_params = info.get('default_params', {})
            
            result['checks']['has_param_info'] = 'param_info' in info
            result['checks']['has_default_params'] = 'default_params' in info
            
            result['details']['param_info_keys'] = list(param_info.keys())
            result['details']['default_params_keys'] = list(default_params.keys())
            result['details']['param_count'] = len(param_info)
            
            print(f"  [✓] Param info: {len(param_info)} parameters")
            print(f"  [✓] Default params: {len(default_params)} parameters")
            
            # CHECK 4: Parameter consistency
            default_keys = set(default_params.keys())
            param_info_keys = set(param_info.keys())
            
            missing_in_info = default_keys - param_info_keys
            missing_in_defaults = param_info_keys - default_keys
            
            if missing_in_info:
                result['warnings'] += 1
                warning_msg = f"Default params not in param_info: {missing_in_info}"
                result['errors'].append(warning_msg)
                self._log_issue('warnings', method_id, warning_msg)
                print(f"  [⚠] {warning_msg}")
            
            if missing_in_defaults:
                result['warnings'] += 1
                warning_msg = f"Param_info not in defaults: {missing_in_defaults}"
                result['errors'].append(warning_msg)
                self._log_issue('warnings', method_id, warning_msg)
                print(f"  [⚠] {warning_msg}")
            
            if not missing_in_info and not missing_in_defaults:
                result['checks']['params_match'] = True
                print(f"  [✓] Parameter definitions match")
            
            # CHECK 5: Validate parameter ranges
            invalid_ranges = []
            for param_name, param_spec in param_info.items():
                param_range = param_spec.get('range', None)
                if param_range:
                    if not isinstance(param_range, list):
                        invalid_ranges.append(f"{param_name}: not a list")
                    elif len(param_range) == 2:
                        if param_range[0] >= param_range[1]:
                            invalid_ranges.append(f"{param_name}: min >= max {param_range}")
                    elif len(param_range) != 1:  # Allow single-element lists for choices
                        invalid_ranges.append(f"{param_name}: invalid length {len(param_range)}")
            
            if invalid_ranges:
                result['warnings'] += len(invalid_ranges)
                for inv_range in invalid_ranges:
                    result['errors'].append(f"Invalid range: {inv_range}")
                    self._log_issue('warnings', method_id, f"Invalid range: {inv_range}")
                    print(f"  [⚠] Invalid range: {inv_range}")
            else:
                result['checks']['ranges_valid'] = True
                print(f"  [✓] All parameter ranges valid")
            
            # CHECK 6: Try instantiation
            try:
                method_instance = self.registry.create_method_instance(
                    category, 
                    method_name, 
                    default_params
                )
                result['checks']['instantiation'] = True
                result['details']['instance_type'] = type(method_instance).__name__
                print(f"  [✓] Instantiation successful: {type(method_instance).__name__}")
                
            except Exception as inst_error:
                result['status'] = 'FAILED'
                error_msg = f"Instantiation failed: {str(inst_error)}"
                result['errors'].append(error_msg)
                self._log_issue('critical', method_id, error_msg)
                print(f"  [✗] {error_msg}")
            
            # Summary for this method
            checks_passed = sum(1 for v in result['checks'].values() if v)
            checks_total = len(result['checks'])
            print(f"\n  Summary: {checks_passed}/{checks_total} checks passed")
            
            if result['status'] == 'PASSED' and result['warnings'] == 0:
                print(f"  [SUCCESS] All checks passed")
            elif result['status'] == 'PASSED':
                print(f"  [PASSED] With {result['warnings']} warnings")
            else:
                print(f"  [FAILED] {len(result['errors'])} errors")
            
        except Exception as e:
            result['status'] = 'FAILED'
            error_msg = f"Test exception: {str(e)}"
            result['errors'].append(error_msg)
            self._log_issue('critical', method_id, error_msg)
            print(f"  [✗] Test failed with exception: {e}")
        
        self.results['methods'][method_id] = result
        return result
    
    def _log_issue(self, severity: str, method_id: str, message: str):
        """Log an issue to the issues tracker."""
        issue = {
            'method': method_id,
            'message': message,
            'severity': severity
        }
        self.results['issues'][severity].append(issue)
    
    def _generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*100)
        print(" FINAL TEST REPORT ".center(100, "="))
        print("="*100)
        
        summary = self.results['summary']
        print(f"\nTest Summary:")
        print(f"  Total Methods Tested: {summary['total_methods']}")
        print(f"  Passed: {summary['passed']} ({summary['passed']/summary['total_methods']*100:.1f}%)")
        print(f"  Failed: {summary['failed']} ({summary['failed']/summary['total_methods']*100:.1f}%)")
        print(f"  Total Warnings: {summary['warnings']}")
        
        # Category breakdown
        print(f"\nCategory Breakdown:")
        for category, cat_results in self.results['categories'].items():
            print(f"  {category:25} {cat_results['passed']:3}/{cat_results['total']:3} passed, "
                  f"{cat_results['warnings']:3} warnings")
        
        # Critical issues
        if self.results['issues']['critical']:
            print(f"\nCritical Issues ({len(self.results['issues']['critical'])}):")
            for issue in self.results['issues']['critical'][:10]:  # Show first 10
                print(f"  • {issue['method']}: {issue['message']}")
            if len(self.results['issues']['critical']) > 10:
                print(f"  ... and {len(self.results['issues']['critical']) - 10} more")
        
        # Save detailed report
        output_dir = os.path.join(os.path.dirname(__file__), 'test_script')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f'test_results_{self.timestamp}.txt')
        json_file = os.path.join(output_dir, f'test_results_{self.timestamp}.json')
        
        # Save text report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE PREPROCESSING TEST RESULTS\n")
            f.write("="*100 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total Methods: {summary['total_methods']}\n")
            f.write(f"Passed: {summary['passed']}\n")
            f.write(f"Failed: {summary['failed']}\n")
            f.write(f"Warnings: {summary['warnings']}\n\n")
            
            f.write("\nDETAILED RESULTS BY METHOD\n")
            f.write("="*100 + "\n")
            
            for method_id, method_result in sorted(self.results['methods'].items()):
                f.write(f"\n{method_id}\n")
                f.write(f"  Status: {method_result['status']}\n")
                f.write(f"  Warnings: {method_result['warnings']}\n")
                
                # Checks
                f.write(f"  Checks:\n")
                for check_name, check_result in method_result['checks'].items():
                    status = "✓" if check_result else "✗"
                    f.write(f"    {status} {check_name}\n")
                
                # Errors
                if method_result['errors']:
                    f.write(f"  Errors:\n")
                    for error in method_result['errors']:
                        f.write(f"    - {error}\n")
                
                f.write("\n")
        
        # Save JSON report
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nDetailed reports saved:")
        print(f"  Text: {output_file}")
        print(f"  JSON: {json_file}")
        
        print("\n" + "="*100)
        
        # Return exit code
        return 0 if summary['failed'] == 0 else 1


def main():
    """Main test execution."""
    print(f"\nStarting comprehensive preprocessing test...")
    print(f"Output directory: test_script/\n")
    
    tester = ComprehensivePreprocessingTester()
    exit_code = tester.run_comprehensive_test()
    
    if exit_code == 0:
        print("\n[SUCCESS] All tests passed!")
    else:
        print(f"\n[FAILURE] {tester.results['summary']['failed']} methods failed")
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
