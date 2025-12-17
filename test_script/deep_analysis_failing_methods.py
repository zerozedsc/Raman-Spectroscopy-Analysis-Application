"""
DEEP ANALYSIS OF FAILING PREPROCESSING METHODS
==============================================
Performs comprehensive signature verification and root cause analysis
for all methods that fail functional testing.

Author: MUHAMMAD HELMI BIN ROZAIN
Date: 2025-10-14
Environment: UV Python environment
Output: Detailed analysis report with fix recommendations
"""

import sys
import os
import inspect
import warnings
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import ramanspy as rp
    import numpy as np
    RAMANSPY_AVAILABLE = True
except ImportError:
    RAMANSPY_AVAILABLE = False
    print("ERROR: ramanspy not available")
    sys.exit(1)

from functions.preprocess.registry import PreprocessingStepRegistry


class MethodSignatureAnalyzer:
    """Analyze method signatures and compare with registry definitions."""
    
    def __init__(self):
        self.registry = PreprocessingStepRegistry()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.analysis_results = {}
        
    def analyze_method(self, category: str, method: str) -> Dict[str, Any]:
        """
        Deep analysis of a single method.
        
        Returns dict with:
        - actual_signature: Parameters from inspect.signature()
        - registry_params: Parameters defined in registry
        - mismatches: List of parameter mismatches
        - missing_in_registry: Parameters in class but not in registry
        - extra_in_registry: Parameters in registry but not in class
        - instantiation_test: Can we create an instance?
        - call_test: Can we call it with test data?
        """
        result = {
            'category': category,
            'method': method,
            'status': 'unknown',
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Get registry info
            method_info = self.registry.get_method_info(category, method)
            if not method_info:
                result['status'] = 'not_in_registry'
                result['issues'].append('Method not found in registry')
                return result
            
            method_class = method_info.get('class')
            registry_params = method_info.get('param_info', {})
            default_params = method_info.get('default_params', {})
            param_aliases = method_info.get('param_aliases', {})
            
            # Get actual signature
            try:
                sig = inspect.signature(method_class.__init__)
                actual_params = {
                    name: param 
                    for name, param in sig.parameters.items() 
                    if name != 'self'
                }
                result['actual_signature'] = list(actual_params.keys())
                result['registry_params'] = list(registry_params.keys())
                
                # Check for mismatches
                actual_param_names = set(actual_params.keys())
                registry_param_names = set(registry_params.keys()) - set(param_aliases.keys())  # Exclude aliases
                
                missing_in_registry = actual_param_names - registry_param_names
                extra_in_registry = registry_param_names - actual_param_names
                
                if missing_in_registry:
                    result['issues'].append(f"Parameters in class but not in registry: {missing_in_registry}")
                    result['recommendations'].append(f"Add to registry param_info: {missing_in_registry}")
                
                if extra_in_registry:
                    result['issues'].append(f"Parameters in registry but not in class: {extra_in_registry}")
                    result['recommendations'].append(f"Remove from registry or fix class: {extra_in_registry}")
                
                # Test instantiation with default params
                try:
                    # Filter default params to only include valid ones
                    valid_defaults = {
                        k: v for k, v in default_params.items()
                        if k in actual_param_names
                    }
                    instance = method_class(**valid_defaults)
                    result['instantiation'] = 'SUCCESS'
                except Exception as e:
                    result['instantiation'] = f'FAILED: {str(e)}'
                    result['issues'].append(f'Cannot instantiate with defaults: {str(e)}')
                    result['recommendations'].append('Fix default_params in registry')
                
                # Test with synthetic data
                try:
                    test_spectrum = np.random.rand(100)
                    test_axis = np.linspace(400, 1800, 100)
                    container = rp.SpectralContainer(test_spectrum.reshape(1, -1), test_axis)
                    
                    # Try to apply
                    if hasattr(instance, '__call__'):
                        try:
                            processed = instance(container)
                            result['call_test'] = 'SUCCESS'
                        except TypeError as e:
                            if 'missing' in str(e) and 'required positional argument' in str(e):
                                result['call_test'] = f'REQUIRES_RUNTIME_INPUT: {str(e)}'
                                result['issues'].append(f'Method requires runtime input: {str(e)}')
                                result['recommendations'].append('Make parameter optional with default=None or redesign API')
                            else:
                                result['call_test'] = f'TYPE_ERROR: {str(e)}'
                                result['issues'].append(f'Call failed: {str(e)}')
                        except Exception as e:
                            result['call_test'] = f'ERROR: {str(e)}'
                            result['issues'].append(f'Call failed: {str(e)}')
                    else:
                        result['call_test'] = 'NO_CALL_METHOD'
                        result['issues'].append('Instance does not have __call__ method')
                    
                except Exception as e:
                    result['call_test'] = f'ERROR: {str(e)}'
                    result['issues'].append(f'Test execution failed: {str(e)}')
                
                # Overall status
                if not result['issues']:
                    result['status'] = 'HEALTHY'
                elif result.get('instantiation') == 'SUCCESS' and result.get('call_test') == 'SUCCESS':
                    result['status'] = 'WORKING_WITH_WARNINGS'
                else:
                    result['status'] = 'BROKEN'
                    
            except Exception as e:
                result['status'] = 'SIGNATURE_ERROR'
                result['issues'].append(f'Cannot inspect signature: {str(e)}')
                
        except Exception as e:
            result['status'] = 'ANALYSIS_ERROR'
            result['issues'].append(f'Analysis failed: {str(e)}')
        
        return result
    
    def analyze_failing_methods(self):
        """Analyze all methods that are known to fail functional tests."""
        failing_methods = [
            # From test results - individual method failures
            ('miscellaneous', 'Cropper'),
            ('miscellaneous', 'BackgroundSubtractor'),
            ('miscellaneous', 'PeakRatioFeatures'),
            ('calibration', 'WavenumberCalibration'),
            ('calibration', 'IntensityCalibration'),
            ('denoising', 'Kernel'),
            ('cosmic_ray_removal', 'WhitakerHayes'),
            ('cosmic_ray_removal', 'Gaussian'),
            ('cosmic_ray_removal', 'MedianDespike'),
            ('baseline_correction', 'FABC'),
            ('normalisation', 'MaxIntensity'),
            ('normalisation', 'AUC'),
            ('normalisation', 'MSC'),
            ('normalisation', 'QuantileNormalization'),
            ('normalisation', 'RankTransform'),
            ('normalisation', 'ProbabilisticQuotientNormalization'),
        ]
        
        print("="*80)
        print("DEEP ANALYSIS OF FAILING PREPROCESSING METHODS")
        print("="*80)
        print(f"Timestamp: {self.timestamp}")
        print(f"Total methods to analyze: {len(failing_methods)}")
        print()
        
        results_by_category = {}
        
        for category, method in failing_methods:
            print(f"\nAnalyzing {category}.{method}...")
            result = self.analyze_method(category, method)
            self.analysis_results[f"{category}.{method}"] = result
            
            if category not in results_by_category:
                results_by_category[category] = []
            results_by_category[category].append(result)
            
            # Print immediate findings
            print(f"  Status: {result['status']}")
            if result.get('actual_signature'):
                print(f"  Actual params: {result['actual_signature']}")
            if result.get('registry_params'):
                print(f"  Registry params: {result['registry_params']}")
            if result.get('instantiation'):
                print(f"  Instantiation: {result['instantiation']}")
            if result.get('call_test'):
                print(f"  Call test: {result['call_test']}")
            if result['issues']:
                print(f"  Issues ({len(result['issues'])}):")
                for issue in result['issues']:
                    print(f"    - {issue}")
            if result['recommendations']:
                print(f"  Recommendations:")
                for rec in result['recommendations']:
                    print(f"    => {rec}")
        
        # Generate summary report
        self._generate_report(results_by_category)
        
    def _generate_report(self, results_by_category: Dict[str, List[Dict]]):
        """Generate comprehensive analysis report."""
        report_filename = f"deep_analysis_report_{self.timestamp}.md"
        report_path = os.path.join(os.path.dirname(__file__), 'results', report_filename)
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Deep Analysis Report: Failing Preprocessing Methods\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Executive summary
            f.write("## Executive Summary\n\n")
            total_analyzed = sum(len(methods) for methods in results_by_category.values())
            status_counts = {}
            for methods in results_by_category.values():
                for method in methods:
                    status = method['status']
                    status_counts[status] = status_counts.get(status, 0) + 1
            
            f.write(f"- **Total Methods Analyzed**: {total_analyzed}\n")
            for status, count in sorted(status_counts.items()):
                f.write(f"- **{status}**: {count}\n")
            f.write("\n")
            
            # Detailed analysis by category
            f.write("## Detailed Analysis by Category\n\n")
            
            for category, methods in sorted(results_by_category.items()):
                f.write(f"### {category.upper()}\n\n")
                
                for method_result in methods:
                    method_name = method_result['method']
                    f.write(f"#### {method_name}\n\n")
                    f.write(f"**Status**: `{method_result['status']}`\n\n")
                    
                    if method_result.get('actual_signature'):
                        f.write(f"**Actual Signature**: `{', '.join(method_result['actual_signature'])}`\n\n")
                    
                    if method_result.get('registry_params'):
                        f.write(f"**Registry Parameters**: `{', '.join(method_result['registry_params'])}`\n\n")
                    
                    if method_result.get('instantiation'):
                        f.write(f"**Instantiation Test**: {method_result['instantiation']}\n\n")
                    
                    if method_result.get('call_test'):
                        f.write(f"**Call Test**: {method_result['call_test']}\n\n")
                    
                    if method_result['issues']:
                        f.write("**Issues Found**:\n")
                        for issue in method_result['issues']:
                            f.write(f"- {issue}\n")
                        f.write("\n")
                    
                    if method_result['recommendations']:
                        f.write("**Recommendations**:\n")
                        for rec in method_result['recommendations']:
                            f.write(f"1. {rec}\n")
                        f.write("\n")
                    
                    f.write("---\n\n")
            
            # Root cause analysis
            f.write("## Root Cause Analysis\n\n")
            f.write("### Common Issues\n\n")
            
            # Categorize issues
            requires_runtime_input = []
            parameter_mismatch = []
            call_failures = []
            
            for method_name, result in self.analysis_results.items():
                if any('REQUIRES_RUNTIME_INPUT' in str(result.get('call_test', '')) for _ in [1]):
                    requires_runtime_input.append(method_name)
                if any('not in registry' in issue or 'not in class' in issue for issue in result['issues']):
                    parameter_mismatch.append(method_name)
                if result.get('call_test', '').startswith('ERROR') or result.get('call_test', '').startswith('TYPE_ERROR'):
                    call_failures.append(method_name)
            
            if requires_runtime_input:
                f.write(f"#### Methods Requiring Runtime Input ({len(requires_runtime_input)})\n\n")
                f.write("These methods need additional data at call time that cannot be provided in pipeline definition:\n\n")
                for method in requires_runtime_input:
                    f.write(f"- {method}\n")
                f.write("\n**Fix**: Redesign to accept optional parameters with sensible defaults or None values.\n\n")
            
            if parameter_mismatch:
                f.write(f"#### Parameter Mismatch ({len(parameter_mismatch)})\n\n")
                f.write("Registry definitions do not match actual class signatures:\n\n")
                for method in parameter_mismatch:
                    f.write(f"- {method}\n")
                f.write("\n**Fix**: Use `inspect.signature()` to verify and update registry definitions.\n\n")
            
            if call_failures:
                f.write(f"#### Call Failures ({len(call_failures)})\n\n")
                f.write("Methods fail when called with test data:\n\n")
                for method in call_failures:
                    f.write(f"- {method}\n")
                f.write("\n**Fix**: Debug individual method implementations.\n\n")
        
        print(f"\n{'='*80}")
        print(f"[OK] Deep analysis report saved to: {report_filename}")
        print(f"[OK] Location: test_script/results/")
        print(f"{'='*80}")


def main():
    """Run deep analysis."""
    analyzer = MethodSignatureAnalyzer()
    analyzer.analyze_failing_methods()
    return 0


if __name__ == "__main__":
    exit(main())
