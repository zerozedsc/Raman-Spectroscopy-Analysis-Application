"""
Comprehensive Parameter Type Validation for All Preprocessing Methods
======================================================================
This script checks EVERY preprocessing method for parameter type issues.

Date: October 15, 2025
"""

import sys
import os
from pathlib import Path
import inspect
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from functions.preprocess.registry import PreprocessingStepRegistry

print("="*100)
print(" COMPREHENSIVE PARAMETER TYPE VALIDATION ".center(100, "="))
print("="*100)
print()

registry = PreprocessingStepRegistry()
all_methods = registry.get_all_methods()

total_issues = 0
methods_checked = 0

for category, methods in all_methods.items():
    print(f"\n{'='*100}")
    print(f" CATEGORY: {category.upper()} ".center(100, "="))
    print(f"{'='*100}\n")
    
    for method_name, method_info in methods.items():
        methods_checked += 1
        print(f"[{methods_checked}] Checking: {category}/{method_name}")
        print("-" * 80)
        
        # Get method class
        method_class = method_info.get('class')
        if method_class is None:
            print("   ❌ ERROR: No class found")
            total_issues += 1
            continue
        
        # Get signatures
        try:
            sig = inspect.signature(method_class.__init__)
            params = list(sig.parameters.keys())[1:]  # Skip 'self'
            print(f"   Method parameters: {params}")
        except Exception as e:
            print(f"   ⚠️  Could not inspect signature: {e}")
            params = []
        
        # Get registry parameter info
        param_info = method_info.get('param_info', {})
        default_params = method_info.get('default_params', {})
        
        print(f"   Registry param_info: {list(param_info.keys())}")
        print(f"   Registry defaults: {default_params}")
        
        # Check for type mismatches
        issues_found = []
        
        for param_name, param_details in param_info.items():
            param_type = param_details.get('type')
            param_range = param_details.get('range')
            default_value = default_params.get(param_name)
            
            # Check integer parameters
            if param_type == 'int':
                # Check if range has float step
                if param_range and isinstance(param_range, list) and len(param_range) >= 2:
                    min_val, max_val = param_range[0], param_range[1]
                    if isinstance(min_val, float) or isinstance(max_val, float):
                        issues_found.append(f"INT param '{param_name}' has float range: {param_range}")
                
                # Check if default is float
                if default_value is not None and isinstance(default_value, float) and default_value != int(default_value):
                    issues_found.append(f"INT param '{param_name}' has float default: {default_value}")
            
            # Check for missing step in float parameters
            if param_type == 'float':
                if 'step' not in param_details:
                    issues_found.append(f"FLOAT param '{param_name}' missing 'step' specification")
        
        # Report issues
        if issues_found:
            print(f"\n   🔴 ISSUES FOUND ({len(issues_found)}):")
            for issue in issues_found:
                print(f"      - {issue}")
            total_issues += len(issues_found)
        else:
            print(f"   ✅ No type issues detected")
        
        # Test instantiation with default params
        print(f"\n   Testing instantiation...")
        try:
            instance = registry.create_method_instance(category, method_name, default_params)
            print(f"   ✅ Instantiation: SUCCESS")
        except Exception as e:
            print(f"   ❌ Instantiation FAILED: {e}")
            total_issues += 1
        
        print()

print("\n" + "="*100)
print(" SUMMARY ".center(100, "="))
print("="*100)
print(f"Total methods checked: {methods_checked}")
print(f"Total issues found: {total_issues}")

if total_issues > 0:
    print(f"\n🔴 ATTENTION REQUIRED: {total_issues} type issues need fixing!")
else:
    print(f"\n✅ ALL METHODS PASS: No type issues detected!")

print("="*100)
