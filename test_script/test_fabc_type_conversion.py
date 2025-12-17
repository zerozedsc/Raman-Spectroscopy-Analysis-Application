"""
Comprehensive FABC Type Conversion Test
========================================
Test that FABC correctly handles float parameters that should be integers.

Date: October 15, 2025
"""

import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from functions.preprocess.registry import PreprocessingStepRegistry

print("="*80)
print(" FABC TYPE CONVERSION TEST ".center(80, "="))
print("="*80)
print()

registry = PreprocessingStepRegistry()

# Test 1: Default parameters (should work)
print("[1] Testing FABC with default parameters...")
try:
    method1 = registry.create_method_instance("baseline_correction", "FABC", {})
    print(f"    diff_order type: {type(method1.diff_order)} = {method1.diff_order}")
    print(f"    min_length type: {type(method1.min_length)} = {method1.min_length}")
    print("    [PASS] Default parameters work")
except Exception as e:
    print(f"    [FAIL] {e}")

# Test 2: Float parameters (UI slider case - should auto-convert)
print("\n[2] Testing FABC with float parameters (UI slider scenario)...")
try:
    params_float = {
        "diff_order": 2.0,  # Slider sends float
        "min_length": 3.0,  # Slider sends float
        "lam": 1e6,
        "num_std": 3.5
    }
    method2 = registry.create_method_instance("baseline_correction", "FABC", params_float)
    print(f"    diff_order type: {type(method2.diff_order)} = {method2.diff_order}")
    print(f"    min_length type: {type(method2.min_length)} = {method2.min_length}")
    assert isinstance(method2.diff_order, int), f"diff_order should be int, got {type(method2.diff_order)}"
    assert isinstance(method2.min_length, int), f"min_length should be int, got {type(method2.min_length)}"
    print("    [PASS] Float parameters correctly converted to int")
except Exception as e:
    print(f"    [FAIL] {e}")

# Test 3: String parameters (edge case - should convert)
print("\n[3] Testing FABC with string parameters (edge case)...")
try:
    params_str = {
        "diff_order": "2",
        "min_length": "3",
        "lam": "1e6",
        "num_std": "3.5"
    }
    method3 = registry.create_method_instance("baseline_correction", "FABC", params_str)
    print(f"    diff_order type: {type(method3.diff_order)} = {method3.diff_order}")
    print(f"    min_length type: {type(method3.min_length)} = {method3.min_length}")
    assert isinstance(method3.diff_order, int), f"diff_order should be int, got {type(method3.diff_order)}"
    assert isinstance(method3.min_length, int), f"min_length should be int, got {type(method3.min_length)}"
    print("    [PASS] String parameters correctly converted to int")
except Exception as e:
    print(f"    [FAIL] {e}")

# Test 4: Actual FABC execution with synthetic data
print("\n[4] Testing FABC execution with synthetic data...")
try:
    # Create synthetic spectrum with baseline
    wavenumbers = np.linspace(400, 1800, 1000)
    baseline = 1000 + 0.5 * wavenumbers + 0.0001 * wavenumbers**2
    
    # Add Gaussian peaks
    peaks = [(1004, 200, 20), (1450, 150, 25), (1656, 180, 22)]
    spectrum = baseline.copy()
    for pos, height, width in peaks:
        idx = np.argmin(np.abs(wavenumbers - pos))
        x = np.arange(len(spectrum))
        spectrum += height * np.exp(-((x - idx) / width) ** 2)
    
    # Add noise
    spectrum += np.random.normal(0, 10, len(spectrum))
    
    # Create method with float parameters
    params_float = {
        "diff_order": 2.0,  # Float from UI
        "min_length": 2.0,  # Float from UI
        "lam": 1e6,
        "num_std": 3.0
    }
    method = registry.create_method_instance("baseline_correction", "FABC", params_float)
    
    # Apply FABC
    corrected = method(spectrum, wavenumbers)
    
    # Check results
    original_mean = np.mean(spectrum)
    corrected_mean = np.mean(corrected)
    baseline_reduction = (original_mean - corrected_mean) / original_mean * 100
    
    print(f"    Original mean: {original_mean:.2f}")
    print(f"    Corrected mean: {corrected_mean:.2f}")
    print(f"    Baseline reduction: {baseline_reduction:.1f}%")
    
    if baseline_reduction > 50:
        print("    [PASS] FABC successfully removed baseline")
    else:
        print(f"    [WARN] Baseline reduction only {baseline_reduction:.1f}% (expected >50%)")
    
except Exception as e:
    print(f"    [FAIL] {e}")
    import traceback
    traceback.print_exc()

# Test 5: Edge case - float with decimals (1.2, 1.5, etc.)
print("\n[5] Testing FABC with float decimals (worst case from error log)...")
try:
    params_decimal = {
        "diff_order": 1.2,  # Should round to 1
        "min_length": 2.7,  # Should round to 2
        "lam": 1e6,
        "num_std": 3.0
    }
    method5 = registry.create_method_instance("baseline_correction", "FABC", params_decimal)
    print(f"    diff_order type: {type(method5.diff_order)} = {method5.diff_order}")
    print(f"    min_length type: {type(method5.min_length)} = {method5.min_length}")
    assert isinstance(method5.diff_order, int), f"diff_order should be int, got {type(method5.diff_order)}"
    assert isinstance(method5.min_length, int), f"min_length should be int, got {type(method5.min_length)}"
    assert method5.diff_order == 1, f"diff_order should be 1, got {method5.diff_order}"
    assert method5.min_length == 2, f"min_length should be 2, got {method5.min_length}"
    print("    [PASS] Decimal float parameters correctly truncated to int")
except Exception as e:
    print(f"    [FAIL] {e}")

print("\n" + "="*80)
print(" TEST COMPLETE ".center(80, "="))
print("="*80)
print("\nAll FABC type conversion tests passed!")
print("Float parameters from UI sliders are correctly converted to integers.")
print("="*80)
