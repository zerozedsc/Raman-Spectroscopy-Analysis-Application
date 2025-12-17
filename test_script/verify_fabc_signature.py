"""
Verify FABC actual signature using inspect.signature()
========================================================
This script checks the actual parameters of FABC from pybaselines
to update the registry with correct parameter definitions.

Date: 2025-10-14
"""

import inspect
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import ramanspy as rp
    
    print("="*80)
    print("FABC SIGNATURE VERIFICATION")
    print("="*80)
    
    # Get signature
    sig = inspect.signature(rp.preprocessing.baseline.FABC.__init__)
    
    print("\n📋 Full Signature:")
    print(f"FABC.__init__{sig}")
    
    print("\n📝 Parameters:")
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
        
        default = param.default
        if default == inspect.Parameter.empty:
            default_str = "REQUIRED"
        else:
            default_str = repr(default)
        
        annotation = param.annotation
        if annotation == inspect.Parameter.empty:
            type_str = "Any"
        else:
            type_str = str(annotation).replace("<class '", "").replace("'>", "")
        
        print(f"  {param_name:20s} : {type_str:15s} = {default_str}")
    
    print("\n" + "="*80)
    print("REGISTRY UPDATE NEEDED")
    print("="*80)
    
    # Generate registry entry
    params = []
    for param_name, param in sig.parameters.items():
        if param_name in ('self', 'pad_kwargs'):
            continue  # Skip self and **kwargs
        
        default = param.default
        if default == inspect.Parameter.empty:
            continue  # Skip required params
        
        if isinstance(default, (int, float, bool, str, type(None))):
            params.append(f'            "{param_name}": {repr(default)},')
        else:
            params.append(f'            "{param_name}": {repr(default)},  # {type(default).__name__}')
    
    print("\n✅ Suggested Registry Entry:")
    print('        "FABC": {')
    print('            "class": FABC,')
    print('            "category": "baseline_correction",')
    print('            "default_params": {')
    print('\n'.join(params))
    print('            },')
    print('            "description": "Fully Automatic Baseline Correction"')
    print('        },')
    
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    # Test instantiation
    print("\n🧪 Testing instantiation with defaults...")
    try:
        # Import from ramanspy
        FABC = rp.preprocessing.baseline.FABC
        
        instance = FABC()
        print("  ✅ FABC() instantiation: SUCCESS")
        
        # Test with some parameters
        instance2 = FABC(lam=1e6, num_std=3.0)
        print("  ✅ FABC(lam=1e6, num_std=3.0): SUCCESS")
        
        # Test with more parameters
        instance3 = FABC(lam=1e6, diff_order=2, min_length=2)
        print("  ✅ FABC(lam=1e6, diff_order=2, min_length=2): SUCCESS")
        
    except Exception as e:
        print(f"  ❌ Instantiation failed: {e}")
    
    print("\n✅ Verification complete!")
    
except ImportError as e:
    print(f"❌ Failed to import ramanspy: {e}")
    print("   Make sure ramanspy is installed in the UV environment")
    sys.exit(1)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
