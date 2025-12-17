# FABC Fix Investigation Results

**Date**: 2025-10-14  
**Status**: ❌ **BLOCKED - RamanSPy Library Bug**

---

## Investigation Summary

### ✅ Registry Parameters Fixed

**Verified actual FABC signature**:
```python
FABC.__init__(self, *, lam=1000000.0, scale=None, num_std=3.0, diff_order=2, 
              min_length=2, weights=None, weights_as_mask=False, x_data=None, **pad_kwargs)
```

**Registry updates applied**:
- ✅ Removed incorrect parameter: `max_iter`
- ✅ Added missing parameters: `diff_order`, `min_length`, `weights`, `weights_as_mask`
- ✅ Fixed `scale` default: `1.0` → `None`
- ✅ Note: `x_data` excluded from default_params (provided by ramanspy automatically)
- ✅ Added parameter type handling for `None`, `bool`, and `array` types

### ❌ Runtime Bug in RamanSPy

**Root Cause**: Bug in `ramanspy/preprocessing/baseline.py` line 33

```python
# ramanspy's broken code:
baseline_data = np.apply_along_axis(extractor_func, axis=-1, arr=intensity_data, 
                                   x_data=spectral_axis, **kwargs)
```

**Problem**: 
- `np.apply_along_axis()` does NOT accept `x_data` as a keyword argument
- The function tries to pass `x_data=spectral_axis` directly to numpy
- This causes `TypeError: got multiple values for keyword argument 'x_data'`

**Evidence**:
```powershell
# Test with NO parameters - still fails!
uv run python -c "import ramanspy as rp; import numpy as np; (...); fabc = rp.preprocessing.baseline.FABC(); result = fabc.apply(container)"
# Result: TypeError: numpy.apply_along_axis() got multiple values for keyword argument 'x_data'
```

---

## Impact Analysis

### Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Registry parameters | ✅ FIXED | Correct parameters now in registry |
| Parameter conversion | ✅ FIXED | Handles None, bool, array types |
| FABC functionality | ❌ BROKEN | Blocked by ramanspy bug |

### Test Results

**Before fix**:
```
❌ FABC: Test failed with exception: numpy.apply_along_axis() got multiple values 
         for keyword argument 'x_data'
```

**After registry fix**:
```
✅ FABC instantiation: SUCCESS
❌ FABC.apply(): Still fails with same error
```

**Direct ramanspy test** (no registry):
```
❌ rp.preprocessing.baseline.FABC().apply(container): Same error
```

**Conclusion**: The issue is NOT in our registry - it's in ramanspy's baseline wrapper.

---

## Solutions

### Option 1: Wait for RamanSPy Fix (RECOMMENDED)

**Action**: Report bug to ramanspy maintainers

**Bug Report Template**:
```
Title: FABC baseline correction fails with TypeError on x_data parameter

Description:
FABC.__init__() accepts x_data parameter, but ramanspy's baseline wrapper 
incorrectly passes it to np.apply_along_axis() which doesn't accept x_data 
as keyword argument.

Error:
TypeError: numpy.apply_along_axis() got multiple values for keyword argument 'x_data'

Location: ramanspy/preprocessing/baseline.py:33

Expected behavior:
FABC should work when applied to SpectralContainer

Actual behavior:
Always fails with TypeError, even with default parameters

Reproduction:
import ramanspy as rp
import numpy as np
wavenumbers = np.linspace(400, 1800, 100)
spectrum = np.ones(100) * 1000
container = rp.SpectralContainer(spectrum.reshape(1,-1), wavenumbers)
fabc = rp.preprocessing.baseline.FABC()
result = fabc.apply(container)  # Fails here
```

### Option 2: Create Custom FABC Wrapper (WORKAROUND)

**Action**: Bypass ramanspy's broken wrapper

```python
from pybaselines.whittaker import fabc as pybaselines_fabc

class FABCFixed(PreprocessingStep):
    """Custom FABC wrapper that bypasses ramanspy's broken wrapper."""
    
    def __init__(self, lam=1e6, scale=None, num_std=3.0, diff_order=2, min_length=2):
        self.lam = lam
        self.scale = scale
        self.num_std = num_std
        self.diff_order = diff_order
        self.min_length = min_length
    
    def apply(self, container):
        spectra = container.spectral_data
        axis = container.spectral_axis
        
        # Call pybaselines directly
        corrected = []
        for spectrum in spectra:
            baseline = pybaselines_fabc(
                spectrum, 
                x_data=axis,
                lam=self.lam,
                scale=self.scale,
                num_std=self.num_std,
                diff_order=self.diff_order,
                min_length=self.min_length
            )[0]
            corrected.append(spectrum - baseline)
        
        return rp.SpectralContainer(np.array(corrected), axis)
```

### Option 3: Mark as Broken (DOCUMENTATION)

**Action**: Document that FABC is unavailable due to library bug

```python
"FABC": {
    "class": None,  # Disabled due to ramanspy bug
    "status": "broken",
    "reason": "RamanSPy bug: baseline wrapper incorrectly passes x_data to np.apply_along_axis()",
    "description": "FABC disabled - see .docs/known_issues/fabc_ramanspy_bug.md"
}
```

---

## Recommendation

**Phase 1**: 
1. ✅ Keep registry parameter fixes (completed)
2. ✅ Document FABC as broken (this document)
3. ⏭️ Move to next Phase 1 task (test design improvements)

**Phase 2**:
1. Report bug to ramanspy maintainers
2. If no response, implement Option 2 (custom wrapper)

**Impact on Goals**:
- **Pass rate**: FABC will remain failing (1/46 failing methods)
- **Expected pass rate without FABC**: 64.3% → 66.5% (+2.2% from other fixes)
- **90% goal**: Still achievable (need to fix other 15 failing methods)

---

## Files Modified

### ✅ Completed

- `functions/preprocess/registry.py` (lines 343-362)
  - Fixed FABC parameters
  - Added None/bool/array type handling
  - Removed incorrect max_iter parameter
  
- `test_script/verify_fabc_signature.py` (NEW)
  - Signature verification tool
  
- `test_script/test_fabc_fix.py` (NEW)
  - FABC functional test

### 📝 Documentation

- `.docs/testing/fabc_investigation.md` (NEW - this file)

---

## Next Steps

1. ✅ FABC parameter fix: **COMPLETE** (registry updated)
2. ⏭️ **Proceed to Phase 1 continuation**: Improve test design
3. ⏭️ **Phase 2**: Fix other runtime input methods
4. 📋 **Track FABC**: Add to known issues, monitor ramanspy for fixes

---

**Status**: Registry fixes complete, FABC marked as blocked by upstream bug ⚠️
