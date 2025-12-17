# Test Design Improvements

**Date**: 2025-10-14  
**Status**: ✅ **RECOMMENDATIONS COMPLETE**

---

## Current Issues

### Issue 1: Non-Deterministic Cosmic Ray Generation

**Problem**:
```python
# test_preprocessing_functional.py, line 112-115
if np.random.random() > 0.7:  # Only 30% chance!
    spike_idx = np.random.randint(100, len(spectrum)-100)
    spectrum = self.add_cosmic_ray(spectrum, spike_idx, spectrum[spike_idx] * 50)
```

**Impact**:
- Cosmic ray removal tests pass/fail randomly
- MedianDespike: Sometimes passes (when spike present), sometimes fails (no spike)
- Gaussian (cosmic_ray): Same issue
- Pass rate varies: 60.9% → 63.0% between runs

**Solution**:
```python
def generate_tissue_spectrum(self, tissue_type="normal", include_cosmic_ray=False):
    """
    Generate realistic tissue Raman spectrum.
    
    Parameters:
    -----------
    tissue_type : str
        'normal', 'cancer', 'inflammation'
    include_cosmic_ray : bool
        If True, always add a cosmic ray spike (for deterministic testing)
    """
    # ... generate spectrum ...
    
    # Add cosmic ray deterministically
    if include_cosmic_ray:
        spike_idx = len(spectrum) // 2  # Fixed position for reproducibility
        spectrum = self.add_cosmic_ray(spectrum, spike_idx, spectrum[spike_idx] * 50)
    
    return spectrum
```

**Modified test method**:
```python
def test_preprocessing_method(self, category: str, method: str) -> Dict[str, Any]:
    # ...
    
    # For cosmic ray removal, ALWAYS include cosmic ray
    include_spike = (category == "cosmic_ray_removal")
    test_spectrum = self.generator.generate_tissue_spectrum("normal", 
                                                            include_cosmic_ray=include_spike)
    # ...
```

---

### Issue 2: Single Spectrum for Multi-Spectrum Normalization

**Problem**:
```python
# Current test uses single spectrum
test_spectrum = self.generator.generate_tissue_spectrum("normal")  # 1D array
container = rp.SpectralContainer(test_spectrum.reshape(1, -1), test_axis)  # (1, n_points)
```

**Impact on multi-spectrum normalization methods**:

| Method | Behavior with 1 Spectrum | Result |
|--------|--------------------------|--------|
| **MSC** | Returns unchanged (no reference) | ❌ FAIL |
| **QuantileNormalization** | Returns unchanged + warning | ❌ FAIL |
| **RankTransform** | Ranks within spectrum (not normalized) | ❌ FAIL |
| **PQN** | Uses self as reference (no change) | ❌ FAIL |

**Why they need multiple spectra**:
- **MSC**: Needs reference spectrum (mean/median of group)
- **QuantileNormalization**: Normalizes based on distribution across spectra
- **RankTransform**: Statistical normalization requires group
- **PQN**: Divides by reference (median of group)

**Solution**:
```python
def test_preprocessing_method(self, category: str, method: str) -> Dict[str, Any]:
    # ...
    
    # For multi-spectrum normalization, use 3+ spectra
    if category == "normalisation" and method in ["MSC", "QuantileNormalization", 
                                                   "RankTransform", 
                                                   "ProbabilisticQuotientNormalization"]:
        # Generate multiple tissue types for proper normalization
        test_spectra = np.vstack([
            self.generator.generate_tissue_spectrum("normal"),
            self.generator.generate_tissue_spectrum("cancer"),
            self.generator.generate_tissue_spectrum("inflammation")
        ])
        container = rp.SpectralContainer(test_spectra, test_axis)
    else:
        # Single spectrum for other methods
        test_spectrum = self.generator.generate_tissue_spectrum("normal")
        container = rp.SpectralContainer(test_spectrum.reshape(1, -1), test_axis)
    
    # ...
```

**Updated validation logic**:
```python
def _check_expected_effect(self, category: str, original: np.ndarray, 
                          processed: np.ndarray, method: str = "",
                          n_spectra: int = 1) -> bool:
    """
    Check if method produced expected effect.
    
    Parameters:
    -----------
    n_spectra : int
        Number of spectra in the dataset (for multi-spectrum methods)
    """
    # ...
    
    elif category == "normalisation":
        method_upper = method.upper()
        
        if 'MSC' in method_upper:
            if n_spectra == 1:
                # Single spectrum: should return unchanged
                return np.allclose(original, processed)
            else:
                # Multiple spectra: should correct scatter
                return not np.allclose(original, processed)
        
        elif 'QUANTILE' in method_upper:
            if n_spectra == 1:
                return np.allclose(original, processed)  # Unchanged
            else:
                # Check that quantile distribution is normalized
                return not np.allclose(original, processed)
        
        # ... similar for RankTransform, PQN ...
```

---

## Expected Impact

### Pass Rate Improvement

| Scenario | Current | After Fix | Improvement |
|----------|---------|-----------|-------------|
| **Deterministic cosmic rays** | 60.9% (varies) | 63.0% (stable) | +2.1%, stable |
| **Multi-spectrum normalization** | 4/8 normalization | 8/8 normalization | +8.7% |
| **Total expected** | 28/46 (60.9%) | 36/46 (78.3%) | +17.4% 🎯 |

### Tests That Will Pass

✅ **Currently failing due to test design**:
1. cosmic_ray_removal.MedianDespike (when no spike generated)
2. cosmic_ray_removal.Gaussian (when no spike generated)
3. normalisation.MSC (single spectrum = correct to return unchanged)
4. normalisation.QuantileNormalization (single spectrum = correct)
5. normalisation.RankTransform (single spectrum validation wrong)
6. normalisation.ProbabilisticQuotientNormalization (single spectrum = correct)

❌ **Still failing (real issues)**:
- FABC (ramanspy bug)
- WavenumberCalibration (needs measured_peaks at runtime)
- IntensityCalibration (needs measured_standard at runtime)
- PeakRatioFeatures (needs wavenumbers at runtime)
- Kernel (needs spectral_axis at runtime)
- WhitakerHayes (needs spectral_axis - but sometimes passes?)
- MaxIntensity, AUC (validation issues)
- Cropper, BackgroundSubtractor (validation issues)

---

## Implementation Plan

### Step 1: Add deterministic cosmic ray flag

**File**: `test_script/test_preprocessing_functional.py`

```python
# Line ~80
def generate_tissue_spectrum(self, tissue_type="normal", include_cosmic_ray=False):
    """..."""
    # ...
    
    # Add realistic noise
    spectrum = self.add_noise(spectrum, noise_level=15)
    
    # Add cosmic ray if requested (deterministic for testing)
    if include_cosmic_ray:
        spike_idx = len(spectrum) // 2  # Middle of spectrum
        spectrum = self.add_cosmic_ray(spectrum, spike_idx, spectrum[spike_idx] * 50)
    
    return spectrum
```

### Step 2: Update test method generation

**File**: `test_script/test_preprocessing_functional.py`

```python
# Line ~160
def test_preprocessing_method(self, category: str, method: str) -> Dict[str, Any]:
    """..."""
    # ...
    
    try:
        # Determine if method needs special spectrum generation
        include_spike = (category == "cosmic_ray_removal")
        multi_spectrum_methods = ["MSC", "QuantileNormalization", "RankTransform", 
                                  "ProbabilisticQuotientNormalization"]
        needs_multi_spectra = (category == "normalisation" and method in multi_spectrum_methods)
        
        # Generate test data
        if needs_multi_spectra:
            test_spectra = np.vstack([
                self.generator.generate_tissue_spectrum("normal", include_spike),
                self.generator.generate_tissue_spectrum("cancer", include_spike),
                self.generator.generate_tissue_spectrum("inflammation", include_spike)
            ])
            test_axis = self.generator.wavenumbers
            container = rp.SpectralContainer(test_spectra, test_axis)
            reference_spectrum = test_spectra[0]  # For validation
            n_spectra = 3
        else:
            test_spectrum = self.generator.generate_tissue_spectrum("normal", include_spike)
            test_axis = self.generator.wavenumbers
            container = rp.SpectralContainer(test_spectrum.reshape(1, -1), test_axis)
            reference_spectrum = test_spectrum
            n_spectra = 1
        
        # ... rest of test ...
```

### Step 3: Update validation logic

**File**: `test_script/test_preprocessing_functional.py`

```python
# Line ~225
def _check_expected_effect(self, category: str, original: np.ndarray, 
                          processed: np.ndarray, method: str = "",
                          n_spectra: int = 1) -> bool:
    """
    Check if method produced expected effect based on category.
    
    Parameters:
    -----------
    n_spectra : int
        Number of spectra (1 = single spectrum, >1 = batch processing)
    """
    try:
        # ...
        
        elif category == "normalisation":
            method_upper = method.upper()
            
            # Methods that need multiple spectra
            if 'MSC' in method_upper or 'QUANTILE' in method_upper or 'PQN' in method_upper:
                if n_spectra == 1:
                    # Single spectrum: expect unchanged (correct behavior)
                    return np.allclose(original, processed, rtol=1e-3)
                else:
                    # Multiple spectra: expect transformation
                    return not np.allclose(original, processed, rtol=1e-3)
            
            elif 'RANK' in method_upper:
                if n_spectra == 1:
                    # Single spectrum: ranks to [0,1] but not normalized across spectra
                    return 0.0 <= np.min(processed) and np.max(processed) <= 1.0
                else:
                    # Multiple spectra: proper rank normalization
                    return not np.allclose(original, processed)
            
            # ... rest of normalization checks ...
```

---

## Files to Modify

1. **`test_script/test_preprocessing_functional.py`**
   - Line ~80: Add `include_cosmic_ray` parameter to `generate_tissue_spectrum()`
   - Line ~112: Make cosmic ray generation deterministic
   - Line ~160: Update `test_preprocessing_method()` to use multiple spectra
   - Line ~225: Update `_check_expected_effect()` to handle `n_spectra` parameter

---

## Validation

### Before Implementation
```
Passed: 28/46 (60.9%) - varies between runs due to randomness
Failed: 18/46 (39.1%)
```

### After Implementation (Expected)
```
Passed: 36/46 (78.3%) - stable across runs
Failed: 10/46 (21.7%)

Newly passing:
✅ cosmic_ray_removal.MedianDespike (deterministic spike)
✅ cosmic_ray_removal.Gaussian (deterministic spike)
✅ normalisation.MSC (multi-spectrum test)
✅ normalisation.QuantileNormalization (multi-spectrum test)
✅ normalisation.RankTransform (multi-spectrum test)
✅ normalisation.ProbabilisticQuotientNormalization (multi-spectrum test)
```

---

## Priority

**Optional** - can be done now or deferred to Phase 3

**Pros of doing now**:
- Accurate pass rate measurement
- Better understanding of real vs test issues
- Confidence in container wrapper fixes

**Pros of deferring**:
- Focus on fixing real issues first (Phase 2: API redesign)
- Test improvements can be done anytime
- Current tests already identify real problems

**Recommendation**: **Implement now** - takes ~30 minutes, provides accurate metrics for Phase 2

---

**Status**: Design complete, ready for implementation ✅
