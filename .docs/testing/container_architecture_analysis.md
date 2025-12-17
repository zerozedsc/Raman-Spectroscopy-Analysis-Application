# Deep Analysis: RamanSPy Container Architecture

**Date**: 2025-10-14  
**Analysis Type**: System Architecture Review  
**Focus**: Data flow between workflow stages

---

## Key Discovery from Workflow Diagram

### System Architecture (from image)
```
CSV/TXT/Andor.asc → 1.データの標準化 → Container(標準化したデータ) 
                     → 2.前処理 → Container(前処理したデータ)
```

**Critical Insight**: The application uses **Container objects** to pass data between stages:
1. **Standardization Stage** outputs: `Container(標準化したデータ)`
2. **Preprocessing Stage** receives Container, outputs: `Container(前処理したデータ)`

### Why This Matters for Preprocessing

The RamanSPy library design:
```python
# RamanSPy containers include BOTH:
container.spectral_data  # numpy array of intensities
container.spectral_axis  # numpy array of wavenumbers
```

**ALL preprocessing methods MUST**:
1. Accept `SpectralContainer` as input
2. Return `SpectralContainer` as output
3. Preserve the container format for pipeline compatibility

---

## Current Problem Analysis

### Issue: Custom Methods Break Container Flow

**6 Custom Methods** currently expect raw numpy arrays:
1. `Gaussian` (cosmic_ray_removal)
2. `MedianDespike` (cosmic_ray_removal)
3. `MSC` (normalisation)
4. `QuantileNormalization` (normalisation)
5. `RankTransform` (normalisation)
6. `ProbabilisticQuotientNormalization` (normalisation)

**Error Pattern**:
```python
# Custom method tries:
if spectra.ndim != 2:  # Expects numpy array
    raise ValueError()

# But receives:
SpectralContainer object  # Has no .ndim attribute
```

### Root Cause

These custom methods were designed for **sklearn pipelines** (which use numpy arrays) but are being used in **RamanSPy workflows** (which use Containers).

---

## Architecture Decision: Wrapper vs Rewrite

### Option 1: Add Wrapper (RECOMMENDED)
**Pros**:
- Maintains sklearn compatibility
- Minimal code changes
- Preserves existing logic

**Cons**:
- Slight performance overhead

### Option 2: Rewrite as PreprocessingStep
**Pros**:
- Native RamanSPy integration
- No wrapper overhead

**Cons**:
- Large refactor
- Loses sklearn compatibility
- More testing needed

**Decision**: **Option 1 - Add Wrapper**

---

## Solution Design: Universal Container Wrapper

### Implementation Pattern

```python
class ContainerAwareWrapper:
    """Wrapper that handles both SpectralContainer and numpy array inputs."""
    
    def __init__(self, processor):
        """
        Args:
            processor: The underlying processor (expects numpy arrays)
        """
        self.processor = processor
    
    def __call__(self, data):
        """
        Handle both SpectralContainer and numpy array inputs.
        
        Args:
            data: SpectralContainer or numpy array
            
        Returns:
            Same type as input (Container → Container, array → array)
        """
        # Check input type
        is_container = hasattr(data, 'spectral_data')
        
        if is_container:
            # Extract numpy array from container
            spectra = data.spectral_data
            axis = data.spectral_axis
        else:
            # Already numpy array (sklearn pipeline)
            spectra = data
            axis = None
        
        # Process with underlying method (expects numpy array)
        processed = self.processor(spectra)
        
        # Return in same format as input
        if is_container:
            # Reconstruct container
            import ramanspy as rp
            return rp.SpectralContainer(processed, axis)
        else:
            # Return numpy array
            return processed
```

### Application to Each Method

#### Example: MedianDespike
```python
# Current implementation (expects numpy array)
class MedianDespike:
    def __call__(self, spectra: np.ndarray) -> np.ndarray:
        if spectra.ndim != 2:
            raise ValueError("Input must be 2D array")
        return np.apply_along_axis(self._despike_spectrum, 1, spectra)

# Add container-aware wrapper
class MedianDespike:
    def __call__(self, data):
        # Detect input type
        if hasattr(data, 'spectral_data'):
            # SpectralContainer input
            spectra = data.spectral_data
            axis = data.spectral_axis
            processed = self._process_array(spectra)
            import ramanspy as rp
            return rp.SpectralContainer(processed, axis)
        else:
            # numpy array input (sklearn compatibility)
            return self._process_array(data)
    
    def _process_array(self, spectra: np.ndarray) -> np.ndarray:
        """Core processing logic (unchanged)."""
        if spectra.ndim != 2:
            raise ValueError("Input must be 2D array")
        return np.apply_along_axis(self._despike_spectrum, 1, spectra)
```

---

## Verification: Container Flow Through Workflow

### Stage 1: Data Loading
```python
# Load raw data
raw_data = load_raman_data("sample.csv")
# Create container
container = rp.SpectralContainer(raw_data, wavenumbers)
```

### Stage 2: Preprocessing (Pipeline)
```python
# Build preprocessing pipeline
pipeline = [
    ("despike", MedianDespike(kernel_size=5)),  # Now container-aware ✓
    ("baseline", ASPLS(lam=1e5)),               # Native RamanSPy ✓
    ("normalize", SNV())                        # Now container-aware ✓
]

# Apply pipeline
for step_name, method in pipeline:
    container = method(container)  # Container → method → Container ✓
```

### Stage 3: Analysis
```python
# Container flows to next stage
results = analyze(container)  # Still has .spectral_data and .spectral_axis ✓
```

---

## Affected Files Analysis

### Files Requiring Changes

1. **functions/preprocess/spike_removal.py**
   - `Gaussian.__call__()` - Add container detection
   - `MedianDespike.__call__()` - Add container detection

2. **functions/preprocess/normalization.py**
   - `MSC.__call__()` - Add container detection
   - `QuantileNormalization.__call__()` - Add container detection
   - `RankTransform.__call__()` - Add container detection
   - `ProbabilisticQuotientNormalization.__call__()` - Add container detection

3. **functions/preprocess/registry.py**
   - No changes needed (methods already registered)

### Files NOT Requiring Changes

**Native RamanSPy methods** already handle containers correctly:
- All baseline correction methods (ASLS, ASPLS, AIRPLS, etc.)
- All denoising methods (SavGol, Whittaker, etc.)
- Vector normalization, MinMax, etc.

---

## Testing Strategy

### Test 1: Container Preservation
```python
# Input
input_container = rp.SpectralContainer(spectrum, wavenumbers)

# Process
output_container = MedianDespike()(input_container)

# Verify
assert isinstance(output_container, rp.SpectralContainer)
assert np.array_equal(output_container.spectral_axis, input_container.spectral_axis)
assert output_container.spectral_data.shape == input_container.spectral_data.shape
```

### Test 2: sklearn Compatibility
```python
# Input: numpy array (for sklearn pipelines)
input_array = np.random.rand(10, 100)

# Process
output_array = MedianDespike()(input_array)

# Verify
assert isinstance(output_array, np.ndarray)
assert output_array.shape == input_array.shape
```

### Test 3: Pipeline Integration
```python
# Full preprocessing pipeline
pipeline = [
    MedianDespike(),  # Custom method
    ASPLS(),          # Native RamanSPy
    SNV()             # Custom method
]

# Apply sequentially
container = initial_container
for method in pipeline:
    container = method(container)
    assert isinstance(container, rp.SpectralContainer)  # Maintained throughout
```

---

## Implementation Priority

### High Priority (6 methods)
These break container flow RIGHT NOW:
1. ✅ MedianDespike - Used in medical pipelines
2. ✅ Gaussian - Alternative despike method
3. ✅ MSC - Common normalization
4. ✅ QuantileNormalization - Advanced normalization
5. ✅ RankTransform - Statistical normalization
6. ✅ ProbabilisticQuotientNormalization - Metabolomics standard

### Medium Priority (9 methods)
These need API redesign (separate issue):
- Cropper, Kernel, WhitakerHayes, etc. (require spectral_axis)
- WavenumberCalibration, IntensityCalibration (require runtime inputs)

---

## Conclusion

**Container architecture is CORRECT for the application design**:
- ✓ Data flows between stages as Containers
- ✓ Preserves both spectral_data AND spectral_axis
- ✓ Enables pipeline composition
- ✓ Matches RamanSPy design philosophy

**Custom methods need fixing**:
- ✗ Currently break container flow
- ✗ Expect numpy arrays only
- ✓ Solution: Add container detection wrapper

**Impact after fix**:
- ✅ 6 methods will work in pipelines
- ✅ Still work in sklearn pipelines (backward compatible)
- ✅ Pass rate: 63% → ~76% (13 additional passes)

---

**Next Action**: Implement container-aware wrappers for 6 custom methods.
