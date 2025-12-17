# Deep Analysis Report: Failing Preprocessing Methods

**Generated**: 2025-10-14 22:55:28

---

## Executive Summary

- **Total Methods Analyzed**: 16
- **BROKEN**: 15
- **HEALTHY**: 1

## Detailed Analysis by Category

### BASELINE_CORRECTION

#### FABC

**Status**: `BROKEN`

**Actual Signature**: `lam, scale, num_std, diff_order, min_length, weights, weights_as_mask, x_data, pad_kwargs`

**Registry Parameters**: `lam, scale, num_std, max_iter`

**Instantiation Test**: SUCCESS

**Call Test**: REQUIRES_RUNTIME_INPUT: PreprocessingStep.__call__() missing 1 required positional argument: 'spectral_axis'

**Issues Found**:
- Parameters in class but not in registry: {'weights_as_mask', 'x_data', 'min_length', 'weights', 'diff_order', 'pad_kwargs'}
- Parameters in registry but not in class: {'max_iter'}
- Method requires runtime input: PreprocessingStep.__call__() missing 1 required positional argument: 'spectral_axis'

**Recommendations**:
1. Add to registry param_info: {'weights_as_mask', 'x_data', 'min_length', 'weights', 'diff_order', 'pad_kwargs'}
1. Remove from registry or fix class: {'max_iter'}
1. Make parameter optional with default=None or redesign API

---

### CALIBRATION

#### WavenumberCalibration

**Status**: `BROKEN`

**Actual Signature**: `reference_peaks, poly_order`

**Registry Parameters**: `reference_peaks, poly_order`

**Instantiation Test**: SUCCESS

**Call Test**: REQUIRES_RUNTIME_INPUT: WavenumberCalibration.__call__() missing 1 required positional argument: 'measured_peaks'

**Issues Found**:
- Method requires runtime input: WavenumberCalibration.__call__() missing 1 required positional argument: 'measured_peaks'

**Recommendations**:
1. Make parameter optional with default=None or redesign API

---

#### IntensityCalibration

**Status**: `BROKEN`

**Actual Signature**: `reference`

**Registry Parameters**: `reference`

**Instantiation Test**: SUCCESS

**Call Test**: REQUIRES_RUNTIME_INPUT: IntensityCalibration.__call__() missing 1 required positional argument: 'measured_standard'

**Issues Found**:
- Method requires runtime input: IntensityCalibration.__call__() missing 1 required positional argument: 'measured_standard'

**Recommendations**:
1. Make parameter optional with default=None or redesign API

---

### COSMIC_RAY_REMOVAL

#### WhitakerHayes

**Status**: `BROKEN`

**Actual Signature**: `kernel_size, threshold`

**Registry Parameters**: `kernel_size, threshold`

**Instantiation Test**: SUCCESS

**Call Test**: REQUIRES_RUNTIME_INPUT: PreprocessingStep.__call__() missing 1 required positional argument: 'spectral_axis'

**Issues Found**:
- Method requires runtime input: PreprocessingStep.__call__() missing 1 required positional argument: 'spectral_axis'

**Recommendations**:
1. Make parameter optional with default=None or redesign API

---

#### Gaussian

**Status**: `BROKEN`

**Actual Signature**: `kernel, threshold`

**Registry Parameters**: `kernel, threshold`

**Instantiation Test**: SUCCESS

**Call Test**: ERROR: 'SpectralContainer' object has no attribute 'ndim'

**Issues Found**:
- Call failed: 'SpectralContainer' object has no attribute 'ndim'

---

#### MedianDespike

**Status**: `BROKEN`

**Actual Signature**: `kernel_size, threshold`

**Registry Parameters**: `kernel_size, threshold`

**Instantiation Test**: SUCCESS

**Call Test**: ERROR: 'SpectralContainer' object has no attribute 'ndim'

**Issues Found**:
- Call failed: 'SpectralContainer' object has no attribute 'ndim'

---

### DENOISING

#### Kernel

**Status**: `BROKEN`

**Actual Signature**: `kernel_type, kernel_size`

**Registry Parameters**: `kernel_type, kernel_size`

**Instantiation Test**: SUCCESS

**Call Test**: REQUIRES_RUNTIME_INPUT: PreprocessingStep.__call__() missing 1 required positional argument: 'spectral_axis'

**Issues Found**:
- Method requires runtime input: PreprocessingStep.__call__() missing 1 required positional argument: 'spectral_axis'

**Recommendations**:
1. Make parameter optional with default=None or redesign API

---

### MISCELLANEOUS

#### Cropper

**Status**: `BROKEN`

**Actual Signature**: `region`

**Registry Parameters**: `region`

**Instantiation Test**: SUCCESS

**Call Test**: REQUIRES_RUNTIME_INPUT: PreprocessingStep.__call__() missing 1 required positional argument: 'spectral_axis'

**Issues Found**:
- Method requires runtime input: PreprocessingStep.__call__() missing 1 required positional argument: 'spectral_axis'

**Recommendations**:
1. Make parameter optional with default=None or redesign API

---

#### BackgroundSubtractor

**Status**: `HEALTHY`

**Actual Signature**: `background`

**Registry Parameters**: `background`

**Instantiation Test**: SUCCESS

**Call Test**: SUCCESS

---

#### PeakRatioFeatures

**Status**: `BROKEN`

**Actual Signature**: `peak_positions, window_size, extraction_method, ratio_mode, epsilon`

**Registry Parameters**: `window_size, extraction_method, ratio_mode, epsilon`

**Instantiation Test**: SUCCESS

**Call Test**: REQUIRES_RUNTIME_INPUT: PeakRatioFeatures.__call__() missing 1 required positional argument: 'wavenumbers'

**Issues Found**:
- Parameters in class but not in registry: {'peak_positions'}
- Method requires runtime input: PeakRatioFeatures.__call__() missing 1 required positional argument: 'wavenumbers'

**Recommendations**:
1. Add to registry param_info: {'peak_positions'}
1. Make parameter optional with default=None or redesign API

---

### NORMALISATION

#### MaxIntensity

**Status**: `BROKEN`

**Actual Signature**: `pixelwise`

**Registry Parameters**: `pixelwise`

**Instantiation Test**: SUCCESS

**Call Test**: REQUIRES_RUNTIME_INPUT: PreprocessingStep.__call__() missing 1 required positional argument: 'spectral_axis'

**Issues Found**:
- Method requires runtime input: PreprocessingStep.__call__() missing 1 required positional argument: 'spectral_axis'

**Recommendations**:
1. Make parameter optional with default=None or redesign API

---

#### AUC

**Status**: `BROKEN`

**Actual Signature**: `pixelwise`

**Registry Parameters**: `pixelwise`

**Instantiation Test**: SUCCESS

**Call Test**: REQUIRES_RUNTIME_INPUT: PreprocessingStep.__call__() missing 1 required positional argument: 'spectral_axis'

**Issues Found**:
- Method requires runtime input: PreprocessingStep.__call__() missing 1 required positional argument: 'spectral_axis'

**Recommendations**:
1. Make parameter optional with default=None or redesign API

---

#### MSC

**Status**: `BROKEN`

**Instantiation Test**: SUCCESS

**Call Test**: ERROR: 'SpectralContainer' object has no attribute 'ndim'

**Issues Found**:
- Call failed: 'SpectralContainer' object has no attribute 'ndim'

---

#### QuantileNormalization

**Status**: `BROKEN`

**Actual Signature**: `method`

**Registry Parameters**: `method`

**Instantiation Test**: SUCCESS

**Call Test**: ERROR: 'SpectralContainer' object has no attribute 'ndim'

**Issues Found**:
- Call failed: 'SpectralContainer' object has no attribute 'ndim'

---

#### RankTransform

**Status**: `BROKEN`

**Actual Signature**: `scale_range, standardize`

**Registry Parameters**: `scale_range, standardize`

**Instantiation Test**: SUCCESS

**Call Test**: ERROR: 'SpectralContainer' object has no attribute 'ndim'

**Issues Found**:
- Call failed: 'SpectralContainer' object has no attribute 'ndim'

---

#### ProbabilisticQuotientNormalization

**Status**: `BROKEN`

**Actual Signature**: `reference_spectrum`

**Instantiation Test**: SUCCESS

**Call Test**: ERROR: 'SpectralContainer' object has no attribute 'ndim'

**Issues Found**:
- Parameters in class but not in registry: {'reference_spectrum'}
- Call failed: 'SpectralContainer' object has no attribute 'ndim'

**Recommendations**:
1. Add to registry param_info: {'reference_spectrum'}

---

## Root Cause Analysis

### Common Issues

#### Methods Requiring Runtime Input (9)

These methods need additional data at call time that cannot be provided in pipeline definition:

- miscellaneous.Cropper
- miscellaneous.PeakRatioFeatures
- calibration.WavenumberCalibration
- calibration.IntensityCalibration
- denoising.Kernel
- cosmic_ray_removal.WhitakerHayes
- baseline_correction.FABC
- normalisation.MaxIntensity
- normalisation.AUC

**Fix**: Redesign to accept optional parameters with sensible defaults or None values.

#### Parameter Mismatch (3)

Registry definitions do not match actual class signatures:

- miscellaneous.PeakRatioFeatures
- baseline_correction.FABC
- normalisation.ProbabilisticQuotientNormalization

**Fix**: Use `inspect.signature()` to verify and update registry definitions.

#### Call Failures (6)

Methods fail when called with test data:

- cosmic_ray_removal.Gaussian
- cosmic_ray_removal.MedianDespike
- normalisation.MSC
- normalisation.QuantileNormalization
- normalisation.RankTransform
- normalisation.ProbabilisticQuotientNormalization

**Fix**: Debug individual method implementations.

