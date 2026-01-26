# Preprocessing Methods Reference

Comprehensive reference for all 40+ preprocessing methods available in the application.

## Table of Contents
- [Baseline Correction](#baseline-correction)
- [Smoothing and Denoising](#smoothing-and-denoising)
- [Normalization](#normalization)
- [Derivatives](#derivatives)
- [Feature Engineering](#feature-engineering)
- [Advanced Methods](#advanced-methods)

---

## Baseline Correction

Remove fluorescence background and baseline drift from spectra.

### AsLS (Asymmetric Least Squares)

**Full Name**: Asymmetric Least Squares Smoothing

**Purpose**: Remove baseline while preserving peaks using asymmetric weighting

**Theory**:
Fits a smoothed baseline by minimizing:
```
Σ w_i(y_i - z_i)² + λ Σ (Δ²z_i)²
```
where `w_i` are asymmetric weights (lower for peaks, higher for baseline)

**Parameters**:

| Parameter    | Type  | Range       | Default | Description                                      |
| ------------ | ----- | ----------- | ------- | ------------------------------------------------ |
| `lambda` (λ) | float | 1e2 - 1e9   | 1e5     | Smoothness parameter. Higher = smoother baseline |
| `p`          | float | 0.001 - 0.1 | 0.01    | Asymmetry parameter. Lower = fit valleys better  |
| `max_iter`   | int   | 5 - 50      | 10      | Maximum iterations for convergence               |
| `tol`        | float | 1e-6 - 1e-3 | 1e-6    | Convergence tolerance                            |

**Parameter Guide**:
```python
# Conservative (preserves more peaks)
lambda = 1e5, p = 0.01

# Standard (balanced)
lambda = 1e6, p = 0.01

# Aggressive (removes more baseline)
lambda = 1e7, p = 0.001
```

**Usage Example**:
```python
from functions.preprocess import apply_asls

# Apply AsLS baseline correction
corrected = apply_asls(
    spectra,
    lam=1e5,
    p=0.01,
    max_iter=10
)
```

**Interpretation**:
- **λ (lambda)**: Controls baseline smoothness
  - Too low → Follows peaks (underfitting)
  - Too high → Over-smooths (may miss real baseline curvature)
  - Sweet spot: 1e5 - 1e6 for most Raman spectra

- **p**: Controls asymmetry
  - Lower → Treats peaks as outliers (good for sharp peaks)
  - Higher → Fits through peaks (use if baseline has structure)

**Common Issues**:
1. **Peaks removed**: λ too high or p too high → Reduce λ or p
2. **Baseline remains**: λ too low → Increase λ
3. **Slow convergence**: Increase tol or reduce max_iter
4. **Oscillations**: p too low → Increase p slightly

**When to Use**:
- ✓ General-purpose baseline correction
- ✓ Fluorescence background
- ✓ Unknown baseline shape
- ✗ Not for: Very noisy data (smooth first)

**Reference**: 
Eilers & Boelens (2005). "Baseline Correction with Asymmetric Least Squares Smoothing"

---

### AirPLS (Adaptive Iteratively Reweighted Penalized Least Squares)

**Full Name**: Adaptive Iteratively Reweighted Penalized Least Squares

**Purpose**: Automatic baseline correction with minimal parameter tuning

**Theory**:
Iteratively fits baseline by:
1. Penalized least squares fitting
2. Adaptive weighting based on residuals
3. Iteration until convergence

**Parameters**:

| Parameter    | Type  | Range       | Default | Description                  |
| ------------ | ----- | ----------- | ------- | ---------------------------- |
| `lambda` (λ) | float | 1e2 - 1e7   | 1e5     | Smoothness parameter         |
| `porder`     | int   | 1, 2        | 1       | Difference order for penalty |
| `max_iter`   | int   | 10 - 100    | 15      | Maximum iterations           |
| `min_diff`   | float | 1e-6 - 1e-3 | 1e-5    | Convergence criterion        |

**Parameter Guide**:
```python
# Fast (fewer iterations)
lambda = 1e4, max_iter = 10

# Balanced (recommended)
lambda = 1e5, max_iter = 15

# Thorough (more iterations)
lambda = 1e6, max_iter = 30
```

**Usage Example**:
```python
from functions.preprocess import apply_airpls

corrected = apply_airpls(
    spectra,
    lam=1e5,
    porder=1,
    max_iter=15
)
```

**Interpretation**:
- **Automatic adaptation**: Weights adjust to separate peaks from baseline
- **Less sensitive to p**: No asymmetry parameter needed
- **porder=1**: First-order differences (smoother)
- **porder=2**: Second-order differences (more flexible)

**Common Issues**:
1. **Negative baseline**: Normal behavior, corrects fluorescence
2. **Slow**: Reduce max_iter or increase λ
3. **Incomplete correction**: Increase max_iter or reduce λ

**When to Use**:
- ✓ Automatic baseline correction
- ✓ Batch processing
- ✓ When AsLS parameters unclear
- ✓ Complex baseline shapes

**Reference**:
Zhang et al. (2010). "Baseline correction using adaptive iteratively reweighted penalized least squares"

---

### Polynomial Baseline

**Purpose**: Fit and subtract polynomial baseline

**Theory**:
Fits polynomial of degree `n`:
```
baseline = a₀ + a₁x + a₂x² + ... + aₙxⁿ
```

**Parameters**:

| Parameter | Type | Range  | Default | Description       |
| --------- | ---- | ------ | ------- | ----------------- |
| `degree`  | int  | 1 - 10 | 3       | Polynomial degree |

**Parameter Guide**:
```python
degree = 1  # Linear baseline
degree = 2  # Quadratic
degree = 3  # Cubic (most common)
degree = 4-5  # Higher order (flexible)
degree > 5  # Risky (may overfit)
```

**Usage Example**:
```python
from functions.preprocess import apply_polynomial_baseline

corrected = apply_polynomial_baseline(
    spectra,
    degree=3
)
```

**Common Issues**:
1. **Underfitting**: degree too low → Increase degree
2. **Overfitting**: degree too high → Reduce degree
3. **Peaks affected**: Use robust fitting or lower degree

**When to Use**:
- ✓ Simple, smooth baseline
- ✓ Known polynomial shape
- ✓ Fast processing needed
- ✗ Not for: Complex baseline, fluorescence

---

### Whittaker Smoothing

**Purpose**: Smooth baseline using penalized least squares

**Parameters**:

| Parameter     | Type  | Range     | Default | Description          |
| ------------- | ----- | --------- | ------- | -------------------- |
| `lambda` (λ)  | float | 1e2 - 1e9 | 1e5     | Smoothness parameter |
| `differences` | int   | 1, 2, 3   | 2       | Order of differences |

**Usage Example**:
```python
from functions.preprocess import apply_whittaker

corrected = apply_whittaker(
    spectra,
    lam=1e5,
    differences=2
)
```

**When to Use**:
- ✓ Smooth baseline
- ✓ Preserving peak shapes
- ✓ Alternative to polynomial

**Reference**:
Eilers (2003). "A Perfect Smoother"

---

### FABC (Fully Automatic Baseline Correction)

**Purpose**: Completely automatic baseline correction with no tuning

**Parameters**:

| Parameter       | Type | Range     | Default | Description                     |
| --------------- | ---- | --------- | ------- | ------------------------------- |
| `window_length` | int  | 100 - 500 | 200     | Window size for local fitting   |
| `iterations`    | int  | 1 - 20    | 10      | Number of correction iterations |

**Usage Example**:
```python
from functions.preprocess.fabc_fixed import apply_fabc

corrected = apply_fabc(
    spectra,
    window_length=200,
    iterations=10
)
```

**When to Use**:
- ✓ No user expertise required
- ✓ Batch processing
- ✓ Standardized pipelines
- ✓ Unknown baseline characteristics

---

### Butterworth High-Pass Filter

**Purpose**: Remove low-frequency baseline components using frequency domain filtering

**Parameters**:

| Parameter | Type  | Range       | Default | Description                   |
| --------- | ----- | ----------- | ------- | ----------------------------- |
| `cutoff`  | float | 0.001 - 0.1 | 0.01    | Cutoff frequency (normalized) |
| `order`   | int   | 1 - 10      | 4       | Filter order                  |

**Usage Example**:
```python
from functions.preprocess import apply_butterworth_highpass

corrected = apply_butterworth_highpass(
    spectra,
    cutoff=0.01,
    order=4
)
```

**When to Use**:
- ✓ Frequency-domain baseline removal
- ✓ Uniform low-frequency drift
- ✗ Not ideal for: Non-uniform baseline

---

## Smoothing and Denoising

Reduce noise while preserving spectral features.

### Savitzky-Golay Filter

**Purpose**: Polynomial smoothing that preserves peak shape and height

**Theory**:
Fits local polynomials using least squares within a moving window

**Parameters**:

| Parameter       | Type | Range        | Default | Description                                           |
| --------------- | ---- | ------------ | ------- | ----------------------------------------------------- |
| `window_length` | int  | 5 - 51 (odd) | 11      | Size of smoothing window                              |
| `polyorder`     | int  | 2 - 5        | 3       | Polynomial order                                      |
| `deriv`         | int  | 0 - 2        | 0       | Derivative order (0=smooth, 1=1st deriv, 2=2nd deriv) |

**Parameter Guide**:
```python
# Light smoothing
window_length = 7, polyorder = 3

# Moderate smoothing (recommended)
window_length = 11, polyorder = 3

# Heavy smoothing
window_length = 21, polyorder = 3

# Peak sharpening (1st derivative)
window_length = 11, polyorder = 3, deriv = 1

# Peak resolution (2nd derivative)
window_length = 11, polyorder = 3, deriv = 2
```

**Usage Example**:
```python
from functions.preprocess import apply_savgol

# Smoothing
smoothed = apply_savgol(
    spectra,
    window_length=11,
    polyorder=3,
    deriv=0
)

# First derivative
derivative = apply_savgol(
    spectra,
    window_length=11,
    polyorder=3,
    deriv=1
)
```

**Interpretation**:
- **window_length**: Larger window = more smoothing, but peak broadening
- **polyorder**: Higher order preserves sharp features better
- **deriv=1**: Converts peaks to zero-crossings, removes baseline
- **deriv=2**: Converts peaks to negative dips, enhances resolution

**Common Issues**:
1. **Over-smoothing**: Window too large → Reduce window
2. **Peak broadening**: Window too large → Use window ≤ 11
3. **Noisy derivatives**: Smooth first, then derivative
4. **Oscillations**: polyorder too high → Reduce to 2 or 3

**When to Use**:
- ✓ General smoothing
- ✓ Peak-preserving smoothing
- ✓ Derivatives for baseline removal
- ✓ Most common choice

**Constraints**:
- window_length > polyorder
- window_length must be odd

**Reference**:
Savitzky & Golay (1964). "Smoothing and Differentiation of Data by Simplified Least Squares Procedures"

---

### Gaussian Smoothing

**Purpose**: Strong smoothing using Gaussian kernel

**Theory**:
Convolves spectrum with Gaussian kernel:
```
K(x) = (1/√(2πσ²)) exp(-x²/2σ²)
```

**Parameters**:

| Parameter | Type  | Range     | Default | Description                           |
| --------- | ----- | --------- | ------- | ------------------------------------- |
| `sigma`   | float | 0.5 - 5.0 | 2.0     | Standard deviation of Gaussian kernel |

**Parameter Guide**:
```python
sigma = 1.0  # Light smoothing
sigma = 2.0  # Moderate (default)
sigma = 3.0  # Heavy smoothing
sigma > 4.0  # Very strong (may blur peaks)
```

**Usage Example**:
```python
from functions.preprocess import apply_gaussian

smoothed = apply_gaussian(
    spectra,
    sigma=2.0
)
```

**Common Issues**:
1. **Peak broadening**: sigma too high → Reduce sigma
2. **Insufficient smoothing**: sigma too low → Increase sigma

**When to Use**:
- ✓ Very noisy data
- ✓ When peak shape less critical
- ✓ Visualization
- ✗ Not for: Quantitative analysis (peak heights change)

---

### Moving Average

**Purpose**: Simple uniform smoothing

**Parameters**:

| Parameter     | Type | Range        | Default | Description |
| ------------- | ---- | ------------ | ------- | ----------- |
| `window_size` | int  | 3 - 21 (odd) | 5       | Window size |

**Usage Example**:
```python
from functions.preprocess import apply_moving_average

smoothed = apply_moving_average(
    spectra,
    window_size=5
)
```

**When to Use**:
- ✓ Quick smoothing
- ✓ Preliminary exploration
- ✗ Not recommended for: Publication (use Savitzky-Golay)

---

### Median Filter

**Purpose**: Remove spike noise (cosmic rays, detector artifacts)

**Theory**:
Replaces each point with median of surrounding window

**Parameters**:

| Parameter     | Type | Range        | Default | Description |
| ------------- | ---- | ------------ | ------- | ----------- |
| `window_size` | int  | 3 - 11 (odd) | 5       | Window size |

**Usage Example**:
```python
from functions.preprocess import apply_median_filter

despike = apply_median_filter(
    spectra,
    window_size=5
)
```

**Common Issues**:
1. **Peak clipping**: Window too large → Use window=3 or 5
2. **Spikes remain**: Window too small → Increase to 7

**When to Use**:
- ✓ Cosmic ray removal
- ✓ Spike artifacts
- ✓ Before other preprocessing
- ✓ CCD detector noise

**Best Practice**: Apply BEFORE baseline correction and smoothing

---

### Kernel Denoising

**Purpose**: Advanced denoising using various kernel functions

**Available Kernels**: Gaussian, Epanechnikov, Tricube, Triweight

**Parameters**:

| Parameter   | Type  | Range     | Default    | Description      |
| ----------- | ----- | --------- | ---------- | ---------------- |
| `kernel`    | str   | -         | 'gaussian' | Kernel type      |
| `bandwidth` | float | 0.5 - 5.0 | 1.0        | Kernel bandwidth |

**Usage Example**:
```python
from functions.preprocess.kernel_denoise import apply_kernel_denoise

denoised = apply_kernel_denoise(
    spectra,
    kernel='gaussian',
    bandwidth=1.0
)
```

**When to Use**:
- ✓ Alternative to Gaussian smoothing
- ✓ Experimenting with different kernels

---

## Normalization

Scale spectra to comparable ranges.

### Vector Normalization (L2 Norm)

**Purpose**: Normalize to unit length (most common)

**Theory**:
```
normalized = spectrum / ||spectrum||₂
where ||spectrum||₂ = √(Σ x²)
```

**Parameters**: None

**Usage Example**:
```python
from functions.preprocess import apply_vector_norm

normalized = apply_vector_norm(spectra)
```

**Effect**:
- Makes all spectra have same total "energy"
- Removes intensity variations
- Preserves relative peak ratios

**When to Use**:
- ✓ Most common choice
- ✓ Classification tasks
- ✓ Removing concentration effects
- ✓ SVM, neural networks
- ✓ After baseline correction

---

### Min-Max Normalization

**Purpose**: Scale to [0, 1] range

**Theory**:
```
normalized = (spectrum - min) / (max - min)
```

**Parameters**:

| Parameter       | Type  | Range | Default | Description             |
| --------------- | ----- | ----- | ------- | ----------------------- |
| `feature_range` | tuple | -     | (0, 1)  | Target range (min, max) |

**Usage Example**:
```python
from functions.preprocess import apply_minmax_norm

normalized = apply_minmax_norm(
    spectra,
    feature_range=(0, 1)
)
```

**When to Use**:
- ✓ Neural networks
- ✓ Visualization
- ✓ Bounded input required
- ✗ Not for: Preserving absolute intensities

---

### Area Normalization

**Purpose**: Normalize by area under curve

**Theory**:
```
normalized = spectrum / Σ|spectrum|
```

**Parameters**: None

**Usage Example**:
```python
from functions.preprocess import apply_area_norm

normalized = apply_area_norm(spectra)
```

**When to Use**:
- ✓ Concentration normalization
- ✓ Comparing relative peak heights
- ✓ Eliminating total intensity differences
- ✗ Not for: Absolute quantification

---

### Standard Normal Variate (SNV)

**Purpose**: Remove multiplicative scatter effects

**Theory**:
```
SNV = (spectrum - mean) / std
```

**Parameters**: None

**Usage Example**:
```python
from functions.preprocess import apply_snv

normalized = apply_snv(spectra)
```

**When to Use**:
- ✓ Solid samples with scattering
- ✓ Particle size variations
- ✓ NIR spectroscopy (also useful for Raman)
- ✓ Removing multiplicative effects

**Reference**:
Barnes et al. (1989). "Standard Normal Variate Transformation"

---

### Multiplicative Scatter Correction (MSC)

**Purpose**: Correct for light scattering variations

**Theory**:
Fits each spectrum to a reference (mean spectrum):
```
spectrum_i = a + b × reference
corrected = (spectrum_i - a) / b
```

**Parameters**:

| Parameter   | Type  | Default | Description                               |
| ----------- | ----- | ------- | ----------------------------------------- |
| `reference` | array | None    | Reference spectrum (default: mean of all) |

**Usage Example**:
```python
from functions.preprocess import apply_msc

normalized = apply_msc(
    spectra,
    reference=None  # Auto: use mean
)
```

**When to Use**:
- ✓ Diffuse reflectance spectroscopy
- ✓ Particle size effects
- ✓ Scattering correction
- ✓ Quantitative analysis

**Requirement**: Need reference spectrum (usually mean)

**Reference**:
Geladi et al. (1985). "Linearization and Scatter-Correction for Near-Infrared Reflectance Spectra"

---

### Quantile Normalization

**Purpose**: Match distributions across spectra

**Parameters**:

| Parameter     | Type | Range      | Default | Description         |
| ------------- | ---- | ---------- | ------- | ------------------- |
| `n_quantiles` | int  | 100 - 1000 | 1000    | Number of quantiles |

**Usage Example**:
```python
from functions.preprocess.advanced_normalization import apply_quantile_norm

normalized = apply_quantile_norm(
    spectra,
    n_quantiles=1000
)
```

**When to Use**:
- ✓ Batch effect correction
- ✓ Making distributions identical
- ✗ Less common in Raman

---

### Probabilistic Quotient Normalization (PQN)

**Purpose**: Dilution correction for metabolomics

**Theory**:
```
1. Calculate reference (median spectrum)
2. Calculate quotients: spectrum / reference
3. Normalize by median quotient
```

**Parameters**:

| Parameter   | Type  | Default | Description        |
| ----------- | ----- | ------- | ------------------ |
| `reference` | array | None    | Reference spectrum |

**Usage Example**:
```python
from functions.preprocess.advanced_normalization import apply_pqn

normalized = apply_pqn(
    spectra,
    reference=None
)
```

**When to Use**:
- ✓ Metabolomics
- ✓ Dilution correction
- ✓ Biological fluids

**Reference**:
Dieterle et al. (2006). "Probabilistic Quotient Normalization"

---

### Rank Transformation

**Purpose**: Convert to ranks (non-parametric)

**Parameters**: None

**Usage Example**:
```python
from functions.preprocess.advanced_normalization import apply_rank_transform

ranked = apply_rank_transform(spectra)
```

**When to Use**:
- ✓ Non-parametric analysis
- ✓ Outlier-resistant
- ✓ When distribution doesn't matter

---

## Derivatives

Calculate spectral derivatives for baseline removal and peak resolution.

### First Derivative (Savitzky-Golay)

**Purpose**: Remove baseline, enhance peak differences

**Theory**:
First derivative of smoothed spectrum using Savitzky-Golay

**Parameters**: Same as Savitzky-Golay + `deriv=1`

**Usage Example**:
```python
from functions.preprocess import apply_savgol

derivative1 = apply_savgol(
    spectra,
    window_length=11,
    polyorder=3,
    deriv=1
)
```

**Effect**:
- Peaks become positive/negative transitions
- Baseline removed (constant → zero)
- Peak maxima → zero-crossings

**When to Use**:
- ✓ Alternative to baseline correction
- ✓ Overlapping peak resolution
- ✓ Chemometric analysis
- ✗ Amplifies noise (smooth first!)

---

### Second Derivative

**Purpose**: Sharpen peaks, resolve overlaps

**Theory**:
Second derivative of smoothed spectrum

**Parameters**: Same as Savitzky-Golay + `deriv=2`

**Usage Example**:
```python
from functions.preprocess import apply_savgol

derivative2 = apply_savgol(
    spectra,
    window_length=11,
    polyorder=3,
    deriv=2
)
```

**Effect**:
- Peaks become negative dips
- Sharpens overlapping peaks
- Greatly amplifies noise

**When to Use**:
- ✓ Overlapping peak resolution
- ✓ Peak identification
- ✓ Advanced analysis
- ⚠️ Warning: Requires good smoothing, amplifies noise significantly

---

## Feature Engineering

Create new features from spectra.

### Peak Ratio

**Purpose**: Calculate ratio between two peaks

**Parameters**:

| Parameter     | Type  | Description                        |
| ------------- | ----- | ---------------------------------- |
| `peak1_range` | tuple | (start, end) wavenumber for peak 1 |
| `peak2_range` | tuple | (start, end) wavenumber for peak 2 |

**Usage Example**:
```python
from functions.preprocess.feature_engineering import calculate_peak_ratio

# Amide I / CH2 ratio
ratio = calculate_peak_ratio(
    spectra,
    wavenumbers,
    peak1_range=(1645, 1665),  # Amide I
    peak2_range=(1440, 1460)   # CH2
)
```

**When to Use**:
- ✓ Known biomarker ratios
- ✓ Creating interpretable features
- ✓ Reducing dimensionality
- ✓ Band ratio analysis

**Example Ratios**:
```
I₁₆₅₅/I₁₄₄₅ (Amide I / CH₂)
I₁₂₉₀/I₁₂₄₀ (Amide III ratio)
I₁₀₀₀/I₁₆₀₀ (Custom biomarkers)
```

---

### Wavelet Transform

**Purpose**: Multi-resolution decomposition for denoising

**Parameters**:

| Parameter   | Type | Range  | Default | Description                     |
| ----------- | ---- | ------ | ------- | ------------------------------- |
| `wavelet`   | str  | -      | 'db4'   | Wavelet type (db4, sym8, coif5) |
| `level`     | int  | 1 - 10 | 4       | Decomposition level             |
| `threshold` | str  | -      | 'soft'  | Thresholding type (soft, hard)  |

**Usage Example**:
```python
from functions.preprocess.feature_engineering import apply_wavelet_transform

denoised = apply_wavelet_transform(
    spectra,
    wavelet='db4',
    level=4,
    threshold='soft'
)
```

**When to Use**:
- ✓ Complex noise patterns
- ✓ Multi-scale features
- ✓ Non-stationary noise
- ✗ Computationally expensive

**Reference**:
Mallat (1989). "A theory for multiresolution signal decomposition: the wavelet representation"

---

## Advanced Methods

Specialized preprocessing methods.

### Convolutional Denoising Autoencoder (CDAE)

**Purpose**: Deep learning-based denoising

**Theory**:
Neural network trained to reconstruct clean spectra from noisy input

**Parameters**:

| Parameter    | Type | Default | Description               |
| ------------ | ---- | ------- | ------------------------- |
| `model_path` | str  | None    | Path to trained model     |
| `batch_size` | int  | 32      | Batch size for prediction |

**Requirements**:
- PyTorch installed
- GPU recommended
- Pre-trained model

**Usage Example**:
```python
from functions.preprocess.deep_learning import apply_cdae

denoised = apply_cdae(
    spectra,
    model_path='models/cdae_raman.pth',
    batch_size=32
)
```

**When to Use**:
- ✓ After training on your data type
- ✓ Complex noise patterns
- ✓ Large datasets
- ✗ Requires: Training data, GPU, expertise

---

### Background Subtraction

**Purpose**: Subtract background/blank spectrum

**Parameters**:

| Parameter    | Type  | Description                     |
| ------------ | ----- | ------------------------------- |
| `background` | array | Background spectrum to subtract |

**Usage Example**:
```python
from functions.preprocess import subtract_background

corrected = subtract_background(
    spectra,
    background=blank_spectrum
)
```

**When to Use**:
- ✓ Measured blank/background available
- ✓ Removing substrate contribution
- ✓ Consistent background across samples

---

### Calibration

**Purpose**: Wavenumber axis calibration

**Types**:
1. **Linear Shift**: Single reference peak
2. **Polynomial**: Multiple reference peaks

**Parameters**:

| Parameter         | Type | Description              |
| ----------------- | ---- | ------------------------ |
| `reference_peaks` | list | Expected peak positions  |
| `measured_peaks`  | list | Measured peak positions  |
| `method`          | str  | 'linear' or 'polynomial' |

**Usage Example**:
```python
from functions.preprocess.calibration import apply_calibration

# Calibrate using silicon peak
calibrated_wn = apply_calibration(
    wavenumbers,
    reference_peaks=[520.7],  # Silicon
    measured_peaks=[522.3],    # Measured
    method='linear'
)
```

**When to Use**:
- ✓ Wavenumber axis errors detected
- ✓ Instrument drift correction
- ✓ Before combining datasets

**Reference Standards**:
- Silicon: 520.7 cm⁻¹
- Polystyrene: 1001, 1031, 1602 cm⁻¹
- Diamond: 1332 cm⁻¹

---

## Method Selection Guide

### Decision Matrix

| Goal                     | Recommended Method(s) | Parameters         |
| ------------------------ | --------------------- | ------------------ |
| **Remove baseline**      | AsLS                  | λ=1e5, p=0.01      |
|                          | AirPLS                | λ=1e5              |
|                          | Polynomial            | degree=3           |
| **Reduce noise**         | Savitzky-Golay        | window=11, order=3 |
|                          | Gaussian              | sigma=2.0          |
|                          | Median (spikes)       | window=5           |
| **Normalize intensity**  | Vector (L2)           | -                  |
|                          | SNV                   | -                  |
|                          | Area                  | -                  |
| **Remove scatter**       | MSC                   | reference=mean     |
|                          | SNV                   | -                  |
| **Baseline alternative** | 1st Derivative        | SavGol deriv=1     |
| **Peak resolution**      | 2nd Derivative        | SavGol deriv=2     |
| **Create features**      | Peak Ratios           | Custom ranges      |

(recommended-pipelines)=

### Common Pipelines

**Standard Pipeline**:
```python
1. AsLS (λ=1e5, p=0.01)
2. Savitzky-Golay (w=11, order=3)
3. Vector Normalization
```

**High-Noise Pipeline**:
```python
1. Median Filter (w=5)
2. AirPLS (λ=1e6)
3. Gaussian (σ=2.0)
4. SNV
```

**Derivative Pipeline**:
```python
1. AsLS (λ=1e5, p=0.01)
2. Savitzky-Golay 1st Derivative (w=11, order=3, deriv=1)
3. Vector Normalization
```

**Quantitative Pipeline**:
```python
1. AsLS (λ=1e6, p=0.001)
2. MSC (reference=mean)
3. Savitzky-Golay (w=9, order=3)
4. Area Normalization
```


---

(validation)=

## Parameter Constraints

### Automatic Validation

All methods include automatic parameter validation:

**Type Checking**:
```text
# Integer parameters converted if needed
window_length = "11" → 11 (converted)
polyorder = 3.0 → 3 (converted)

# Float parameters validated
lambda = "1e5" → 100000.0 (converted)
p = 0.01 (valid)
```

**Range Validation**:
```text
# Values clamped to valid ranges
lambda = 1e10 → 1e9 (max allowed)
window_length = 3 → 5 (min allowed)
p = -0.01 → 0.001 (min allowed)
```

**Logical Validation**:
```python
# Constraints enforced
if window_length <= polyorder:
    window_length = polyorder + 2
    
if window_length % 2 == 0:
    window_length += 1  # Make odd
```

---

## Best Practices

### General Guidelines

1. **Order Matters**:
   ```
   Correct: Spike removal → Baseline → Smoothing → Normalization
   Wrong: Normalization → Baseline (affects baseline estimation)
   ```

2. **Less is More**:
   - Use minimal necessary steps
   - Over-processing loses information
   - Validate each step visually

3. **Parameter Tuning**:
   - Start with defaults
   - Adjust based on visual inspection
   - Test on representative subset
   - Document final parameters

4. **Validation**:
   - Always preview before applying
   - Check multiple spectra
   - Compare before/after
   - Verify peak preservation

### Method-Specific Tips

**AsLS**:
- Start with λ=1e5, adjust if needed
- Lower p for sharper peaks
- Check that real peaks aren't removed

**Savitzky-Golay**:
- Window ≤ 11 for most cases
- polyorder = 3 is usually optimal
- Don't over-smooth

**Derivatives**:
- Always smooth before derivative
- 1st derivative: window ≥ 11
- 2nd derivative: window ≥ 15, very noisy

**Normalization**:
- Apply AFTER baseline and smoothing
- Vector norm: most common choice
- SNV: for scattering issues

---

## Troubleshooting

### Common Issues

| Issue            | Cause                  | Solution                        |
| ---------------- | ---------------------- | ------------------------------- |
| Peaks removed    | λ too high in AsLS     | Reduce λ to 1e4-1e5             |
| Baseline remains | λ too low              | Increase λ to 1e6-1e7           |
| Over-smoothed    | Window too large       | Reduce SavGol window to 7-9     |
| Noisy            | Insufficient smoothing | Increase window or sigma        |
| Negative values  | Normal after baseline  | Use area/vector norm            |
| Spikes remain    | Window too small       | Use median filter window=5-7    |
| Slow processing  | Too many iterations    | Reduce max_iter or increase tol |

---

## References

1. **Eilers & Boelens (2005)**: AsLS method
2. **Zhang et al. (2010)**: AirPLS method
3. **Savitzky & Golay (1964)**: SavGol filter
4. **Barnes et al. (1989)**: SNV normalization
5. **Geladi et al. (1985)**: MSC method
6. **Dieterle et al. (2006)**: PQN normalization
7. **Eilers (2003)**: Whittaker smoother
8. **Mallat (1989)**: Wavelet theory

See [References](../references.md) for complete citations.

---

## See Also

- [Preprocessing User Guide](../user-guide/preprocessing.md) - Step-by-step tutorials
- [Best Practices](../user-guide/best-practices.md) - Preprocessing strategies
- [FAQ - Preprocessing](../faq.md#preprocessing-questions) - Common questions
- [Glossary](../glossary.md) - Term definitions

---

**Total Methods Documented**: 40+
**Last Updated**: 2026-01-24
