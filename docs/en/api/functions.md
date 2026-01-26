# Functions API

Reference documentation for processing functions.

## Table of Contents
- [Data Loading Functions](#data-loading-functions)
- [Preprocessing Functions](#preprocessing-functions)
  - [Baseline Correction](#baseline-correction)
  - [Smoothing](#smoothing)
  - [Normalization](#normalization)
  - [Derivatives](#derivatives)
  - [Advanced Processing](#advanced-processing)
- [Analysis Functions](#analysis-functions)
  - [Dimensionality Reduction](#dimensionality-reduction)
  - [Clustering](#clustering)
  - [Statistical Tests](#statistical-tests)
- [Machine Learning Functions](#machine-learning-functions)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Feature Importance](#feature-importance)
- [Utility Functions](#utility-functions)

---

## Data Loading Functions

### functions/data_loader.py

#### load_spectra_from_csv()

Load Raman spectra from CSV file.

```python
from functions.data_loader import load_spectra_from_csv

data = load_spectra_from_csv('data/samples.csv')

# Returns:
# {
#     'wavenumbers': np.array([400, 401, 402, ...]),
#     'spectra': np.array([[...], [...], ...]),
#     'labels': ['Sample1', 'Sample2', ...],
#     'groups': ['Control', 'Control', 'Treatment', ...],
#     'metadata': {
#         'acquisition_date': '2026-01-24',
#         'instrument': 'Renishaw inVia',
#         'laser_power': 50,
#         'exposure_time': 10
#     }
# }
```

**Parameters**:
- `filepath` (str): Path to CSV file
- `has_header` (bool, default=True): First row contains headers
- `wavenumber_column` (int or str, default=0): Wavenumber column index/name
- `delimiter` (str, default=','): Column delimiter

**Returns**: dict - Spectral data dictionary

**CSV Format**:
```text
Wavenumber,Sample1,Sample2,Sample3
400.0,0.12,0.15,0.11
401.0,0.13,0.16,0.12
402.0,0.14,0.17,0.13
...
```

**Raises**:
- `FileNotFoundError`: File doesn't exist
- `ValueError`: Invalid file format
- `DataError`: Inconsistent data dimensions

#### save_spectra_to_csv()

Save spectra to CSV file.

```python
from functions.data_loader import save_spectra_to_csv

save_spectra_to_csv(
    data={
        'wavenumbers': wavenumbers,
        'spectra': spectra,
        'labels': labels
    },
    filepath='output/preprocessed.csv',
    include_metadata=True
)
```

**Parameters**:
- `data` (dict): Spectral data dictionary
- `filepath` (str): Output file path
- `include_metadata` (bool, default=True): Include metadata as comments
- `delimiter` (str, default=','): Column delimiter
- `float_format` (str, default='%.6f'): Number formatting

**Side Effects**: Creates file at specified path

#### load_raman_peaks()

Load reference Raman peak database.

```python
from functions.data_loader import load_raman_peaks

peaks = load_raman_peaks()

# Returns:
# {
#     'proteins': {
#         'phenylalanine': [1004, 1033],
#         'amide_I': [1650, 1680],
#         'amide_III': [1230, 1300]
#     },
#     'lipids': {
#         'CH2_stretch': [2850, 2900],
#         'CH2_bend': [1440, 1470]
#     },
#     'nucleic_acids': {
#         'DNA_backbone': [785, 810],
#         'RNA_bases': [810, 850]
#     }
# }
```

**Returns**: dict - Peak database by biomolecule category

**Source**: `assets/data/raman_peaks.json`

---

## Preprocessing Functions

### Baseline Correction

#### apply_asls()

Apply Asymmetric Least Squares (AsLS) baseline correction.

```python
from functions.preprocess.baseline import apply_asls

corrected = apply_asls(
    spectrum=spectrum,
    lambda_=1e5,    # Smoothness
    p=0.01,         # Asymmetry
    max_iter=10
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum (1D array)
- `lambda_` (float): Smoothness parameter (1e2 - 1e9)
  - Lower → follows peaks more closely
  - Higher → smoother baseline
- `p` (float): Asymmetry parameter (0.001 - 0.1)
  - Lower → fits valleys more closely
  - Higher → baseline goes through peaks
- `max_iter` (int, default=10): Maximum iterations

**Returns**: np.ndarray - Baseline-corrected spectrum

**Algorithm**: Iteratively weighted least squares with asymmetric weights

**Reference**: Eilers & Boelens (2005) Baseline Correction with Asymmetric Least Squares Smoothing

#### apply_airpls()

Apply Adaptive Iteratively Reweighted Penalized Least Squares (airPLS).

```python
from functions.preprocess.baseline import apply_airpls

corrected = apply_airpls(
    spectrum=spectrum,
    lambda_=100,
    porder=1,
    max_iter=15
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `lambda_` (float): Smoothness parameter (1 - 1e6)
- `porder` (int): Difference order (1 or 2)
- `max_iter` (int, default=15): Maximum iterations

**Returns**: np.ndarray - Baseline-corrected spectrum

**Algorithm**: Adaptive weighting based on residual signs

**Reference**: Zhang et al. (2010) Baseline correction using adaptive iteratively reweighted penalized least squares

#### apply_polynomial_baseline()

Apply polynomial baseline fitting.

```python
from functions.preprocess.baseline import apply_polynomial_baseline

corrected = apply_polynomial_baseline(
    spectrum=spectrum,
    degree=3,
    mask_peaks=True,
    threshold=0.8
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `degree` (int): Polynomial degree (1-10)
  - 1: Linear baseline
  - 2-3: Gentle curvature
  - 4-6: Complex baseline
  - 7+: Risk of overfitting
- `mask_peaks` (bool, default=False): Exclude peaks from fit
- `threshold` (float, default=0.8): Peak detection threshold

**Returns**: np.ndarray - Baseline-corrected spectrum

**Algorithm**: Least squares polynomial fitting

#### apply_whittaker_baseline()

Apply Whittaker smoothing baseline.

```python
from functions.preprocess.baseline import apply_whittaker_baseline

corrected = apply_whittaker_baseline(
    spectrum=spectrum,
    lambda_=1000,
    differences=2
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `lambda_` (float): Smoothness parameter (1 - 1e6)
- `differences` (int): Order of differences (1 or 2)

**Returns**: np.ndarray - Baseline-corrected spectrum

**Algorithm**: Penalized least squares with difference penalty

#### apply_fabc()

Apply Fully Automatic Baseline Correction (FABC).

```python
from functions.preprocess.fabc_fixed import apply_fabc

corrected = apply_fabc(
    spectrum=spectrum,
    window_length=51,
    iterations=10
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `window_length` (int): Smoothing window size (5-201, odd)
- `iterations` (int, default=10): Number of iterations

**Returns**: np.ndarray - Baseline-corrected spectrum

**Algorithm**: Iterative morphological operations

#### apply_butterworth_filter()

Apply Butterworth high-pass filter for baseline removal.

```python
from functions.preprocess.baseline import apply_butterworth_filter

corrected = apply_butterworth_filter(
    spectrum=spectrum,
    cutoff=0.01,
    order=4,
    sampling_rate=1.0
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `cutoff` (float): Cutoff frequency (0.001 - 0.1)
- `order` (int): Filter order (2-10)
- `sampling_rate` (float): Sampling rate

**Returns**: np.ndarray - High-pass filtered spectrum

**Algorithm**: Butterworth high-pass filter (removes low-frequency baseline)

### Smoothing

#### apply_savgol()

Apply Savitzky-Golay smoothing.

```python
from functions.preprocess.kernel_denoise import apply_savgol

smoothed = apply_savgol(
    spectrum=spectrum,
    window_length=11,
    polyorder=3,
    deriv=0
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `window_length` (int): Window size (5-51, odd, > polyorder)
  - Smaller → less smoothing, preserves peaks
  - Larger → more smoothing, may broaden peaks
- `polyorder` (int): Polynomial order (2-5)
  - 2-3: Most common
  - 4-5: More flexible but risk of artifacts
- `deriv` (int, default=0): Derivative order (0, 1, or 2)
  - 0: Smoothing only
  - 1: First derivative
  - 2: Second derivative

**Returns**: np.ndarray - Smoothed spectrum

**Algorithm**: Local polynomial regression within sliding window

**Reference**: Savitzky & Golay (1964) Smoothing and Differentiation of Data by Simplified Least Squares Procedures

#### apply_gaussian_filter()

Apply Gaussian smoothing.

```python
from functions.preprocess.kernel_denoise import apply_gaussian_filter

smoothed = apply_gaussian_filter(
    spectrum=spectrum,
    sigma=2.0,
    mode='reflect'
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `sigma` (float): Standard deviation (0.5 - 5.0)
  - Smaller → less smoothing
  - Larger → more smoothing
- `mode` (str, default='reflect'): Edge handling mode

**Returns**: np.ndarray - Smoothed spectrum

**Algorithm**: Convolution with Gaussian kernel

#### apply_moving_average()

Apply moving average smoothing.

```python
from functions.preprocess.kernel_denoise import apply_moving_average

smoothed = apply_moving_average(
    spectrum=spectrum,
    window_size=5,
    mode='same'
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `window_size` (int): Window size (3-21, odd)
- `mode` (str, default='same'): Output size mode

**Returns**: np.ndarray - Smoothed spectrum

**Algorithm**: Uniform weighting within sliding window

#### apply_median_filter()

Apply median filter for noise removal.

```python
from functions.preprocess.kernel_denoise import apply_median_filter

filtered = apply_median_filter(
    spectrum=spectrum,
    kernel_size=5
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `kernel_size` (int): Kernel size (3-15, odd)

**Returns**: np.ndarray - Filtered spectrum

**Algorithm**: Median value within sliding window (robust to outliers)

#### apply_kernel_denoise()

Apply adaptive kernel denoising.

```python
from functions.preprocess.kernel_denoise import apply_kernel_denoise

denoised = apply_kernel_denoise(
    spectrum=spectrum,
    kernel_type='gaussian',
    bandwidth=2.0,
    iterations=1
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `kernel_type` (str): Kernel type ('gaussian', 'epanechnikov', 'uniform')
- `bandwidth` (float): Kernel bandwidth (0.5 - 5.0)
- `iterations` (int, default=1): Number of passes

**Returns**: np.ndarray - Denoised spectrum

**Algorithm**: Kernel regression with adaptive bandwidth

### Normalization

#### apply_vector_norm()

Apply vector (L2) normalization.

```python
from functions.preprocess.normalization import apply_vector_norm

normalized = apply_vector_norm(spectrum)

# Spectrum now has unit L2 norm: np.linalg.norm(normalized) == 1.0
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum

**Returns**: np.ndarray - Normalized spectrum

**Formula**: $x_{norm} = \frac{x}{\sqrt{\sum x_i^2}}$

**Use Case**: Makes spectra comparable regardless of absolute intensity

#### apply_minmax_norm()

Apply Min-Max normalization.

```python
from functions.preprocess.normalization import apply_minmax_norm

normalized = apply_minmax_norm(
    spectrum=spectrum,
    feature_range=(0, 1)
)

# Values now in [0, 1]: min=0, max=1
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `feature_range` (tuple, default=(0, 1)): Target range (min, max)

**Returns**: np.ndarray - Normalized spectrum

**Formula**: $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}} \times (max - min) + min$

**Use Case**: Scale to specific range, preserves distribution shape

#### apply_area_norm()

Apply area (sum) normalization.

```python
from functions.preprocess.normalization import apply_area_norm

normalized = apply_area_norm(spectrum)

# Total area is now 1.0: np.sum(normalized) == 1.0
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum

**Returns**: np.ndarray - Normalized spectrum

**Formula**: $x_{norm} = \frac{x}{\sum x_i}$

**Use Case**: Normalize by total signal intensity

#### apply_snv()

Apply Standard Normal Variate (SNV) normalization.

```python
from functions.preprocess.normalization import apply_snv

normalized = apply_snv(spectrum)

# Mean=0, Std=1: np.mean(normalized) ≈ 0, np.std(normalized) ≈ 1
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum

**Returns**: np.ndarray - Normalized spectrum

**Formula**: $x_{norm} = \frac{x - \bar{x}}{\sigma_x}$

**Use Case**: Remove multiplicative scatter effects, common in NIR/Raman

#### apply_msc()

Apply Multiplicative Scatter Correction (MSC).

```python
from functions.preprocess.normalization import apply_msc

# For single spectrum
normalized = apply_msc(
    spectrum=spectrum,
    reference=mean_spectrum
)

# For multiple spectra (calculates mean reference automatically)
from functions.preprocess.normalization import apply_msc_batch

normalized_spectra = apply_msc_batch(spectra)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `reference` (np.ndarray): Reference spectrum (typically mean of all spectra)

**Returns**: np.ndarray - Normalized spectrum

**Algorithm**: Linear regression to reference, then correction

**Formula**: 
- Fit: $x = a + b \cdot x_{ref}$
- Correct: $x_{corr} = \frac{x - a}{b}$

**Use Case**: Correct for scatter differences between samples

#### apply_quantile_norm()

Apply quantile normalization.

```python
from functions.preprocess.advanced_normalization import apply_quantile_norm

normalized = apply_quantile_norm(
    spectra=spectra,  # 2D array: (n_samples, n_features)
    n_quantiles=1000
)
```

**Parameters**:
- `spectra` (np.ndarray): Multiple spectra (2D array)
- `n_quantiles` (int, default=1000): Number of quantiles

**Returns**: np.ndarray - Normalized spectra

**Algorithm**: Rank-based transformation to common distribution

**Use Case**: Make distributions identical across samples

#### apply_pqn()

Apply Probabilistic Quotient Normalization (PQN).

```python
from functions.preprocess.advanced_normalization import apply_pqn

normalized = apply_pqn(
    spectrum=spectrum,
    reference=reference_spectrum
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `reference` (np.ndarray): Reference spectrum

**Returns**: np.ndarray - Normalized spectrum

**Algorithm**: Quotient to reference, then scale by median quotient

**Use Case**: Robust normalization for metabolomics/lipidomics

#### apply_rank_transform()

Apply rank transformation.

```python
from functions.preprocess.advanced_normalization import apply_rank_transform

ranked = apply_rank_transform(spectrum)

# Values replaced by ranks: 1, 2, 3, ..., n
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum

**Returns**: np.ndarray - Rank-transformed spectrum

**Use Case**: Non-parametric transformation, robust to outliers

### Derivatives

#### apply_first_derivative()

Calculate first derivative.

```python
from functions.preprocess.derivatives import apply_first_derivative

deriv1 = apply_first_derivative(
    spectrum=spectrum,
    method='savgol',
    window_length=11,
    polyorder=3
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `method` (str): Derivative method ('savgol', 'gradient', 'diff')
- `window_length` (int): Window size for Savitzky-Golay
- `polyorder` (int): Polynomial order for Savitzky-Golay

**Returns**: np.ndarray - First derivative

**Use Case**: Enhance peak resolution, remove baseline drift

#### apply_second_derivative()

Calculate second derivative.

```python
from functions.preprocess.derivatives import apply_second_derivative

deriv2 = apply_second_derivative(
    spectrum=spectrum,
    method='savgol',
    window_length=11,
    polyorder=3
)
```

**Parameters**: Same as `apply_first_derivative()`

**Returns**: np.ndarray - Second derivative

**Use Case**: Sharpen overlapping peaks, enhance fine structure

### Advanced Processing

#### apply_cdae()

Apply Convolutional Denoising Autoencoder (CDAE).

```python
from functions.preprocess.deep_learning import apply_cdae

denoised = apply_cdae(
    spectrum=spectrum,
    model_path='models/cdae_raman.pth',
    batch_size=32,
    device='cuda'  # or 'cpu'
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `model_path` (str): Path to trained CDAE model
- `batch_size` (int, default=32): Batch size for inference
- `device` (str, default='cpu'): Computation device

**Returns**: np.ndarray - Denoised spectrum

**Algorithm**: Deep learning-based denoising via autoencoder

**Requirements**: PyTorch, trained CDAE model

#### apply_background_subtraction()

Subtract background spectrum.

```python
from functions.preprocess.background_subtraction import apply_background_subtraction

corrected = apply_background_subtraction(
    spectrum=spectrum,
    background=background_spectrum,
    method='direct',
    scale_factor=1.0
)
```

**Parameters**:
- `spectrum` (np.ndarray): Sample spectrum
- `background` (np.ndarray): Background spectrum
- `method` (str): Subtraction method ('direct', 'scaled', 'weighted')
- `scale_factor` (float, default=1.0): Background scaling factor

**Returns**: np.ndarray - Background-corrected spectrum

#### apply_wavelength_calibration()

Apply wavelength/wavenumber calibration.

```python
from functions.preprocess.calibration import apply_wavelength_calibration

calibrated = apply_wavelength_calibration(
    wavenumbers=wavenumbers,
    reference_peaks=[1004, 1445, 1660],
    measured_peaks=[1002, 1443, 1658],
    method='linear'
)
```

**Parameters**:
- `wavenumbers` (np.ndarray): Original wavenumber axis
- `reference_peaks` (List[float]): Known reference peak positions
- `measured_peaks` (List[float]): Measured peak positions
- `method` (str): Calibration method ('linear', 'polynomial', 'spline')

**Returns**: np.ndarray - Calibrated wavenumber axis

#### apply_peak_ratio()

Calculate peak intensity ratios.

```python
from functions.preprocess.feature_engineering import apply_peak_ratio

ratio = apply_peak_ratio(
    spectrum=spectrum,
    wavenumbers=wavenumbers,
    band1_range=(1640, 1680),  # Amide I
    band2_range=(2840, 2900),  # CH2 stretch
    integration_method='trapz'
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `wavenumbers` (np.ndarray): Wavenumber axis
- `band1_range` (tuple): First band (min, max) wavenumbers
- `band2_range` (tuple): Second band (min, max) wavenumbers
- `integration_method` (str): Integration method ('trapz', 'simps', 'max')

**Returns**: float - Peak ratio (band1 / band2)

**Use Case**: Create discriminative features for classification

#### apply_wavelet_transform()

Apply wavelet transform for denoising or feature extraction.

```python
from functions.preprocess.feature_engineering import apply_wavelet_transform

coeffs, denoised = apply_wavelet_transform(
    spectrum=spectrum,
    wavelet='db4',
    level=5,
    threshold_method='soft',
    threshold_value='auto'
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `wavelet` (str): Wavelet type ('db4', 'sym5', 'coif3')
- `level` (int): Decomposition level (1-10)
- `threshold_method` (str): Thresholding ('soft', 'hard', 'garrote')
- `threshold_value` (str or float): Threshold ('auto' or specific value)

**Returns**: Tuple[list, np.ndarray] - (wavelet_coefficients, reconstructed_signal)

**Use Case**: Multi-resolution denoising, feature extraction

---

## Analysis Functions

### Dimensionality Reduction

#### apply_pca()

Perform Principal Component Analysis.

```python
from functions.ML.linear_regression import apply_pca

result = apply_pca(
    spectra=spectra,
    n_components=2,
    whiten=False,
    random_state=42
)

# Returns:
# {
#     'scores': np.array([[...], [...]]),  # PC scores
#     'loadings': np.array([[...], [...]]),  # PC loadings
#     'explained_variance': np.array([...]),
#     'explained_variance_ratio': np.array([...]),
#     'model': fitted_pca_model
# }
```

**Parameters**:
- `spectra` (np.ndarray): Input spectra (n_samples, n_features)
- `n_components` (int or float): Number of components or variance to retain
  - int: Specific number of PCs
  - float (0-1): Retain components explaining this variance fraction
- `whiten` (bool, default=False): Whiten components (unit variance)
- `random_state` (int, optional): Random seed for reproducibility

**Returns**: dict - PCA results

**Algorithm**: Singular Value Decomposition (SVD)

**Use Case**: Dimensionality reduction, visualization, noise reduction

#### apply_umap()

Perform UMAP (Uniform Manifold Approximation and Projection).

```python
from functions.ML.linear_regression import apply_umap

result = apply_umap(
    spectra=spectra,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    random_state=42
)

# Returns:
# {
#     'embedding': np.array([[...], [...]]),
#     'model': fitted_umap_model
# }
```

**Parameters**:
- `spectra` (np.ndarray): Input spectra
- `n_neighbors` (int, default=15): Neighborhood size (2-200)
- `min_dist` (float, default=0.1): Minimum distance between points (0.0-0.99)
- `n_components` (int, default=2): Embedding dimensions
- `metric` (str, default='euclidean'): Distance metric
- `random_state` (int, optional): Random seed

**Returns**: dict - UMAP results

**Use Case**: Non-linear dimensionality reduction, preserves both local and global structure

#### apply_tsne()

Perform t-SNE (t-Distributed Stochastic Neighbor Embedding).

```python
from functions.ML.linear_regression import apply_tsne

result = apply_tsne(
    spectra=spectra,
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)

# Returns:
# {
#     'embedding': np.array([[...], [...]]),
#     'model': fitted_tsne_model
# }
```

**Parameters**:
- `spectra` (np.ndarray): Input spectra
- `n_components` (int, default=2): Embedding dimensions
- `perplexity` (float, default=30): Perplexity parameter (5-50)
- `learning_rate` (float, default=200): Learning rate (10-1000)
- `n_iter` (int, default=1000): Maximum iterations (250-5000)
- `random_state` (int, optional): Random seed

**Returns**: dict - t-SNE results

**Use Case**: Visualization, preserves local structure (neighborhoods)

### Clustering

#### apply_kmeans()

Perform K-means clustering.

```python
from functions.ML.linear_regression import apply_kmeans

result = apply_kmeans(
    spectra=spectra,
    n_clusters=3,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)

# Returns:
# {
#     'labels': np.array([0, 0, 1, 1, 2, 2, ...]),
#     'centers': np.array([[...], [...], [...]]),
#     'inertia': 123.45,
#     'model': fitted_kmeans_model
# }
```

**Parameters**:
- `spectra` (np.ndarray): Input spectra
- `n_clusters` (int): Number of clusters
- `init` (str, default='k-means++'): Initialization method
- `n_init` (int, default=10): Number of initializations
- `max_iter` (int, default=300): Maximum iterations
- `random_state` (int, optional): Random seed

**Returns**: dict - Clustering results

**Use Case**: Partition data into K groups

#### apply_hierarchical_clustering()

Perform hierarchical clustering.

```python
from functions.ML.linear_regression import apply_hierarchical_clustering

result = apply_hierarchical_clustering(
    spectra=spectra,
    n_clusters=3,
    linkage='ward',
    metric='euclidean'
)

# Returns:
# {
#     'labels': np.array([0, 0, 1, 1, 2, 2, ...]),
#     'linkage_matrix': np.array([[...], [...]]),
#     'cophenetic_corr': 0.85,
#     'dendrogram': {...}
# }
```

**Parameters**:
- `spectra` (np.ndarray): Input spectra
- `n_clusters` (int, optional): Number of clusters (if None, returns full tree)
- `linkage` (str, default='ward'): Linkage criterion ('ward', 'complete', 'average', 'single')
- `metric` (str, default='euclidean'): Distance metric

**Returns**: dict - Clustering results with dendrogram

**Use Case**: Hierarchical grouping, visualize cluster relationships

#### apply_dbscan()

Perform DBSCAN (Density-Based Spatial Clustering).

```python
from functions.ML.linear_regression import apply_dbscan

result = apply_dbscan(
    spectra=spectra,
    eps=0.5,
    min_samples=5,
    metric='euclidean'
)

# Returns:
# {
#     'labels': np.array([0, 0, -1, 1, 1, ...]),  # -1 = noise
#     'core_samples': np.array([True, True, False, ...]),
#     'n_clusters': 2,
#     'n_noise': 15,
#     'model': fitted_dbscan_model
# }
```

**Parameters**:
- `spectra` (np.ndarray): Input spectra
- `eps` (float): Maximum distance between neighbors (0.1-10.0)
- `min_samples` (int): Minimum samples in neighborhood (3-20)
- `metric` (str, default='euclidean'): Distance metric

**Returns**: dict - Clustering results

**Use Case**: Find arbitrary-shaped clusters, detect outliers

### Statistical Tests

#### apply_ttest()

Perform t-test between two groups.

```python
from functions.utils import apply_ttest

result = apply_ttest(
    group1=control_spectra,
    group2=treatment_spectra,
    equal_var=False,  # Welch's t-test
    alternative='two-sided'
)

# Returns:
# {
#     'statistic': np.array([...]),  # t-statistic per feature
#     'pvalue': np.array([...]),     # p-value per feature
#     'significant_features': np.array([100, 234, 567, ...]),
#     'effect_size': np.array([...])  # Cohen's d per feature
# }
```

**Parameters**:
- `group1` (np.ndarray): First group spectra (n_samples, n_features)
- `group2` (np.ndarray): Second group spectra
- `equal_var` (bool, default=False): Assume equal variances
  - True: Student's t-test
  - False: Welch's t-test (recommended)
- `alternative` (str, default='two-sided'): Alternative hypothesis ('two-sided', 'less', 'greater')

**Returns**: dict - Test results

**Use Case**: Compare means between two groups at each wavenumber

#### apply_mannwhitneyu()

Perform Mann-Whitney U test (non-parametric t-test alternative).

```python
from functions.utils import apply_mannwhitneyu

result = apply_mannwhitneyu(
    group1=control_spectra,
    group2=treatment_spectra,
    alternative='two-sided'
)

# Returns similar structure to apply_ttest()
```

**Parameters**:
- `group1` (np.ndarray): First group spectra
- `group2` (np.ndarray): Second group spectra
- `alternative` (str, default='two-sided'): Alternative hypothesis

**Returns**: dict - Test results

**Use Case**: Compare distributions when data is non-normal

#### apply_anova()

Perform one-way ANOVA for multiple groups.

```python
from functions.utils import apply_anova

result = apply_anova(
    *groups,  # Variable number of group arrays
    post_hoc='tukey'
)

# Example:
result = apply_anova(
    control_spectra,
    treatment_a_spectra,
    treatment_b_spectra,
    post_hoc='tukey'
)

# Returns:
# {
#     'f_statistic': np.array([...]),
#     'pvalue': np.array([...]),
#     'significant_features': np.array([...]),
#     'effect_size': np.array([...]),  # eta-squared
#     'post_hoc_results': {...}  # if post_hoc specified
# }
```

**Parameters**:
- `*groups` (np.ndarray): Multiple group arrays
- `post_hoc` (str, optional): Post-hoc test ('tukey', 'bonferroni', 'holm')

**Returns**: dict - ANOVA results

**Use Case**: Compare means across multiple groups

#### apply_kruskal()

Perform Kruskal-Wallis test (non-parametric ANOVA alternative).

```python
from functions.utils import apply_kruskal

result = apply_kruskal(
    *groups,
    post_hoc='dunn'
)
```

**Parameters**:
- `*groups` (np.ndarray): Multiple group arrays
- `post_hoc` (str, optional): Post-hoc test ('dunn')

**Returns**: dict - Test results

**Use Case**: Compare distributions across multiple groups (non-parametric)

#### apply_correlation()

Calculate correlation matrix.

```python
from functions.utils import apply_correlation

result = apply_correlation(
    spectra=spectra,
    method='pearson'
)

# Returns:
# {
#     'correlation_matrix': np.array([[1.0, 0.8, ...], [0.8, 1.0, ...], ...]),
#     'pvalue_matrix': np.array([[0.0, 0.001, ...], [0.001, 0.0, ...], ...])
# }
```

**Parameters**:
- `spectra` (np.ndarray): Input spectra (n_samples, n_features)
- `method` (str): Correlation method ('pearson', 'spearman', 'kendall')

**Returns**: dict - Correlation results

**Use Case**: Find relationships between samples or features

#### apply_multiple_testing_correction()

Apply correction for multiple hypothesis testing.

```python
from functions.utils import apply_multiple_testing_correction

corrected_pvalues = apply_multiple_testing_correction(
    pvalues=raw_pvalues,
    method='fdr_bh',
    alpha=0.05
)

# Returns:
# {
#     'corrected_pvalues': np.array([...]),
#     'reject': np.array([True, False, True, ...]),
#     'significant_count': 42
# }
```

**Parameters**:
- `pvalues` (np.ndarray): Uncorrected p-values
- `method` (str): Correction method
  - `'bonferroni'`: Most conservative
  - `'holm'`: Step-down Bonferroni
  - `'fdr_bh'`: Benjamini-Hochberg FDR (recommended)
  - `'fdr_by'`: Benjamini-Yekutieli FDR
- `alpha` (float, default=0.05): Significance level

**Returns**: dict - Corrected results

**Use Case**: Control false positives when testing many hypotheses

---

## Machine Learning Functions

### Model Training

#### train_svm()

Train Support Vector Machine classifier.

```python
from functions.ML.svm import train_svm

result = train_svm(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    kernel='rbf',
    C=1.0,
    gamma='scale',
    cv=5
)

# Returns:
# {
#     'model': fitted_svm_model,
#     'train_score': 0.98,
#     'test_score': 0.92,
#     'cv_scores': np.array([0.90, 0.93, 0.91, 0.94, 0.89]),
#     'confusion_matrix': np.array([[45, 5], [3, 47]]),
#     'classification_report': {...},
#     'support_vectors': np.array([[...], ...])
# }
```

**Parameters**:
- `X_train` (np.ndarray): Training features (n_samples, n_features)
- `y_train` (np.ndarray): Training labels
- `X_test` (np.ndarray, optional): Test features
- `y_test` (np.ndarray, optional): Test labels
- `kernel` (str, default='rbf'): Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
- `C` (float, default=1.0): Regularization parameter (0.1-100)
- `gamma` (str or float, default='scale'): Kernel coefficient
- `cv` (int, default=5): Cross-validation folds

**Returns**: dict - Training results

**Use Case**: Binary or multi-class classification with kernel trick

#### train_random_forest()

Train Random Forest classifier.

```python
from functions.ML.random_forest import train_random_forest

result = train_random_forest(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    cv=5,
    random_state=42
)

# Returns similar structure to train_svm() plus:
# {
#     ...,
#     'feature_importances': np.array([...]),
#     'oob_score': 0.91  # if oob_score=True
# }
```

**Parameters**:
- `X_train`, `y_train`, `X_test`, `y_test`: Data arrays
- `n_estimators` (int, default=100): Number of trees (100-1000)
- `max_depth` (int, optional): Maximum tree depth
- `min_samples_split` (int, default=2): Minimum samples to split (2-20)
- `min_samples_leaf` (int, default=1): Minimum samples per leaf
- `max_features` (str or int, default='sqrt'): Features per split
- `cv` (int, default=5): Cross-validation folds
- `random_state` (int, optional): Random seed

**Returns**: dict - Training results

**Use Case**: Robust ensemble classifier, handles non-linearity

#### train_xgboost()

Train XGBoost classifier.

```python
from functions.ML.xgboost import train_xgboost

result = train_xgboost(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    cv=5,
    early_stopping_rounds=10,
    random_state=42
)

# Returns similar structure plus:
# {
#     ...,
#     'feature_importances': {
#         'gain': np.array([...]),
#         'weight': np.array([...]),
#         'cover': np.array([...])
#     },
#     'best_iteration': 87
# }
```

**Parameters**:
- `X_train`, `y_train`, `X_test`, `y_test`: Data arrays
- `n_estimators` (int, default=100): Number of boosting rounds (100-1000)
- `learning_rate` (float, default=0.1): Step size shrinkage (0.01-0.3)
- `max_depth` (int, default=6): Tree depth (3-10)
- `subsample` (float, default=0.8): Sample fraction (0.5-1.0)
- `colsample_bytree` (float, default=0.8): Feature fraction (0.5-1.0)
- `gamma` (float, default=0): Minimum split loss (0-10)
- `reg_lambda` (float, default=1): L2 regularization (0-10)
- `reg_alpha` (float, default=0): L1 regularization (0-10)
- `cv` (int, default=5): Cross-validation folds
- `early_stopping_rounds` (int, optional): Stop if no improvement
- `random_state` (int, optional): Random seed

**Returns**: dict - Training results

**Use Case**: State-of-the-art gradient boosting, competitions

#### train_logistic_regression()

Train Logistic Regression classifier.

```python
from functions.ML.logistic_regression import train_logistic_regression

result = train_logistic_regression(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    cv=5,
    random_state=42
)

# Returns similar structure plus:
# {
#     ...,
#     'coefficients': np.array([[...], ...]),
#     'intercept': np.array([...]),
#     'odds_ratios': np.array([...])
# }
```

**Parameters**:
- `X_train`, `y_train`, `X_test`, `y_test`: Data arrays
- `C` (float, default=1.0): Inverse regularization strength (0.01-100)
- `penalty` (str, default='l2'): Regularization type ('l1', 'l2', 'elasticnet', 'none')
- `solver` (str, default='lbfgs'): Optimization algorithm
- `max_iter` (int, default=100): Maximum iterations (100-10000)
- `cv` (int, default=5): Cross-validation folds
- `random_state` (int, optional): Random seed

**Returns**: dict - Training results

**Use Case**: Interpretable linear classifier with probabilities

#### train_mlp()

Train Multi-Layer Perceptron (Neural Network) classifier.

```python
from functions.ML.logistic_regression import train_mlp

result = train_mlp(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    hidden_layer_sizes=(100, 50),
    activation='relu',
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    cv=5,
    random_state=42
)

# Returns similar structure plus:
# {
#     ...,
#     'loss_curve': [0.8, 0.6, 0.5, 0.4, ...],
#     'best_loss': 0.35
# }
```

**Parameters**:
- `X_train`, `y_train`, `X_test`, `y_test`: Data arrays
- `hidden_layer_sizes` (tuple, default=(100,)): Neurons per hidden layer
- `activation` (str, default='relu'): Activation function ('relu', 'tanh', 'logistic')
- `alpha` (float, default=0.0001): L2 regularization (0.0001-0.01)
- `learning_rate_init` (float, default=0.001): Initial learning rate (0.0001-0.01)
- `max_iter` (int, default=200): Maximum epochs (200-1000)
- `early_stopping` (bool, default=True): Stop if validation score doesn't improve
- `cv` (int, default=5): Cross-validation folds
- `random_state` (int, optional): Random seed

**Returns**: dict - Training results

**Use Case**: Flexible non-linear classifier, deep learning

### Model Evaluation

#### evaluate_model()

Comprehensive model evaluation.

```python
from functions.ML.utils import evaluate_model

metrics = evaluate_model(
    model=trained_model,
    X_test=X_test,
    y_test=y_test,
    average='weighted'
)

# Returns:
# {
#     'accuracy': 0.92,
#     'precision': 0.91,
#     'recall': 0.93,
#     'f1_score': 0.92,
#     'roc_auc': 0.95,
#     'confusion_matrix': np.array([[45, 5], [3, 47]]),
#     'classification_report': {
#         'Control': {'precision': 0.94, 'recall': 0.90, 'f1-score': 0.92},
#         'Treatment': {'precision': 0.90, 'recall': 0.94, 'f1-score': 0.92}
#     }
# }
```

**Parameters**:
- `model`: Trained scikit-learn model
- `X_test` (np.ndarray): Test features
- `y_test` (np.ndarray): True labels
- `average` (str, default='weighted'): Averaging method for multi-class

**Returns**: dict - Evaluation metrics

#### plot_confusion_matrix()

Plot confusion matrix heatmap.

```python
from functions.visualization.plots import plot_confusion_matrix

fig = plot_confusion_matrix(
    y_true=y_test,
    y_pred=predictions,
    class_names=['Control', 'Treatment'],
    normalize=True,
    cmap='Blues'
)
```

**Parameters**:
- `y_true` (np.ndarray): True labels
- `y_pred` (np.ndarray): Predicted labels
- `class_names` (List[str], optional): Class labels
- `normalize` (bool, default=False): Normalize by row (true labels)
- `cmap` (str, default='Blues'): Colormap

**Returns**: matplotlib.figure.Figure

#### plot_roc_curve()

Plot ROC curve.

```python
from functions.visualization.plots import plot_roc_curve

fig = plot_roc_curve(
    y_true=y_test,
    y_scores=y_proba,
    class_names=['Control', 'Treatment']
)
```

**Parameters**:
- `y_true` (np.ndarray): True labels
- `y_scores` (np.ndarray): Predicted probabilities
- `class_names` (List[str], optional): Class labels

**Returns**: matplotlib.figure.Figure

**Displays**: ROC curve with AUC score

#### plot_learning_curve()

Plot learning curve (train/validation score vs. training size).

```python
from functions.visualization.plots import plot_learning_curve

fig = plot_learning_curve(
    model=model,
    X=X_train,
    y=y_train,
    cv=5,
    scoring='accuracy'
)
```

**Parameters**:
- `model`: Scikit-learn estimator
- `X` (np.ndarray): Training features
- `y` (np.ndarray): Training labels
- `cv` (int, default=5): Cross-validation folds
- `scoring` (str, default='accuracy'): Metric to plot

**Returns**: matplotlib.figure.Figure

**Use Case**: Diagnose overfitting/underfitting

### Feature Importance

#### get_feature_importance()

Extract feature importance from model.

```python
from functions.ML.utils import get_feature_importance

importance = get_feature_importance(
    model=trained_model,
    method='default',
    wavenumbers=wavenumbers
)

# Returns:
# {
#     'importance': np.array([...]),
#     'feature_indices': np.array([...]),
#     'wavenumbers': np.array([...]),
#     'top_features': {
#         'indices': [234, 567, 890],
#         'wavenumbers': [1004, 1445, 1660],
#         'importance': [0.15, 0.12, 0.10]
#     }
# }
```

**Parameters**:
- `model`: Trained model
- `method` (str, default='default'): Importance method
  - `'default'`: Use model's native importance
  - `'permutation'`: Permutation importance (model-agnostic)
  - `'shap'`: SHAP values (requires shap library)
- `wavenumbers` (np.ndarray, optional): Wavenumber axis

**Returns**: dict - Feature importance results

#### plot_feature_importance()

Plot feature importance as bar plot or spectrum overlay.

```python
from functions.visualization.plots import plot_feature_importance

fig = plot_feature_importance(
    importance=importance_array,
    wavenumbers=wavenumbers,
    top_n=20,
    plot_type='spectrum'
)
```

**Parameters**:
- `importance` (np.ndarray): Feature importance values
- `wavenumbers` (np.ndarray): Wavenumber axis
- `top_n` (int, default=20): Number of top features to highlight
- `plot_type` (str, default='bar'): Plot type ('bar', 'spectrum', 'both')

**Returns**: matplotlib.figure.Figure

#### calculate_permutation_importance()

Calculate permutation feature importance (model-agnostic).

```python
from functions.ML.utils import calculate_permutation_importance

importance = calculate_permutation_importance(
    model=trained_model,
    X=X_test,
    y=y_test,
    n_repeats=10,
    random_state=42,
    scoring='accuracy'
)

# Returns:
# {
#     'importances_mean': np.array([...]),
#     'importances_std': np.array([...]),
#     'importances': np.array([[...], [...], ...])  # shape: (n_features, n_repeats)
# }
```

**Parameters**:
- `model`: Trained model
- `X` (np.ndarray): Test features
- `y` (np.ndarray): True labels
- `n_repeats` (int, default=10): Number of permutations
- `random_state` (int, optional): Random seed
- `scoring` (str or callable, default='accuracy'): Metric

**Returns**: dict - Permutation importance results

**Algorithm**: Shuffle each feature and measure score drop

---

## Utility Functions

### functions/utils.py

#### validate_spectra_data()

Validate spectral data structure.

```python
from functions.utils import validate_spectra_data

is_valid, errors = validate_spectra_data(data)

if not is_valid:
    for error in errors:
        print(f"Validation error: {error}")
```

**Parameters**:
- `data` (dict): Data dictionary

**Returns**: Tuple[bool, List[str]] - (is_valid, error_messages)

**Checks**:
- Required keys present
- Correct data types
- Consistent dimensions
- No NaN/Inf values
- Valid wavenumber range

#### generate_mock_spectra()

Generate synthetic spectra for testing.

```python
from functions.utils import generate_mock_spectra

data = generate_mock_spectra(
    n_samples=100,
    n_features=1000,
    n_groups=3,
    noise_level=0.05,
    random_state=42
)

# Returns standard data dictionary
```

**Parameters**:
- `n_samples` (int): Number of spectra
- `n_features` (int): Number of wavenumbers
- `n_groups` (int): Number of groups
- `noise_level` (float): Noise standard deviation (0-1)
- `random_state` (int, optional): Random seed

**Returns**: dict - Synthetic spectral data

**Use Case**: Testing, prototyping, demonstrations

#### calculate_snr()

Calculate Signal-to-Noise Ratio.

```python
from functions.utils import calculate_snr

snr = calculate_snr(
    spectrum=spectrum,
    signal_range=(1640, 1680),  # Amide I region
    noise_range=(1800, 2000)    # Baseline region
)

print(f"SNR: {snr:.2f} dB")
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `signal_range` (tuple): (min, max) wavenumbers for signal
- `noise_range` (tuple): (min, max) wavenumbers for noise estimation

**Returns**: float - SNR in decibels (dB)

**Formula**: $SNR = 20 \log_{10}(\frac{\mu_{signal}}{\sigma_{noise}})$

#### find_peaks()

Find peaks in spectrum.

```python
from functions.utils import find_peaks

peaks = find_peaks(
    spectrum=spectrum,
    wavenumbers=wavenumbers,
    prominence=0.05,
    distance=10,
    width=5
)

# Returns:
# {
#     'peak_indices': np.array([...]),
#     'peak_wavenumbers': np.array([...]),
#     'peak_intensities': np.array([...]),
#     'properties': {
#         'prominences': np.array([...]),
#         'widths': np.array([...]),
#         'left_bases': np.array([...]),
#         'right_bases': np.array([...])
#     }
# }
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `wavenumbers` (np.ndarray): Wavenumber axis
- `prominence` (float, optional): Minimum peak prominence
- `distance` (int, optional): Minimum distance between peaks
- `width` (int or tuple, optional): Required peak width
- `height` (float or tuple, optional): Required peak height

**Returns**: dict - Peak information

**Use Case**: Peak identification, peak tracking

#### integrate_region()

Integrate spectrum over wavenumber range.

```python
from functions.utils import integrate_region

area = integrate_region(
    spectrum=spectrum,
    wavenumbers=wavenumbers,
    region=(1640, 1680),
    method='trapz'
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `wavenumbers` (np.ndarray): Wavenumber axis
- `region` (tuple): (min, max) wavenumbers
- `method` (str, default='trapz'): Integration method ('trapz', 'simps')

**Returns**: float - Integrated area

**Use Case**: Quantification, peak ratios

#### resample_spectrum()

Resample spectrum to new wavenumber axis.

```python
from functions.utils import resample_spectrum

resampled = resample_spectrum(
    spectrum=spectrum,
    old_wavenumbers=old_wn,
    new_wavenumbers=new_wn,
    method='linear'
)
```

**Parameters**:
- `spectrum` (np.ndarray): Input spectrum
- `old_wavenumbers` (np.ndarray): Original wavenumber axis
- `new_wavenumbers` (np.ndarray): Target wavenumber axis
- `method` (str, default='linear'): Interpolation method ('linear', 'cubic', 'spline')

**Returns**: np.ndarray - Resampled spectrum

**Use Case**: Align spectra with different wavenumber axes

#### split_train_test()

Split data into train/test sets with stratification.

```python
from functions.utils import split_train_test

X_train, X_test, y_train, y_test = split_train_test(
    X=spectra,
    y=labels,
    test_size=0.2,
    stratify=True,
    random_state=42
)
```

**Parameters**:
- `X` (np.ndarray): Features
- `y` (np.ndarray): Labels
- `test_size` (float, default=0.2): Test set fraction
- `stratify` (bool, default=True): Preserve class distribution
- `random_state` (int, optional): Random seed

**Returns**: Tuple[np.ndarray, ...] - X_train, X_test, y_train, y_test

#### export_pipeline()

Export preprocessing pipeline to JSON.

```python
from functions.utils import export_pipeline

export_pipeline(
    pipeline=[
        {'method': 'asls', 'params': {'lambda': 1e5, 'p': 0.01}},
        {'method': 'savgol', 'params': {'window_length': 11, 'polyorder': 3}},
        {'method': 'vector_norm', 'params': {}}
    ],
    filepath='pipelines/my_pipeline.json',
    metadata={
        'description': 'Standard preprocessing for tissue samples',
        'author': 'Researcher Name',
        'created': '2026-01-24'
    }
)
```

**Parameters**:
- `pipeline` (List[dict]): Pipeline steps
- `filepath` (str): Output file path
- `metadata` (dict, optional): Additional metadata

**Side Effects**: Creates JSON file

#### load_pipeline()

Load preprocessing pipeline from JSON.

```python
from functions.utils import load_pipeline

pipeline, metadata = load_pipeline('pipelines/my_pipeline.json')

# pipeline: [{'method': 'asls', 'params': {...}}, ...]
# metadata: {'description': '...', 'author': '...', 'created': '...'}
```

**Parameters**:
- `filepath` (str): Pipeline file path

**Returns**: Tuple[List[dict], dict] - (pipeline_steps, metadata)

---

## Pipeline Execution

### execute_preprocessing_pipeline()

Execute complete preprocessing pipeline.

```python
from functions.preprocess.pipeline import execute_preprocessing_pipeline

result = execute_preprocessing_pipeline(
    spectra=raw_spectra,
    pipeline=[
        {'method': 'asls', 'params': {'lambda': 1e5, 'p': 0.01}},
        {'method': 'savgol', 'params': {'window_length': 11, 'polyorder': 3}},
        {'method': 'vector_norm', 'params': {}}
    ],
    progress_callback=lambda i, total, msg: print(f"{i}/{total}: {msg}")
)

# Returns:
# {
#     'preprocessed_spectra': np.array([[...], [...]]),
#     'success': True,
#     'execution_time': 2.34,
#     'steps_applied': ['asls', 'savgol', 'vector_norm'],
#     'intermediate_results': {...}  # if return_intermediate=True
# }
```

**Parameters**:
- `spectra` (np.ndarray): Input spectra (n_samples, n_features)
- `pipeline` (List[dict]): Pipeline steps
- `progress_callback` (callable, optional): Progress function(current, total, message)
- `return_intermediate` (bool, default=False): Return results after each step
- `parallel` (bool, default=False): Process spectra in parallel
- `n_jobs` (int, default=-1): Number of parallel jobs (-1 = all CPUs)

**Returns**: dict - Processing results

**Raises**:
- `PreprocessingError`: If any step fails
- `ValueError`: If pipeline configuration is invalid

---

## Best Practices

### Function Usage

```python
# Good: Clear, explicit parameters
corrected = apply_asls(
    spectrum=spectrum,
    lambda_=1e5,
    p=0.01,
    max_iter=10
)

# Avoid: Relying on defaults without understanding
corrected = apply_asls(spectrum)  # May not be appropriate for your data
```

### Error Handling

```python
from functions.preprocess.baseline import apply_asls
from functions._utils_ import PreprocessingError

try:
    corrected = apply_asls(spectrum, lambda_=1e5, p=0.01)
except PreprocessingError as e:
    print(f"Preprocessing failed: {e}")
    # Handle error or use fallback method
except ValueError as e:
    print(f"Invalid parameters: {e}")
```

### Pipeline Design

```python
# Good: Logical order
pipeline = [
    {'method': 'asls', 'params': {...}},      # 1. Baseline correction
    {'method': 'savgol', 'params': {...}},    # 2. Smoothing
    {'method': 'vector_norm', 'params': {}}   # 3. Normalization
]

# Avoid: Illogical order
pipeline = [
    {'method': 'vector_norm', 'params': {}},  # Normalizing before baseline correction
    {'method': 'asls', 'params': {...}}       # May normalize baseline as well
]
```

### Performance Optimization

```python
# For large datasets, use parallel processing
result = execute_preprocessing_pipeline(
    spectra=large_dataset,
    pipeline=my_pipeline,
    parallel=True,
    n_jobs=-1  # Use all CPU cores
)

# For repeated operations, cache intermediate results
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_preprocessing(spectrum_tuple, pipeline_hash):
    spectrum = np.array(spectrum_tuple)
    return apply_preprocessing(spectrum, pipeline)
```

---

## See Also

- [Core API](core.md) - Core application modules
- [Pages API](pages.md) - Application pages
- [Components API](components.md) - UI components
- [Analysis Methods](../analysis-methods/index.md) - Detailed method documentation
- [Best Practices](../user-guide/best-practices.md) - Guidelines and recommendations

---

**Last Updated**: 2026-01-24
