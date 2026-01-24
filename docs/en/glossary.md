# Glossary

## A

**Amide I Band**
: Raman peak around 1650 cm⁻¹ corresponding to C=O stretching vibrations in protein backbones. Strong marker for protein content.

**Amide II Band**
: Raman peak around 1550 cm⁻¹ from N-H bending and C-N stretching in proteins. Used for protein structure analysis.

**Asymmetric Least Squares (AsLS)**
: Baseline correction algorithm that fits asymmetric polynomials, giving more weight to points below the spectrum. Fast and effective for smooth baselines.

**API**
: Application Programming Interface - the set of functions and classes that developers can use to extend or interact with the application programmatically.

---

## B

**Baseline Correction**
: Preprocessing step that removes additive background fluorescence from Raman spectra, revealing the true Raman peaks.

**Batch Processing**
: Processing multiple datasets or spectra simultaneously with the same pipeline or analysis method.

**Biomarker**
: A measurable indicator (e.g., specific Raman peak or ratio) that characterizes biological state or disease condition.

---

## C

**Calibration**
: Process of correcting spectral axis (wavenumber) or intensity axis using known standards to ensure measurement accuracy.

**Class Imbalance**
: When one class has many more samples than another (e.g., 90% healthy, 10% disease), requiring special handling in machine learning.

**Confusion Matrix**
: Table showing model predictions vs actual labels, used to evaluate classification performance (True Positive, False Positive, etc.).

**Cosmic Ray**
: Sharp intensity spike caused by high-energy cosmic ray hitting the detector. Must be removed before analysis.

**Cross-Validation**
: Technique for evaluating model performance by splitting data into training/test sets multiple times and averaging results.

---

## D

**Data Leakage**
: When information from test set inadvertently influences training, causing overoptimistic performance estimates. Critical to avoid.

**Dataset**
: Collection of Raman spectra imported and managed together. Can contain multiple groups.

**Dimensionality Reduction**
: Techniques (PCA, UMAP, t-SNE) that reduce high-dimensional spectral data (1000+ wavenumbers) to 2-3 dimensions for visualization.

---

## E

**Effect Size**
: Magnitude of difference between groups, independent of sample size. Cohen's d is common effect size measure.

**Endmember**
: Pure component spectrum in spectral unmixing (MCR-ALS). Mixture spectra are combinations of endmembers.

---

## F

**False Discovery Rate (FDR)**
: Expected proportion of false positives among all positive results. FDR correction (Benjamini-Hochberg) controls this rate.

**Feature**
: In machine learning context, each wavenumber intensity value is a feature. Raman spectra have ~1000-2000 features.

**Feature Engineering**
: Creating new features from raw data (e.g., peak ratios, derivatives) to improve model performance.

**Feature Importance**
: Measure of how much each feature (wavenumber) contributes to model predictions. SHAP and permutation importance are common methods.

**Fluorescence**
: Broad background signal in Raman spectra caused by sample autofluorescence. Removed by baseline correction.

---

## G

**Group**
: User-defined collection of spectra (e.g., "Healthy", "Disease") used for comparative analysis and machine learning.

**GroupKFold**
: Cross-validation strategy ensuring all spectra from same patient/sample stay in same fold, preventing data leakage.

---

## H

**Hyperparameter**
: Model parameter set by user (not learned from data), such as number of trees in Random Forest or C in SVM.

**Hyperparameter Tuning**
: Process of finding optimal hyperparameter values using grid search, random search, or optimization algorithms.

---

## I

**Intensity**
: Raman scattering signal strength, proportional to molecular concentration and scattering cross-section.

**Interpolation**
: Process of estimating spectrum values at new wavenumber points based on existing data. Used to align spectra with different sampling.

---

## K

**K-Fold Cross-Validation**
: Splitting data into K parts, training on K-1 parts and testing on remaining part, repeated K times.

---

## L

**Lambda (λ)**
: Smoothness parameter in baseline correction algorithms. Higher λ = smoother baseline.

**Leave-One-Patient-Out Cross-Validation (LOPOCV)**
: Cross-validation where each patient's data is test set once, ensuring patient-level separation.

**Loading**
: In PCA, the contribution of each original variable (wavenumber) to a principal component. Loadings plot shows which peaks drive PCs.

---

## M

**Machine Learning (ML)**
: Using algorithms to learn patterns from data and make predictions on new data without explicit programming.

**MCR-ALS**
: Multivariate Curve Resolution - Alternating Least Squares. Method for decomposing mixture spectra into pure component spectra.

**MGUS**
: Monoclonal Gammopathy of Undetermined Significance. Pre-cancerous condition that may progress to multiple myeloma (MM).

**Multiple Testing Correction**
: Statistical adjustment for performing many hypothesis tests simultaneously, controlling false positive rate.

---

## N

**Normalization**
: Scaling spectra to remove multiplicative intensity variations, enabling fair comparison. Common methods: Vector, SNV, Min-Max, Area.

**NumPy**
: Python library for numerical computing, providing arrays and mathematical functions. Core dependency for this application.

---

## O

**Outlier**
: Spectrum significantly different from others, possibly due to measurement error or unusual sample. Should be identified and often removed.

**Overfitting**
: When model learns training data too well, including noise, causing poor performance on new data.

---

## P

**P-value**
: Probability of observing data as extreme as measured, assuming null hypothesis is true. p < 0.05 traditionally considered significant.

**PCA (Principal Component Analysis)**
: Unsupervised dimensionality reduction finding directions of maximum variance. First step for most analyses.

**Peak**
: Local maximum in Raman spectrum corresponding to specific molecular vibration.

**Pipeline**
: Sequence of preprocessing steps applied in order (e.g., Baseline → Smooth → Normalize).

**PLS-DA (Partial Least Squares Discriminant Analysis)**
: Supervised dimensionality reduction maximizing separation between known groups.

**Preprocessing**
: Data transformation steps applied before analysis (baseline correction, smoothing, normalization, etc.).

---

## Q

**Quality Control (QC)**
: Procedures ensuring data quality, including outlier detection, cosmic ray removal, and baseline verification.

**Quantile Normalization**
: Advanced normalization making intensity distributions identical across spectra.

---

## R

**Raman Scattering**
: Inelastic scattering of photons by molecules, providing fingerprint of molecular structure.

**Raman Shift**
: Energy difference between incident and scattered photons, measured in wavenumbers (cm⁻¹).

**Random Forest (RF)**
: Ensemble machine learning algorithm using multiple decision trees. Robust and interpretable.

**Regularization**
: Adding penalty to model complexity to prevent overfitting. L1 (Lasso) and L2 (Ridge) are common types.

**ROC Curve**
: Receiver Operating Characteristic curve plotting True Positive Rate vs False Positive Rate. ROC-AUC measures classification performance.

---

## S

**Savitzky-Golay Filter**
: Smoothing method fitting polynomials to local windows. Preserves peak shapes better than simple averaging.

**Scree Plot**
: Plot of explained variance vs principal component number. Used to select number of PCs to keep.

**SHAP (SHapley Additive exPlanations)**
: Method for interpreting machine learning models by assigning importance value to each feature for each prediction.

**Smoothing**
: Reducing noise by averaging neighboring points. Savitzky-Golay and Gaussian are common methods.

**SNV (Standard Normal Variate)**
: Normalization method centering and scaling each spectrum independently. Robust for biological samples.

**Spectrum (plural: Spectra)**
: Plot of Raman intensity vs wavenumber for a single measurement.

**Stratified Sampling**
: Splitting data while maintaining class proportions in train and test sets. Important for imbalanced data.

**SVM (Support Vector Machine)**
: Machine learning algorithm finding optimal separating hyperplane between classes. Effective for high-dimensional data.

---

## T

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
: Non-linear dimensionality reduction emphasizing local structure. Good for visualizing clusters.

**Test Set**
: Data held out during training, used only for final performance evaluation.

**Training Set**
: Data used to train machine learning model. Model learns patterns from training set.

---

## U

**UMAP (Uniform Manifold Approximation and Projection)**
: Non-linear dimensionality reduction preserving both local and global structure. Faster than t-SNE.

**Underfitting**
: When model is too simple to capture data patterns, causing poor performance on both training and test data.

---

## V

**Validation Set**
: Data used during training for hyperparameter tuning and model selection (separate from test set).

**Vector Normalization**
: Scaling each spectrum to unit length (Euclidean norm = 1). Most common normalization for Raman data.

---

## W

**Wavenumber**
: Unit for Raman shift, measured in cm⁻¹ (inverse centimeters). Proportional to vibrational energy.

**Whittaker Smoothing**
: Smoothing method using penalized least squares. Controlled by lambda parameter.

---

## X

**XGBoost (eXtreme Gradient Boosting)**
: Advanced gradient boosting machine learning algorithm. Often achieves highest accuracy but requires careful tuning.

---

## Japanese Terms / 日本語用語

**未病 (Mibyō)**
: Pre-disease state - condition before full disease manifestation. Focus of early detection research.

**臨床光情報工学研究室 (Rinsho Hikari Jōhō Kōgaku Kenkyūshitsu)**
: Laboratory for Clinical Photonics and Information Engineering at University of Toyama.

---

## Abbreviations

| Abbreviation | Full Name                                          |
| ------------ | -------------------------------------------------- |
| AI           | Artificial Intelligence                            |
| ALS          | Alternating Least Squares                          |
| ANOVA        | Analysis of Variance                               |
| API          | Application Programming Interface                  |
| AsLS         | Asymmetric Least Squares                           |
| AUC          | Area Under Curve                                   |
| CDAE         | Convolutional Denoising Autoencoder                |
| CI           | Confidence Interval                                |
| CSV          | Comma-Separated Values                             |
| DPI          | Dots Per Inch                                      |
| EM           | Expectation-Maximization                           |
| FABC         | Fixed-Anchor Baseline Correction                   |
| FDR          | False Discovery Rate                               |
| FN           | False Negative                                     |
| FP           | False Positive                                     |
| GUI          | Graphical User Interface                           |
| ICA          | Independent Component Analysis                     |
| KNN          | K-Nearest Neighbors                                |
| LOPOCV       | Leave-One-Patient-Out Cross-Validation             |
| MAT          | MATLAB File Format                                 |
| MCR          | Multivariate Curve Resolution                      |
| MGUS         | Monoclonal Gammopathy of Undetermined Significance |
| ML           | Machine Learning                                   |
| MM           | Multiple Myeloma                                   |
| MSC          | Multiplicative Scatter Correction                  |
| NMF          | Non-negative Matrix Factorization                  |
| PC           | Principal Component                                |
| PCA          | Principal Component Analysis                       |
| PLS          | Partial Least Squares                              |
| PLS-DA       | Partial Least Squares Discriminant Analysis        |
| PQN          | Probabilistic Quotient Normalization               |
| QC           | Quality Control                                    |
| RF           | Random Forest                                      |
| ROC          | Receiver Operating Characteristic                  |
| SMOTE        | Synthetic Minority Over-sampling Technique         |
| SNV          | Standard Normal Variate                            |
| SVM          | Support Vector Machine                             |
| t-SNE        | t-Distributed Stochastic Neighbor Embedding        |
| TN           | True Negative                                      |
| TP           | True Positive                                      |
| UMAP         | Uniform Manifold Approximation and Projection      |
| UV           | uv package manager                                 |

---

## Contributing

Found a term that should be added? 

1. Fork the [repository](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application)
2. Edit `docs/glossary.md`
3. Add term in alphabetical order with clear definition
4. Submit pull request

Please include:
- **Term** in bold
- Clear, concise definition
- Related terms or examples where helpful
