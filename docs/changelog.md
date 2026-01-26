# Changelog

All notable changes to the Raman Spectroscopy Analysis Application will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation
- Comprehensive ReadTheDocs documentation system
- User guides for all major features
- Analysis methods reference documentation
- API documentation for developers
- Japanese translation (in progress)

## [1.0.0-alpha] - 2026-01-24

### Added

#### Core Features
- Complete desktop application with PySide6/Qt6 interface
- Multi-language support (English, Japanese)
- Project management system with workspace organization
- Data package management for multiple datasets
- Group-based sample organization

#### Preprocessing (40+ Methods)
- **Baseline Correction**: AsLS, AirPLS, Polynomial, Whittaker, FABC, Butterworth High-Pass
- **Smoothing**: Savitzky-Golay, Gaussian, Moving Average, Median Filter
- **Normalization**: Vector, Min-Max, Area, SNV, MSC, Quantile, PQN, Rank Transform
- **Derivatives**: 1st and 2nd order Savitzky-Golay
- **Feature Engineering**: Peak Ratio, Wavelet Transform
- **Advanced**: Convolutional Denoising Autoencoder (CDAE)
- **Pipeline System**: Save, load, and share preprocessing pipelines
- **Real-time Preview**: See effects before applying

#### Analysis Methods
- **Exploratory**:
  - Principal Component Analysis (PCA) with loadings and scree plots
  - UMAP for non-linear dimensionality reduction
  - t-SNE for cluster visualization
  - Hierarchical Clustering with dendrograms
  - K-means Clustering with elbow method
- **Statistical**:
  - Pairwise tests (t-test, Mann-Whitney U)
  - Multi-group comparisons (ANOVA, Kruskal-Wallis)
  - Correlation analysis (Pearson, Spearman, Kendall)
  - Band ratio analysis with customizable ranges
  - Peak detection and identification
- **Visualization**:
  - Interactive heatmaps
  - Waterfall plots
  - Overlaid spectra with group coloring
  - Peak scatter plots
  - Correlation matrices

#### Machine Learning
- **Algorithms**:
  - Support Vector Machine (SVM)
  - Random Forest (RF)
  - XGBoost
  - Logistic Regression
  - Multi-Layer Perceptron (MLP)
  - PLS-DA (planned)
- **Validation**:
  - GroupKFold cross-validation (patient-level splitting)
  - Leave-One-Patient-Out (LOPOCV)
  - Stratified K-Fold
  - Hold-out test sets
- **Evaluation**:
  - ROC curves with AUC scores
  - Confusion matrices
  - Classification reports (precision, recall, F1)
  - Calibration curves
- **Interpretability**:
  - SHAP values for feature importance
  - Permutation importance
  - Feature importance plots mapped to wavenumbers
- **Export**: Save trained models in pickle or ONNX format

#### Build System
- **Windows Portable**: Single executable (~375 MB) with all dependencies
- **Windows Installer**: NSIS-based professional installer
- **Build Scripts**: Automated PowerShell build system
- **Testing**: Comprehensive test suite for executables

### Fixed

#### January 2026
- **Analysis Page Stability** (2026-01-23):
  - Two-stage stop mechanism for safe cancellation
  - Improved outlier detection performance with PCA-reduced space
  - Fixed correlation heatmap tick clipping
  - Registered bundled fonts for proper EN/JA plot rendering
  - Fixed ML dropdown black popup styling
  - Improved band ratio plot embedding with proper PathPatch preservation

- **ML Evaluation Enhancements** (2026-01-22):
  - Added comprehensive evaluation summary tab
  - Implemented data leakage warnings
  - Dataset-level performance metrics

- **Grouped Mode Analysis** (2026-01-21):
  - Fixed PCA plots in grouped mode with proper tab organization
  - Improved runtime performance with shared component sync

- **ML UI Improvements** (2026-01-20):
  - Major ML page UI refactor for better usability
  - Enhanced group management with drag-and-drop
  - Improved i18n coverage across ML interface
  - Optimized performance for large datasets

#### October 2025
- **Parameter Type Validation** (2025-10-15):
  - Fixed FABC baseline correction integer conversion
  - Implemented two-layer type validation system
  - All 40 preprocessing methods validated (100% pass rate)
  - Zero breaking changes, full backwards compatibility

- **Preprocessing UI/UX** (2025-10-08):
  - Fixed pipeline eye button crash (index out of range)
  - Fixed derivative order parameter empty field
  - Fixed feature engineering enumerate bug
  - Fixed deep learning module syntax errors

- **UI Polish** (2025-10-07):
  - Input datasets layout optimization (show 3-4 items minimum)
  - Pipeline step selection visual feedback (darker background)
  - Changed pipeline add button color (blue → green)
  - Section title standardization across pages

### Changed

- **Documentation Structure**: Separated public docs/ from local .docs/ for better organization
- **README Organization**: Removed issues/problems from README, moved to troubleshooting docs
- **Build System**: Upgraded to PyInstaller 6.16.0+ for better compatibility

### Security

- **Type Validation**: Comprehensive parameter type checking prevents injection attacks
- **Path Validation**: All file operations validate paths to prevent directory traversal

## [0.1.0] - 2025-10-01

### Added
- Initial alpha release
- Basic preprocessing pipeline
- PCA analysis
- Simple machine learning integration

## Release Notes

### Version 1.0.0-alpha

This is the first alpha release of the Raman Spectroscopy Analysis Application, developed as a final year project at the University of Toyama.

**Status**: Alpha - Feature complete but undergoing testing and refinement

**Recommended for**:
- Research laboratories
- Academic institutions
- Method development and validation

**Not recommended for**:
- Clinical diagnostic use (not approved)
- Production medical systems

**Known Limitations**:
1. Some preprocessing methods require further validation
2. Deep learning features require GPU for practical use
3. Large datasets (>5000 spectra) may require optimization
4. Some advanced features are planned but not yet implemented

**Upcoming Features** (v1.1.0):
- Additional spectral unmixing methods (NMF, ICA)
- Enhanced batch processing capabilities
- REST API for remote processing
- Command-line interface
- Additional file format support (SPC, WDF)
- Video tutorials
- Japanese documentation completion
- Malay translation

### Migration Guides

#### From v0.1.0 to v1.0.0-alpha

**Breaking Changes**: None - v1.0.0 is backwards compatible

**New Features to Note**:
1. **GroupKFold is now default** for ML validation - ensures patient-level splitting
2. **SHAP values** now available for all tree-based models
3. **New preprocessing methods** - Review [Preprocessing Guide](docs/user-guide/preprocessing.md)

**Recommended Actions**:
1. Update preprocessing pipelines to use new methods
2. Re-run ML experiments with GroupKFold validation
3. Export results with SHAP interpretability

## Contributors

### Core Development
- **Muhammad Helmi bin Rozain** - Lead Developer, BSc Student
  - GitHub: [@zerozedsc](https://github.com/zerozedsc)
  - University of Toyama (富山大学)

### Supervision
- **大嶋 佑介 (Oshima Yusuke)** - Primary Supervisor
- **竹谷 皓規 (Taketani Akinori)** - Co-Supervisor
- **Laboratory**: [Clinical Photonics and Information Engineering](http://www3.u-toyama.ac.jp/medphoto/)

### Acknowledgments
- University of Toyama for research facilities and support
- Laboratory members for testing and feedback
- Open-source community for underlying libraries:
  - PySide6/Qt - UI framework
  - RamanSPy - Raman spectroscopy tools
  - scikit-learn - Machine learning
  - pybaselines - Baseline correction
  - NumPy, SciPy, pandas - Numerical computing
  - matplotlib - Visualization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{rozain2025raman,
  author = {Rozain, Muhammad Helmi bin},
  title = {Raman Spectroscopy Analysis Application: A Comprehensive Platform for Real-Time Spectral Classification},
  year = {2025},
  version = {1.0.0-alpha},
  publisher = {GitHub},
  url = {https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application},
  institution = {University of Toyama, Laboratory for Clinical Photonics and Information Engineering}
}
```

## Support

- **Documentation**: [ReadTheDocs](https://raman-spectroscopy-analysis-application.readthedocs.io/en/latest/)
- **Issues**: [GitHub Issues](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/discussions)
- **Email**: Contact via [@zerozedsc](https://github.com/zerozedsc)

---

**Note**: This software is intended for **research use only** and is not approved for clinical diagnostic purposes.
