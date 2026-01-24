# Frequently Asked Questions (FAQ)

## General Questions

### What is this application for?

The Raman Spectroscopy Analysis Application is designed for analyzing Raman spectroscopy data, particularly for disease detection and biomedical research. It provides preprocessing, exploratory analysis, statistical testing, and machine learning classification in an easy-to-use desktop interface.

### Who developed this software?

This software was developed by Muhammad Helmi bin Rozain as a final year project at the University of Toyama, under the Laboratory for Clinical Photonics and Information Engineering (臨床光情報工学研究室), supervised by 大嶋 佑介 (Oshima Yusuke) and 竹谷 皓規 (Taketani Akinori).

### Is this software free?

Yes, this software is open-source and released under the MIT License. You can use it freely for academic research, commercial applications, and personal projects.

### Can I use this for clinical diagnosis?

**No**. This software is intended for **research use only** and is **not approved for clinical diagnostic purposes**. Always consult qualified medical professionals for medical decisions.

### What platforms are supported?

- **Windows**: 10/11 (fully supported, installer and portable versions available)
- **macOS**: 10.14+ (supported from source)
- **Linux**: Ubuntu 18.04+ (supported from source)

## Installation Questions

### Do I need Python installed?

- **For source installation**: Yes, Python 3.12+ is required
- **For Windows executable/installer**: No, Python is bundled

### Why is the executable so large (375 MB)?

The executable bundles Python, all libraries, and dependencies for complete portability. This ensures it works on any Windows system without installation.

### How do I update to a new version?

**From source:**
```bash
git pull origin main
uv pip install -e .
```

**From executable:**
Download the latest version from [Releases](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/releases) and replace the old executable.

### Can I install on a computer without internet?

Yes, the portable executable runs completely offline once downloaded. For source installation, download dependencies on a connected computer and transfer them.

## Data Questions

### What file formats are supported?

**Fully supported:**
- CSV (comma-separated values)
- TXT (tab or space-separated)
- MAT (MATLAB format)

**Planned support:**
- SPC (Galactic)
- WDF (Renishaw WiRE)
- Custom binary formats

### What data structure is required?

**Recommended format:**
- **Rows**: Wavenumber values (cm⁻¹)
- **Columns**: Individual spectra
- **First row**: Optional column headers (spectrum IDs)
- **First column**: Wavenumber values

**Example CSV:**
```
Wavenumber,Sample1,Sample2,Sample3
400,125.3,134.2,128.7
401,126.1,135.4,129.3
...
```

### My data has x-axis in nm, not cm⁻¹. What should I do?

Convert wavelength (nm) to wavenumber (cm⁻¹):

**Formula:** Wavenumber = 10,000,000 / wavelength (nm)

**Example:** 
- 785 nm laser → 12,738 cm⁻¹
- For Raman shift, subtract excitation wavenumber

### Can I import multiple files at once?

Yes, use the batch import feature:
1. Click **Import Data** in the Data Package tab
2. Select multiple files (Ctrl+Click or Shift+Click)
3. All files will be imported as separate datasets

### How do I handle replicates?

**Option 1: Average replicates during import**
- Select "Average Replicates" in import dialog
- Specify replicate pattern (e.g., Sample1_rep1, Sample1_rep2)

**Option 2: Keep replicates separate**
- Import all spectra individually
- Use **Groups** to organize (e.g., "Sample1" group contains all Sample1 replicates)

## Preprocessing Questions

### What preprocessing should I use?

**Minimum recommended pipeline:**
1. Baseline Correction (AsLS or AirPLS)
2. Smoothing (Savitzky-Golay, window=11)
3. Normalization (Vector or SNV)

See [Preprocessing Guide](user-guide/preprocessing.md#recommended-pipelines) for specific use cases.

### What is the difference between AsLS and AirPLS?

- **AsLS (Asymmetric Least Squares)**: Fast, works well for smooth baselines
- **AirPLS (Adaptive Iteratively Reweighted Penalized Least Squares)**: Better for complex baselines with sharp peaks

Try both and compare the preview. AsLS is a good starting point.

### Should I normalize before or after baseline correction?

**Always baseline correction first**, then normalization:
1. Baseline correction removes additive background
2. Normalization handles multiplicative intensity differences

Reversing this order produces incorrect results.

### Can I save my preprocessing pipeline?

Yes:
1. Build your pipeline in the Preprocessing tab
2. Click **Save Pipeline**
3. Give it a descriptive name (e.g., "MGUS_Classification_Pipeline")
4. Load it later with **Load Pipeline** button

### My preview shows all zeros after preprocessing. What's wrong?

**Common causes:**
1. **Parameter out of range**: Check parameter values are within valid ranges
2. **Incorrect order**: Ensure baseline correction comes before normalization
3. **Data already processed**: Don't apply preprocessing twice
4. **Negative intensities**: Some methods require positive intensities only

Check the console/log for error messages.

## Analysis Questions

### PCA shows no group separation. What should I do?

**Possible reasons and solutions:**

1. **Groups are actually similar**
   - Try supervised method (PLS-DA) instead
   - Check if differences are subtle (small effect size)

2. **Preprocessing issue**
   - Verify baseline is removed
   - Check normalization is applied
   - Try different preprocessing pipeline

3. **Outliers dominating**
   - Use outlier detection and remove bad spectra
   - Check for cosmic rays

4. **Need more components**
   - Try PC2 vs PC3 plot
   - Examine scree plot for variance distribution

### How many principal components should I use?

**For visualization:** 2-3 components (PC1 vs PC2 plot)

**For analysis:** Keep components until cumulative explained variance > 80-90%

**For classification:** Use scree plot elbow point (typically 5-10 components for Raman data)

### What statistical test should I use?

| Scenario                           | Test                                  |
| ---------------------------------- | ------------------------------------- |
| 2 groups, normal distribution      | Independent t-test                    |
| 2 groups, non-normal distribution  | Mann-Whitney U test                   |
| 2 groups, paired samples           | Paired t-test or Wilcoxon signed-rank |
| >2 groups, normal distribution     | One-way ANOVA                         |
| >2 groups, non-normal distribution | Kruskal-Wallis test                   |

**Tip:** Mann-Whitney U is robust and works for most cases. When in doubt, use it.

### What is "multiple testing correction" and do I need it?

When testing at many wavenumbers (~1400 points), you risk false positives. **Multiple testing correction** (e.g., FDR, Bonferroni) adjusts p-values to control false discovery rate.

**Always use correction** when testing across full spectrum. Bonferroni is conservative, FDR (Benjamini-Hochberg) is balanced.

## Machine Learning Questions

### What algorithm should I choose?

**For beginners:** Random Forest
- Easy to use
- Robust to overfitting
- Provides feature importance
- Few hyperparameters to tune

**For best performance:** XGBoost
- Often highest accuracy
- Requires careful hyperparameter tuning
- May overfit on small datasets

**For interpretability:** Logistic Regression
- Simple, transparent
- Works well for linearly separable data
- Fast training

### What is GroupKFold and why should I use it?

GroupKFold ensures **all spectra from the same patient/sample** stay together in either training or test set. This prevents data leakage when you have multiple spectra per patient.

**Always use GroupKFold** if you have multiple spectra per patient/biological sample.

### How much data do I need?

**Minimum:**
- 20-30 samples per group
- At least 5 patients per group (if using LOPOCV)

**Recommended:**
- 50-100 samples per group
- 10+ patients per group

**For deep learning:**
- 100+ samples per group minimum
- 500+ samples for robust models

### My model has 100% accuracy. Is that good?

**Probably not!** 100% accuracy often indicates:
1. **Data leakage** - Test data contaminated training
2. **Overfitting** - Model memorized training data
3. **Too easy problem** - Groups are perfectly separable (rare)

**Check:**
- Use external validation set
- Verify GroupKFold is used correctly
- Simplify model and see if performance drops
- Check confusion matrix for patterns

### What is SHAP and how do I use it?

SHAP (SHapley Additive exPlanations) shows which wavenumbers (features) are most important for model predictions.

**How to interpret SHAP plots:**
- **Higher absolute SHAP value** = more important feature
- **Positive SHAP** = increases probability of class 1
- **Negative SHAP** = decreases probability of class 1

Map important wavenumbers to biochemical assignments (e.g., 1650 cm⁻¹ = Amide I proteins).

## Export and Results Questions

### How do I export results?

Each analysis method has an **Export Results** button that saves:
- **Figures**: High-resolution PNG images (300 DPI)
- **Data**: CSV files with numerical results
- **Report**: Text file with analysis summary

### Can I get publication-quality figures?

Yes! Figures are exported at 300 DPI (publication quality). You can also customize:
- Figure size
- Font sizes
- Color schemes
- Line widths

In the Settings menu, adjust "Figure Export Settings".

### How do I cite this software?

```bibtex
@software{rozain2025raman,
  author = {Rozain, Muhammad Helmi bin},
  title = {Raman Spectroscopy Analysis Application},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application},
  institution = {University of Toyama}
}
```

## Performance Questions

### The application is slow. How can I speed it up?

**For preprocessing:**
- Use fewer CPU cores (Settings → Processing → CPU Cores)
- Process smaller batches
- Reduce smoothing window size

**For analysis:**
- Reduce number of components (PCA)
- Use smaller sample size for preview
- Close other applications

**For machine learning:**
- Reduce cross-validation folds
- Use simpler model (Logistic Regression instead of XGBoost)
- Reduce grid search parameters

### How much RAM do I need?

**Minimum:** 4 GB RAM
**Recommended:** 8 GB RAM
**For large datasets (>1000 spectra):** 16 GB RAM

## Language and Localization Questions

### Can I use the interface in Japanese?

Yes! Go to **Settings** → **Interface** → **Language** → **日本語**

The application supports:
- English (default)
- Japanese (日本語)
- Malay (planned)

### Some text is still in English after changing language. Why?

Some elements (error messages, console output) may not be fully translated yet. We're continuously improving localization. Report incomplete translations via [GitHub Issues](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues).

## Still Have Questions?

### Documentation

- [User Guide](user-guide/index.md) - Comprehensive tutorials
- [Analysis Methods Reference](analysis-methods/index.md) - Detailed method documentation
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions

### Community

- **GitHub Discussions**: [Ask questions](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/discussions)
- **GitHub Issues**: [Report bugs](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues)
- **Email**: Contact via [@zerozedsc](https://github.com/zerozedsc)

### Contributing

Found an error in the FAQ? Want to add a question?

1. Fork the [repository](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application)
2. Edit `docs/faq.md`
3. Submit a pull request

Your contributions help everyone!
