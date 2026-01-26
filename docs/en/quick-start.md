# Quick Start

This quick start guide will help you perform your first complete analysis in **15 minutes**.

## Prerequisites

- Application installed (see [Installation Guide](installation.md))
- Sample Raman spectroscopy data (CSV, TXT, or MAT format)
- Basic understanding of Raman spectroscopy

## Tutorial: Analyzing Blood Plasma Samples

This tutorial demonstrates a complete workflow for comparing healthy vs disease samples.

### Step 1: Launch and Create Project (2 minutes)

1. **Launch the application**
   ```bash
   uv run python main.py  # From source
   # OR
   # Double-click RamanApp.exe  # Portable/Installer
   ```

2. **Create a new project**
   - Click **New Project** on the Home page
   - **Project Name**: `Blood Plasma Analysis`
   - **Location**: Choose a folder (default is fine)
   - Click **Create**

3. **Verify project creation**
   - You should see the project name in the title bar
   - All tabs (Home, Data, Preprocessing, Analysis, ML) should be visible

### Step 2: Import Data (3 minutes)

1. **Navigate to Data Package tab**
   - Click the **Data Package** tab at the top

2. **Import your spectra**
   - Click **Import Data** button
   - Select your data files:
     - **CSV**: Each column is a spectrum, rows are wavenumbers
     - **TXT**: Tab or space-separated values
     - **MAT**: MATLAB format with spectrum arrays
   - Click **Open**

3. **Create groups**
   - Click **Create Group** in the left panel
   - **Group Name**: `Healthy`
   - Select spectra from healthy samples
   - Click **Add to Group**
   - Repeat for `Disease` group

4. **Verify data**
   - Preview pane should show all imported spectra
   - Check that wavenumber range is correct (typically 400-1800 cm⁻¹)
   - Verify spectrum count matches your expectations

### Step 3: Preprocess Data (5 minutes)

1. **Navigate to Preprocessing tab**

2. **Add baseline correction**
   - Click **➕ Add Step** button
   - **Category**: Baseline Correction
   - **Method**: AsLS (Asymmetric Least Squares)
   - **Parameters**:
     - Lambda: `1e6` (smoothness)
     - P: `0.001` (asymmetry)
   - **Preview**: Check that fluorescence background is removed

3. **Add smoothing**
   - Click **➕ Add Step**
   - **Category**: Smoothing
   - **Method**: Savitzky-Golay
   - **Parameters**:
     - Window Length: `11` (must be odd)
     - Polynomial Order: `3`
   - **Preview**: Check that noise is reduced without losing peaks

4. **Add normalization**
   - Click **➕ Add Step**
   - **Category**: Normalization
   - **Method**: Vector Normalization
   - **Preview**: Check that all spectra have similar intensity scales

5. **Apply pipeline**
   - Review the preview of all steps
   - Click **Apply Pipeline** button
   - **Output Name**: `Preprocessed_Spectra`
   - Select **All Datasets**
   - Click **Confirm**
   - Wait for processing to complete (~10-30 seconds)

6. **Verify results**
   - New dataset `Preprocessed_Spectra` should appear in Data Package
   - Inspect spectra visually - should be clean and normalized

### Step 4: Exploratory Analysis with PCA (3 minutes)

1. **Navigate to Analysis tab**

2. **Select PCA method**
   - In the method list, click **PCA (Principal Component Analysis)**

3. **Configure parameters**
   - **Dataset**: Select `Preprocessed_Spectra`
   - **Number of Components**: `3`
   - **Scaling Method**: `StandardScaler` (recommended)
   - **Show 95% Confidence Ellipses**: ✓ Enable
   - **Show Loadings Plot**: ✓ Enable

4. **Run analysis**
   - Click **Run Analysis** button
   - Wait for computation (~5-15 seconds)

5. **Interpret results**
   - **Scores Plot (PC1 vs PC2)**:
     - Do Healthy and Disease groups separate?
     - Are there any outliers?
   - **Scree Plot**:
     - How much variance do PC1 and PC2 explain?
     - Typically want >60% for PC1+PC2
   - **Loadings Plot**:
     - Which wavenumbers (Raman bands) drive the separation?
     - Match peaks to biochemical assignments

6. **Export results**
   - Click **Export Results** button
   - Choose location and filename
   - Saves figures (PNG) and data (CSV)

### Step 5: Statistical Testing (2 minutes)

1. **Select statistical test**
   - In the method list, click **Pairwise Statistical Tests**

2. **Configure parameters**
   - **Dataset**: `Preprocessed_Spectra`
   - **Group 1**: `Healthy`
   - **Group 2**: `Disease`
   - **Test Method**: `Mann-Whitney U` (non-parametric, recommended)
   - **Multiple Testing Correction**: `FDR (Benjamini-Hochberg)`
   - **Significance Level**: `0.05`

3. **Run test**
   - Click **Run Analysis**
   - Results show:
     - P-value heatmap across wavenumbers
     - Significant regions highlighted
     - Effect sizes

4. **Interpret results**
   - Which wavenumber regions show significant differences?
   - Map significant peaks to biochemical components:
     - 1650 cm⁻¹ → Amide I (proteins)
     - 1440 cm⁻¹ → CH₂ deformation (lipids)
     - 1000 cm⁻¹ → Phenylalanine (aromatic amino acids)

## Optional: Machine Learning Classification

If you want to build a classification model:

### Step 6: Train ML Model (Optional, +10 minutes)

1. **Navigate to Machine Learning tab**

2. **Configure dataset**
   - Select `Preprocessed_Spectra`
   - **Groups**: Ensure `Healthy` and `Disease` are defined

3. **Choose algorithm**
   - **Algorithm**: Random Forest (recommended for beginners)
   - **Parameters**: Use defaults

4. **Configure validation**
   - **Method**: GroupKFold (prevents data leakage)
   - **Number of Folds**: `5`
   - **Test Set Size**: `20%`

5. **Train model**
   - Click **Train Model**
   - Wait for training (~30 seconds to 2 minutes)

6. **Evaluate results**
   - **ROC Curve**: Check AUC score (>0.90 is excellent)
   - **Confusion Matrix**: Check classification accuracy
   - **SHAP Values**: Identify most important wavenumbers

7. **Export model**
   - Click **Export Model**
   - Save trained model for future use

## Next Steps

Congratulations! You've completed your first analysis. Now explore:

### Learn More About Methods

- [Preprocessing Methods](analysis-methods/preprocessing.md) - Complete preprocessing reference
- {ref}`PCA Guide <pca>` - Deep dive into PCA theory and interpretation
- [Statistical Tests](analysis-methods/statistical.md) - All available statistical methods
- [Machine Learning](analysis-methods/machine-learning.md) - Complete ML pipeline guide

### Advanced Workflows

- [Multi-Group Comparison](user-guide/analysis.md) - Compare >2 groups
- [Custom Pipelines](user-guide/preprocessing.md) - Build complex preprocessing workflows
- [Batch Processing](user-guide/preprocessing.md) - Process multiple datasets
- [Hyperparameter Optimization](user-guide/machine-learning.md) - Optimize ML models

### Best Practices

- [Data Quality](user-guide/best-practices.md#data-quality) - Ensure clean data
- {ref}`Avoiding Data Leakage <data-leakage>` - Proper train/test splitting
- [Publication-Ready Figures](user-guide/best-practices.md#figures) - Export high-quality plots
- [Reproducible Workflows](user-guide/best-practices.md#reproducibility) - Document your analysis

## Common Issues

### Data Import Problems

**Issue**: "Unable to read file"  
**Solution**: 
- Check file format (CSV with headers, TXT tab-separated)
- Ensure numeric data only (remove text annotations)
- Verify wavenumber range is in first column/row

**Issue**: "Dimension mismatch"  
**Solution**:
- All spectra must have same wavenumber range
- Check for missing data points
- Ensure consistent sampling intervals

### Preprocessing Errors

**Issue**: "Baseline correction failed"  
**Solution**:
- Try different method (AsLS, AirPLS, Polynomial)
- Adjust lambda parameter (increase for smoother baseline)
- Check for cosmic rays or spikes in raw data

**Issue**: "Preview is blank"  
**Solution**:
- Check that input dataset is selected
- Verify preprocessing parameters are valid
- Look for error messages in console/log

### Analysis Issues

**Issue**: "Groups don't separate in PCA"  
**Solution**:
- Ensure preprocessing is correct (baseline + normalization)
- Check for outliers and remove bad spectra
- Try supervised method (PLS-DA) instead of PCA
- Consider that groups may actually be similar

**Issue**: "No significant differences found"  
**Solution**:
- Check sample size (n ≥ 5 per group recommended)
- Verify groups are correctly assigned
- Consider more sensitive statistical tests
- Groups may genuinely not differ

## Getting Help

If you encounter issues not covered here:

1. **Check documentation**: [User Guide](user-guide/index.md) and [Troubleshooting](troubleshooting.md)
2. **Search issues**: [GitHub Issues](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues)
3. **Ask community**: [GitHub Discussions](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/discussions)
4. **Report bug**: Create new issue with:
   - Steps to reproduce
   - Error messages
   - Sample data (if possible)

## Feedback

Help us improve this quick start guide! Submit suggestions via [GitHub Issues](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues) with the label `documentation`.
