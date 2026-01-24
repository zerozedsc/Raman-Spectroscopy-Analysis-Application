# Troubleshooting Guide

This guide helps you diagnose and fix common issues with the Raman Spectroscopy Analysis Application.

## Quick Diagnostic Steps

Before diving into specific issues:

1. **Check the logs**
   - Location: `logs/` folder in project directory
   - Key files: `application.log`, `PreprocessPage.log`, `RamanPipeline.log`

2. **Verify installation**
   ```bash
   python -c "import PySide6; print('PySide6 OK')"
   python -c "import ramanspy; print('RamanSPy OK')"
   ```

3. **Update to latest version**
   ```bash
   git pull origin main
   uv pip install -e .
   ```

4. **Check system resources**
   - RAM usage (Task Manager / Activity Monitor)
   - Disk space (>500 MB free recommended)
   - CPU usage

## Installation Issues

### Python Version Error

**Error:** `Python 3.12 or higher is required`

**Diagnosis:**
```bash
python --version
```

**Solution:**
1. Install Python 3.12+ from [python.org](https://www.python.org/downloads/)
2. On Windows, check "Add Python to PATH" during installation
3. Verify installation: `python --version`
4. Reinstall dependencies: `uv pip install -e .`

---

### Module Not Found

**Error:** `ModuleNotFoundError: No module named 'PySide6'` (or other modules)

**Diagnosis:**
- Check if virtual environment is activated
- Verify dependencies are installed

**Solution:**
```bash
# Ensure virtual environment is activated
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Reinstall dependencies
uv pip install -e .
# OR
pip install -r requirements.txt

# Verify installation
python -c "import PySide6; print('OK')"
```

---

### UV Installation Fails

**Error:** `pip install uv` fails or UV commands don't work

**Solution:**
Use traditional virtual environment instead:
```bash
# Create venv
python -m venv .venv

# Activate
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

---

### Permission Denied (Linux/macOS)

**Error:** `PermissionError: [Errno 13] Permission denied`

**Cause:** Trying to install globally without sudo (bad practice)

**Solution:**
```bash
# Never use sudo with pip in virtual environments
# Instead, activate virtual environment first:
source .venv/bin/activate

# Then install normally
pip install -r requirements.txt
```

---

### Windows SmartScreen Blocks Executable

**Warning:** "Windows protected your PC" when running executable

**Solution:**
1. Click **More info**
2. Click **Run anyway**

**Why this happens:** The executable is not digitally signed (requires expensive code signing certificate). The software is safe to run.

**Alternative:** Run from source using Python to avoid this warning.

## Application Launch Issues

### Application Doesn't Start

**Symptoms:** Double-click executable or run `python main.py` - nothing happens

**Diagnosis:**
```bash
# Run with verbose logging
python main.py --verbose

# Check logs
cat logs/application.log  # macOS/Linux
type logs\application.log  # Windows
```

**Common causes:**

1. **Missing configuration files**
   - Check `configs/app_configs.json` exists
   - If missing, application will create default on first run

2. **Port conflict**
   - Check if another instance is running
   - Close duplicate instances

3. **Graphics driver issue**
   - Update graphics drivers
   - Try software rendering: `export QT_QPA_PLATFORM=offscreen` (Linux)

---

### Application Crashes on Startup

**Error:** Immediate crash or "Segmentation fault"

**Diagnosis:**
Look for error in `logs/application.log`

**Solutions by error type:**

**Qt Platform Error:**
```bash
# Linux: Install required Qt libraries
sudo apt-get install libxcb-xinerama0 libxcb-cursor0

# macOS: Reinstall PySide6
pip uninstall PySide6
pip install PySide6
```

**OpenGL Error:**
```bash
# Use software rendering (Linux)
export QT_QPA_PLATFORM=offscreen

# Or force specific Qt backend
export QT_QPA_PLATFORM=xcb
```

**Font Error:**
```bash
# Install font packages
# Linux:
sudo apt-get install fonts-noto fonts-noto-cjk

# macOS: Fonts should be included
```

---

### Black/Blank Window

**Symptoms:** Application opens but window is completely black or blank

**Causes:**
1. Graphics driver incompatibility
2. High DPI scaling issues
3. Qt platform plugin issues

**Solutions:**

**Try software rendering:**
```bash
# Linux
export QT_QPA_PLATFORM=xcb
python main.py

# Windows
set QT_QPA_PLATFORM=windows
python main.py
```

**Adjust DPI scaling (Windows):**
1. Right-click application → Properties
2. Compatibility tab
3. Check "Override high DPI scaling behavior"
4. Select "Application" from dropdown

**Update graphics drivers:**
- NVIDIA: GeForce Experience
- AMD: Radeon Software
- Intel: Intel Driver & Support Assistant

## Data Import Issues

### File Not Recognized

**Error:** "Unable to read file" or "Unsupported file format"

**Diagnosis:**
- Check file extension (.csv, .txt, .mat)
- Open file in text editor to inspect structure

**Solutions:**

1. **For CSV/TXT files:**
   - Ensure proper delimiter (comma for CSV, tab for TXT)
   - Remove non-numeric data (text annotations, units)
   - Verify first row/column structure

2. **For MATLAB .mat files:**
   - Check MATLAB version compatibility (v7.3 or older)
   - Verify variable names are standard
   - Re-save in older format if needed

**Working CSV example:**
```
Wavenumber,Spectrum1,Spectrum2
400,125.3,134.2
401,126.1,135.4
402,127.3,136.8
```

---

### Dimension Mismatch Error

**Error:** "Shape mismatch" or "Dimension error"

**Cause:** Not all spectra have same wavenumber range

**Diagnosis:**
Check each file:
```python
import pandas as pd
df1 = pd.read_csv('file1.csv')
df2 = pd.read_csv('file2.csv')
print(df1.shape, df2.shape)  # Should be same
```

**Solution:**
1. Ensure all spectra have identical wavenumber range
2. Use interpolation if ranges differ slightly:
   - Preprocessing → Interpolation → Select common range

---

### Data Looks Wrong After Import

**Symptoms:** Spectra inverted, flipped, or nonsensical

**Diagnosis:**
- Check if wavenumber values are in rows or columns
- Verify intensity values are positive
- Look for missing data (NaN, null values)

**Solution:**
1. **Transpose data if needed:**
   - Data Package → Select dataset → **Transpose**

2. **Handle missing values:**
   - Replace NaN with zeros or interpolate
   - Or remove affected spectra

3. **Check wavenumber order:**
   - Should be ascending (400 → 1800) or descending (1800 → 400)
   - Use **Reverse Wavenumber Axis** if needed

## Preprocessing Issues

### Preview Shows All Zeros

**Symptoms:** After adding preprocessing step, preview is blank or all zeros

**Causes:**
1. Parameter out of valid range
2. Input data incompatible with method
3. Method requires positive intensities (has negatives)

**Diagnosis:**
- Check console/log for error messages
- Try with default parameters
- Preview intermediate steps

**Solutions:**

1. **Reset parameters to defaults:**
   - Click parameter label to reset

2. **Check input data:**
   - Ensure baseline correction is done first
   - Verify no negative intensities

3. **Try different method:**
   - If AsLS fails, try AirPLS
   - If method-specific, use alternative

---

### Baseline Correction Not Working

**Symptoms:** Fluorescence background still present after baseline correction

**Diagnosis:**
Check preview - is baseline slightly improved or completely unchanged?

**Solutions:**

**If slightly improved:**
- Increase lambda parameter (try 1e7, 1e8)
- Try different method (AirPLS instead of AsLS)

**If completely unchanged:**
- Check that method is actually applied (green checkmark)
- Verify input data is loaded
- Try simpler method (Polynomial baseline)

**For strong fluorescence:**
```
Method: AirPLS
Lambda: 1e6 or higher
Max Iterations: 50
```

---

### Smoothing Removes Peaks

**Symptoms:** Important peaks disappear after smoothing

**Cause:** Window size too large

**Solution:**
Reduce smoothing window:
```
For sharp peaks: window = 5
For normal peaks: window = 7-11
For broad peaks: window = 11-15
```

**Rule:** Window must be odd number and smaller than peak width.

---

### Normalization Produces Strange Results

**Symptoms:** Spectra become extremely small/large or unrecognizable

**Causes:**
1. Baseline not removed first
2. Negative intensities present
3. Wrong normalization method for data type

**Solutions:**

1. **Always baseline correct before normalizing**
2. **Use appropriate normalization:**
   - Vector: Most Raman data
   - SNV: Biological samples
   - Min-Max: When absolute intensities matter
   - Area: For quantitative comparisons

3. **Remove negative values:**
   - Add offset: minimum value + small epsilon
   - Or use ReLU: replace negatives with 0

---

### Pipeline Fails to Execute

**Error:** "Pipeline execution failed" or individual steps fail

**Diagnosis:**
1. Check which step fails (look at step number in error)
2. Verify parameters for that step
3. Test step individually

**Common issues:**

1. **Step order wrong:**
   - Baseline → Smoothing → Normalization
   - Never normalize before baseline

2. **Incompatible steps:**
   - Some methods require specific input ranges
   - Check method documentation

3. **Resource exhaustion:**
   - Too many spectra for available RAM
   - Process in smaller batches

**Solution:**
- Test each step individually
- Check parameter constraints
- Reduce batch size if memory issue

## Analysis Issues

### PCA Shows No Group Separation

**Symptoms:** All groups overlap in PCA plot

**NOT necessarily an error!** This may indicate:
1. Groups are genuinely similar
2. Preprocessing removes discriminative features
3. Variance is in other components (PC3, PC4)

**Diagnosis checklist:**

- [ ] Baseline correction applied?
- [ ] Normalization applied?
- [ ] Outliers removed?
- [ ] Checked PC2 vs PC3 plot?
- [ ] Examined scree plot (variance explained)?

**Solutions:**

1. **Try supervised method:**
   - Use PLS-DA instead of PCA
   - PLS-DA maximizes group separation

2. **Improve preprocessing:**
   - Ensure baseline is fully removed
   - Try different normalization (SNV vs Vector)
   - Add cosmic ray removal

3. **Check other components:**
   - Plot PC2 vs PC3
   - Look for separation in higher PCs

4. **Remove outliers:**
   - Use outlier detection
   - Remove spectra with poor quality

5. **Consider reality:**
   - Groups may actually be similar
   - Small effect size requires larger sample

---

### Statistical Tests Show No Significant Differences

**Symptoms:** All p-values > 0.05, no significant regions

**Possible causes:**
1. Small sample size (low statistical power)
2. High variability within groups
3. Groups are actually similar
4. Overly strict correction (Bonferroni)

**Solutions:**

1. **Check sample size:**
   - Need n ≥ 5 per group minimum
   - n ≥ 10 per group recommended
   - Increase sample size if possible

2. **Use appropriate test:**
   - Mann-Whitney for non-normal data
   - Use less conservative correction (FDR instead of Bonferroni)

3. **Check effect size:**
   - Even non-significant p-values can have large effect sizes
   - Report Cohen's d or similar

4. **Explore data:**
   - Use PCA to visualize
   - Check if differences exist at all

---

### Analysis Takes Forever

**Symptoms:** Analysis runs for >5 minutes without completing

**Causes:**
1. Large dataset (>1000 spectra)
2. Computationally intensive method (UMAP, t-SNE)
3. Too many cross-validation folds

**Solutions:**

1. **Reduce data size:**
   - Use subset for testing parameters
   - Then run full analysis overnight

2. **Optimize parameters:**
   - UMAP/t-SNE: Reduce n_neighbors, n_iterations
   - Clustering: Subsample data

3. **Use faster alternative:**
   - PCA instead of UMAP for quick exploration
   - Random sample for preview

4. **Increase CPU usage:**
   - Settings → Processing → CPU Cores: Set to max

---

### Plots Don't Appear

**Symptoms:** Analysis completes but no plots shown

**Diagnosis:**
- Check if results tab is created
- Look for error in console
- Try exporting results (might show even if not displayed)

**Solutions:**

1. **Check matplotlib backend:**
   ```python
   import matplotlib
   print(matplotlib.get_backend())  # Should be Qt5Agg or similar
   ```

2. **Try refreshing:**
   - Switch to different tab and back
   - Close and reopen results panel

3. **Export and view externally:**
   - Use Export Results button
   - Open saved PNG files

4. **Reinstall matplotlib:**
   ```bash
   pip uninstall matplotlib
   pip install matplotlib
   ```

## Machine Learning Issues

### Model Training Fails

**Error:** "Training failed" or crash during training

**Common causes:**

1. **Insufficient data:**
   - Need at least 20 samples per class
   - Use simpler model or collect more data

2. **NaN/Inf in features:**
   - Check preprocessing produces valid numbers
   - Remove or impute missing values

3. **Memory error:**
   - Reduce dataset size
   - Use simpler model
   - Close other applications

**Solutions:**

1. **Start simple:**
   - Use Logistic Regression first
   - If works, try more complex models

2. **Check data quality:**
   ```python
   # Check for NaN
   import numpy as np
   print(np.isnan(data).any())
   
   # Check for Inf
   print(np.isinf(data).any())
   ```

3. **Reduce complexity:**
   - Fewer cross-validation folds (3 instead of 5)
   - Smaller grid search space
   - Simpler model architecture

---

### 100% Training Accuracy, Poor Test Accuracy

**Symptoms:** Training accuracy = 100%, test accuracy = 50-70%

**Diagnosis:** Classic **overfitting**

**Solutions:**

1. **Simplify model:**
   - Random Forest: Reduce max_depth
   - XGBoost: Increase reg_alpha, reg_lambda
   - Neural Network: Reduce layers/neurons

2. **Increase regularization:**
   - SVM: Reduce C parameter
   - Logistic Regression: Increase penalty

3. **Collect more data:**
   - More samples reduces overfitting
   - Aim for 50+ samples per class

4. **Feature selection:**
   - Use only most important features
   - Reduces model complexity

5. **Ensemble methods:**
   - Use Random Forest (naturally regularized)
   - Averaging reduces overfitting

---

### Groups Imbalanced (90% vs 10%)

**Symptoms:** Model always predicts majority class

**Solutions:**

1. **Use stratified sampling:**
   - Ensures both classes in train/test
   - Already default in application

2. **Use appropriate metrics:**
   - **NOT accuracy** (misleading)
   - Use ROC-AUC, F1-score, balanced accuracy

3. **Class weighting:**
   - Set `class_weight='balanced'` in model parameters
   - Penalizes misclassifying minority class more

4. **Resampling:**
   - Oversample minority class (SMOTE)
   - Undersample majority class
   - Or both (SMOTEENN)

5. **Threshold adjustment:**
   - Default threshold = 0.5
   - Adjust based on ROC curve
   - Optimize F1-score

---

### SHAP Values Take Forever

**Symptoms:** SHAP interpretation runs for >30 minutes

**Cause:** SHAP is computationally expensive, especially for tree models

**Solutions:**

1. **Use subset:**
   - Calculate SHAP for 100 samples only
   - Still provides good interpretation

2. **Use TreeExplainer:**
   - Faster for tree-based models (RF, XGBoost)
   - Automatically used when available

3. **Use KernelExplainer (last resort):**
   - Model-agnostic but slow
   - Use smallest possible sample

4. **Alternative: Permutation Importance:**
   - Much faster
   - Less detailed but still useful

---

### Can't Export Trained Model

**Error:** Export fails or file not created

**Diagnosis:**
- Check write permissions on output folder
- Verify model training completed successfully
- Look for error message

**Solutions:**

1. **Check folder permissions:**
   - Ensure you can write to selected folder
   - Try different location (Desktop, Documents)

2. **Check model size:**
   - Large models (>2GB) may fail
   - Use simpler model or feature selection

3. **Try different format:**
   - Default: pickle (.pkl)
   - Alternative: ONNX (.onnx) - more portable

## Performance Issues

### Application Runs Slowly

**General performance optimization:**

1. **Close other applications:**
   - Free up RAM and CPU
   - Especially other Python applications

2. **Reduce CPU usage:**
   - Settings → Processing → CPU Cores
   - Set to `max - 1` to keep system responsive

3. **Process smaller batches:**
   - Split large datasets into smaller groups
   - Process sequentially

4. **Disable real-time preview:**
   - Preview requires continuous computation
   - Disable while building pipeline

5. **Update Python and libraries:**
   ```bash
   pip install --upgrade pip
   pip install --upgrade numpy scipy scikit-learn
   ```

---

### High RAM Usage

**Symptoms:** Application uses >4 GB RAM or crashes with "MemoryError"

**Solutions:**

1. **Process fewer spectra at once:**
   - Split dataset into batches of 100-500 spectra
   - Process and save each batch

2. **Reduce data precision:**
   - Convert float64 to float32 (half memory)
   - Acceptable for most analyses

3. **Close unused results:**
   - Each analysis result keeps figures in memory
   - Close old results before starting new analyses

4. **Restart application periodically:**
   - Memory leaks may accumulate
   - Restart after several hours of use

---

### Disk Space Issues

**Symptoms:** "No space left on device" or save operations fail

**Diagnosis:**
```bash
# Check disk space
df -h  # Linux/macOS
wmic logicaldisk get size,freespace  # Windows
```

**Solutions:**

1. **Clean up old projects:**
   - Delete or archive completed projects
   - Projects can be large (>1 GB with results)

2. **Clear logs:**
   - Delete old log files in `logs/` folder
   - Keep only recent logs

3. **Export and delete results:**
   - Export important results (figures, data)
   - Delete from application

4. **Move projects to external drive:**
   - Projects are portable
   - Move entire project folder

## UI Issues

### Text Too Small/Large

**Cause:** High DPI scaling on Windows

**Solution:**
1. Settings → Interface → Font Size
2. Choose: Small, Medium, Large, Extra Large
3. Restart application

**Windows scaling issue:**
1. Right-click application → Properties → Compatibility
2. Check "Override high DPI scaling behavior"
3. Select "Application"

---

### Japanese Text Shows as Boxes (□□□)

**Cause:** Missing Japanese font

**Solutions:**

**Windows:**
1. Control Panel → Fonts
2. Ensure "MS Gothic" or "Meiryo" is installed
3. If missing, install from Windows Features

**macOS:**
1. Usually includes Japanese fonts by default
2. If issue persists, install "Hiragino Sans"

**Linux:**
```bash
sudo apt-get install fonts-noto-cjk
```

Then restart application.

---

### Buttons Not Responding

**Symptoms:** Clicking buttons does nothing

**Diagnosis:**
- Check if application is busy (loading icon)
- Look for error in console
- Check logs for exceptions

**Solutions:**

1. **Wait for current operation:**
   - Some operations block UI temporarily
   - Wait 10-30 seconds

2. **Cancel running operation:**
   - Click Stop button if available
   - Or close and reopen application

3. **Check for modal dialogs:**
   - Another dialog may be open behind main window
   - Alt+Tab to find it

4. **Restart application:**
   - Last resort if UI completely frozen
   - Check logs after restart

## Getting More Help

### Collect Diagnostic Information

When reporting issues, include:

1. **System information:**
   ```bash
   python --version
   pip list > installed_packages.txt
   ```

2. **Error logs:**
   - Attach `logs/application.log`
   - Include error traceback

3. **Screenshots:**
   - Show the issue visually
   - Include error messages

4. **Steps to reproduce:**
   - Detailed step-by-step
   - Sample data if possible

### Where to Get Help

1. **Search documentation:**
   - [User Guide](user-guide/index.md)
   - [FAQ](faq.md)

2. **Search existing issues:**
   - [GitHub Issues](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues)
   - Many problems already solved

3. **Ask community:**
   - [GitHub Discussions](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/discussions)
   - Other users may have encountered same issue

4. **Report bug:**
   - [Create new issue](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues/new)
   - Include diagnostic information

5. **Contact developer:**
   - [@zerozedsc](https://github.com/zerozedsc)
   - For complex issues

### Emergency: Application Completely Broken

**Nuclear option - Full reinstall:**

```bash
# 1. Backup your projects folder
cp -r projects/ /backup/location/

# 2. Remove application completely
rm -rf .venv/
rm -rf logs/
rm -rf __pycache__/

# 3. Fresh install
git pull origin main
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 4. Test with simple operation
python main.py

# 5. Restore projects
cp -r /backup/location/projects/ ./
```

This should resolve almost any installation corruption issue.

## Contributing to This Guide

Found a solution not listed here? 

1. Fork the [repository](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application)
2. Edit `docs/troubleshooting.md`
3. Add your solution with clear title, diagnosis, and solution steps
4. Submit pull request

Help others avoid the same issues you faced!
