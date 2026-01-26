# Getting Started

Welcome to the Raman Spectroscopy Analysis Application! This guide will help you get up and running quickly.

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.12 or higher
- **RAM**: 4 GB (8 GB recommended for large datasets)
- **Storage**: 500 MB free space
- **Display**: 1280×720 resolution (1920×1080 recommended)

### Recommended Requirements

- **RAM**: 8 GB or more
- **Display**: 1920×1080 or higher resolution
- **GPU**: NVIDIA GPU with CUDA support (for deep learning preprocessing)

## Installation Options

There are three ways to install and run the application:

### 1. Source Installation (Recommended for Developers)

Best for users who want to:
- Contribute to development
- Customize the application
- Stay on the latest version
- Debug or extend functionality

**Time**: ~10 minutes  
**Difficulty**: Intermediate (requires familiarity with Python)

See {ref}`Installation Guide <from-source>` for detailed steps.

### 2. Portable Executable (Windows Only)

Best for users who want to:
- Run without installing Python
- Quick testing or evaluation
- Run from USB drive
- Avoid system modifications

**Time**: ~2 minutes  
**Difficulty**: Easy

See {ref}`Installation Guide <portable-executable>` for detailed steps.

### 3. Installer (Windows Only)

Best for users who want to:
- Professional installation experience
- Start Menu integration
- File associations for project files
- Standard Windows uninstallation

**Time**: ~5 minutes  
**Difficulty**: Easy

See {ref}`Installation Guide <installer>` for detailed steps.

## Quick Start Tutorial

### Step 1: Launch the Application

**From Source:**
```bash
cd Raman-Spectroscopy-Analysis-Application
uv run python main.py
```

**From Executable:**
Double-click `RamanApp.exe`

**From Installer:**
Launch from Start Menu or desktop shortcut

### Step 2: Create Your First Project

1. On the **Home Page**, click **New Project**
2. Enter a project name (e.g., "My First Analysis")
3. Select a project folder location
4. Click **Create**

### Step 3: Import Data

1. Navigate to the **Data Package** tab
2. Click **Import Data**
3. Select your Raman spectroscopy data files (CSV, TXT, ASC/ASCII, or PKL format)
4. Review the imported spectra in the preview panel

### Step 4: Preprocess Your Data

1. Navigate to the **Preprocessing** tab
2. Add preprocessing steps:
   - **Baseline Correction** → Select "AsLS" (default settings)
   - **Smoothing** → Select "Savitzky-Golay" (window=11, poly=3)
   - **Normalization** → Select "Vector Normalization"
3. Preview the effects in real-time
4. Click **Apply Pipeline** to process all spectra

### Step 5: Analyze Your Data

1. Navigate to the **Analysis** tab
2. Select an analysis method:
   - **PCA** for exploratory analysis
   - **Band Ratio** for specific biomarker analysis
   - **Statistical Tests** for group comparisons
3. Configure parameters
4. Click **Run Analysis**
5. View and export results

### Step 6: Machine Learning (Optional)

1. Navigate to the **Machine Learning** tab
2. Configure your dataset:
   - Define groups (Control vs Disease)
   - Select validation method (GroupKFold recommended)
3. Choose an algorithm (Random Forest recommended for beginners)
4. Click **Train Model**
5. Evaluate results (ROC curves, confusion matrix, SHAP plots)

## Next Steps

Now that you've completed your first analysis, explore these topics:

- [Interface Overview](user-guide/interface-overview.md) - Learn about all interface elements
- [Data Import Guide](user-guide/data-import.md) - Supported formats and data organization
- [Preprocessing Tutorial](user-guide/preprocessing.md) - Detailed preprocessing pipeline guide
- [Analysis Methods](analysis-methods/index.md) - Comprehensive method documentation
- [Best Practices](user-guide/best-practices.md) - Tips for Raman spectroscopy analysis

## Getting Help

### Documentation Resources

- [User Guide](user-guide/index.md) - Comprehensive tutorials and walkthroughs
- [Analysis Methods Reference](analysis-methods/index.md) - Theory and parameter guidance
- [API Documentation](api/index.md) - For developers and advanced users
- [FAQ](faq.md) - Frequently asked questions
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

### Community Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues)
- **GitHub Discussions**: [Ask questions and share experiences](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/discussions)
- **Email**: Contact the development team via [@zerozedsc](https://github.com/zerozedsc)

## Video Tutorials (Coming Soon)

We're working on video tutorials covering:
- Complete installation walkthrough
- First project and data import
- Building a preprocessing pipeline
- Performing PCA analysis
- Training and evaluating ML models

Subscribe to updates on our [GitHub repository](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application) to be notified when videos are available.

## Feedback

Your feedback helps us improve! Please share:
- Feature requests
- Usability suggestions
- Documentation improvements
- Bug reports

Submit feedback via [GitHub Issues](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues).
