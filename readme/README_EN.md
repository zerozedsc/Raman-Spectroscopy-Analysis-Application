# Raman Spectroscopy Analysis Application
## Complete English Documentation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)](https://www.qt.io/qt-for-python)

**Version:** 1.0.0  
**Last Updated:** January 2026  
**Language:** English

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Features](#features)
5. [User Interface Guide](#user-interface-guide)
6. [Preprocessing Methods](#preprocessing-methods)
7. [Analysis Methods](#analysis-methods)
8. [Development](#development)
9. [Contributing](#contributing)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)
12. [License](#license)

---

## Introduction

### About the Project

The Raman Spectroscopy Analysis Application is a comprehensive desktop software designed for **real-time classification and disease detection** using Raman spectroscopy. This project was developed as a **final year project** at the **University of Toyama**, under the supervision of the **Laboratory for Clinical Photonics and Information Engineering**.

<div align="center">
  <img src="images/app-overview.png" alt="Application overview" width="800"/>
</div>

### Research Background

#### Problem Statement

Current challenges in Raman spectroscopy analysis:

1. **Manual Processing Required**
   - Researchers must manually process spectra using MATLAB or Python scripts
   - Time-consuming and prone to human error
   - Requires programming expertise

2. **Proprietary Software**
   - Existing medical/biological spectroscopy software requires expensive licenses
   - Limited customization options
   - Vendor lock-in

3. **Lack of Open-Source Solutions**
   - Few open-source GUI applications available
   - Limited community-driven development
   - Poor integration with modern machine learning tools

#### Project Goals

This project aims to address these challenges by:

1. **Providing Comprehensive Analysis Tools**
   - Implement a full preprocessing pipeline
   - Support both classical and modern classification algorithms
   - Enable custom pipeline configuration

2. **Creating User-Friendly Software**
   - Develop an intuitive GUI for non-programmers
   - Support real-time processing and visualization
   - Ensure cross-platform compatibility

3. **Enabling Research and Clinical Use**
   - Implement explainability features for medical applications
   - Provide detailed result interpretation
   - Support clinical decision-making workflows

### Academic Information

**Student:** Muhamad Helmi bin Rozain (ムハマドヘルミビンロザイン)  
**Student ID:** 12270294  
**Institution:** University of Toyama (富山大学)  
**Laboratory:** [Clinical Photonics and Information Engineering](http://www3.u-toyama.ac.jp/medphoto/)

**Supervisors:**
- 大嶋　佑介 (Yusuke Oshima)
- 竹谷　皓規 (Hironori Taketani)

### Key Features

- ✅ **40+ Preprocessing Methods** - Research-validated algorithms
- ✅ **Real-Time Analysis** - Interactive visualization and classification
- ✅ **Modern GUI** - Intuitive PySide6/Qt6 interface
- ✅ **Multi-Language** - English and Japanese support
- ✅ **Open Source** - MIT License for academic and commercial use
- ✅ **Cross-Platform** - Windows, macOS, and Linux support

---

## Installation

### System Requirements

**Minimum Requirements:**
- **OS:** Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python:** 3.8 or higher
- **RAM:** 4 GB (8 GB recommended for large datasets)
- **Storage:** 500 MB free space
- **Display:** 1280x720 resolution (1920x1080 recommended)

**Optional Requirements:**
- **GPU:** NVIDIA GPU with CUDA support (for deep learning features)
- **Spectrometer:** Andor-compatible spectrometer (for real-time acquisition)

### Installation Methods

#### Method 1: Source Installation (Recommended for Developers)

```bash
# 1. Clone the repository
git clone https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application.git
cd Raman-Spectroscopy-Analysis-Application

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
python main.py
```

#### Method 2: Using UV Package Manager (Recommended for Users)

```bash
# 1. Install UV package manager
pip install uv

# 2. Clone and navigate to repository
git clone https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application.git
cd Raman-Spectroscopy-Analysis-Application

# 3. Create environment and install dependencies
uv venv
uv pip install -e .

# 4. Run the application
uv run python main.py
```

#### Method 3: Portable Executable (Windows Only)

For clinical deployment without Python installation:

1. Download the latest portable executable from [Releases](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/releases)
2. Extract the ZIP file to your desired location
3. Run `RamanApp.exe`

**Features:**
- ✅ No installation required
- ✅ All dependencies bundled
- ✅ Single executable file (375 MB)
- ✅ Portable - run from USB drive

#### Method 4: Installer (Windows Only)

For permanent installation on Windows:

1. Download the installer from [Releases](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/releases)
2. Run the `.exe` installer
3. Follow the installation wizard
4. Launch from Start Menu or Desktop shortcut

**Features:**
- ✅ Professional installation experience
- ✅ Start Menu integration
- ✅ File association for project files
- ✅ Easy uninstallation

### Verifying Installation

After installation, verify everything works:

```bash
# Run a quick test
python -c "import PySide6; print('PySide6 OK')"
python -c "import ramanspy; print('RamanSPy OK')"
python -c "import numpy; print('NumPy OK')"

# Or run the application
python main.py
```

---

## Getting Started

### First Launch

When you first launch the application:

1. **Language Selection**
   - Choose your preferred language (English/Japanese)
   - Can be changed later in settings

2. **Welcome Screen**
   - Overview of main features
   - Quick tutorial option

3. **Create Your First Project**
   - Click "New Project" button
   - Enter project name and description
   - Choose project location

<div align="center">
  <img src="images/first-launch.png" alt="First launch screen" width="700"/>
</div>

### Basic Workflow

#### 1. Create or Open Project

```
Home Page → New Project
- Enter project name (e.g., "MGUS Classification")
- Add description (optional)
- Choose save location
- Click "Create"
```

#### 2. Load Spectral Data

```
Data Package Page → Import Data
- Supported formats: CSV, Excel, TXT, .spc
- Single file or batch import
- Automatic format detection
```

#### 3. Preprocess Spectra

```
Preprocessing Page
- Add preprocessing steps to pipeline
- Configure parameters for each step
- Preview results in real-time
- Export processed data
```

#### 4. Analyze Results

```
Analysis Page
- Choose analysis method (PCA, clustering, etc.)
- Select datasets to analyze
- View interactive visualizations
- Export results
```

### Quick Example: Preprocessing a Single Spectrum

```python
# This workflow is done through the GUI, but here's the conceptual flow:

1. Load Data
   - File → Import → Select "sample.csv"
   
2. Add Preprocessing Steps
   - Click "+" button
   - Select "Baseline Correction" → "ASLS"
   - Set lambda=1e6, p=0.05
   
3. Add More Steps
   - Click "+" again
   - Select "Normalization" → "Vector Norm"
   - Set norm_type="L2"
   
4. Preview Results
   - See before/after comparison automatically
   - Adjust parameters if needed
   
5. Apply and Export
   - Click "Apply to All"
   - Export → "Save Processed Data"
```

---

## Features

### 1. Preprocessing Pipeline

#### Overview

The preprocessing pipeline allows you to chain multiple processing methods together, with real-time preview and parameter adjustment.

<div align="center">
  <img src="images/preprocessing-pipeline-detail.png" alt="Preprocessing pipeline" width="750"/>
</div>

#### Available Categories

**Baseline Correction**
- **ASLS** (Asymmetric Least Squares)
- **Polynomial Baseline**
- **IASLS** (Improved ASLS)
- **Butterworth High-Pass Filter**

**Normalization**
- **Vector Normalization** (L1, L2, Max)
- **MinMax Scaling**
- **Z-Score Standardization**
- **Quantile Normalization**
- **Probabilistic Quotient Normalization (PQN)**
- **Rank Transform**

**Smoothing & Derivatives**
- **Savitzky-Golay Smoothing**
- **Savitzky-Golay Derivatives** (1st, 2nd order)
- **Moving Average**
- **Gaussian Filter**

**Feature Engineering**
- **Peak-Ratio Features** (for MGUS/MM classification)
- **Peak Detection and Integration**
- **Spectral Binning**

**Deep Learning**
- **Convolutional Autoencoder (CDAE)**
- Unified denoising and baseline removal

**Advanced Methods**
- **Cosmic Ray Removal**
- **Spike Detection**
- **Noise Reduction**

#### Pipeline Features

- **Drag-and-Drop Reordering** - Change step order easily
- **Enable/Disable Steps** - Toggle steps without removing them
- **Parameter Persistence** - Settings saved automatically
- **Batch Processing** - Apply pipeline to multiple spectra
- **Export Pipeline** - Save and share pipeline configurations

### 2. Analysis Methods

#### Exploratory Analysis

**Principal Component Analysis (PCA)**
- Dimensionality reduction
- Variance explanation
- Biplot visualization
- Loadings analysis
- Score distributions with statistical tests

<div align="center">
  <img src="images/pca-analysis.png" alt="PCA analysis results" width="700"/>
</div>

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- Non-linear dimensionality reduction
- Cluster visualization
- Perplexity optimization

**UMAP (Uniform Manifold Approximation and Projection)**
- Modern dimensionality reduction
- Faster than t-SNE
- Preserves both local and global structure

#### Clustering Analysis

**K-Means Clustering**
- Automatic cluster number selection (elbow method)
- Silhouette analysis
- Cluster visualization

**Hierarchical Clustering**
- Dendrogram visualization
- Multiple linkage methods
- Cut-off optimization

#### Statistical Analysis

**ANOVA Testing (not working well yet)**
- One-way and two-way ANOVA
- Post-hoc tests (Tukey HSD)
- Effect size calculation

**Correlation Analysis**
- Pearson and Spearman correlation
- Heatmap visualization
- Significance testing

**Mann-Whitney U Test (not working well yet)**
- Non-parametric comparison
- Effect size (Cohen's d)
- Confidence intervals

### 3. Visualization Tools

#### Interactive Plots

All plots are interactive with:
- **Zoom** - Mouse wheel or box selection
- **Pan** - Click and drag
- **Export** - Save as PNG, SVG, PDF
- **Customize** - Colors, markers, labels

<div align="center">
  <img src="images/interactive-plots.png" alt="Interactive plotting features" width="650"/>
</div>

#### Plot Types

- **Line Plots** - Spectral overlays
- **Scatter Plots** - PCA/t-SNE results
- **Heatmaps** - Correlation matrices
- **Box Plots** - Statistical comparisons
- **Violin Plots** - Distribution visualization
- **Dendrograms** - Hierarchical clustering

### 4. Project Management

#### Project Structure

```
MyProject/
├── project.json          # Project metadata
├── raw_data/            # Original spectral files
├── processed_data/      # Preprocessed spectra
├── pipelines/           # Saved preprocessing pipelines
├── analyses/            # Analysis results and figures
└── exports/             # Exported data and reports
```

#### Features

- **Recent Projects** - Quick access sidebar
- **Auto-Save** - Automatic project state saving
- **Version Control** - Track changes to pipelines
- **Export/Import** - Share projects with collaborators

---

## User Interface Guide

### Main Window Layout

<div align="center">
  <img src="images/ui-layout-annotated.png" alt="User interface layout" width="800"/>
</div>

#### 1. Navigation Bar (Left Side)

- **Home** - Project management
- **Data Package** - Import and organize data
- **Preprocessing** - Configure processing pipeline
- **Analysis** - Run analysis methods
- **Workspace** - View all project files
- **Settings** - Application preferences

#### 2. Main Content Area (Center)

Displays the active page content:
- Data tables
- Interactive plots
- Parameter controls
- Results visualization

#### 3. Sidebar (Right Side)

Context-sensitive information:
- Recent projects (Home page)
- Dataset list (Preprocessing page)
- Analysis history (Analysis page)
- Parameter hints

#### 4. Status Bar (Bottom)

- Current project name
- Processing status
- Memory usage
- Error notifications

### Home Page

#### Overview

<div align="center">
  <img src="images/home-page.png" alt="Home page" width="700"/>
</div>

#### Actions

**Create New Project**
1. Click "New Project" button
2. Enter project details
3. Choose location
4. Click "Create"

**Open Existing Project**
1. Click on recent project in sidebar
2. Or use "Open Project" → Browse
3. Project loads automatically

**Project Cards**
- Display project metadata
- Show last modified date
- Quick actions (Open, Delete, Export)

### Data Package Page

#### Overview

Import and manage spectral data files.

<div align="center">
  <img src="images/data-package-page.png" alt="Data package page" width="700"/>
</div>

#### Features

**Import Data**
- Single file import
- Batch folder import
- Drag-and-drop support
- Format auto-detection

**Supported Formats**
- CSV (comma-separated values)
- Excel (.xlsx, .xls)
- Plain text (.txt)
- SPC (binary spectroscopy format)
- Custom formats (via plugins)

**Data Organization**
- Group datasets by type
- Tag and label files
- Add metadata
- Create subsets

### Preprocessing Page

#### Overview

Build and execute preprocessing pipelines with real-time preview.

<div align="center">
  <img src="images/preprocessing-page-full.png" alt="Preprocessing page" width="800"/>
</div>

#### Layout Sections

**1. Input Datasets (Top Left)**
- Select datasets to process
- Multi-selection support
- Preview selected data

**2. Preprocessing Pipeline (Bottom Left)**
- Add steps with "+" button
- Drag to reorder
- Enable/disable with eye icon
- Remove with "×" button

**3. Parameters (Center)**
- Dynamic parameter widgets
- Real-time validation
- Research-based hints
- Save/load presets

**4. Preview (Right)**
- Before/after comparison
- Real-time updates
- Zoom and pan
- Multiple view modes

#### Building a Pipeline

**Step 1: Add Methods**
```
Click "+" → Select Category → Choose Method
```

**Step 2: Configure Parameters**
```
Adjust sliders, inputs, and dropdowns
See instant preview updates
```

**Step 3: Reorder if Needed**
```
Drag steps to change order
Order matters! (e.g., baseline before normalization)
```

**Step 4: Test and Refine**
```
Toggle steps on/off to compare
Adjust parameters while watching preview
```

**Step 5: Apply**
```
"Apply to Selected" → Process chosen datasets
"Apply to All" → Process entire project
"Export Pipeline" → Save configuration
```

### Analysis Page

#### Overview

Perform multivariate analysis and classification.

<div align="center">
  <img src="images/analysis-page-full.png" alt="Analysis page" width="800"/>
</div>

#### Workflow

**1. Select Method Category**
- Exploratory (PCA, t-SNE, UMAP)
- Clustering (K-Means, Hierarchical)
- Statistical (ANOVA, Correlation)
- Classification (coming soon)

**2. Choose Datasets**
- Select one or more datasets
- Option to create groups
- Define control/test groups

**3. Configure Parameters**
- Method-specific settings
- Validation options
- Visualization preferences

**4. Run Analysis**
- Click "Start Analysis"
- Progress indicator shown
- Results appear in tabs

**5. View Results**
- Multiple result tabs
- Interactive plots
- Statistical summaries
- Export options

---

## Preprocessing Methods

### Baseline Correction

#### ASLS (Asymmetric Least Squares)

**Purpose:** Remove baseline fluorescence while preserving peaks.

**Parameters:**
- `lambda` (λ): Smoothness (1e3 to 1e10)
  - Lower = follows data closely
  - Higher = smoother baseline
  - Typical: 1e6 for Raman spectra

- `p`: Asymmetry (0.001 to 0.1)
  - Controls peak vs. baseline weighting
  - Typical: 0.05 for biological samples

**Use Cases:**
- Biological samples with strong fluorescence
- Mineral samples with broad backgrounds
- Any sample with non-flat baseline

**Mathematical Background:**
```
Minimize: ||y - z||² + λ||D²z||²
Subject to: weights w_i based on residuals
```

**References:**
- Eilers & Boelens (2005) "Baseline correction with asymmetric least squares smoothing"

#### Butterworth High-Pass Filter

**Purpose:** Remove baseline using frequency-domain filtering.

**Parameters:**
- `cutoff_freq`: Cutoff frequency (0.001 to 0.5 Hz)
  - Lower = removes only very broad baselines
  - Higher = removes more baseline features
  - Typical: 0.01 Hz

- `order`: Filter order (1 to 10)
  - Higher = sharper cutoff
  - Typical: 4

**Use Cases:**
- Spectra with very broad, smooth baselines
- When ASLS is too slow for large datasets
- When digital filtering is preferred

**Advantages:**
- Fast computation (IIR filter)
- Sharp frequency cutoff
- No iterative fitting required

### Normalization

#### Probabilistic Quotient Normalization (PQN)

**Purpose:** Correct for sample dilution effects.

**Parameters:**
- `reference`: Reference spectrum
  - "median": Use median spectrum (recommended)
  - "mean": Use mean spectrum
  - Custom: Provide specific spectrum

**Algorithm:**
```python
1. Choose reference spectrum
2. Calculate quotients: q_i = sample_i / reference_i
3. Find median quotient: Q = median(q_i)
4. Normalize: normalized = sample / Q
```

**Use Cases:**
- Biofluid analysis (blood, urine)
- Cell culture samples with variable density
- Any samples with dilution variation

**References:**
- Dieterle et al. (2006) "Probabilistic quotient normalization as robust method to account for dilution of complex biological mixtures"

#### Quantile Normalization

**Purpose:** Align intensity distributions across samples.

**Parameters:**
- `n_quantiles`: Number of quantiles (10 to 1000)
  - More quantiles = finer alignment
  - Typical: 100

- `reference`: Reference distribution
  - "median": Median across all samples
  - "mean": Mean across all samples

**Algorithm:**
```python
1. Sort each spectrum
2. Calculate reference quantiles
3. Map each spectrum to reference
4. Restore original order
```

**Use Cases:**
- Cross-platform normalization
- Batch effect removal
- Large-scale studies with multiple instruments

**References:**
- Bolstad et al. (2003) "A comparison of normalization methods for high density oligonucleotide array data"

### Feature Engineering

#### Peak-Ratio Features

**Purpose:** Extract discriminative peak ratios for classification.

**Parameters:**
- `peak_indices`: Wavenumber regions of interest
  - Custom list: e.g., [1000, 1200, 1400]
  - Auto: Detect peaks automatically

- `extraction_method`:
  - "local_max": Maximum intensity
  - "local_integral": Integrate peak area
  - "gaussian_fit": Fit Gaussian and use amplitude

- `ratio_type`:
  - "all_pairs": All possible ratios (n² features)
  - "sequential": Adjacent peak ratios
  - "relative_to_first": All ratios to first peak

**Mathematical Background:**
```
Feature_ij = Peak_i / Peak_j

For n peaks:
- all_pairs: n(n-1)/2 features
- sequential: n-1 features
- relative_to_first: n-1 features
```

**Use Cases:**
- MGUS vs. MM classification
- Cancer detection
- Material phase identification
- Any classification task with known peak regions

**Example:**
```
Peaks at: 1000, 1200, 1400 cm⁻¹

all_pairs ratios:
- I₁₀₀₀/I₁₂₀₀
- I₁₀₀₀/I₁₄₀₀
- I₁₂₀₀/I₁₄₀₀

These ratios often show better class separation
than raw intensities.
```

**References:**
- Deeley et al. (2010) "Using Raman spectroscopy to elucidate the pathogenesis of multiple myeloma"

### Deep Learning

#### Convolutional Autoencoder (CDAE)

**Purpose:** Unified denoising and baseline removal using deep learning.

**Architecture:**
```
Encoder: Conv1D → ReLU → MaxPool → ... → Latent Space
Decoder: Conv1D → ReLU → Upsample → ... → Reconstructed
```

**Parameters:**
- `latent_dim`: Latent space dimensions (16 to 256)
- `n_layers`: Number of convolutional layers (2 to 5)
- `kernel_size`: Convolution kernel size (3 to 15)
- `learning_rate`: Training learning rate (1e-5 to 1e-2)
- `num_epochs`: Training epochs (10 to 200)
- `batch_size`: Training batch size (8 to 128)

**Training Process:**
```python
1. Split data into train/validation
2. Train autoencoder to reconstruct clean spectra
3. Encoder compresses to latent space
4. Decoder reconstructs denoised signal
5. Validate on held-out data
```

**Use Cases:**
- Highly noisy spectra
- Complex baseline shapes
- Large datasets (>1000 spectra)
- When traditional methods fail

**Requirements:**
- PyTorch installed
- GPU recommended (optional)
- Training data (100+ spectra)

**Advantages:**
- Learns optimal preprocessing from data
- Handles complex noise patterns
- No manual parameter tuning after training

**References:**
- Vincent et al. (2010) "Stacked denoising autoencoders"

---

## Analysis Methods

### Principal Component Analysis (PCA)

#### Overview

PCA reduces high-dimensional spectral data to a few principal components that capture the most variance.

<div align="center">
  <img src="images/pca-explained.png" alt="PCA explanation" width="700"/>
</div>

#### Parameters

**n_components:** Number of principal components
- Default: 3
- Range: 1 to min(n_samples, n_features)
- Recommendation: Start with 2-3 for visualization

**scaling:** Data scaling method
- "StandardScaler": Mean=0, Std=1 (recommended)
- "MinMaxScaler": Scale to [0, 1]
- "RobustScaler": Robust to outliers

**show_ellipses:** Confidence ellipses
- true: Show 95% confidence ellipses
- false: Show points only

**show_loadings:** Show loadings plot
- true: Display PC loadings
- false: Skip loadings

**show_scree:** Show scree plot
- true: Display variance explained
- false: Skip scree plot

**n_distribution_components:** Number of PCs for distribution plots
- Default: 3
- Shows statistical tests between groups

#### Interpretation

**Score Plot:**
- Each point = one spectrum
- Distance = similarity
- Clusters = groups with similar spectra
- Confidence ellipses = 95% CI

**Loadings Plot:**
- Shows which wavenumbers contribute to each PC
- Peaks = important spectral features
- Sign indicates positive/negative correlation

**Scree Plot:**
- Shows variance explained by each PC
- "Elbow" indicates optimal number of PCs
- Cumulative variance should be >80%

**Distribution Plots:**
- Shows PC score distributions for each group
- Statistical tests (Mann-Whitney U)
- Effect size (Cohen's d)
- Helps assess group separation

#### Example Use Case

**MGUS vs. MM Classification:**
```
1. Load preprocessed spectra (baseline + norm)
2. Select MGUS and MM groups
3. Run PCA with n_components=3
4. Observe:
   - PC1 separates groups (variance = 45%)
   - PC2 shows within-group variation (20%)
   - Mann-Whitney p < 0.001 (strong separation)
5. Export loadings to identify discriminative peaks
```

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

#### Overview

t-SNE creates a 2D or 3D visualization where similar spectra are grouped together, preserving local structure.

#### Parameters

**perplexity:** Balance between local and global structure
- Default: 30
- Range: 5 to 50
- Lower = emphasizes local structure
- Higher = emphasizes global structure

**n_iter:** Number of iterations
- Default: 1000
- More iterations = better convergence

**learning_rate:** Optimization learning rate
- Default: 200
- Range: 10 to 1000

#### Advantages Over PCA

- Non-linear dimensionality reduction
- Better cluster visualization
- Preserves local neighborhood structure

#### Disadvantages

- Slower than PCA
- Non-deterministic (results vary between runs)
- Distances between clusters not meaningful

#### Use Cases

- Visualizing complex, non-linear relationships
- Cluster detection
- Exploratory analysis of large datasets

### UMAP (Uniform Manifold Approximation and Projection)

#### Overview

UMAP is a modern alternative to t-SNE, faster and better at preserving both local and global structure.

#### Parameters

**n_neighbors:** Local neighborhood size
- Default: 15
- Range: 2 to 100
- Smaller = local structure
- Larger = global structure

**min_dist:** Minimum distance between points
- Default: 0.1
- Range: 0.0 to 0.99
- Smaller = tighter clusters
- Larger = more spread out

**metric:** Distance metric
- "euclidean": Standard distance
- "cosine": Angular similarity
- "manhattan": City-block distance

#### Advantages

- Faster than t-SNE
- Preserves both local and global structure
- More consistent results

#### Use Cases

- Large datasets (>10,000 spectra)
- When global relationships matter
- Production pipelines (more stable)

---

## Development

### Setting Up Development Environment

#### Prerequisites

- Python 3.8 or higher
- Git
- Text editor or IDE (VS Code, PyCharm recommended)

#### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application.git
cd Raman-Spectroscopy-Analysis-Application

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Project Structure

```
raman-app/
├── .AGI-BANKS/              # AI agent knowledge base
├── .docs/                   # Detailed documentation
├── assets/                  # Static resources
│   ├── fonts/              # Custom fonts
│   ├── icons/              # UI icons
│   ├── images/             # Splash screens, logos
│   └── locales/            # Translation files
│       ├── en.json
│       └── ja.json
├── build_scripts/          # Build and packaging scripts
├── components/             # Reusable UI components
│   └── widgets/           # Custom widgets
├── configs/                # Configuration management
│   └── style/             # Stylesheets
├── functions/              # Core processing logic
│   ├── preprocess/        # Preprocessing methods
│   ├── visualization/     # Plotting utilities
│   └── andorsdk/         # Hardware integration
├── pages/                  # Application pages
│   ├── home_page.py
│   ├── data_package_page.py
│   ├── preprocess_page.py
│   └── analysis_page.py
├── tests/                  # Test suite
├── main.py                # Application entry point
├── pyproject.toml         # Project metadata and dependencies
└── README.md              # This file
```

### Coding Standards

Follow the guidelines in [.AGI-BANKS/DEVELOPMENT_GUIDELINES.md](../.AGI-BANKS/DEVELOPMENT_GUIDELINES.md):

- **PEP 8** style guide
- **Type hints** for function signatures
- **Docstrings** for all public functions
- **Comments** for complex logic
- **Logging** instead of print statements

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py

# Run specific test
pytest tests/test_preprocessing.py::test_asls_baseline
```

### Building Executables

#### Windows Portable Executable

```bash
# Run build script
cd build_scripts
.\build_portable.ps1

# Output: dist/raman_app.exe
```

#### Windows Installer

```bash
# Run installer build script
cd build_scripts
.\build_installer.ps1

# Requires NSIS installed
# Output: dist/RamanApp-Setup.exe
```

### Adding New Preprocessing Methods

#### Step 1: Create Method Class

```python
# functions/preprocess/your_method.py

from .base import PreprocessingMethod
import numpy as np

class YourMethodFixed(PreprocessingMethod):
    """
    Your method description.
    
    Parameters
    ----------
    param1 : float
        Description of param1
    param2 : int
        Description of param2
    """
    
    def __init__(self, param1: float = 1.0, param2: int = 10):
        self.param1 = float(param1)
        self.param2 = int(param2)
    
    def fit(self, X: np.ndarray, y=None):
        """Fit method (if needed)."""
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply transformation."""
        # Your implementation here
        result = X * self.param1  # Example
        return result
```

#### Step 2: Register Method

```python
# functions/preprocess/__init__.py

from .your_method import YourMethodFixed

PREPROCESSING_METHODS = {
    "your_category": {
        "Your Method": {
            "class": YourMethodFixed,
            "parameters": [
                {
                    "name": "param1",
                    "type": "float",
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "description": "Description"
                },
                {
                    "name": "param2",
                    "type": "int",
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "description": "Description"
                }
            ]
        }
    }
}
```

#### Step 3: Add Tests

```python
# tests/test_your_method.py

import numpy as np
from functions.preprocess.your_method import YourMethodFixed

def test_your_method():
    """Test your method."""
    # Create test data
    X = np.random.randn(100, 1000)
    
    # Create and apply method
    method = YourMethodFixed(param1=2.0, param2=20)
    method.fit(X)
    result = method.transform(X)
    
    # Assertions
    assert result.shape == X.shape
    assert not np.any(np.isnan(result))
    # Add more specific tests
```

---

## Contributing

We welcome contributions from the research community!

### How to Contribute

#### 1. Fork the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Raman-Spectroscopy-Analysis-Application.git
cd Raman-Spectroscopy-Analysis-Application
```

#### 2. Create a Branch

```bash
# Feature branch
git checkout -b feature/your-feature-name

# Bug fix branch
git checkout -b fix/bug-description

# Documentation branch
git checkout -b docs/what-you-are-documenting
```

#### 3. Make Your Changes

- Write clear, commented code
- Follow coding standards
- Add tests for new features
- Update documentation

#### 4. Test Your Changes

```bash
# Run tests
pytest

# Run linter
flake8 .

# Check types
mypy .
```

#### 5. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with clear message
git commit -m "feat: add new preprocessing method"

# Follow conventional commits:
# feat: new feature
# fix: bug fix
# docs: documentation
# test: test additions
# refactor: code refactoring
```

#### 6. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Go to GitHub and create Pull Request
# Describe your changes clearly
# Reference any related issues
```

### Contribution Guidelines

#### Code Review Process

1. **Automated Checks**
   - Tests must pass
   - Code coverage maintained
   - Linting passes

2. **Manual Review**
   - Code quality
   - Documentation completeness
   - Design patterns followed

3. **Approval and Merge**
   - At least one maintainer approval required
   - Squash and merge preferred

#### What to Contribute

**High Priority:**
- 🐛 Bug fixes
- 📖 Documentation improvements
- 🧪 Test coverage
- 🌍 Translations

**Medium Priority:**
- ✨ New preprocessing methods
- 📊 New analysis methods
- 🎨 UI improvements

**Welcome:**
- 🔬 Research validations
- 📝 Tutorial creation
- 🎓 Educational materials
- 💡 Feature suggestions

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on technical merits
- Help newcomers
- Follow project guidelines

---

## Troubleshooting

### Installation Issues

#### Problem: "ModuleNotFoundError: No module named 'PySide6'"

**Solution:**
```bash
pip install PySide6
# Or reinstall all dependencies
pip install -r requirements.txt
```

#### Problem: "ImportError: DLL load failed" (Windows)

**Solution:**
- Install Visual C++ Redistributable
- Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

#### Problem: "Permission denied" when installing

**Solution:**
```bash
# Use virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Or install with --user flag
pip install --user -r requirements.txt
```

### Runtime Issues

#### Problem: Application crashes on startup

**Solutions:**
1. Check log files in `logs/` directory
2. Run in debug mode:
   ```bash
   python main.py --debug
   ```
3. Check for conflicting Qt installations
4. Verify Python version >= 3.8

#### Problem: "OpenGL errors" or graphics issues

**Solutions:**
1. Update graphics drivers
2. Set environment variable:
   ```bash
   # Windows PowerShell
   $env:QT_OPENGL="software"
   python main.py
   
   # Linux/macOS
   export QT_OPENGL=software
   python main.py
   ```
3. Use software rendering in settings

#### Problem: Slow preprocessing with large datasets

**Solutions:**
1. Enable parallel processing in settings
2. Reduce preview update frequency
3. Process in batches
4. Use faster methods (e.g., Butterworth instead of ASLS)

### Data Import Issues

#### Problem: "Unsupported file format"

**Solutions:**
1. Check file extension (.csv, .xlsx, .txt, .spc)
2. Verify file is not corrupted
3. Try converting to CSV format
4. Check file encoding (UTF-8 recommended)

#### Problem: "Data format not recognized"

**Solutions:**
1. Ensure first column contains wavenumbers
2. Each subsequent column should be one spectrum
3. No missing values or non-numeric data
4. Header row should contain dataset names

### Analysis Issues

#### Problem: PCA returns all NaN values

**Solutions:**
1. Check for NaN or Inf values in input data
2. Remove constant features (zero variance)
3. Apply appropriate preprocessing (baseline, norm)
4. Ensure sufficient number of samples

#### Problem: Memory error with large datasets

**Solutions:**
1. Process subsets of data
2. Increase virtual memory/swap
3. Use dimensionality reduction first
4. Consider using streaming processing

### Getting Help

If you encounter issues not listed here:

1. **Check Documentation**
   - [English docs](README_EN.md)
   - [Japanese docs](README_JA.md)

2. **Search Existing Issues**
   - [GitHub Issues](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues)

3. **Create New Issue**
   - Provide Python version
   - Include error messages
   - Describe steps to reproduce
   - Attach log files if possible

4. **Ask in Discussions**
   - [GitHub Discussions](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/discussions)

---

## API Reference

### Core Modules

#### `functions.preprocess`

**BaselineCorrection Methods:**
```python
from functions.preprocess import ASLSBaseline

# ASLS baseline correction
asls = ASLSBaseline(lam=1e6, p=0.05)
asls.fit(X)
corrected = asls.transform(X)
```

**Normalization Methods:**
```python
from functions.preprocess import (
    VectorNormalization,
    QuantileNormalization,
    PQNormalization
)

# Vector normalization
vn = VectorNormalization(norm='L2')
normalized = vn.fit_transform(X)

# Quantile normalization
qn = QuantileNormalization(n_quantiles=100)
normalized = qn.fit_transform(X)

# PQN
pqn = PQNormalization(reference='median')
normalized = pqn.fit_transform(X)
```

**Feature Engineering:**
```python
from functions.preprocess import PeakRatioFeatures

# Extract peak ratios
prf = PeakRatioFeatures(
    peak_indices=[1000, 1200, 1400],
    extraction_method='local_max',
    ratio_type='all_pairs'
)
features = prf.fit_transform(X, wavenumbers)
```

#### `functions.visualization`

**Plotting Functions:**
```python
from functions.visualization import (
    plot_spectra,
    plot_pca,
    plot_heatmap
)

# Plot multiple spectra
fig = plot_spectra(
    X, 
    wavenumbers,
    labels=['Sample 1', 'Sample 2'],
    title='Raman Spectra'
)

# PCA visualization
fig = plot_pca(
    pca_model,
    labels,
    show_ellipses=True
)

# Correlation heatmap
fig = plot_heatmap(
    correlation_matrix,
    labels,
    cmap='coolwarm'
)
```

### GUI Components

#### `components.widgets.ParameterWidget`

```python
from components.widgets import ParameterWidget

# Create parameter widget
widget = ParameterWidget(
    name='lambda',
    param_type='float',
    default=1e6,
    min_val=1e3,
    max_val=1e10,
    scientific=True
)

# Get current value
value = widget.get_value()

# Set value programmatically
widget.set_value(1e7)

# Connect to signal
widget.value_changed.connect(on_parameter_changed)
```

#### `pages.PreprocessPage`

```python
from pages import PreprocessPage

# Access preprocessing page
preprocess_page = main_window.preprocess_page

# Get current pipeline
pipeline = preprocess_page.get_pipeline()

# Apply pipeline to data
processed = preprocess_page.apply_pipeline(X)

# Export pipeline
preprocess_page.export_pipeline('my_pipeline.json')

# Load pipeline
preprocess_page.load_pipeline('my_pipeline.json')
```

### Configuration

#### `configs.configs`

```python
from configs.configs import (
    load_config,
    LocalizationManager,
    create_logs
)

# Load application configuration
config = load_config('configs/app_configs.json')

# Initialize localization
localize = LocalizationManager(
    locale_dir='assets/locales',
    default_lang='en',
    initial_lang='en'
)

# Get translated string
text = localize.get('PREPROCESS_PAGE.title')

# Create log entry
create_logs(
    log_name='MyModule',
    filename='module_logs',
    log_message='Processing complete',
    status='info'
)
```

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024-2026 Muhamad Helmi bin Rozain
Laboratory for Clinical Photonics and Information Engineering
University of Toyama

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Third-Party Licenses

This software uses the following open-source libraries:

- **PySide6** - LGPL v3
- **NumPy** - BSD License
- **SciPy** - BSD License
- **matplotlib** - PSF License
- **pandas** - BSD License
- **scikit-learn** - BSD License
- **RamanSPy** - BSD License

See [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) for complete details.

---

## Acknowledgments

### Academic Support

**University of Toyama** (富山大学)
- Providing research facilities and computational resources
- Supporting open-source development in academia

**Laboratory for Clinical Photonics and Information Engineering**
- Website: http://www3.u-toyama.ac.jp/medphoto/
- Guidance on clinical applications
- Access to spectroscopy equipment
- Research collaboration

**Supervisors:**
- **大嶋　佑介** (Yusuke Oshima) - Technical guidance and project supervision
- **竹谷　皓規** (Hironori Taketani) - Clinical insights and validation

### Open Source Community

**Framework and Libraries:**
- Qt Company for PySide6/Qt framework
- NumPy and SciPy communities
- matplotlib development team
- scikit-learn contributors
- RamanSPy developers

**Research Community:**
- Authors of cited papers and methods
- Open-access journal publishers
- Dataset contributors

### Contributors

See [CONTRIBUTORS.md](../CONTRIBUTORS.md) for list of project contributors.

---

## Citation

If you use this software in your research, please cite:

### Software Citation

```bibtex
@software{helmi2024raman,
  author = {Rozain, Muhamad Helmi bin},
  title = {Raman Spectroscopy Analysis Application: 
           Real-Time Classification Software for Disease Detection},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application},
  version = {1.0.0},
  institution = {University of Toyama, 
                 Laboratory for Clinical Photonics and Information Engineering}
}
```

### Academic Paper (If Applicable)

```bibtex
@article{helmi2024raman_paper,
  author = {Rozain, Muhamad Helmi bin and Oshima, Yusuke and Taketani, Hironori},
  title = {Open-Source Software for Real-Time Raman Spectroscopy Classification 
           and Disease Detection},
  journal = {Journal Name},
  year = {2024},
  volume = {XX},
  pages = {XX-XX},
  doi = {XX.XXXX/XXXXXX}
}
```

---

## Contact

### Project Maintainer

**Muhamad Helmi bin Rozain** (ムハマドヘルミビンロザイン)  
Student ID: 12270294  
University of Toyama  
Laboratory for Clinical Photonics and Information Engineering

**Contact Methods:**
- GitHub: [@zerozedsc](https://github.com/zerozedsc)
- Email: [Contact via GitHub](https://github.com/zerozedsc)
- Laboratory: http://www3.u-toyama.ac.jp/medphoto/

### Supervisors

**大嶋　佑介** (Yusuke Oshima)  
Laboratory for Clinical Photonics and Information Engineering

**竹谷　皓規** (Hironori Taketani)  
Laboratory for Clinical Photonics and Information Engineering

### Reporting Issues

**Bug Reports:**
- [GitHub Issues](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues)
- Include: Python version, OS, error message, steps to reproduce

**Feature Requests:**
- [GitHub Discussions](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/discussions)
- Describe: Use case, expected behavior, potential implementation

**Security Issues:**
- Contact maintainer directly
- Do not post publicly until fixed

### Community

**Discussions:** [GitHub Discussions](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/discussions)  
**Wiki:** [GitHub Wiki](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/wiki)  
**Updates:** Watch repository for new releases

---

## Changelog

See [CHANGELOG.md](../CHANGELOG.md) for detailed version history.

### Latest Version: 1.0.0 (January 2026)

**Major Features:**
- ✨ Complete preprocessing pipeline (40+ methods)
- ✨ Real-time analysis and visualization
- ✨ Multi-language support (EN/JA)
- ✨ Project management system
- ✨ Portable executable distribution

**Recent Updates:**
- 🔧 Multi-group dialog theme refactor
- 🔧 Spectrum preview tab fixes
- 🔧 Matplotlib legend crash fix
- 🔧 Console logging improvements
- 📖 Comprehensive bilingual documentation

---

<div align="center">
  <p><strong>Thank you for using the Raman Spectroscopy Analysis Application!</strong></p>
  <p>Developed with ❤️ for the scientific and medical research community</p>
  <p>
    <a href="http://www3.u-toyama.ac.jp/medphoto/">Laboratory for Clinical Photonics and Information Engineering</a> •
    <a href="https://www.u-toyama.ac.jp/">University of Toyama</a>
  </p>
  <p><strong>富山大学 臨床フォトニクスおよび情報工学研究室</strong></p>
</div>
