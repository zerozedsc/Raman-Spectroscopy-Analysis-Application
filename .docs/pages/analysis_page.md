# Analysis Page - Comprehensive Documentation

> **Complete guide to the Analysis Page interface and functionality**  
> **Created**: December 28, 2024 | **Status**: Complete Implementation  
> **Last Major Update**: December 28, 2024 - Visual Styling & Localization Fixes

## Recent Updates (December 28, 2024)

### Visual Design Improvements
- ✅ Added **hint buttons** (blue themed, 20x20px) to all section headers:
  - Dataset Selection hint button with tooltip
  - Method Selection hint button with comprehensive tooltip
  - Parameters hint button explaining dynamic parameter system
  - Quick Stats hint button describing displayed information
- ✅ Implemented **title bar widgets** matching preprocess page design pattern
- ✅ Added **refresh button** with hover effects to dataset selection
- ✅ Applied consistent **label styling** (font-weight, colors, sizes)
- ✅ Matched **spacing and margins** with preprocess page standards

### Localization Completeness
- ✅ Added 25+ missing translation keys to both English and Japanese locales
- ✅ All warnings resolved - no missing translation keys
- ✅ Complete coverage for all UI elements and tooltips
- ✅ Added `quick_stats_hint` key explaining statistics display

### Implementation Verification
- ✅ All 14 analysis methods implemented and tested:
  - 5 Exploratory methods (PCA, UMAP, t-SNE, Hierarchical Clustering, K-means)
  - 4 Statistical methods (Spectral Comparison, Peak Analysis, Correlation, ANOVA)
  - 5 Visualization methods (Heatmap, Overlay, Waterfall, Correlation Heatmap, Peak Scatter)
- ✅ All analysis_page_utils modules complete:
  - `result.py` - AnalysisResult dataclass ✅
  - `registry.py` - 15 method definitions ✅
  - `thread.py` - Background analysis threading ✅
  - `widgets.py` - Dynamic parameter widget factory ✅
  - `methods/` folder - All 14 method implementations ✅

### Design Pattern Consistency
The Analysis Page now perfectly matches the Preprocess Page design patterns:
- **Hint buttons**: 20x20px, blue theme (#e7f3ff background, #0078d4 color)
- **Hover effects**: Reverse colors on hover (#0078d4 background, white color)
- **Action buttons**: 24x24px, transparent background, hover effect with #e7f3ff
- **Title labels**: font-weight 600, font-size 13px, color #2c3e50
- **Secondary labels**: font-weight 500, font-size 11px, color #495057
- **Spacing**: 8-12px between elements, 12px margins

## Table of Contents

1. [Overview](#overview)
2. [User Interface](#user-interface)
3. [Analysis Methods](#analysis-methods)
4. [Usage Workflows](#usage-workflows)
5. [Export Options](#export-options)
6. [Technical Details](#technical-details)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The Analysis Page provides a comprehensive interface for exploring and analyzing Raman spectroscopy data. It supports 15+ analysis methods across three categories: exploratory, statistical, and visualization.

### Key Features

- **Multi-dataset Support**: Analyze one or multiple datasets simultaneously
- **Dataset Filtering**: Filter by Raw, Preprocessed, or All datasets
- **Dynamic Parameters**: Parameters automatically update based on selected method
- **Background Processing**: Threading ensures UI remains responsive during analysis
- **Result Caching**: Avoid recomputation of identical analyses
- **Comprehensive Export**: PNG, SVG, CSV, and full report options

### Design Philosophy

The Analysis Page follows the same design patterns as the Preprocessing Page:
- Left panel (400px fixed) for controls
- Right panel (expanding) for visualizations and data
- Consistent styling and localization
- Real-time progress feedback

---

## User Interface

### Layout Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│  Analysis Page                                                       │
├──────────────────┬──────────────────────────────────────────────────┤
│  Left Panel      │  Right Panel                                     │
│  (400px)         │  (Expanding)                                     │
│                  │                                                  │
│  📊 Dataset      │  📈 Primary Visualization                        │
│     Selection    │     - Main plot/chart                           │
│                  │                                                  │
│  🔬 Method       │  📊 Secondary Visualization                      │
│     Selection    │     - Additional plots (if applicable)          │
│                  │                                                  │
│  ⚙️  Parameters   │  📄 Data Table                                  │
│                  │     - Tabular results                           │
│  📊 Quick Stats  │                                                  │
│                  │                                                  │
│  ▶️  Run Controls │                                                  │
└──────────────────┴──────────────────────────────────────────────────┘
```

### Left Panel Components

#### 1. Dataset Selection
- **Purpose**: Select one or more datasets for analysis
- **Features**:
  - Multi-select list with checkboxes
  - Filter dropdown (All / Raw / Preprocessed)
  - Select All / Deselect All buttons
  - Visual indication of dataset type (icon)
- **Usage**: Click checkboxes or use Ctrl/Cmd for multiple selection

#### 2. Method Selection
- **Purpose**: Choose analysis category and specific method
- **Components**:
  - Category dropdown (Exploratory / Statistical / Visualization)
  - Method dropdown (dynamically populated based on category)
- **Behavior**: Method list updates when category changes

#### 3. Parameters Section
- **Purpose**: Configure method-specific parameters
- **Features**:
  - Dynamically generated based on selected method
  - Supports spinbox, double spinbox, combo, and checkbox widgets
  - Each parameter has appropriate range and default value
- **Note**: Empty when no method is selected

#### 4. Quick Statistics
- **Displays**:
  - Number of selected datasets
  - Total number of spectra
  - Wavenumber range (if datasets selected)
- **Updates**: Real-time as dataset selection changes

#### 5. Run Controls
- **Buttons**:
  - **Run Analysis**: Execute selected method with current parameters
  - **Cancel**: Stop running analysis
  - **Export Results**: Save analysis outputs
  - **Clear Results**: Remove displayed results
- **Progress Bar**: Shows analysis progress (0-100%)

### Right Panel Tabs

#### Tab 1: Primary Visualization
- Main plot or chart from analysis
- Matplotlib canvas with zoom/pan tools
- Auto-resizes to fit panel

#### Tab 2: Secondary Visualization
- Additional plots (method-dependent)
- Not all methods produce secondary visualizations
- Examples: PCA loadings plot, scree plot, p-value plot

#### Tab 3: Data Table
- Tabular results from analysis
- Sortable columns
- Copy-pasteable data
- Export to CSV functionality

---

## Analysis Methods

### Exploratory Analysis

#### 1. PCA (Principal Component Analysis)

**Purpose**: Reduce dimensionality and explain variance in spectral data.

**Parameters**:
- `n_components` (2-10): Number of principal components to compute
- `scaling`: Data scaling method
  - StandardScaler: Zero mean, unit variance
  - MinMaxScaler: Scale to [0, 1]
  - None: No scaling
- `show_loadings` (bool): Display PC loadings plot
- `show_scree` (bool): Display scree plot (variance explained)

**Outputs**:
- **Primary**: PC1 vs PC2 scores plot (colored by dataset)
- **Secondary**: Loadings plot and/or scree plot
- **Table**: PC scores for all spectra

**Use Cases**:
- Identify major sources of variation
- Detect outliers or sample groupings
- Reduce data for further analysis

**Interpretation**:
- Points close together are similar spectra
- PC1 explains most variance, PC2 second most
- Loadings show which wavenumbers contribute to each PC

---

#### 2. UMAP Projection

**Purpose**: Non-linear dimensionality reduction for visualization.

**Parameters**:
- `n_neighbors` (5-100): Balance local vs global structure
  - Low: Focus on local structure
  - High: Preserve global structure
- `min_dist` (0.0-1.0): Minimum distance between points
  - Low: Tight clusters
  - High: Dispersed points
- `n_components` (2-3): Output dimensions
- `metric`: Distance metric
  - euclidean: Standard Euclidean distance
  - cosine: Cosine similarity
  - manhattan: L1 distance
  - correlation: Pearson correlation

**Outputs**:
- **Primary**: 2D or 3D UMAP embedding (colored by dataset)
- **Table**: UMAP coordinates

**Use Cases**:
- Visualize complex relationships
- Identify non-linear patterns
- Alternative to PCA when linear assumptions fail

**Note**: Requires `umap-learn` package. Install with: `pip install umap-learn`

---

#### 3. t-SNE Projection

**Purpose**: Visualize high-dimensional data in 2D space.

**Parameters**:
- `perplexity` (5-100): Balance local vs global structure
  - Typical range: 5-50
  - Higher for large datasets
- `learning_rate` (10-1000): Optimization speed
  - Too low: Slow convergence
  - Too high: Unstable results
- `n_iter` (250-5000): Number of iterations
  - More iterations = better convergence

**Outputs**:
- **Primary**: 2D t-SNE embedding (colored by dataset)
- **Table**: t-SNE coordinates

**Use Cases**:
- Visualize clusters and groupings
- Explore local structure in data
- Complement PCA analysis

**Caution**: 
- Sensitive to parameter choices
- Not deterministic (use random_state=42 for reproducibility)
- Distances between clusters not meaningful

---

#### 4. Hierarchical Clustering

**Purpose**: Build hierarchy of sample relationships.

**Parameters**:
- `linkage_method`: How to measure cluster distance
  - ward: Minimize within-cluster variance (default)
  - average: Average distance between points
  - complete: Maximum distance between points
  - single: Minimum distance between points
- `distance_metric`: Distance measure
  - euclidean: Standard distance
  - correlation: Based on correlation
  - cosine: Cosine similarity

**Outputs**:
- **Primary**: Dendrogram with sample labels
- Horizontal lines show cluster merges
- Y-axis shows distance at merge

**Use Cases**:
- Identify hierarchical relationships
- Determine optimal number of clusters (cut dendrogram)
- Visualize sample similarity structure

**Interpretation**:
- Branches close together = similar samples
- Height of merge = dissimilarity
- Can cut dendrogram at any height to get clusters

---

#### 5. K-means Clustering

**Purpose**: Partition samples into K distinct clusters.

**Parameters**:
- `n_clusters` (2-10): Number of clusters
- `max_iter` (100-1000): Maximum iterations
- `n_init` (1-20): Number of random initializations
- `show_pca` (bool): Visualize in PCA space

**Outputs**:
- **Primary**: Cluster visualization (PCA projection or bar chart)
- **Table**: Cluster assignments for each spectrum

**Use Cases**:
- Group similar spectra
- Identify sample categories
- Prepare data for supervised learning

**Tips**:
- Use elbow method or silhouette score to choose K
- Multiple runs recommended (n_init > 1)
- PCA visualization helps interpret clusters

---

### Statistical Analysis

#### 6. Spectral Comparison

**Purpose**: Statistically compare two datasets.

**Parameters**:
- `confidence_level` (0.80-0.99): Confidence for significance
  - 0.95 = 95% confidence (common)
- `fdr_correction` (bool): Apply False Discovery Rate correction
- `show_ci` (bool): Show confidence intervals
- `highlight_significant` (bool): Highlight significant regions

**Outputs**:
- **Primary**: Mean spectra with confidence intervals
- **Secondary**: P-value plot (-log10 scale)
- **Table**: Statistics for each wavenumber

**Use Cases**:
- Compare healthy vs disease samples
- Validate preprocessing effects
- Identify spectral biomarkers

**Interpretation**:
- Yellow regions = statistically significant differences
- Higher -log10(p) = more significant
- FDR correction reduces false positives

---

#### 7. Peak Analysis

**Purpose**: Detect and characterize spectral peaks.

**Parameters**:
- `prominence_threshold` (0.01-1.0): Minimum peak prominence
  - Higher = fewer, more prominent peaks
- `width_min` (1-50): Minimum peak width (data points)
- `top_n_peaks` (5-100): Number of peaks to analyze
- `show_assignments` (bool): Annotate peak positions

**Outputs**:
- **Primary**: Spectrum with annotated peaks
- **Secondary**: Peak intensity distribution
- **Table**: Peak positions, intensities, widths

**Use Cases**:
- Identify characteristic peaks
- Compare peak intensities across samples
- Assign peaks to molecular vibrations

**Tips**:
- Adjust prominence for noisy data
- Use on preprocessed/smoothed spectra for best results
- Cross-reference peaks with literature databases

---

#### 8. Correlation Analysis

**Purpose**: Measure similarity between all spectra.

**Parameters**:
- `method`: Correlation method
  - pearson: Linear correlation
  - spearman: Rank correlation (non-linear)
- `show_pvalues` (bool): Display p-value matrix

**Outputs**:
- **Primary**: Correlation heatmap
- **Table**: Full correlation matrix

**Use Cases**:
- Identify replicate quality
- Detect outliers (low correlation with others)
- Understand sample relationships

**Interpretation**:
- Red = positive correlation
- Blue = negative correlation
- Dark red (≈1) = very similar spectra

---

#### 9. ANOVA Test

**Purpose**: Compare three or more datasets statistically.

**Parameters**:
- `alpha` (0.01-0.10): Significance level
  - 0.05 = 5% significance (standard)
- `post_hoc` (bool): Perform post-hoc tests

**Outputs**:
- **Primary**: F-statistic and p-value plots
- **Secondary**: Mean spectra by group
- **Table**: ANOVA results for each wavenumber

**Use Cases**:
- Compare multiple treatment groups
- Identify wavenumbers with significant differences
- Multi-group classification preparation

**Requirements**: At least 3 datasets required

**Interpretation**:
- High F-statistic = larger between-group variance
- Low p-value = significant difference
- Yellow regions = significant wavenumbers

---

### Visualization Methods

#### 10. Spectral Heatmap

**Purpose**: Overview of all spectra in matrix form.

**Parameters**:
- `cluster_rows` (bool): Cluster spectra hierarchically
- `cluster_cols` (bool): Cluster wavenumbers hierarchically
- `colormap`: Color scheme (viridis, plasma, inferno, magma, etc.)
- `normalize` (bool): Normalize each spectrum to [0, 1]
- `show_dendrograms` (bool): Display clustering dendrograms

**Outputs**:
- **Primary**: Heatmap with optional dendrograms

**Use Cases**:
- Visual overview of entire dataset
- Identify patterns across wavenumbers
- Detect data quality issues

**Tips**:
- Normalization recommended for comparing spectra
- Clustering reveals hidden groupings
- Use diverging colormap (RdBu) for centered data

---

#### 11. Mean Spectra Overlay

**Purpose**: Compare mean signatures across datasets.

**Parameters**:
- `show_std` (bool): Display standard deviation bands
- `show_individual` (bool): Plot individual spectra
- `alpha_individual` (0.0-1.0): Transparency for individual spectra
- `normalize` (bool): Normalize each dataset

**Outputs**:
- **Primary**: Overlay plot with means ± std

**Use Cases**:
- Quick comparison of dataset signatures
- Assess variability within datasets
- Identify characteristic spectral features

**Tips**:
- Use std bands to assess reproducibility
- Individual spectra show data distribution
- Normalization useful for different intensity scales

---

#### 12. Waterfall Plot

**Purpose**: Visualize many spectra simultaneously with vertical offset.

**Parameters**:
- `offset_scale` (0.1-5.0): Vertical spacing between spectra
- `max_spectra` (10-200): Maximum number to plot
- `colormap`: Color gradient for spectra
- `reverse_order` (bool): Plot from bottom to top

**Outputs**:
- **Primary**: Stacked spectra plot

**Use Cases**:
- Show time series or sequential measurements
- Visualize sample progression
- Display replicate variability

**Tips**:
- Adjust offset_scale for readability
- Use gradient colormap to show sequence
- Limit max_spectra to avoid clutter

---

#### 13. Correlation Heatmap

**Purpose**: Visualize wavenumber-wavenumber correlations.

**Parameters**:
- `method`: Correlation method (pearson / spearman)
- `colormap`: Color scheme (default: RdBu_r)
- `cluster` (bool): Apply hierarchical clustering

**Outputs**:
- **Primary**: Correlation matrix heatmap
- **Table**: Correlation values

**Use Cases**:
- Identify correlated spectral regions
- Understand band relationships
- Feature selection for ML

**Interpretation**:
- Red = positive correlation (co-varying peaks)
- Blue = negative correlation (inverse relationship)
- Off-diagonal patterns reveal band coupling

---

#### 14. Peak Intensity Scatter

**Purpose**: Compare specific peak intensities across datasets.

**Parameters**:
- `peak_positions`: List of wavenumbers (auto-detect if empty)
- `tolerance` (1-20 cm⁻¹): Matching tolerance
- `prominence` (0.01-1.0): Auto-detection threshold

**Outputs**:
- **Primary**: Scatter plots for each peak
- **Table**: Peak intensities for all spectra

**Use Cases**:
- Compare biomarker peaks across groups
- Assess peak intensity distributions
- Identify outliers at specific wavenumbers

**Tips**:
- Manual peak selection more precise than auto-detect
- Use for known biomarker positions
- Scatter shows within-group variability

---

## Usage Workflows

### Workflow 1: Exploratory Data Analysis

**Goal**: Understand dataset structure and relationships.

**Steps**:
1. Select all datasets (or subset of interest)
2. Start with PCA:
   - Use StandardScaler
   - Set n_components=3
   - Enable show_loadings and show_scree
3. Examine scores plot:
   - Look for sample groupings
   - Identify outliers
4. Check variance explained:
   - If first 2 PCs < 60%, consider UMAP or t-SNE
5. Follow up with clustering:
   - Try hierarchical clustering for hierarchy
   - Or K-means for distinct groups

**Interpretation Tips**:
- PCA works best for linear relationships
- UMAP better for complex non-linear data
- Clustering confirms visual groupings

---

### Workflow 2: Statistical Comparison

**Goal**: Determine if datasets differ significantly.

**Steps**:
1. For 2 datasets:
   - Use Spectral Comparison
   - Set confidence_level=0.95
   - Enable fdr_correction
   - Highlight significant regions
2. For 3+ datasets:
   - Use ANOVA Test
   - Set alpha=0.05
   - Enable post_hoc tests
3. Follow up with Peak Analysis:
   - Focus on significant wavenumbers
   - Characterize peaks

**Statistical Considerations**:
- FDR correction recommended for multiple comparisons
- Check assumption: normal distribution and equal variance
- Use post-hoc tests to identify which groups differ

---

### Workflow 3: Visualization for Publication

**Goal**: Create publication-ready figures.

**Steps**:
1. Choose appropriate visualization:
   - Heatmap: Overview of all data
   - Mean Overlay: Compare group signatures
   - Waterfall: Show sequential trends
2. Configure parameters:
   - Use publication-quality colormaps (viridis, plasma)
   - Enable normalization if intensity scales differ
   - Adjust font sizes in matplotlib settings
3. Export:
   - PNG for general use (300 DPI)
   - SVG for vector graphics (scalable)
4. Post-processing:
   - Use exported SVG in Inkscape or Illustrator
   - Adjust labels, legends, colors as needed

**Recommended Colormaps**:
- **Sequential**: viridis, plasma, inferno
- **Diverging**: RdBu_r, coolwarm, seismic
- **Qualitative**: tab10, Set2, Paired

---

### Workflow 4: Biomarker Discovery

**Goal**: Identify peaks that discriminate between groups.

**Steps**:
1. Perform Spectral Comparison or ANOVA
2. Identify significant wavenumber regions
3. Use Peak Analysis on significant regions
4. Create Peak Intensity Scatter for top peaks
5. Validate with correlation analysis
6. Export peak table for quantification

**Validation**:
- Cross-validate findings with independent dataset
- Check biological relevance (literature)
- Assess reproducibility across replicates

---

## Export Options

### Export Format Comparison

| Format | Use Case | Pros | Cons |
|--------|----------|------|------|
| **PNG** | Quick sharing, presentations | Universal compatibility, good quality | Not scalable, larger file size |
| **SVG** | Publications, posters | Scalable, editable | Requires vector software |
| **CSV** | Further analysis, statistics | Raw data, portable | No visualization |
| **Full Report** | Documentation, archival | Complete record | Large file size |

### Export Workflow

1. Complete analysis
2. Click "Export Results..." button
3. Choose format:
   - **PNG**: Select primary or secondary figure, choose resolution (default 300 DPI)
   - **SVG**: Select figure, save with .svg extension
   - **CSV**: Export data table as comma-separated values
   - **Full Report**: Generate HTML/PDF with all figures and tables
4. Choose save location
5. Verify exported file

### Export Tips

- **PNG**: Use 300 DPI for publication, 150 DPI for web
- **SVG**: Edit text and colors in vector graphics software
- **CSV**: Open in Excel, pandas, or R for further analysis
- **Reports**: Include analysis parameters for reproducibility

---

## Technical Details

### Threading Architecture

**Purpose**: Keep UI responsive during long computations.

**Implementation**:
```python
class AnalysisThread(QThread):
    progress = Signal(int)      # 0-100
    finished = Signal(AnalysisResult)
    error = Signal(str)
```

**Lifecycle**:
1. User clicks "Run Analysis"
2. AnalysisThread created with parameters
3. Thread started (`thread.start()`)
4. Progress updates sent via signal
5. Result or error emitted on completion
6. UI updates with result

**Cancellation**:
- "Cancel" button terminates thread
- Cleanup performed automatically
- UI returns to ready state

---

### Result Caching

**Purpose**: Avoid recomputing identical analyses.

**Cache Key**:
```python
cache_key = hash((
    category,
    method_key,
    frozenset(dataset_names),
    frozenset(params.items())
))
```

**Behavior**:
- Cache checked before running analysis
- If hit: Display cached result immediately
- If miss: Run analysis and cache result
- Cache cleared when:
  - User clicks "Clear Results"
  - Project changes
  - Application restarts

**Limitations**:
- Cache not persisted to disk
- Memory usage scales with result size
- Cleared on page switch

---

### Parameter Validation

**Validation Rules**:
- Numeric ranges enforced by widget configuration
- Type checking before analysis execution
- Required parameters must have values

**Error Handling**:
- Invalid parameters prevent analysis execution
- User notified via error message
- Parameters highlighted in red (future feature)

---

### Localization System

**Key Structure**:
```json
"ANALYSIS_PAGE": {
    "page_title": "Data Analysis",
    "METHODS": {
        "pca": "PCA (Principal Component Analysis)"
    },
    "PARAMS": {
        "n_components": "Number of Components"
    }
}
```

**Usage**:
```python
title = self.LOCALIZE("ANALYSIS_PAGE.page_title")
method_name = self.LOCALIZE(f"ANALYSIS_PAGE.METHODS.{method_key}")
```

**Supported Languages**:
- English (en)
- Japanese (ja)

---

## Troubleshooting

### Common Issues

#### Issue 1: No Methods Available

**Symptoms**: Method dropdown is empty after selecting category.

**Causes**:
- Registry not loaded correctly
- Import error in analysis_page_utils

**Solutions**:
1. Check console for import errors
2. Verify `analysis_page_utils/__init__.py` exports
3. Ensure all method modules are present

---

#### Issue 2: UMAP Not Available

**Symptoms**: Error when selecting UMAP method.

**Cause**: `umap-learn` package not installed

**Solution**:
```bash
pip install umap-learn
```

Or use conda:
```bash
conda install -c conda-forge umap-learn
```

---

#### Issue 3: Analysis Hangs

**Symptoms**: Progress bar stuck, UI unresponsive.

**Causes**:
- Large dataset with intensive computation
- Infinite loop in analysis function (bug)

**Solutions**:
1. Click "Cancel" button
2. Reduce dataset size or parameters
3. Check console for error messages
4. Report bug if reproducible

---

#### Issue 4: Export Fails

**Symptoms**: Error message when exporting results.

**Causes**:
- Invalid file path (special characters)
- Insufficient permissions
- Disk space full

**Solutions**:
1. Choose different save location
2. Check file path for invalid characters
3. Verify write permissions
4. Free up disk space

---

#### Issue 5: Results Look Wrong

**Symptoms**: Unexpected plots or data.

**Causes**:
- Incorrect parameters
- Wrong datasets selected
- Data quality issues

**Solutions**:
1. Verify dataset selection (check filter)
2. Review parameter values
3. Check raw data quality
4. Try different analysis method
5. Compare with literature/expected results

---

### Debug Mode

**Enable Logging**:
```python
# In main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Log Locations**:
- Console output
- `logs/` folder (if configured)

**What to Log**:
- Parameter values before analysis
- Dataset shapes and ranges
- Intermediate computation results
- Error tracebacks

---

### Performance Optimization

**Large Datasets**:
- Use filters to reduce dataset count
- Limit max_spectra in waterfall plots
- Consider downsampling wavenumbers (preprocessing)

**Memory Usage**:
- Clear results cache periodically
- Close unused figure windows
- Restart application if memory grows

**Computation Time**:
- Use faster methods (PCA vs t-SNE)
- Reduce iterations (t-SNE, K-means)
- Use smaller n_neighbors (UMAP)

---

## Appendix

### Method Comparison Table

| Method | Speed | Scalability | Linear | Interpretability |
|--------|-------|-------------|--------|-----------------|
| PCA | Fast | Excellent | Yes | High |
| UMAP | Moderate | Good | No | Moderate |
| t-SNE | Slow | Poor | No | Low |
| Hierarchical | Moderate | Moderate | N/A | High |
| K-means | Fast | Excellent | N/A | Moderate |

### Recommended Reading

**Dimensionality Reduction**:
- Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer.
- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. arXiv:1802.03426.

**Clustering**:
- MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations*.
- Ward Jr, J. H. (1963). *Hierarchical grouping to optimize an objective function*. JASA.

**Statistical Methods**:
- Student (1908). *The probable error of a mean*. Biometrika.
- Fisher, R. A. (1925). *Statistical Methods for Research Workers*.

---

**Document Version**: 1.0  
**Last Updated**: December 28, 2024  
**Maintainer**: AI Agent (GitHub Copilot)
