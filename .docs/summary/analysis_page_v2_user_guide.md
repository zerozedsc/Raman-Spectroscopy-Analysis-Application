# Analysis Page V2.0: User Guide

**Version**: 2.0.0  
**Release Date**: December 18, 2024  
**Platform**: Raman Spectroscopy Analysis Application

---

## What's New in V2.0?

The Analysis Page has been completely redesigned with a modern, intuitive interface that makes exploring and running analyses easier than ever. Here's what you'll love:

### 🎴 Card-Based Method Selection
Browse all available analysis methods at a glance with visual cards organized by category. Each card shows:
- Method name and description
- "Start Analysis" button for quick access
- Hover effects for better interactivity

### 📊 Three Analysis Categories
Methods are now organized into three clear categories:
- **Exploratory**: PCA, UMAP, t-SNE, Clustering
- **Statistical**: Spectral Comparison, Peak Analysis, ANOVA
- **Visualization**: Heatmaps, Overlay Plots, Waterfall Plots

### ✨ Split-View Analysis Interface
When you select a method, you'll see:
- **Left Panel**: Configure your analysis with dataset selection and parameters
- **Right Panel**: View results in real-time with multiple tabs

### 📜 Analysis History
Never lose track of your work! The sidebar keeps a running history of all analyses in your session:
- Click any history item to instantly restore previous results
- See timestamps, method names, and datasets used
- Clear history when you want a fresh start

### 💾 Comprehensive Export Options
Export your results in multiple formats:
- **PNG**: High-resolution images (300 DPI)
- **SVG**: Vector graphics for publications
- **CSV**: Data tables for further analysis
- **Full Report**: Complete analysis package with plots, data, and metadata

---

## Getting Started

### Step 1: Open the Analysis Page

Click on the "Analysis" tab in the main navigation to access the new interface.

### Step 2: Browse Methods

You'll see the startup view with all available analysis methods displayed as cards. Take your time to explore:
- **Exploratory Analysis**: Discover patterns and structures in your data
- **Statistical Analysis**: Test hypotheses and compare groups
- **Visualization**: Create compelling visual representations

### Step 3: Select a Method

Click on any method card to begin. The card will show:
- 📝 Method name (e.g., "PCA - Principal Component Analysis")
- 📖 Brief description of what the method does
- ▶️ "Start Analysis" button

### Step 4: Configure Your Analysis

After selecting a method, you'll see the method view with two panels:

**Left Panel (Configuration)**:
1. **Select Dataset**: Choose which dataset to analyze from the dropdown
2. **Set Parameters**: Adjust method-specific parameters (e.g., number of components, normalization options)
3. **Run**: Click the "Run Analysis" button when ready

**Right Panel (Results)**:
- Initially shows "Run analysis to see results here"
- Updates automatically when analysis completes
- Displays multiple tabs with different result views

### Step 5: View Results

Results appear in organized tabs:
- **📈 Plot**: Visual representation of your analysis
- **📋 Data Table**: Numerical results in table format
- **📝 Summary**: Text summary of key findings
- **🔍 Diagnostics**: Additional diagnostic information (when available)

### Step 6: Export or Save

When you're satisfied with your results:
1. Click one of the export buttons at the top of the results panel:
   - "Export PNG" for images
   - "Export SVG" for vector graphics
   - "Export CSV" for data tables
2. Choose your save location
3. Get a confirmation message when export completes

---

## Interface Tour

### Startup View

```
┌─────────────────────────────────────────────────────────────────┐
│  📊 Choose Your Analysis Method                                 │
│  Select from our comprehensive suite of analysis tools          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🔍 Exploratory Analysis                                        │
│  ┌────────┐  ┌────────┐  ┌────────┐                           │
│  │  PCA   │  │ UMAP   │  │ t-SNE  │                           │
│  │  ...   │  │  ...   │  │  ...   │                           │
│  │ [Start]│  │ [Start]│  │ [Start]│                           │
│  └────────┘  └────────┘  └────────┘                           │
│                                                                 │
│  📊 Statistical Analysis                                        │
│  ┌────────┐  ┌────────┐  ┌────────┐                           │
│  │Spectral│  │  Peak  │  │ ANOVA  │                           │
│  │Compare │  │Analysis│  │  ...   │                           │
│  │ [Start]│  │ [Start]│  │ [Start]│                           │
│  └────────┘  └────────┘  └────────┘                           │
│                                                                 │
│  🎨 Visualization                                               │
│  ┌────────┐  ┌────────┐  ┌────────┐                           │
│  │Heatmap │  │Overlay │  │Waterfall│                          │
│  │  ...   │  │  ...   │  │  ...   │                           │
│  │ [Start]│  │ [Start]│  │ [Start]│                           │
│  └────────┘  └────────┘  └────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### Method View

```
┌─────────────────────────────────────────────────────────────────┐
│  ← Back    📊 PCA (Principal Component Analysis)        [+New]  │
├──────────────────┬──────────────────────────────────────────────┤
│ 📜 History       │ INPUT FORM          │  RESULTS               │
│ ┌──────────────┐ │ ┌───────────────┐  │ ┌───────────────────┐ │
│ │ 🕐 14:35     │ │ │ Select Dataset│  │ │ [Export PNG/SVG]  │ │
│ │ PCA          │ │ │ [Dropdown]    │  │ │ ┌───┬───┬───┬───┐ │ │
│ │ Dataset 1    │ │ │               │  │ │ │📈 │📋 │📝 │🔍 │ │ │
│ ├──────────────┤ │ │ Parameters    │  │ │ └───┴───┴───┴───┘ │ │
│ │ 🕐 14:20     │ │ │ Components: 3 │  │ │                   │ │
│ │ UMAP         │ │ │ Normalize: ☑  │  │ │   [Plot Display]  │ │
│ │ Dataset 1    │ │ │               │  │ │                   │ │
│ └──────────────┘ │ │ [← Back]      │  │ │                   │ │
│ [Clear History]  │ │ [Run Analysis]│  │ │                   │ │
│                  │ └───────────────┘  │ └───────────────────┘ │
└──────────────────┴────────────────────┴──────────────────────────┘
```

---

## Tips & Tricks

### 💡 Quick Method Access
Hover over any method card to see the border highlight in blue. The entire card is clickable - no need to aim for the button!

### 💡 History Navigation
Click on any previous analysis in the history sidebar to:
- Restore the exact parameters you used
- View the cached results instantly (no need to re-run)
- Continue from where you left off

### 💡 Multiple Exports
You can export the same results in multiple formats:
1. Export PNG for presentations
2. Export SVG for publications
3. Export CSV for further analysis in Excel/Python

### 💡 Parameter Exploration
Adjust parameters and re-run the same analysis to:
- Compare different component numbers (PCA)
- Test various neighbor counts (UMAP, t-SNE)
- Explore normalization options

### 💡 Keyboard Navigation
Use Tab to navigate between form fields and Shift+Tab to go backwards. Press Enter on the "Run Analysis" button to start.

### 💡 Session History
Your analysis history persists throughout your current session. When you close and reopen the app, history is cleared for a fresh start.

---

## Frequently Asked Questions

### Q: Where did the old interface go?
**A**: The old interface has been completely replaced with this new design. All the same analysis methods are available, just in a more intuitive layout. If you need to rollback, see the technical documentation.

### Q: Can I run multiple analyses at once?
**A**: Currently, you can run one analysis at a time. While an analysis is running, the "Run Analysis" button is disabled and shows progress.

### Q: How do I compare two analyses?
**A**: Run your first analysis, then click the "+ New Analysis" button (or back button) to return to method selection. Run your second analysis and use the history sidebar to switch between them.

### Q: What happens when I export?
**A**: A file dialog opens asking where to save the file. Choose your location and click "Save". You'll see a success message when the export completes.

### Q: Can I save analyses to my project?
**A**: Yes! Use the "Export Full Report" option or the programmatic `save_to_project()` method. Reports are saved to your project's `analyses/` folder with a timestamped subfolder.

### Q: Why is the "Run Analysis" button disabled?
**A**: This means either:
- An analysis is currently running (wait for completion)
- No datasets are loaded (go to Data Package page to load data)

### Q: How do I clear my history?
**A**: Click the "Clear History" button at the bottom of the history sidebar. You'll see a confirmation dialog before the history is cleared.

### Q: What if analysis fails?
**A**: You'll see an error dialog with details about what went wrong. Common issues:
- No datasets loaded
- Invalid parameters
- Insufficient data for the method

### Q: Can I use this in Japanese?
**A**: Yes! The entire interface is fully localized in Japanese. Change your application language setting to see all text in Japanese.

---

## Analysis Method Reference

### Exploratory Analysis

**PCA (Principal Component Analysis)**
- **Purpose**: Reduce dimensionality and identify main variance components
- **Use When**: You want to see overall patterns or reduce noise
- **Parameters**: Number of components, normalization options

**UMAP (Uniform Manifold Approximation and Projection)**
- **Purpose**: Non-linear dimensionality reduction for visualization
- **Use When**: Linear methods don't reveal structure
- **Parameters**: Number of neighbors, minimum distance

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- **Purpose**: Visualize high-dimensional data in 2D/3D
- **Use When**: You want to see clusters and local structure
- **Parameters**: Perplexity, number of iterations

**Hierarchical Clustering**
- **Purpose**: Group similar spectra into a tree structure
- **Use When**: You want to see relationships between groups
- **Parameters**: Linkage method, distance metric

**K-Means Clustering**
- **Purpose**: Partition data into K clusters
- **Use When**: You know approximately how many groups exist
- **Parameters**: Number of clusters, initialization method

### Statistical Analysis

**Spectral Comparison**
- **Purpose**: Compare spectra between groups
- **Use When**: Testing for differences between conditions
- **Parameters**: Statistical test, significance level

**Peak Analysis**
- **Purpose**: Identify and quantify spectral peaks
- **Use When**: Looking for specific chemical signatures
- **Parameters**: Peak detection threshold, width

**Correlation Analysis**
- **Purpose**: Find correlations between spectra or variables
- **Use When**: Exploring relationships in your data
- **Parameters**: Correlation method

**ANOVA (Analysis of Variance)**
- **Purpose**: Test for differences across multiple groups
- **Use When**: Comparing 3+ experimental conditions
- **Parameters**: Factor variables, post-hoc tests

### Visualization

**Heatmap**
- **Purpose**: Display intensity patterns across spectra
- **Use When**: You want to see overall patterns at a glance
- **Parameters**: Color scheme, clustering

**Mean Spectra Overlay**
- **Purpose**: Compare average spectra between groups
- **Use When**: Showing group differences visually
- **Parameters**: Groups to compare, error bars

**Waterfall Plot**
- **Purpose**: 3D-style visualization of multiple spectra
- **Use When**: Showing many spectra simultaneously
- **Parameters**: Offset amount, color scheme

**Correlation Heatmap**
- **Purpose**: Visualize correlation matrix
- **Use When**: Identifying related variables
- **Parameters**: Clustering, annotation

**Peak Scatter**
- **Purpose**: Plot identified peaks across samples
- **Use When**: Comparing peak positions/intensities
- **Parameters**: Peak selection, axes

---

## Troubleshooting

### Problem: Method cards are not visible
**Solution**: Make sure you've scrolled down. With 15 methods, some cards may be below the fold. Use the scroll bar on the right side.

### Problem: Can't select a dataset
**Solution**: Go to the Data Package page and load at least one dataset first. The Analysis page requires data to be loaded.

### Problem: Parameters are confusing
**Solution**: Hover over parameter labels to see tooltips with more information. Consult the method documentation for detailed explanations.

### Problem: Analysis takes too long
**Solution**: Some methods (UMAP, t-SNE) can be computationally intensive with large datasets. Try:
- Reducing the number of spectra
- Using PCA first to reduce dimensionality
- Adjusting method-specific parameters (fewer iterations, etc.)

### Problem: Export button is grayed out
**Solution**: Export buttons only become active after an analysis has completed successfully. Run an analysis first, then try exporting.

### Problem: History items show wrong parameters
**Solution**: History captures parameters at the time of analysis. If you've changed parameters since, the history will show the old values. Re-run the analysis with new parameters to update.

---

## Keyboard Shortcuts (Planned for Future)

_These shortcuts will be added in a future update:_

- `Ctrl+N`: New Analysis (return to startup)
- `Ctrl+R`: Run Analysis
- `Ctrl+E`: Export PNG
- `Ctrl+H`: Toggle History Sidebar
- `Escape`: Cancel running analysis
- `Ctrl+Z`: Back to startup view

---

## Feedback & Support

We'd love to hear your thoughts on the new Analysis Page!

**Found a bug?** Report it through the application's feedback system.  
**Have a suggestion?** Let us know what features you'd like to see next.  
**Need help?** Consult the technical documentation or contact support.

---

## Version History

### Version 2.0.0 (December 18, 2024)
- 🆕 Card-based startup view with method gallery
- 🆕 Split-view analysis interface
- 🆕 Analysis history sidebar
- 🆕 Comprehensive export options (PNG, SVG, CSV, reports)
- 🆕 Enhanced visual design with hover effects
- 🆕 Full Japanese localization
- ✨ Improved error handling and notifications
- ✨ Modular architecture for better maintainability

### Version 1.0.0 (Previous)
- Basic left-panel/right-panel layout
- 15 analysis methods
- Simple export functionality

---

**Enjoy the new Analysis Page! Happy analyzing! 🎉**
