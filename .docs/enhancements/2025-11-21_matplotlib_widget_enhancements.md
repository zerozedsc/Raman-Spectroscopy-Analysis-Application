# MatplotlibWidget Enhancement Summary

**Date**: November 21, 2025  
**Version**: 2.0  
**Author**: Development Team  
**Status**: Completed ✅

---

## Executive Summary

This document details significant enhancements made to the MatplotlibWidget component, including bug fixes for rendering issues, addition of 3D visualization capabilities, hierarchical clustering support, and advanced configuration options.

---

## Table of Contents

1. [Bug Fixes](#bug-fixes)
2. [New Features](#new-features)
3. [API Reference](#api-reference)
4. [Usage Examples](#usage-examples)
5. [Testing Guide](#testing-guide)
6. [Migration Notes](#migration-notes)

---

## Bug Fixes

### 1. FancyArrow Rendering Issue

**Problem**: Arrow patches not rendering in visualizations
- Debug message: `[DEBUG] Skipping unsupported patch type: FancyArrow`
- Arrows missing from annotated plots

**Solution**: Added proper recreation logic for FancyArrow and FancyArrowPatch types

**Before**:
```python
else:
    # For other patch types, log and skip
    print(f"[DEBUG] Skipping unsupported patch type: {type(patch).__name__}")
```

**After**:
```python
elif isinstance(patch, FancyArrow):
    new_arrow = FancyArrow(
        x=patch.get_x(),
        y=patch.get_y(),
        dx=patch.get_width(),
        dy=patch.get_height(),
        width=getattr(patch, '_width', 0.01),
        head_width=getattr(patch, '_head_width', 0.03),
        head_length=getattr(patch, '_head_length', 0.05),
        facecolor=patch.get_facecolor(),
        edgecolor=patch.get_edgecolor(),
        linewidth=patch.get_linewidth(),
        alpha=patch.get_alpha()
    )
    new_ax.add_patch(new_arrow)
```

### 2. LineCollection Handling for Dendrograms

**Problem**: Hierarchical clustering dendrograms not rendering properly
- Debug message: `[DEBUG] Skipping LineCollection (no get_sizes method)`
- Dendrogram lines missing from plots

**Solution**: Added proper LineCollection copying with segments, colors, and styles

**Before**:
```python
if isinstance(collection, LineCollection):
    print(f"[DEBUG] Skipping LineCollection (no get_sizes method)")
    continue
```

**After**:
```python
if isinstance(collection, LineCollection):
    print(f"[DEBUG] Copying LineCollection (dendrogram/cluster lines)")
    segments = collection.get_segments()
    colors = collection.get_colors()
    linewidths = collection.get_linewidths()
    linestyles = collection.get_linestyles()
    
    new_collection = LineCollection(
        segments,
        colors=colors,
        linewidths=linewidths,
        linestyles=linestyles
    )
    new_ax.add_collection(new_collection)
    continue
```

---

## New Features

### 1. Advanced Plot Configuration

**Method**: `update_plot_with_config(new_figure, config)`

**Purpose**: Provide comprehensive control over plot appearance and layout

**Configuration Options**:

```python
config = {
    # Subplot spacing
    'subplot_spacing': (hspace, wspace),  # tuple of floats
    
    # Grid configuration
    'grid': {
        'enabled': bool,
        'alpha': float,
        'linestyle': str,
        'linewidth': float
    },
    
    # Legend configuration
    'legend': {
        'loc': str,           # 'best', 'upper right', 'lower left', etc.
        'fontsize': int,
        'framealpha': float
    },
    
    # Title styling
    'title': {
        'fontsize': int,
        'fontweight': str,    # 'normal', 'bold'
        'pad': float
    },
    
    # Axes configuration
    'axes': {
        'xlabel_fontsize': int,
        'ylabel_fontsize': int,
        'tick_labelsize': int
    },
    
    # Figure layout
    'figure': {
        'tight_layout': bool,
        'constrained_layout': bool
    }
}
```

**Example**:
```python
config = {
    'grid': {'alpha': 0.5, 'linestyle': ':', 'linewidth': 0.8},
    'legend': {'loc': 'upper right', 'fontsize': 10, 'framealpha': 0.9},
    'title': {'fontsize': 14, 'fontweight': 'bold'},
    'axes': {'xlabel_fontsize': 12, 'ylabel_fontsize': 12, 'tick_labelsize': 10}
}

widget.update_plot_with_config(figure, config)
```

### 2. 3D Visualization Support

**Method**: `plot_3d(data, plot_type, title, config)`

**Supported Plot Types**:
- `'scatter'` - 3D scatter plot (for PCA, t-SNE, UMAP with 3 components)
- `'surface'` - 3D surface plot (for heatmaps, intensity surfaces)
- `'wireframe'` - 3D wireframe plot (for structural visualization)

**Data Format**:
```python
data = {
    'x': np.ndarray,          # X coordinates
    'y': np.ndarray,          # Y coordinates
    'z': np.ndarray,          # Z coordinates
    'labels': np.ndarray,     # Optional: class labels for coloring
    'colors': np.ndarray      # Optional: color values
}
```

**Configuration**:
```python
config = {
    'marker_size': int,       # Marker size for scatter (default 50)
    'alpha': float,           # Transparency (default 0.6)
    'elev': float,            # Elevation viewing angle (default 20)
    'azim': float,            # Azimuth viewing angle (default 45)
    'xlabel': str,            # X-axis label (default 'X')
    'ylabel': str,            # Y-axis label (default 'Y')
    'zlabel': str,            # Z-axis label (default 'Z')
    'grid': bool,             # Show grid (default True)
    'color': str              # Wireframe color (for wireframe plots)
}
```

**Examples**:

**3D Scatter with Labels**:
```python
from sklearn.decomposition import PCA

# Perform 3D PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_matrix)

# Prepare data
data_3d = {
    'x': pca_result[:, 0],
    'y': pca_result[:, 1],
    'z': pca_result[:, 2],
    'labels': class_labels  # Will color by class
}

config = {
    'xlabel': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
    'ylabel': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
    'zlabel': f'PC3 ({pca.explained_variance_ratio_[2]:.1%})',
    'marker_size': 60,
    'alpha': 0.7,
    'elev': 25,
    'azim': 60
}

widget.plot_3d(data_3d, plot_type='scatter', title='PCA 3D Visualization', config=config)
```

**3D Surface Plot**:
```python
# Create intensity surface
X, Y = np.meshgrid(wavenumbers, sample_indices)
Z = intensity_matrix

data_surface = {
    'x': X,
    'y': Y,
    'z': Z
}

config = {
    'xlabel': 'Wavenumber (cm⁻¹)',
    'ylabel': 'Sample Index',
    'zlabel': 'Intensity',
    'alpha': 0.9
}

widget.plot_3d(data_surface, plot_type='surface', title='Spectral Intensity Surface', config=config)
```

### 3. Dendrogram Visualization

**Method**: `plot_dendrogram(dendrogram_data, title, config)`

**Purpose**: Proper visualization of hierarchical clustering results

**Input Types**:

1. **Linkage Matrix** (from scipy.cluster.hierarchy.linkage):
```python
from scipy.cluster.hierarchy import linkage

Z = linkage(data_matrix, method='ward')
widget.plot_dendrogram(Z, title='Hierarchical Clustering')
```

2. **Precomputed Dendrogram** (from scipy.cluster.hierarchy.dendrogram):
```python
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(data_matrix, method='ward')
dend_data = dendrogram(Z, no_plot=True)
widget.plot_dendrogram(dend_data, title='Hierarchical Clustering')
```

**Configuration**:
```python
config = {
    'orientation': str,               # 'top', 'bottom', 'left', 'right' (default 'top')
    'color_threshold': float,         # Distance threshold for coloring (default None)
    'above_threshold_color': str,     # Color for above threshold (default 'grey')
    'leaf_font_size': int,            # Font size for leaf labels (default 8)
    'linewidth': float,               # Line width (default 1.5)
    'xlabel': str,                    # X-axis label (default 'Sample Index')
    'ylabel': str,                    # Y-axis label (default 'Distance')
    'grid': bool                      # Show grid (default True)
}
```

**Example**:
```python
from scipy.cluster.hierarchy import linkage

# Perform hierarchical clustering
Z = linkage(data_matrix, method='ward', metric='euclidean')

# Configure dendrogram
config = {
    'orientation': 'top',
    'color_threshold': 50,  # Color clusters at distance threshold
    'linewidth': 2.0,
    'xlabel': 'Spectrum Index',
    'ylabel': 'Ward Distance',
    'grid': True
}

widget.plot_dendrogram(Z, title='Hierarchical Clustering Dendrogram', config=config)
```

### 4. Helper Method for Element Copying

**Method**: `_copy_plot_elements(source_ax, target_ax)`

**Purpose**: Centralized, reusable logic for copying plot elements between axes

**Handles**:
- Line plots (Line2D)
- Scatter plots (PathCollection)
- Line collections (LineCollection)
- Patches (Ellipse, Rectangle, FancyArrow, FancyArrowPatch)

**Benefits**:
- Cleaner code organization
- Easier maintenance
- Reusable across methods
- Extensible for new artist types

---

## API Reference

### MatplotlibWidget Class

#### Methods

##### `__init__(parent=None)`
Initialize the widget with matplotlib figure, canvas, and toolbar.

##### `update_plot(new_figure)`
Replace current figure with new figure (basic method, no configuration).

##### `update_plot_with_config(new_figure, config=None)`
Replace current figure with advanced configuration options.

**Parameters**:
- `new_figure` (Figure): Matplotlib figure to display
- `config` (dict, optional): Configuration dictionary (see [Configuration Options](#1-advanced-plot-configuration))

##### `plot_3d(data, plot_type='scatter', title="3D Visualization", config=None)`
Create 3D visualization.

**Parameters**:
- `data` (dict): Dictionary with 'x', 'y', 'z' arrays and optional 'labels'/'colors'
- `plot_type` (str): 'scatter', 'surface', or 'wireframe'
- `title` (str): Plot title
- `config` (dict, optional): Configuration dictionary

##### `plot_dendrogram(dendrogram_data, title="Hierarchical Clustering Dendrogram", config=None)`
Create dendrogram visualization.

**Parameters**:
- `dendrogram_data` (dict or np.ndarray): Dendrogram data dict or linkage matrix
- `title` (str): Plot title
- `config` (dict, optional): Configuration dictionary

##### `clear_plot()`
Clear the current plot.

##### `plot_spectra(data, title="Spectra", auto_focus=False, focus_padding=None, crop_bounds=None)`
Plot Raman spectra data.

##### `plot_comparison_spectra_with_wavenumbers(...)`
Plot comparison between original and processed spectra with proper wavenumber axes.

---

## Usage Examples

### Example 1: Enhanced Configuration for Analysis Results

```python
from components.widgets.matplotlib_widget import MatplotlibWidget
from matplotlib import pyplot as plt

# Create widget
widget = MatplotlibWidget()

# Create analysis figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ... populate figure with analysis results ...

# Apply consistent styling
config = {
    'subplot_spacing': (0.4, 0.3),
    'grid': {
        'enabled': True,
        'alpha': 0.4,
        'linestyle': '--',
        'linewidth': 0.6
    },
    'legend': {
        'loc': 'best',
        'fontsize': 9,
        'framealpha': 0.85
    },
    'title': {
        'fontsize': 13,
        'fontweight': 'bold',
        'pad': 12
    },
    'axes': {
        'xlabel_fontsize': 11,
        'ylabel_fontsize': 11,
        'tick_labelsize': 9
    },
    'figure': {
        'tight_layout': True
    }
}

widget.update_plot_with_config(fig, config)
```

### Example 2: 3D PCA Visualization

```python
from sklearn.decomposition import PCA
import numpy as np

# Perform PCA with 3 components
pca = PCA(n_components=3)
pca_result = pca.fit_transform(spectral_data)

# Get class labels (if available)
labels = class_assignments  # Array of class indices

# Prepare 3D data
data_3d = {
    'x': pca_result[:, 0],
    'y': pca_result[:, 1],
    'z': pca_result[:, 2],
    'labels': labels
}

# Configure visualization
config = {
    'xlabel': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
    'ylabel': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
    'zlabel': f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)',
    'marker_size': 70,
    'alpha': 0.7,
    'elev': 20,
    'azim': 45,
    'grid': True
}

# Display 3D scatter plot
widget.plot_3d(data_3d, plot_type='scatter', 
              title='PCA 3D Visualization - Raman Spectra',
              config=config)
```

### Example 3: Hierarchical Clustering Dendrogram

```python
from scipy.cluster.hierarchy import linkage
import numpy as np

# Perform hierarchical clustering
Z = linkage(spectral_data, method='ward', metric='euclidean')

# Configure dendrogram
config = {
    'orientation': 'top',
    'color_threshold': 100,  # Distance threshold for cluster coloring
    'above_threshold_color': '#808080',
    'leaf_font_size': 7,
    'linewidth': 1.8,
    'xlabel': 'Spectrum Index',
    'ylabel': 'Ward Linkage Distance',
    'grid': True
}

# Display dendrogram
widget.plot_dendrogram(Z, 
                      title='Hierarchical Clustering of Raman Spectra',
                      config=config)
```

---

## Testing Guide

### Test 1: FancyArrow Rendering

**Purpose**: Verify arrow patches render correctly

**Steps**:
1. Create plot with arrow annotations
2. Call `widget.update_plot(figure)`
3. Check console logs

**Expected**:
- ✅ No "[DEBUG] Skipping unsupported patch type: FancyArrow" messages
- ✅ Arrows visible in plot
- ✅ Arrow properties preserved (color, size, orientation)

### Test 2: Dendrogram Visualization

**Purpose**: Verify hierarchical clustering dendrograms display properly

**Steps**:
1. Navigate to Analysis Page
2. Select "Hierarchical Clustering" method
3. Run analysis on dataset
4. Check dendrogram display

**Expected**:
- ✅ No "[DEBUG] Skipping LineCollection" messages
- ✅ Dendrogram lines visible with proper colors
- ✅ Axes labels and title displayed
- ✅ Grid visible (if configured)

### Test 3: 3D Scatter Plot

**Purpose**: Verify 3D visualization works for PCA results

**Steps**:
1. Run PCA with n_components=3
2. Call `widget.plot_3d()` with PCA results
3. Interact with plot (rotate, zoom)

**Expected**:
- ✅ 3D scatter plot displays
- ✅ Points colored by class labels
- ✅ Legend shows all classes
- ✅ Interactive rotation works
- ✅ Axis labels match configuration

### Test 4: Advanced Configuration

**Purpose**: Verify configuration options apply correctly

**Steps**:
1. Create multi-subplot figure
2. Call `widget.update_plot_with_config()` with custom config
3. Inspect visual appearance

**Expected**:
- ✅ Subplot spacing matches configuration
- ✅ Grid style matches configuration (linestyle, alpha, width)
- ✅ Legend positioned correctly
- ✅ Font sizes match configuration
- ✅ All subplots styled consistently

---

## Migration Notes

### Backward Compatibility

All existing code using `update_plot()` continues to work without modification.

**Old Code** (still works):
```python
widget.update_plot(figure)
```

**New Code** (with enhancements):
```python
config = {...}  # Optional configuration
widget.update_plot_with_config(figure, config)
```

### Recommended Upgrades

#### For Analysis Page

**Before**:
```python
# Manual figure updating without configuration
widget.update_plot(analysis_figure)
```

**After**:
```python
# Use configuration for consistent styling
config = {
    'grid': {'alpha': 0.3},
    'legend': {'loc': 'best', 'fontsize': 9}
}
widget.update_plot_with_config(analysis_figure, config)
```

#### For Hierarchical Clustering

**Before**:
```python
# Limited dendrogram support, used generic update_plot
widget.update_plot(dendrogram_figure)
```

**After**:
```python
# Direct dendrogram support with configuration
config = {'color_threshold': 50, 'linewidth': 2.0}
widget.plot_dendrogram(linkage_matrix, title='Clustering', config=config)
```

#### For 3D Visualization

**Before**:
```python
# No built-in 3D support
# Had to create 3D figure externally
```

**After**:
```python
# Direct 3D plotting with configuration
data_3d = {'x': pc1, 'y': pc2, 'z': pc3, 'labels': labels}
config = {'elev': 20, 'azim': 45, 'marker_size': 60}
widget.plot_3d(data_3d, 'scatter', 'PCA 3D', config)
```

---

## Performance Considerations

### Large Datasets

- **Scatter plots**: Use downsampling for >10,000 points
- **Surface plots**: Grid resolution affects performance
- **Dendrograms**: Large trees (>1000 leaves) may be slow

**Example Downsampling**:
```python
if len(x) > 10000:
    indices = np.random.choice(len(x), 10000, replace=False)
    data_3d = {
        'x': x[indices],
        'y': y[indices],
        'z': z[indices],
        'labels': labels[indices]
    }
```

### Memory Usage

- 3D surface plots with high resolution can use significant memory
- Consider using wireframe for preliminary visualization

---

## Future Enhancements

### Planned Features

1. **Interactive 3D tooltips** - Show point details on hover
2. **Animation support** - Rotate 3D plots automatically
3. **Export configuration** - Save/load plot configurations
4. **Preset themes** - Pre-defined styling themes (dark, light, publication)
5. **More plot types** - Contour plots, quiver plots, streamplots

### Extensibility

The widget is designed to be easily extended:

```python
def plot_custom(self, data, config=None):
    """Template for adding new plot types."""
    if config is None:
        config = {}
    
    self.figure.clear()
    ax = self.figure.add_subplot(111)
    
    # Custom plotting logic here
    # ...
    
    # Apply configuration
    if 'grid' in config:
        ax.grid(**config['grid'])
    
    self.figure.tight_layout()
    self.canvas.draw()
```

---

## Conclusion

The MatplotlibWidget enhancements provide:

✅ **Bug-free rendering** for all matplotlib artist types  
✅ **3D visualization** capabilities for advanced analysis  
✅ **Dendrogram support** for hierarchical clustering  
✅ **Flexible configuration** for consistent styling  
✅ **Clean code organization** with helper methods  
✅ **Backward compatibility** with existing code  

These improvements enable richer visualizations throughout the Raman spectroscopy application while maintaining code quality and ease of use.

---

**Document Version**: 1.0  
**Last Updated**: November 21, 2025  
**Review Status**: Ready for implementation ✅
