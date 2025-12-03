# MatplotlibWidget Heatmap Support

> **Date**: December 4, 2025  
> **Component**: `components/widgets/matplotlib_widget.py`  
> **Priority**: Critical

## Overview

This document describes the heatmap display support added to `MatplotlibWidget` to enable proper rendering of correlation heatmaps, spectral heatmaps with clustering, and other visualization types that use `imshow()`.

## Problem Statement

### Symptom
- Correlation Heatmap displayed blank with debug message: `[DEBUG] Found 0 patches on axis`
- Spectral Heatmap with Clustering showed only the dendrogram, not the heatmap

### Root Cause
The `update_plot()` method in `matplotlib_widget.py` copies figure content from the source figure to the widget's internal figure. However, it only handled:
- Lines (`ax.get_lines()`)
- Scatter plots/collections (`ax.collections`)
- Patches (`ax.patches`)
- Annotations (`ax.texts`)

It **did NOT handle** `AxesImage` objects, which are created by `imshow()` for heatmaps.

## Solution Implementation

### 1. Import AxesImage Type

```python
from matplotlib.image import AxesImage
```

### 2. Add Image Copying in update_plot()

```python
# Copy images (imshow data - for heatmaps)
images = ax.get_images()
if images:
    for img in images:
        img_data = img.get_array()
        extent = img.get_extent()
        cmap = img.get_cmap()
        clim = img.get_clim()
        
        # Get image properties
        alpha = img.get_alpha() if img.get_alpha() is not None else 1.0
        interpolation = img.get_interpolation()
        origin = 'upper'  # Default for imshow
        
        # Recreate image on new axis
        new_img = new_ax.imshow(
            img_data,
            extent=extent,
            cmap=cmap,
            aspect='auto',
            interpolation=interpolation,
            alpha=alpha,
            origin=origin
        )
        new_img.set_clim(clim)
        
        has_images = True
```

### 3. Add Colorbar Support

```python
# Add colorbar for images if the source figure has one
if has_images and source_has_colorbar:
    last_img = new_ax.get_images()[-1]
    self.figure.colorbar(last_img, ax=new_ax)
```

### 4. Preserve Complex Layout Positions

For GridSpec layouts (like dendrograms + heatmap), preserve the exact position:

```python
# Get original position from source axis
pos = ax.get_position()

# Recreate axis at same position
new_ax = self.figure.add_axes([pos.x0, pos.y0, pos.width, pos.height])
```

## Affected Functions

### `update_plot(self, new_figure: Figure, layout_rect=None)`
- Added AxesImage detection and copying
- Added colorbar recreation logic
- Improved position preservation for GridSpec layouts

### `_copy_plot_elements(source_ax, target_ax, copy_data=True, copy_annotations=True)`
- Added AxesImage handling
- Returns `has_images` boolean for colorbar handling

### `update_plot_with_config(self, new_figure: Figure, ...)`
- Added colorbar handling based on `_copy_plot_elements()` return value

## Visualization Methods Affected

These analysis page methods now work correctly:
- `create_correlation_heatmap()` - Correlation matrix visualization
- `create_spectral_heatmap()` - Spectral heatmap with hierarchical clustering
- Any future method using `imshow()` for visualization

## Testing

### Manual Test Steps
1. Load a dataset in Analysis page
2. Select "Correlation Heatmap" from visualization dropdown
3. Verify the heatmap displays with colorbar
4. Select "Spectral Heatmap with Clustering"
5. Verify both dendrogram and heatmap display correctly

### Expected Result
- Heatmap color gradient visible (not blank)
- Colorbar displayed alongside heatmap
- Proper aspect ratio and axis labels

## Code Pattern Reference

### CORRECT Pattern
```python
# Always check for images when copying figures
images = ax.get_images()
if images:
    for img in images:
        # Copy all image properties
        new_ax.imshow(
            img.get_array(),
            extent=img.get_extent(),
            cmap=img.get_cmap(),
            aspect='auto'
        )
```

### WRONG Pattern (Causes Blank Heatmaps)
```python
# Only copying lines and collections - MISSES imshow data!
for line in ax.get_lines():
    new_ax.plot(...)
for coll in ax.collections:
    ...
# ❌ No image handling = blank heatmaps
```

## Related Files

- `components/widgets/matplotlib_widget.py` - Main widget implementation
- `functions/visualization/analysis_plots.py` - Heatmap generation functions
- `.AGI-BANKS/BASE_MEMORY.md` - Pattern documentation
- `.AGI-BANKS/RECENT_CHANGES/2025-12-04_0003_heatmap_display_fix.md` - Change log

## Future Considerations

1. **Performance**: For very large heatmaps, consider lazy loading or downsampling
2. **Interactive Features**: Add zoom/pan support for heatmaps
3. **Export**: Ensure heatmap exports maintain resolution and colorbar
