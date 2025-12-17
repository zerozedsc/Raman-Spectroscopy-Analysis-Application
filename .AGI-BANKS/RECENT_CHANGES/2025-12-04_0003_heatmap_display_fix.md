---
Title: Heatmap Display Fix for Correlation Heatmap and Spectral Heatmap with Clustering
Date: 2025-12-04T16:00:00Z
Author: agent/auto
Tags: bugfix, heatmap, matplotlib, imshow, correlation, spectral-heatmap, colorbar
RelatedFiles:
  - components/widgets/matplotlib_widget.py
  - functions/visualization/analysis_plots.py
Priority: P0 (Critical)
---

# Heatmap Display Fix for Analysis Page

## Summary

Fixed critical bugs causing blank/incorrect heatmap displays in:
1. **Correlation Heatmap** - Displayed blank/empty axes with only labels
2. **Spectral Heatmap with Clustering** - Displayed weird overlapping graphs instead of heatmap

---

## Root Cause Analysis

### Problem: `update_plot()` Did Not Copy `AxesImage` Objects

The `update_plot()` method in `matplotlib_widget.py` is responsible for copying figure contents from analysis methods to the display canvas. It handles:
- ✅ Line plots (`ax.get_lines()`)
- ✅ Scatter plots (`PathCollection`)
- ✅ Dendrogram lines (`LineCollection`)
- ✅ Patches (Ellipse, Rectangle, FancyArrow)
- ❌ **AxesImage objects (imshow/heatmaps)** - WAS MISSING!

When `imshow()` creates a heatmap, it returns an `AxesImage` object. The `update_plot()` method never copied these objects, resulting in blank displays.

### Debug Log Evidence
```
[DEBUG] Found 0 patches on axis
[DEBUG] Recreating 0 patches on new axis
[DEBUG] Copying LineCollection (dendrogram/cluster lines)
```
Notice: No image copying was happening!

---

## Solution Implemented

### 1. Added AxesImage Handling in `update_plot()`

```python
# ✅ FIX: Copy AxesImage objects (imshow/heatmaps) - CRITICAL FOR HEATMAPS
from matplotlib.image import AxesImage
images = ax.get_images()
if images:
    print(f"[DEBUG] Found {len(images)} AxesImage objects (heatmaps/imshow)")
    for img in images:
        try:
            img_data = img.get_array()
            extent = img.get_extent()
            cmap = img.get_cmap()
            alpha = img.get_alpha()
            interpolation = img.get_interpolation()
            clim = img.get_clim()
            
            new_img = new_ax.imshow(
                img_data,
                extent=extent,
                cmap=cmap,
                alpha=alpha if alpha is not None else 1.0,
                interpolation=interpolation,
                aspect='auto',
                vmin=clim[0],
                vmax=clim[1]
            )
            print(f"[DEBUG] Successfully copied AxesImage: shape={img_data.shape}")
        except Exception as e:
            print(f"[DEBUG] Failed to copy AxesImage: {e}")
```

### 2. Fixed Axes Position Preservation

Changed from simple sequential subplot creation to position-preserving layout:

```python
# ✅ FIX: Preserve complex subplot layouts (GridSpec, dendrograms with heatmaps)
for i, ax in enumerate(axes_list):
    try:
        pos = ax.get_position()
        new_ax = self.figure.add_axes([pos.x0, pos.y0, pos.width, pos.height])
    except Exception as e:
        # Fallback to simple layout
        if len(axes_list) == 1:
            new_ax = self.figure.add_subplot(111)
        else:
            new_ax = self.figure.add_subplot(len(axes_list), 1, i+1)
```

This preserves the original GridSpec layout with dendrograms positioned correctly around the main heatmap.

### 3. Added Colorbar Support

```python
# ✅ FIX: Add colorbar for heatmaps/imshow if images were copied
if images:
    last_img = new_ax.get_images()
    if last_img:
        try:
            cbar = self.figure.colorbar(last_img[-1], ax=new_ax)
            print(f"[DEBUG] Added colorbar for heatmap")
        except Exception as e:
            print(f"[DEBUG] Failed to add colorbar: {e}")
```

### 4. Disabled Grid for Heatmaps

Grids look bad on heatmaps, so we skip them:

```python
# Add grid (but not for heatmaps - they look bad with grid)
if not images:
    new_ax.grid(True, which='both', linestyle='--', linewidth=0.5)
```

### 5. Updated `_copy_plot_elements()` Helper

Also added AxesImage handling to the helper method used by `update_plot_with_config()`.

---

## Files Modified

| File | Changes |
|------|---------|
| `components/widgets/matplotlib_widget.py` | Added AxesImage copying, position preservation, colorbar support, grid skip for heatmaps |

---

## Methods Affected

All methods that use `imshow()` for visualization:
- `create_spectral_heatmap()` in `functions/visualization/analysis_plots.py`
- `create_correlation_heatmap()` in `functions/visualization/analysis_plots.py`
- `perform_correlation_analysis()` in `pages/analysis_page_utils/methods/statistical.py`

---

## Testing Recommendations

1. **Correlation Heatmap**:
   - Select 2+ datasets
   - Run Correlation Heatmap analysis
   - Verify: Colored heatmap displays (not blank)
   - Verify: Colorbar shows correlation coefficient scale (-1 to 1)

2. **Spectral Heatmap with Clustering**:
   - Select datasets
   - Enable cluster_rows and show_dendrograms
   - Verify: Heatmap displays with proper colormap
   - Verify: Dendrogram displays on left side
   - Verify: Layout is preserved (dendrogram + heatmap side by side)

3. **Debug Log Verification**:
   - Should see: `[DEBUG] Found N AxesImage objects (heatmaps/imshow)`
   - Should see: `[DEBUG] Successfully copied AxesImage: shape=...`
   - Should see: `[DEBUG] Added colorbar for heatmap`

---

## Technical Notes

### Why Images Are Different from Other Plot Elements

1. **Line plots**: Store x/y coordinate arrays → Can be re-plotted with `plot()`
2. **Scatter plots**: Store offsets and colors → Can be re-created with `scatter()`
3. **Images (imshow)**: Store 2D/3D numpy array data with colormap mapping → Need to preserve:
   - Raw data array (`get_array()`)
   - Color limits (`get_clim()`)
   - Colormap (`get_cmap()`)
   - Extent (data coordinates)
   - Interpolation method

### Position Preservation for GridSpec

Spectral Heatmap with Clustering uses GridSpec with multiple axes:
```
[dendrogram_col] [heatmap    ]
                 [dendrogram_row]
```

The old code created simple 1xN or Nx1 subplots, breaking this layout.
New code uses `ax.get_position()` to get exact normalized coordinates and recreates axes at the same positions.

---

## Related Issues

- Prior dimension mismatch fix: `2025-12-04_0002_dimension_mismatch_and_tsne_colors.md`
- Prior PCA visualization fixes: `2025-12-04_0001_pca_viz_enhancements.md`
