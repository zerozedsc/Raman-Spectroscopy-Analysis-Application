# Analysis Page Fixes and Improvements (December 3, 2025)

## Overview
Addressed several issues in the Analysis Page visualization and configuration, specifically for PCA, UMAP, t-SNE, and Clustering methods.

## Key Changes

### 1. Matplotlib Widget
- **Tight Layout**: Enforced `tight_layout()` by default in `MatplotlibWidget` to ensure plots are properly sized and labels are not cut off.

### 2. PCA (Principal Component Analysis)
- **Distribution Plots**: Added `n_distribution_components` parameter (default 3, max 6).
- **Grid Layout**: Updated distribution visualization to use a grid layout (up to 3 rows x 2 columns) to show more components as requested.
- **User Control**: Users can now choose how many components to display in the distributions tab.

### 3. t-SNE
- **Group Mode**: Enabled multi-dataset selection for t-SNE to allow group comparisons (Control vs Disease).
- **Perplexity Fix**: Added robust check for perplexity parameter. If `perplexity >= n_samples`, it is automatically adjusted to `n_samples - 1` to prevent `ValueError`.
- **Error Handling**: Fixed missing `loadings_figure` key causing debug errors.

### 4. UMAP
- **Error Handling**: Fixed missing `loadings_figure` key in return dictionary to prevent "Loadings tab NOT created" debug warnings.

### 5. Hierarchical Clustering
- **Dendrogram Improvement**: 
  - Added color thresholding (70% of max distance) to visually distinguish clusters.
  - Added threshold line.
  - Improved axis labels and grid.
- **Error Handling**: Fixed missing `loadings_figure` key.

### 6. K-Means Clustering
- **Error Handling**: Fixed missing `loadings_figure` key.

## Technical Details

### Registry Updates (`pages/analysis_page_utils/registry.py`)
- **PCA**: Added `n_distribution_components` spinbox parameter.
- **t-SNE**: Changed `dataset_selection_mode` to `"multi"` and `max_datasets` to `None`.

### Method Updates (`pages/analysis_page_utils/methods/exploratory.py`)
- **perform_pca_analysis**: Implemented dynamic grid plotting for distributions.
- **perform_tsne_analysis**: Added group label handling and perplexity validation.
- **perform_hierarchical_clustering**: Enhanced `dendrogram` call with `color_threshold`.
- **All Methods**: Ensured `loadings_figure: None` is returned where applicable.

## Verification
- **PCA**: Verify distribution tab shows up to 6 components in grid.
- **t-SNE**: Verify group mode works with multiple datasets and no perplexity crash on small datasets.
- **Clustering**: Verify dendrogram looks better and no "loadings_figure" errors in logs.
