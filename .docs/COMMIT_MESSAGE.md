fix(analysis): enhance PCA visualization, fix memory leaks, and add spectrum preview

## Major Changes

### 1. MatplotlibWidget Stability & Layout
- **Fixed Memory Leak**: Implemented `plt.close()` in `update_plot` and `update_plot_with_config` to properly release memory after plot updates.
- **Fixed Crashes**: Added robust error handling for `FancyArrow` and Annotation copying, preventing crashes during Biplot and Peak Analysis.
- **Layout Improvement**: Ensured `tight_layout()` is applied by default for better space utilization.
- **Custom Toolbar**: Added `add_custom_toolbar` method to support adding widgets (like the component selector) to the plot area.

### 2. New Spectrum Tab
- **Added**: A new "Spectrum Preview" tab is now available in all analysis results.
- **Functionality**: It displays the mean spectrum ± standard deviation for each dataset, allowing quick verification of the input data.
- **Backend**: Updated `AnalysisResult` and `AnalysisThread` to propagate `dataset_data` to the results.

### 3. PCA Visualization Enhancements
- **Biplot**:
    - Arrows are now thinner and cleaner (adjusted `head_width`, `linewidth`).
    - Text labels are improved (thinner font, no box) for better readability.
- **Loadings Plot**:
    - **Component Selection**: Added a dropdown menu to select which Principal Component (PC) to display.
    - **Cleaner Look**: Removed x-axis wavenumber labels to reduce clutter.
- **Distributions Plot**:
    - **Fixed**: Resolved the issue where empty graphs appeared at the end of the grid by explicitly hiding unused axes.

## Files Modified

### Core Components
- `components/widgets/matplotlib_widget.py`: Critical fixes for memory leaks, copying crashes, and layout.

### Analysis Logic
- `pages/analysis_page_utils/result.py`: Added `dataset_data` field.
- `pages/analysis_page_utils/thread.py`: Passed `dataset_data` to result.
- `pages/analysis_page_utils/methods/exploratory.py`:
    - Implemented `create_spectrum_preview_figure`.
    - Enhanced PCA plotting logic (Biplot, Loadings, Distributions).

### Analysis UI
- `pages/analysis_page_utils/method_view.py`:
    - Integrated "Spectrum Preview" tab.
    - Implemented Component Selector for PCA Loadings.

## TODOs / Next Steps

- [ ] **Thread Safety**: Audit `AnalysisThread` usage to ensure NO heavy computation happens on the main thread (UI freezing prevention).
- [ ] **Subplot Layout**: Implement a generic `_calculate_subplot_grid` helper in `MatplotlibWidget` to standardize grid creation.
- [ ] **Logging**: Add comprehensive logging to `matplotlib_widget.py` to catch silent failures in the future.
- [ ] **Configuration**: Move hardcoded alpha values and colors to a configuration file or theme system.
- [ ] **Refactoring**: Rename `plot_spectra` in `matplotlib_widget.py` to `create_spectra_figure` to avoid naming conflicts.
- [ ] **Testing**: Perform full regression testing on all other analysis methods (t-SNE, UMAP, Clustering) to ensure no regressions.
