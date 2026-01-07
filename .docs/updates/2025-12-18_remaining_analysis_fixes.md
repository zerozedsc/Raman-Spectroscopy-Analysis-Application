# Critical Fixes Required - Analysis Page Issues

## Status: PARTIAL - ANOVA Disabled, Dropdown Backgrounds Fixed

### Completed Fixes

1. ✅ **ANOVA Disabled** - Commented out in registry.py
2. ✅ **Dropdown Background** - Set to white in all QComboBox instances

### Remaining Critical Issues

#### 1. Create Group Dialog Issue
**Problem**: Still showing old single-input dialog ("グループ作成") instead of multi-group dialog ("複数グループを一度に作成")

**Hypothesis**: The dialog title suggests an old cached instance or import issue.

**Required Actions**:
- Clear Python cache: `find . -type d -name "__pycache__" -exec rm -rf {} +`
- Restart application completely
- Check console for debug messages when clicking Create Group button
- If still failing, the multi_group_dialog.py might have initialization errors

**Debug Commands to Add**:
```python
# In group_assignment_table.py, line 131
print(f"[DEBUG] Button object: {multi_group_action}")
print(f"[DEBUG] Connection test: {multi_group_action.triggered}")
```

#### 2. PCA Loadings Grid Dimension Mismatch
**Error**: `x and y must have same first dimension, but have shapes (883,) and (882,)`

**Location**: Occurs when selecting PCA components in the grid

**Fix Needed**: In `exploratory.py`, the PC loadings plot generation needs to handle edge cases where wavenumber arrays don't match perfectly. Add array length checks and truncate to minimum length.

#### 3. Missing loadings_figure for Multiple Methods

The following methods need to return `loadings_figure=None` or a placeholder:

- **UMAP** (`perform_umap_analysis`)
- **t-SNE** (`perform_tsne_analysis`)  
- **K-Means** (`perform_kmeans_clustering`)
- **Hierarchical Clustering** (`perform_hierarchical_clustering`)
- **Group Mean Spectral Comparison** (fix to support >2 groups)
- **Peak Detection** (remove spectrum preview)
- **Spectral Correlation** (fix wrong graph type)
- **Waterfall Plot** (3D not working)

**Pattern to Add**:
```python
# At end of each function
return AnalysisResult(
    main_figure=fig,
    loadings_figure=None,  # ADD THIS LINE
    distributions_figure=None,
    # ... other fields
)
```

#### 4. Hierarchical Clustering - Missing Labels/Legend
**Problem**: Dendrogram has no labels showing which clusters are which

**Fix**: Add color-coded labels and legend to dendrogram

#### 5. Group Mean Spectral Comparison - Only Shows 2 Lines
**Problem**: When 3 datasets selected, only 2 lines appear

**Fix**: Check loop logic in `perform_group_mean_comparison` to ensure all datasets are plotted

#### 6. Peak Detection - Remove Spectrum Preview
**Fix**: Set `show_spectrum_preview=False` in result or don't generate preview figure

#### 7. Spectral Correlation - Wrong Graph Type
**Problem**: Showing wrong visualization (image 4)

**Fix**: Verify correlation matrix heatmap is being generated correctly

#### 8. Waterfall Plot - 3D Not Working
**Fix**: Check 3D axes initialization and matplotlib backend compatibility

## Implementation Priority

**P0 (Critical - Blocking User)**:
1. PCA loadings grid dimension fix
2. Create Group dialog (if debug shows it's really calling wrong function)
3. Group Mean Spectral Comparison (3+ groups)

**P1 (High - User Experience)**:
1. Add loadings_figure=None to all methods
2. Hierarchical Clustering labels
3. Peak Detection - remove preview

**P2 (Medium - Polish)**:
1. Spectral Correlation graph fix
2. Waterfall 3D fix

## Next Steps

1. User should check console output when clicking "Create Group"
2. Share the full error traceback for PCA dimension mismatch
3. I'll implement remaining fixes in next response once we confirm priorities
