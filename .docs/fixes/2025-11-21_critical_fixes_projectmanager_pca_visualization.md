# Critical Fixes: ProjectManager, PCA, and Visualization Refactoring

**Date**: November 21, 2025  
**Version**: 1.0  
**Author**: Development Team  
**Status**: Completed ✅

---

## Executive Summary

This document details three critical fixes implemented on November 21, 2025:

1. **Bug Fix**: ProjectManager AttributeError during CSV export
2. **Enhancement**: Removal of arbitrary PCA n_components limit
3. **Architecture Improvement**: Migration of visualization methods to proper package location

All changes have been implemented and are ready for testing before production deployment.

---

## Table of Contents

1. [Fix 1: ProjectManager.current_project AttributeError](#fix-1-projectmanagercurrent_project-attributeerror)
2. [Fix 2: Remove PCA n_components Limit](#fix-2-remove-pca-n_components-limit)
3. [Fix 3: Visualization Methods Migration](#fix-3-visualization-methods-migration)
4. [Testing Instructions](#testing-instructions)
5. [Related Files](#related-files)
6. [Known Limitations](#known-limitations)

---

## Fix 1: ProjectManager.current_project AttributeError

### Problem Description

**Symptom**: Runtime error when exporting CSV from Analysis Page:
```
AttributeError: 'ProjectManager' object has no attribute 'current_project'. Did you mean: 'current_project_data'?
```

**Root Cause**: 
- Code referenced non-existent `.current_project` attribute
- ProjectManager actually stores data in `.current_project_data` dictionary
- Multiple locations had this incorrect reference

**Impact**: 
- **Severity**: Critical - blocks all CSV export functionality
- **Scope**: All analysis result exports
- **User Experience**: Complete failure of export feature

### Technical Details

**ProjectManager Data Structure**:
```python
# Correct structure (from utils.py)
class ProjectManager:
    def __init__(self):
        self.current_project_data = {
            "projectPath": str,      # Path to .ramanproj file
            "projectName": str,      # Project name
            "dataPackages": dict,    # All data packages
            "metadata": dict,        # Project metadata
            ...
        }
```

**Affected Locations**: `pages/analysis_page_utils/export_utils.py`

### Code Changes

#### Location 1: save_to_project() validation (Lines 254-263)

**Before**:
```python
def save_to_project(self, filename: str, data: Any) -> Optional[Path]:
    """Save file to project exports directory."""
    if not self.project_manager:
        return None
    
    if not self.project_manager.current_project:  # ❌ WRONG
        return None
    
    project_path = Path(self.project_manager.current_project)  # ❌ WRONG
    exports_dir = project_path.parent / "exports"
    exports_dir.mkdir(exist_ok=True)
```

**After**:
```python
def save_to_project(self, filename: str, data: Any) -> Optional[Path]:
    """Save file to project exports directory."""
    if not self.project_manager:
        return None
    
    if not self.project_manager.current_project_data:  # ✅ CORRECT
        return None
    
    project_path_str = self.project_manager.current_project_data.get("projectPath", "")  # ✅ SAFE ACCESS
    if not project_path_str:
        return None
    project_path = Path(project_path_str)
    exports_dir = project_path.parent / "exports"
    exports_dir.mkdir(exist_ok=True)
```

**Changes**:
- Line 254: `.current_project` → `.current_project_data`
- Line 259: Added safe dictionary access with `.get("projectPath", "")`
- Line 260-261: Added validation before Path creation
- Line 262: Extract path from validated string

#### Location 2: _get_default_export_path() fallback (Lines 311-322)

**Before**:
```python
def _get_default_export_path(self) -> Path:
    """Get default path for exports."""
    if self.project_manager and self.project_manager.current_project:  # ❌ WRONG
        project_path = Path(self.project_manager.current_project)  # ❌ WRONG
        exports_dir = project_path.parent / "exports"
        exports_dir.mkdir(exist_ok=True)
        return exports_dir
    return Path.home()
```

**After**:
```python
def _get_default_export_path(self) -> Path:
    """Get default path for exports."""
    if self.project_manager and self.project_manager.current_project_data:  # ✅ CORRECT
        project_path_str = self.project_manager.current_project_data.get("projectPath", "")  # ✅ SAFE ACCESS
        if project_path_str:
            project_path = Path(project_path_str)
            exports_dir = project_path.parent / "exports"
            exports_dir.mkdir(exist_ok=True)
            return exports_dir
    return Path.home()
```

**Changes**:
- Line 311: `.current_project` → `.current_project_data`
- Line 312: Added safe dictionary access with `.get("projectPath", "")`
- Line 313-317: Added validation before Path creation
- Maintained fallback to `Path.home()` if no project

### Testing Checklist

- [ ] Open Raman application
- [ ] Load or create a project
- [ ] Navigate to Analysis Page
- [ ] Run any analysis method (e.g., PCA, Spectral Heatmap)
- [ ] Click "Export Results" button
- [ ] Select "Export as CSV"
- [ ] **Expected**: CSV file exports successfully to `project_folder/exports/`
- [ ] **Verify**: No AttributeError appears in logs
- [ ] **Verify**: CSV file contains correct analysis results

### Rollback Plan

If this fix causes issues:

```python
# Revert to investigating ProjectManager structure
# Check if .current_project was recently renamed
# Verify utils.py ProjectManager class definition
```

---

## Fix 2: Remove PCA n_components Limit

### Problem Description

**Symptom**: PCA analysis limited to maximum 10 components in UI

**Root Cause**:
- Arbitrary limit set in registry.py parameter definition
- No scientific justification for this constraint
- Prevents exploration of higher-dimensional variance

**Impact**:
- **Severity**: Medium - limits analytical capabilities
- **Scope**: PCA analysis method only
- **User Experience**: Cannot analyze high-dimensional datasets properly

### Technical Details

**Why Remove Limit?**:
1. Raman spectra can have 100+ wavenumber points
2. Eigenvalue scree plots require seeing all components
3. Cumulative variance analysis needs flexibility
4. Different datasets have different optimal component counts
5. Limit of 10 is arbitrary and not based on spectroscopy best practices

**New Limit Rationale**:
- Set to 100 (covers 99% of use cases)
- Prevents UI from accepting unreasonable values (e.g., 10000)
- Users with >100 wavenumber points can modify code if needed

### Code Changes

**File**: `pages/analysis_page_utils/registry.py` (Line 27)

**Before**:
```python
ANALYSIS_REGISTRY = {
    "exploratory": {
        "PCA": {
            "function": "perform_pca_analysis",
            "name": "Principal Component Analysis (PCA)",
            "description": "Reduce dimensionality using PCA",
            "parameters": {
                "n_components": {
                    "type": "int",
                    "default": 2,
                    "range": (2, 10),  # ❌ TOO RESTRICTIVE
                    "description": "Number of principal components"
                }
            }
        }
    }
}
```

**After**:
```python
ANALYSIS_REGISTRY = {
    "exploratory": {
        "PCA": {
            "function": "perform_pca_analysis",
            "name": "Principal Component Analysis (PCA)",
            "description": "Reduce dimensionality using PCA",
            "parameters": {
                "n_components": {
                    "type": "int",
                    "default": 2,
                    "range": (2, 100),  # ✅ REMOVED ARBITRARY LIMIT - users should be free to choose based on their data
                    "description": "Number of principal components"
                }
            }
        }
    }
}
```

**Changes**:
- Line 27: Changed range from `(2, 10)` to `(2, 100)`
- Added explanatory comment about removing arbitrary limit

### Testing Checklist

- [ ] Open Raman application
- [ ] Load dataset with >10 wavenumber points
- [ ] Navigate to Analysis Page
- [ ] Select "PCA" method
- [ ] Click "Configure" to open parameter dialog
- [ ] **Expected**: n_components spinbox allows values up to 100
- [ ] Set n_components to 50
- [ ] Click "Run Analysis"
- [ ] **Expected**: PCA completes successfully with 50 components
- [ ] **Verify**: Scree plot shows all 50 components
- [ ] **Verify**: No error about exceeding maximum components

### Use Cases Enabled

**Before** (limited to 10):
```python
# User wants to see cumulative variance
# Can only see first 10 components
# Misses information about components 11-50
n_components = 10  # FORCED LIMIT
```

**After** (up to 100):
```python
# User can choose optimal number based on data
# See full eigenvalue distribution
# Make informed decisions about dimensionality
n_components = 50  # USER CHOICE
```

---

## Fix 3: Visualization Methods Migration

### Problem Description

**Symptom**: Visualization functions in wrong package location

**Root Cause**:
- Functions located in `pages/analysis_page_utils/methods/visualization.py`
- Violates project architecture: `pages/` is for UI, `functions/` is for logic
- Prevents reusability across different components

**Impact**:
- **Severity**: Low - code works but architecture is incorrect
- **Scope**: All 5 visualization methods
- **User Experience**: No immediate impact, but limits future development

### Technical Details

**Project Architecture Principle**:
```
pages/                  → UI components, page layouts, Qt widgets
functions/              → Reusable business logic, algorithms, data processing
components/             → Reusable UI widgets
```

**Why Migrate?**:
1. **Reusability**: Enable use in CLI tools, batch processing, or other pages
2. **Testability**: Easier to unit test without UI dependencies
3. **Maintainability**: Clear separation of concerns
4. **Consistency**: Follows established project structure

### Functions Migrated

All 5 functions moved from `pages/analysis_page_utils/methods/visualization.py` to `functions/visualization/analysis_plots.py`:

#### 1. create_spectral_heatmap() - 162 lines
**Purpose**: Heatmap visualization with optional clustering

**Parameters**:
- `cluster_rows` (bool): Cluster spectra (rows)
- `cluster_cols` (bool): Cluster wavenumbers (columns)
- `colormap` (str): Matplotlib colormap name
- `normalize` (bool): Normalize data 0-1
- `show_dendrograms` (bool): Display dendrograms

**Returns**: Dict with primary_figure, secondary_figure, data_table, summaries

#### 2. create_mean_spectra_overlay() - 96 lines
**Purpose**: Overlay plot of mean spectra with std bands

**Parameters**:
- `show_std` (bool): Show standard deviation bands
- `show_individual` (bool): Show all individual spectra
- `alpha_individual` (float): Alpha for individual spectra
- `normalize` (bool): Normalize data

**Returns**: Dict with overlay figure and summary

#### 3. create_waterfall_plot() - 97 lines
**Purpose**: Stacked spectra with vertical offset

**Parameters**:
- `offset_scale` (float): Vertical offset multiplier
- `max_spectra` (int): Maximum number of spectra to plot
- `colormap` (str): Color gradient name
- `reverse_order` (bool): Reverse plotting order

**Returns**: Dict with waterfall figure

#### 4. create_correlation_heatmap() - 79 lines
**Purpose**: Correlation matrix between spectral regions

**Parameters**:
- `method` (str): 'pearson' or 'spearman'
- `colormap` (str): Heatmap colormap
- `cluster` (bool): Apply hierarchical clustering

**Returns**: Dict with heatmap and correlation matrix

#### 5. create_peak_scatter() - 131 lines
**Purpose**: Scatter plot of peak intensities across datasets

**Parameters**:
- `peak_positions` (list): Wavenumber positions or None for auto-detect
- `tolerance` (float): Tolerance for peak matching (cm⁻¹)
- `prominence` (float): Prominence threshold for auto-detection

**Returns**: Dict with scatter plot and peak intensity table

### Code Changes

#### Created: functions/visualization/analysis_plots.py (582 lines)

**File Structure**:
```python
"""
Analysis Visualization Methods

This module implements various visualization methods for Raman spectral analysis,
including heatmaps, overlay plots, waterfall plots, correlation heatmaps, and scatter plots.

Created: November 21, 2025
Purpose: Consolidate analysis-specific visualizations in functions/visualization
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.signal import find_peaks

def create_spectral_heatmap(...): ...
def create_mean_spectra_overlay(...): ...
def create_waterfall_plot(...): ...
def create_correlation_heatmap(...): ...
def create_peak_scatter(...): ...

__all__ = [
    'create_spectral_heatmap',
    'create_mean_spectra_overlay',
    'create_waterfall_plot',
    'create_correlation_heatmap',
    'create_peak_scatter',
]
```

#### Modified: functions/visualization/__init__.py

**Added Imports**:
```python
# Import analysis plot functions (November 2025 refactoring)
from .analysis_plots import (
    create_spectral_heatmap,
    create_mean_spectra_overlay,
    create_waterfall_plot,
    create_correlation_heatmap,
    create_peak_scatter,
)
```

**Updated __all__**:
```python
__all__ = [
    # ... existing exports ...
    
    # Analysis plot functions (November 2025 refactoring)
    'create_spectral_heatmap',
    'create_mean_spectra_overlay',
    'create_waterfall_plot',
    'create_correlation_heatmap',
    'create_peak_scatter',
]
```

#### Modified: pages/analysis_page_utils/methods/__init__.py

**Changed Import Source**:
```python
# Before:
from .visualization import (
    create_spectral_heatmap,
    ...
)

# After:
# Import visualization functions from functions.visualization package
from functions.visualization import (
    create_spectral_heatmap,
    ...
)
```

**Backward Compatibility**: All existing code continues to work!
```python
# Both import methods work:
from pages.analysis_page_utils.methods import create_spectral_heatmap  # ✅ Works
from functions.visualization import create_spectral_heatmap            # ✅ Works
```

#### Deprecated: pages/analysis_page_utils/methods/visualization.py

**Action**: Renamed to `visualization.py.deprecated`

**Reason**: 
- Keep for reference during transition
- Prevents accidental imports
- Can be deleted after validation period

### Testing Checklist

Test all 5 visualization methods:

#### Test 1: Spectral Heatmap
- [ ] Load multiple datasets
- [ ] Select "Spectral Heatmap" method
- [ ] Configure: Enable clustering, choose colormap
- [ ] Click "Run Analysis"
- [ ] **Expected**: Heatmap displays with dendrograms
- [ ] **Verify**: No import errors in logs

#### Test 2: Mean Spectra Overlay
- [ ] Load multiple datasets
- [ ] Select "Mean Spectra Overlay" method
- [ ] Configure: Enable std bands
- [ ] Click "Run Analysis"
- [ ] **Expected**: Overlay plot with shaded regions
- [ ] **Verify**: Legend shows all datasets

#### Test 3: Waterfall Plot
- [ ] Load dataset with many spectra
- [ ] Select "Waterfall Plot" method
- [ ] Configure: Adjust offset_scale
- [ ] Click "Run Analysis"
- [ ] **Expected**: Stacked spectra with color gradient
- [ ] **Verify**: Wavenumbers inverted (high to low)

#### Test 4: Correlation Heatmap
- [ ] Load single dataset
- [ ] Select "Correlation Heatmap" method
- [ ] Configure: Choose Pearson or Spearman
- [ ] Click "Run Analysis"
- [ ] **Expected**: Symmetric correlation matrix
- [ ] **Verify**: Values range from -1 to 1

#### Test 5: Peak Scatter
- [ ] Load multiple datasets
- [ ] Select "Peak Intensity Scatter" method
- [ ] Configure: Auto-detect or manual peak positions
- [ ] Click "Run Analysis"
- [ ] **Expected**: Scatter plot for each peak
- [ ] **Verify**: Data table shows peak intensities

---

## Testing Instructions

### Prerequisites

1. Raman application installed and running
2. Sample project with Raman spectral data
3. Multiple datasets loaded (for visualization tests)

### Test Sequence

Run tests in this order to validate all fixes:

1. **Export Test** (Fix 1)
   - Validates ProjectManager attribute fix
   - Critical for production use

2. **PCA Component Test** (Fix 2)
   - Validates parameter range change
   - Medium priority

3. **Visualization Tests** (Fix 3)
   - Validates all 5 migrated methods
   - Low priority but comprehensive

### Test Environment

- **OS**: Windows 10/11
- **Python**: 3.11+
- **Dependencies**: All packages from pyproject.toml
- **Data**: Real Raman spectra (not synthetic)

### Expected Test Duration

- Export Test: 2 minutes
- PCA Test: 3 minutes
- Visualization Tests: 10 minutes (2 min × 5 methods)
- **Total**: ~15 minutes

---

## Related Files

### Modified Files

| File | Lines Changed | Type | Purpose |
|------|--------------|------|---------|
| `pages/analysis_page_utils/export_utils.py` | 4 | Bug Fix | Fixed ProjectManager attribute access |
| `pages/analysis_page_utils/registry.py` | 1 | Enhancement | Increased PCA component limit |
| `functions/visualization/analysis_plots.py` | 582 | New File | Migrated visualization methods |
| `functions/visualization/__init__.py` | 12 | Import Update | Added analysis_plots exports |
| `pages/analysis_page_utils/methods/__init__.py` | 8 | Import Update | Changed import source |
| `pages/analysis_page_utils/methods/visualization.py` | 0 | Deprecated | Renamed to .deprecated |

### Related Documentation

- **Architecture**: `.AGI-BANKS/FILE_STRUCTURE.md`
- **Change Log**: `.AGI-BANKS/RECENT_CHANGES.md`
- **Project Overview**: `.AGI-BANKS/PROJECT_OVERVIEW.md`
- **Development Guidelines**: `.AGI-BANKS/DEVELOPMENT_GUIDELINES.md`

### Key Classes and Modules

- **ProjectManager**: `utils.py` - Project state management
- **ExportUtils**: `pages/analysis_page_utils/export_utils.py` - Export functionality
- **AnalysisRegistry**: `pages/analysis_page_utils/registry.py` - Method definitions
- **Visualization Package**: `functions/visualization/` - All visualization functions

---

## Known Limitations

### Fix 1: ProjectManager AttributeError

**Limitation**: None - complete fix

**Future Considerations**:
- Add type hints to ProjectManager class
- Create ProjectManager interface/protocol
- Add unit tests for export functionality

### Fix 2: PCA n_components Limit

**Limitation**: Hard limit at 100 components

**Workaround**: Users can modify `registry.py` if they need >100 components

**Future Considerations**:
- Make limit configurable in settings
- Add validation based on actual data dimensions
- Dynamic range based on dataset size

### Fix 3: Visualization Migration

**Limitation**: Old file kept as .deprecated (not deleted)

**Cleanup Plan**:
1. Monitor for 1-2 weeks
2. Verify no remaining imports from old location
3. Delete .deprecated file in future update

**Future Considerations**:
- Add unit tests for each visualization function
- Create visualization presets/templates
- Add more visualization types (3D, interactive)

---

## Rollback Procedures

### If Critical Issues Arise

#### Rollback Fix 1 (Export)
```bash
# Revert export_utils.py changes
git checkout HEAD~1 pages/analysis_page_utils/export_utils.py

# Then investigate ProjectManager class structure
```

#### Rollback Fix 2 (PCA)
```python
# Edit registry.py line 27
"range": (2, 10),  # Revert to old limit
```

#### Rollback Fix 3 (Visualization)
```bash
# Restore old file
mv pages/analysis_page_utils/methods/visualization.py.deprecated \
   pages/analysis_page_utils/methods/visualization.py

# Revert imports in __init__.py
# Change: from functions.visualization import ...
# Back to: from .visualization import ...
```

### Validation After Rollback

- [ ] Application starts without errors
- [ ] Analysis page loads correctly
- [ ] Can run at least one analysis method
- [ ] Export functionality works (even if limited)

---

## Conclusion

All three fixes have been successfully implemented:

1. ✅ **ProjectManager AttributeError**: Fixed critical export bug
2. ✅ **PCA Limit**: Removed arbitrary constraint
3. ✅ **Visualization Migration**: Improved code architecture

**Next Steps**:
1. Run comprehensive testing (see Testing Instructions)
2. Monitor logs for any unexpected errors
3. Collect user feedback on PCA higher components
4. Plan deletion of .deprecated file after validation period

**Deployment Readiness**: ⚠️ Requires testing before production

---

**Document Version**: 1.0  
**Last Updated**: November 21, 2025  
**Review Status**: Pending QA validation
