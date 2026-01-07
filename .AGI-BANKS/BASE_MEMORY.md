# Base Memory - AI Agent Knowledge Base

> **Core knowledge and reference system for AI-assisted development**  
> **Last Updated**: December 18, 2025 - Dynamic Colormaps, Loading Overlay, Tab Preservation, Multi-Format Export

## 🎨 DYNAMIC COLOR PALETTE PATTERN (December 18, 2025) ⭐ CRITICAL

### Never Use Hardcoded Color Lists for Multi-Dataset Visualization

**CRITICAL**: Always use matplotlib colormaps for dynamic color generation to avoid `IndexError` with variable dataset counts:

```python
# ❌ WRONG (causes IndexError with >5 datasets)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, dataset in enumerate(datasets):
    ax.plot(..., color=colors[i])  # CRASHES if i >= 5!

# ✅ CORRECT (handles any number of datasets)
n_datasets = len(datasets)
cmap = plt.get_cmap('tab20', max(n_datasets, 10))
colors = [cmap(i) for i in range(n_datasets)]

for i, dataset in enumerate(datasets):
    ax.plot(..., color=colors[i])  # Always works!
```

**Recommended Colormaps**:
- `tab10`: Up to 10 distinct categories (default Matplotlib palette)
- `tab20`: Up to 20 distinct categories (best for research)
- `Set1`, `Set2`, `Set3`: Qualitative palettes (8-12 colors)

**Benefits**:
- Handles unlimited datasets (automatic cycling after max colors)
- Consistent with Matplotlib standards
- Better accessibility (colorblind-safe options available)
- No hardcoded maintenance

**Locations Implemented**:
- `pages/analysis_page_utils/methods/exploratory.py` - All multi-dataset plots
- `pages/analysis_page_utils/method_view.py` - Loading plot component colors

---

## 🔄 TAB PRESERVATION PATTERN (December 18, 2025) ⭐ CRITICAL

### Preserve User Context Across Results Updates

**CRITICAL**: When repopulating tabs (e.g., after rerunning analysis), preserve the current tab index:

```python
def populate_results_tabs(results_panel, result, ...):
    tab_widget = results_panel.tab_widget
    
    # ✅ PRESERVE current tab index BEFORE clearing
    current_tab_index = tab_widget.currentIndex() if tab_widget.count() > 0 else 0
    
    # Clear and repopulate tabs
    while tab_widget.count() > 0:
        tab_widget.removeTab(0)
    
    # ... add new tabs ...
    
    # ✅ RESTORE the previous tab index
    if current_tab_index < tab_widget.count():
        tab_widget.setCurrentIndex(current_tab_index)
    else:
        tab_widget.setCurrentIndex(0)  # Fallback if out of bounds
```

**Why This Matters**:
- Users often iterate on analysis parameters while viewing specific tabs
- Jumping to first tab on each run disrupts workflow
- Especially important for Loading Plot, Distributions, or secondary visualizations

**Location**: `pages/analysis_page_utils/method_view.py` - `populate_results_tabs()`

---

## ⏳ LOADING OVERLAY PATTERN (December 18, 2025) ⭐ CRITICAL

### Stacked Layout for Non-Blocking Visual Feedback

**CRITICAL**: Use `QStackedLayout` with `StackAll` mode to overlay loading indicator over content:

```python
from PySide6.QtWidgets import QStackedLayout, QWidget, QVBoxLayout, QLabel

# Create stacked layout for overlay support
stacked_layout = QStackedLayout(results_panel)
stacked_layout.setStackingMode(QStackedLayout.StackAll)  # ← CRITICAL

# Index 0: Main content
main_widget = QWidget()
main_layout = QVBoxLayout(main_widget)
# ... add normal widgets ...
stacked_layout.addWidget(main_widget)

# Index 1: Loading overlay (semi-transparent)
loading_overlay = QWidget()
loading_overlay.setStyleSheet("""
    QWidget {
        background-color: rgba(255, 255, 255, 0.85);
    }
""")
loading_layout = QVBoxLayout(loading_overlay)
loading_layout.setAlignment(Qt.AlignCenter)

loading_label = QLabel("⏳ Analyzing...")
loading_label.setStyleSheet("""
    font-size: 18px;
    font-weight: 600;
    color: #0078d4;
    background-color: white;
    padding: 20px 40px;
    border-radius: 8px;
    border: 2px solid #0078d4;
""")
loading_layout.addWidget(loading_label)

stacked_layout.addWidget(loading_overlay)
loading_overlay.setVisible(False)  # Hidden by default

# Helper methods
def show_loading():
    loading_overlay.setVisible(True)
    loading_overlay.raise_()

def hide_loading():
    loading_overlay.setVisible(False)

results_panel.show_loading = show_loading
results_panel.hide_loading = hide_loading
```

**Usage Pattern**:
```python
# Start long operation
results_panel.show_loading()

# ... run analysis in thread ...

# Finish operation
results_panel.hide_loading()
```

**Key Points**:
- Use `StackAll` mode (not `StackOne`)
- Semi-transparent background (85% opacity recommended)
- Call `raise_()` to ensure overlay is on top
- Always pair show/hide in try-finally or signal handlers

**Locations**:
- `pages/analysis_page_utils/method_view.py` - Overlay creation
- `pages/analysis_page.py` - Show/hide calls in `_run_analysis()` and `_on_analysis_finished()`

---

## 📤 MULTI-FORMAT EXPORT PATTERN (December 18, 2025) ⭐

### Format Selection Dialog Before File Export

**Pattern for supporting multiple export formats (CSV, Excel, JSON, etc.)**:

```python
from PySide6.QtWidgets import QDialog, QComboBox, QVBoxLayout, QHBoxLayout, QDialogButtonBox

def export_data_multi_format(data_table, default_filename="data"):
    # 1. Show format selection dialog
    dialog = QDialog()
    dialog.setWindowTitle("Export Data - Select Format")
    
    format_combo = QComboBox()
    formats = [
        ("csv", "CSV (Comma-Separated Values)"),
        ("xlsx", "Excel Spreadsheet (.xlsx)"),
        ("json", "JSON (JavaScript Object Notation)"),
        ("txt", "Text File (Tab-delimited)"),
        ("pkl", "Pickle (Python Binary)")
    ]
    for fmt_key, fmt_label in formats:
        format_combo.addItem(fmt_label, fmt_key)
    
    # ... add to dialog layout ...
    
    if dialog.exec() != QDialog.Accepted:
        return False
    
    # 2. Get file save path with appropriate filter
    selected_format = format_combo.currentData()
    file_path, _ = QFileDialog.getSaveFileName(
        parent,
        f"Export Data as {selected_format.upper()}",
        f"{default_filename}.{selected_format}",
        format_filters[selected_format]
    )
    
    # 3. Export based on format
    import pandas as pd
    df = pd.DataFrame(data_table) if isinstance(data_table, dict) else data_table
    
    if selected_format == "csv":
        df.to_csv(file_path, index=False)
    elif selected_format == "xlsx":
        df.to_excel(file_path, index=False, engine='openpyxl')
    elif selected_format == "json":
        df.to_json(file_path, orient='records', indent=2, force_ascii=False)
    elif selected_format == "txt":
        df.to_csv(file_path, sep='\\t', index=False)
    elif selected_format == "pkl":
        df.to_pickle(file_path)
    
    return True
```

**Requirements**:
- `openpyxl>=3.0.0` for Excel support
- Format dialog must match app styling
- Handle encoding for JSON (force_ascii=False for international characters)

**Location**: `pages/analysis_page_utils/export_utils.py` - `export_data_multi_format()`

---

## 📊 ANALYSIS PAGE VISUALIZATION PATTERNS (December 4, 2025) ⭐ CRITICAL

### DataFrame Row Iteration for QTableWidget

**CRITICAL**: When populating `QTableWidget` from `pd.DataFrame`, NEVER use `df.iterrows()` for row indices:

```python
# ❌ WRONG (causes TypeError when index is non-integer like "Peak_1000")
for i, row in df.iterrows():  # i could be string!
    for j, value in enumerate(row):
        table_widget.setItem(i, j, item)  # TypeError!

# ✅ CORRECT (enumerate always gives integer indices)
for row_idx, row_data in enumerate(df.values):  # row_idx is always int
    for col_idx, value in enumerate(row_data):
        item = QTableWidgetItem(str(value))
        table_widget.setItem(row_idx, col_idx, item)  # Works correctly
```

**Location**: `pages/analysis_page_utils/method_view.py` - `create_data_table_tab()`

### ComboBox Dropdown Styling (QAbstractItemView)

**CRITICAL**: ComboBox dropdown popups need explicit `QAbstractItemView` styling:

```python
QComboBox QAbstractItemView {
    background-color: white;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    selection-background-color: #e7f3ff;
    selection-color: #0078d4;
    outline: none;
    padding: 2px;
}

QComboBox QAbstractItemView::item {
    padding: 6px 10px;
    min-height: 24px;
}

QComboBox QAbstractItemView::item:hover {
    background-color: #f0f6fc;
}
```

### 3D Surface Plots with matplotlib Axes3D

**Pattern for 3D spectral visualization**:

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for surface
WN, SPEC = np.meshgrid(wavenumbers, range(n_spectra))

# Surface plot
surf = ax.plot_surface(WN, SPEC, data_matrix,
                       cmap=colormap, edgecolor='none', alpha=0.85)

# Add colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Intensity')

# Labels
ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11, labelpad=10)
ax.set_ylabel('Spectrum Index', fontsize=11, labelpad=10)
ax.set_zlabel('Intensity', fontsize=11, labelpad=10)
```

**Implemented in**: Waterfall Plot, Peak Intensity Scatter Plot

### Peak Intensity Statistics for Research

**Comprehensive statistics pattern**:

```python
# Calculate CV% for reproducibility assessment
cv_percent = (std_val / mean_val * 100) if mean_val != 0 else 0

# Interpretation for research notes
cv_interpretation = (
    "low variability" if cv < 10 else 
    "moderate variability" if cv < 25 else 
    "high variability"
)

# Statistics table columns
stats_columns = ['Dataset', 'Peak (cm⁻¹)', 'N', 'Mean', 'Std', 'CV (%)', 'Min', 'Max', 'Range']
```

**Research Value**: CV% < 10% indicates good measurement reproducibility.

---

## 🖼️ MATPLOTLIB WIDGET HEATMAP PATTERN (December 4, 2025) ⭐ CRITICAL

### AxesImage Handling in update_plot()

**CRITICAL**: When copying figures in `matplotlib_widget.py`, you MUST handle `AxesImage` objects (from `imshow()`):

```python
# matplotlib_widget.py - CORRECT PATTERN
from matplotlib.image import AxesImage

def update_plot(self, new_figure: Figure):
    # ... existing line/scatter/patch copying ...
    
    # ✅ MUST COPY: AxesImage objects (imshow/heatmaps)
    images = ax.get_images()
    if images:
        for img in images:
            img_data = img.get_array()
            extent = img.get_extent()
            cmap = img.get_cmap()
            clim = img.get_clim()
            
            new_ax.imshow(
                img_data,
                extent=extent,
                cmap=cmap,
                aspect='auto',
                vmin=clim[0],
                vmax=clim[1]
            )
        
        # Add colorbar for heatmaps
        self.figure.colorbar(new_ax.get_images()[-1], ax=new_ax)
```

**Common Mistake**:
```python
# ❌ WRONG: Only copying lines and collections (heatmaps display blank!)
for line in ax.get_lines():
    new_ax.plot(...)
for collection in ax.collections:
    # scatter plots, line collections...

# ✅ CORRECT: Also copy images
images = ax.get_images()  # Don't forget this!
```

### Position Preservation for Complex Layouts

For GridSpec layouts (e.g., dendrograms + heatmap), preserve original positions:

```python
# ✅ CORRECT: Preserve axes positions
pos = ax.get_position()
new_ax = self.figure.add_axes([pos.x0, pos.y0, pos.width, pos.height])

# ❌ WRONG: Simple sequential subplots (breaks GridSpec layouts)
new_ax = self.figure.add_subplot(len(axes_list), 1, i+1)
```

---

## 📊 MULTI-DATASET ANALYSIS PATTERNS (December 4, 2025) ⭐ CRITICAL

### Wavenumber Interpolation for Multi-Dataset Analysis

**CRITICAL**: When combining datasets with different wavenumber ranges, use `interpolate_to_common_wavenumbers()`:

```python
# exploratory.py - CORRECT PATTERN
from scipy.interpolate import interp1d

def interpolate_to_common_wavenumbers_with_groups(
    dataset_data: Dict[str, pd.DataFrame],
    group_labels_map: Optional[Dict[str, str]] = None,
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Interpolate datasets to common wavenumber grid."""
    # 1. Find common range (intersection)
    common_min = max(df.index.min() for df in dataset_data.values())
    common_max = min(df.index.max() for df in dataset_data.values())
    
    # 2. Create common grid
    common_wn = np.linspace(common_min, common_max, n_points)
    
    # 3. Interpolate each spectrum
    for df in dataset_data.values():
        interp_func = interp1d(df.index, spectrum, kind='linear')
        interpolated = interp_func(common_wn)
    
    return common_wn, X, labels
```

**Usage in Analysis Methods**:
```python
# ✅ CORRECT: Use interpolation
wavenumbers, X, labels = interpolate_to_common_wavenumbers_with_groups(
    dataset_data, group_labels_map=group_labels_map
)

# ❌ WRONG: Direct vstack (fails with different wavenumber ranges)
X = np.vstack([df.values.T for df in dataset_data.values()])
```

### High-Contrast Color Palette

Use `get_high_contrast_colors()` for all multi-group visualizations:

```python
def get_high_contrast_colors(num_groups: int) -> List[str]:
    if num_groups == 2:
        return ['#0066cc', '#ff4444']  # Blue, Red
    elif num_groups == 3:
        return ['#0066cc', '#ff4444', '#00cc66']  # Blue, Red, Green
    elif num_groups == 4:
        return ['#0066cc', '#ff4444', '#00cc66', '#ff9900']
    # ... etc for larger groups
```

---

## 🌐 LOCALIZATION PATTERN (November 19, 2025) ⭐ CRITICAL

### Single Initialization Rule
**CRITICAL**: LocalizationManager must be initialized **ONLY ONCE** in `utils.py`

**Correct Pattern**:
```python
# utils.py (SINGLE SOURCE OF TRUTH)
arg_lang = args.lang  # Command-line argument (default: 'ja')
CONFIGS = load_config()
LOCALIZEMANAGER = LocalizationManager(default_lang=arg_lang)

def LOCALIZE(key, **kwargs):
    return LOCALIZEMANAGER.get(key, **kwargs)
```

**Usage in Other Files**:
```python
# ANY other Python file
from utils import LOCALIZE  # ✅ USE THIS

# NOT this:
from configs.configs import LocalizationManager  # ❌ NEVER IMPORT THIS
localize = LocalizationManager()  # ❌ NEVER INSTANTIATE
```

### Common Mistakes to Avoid

❌ **WRONG**: Creating new instance
```python
from configs.configs import LocalizationManager
localize = LocalizationManager()  # Creates duplicate!
```

❌ **WRONG**: Calling set_language() after initialization
```python
# main.py
language = CONFIGS.get("language", "ja")
LOCALIZEMANAGER.set_language(language)  # Overrides command-line arg!
```

✅ **CORRECT**: Use global LOCALIZE function
```python
from utils import LOCALIZE
text = LOCALIZE("KEY.subkey")
```

### Why This Matters
- Command-line arguments (`--lang en`) must take priority over config files
- Multiple initializations cause language to be overridden
- Global state should be initialized once at module import
- Prevents "Successfully loaded language" appearing multiple times in logs

### Priority Hierarchy
1. **Command-line arguments** (highest priority) - set in utils.py
2. **Config file** (fallback) - currently not used
3. **Code defaults** (last resort) - argparse default='ja'

---

## 🔧 BUILD SYSTEM (October 21, 2025) ⭐ UPDATED

### Critical Build Workflow (Part 7) ⚠️ IMPORTANT

**MUST REBUILD After Spec Changes**:
```powershell
# 1. Edit spec file (add hidden imports, change config)
# 2. REBUILD (don't skip this!)
.\build_portable.ps1 -Clean
# 3. Test NEW exe (not old one!)
.\dist\raman_app\raman_app.exe
```

**Common Mistake**:
- Update spec file ✅
- Forget to rebuild ❌
- Test old exe ❌
- Report "fix doesn't work" ❌

**Verification**:
```powershell
# Check if rebuild needed:
(Get-Item build_scripts\raman_app.spec).LastWriteTime  # Spec modified
(Get-Item dist\raman_app).LastWriteTime  # Dist created
# If spec is newer → MUST REBUILD
```

**PowerShell Syntax**:
```powershell
# ✅ Correct:
.\build_portable.ps1 -Clean
.\build_portable.ps1 -Debug

# ❌ Wrong (Linux style):
.\build_portable.ps1 --clean
.\build_portable.ps1 --debug
```

**Path Handling**:
```powershell
# ✅ Use -LiteralPath for Unicode/special chars:
Get-Item -LiteralPath $path
Get-ChildItem -LiteralPath $path -Recurse

# ❌ Fails with Chinese/Unicode:
Get-Item $path
Get-ChildItem -Path $path
```

**Troubleshooting Reference**: `.docs/building/TROUBLESHOOTING.md`

### Build System Enhancements (Part 6)

**Critical Fixes**:
1. **PowerShell Path Handling**: Use `-LiteralPath` for paths with non-ASCII characters
2. **Scipy Hidden Imports**: Added scipy.stats submodules to prevent runtime errors
3. **Test Validation**: Check both direct and `_internal/` paths for assets
4. **Backup System**: Automatic timestamped backups before cleaning builds

**Backup System**:
```powershell
# Automatically creates backup_YYYYMMDD_HHmmss folder
# Moves (not deletes) existing build/dist before rebuild
# Example: build_backups/backup_20251021_150732/
```

**Scipy Runtime Fix**:
```python
# Required hidden imports for scipy.stats
hiddenimports += [
    'scipy.stats',
    'scipy.stats._stats_py',
    'scipy.stats.distributions',
    'scipy.stats._distn_infrastructure',
]
```

### PyInstaller Configuration
**Windows Builds**: Portable executable and NSIS installer

**Spec Files**:
- `raman_app.spec` (Portable) - 100 lines
- `raman_app_installer.spec` (Installer) - 100 lines

**Key Configuration**:
```python
# Hidden imports for complex packages
hiddenimports = [
    'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets',  # Qt6
    'numpy', 'pandas', 'scipy',  # Data processing
    'matplotlib.backends.backend_qt5agg',  # Visualization
    'ramanspy', 'pybaselines',  # Spectroscopy
    'torch', 'sklearn',  # ML (optional)
]

# Data files collection
datas = [
    ('assets/icons', 'assets/icons'),
    ('assets/fonts', 'assets/fonts'),
    ('assets/locales', 'assets/locales'),
]

# Binary files
binaries = [
    ('drivers/atmcd32d.dll', 'drivers'),  # Andor SDK
    ('drivers/atmcd64d.dll', 'drivers'),
]
```

**Build Scripts** (PowerShell):
- `build_portable.ps1` - Automated portable build (190 lines)
- `build_installer.ps1` - Installer staging build (180 lines)

**Test Suite**:
- `test_build_executable.py` - Comprehensive validation (500+ lines)

**NSIS Template**:
- `raman_app_installer.nsi` - Windows installer (100 lines)

**Build Workflow**:
```powershell
# 1. Build portable
.\build_portable.ps1

# 2. Test
python test_build_executable.py

# 3. Build installer (optional)
.\build_installer.ps1

# Output:
# - dist/raman_app/ (50-80 MB)
# - dist_installer/raman_app_installer_staging/ (same size)
# - raman_app_installer.exe (30-50 MB, if NSIS available)
```

**Documentation**:
- `.docs/building/PYINSTALLER_GUIDE.md` (600+ lines)

**Typical Output Sizes**:
- Portable .exe: 50-80 MB
- Total distribution: 60-100 MB
- Installer .exe: 30-50 MB (compressed)

**Testing Validation**:
- ✓ Executable structure
- ✓ Required directories (assets, PySide6, _internal)
- ✓ Asset files (icons, fonts, locales, data)
- ✓ Binary/DLL files
- ✓ Executable launch
- ✓ Performance baseline

### Build Script Fixes (October 21, 2025 - Part 3) ✅

**PowerShell Script Corrections**:
1. **build_portable.ps1**
   - Fixed try-catch block structure
   - Removed problematic emoji characters from code
   - Improved error handling
   - Status: ✅ Syntax correct, runs without errors

2. **build_installer.ps1**
   - Fixed nested block structure
   - Corrected variable interpolation
   - Improved parenthesis matching
   - Status: ✅ All parsing errors resolved

3. **test_build_executable.py**
   - Enhanced error messages with build command hints
   - Better user guidance
   - Status: ✅ Improved user experience

**Verification** ✅:
- [x] PowerShell scripts run without syntax errors
- [x] Error messages provide clear next steps
- [x] All blocks properly closed and structured

### Path Resolution Fix (October 21, 2025 - Part 4) ✅

**Problem**: Build scripts in `build_scripts/` subfolder couldn't locate parent directory files

**Solution Implemented**:

1. **Spec Files** (`raman_app.spec`, `raman_app_installer.spec`)
   ```python
   # Get project root (parent of build_scripts/)
   spec_dir = os.path.dirname(os.path.abspath(__file__))
   project_root = os.path.dirname(spec_dir)
   sys.path.insert(0, project_root)
   ```

2. **PowerShell Scripts** (`build_portable.ps1`, `build_installer.ps1`)
   ```powershell
   # Detect script location and project root
   $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
   $ProjectRoot = Split-Path -Parent $ScriptDir
   
   # Change to project root
   Push-Location $ProjectRoot
   # ... build ...
   Pop-Location  # Restore directory
   ```

3. **Python Test** (`test_build_executable.py`)
   ```python
   script_dir = os.path.dirname(os.path.abspath(__file__))
   project_root = os.path.dirname(script_dir)
   os.chdir(project_root)
   ```

**Benefits**:
- ✅ Scripts work from any directory
- ✅ Relative paths resolve correctly
- ✅ No hardcoded paths needed
- ✅ Proper error handling
- ✅ Directory restored after build

**Status** ✅:
- [x] Spec files detect project root correctly
- [x] Build scripts use Push/Pop-Location
- [x] Test script changes to project root
- [x] All relative paths resolve properly
- [x] Error handling with directory restoration

### PyInstaller CLI Argument Fix (October 21, 2025 - Part 5) ✅

**Problem**: PyInstaller 6.16.0 rejected the unsupported `--buildpath` option used by both build scripts, preventing builds from running.

**Solution**:
```powershell
$BuildArgs = @(
    '--distpath', $OutputDir,
    '--workpath', 'build'
)

$BuildArgs += 'raman_app.spec'
```
- Removed `--buildpath` and ensured all options precede the spec file
- Mirrored the fix in `build_installer.ps1` and appended `raman_app_installer.spec` last
- Cleanup lists now target only directories that PyInstaller actually generates (`build`, `dist`, `build_installer`, `dist_installer`)
- Spec files now compute their location via `Path(sys.argv[0])` fallback when `__file__` is missing

**Benefits**:
- ✅ PyInstaller CLI invocation succeeds without errors
- ✅ Works reliably with PyInstaller 6.16.0 behaviour
- ✅ Consistent command structure between portable and installer builds
- ✅ Clearer BuildArgs assembly with inline comment guidance
- ⚠️ Build currently fails later during module data collection (follow-up fix required for `collect_data_files` usage)

---

## 🎯 Purpose

This document serves as the foundational knowledge base for AI agents working on the Raman Spectroscopy Analysis Application. It provides quick access to essential information and references to detailed documentation.

## 🧪 Testing Standards (Updated October 15, 2025)

### Critical Testing Principles

#### 0. ALWAYS Implement Robust Parameter Type Validation (NEW - October 15, 2025) 🔒
**MANDATORY**: ALL preprocessing methods MUST handle parameter type conversions robustly.

**Why This Matters**:
- UI sliders send FLOATS (1.0, 1.2) but libraries may expect INTEGERS
- User input may come as STRINGS from text fields
- Optional parameters may be None but need proper handling
- Type mismatches cause RuntimeWarnings and failures

**Critical Pattern - Two-Layer Type Validation**:

```python
# Layer 1: Registry Level (functions/preprocess/registry.py)
def create_method_instance(self, category: str, method: str, params: Dict[str, Any] = None):
    """Create instance with robust type conversion."""
    if param_type == "int":
        # CRITICAL: Two-stage conversion handles all cases
        if value is None:
            converted_params[actual_key] = None
        else:
            converted_params[actual_key] = int(float(value))  # float() handles strings, int() converts
    
    elif param_type in ("float", "scientific"):
        if value is None:
            converted_params[actual_key] = None
        else:
            converted_params[actual_key] = float(value)
    
    elif param_type == "choice":
        choices = param_info[actual_key].get("choices", [])
        if choices and isinstance(choices[0], int):
            converted_params[actual_key] = int(float(value))  # Type-aware conversion
        elif choices and isinstance(choices[0], float):
            converted_params[actual_key] = float(value)
        else:
            converted_params[actual_key] = value
    
    elif param_type == "bool":
        if isinstance(value, bool):
            converted_params[actual_key] = value
        elif isinstance(value, str):
            converted_params[actual_key] = value.lower() in ('true', '1', 'yes')
        else:
            converted_params[actual_key] = bool(value)

# Layer 2: Class Level (defensive programming)
class FABCFixed:
    def __init__(self, lam=1e6, scale=None, num_std=3.0, diff_order=2, min_length=2, ...):
        # CRITICAL: Explicit type conversion at class level
        self.lam = float(lam)  # Ensure float
        self.scale = None if scale is None else float(scale)  # None-safe
        self.num_std = float(num_std)
        self.diff_order = int(diff_order)  # MUST be int!
        self.min_length = int(min_length)  # MUST be int!
        self.weights_as_mask = bool(weights_as_mask)
```

**Testing Requirements**:
1. Test with default parameters
2. Test with float parameters (UI slider case): 1.0, 2.0
3. Test with decimal floats (worst case): 1.2, 2.7
4. Test with string parameters: "1", "2.0"
5. Test with None for optional parameters
6. Test actual execution with synthetic data

**Common Failure Patterns**:
- `expected a sequence of integers or a single integer, got '1.0'` → Need int() conversion
- `extrapolate_window must be greater than 0` → Parameter passed as wrong type
- Type conversion only at one layer → Need defensive programming at both layers

#### 1. ALWAYS Verify Library Signatures Before Registry Updates
**MANDATORY PROCESS**: When adding/updating preprocessing methods in registry, MUST verify actual library signatures:

```python
# Example: Verify ASPLS parameters
import inspect
from ramanspy.preprocessing.baseline import ASPLS
sig = inspect.signature(ASPLS.__init__)
print('ASPLS parameters:', list(sig.parameters.keys()))
```

**Why This Matters**:
- ramanspy wrappers may NOT expose all pybaselines parameters
- Documentation can be misleading or outdated
- ASPLS issue: registry had `p_initial` and `asymmetric_coef`, but ramanspy only accepts `lam, diff_order, max_iter, tol, weights, alpha`
- Result: 50% of methods were broken due to incorrect parameter definitions

**Process**:
1. Import the actual class from ramanspy/pybaselines
2. Use `inspect.signature()` to get real parameters
3. Update registry to match EXACT library signature
4. Add parameter aliases ONLY for backward compatibility
5. Test with functional data, not just instantiation

#### 1.1 Bypassing Broken ramanspy Wrappers (FABC Pattern)
**WHEN TO USE**: If ramanspy wrapper has bugs that cannot be fixed by parameter adjustments

**FABC Case Study**: ramanspy.FABC has upstream bug (line 33 passes x_data incorrectly)
- **Problem**: `np.apply_along_axis(self.method, axis, data.spectral_data, x_data)` 
  - Passes x_data to function call, but pybaselines.fabc doesn't accept it in call signature
  - x_data should be in Baseline initialization, not method call
- **Solution**: Create custom wrapper calling pybaselines.api directly

**Pattern**:
```python
# functions/preprocess/fabc_fixed.py
from pybaselines import api
import numpy as np

class FABCFixed:
    """Custom FABC bypassing ramanspy wrapper bug."""
    
    def __init__(self, lam=1e5, scale=0.5, num_std=3, diff_order=2, min_length=2):
        self.lam = lam
        self.scale = scale
        self.num_std = num_std
        self.diff_order = diff_order
        self.min_length = min_length
    
    def _get_baseline_fitter(self, x_data: np.ndarray):
        # CORRECT: x_data in initialization
        return api.Baseline(x_data=x_data)
    
    def _process_spectrum(self, spectrum, x_data):
        fitter = self._get_baseline_fitter(x_data)
        # CORRECT: No x_data in method call
        baseline, params = fitter.fabc(
            data=spectrum,
            lam=self.lam,
            scale=self.scale,
            num_std=self.num_std,
            diff_order=self.diff_order,
            min_length=self.min_length
        )
        return spectrum - baseline
    
    def __call__(self, data, spectral_axis=None):
        # Container-aware wrapper
        # Handles both SpectralContainer and numpy arrays
        # Returns same type as input
```

**Registry Integration**:
```python
# functions/preprocess/registry.py
from .fabc_fixed import FABCFixed

"FABC": {
    "class": FABCFixed,  # Use custom wrapper, not ramanspy.FABC
    "description": "Fixed FABC implementation (bypassing ramanspy bug)",
    # ... parameters
}
```

**Testing Requirements**:
- Test baseline reduction (should be >95% for fluorescence)
- Test with SpectralContainer and numpy arrays
- Verify container type preserved
- Compare with pybaselines direct call (should match exactly)

#### 2. Testing Documentation Location
**CRITICAL**: Documentation location rules:
- **Test summaries/reports (.md)**: Save in `.docs/testing/` folder
- **Test scripts (.py)**: Save in `test_script/` folder
- **Test results (.txt, .json)**: Save in `test_script/results/` folder

**Directory Structure**:
```
.docs/
└── testing/
    ├── session2_comprehensive_testing.md
    ├── session3_functional_testing.md
    └── priority_fixes_progress.md

test_script/
├── test_preprocessing_comprehensive.py
├── test_preprocessing_functional.py
├── results/
│   ├── comprehensive_test_results_20251014_220536.txt
│   └── functional_test_results_20251014_224048.json
└── README.md
```

### Test Script Location
**CRITICAL**: ALL test scripts MUST be saved in the `test_script/` folder with their output results organized in subdirectories.

### Environment Requirements
1. **Always check the Python environment**: This project uses **UV** (not pip, conda, or poetry)
2. **Run scripts with UV**: Use `uv run python <script_name>` to ensure correct environment
3. **Example**:
   ```bash
   cd test_script
   uv run python test_preprocessing_comprehensive.py
   ```

## Testing Standards

### Test Script Organization
- **Scripts Location**: All test scripts MUST be in `test_script/` folder
- **Results Location**: Timestamped outputs in `test_script/results/` subfolder
- **Documentation Location**: Test summaries/reports in `.docs/testing/` folder
- **Environment**: ALWAYS use UV package manager (`uv run python test_script.py`)
- **Outputs**: Save timestamped results with format `test_results_YYYYMMDD_HHMMSS.txt`

### Test Coverage Requirements

#### 1. Structural Testing (Necessary but NOT Sufficient)
- ✓ Method exists in registry
- ✓ Parameters defined correctly
- ✓ Can instantiate class
- ✗ Does NOT guarantee functionality

#### 2. Functional Testing (REQUIRED for Production)
- ✓ Apply methods to realistic synthetic data
- ✓ Validate expected transformations
- ✓ Check output quality (no NaN/Inf)
- ✓ Test complete workflows (pipelines)
- ✓ Measure domain-appropriate metrics

**Key Principle**: "If it instantiates but doesn't work on real data, it's still broken!"

#### 3. Library Signature Verification (MANDATORY for Registry Updates)
- ✓ Use `inspect.signature()` to verify actual parameters
- ✓ Compare with ramanspy wrapper (may differ from pybaselines)
- ✓ Test with actual data after registry updates
- ✓ Document any wrapper limitations

### Current Test Scripts

1. **test_preprocessing_comprehensive.py** (Session 2)
   - Structural validation
   - 40/40 methods pass instantiation
   - Does NOT test functionality

2. **test_preprocessing_functional.py** (Session 3) ⭐ CRITICAL
   - Functional validation with synthetic Raman spectra
   - Tests tissue-realistic data
   - **FOUND**: 50% of methods have functional issues
   - **CRITICAL**: ASPLS parameter bug blocks ALL medical pipelines

### Test Data Generation Standards

For Raman preprocessing tests:
```python
# REQUIRED: Generate realistic synthetic spectra
- Fluorescence baseline (tissue autofluorescence)
- Gaussian peaks at biomolecular positions
- Realistic noise levels
- Occasional cosmic ray spikes
- Multiple tissue types (normal, cancer, inflammation)
```

### Validation Metrics (Category-Specific)

**DO NOT use universal metrics for all methods!**

```python
# Baseline Correction:
✓ Check residual variance reduction
✓ Validate baseline removed
✗ Don't expect SNR increase (removes mean!)

# Normalization:
✓ Check appropriate scaling (SNV: mean=0, std=1)
✓ Validate data range
✗ Don't use vector norm for all types!

# Denoising:
✓ Check high-frequency noise reduction
✓ Measure signal smoothing
✓ SNR improvement valid here

# Cosmic Ray Removal:
✓ Check spike elimination
✓ Validate outlier reduction
✓ SNR improvement expected

# Derivatives:
✓ Check zero-crossings
✓ Validate peak enhancement
✗ Don't expect positive SNR (emphasizes differences!)
```

### Medical Pipeline Testing

**REQUIRED for all medical diagnostic applications**:

1. Test complete workflows end-to-end
2. Use multiple tissue types
3. Validate tissue separability after preprocessing
4. Check each step for cascading failures
5. Document expected outcomes

**Critical Lesson**: One broken step blocks entire medical workflow!

### Known Critical Issues (October 2025)

**BLOCKER BUGS**:
1. **ASPLS Parameter Naming** 🚨
   - Registry uses 'p_initial', users expect 'p'
   - Blocks 3/6 medical pipelines
   - FIX: Accept both parameter names

2. **Method Name Inconsistencies**:
   - IAsLS vs IASLS
   - AirPLS vs AIRPLS
   - ArPLS vs ARPLS
   - FIX: Add aliases

3. **Calibration Methods**:
   - Need optional runtime inputs for testing
   - Cannot use in automated pipelines currently

### Test Execution Standards
1. **Comprehensive Coverage**: Test ALL implemented methods/features, not just a subset
2. **Deep Analysis**: Perform thorough validation including:
   - Parameter definitions and consistency
   - Method instantiation
   - Range validation
   - Error handling
3. **Output Reports**: Generate both text and JSON reports with timestamps
4. **Results Tracking**: Save results to `test_script/test_results_TIMESTAMP.txt`

### Current Test Scripts
- `test_preprocessing_comprehensive.py`: Tests all 40 preprocessing methods
  - Last run: 2025-10-14, Result: 40/40 passed (100%)
  - Output: `test_results_20251014_220536.txt`

## ⚠️ CRITICAL: Non-Maximized Window Design Constraint

**Updated**: October 6, 2025 (Evening #2)

### Design Principle
The application **MUST** work well in non-maximized window mode (e.g., 800x600 resolution). All UI sections must be optimized for smaller heights.

### Height Management Guidelines
1. **List Widgets**:
   - Calculate height based on actual item size measurement
   - **Dataset lists**: Show max **4 items** before scroll (**120px** height)
   - **Pipeline lists**: Show max **5 steps** before scroll (215px height)
   - **Item height**: Dataset items ~28px, Pipeline items ~40px

2. **Compact Controls**:
   - Button sizes: **28x28px** for compact headers (not 32x32px)
   - Icon sizes: **14x14px** in compact buttons (not 16x16px)
   - Spacing: **8px** between controls in compact layouts
   - Font sizes: Reduce by 1-2px in compact areas

3. **Section Headers**:
   - Explicit margins: **12px** all sides
   - Minimal spacing: **8px** between elements
   - No extra label containers (combine text + controls directly)

4. **Main Window**:
   - Minimum height: **600px** (for non-maximized mode)
   - Minimum width: **1000px**
   - Default size: 1440x900

5. **Applied in**:
   - ✅ Input Dataset Section: 100-120px (shows 4 items)
   - ✅ Pipeline Construction: 180-215px (shows 5 steps)
   - ✅ Visualization Header: Compact 28px buttons, 8px spacing
   - ✅ Visualization Plot: 300px minimum (reduced from 400px)
   - ✅ Main Window: 600px minimum height
   - ✅ Data Package Page: 12px/16px margins (Oct 14, 2025)

### Example Implementation
```python
# Dataset list - shows 4 items (actual item height ~28px)
list_widget.setMinimumHeight(100)
list_widget.setMaximumHeight(120)

# Pipeline list - shows 5 steps (actual item height ~40px)
self.pipeline_list.setMinimumHeight(180)
self.pipeline_list.setMaximumHeight(215)

# Visualization plot - compact
self.plot_widget.setMinimumHeight(300)  # Reduced from 400

# Main window constraints
self.setMinimumHeight(600)
self.setMinimumWidth(1000)

# Compact buttons
button.setFixedSize(28, 28)
icon = load_svg_icon(path, color, QSize(14, 14))
```

## 📋 Current Development Focus

### Active Tasks (See `.docs/TODOS.md` for details)
1. ✅ **Data Package Page Major Redesign** - COMPLETE (Oct 14, 2025)
   - Modern UI matching preprocessing page design
   - Multiple folder batch import (180x faster for 118 datasets)
   - Automatic metadata loading from JSON files
   - Real-time auto-preview with toggle control
   - Dataset selector for multiple dataset preview
   - Full localization (English + Japanese)
   - 456 lines of new code, production-ready
2. ✅ **Preprocessing Page Enhancements** - COMPLETE (Oct 10, 2025)
   - Dynamic parameter titles show category + step name
   - Gray border selection feedback for pipeline steps
   - Hint buttons added to all major sections
   - Complete pipeline import/export functionality
   - Clean code, no debug statements, production-ready
3. ✅ **Visualization Package Refactoring** - COMPLETE (Oct 1, 2025)
   - Extracted FigureManager to separate file (387 lines)
   - Reduced core.py to 4,405 lines (from 4,812)
   - Full backward compatibility maintained
   - Application tested and working
4. 📋 **RamanVisualizer Modularization** - PLANNED (See `.docs/functions/RAMAN_VISUALIZER_REFACTORING_PLAN.md`)
   - Phase 1-3 extraction (13-18 hours estimated)
   - Deferred to future sprint
5. ✅ **UI Improvements** - COMPLETE
   - Dataset list enhancement (4-6 items visible)
   - Export button styling (green, SVG icon)
   - Preview button width fix
6. ✅ **Export Feature Enhancements** - COMPLETE (Oct 3, 2025)
   - Metadata JSON export alongside datasets
   - Location validation with warning dialog
   - Default location persistence across exports
   - Multiple dataset batch export support

### Recent Completions (October 2025)
- ✅ **Data Package Page Major Redesign (Oct 14, 2025)**:
  - Batch import: 180x faster (30 min → 10 sec for 118 datasets)
  - Auto-metadata loading from JSON files
  - Real-time auto-preview with eye icon toggle
  - Dataset selector dropdown for multiple previews
  - Modern UI with hint buttons matching preprocessing page
  - 10 new localization keys (English + Japanese)
- ✅ **Preprocessing Page Enhancements (Oct 10, 2025)**:
  - Dynamic parameter section titles
  - Visual selection feedback with gray borders
  - Hint buttons for all major sections
  - Complete pipeline import/export system
  - Saved pipelines in projects/{project_name}/pipelines/
  - External pipeline import support
  - Rich pipeline preview in import dialog
- ✅ **Export Functionality (Oct 3, 2025)**:
  - Automatic metadata export in JSON format
  - Smart location validation and warnings
  - Last-used location persistence
  - Multi-dataset batch export capability
  - Comprehensive localization (EN/JA)
- ✅ Visualization package refactoring (visualization.py → visualization/)
- ✅ Removed original visualization.py file
- ✅ Comprehensive testing and validation
- ✅ Documentation reorganization (.docs/ structure)
- ✅ UI improvements (dataset list, export button, preview button)
- ✅ Fixed xlim padding (±50 wavenumber units)
- ✅ Enhanced parameter persistence
- ✅ Removed debug logging the Raman Spectroscopy Analysis Application. It provides quick access to essential information and references to detailed documentation.

## 📚 Documentation Structure

### Primary Documentation Hub: `.docs/`
All detailed documentation is centralized in the `.docs/` folder:
- **Task Management**: `.docs/TODOS.md` - Start here for current tasks
- **Architecture**: `.docs/main.md` - Application structure
- **Pages**: `.docs/pages/` - Page-specific documentation
- **Widgets**: `.docs/widgets/` - Widget system details
- **Functions**: `.docs/functions/` - Function library docs
- **Testing**: `.docs/testing/` - Test documentation

### AI Agent Knowledge Base: `.AGI-BANKS/`
High-level context and patterns (this folder):
- `BASE_MEMORY.md` - This file (quick reference)
- `PROJECT_OVERVIEW.md` - Architecture and design patterns
- `FILE_STRUCTURE.md` - Codebase organization
- `IMPLEMENTATION_PATTERNS.md` - Common coding patterns
- `RECENT_CHANGES.md` - Latest updates and fixes
- `DEVELOPMENT_GUIDELINES.md` - Coding standards
- `HISTORY_PROMPT.md` - Completed work archive

## 🚀 Quick Start

### For New Tasks
```
1. Check .docs/TODOS.md for current task details
2. Review PROJECT_OVERVIEW.md for architecture context
3. Check IMPLEMENTATION_PATTERNS.md for coding patterns
4. Review relevant .docs/ files for implementation details
5. Implement changes following DEVELOPMENT_GUIDELINES.md
6. Update both .docs/ and .AGI-BANKS as needed
```

### For Bug Fixes
```
1. Check RECENT_CHANGES.md for recent modifications
2. Review relevant .docs/pages/ or .docs/widgets/ files
3. Check FILE_STRUCTURE.md for file locations
4. Implement fix following patterns
5. Update TODOS.md and RECENT_CHANGES.md
```

## 🏗️ Project Architecture

### Technology Stack
- **Frontend**: PySide6 (Qt6) - Modern cross-platform GUI
- **Visualization**: matplotlib with Qt backend
- **Data Processing**: pandas, numpy, scipy
- **Configuration**: JSON-based with live reloading
- **Internationalization**: Multi-language support (EN/JA)

### Key Directories
```
raman-app/
├── main.py                    # Application entry point
├── pages/                     # Application pages (UI views)
│   ├── preprocess_page.py     # Main preprocessing interface
│   ├── data_package_page.py   # Data import/management
│   ├── analysis_page.py       # ⭐ NEW: Data analysis interface (Dec 2024)
│   │   └── analysis_page_utils/  # Analysis support modules
│   │       ├── result.py         # AnalysisResult dataclass
│   │       ├── registry.py       # 15+ method definitions
│   │       ├── thread.py         # Background processing
│   │       ├── widgets.py        # Parameter widget factory
│   │       └── methods/          # Analysis implementations
│   │           ├── exploratory.py    # PCA, UMAP, t-SNE, clustering
│   │           ├── statistical.py    # Comparison, peaks, ANOVA
│   │           └── visualization.py  # Heatmaps, overlays, plots
│   └── home_page.py           # Project management
├── components/                # Reusable UI components
│   ├── app_tabs.py           # Tab navigation
│   ├── toast.py              # Notifications
│   └── widgets/              # Custom widget library
├── functions/                 # Core processing functions
│   ├── data_loader.py        # File loading/parsing
│   ├── visualization/        # 📦 Visualization package (Oct 2025 refactor)
│   │   ├── __init__.py       # Package exports
│   │   ├── core.py           # RamanVisualizer class
│   │   └── figure_manager.py # Figure management
│   ├── preprocess/           # Preprocessing algorithms
│   └── ML.py                 # Machine learning
├── configs/                   # Configuration management
├── assets/                    # Static resources
│   ├── icons/                # SVG icons
│   ├── locales/              # EN/JA translations (ANALYSIS_PAGE keys added Dec 2024)
│   └── fonts/                # Typography
├── .docs/                     # 📚 Detailed documentation hub
└── .AGI-BANKS/               # 🤖 AI agent knowledge base
```

## 🎨 UI/UX Patterns

### Analysis Page Structure (NEW - December 2024) ⭐
- **Left Panel (400px)**: 
  - Dataset selection with Raw/Preprocessed/All filters
  - Method selection (category + method dropdowns)
  - Dynamic parameters (changes based on selected method)
  - Quick statistics (datasets, spectra, wavenumber range)
  - Run controls (Run, Cancel, Export, Clear)
- **Right Panel (expanding)**: 
  - Primary Visualization tab
  - Secondary Visualization tab (method-dependent)
  - Data Table tab
- **Key Features**: 
  - Multi-dataset selection with filters
  - 15+ analysis methods (PCA, UMAP, clustering, statistical tests, visualizations)
  - Background threading for responsive UI
  - Result caching to avoid recomputation
  - Comprehensive export (PNG, SVG, CSV, reports)

### Preprocessing Page Structure
- **Left Panel**: Dataset selection, pipeline building, output config
- **Right Panel**: Parameter controls, visualization
- **Key Features**: Global pipeline memory, parameter persistence, real-time preview

### Color Scheme
- **Primary**: Blue (`#1976d2`) - Active/enabled states
- **Secondary**: Light blue (`#64b5f6`) - Disabled states  
- **Success**: Green (`#2e7d32`)
- **Warning**: Orange (minimal use)
- **Error**: Red (`#dc3545`)
- **Neutral**: Grays for backgrounds and borders

### Widget Patterns
- Enhanced parameter widgets with validation
- Real-time value updates
- Visual feedback for errors
- Tooltip-based help system

## 🔧 Common Patterns

### Data Flow
```
1. Data Loading (data_loader.py)
   ↓
2. RAMAN_DATA global store (utils.py)
   ↓
3. Page UI (e.g., preprocess_page.py)
   ↓
4. Processing Pipeline (functions/preprocess/)
   ↓
5. Visualization (matplotlib_widget.py)
```

### Pipeline System
```
1. User selects category & method
2. PipelineStep created with parameters
3. Steps stored in pipeline_steps list
4. Global memory preserves across dataset switches
5. Real-time preview updates on changes
```

### File Operations
```python
# Import Pattern (data_loader.py)
- Support: CSV, TXT, ASC, Pickle
- Directory or single file
- Wavenumber as index, spectra as columns

# Export Pattern (to be implemented)
- Similar formats as import
- Selected dataset required
- Preserve metadata where possible
```

## � Critical Development Context

### GUI Application Architecture
**Important**: This is a GUI application built with PySide6. Output and debugging require:
1. **Log Files**: Check `logs/` folder for runtime information
   - `PreprocessPage.log` - Preprocessing operations
   - `data_loading.log` - Data import/export
   - `RamanPipeline.log` - Pipeline execution
   - `config.log` - Configuration changes
2. **Terminal Output**: Run `uv run main.py` to see console output
3. **No Direct Print**: GUI apps don't show print() in typical execution

### Environment Management
**Always check current environment before operations:**
- Project uses **uv** package manager (pyproject.toml)
- Commands: `uv run python script.py` or `uv run main.py`
- Virtual environment managed by uv automatically
- Dependencies: See `pyproject.toml` and `uv.lock`

### Documentation Standards (Required)
**All functions, classes, and features MUST include docstrings in this format:**

```python
def function_name(param1, param2):
    """
    Brief description of what the function does.
    
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
    
    Returns:
        type: Description of return value
    
    Raises:
        ExceptionType: When this exception is raised
    """
```

**For classes:**
```python
class ClassName:
    """
    Brief description of the class purpose.
    
    Attributes:
        attr1 (type): Description of attr1
        attr2 (type): Description of attr2
    """
```

Refer to existing code for examples of proper documentation style.

## �📋 Current Development Focus

### Active Tasks (See `.docs/TODOS.md` for details)
1. **Dataset Selection Highlighting** - Improve visual feedback (show 4-6 items with scroll)
2. **Export Button Enhancement** - Use SVG icon, green color, simplified text
3. **Preview Button Sizing** - Dynamic width based on text content
4. **Visualization.py Refactoring** - Convert to package folder for better organization
5. **Testing & Debugging** - Deep analysis and problem identification

### Recent Completions
- Fixed xlim padding (±50 wavenumber units)
- Enhanced parameter persistence
- Removed debug logging
- Organized documentation structure

## 🌐 Internationalization

### Locale System
- **Files**: `assets/locales/en.json`, `assets/locales/ja.json`
- **Usage**: `LOCALIZE("KEY.SUBKEY", param=value)`
- **Pattern**: Hierarchical keys (e.g., `PREPROCESS.input_datasets_title`)

### Adding Localized Strings
```json
// en.json
{
  "PREPROCESS": {
    "export_button": "Export Dataset",
    "export_formats": "Select Format"
  }
}

// ja.json  
{
  "PREPROCESS": {
    "export_button": "データセットをエクスポート",
    "export_formats": "形式を選択"
  }
}
```

## 🧪 Testing Protocol

### Validation Process
1. Create test documentation in `.docs/testing/feature-name/`
2. Implement feature with debug logging
3. Run terminal validation (45-second observation periods)
4. Document results (screenshots, terminal output)
5. Clean up test artifacts
6. Update `.docs/TODOS.md` and `.AGI-BANKS/RECENT_CHANGES.md`

### Test Documentation Structure
```
.docs/testing/
└── feature-name/
    ├── TEST_PLAN.md          # What to test
    ├── RESULTS.md            # Test outcomes
    ├── terminal_output.txt   # Console logs
    └── screenshots/          # Visual evidence
```

## 🔗 Key References

### Must-Read Documentation
- `.docs/TODOS.md` - Current tasks and priorities
- `.docs/pages/preprocess_page.md` - Main UI documentation
- `PROJECT_OVERVIEW.md` - Architecture overview
- `IMPLEMENTATION_PATTERNS.md` - Coding patterns

### External Resources
- PySide6 Documentation: https://doc.qt.io/qtforpython-6/
- matplotlib Qt Backend: https://matplotlib.org/stable/users/explain/backends.html
- pandas API: https://pandas.pydata.org/docs/

## 📝 Documentation Updates

### When to Update
- **Immediately**: Critical bugs, breaking changes
- **Per Feature**: New capabilities, UI changes
- **Per Task**: Task completion, progress updates
- **Weekly**: Review and consolidate changes

### What to Update
1. `.docs/TODOS.md` - Task progress
2. Relevant `.docs/` component files - Implementation details
3. `RECENT_CHANGES.md` - Summary of changes
4. This file (BASE_MEMORY.md) - If core patterns change

## 🎓 Development Guidelines

### Code Quality
- Follow PEP 8 style guide
- Use type hints where beneficial
- Document complex logic
- Keep functions focused and modular

### UI Development
- Maintain consistent styling
- Support both languages (EN/JA)
- Provide clear visual feedback
- Ensure accessibility

### Git Workflow
- Descriptive commit messages
- Reference issues/tasks in commits
- Keep commits focused
- Regular pushes to backup work

## ⚠️ CRITICAL: Project Loading & Memory Management (Oct 8, 2025)

### Project Loading Flow
**ALWAYS follow this exact sequence:**
```python
# In workspace_page.py load_project():
1. Clear all pages → clear_project_data() on each page
2. Load project → PROJECT_MANAGER.load_project(project_path)  # ← CRITICAL
3. Refresh pages → load_project_data() on each page
```

### Common Mistakes to Avoid
❌ **WRONG**: Calling non-existent `set_current_project()`
❌ **WRONG**: Not calling `PROJECT_MANAGER.load_project()` before `load_project_data()`
❌ **WRONG**: Forgetting to clear `RAMAN_DATA` in `clear_project_data()`

✅ **CORRECT**: `PROJECT_MANAGER.load_project(project_path)` populates `RAMAN_DATA`
✅ **CORRECT**: Clear pages → Load project → Refresh pages
✅ **CORRECT**: `RAMAN_DATA.clear()` in every `clear_project_data()` method

### Global State Management
- **RAMAN_DATA**: Dict[str, pd.DataFrame] in `utils.py` (line 16)
- **PROJECT_MANAGER**: Singleton instance in `utils.py` (line 219)
- **load_project()**: Reads pickle files, populates RAMAN_DATA (utils.py line 156)
- **clear_project_data()**: Must clear RAMAN_DATA explicitly

### Pipeline Index Safety
**ALWAYS access full pipeline list, then check if in filtered list:**
```python
# ❌ WRONG (causes index out of range):
current_step = steps[current_row]  # steps = enabled only

# ✅ CORRECT:
if current_row < len(self.pipeline_steps):
    current_step = self.pipeline_steps[current_row]  # Full list
    if current_step in steps:  # Check if enabled
        # Update parameters...
```

### Parameter Type Conversion
**Registry handles these types automatically:**
- `int` → int(value)
- `float` → float(value)
- `scientific` → float(value)  # 1e6 → 1000000.0
- `list` → ast.literal_eval(value)  # "[5,11,21]" → [5,11,21]
- `choice` → type detection from choices[0]

### Ramanspy Library Wrappers
**Created wrappers for buggy ramanspy methods:**
- `functions/preprocess/kernel_denoise.py` - Fixes numpy.uniform → numpy.random.uniform
- `functions/preprocess/background_subtraction.py` - Fixes array comparison issues

## 🆘 Troubleshooting

### Common Issues
1. **Import Errors**: Check sys.path manipulation in files
2. **Locale Missing**: Add keys to both en.json and ja.json
3. **UI Not Updating**: Check signal/slot connections
4. **Preview Issues**: Verify global memory persistence
5. **Project Won't Load**: Check PROJECT_MANAGER.load_project() is called
6. **Memory Persists**: Ensure RAMAN_DATA.clear() in clear_project_data()
7. **Pipeline Errors**: Validate index access patterns (see above)
8. **Parameter Errors**: Check param_info type definitions in registry

### Where to Look
- **UI Issues**: `.docs/pages/` documentation
- **Data Issues**: `.docs/functions/` and `data_loader.py`
- **Widget Issues**: `.docs/widgets/` documentation
- **Style Issues**: `configs/style/stylesheets.py`
- **Project Loading**: `utils.py` (ProjectManager class)
- **Pipeline Issues**: `pages/preprocess_page.py`, `pages/preprocess_page_utils/pipeline.py`
- **Type Conversion**: `functions/preprocess/registry.py`

---

**Version**: 1.1  
**Last Updated**: October 8, 2025  
**Next Review**: After full system testing

**Quick Links**:
- [TODOS](./../.docs/TODOS.md)
- [Project Overview](./PROJECT_OVERVIEW.md)
- [Recent Changes](./RECENT_CHANGES.md)
- [File Structure](./FILE_STRUCTURE.md)
- [Bug Fixes Report](../FINAL_BUG_FIX_REPORT.md)

---

##  MATPLOTLIB PATCH MANAGEMENT (November 20, 2025)  CRITICAL

### Patches Cannot Be Transferred Between Figures

**CRITICAL RULE**: Matplotlib patches (Ellipse, Rectangle, Polygon) belong to ONE figure and CANNOT be moved to another figure.

**The Problem**:
`python
#  WRONG (causes RuntimeError)
ellipse = Ellipse(xy=(x, y), ...)
ax1.add_patch(ellipse)  # Ellipse belongs to fig1

# Later, trying to display in new figure:
new_ax.add_patch(ellipse)  # RuntimeError: Can not put single artist in more than one figure
`

**The Solution - Recreate Patches**:
`python
#  CORRECT (recreate patch with same properties)
for patch in old_ax.patches:
    if isinstance(patch, Ellipse):
        new_ellipse = Ellipse(
            xy=patch.center,
            width=patch.width,
            height=patch.height,
            angle=patch.angle,
            facecolor=patch.get_facecolor(),
            edgecolor=patch.get_edgecolor(),
            linestyle=patch.get_linestyle(),
            linewidth=patch.get_linewidth(),
            alpha=patch.get_alpha(),
            label=patch.get_label()
        )
        new_ax.add_patch(new_ellipse)
`

### Implementation Pattern

**Location**: components/widgets/matplotlib_widget.py lines 200-240

**Type Checking Required**:
- Ellipse - PCA confidence intervals (CRITICAL for Chemometrics)
- Rectangle - Bar charts, histograms
- Polygon - Custom shapes (not yet implemented)

**Debug Logging**:
`python
print(f"[DEBUG] Recreating {len(ax.patches)} patches on new axis")
print(f"[DEBUG] Recreated ellipse at {patch.center} on new axis")
`

### Scientific Context

**Why This Matters**:
- Confidence ellipses are **mandatory** in Chemometrics PCA plots
- Prove statistical separation between groups (e.g., MM vs MGUS)
- Without ellipses, plots are scientifically incomplete
- Medical diagnostics depend on visual separation proof

**Previous Workaround**: Skip patches entirely  Loss of critical information
**Current Solution**: Recreate patches  Full functionality restored

### Common Failure Modes

1. **Attempting to copy patches directly**:
   `python
   new_ax.add_patch(patch)  # RuntimeError!
   `

2. **Not handling patch types**:
   `python
   # Ellipse and Rectangle have different constructors
   # Must use isinstance() to check type
   `

3. **Forgetting to preserve properties**:
   `python
   # Must copy: color, alpha, linewidth, linestyle, label
   `

### Testing Requirements

1. Create PCA plot with confidence ellipses
2. Display in matplotlib_widget
3. Verify terminal shows "Recreated ellipse" messages
4. Verify ellipses visible in UI
5. No RuntimeError


---

## 🎨 PCA VISUALIZATION BEST PRACTICES (December 2025) ⭐ CRITICAL

### Dual-Layer Ellipse Pattern

**CRITICAL RULE**: Confidence ellipses in PCA score plots should use a DUAL-LAYER pattern for optimal visibility.

**The Problem**:
```python
# ❌ WRONG (single layer with α=0.6 creates dark overlapping blobs)
ellipse = Ellipse(xy=(x, y), alpha=0.6, ...)
```

**The Solution - Dual-Layer Pattern**:
```python
# ✅ CORRECT - Two ellipses with different purposes
# Layer 1: Ultra-light fill (barely visible)
ellipse_fill = Ellipse(
    xy=(mean_x, mean_y), 
    facecolor=color,
    edgecolor='none',  # No edge on fill layer
    alpha=0.08,  # 8% opacity - barely visible fill
    zorder=5
)
ax.add_patch(ellipse_fill)

# Layer 2: Bold visible edge (clear boundary)
ellipse_edge = Ellipse(
    xy=(mean_x, mean_y), 
    facecolor='none',  # No fill on edge layer
    edgecolor=color,
    linewidth=2.5,  # Thick edge for visibility
    alpha=0.85,  # Strong edge visibility
    label=label,  # Only edge gets legend label
    zorder=15  # Above scatter points
)
ax.add_patch(ellipse_edge)
```

### MatplotlibWidget Ellipse Recreation

**Location**: `components/widgets/matplotlib_widget.py`

**CRITICAL**: When copying ellipses from source figures, must preserve the dual-layer pattern:
- Detect layer type (fill vs edge) based on alpha and colors
- Preserve zorder for proper layering
- Keep original alpha values intact

### Spectrum Preview Best Practices

**Mean-Only Display Pattern**:
```python
# ✅ CORRECT - Show only mean spectra with vertical offset
for idx, (name, df) in enumerate(dataset_data.items()):
    mean_spectrum = df.mean(axis=1).values
    mean_with_offset = mean_spectrum + offset
    
    ax.plot(wavenumbers, mean_with_offset,
            linewidth=2.8,
            label=f'{name} (mean, n={df.shape[1]})')
    
    # Ultra-subtle ±0.5σ envelope (optional)
    ax.fill_between(wavenumbers,
                    mean_with_offset - std * 0.5,
                    mean_with_offset + std * 0.5,
                    alpha=0.08)  # Barely visible
    
    offset = max_intensity * 1.15  # 15% spacing (RAMANMETRIX standard)
```

### Loading Plot Annotations

**Top 5 Peaks Per Component** (increased from 3):
```python
# Annotate top 5 peak positions for each PC
top_indices = np.argsort(np.abs(loadings))[-5:]

for peak_idx in top_indices:
    ax.annotate(f'{wavenumber:.0f}', 
               xy=(wavenumber, loading_value),
               xytext=(0, 10 if loading_value > 0 else -15),
               textcoords='offset points',
               fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', 
                        facecolor='white', alpha=0.8))
```

### Scree Plot Layout

**Side-by-Side GridSpec Pattern**:
```python
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(14, 5.5))
gs = GridSpec(1, 2, figure=fig, wspace=0.25)

# LEFT: Individual variance bar chart
ax_bar = fig.add_subplot(gs[0, 0])

# RIGHT: Cumulative variance line plot
ax_cum = fig.add_subplot(gs[0, 1])
```

### tight_layout Best Practices

**CRITICAL**: Call tight_layout AFTER all artists are added:
```python
# ✅ CORRECT order
for ax in axes:
    # ... add all lines, patches, annotations ...
    pass

# Then apply layout
try:
    fig.tight_layout(pad=1.2, rect=[0, 0.03, 1, 0.95])
except:
    fig.set_constrained_layout(True)  # Fallback

canvas.draw()
plt.close(source_figure)  # Prevent memory leak
```

### Legend Clarity for Confidence Ellipses

**Use descriptive labels**:
```python
# ❌ WRONG (ambiguous)
label=f'{dataset_label} 95% CI'

# ✅ CORRECT (clear)
label=f'{dataset_label} (95% Conf. Ellipse)'

# Add footnote for scientific papers
ax.text(0.02, 0.02,
        "95% Confidence Ellipses calculated using Hotelling's T² (1.96σ)",
        transform=ax.transAxes, fontsize=9, style='italic')
```

---

### Why Qt Signals Fail Silently

**CRITICAL ISSUE**: Qt signal connections can fail without Python exceptions. Only way to diagnose is comprehensive debug logging.

**The Problem**:
`python
# Signal connection might fail silently:
button_group.buttonClicked.connect(handler)
# User clicks button  NOTHING HAPPENS
# No error, no exception, no feedback
`

### Comprehensive Debug Strategy

**5-Stage Debug Logging** (Required for diagnosing button issues):

#### Stage 1: Button Creation
`python
button = QRadioButton("Label")
button.setObjectName("my_button")  # CRITICAL for debugging
print("[DEBUG] Created my_button")
`

#### Stage 2: Individual Click Handlers
`python
def on_button_clicked():
    print("[DEBUG] button.clicked() signal fired!")
    print(f"[DEBUG] Button is checked: {button.isChecked()}")

button.clicked.connect(on_button_clicked)
print("[DEBUG] Connected button.clicked signal")
`

**Why Individual Handlers**: 
- clicked() signal fires BEFORE uttonClicked() on button group
- If clicked() works but not uttonClicked(), group connection is broken
- If neither fires, button click not reaching Qt event loop (CSS/layout issue)

#### Stage 3: Button Group Setup
`python
button_group = QButtonGroup()
print("[DEBUG] Creating QButtonGroup")

button_group.addButton(button1, 0)
print(f"[DEBUG] Added button1 with ID 0")

button_group.addButton(button2, 1)
print(f"[DEBUG] Added button2 with ID 1")

print(f"[DEBUG] Button group contains {len(button_group.buttons())} buttons")
`

#### Stage 4: Signal Connection Verification
`python
button_group.buttonClicked.connect(toggle_function)
print("[DEBUG] Signal connected")
print(f"[DEBUG] Signal object: {button_group.buttonClicked}")
print(f"[DEBUG] Handler function: {toggle_function}")
print(f"[DEBUG] button1 checked: {button1.isChecked()}")
print(f"[DEBUG] button2 checked: {button2.isChecked()}")
`

#### Stage 5: Handler Function Logging
`python
def toggle_function(button):
    print("[DEBUG] toggle_function called")
    print(f"[DEBUG] Button clicked: {button}")
    print(f"[DEBUG] Button objectName: {button.objectName()}")
    
    if button == button1:
        print("[DEBUG] Switching to mode 1")
        # ... action
    elif button == button2:
        print("[DEBUG] Switching to mode 2")
        # ... action
    else:
        print("[DEBUG] WARNING: Button not recognized!")
`

### Diagnostic Flow

**Expected Terminal Output When Button Works**:
`
[DEBUG] button2.clicked() signal fired!
[DEBUG] Button is checked: True
[DEBUG] toggle_function called
[DEBUG] Button clicked: <PySide6.QtWidgets.QRadioButton(0x...) at 0x...>
[DEBUG] Button objectName: button2
[DEBUG] Switching to mode 2
`

**Troubleshooting Decision Tree**:

| Symptom                                  | Diagnosis                         | Solution                              |
| ---------------------------------------- | --------------------------------- | ------------------------------------- |
| No clicked() signal                      | Button click not reaching Qt      | Check CSS pointer-events, isEnabled() |
| clicked() fires, but not buttonClicked() | Button group connection broken    | Verify addButton() calls              |
| buttonClicked() fires, but wrong action  | Button identity comparison failed | Use objectName or button IDs          |
| All signals fire, but UI doesn't change  | Widget/state issue                | Check actual action code              |

### Common Failure Patterns

#### 1. Missing objectName
`python
#  WRONG (can't debug button identity)
button = QRadioButton("Label")

#  CORRECT (can identify in debug output)
button = QRadioButton("Label")
button.setObjectName("classification_button")
`

#### 2. No Individual Click Handlers
`python
#  WRONG (can't detect if click reaches Qt)
button_group.buttonClicked.connect(handler)

#  CORRECT (can detect click even if group fails)
button.clicked.connect(lambda: print("[DEBUG] Click detected"))
button_group.buttonClicked.connect(handler)
`

#### 3. No Button Group Verification
`python
#  WRONG (don't know if buttons added)
button_group.addButton(button1)
button_group.addButton(button2)

#  CORRECT (verify button count)
button_group.addButton(button1, 0)
print(f"[DEBUG] Added button1, group has {len(button_group.buttons())} buttons")
button_group.addButton(button2, 1)
print(f"[DEBUG] Added button2, group has {len(button_group.buttons())} buttons")
`

#### 4. Missing Initial State Verification
`python
#  WRONG (don't know initial state)
button1.setChecked(True)

#  CORRECT (verify state was set)
button1.setChecked(True)
print(f"[DEBUG] button1 checked: {button1.isChecked()}")
print(f"[DEBUG] button2 checked: {button2.isChecked()}")
`

### Qt Signal/Slot Architecture

**Signal Types**:
- **Widget-specific**: clicked(), 	oggled(), alueChanged()
- **Button group**: uttonClicked(QAbstractButton*)

**Signal Timing**:
1. User clicks button
2. Button's clicked() signal fires
3. Button group's uttonClicked() signal fires
4. Widget updates

**Connection Failures**:
- Type mismatch (signal emits int, slot expects str)
- Object deleted (button destroyed before signal fires)
- Lambda capture issues (wrong variable captured)

### Implementation Location

**File**: pages/analysis_page_utils/method_view.py lines 128-338

**Debug Statements Added**: 23 total
- Button creation: 2
- Individual click handlers: 4
- Button group setup: 8
- Signal connection: 7
- Initial state: 2

### Testing Requirements

1. Run app, navigate to page with button group
2. Verify terminal shows all creation messages
3. Click each button
4. Verify terminal shows:
   - clicked() signal fired!
   - 	oggle_function called
   - Correct button identification
   - Expected action messages
5. Verify UI changes as expected

### Scientific Impact

**Why This Matters for Raman App**:
- Classification mode button controls PCA group assignment
- Group assignment critical for medical diagnostics (MM vs MGUS vs Normal)
- Silent failure blocks entire analysis workflow
- Debug logging enables rapid troubleshooting during experiments

