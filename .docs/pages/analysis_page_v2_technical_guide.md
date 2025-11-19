# Analysis Page V2.0: Card-Based Architecture Guide

**Last Updated**: 2024-12-18  
**Type**: Implementation Documentation  
**Audience**: Developers and Maintainers

---

## Overview

This document provides comprehensive technical documentation for the Analysis Page V2.0, which implements a modern card-based architecture for Raman spectroscopy data analysis. The redesign focuses on improved user experience, modular code structure, and enhanced maintainability.

---

## Architecture

### Design Philosophy

The V2.0 architecture follows these principles:

1. **Separation of Concerns**: UI components are modularized into specialized utility modules
2. **View-Based State Management**: Clean transitions between startup and method views
3. **Lazy Loading**: Views are created only when needed to optimize performance
4. **Immutable History**: Analysis results are cached for instant recall without recomputation
5. **Declarative UI**: Component creation functions return fully configured widgets

### Component Hierarchy

```
AnalysisPage (Main Orchestration Layer)
├── Top Bar (Navigation)
│   ├── Back Button (← icon)
│   ├── Title Label (dynamic method name)
│   └── New Analysis Button (+ icon)
│
├── Content Splitter (Horizontal)
│   ├── History Sidebar (280px, collapsible)
│   │   ├── History Header
│   │   ├── History List Widget
│   │   └── Clear History Button
│   │
│   └── View Stack (QStackedWidget)
│       ├── [0] Startup View
│       │   ├── Welcome Header
│       │   ├── Subtitle Label
│       │   └── Scroll Area
│       │       └── Category Sections (3)
│       │           ├── Exploratory (5 method cards)
│       │           ├── Statistical (4 method cards)
│       │           └── Visualization (6 method cards)
│       │
│       └── [1] Method View (created dynamically)
│           └── Results Splitter (Horizontal)
│               ├── Input Form Panel (400px)
│               │   ├── Method Name + Description
│               │   ├── Dataset Selection Group
│               │   ├── Parameters Group
│               │   │   └── Dynamic Parameter Widgets
│               │   └── Action Buttons (Back, Run)
│               │
│               └── Results Panel (600px+)
│                   ├── Results Header + Export Buttons
│                   └── Results Tab Widget
│                       ├── Plot Tab (MatplotlibWidget)
│                       ├── Data Table Tab (QTableWidget)
│                       ├── Summary Tab (QTextEdit)
│                       └── Diagnostics Tab (QTextEdit, optional)
```

---

## Module Structure

### Core Module: `pages/analysis_page.py`

**Purpose**: Main orchestration and state management  
**Lines of Code**: ~700  
**Key Responsibilities**:
- View state transitions (startup ↔ method)
- Analysis execution coordination
- History management
- Dataset change handling
- Export operation delegation

**Key Classes**:

#### `AnalysisHistoryItem` (dataclass)
```python
@dataclass
class AnalysisHistoryItem:
    timestamp: datetime
    category: str
    method_key: str
    method_name: str
    dataset_name: str
    parameters: Dict[str, Any]
    result: Optional[AnalysisResult] = None
```

**Purpose**: Represents a single analysis in session history with full context for restoration.

#### `AnalysisPage` (QWidget)
```python
class AnalysisPage(QWidget):
    # Signals
    analysis_started = Signal(str, str)
    analysis_finished = Signal(str, str, object)
    error_occurred = Signal(str)
    
    # State
    current_view: str  # 'startup' | 'method'
    current_category: Optional[str]
    current_method_key: Optional[str]
    current_result: Optional[AnalysisResult]
    analysis_history: List[AnalysisHistoryItem]
    analysis_thread: Optional[AnalysisThread]
```

**Key Methods**:
- `_setup_ui()`: Initialize layout with top bar, sidebar, and view stack
- `_show_startup_view()`: Switch to card gallery view
- `_show_method_view(category, method_key)`: Create and display method view
- `_run_analysis(...)`: Execute analysis with threading
- `_on_analysis_finished(...)`: Handle results and update history
- `_export_png/svg/csv()`: Delegate to ExportManager
- `export_full_report()`: Public method for full report generation
- `save_to_project()`: Public method for project folder saving

---

### Utility Module: `pages/analysis_page_utils/views.py`

**Purpose**: Startup view, category sections, method cards, history sidebar, top bar  
**Lines of Code**: ~350

**Key Functions**:

#### `create_startup_view(localize_func, on_method_selected) -> QWidget`
Returns the main startup view with:
- Welcome header and subtitle
- Scroll area containing category sections
- 3-column grid layout for method cards
- Automatic card generation for all 15 methods

**Parameters**:
- `localize_func`: Localization function (LocalizationManager.get_text)
- `on_method_selected`: Callback (category, method_key) when card clicked

**Returns**: QWidget with `startupView` object name

#### `create_category_section(category_key, localize_func, on_method_selected) -> QWidget`
Returns a category section with:
- Category header (emoji + localized name)
- Grid layout with method cards (3 per row)
- Styled category divider

**Parameters**:
- `category_key`: 'exploratory' | 'statistical' | 'visualization'
- `localize_func`: Localization function
- `on_method_selected`: Card click callback

#### `create_method_card(category, method_key, method_info, localize_func, on_method_selected) -> QFrame`
Returns an individual method card with:
- Method name label (bold, 15px)
- Description label (gray, 12px, word-wrapped)
- Start Analysis button (primary blue)
- Hover effect (border color + shadow)
- Click handler (entire card clickable)

**Styling**:
```css
Background: #ffffff
Border: 1px solid #e0e0e0
Border Radius: 8px
Padding: 16px
Min Width: 280px
Max Width: 400px
Min Height: 180px

Hover:
Border Color: #0078d4
Box Shadow: 0 2px 8px rgba(0, 120, 212, 0.15)
```

#### `create_history_sidebar(localize_func) -> QWidget`
Returns history sidebar with:
- History title label
- QListWidget for history items
- Clear history button
- Stored references: `history_list`, `clear_btn`

**Width**: 280px (max), 200px (min)

#### `create_top_bar(localize_func, on_new_analysis) -> QWidget`
Returns top navigation bar with:
- Back button (← icon, hidden by default)
- Title label (📊 + method name)
- New Analysis button (+ icon, hidden in startup)
- Stored references: `new_analysis_btn`, `back_btn`, `title_label`

---

### Utility Module: `pages/analysis_page_utils/method_view.py`

**Purpose**: Method-specific input forms and results display  
**Lines of Code**: ~400

**Key Functions**:

#### `create_method_view(category, method_key, datasets, localize_func, on_run_analysis, on_back) -> QWidget`
Returns split-view method interface with:
- **Left Panel**: Input form with dataset selector, parameter widgets, action buttons
- **Right Panel**: Results display with export buttons and tab widget
- **Splitter**: Horizontal with sizes [400, 600]

**Stored Attributes**:
```python
method_widget.dataset_combo: QComboBox
method_widget.param_widgets: Dict[str, QWidget]
method_widget.run_btn: QPushButton
method_widget.back_btn: QPushButton
method_widget.results_panel: QWidget
method_widget.category: str
method_widget.method_key: str
```

**Parameter Widgets**:
Generated dynamically using `create_parameter_widgets()` from registry:
- `spinbox`: QSpinBox for integers
- `double_spinbox`: QDoubleSpinBox for floats
- `combo`: QComboBox for selections
- `checkbox`: QCheckBox for booleans

#### `create_results_panel(localize_func) -> QWidget`
Returns results display panel with:
- Results title label
- Export buttons row (PNG, SVG, CSV - hidden initially)
- QTabWidget for multiple result views
- Placeholder tab (shown before analysis runs)

**Stored Attributes**:
```python
results_panel.tab_widget: QTabWidget
results_panel.export_png_btn: QPushButton
results_panel.export_svg_btn: QPushButton
results_panel.export_data_btn: QPushButton
results_panel.results_title: QLabel
```

#### `populate_results_tabs(results_panel, result, localize_func, matplotlib_widget_class)`
Populates results tabs with analysis output:
- Removes placeholder tabs
- Shows export buttons
- Creates Plot tab (if figure exists)
- Creates Data Table tab (if data_table exists)
- Creates Summary tab (if summary exists)
- Creates Diagnostics tab (if diagnostics exists)

**Tab Types**:
1. **Plot Tab**: MatplotlibWidget with result.figure
2. **Data Table Tab**: QTableWidget with pandas DataFrame
3. **Summary Tab**: QTextEdit with monospace font
4. **Diagnostics Tab**: QTextEdit for diagnostic info

#### `create_data_table_tab(data_table) -> QWidget`
Converts pandas DataFrame or dict to styled QTableWidget with:
- Horizontal header labels
- Auto-sized columns
- Gridline styling
- Read-only cells

#### `create_summary_tab(summary_text) -> QWidget`
Creates read-only QTextEdit with:
- Monospace font (Consolas, Monaco)
- Plain text display
- 12px font size
- White background

---

### Utility Module: `pages/analysis_page_utils/export_utils.py`

**Purpose**: Export operations for all output formats  
**Lines of Code**: ~350

**Key Class**: `ExportManager(QObject)`

**Constructor**:
```python
def __init__(self, parent_widget, localize_func, project_manager):
    self.parent = parent_widget
    self.localize = localize_func
    self.project_manager = project_manager
```

**Key Methods**:

#### `export_plot_png(figure, default_filename='analysis_plot.png') -> bool`
Exports matplotlib figure to PNG with:
- DPI: 300 (high resolution)
- BBox: 'tight' (no whitespace)
- File dialog with PNG filter
- Success/error notifications

**Returns**: True if export succeeded

#### `export_plot_svg(figure, default_filename='analysis_plot.svg') -> bool`
Exports matplotlib figure to SVG (vector format) with:
- Format: SVG
- BBox: 'tight'
- File dialog with SVG filter

#### `export_data_csv(data_table, default_filename='analysis_data.csv') -> bool`
Exports data table to CSV with:
- Supports pandas DataFrame or dict
- Index excluded (index=False)
- UTF-8 encoding
- File dialog with CSV filter

#### `export_full_report(result, method_name, parameters, dataset_name) -> bool`
Creates complete analysis report folder with:
- `plot.png`: High-res plot (300 DPI)
- `data.csv`: Data table
- `report.txt`: Formatted text report with:
  - Method name and timestamp
  - Dataset information
  - Parameters used
  - Summary and diagnostics

**Folder Name**: `{method_name}_{YYYYMMDD_HHMMSS}/`

#### `save_to_project(result, method_name, parameters, dataset_name) -> bool`
Saves analysis to project folder with:
- Project path: `{project}/analyses/{method_name}_{timestamp}/`
- Files: `plot.png`, `data.csv`, `metadata.json`
- Metadata: JSON with method, dataset, parameters, timestamp, summary

#### `_get_default_export_path(filename) -> str`
Returns smart default export path:
- If project open: `{project}/exports/{filename}`
- Otherwise: `{home}/{filename}`

#### `_show_success(message)` / `_show_error(message)`
Display QMessageBox notifications with localized titles.

---

## State Management

### View States

```python
# Startup View
current_view = "startup"
top_bar.new_analysis_btn.visible = False
top_bar.back_btn.visible = False
view_stack.current_index = 0 (startup_view)

# Method View
current_view = "method"
current_category = "exploratory"
current_method_key = "pca"
top_bar.new_analysis_btn.visible = True
top_bar.back_btn.visible = True
view_stack.current_index = 1 (method_view)
```

### History Management

**Data Structure**:
```python
analysis_history: List[AnalysisHistoryItem] = [
    AnalysisHistoryItem(
        timestamp=datetime(2024, 12, 18, 14, 35, 20),
        category="exploratory",
        method_key="pca",
        method_name="PCA (Principal Component Analysis)",
        dataset_name="Control Group A",
        parameters={"n_components": 3, "normalize": True},
        result=AnalysisResult(...)  # Cached result
    ),
    ...
]
```

**Operations**:
- **Add**: Append new item on analysis completion
- **Display**: Reverse chronological order in sidebar
- **Restore**: Click item → show method view → set parameters → display cached result
- **Clear**: Confirmation dialog → empty list → update UI

### Result Caching

Benefits:
✅ Instant recall from history (no recomputation)  
✅ Preserves exact analysis state  
✅ Enables result comparison  
✅ Reduces computational overhead  

Trade-offs:
⚠️ Memory usage grows with history size  
⚠️ Not persisted across app restarts  

**Future Enhancement**: Implement cache size limit with LRU eviction

---

## Analysis Execution Flow

### 1. User Initiates Analysis

```
User clicks "Run Analysis" button
    ↓
_run_analysis(category, method_key, dataset_name, param_widgets)
    ↓
Validate dataset exists
    ↓
Extract parameters from widgets
    ↓
Disable run button, show "Running..." text
    ↓
Create AnalysisThread(category, method_key, dataset, parameters)
    ↓
Connect signals: finished, error, progress
    ↓
Emit analysis_started signal
    ↓
Start thread
```

### 2. Thread Execution

```
AnalysisThread.run()
    ↓
Get method function from registry
    ↓
Call method_func(dataset, **parameters)
    ↓
Emit progress signals (0% → 100%)
    ↓
Return AnalysisResult object
```

### 3. Result Handling

```
Thread emits finished signal with result
    ↓
_on_analysis_finished(result, category, method_key, dataset_name, parameters)
    ↓
Re-enable run button
    ↓
Store result in self.current_result
    ↓
Call populate_results_tabs(results_panel, result, ...)
    ↓
Create AnalysisHistoryItem
    ↓
Append to analysis_history
    ↓
Update history sidebar
    ↓
Emit analysis_finished signal
```

### 4. Error Handling

```
Thread emits error signal with message
    ↓
_on_analysis_error(error_msg)
    ↓
Re-enable run button
    ↓
Show QMessageBox.critical with error
    ↓
Emit error_occurred signal
```

---

## Localization Strategy

### Key Structure

All UI strings are localized using:
```python
self.localize = LocalizationManager.get_text
text = self.localize("ANALYSIS_PAGE.welcome_title")
```

### Naming Convention

```
{SECTION}.{component}_{type}
```

Examples:
- `ANALYSIS_PAGE.welcome_title`: Main title
- `ANALYSIS_PAGE.start_analysis_button`: Button text
- `ANALYSIS_PAGE.export_png_title`: Dialog title
- `ANALYSIS_PAGE.error_title`: Error dialog title

### String Formatting

```python
# Parameterized strings use {0}, {1}, etc.
message = self.localize("ANALYSIS_PAGE.export_success").format(file_path)
# "Successfully exported to: {0}" → "Successfully exported to: /path/to/file"

error = self.localize("ANALYSIS_PAGE.analysis_error").format(str(e))
# "Analysis failed: {0}" → "Analysis failed: Division by zero"
```

### Language Support

- **English** (`en.json`): Primary language, 37 new keys
- **Japanese** (`ja.json`): Full translation, 37 new keys

**Translation Quality**: Native-level Japanese translations with appropriate honorifics and technical terminology.

---

## Styling Guidelines

### Color Palette

```python
PRIMARY_BLUE = "#0078d4"       # Microsoft blue (buttons, links)
PRIMARY_BLUE_HOVER = "#006abc" # Darker blue (hover state)
PRIMARY_BLUE_PRESS = "#005a9e" # Darkest blue (pressed state)

TEXT_PRIMARY = "#2c3e50"       # Dark gray (headers)
TEXT_SECONDARY = "#6c757d"     # Medium gray (descriptions)
TEXT_DISABLED = "#c0c0c0"      # Light gray (disabled)

BORDER_DEFAULT = "#e0e0e0"     # Light border
BORDER_HOVER = "#0078d4"       # Highlighted border

BG_WHITE = "#ffffff"           # Card background
BG_GRAY = "#f8f9fa"            # Sidebar background
BG_HOVER = "#e7f3ff"           # Light blue hover
```

### Typography Scale

```python
FONT_SIZES = {
    "title": 24,      # Welcome header
    "heading": 18,    # Section headers
    "subheading": 15, # Method names
    "body": 13,       # Descriptions
    "small": 12,      # Data table, code
    "icon": 20        # Icon buttons
}

FONT_WEIGHTS = {
    "normal": 400,
    "medium": 500,
    "semibold": 600,
    "bold": 700
}
```

### Component Styling

#### Method Cards
```css
background-color: #ffffff
border: 1px solid #e0e0e0
border-radius: 8px
padding: 16px
min-width: 280px
max-width: 400px
min-height: 180px
cursor: pointer

:hover
    border-color: #0078d4
    box-shadow: 0 2px 8px rgba(0, 120, 212, 0.15)
```

#### Primary Buttons
```css
background-color: #0078d4
color: white
border: none
border-radius: 4px
font-weight: 600
padding: 8px 16px
min-height: 36px

:hover
    background-color: #006abc

:pressed
    background-color: #005a9e

:disabled
    background-color: #c0c0c0
```

#### Group Boxes (Parameters, Dataset Selection)
```css
border: 1px solid #e0e0e0
border-radius: 4px
padding-top: 16px
margin-top: 8px
font-weight: 600

::title
    padding: 0 4px
    left: 12px
```

---

## Error Handling

### Validation Layers

1. **Pre-Analysis Validation**
   - Dataset existence check
   - Dataset not found → QMessageBox.critical
   
2. **Parameter Extraction**
   - Try-except blocks for each widget
   - Failed extraction → Skip parameter (non-blocking)
   - Log error to console
   
3. **Thread Execution**
   - Thread catches all exceptions
   - Emits error signal with traceback
   - Main thread shows QMessageBox.critical
   
4. **Export Operations**
   - File dialog cancellation → Silent return
   - Export errors → QMessageBox.critical with error details
   - Success → QMessageBox.information

### User Notifications

All error messages follow this pattern:
```python
QMessageBox.critical(
    self,
    self.localize("ANALYSIS_PAGE.error_title"),      # "Analysis Error"
    self.localize("ANALYSIS_PAGE.analysis_error").format(error_msg)
)
```

Success notifications:
```python
QMessageBox.information(
    self,
    self.localize("ANALYSIS_PAGE.export_success_title"),
    self.localize("ANALYSIS_PAGE.export_success").format(file_path)
)
```

---

## Performance Considerations

### Optimizations

1. **Lazy View Creation**
   - Method view created only when method selected
   - Old method view destroyed before creating new one
   - Reduces memory footprint

2. **Result Caching**
   - History items store full AnalysisResult
   - Click history → instant display (no recomputation)
   - Trade-off: Memory vs. speed

3. **Threaded Analysis**
   - AnalysisThread prevents UI blocking
   - Progress signals update UI (0% → 100%)
   - Cancellation support (future enhancement)

4. **Dynamic Widget Generation**
   - Parameters generated only for selected method
   - Reduces startup time
   - Enables flexible method definitions

5. **Efficient Card Layout**
   - 3-column grid with QGridLayout
   - Scroll area for overflow
   - Fixed card sizes prevent relayout

### Bottlenecks

- **Large Datasets**: matplotlib rendering can be slow (>1000 spectra)
  - Solution: Implement downsampling or preview mode
  
- **History Growth**: Unbounded list with cached results
  - Solution: Implement LRU cache with size limit (e.g., 50 items)
  
- **Export Operations**: Large PNG exports take time
  - Solution: Show progress dialog for operations >2s

---

## Testing Strategy

### Unit Tests (Recommended)

```python
# Test view creation
def test_create_startup_view():
    view = create_startup_view(mock_localize, mock_callback)
    assert view.objectName() == "startupView"
    
def test_create_method_card():
    card = create_method_card("exploratory", "pca", {...}, mock_localize, mock_callback)
    assert card.minimumWidth() == 280

# Test state transitions
def test_show_startup_view():
    page = AnalysisPage()
    page._show_startup_view()
    assert page.current_view == "startup"
    assert not page.top_bar.new_analysis_btn.isVisible()

# Test export operations
def test_export_png_success(tmp_path):
    manager = ExportManager(mock_parent, mock_localize, mock_project_manager)
    result = manager.export_plot_png(mock_figure, str(tmp_path / "test.png"))
    assert result == True
    assert (tmp_path / "test.png").exists()
```

### Integration Tests

```python
def test_full_analysis_workflow():
    page = AnalysisPage()
    
    # Start from startup
    assert page.current_view == "startup"
    
    # Select method
    page._show_method_view("exploratory", "pca")
    assert page.current_view == "method"
    assert page.current_category == "exploratory"
    
    # Run analysis
    page._run_analysis("exploratory", "pca", "Dataset 1", mock_widgets)
    # Wait for thread completion...
    
    # Check history
    assert len(page.analysis_history) == 1
    assert page.analysis_history[0].method_key == "pca"
```

### Manual Testing Checklist

See `.AGI-BANKS/RECENT_CHANGES_analysis_page_v2.md` for comprehensive checklist.

---

## Troubleshooting

### Common Issues

**Issue**: Method cards not displaying  
**Cause**: ANALYSIS_METHODS registry empty or malformed  
**Solution**: Check `analysis_page_utils/registry.py` for correct structure

**Issue**: Parameters not generating  
**Cause**: `create_parameter_widgets()` not found  
**Solution**: Ensure `components/widgets/parameter_widgets.py` is imported

**Issue**: Export buttons not showing after analysis  
**Cause**: `populate_results_tabs()` not setting visibility  
**Solution**: Check that `results_panel.export_*_btn.setVisible(True)` is called

**Issue**: History items not clickable  
**Cause**: Signal not connected  
**Solution**: Verify `history_list.itemClicked.connect(self._on_history_item_clicked)`

**Issue**: Localization keys missing  
**Cause**: Keys not added to en.json/ja.json  
**Solution**: Add missing keys following naming convention

---

## Future Development

### Planned Features

1. **Splitter Position Persistence**
   ```python
   # Save splitter state
   QSettings().setValue("analysis_page/splitter_sizes", splitter.sizes())
   
   # Restore on startup
   sizes = QSettings().value("analysis_page/splitter_sizes", [400, 600])
   splitter.setSizes(sizes)
   ```

2. **Favorites System**
   - Star icon on method cards
   - "Favorites" category at top of startup view
   - Persistence in QSettings

3. **Batch Analysis**
   - Multi-dataset selection in method view
   - Run analysis on all selected datasets
   - Combined results view

4. **Comparison View**
   - Side-by-side result display
   - Difference highlighting
   - Synchronized scrolling

5. **Custom Methods**
   - Plugin system for user-defined methods
   - Method definition wizard
   - Python script integration

---

## API Reference

### Public Methods

```python
class AnalysisPage(QWidget):
    # Signals
    analysis_started: Signal(str, str)
    analysis_finished: Signal(str, str, object)
    error_occurred: Signal(str)
    
    # Public Methods
    def export_full_report() -> None:
        """Export complete analysis report with user file dialog."""
        
    def save_to_project() -> None:
        """Save current analysis results to project folder."""
```

### Public Functions (Utilities)

```python
# views.py
def create_startup_view(localize_func, on_method_selected) -> QWidget
def create_category_section(category_key, localize_func, on_method_selected) -> QWidget
def create_method_card(...) -> QFrame
def create_history_sidebar(localize_func) -> QWidget
def create_top_bar(localize_func, on_new_analysis) -> QWidget

# method_view.py
def create_method_view(...) -> QWidget
def create_results_panel(localize_func) -> QWidget
def populate_results_tabs(results_panel, result, localize_func, matplotlib_widget_class) -> None
def create_data_table_tab(data_table) -> QWidget
def create_summary_tab(summary_text) -> QWidget

# export_utils.py
class ExportManager:
    def export_plot_png(figure, default_filename) -> bool
    def export_plot_svg(figure, default_filename) -> bool
    def export_data_csv(data_table, default_filename) -> bool
    def export_full_report(result, method_name, parameters, dataset_name) -> bool
    def save_to_project(result, method_name, parameters, dataset_name) -> bool
```

---

## Changelog

### Version 2.0.0 (2024-12-18)

**Added**:
- Card-based startup view with method gallery
- Category-based method organization
- Split-view method interface (input + results)
- Comprehensive export functionality (PNG/SVG/CSV/reports)
- Analysis history sidebar with session tracking
- Modular component architecture
- 37 new localization keys (English + Japanese)
- Enhanced error handling and notifications

**Changed**:
- Redesigned main layout (stacked widgets vs. fixed panels)
- Improved visual design with hover effects
- Enhanced state management
- Better separation of concerns

**Deprecated**:
- Old left-panel/right-panel layout (preserved in backup file)

**Removed**:
- None (backward compatible)

---

**Document Version**: 1.0  
**Last Updated**: 2024-12-18  
**Maintainer**: Development Team
