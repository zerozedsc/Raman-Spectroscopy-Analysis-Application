# Implementation Patterns and Best Practices

## Code Architecture Patterns

### 0.0.0. Method Card with Preview Images Pattern (November 19, 2025) 🆕⭐🖼️
**PURPOSE**: Display analysis method cards with preview images showing typical visualization outputs  
**CONTEXT**: Analysis page method selection cards, any card-based method gallery interface

**✅ CORRECT Pattern - Method Card with Image**:
```python
import os
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap

# Step 1: Create method-to-image mapping dictionary
METHOD_IMAGES = {
    "pca": "pca_analysis.png",
    "umap": "umap.png",
    "tsne": "t-sne.png",
    "spectral_heatmap": "spectral_heatmap.png",
    "waterfall_plot": "waterfall.png",
    # ... more mappings
}

def create_method_card(method_key, method_info, localize_func, on_click):
    """Create card with preview image at top."""
    card = QFrame()
    card.setObjectName("methodCard")
    card.setMinimumHeight(200)  # Increased for image
    card.setCursor(Qt.PointingHandCursor)
    
    layout = QVBoxLayout(card)
    layout.setSpacing(8)
    
    # Step 2: Add image if available (TOP of card)
    if method_key in METHOD_IMAGES:
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        
        # Resolve path relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        image_path = os.path.join(base_dir, "assets", "image", 
                                 METHOD_IMAGES[method_key])
        
        # Load and scale image
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # Scale with quality (max 240×120px)
                scaled_pixmap = pixmap.scaled(
                    240, 120,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                image_label.setPixmap(scaled_pixmap)
                image_label.setStyleSheet("""
                    QLabel {
                        background-color: #f8f9fa;
                        border-radius: 4px;
                        padding: 4px;
                    }
                """)
                layout.addWidget(image_label)
    
    # Step 3: Add method name (after image)
    name_label = QLabel(localize_func(f"METHOD.{method_key}"))
    name_label.setStyleSheet("font-weight: 600; font-size: 14px;")
    layout.addWidget(name_label)
    
    # Step 4: Add description
    desc_label = QLabel(localize_func(f"METHOD_DESC.{method_key}"))
    desc_label.setWordWrap(True)
    layout.addWidget(desc_label)
    
    layout.addStretch()
    
    # Step 5: Add action button (bottom)
    btn = QPushButton(localize_func("start_analysis"))
    btn.clicked.connect(lambda: on_click(method_key))
    layout.addWidget(btn)
    
    return card
```

**Image Specifications**:
- **Location**: `assets/image/` folder (PNG files)
- **Max Dimensions**: 240px width × 120px height
- **Scaling**: Maintain aspect ratio with smooth transformation
- **Background**: Light gray (#f8f9fa) with 4px border radius
- **Positioning**: Top of card, centered
- **Padding**: 4px around image

**Card Layout Order** (Top to Bottom):
1. Preview Image (if available)
2. Method Name (bold, 14px)
3. Description (word-wrapped)
4. Stretch spacer
5. Action Button

**Why This Pattern Works**:
1. **Visual Learning**: Users see output before selecting method
2. **Fast Recognition**: Images faster to scan than text
3. **Professional Look**: Images make interface more polished
4. **Clear Expectations**: Users know what method produces
5. **Graceful Degradation**: Works even if images missing

**Benefits**:
- Improved user experience with visual previews
- Faster method selection
- Reduced learning curve for new users
- Professional, modern interface design

**❌ INCORRECT Pattern - Don't hardcode paths**:
```python
# DON'T: Hardcode absolute paths
image_path = "C:/Users/me/project/assets/image/pca.png"

# DON'T: Use relative paths without resolution
image_path = "../../../assets/image/pca.png"

# DON'T: Skip existence checks
pixmap = QPixmap(image_path)
image_label.setPixmap(pixmap)  # May fail silently
```

**Best Practices**:
1. Always resolve paths relative to project root
2. Check file existence before loading
3. Validate pixmap is not null before displaying
4. Use smooth transformation for high quality scaling
5. Maintain aspect ratio to prevent distortion
6. Provide light background for image separation
7. Position images at top for maximum impact

**Related Patterns**:
- Icon Loading Pattern (0.0)
- Card-Based UI Pattern (Section 3)
- Modern Card-Based UI Pattern (October 2025)

---

### 0.0. Icon Loading Pattern (October 16, 2025) 🆕⭐
**PURPOSE**: Standardized icon loading using centralized icon management  
**CONTEXT**: All UI components requiring icons (buttons, toolbars, widgets)

**✅ CORRECT Pattern - Use icons.py API**:
```python
from components.widgets.icons import load_icon, create_button_icon, get_icon_path
from PySide6.QtCore import QSize

# Method 1: Load icon with color customization (most common)
icon = load_icon("icon_name", QSize(width, height), "#hex_color")

# Method 2: Load icon without color (use original SVG colors)
icon = load_icon("icon_name", QSize(width, height))

# Method 3: Use predefined size string keys
icon = load_icon("icon_name", "button")  # Uses DEFAULT_SIZES["button"] = QSize(16, 16)
icon = load_icon("icon_name", "toolbar")  # Uses DEFAULT_SIZES["toolbar"] = QSize(24, 24)

# Method 4: Specialized button icon creation
icon = create_button_icon("icon_name", "#0078d4")

# Examples from codebase:
import_icon = load_icon("load_project", QSize(14, 14), "#28a745")
export_icon = load_icon("export", QSize(14, 14), "#0078d4")
plus_icon = load_icon("plus", QSize(24, 24), "white")
trash_icon = load_icon("trash_bin", QSize(14, 14), "#dc3545")
```

**❌ INCORRECT Pattern - Don't use low-level utils directly**:
```python
# DON'T: Direct path-based loading (deprecated)
from utils import load_svg_icon
icon = load_svg_icon(get_icon_path("icon_name"), "#color", QSize(w, h))

# DON'T: Using ICON_PATHS dictionary directly
icon = load_svg_icon(ICON_PATHS["icon_name"], "#color", QSize(w, h))

# DON'T: Manual path construction
icon_path = os.path.join(os.path.dirname(__file__), "assets", "icons", "icon.svg")
icon = load_svg_icon(icon_path, "#color", QSize(w, h))
```

**Why Use icons.py API?**:
1. **Name-based loading**: No path manipulation needed
2. **Consistent API**: Single source of truth for icon loading
3. **Type safety**: Better IDE support and error detection
4. **Maintainability**: Centralized icon management
5. **Flexibility**: Supports size presets, color customization

**Parameter Order Note**:
- `load_icon(icon_name, size, color)` - Note: size comes before color!
- This differs from the old `load_svg_icon(path, color, size)` pattern

**Internal Architecture** (For Reference Only):
```python
# utils.py - Low-level function (used internally by icons.py)
def load_svg_icon(path: str, color: Qt.GlobalColor = None, size: QSize = QSize(48, 48)) -> QIcon:
    # Handles SVG rendering with color customization
    pass

# components/widgets/icons.py - High-level API (USE THIS)
def load_icon(icon_name: str, size=None, color: Optional[str] = None) -> QIcon:
    """
    Load icon by name with optional size and color customization.
    
    Args:
        icon_name: Icon identifier (key from ICON_PATHS)
        size: QSize object or string key from DEFAULT_SIZES
        color: Optional hex color (#RRGGBB) or Qt color name
    """
    icon_path = get_icon_path(icon_name)
    if color is not None:
        return load_svg_icon(icon_path, color, size)  # Calls utils internally
    else:
        return QIcon(icon_path)
```

**Migration Checklist**:
- [x] Replace all `load_svg_icon(get_icon_path("name"), color, size)` calls
- [x] Replace all `load_svg_icon(ICON_PATHS["name"], color, size)` calls
- [x] Replace all manual path construction with `load_icon("name", size, color)`
- [x] Update imports: `from components.widgets.icons import load_icon`
- [x] Remove unused `from utils import load_svg_icon` imports
- [x] Verify all icons display correctly after migration

---

### 0.1. Tab-Aware Selection Pattern (October 16, 2025) 🆕⭐
**PURPOSE**: Implement selection operations that respect active tab context  
**CONTEXT**: Multi-tab interfaces where each tab has its own dataset/item list

**Pattern Implementation**:
```python
class PreprocessPage:
    def __init__(self):
        # Create separate list widgets for each tab
        self.dataset_list_all = QListWidget()
        self.dataset_list_raw = QListWidget()
        self.dataset_list_preprocessed = QListWidget()
        
        # Reference to current tab's list (updated on tab change)
        self.dataset_list = self.dataset_list_all
        
        # Connect tab change signal
        self.dataset_tabs.currentChanged.connect(self._on_dataset_tab_changed)
    
    def _on_dataset_tab_changed(self, index: int):
        """Update active list reference when tab changes."""
        if index == 0:
            self.dataset_list = self.dataset_list_all
        elif index == 1:
            self.dataset_list = self.dataset_list_raw
        elif index == 2:
            self.dataset_list = self.dataset_list_preprocessed
        
        # Update any tab-dependent UI state here
        self._update_tab_dependent_state(index)
    
    def _toggle_select_all_datasets(self):
        """Select/deselect all items in current tab only."""
        current_list = self.dataset_list  # Uses current tab's list
        
        total_items = current_list.count()
        if total_items == 0:
            return
        
        selected_items = current_list.selectedItems()
        all_selected = len(selected_items) == total_items
        
        if all_selected:
            current_list.clearSelection()  # Deselect all
        else:
            current_list.selectAll()  # Select all
```

**Key Principles**:
1. **Single Reference**: Maintain one `self.dataset_list` reference that points to current tab's list
2. **Tab Change Handler**: Update reference in `_on_dataset_tab_changed()`
3. **Toggle Logic**: Check if all selected → deselect, otherwise → select all
4. **Zero Items Check**: Always check if list is empty before operations

**UI Integration**:
```python
# Add select all button to title bar
select_all_btn = self._create_action_icon_button(
    "checkmark", "#0078d4", LOCALIZE("PREPROCESS.select_all_tooltip")
)
select_all_btn.clicked.connect(self._toggle_select_all_datasets)
title_layout.addWidget(select_all_btn)
```

**Localization Keys**:
- `select_all_tooltip`: "Select/deselect all datasets in current tab"
- JA: "現在のタブのすべてのデータセットを選択/選択解除"

**Related Files**:
- `pages/preprocess_page.py` (lines 485-513, 658-695)
- `assets/locales/en.json` (line 221)
- `assets/locales/ja.json` (line 200)

---

### 0.2. Intelligent State Management Pattern (October 16, 2025) 🆕⭐
**PURPOSE**: Automatically adjust UI state based on context changes (e.g., tab switches)  
**CONTEXT**: Features that should behave differently based on data type or context

**Pattern: Preview Toggle with Tab-Aware Defaults**
```python
def _on_dataset_tab_changed(self, index: int):
    """Update UI state based on tab type."""
    # Update active reference
    if index == 0:
        self.dataset_list = self.dataset_list_all
    elif index == 1:
        self.dataset_list = self.dataset_list_raw
    elif index == 2:
        self.dataset_list = self.dataset_list_preprocessed
    
    # Adjust preview toggle based on tab type
    # Raw datasets (tabs 0 & 1) should have preview ON
    # Preprocessed datasets (tab 2) should have preview OFF
    if index in [0, 1]:  # All or Raw datasets
        if not self.preview_toggle_btn.isChecked():
            self.preview_toggle_btn.blockSignals(True)
            self.preview_toggle_btn.setChecked(True)
            self.preview_toggle_btn.blockSignals(False)
            self._update_preview_toggle_button_style()
    elif index == 2:  # Preprocessed datasets
        if self.preview_toggle_btn.isChecked():
            self.preview_toggle_btn.blockSignals(True)
            self.preview_toggle_btn.setChecked(False)
            self.preview_toggle_btn.blockSignals(False)
            self._update_preview_toggle_button_style()
    
    # Trigger selection changed event
    self._on_dataset_selection_changed()
```

**Key Principles**:
1. **Context-Dependent Defaults**: State depends on current context (tab or dataset type)
2. **Signal Blocking**: Use `blockSignals(True/False)` to prevent unwanted events
3. **State Consistency**: Update related UI elements after state change
4. **Rationale Documentation**: Comment explaining why state differs per context
5. **Dataset Type Detection**: Check metadata to determine raw vs preprocessed

**Enhanced Pattern: Dataset Type-Aware State Management**:
```python
def _on_dataset_selection_changed(self):
    """Adjust preview toggle based on selected dataset type."""
    selected_items = self.dataset_list.selectedItems()
    
    if len(selected_items) == 1:
        dataset_name = self._clean_dataset_name(selected_items[0].text())
        metadata = PROJECT_MANAGER.get_dataframe_metadata(dataset_name)
        is_preprocessed = metadata and metadata.get('is_preprocessed', False)
        
        if is_preprocessed:
            # Preprocessed dataset → Force preview OFF
            if self.preview_toggle_btn.isChecked():
                self.preview_toggle_btn.blockSignals(True)
                self.preview_toggle_btn.setChecked(False)
                self.preview_toggle_btn.blockSignals(False)
                self._update_preview_toggle_button_style()
            self._last_selected_was_preprocessed = True
        else:
            # Raw dataset → Force preview ON
            # Works for: first load, raw-to-raw, preprocessed-to-raw
            if not self.preview_toggle_btn.isChecked():
                self.preview_toggle_btn.blockSignals(True)
                self.preview_toggle_btn.setChecked(True)
                self.preview_toggle_btn.blockSignals(False)
                self._update_preview_toggle_button_style()
            self._last_selected_was_preprocessed = False
```

**State Decision Matrix**:
| Context | Tab | Dataset Type | Preview State |
|---------|-----|--------------|---------------|
| Tab change | All/Raw | N/A | ON |
| Tab change | Preprocessed | N/A | OFF |
| Selection | Any | Raw | ON |
| Selection | Any | Preprocessed | OFF |
| First load | Any | Raw | ON |
| First load | Any | Preprocessed | OFF |

**Rationale Example**:
```python
# Default state: ON for raw datasets, OFF for preprocessed
# Reasoning:
# - Raw data needs preview to show processing effects
# - Preprocessed data should NOT be previewed to avoid double processing
# - Tab changes AND dataset selection both trigger state updates
self.preview_toggle_btn.setChecked(True)  # Default for raw datasets
```

**Signal Management**:
```python
# Always block signals during programmatic changes
widget.blockSignals(True)
widget.setValue(new_value)
widget.blockSignals(False)

# Then manually trigger any required updates
self._update_dependent_ui()
```

**Related Files**:
- `pages/preprocess_page.py` (lines 658-681 tab change, 707-798 selection change, 1007-1010 initialization)

---

### 0.3. Dialog Styling Consistency Pattern (October 16, 2025) 🆕⭐
**PURPOSE**: Maintain consistent visual design across all dialogs  
**CONTEXT**: QDialog instances need unified styling matching application theme

**Standard Dialog Styling**:
```python
def _create_styled_dialog(self, title_key: str, min_width: int = 500, 
                         min_height: int = None) -> QDialog:
    """Create dialog with standardized styling."""
    dialog = QDialog(self)
    dialog.setWindowTitle(LOCALIZE(title_key))
    dialog.setModal(True)
    dialog.setMinimumWidth(min_width)
    if min_height:
        dialog.setMinimumHeight(min_height)
    
    # Standard dialog stylesheet
    dialog.setStyleSheet("""
        QDialog {
            background-color: #ffffff;
        }
        QLabel {
            color: #2c3e50;
        }
        QLineEdit, QTextEdit {
            padding: 8px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: #ffffff;
            color: #2c3e50;
        }
        QLineEdit:focus, QTextEdit:focus {
            border-color: #0078d4;
        }
        QListWidget {
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: #ffffff;
        }
        QListWidget::item {
            padding: 4px;
            border-bottom: 1px solid #e9ecef;
        }
        QListWidget::item:selected {
            background-color: #e7f3ff;
            border-left: 3px solid #0078d4;
        }
        QListWidget::item:hover {
            background-color: #f8f9fa;
        }
        QPushButton {
            padding: 8px 16px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: #f8f9fa;
            color: #2c3e50;
        }
        QPushButton:hover {
            background-color: #e9ecef;
            border-color: #adb5bd;
        }
        QPushButton#ctaButton {
            background-color: #0078d4;
            color: white;
            border: none;
        }
        QPushButton#ctaButton:hover {
            background-color: #006abc;
        }
    """)
    
    return dialog
```

**Color Scheme**:
- **Background**: `#ffffff` (white)
- **Text**: `#2c3e50` (dark gray)
- **Borders**: `#ced4da` (light gray)
- **Primary Actions**: `#0078d4` (blue)
- **Hover States**: `#e9ecef` (lighter gray)
- **Selection**: `#e7f3ff` (light blue)

**Button Patterns**:
```python
# Primary CTA button (blue)
cta_btn = QPushButton(LOCALIZE("action_label"))
cta_btn.setObjectName("ctaButton")
cta_btn.setDefault(True)

# Secondary cancel button (gray)
cancel_btn = QPushButton(LOCALIZE("COMMON.cancel"))
# No special objectName needed
```

**Layout Pattern**:
```python
layout = QVBoxLayout(dialog)
layout.setContentsMargins(20, 20, 20, 20)
layout.setSpacing(16)

# Add content widgets...

# Button layout (always at bottom)
button_layout = QHBoxLayout()
button_layout.addStretch()  # Push buttons right
button_layout.addWidget(cancel_btn)
button_layout.addWidget(cta_btn)
layout.addLayout(button_layout)
```

**Related Files**:
- `pages/preprocess_page.py` (lines 1903-1955, 2079-2125)

---

### 0.4. Preview Toggle State Synchronization Pattern (October 21, 2025) 🆕⭐
**PURPOSE**: Synchronize toggle button visual state, checked state, and internal flags  
**CONTEXT**: Preview systems with ON/OFF toggle that must maintain consistent state across UI and logic

**🚨 Common Anti-Pattern - Missing State Synchronization**:
```python
# ❌ INCORRECT - Only updates checked state, visual out of sync
if condition:
    self.preview_toggle_btn.setChecked(True)
    # Missing: Button text still shows "OFF", preview_enabled flag not updated

# ❌ INCORRECT - Calls non-existent method
self.preview_toggle_btn.setChecked(False)
self._update_preview_toggle_button_style()  # Method doesn't exist!

# ❌ INCORRECT - Duplicate method definitions (Python uses last one)
def _update_preview_button_state(self, enabled):
    # First definition with hardcoded text
    self.preview_toggle_btn.setText("プレビュー オン")
    
def _update_preview_button_state(self, enabled):
    # Second definition with localization (this one is used)
    text = LOCALIZE("PREPROCESS.UI.preview_on" if enabled else "PREPROCESS.UI.preview_off")
    self.preview_toggle_btn.setText(text)
```

**✅ CORRECT Pattern - Three-Way Synchronization**:
```python
# Pattern: Always update ALL THREE components together
def _update_preview_state(self, enabled: bool):
    """Update preview state with full synchronization."""
    # 1. Update button checked state
    self.preview_toggle_btn.blockSignals(True)
    self.preview_toggle_btn.setChecked(enabled)
    self.preview_toggle_btn.blockSignals(False)
    
    # 2. Update button visual (text, icon, style)
    self._update_preview_button_state(enabled)
    
    # 3. Update internal flag
    self.preview_enabled = enabled

def _update_preview_button_state(self, enabled: bool):
    """Update button visual appearance only."""
    # Use localized text
    text = LOCALIZE("PREPROCESS.UI.preview_on" if enabled else "PREPROCESS.UI.preview_off")
    self.preview_toggle_btn.setText(text)
    
    # Optional: Update icon or stylesheet based on state
    # icon = load_icon("eye-open" if enabled else "eye-close", QSize(16, 16))
    # self.preview_toggle_btn.setIcon(icon)
```

**Real-World Example - Dataset Type Detection**:
```python
def _on_dataset_selection_changed(self):
    """Handle dataset selection with automatic preview state adjustment."""
    selected_items = self.dataset_list.selectedItems()
    if not selected_items:
        return
    
    # Detect if dataset is preprocessed
    first_item = selected_items[0]
    dataset_name = self._clean_dataset_name(first_item.text())
    is_preprocessed = (
        dataset_name in self.project_manager.datasets 
        and self.project_manager.datasets[dataset_name].get("is_preprocessed", False)
    )
    
    # Auto-adjust preview state based on dataset type
    if is_preprocessed:
        # Preprocessed dataset → Force preview OFF (avoid double preprocessing)
        if self.preview_toggle_btn.isChecked():
            self.preview_toggle_btn.blockSignals(True)
            self.preview_toggle_btn.setChecked(False)
            self.preview_toggle_btn.blockSignals(False)
            self._update_preview_button_state(False)  # ✅ Update visual
            self.preview_enabled = False  # ✅ Update flag
    else:
        # Raw dataset → Force preview ON (show processing effects)
        if not self.preview_toggle_btn.isChecked():
            self.preview_toggle_btn.blockSignals(True)
            self.preview_toggle_btn.setChecked(True)
            self.preview_toggle_btn.blockSignals(False)
            self._update_preview_button_state(True)  # ✅ Update visual
            self.preview_enabled = True  # ✅ Update flag
    
    # Update preview display
    self._update_preview()
```

**Tab Change Pattern**:
```python
def _on_dataset_tab_changed(self, index: int):
    """Adjust preview defaults based on active tab."""
    # Tab 0 (All), Tab 1 (Raw) → Preview ON
    if index in [0, 1]:
        if not self.preview_toggle_btn.isChecked():
            self.preview_toggle_btn.blockSignals(True)
            self.preview_toggle_btn.setChecked(True)
            self.preview_toggle_btn.blockSignals(False)
            self._update_preview_button_state(True)
            self.preview_enabled = True
            
    # Tab 2 (Preprocessed) → Preview OFF
    elif index == 2:
        if self.preview_toggle_btn.isChecked():
            self.preview_toggle_btn.blockSignals(True)
            self.preview_toggle_btn.setChecked(False)
            self.preview_toggle_btn.blockSignals(False)
            self._update_preview_button_state(False)
            self.preview_enabled = False
    
    self._update_preview()
```

**Preview Display Logic**:
```python
def _update_preview(self):
    """Update preview based on current state."""
    # Always show something - never blank the plot!
    
    if not self.preview_enabled:
        # Preview OFF → Show original/current dataset state
        dataset_name = self._get_selected_dataset_name()
        if dataset_name in RAMAN_DATA:
            self.original_data = RAMAN_DATA[dataset_name]
            self._show_original_data()  # ✅ Shows graph with original data
        return
    
    # Preview ON → Show processed data with real-time pipeline
    self._show_processed_preview()
```

**User Confirmation Pattern (Optional)**:
```python
def _on_preview_toggle_clicked(self):
    """Handle manual preview toggle with warnings."""
    enabled = self.preview_toggle_btn.isChecked()
    
    # Warn when disabling preview on pipelines with steps
    if not enabled and self.pipeline_steps:
        result = QMessageBox.question(
            self,
            LOCALIZE("PREPROCESS.UI.warning"),
            LOCALIZE("PREPROCESS.UI.disable_preview_warning"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if result == QMessageBox.No:
            # User cancelled - revert toggle
            self.preview_toggle_btn.blockSignals(True)
            self.preview_toggle_btn.setChecked(True)
            self.preview_toggle_btn.blockSignals(False)
            return
    
    # Update all three components
    self._update_preview_button_state(enabled)
    self.preview_enabled = enabled
    self._update_preview()
```

**Initialization Pattern**:
```python
def __init__(self):
    # Create toggle button
    self.preview_toggle_btn = QPushButton()
    self.preview_toggle_btn.setCheckable(True)
    self.preview_toggle_btn.clicked.connect(self._on_preview_toggle_clicked)
    
    # Set initial state (all three components)
    initial_enabled = True
    self.preview_toggle_btn.setChecked(initial_enabled)
    self._update_preview_button_state(initial_enabled)
    self.preview_enabled = initial_enabled
```

**Debug Checklist**:
- [ ] Does `setChecked()` match `preview_enabled` flag?
- [ ] Does button text/icon match `setChecked()` state?
- [ ] Is `_update_preview_button_state()` called after every `setChecked()`?
- [ ] Are signals blocked during programmatic state changes?
- [ ] Does the preview display logic check `preview_enabled` flag?
- [ ] Is there only ONE method definition (no duplicates)?
- [ ] Are all method names spelled correctly (no typos)?

**Benefits**:
- ✅ Prevents visual-state desynchronization bugs
- ✅ Ensures consistent user experience
- ✅ Makes debugging easier (single source of truth)
- ✅ Avoids duplicate code with centralized update method
- ✅ Supports localization through proper method structure

**Anti-Patterns to Avoid**:
1. **Calling non-existent methods** - Use IDE autocomplete to verify
2. **Duplicate method definitions** - Search codebase before creating new methods
3. **Updating only 1-2 of 3 components** - Always update checked + visual + flag
4. **Hardcoded text in methods** - Use localization system
5. **Not blocking signals** - Prevents infinite loops during programmatic changes
6. **Forgetting to update preview display** - Call `_update_preview()` after state changes

**Related Files**:
- `pages/preprocess_page.py` (lines 669-683, 734-765, 3951-3971)

---

### 0.0. Standardized Section Title Bar Pattern (October 14, 2025) 🆕⭐
**PURPOSE**: Maintain visual consistency across all pages with standardized title bars  
**GUIDELINE**: See `.AGI-BANKS/UI_TITLE_BAR_STANDARD.md` for comprehensive documentation

**Quick Pattern**:
```python
def _create_section_with_standard_title(self, title_key: str, buttons: list = None) -> QGroupBox:
    """Create section group with standardized title bar."""
    section_group = QGroupBox()
    section_group.setObjectName("modernSectionGroup")
    
    layout = QVBoxLayout(section_group)
    layout.setContentsMargins(12, 4, 12, 12)
    layout.setSpacing(10)
    
    # === STANDARDIZED TITLE BAR ===
    title_widget = QWidget()
    title_layout = QHBoxLayout(title_widget)
    title_layout.setContentsMargins(0, 0, 0, 0)
    title_layout.setSpacing(8)
    
    # Title label (always first)
    title_label = QLabel(LOCALIZE(title_key))
    title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
    title_layout.addWidget(title_label)
    
    # Stretch to push buttons right
    title_layout.addStretch()
    
    # Add optional buttons (hint, action, save, toggle)
    if buttons:
        for button in buttons:
            title_layout.addWidget(button)
    
    layout.addWidget(title_widget)
    
    # Add section content below
    # ... your content here ...
    
    return section_group

# Button Patterns
def _create_hint_button(self, tooltip: str) -> QPushButton:
    """Create standardized hint button (20x20px blue)."""
    hint_btn = QPushButton("?")
    hint_btn.setObjectName("hintButton")
    hint_btn.setFixedSize(20, 20)
    hint_btn.setToolTip(tooltip)
    hint_btn.setCursor(Qt.PointingHandCursor)
    hint_btn.setStyleSheet("""
        QPushButton#hintButton {
            background-color: #e7f3ff;
            color: #0078d4;
            border: 1px solid #90caf9;
            border-radius: 10px;
            font-weight: bold;
            font-size: 11px;
        }
        QPushButton#hintButton:hover {
            background-color: #0078d4;
            color: white;
        }
    """)
    return hint_btn

def _create_action_icon_button(self, icon_name: str, color: str, tooltip: str) -> QPushButton:
    """Create standardized action icon button (24x24px with 14x14px icon)."""
    btn = QPushButton()
    btn.setObjectName("titleBarButton")
    icon = load_svg_icon(get_icon_path(icon_name), color, QSize(14, 14))
    btn.setIcon(icon)
    btn.setIconSize(QSize(14, 14))
    btn.setFixedSize(24, 24)
    btn.setToolTip(tooltip)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setStyleSheet("""
        QPushButton#titleBarButton {
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 3px;
            padding: 2px;
        }
        QPushButton#titleBarButton:hover {
            background-color: #e7f3ff;
            border-color: #0078d4;
        }
    """)
    return btn

def _create_save_icon_button(self, tooltip: str) -> QPushButton:
    """Create standardized save icon button (24x24px green theme)."""
    btn = QPushButton()
    btn.setObjectName("titleBarButtonGreen")
    icon = load_svg_icon(get_icon_path("save"), "#28a745", QSize(14, 14))
    btn.setIcon(icon)
    btn.setIconSize(QSize(14, 14))
    btn.setFixedSize(24, 24)
    btn.setToolTip(tooltip)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setStyleSheet("""
        QPushButton#titleBarButtonGreen {
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 3px;
            padding: 2px;
        }
        QPushButton#titleBarButtonGreen:hover {
            background-color: #d4edda;
            border-color: #28a745;
        }
    """)
    return btn

def _create_delete_icon_button(self, tooltip: str) -> QPushButton:
    """Create standardized delete icon button (24x24px red theme)."""
    btn = QPushButton()
    btn.setObjectName("titleBarButtonRed")
    icon = load_svg_icon(get_icon_path("delete_all"), "#dc3545", QSize(14, 14))
    btn.setIcon(icon)
    btn.setIconSize(QSize(14, 14))
    btn.setFixedSize(24, 24)
    btn.setToolTip(tooltip)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setStyleSheet("""
        QPushButton#titleBarButtonRed {
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 3px;
            padding: 2px;
        }
        QPushButton#titleBarButtonRed:hover {
            background-color: #f8d7da;
            border-color: #dc3545;
        }
        QPushButton#titleBarButtonRed:pressed {
            background-color: #f5c6cb;
        }
    """)
    return btn
```

**Design Specs**:
- **Title**: 13px, font-weight 600, color #2c3e50
- **Margins**: (12, 4, 12, 12)
- **Spacing**: 8px between title elements
- **Button Order**: [Title] [Stretch] [Hint] [Toggle] [Action] [Save] [Delete]
- **Icon Sizes**: Hint 20x20px (no icon), Action/Save/Delete 24x24px (14x14px icon)
- **Colors**: Blue #0078d4 (primary), Green #28a745 (save), Red #dc3545 (delete), Gray #6c757d (disabled)

**Pages Compliant**:
- ✅ Preprocessing Page (5 sections)
- ✅ Data Package Page (4 sections)
- 📋 ML/Analysis/Visualization/Real-Time pages (need update)

**Reference**: `.AGI-BANKS/UI_TITLE_BAR_STANDARD.md` - Full specification and examples

---

### 0.1. Batch Import with Progress Dialog Pattern (October 14, 2025) 🆕
**PURPOSE**: Import multiple datasets without freezing UI  
**SOLUTION**: Modal progress dialog with real-time updates

```python
class BatchImportProgressDialog(QDialog):
    """Progress dialog for batch import operations."""
    def __init__(self, parent=None, total=0):
        super().__init__(parent)
        self.setWindowTitle(LOCALIZE("DATA_PACKAGE_PAGE.batch_import_progress_title"))
        self.setModal(True)
        self.setFixedSize(400, 150)
        
        layout = QVBoxLayout(self)
        
        # Message label
        message_label = QLabel(LOCALIZE("DATA_PACKAGE_PAGE.batch_import_progress_message"))
        layout.addWidget(message_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Current folder label
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel(LOCALIZE("DATA_PACKAGE_PAGE.processing_folder")))
        self.current_folder_label = QLabel("")
        self.current_folder_label.setStyleSheet("font-weight: bold;")
        folder_layout.addWidget(self.current_folder_label)
        folder_layout.addStretch()
        layout.addLayout(folder_layout)
        
        # Status label (✓ X | ✗ Y format)
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel(LOCALIZE("DATA_PACKAGE_PAGE.import_status")))
        self.status_label = QLabel("✓ 0 | ✗ 0")
        self.status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)
    
    def update_progress(self, current: int, folder_name: str, success_count: int, fail_count: int):
        """Update progress with real-time info."""
        self.progress_bar.setValue(current)
        self.current_folder_label.setText(folder_name)
        self.status_label.setText(f"✓ {success_count} | ✗ {fail_count}")
        QApplication.processEvents()  # CRITICAL: Keep UI responsive

# Usage in batch import
def _handle_batch_import(self, parent_path: str, subfolders: list):
    """Handle batch import with progress dialog."""
    # Create and show progress dialog
    progress_dialog = BatchImportProgressDialog(self, total=len(subfolders))
    progress_dialog.show()
    
    self.pending_datasets = {}
    success_count = 0
    failed_count = 0
    
    for i, folder_name in enumerate(subfolders):
        # Update progress dialog
        progress_dialog.update_progress(i + 1, folder_name, success_count, failed_count)
        
        folder_path = os.path.join(parent_path, folder_name)
        
        try:
            # Load data (your existing loading logic)
            df = load_data_from_path(folder_path)
            if isinstance(df, str):
                failed_count += 1
                continue
            
            # Store dataset
            self.pending_datasets[folder_name] = {
                'df': df,
                'metadata': {},
                'path': folder_path
            }
            success_count += 1
            
        except Exception as e:
            failed_count += 1
            continue
    
    # Close progress dialog
    progress_dialog.close()
    
    # Show results
    if success_count > 0:
        self.showNotification.emit(
            LOCALIZE("batch_import_success", count=success_count),
            "success"
        )
```

**Key Points**:
- **Modal Dialog**: Prevents user interaction during import
- **Real-time Updates**: Shows current folder being processed
- **Success/Fail Counter**: Visual feedback on import results
- **QApplication.processEvents()**: CRITICAL for keeping UI responsive
- **Fixed Size**: Prevents dialog resize during updates

---

### 0.2. Delete All with Confirmation Pattern (October 14, 2025) 🆕
**PURPOSE**: Safely delete all items with user confirmation  
**USE CASE**: Bulk operations that are destructive and irreversible

```python
def _handle_delete_all_datasets(self):
    """Delete all datasets from project with confirmation dialog."""
    # Check if there's anything to delete
    if not RAMAN_DATA:
        self.showNotification.emit(
            LOCALIZE("DATA_PACKAGE_PAGE.no_datasets_to_delete"),
            "warning"
        )
        return
    
    # Count items for confirmation message
    count = len(RAMAN_DATA)
    
    # Show confirmation dialog
    reply = QMessageBox.question(
        self,
        LOCALIZE("DATA_PACKAGE_PAGE.delete_all_confirm_title"),
        LOCALIZE("DATA_PACKAGE_PAGE.delete_all_confirm_text", count=count),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No  # Default to No for safety
    )
    
    if reply == QMessageBox.StandardButton.Yes:
        # Delete all items
        success_count = 0
        for name in list(RAMAN_DATA.keys()):  # Use list() to avoid dict size change during iteration
            if PROJECT_MANAGER.remove_dataframe_from_project(name):
                success_count += 1
        
        # Show success notification
        self.showNotification.emit(
            LOCALIZE("DATA_PACKAGE_PAGE.delete_all_success", count=success_count),
            "success"
        )
        
        # Refresh UI
        self.load_project_data()

# Button in title bar (red theme)
delete_all_btn = QPushButton()
delete_all_btn.setObjectName("titleBarButtonRed")
delete_all_icon = load_svg_icon(get_icon_path("delete_all"), "#dc3545", QSize(14, 14))
delete_all_btn.setIcon(delete_all_icon)
delete_all_btn.setFixedSize(24, 24)
delete_all_btn.setToolTip(LOCALIZE("DATA_PACKAGE_PAGE.delete_all_tooltip"))
delete_all_btn.clicked.connect(self._handle_delete_all_datasets)
```

**Key Points**:
- **Safety Check**: Verify items exist before showing dialog
- **Count Display**: Show number of items in confirmation message
- **Default to No**: Safe default prevents accidental deletion
- **List Conversion**: `list(dict.keys())` prevents iteration errors
- **Success Feedback**: Notify user of operation result
- **UI Refresh**: Update interface after deletion

**Localization Required**:
- `no_datasets_to_delete`: "No datasets to delete"
- `delete_all_confirm_title`: "Confirm Delete All"
- `delete_all_confirm_text`: "Are you sure you want to delete all {count} datasets?"
- `delete_all_success`: "Successfully deleted {count} dataset(s)"
- `delete_all_tooltip`: "Delete all datasets from project"

---

### 0.3. Optimized Preview Layout Pattern (October 14, 2025) 🆕
**PURPOSE**: Maximize graph visibility by removing unnecessary wrappers  
**ANTI-PATTERN**: Wrapping plot widget in QFrame reduces available space

```python
# ❌ ANTI-PATTERN: QFrame wrapper shrinks graph
def _create_preview_group_bad(self) -> QGroupBox:
    preview_layout = QVBoxLayout()
    
    # Wrapper frame reduces space
    preview_frame = QFrame()
    preview_frame.setFrameShape(QFrame.StyledPanel)
    preview_layout.addWidget(preview_frame)  # No stretch factor
    
    plot_layout = QVBoxLayout(preview_frame)
    plot_layout.addWidget(self.plot_widget)  # Graph constrained by wrapper
    
    return preview_group

# ✅ CORRECT PATTERN: Direct widget placement with stretch factor
def _create_preview_group_good(self) -> QGroupBox:
    preview_group = QGroupBox()
    preview_layout = QVBoxLayout(preview_group)
    preview_layout.setContentsMargins(12, 4, 12, 12)
    preview_layout.setSpacing(10)
    
    # Add standardized title bar
    # ... title bar code ...
    
    # Add plot widget directly with stretch factor
    preview_layout.addWidget(self.plot_widget, 1)  # Stretch factor 1
    self.plot_widget.setMinimumHeight(300)  # Ensure readable minimum
    
    return preview_group
```

**Benefits**:
- Graph uses all available vertical space
- No unnecessary frame styling
- Better readability for data analysis
- Maintains minimum readable size

---

### 0.4. Smart Drag-Drop Detection Pattern (October 14, 2025) 🆕
**PURPOSE**: Auto-detect whether dropped file is metadata or data  
**USE CASE**: Enable drag-drop on entire groupbox, not just specific labels

```python
def _on_drag_enter(self, event):
    """Handle drag enter for groupbox."""
    if event.mimeData().hasUrls():
        event.acceptProposedAction()

def _on_drop(self, event):
    """Handle drop with smart detection."""
    urls = event.mimeData().urls()
    if not urls:
        return
    
    path = urls[0].toLocalFile()
    
    # Smart detection: metadata.json vs data
    if os.path.basename(path).lower() == "metadata.json":
        # It's metadata
        self.meta_path_input.setText(path)
        self.showNotification.emit(
            LOCALIZE("metadata_loaded"),
            "success"
        )
    else:
        # It's data (file or folder)
        self.data_path_input.setText(path)
        self.showNotification.emit(
            LOCALIZE("data_path_loaded"),
            "success"
        )
    
    event.acceptProposedAction()

# Enable drag-drop on groupbox
importer_group.setAcceptDrops(True)
importer_group.dragEnterEvent = self._on_drag_enter
importer_group.dropEvent = self._on_drop
```

**Benefits**:
- No need for separate drag-drop areas
- User can drop anywhere in section
- Automatic detection of file type
- Better UX with visual feedback

---

### 0.5. Batch Import Detection Pattern (October 14, 2025) 🆕
Pattern for detecting and importing multiple datasets from parent folder containing subfolders.

```python
# Detection Pattern
def _check_if_batch_import(self, parent_path: str, subfolders: list) -> bool:
    """Check if selected path is batch import scenario."""
    # Sample first few subfolders (min 3, avoid checking all for performance)
    check_count = min(3, len(subfolders))
    folders_with_data = 0
    
    for folder in subfolders[:check_count]:
        folder_path = os.path.join(parent_path, folder)
        # Check for supported data file extensions
        has_data = any(
            any(f.endswith(ext) for ext in ['.txt', '.asc', '.csv', '.pkl'])
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        )
        if has_data:
            folders_with_data += 1
    
    # If majority of sampled folders contain data, treat as batch import
    return folders_with_data >= check_count * 0.5

# Batch Loading Pattern
def _handle_batch_import(self, parent_path: str, subfolders: list):
    """Handle batch import of multiple datasets."""
    self.pending_datasets = {}  # Store pending imports
    success_count = 0
    failed_count = 0
    
    for folder_name in subfolders:
        folder_path = os.path.join(parent_path, folder_name)
        
        try:
            # Load data from subfolder using existing loader
            df = load_data_from_path(folder_path)
            if isinstance(df, str):  # Error returned as string
                failed_count += 1
                continue
            
            # Check for metadata.json in subfolder
            metadata = {}
            metadata_path = os.path.join(folder_path, "metadata.json")
            if os.path.exists(metadata_path):
                meta = load_metadata_from_json(metadata_path)
                if not isinstance(meta, str):
                    metadata = meta
            
            # Store in pending datasets dictionary
            self.pending_datasets[folder_name] = {
                'df': df,
                'metadata': metadata,
                'path': folder_path
            }
            success_count += 1
            
        except Exception as e:
            failed_count += 1
            continue
    
    if success_count > 0:
        # Populate dataset selector for preview
        self.dataset_selector.clear()
        self.dataset_selector.addItems(sorted(self.pending_datasets.keys()))
        self.dataset_selector.setVisible(True)
        
        # Show first dataset preview
        first_dataset = self.dataset_selector.currentText()
        dataset_info = self.pending_datasets[first_dataset]
        self.update_preview_display(dataset_info['df'], dataset_info.get('metadata', {}))
        
        # Notify user
        self.showNotification.emit(
            LOCALIZE("batch_import_info", count=success_count),
            "success"
        )

# Batch Add to Project Pattern
def _handle_batch_add_to_project(self):
    """Add all pending datasets to project."""
    success_count = 0
    
    for dataset_name, dataset_info in self.pending_datasets.items():
        df = dataset_info.get('df')
        metadata = dataset_info.get('metadata', {})
        
        # Handle name conflicts with auto-suffix
        if dataset_name in RAMAN_DATA:
            base_name = dataset_name
            counter = 1
            while f"{base_name}_{counter}" in RAMAN_DATA:
                counter += 1
            dataset_name = f"{base_name}_{counter}"
        
        # Add to project
        if PROJECT_MANAGER.add_dataframe_to_project(dataset_name, df, metadata):
            success_count += 1
    
    # Notify and refresh
    if success_count > 0:
        self.showNotification.emit(
            LOCALIZE("batch_import_success", count=success_count),
            "success"
        )
        self.load_project_data()
        self.clear_importer_fields()
```

**Key Principles:**
1. **Sampling for Performance:** Check only first 3 folders instead of all
2. **Graceful Degradation:** Count successes/failures, continue on errors
3. **Auto-Conflict Resolution:** Add suffix for duplicate names
4. **Metadata Preservation:** Auto-load metadata.json from each folder
5. **User Feedback:** Show clear notifications with counts

**Use Cases:**
- Medical imaging datasets (100+ patient folders)
- Multi-sample experiments (many measurement runs)
- Time-series data (daily/weekly folders)
- Multi-instrument data (folder per device)

### 0.2. Auto-Preview Pattern (October 14, 2025) 🆕
Pattern for automatic data preview with user control toggle.

```python
# Toggle State Management
class DataPage(QWidget):
    def __init__(self):
        self.auto_preview_enabled = True  # Feature flag
        
    def _toggle_auto_preview(self):
        """Toggle auto-preview on/off."""
        self.auto_preview_enabled = not self.auto_preview_enabled
        self._update_auto_preview_icon()
    
    def _update_auto_preview_icon(self):
        """Update icon based on state."""
        if self.auto_preview_enabled:
            icon = load_svg_icon(get_icon_path("eye_open"), "#0078d4", QSize(14, 14))
            tooltip = LOCALIZE("auto_preview_enabled")
        else:
            icon = load_svg_icon(get_icon_path("eye_close"), "#6c757d", QSize(14, 14))
            tooltip = LOCALIZE("auto_preview_disabled")
        self.auto_preview_btn.setIcon(icon)
        self.auto_preview_btn.setToolTip(tooltip)

# Auto-Trigger Pattern
def _set_data_path(self, path: str):
    """Set path and auto-preview if enabled."""
    self.data_path_edit.setText(path)
    
    # Trigger auto-preview conditionally
    if self.auto_preview_enabled and path:
        self.handle_preview_data()

# Manual Fallback
self.preview_button.clicked.connect(self.handle_preview_data)  # Always available
```

**Key Principles:**
1. **User Control:** Toggle button for enabling/disabling
2. **Visual Feedback:** Eye icon changes (open/closed)
3. **Manual Fallback:** Preview button always works regardless of toggle
4. **State Persistence:** Flag stored in instance attribute
5. **Clear Indication:** Tooltip shows current state

### 0.3. Metadata Auto-Fill Pattern (October 14, 2025) 🆕
Pattern for automatic metadata detection and loading from JSON files.

```python
# Auto-Detection Pattern
def _handle_single_import(self, data_path: str):
    """Import with auto-metadata detection."""
    df = load_data_from_path(data_path)
    
    # Check for metadata.json in same folder as data
    if os.path.isdir(data_path):
        auto_meta_path = os.path.join(data_path, "metadata.json")
        if os.path.exists(auto_meta_path):
            meta = load_metadata_from_json(auto_meta_path)
            if not isinstance(meta, str):  # Success (not error string)
                self.preview_metadata = meta
                self.meta_path_edit.setText(auto_meta_path)
                self.showNotification.emit(
                    LOCALIZE("metadata_autofilled"),
                    "info"
                )
                # Auto-fill editor fields
                self._set_metadata_in_editor(meta)
            else:
                self.preview_metadata = {}
        else:
            self.showNotification.emit(
                LOCALIZE("no_metadata_found"),
                "info"
            )
            self.preview_metadata = {}

# Manual Override Pattern
# User can still manually select different metadata file
browse_meta_btn.clicked.connect(self.browse_for_metadata)

# Editor Update Pattern
if self.meta_editor_group.isChecked():
    # User edited metadata takes precedence
    self.preview_metadata = self._get_metadata_from_editor()
```

**Key Principles:**
1. **Non-Intrusive:** Only auto-fills if metadata.json exists
2. **Clear Notification:** Tell user when metadata was auto-filled
3. **Manual Override:** User can browse for different metadata file
4. **Editor Priority:** Manual edits take precedence over auto-filled
5. **Graceful Fallback:** Empty dict if no metadata found

### 0. Dynamic Section Title Pattern (October 10, 2025) 🆕
Dynamic updating of section titles to reflect current context/state.

```python
# Store reference to title label during UI setup
self.params_title_label = QLabel(LOCALIZE("PREPROCESS.parameters_title"))
self.params_title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
params_title_layout.addWidget(self.params_title_label)

# Update title dynamically when state changes
def _show_parameter_widget(self, step: PipelineStep):
    # ... create parameter widget ...
    
    # Update title with context
    category_display = step.category.replace('_', ' ').title()
    self.params_title_label.setText(
        f"{LOCALIZE('PREPROCESS.parameters_title')} - {category_display}: {step.method}"
    )

# Reset title when clearing
def _clear_parameter_widget(self):
    # ... clear widget ...
    self.params_title_label.setText(LOCALIZE("PREPROCESS.parameters_title"))
```

### 1. Pipeline Import/Export Pattern (October 10, 2025) 🆕
Complete workflow for saving and loading complex configuration data.

```python
# Export Pattern
def export_pipeline(self):
    # 1. Validate data exists
    if not self.pipeline_steps:
        self.showNotification.emit("No pipeline to export", "warning")
        return
    
    # 2. Show user-friendly dialog
    dialog = QDialog(self)
    name_edit = QLineEdit()
    desc_edit = QTextEdit()
    # ... build dialog UI ...
    
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return
    
    # 3. Get project directory
    project_name = PROJECT_MANAGER.current_project_data.get("projectName", "").replace(' ', '_').lower()
    project_root = os.path.join(PROJECT_MANAGER.projects_dir, project_name)
    pipelines_dir = os.path.join(project_root, "pipelines")
    os.makedirs(pipelines_dir, exist_ok=True)
    
    # 4. Build data structure
    pipeline_data = {
        "name": pipeline_name,
        "description": pipeline_description,
        "created_date": datetime.datetime.now().isoformat(),
        "step_count": len(self.pipeline_steps),
        "steps": []
    }
    
    for step in self.pipeline_steps:
        pipeline_data["steps"].append({
            "category": step.category,
            "method": step.method,
            "params": step.params,
            "enabled": step.enabled
        })
    
    # 5. Save to file
    filename = f"{pipeline_name.replace(' ', '_').lower()}.json"
    filepath = os.path.join(pipelines_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(pipeline_data, f, indent=2, ensure_ascii=False)
    
    # 6. Provide feedback
    self.showNotification.emit(f"Pipeline '{pipeline_name}' exported successfully", "success")

# Import Pattern
def import_pipeline(self):
    # 1. Load saved pipelines
    pipelines_dir = os.path.join(project_root, "pipelines")
    saved_pipelines = []
    
    for filename in os.listdir(pipelines_dir):
        if filename.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                pipeline_data = json.load(f)
                saved_pipelines.append({
                    'name': pipeline_data.get('name'),
                    'filepath': filepath,
                    'data': pipeline_data
                })
    
    # 2. Show selection dialog with rich preview
    dialog = QDialog(self)
    pipeline_list = QListWidget()
    
    for pipeline in saved_pipelines:
        item = QListWidgetItem()
        # Create custom widget with name, step count, description
        widget = create_pipeline_preview_widget(pipeline)
        item.setData(Qt.ItemDataRole.UserRole, pipeline)
        pipeline_list.addItem(item)
        pipeline_list.setItemWidget(item, widget)
    
    # 3. Confirm replacement if needed
    if self.pipeline_steps:
        confirm = QMessageBox.question(self, "Replace Pipeline?", 
                                      f"Replace {len(self.pipeline_steps)} steps?")
        if confirm != QMessageBox.StandardButton.Yes:
            return
    
    # 4. Load selected pipeline
    selected_pipeline = pipeline_list.currentItem().data(Qt.ItemDataRole.UserRole)
    self._load_pipeline_from_data(selected_pipeline['data'])
    
    self.showNotification.emit(f"Pipeline imported successfully", "success")

# External import support
def _import_external_pipeline(self, parent_dialog):
    from PySide6.QtWidgets import QFileDialog
    
    filepath, _ = QFileDialog.getOpenFileName(
        parent_dialog, "Select Pipeline File", "", "JSON Files (*.json)"
    )
    
    if filepath:
        with open(filepath, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)
        self._load_pipeline_from_data(pipeline_data)
        parent_dialog.accept()
```

### 2. Gray Border Selection Pattern (October 10, 2025) 🆕
Subtle selection indicator that preserves item's original appearance.

```python
def _update_appearance(self):
    """Update visual appearance with selection support."""
    # Determine background based on step state
    if not self.step.enabled:
        bg_color = "#f8f9fa"  # Disabled
    elif self.step.is_existing:
        bg_color = "#e3f2fd"  # Existing
    else:
        bg_color = "white"  # New
    
    # Apply normal styling
    self.setStyleSheet(f"background-color: {bg_color}; border: 1px solid #dee2e6;")
    
    # Override with selection if selected (gray border, keep background)
    if self.is_selected:
        selected_style = f"""
            QWidget {{
                background-color: {bg_color};
                border: 2px solid #6c757d;  /* Gray selection border */
                border-radius: 6px;
            }}
        """
        self.setStyleSheet(selected_style)
```

### 3. Title Bar Action Buttons Pattern (October 7, 2025)
Compact action buttons integrated into section title bars for space-efficient UI design.

```python
# Create title widget with action buttons in title bar
title_widget = QWidget()
title_layout = QHBoxLayout(title_widget)
title_layout.setContentsMargins(0, 0, 0, 0)
title_layout.setSpacing(8)

title_label = QLabel(LOCALIZE("SECTION.title"))
title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
title_layout.addWidget(title_label)

# Add hint button if needed
hint_btn = QPushButton("?")
hint_btn.setFixedSize(20, 20)
# ... hint button styling ...
title_layout.addWidget(hint_btn)

title_layout.addStretch()  # Push action buttons to the right

# Compact action buttons (24px)
refresh_btn = QPushButton()
refresh_btn.setObjectName("titleBarButton")
refresh_btn.setIcon(load_svg_icon(get_icon_path("reload"), "#0078d4", QSize(14, 14)))
refresh_btn.setFixedSize(24, 24)
refresh_btn.setStyleSheet("""
    QPushButton#titleBarButton {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 3px;
        padding: 2px;
    }
    QPushButton#titleBarButton:hover {
        background-color: #e7f3ff;
        border-color: #0078d4;
    }
""")
title_layout.addWidget(refresh_btn)
```

### 1. Pipeline Step Selection Visual Feedback Pattern (October 7, 2025) 🆕
Implementing visual selection state for custom widgets in list views.

```python
class PipelineStepWidget(QWidget):
    def __init__(self, step, step_index, parent=None):
        super().__init__(parent)
        self.is_selected = False  # Track selection state
        self._setup_ui()
        
    def set_selected(self, selected: bool):
        """Set selection state and update appearance."""
        self.is_selected = selected
        self._update_appearance()
        
    def _update_appearance(self):
        """Update visual appearance with selection override."""
        # ... normal state styling ...
        
        # Override with selection styling if selected
        if self.is_selected:
            selected_style = """
                QWidget {
                    background-color: #d4e6f7;
                    border: 2px solid #0078d4;
                    border-radius: 6px;
                }
            """
            self.setStyleSheet(selected_style)

# In parent widget's selection handler
def on_item_selected(self, current, previous):
    # Update visual selection state for all widgets
    for i in range(self.list_widget.count()):
        item = self.list_widget.item(i)
        widget = self.list_widget.itemWidget(item)
        if widget and hasattr(widget, 'set_selected'):
            widget.set_selected(item == current)
```

### 2. Choice Parameter Type Conversion Pattern (October 7, 2025) 🆕
Proper handling of choice parameters with type preservation in dynamic UI generation.

```python
class DynamicParameterWidget(QWidget):
    def _create_parameter_widget(self, param_name: str, info: Dict, default_value: Any):
        if param_type == "choice":
            widget = QComboBox()
            choices = info.get("choices", [])
            
            # Create mapping to preserve original types
            choice_mapping = {}
            for choice in choices:
                str_choice = str(choice)
                widget.addItem(str_choice)
                choice_mapping[str_choice] = choice
            
            widget.choice_mapping = choice_mapping  # Store for extraction
            
    def get_parameters(self) -> Dict[str, Any]:
        # Extract with proper type conversion
        if param_type == "choice":
            current_text = widget.currentText()
            if hasattr(widget, 'choice_mapping'):
                params[param_name] = widget.choice_mapping[current_text]
            else:
                # Fallback type conversion
                try:
                    params[param_name] = int(current_text)
                except ValueError:
                    params[param_name] = current_text
```

### 3. Robust Signal Handling with Dynamic Index Resolution (October 7, 2025) 🆕
Error-safe signal handling for widgets that may have stale indices.

```python
def on_step_toggled(self, step_index: int, enabled: bool):
    """Handle step toggle with robust index validation."""
    # Find actual index by searching for sender widget
    actual_step_index = None
    sender_widget = self.sender()
    
    for i in range(self.list_widget.count()):
        widget = self.list_widget.itemWidget(self.list_widget.item(i))
        if widget == sender_widget:
            actual_step_index = i
            break
    
    # Use provided index as fallback
    if actual_step_index is None:
        actual_step_index = step_index
    
    # Validate bounds
    if not (0 <= actual_step_index < len(self.data_list)):
        create_logs("error", "Invalid index", f"Index {actual_step_index} out of bounds")
        return
    
    # Proceed with safe index
    self.data_list[actual_step_index].enabled = enabled
```

### 4. Hint Button in Title Pattern (October 6 Evening, 2025) 🆕
Space-efficient hint/info button integrated into section titles, combining multiple hints in one compact button.

```python
# Create custom title widget with hint button
title_widget = QWidget()
title_layout = QHBoxLayout(title_widget)
title_layout.setContentsMargins(0, 0, 0, 0)
title_layout.setSpacing(8)

# Title label
title_label = QLabel(LOCALIZE("SECTION.title"))
title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
title_layout.addWidget(title_label)

# Hint button with ? icon
hint_btn = QPushButton("?")
hint_btn.setObjectName("hintButton")
hint_btn.setFixedSize(20, 20)
hint_btn.setToolTip(
    LOCALIZE("SECTION.hint_1") + "\\n\\n" +
    LOCALIZE("SECTION.hint_2")  # Combine multiple hints
)
hint_btn.setCursor(Qt.PointingHandCursor)
hint_btn.setStyleSheet("""
    QPushButton#hintButton {
        background-color: #e7f3ff;
        color: #0078d4;
        border: 1px solid #90caf9;
        border-radius: 10px;
        font-weight: bold;
        font-size: 11px;
        padding: 0px;
    }
    QPushButton#hintButton:hover {
        background-color: #0078d4;
        color: white;
        border-color: #0078d4;
    }
""")
title_layout.addWidget(hint_btn)
title_layout.addStretch()

# Use custom title in QGroupBox
group_box = QGroupBox()
group_box.setTitle("")  # Empty title since using custom widget
layout = QVBoxLayout(group_box)
layout.addWidget(title_widget)
```

**Benefits**:
- Saves vertical space (no separate info row needed)
- Consolidates multiple hints in one button
- Professional appearance (? icon vs emoji)
- Clear visual hierarchy in title bar

---

### 1. Icon-Only Button Pattern (October 6, 2025) 🆕
Space-efficient button design using SVG icons with hover tooltips, eliminating text labels for compact UI.

```python
# Import required functions
from utils import load_svg_icon
from components.widgets.icons import get_icon_path

# Create icon-only button with hover tooltip
button = QPushButton()
button.setObjectName("iconOnlyButton")

# Load and set icon (with color customization)
icon = load_svg_icon(get_icon_path("icon_name"), "#0078d4", QSize(18, 18))
button.setIcon(icon)
button.setIconSize(QSize(18, 18))
button.setFixedSize(36, 36)

# Add tooltip for text (shows on hover)
button.setToolTip(LOCALIZE("PREPROCESS.button_tooltip"))
button.setCursor(Qt.PointingHandCursor)

# Apply inline or stylesheet-based styling
button.setStyleSheet("""
    QPushButton#iconOnlyButton {
        background-color: #ffffff;
        border: 2px solid #dee2e6;
        border-radius: 6px;
        padding: 6px;
    }
    QPushButton#iconOnlyButton:hover {
        background-color: #e7f3ff;
        border-color: #0078d4;
    }
    QPushButton#iconOnlyButton:pressed {
        background-color: #d0e7ff;
        border-color: #005a9e;
    }
""")

# Connect to action
button.clicked.connect(self.action_method)
```

**Variant: Professional SVG Icon Buttons (Replaces Emoji)**
```python
# Plus button with SVG icon (instead of ➕ emoji)
add_btn = QPushButton()
add_btn.setObjectName("addStepButton")
plus_icon = load_svg_icon(get_icon_path("plus"), "white", QSize(24, 24))
add_btn.setIcon(plus_icon)
add_btn.setIconSize(QSize(24, 24))
add_btn.setFixedSize(60, 50)

# Trash button with SVG icon (instead of 🗑️ emoji)
remove_btn = QPushButton()
remove_btn.setObjectName("compactButton")
trash_icon = load_svg_icon(get_icon_path("trash_bin"), "#dc3545", QSize(14, 14))
remove_btn.setIcon(trash_icon)
remove_btn.setIconSize(QSize(14, 14))
remove_btn.setFixedHeight(28)
```

**Available SVG Icons** (from `components/widgets/icons.py`):
- `plus` → plus.svg (add/create actions)
- `minus` → minus.svg (decrease/remove actions)
- `trash_bin` → trash-bin.svg (delete actions)
- `trash` → trash-xmark.svg (clear/reset actions)
- `reload` → reload.svg (refresh actions)
- `export` → export-button.svg (export actions)
- `eye_open` → eye-open.svg (show/visibility)
- `eye_close` → eye-close.svg (hide actions)
- `chevron_down` → chevron-down.svg (dropdown indicators)

**Variant: Colored Icon Button (e.g., Green for Export)**
```python
export_btn = QPushButton()
export_btn.setObjectName("iconOnlyButtonGreen")
export_icon = load_svg_icon(get_icon_path("export"), "#2e7d32", QSize(18, 18))
export_btn.setIcon(export_icon)
export_btn.setIconSize(QSize(18, 18))
export_btn.setFixedSize(36, 36)
export_btn.setToolTip(LOCALIZE("PREPROCESS.export_button_tooltip"))
export_btn.setCursor(Qt.PointingHandCursor)
export_btn.setStyleSheet("""
    QPushButton#iconOnlyButtonGreen {
        background-color: #4caf50;
        border: 2px solid #4caf50;
        border-radius: 6px;
        padding: 6px;
    }
    QPushButton#iconOnlyButtonGreen:hover {
        background-color: #45a049;
        border-color: #45a049;
    }
    QPushButton#iconOnlyButtonGreen:pressed {
        background-color: #3d8b40;
        border-color: #3d8b40;
    }
    QPushButton#iconOnlyButtonGreen:disabled {
        background-color: #a5d6a7;
        border-color: #a5d6a7;
    }
""")
export_btn.clicked.connect(self.export_dataset)
```

**Variant: Compact Control Buttons (Emoji Icons)**
```python
# For pipeline controls (Remove, Clear, Toggle)
remove_btn = QPushButton("🗑️")
remove_btn.setObjectName("compactButton")
remove_btn.setFixedHeight(28)
remove_btn.setStyleSheet("""
    QPushButton#compactButton {
        background-color: #f8f9fa;
        color: #495057;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 4px 10px;
        font-weight: 500;
        font-size: 14px;
    }
    QPushButton#compactButton:hover {
        background-color: #e9ecef;
        border-color: #adb5bd;
    }
    QPushButton#compactButton:pressed {
        background-color: #dee2e6;
    }
""")
remove_btn.setToolTip(LOCALIZE("PREPROCESS.remove_step"))
remove_btn.setCursor(Qt.PointingHandCursor)
remove_btn.clicked.connect(self.remove_pipeline_step)
```

**Key Features:**
- Saves 60-80% horizontal space compared to text buttons
- Text visible only on hover via tooltip
- Consistent size (36x36px for icon buttons, 28px height for compact)
- Clear visual feedback (hover and pressed states)
- Accessible through tooltips
- Pointer cursor indicates interactivity

**Icon Loading Best Practices:**
- Use centralized `get_icon_path()` for path management
- Use `load_svg_icon()` for color customization
- Common icon sizes: 16x16, 18x18, 20x20 pixels
- Button sizes: 32x32, 36x36, 40x40 pixels
- Match icon color to button theme

**Color Schemes:**
- **Primary (Blue)**: #0078d4 icon, white/light blue background
- **Success (Green)**: #2e7d32 icon, green background (#4caf50)
- **Neutral (Gray)**: #495057 icon/text, light gray background (#f8f9fa)
- **Hover states**: Slightly darker/lighter shade of base color

**Use Cases:**
- Toolbar buttons with limited space
- Action buttons in compact layouts
- Control buttons (add, remove, clear, toggle)
- Export, import, reload, refresh actions
- Navigation buttons

**Best Practices:**
- Always provide tooltip for accessibility
- Use recognizable icons (standard icons or emoji)
- Set cursor to pointer for click indication
- Provide clear hover/pressed states
- Group related icon buttons together
- Maintain consistent sizing within a group
- Test icon visibility on different backgrounds

**Available Icons** (from components/widgets/icons.py):
- `reload`: Refresh/reload icon
- `export`: Export/download icon
- `trash_bin`: Delete/remove icon
- `plus`: Add/create icon
- `minus`: Remove/subtract icon
- `eye_open`: Show/visible icon
- `eye_close`: Hide/invisible icon
- `focus_horizontal`: Focus/zoom icon
- `new_project`: New file/project icon
- `load_project`: Open file/project icon

---

### 1. Hover Tooltip System Pattern (October 2025)
Space-efficient information display using interactive icons with rich HTML tooltips.

```python
# Create info icons row with hover tooltips
info_row = QHBoxLayout()
info_row.setSpacing(12)
info_row.setContentsMargins(4, 4, 4, 4)

# Multi-selection hint icon
multi_select_icon = QLabel("ℹ️")
multi_select_icon.setStyleSheet("""
    QLabel {
        color: #0078d4;
        font-size: 14px;
        padding: 2px;
        border-radius: 3px;
    }
    QLabel:hover {
        background-color: #e7f3ff;
    }
""")
multi_select_icon.setToolTip(LOCALIZE("PREPROCESS.multi_select_hint"))
multi_select_icon.setCursor(Qt.PointingHandCursor)
info_row.addWidget(multi_select_icon)

# Multi-dataset processing hint icon
multi_dataset_icon = QLabel("💡")
multi_dataset_icon.setStyleSheet("""
    QLabel {
        color: #0078d4;
        font-size: 14px;
        padding: 2px;
        border-radius: 3px;
    }
    QLabel:hover {
        background-color: #e7f3ff;
    }
""")
multi_dataset_icon.setToolTip(LOCALIZE("PREPROCESS.multi_dataset_hint"))
multi_dataset_icon.setCursor(Qt.PointingHandCursor)
info_row.addWidget(multi_dataset_icon)

info_row.addStretch()
layout.addLayout(info_row)
```

**Locale String Format (HTML-formatted):**
```json
{
  "multi_select_hint": "<b>Multi-Selection:</b><br>• Hold <b>Ctrl</b> (or <b>Cmd</b>) + Click to select multiple datasets<br>• Hold <b>Shift</b> + Click to select a range",
  "multi_dataset_hint": "<b>Multi-Dataset Processing:</b><br>When multiple datasets are selected, they will be combined and processed together as one output dataset. The preprocessing pipeline applies to all selected data simultaneously."
}
```

**Key Features:**
- Saves vertical space compared to always-visible labels
- Rich HTML formatting in tooltips (bold text, bullet points, line breaks)
- Interactive hover states with visual feedback
- Medical theme colors (#0078d4 blue, #e7f3ff light blue hover)
- Pointer cursor indicates interactivity
- Icon-based design for quick recognition (ℹ️ = info, 💡 = insight)

**Best Practices:**
- Use HTML formatting for complex tooltip content
- Provide hover state styling for visual feedback
- Use emoji icons for better visual communication
- Keep tooltip text concise but informative
- Group related info icons together in horizontal layout
- Maintain consistent styling across all tooltip icons

---

### 2. Unified Selection Handler Pattern (October 2025)
Cross-tab selection synchronization with single handler for multiple QListWidget instances.

```python
# In __init__ or creation method:
# Create three separate list widgets for different tabs
self.dataset_list_all = QListWidget()
self.dataset_list_raw = QListWidget()
self.dataset_list_preprocessed = QListWidget()

# Configure and connect all list widgets to same handler
for list_widget in [self.dataset_list_all, self.dataset_list_raw, 
                    self.dataset_list_preprocessed]:
    list_widget.setObjectName("datasetList")
    list_widget.setSelectionMode(QListWidget.ExtendedSelection)
    # Connect to unified handler
    list_widget.itemSelectionChanged.connect(self._on_dataset_selection_changed)

# Keep reference to active list for backward compatibility
self.dataset_list = self.dataset_list_all

# Connect tab change signal
self.dataset_tabs.currentChanged.connect(self._on_dataset_tab_changed)

def _on_dataset_tab_changed(self, index: int):
    """Update active dataset list reference when tab changes."""
    if index == 0:
        self.dataset_list = self.dataset_list_all
    elif index == 1:
        self.dataset_list = self.dataset_list_raw
    elif index == 2:
        self.dataset_list = self.dataset_list_preprocessed
    
    # Trigger selection event for newly active tab
    self._on_dataset_selection_changed()

def _on_dataset_selection_changed(self):
    """Handle dataset selection changes across all tabs."""
    selected_items = self.dataset_list.selectedItems()
    
    if not selected_items:
        # Clear visualization and state
        self.plot_widget.clear_plot()
        return
    
    # Load and display selected data
    all_dfs = []
    for item in selected_items:
        dataset_name = self._clean_dataset_name(item.text())
        if dataset_name in RAMAN_DATA:
            all_dfs.append(RAMAN_DATA[dataset_name])
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, axis=1)
        self.original_data = combined_df
        # Update visualization
        self._schedule_preview_update()
```

**Key Features:**
- Single handler for all tab list widgets (DRY principle)
- Tab switching automatically triggers visualization update
- Backward compatible with existing `self.dataset_list` reference
- Consistent behavior across All/Raw/Preprocessed tabs
- Signal-based architecture for loose coupling

**Best Practices:**
- Connect all related widgets to same handler for consistency
- Update active reference on tab change for backward compatibility
- Trigger selection event when switching tabs
- Use `itemSelectionChanged` signal for immediate response
- Handle empty selection gracefully



def _on_dataset_tab_changed(self, index: int):
    """Update active dataset list reference when tab changes."""
    if index == 0:
        self.dataset_list = self.dataset_list_all
    elif index == 1:
        self.dataset_list = self.dataset_list_raw
    elif index == 2:
        self.dataset_list = self.dataset_list_preprocessed
    
    # Trigger selection event for newly active tab
    self._on_dataset_selection_changed()

def _on_dataset_selection_changed(self):
    """Handle dataset selection changes across all tabs."""
    selected_items = self.dataset_list.selectedItems()
    
    if not selected_items:
        # Clear visualization and state
        self.plot_widget.clear_plot()
        return
    
    # Load and display selected data
    all_dfs = []
    for item in selected_items:
        dataset_name = self._clean_dataset_name(item.text())
        if dataset_name in RAMAN_DATA:
            all_dfs.append(RAMAN_DATA[dataset_name])
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, axis=1)
        self.original_data = combined_df
        # Update visualization
        self._schedule_preview_update()
```

**Key Features:**
- Single handler for all tab list widgets (DRY principle)
- Tab switching automatically triggers visualization update
- Backward compatible with existing `self.dataset_list` reference
- Consistent behavior across All/Raw/Preprocessed tabs
- Signal-based architecture for loose coupling

**Best Practices:**
- Connect all related widgets to same handler for consistency
- Update active reference on tab change for backward compatibility
- Trigger selection event when switching tabs
- Use `itemSelectionChanged` signal for immediate response
- Handle empty selection gracefully

---

### 3. Modern Card-Based UI Pattern (October 2025)
Clean, professional card-based design for metrics and controls.

```python
def _create_metric_item(self, icon: str, value: str, label: str) -> QFrame:
    """Create a modern metric display item with icon, value, and label."""
    metric = QFrame()
    metric.setObjectName("metricItem")
    metric_layout = QVBoxLayout(metric)
    metric_layout.setContentsMargins(16, 12, 16, 12)
    metric_layout.setSpacing(6)
    
    # Icon row (centered)
    icon_label = QLabel(icon)
    icon_label.setStyleSheet("font-size: 20px;")
    icon_label.setAlignment(Qt.AlignCenter)
    metric_layout.addWidget(icon_label)
    
    # Value (large and prominent, centered)
    value_label = QLabel(value)
    value_label.setObjectName("metricValue")
    value_label.setAlignment(Qt.AlignCenter)
    metric_layout.addWidget(value_label)
    
    # Label (small description, centered)
    label_widget = QLabel(label)
    label_widget.setObjectName("metricLabel")
    label_widget.setAlignment(Qt.AlignCenter)
    label_widget.setWordWrap(True)
    metric_layout.addWidget(label_widget)
    
    return metric

# In confirmation dialog header:
metrics_grid = QGridLayout()
metrics_grid.setSpacing(16)

input_metric = self._create_metric_item("📊", str(len(datasets)), 
                                        LOCALIZE("datasets_label"))
metrics_grid.addWidget(input_metric, 0, 0)

steps_metric = self._create_metric_item("⚙️", str(len(steps)), 
                                        LOCALIZE("steps_label"))
metrics_grid.addWidget(steps_metric, 0, 1)

output_metric = self._create_metric_item("💾", output_name, 
                                         LOCALIZE("output_label"))
metrics_grid.addWidget(output_metric, 0, 2)

# Stylesheet for metrics:
"""
#metricItem {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #ffffff, stop:1 #fafcfe);
    border: 1px solid #e1e8ed;
    border-radius: 8px;
    min-width: 140px;
    max-width: 200px;
}

#metricItem:hover {
    border-color: #90caf9;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #f8fbff, stop:1 #e3f2fd);
}

#metricValue {
    font-size: 24px;
    font-weight: 700;
    color: #0078d4;
    letter-spacing: -0.5px;
}

#metricLabel {
    font-size: 11px;
    font-weight: 500;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
"""
```

**Key Features:**
- Vertical layout: Icon → Value → Label
- Centered alignment for clean appearance
- Hover states with border and gradient changes
- Grid layout for responsive metric display
- Professional typography with size/weight hierarchy
- Medical theme colors and subtle gradients

**Best Practices:**
- Use QGridLayout for responsive metric arrangement
- Implement hover states for interactivity
- Maintain consistent spacing (padding 16/12, spacing 6)
- Use uppercase labels with letter spacing for professional look
- Limit card width for better readability (140-200px)
- Center-align all content within metric cards

---

### 4. Export with Metadata Pattern (October 2025)
Comprehensive data export with automatic metadata serialization and batch processing support.

```python
def export_dataset(self):
    """Export selected dataset(s) with optional metadata."""
    selected_items = self.dataset_list.selectedItems()
    dataset_names = [self._clean_dataset_name(item.text()) for item in selected_items]
    
    # Create export dialog with dynamic UI
    if len(dataset_names) > 1:
        # Multiple export mode: show count, hide filename field
        info_label = QLabel(LOCALIZE("PREPROCESS.export_multiple_info", count=len(dataset_names)))
        layout.addWidget(info_label)
    else:
        # Single export mode: show filename field
        filename_edit = QLineEdit()
        filename_edit.setText(dataset_names[0])
    
    # Load and pre-fill last location (session persistence)
    last_export_path = getattr(self, '_last_export_location', None)
    if last_export_path and os.path.exists(last_export_path):
        location_edit.setText(last_export_path)
    
    # Validation before export
    if not export_path:
        QMessageBox.warning(self, 
                          LOCALIZE("PREPROCESS.export_warning_title"),
                          LOCALIZE("PREPROCESS.export_no_location_warning"))
        return
    
    # Store location for next time
    self._last_export_location = export_path
    
    # Export with metadata
    for dataset_name in dataset_names:
        self._export_single_dataset(dataset_name, export_path, 
                                    dataset_name, export_format, 
                                    export_metadata=True)

def _export_metadata_json(self, metadata: Dict, export_path: str, 
                          filename: str, data_shape: tuple):
    """Export preprocessing metadata to JSON."""
    export_meta = {
        "export_info": {
            "export_date": datetime.now().isoformat(),
            "dataset_name": filename,
            "data_shape": {"rows": data_shape[0], "columns": data_shape[1]}
        },
        "preprocessing": {
            "is_preprocessed": metadata.get("is_preprocessed", False),
            "pipeline": metadata.get("preprocessing_pipeline", []),
            "source_datasets": metadata.get("source_datasets", [])
        },
        "spectral_info": {
            "num_spectra": metadata.get("num_spectra", data_shape[1]),
            "spectral_axis_start": metadata.get("spectral_axis", [None])[0],
            "spectral_axis_end": metadata.get("spectral_axis", [None])[-1]
        }
    }
    
    metadata_path = os.path.join(export_path, f"{filename}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(export_meta, f, indent=2, ensure_ascii=False)
```

**Key Features:**
- Dynamic UI based on single vs. multiple dataset selection
- Location persistence across export operations (session-level)
- Explicit validation with user-friendly warning dialogs
- Structured metadata JSON with export info, preprocessing, and spectral data
- Batch processing with individual error handling
- Comprehensive localization support

**Best Practices:**
- Separate helper methods for single dataset export and metadata export
- Validate all inputs before file operations
- Store last location using `getattr()` with default to handle first use
- Use ISO format for timestamps in metadata
- Include both human-readable and machine-parseable data in JSON
- Emit appropriate notifications: success, warning (partial), error

**Benefits:**
- Improved data traceability with metadata export
- Better UX with location persistence and validation
- Efficient batch processing for multiple datasets
- Robust error handling with graceful degradation
- Full internationalization support

---

### 2. Icon Management Pattern (Updated September 2025)
Centralized icon management with comprehensive path registry and utility functions.

```python
# components/widgets/icons.py
ICON_PATHS = {
    "eye_open": "eye-open.svg",
    "eye_close": "eye-close.svg",
    "minus": "minus.svg",
    "plus": "plus.svg",
    # ... more icons
}

def load_icon(icon_name: str, size: Optional[Union[QSize, str]] = None, color: Optional[str] = None) -> QIcon:
    """Load an icon with optional size and color customization."""
    icon_path = get_icon_path(icon_name)
    
    if size is None:
        size = DEFAULT_SIZES["button"]
    elif isinstance(size, str):
        size = DEFAULT_SIZES[size]
    
    if color is not None:
        return load_svg_icon(icon_path, color, size)
    else:
        return QIcon(icon_path)
```

**Best Practices:**
- Use centralized icon registry to avoid hardcoded paths
- Provide size presets ("button", "toolbar", "large") for consistency
- Support color customization for SVG icons
- Include backward compatibility aliases for existing usage

### 2. State-Aware Toggle Buttons Pattern
Interactive buttons that change appearance and behavior based on internal state.

```python
def _update_enable_button(self):
    """Update button icon and tooltip based on current state."""
    if self.step.enabled:
        # State is enabled, show action to disable
        icon = load_icon("eye_close", "button")
        tooltip = LOCALIZE("PREPROCESS.disable_step_tooltip")
    else:
        # State is disabled, show action to enable
        icon = load_icon("eye_open", "button")
        tooltip = LOCALIZE("PREPROCESS.enable_step_tooltip")
    
    self.enable_toggle_btn.setIcon(icon)
    self.enable_toggle_btn.setToolTip(tooltip)

def _toggle_enabled(self):
    """Toggle state and update UI accordingly."""
    self.step.enabled = not self.step.enabled
    self._update_enable_button()
    self._update_appearance()
    # Notify parent of state change
    self.toggled.emit(self.step_index, self.step.enabled)
```

**Key Principles:**
- Icon represents the action, not the current state
- Tooltip describes what will happen when clicked
- Emit signals to notify parent components of state changes
- Update multiple UI elements consistently when state changes

### 3. Dataset Type Tracking Pattern
Track user navigation between different data types to provide contextual behavior.

```python
def __init__(self):
    # Track dataset selection for pipeline transfer logic
    self._last_selected_was_preprocessed = False

def handle_dataset_selection(self, dataset_name: str):
    """Handle dataset selection with context-aware logic."""
    metadata = PROJECT_MANAGER.get_dataframe_metadata(dataset_name)
    is_preprocessed = metadata and metadata.get('is_preprocessed', False)
    
    if is_preprocessed:
        # Check if moving between preprocessed datasets
        if self._last_selected_was_preprocessed and len(self.pipeline_steps) > 0:
            self._show_pipeline_transfer_dialog(dataset_name)
        else:
            self._load_preprocessing_pipeline(metadata.get('preprocessing_pipeline', []))
        self._last_selected_was_preprocessed = True
    else:
        # Raw dataset selected - always clear
        self._clear_preprocessing_history()
        self._last_selected_was_preprocessed = False
```

**Benefits:**
- Provides contextual user interactions
- Prevents unwanted dialogs in wrong scenarios
- Maintains state across user navigation
- Enables intelligent pipeline management

### 4. Centralized Dataset Name Cleaning Pattern
Consistent emoji and prefix removal across the application.

```python
def _clean_dataset_name(self, item_text: str) -> str:
    """Clean dataset name by removing UI prefixes like emojis."""
    return item_text.replace("📊 ", "").replace("🔬 ", "").strip()

# Usage throughout the application
dataset_name = self._clean_dataset_name(first_item.text())  # Instead of hardcoded replace()
```

**Advantages:**
- Single source of truth for dataset name cleaning
- Easy to add new emoji types or prefixes
- Prevents bugs from missed emoji types
- Consistent behavior across all components

### 5. Widget Component Pattern
Used throughout the application for parameter input widgets and UI components.

```python
class BaseParameterWidget(QWidget):
    """Base class for all parameter input widgets."""
    
    value_changed = Signal()  # Real-time value change notification
    
    def __init__(self, param_name: str, config: dict, parent=None):
        super().__init__(parent)
        self.param_name = param_name
        self.config = config
        self._setup_ui()
        self._connect_signals()
    
    def get_value(self):
        """Get current widget value - must be implemented by subclasses."""
        raise NotImplementedError
    
    def set_value(self, value):
        """Set widget value - must be implemented by subclasses."""
        raise NotImplementedError
    
    def validate(self) -> bool:
        """Validate current value - returns True if valid."""
        return True
    
    def _setup_ui(self):
        """Setup widget UI - implemented by subclasses."""
        pass
    
    def _connect_signals(self):
        """Connect internal signals - implemented by subclasses."""
        pass
```

**Usage Pattern:**
- Inherit from base class for consistency
- Implement required methods (get_value, set_value)
- Emit value_changed signal for real-time updates
- Use validate() method for input validation
- Apply consistent styling using configuration

### 2. Method Registry Pattern
Used for dynamic method discovery and instantiation in preprocessing pipeline.

```python
class MethodRegistry:
    """Registry for preprocessing methods with automatic discovery."""
    
    def __init__(self):
        self._methods = {}  # category -> {method_name: method_info}
    
    def register_method(self, category: str, name: str, method_class, **kwargs):
        """Register a preprocessing method."""
        if category not in self._methods:
            self._methods[category] = {}
        
        self._methods[category][name] = {
            'class': method_class,
            'name': name,
            'category': category,
            **kwargs
        }
    
    def get_method_info(self, category: str, method: str) -> dict:
        """Get method information."""
        return self._methods.get(category, {}).get(method)
    
    def create_method_instance(self, category: str, method: str, params: dict):
        """Create method instance with parameters."""
        method_info = self.get_method_info(category, method)
        if not method_info:
            raise ValueError(f"Method {category}.{method} not found")
        
        method_class = method_info['class']
        return method_class(**params)
```

**Usage Benefits:**
- Dynamic method discovery
- Consistent method interfaces
- Easy addition of new methods
- Automatic parameter handling

### 2.1. Custom Wrapper Pattern for Broken Library Methods (pybaselines.api Pattern)
**WHEN TO USE**: When ramanspy wrapper has bugs that cannot be fixed by parameter adjustments.

**Pattern**: Bypass ramanspy wrapper, call pybaselines.api directly while maintaining Container compatibility.

**Example**: FABC wrapper (ramanspy bug on line 33)

```python
# functions/preprocess/fabc_fixed.py
from pybaselines import api
import numpy as np
import ramanspy as rp

class FABCFixed:
    """
    Custom FABC implementation bypassing ramanspy wrapper bug.
    
    Problem: ramanspy.FABC passes x_data incorrectly in apply_along_axis call.
    Solution: Call pybaselines.api.Baseline directly with correct initialization.
    """
    
    def __init__(self, lam=1e5, scale=0.5, num_std=3, diff_order=2, min_length=2, 
                 weights=None, weights_as_mask=False):
        """Initialize with pybaselines.fabc parameters."""
        self.lam = lam
        self.scale = scale
        self.num_std = num_std
        self.diff_order = diff_order
        self.min_length = min_length
        self.weights = weights
        self.weights_as_mask = weights_as_mask
    
    def _get_baseline_fitter(self, x_data: np.ndarray):
        """Create baseline fitter with x_data (CORRECT approach)."""
        return api.Baseline(x_data=x_data)
    
    def _process_spectrum(self, spectrum, x_data):
        """Process single spectrum (no x_data in method call)."""
        fitter = self._get_baseline_fitter(x_data)
        baseline, params = fitter.fabc(
            data=spectrum,
            lam=self.lam,
            scale=self.scale,
            num_std=self.num_std,
            diff_order=self.diff_order,
            min_length=self.min_length,
            weights=self.weights,
            weights_as_mask=self.weights_as_mask
        )
        return spectrum - baseline
    
    def __call__(self, data, spectral_axis=None):
        """
        Container-aware wrapper maintaining ramanspy interface.
        
        Handles both:
        - ramanspy.SpectralContainer (returns Container)
        - numpy arrays (returns array)
        """
        is_container = hasattr(data, 'spectral_data')
        
        if is_container:
            spectral_data = data.spectral_data
            x_data = data.spectral_axis if hasattr(data, 'spectral_axis') else spectral_axis
        else:
            spectral_data = data
            x_data = spectral_axis
        
        if x_data is None:
            raise ValueError("spectral_axis required")
        
        # Process
        if spectral_data.ndim == 1:
            corrected = self._process_spectrum(spectral_data, x_data)
        else:
            corrected = np.array([
                self._process_spectrum(spectrum, x_data)
                for spectrum in spectral_data
            ])
        
        # Return same type as input
        if is_container:
            return rp.SpectralContainer(corrected, x_data)
        return corrected
    
    def apply(self, container):
        """ramanspy compatibility: apply() method."""
        return self(container)
```

**Registry Integration**:
```python
# functions/preprocess/registry.py
from .fabc_fixed import FABCFixed

"FABC": {
    "class": FABCFixed,  # Custom wrapper, not ramanspy.FABC
    "description": "Fixed FABC implementation (bypassing ramanspy bug)",
    "category": "baseline_correction",
    "param_info": {
        "lam": {"type": "float", "min": 1e3, "max": 1e7, "default": 1e5},
        "scale": {"type": "float", "min": 0.1, "max": 2.0, "default": 0.5},
        "num_std": {"type": "float", "min": 1.0, "max": 10.0, "default": 3.0},
        "diff_order": {"type": "int", "min": 1, "max": 3, "default": 2},
        "min_length": {"type": "int", "min": 1, "max": 100, "default": 2}
    },
    "default_params": {
        "lam": 1e5, "scale": 0.5, "num_std": 3.0,
        "diff_order": 2, "min_length": 2
    }
}
```

**Testing Requirements**:
```python
# test_script/test_fabc_fix.py
def test_fabc_custom_implementation():
    """Test custom FABC wrapper."""
    # 1. Registry instantiation
    registry = PreprocessingStepRegistry()
    method = registry.create_method_instance('baseline_correction', 'FABC', {})
    
    # 2. Synthetic data (fluorescence + peaks)
    wavenumbers = np.linspace(400, 1800, 1000)
    spectrum = generate_tissue_spectrum()  # With baseline
    
    # 3. Apply FABC
    corrected = method(spectrum, wavenumbers)
    
    # 4. Validate baseline reduction (should be >95%)
    original_mean = np.mean(spectrum)
    corrected_mean = np.mean(corrected)
    baseline_reduction = (original_mean - corrected_mean) / original_mean
    assert baseline_reduction > 0.95, f"Expected >95% reduction, got {baseline_reduction:.1%}"
    
    # 5. Container compatibility
    container = rp.SpectralContainer(spectrum.reshape(1, -1), wavenumbers)
    corrected_container = method.apply(container)
    assert isinstance(corrected_container, rp.SpectralContainer)
```

**Key Principles**:
1. **x_data in initialization**: `api.Baseline(x_data=x)`, NOT in method call
2. **Container awareness**: Detect input type, return same type
3. **Interface compatibility**: Implement both `__call__()` and `apply()`
4. **Functional testing**: Test with realistic synthetic data, verify >95% baseline reduction
5. **Type preservation**: numpy in → numpy out, Container in → Container out

### 2.2. Robust Parameter Type Validation Pattern (October 15, 2025) 🔒⭐
**CRITICAL**: ALL preprocessing methods MUST implement robust parameter type conversion.

**Problem**: UI frameworks send unexpected types:
- Qt sliders emit floats (2.0) when integers expected (2)
- Text fields return strings ("2") instead of numbers
- Decimal adjustments create problematic values (1.2, 2.7)

**Solution**: Two-layer type validation system

#### Layer 1: Registry Type Conversion (Universal)

**Location**: `functions/preprocess/registry.py` → `create_method_instance()`

```python
def create_method_instance(self, category: str, method: str, params: Dict[str, Any] = None):
    """Create instance with robust type conversion."""
    
    param_type = param_info[actual_key].get("type")
    
    # INTEGER CONVERSION (Critical for pybaselines)
    if param_type == "int":
        if value is None:
            converted_params[actual_key] = None
        else:
            # CRITICAL: Two-stage conversion handles strings and decimals
            converted_params[actual_key] = int(float(value))
            # Step 1: float("2") handles strings → 2.0
            # Step 2: int(2.0) converts to integer → 2
            # Works for: "2", 2.0, 1.2 (truncates to 1)
    
    # FLOAT CONVERSION
    elif param_type in ("float", "scientific"):
        if value is None:
            converted_params[actual_key] = None
        else:
            converted_params[actual_key] = float(value)
    
    # CHOICE CONVERSION (Type-aware based on choices)
    elif param_type == "choice":
        choices = param_info[actual_key].get("choices", [])
        if choices and isinstance(choices[0], int):
            # Integer choices: convert float→int
            converted_params[actual_key] = int(float(value))
        elif choices and isinstance(choices[0], float):
            converted_params[actual_key] = float(value)
        else:
            converted_params[actual_key] = value
    
    # BOOLEAN CONVERSION
    elif param_type == "bool":
        if isinstance(value, bool):
            converted_params[actual_key] = value
        elif isinstance(value, str):
            # Handle string booleans from text input
            converted_params[actual_key] = value.lower() in ('true', '1', 'yes')
        else:
            converted_params[actual_key] = bool(value)
    
    # ARRAY CONVERSION (None or ndarray pass-through)
    elif param_type == "array":
        converted_params[actual_key] = value
```

**Why Two-Stage Int Conversion?**
```python
# ❌ Bad: Fails on string decimals
int("1.2")  # ValueError: invalid literal for int()

# ✅ Good: Handles all cases
int(float("1.2"))  # 1 (works!)
int(float("2"))    # 2
int(float(2.0))    # 2
int(float(1.2))    # 1 (truncation)
```

#### Layer 2: Defensive Type Conversion (Class-Level)

**When to Use**: Methods with strict type requirements (especially pybaselines wrappers)

```python
class FABCFixed:
    """Custom FABC with defensive type conversion."""
    
    def __init__(self, lam=1e6, scale=None, num_std=3.0, diff_order=2, min_length=2,
                 weights=None, weights_as_mask=False):
        """Initialize with explicit type conversion."""
        
        # CRITICAL: Explicit type enforcement
        self.lam = float(lam)  # Ensure float
        self.scale = None if scale is None else float(scale)  # None-safe
        self.num_std = float(num_std)  # Ensure float
        self.diff_order = int(diff_order)  # MUST be int, not float!
        self.min_length = int(min_length)  # MUST be int, not float!
        self.weights = weights  # Can be None or ndarray
        self.weights_as_mask = bool(weights_as_mask)  # Ensure bool
```

**Benefits of Two Layers**:
1. **Registry Layer**: Universal protection for all methods
2. **Class Layer**: Defensive programming for critical methods
3. **Redundancy**: Protection even if registry is bypassed
4. **Documentation**: Makes type requirements explicit in code

#### Testing Requirements

```python
# Test 1: Default parameters (baseline)
method = registry.create_method_instance("category", "Method", {})
assert isinstance(method.int_param, int)

# Test 2: Float from UI slider
params = {"int_param": 2.0, "float_param": 3.5}
method = registry.create_method_instance("category", "Method", params)
assert isinstance(method.int_param, int)  # 2.0 → 2

# Test 3: Decimal float (worst case)
params = {"int_param": 1.2}
method = registry.create_method_instance("category", "Method", params)
assert method.int_param == 1  # Truncation is intentional

# Test 4: String from text field
params = {"int_param": "2", "float_param": "3.5"}
method = registry.create_method_instance("category", "Method", params)
assert isinstance(method.int_param, int)  # "2" → 2

# Test 5: None for optional parameter
params = {"optional_param": None}
method = registry.create_method_instance("category", "Method", params)
assert method.optional_param is None  # None preserved
```

#### Common Type Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| UI slider float | `got '2.0', expected int` | `int(float(value))` |
| Decimal adjustment | `got '1.2', expected int` | `int(float(value))` → 1 |
| String input | `got '2', expected int` | `int(float(value))` |
| None optional | NoneType error | `None if val is None else type(val)` |
| Boolean string | `got 'true', expected bool` | `val.lower() in ('true', '1', 'yes')` |

#### Edge Cases Covered

```python
# Case 1: UI Slider Floats
# Input: diff_order=2.0
# Registry: int(float(2.0)) → 2 ✅
# Class: int(2) → 2 ✅

# Case 2: Decimal Float Adjustments
# Input: diff_order=1.2
# Registry: int(float(1.2)) → 1 ✅ (truncation)
# Class: int(1) → 1 ✅

# Case 3: String Input
# Input: diff_order="2"
# Registry: int(float("2")) → 2 ✅
# Class: int(2) → 2 ✅

# Case 4: None Optional
# Input: scale=None
# Registry: None (preserved) ✅
# Class: None if None else float() ✅

# Case 5: Boolean String
# Input: enabled="true"
# Registry: "true".lower() in (...) → True ✅
# Class: bool(True) → True ✅
```

**Best Practices**:
- Always use `int(float(value))` for integer parameters
- Always handle None for optional parameters
- Always implement type conversion at both layers (registry + class)
- Always test with all input types (int, float, string, None)
- Never assume parameter types from UI match method signatures

---

### 3. Pipeline Step Pattern
Used for preprocessing pipeline management with enable/disable functionality.

```python
@dataclass
class PipelineStep:
    """Represents a step in the preprocessing pipeline."""
    
    category: str
    method: str
    params: dict
    enabled: bool = True
    order: int = 0
    
    def execute(self, data):
        """Execute this step on the provided data."""
        if not self.enabled:
            return data
        
        method_instance = REGISTRY.create_method_instance(
            self.category, self.method, self.params
        )
        return method_instance.process(data)
    
    def validate_params(self) -> bool:
        """Validate step parameters."""
        method_info = REGISTRY.get_method_info(self.category, self.method)
        if not method_info:
            return False
        
        # Validate parameters against method requirements
        param_info = method_info.get('param_info', {})
        for param_name, requirements in param_info.items():
            if param_name not in self.params:
                if requirements.get('required', False):
                    return False
        return True
```

### 4. Observer Pattern for Real-time Updates
Used for parameter widgets and preview generation.

```python
class PreviewManager:
    """Manages real-time preview updates."""
    
    def __init__(self):
        self._observers = []
        self._data = None
        self._pipeline = []
    
    def add_observer(self, callback):
        """Add observer for preview updates."""
        self._observers.append(callback)
    
    def remove_observer(self, callback):
        """Remove observer."""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def notify_observers(self, event_type: str, data=None):
        """Notify all observers of changes."""
        for callback in self._observers:
            try:
                callback(event_type, data)
            except Exception as e:
                print(f"Observer error: {e}")
    
    def update_pipeline(self, pipeline_steps):
        """Update pipeline and trigger preview update."""
        self._pipeline = pipeline_steps
        self._generate_preview()
    
    def _generate_preview(self):
        """Generate preview data and notify observers."""
        if self._data is None:
            return
        
        preview_data = self._apply_pipeline(self._data, self._pipeline)
        self.notify_observers('preview_updated', preview_data)
```

## UI Development Patterns

### 1. Dynamic Widget Generation
Pattern for automatically generating parameter widgets based on method signatures.

```python
def create_parameter_widget(param_name: str, param_info: dict, parent=None):
    """Factory function for creating parameter widgets."""
    
    param_type = param_info.get('type', 'float')
    widget_config = {
        'param_name': param_name,
        'label': param_info.get('label', param_name.title()),
        'tooltip': param_info.get('description', ''),
        'units': param_info.get('units', ''),
    }
    
    if param_type == 'int':
        widget = IntParameterWidget(**widget_config, parent=parent)
        if 'min' in param_info:
            widget.set_minimum(param_info['min'])
        if 'max' in param_info:
            widget.set_maximum(param_info['max'])
    
    elif param_type == 'float':
        widget = FloatParameterWidget(**widget_config, parent=parent)
        widget.set_precision(param_info.get('precision', 2))
        if 'min' in param_info:
            widget.set_minimum(param_info['min'])
        if 'max' in param_info:
            widget.set_maximum(param_info['max'])
    
    elif param_type == 'range':
        widget = RangeParameterWidget(**widget_config, parent=parent)
        widget.set_range(param_info.get('min', 0), param_info.get('max', 100))
    
    elif param_type == 'choice':
        widget = ChoiceParameterWidget(**widget_config, parent=parent)
        widget.set_choices(param_info.get('choices', []))
    
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")
    
    # Set default value if provided
    if 'default' in param_info:
        widget.set_value(param_info['default'])
    
    return widget
```

### 2. Color-Coded Status Indicators
Pattern for providing visual feedback on widget states.

```python
class StatusMixin:
    """Mixin for adding status indication to widgets."""
    
    STATUS_COLORS = {
        'valid': '#2ecc71',    # Green
        'invalid': '#e74c3c',  # Red
        'warning': '#f39c12',  # Orange
        'neutral': '#95a5a6'   # Gray
    }
    
    def set_status(self, status: str, message: str = ''):
        """Set widget status with visual feedback."""
        color = self.STATUS_COLORS.get(status, self.STATUS_COLORS['neutral'])
        
        # Update border color
        self.setStyleSheet(f"""
            QWidget {{
                border: 2px solid {color};
                border-radius: 4px;
            }}
        """)
        
        # Update tooltip with status message
        if message:
            self.setToolTip(f"Status: {status.title()}\n{message}")
        
        # Emit status change signal
        if hasattr(self, 'status_changed'):
            self.status_changed.emit(status, message)
```

### 3. Matplotlib Integration Pattern
Pattern for integrating matplotlib with PySide6 for scientific plotting.

```python
class MatplotlibWidget(QWidget):
    """Widget for embedding matplotlib plots in PySide6."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Configure matplotlib for Qt
        self._configure_matplotlib()
    
    def _configure_matplotlib(self):
        """Configure matplotlib for optimal Qt integration."""
        # Use Qt-specific backend
        matplotlib.use('Qt5Agg')
        
        # Configure for scientific plotting
        plt.style.use('default')
        self.figure.patch.set_facecolor('white')
        
        # Configure for high DPI displays
        self.canvas.setStyleSheet("background-color: white;")
    
    def plot_spectra(self, wavenumbers, intensities, auto_focus=False, **kwargs):
        """Plot spectral data with optional auto-focus."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Plot data
        ax.plot(wavenumbers, intensities, **kwargs)
        
        # Apply auto-focus if requested
        if auto_focus:
            focus_range = self.detect_signal_range(wavenumbers, intensities)
            if focus_range:
                ax.set_xlim(focus_range)
        
        # Configure axes
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Intensity')
        ax.grid(True, alpha=0.3)
        
        # Update display
        self.figure.tight_layout()
        self.canvas.draw()
```

### 4. Dynamic UI Sizing Pattern
Pattern for creating UI elements that adapt to content length, especially for internationalization.

```python
def adjust_button_width_to_text(button: QPushButton, min_width: int = 80):
    """Adjust button width dynamically based on text content."""
    from PySide6.QtGui import QFontMetrics
    
    # Get current text and font
    text = button.text()
    font = button.font()
    
    # Calculate text width using font metrics
    font_metrics = QFontMetrics(font)
    text_width = font_metrics.horizontalAdvance(text)
    
    # Account for additional UI elements
    icon_width = 16 if button.icon() else 0
    spacing = 8 if text.strip() and icon_width > 0 else 0
    padding = 16  # CSS padding (8px left + 8px right)
    border = 4    # CSS border (2px left + 2px right)
    
    # Calculate total width
    total_width = text_width + icon_width + spacing + padding + border
    dynamic_width = max(min_width, total_width)
    
    button.setFixedWidth(dynamic_width)

# Integration example
class LocalizedToggleButton(QPushButton):
    """Button that auto-adjusts width for localized text."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedHeight(32)  # Fixed height, dynamic width
        
    def setText(self, text: str):
        """Override setText to trigger width adjustment."""
        super().setText(text)
        self.adjust_width_to_text()
        
    def adjust_width_to_text(self):
        """Adjust width based on current text content."""
        adjust_button_width_to_text(self, min_width=80)
```

**Benefits:**
- Responsive UI that adapts to different languages
- Prevents text truncation in longer translations
- Maintains visual consistency across locales
- Automatic adjustment without manual sizing

**Use Cases:**
- Toggle buttons with ON/OFF states
- Buttons with localized text
- Dynamic labels that change content
- UI elements with varying text lengths

## Error Handling Patterns

### 1. Graceful Degradation Pattern
Used throughout the application to handle errors without crashing.

```python
def safe_execute(func, fallback_value=None, log_errors=True):
    """Execute function with graceful error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_errors:
                create_logs("safe_execute", func.__name__, 
                           f"Error in {func.__name__}: {e}", status='error')
            return fallback_value
    return wrapper

# Usage example
@safe_execute
def process_spectral_data(data, method_params):
    """Process spectral data with error handling."""
    # Processing logic here
    return processed_data
```

### 2. Validation Pattern
Used for parameter validation with user feedback.

```python
class ValidationResult:
    """Result of validation operation."""
    
    def __init__(self, is_valid: bool, message: str = '', value=None):
        self.is_valid = is_valid
        self.message = message
        self.value = value

def validate_parameter(value, param_info: dict) -> ValidationResult:
    """Validate parameter against requirements."""
    
    # Type validation
    expected_type = param_info.get('type', 'float')
    if expected_type == 'int':
        try:
            value = int(value)
        except (ValueError, TypeError):
            return ValidationResult(False, "Must be an integer")
    
    elif expected_type == 'float':
        try:
            value = float(value)
        except (ValueError, TypeError):
            return ValidationResult(False, "Must be a number")
    
    # Range validation
    if 'min' in param_info and value < param_info['min']:
        return ValidationResult(False, f"Must be >= {param_info['min']}")
    
    if 'max' in param_info and value > param_info['max']:
        return ValidationResult(False, f"Must be <= {param_info['max']}")
    
    return ValidationResult(True, "Valid", value)
```

## Performance Optimization Patterns

### 1. Caching Pattern
Used for expensive operations like data processing.

```python
from functools import lru_cache
import hashlib

class DataCache:
    """Cache for processed data with automatic invalidation."""
    
    def __init__(self, max_size=100):
        self._cache = {}
        self._max_size = max_size
        self._access_order = []
    
    def get_cache_key(self, data, pipeline_steps):
        """Generate cache key from data and pipeline."""
        # Create hash from data shape and pipeline configuration
        data_hash = hashlib.md5(str(data.shape).encode()).hexdigest()
        pipeline_hash = hashlib.md5(str(pipeline_steps).encode()).hexdigest()
        return f"{data_hash}_{pipeline_hash}"
    
    def get(self, key):
        """Get cached result."""
        if key in self._cache:
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def set(self, key, value):
        """Store result in cache."""
        # Remove oldest if cache is full
        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        
        self._cache[key] = value
        if key not in self._access_order:
            self._access_order.append(key)
```

### 2. Lazy Loading Pattern
Used for components that are expensive to initialize.

```python
class LazyLoader:
    """Lazy loader for expensive components."""
    
    def __init__(self, loader_func, *args, **kwargs):
        self._loader_func = loader_func
        self._args = args
        self._kwargs = kwargs
        self._instance = None
        self._loaded = False
    
    def __call__(self):
        """Get or create instance."""
        if not self._loaded:
            self._instance = self._loader_func(*self._args, **self._kwargs)
            self._loaded = True
        return self._instance
    
    def is_loaded(self):
        """Check if instance has been loaded."""
        return self._loaded

# Usage example
def create_expensive_component():
    # Expensive initialization here
    return ExpensiveComponent()

lazy_component = LazyLoader(create_expensive_component)

# Component is only created when first accessed
component = lazy_component()
```

## Configuration Management Patterns

### 1. Configuration Validation Pattern
Used for ensuring configuration integrity.

```python
import json
from pathlib import Path

class ConfigurationManager:
    """Manages application configuration with validation."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._config = {}
        self._defaults = self._get_defaults()
        self.load_config()
    
    def _get_defaults(self):
        """Define default configuration values."""
        return {
            'ui': {
                'theme': 'light',
                'auto_save': True,
                'preview_update_delay': 500  # ms
            },
            'processing': {
                'cache_size': 100,
                'parallel_processing': True,
                'auto_focus': {
                    'enabled': True,
                    'threshold': 0.1
                }
            },
            'data': {
                'default_format': 'csv',
                'backup_enabled': True
            }
        }
    
    def load_config(self):
        """Load configuration from file with fallback to defaults."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                self._config = self._merge_configs(self._defaults, file_config)
            else:
                self._config = self._defaults.copy()
                self.save_config()  # Create default config file
        except Exception as e:
            print(f"Error loading config: {e}")
            self._config = self._defaults.copy()
    
    def _merge_configs(self, defaults, user_config):
        """Recursively merge user config with defaults."""
        result = defaults.copy()
        for key, value in user_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self._config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set target value
        config[keys[-1]] = value
        self.save_config()
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")
```

These patterns provide a solid foundation for maintaining code quality, consistency, and extensibility throughout the Raman spectroscopy application. Each pattern addresses specific architectural needs while maintaining the overall design philosophy of the project.