# UI Section Title Bar Standardization Guideline

**Version:** 1.0  
**Date:** 2025-10-14  
**Status:** Active Standard

## Purpose

This document defines the standardized approach for creating section title bars across all pages in the Real-time Raman Spectral Classifier application. Consistency in title bar design improves user experience, maintains visual hierarchy, and enables uniform placement of controls.

## Standard Title Bar Pattern

### Visual Design
- **Title Text:** Font weight 600, 13px, color #2c3e50
- **Background:** Transparent (inherits from parent)
- **Layout:** Horizontal with left-aligned title and right-aligned controls
- **Spacing:** 8px between elements, 0px margins on container
- **Height:** Auto (typically ~24-28px)

### Code Pattern

```python
def _create_section_group_with_standard_title(self) -> QGroupBox:
    """Create section group with standardized title bar."""
    section_group = QGroupBox()
    section_group.setObjectName("modernSectionGroup")  # Or appropriate name
    
    layout = QVBoxLayout(section_group)
    layout.setContentsMargins(12, 4, 12, 12)  # Standard margins
    layout.setSpacing(10)  # Standard spacing
    
    # === STANDARDIZED TITLE BAR ===
    title_widget = QWidget()
    title_layout = QHBoxLayout(title_widget)
    title_layout.setContentsMargins(0, 0, 0, 0)
    title_layout.setSpacing(8)
    
    # Title label
    title_label = QLabel(LOCALIZE("SECTION.title_key"))
    title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
    title_layout.addWidget(title_label)
    
    # Optional: Store reference for dynamic title updates
    self.section_title_label = title_label
    
    # Stretch to push controls to the right
    title_layout.addStretch()
    
    # Add control buttons here (see Control Button Patterns below)
    
    layout.addWidget(title_widget)
    
    # Add section content below title
    # ... rest of section layout ...
    
    return section_group
```

## Control Button Patterns

### 1. Hint Button (Blue "?" Icon)
**Use Case:** Provide contextual help/tooltip

```python
hint_btn = QPushButton("?")
hint_btn.setObjectName("hintButton")
hint_btn.setFixedSize(20, 20)
hint_btn.setToolTip(LOCALIZE("SECTION.hint_text"))
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
```

### 2. Action Icon Button (Blue Theme)
**Use Case:** Primary actions (export, import, reload, etc.)

```python
action_btn = QPushButton()
action_btn.setObjectName("titleBarButton")
action_icon = load_svg_icon(get_icon_path("icon_name"), "#0078d4", QSize(14, 14))
action_btn.setIcon(action_icon)
action_btn.setIconSize(QSize(14, 14))
action_btn.setFixedSize(24, 24)
action_btn.setToolTip(LOCALIZE("SECTION.action_tooltip"))
action_btn.setCursor(Qt.PointingHandCursor)
action_btn.setStyleSheet("""
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
    QPushButton#titleBarButton:pressed {
        background-color: #d0e7ff;
    }
""")
action_btn.clicked.connect(self.action_handler)
title_layout.addWidget(action_btn)
```

### 3. Action Icon Button (Green Theme)
**Use Case:** Save/create actions

```python
save_btn = QPushButton()
save_btn.setObjectName("titleBarButtonGreen")
save_icon = load_svg_icon(get_icon_path("save"), "#28a745", QSize(14, 14))
save_btn.setIcon(save_icon)
save_btn.setIconSize(QSize(14, 14))
save_btn.setFixedSize(24, 24)
save_btn.setToolTip(LOCALIZE("SECTION.save_tooltip"))
save_btn.setCursor(Qt.PointingHandCursor)
save_btn.setStyleSheet("""
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
    QPushButton#titleBarButtonGreen:pressed {
        background-color: #c3e6cb;
    }
""")
save_btn.clicked.connect(self.save_handler)
title_layout.addWidget(save_btn)
```

### 4. Toggle Button (State-Dependent Icon)
**Use Case:** Enable/disable features (e.g., auto-preview, real-time mode)

```python
toggle_btn = QPushButton()
toggle_btn.setObjectName("titleBarButton")
toggle_btn.setFixedSize(24, 24)
toggle_btn.setToolTip(LOCALIZE("SECTION.toggle_tooltip"))
toggle_btn.setCursor(Qt.PointingHandCursor)
self._update_toggle_icon()  # Set initial icon based on state
toggle_btn.clicked.connect(self._toggle_feature)
toggle_btn.setStyleSheet("""
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
    QPushButton#titleBarButton:pressed {
        background-color: #d0e7ff;
    }
""")
title_layout.addWidget(toggle_btn)

def _update_toggle_icon(self):
    """Update toggle button icon based on state."""
    if self.feature_enabled:
        icon = load_svg_icon(get_icon_path("eye_open"), "#0078d4", QSize(14, 14))
        tooltip = LOCALIZE("SECTION.feature_enabled")
    else:
        icon = load_svg_icon(get_icon_path("eye_close"), "#6c757d", QSize(14, 14))
        tooltip = LOCALIZE("SECTION.feature_disabled")
    toggle_btn.setIcon(icon)
    toggle_btn.setIconSize(QSize(14, 14))
    toggle_btn.setToolTip(tooltip)
```

## Button Ordering Convention

**Left to Right:**
1. Title Label (always first)
2. `addStretch()` (push buttons to right)
3. Hint Button (if present) - informational
4. State/Toggle Buttons (if present) - feature controls
5. Action Buttons (if present) - operations
6. Save/Create Buttons (if present) - data persistence

**Example:**
```
[Title] ---------------- [?] [👁️] [↻] [📤] [💾]
```

## Icon Sizes and Colors

### Icon Sizes
- **Hint Button:** 20x20px outer, no icon (text "?")
- **Action Buttons:** 24x24px outer, 14x14px icon

### Icon Colors by Theme
| Theme | Color Code | Use Case |
|-------|------------|----------|
| Blue | #0078d4 | Primary actions, enabled states |
| Gray | #6c757d | Disabled states, neutral actions |
| Green | #28a745 | Save, create, success actions |
| Red | #dc3545 | Delete, remove, error actions |

## Implementation Checklist

When adding a new section or updating an existing one:

- [ ] Use standardized title bar pattern
- [ ] Apply standard title label styling (`font-weight: 600; font-size: 13px; color: #2c3e50;`)
- [ ] Set layout margins to `(12, 4, 12, 12)`
- [ ] Set layout spacing to `10`
- [ ] Add `addStretch()` after title label
- [ ] Use appropriate button pattern for controls
- [ ] Set button sizes to 24x24px (or 20x20px for hint)
- [ ] Set icon sizes to 14x14px
- [ ] Apply cursor pointer to buttons
- [ ] Add tooltips to all buttons
- [ ] Connect button signals to handlers
- [ ] Use localized strings for all text
- [ ] Test hover and pressed states

## Pages Currently Using This Standard

### ✅ Fully Compliant
1. **Preprocessing Page** (`pages/preprocess_page.py`)
   - Input Datasets section
   - Pipeline Building section
   - Output Configuration section
   - Parameters section
   - Visualization section

2. **Data Package Page** (`pages/data_package_page.py`)
   - Import New Dataset section
   - Project Datasets section
   - Data Preview section
   - Metadata section

### 📋 Requires Update
1. **Machine Learning Page** - Needs standardization
2. **Analysis Page** - Needs standardization
3. **Visualization Page** - Needs standardization
4. **Real-Time Page** - Needs standardization

## Dynamic Title Updates

For sections where the title needs to reflect current state:

```python
# Store reference during creation
self.section_title_label = title_label

# Update dynamically
def _update_section_title(self, context):
    """Update title with context information."""
    base_title = LOCALIZE("SECTION.title_key")
    contextual_info = f" - {context}"
    self.section_title_label.setText(f"{base_title}{contextual_info}")

# Reset to base title
def _reset_section_title(self):
    """Reset title to base text."""
    self.section_title_label.setText(LOCALIZE("SECTION.title_key"))
```

## Accessibility Considerations

- **Tooltips:** All buttons must have descriptive tooltips
- **Cursor:** Use pointer cursor for all interactive elements
- **Color Contrast:** Ensure 4.5:1 contrast ratio for text
- **Keyboard Navigation:** Buttons should be keyboard accessible
- **Screen Readers:** Use descriptive button labels

## Examples from Codebase

### Preprocessing Page - Pipeline Building Section
```python
# Title bar with hint + import + export buttons
title_label = QLabel(LOCALIZE("PREPROCESS.pipeline_building_title"))
title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
title_layout.addWidget(title_label)

# Hint button
pipeline_hint_btn = QPushButton("?")
# ... (hint button styling)
title_layout.addWidget(pipeline_hint_btn)

title_layout.addStretch()

# Import button (green)
import_btn = QPushButton()
import_btn.setObjectName("titleBarButtonGreen")
# ... (import button setup)
title_layout.addWidget(import_btn)

# Export button (blue)
export_btn = QPushButton()
export_btn.setObjectName("titleBarButton")
# ... (export button setup)
title_layout.addWidget(export_btn)
```

### Data Package Page - Preview Section
```python
# Title bar with auto-preview toggle
title_label = QLabel(LOCALIZE("DATA_PACKAGE_PAGE.preview_title"))
title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
title_layout.addWidget(title_label)

title_layout.addStretch()

# Auto-preview toggle (eye icon)
self.auto_preview_btn = QPushButton()
self.auto_preview_btn.setObjectName("titleBarButton")
# ... (toggle button setup with dynamic icon)
title_layout.addWidget(self.auto_preview_btn)
```

### Data Package Page - Metadata Section
```python
# Title bar with save button
title_label = QLabel(LOCALIZE("DATA_PACKAGE_PAGE.metadata_editor_title"))
title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
title_layout.addWidget(title_label)

title_layout.addStretch()

# Save button (green)
self.save_meta_button = QPushButton()
self.save_meta_button.setObjectName("titleBarButtonGreen")
save_icon = load_svg_icon(get_icon_path("save"), "#28a745", QSize(14, 14))
# ... (save button setup)
title_layout.addWidget(self.save_meta_button)
```

## Benefits of Standardization

1. **Visual Consistency:** All sections look cohesive and professional
2. **User Experience:** Users know where to find controls
3. **Development Speed:** Copy-paste patterns reduce development time
4. **Maintainability:** Changes can be applied uniformly
5. **Accessibility:** Consistent patterns improve accessibility
6. **Scalability:** Easy to add new sections following the same pattern

## Migration Guide

To update an existing section to use the standard:

1. **Backup Current Code:** Save a copy before modifying
2. **Identify Title Area:** Locate current title/header implementation
3. **Replace with Standard Pattern:** Use code template from this guide
4. **Migrate Controls:** Move existing buttons to title bar
5. **Test Functionality:** Verify all buttons still work
6. **Update Styling:** Apply standard colors and sizes
7. **Test Responsiveness:** Check layout at different window sizes
8. **Document Changes:** Update page documentation

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-14 | Initial standard based on Preprocessing and Data Package pages |

## Related Documentation

- `.AGI-BANKS/IMPLEMENTATION_PATTERNS.md` - Implementation patterns
- `.docs/pages/data_package_page.md` - Data Package Page documentation
- `.docs/pages/preprocess_page.md` - Preprocessing Page documentation
- `BASE_MEMORY.md` - Height optimization constraints

---

**Approved by:** AI Agent (GitHub Copilot)  
**Next Review:** 2025-11-14
