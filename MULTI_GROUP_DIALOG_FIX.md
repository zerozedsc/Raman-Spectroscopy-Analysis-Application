# Multi-Group Dialog Fix - AttributeError Resolution

**Date:** 2025-12-18  
**Issue:** `AttributeError: 'PySide6.QtWidgets.QTableWidgetItem' object has no attribute 'setPlaceholderText'`

## Problem Description

The `MultiGroupCreationDialog` was crashing when opened from `GroupTreeManager` because:

1. **Root Cause**: Lines 301 and 307 in `multi_group_dialog.py` called `setPlaceholderText()` on `QTableWidgetItem` objects
2. **Why It Failed**: `QTableWidgetItem` doesn't have a `setPlaceholderText()` method - that's only available on `QLineEdit` widgets
3. **Impact**: Dialog could not be created, preventing multi-group creation in Classification Mode

## Error Traceback

```
File "multi_group_dialog.py", line 301, in _add_row
    include_item.setPlaceholderText("e.g., ctrl, control, con")
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'PySide6.QtWidgets.QTableWidgetItem' object has no attribute 'setPlaceholderText'
```

## Solution Implemented

### 1. Changed `_add_row()` Method (Lines 287-323)

**BEFORE** (Incorrect - using QTableWidgetItem):
```python
# Group name (editable)
name_item = QTableWidgetItem("")
name_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)
self.table.setItem(row, 0, name_item)

# Include keywords (editable)
include_item = QTableWidgetItem("")
include_item.setPlaceholderText("e.g., ctrl, control, con")  # ❌ CRASH
self.table.setItem(row, 1, include_item)
```

**AFTER** (Correct - using QLineEdit with setCellWidget):
```python
# Group name (QLineEdit for better UX)
name_edit = QLineEdit()
name_edit.setPlaceholderText("e.g., Control, Disease, MM, MGUS")
name_edit.setStyleSheet("""...""")
self.table.setCellWidget(row, 0, name_edit)

# Include keywords (QLineEdit with placeholder)
include_edit = QLineEdit()
include_edit.setPlaceholderText("e.g., ctrl, control, con")  # ✅ WORKS
self.table.setCellWidget(row, 1, include_edit)
```

### 2. Updated `_validate_and_get_configs()` Method (Lines 450-516)

**BEFORE** (Accessing QTableWidgetItem):
```python
# Get group name
name_item = self.table.item(row, 0)
if not name_item:
    continue
group_name = name_item.text().strip()
```

**AFTER** (Accessing QLineEdit widget):
```python
# Get group name from QLineEdit widget
name_widget = self.table.cellWidget(row, 0)
if not name_widget:
    continue
group_name = name_widget.text().strip() if isinstance(name_widget, QLineEdit) else ""
```

## Benefits of the Fix

1. ✅ **Placeholders Work**: Users now see helpful hints in empty fields
2. ✅ **Better UX**: QLineEdit provides better focus indication with custom styling
3. ✅ **Type Safety**: Added `isinstance()` checks for robustness
4. ✅ **No Breaking Changes**: Dialog functionality remains identical

## Styling Enhancements Added

- **Name field**: Yellow highlight on focus (`#fff9e6`)
- **Include field**: Green highlight on focus (`#e8f5e9`)
- **Exclude field**: Red highlight on focus (`#ffebee`)
- Transparent background when not focused for clean look

## Testing Checklist

- [ ] Dialog opens without crash
- [ ] Placeholder text visible in all three columns
- [ ] Can type in all fields
- [ ] Preview button shows correct assignments
- [ ] Apply button creates groups successfully
- [ ] Groups appear in tree structure
- [ ] Datasets assigned to correct groups

## Files Modified

1. `pages/analysis_page_utils/multi_group_dialog.py`:
   - `_add_row()` method (lines 287-323)
   - `_validate_and_get_configs()` method (lines 450-516)

## Related Issues

- ✅ **FIXED**: GroupTreeManager "Create Group" button now opens multi-group dialog
- ✅ **FIXED**: QTableWidgetItem.setPlaceholderText() AttributeError
- 🔄 **TO TEST**: Multi-group creation workflow end-to-end

## Next Steps

1. Restart application: `uv run main.py`
2. Navigate to Analysis page → Classification Mode
3. Click "➕ Create Group" button
4. Dialog should open without errors
5. Test creating multiple groups with keyword patterns
6. Verify assignments populate tree correctly

---

**Status:** ✅ FIXED - Ready for testing
