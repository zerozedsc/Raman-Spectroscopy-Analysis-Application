# Deferred Task: Multi-Group Creation Enhancement

**Task ID**: Task-7-Multi-Group-Creation  
**Date Created**: 2025-01-21  
**Priority**: P2 (Feature Enhancement)  
**Status**: ⏸️ Deferred  
**Estimated Effort**: 4-6 hours  
**Complexity**: High

---

## Original Request

> **Improve group creation - allow multiple groups at once with include/exclude keywords and smart auto-assign**

---

## Background

Currently, the group assignment workflow in the Analysis Page requires creating groups one at a time. Users must manually:
1. Enter a group name
2. Select datasets one by one
3. Click "Add Group"
4. Repeat for each additional group

For large datasets with many groups (e.g., 10+ sample types), this becomes tedious and error-prone.

---

## Proposed Enhancement

### High-Level Goals
1. **Batch Creation**: Create multiple groups in a single dialog
2. **Keyword Filtering**: Include/exclude patterns for automatic dataset matching
3. **Smart Auto-Assign**: Intelligently assign datasets to groups based on name patterns
4. **Validation**: Prevent conflicts and overlaps with clear feedback

### User Workflow (Proposed)

#### Current Workflow
```
1. Enter group name: "Control"
2. Select datasets: dataset1, dataset2
3. Click "Add Group"
4. Enter group name: "Treatment"
5. Select datasets: dataset3, dataset4
6. Click "Add Group"
... (repeat for each group)
```

#### New Workflow
```
1. Click "Create Multiple Groups" button
2. Dialog opens with table:
   ┌─────────────┬──────────────┬──────────────┬─────────────┐
   │ Group Name  │ Include      │ Exclude      │ Auto-Assign │
   ├─────────────┼──────────────┼──────────────┼─────────────┤
   │ Control     │ ctrl, con    │              │ [✓]         │
   │ Treatment   │ treat, tx    │              │ [✓]         │
   │ Disease     │ disease, dis │ ctrl         │ [✓]         │
   └─────────────┴──────────────┴──────────────┴─────────────┘
3. Click "Preview Assignments" → see which datasets match each group
4. Click "Apply" → all groups created at once
```

---

## Technical Requirements

### 1. UI Components

#### Multi-Group Dialog
```python
class MultiGroupCreationDialog(QDialog):
    """
    Dialog for creating multiple groups at once with keyword matching.
    """
    def __init__(self, available_datasets: List[str]):
        # Table with columns:
        # - Group Name (QLineEdit)
        # - Include Keywords (QLineEdit, comma-separated)
        # - Exclude Keywords (QLineEdit, comma-separated)
        # - Auto-Assign (QCheckBox)
        pass
    
    def add_row(self):
        """Add new row to table."""
        pass
    
    def remove_row(self, index: int):
        """Remove row from table."""
        pass
    
    def preview_assignments(self) -> Dict[str, List[str]]:
        """
        Calculate which datasets would be assigned to each group.
        Returns: {group_name: [dataset1, dataset2, ...]}
        """
        pass
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate configuration:
        - No empty group names
        - No duplicate group names
        - No dataset assigned to multiple groups (if strict mode)
        - At least one keyword or manual selection per group
        Returns: (is_valid, error_message)
        """
        pass
    
    def apply(self) -> List[Dict]:
        """
        Return list of group configurations:
        [
            {
                "name": "Control",
                "datasets": ["dataset1", "dataset2"],
                "keywords": {"include": ["ctrl", "con"], "exclude": []}
            },
            ...
        ]
        """
        pass
```

#### Preview Widget
```python
class AssignmentPreviewWidget(QWidget):
    """
    Shows preview of which datasets will be assigned to which groups.
    """
    def __init__(self):
        # Tree or list view:
        # ├─ Control (3 datasets)
        # │  ├─ dataset1_ctrl
        # │  ├─ dataset2_control
        # │  └─ dataset3_con
        # ├─ Treatment (2 datasets)
        # │  ├─ dataset4_treat
        # │  └─ dataset5_tx
        # └─ Unassigned (1 dataset)
        #    └─ dataset6_other
        pass
```

### 2. Matching Algorithm

```python
def match_datasets_to_groups(
    datasets: List[str],
    group_configs: List[Dict]
) -> Dict[str, List[str]]:
    """
    Match datasets to groups based on include/exclude keywords.
    
    Args:
        datasets: List of available dataset names
        group_configs: List of {
            "name": str,
            "include": List[str],  # keywords to match
            "exclude": List[str],  # keywords to reject
            "auto_assign": bool
        }
    
    Returns:
        {group_name: [matched_datasets]}
    
    Algorithm:
        1. For each dataset:
            - Check each group's include keywords (case-insensitive)
            - If ANY include keyword matches → candidate
            - If ANY exclude keyword matches → reject
            - If multiple groups match → assign to first match (or flag conflict)
        2. Return assignments
    
    Example:
        datasets = ["ctrl_sample1", "ctrl_sample2", "treat_sample1", "other"]
        groups = [
            {"name": "Control", "include": ["ctrl", "con"], "exclude": [], "auto_assign": True},
            {"name": "Treatment", "include": ["treat", "tx"], "exclude": ["ctrl"], "auto_assign": True}
        ]
        
        Result:
        {
            "Control": ["ctrl_sample1", "ctrl_sample2"],
            "Treatment": ["treat_sample1"],
            "Unassigned": ["other"]
        }
    """
    pass
```

### 3. Integration with Existing Code

**Location**: `pages/analysis_page_utils/method_view.py`  
**Function**: `create_dataset_selector_with_groups()` (around line 500-800)

**Changes Needed**:
1. Add "Create Multiple Groups" button next to "Add Group"
2. Open MultiGroupCreationDialog when clicked
3. Process returned group configs
4. Update group list display
5. Trigger plot update

**Pseudocode**:
```python
# In create_dataset_selector_with_groups()

# Existing "Add Group" button
add_group_btn = QPushButton(localize_func("ANALYSIS_PAGE.add_group"))
add_group_btn.clicked.connect(add_single_group)

# NEW: "Create Multiple Groups" button
add_multi_groups_btn = QPushButton(localize_func("ANALYSIS_PAGE.create_multiple_groups"))
add_multi_groups_btn.setIcon(QIcon.fromTheme("list-add-multiple"))
add_multi_groups_btn.clicked.connect(open_multi_group_dialog)

def open_multi_group_dialog():
    available_datasets = get_available_dataset_names()
    dialog = MultiGroupCreationDialog(available_datasets)
    
    if dialog.exec() == QDialog.Accepted:
        group_configs = dialog.apply()
        
        # Clear existing groups if desired
        # (or add option in dialog: "Replace existing groups" checkbox)
        
        # Add all groups
        for config in group_configs:
            add_group_to_ui(
                name=config["name"],
                datasets=config["datasets"]
            )
        
        # Trigger update
        update_analysis_selector()
```

---

## Localization Requirements

### New Keys Needed

#### English (`en.json`)
```json
"ANALYSIS_PAGE": {
    "create_multiple_groups": "Create Multiple Groups",
    "multi_group_dialog_title": "Create Multiple Groups at Once",
    "group_name_column": "Group Name",
    "include_keywords_column": "Include Keywords",
    "exclude_keywords_column": "Exclude Keywords",
    "auto_assign_column": "Auto-Assign",
    "add_row_button": "Add Row",
    "remove_row_button": "Remove",
    "preview_assignments_button": "Preview Assignments",
    "apply_groups_button": "Apply",
    "cancel_button": "Cancel",
    "keywords_hint": "Comma-separated, case-insensitive",
    "preview_title": "Assignment Preview",
    "unassigned_group": "Unassigned",
    "conflict_warning": "Warning: Some datasets match multiple groups",
    "no_matches_warning": "Warning: Group '{group}' has no matching datasets",
    "duplicate_name_error": "Error: Duplicate group name '{name}'",
    "empty_name_error": "Error: Group name cannot be empty"
}
```

#### Japanese (`ja.json`)
```json
"ANALYSIS_PAGE": {
    "create_multiple_groups": "複数グループを作成",
    "multi_group_dialog_title": "複数グループを一度に作成",
    "group_name_column": "グループ名",
    "include_keywords_column": "キーワード（含む）",
    "exclude_keywords_column": "キーワード（除外）",
    "auto_assign_column": "自動割り当て",
    "add_row_button": "行を追加",
    "remove_row_button": "削除",
    "preview_assignments_button": "割り当てをプレビュー",
    "apply_groups_button": "適用",
    "cancel_button": "キャンセル",
    "keywords_hint": "カンマ区切り、大文字小文字を区別しない",
    "preview_title": "割り当てプレビュー",
    "unassigned_group": "未割り当て",
    "conflict_warning": "警告: 一部のデータセットが複数グループに一致",
    "no_matches_warning": "警告: グループ '{group}' に一致するデータセットなし",
    "duplicate_name_error": "エラー: 重複するグループ名 '{name}'",
    "empty_name_error": "エラー: グループ名を空にできません"
}
```

---

## Testing Strategy

### Unit Tests

```python
def test_keyword_matching_basic():
    """Test basic include keyword matching."""
    datasets = ["ctrl_sample1", "treat_sample2", "other"]
    groups = [{"name": "Control", "include": ["ctrl"], "exclude": [], "auto_assign": True}]
    
    result = match_datasets_to_groups(datasets, groups)
    
    assert "ctrl_sample1" in result["Control"]
    assert "treat_sample2" not in result["Control"]

def test_keyword_matching_exclude():
    """Test exclude keyword logic."""
    datasets = ["ctrl_sample1", "ctrl_treat_sample2"]
    groups = [
        {"name": "Control", "include": ["ctrl"], "exclude": ["treat"], "auto_assign": True}
    ]
    
    result = match_datasets_to_groups(datasets, groups)
    
    assert "ctrl_sample1" in result["Control"]
    assert "ctrl_treat_sample2" not in result["Control"]  # Excluded

def test_multiple_groups_priority():
    """Test which group wins when multiple match."""
    datasets = ["ctrl_treat_sample"]
    groups = [
        {"name": "Control", "include": ["ctrl"], "exclude": [], "auto_assign": True},
        {"name": "Treatment", "include": ["treat"], "exclude": [], "auto_assign": True}
    ]
    
    result = match_datasets_to_groups(datasets, groups)
    
    # Should assign to first matching group (Control)
    assert "ctrl_treat_sample" in result["Control"]
    assert "ctrl_treat_sample" not in result["Treatment"]

def test_case_insensitive_matching():
    """Test case-insensitive keyword matching."""
    datasets = ["CTRL_Sample", "ctrl_sample", "Ctrl_SAMPLE"]
    groups = [{"name": "Control", "include": ["ctrl"], "exclude": [], "auto_assign": True}]
    
    result = match_datasets_to_groups(datasets, groups)
    
    assert len(result["Control"]) == 3

def test_validation_duplicate_names():
    """Test validation catches duplicate group names."""
    configs = [
        {"name": "Control", "include": ["ctrl"], "exclude": []},
        {"name": "Control", "include": ["con"], "exclude": []}
    ]
    
    is_valid, error = validate_group_configs(configs)
    
    assert not is_valid
    assert "duplicate" in error.lower()

def test_validation_empty_name():
    """Test validation catches empty group names."""
    configs = [{"name": "", "include": ["ctrl"], "exclude": []}]
    
    is_valid, error = validate_group_configs(configs)
    
    assert not is_valid
    assert "empty" in error.lower()
```

### Integration Tests

```python
def test_multi_group_dialog_workflow():
    """Test full dialog workflow."""
    datasets = ["ctrl1", "ctrl2", "treat1", "treat2", "other"]
    
    dialog = MultiGroupCreationDialog(datasets)
    
    # Add two groups
    dialog.add_row()
    dialog.set_group_config(0, name="Control", include="ctrl", exclude="", auto_assign=True)
    
    dialog.add_row()
    dialog.set_group_config(1, name="Treatment", include="treat", exclude="", auto_assign=True)
    
    # Preview
    preview = dialog.preview_assignments()
    
    assert len(preview["Control"]) == 2
    assert len(preview["Treatment"]) == 2
    assert len(preview["Unassigned"]) == 1
    
    # Apply
    configs = dialog.apply()
    
    assert len(configs) == 2
    assert configs[0]["name"] == "Control"
    assert len(configs[0]["datasets"]) == 2
```

### UI Tests

```python
def test_ui_create_multiple_groups_button_exists():
    """Verify button appears in UI."""
    page = create_analysis_page()
    button = page.findChild(QPushButton, name="create_multiple_groups_btn")
    
    assert button is not None
    assert button.text() == localize_func("ANALYSIS_PAGE.create_multiple_groups")

def test_ui_dialog_opens():
    """Verify dialog opens when button clicked."""
    page = create_analysis_page()
    button = page.findChild(QPushButton, name="create_multiple_groups_btn")
    
    # Simulate click
    button.click()
    
    # Verify dialog opened
    dialog = QApplication.activeModalWidget()
    assert isinstance(dialog, MultiGroupCreationDialog)
```

---

## Implementation Plan

### Phase 1: Core Algorithm (2 hours)
1. Implement `match_datasets_to_groups()` function
2. Implement `validate_group_configs()` function
3. Write and pass all unit tests
4. Handle edge cases:
   - Empty include lists
   - Conflicting matches
   - Case-insensitive matching
   - Special characters in dataset names

### Phase 2: Dialog UI (2 hours)
1. Create `MultiGroupCreationDialog` class
2. Implement table with editable columns
3. Add "Add Row" / "Remove Row" buttons
4. Implement "Preview Assignments" with tree view
5. Add validation with error messages
6. Apply localization to all strings

### Phase 3: Integration (1 hour)
1. Add "Create Multiple Groups" button to analysis page
2. Wire up dialog to existing group management system
3. Test with real datasets
4. Handle group replacement vs. addition
5. Update documentation

### Phase 4: Testing & Polish (1 hour)
1. Run all automated tests
2. Manual testing with various dataset patterns
3. Test with Japanese locale
4. Fix any bugs found
5. Update .AGI-BANKS and .docs

---

## Design Mockups

### Multi-Group Dialog Layout

```
┌────────────────────────────────────────────────────────────┐
│  Create Multiple Groups at Once                        × │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Define groups with keyword patterns:                     │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Group Name │ Include Keywords │ Exclude │ Auto ▼│×│  │ │
│  ├────────────┼──────────────────┼─────────┼──────┼─┤  │ │
│  │ Control    │ ctrl, con        │         │ [✓]  │ │  │ │
│  │ Treatment  │ treat, tx        │         │ [✓]  │ │  │ │
│  │ Disease    │ disease, dis     │ ctrl    │ [✓]  │ │  │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  [+ Add Row]                                               │
│                                                            │
│  💡 Keywords are comma-separated, case-insensitive        │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Preview Assignments                                   │ │
│  ├──────────────────────────────────────────────────────┤ │
│  │ ├─ Control (3 datasets)                              │ │
│  │ │  ├─ dataset1_ctrl                                  │ │
│  │ │  ├─ dataset2_control                               │ │
│  │ │  └─ dataset3_con                                   │ │
│  │ ├─ Treatment (2 datasets)                            │ │
│  │ │  ├─ dataset4_treat                                 │ │
│  │ │  └─ dataset5_tx                                    │ │
│  │ ├─ Disease (1 dataset)                               │ │
│  │ │  └─ dataset6_disease                               │ │
│  │ └─ Unassigned (1 dataset)                            │ │
│  │    └─ dataset7_other                                 │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│                            [Preview] [Apply] [Cancel]     │
└────────────────────────────────────────────────────────────┘
```

---

## Potential Challenges

### 1. Conflict Resolution
**Issue**: What if a dataset matches multiple groups?

**Options**:
- **Option A**: Assign to first matching group (predictable, simple)
- **Option B**: Flag as conflict, require manual resolution (safest)
- **Option C**: Allow multi-group assignment (complex, may confuse users)

**Recommendation**: Option A with warning in preview

### 2. Performance
**Issue**: Large datasets (100+ datasets) may slow down matching

**Solutions**:
- Cache compiled regex patterns
- Use async preview updates
- Limit preview to first 50 matches per group

### 3. Localization
**Issue**: Keywords may need to work in multiple languages

**Solutions**:
- Support Unicode keywords (Japanese, Chinese, etc.)
- Document that keywords are literal string matching (not translated)
- Consider adding preset keyword templates

### 4. UX Complexity
**Issue**: Dialog may be too complex for simple use cases

**Solutions**:
- Keep "Add Group" button for simple single-group workflow
- Add "Import from CSV" for advanced users
- Provide example templates in tooltip

---

## Dependencies

### Required
- Existing group management system (already implemented)
- Localization framework (already implemented)
- QTableWidget or QTableView for editable table

### Optional
- CSV import/export for group configs
- Regex support for advanced pattern matching
- Group templates (save and load common patterns)

---

## Success Criteria

1. ✅ User can create 3+ groups in a single dialog
2. ✅ Keyword matching works case-insensitively
3. ✅ Exclude keywords properly filter datasets
4. ✅ Preview accurately shows assignments before applying
5. ✅ Validation prevents errors (duplicate names, empty fields)
6. ✅ All strings are localized (EN + JA)
7. ✅ Performance acceptable with 100+ datasets
8. ✅ No regression in existing single-group workflow

---

## Related Files

- `pages/analysis_page_utils/method_view.py` - Main integration point
- `assets/locales/en.json` - English strings
- `assets/locales/ja.json` - Japanese strings
- `.docs/summary/2025-01-21_pca-ui-improvements-localization.md` - Related improvements

---

## Status Updates

| Date       | Status     | Notes                                                       |
| ---------- | ---------- | ----------------------------------------------------------- |
| 2025-01-21 | ⏸️ Deferred | Tasks 1-6 prioritized; Task 7 deferred to dedicated session |

---

## When to Resume

**Triggers**:
1. User explicitly requests multi-group functionality
2. Feedback indicates single-group workflow is too slow
3. Available development time slot (4-6 hours)

**Prerequisites**:
1. Tasks 1-6 tested and validated
2. No critical bugs in current implementation
3. Localization framework stable

---

**End of TODO Document**

*Created by: Beast Mode Agent v4.1*  
*Last Updated: 2025-01-21*
