# Data Package Page - Testing Guide

**Date**: 2025-10-14  
**Version**: 1.0  
**Status**: Production Testing Required

## Overview

This document provides comprehensive testing procedures for the Data Package Page after the October 14, 2025 UX fixes. All 6 critical issues have been addressed and require validation.

## Fixed Issues Summary

1. ✅ **Y-axis visibility** - Enhanced matplotlib margins
2. ✅ **Preview title prefix** - Removed "Preview:" prefix
3. ✅ **Title update on selection** - Title changes when selecting datasets
4. ✅ **Metadata editor** - Full redesign with edit/view/save/export functionality
5. ✅ **Documentation** - Complete technical documentation in RECENT_CHANGES.md
6. ⏳ **Testing** - Requires end-to-end validation

## Test Environment Setup

### Prerequisites
- Python 3.10+ with uv or venv
- All dependencies installed (PySide6, pandas, matplotlib, etc.)
- At least 3 sample Raman spectroscopy datasets
- Sample metadata JSON files

### Preparation
1. Launch application: `uv run python main.py`
2. Create or load a test project
3. Have sample data ready:
   - Single dataset files (TXT, CSV, ASC, PKL)
   - Folder with multiple data files
   - Parent folder with multiple dataset subfolders (for batch import)
   - metadata.json files (optional)

## Test Cases

### Test Case 1: Y-Axis Visibility ✅

**Objective**: Verify y-axis labels are clearly visible

**Steps**:
1. Import any dataset
2. Click "Preview" button
3. Observe the data preview graph

**Expected Results**:
- [ ] Y-axis labels (intensity values) fully visible
- [ ] No label cutoff on the left side
- [ ] Font size consistent and readable (10pt)
- [ ] Proper spacing between axis and labels

**Validation Criteria**:
- Left margin: 12% of plot width
- Labels not overlapping with plot area
- All numeric values clearly readable

---

### Test Case 2: Preview Title (Import) ✅

**Objective**: Verify preview title shows dataset name without "Preview:" prefix

**Steps**:
1. Go to Data Package Page
2. Import a single dataset (e.g., "20220221_MM01_B")
3. Click "Preview"
4. Check the preview title

**Expected Results**:
- [ ] Title shows only dataset name: "20220221_MM01_B"
- [ ] No "Preview:" prefix
- [ ] Title updates after preview loads

**Validation Criteria**:
- Format: `{dataset_name}` only
- No extra text or prefixes

---

### Test Case 3: Preview Title (Selection) ✅

**Objective**: Verify preview title updates when selecting different datasets

**Steps**:
1. Load project with at least 3 datasets
2. Go to Data Package Page
3. Click first dataset in "Project Datasets" list
4. Observe preview title
5. Click second dataset
6. Observe title change
7. Click third dataset
8. Observe title change

**Expected Results**:
- [ ] Title updates to match selected dataset (1st dataset)
- [ ] Title changes when selecting 2nd dataset
- [ ] Title changes when selecting 3rd dataset
- [ ] Title clears when deselecting all

**Validation Criteria**:
- Title always reflects currently displayed dataset
- Immediate update on selection change

---

### Test Case 4: Metadata Editor - View Mode 📝

**Objective**: Verify metadata displays in read-only mode for loaded datasets

**Steps**:
1. Load project with datasets that have metadata
2. Select a dataset from "Project Datasets" list
3. Check Metadata section

**Expected Results**:
- [ ] Metadata section displays metadata content
- [ ] All fields are read-only (cannot type)
- [ ] Edit button (pencil icon) visible in title bar
- [ ] Save button NOT visible
- [ ] Export button visible

**Validation Criteria**:
- Fields show stored metadata values
- Clicking in fields shows read-only cursor
- Edit button has tooltip: "Edit metadata for selected dataset"

---

### Test Case 5: Metadata Editor - Edit Mode 📝

**Objective**: Verify metadata can be edited when edit button is clicked

**Steps**:
1. Select a dataset with metadata
2. Click the Edit button (pencil icon) in Metadata title bar
3. Observe changes
4. Try editing metadata fields

**Expected Results**:
- [ ] Edit button turns blue (checked state)
- [ ] All metadata fields become editable
- [ ] Save button (green checkmark) appears
- [ ] Can type in text fields
- [ ] Can edit all tabs (Sample, Instrument, Measurement, Notes)

**Validation Criteria**:
- Edit button tooltip changes to: "View mode (read-only)"
- All QLineEdit and QTextEdit widgets accept input
- Save button tooltip: "Save metadata to JSON file"

---

### Test Case 6: Metadata Editor - Save to Project 💾

**Objective**: Verify metadata saves to project correctly

**Steps**:
1. Select a dataset
2. Click Edit button
3. Modify metadata (e.g., change "Patient ID" field)
4. Click Save button (green checkmark)
5. Observe notification
6. Check if edit mode exits
7. Re-select the same dataset
8. Verify changes persisted

**Expected Results**:
- [ ] Success notification: "Metadata for '{name}' saved successfully"
- [ ] Edit button unchecks (returns to view mode)
- [ ] Save button disappears
- [ ] Fields return to read-only
- [ ] Re-selecting dataset shows saved changes
- [ ] Project JSON file updated

**Validation Criteria**:
- Metadata saved to `PROJECT_MANAGER.current_project_data`
- Changes persist after app restart
- No data loss

---

### Test Case 7: Metadata Editor - Export to JSON 📤

**Objective**: Verify metadata exports to external JSON file

**Steps**:
1. Select a dataset with metadata
2. Click Export button (orange icon) in Metadata title bar
3. Choose save location and filename
4. Save file
5. Open exported JSON file

**Expected Results**:
- [ ] File dialog opens with suggested filename
- [ ] Default location: project data folder
- [ ] JSON file created successfully
- [ ] Success notification: "Metadata successfully saved to file"
- [ ] JSON file contains all metadata fields
- [ ] JSON properly formatted (indented, UTF-8)

**Validation Criteria**:
- File format: `{dataset_name}_metadata.json`
- Valid JSON structure with all tabs and fields
- Unicode characters preserved (Japanese, etc.)

---

### Test Case 8: Single Dataset Import 📥

**Objective**: Verify single dataset import workflow

**Steps**:
1. Click Browse button
2. Select "Select Folder" option
3. Choose a data folder
4. Click Preview
5. Check preview displays
6. Enter dataset name
7. Click "Add to Project"

**Expected Results**:
- [ ] Browse dialog shows file/folder choice
- [ ] Preview loads correctly
- [ ] Preview title shows dataset name (no "Preview:")
- [ ] Graph displays with visible y-axis
- [ ] Dataset added to project
- [ ] Dataset appears in "Project Datasets" list

**Validation Criteria**:
- All preview features working
- Dataset saved to project
- Metadata imported if metadata.json present

---

### Test Case 9: Batch Dataset Import 📥📥📥

**Objective**: Verify batch import of multiple datasets

**Steps**:
1. Prepare parent folder with 3+ dataset subfolders
2. Click Browse → Select Folder
3. Choose parent folder
4. Wait for batch import progress dialog
5. Check dataset selector dropdown
6. Switch between datasets in selector
7. Click "Add to Project"

**Expected Results**:
- [ ] Progress dialog shows import status
- [ ] Success count and failed count displayed
- [ ] Dataset selector populated with all folders
- [ ] Switching datasets updates preview and title
- [ ] All datasets added to project in one click

**Validation Criteria**:
- Multiple datasets imported simultaneously
- Each dataset has its own metadata (if available)
- Preview works for each dataset

---

### Test Case 10: Metadata Auto-Detection 🔍

**Objective**: Verify automatic metadata loading from metadata.json files

**Steps**:
1. Create dataset folder with data files + metadata.json
2. Import the folder
3. Click Preview
4. Check Metadata section

**Expected Results**:
- [ ] Metadata path field populated automatically
- [ ] Notification: "Metadata auto-filled from JSON"
- [ ] Metadata displays in editor
- [ ] All fields populated from JSON

**Validation Criteria**:
- Metadata loaded without manual browsing
- Correct mapping of JSON keys to fields

---

### Test Case 11: End-to-End Workflow 🔄

**Objective**: Complete workflow from import to export

**Steps**:
1. Import 3 datasets (single + batch)
2. Select each dataset and view metadata
3. Edit metadata for one dataset
4. Save edited metadata
5. Export metadata to JSON
6. Verify all functionality

**Expected Results**:
- [ ] All imports successful
- [ ] All previews working
- [ ] Metadata editing works
- [ ] Metadata saving works
- [ ] Metadata export works
- [ ] No errors in any step

**Validation Criteria**:
- Complete workflow without errors
- All features integrated properly

---

## Regression Testing

### Areas to Check
- [ ] Home page still loads correctly
- [ ] Preprocessing page unaffected
- [ ] Other tabs functional
- [ ] Project loading/saving works
- [ ] No console errors
- [ ] No crashes

---

## Known Limitations

1. **Metadata structure**: Fixed by app configuration (not user-customizable)
2. **Export format**: JSON only (no CSV or Excel)
3. **Batch import**: Limited to folders containing data files (not recursive)

---

## Bug Report Template

If issues found during testing:

```markdown
### Bug Report

**Issue Title**: [Brief description]

**Test Case**: [Which test case failed]

**Steps to Reproduce**:
1. 
2. 
3. 

**Expected Behavior**: 

**Actual Behavior**: 

**Screenshots**: [If applicable]

**Environment**:
- OS: 
- Python Version: 
- Dependencies: 

**Error Messages**: [If any]

**Severity**: [Critical / High / Medium / Low]
```

---

## Completion Checklist

### Before Release
- [ ] All 11 test cases passed
- [ ] No regression issues
- [ ] Documentation updated
- [ ] User guide updated
- [ ] Changelog entry added
- [ ] Git commit with descriptive message

### Performance Metrics
- [ ] Import speed acceptable (< 2s for single dataset)
- [ ] Batch import shows progress (no freeze)
- [ ] Preview rendering smooth (< 1s)
- [ ] Metadata save instant (< 500ms)
- [ ] No memory leaks after 10+ imports

### User Acceptance
- [ ] All user-reported issues resolved
- [ ] UI/UX improvements validated
- [ ] No new issues introduced

---

## Contact & Support

**Developer**: AI Agent (GitHub Copilot)  
**Date**: 2025-10-14  
**Documentation**: `.AGI-BANKS/RECENT_CHANGES.md`  
**Issue Tracker**: See `.docs/TODOS.md`

---

## Appendix: Test Data Samples

### Sample metadata.json
```json
{
    "sample": {
        "patient_id": "P001",
        "sample_id": "S001",
        "sample_source": "Blood",
        "sample_prep": "Centrifuged"
    },
    "instrument": {
        "instrument_model": "Renishaw InVia",
        "laser_wavelength": "532",
        "laser_power": "10",
        "objective": "50x",
        "grating": "1800"
    },
    "measurement": {
        "integration_time": "1.0",
        "accumulations": "3",
        "map_area": "100x100",
        "step_size": "1.0",
        "calibration": "Silicon peak at 520.5 cm⁻¹",
        "operator": "Researcher A",
        "acquisition_date": "2025-10-14"
    },
    "notes": {
        "notes": "Test sample for quality control"
    }
}
```

### Folder Structure Example
```
test_data/
├── dataset1/
│   ├── spectrum_001.txt
│   ├── spectrum_002.txt
│   └── metadata.json
├── dataset2/
│   ├── data.csv
│   └── metadata.json
└── dataset3/
    └── raman_data.asc
```

---

**END OF TESTING GUIDE**
