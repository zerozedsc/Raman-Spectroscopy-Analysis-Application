# Testing Instructions: Preprocess Page UI Enhancements
**Date**: October 16, 2025  
**Features**: Select All Button, Preview Toggle Defaults, Pipeline Dialog Styling  
**Status**: Ready for Testing  
**Priority**: High

---

## 📋 Overview

This document provides comprehensive testing instructions for three UI enhancements to the preprocess page:
1. **Tab-Aware Select All Button** - Quick selection toggle per tab
2. **Intelligent Preview Toggle Defaults** - Context-aware preview behavior
3. **Enhanced Pipeline Dialog Styling** - Professional dialog appearance

---

## 🔧 Pre-Testing Setup

### Requirements
- ✅ Application running with no syntax errors
- ✅ Project loaded with multiple datasets (both raw and preprocessed)
- ✅ At least 3+ datasets in each tab for meaningful testing

### Test Environment
1. **Language**: Test in both English and Japanese
2. **Datasets**: Prepare test data:
   - **Raw datasets**: 5+ raw spectral files
   - **Preprocessed datasets**: 3+ preprocessed files from project
3. **Pipeline**: Create at least one preprocessing pipeline with 2+ steps

### Before You Start
- Note current application version
- Clear any error logs
- Take screenshots of issues found
- Document exact steps to reproduce any bugs

---

## 🧪 Test Suite 1: Tab-Aware Select All Button

### Location
**Preprocess Page** → **Input Datasets Section** → **Title Bar** (checkmark icon button)

### Test 1.1: Visual Verification ✅
**Objective**: Verify button appearance and positioning

**Steps**:
1. Open preprocess page
2. Locate input datasets section title bar
3. Find checkmark icon button (should be before refresh button)

**Expected Results**:
- ✅ Button is 24x24 pixels with 14x14px checkmark icon
- ✅ Button is positioned in title bar before refresh button
- ✅ Checkmark icon is blue (#0078d4)
- ✅ Hover shows light blue background (#e7f3ff)
- ✅ Tooltip appears on hover

**Screenshot**: Take screenshot showing button in title bar

---

### Test 1.2: Tooltip Localization ✅
**Objective**: Verify tooltip text in both languages

**Steps**:
1. **English Mode**:
   - Hover over select all button
   - Read tooltip text
2. **Japanese Mode** (switch language in settings if available):
   - Hover over select all button
   - Read tooltip text

**Expected Results**:
- ✅ EN: "Select/deselect all datasets in current tab"
- ✅ JA: "現在のタブのすべてのデータセットを選択/選択解除"

---

### Test 1.3: Select All in "All" Tab ✅
**Objective**: Test selection behavior in All tab

**Steps**:
1. Switch to "All" tab
2. Ensure no datasets are selected (click empty area)
3. Click select all button
4. Verify selection state
5. Click select all button again
6. Verify deselection state

**Expected Results**:
- ✅ First click: All datasets in All tab become selected
- ✅ Datasets in other tabs remain unaffected
- ✅ Second click: All datasets in All tab become deselected
- ✅ Selection count updates correctly

**Report**: Number of datasets selected: _____

---

### Test 1.4: Select All in "Raw" Tab ✅
**Objective**: Test selection behavior in Raw tab

**Steps**:
1. Switch to "Raw" tab
2. Manually select 2 datasets
3. Click select all button (should select remaining)
4. Verify all raw datasets selected
5. Click select all button again
6. Verify all deselected

**Expected Results**:
- ✅ First click: All remaining raw datasets become selected
- ✅ Previously selected datasets remain selected
- ✅ Second click: All raw datasets deselected
- ✅ Preprocessed tab datasets remain unchanged

**Report**: Number of raw datasets: _____

---

### Test 1.5: Select All in "Preprocessed" Tab ✅
**Objective**: Test selection behavior in Preprocessed tab

**Steps**:
1. Switch to "Preprocessed" tab
2. Click select all button
3. Verify selection
4. Switch to "Raw" tab
5. Verify raw tab selection unchanged
6. Return to "Preprocessed" tab
7. Click select all button again

**Expected Results**:
- ✅ First click: All preprocessed datasets selected
- ✅ Raw tab datasets remain unchanged
- ✅ Second click: All preprocessed datasets deselected
- ✅ Tab switching preserves selection state

**Report**: Number of preprocessed datasets: _____

---

### Test 1.6: Edge Case - Empty List ✅
**Objective**: Test behavior with no datasets

**Steps**:
1. Create new project or clear all datasets
2. Open preprocess page
3. Click select all button in each tab

**Expected Results**:
- ✅ Button clickable but no action (list is empty)
- ✅ No errors or crashes
- ✅ Tooltip still displays correctly

---

### Test 1.7: Edge Case - Single Item ✅
**Objective**: Test toggle behavior with one dataset

**Steps**:
1. Ensure tab has only one dataset
2. Click select all button (should select)
3. Click again (should deselect)
4. Repeat 3 times

**Expected Results**:
- ✅ First click: Single item selected
- ✅ Second click: Single item deselected
- ✅ Consistent toggle behavior
- ✅ No visual glitches

---

### Test 1.8: Interaction with Manual Selection ✅
**Objective**: Verify button works correctly after manual selection

**Steps**:
1. Manually select 3 datasets
2. Click select all button
3. Manually deselect 1 dataset
4. Click select all button
5. Verify state

**Expected Results**:
- ✅ Step 2: Remaining datasets become selected (all selected)
- ✅ Step 4: Deselects all (because not all were selected)
- ✅ Consistent with "all selected → deselect, otherwise → select" logic

---

## 🧪 Test Suite 2: Intelligent Preview Toggle Defaults

### Location
**Preprocess Page** → **Visualization Section** → **Preview Toggle Button** (eye icon)

### Test 2.1: Initial State - First Open ✅
**Objective**: Verify correct default state on first application launch

**Steps**:
1. Close application completely
2. Launch application fresh
3. Load project
4. Open preprocess page
5. Check preview toggle state (should start on "All" or "Raw" tab)

**Expected Results**:
- ✅ Preview toggle is ON (eye open icon)
- ✅ Toggle shows "Preview ON" text
- ✅ Correct for raw data (preview needed to see effects)

**Screenshot**: Take screenshot of initial state

---

### Test 2.2: Tab Switch - Raw to Preprocessed ✅
**Objective**: Test preview toggle updates when switching to preprocessed

**Steps**:
1. Start on "Raw" tab (preview should be ON)
2. Click "Preprocessed" tab
3. Observe preview toggle state change

**Expected Results**:
- ✅ Preview toggle automatically switches to OFF
- ✅ Eye icon changes to eye-closed
- ✅ Text changes to "Preview OFF"
- ✅ No manual click needed

**Screenshot**: Before and after tab switch

---

### Test 2.3: Tab Switch - Preprocessed to Raw ✅
**Objective**: Test preview toggle updates when switching to raw

**Steps**:
1. Start on "Preprocessed" tab (preview should be OFF)
2. Click "Raw" tab
3. Observe preview toggle state change

**Expected Results**:
- ✅ Preview toggle automatically switches to ON
- ✅ Eye icon changes to eye-open
- ✅ Text changes to "Preview ON"
- ✅ Automatic adjustment prevents double preprocessing

---

### Test 2.4: All Tab Behavior ✅
**Objective**: Verify "All" tab treated as raw data

**Steps**:
1. Switch to "All" tab
2. Check preview toggle state

**Expected Results**:
- ✅ Preview toggle is ON
- ✅ Treated same as raw data tab
- ✅ Switching from "Preprocessed" to "All" enables preview

---

### Test 2.5: Manual Override Persistence ✅
**Objective**: Test if manual changes persist within tab

**Steps**:
1. On "Raw" tab, manually turn preview OFF
2. Select different dataset in same tab
3. Check if preview stays OFF
4. Switch to "Preprocessed" tab
5. Switch back to "Raw" tab
6. Check preview state

**Expected Results**:
- ✅ Step 3: Preview stays OFF (manual override respected)
- ✅ Step 6: Preview resets to ON (tab change resets to default)
- ✅ Consistent behavior across tab switches

---

### Test 2.6: Preview Behavior Verification ✅
**Objective**: Confirm preview actually affects visualization

**Steps**:
1. **Raw Tab Test**:
   - Select raw dataset
   - Preview ON: Should show processed data
   - Preview OFF: Should show raw data
2. **Preprocessed Tab Test**:
   - Select preprocessed dataset
   - Preview OFF: Should show preprocessed data as-is
   - Preview ON (if manually enabled): Should show double-processed data

**Expected Results**:
- ✅ Preview toggle actually changes visualization
- ✅ Preprocessed data not double-processed when preview OFF
- ✅ Raw data shows effects when preview ON

---

### Test 2.7: Multiple Rapid Tab Switches ✅
**Objective**: Test stability during rapid tab switching

**Steps**:
1. Rapidly switch between tabs: Raw → Preprocessed → All → Raw → Preprocessed
2. Repeat 5 times
3. Check for visual glitches or errors

**Expected Results**:
- ✅ Preview toggle updates correctly each time
- ✅ No visual glitches or lag
- ✅ No error messages
- ✅ Icons and text always synchronized

---

## 🧪 Test Suite 3: Enhanced Pipeline Dialog Styling

### Location
**Preprocess Page** → **Pipeline Building Section**

### Test 3.1: Export Pipeline Dialog Visual Check ✅
**Objective**: Verify export dialog appearance and styling

**Steps**:
1. Create pipeline with 2+ steps
2. Click export pipeline button
3. Observe dialog appearance

**Expected Results**:
- ✅ White background (#ffffff)
- ✅ Clean title bar with proper text
- ✅ "Pipeline Name" label in bold (font-weight: 600)
- ✅ Name input field with border (1px solid #ced4da)
- ✅ Name placeholder: "e.g., MGUS Classification Pipeline"
- ✅ "Description" label in bold
- ✅ Description text area (max height 100px)
- ✅ Description placeholder: "Describe the purpose and use case..."
- ✅ Pipeline info box: "📊 X steps in current pipeline" (gray background)
- ✅ Cancel button (gray style)
- ✅ Export button (blue CTA style)

**Screenshot**: Full dialog view

---

### Test 3.2: Export Dialog Interaction ✅
**Objective**: Test input fields and focus states

**Steps**:
1. Click in name field
2. Observe focus state (border should turn blue)
3. Type test name
4. Click in description field
5. Observe focus state
6. Type test description
7. Hover over Cancel button
8. Hover over Export button

**Expected Results**:
- ✅ Focus state: Border changes to #0078d4 (blue)
- ✅ Text entry works smoothly
- ✅ Cancel hover: Background changes to #e9ecef
- ✅ Export hover: Background changes to #006abc (darker blue)
- ✅ Cursor changes to pointer on buttons

**Screenshot**: Focus and hover states

---

### Test 3.3: Export Dialog Functionality ✅
**Objective**: Test export process

**Steps**:
1. Open export dialog
2. Enter name: "Test Pipeline Export"
3. Enter description: "Testing export functionality"
4. Click Export button
5. Check for success notification
6. Verify file created in project pipelines folder

**Expected Results**:
- ✅ Dialog closes on export
- ✅ Success notification appears: "Pipeline 'Test Pipeline Export' exported successfully"
- ✅ JSON file created in `projects/<project_name>/pipelines/`
- ✅ File contains correct data

**Report**: File path: _____________________

---

### Test 3.4: Export Dialog Validation ✅
**Objective**: Test empty name validation

**Steps**:
1. Open export dialog
2. Leave name field empty
3. Click Export button

**Expected Results**:
- ✅ Warning notification: "Please provide a name for the pipeline"
- ✅ Dialog remains open
- ✅ User can correct and retry

---

### Test 3.5: Import Pipeline Dialog Visual Check ✅
**Objective**: Verify import dialog appearance and styling

**Steps**:
1. Ensure at least one saved pipeline exists
2. Click import pipeline button
3. Observe dialog appearance

**Expected Results**:
- ✅ White background
- ✅ Title: "Import Preprocessing Pipeline"
- ✅ "Saved Pipelines" label in bold
- ✅ List widget with border (1px solid #ced4da)
- ✅ Pipeline items show:
  - Name in bold
  - Step count and date (gray text, 11px)
  - Description (italic, gray)
- ✅ "Import from External File..." button
- ✅ Cancel button (gray)
- ✅ Import Pipeline button (blue CTA)

**Screenshot**: Full dialog view with pipeline list

---

### Test 3.6: Import Dialog List Interaction ✅
**Objective**: Test list selection and hover states

**Steps**:
1. Hover over pipeline items
2. Click to select pipeline
3. Observe selection state
4. Select different pipeline

**Expected Results**:
- ✅ Hover: Background changes to #f8f9fa
- ✅ Selection: Background changes to #e7f3ff (light blue)
- ✅ Selection: Left border appears (3px solid #0078d4)
- ✅ Only one pipeline selectable at a time
- ✅ Visual feedback is smooth

**Screenshot**: Hover and selection states

---

### Test 3.7: Import Dialog Functionality ✅
**Objective**: Test import from saved pipeline

**Steps**:
1. Open import dialog
2. Select a saved pipeline
3. Click Import button
4. Confirm replacement if prompted
5. Check pipeline loaded

**Expected Results**:
- ✅ Confirmation dialog if current pipeline exists
- ✅ Success notification: "Pipeline 'X' imported successfully (Y steps)"
- ✅ Pipeline steps appear in pipeline builder
- ✅ Step count matches imported pipeline

---

### Test 3.8: Import External File ✅
**Objective**: Test importing from external JSON file

**Steps**:
1. Open import dialog
2. Click "Import from External File..." button
3. File picker dialog appears
4. Select external pipeline JSON file
5. Confirm import

**Expected Results**:
- ✅ File picker shows "JSON Files (*.json)" filter
- ✅ Valid file imports successfully
- ✅ Invalid file shows error message
- ✅ Pipeline steps loaded correctly

---

### Test 3.9: Import Dialog - Empty State ✅
**Objective**: Test dialog when no saved pipelines exist

**Steps**:
1. Create new project or clear pipelines folder
2. Open import dialog

**Expected Results**:
- ✅ Message: "No saved pipelines found in this project"
- ✅ Message centered with gray color (#6c757d)
- ✅ "Import from External File..." button still available
- ✅ Import button present but no pipeline to select

---

### Test 3.10: Dialog Styling Consistency ✅
**Objective**: Verify consistent styling across dialogs

**Steps**:
1. Open export dialog, take note of styling
2. Close and open import dialog
3. Compare visual elements

**Expected Results**:
- ✅ Both use same background color
- ✅ Both use same button styling
- ✅ Both use same border radius (4px)
- ✅ Both use same color scheme (blue primary, gray secondary)
- ✅ Consistent padding and spacing

---

## 🧪 Test Suite 4: Integration Tests

### Test 4.1: Combined Feature Test ✅
**Objective**: Test all three features together

**Steps**:
1. Open preprocess page
2. Use select all button to select datasets in Raw tab
3. Verify preview toggle is ON
4. Switch to Preprocessed tab
5. Verify preview toggle turns OFF
6. Use select all button in Preprocessed tab
7. Create pipeline with selected data
8. Export pipeline using styled dialog
9. Clear pipeline
10. Import pipeline using styled dialog

**Expected Results**:
- ✅ All features work seamlessly together
- ✅ No conflicts or unexpected behavior
- ✅ Smooth workflow from selection to pipeline management

---

### Test 4.2: Performance Test ✅
**Objective**: Test with large dataset counts

**Steps**:
1. Load project with 50+ datasets
2. Test select all button (should be instant)
3. Switch tabs multiple times (should be smooth)
4. Open dialogs (should open quickly)

**Expected Results**:
- ✅ Select all completes instantly (< 100ms)
- ✅ Tab switching remains smooth
- ✅ Dialog opening is responsive
- ✅ No lag or freezing

**Report**: Number of datasets tested: _____

---

### Test 4.3: Language Switch Test ✅
**Objective**: Verify all features work after language change

**Steps**:
1. Test all features in English
2. Switch language to Japanese (if available)
3. Test all features in Japanese
4. Switch back to English

**Expected Results**:
- ✅ All tooltips update correctly
- ✅ Dialog titles and labels translate
- ✅ Functionality unchanged by language switch
- ✅ No missing translations

---

## 📊 Test Report Template

### Test Execution Summary
- **Tester Name**: _______________________
- **Test Date**: _______________________
- **Application Version**: _______________________
- **OS**: _______________________
- **Python Version**: _______________________

### Results Overview
| Test Suite | Tests Passed | Tests Failed | Not Tested |
|------------|-------------|--------------|------------|
| Suite 1: Select All | __ / 8 | __ | __ |
| Suite 2: Preview Toggle | __ / 7 | __ | __ |
| Suite 3: Pipeline Dialogs | __ / 10 | __ | __ |
| Suite 4: Integration | __ / 3 | __ | __ |
| **TOTAL** | **__ / 28** | **__** | **__** |

### Critical Issues Found
1. **Issue**: _______________________
   - **Severity**: High / Medium / Low
   - **Steps to Reproduce**: _______________________
   - **Screenshot**: _______________________

2. **Issue**: _______________________
   - **Severity**: High / Medium / Low
   - **Steps to Reproduce**: _______________________
   - **Screenshot**: _______________________

### Minor Issues / Suggestions
1. _______________________
2. _______________________

### Screenshots Checklist
- [ ] Select all button in title bar
- [ ] Preview toggle ON state
- [ ] Preview toggle OFF state
- [ ] Export pipeline dialog
- [ ] Import pipeline dialog
- [ ] Pipeline list with selection
- [ ] Any bugs or visual glitches

---

## 🐛 Bug Reporting Template

If you find any issues, please report using this format:

```
**Bug Title**: [Short description]

**Feature**: [Select All / Preview Toggle / Pipeline Dialogs]

**Severity**: [Critical / High / Medium / Low]

**Steps to Reproduce**:
1. Step one
2. Step two
3. Step three

**Expected Behavior**:
[What should happen]

**Actual Behavior**:
[What actually happens]

**Screenshots**:
[Attach screenshots if applicable]

**Console Errors**:
[Any error messages from logs]

**Environment**:
- OS: [Windows / Mac / Linux]
- Python Version: [e.g., 3.11.5]
- Application Version: [Check about page]
```

---

## ✅ Final Checklist

Before reporting results, ensure:
- [ ] All 28 tests executed
- [ ] Screenshots captured for visual tests
- [ ] Any bugs documented with reproduction steps
- [ ] Test report template filled out
- [ ] Console logs checked for warnings/errors
- [ ] Both English and Japanese tested (if applicable)

---

## 📝 Notes

- **Testing Time Estimate**: 45-60 minutes for complete suite
- **Priority**: Focus on Suite 1 and 2 first (core functionality)
- **Known Limitations**: None currently documented
- **Future Enhancements**: Keyboard shortcuts for select all (Ctrl+A)

---

## 📚 Related Documentation

- `.AGI-BANKS/RECENT_CHANGES.md` - Implementation details
- `.AGI-BANKS/IMPLEMENTATION_PATTERNS.md` - Design patterns used
- `.AGI-BANKS/UI_TITLE_BAR_STANDARD.md` - UI standards
- `.docs/pages/preprocess_page.md` - Page documentation

---

**End of Testing Instructions**  
**Questions?** Contact development team or create issue with `testing` tag.
