# Quick Testing Guide - PCA Improvements

## How to Test the New Features

### 1. Spectrum Preview Display Options

**Steps**:
1. Run PCA analysis with 2-3 datasets
2. Go to "Spectrum Preview" tab
3. Look for "⚙️ Display Options" button in the toolbar (top of the plot area)
4. Click the button to open the sidebar

**Expected Result**:
- Sidebar slides in from the left (220px width)
- Shows "Select All" checkbox (should be checked by default)
- Shows list of datasets with checkboxes (all checked by default)
  - Example: "☑ Glucose (15 spectra)"
- Shows "Display Mode" dropdown at top

**Test Actions**:
- ✅ Uncheck one dataset → Plot should update to show only remaining datasets
- ✅ Uncheck all → Plot should be empty or show placeholder
- ✅ Check multiple datasets → Plot should show selected datasets overlaid
- ✅ Click "Select All" → All checkboxes should toggle
- ✅ Change Display Mode to "All Spectra" → Should show all individual spectra instead of means
- ✅ Click toggle button again → Sidebar should close

---

### 2. Loading Plot 2-Column Grid

**Steps**:
1. Run PCA analysis with 5+ components
2. Go to "Loading Plot" tab
3. Look for "🔧 Show Components" button in the toolbar
4. Click the button to open the sidebar

**Expected Result**:
- Sidebar slides in from the left (220px width)
- Shows "Select All" checkbox
- Shows list of PC components with explained variance
  - Example: "☑ PC1 (45.2%)", "☑ PC2 (25.8%)", etc.
- Default: First 4 components checked
- Plot shows components in 2-column grid

**Test Actions**:
- ✅ Initial view: Should show 2×2 grid (PC1-PC4) by default
- ✅ Check 6 components → Should see 3×2 grid
- ✅ Check 8 components → Should see 4×2 grid
- ✅ Check 5 components → Should see 3×2 grid (last row has 1 plot)
- ✅ **IMPORTANT**: Check x-axis labels:
  - Top 2 plots (PC1, PC2): NO x-axis labels ✅
  - Middle plots: NO x-axis labels ✅
  - Bottom row: ONLY bottom plots should show "Wavenumber (cm⁻¹)" ✅
- ✅ Colors: Each component should have a distinct color
- ✅ Peak annotations: Top 3 peaks should have dots

---

### 3. Distributions Component Selector

**Steps**:
1. Run PCA analysis
2. Go to "Distributions" tab
3. Look for "📊 Show Components" button in the toolbar
4. Click the button to open the sidebar

**Expected Result**:
- Sidebar slides in from the left (220px width)
- Shows "Select All" checkbox
- Shows list of PC components
- Default: First 3 components checked
- Plot shows PC score distributions with KDE curves + histograms

**Test Actions**:
- ✅ Initial view: Should show 2×2 grid (PC1-PC3, with PC3 in bottom-left)
- ✅ Check 6 components → Should see 3×2 grid with all 6 distributions
- ✅ Uncheck PC1 → Plot should regenerate without PC1
- ✅ Check only PC2 and PC5 → Should show 1×2 grid (2 plots side-by-side)
- ✅ Each subplot should show:
  - Title: "PC# (XX.X%)"
  - X-label: "PC# Score"
  - Y-label: "Density"
  - KDE curves for each dataset (colored lines)
  - Histograms behind (semi-transparent)
  - Legend showing dataset names
- ✅ Colors should match the Score Plot colors

---

### 4. Hierarchical Clustering Legend

**Steps**:
1. Run Hierarchical Clustering analysis with 3 datasets
2. Go to the dendrogram plot

**Expected Result**:
- Dendrogram plot with colored branches
- **Top-left corner**: Text box with legend
  - Title: "Dataset Ranges:"
  - Lines like: "Glucose: samples 0-14 (n=15)"
  - Background: Wheat/tan color with rounded corners

**Test Actions**:
- ✅ Verify sample ranges match dataset sizes
  - Example: If Glucose has 15 spectra, range should be 0-14
  - If Lactose has 12 spectra, range should be 15-26 (starting after Glucose)
- ✅ Check if legend text is readable (not cut off)
- ✅ Verify legend box doesn't overlap important dendrogram parts

**Understanding the Legend**:
- The legend shows **sample index ranges**, not colors
- Dendrogram colors are assigned automatically based on hierarchical distance
- Use the legend to identify **which samples belong to which dataset**
- If samples from the same dataset cluster together → Good separation
- If samples are mixed across clusters → Poor separation

---

### 5. Tab Preservation

**Steps**:
1. Run PCA analysis
2. Switch to "Loading Plot" tab (Tab 3)
3. Click "Run Analysis" again (without changing parameters)

**Expected Result**:
- After analysis completes, you should STILL be on "Loading Plot" tab
- Should NOT jump back to first tab (Score Plot)

---

### 6. Multi-Format Export

**Steps**:
1. Run any analysis (PCA, UMAP, etc.)
2. Look for "Export Data" button in the results panel
3. Click "Export Data"

**Expected Result**:
- Dialog appears asking to select export format:
  - CSV (Comma-separated values)
  - Excel (XLSX)
  - JSON (JavaScript Object Notation)
  - TXT (Tab-separated text)
  - Pickle (Python binary format)
- Select a format → File save dialog appears
- Save file → File should be created with correct format

**Test Actions**:
- ✅ Export to CSV → Open in Excel/Notepad, verify data is correct
- ✅ Export to Excel → Open in Microsoft Excel, verify formatting
- ✅ Export to JSON → Open in text editor, verify valid JSON
- ✅ Try all formats → All should work without errors

---

## Common Issues to Watch For

### Issue 1: Sidebar Not Appearing
**Symptom**: Click toggle button, but sidebar doesn't show
**Check**: Look for console errors, verify button is checkable

### Issue 2: Plot Doesn't Update
**Symptom**: Toggle checkboxes, but plot stays the same
**Check**: Look for errors in console like "KeyError" or "AttributeError"

### Issue 3: X-Axis Labels on Top Plots
**Symptom**: Loading Plot shows "Wavenumber" on all subplots
**Fix**: This should NOT happen after the update. Report if you see it.

### Issue 4: Grid Layout Wrong
**Symptom**: 6 components shown in 1 column instead of 2 columns
**Check**: Verify n_cols calculation in code

### Issue 5: Colors Not Matching
**Symptom**: Distributions use different colors than Score Plot
**Check**: Verify raw_results["colors"] is being used

### Issue 6: KDE Computation Error
**Symptom**: Distributions update fails with "singular matrix" error
**Reason**: Too few data points for KDE, happens with <3 spectra per dataset
**Expected**: Fallback to histogram only (should not crash)

---

## Performance Expectations

### Fast Operations:
- Sidebar toggle (instant)
- Checkbox state changes (instant)
- Plot updates for Spectrum Preview (< 1 second)

### Moderate Operations:
- Loading Plot update with 4 components (< 2 seconds)
- Distributions update with 3 components (< 3 seconds)

### Slow Operations:
- Loading Plot with 8 components (2-5 seconds)
- Distributions with 6 components and large datasets (3-10 seconds)
  - This is expected due to KDE computation

---

## Success Criteria

✅ **All 8 features working**:
1. Spectrum Preview: Multi-select datasets with collapsible sidebar
2. Loading Plot: 2-column grid layout
3. Loading Plot: X-axis labels only on bottom row
4. Loading Plot: Checkbox component selector
5. Distributions: Checkbox component selector
6. Hierarchical Clustering: Legend with dataset ranges
7. Tab Preservation: Stays on current tab after analysis restart
8. Multi-format Export: CSV/Excel/JSON/TXT/Pickle all work

✅ **No errors in console**
✅ **Smooth sidebar animations**
✅ **Dynamic plot updates work correctly**
✅ **Colors consistent across plots**

---

## Quick Keyboard Navigation Tips

### Future Enhancement Ideas (Not Implemented Yet):
- `Ctrl+D` to toggle Display Options sidebar
- `Ctrl+C` to toggle Components sidebar
- `Ctrl+S` to toggle Select All checkboxes
- `Ctrl+E` to open Export dialog

---

## Reporting Issues

If you find any bugs, please note:
1. **What you did** (exact steps)
2. **What you expected** to happen
3. **What actually happened**
4. **Error messages** in console (if any)
5. **Screenshot** if possible

Example bug report:
```
BUG: Loading Plot X-Axis Labels

Steps:
1. Run PCA with 3 datasets, 5 components
2. Go to Loading Plot tab
3. Open sidebar, check all 5 components

Expected:
- Top 2 plots: No x-axis labels
- Middle 2 plots: No x-axis labels  
- Bottom 1 plot: Shows "Wavenumber (cm⁻¹)"

Actual:
- All 5 plots show x-axis labels

Error in console:
None

Screenshot:
[attach screenshot]
```

---

## Quick Test Script

For rapid testing, run this sequence:

1. **Launch app** → `uv run main.py`
2. **Load 3 datasets** (Glucose, Lactose, Protein)
3. **Run PCA** → n_components=5
4. **Test Spectrum Preview**:
   - Toggle sidebar ✅
   - Select 2 datasets ✅
   - Verify plot shows both ✅
5. **Test Loading Plot**:
   - Toggle sidebar ✅
   - Select 6 components ✅
   - Verify 3×2 grid ✅
   - Verify x-axis labels only on bottom ✅
6. **Test Distributions**:
   - Toggle sidebar ✅
   - Select 4 components ✅
   - Verify 2×2 grid with KDE curves ✅
7. **Run Hierarchical Clustering**:
   - Verify legend shows dataset ranges ✅
8. **Test Tab Preservation**:
   - Switch to Loading Plot tab
   - Re-run PCA
   - Verify stays on Loading Plot ✅
9. **Test Export**:
   - Export to CSV ✅
   - Export to Excel ✅

**Total time**: ~5 minutes

---

**Testing Guide Complete**
