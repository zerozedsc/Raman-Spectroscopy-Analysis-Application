═══════════════════════════════════════════════════════════════
🧪 TEST INSTRUCTIONS - Multi-Group Dialog Fix
═══════════════════════════════════════════════════════════════

## ✅ Issue Fixed

**Problem:** `AttributeError: 'PySide6.QtWidgets.QTableWidgetItem' object has no attribute 'setPlaceholderText'`

**Cause:** Incorrect use of `QTableWidgetItem` instead of `QLineEdit` for editable table cells

**Solution:** Changed to `QLineEdit` widgets with `setCellWidget()` method

═══════════════════════════════════════════════════════════════

## 🔄 Testing Steps

### 1️⃣ Restart Application

```powershell
uv run main.py
```

### 2️⃣ Navigate to Classification Mode

1. Go to **Analysis** page (tab on left sidebar)
2. Ensure you're in **Classification Mode** (group tree should be visible)
3. Locate the "➕ Create Group" button in the toolbar

### 3️⃣ Open Multi-Group Dialog

Click "➕ Create Group" button

**Expected Console Output:**
```
[DEBUG] ➕ CREATE GROUP BUTTON CLICKED (GroupTreeManager)
[DEBUG] 🚀 OPENING MULTI-GROUP DIALOG (GroupTreeManager)
[DEBUG] Step 1: Attempting to import MultiGroupCreationDialog...
[DEBUG] ✅ MultiGroupCreationDialog imported successfully!
[DEBUG] Step 2: Creating dialog instance...
[DEBUG] ✅ Dialog created successfully!
[DEBUG] Dialog title: 複数グループを一度に作成
```

**Expected Result:**
✅ Dialog opens successfully (no crash)
✅ Dialog shows table with 4 columns: Group Name | Include | Exclude | Auto-Assign
✅ One empty row is pre-populated
✅ Placeholder text visible in first three columns

### 4️⃣ Test Placeholder Text Visibility

**What to Check:**
- Column 1 (Group Name): Should show "e.g., Control, Disease, MM, MGUS"
- Column 2 (Include Keywords): Should show "e.g., ctrl, control, con"
- Column 3 (Exclude Keywords): Should show "e.g., treatment, test"

**If placeholders NOT visible:** Click in each cell - they might appear on focus

### 5️⃣ Test Input and Focus Styling

1. Click in **Group Name** field → Should highlight with **yellow** background
2. Type "Control" → Yellow highlight remains while typing
3. Click in **Include Keywords** → Should highlight with **green** background
4. Type "ctrl, control" → Green highlight remains
5. Click in **Exclude Keywords** → Should highlight with **red** background
6. Type "treatment" → Red highlight remains

### 6️⃣ Test Add Row Functionality

1. Click "➕ Add Row" button
2. **Expected:** New empty row appears below
3. **Expected:** All placeholder texts visible in new row
4. **Expected:** Auto-Assign checkbox is checked by default

### 7️⃣ Test Remove Row Functionality

1. Click anywhere in the first row to select it
2. Click "➖ Remove Row" button
3. **Expected:** Selected row is removed

### 8️⃣ Test Preview Functionality

1. Enter test group configuration:
   - **Row 1:**
     - Group Name: `Control`
     - Include: `ctrl, control`
     - Exclude: (leave empty)
     - Auto-Assign: ✅ checked
   
   - **Row 2:**
     - Group Name: `Disease`
     - Include: `disease, dis`
     - Exclude: (leave empty)
     - Auto-Assign: ✅ checked

2. Click "👁️ Preview" button

3. **Expected:**
   - Right panel shows tree view with groups
   - Each group shows count of assigned datasets
   - "Unassigned" group shows remaining datasets
   - Status label shows: "⚠️ X dataset(s) unassigned. Y/244 assigned to Z group(s)."

### 9️⃣ Test Apply Functionality

1. With the preview looking correct, click "✅ Apply" button

2. **Expected Console Output:**
```
[DEBUG] ✅ Dialog was ACCEPTED
[DEBUG] Assignments: {'Control': [...], 'Disease': [...]}
```

3. **Expected UI Changes:**
   - Dialog closes
   - Success message box appears: "Successfully created X group(s) with Y dataset(s) assigned."
   - Groups appear in tree view as folders (🧪 icon)
   - Datasets move from "Unassigned" to respective group folders
   - Group tree expands to show all assignments

### 🔟 Test Cancel Functionality

1. Open dialog again
2. Enter some data
3. Click "❌ Cancel" button

**Expected:**
- Dialog closes without applying changes
- No groups created
- Console shows: `[DEBUG] ❌ Dialog was CANCELLED/REJECTED`

═══════════════════════════════════════════════════════════════

## ❌ What to Report if Issues Occur

### If Dialog Still Crashes:
1. Copy **FULL console output** from startup through button click
2. Copy the **complete error traceback**
3. Note at which step the crash occurred

### If Placeholder Text Not Working:
1. Send screenshot of the empty dialog
2. Try clicking in each field and report if they appear on focus
3. Check console for any warnings

### If Preview/Apply Not Working:
1. Copy console output showing what assignments were detected
2. Send screenshot of preview panel
3. Describe expected vs actual behavior

### If Groups Don't Appear in Tree:
1. Check if success message appeared
2. Look for any console errors after clicking Apply
3. Send screenshot of tree view after applying

═══════════════════════════════════════════════════════════════

## 🎯 Success Criteria

✅ Dialog opens without AttributeError
✅ All placeholder texts visible
✅ Focus styling works (yellow/green/red highlights)
✅ Can type in all fields
✅ Add/Remove row buttons work
✅ Preview shows correct group assignments
✅ Apply creates groups in tree structure
✅ Datasets correctly assigned to groups
✅ Cancel works without errors

═══════════════════════════════════════════════════════════════

## 📋 Quick Test Script

Copy-paste this into each row for quick validation:

**Row 1:**
```
Name: MM
Include: mm, mgus
Exclude: control
```

**Row 2:**
```
Name: Control
Include: ctrl, control, con
Exclude: disease, treatment
```

Click Preview → Should show clear separation of MM and Control groups

═══════════════════════════════════════════════════════════════
