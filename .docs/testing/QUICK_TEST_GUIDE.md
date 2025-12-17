# Quick Testing Guide - Preprocess Page UI Enhancements
**For detailed instructions, see**: `.docs/testing/2025-10-16_PREPROCESS_PAGE_UI_ENHANCEMENTS_TESTING.md`

---

## ⚡ Quick Start (5-Minute Test)

### 1. Select All Button ✅
**Location**: Preprocess Page → Input Datasets → Title Bar (checkmark icon)

**Quick Test**:
1. ✅ Click checkmark button → all datasets in current tab selected
2. ✅ Click again → all deselected
3. ✅ Switch to different tab → button only affects that tab
4. ✅ Hover button → see tooltip

**Report if**: Button missing, doesn't work, or affects wrong tab

---

### 2. Preview Toggle ✅
**Location**: Preprocess Page → Visualization → Eye icon button

**Quick Test**:
1. ✅ Start on Raw tab → preview should be ON (eye open)
2. ✅ Switch to Preprocessed tab → preview auto-switches to OFF (eye closed)
3. ✅ Switch back to Raw → preview auto-switches to ON
4. ✅ Check "All" tab → preview should be ON

**Report if**: Preview doesn't change automatically, wrong default state, or stays same

---

### 3. Pipeline Dialogs ✅
**Location**: Preprocess Page → Pipeline Building Section

**Quick Test Export Dialog**:
1. ✅ Create simple pipeline (add 1-2 steps)
2. ✅ Click export button
3. ✅ Check dialog has clean white background, bordered input fields
4. ✅ Enter name "Test" and click Export
5. ✅ See success notification

**Quick Test Import Dialog**:
1. ✅ Click import button
2. ✅ Check dialog has clean white background, styled list
3. ✅ Click on pipeline in list → should highlight with blue left border
4. ✅ Hover over items → should show light gray background
5. ✅ Click Import → pipeline loads

**Report if**: Dialogs look broken, unstyled, or have layout issues

---

## 🐛 How to Report Bugs

**Use this format**:
```
Bug: [Short description]
Steps: 1. Do this 2. Then this 3. See error
Expected: [What should happen]
Actual: [What happened instead]
Screenshot: [Attach if visual issue]
```

**Examples**:
```
Bug: Select all doesn't work in Preprocessed tab
Steps: 1. Switch to Preprocessed tab 2. Click checkmark button
Expected: All preprocessed datasets selected
Actual: Nothing happens
```

```
Bug: Preview stays ON in Preprocessed tab
Steps: 1. Switch to Preprocessed tab 2. Check preview toggle
Expected: Preview OFF (eye closed icon)
Actual: Preview still ON (eye open icon)
```

---

## ✅ What to Check

### Visual Checks
- [ ] Select all button visible and properly sized
- [ ] Preview toggle icon changes (eye open ↔ eye closed)
- [ ] Export dialog has white background and styled inputs
- [ ] Import dialog list has hover and selection effects
- [ ] All buttons have hover effects

### Functional Checks
- [ ] Select all works in all 3 tabs (All, Raw, Preprocessed)
- [ ] Select all toggles correctly (select → deselect → select)
- [ ] Preview toggle changes automatically when switching tabs
- [ ] Export saves pipeline successfully
- [ ] Import loads pipeline successfully

### Edge Cases
- [ ] Select all with empty list (should do nothing, no error)
- [ ] Select all with 1 item (should toggle correctly)
- [ ] Rapid tab switching (should be smooth)
- [ ] Export with empty name (should show warning)

---

## 📊 Quick Report Template

**Tester**: _____________  
**Date**: _____________  
**Version**: _____________

### Feature Status
- [ ] ✅ Select All Button - Working
- [ ] ✅ Preview Toggle - Working
- [ ] ✅ Pipeline Dialogs - Working

### Issues Found
1. ________________________
2. ________________________
3. ________________________

### Overall Assessment
- [ ] Ready to use
- [ ] Minor issues (usable but needs fixes)
- [ ] Major issues (needs work)

---

## 📸 Screenshots Needed

If reporting issues, please capture:
1. **Select all button** - Show in title bar
2. **Preview toggle** - Show ON state and OFF state
3. **Export dialog** - Full dialog view
4. **Import dialog** - Full dialog with list
5. **Any bugs** - Screenshot of the problem

---

## 💡 Tips

- **Test in order**: Select All → Preview Toggle → Dialogs
- **Take notes**: Write down anything unexpected
- **Screenshot everything**: Easier to diagnose with images
- **Check console**: Look for error messages in logs
- **Test both languages**: If you have EN/JA, test both

---

**Full Testing Guide**: `.docs/testing/2025-10-16_PREPROCESS_PAGE_UI_ENHANCEMENTS_TESTING.md`  
**Implementation Details**: `.docs/summaries/2025-10-16_PREPROCESS_PAGE_IMPLEMENTATION_SUMMARY.md`

**Estimated Time**: 5-10 minutes for quick test, 45-60 minutes for full suite
