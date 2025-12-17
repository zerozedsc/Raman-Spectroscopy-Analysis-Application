# Quick Action Guide - What to Do Next
**Date**: October 16, 2025  
**Status**: Fixes Complete - Ready for Testing

---

## ⚠️ CRITICAL FIRST STEP

### 🔄 RESTART THE APPLICATION

**Why**: Localization keys were added to JSON files, but application caches them at startup.

**How**:
1. Close the application completely
2. Restart the application
3. Load your project
4. Open preprocess page

**After restart**:
- ✅ Pipeline dialogs will show proper text (no "DIALOGS" placeholder)
- ✅ All localization warnings will disappear from logs

---

## 🧪 Quick Tests (5 minutes)

### Test 1: Localization ✅
1. Open preprocess page
2. Click "Export Pipeline" button
3. **Check**: Dialog title says "Export Preprocessing Pipeline"
4. **Check**: Field labels show proper text (not "DIALOGS")
5. Close dialog
6. Click "Import Pipeline" button
7. **Check**: Dialog title says "Import Preprocessing Pipeline"

**If wrong**: Keys still missing → Report to me

---

### Test 2: Preview Toggle - First Load ✅
1. **Close application**
2. **Restart application**
3. Load project
4. Open preprocess page
5. **Check first dataset type**:
   - If **raw** → Preview should be **ON** (eye open icon)
   - If **preprocessed** → Preview should be **OFF** (eye closed icon)

**If wrong**: Screenshot and report scenario

---

### Test 3: Preview Toggle - Dataset Selection ✅
1. Click a **raw dataset**
2. **Check**: Preview switches to **ON**
3. Click a **preprocessed dataset**
4. **Check**: Preview switches to **OFF**
5. Click another **raw dataset**
6. **Check**: Preview switches to **ON**

**If wrong**: Report which transition failed

---

### Test 4: Preview Toggle - Tab Switching ✅
1. Go to **Raw tab**
2. **Check**: Preview is **ON**
3. Go to **Preprocessed tab**
4. **Check**: Preview switches to **OFF**
5. Go to **All tab**
6. **Check**: Preview switches to **ON**

**If wrong**: Report which tab transition failed

---

## 📊 Quick Results Form

**Tester**: _______________  
**Date/Time**: _______________

### Results
- [ ] ✅ Localization working (dialogs show proper text)
- [ ] ✅ Preview ON for raw datasets
- [ ] ✅ Preview OFF for preprocessed datasets
- [ ] ✅ Preview adjusts on dataset selection
- [ ] ✅ Preview adjusts on tab switching
- [ ] ✅ Works correctly on first load

### Issues Found
1. ______________________________
2. ______________________________
3. ______________________________

---

## 🐛 Bug Report Template (If Needed)

```
**Problem**: [What's wrong]

**Steps to Reproduce**:
1. Step one
2. Step two
3. See problem

**What I Expected**: 
[Describe correct behavior]

**What Actually Happened**: 
[Describe wrong behavior]

**Context**:
- Dataset name: ____________
- Dataset type: Raw / Preprocessed
- Tab: All / Raw / Preprocessed  
- First load: Yes / No
- Preview state before: ON / OFF
- Preview state after: ON / OFF

**Screenshot**: [Attach if visual issue]
```

---

## 📁 Where to Find Documentation

### Quick Reference
- **This guide**: `.docs/testing/QUICK_ACTION_GUIDE.md`
- **Fix summary**: `.docs/summaries/2025-10-16_FIXES_SUMMARY.md`

### Full Documentation
- **All changes**: `.AGI-BANKS/RECENT_CHANGES.md`
- **Code patterns**: `.AGI-BANKS/IMPLEMENTATION_PATTERNS.md`
- **Full testing**: `.docs/testing/2025-10-16_PREPROCESS_PAGE_UI_ENHANCEMENTS_TESTING.md`

---

## ✅ Success Criteria

Everything works if:
1. ✅ No localization warnings in console/logs
2. ✅ Dialog text displays correctly (no placeholders)
3. ✅ Preview is ON when viewing raw datasets
4. ✅ Preview is OFF when viewing preprocessed datasets
5. ✅ Preview auto-adjusts when switching datasets or tabs
6. ✅ No visual glitches or errors

---

## 🎯 What Was Fixed

### Fix #1: Localization Keys
- **Status**: Keys exist in JSON files
- **Action**: Restart application to reload
- **Files**: No code changes needed

### Fix #2: Preview Toggle Logic
- **Status**: Code updated to detect dataset type
- **Action**: Test all scenarios
- **Files**: `pages/preprocess_page.py` (lines 707-798)

### Fix #3: Documentation
- **Status**: All docs updated
- **Files**: RECENT_CHANGES.md, IMPLEMENTATION_PATTERNS.md, summaries

---

## 💬 Communication

### If All Tests Pass
Report: "✅ All tests passed! Everything working correctly."

### If Issues Found
Use bug report template above with:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Context (dataset type, tab, etc.)
- Screenshot if visual issue

---

**Estimated Time**: 5-10 minutes for all quick tests  
**Status**: Ready for your validation! 🚀
