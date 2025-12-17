# CRITICAL: LocalizationManager Initialization Pattern

**DATE ESTABLISHED**: November 19, 2025  
**SEVERITY**: HIGH - Affects all localization functionality  
**PATTERN TYPE**: Single Source of Truth, Global State Management  

---

## 🚨 THE RULE

### NEVER create LocalizationManager instances except in utils.py

```python
# ❌ FORBIDDEN in ALL files except utils.py
from configs.configs import LocalizationManager
localize = LocalizationManager()

# ✅ CORRECT - use global function
from utils import LOCALIZE
text = LOCALIZE("KEY.subkey")
```

---

## 📋 The One True Initialization

### utils.py (Lines 218-224)

This is the **ONLY** place where LocalizationManager should be instantiated:

```python
# Parse command-line args (handled by argparse)
parser = argparse.ArgumentParser(description="...")
parser.add_argument('--lang', type=str, choices=['en', 'ja'], default='ja')
args = parser.parse_args()

# Single initialization with command-line arg
arg_lang = args.lang
CONFIGS = load_config()
LOCALIZEMANAGER = LocalizationManager(default_lang=arg_lang)
PROJECT_MANAGER = ProjectManager()

# Global wrapper function
def LOCALIZE(key, **kwargs):
    return LOCALIZEMANAGER.get(key, **kwargs)
```

**What This Does**:
- Parses `--lang` argument from command line
- Creates **single global** LocalizationManager instance
- Provides convenient `LOCALIZE()` function for all modules
- Respects command-line arg priority

---

## 🎯 Usage Pattern for ALL Other Files

### Import Pattern

```python
# At top of file
from utils import LOCALIZE  # Import function, not class

# Use anywhere in code
class MyPage(QWidget):
    def __init__(self):
        super().__init__()
        self.title = LOCALIZE("PAGE.title")
        self.button_text = LOCALIZE("PAGE.button", count=5)
```

### Widget Pattern

```python
# For widgets that need localize reference
class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.localize = LOCALIZE  # Store reference to function
        
    def update_text(self):
        self.label.setText(self.localize("WIDGET.label"))
```

### Passing to Functions

```python
# When passing to utility functions
from utils import LOCALIZE

def create_button_widget(localize_func):
    button = QPushButton(localize_func("BUTTON.text"))
    return button

# Call it
button = create_button_widget(LOCALIZE)
```

---

## ❌ ANTI-PATTERNS (DO NOT DO)

### 1. Creating New Instances

```python
# ❌ WRONG - creates duplicate initialization
from configs.configs import LocalizationManager
lm = LocalizationManager()  # Second instance!
text = lm.get("KEY")

# ✅ CORRECT
from utils import LOCALIZE
text = LOCALIZE("KEY")
```

### 2. Calling set_language() After Init

```python
# ❌ WRONG - overrides command-line arg
from utils import LOCALIZEMANAGER
language = CONFIGS.get("language", "ja")
LOCALIZEMANAGER.set_language(language)  # Ignores --lang argument!

# ✅ CORRECT
# Don't call set_language() - language already set in utils.py
# Command-line arg has priority
```

### 3. Importing LocalizationManager Class

```python
# ❌ WRONG - you don't need the class
from configs.configs import LocalizationManager

# ✅ CORRECT - import the function
from utils import LOCALIZE
```

---

## 🔍 Why This Pattern Exists

### The Problem We're Solving

**Before Fix** (November 19, 2025):
```
Terminal: uv run main.py --lang en

Logs:
2025-11-19 20:28:32,644 - LocalizationManager - INFO - Successfully loaded language: en
2025-11-19 20:28:34,188 - LocalizationManager - INFO - Successfully loaded language: ja

Result: App used Japanese despite --lang en ❌
```

**Root Cause**:
1. `utils.py` initialized with command-line arg (`en`)
2. `main.py` called `set_language()` with config file (`ja`)
3. Config overrode command-line arg

**After Fix**:
```
Terminal: uv run main.py --lang en

Logs:
2025-11-19 20:39:37,160 - LocalizationManager - INFO - Successfully loaded language: en

Result: App uses English as specified ✅
```

### Design Principles

1. **Single Source of Truth**: Only utils.py initializes
2. **Priority Hierarchy**: CLI > Config > Defaults
3. **Global State**: One instance shared by all modules
4. **Separation of Concerns**: Init in utils, use everywhere else

---

## 🛠️ Files to Check

### Critical Files (Must Follow Pattern)

- ✅ `utils.py` - **ONLY** file that instantiates LocalizationManager
- ✅ `main.py` - Uses LOCALIZE, never calls set_language()
- ✅ `pages/*.py` - All import LOCALIZE from utils
- ✅ `components/*.py` - All import LOCALIZE from utils

### Files Modified in Fix (November 19, 2025)

1. **utils.py** - Simplified arg parsing
2. **main.py** - Removed redundant set_language() call
3. **analysis_page.py** - Removed unused LocalizationManager import

---

## 🧪 Testing Checklist

When modifying localization code, verify:

- [ ] Only one "Successfully loaded language" log appears
- [ ] Command-line `--lang en` uses English
- [ ] Command-line `--lang ja` uses Japanese  
- [ ] No `set_language()` calls after initialization
- [ ] No `LocalizationManager()` instantiations except utils.py
- [ ] All files import `LOCALIZE` from utils, not `LocalizationManager` from configs

### Test Commands

```powershell
# Test English
uv run main.py --lang en
# Should show only: "Successfully loaded language: en"

# Test Japanese
uv run main.py --lang ja
# Should show only: "Successfully loaded language: ja"

# Test default
uv run main.py
# Should show only: "Successfully loaded language: ja" (argparse default)
```

---

## 📚 Code Search Commands

### Find Violations

```powershell
# Search for LocalizationManager instantiation
rg "LocalizationManager\(" --glob "*.py"
# Should only find utils.py line 220

# Search for set_language calls
rg "set_language\(" --glob "*.py"
# Should only find definition in configs.py, not in application code

# Search for LocalizationManager imports
rg "from configs.configs import.*LocalizationManager" --glob "*.py"
# Should only find utils.py (for type hint if needed)
```

---

## 🚀 Quick Reference Card

| Scenario | Correct Pattern | Reason |
|----------|-----------------|---------|
| Need localization | `from utils import LOCALIZE` | Global function |
| Pass to function | `func(LOCALIZE)` | Pass function reference |
| Store reference | `self.localize = LOCALIZE` | Store function |
| Change language | ❌ Don't do this | Set at startup only |
| New instance | ❌ Never do this | Single global instance |
| Import class | ❌ Don't do this | Use LOCALIZE function |

---

## 💡 Key Takeaways

1. **One Instance Rule**: LocalizationManager instantiated once in utils.py
2. **Function, Not Class**: Always import LOCALIZE function, never LocalizationManager class
3. **No set_language()**: Language set at startup from command-line, never changed
4. **Global State**: All modules share the same LocalizationManager instance
5. **Command-Line Priority**: CLI args override config file settings

---

## 📖 Related Documentation

- `.AGI-BANKS/RECENT_CHANGES.md` - Full fix details (November 19, 2025 entry)
- `.docs/fixes/localization_duplicate_init_fix_20251119.md` - Technical analysis
- `.docs/summary/session_20251119_localization_fix.md` - Session summary
- `.AGI-BANKS/BASE_MEMORY.md` - Quick pattern reference

---

**Enforcement**: This pattern is **MANDATORY** for all localization code.  
**Review**: Check all PRs for LocalizationManager instantiation violations.  
**Testing**: Automated tests should verify single initialization.

---

**Pattern Owner**: AI Agent / Core Team  
**Last Verified**: November 19, 2025  
**Status**: ✅ PRODUCTION ENFORCED
