# Build Troubleshooting Guide

**Last Updated**: October 21, 2025  
**Status**: Active Reference

This document provides solutions to common build problems and runtime errors.

---

## Common Build Errors

### Error 1: "argument --distpath: expected one argument"

**Symptom**:
```
pyinstaller: error: argument --distpath: expected one argument
```

**Cause**: PowerShell parameter parsing issue when using `--clean` flag

**Solution**: Use `-Clean` (PowerShell style) instead of `--clean`:
```powershell
# ❌ Wrong:
.\build_portable.ps1 --clean

# ✅ Correct:
.\build_portable.ps1 -Clean
```

**Explanation**: PowerShell switch parameters use `-` not `--`. The `--` style causes the parameter to be interpreted incorrectly.

---

### Error 2: "A positional parameter cannot be found that accepts argument"

**Symptom**:
```
FATAL ERROR: A positional parameter cannot be found that accepts argument 'raman_app.exe'.
at line 184
```

**Cause**: PowerShell cmdlets can't handle paths with special characters (Unicode, spaces, etc.)

**Solution**: Use `-LiteralPath` parameter instead of `-Path`:
```powershell
# ❌ Wrong:
$Size = (Get-Item $ExePath).Length
$DirSize = (Get-ChildItem -Path $DirPath -Recurse | ...)

# ✅ Correct:
$Size = (Get-Item -LiteralPath $ExePath).Length  
$DirSize = (Get-ChildItem -LiteralPath $DirPath -Recurse | ...)
```

**Status**: ✅ Fixed in build_portable.ps1

---

## Common Runtime Errors

### Error 3: scipy.stats "NameError: name 'obj' is not defined"

**Symptom**:
```python
File "scipy\stats\_distn_infrastructure.py", line 369, in <module>
NameError: name 'obj' is not defined
```

**Cause**: PyInstaller doesn't automatically detect all scipy.stats submodules

**Solution**: Add hidden imports to ALL spec files:
```python
hiddenimports += [
    'scipy.stats',
    'scipy.stats._stats_py',
    'scipy.stats.distributions',
    'scipy.stats._distn_infrastructure',
]
```

**Files to Update**:
- `build_scripts/raman_app.spec` ✅
- `build_scripts/raman_app_installer.spec` ✅
- `raman_app.spec` (root) ✅
- `raman_app_installer.spec` (root) ✅

**Critical**: After updating spec files, you MUST rebuild:
```powershell
.\build_portable.ps1 -Clean
```

**Status**: ✅ Fixed in all spec files (October 21, 2025)

---

### Error 4: Missing Hidden Imports (General Pattern)

**Symptoms**:
```
ERROR: Hidden import 'module_name' not found
ModuleNotFoundError: No module named 'xxx'
ImportError: cannot import name 'yyy'
```

**Diagnosis Steps**:

1. **Check if module is actually needed:**
   - Some "ERROR: Hidden import not found" warnings are safe to ignore
   - Only critical if the exe crashes at runtime

2. **Verify module is installed:**
   ```powershell
   python -c "import module_name; print(module_name.__file__)"
   ```

3. **Add to spec file if needed:**
   ```python
   hiddenimports += [
       'module_name',
       'module_name.submodule',
   ]
   ```

**Common False Alarms** (safe to ignore):
- `torch.nn`, `torch.optim` - Only needed if using ML features
- `requests` - Only for online features  
- `cryptography` - Only for secure connections
- `renishawwire` - Specific file format support

---

## Build vs Runtime Issues

### Critical Distinction

| Issue Type | When It Happens | How to Fix |
|------------|-----------------|------------|
| **Build Error** | During `pyinstaller` execution | Fix script/spec, rebuild |
| **Runtime Error** | When running the .exe | Add hidden imports, rebuild |

**Key Point**: Runtime errors mean the build succeeded but the exe is missing dependencies. You must:
1. Update spec file with missing imports
2. **Rebuild completely** (use `-Clean` flag)
3. Test the new exe

---

## Debugging Workflow

### Step 1: Identify Error Type

```
Is error during build? → Build Error → Check scripts
Is error when running exe? → Runtime Error → Check imports
```

### Step 2: Check Spec File Age

**CRITICAL**: Always verify spec file changes are included in build:

```powershell
# Check when spec was last modified
(Get-Item build_scripts\raman_app.spec).LastWriteTime

# Check when dist was created  
(Get-Item dist\raman_app).LastWriteTime

# If spec is NEWER than dist, you MUST rebuild!
```

**Common Mistake**: Updating spec but testing old exe → appears "not fixed"

### Step 3: Clean Rebuild

```powershell
# Always use -Clean after spec changes
.\build_portable.ps1 -Clean

# This forces PyInstaller to re-analyze all imports
```

### Step 4: Test Immediately

```powershell
# Run exe right after build
.\dist\raman_app\raman_app.exe

# Check for import errors in first 10 seconds
```

---

## Verification Checklist

Before reporting "still broken":

- [ ] Spec file has the fix (check with grep/search)
- [ ] Rebuild was done AFTER spec file update
- [ ] Used `-Clean` flag to force full rebuild
- [ ] Tested the NEWLY built exe (not old one)
- [ ] Checked timestamps (spec vs dist)

---

## Hidden Import Reference

### Currently Required (October 2025)

```python
# PySide6 (GUI)
'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets',
'PySide6.QtOpenGL', 'PySide6.QtPrintSupport',

# Data Science
'numpy', 'pandas',

# Scipy (CRITICAL - all required)
'scipy', 'scipy.integrate', 'scipy.signal',
'scipy.interpolate', 'scipy.optimize', 'scipy.special',
'scipy.stats',  # ← Must have these 4 for stats
'scipy.stats._stats_py',
'scipy.stats.distributions',
'scipy.stats._distn_infrastructure',

# Sklearn
'sklearn', 'sklearn.preprocessing',
'sklearn.decomposition', 'sklearn.linear_model', 'sklearn.ensemble',

# Visualization
'matplotlib', 'matplotlib.pyplot',
'matplotlib.backends.backend_qt5agg',
'matplotlib.figure', 'matplotlib.widgets',
'seaborn', 'PIL', 'imageio',

# Raman Analysis
'ramanspy', 'ramanspy.preprocessing',
'ramanspy.preprocessing.normalise',
'ramanspy.preprocessing.baseline',
'pybaselines', 'pybaselines.api', 'pybaselines.optimizers',

# Optional (only if using these features)
'torch', 'torch.nn', 'torch.optim',  # ML/Deep Learning
'onnx', 'skl2onnx',  # Model export
'requests',  # Online features
'cryptography',  # Secure connections
```

---

## PowerShell Best Practices

### Parameter Syntax

```powershell
# ✅ Correct PowerShell syntax:
.\script.ps1 -ParameterName
.\script.ps1 -Clean
.\script.ps1 -Debug

# ❌ Wrong (Bash/Linux style):
.\script.ps1 --parameter-name
.\script.ps1 --clean
```

### Path Handling

```powershell
# ✅ Always use -LiteralPath for user paths:
Get-Item -LiteralPath $path
Get-ChildItem -LiteralPath $path -Recurse
Test-Path -LiteralPath $path

# ❌ Avoid default -Path with special characters:
Get-Item $path  # Fails with Chinese/Unicode
Get-ChildItem -Path $path  # Fails with spaces
```

---

## Quick Reference Commands

### Build Commands
```powershell
# Clean build (recommended after spec changes)
.\build_portable.ps1 -Clean

# Debug build (verbose output)
.\build_portable.ps1 -Debug

# Build installer
.\build_installer.ps1 -Clean
```

### Verification Commands
```powershell
# Check spec file syntax
python -c "exec(open('build_scripts/raman_app.spec').read())"

# List hidden imports in spec
Select-String -Path "build_scripts\raman_app.spec" -Pattern "hiddenimports"

# Check build timestamps
Get-ChildItem dist, build_scripts\raman_app.spec | Select-Object Name, LastWriteTime
```

### Test Commands
```powershell
# Test executable
.\dist\raman_app\raman_app.exe

# Run test suite
uv run build_scripts\test_build_executable.py

# Check imports in exe
python -c "import PyInstaller.utils.hooks; print(PyInstaller.utils.hooks.collect_submodules('scipy.stats'))"
```

---

## Emergency Recovery

If build is completely broken:

```powershell
# 1. Restore from backup
Copy-Item build_backups\backup_LATEST\* . -Recurse -Force

# 2. Or start fresh
Remove-Item build, dist -Recurse -Force
Remove-Item *.spec -Force
Copy-Item build_scripts\raman_app.spec . -Force

# 3. Verify spec integrity
python build_scripts\test_build_executable.py --check-spec

# 4. Clean rebuild
.\build_portable.ps1 -Clean
```

---

## Contact Points

When asking for help, include:

1. **Error Type**: Build or Runtime?
2. **Full Error Message**: Copy complete traceback
3. **Last Command**: Exact command that failed
4. **Timestamps**: When spec was modified vs when exe was built
5. **Verification**: Did you rebuild after spec changes?

**Template**:
```
Error Type: [Build/Runtime]
Command: .\build_portable.ps1 -Clean
Error: [paste full error]
Spec Modified: [timestamp]
Dist Created: [timestamp]
Rebuilt After Fix: [Yes/No]
```

---

**End of Troubleshooting Guide**
