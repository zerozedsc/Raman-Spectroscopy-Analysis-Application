# Build System Enhancements

**Date**: October 22, 2025  
**Status**: Implemented  
**Version**: 2.0

## Overview

This document describes the enhanced build system for the Raman Spectroscopy Analysis Application, including automated configuration generation, console/debug modes, and logging controls.

---

## 🆕 New Features

### 1. Automated Spec/NSI Generator

**File**: `build_scripts/generate_build_configs.py`

A Python script that automatically generates optimized PyInstaller spec files and NSIS installer configuration based on your current system environment.

**Features**:
- Auto-detects installed Python packages
- Creates optimal hidden imports list
- Generates platform-specific configurations
- Produces comprehensive environment report
- Supports console mode for debugging

**Usage**:
```powershell
# Generate all config files
python build_scripts/generate_build_configs.py

# Generate with console mode enabled
python build_scripts/generate_build_configs.py --console

# Analyze environment only (no file generation)
python build_scripts/generate_build_configs.py --analyze-only

# Custom output directory
python build_scripts/generate_build_configs.py --output-dir build_scripts
```

**Generated Files**:
- `raman_app.spec` - Portable executable configuration
- `raman_app_installer.spec` - Installer staging configuration
- `raman_app_installer.nsi` - NSIS installer script
- `build_config_report.json` - Environment analysis report

**Benefits**:
- ✅ No manual spec file editing required
- ✅ Optimal configuration for current system
- ✅ Automatic package discovery
- ✅ Platform-specific optimizations
- ✅ Detailed environment diagnostics

---

### 2. Console Mode for Debugging

Both `build_portable.ps1` and `build_installer.ps1` now support console mode.

**Usage**:
```powershell
# Build with console window visible (for debugging)
.\build_portable.ps1 -Console

# Build installer with console
.\build_installer.ps1 -Console
```

**What it does**:
- Temporarily modifies spec file to set `console=True`
- Creates executable with visible console window
- Displays real-time logging output
- Shows Python errors and tracebacks
- Cleans up temporary spec file after build

**When to use**:
- ✅ Debugging application crashes
- ✅ Investigating slow startup times
- ✅ Identifying missing imports
- ✅ Checking asset loading issues
- ✅ Development and testing

**When NOT to use**:
- ❌ Production builds for end users
- ❌ Final distribution packages
- ❌ When console output is not needed

---

### 3. Logging Level Control

Control application logging verbosity at build time and runtime.

**Build Script Parameter**:
```powershell
# Build with specific log level
.\build_portable.ps1 -LogLevel DEBUG    # All logs
.\build_portable.ps1 -LogLevel INFO     # Info and above
.\build_portable.ps1 -LogLevel WARNING  # Warnings and errors (default)
.\build_portable.ps1 -LogLevel ERROR    # Errors only
```

**Runtime Environment Variable**:
```powershell
# Set before running executable
$env:RAMAN_LOG_LEVEL = "DEBUG"
.\dist\raman_app\raman_app.exe

# Or in one line
$env:RAMAN_LOG_LEVEL = "INFO"; .\dist\raman_app\raman_app.exe
```

**Log Level Hierarchy**:
- **DEBUG**: All messages (very verbose)
- **INFO**: General information + warnings + errors
- **WARNING**: Warnings + errors (default for production)
- **ERROR**: Errors only (minimal logging)

**Behavior**:
- Console output respects log level
- Log files always contain all levels
- Default is WARNING for performance
- Set to DEBUG for troubleshooting

---

## 📊 Performance Optimizations

### Startup Time Analysis

The slow startup time (45-70s) is likely caused by:

1. **Module Imports**
   - Large scientific libraries (numpy, scipy, pandas)
   - Machine learning frameworks (torch, sklearn)
   - Spectroscopy packages (ramanspy, pybaselines)

2. **Asset Loading**
   - Font registration
   - Icon loading
   - Localization files
   - Configuration files

3. **Initialization**
   - GUI framework setup
   - Database connections
   - Hardware driver checks

### Optimization Strategies

#### 1. Lazy Imports (Recommended)

```python
# BEFORE (main.py)
import torch
import sklearn
import ramanspy

# AFTER (lazy loading)
def load_ml_features():
    """Load ML libraries only when needed."""
    global torch, sklearn
    if 'torch' not in globals():
        import torch
        import sklearn
        import ramanspy
```

#### 2. Splash Screen

Show user feedback during startup:

```python
from PySide6.QtWidgets import QSplashScreen
from PySide6.QtGui import QPixmap

# In main.py __main__ block
splash_pix = QPixmap("assets/splash.png")
splash = QSplashScreen(splash_pix)
splash.show()
splash.showMessage("Loading...")

app.processEvents()

# ... initialization ...

splash.finish(window)
```

#### 3. Configuration Caching

```python
# Cache compiled assets
if not os.path.exists('.cache/app_config.pkl'):
    config = load_and_process_config()
    pickle.dump(config, open('.cache/app_config.pkl', 'wb'))
else:
    config = pickle.load(open('.cache/app_config.pkl', 'rb'))
```

#### 4. Reduce Logging During Startup

Use WARNING level for production:

```python
# Set in main.py before imports
os.environ['RAMAN_LOG_LEVEL'] = 'WARNING'
```

---

## 🐛 Troubleshooting

### Issue 1: Slow Startup Time

**Symptoms**:
- Application takes 45-70 seconds to open
- Console shows many import messages
- High CPU usage during startup

**Diagnosis**:
```powershell
# Build with console and INFO logging
.\build_portable.ps1 -Console -LogLevel INFO

# Run and observe startup
.\dist\raman_app\raman_app.exe
```

**Solutions**:
1. Use WARNING log level for production
2. Implement lazy imports for ML libraries
3. Add splash screen for user feedback
4. Cache configuration files

### Issue 2: Missing Metadata Components

**Symptoms**:
- Metadata section incomplete in data package page
- Components missing compared to `uv run main.py`

**Diagnosis**:
```powershell
# Build with debug mode
.\build_portable.ps1 -Console -LogLevel DEBUG

# Check logs for asset loading errors
Get-Content logs\*.log | Select-String "metadata"
```

**Potential Causes**:
- Asset paths incorrect in frozen app
- JSON files not bundled correctly
- Relative path issues

**Solution**:
```python
# In data_package_page.py
import sys
import os

def get_asset_path(relative_path):
    """Get absolute path to asset, works for dev and frozen app."""
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        base_path = sys._MEIPASS
    else:
        # Running in normal Python
        base_path = os.path.dirname(os.path.dirname(__file__))
    
    return os.path.join(base_path, relative_path)

# Use:
metadata_path = get_asset_path('assets/data/metadata.json')
```

### Issue 3: Window Title Not English

**Symptoms**:
- Window title shows Japanese text instead of English
- Localization not working correctly

**Diagnosis**:
```powershell
# Check localization files bundled
python test_build_executable.py --exe dist/raman_app/raman_app.exe

# Verify assets
dir dist\raman_app\assets\locales
```

**Solution**:
```python
# In main.py, ensure locale loading before window creation
from configs.configs import LocalizationManager

# Force English locale for testing
LocalizationManager().set_language('en')

# Then create window
window = MainWindow()
```

---

## 📋 Build Workflow

### Development Build (with debugging)

```powershell
# 1. Generate configs (optional, if environment changed)
python build_scripts/generate_build_configs.py --console

# 2. Clean build with console and debug logs
.\build_scripts\build_portable.ps1 -Clean -Console -LogLevel DEBUG

# 3. Test
.\dist\raman_app\raman_app.exe

# 4. Review logs
Get-Content logs\*.log | Select-String -Pattern "ERROR|WARNING"
```

### Production Build (optimized)

```powershell
# 1. Generate production configs
python build_scripts/generate_build_configs.py

# 2. Clean build with minimal logging
.\build_scripts\build_portable.ps1 -Clean -LogLevel WARNING

# 3. Test
python test_build_executable.py

# 4. Run
.\dist\raman_app\raman_app.exe
```

### Installer Build

```powershell
# 1. Build staging with production settings
.\build_scripts\build_installer.ps1 -Clean -LogLevel WARNING

# 2. Test staging
python test_build_executable.py --exe dist_installer/raman_app_installer_staging/raman_app.exe

# 3. Create installer (if NSIS available)
# Will automatically run if NSIS detected
```

---

## 🔍 Testing

### Test Logging Levels

```powershell
# Test DEBUG level
$env:RAMAN_LOG_LEVEL = "DEBUG"; .\dist\raman_app\raman_app.exe
# Expect: Verbose output, all messages

# Test INFO level
$env:RAMAN_LOG_LEVEL = "INFO"; .\dist\raman_app\raman_app.exe
# Expect: Info + warnings + errors

# Test WARNING level (default)
$env:RAMAN_LOG_LEVEL = "WARNING"; .\dist\raman_app\raman_app.exe
# Expect: Warnings + errors only

# Test ERROR level
$env:RAMAN_LOG_LEVEL = "ERROR"; .\dist\raman_app\raman_app.exe
# Expect: Errors only
```

### Test Console Mode

```powershell
# Build with console
.\build_portable.ps1 -Console

# Run (should show console window)
.\dist\raman_app\raman_app.exe

# Verify console output visible
# Verify Python errors shown
# Verify logging messages displayed
```

### Automated Testing

```powershell
# Full test suite
python test_build_executable.py --verbose

# Expected results:
# ✓ Executable structure validated
# ✓ Required directories present
# ✓ Asset files included
# ✓ Binaries bundled
# ✓ Launch successful
# ✓ No errors in logs
```

---

## 📝 Best Practices

### For Development

1. **Use Console Mode**: Always build with `-Console` during development
2. **Enable Debug Logs**: Use `-LogLevel DEBUG` for troubleshooting
3. **Test Frequently**: Test after each change
4. **Check Logs**: Review log files for warnings/errors

### For Production

1. **Disable Console**: Build without `-Console` flag
2. **Minimize Logging**: Use `-LogLevel WARNING` or `-LogLevel ERROR`
3. **Test Thoroughly**: Run full test suite
4. **Clean Build**: Always use `-Clean` flag for final builds

### For Distribution

1. **Generate Fresh Configs**: Run `generate_build_configs.py`
2. **Use Production Settings**: WARNING log level, no console
3. **Test on Clean System**: Test on computer without dev tools
4. **Create Installer**: Use NSIS for professional installation

---

## 🆕 Changelog

### Version 2.0 (October 22, 2025)

**Added**:
- Automated spec/NSI generator script
- Console mode for debugging
- Logging level control system
- Environment analysis and reporting

**Improved**:
- Build scripts with new parameters
- Logging function with environment variable support
- Documentation with troubleshooting guide

**Fixed**:
- Manual spec file editing required
- No way to debug frozen application
- Excessive logging in production builds
- Unclear system requirements

---

## 📚 Related Documentation

- [PyInstaller Guide](./PYINSTALLER_GUIDE.md) - Complete build guide
- [Troubleshooting](./TROUBLESHOOTING.md) - Common issues and solutions
- [Base Memory](../../.AGI-BANKS/BASE_MEMORY.md) - Quick reference
- [Recent Changes](../../.AGI-BANKS/RECENT_CHANGES.md) - Latest updates

---

**Last Updated**: October 22, 2025  
**Author**: AI Assistant  
**Version**: 2.0
