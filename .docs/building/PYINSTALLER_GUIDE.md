# PyInstaller Build Guide for Raman Spectroscopy Application

**Last Updated**: October 21, 2025  
**Target OS**: Windows 10/11  
**Python Version**: 3.12+  
**PyInstaller Version**: 6.16.0+

---

## 📋 Quick Start

### Build Portable Executable (Recommended for Testing)
```powershell
# Navigate to project directory
cd path\to\real-time-raman-spectral-classifier-with-gui

# Run build script
.\build_portable.ps1

# Test the executable
.\dist\raman_app\raman_app.exe
```

### Build Installer (Requires NSIS)
```powershell
# Run installer build script
.\build_installer.ps1

# Test staging directory
.\dist_installer\raman_app_installer_staging\raman_app.exe

# (Optional) Create NSIS installer if raman_app_installer.nsi exists
# Output: raman_app_installer.exe
```

### Validate Build
```powershell
# Run comprehensive tests
python test_build_executable.py

# Or with specific executable
python test_build_executable.py --exe dist\raman_app\raman_app.exe --verbose
```

---

## 🔧 Prerequisites

### Required Software
1. **Python 3.12+**
   - Check: `python --version`
   - Download: https://www.python.org

2. **UV Package Manager**
   ```powershell
   pip install uv
   ```

3. **PyInstaller 6.16.0+**
   ```powershell
   pip install pyinstaller
   # Or included in project dependencies:
   uv pip install pyinstaller
   ```

### Optional but Recommended
1. **NSIS (Nullsoft Scriptable Install System)**
   - For creating installer (.exe setup)
   - Download: https://nsis.sourceforge.io/
   - Install path: `C:\Program Files (x86)\NSIS\`

2. **7-Zip** (for compressing distributions)
   - Download: https://www.7-zip.org/

### Dependencies Installation
```powershell
# Option 1: Using UV (Recommended)
uv pip install -e .
uv pip install pyinstaller

# Option 2: Using Pip directly
pip install -e .
pip install pyinstaller
```

---

## 📦 Build Output Structure

### Portable Executable
```
project_root/
├── dist/
│   └── raman_app/                    # Distribution directory
│       ├── raman_app.exe             # Main executable (50-80 MB)
│       ├── _internal/                # PyInstaller runtime
│       ├── PySide6/                  # Qt6 libraries
│       ├── assets/                   # Application assets
│       │   ├── icons/
│       │   ├── fonts/
│       │   ├── locales/
│       │   └── data/
│       ├── drivers/                  # Andor SDK DLLs
│       └── [other libraries and DLLs]
```

### Installer Staging
```
project_root/
└── dist_installer/
    └── raman_app_installer_staging/  # Same as portable
        ├── raman_app.exe
        ├── assets/
        └── [all dependencies]
```

---

## 🏗️ Build Configuration

### Spec File: `raman_app.spec` (Portable)

**Key Configuration Points**:

```python
# 1. Data Files Collection
datas = []
datas += collect_data_files('assets', includes=['icons/*', 'fonts/*', 'locales/*'])
datas += collect_data_files('PySide6')
datas += collect_data_files('functions')

# 2. Hidden Imports (libraries that PyInstaller doesn't automatically detect)
hiddenimports = [
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'numpy',
    'pandas',
    'scipy',
    'matplotlib.backends.backend_qt5agg',
    'ramanspy',
    'pybaselines',
    # ... and more
]

# 3. Binary Files (DLLs)
binaries = [
    (os.path.join(dll_path, 'atmcd32d.dll'), 'drivers'),
    (os.path.join(dll_path, 'atmcd64d.dll'), 'drivers'),
]

# 4. Executable Configuration
exe = EXE(
    ...,
    console=False,  # No console window (GUI only)
    name='raman_app',
    icon=None,      # Add icon path here if needed
)
```

**Modification Tips**:
- Add icon: Set `icon='assets/app_icon.ico'` in EXE()
- Change console behavior: Set `console=True` for debugging
- Exclude modules: Add to `excludedimports` to reduce size
- Custom hooks: Add hook scripts to `hookspath`

### Spec File: `raman_app_installer.spec` (Installer)

Same as portable but outputs to `build_installer/` instead of `dist/`.

---

## 🔨 Build Scripts

### `build_portable.ps1`

**Purpose**: Build standalone portable executable

**Usage**:
```powershell
.\build_portable.ps1                    # Normal build
.\build_portable.ps1 -Clean             # Clean before build
.\build_portable.ps1 -Debug             # Debug mode
.\build_portable.ps1 -OutputDir custom  # Custom output
```

**What It Does**:
1. ✓ Validates environment (Python, PyInstaller)
2. ✓ Checks spec file exists
3. ✓ Optionally cleans previous builds
4. ✓ Validates project files (main.py, assets/)
5. ✓ Runs PyInstaller with spec file
6. ✓ Validates output (executable, size, components)
7. ✓ Generates build report

**Output**:
- `dist/raman_app/` - Portable distribution
- Console output with validation results
- Build time and size metrics

### `build_installer.ps1`

**Purpose**: Build staging files for NSIS installer

**Usage**:
```powershell
.\build_installer.ps1                    # Normal build + NSIS creation
.\build_installer.ps1 -Clean             # Clean before build
.\build_installer.ps1 -BuildOnly         # Skip NSIS creation
.\build_installer.ps1 -Debug             # Debug mode
```

**What It Does**:
1. ✓ Validates PyInstaller and NSIS availability
2. ✓ Builds executable staging files
3. ✓ (Optional) Compiles NSIS installer if script exists
4. ✓ Validates output

**Output**:
- `dist_installer/raman_app_installer_staging/` - Staging directory
- `raman_app_installer.exe` - Installer (if NSIS script exists)
- Console output with validation results

---

## 🧪 Testing Builds

### Automated Test Suite

**Run all tests**:
```powershell
python test_build_executable.py
```

**Run with specific executable**:
```powershell
python test_build_executable.py --exe dist\raman_app\raman_app.exe
```

**Verbose mode**:
```powershell
python test_build_executable.py --verbose
```

### Test Coverage

The `test_build_executable.py` suite validates:

1. **Executable Structure**
   - Executable exists and is valid
   - File size is non-zero
   - Path is correct

2. **Required Directories**
   - `assets/` - Application resources
   - `PySide6/` - Qt6 libraries
   - `_internal/` - PyInstaller runtime

3. **Required Assets**
   - `assets/icons/` - UI icons
   - `assets/fonts/` - Custom fonts
   - `assets/locales/` - Translations
   - `assets/data/` - Data files

4. **Binary Files**
   - DLL count verification
   - Andor SDK DLLs (atmcd32d.dll, atmcd64d.dll)
   - Critical library DLLs

5. **Executable Launch**
   - Launch test with `--help` flag
   - Timeout handling (5 seconds)
   - Return code validation

6. **Performance**
   - Distribution size
   - File access performance
   - Warning for large distributions

### Manual Testing

**Basic functionality test**:
```powershell
# Start the application
.\dist\raman_app\raman_app.exe

# Test:
# 1. UI loads without errors
# 2. Can create new project
# 3. Can load data files
# 4. Can run preprocessing
# 5. Can export results
# 6. Application closes cleanly
```

**Command-line testing**:
```powershell
# Launch with debug output
.\dist\raman_app\raman_app.exe 2>error.log

# Check for errors
type error.log
```

---

## 📊 Build Statistics

### Typical Build Sizes

| Build Type | Size | Time | Notes |
|-----------|------|------|-------|
| Portable Executable | 50-80 MB | 2-5 min | Including all dependencies |
| Installed (via NSIS) | 100-150 MB | 5-10 min | With installer overhead |
| Total Distribution | 60-100 MB | - | Uncompressed |
| Compressed (.7z) | 20-30 MB | - | 7-Zip compression |

### Dependency Sizes (Approximate)

| Component | Size |
|-----------|------|
| PySide6 + Qt6 | 200-250 MB (extracted) |
| NumPy/SciPy/Pandas | 100-150 MB |
| RamanSPy/PyBaselines | 30-50 MB |
| Matplotlib | 30-40 MB |
| PyTorch (if included) | 500+ MB |
| **Total Dependencies** | **1-2 GB** (extracted) |

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "PyInstaller not found"
```powershell
# Install PyInstaller
pip install pyinstaller>=6.16.0

# Or using UV
uv pip install pyinstaller
```

#### 2. "Module not found" errors at runtime
```
Solution: Add to hiddenimports in spec file
Example: If numpy fails to load
  - Add 'numpy' to hiddenimports list
  - Rebuild with: pyinstaller raman_app.spec
```

#### 3. "raman_app.exe has stopped working"
```
Possible causes:
  1. Missing dependency - check test_build_executable.py output
  2. Missing asset file - verify assets/ directory copied
  3. Library version mismatch - check pyproject.toml versions
  
Solution:
  1. Add missing module to hiddenimports
  2. Use -Clean flag: .\build_portable.ps1 -Clean
  3. Rebuild entire distribution
```

#### 4. Large executable size (>100 MB)
```
Optimization tips:
  1. Exclude tkinter: Add to excludedimports
  2. Exclude development tools: pytest, sphinx, etc.
  3. Use UPX compression: Enable in spec file
  4. Remove unnecessary modules from hiddenimports
  5. Compress final .exe with 7-Zip
```

#### 5. NSIS installer creation fails
```powershell
# Check NSIS installation
"C:\Program Files (x86)\NSIS\makensis.exe" --version

# If not found:
1. Download NSIS: https://nsis.sourceforge.io/
2. Install to C:\Program Files (x86)\NSIS\
3. Create raman_app_installer.nsi script
4. Re-run: .\build_installer.ps1
```

#### 6. Assets not loading in executable
```
Causes:
  - Assets not included in datas list
  - Icon paths hardcoded instead of relative
  
Solution:
  1. Add to spec file datas:
     datas += collect_data_files('assets')
  2. Use Path() for relative paths:
     icon_path = Path(__file__).parent / 'assets' / 'icon.svg'
  3. Rebuild
```

#### 7. Antivirus flagging executable as malware
```
Common with PyInstaller apps (false positive)
Solutions:
  1. Sign executable with certificate
  2. Submit to VirusTotal for analysis: virustotal.com
  3. Report to antivirus vendor
  4. Build with --noupx flag (slower, less compression)
```

### Debug Mode

Build in debug mode for detailed error information:

```powershell
# Portable build with debug
.\build_portable.ps1 -Debug

# Installer build with debug
.\build_installer.ps1 -Debug

# Test with verbose output
python test_build_executable.py --verbose
```

---

## 🚀 Deployment

### Distribute Portable Version

**For users**:
1. Create release package:
   ```powershell
   # Compress distribution
   7z a raman_app_v1.0.0.7z .\dist\raman_app\*
   
   # Or ZIP
   Compress-Archive -Path .\dist\raman_app -DestinationPath raman_app_v1.0.0.zip
   ```

2. Upload to release hosting (GitHub, etc.)

3. Users extract and run `raman_app.exe`

### Distribute Installer Version

1. Create NSIS script (`raman_app_installer.nsi`)
2. Build with: `.\build_installer.ps1`
3. Output: `raman_app_installer.exe`
4. Users run installer, application installed to `C:\Program Files\RamanApp\`

### Version Management

Update in `pyproject.toml`:
```toml
[project]
version = "1.0.0"  # Change version here
```

Build includes version in output filenames (optional).

---

## 📝 Build Checklist

Before releasing a build:

- [ ] All tests pass: `python test_build_executable.py`
- [ ] Application launches without errors
- [ ] Can load and preview sample data
- [ ] All preprocessing methods available
- [ ] Export functionality works
- [ ] UI renders correctly (fonts, icons, colors)
- [ ] No console errors in log files
- [ ] Help and about dialogs display correctly
- [ ] File dialogs work (open, save)
- [ ] Language switching works (EN/JP)
- [ ] No temporary files left after exit
- [ ] File size is reasonable (<200 MB)

---

## 🔗 Additional Resources

- **PyInstaller Documentation**: https://pyinstaller.org/
- **PySide6 Guide**: https://doc.qt.io/qtforpython/
- **NSIS Documentation**: https://nsis.sourceforge.io/
- **Windows Signing**: https://docs.microsoft.com/en-us/windows/win32/seccrypto/

---

## 📞 Support

For build issues, check:
1. This guide's troubleshooting section
2. PyInstaller issues: https://github.com/pyinstaller/pyinstaller/issues
3. Project issues: GitHub repository
4. Build logs in `test_script/build_test_results_*.json`

Generated: October 21, 2025
