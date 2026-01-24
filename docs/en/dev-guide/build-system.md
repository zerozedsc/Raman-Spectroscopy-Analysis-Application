# Build System

Guide to building, packaging, and deploying the Raman Spectroscopy Analysis Application.

## Table of Contents
- [Development Environment](#development-environment)
- [Dependency Management](#dependency-management)
- [Building the Application](#building-the-application)
- [Platform-Specific Builds](#platform-specific-builds)
- [Creating Installers](#creating-installers)
- [CI/CD Pipeline](#cicd-pipeline)
- [Documentation Building](#documentation-building)
- [Troubleshooting](#troubleshooting)

---

## Development Environment

### UV Package Manager

**UV** is a fast Python package installer and resolver written in Rust.

#### Installation

**macOS/Linux**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**:
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Verify**:
```bash
uv --version
```

#### Basic Commands

```bash
# Create virtual environment
uv venv

# Install package
uv pip install package-name

# Install from requirements
uv pip install -r requirements.txt

# Install in editable mode
uv pip install -e .

# Sync with lock file (if using uv.lock)
uv sync
```

### Project Configuration

#### pyproject.toml

**File**: `pyproject.toml`

```toml
[project]
name = "raman-app"
version = "1.0.0"
description = "Real-time Raman Spectral Classifier with GUI"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"

dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "matplotlib>=3.7.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "PyQt6>=6.5.0",
    "openpyxl>=3.1.0",
]

[project.optional-dependencies]
ml = [
    "xgboost>=2.0.0",
    "umap-learn>=0.5.3",
]
deep-learning = [
    "torch>=2.0.0",
]
dev = [
    "pytest>=7.4.0",
    "pytest-qt>=4.2.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.0",
    "black>=23.7.0",
    "ruff>=0.0.285",
    "mypy>=1.5.0",
    "pre-commit>=3.3.0",
]
docs = [
    "sphinx>=7.1.0",
    "sphinx-rtd-theme>=1.3.0",
    "myst-parser>=2.0.0",
]

[project.scripts]
raman-app = "main:main"

[tool.setuptools]
packages = [
    "functions",
    "functions.preprocess",
    "functions.ML",
    "functions.visualization",
    "pages",
    "components",
    "components.widgets",
    "configs",
]

[tool.black]
line-length = 100
target-version = ['py310']
include = '\.pyi?$'
extend-exclude = '''
/(
  | .venv
  | build
  | dist
)/
'''

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by black)
    "B008",  # function calls in argument defaults
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--tb=short",
    "--cov=.",
    "--cov-report=html",
    "--cov-report=term-missing",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "ui: marks tests as UI tests",
]

[tool.coverage.run]
source = ["."]
omit = [
    "*/tests/*",
    "*/build/*",
    "*/dist/*",
    "*/.venv/*",
    "*/setup.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
disallow_incomplete_defs = false
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true

[[tool.mypy.overrides]]
module = [
    "scipy.*",
    "sklearn.*",
    "matplotlib.*",
    "pandas.*",
]
ignore_missing_imports = true
```

---

## Dependency Management

### Core Dependencies

#### Required Dependencies

Install with:
```bash
uv pip install -r requirements.txt
```

**requirements.txt**:
```txt
# Core scientific computing
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0

# GUI Framework
PyQt6>=6.5.0

# Machine Learning
scikit-learn>=1.3.0

# Data export
openpyxl>=3.1.0

# Utilities
joblib>=1.3.0
tqdm>=4.66.0
```

#### Optional Dependencies

**Machine Learning (Enhanced)**:
```bash
uv pip install xgboost>=2.0.0 umap-learn>=0.5.3
```

**Deep Learning**:
```bash
uv pip install torch>=2.0.0 torchvision>=0.15.0
```

**Camera Integration**:
```bash
# Andor SDK (requires manual installation)
# See functions/andorsdk/README.md
```

#### Development Dependencies

**requirements-dev.txt**:
```txt
# Testing
pytest>=7.4.0
pytest-qt>=4.2.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
pytest-xdist>=3.3.0  # Parallel testing

# Code Quality
black>=23.7.0
ruff>=0.0.285
mypy>=1.5.0
pre-commit>=3.3.0

# Documentation
sphinx>=7.1.0
sphinx-rtd-theme>=1.3.0
myst-parser>=2.0.0
sphinx-autodoc-typehints>=1.24.0

# Build Tools
pyinstaller>=5.13.0
pyinstaller-hooks-contrib>=2023.7
```

Install with:
```bash
uv pip install -r requirements-dev.txt
```

### Virtual Environment

#### Creating Virtual Environment

```bash
# Create virtual environment
uv venv

# Activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux/macOS:
source .venv/bin/activate
```

#### Installing Dependencies

```bash
# Install core dependencies
uv pip install -r requirements.txt

# Install development dependencies
uv pip install -r requirements-dev.txt

# Install package in editable mode
uv pip install -e .

# Install with optional dependencies
uv pip install -e ".[ml,dev]"
```

#### Managing Dependencies

```bash
# List installed packages
uv pip list

# Show package info
uv pip show package-name

# Update package
uv pip install --upgrade package-name

# Freeze current environment
uv pip freeze > requirements-frozen.txt
```

---

## Building the Application

### PyInstaller Configuration

PyInstaller bundles the application into a standalone executable.

#### Spec File

**File**: `raman_app.spec`

```python
# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Determine platform
is_windows = sys.platform.startswith('win')
is_macos = sys.platform == 'darwin'
is_linux = sys.platform.startswith('linux')

# Application info
app_name = 'RamanApp'
version = '1.0.0'
author = 'Your Organization'

# Collect data files
datas = []
datas += collect_data_files('matplotlib')
datas += collect_data_files('sklearn')
datas += [
    ('assets', 'assets'),
    ('configs/app_configs.json', 'configs'),
    ('assets/locales', 'assets/locales'),
]

# Collect hidden imports
hiddenimports = []
hiddenimports += collect_submodules('sklearn')
hiddenimports += collect_submodules('scipy')
hiddenimports += [
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'matplotlib.backends.backend_qt5agg',
    'numpy',
    'pandas',
    'openpyxl',
]

# Optional: Add XGBoost if available
try:
    import xgboost
    hiddenimports += collect_submodules('xgboost')
except ImportError:
    pass

# Analysis
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'test',
        'tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Remove unnecessary files
def remove_from_list(lst, remove_list):
    return [item for item in lst if item[0] not in remove_list]

# Remove test files
a.datas = remove_from_list(a.datas, ['test', 'tests'])

# PYZ (Python zip archive)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# EXE (Windows) or Binary (Linux/macOS)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icons/app_icon.ico' if is_windows else 'assets/icons/app_icon.icns',
)

# COLLECT (collect all files)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=app_name,
)

# macOS: Create .app bundle
if is_macos:
    app = BUNDLE(
        coll,
        name=f'{app_name}.app',
        icon='assets/icons/app_icon.icns',
        bundle_identifier=f'com.yourorg.{app_name.lower()}',
        info_plist={
            'CFBundleName': app_name,
            'CFBundleDisplayName': app_name,
            'CFBundleVersion': version,
            'CFBundleShortVersionString': version,
            'NSHighResolutionCapable': 'True',
            'NSRequiresAquaSystemAppearance': 'False',
        },
    )
```

#### Building Executable

**Windows**:
```powershell
# Build using spec file
pyinstaller raman_app.spec

# Build directly (creates spec file)
pyinstaller --windowed `
    --name RamanApp `
    --icon assets/icons/app_icon.ico `
    --add-data "assets;assets" `
    --add-data "configs/app_configs.json;configs" `
    --hidden-import PyQt6.QtCore `
    --hidden-import PyQt6.QtGui `
    --hidden-import PyQt6.QtWidgets `
    main.py
```

**Linux/macOS**:
```bash
# Build using spec file
pyinstaller raman_app.spec

# Build directly
pyinstaller --windowed \
    --name RamanApp \
    --icon assets/icons/app_icon.icns \
    --add-data "assets:assets" \
    --add-data "configs/app_configs.json:configs" \
    --hidden-import PyQt6.QtCore \
    --hidden-import PyQt6.QtGui \
    --hidden-import PyQt6.QtWidgets \
    main.py
```

#### Output

```
dist/
└── RamanApp/              # Folder with executable and dependencies
    ├── RamanApp.exe       # Windows
    ├── RamanApp           # Linux
    ├── RamanApp.app/      # macOS (bundle)
    ├── assets/            # Data files
    ├── configs/
    └── ... (libraries and dependencies)
```

### Build Scripts

#### Portable Build Script

**File**: `build_scripts/build_portable.sh` (Linux/macOS)

```bash
#!/bin/bash

# Build portable application (no installer)

set -e  # Exit on error

echo "=== Building Raman App (Portable) ==="

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/

# Build with PyInstaller
echo "Building with PyInstaller..."
pyinstaller raman_app.spec

# Create zip archive
echo "Creating archive..."
cd dist
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: zip .app bundle
    zip -r RamanApp-macos.zip RamanApp.app
else
    # Linux: tar.gz
    tar -czf RamanApp-linux.tar.gz RamanApp/
fi
cd ..

echo "=== Build complete ==="
echo "Output: dist/RamanApp-*.zip or dist/RamanApp-*.tar.gz"
```

**File**: `build_scripts/build_portable.ps1` (Windows)

```powershell
# Build portable application (no installer)

Write-Host "=== Building Raman App (Portable) ===" -ForegroundColor Green

# Clean previous builds
Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
Remove-Item -Path "build", "dist" -Recurse -Force -ErrorAction SilentlyContinue

# Build with PyInstaller
Write-Host "Building with PyInstaller..." -ForegroundColor Yellow
pyinstaller raman_app.spec

# Create zip archive
Write-Host "Creating archive..." -ForegroundColor Yellow
Compress-Archive -Path "dist\RamanApp" -DestinationPath "dist\RamanApp-windows.zip" -Force

Write-Host "=== Build complete ===" -ForegroundColor Green
Write-Host "Output: dist\RamanApp-windows.zip"
```

#### Test Build Script

**File**: `build_scripts/test_build_executable.py`

```python
"""Test built executable"""

import subprocess
import sys
import os
from pathlib import Path

def find_executable():
    """Find built executable"""
    dist_dir = Path('dist')
    
    if sys.platform == 'win32':
        exe = dist_dir / 'RamanApp' / 'RamanApp.exe'
    elif sys.platform == 'darwin':
        exe = dist_dir / 'RamanApp.app' / 'Contents' / 'MacOS' / 'RamanApp'
    else:
        exe = dist_dir / 'RamanApp' / 'RamanApp'
    
    return exe

def test_executable():
    """Test if executable runs"""
    exe = find_executable()
    
    if not exe.exists():
        print(f"❌ Executable not found: {exe}")
        return False
    
    print(f"✓ Executable found: {exe}")
    
    # Try running with --version flag
    try:
        result = subprocess.run(
            [str(exe), '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"✓ Executable runs successfully")
            print(f"  Version: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Executable failed with code {result.returncode}")
            print(f"  Error: {result.stderr}")
            return False
    
    except subprocess.TimeoutExpired:
        print("❌ Executable timed out")
        return False
    except Exception as e:
        print(f"❌ Error running executable: {e}")
        return False

if __name__ == '__main__':
    success = test_executable()
    sys.exit(0 if success else 1)
```

---

## Platform-Specific Builds

### Windows

#### Requirements

- Windows 10 or later
- Python 3.10+
- Visual C++ Redistributable (for some dependencies)

#### Building

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies
uv pip install -r requirements.txt
uv pip install pyinstaller

# Build
pyinstaller raman_app.spec

# Test
python build_scripts/test_build_executable.py

# Output
# dist/RamanApp/RamanApp.exe
```

#### Icon

- **Format**: `.ico`
- **Sizes**: 16x16, 32x32, 48x48, 256x256
- **Location**: `assets/icons/app_icon.ico`

#### Code Signing (Optional)

```powershell
# Sign executable with certificate
signtool sign /f certificate.pfx /p password /tr http://timestamp.digicert.com /td sha256 /fd sha256 dist\RamanApp\RamanApp.exe
```

### macOS

#### Requirements

- macOS 11 (Big Sur) or later
- Python 3.10+
- Xcode Command Line Tools

#### Building

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install pyinstaller

# Build
pyinstaller raman_app.spec

# Test
python build_scripts/test_build_executable.py

# Output
# dist/RamanApp.app
```

#### Icon

- **Format**: `.icns`
- **Sizes**: 16x16 to 1024x1024 (various @1x and @2x)
- **Location**: `assets/icons/app_icon.icns`

**Create .icns from PNG**:
```bash
# Create iconset
mkdir MyIcon.iconset
sips -z 16 16     icon.png --out MyIcon.iconset/icon_16x16.png
sips -z 32 32     icon.png --out MyIcon.iconset/icon_16x16@2x.png
sips -z 32 32     icon.png --out MyIcon.iconset/icon_32x32.png
sips -z 64 64     icon.png --out MyIcon.iconset/icon_32x32@2x.png
sips -z 128 128   icon.png --out MyIcon.iconset/icon_128x128.png
sips -z 256 256   icon.png --out MyIcon.iconset/icon_128x128@2x.png
sips -z 256 256   icon.png --out MyIcon.iconset/icon_256x256.png
sips -z 512 512   icon.png --out MyIcon.iconset/icon_256x256@2x.png
sips -z 512 512   icon.png --out MyIcon.iconset/icon_512x512.png
sips -z 1024 1024 icon.png --out MyIcon.iconset/icon_512x512@2x.png

# Create .icns
iconutil -c icns MyIcon.iconset
```

#### Code Signing and Notarization

**Sign App**:
```bash
# Sign with Developer ID
codesign --force --deep --sign "Developer ID Application: Your Name (TEAM_ID)" dist/RamanApp.app

# Verify
codesign --verify --verbose dist/RamanApp.app
spctl --assess --verbose dist/RamanApp.app
```

**Notarize**:
```bash
# Create zip
ditto -c -k --keepParent dist/RamanApp.app RamanApp.zip

# Submit for notarization
xcrun notarytool submit RamanApp.zip \
    --apple-id your.email@example.com \
    --team-id TEAM_ID \
    --password app-specific-password \
    --wait

# Staple notarization ticket
xcrun stapler staple dist/RamanApp.app

# Verify
spctl --assess --verbose=4 dist/RamanApp.app
```

### Linux

#### Requirements

- Linux (Ubuntu 20.04+ or equivalent)
- Python 3.10+
- Development tools (`build-essential`)

#### Building

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install pyinstaller

# Build
pyinstaller raman_app.spec

# Test
python build_scripts/test_build_executable.py

# Output
# dist/RamanApp/RamanApp
```

#### Icon

- **Format**: `.png` or `.svg`
- **Size**: 512x512 or scalable
- **Location**: `assets/icons/app_icon.png`

#### Creating AppImage

**Tool**: `appimagetool`

```bash
# Download appimagetool
wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x appimagetool-x86_64.AppImage

# Create AppDir
mkdir -p RamanApp.AppDir/usr/bin
cp -r dist/RamanApp/* RamanApp.AppDir/usr/bin/

# Create desktop file
cat > RamanApp.AppDir/RamanApp.desktop << EOF
[Desktop Entry]
Type=Application
Name=Raman App
Exec=RamanApp
Icon=app_icon
Categories=Science;Education;
EOF

# Copy icon
cp assets/icons/app_icon.png RamanApp.AppDir/app_icon.png

# Create AppRun
cat > RamanApp.AppDir/AppRun << 'EOF'
#!/bin/bash
SELF=$(readlink -f "$0")
HERE=${SELF%/*}
exec "${HERE}/usr/bin/RamanApp" "$@"
EOF
chmod +x RamanApp.AppDir/AppRun

# Build AppImage
./appimagetool-x86_64.AppImage RamanApp.AppDir RamanApp-x86_64.AppImage
```

---

## Creating Installers

### Windows Installer (NSIS)

**NSIS** (Nullsoft Scriptable Install System) creates Windows installers.

#### Install NSIS

Download from: https://nsis.sourceforge.io/Download

#### NSIS Script

**File**: `raman_app_installer.nsi`

```nsis
; Raman App Installer Script

!define APP_NAME "Raman App"
!define APP_VERSION "1.0.0"
!define APP_PUBLISHER "Your Organization"
!define APP_URL "https://example.com"
!define APP_EXE "RamanApp.exe"

; Include Modern UI
!include "MUI2.nsh"

; General
Name "${APP_NAME}"
OutFile "RamanApp-${APP_VERSION}-installer.exe"
InstallDir "$PROGRAMFILES64\${APP_NAME}"
InstallDirRegKey HKLM "Software\${APP_NAME}" "InstallDir"
RequestExecutionLevel admin

; Interface Settings
!define MUI_ABORTWARNING
!define MUI_ICON "assets\icons\app_icon.ico"
!define MUI_UNICON "assets\icons\app_icon.ico"

; Pages
!insertmacro MUI_PAGE_LICENSE "LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; Languages
!insertmacro MUI_LANGUAGE "English"

; Installer Sections
Section "Install"
    SetOutPath "$INSTDIR"
    
    ; Copy files
    File /r "dist\RamanApp\*.*"
    
    ; Create shortcuts
    CreateDirectory "$SMPROGRAMS\${APP_NAME}"
    CreateShortCut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"
    CreateShortCut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"
    
    ; Create uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"
    
    ; Registry
    WriteRegStr HKLM "Software\${APP_NAME}" "InstallDir" "$INSTDIR"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
        "DisplayName" "${APP_NAME}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
        "UninstallString" "$INSTDIR\Uninstall.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
        "DisplayVersion" "${APP_VERSION}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
        "Publisher" "${APP_PUBLISHER}"
SectionEnd

; Uninstaller Section
Section "Uninstall"
    ; Remove files
    RMDir /r "$INSTDIR"
    
    ; Remove shortcuts
    Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
    RMDir "$SMPROGRAMS\${APP_NAME}"
    Delete "$DESKTOP\${APP_NAME}.lnk"
    
    ; Remove registry keys
    DeleteRegKey HKLM "Software\${APP_NAME}"
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"
SectionEnd
```

#### Build Installer

```powershell
# Build executable first
pyinstaller raman_app.spec

# Build installer
makensis raman_app_installer.nsi

# Output
# RamanApp-1.0.0-installer.exe
```

#### Build Script

**File**: `build_scripts/build_installer.ps1`

```powershell
# Build Windows installer

Write-Host "=== Building Raman App Installer ===" -ForegroundColor Green

# Build executable
Write-Host "Building executable..." -ForegroundColor Yellow
pyinstaller raman_app.spec

# Build installer
Write-Host "Building installer..." -ForegroundColor Yellow
& "C:\Program Files (x86)\NSIS\makensis.exe" raman_app_installer.nsi

Write-Host "=== Build complete ===" -ForegroundColor Green
Write-Host "Output: RamanApp-1.0.0-installer.exe"
```

### macOS Installer (DMG)

**create-dmg** tool creates DMG installers.

#### Install create-dmg

```bash
brew install create-dmg
```

#### Create DMG

```bash
# Build app first
pyinstaller raman_app.spec

# Create DMG
create-dmg \
    --volname "Raman App" \
    --window-pos 200 120 \
    --window-size 800 400 \
    --icon-size 100 \
    --icon "RamanApp.app" 200 190 \
    --hide-extension "RamanApp.app" \
    --app-drop-link 600 185 \
    "RamanApp-1.0.0.dmg" \
    "dist/"
```

### Linux Package (DEB)

**dpkg-deb** creates Debian packages.

#### Create Package Structure

```bash
# Create directory structure
mkdir -p raman-app_1.0.0/DEBIAN
mkdir -p raman-app_1.0.0/usr/bin
mkdir -p raman-app_1.0.0/usr/share/applications
mkdir -p raman-app_1.0.0/usr/share/icons/hicolor/512x512/apps

# Copy files
cp -r dist/RamanApp/* raman-app_1.0.0/usr/bin/
cp assets/icons/app_icon.png raman-app_1.0.0/usr/share/icons/hicolor/512x512/apps/raman-app.png
```

#### Create Control File

**File**: `raman-app_1.0.0/DEBIAN/control`

```
Package: raman-app
Version: 1.0.0
Section: science
Priority: optional
Architecture: amd64
Depends: python3 (>= 3.10), libpython3.10, libqt6core6
Maintainer: Your Name <your.email@example.com>
Description: Raman Spectroscopy Analysis Application
 Real-time Raman spectral analysis and classification tool
 with advanced preprocessing and machine learning capabilities.
```

#### Create Desktop Entry

**File**: `raman-app_1.0.0/usr/share/applications/raman-app.desktop`

```desktop
[Desktop Entry]
Type=Application
Name=Raman App
Comment=Raman Spectroscopy Analysis
Exec=/usr/bin/RamanApp
Icon=raman-app
Categories=Science;Education;
Terminal=false
```

#### Build Package

```bash
# Build .deb package
dpkg-deb --build raman-app_1.0.0

# Output: raman-app_1.0.0.deb
```

---

## CI/CD Pipeline

### GitHub Actions

**File**: `.github/workflows/build.yml`

```yaml
name: Build and Test

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  release:
    types: [created]

jobs:
  test:
    name: Test on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install UV
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
        if: runner.os != 'Windows'
      
      - name: Install UV (Windows)
        run: |
          irm https://astral.sh/uv/install.ps1 | iex
        if: runner.os == 'Windows'
      
      - name: Install dependencies
        run: |
          uv pip install -r requirements.txt
          uv pip install -r requirements-dev.txt
      
      - name: Run tests
        run: pytest --cov --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
  
  build:
    name: Build on ${{ matrix.os }}
    needs: test
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install uv
          uv pip install -r requirements.txt
          uv pip install pyinstaller
      
      - name: Build executable
        run: pyinstaller raman_app.spec
      
      - name: Test executable
        run: python build_scripts/test_build_executable.py
      
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: raman-app-${{ matrix.os }}
          path: dist/
  
  release:
    name: Create Release
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name == 'release'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Download artifacts
        uses: actions/download-artifact@v3
      
      - name: Create archives
        run: |
          cd raman-app-ubuntu-latest
          tar -czf ../RamanApp-linux.tar.gz RamanApp/
          cd ../raman-app-windows-latest
          zip -r ../RamanApp-windows.zip RamanApp/
          cd ../raman-app-macos-latest
          zip -r ../RamanApp-macos.zip RamanApp.app/
      
      - name: Upload release assets
        uses: softprops/action-gh-release@v1
        with:
          files: |
            RamanApp-linux.tar.gz
            RamanApp-windows.zip
            RamanApp-macos.zip
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## Documentation Building

### Sphinx Configuration

**File**: `docs/conf.py`

```python
# Configuration file for Sphinx documentation

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'Raman App'
copyright = '2024, Your Organization'
author = 'Your Name'
version = '1.0'
release = '1.0.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# MyST Parser configuration
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'substitution',
    'tasklist',
]

# Templates
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '../assets/icons/app_icon.png'
html_favicon = '../assets/icons/app_icon.ico'

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
}

# Autodoc
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
}
```

### Building Documentation

```bash
# Install Sphinx
uv pip install sphinx sphinx-rtd-theme myst-parser

# Build HTML
cd docs
make html

# View documentation
# Open docs/_build/html/index.html

# Build PDF (requires LaTeX)
make latexpdf

# Clean build
make clean
```

### ReadTheDocs Integration

**File**: `.readthedocs.yaml`

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

python:
  install:
    - requirements: requirements.txt
    - requirements: docs/requirements.txt

sphinx:
  configuration: docs/conf.py
```

---

## Troubleshooting

### Common Build Issues

#### Issue: "Module not found" error

**Cause**: Missing hidden import

**Solution**: Add to `hiddenimports` in spec file:
```python
hiddenimports += ['missing_module']
```

#### Issue: Data files not included

**Cause**: Files not added to `datas`

**Solution**: Add to spec file:
```python
datas += [('path/to/data', 'destination')]
```

#### Issue: Large executable size

**Cause**: Including unnecessary libraries

**Solution**:
1. Add to `excludes` in spec file
2. Use `upx=True` for compression
3. Remove test/debug files

#### Issue: Executable crashes on startup

**Causes**:
- Missing DLLs (Windows)
- Code signing issues (macOS)
- Missing libraries (Linux)

**Solutions**:
```bash
# Windows: Check dependencies
dumpbin /dependents RamanApp.exe

# macOS: Check dependencies
otool -L RamanApp.app/Contents/MacOS/RamanApp

# Linux: Check dependencies
ldd RamanApp
```

### Platform-Specific Issues

#### Windows

**Issue**: "VCRUNTIME140.dll not found"

**Solution**: Install Visual C++ Redistributable:
```powershell
# Download from Microsoft
https://aka.ms/vs/17/release/vc_redist.x64.exe
```

#### macOS

**Issue**: "App is damaged and can't be opened"

**Solution**: Code sign the app:
```bash
codesign --force --deep --sign - RamanApp.app
```

#### Linux

**Issue**: "error while loading shared libraries"

**Solution**: Install missing libraries:
```bash
sudo apt-get install libxcb-xinerama0 libxcb-icccm4 libxcb-image0
```

---

## See Also

- [Architecture Guide](architecture.md) - System design
- [Contributing Guide](contributing.md) - Development workflow
- [Testing Guide](testing.md) - Testing strategy
- [API Reference](../api/index.md) - Complete API documentation

---

**Last Updated**: 2026-01-24
