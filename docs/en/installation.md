# Installation Guide

This guide provides detailed installation instructions for all supported platforms and methods.

## Prerequisites

Before installing, ensure your system meets these requirements:

### Python Version

- **Python 3.12** or higher (3.12.0+ recommended)
- Check your Python version:
  ```bash
  python --version
  ```

### System Dependencies

**Windows:**
- No additional dependencies required
- Microsoft Visual C++ Redistributable 2015-2022 (usually pre-installed)

**macOS:**
- Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip build-essential
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install python3-devel python3-pip gcc gcc-c++
```

## Installation Methods

(from-source)=

### Method 1: From Source

Recommended for **developers** and **advanced users** who want the latest features and ability to customize.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application.git
cd Raman-Spectroscopy-Analysis-Application
```

#### Step 2: Create Virtual Environment

**Using UV (Recommended):**

UV is a fast Python package installer and resolver.

```bash
# Install UV if not already installed
pip install uv

# Create virtual environment and install dependencies
uv venv
uv pip install -e .
```

**Using Traditional venv:**

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 3: Verify Installation

```bash
# Test imports
python -c "import PySide6; print('PySide6 OK')"
python -c "import ramanspy; print('RamanSPy OK')"
python -c "import numpy; print('NumPy OK')"
```

All commands should print "OK" without errors.

#### Step 4: Run the Application

**With UV:**
```bash
uv run python main.py
```

**Without UV:**
```bash
# Ensure virtual environment is activated
python main.py
```

#### Updating from Source

To get the latest updates:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
uv pip install -e .  # With UV
# OR
pip install -r requirements.txt  # Without UV
```

---

(portable-executable)=

### Method 2: Portable Executable (Windows Only)

Best for **end users** who want to run the application without installing Python.

#### Step 1: Download

1. Go to [Releases](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/releases)
2. Download the latest portable ZIP file: `RamanApp_portable_vX.X.X.zip`
3. File size: ~375 MB (includes all dependencies)

#### Step 2: Extract

1. Extract the ZIP file to your desired location
2. Recommended locations:
   - Desktop: `C:\Users\YourName\Desktop\RamanApp`
   - Programs: `C:\Program Files\RamanApp`
   - USB Drive: `E:\RamanApp`

#### Step 3: Run

1. Navigate to the extracted folder
2. Double-click `RamanApp.exe`
3. First launch may take 10-20 seconds

#### Features

- ✅ **No installation required** - Run directly from any folder
- ✅ **Portable** - Copy to USB drive and run on any Windows PC
- ✅ **Self-contained** - All dependencies bundled (Python, libraries, drivers)
- ✅ **No Python required** - Works on systems without Python installed
- ✅ **Isolated** - Does not interfere with system Python installations

#### Limitations

- ❌ **Windows only** - Not available for macOS or Linux
- ❌ **Larger file size** - ~375 MB vs ~50 MB source installation
- ❌ **No automatic updates** - Must manually download new versions
- ❌ **Slower startup** - First launch takes longer than source installation

---

(installer)=

### Method 3: Installer (Windows Only)

Best for **permanent installations** with Start Menu integration and file associations.

#### Step 1: Download

1. Go to [Releases](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/releases)
2. Download the latest installer: `RamanApp_installer_vX.X.X.exe`
3. File size: ~375 MB

#### Step 2: Run Installer

1. Double-click the installer executable
2. Windows SmartScreen may appear:
   - Click "More info"
   - Click "Run anyway"
3. Follow the installation wizard:
   - **License Agreement**: Review and accept MIT License
   - **Installation Location**: Default is `C:\Program Files\RamanApp`
   - **Start Menu Folder**: Choose folder name (default: "Raman Spectroscopy Analysis")
   - **Desktop Shortcut**: Optionally create desktop shortcut

#### Step 3: Launch

- From **Start Menu**: Search "Raman" or navigate to your chosen folder
- From **Desktop**: Double-click shortcut (if created)

#### Features

- ✅ **Professional installation** - Standard Windows installation experience
- ✅ **Start Menu integration** - Easy access from Windows menu
- ✅ **Desktop shortcut** - Optional quick access
- ✅ **File associations** - `.rproj` files open with application
- ✅ **Uninstaller** - Clean removal via Windows Settings

#### Uninstallation

**Via Windows Settings:**
1. Open **Settings** → **Apps** → **Installed Apps**
2. Search for "Raman Spectroscopy Analysis"
3. Click **⋮** → **Uninstall**

**Via Control Panel:**
1. Open **Control Panel** → **Programs** → **Uninstall a program**
2. Select "Raman Spectroscopy Analysis"
3. Click **Uninstall**

---

## Post-Installation

### Verify Installation

After installation, verify the application works:

1. **Launch the application** using your preferred method
2. **Create a test project**:
   - Click "New Project"
   - Name: "Installation Test"
   - Click "Create"
3. **Check interface elements**:
   - All tabs should be visible (Home, Data, Preprocessing, Analysis, ML)
   - No error messages in the console/log

### Optional: Install Development Tools

If you plan to develop or contribute:

```bash
# Install development dependencies
uv pip install -e ".[dev]"  # With UV
# OR
pip install -e ".[dev]"  # Without UV

# This installs:
# - pytest (testing framework)
# - black (code formatter)
# - watchdog (file monitoring)
```

### Configure Application Settings

On first launch, you may want to configure:

1. **Language Preference**:
   - Settings → Interface → Language
   - Choose English or Japanese (日本語)

2. **Default Project Location**:
   - Settings → Projects → Default Location
   - Choose where new projects are created

3. **Processing Preferences**:
   - Settings → Processing → Number of CPU Cores
   - Adjust based on your system (default: auto-detect)

## Troubleshooting Installation

### Common Issues

#### Python Version Mismatch

**Error:** `Python 3.12 or higher is required`

**Solution:**
```bash
# Check current version
python --version

# Install Python 3.12+ from https://www.python.org/downloads/
# On Windows, ensure "Add Python to PATH" is checked during installation

# Verify new version
python --version
```

#### Module Not Found Errors

**Error:** `ModuleNotFoundError: No module named 'PySide6'`

**Solution:**
```bash
# Ensure virtual environment is activated
# Then reinstall dependencies
uv pip install -e .  # With UV
# OR
pip install -r requirements.txt  # Without UV
```

#### Permission Denied (Linux/macOS)

**Error:** `PermissionError: [Errno 13] Permission denied`

**Solution:**
```bash
# Do NOT use sudo with pip in virtual environments
# Instead, ensure virtual environment is activated:
source .venv/bin/activate

# Then install normally
pip install -r requirements.txt
```

#### UV Installation Fails

**Error:** `pip install uv` fails

**Solution:**
```bash
# Use traditional venv instead
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

#### Windows Executable Blocked

**Error:** Windows SmartScreen blocks executable

**Solution:**
1. Click "More info"
2. Click "Run anyway"
3. This is expected for unsigned executables

### Getting Help

If installation issues persist:

1. **Check existing issues**: [GitHub Issues](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues)
2. **Create new issue**: Include:
   - Operating system and version
   - Python version (`python --version`)
   - Installation method attempted
   - Full error message
   - Steps to reproduce

## Next Steps

Now that installation is complete:

- [Getting Started Guide](getting-started.md) - Create your first project
- [Quick Start Tutorial](quick-start.md) - 5-minute walkthrough
- [User Guide](user-guide/index.md) - Comprehensive tutorials

## Advanced Installation

### Installing Specific Versions

```bash
# Install specific version from git tag
git clone --branch v1.0.0 https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application.git
cd Raman-Spectroscopy-Analysis-Application
uv pip install -e .
```

### Installing with Optional Dependencies

```bash
# Install with CUDA support for deep learning
uv pip install -e ".[cuda]"

# Install with development + testing tools
uv pip install -e ".[dev,test]"

# Install everything
uv pip install -e ".[all]"
```

### Building from Source (Advanced)

To build your own executable:

```bash
# Install build dependencies
uv pip install -e ".[build]"

# Build portable executable
python build_scripts/build_portable.py

# Build installer
python build_scripts/build_installer.py
```

See [Build System Documentation](dev-guide/build-system.md) for details.
