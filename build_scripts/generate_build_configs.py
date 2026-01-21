#!/usr/bin/env python3
"""
Build Configuration Generator for Raman Spectroscopy Analysis Application
===========================================================================

This script generates optimized PyInstaller spec files and NSIS installer
configuration based on the current system environment and installed packages.

Features:
- Auto-detects Python version and architecture
- Scans installed packages and creates optimal hidden imports list
- Generates platform-specific configurations
- Creates both portable and installer spec files
- Generates NSIS installer script with correct paths

Usage:
    python generate_build_configs.py [--output-dir DIR] [--console] [--analyze-only]

Arguments:
    --output-dir DIR     Output directory for generated files (default: current directory)
    --console            Generate spec files with console window enabled (for debugging)
    --analyze-only       Only analyze environment, don't generate files

Generated Files:
    - raman_app.spec (Portable executable configuration)
    - raman_app_installer.spec (Installer staging configuration)
    - raman_app_installer.nsi (NSIS installer script)
    - build_config_report.json (System analysis report)

Author: MUHAMMAD HELMI BIN ROZAIN
Date: October 22, 2025
"""

import sys
import os
import platform
import json
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
import importlib.util
import importlib.metadata

def _try_import_create_logs():
    """Best-effort import of create_logs.

    This script can be run standalone; if the project root isn't on sys.path,
    we fall back to stdlib logging.
    """
    try:
        from configs.configs import create_logs  # type: ignore

        return create_logs
    except Exception:
        return None


_CREATE_LOGS = _try_import_create_logs()


def _log(status: str, message: str):
    """Log to app logger (if available) and to stdlib logging."""
    if _CREATE_LOGS is not None:
        try:
            _CREATE_LOGS(__name__, __file__, message, status=status)
        except Exception:
            # Never let logging break build config generation.
            pass

    level = status.lower()
    if level == "warn":
        level = "warning"

    log_fn = getattr(logging, level, logging.info)
    log_fn(message)


class BuildConfigGenerator:
    """Generates build configuration files based on system environment."""

    def __init__(self, project_root: Path, console_mode: bool = False):
        self.project_root = project_root
        self.console_mode = console_mode
        self.system_info = self._detect_system()
        self.installed_packages = self._scan_installed_packages()
        self.required_imports = self._analyze_imports()

    def _detect_system(self) -> Dict[str, str]:
        """Detect system information."""
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "platform_release": platform.release(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
            "python_build": platform.python_build()[0],
            "python_executable": sys.executable,
            "bits": "64bit" if sys.maxsize > 2**32 else "32bit",
        }

    def _scan_installed_packages(self) -> Dict[str, str]:
        """Scan and return installed packages with versions."""
        packages = {}
        try:
            for dist in importlib.metadata.distributions():
                packages[dist.name] = dist.version
        except Exception as e:
            _log("warning", f"Could not scan all packages: {e}")

        return packages

    def _analyze_imports(self) -> Dict[str, List[str]]:
        """Analyze project imports and categorize them."""
        categories = {
            "gui": [],
            "data_science": [],
            "visualization": [],
            "spectroscopy": [],
            "ml_dl": [],
            "utilities": [],
        }

        # GUI frameworks
        if "PySide6" in self.installed_packages:
            categories["gui"].extend(
                [
                    "PySide6.QtCore",
                    "PySide6.QtGui",
                    "PySide6.QtWidgets",
                    "PySide6.QtOpenGL",
                    "PySide6.QtPrintSupport",
                    "PySide6.QtSvg",
                    "PySide6.QtSvgWidgets",
                    "shiboken6",
                ]
            )

        # Data science
        for pkg in ["numpy", "pandas", "scipy"]:
            if pkg in self.installed_packages:
                categories["data_science"].append(pkg)
                if pkg == "scipy":
                    categories["data_science"].extend(
                        [
                            "scipy.integrate",
                            "scipy.signal",
                            "scipy.interpolate",
                            "scipy.optimize",
                            "scipy.special",
                            "scipy.stats",
                            "scipy.stats._stats_py",
                            "scipy.stats.distributions",
                            "scipy.stats._distn_infrastructure",
                            "scipy.linalg",
                            "scipy.sparse",
                            "scipy.ndimage",
                        ]
                    )

        # Machine learning
        for pkg in ["sklearn", "scikit-learn"]:
            if pkg in self.installed_packages:
                categories["ml_dl"].extend(
                    [
                        "sklearn",
                        "sklearn.preprocessing",
                        "sklearn.decomposition",
                        "sklearn.linear_model",
                        "sklearn.ensemble",
                        "sklearn.metrics",
                        "sklearn.model_selection",
                    ]
                )
                break

        # Deep learning
        if "torch" in self.installed_packages:
            categories["ml_dl"].extend(
                [
                    "torch",
                    "torch.nn",
                    "torch.optim",
                ]
            )

        # Visualization
        if "matplotlib" in self.installed_packages:
            categories["visualization"].extend(
                [
                    "matplotlib",
                    "matplotlib.pyplot",
                    "matplotlib.backends.backend_qt5agg",
                    "matplotlib.figure",
                    "matplotlib.widgets",
                ]
            )

        if "seaborn" in self.installed_packages:
            categories["visualization"].append("seaborn")

        if "PIL" in self.installed_packages or "Pillow" in self.installed_packages:
            categories["visualization"].append("PIL")

        if "imageio" in self.installed_packages:
            categories["visualization"].append("imageio")

        # Spectroscopy
        for pkg in ["ramanspy", "pybaselines"]:
            if pkg in self.installed_packages:
                categories["spectroscopy"].append(pkg)
                if pkg == "ramanspy":
                    categories["spectroscopy"].extend(
                        [
                            "ramanspy.preprocessing",
                            "ramanspy.preprocessing.normalise",
                            "ramanspy.preprocessing.noise",
                            "ramanspy.preprocessing.baseline",
                        ]
                    )

        # Utilities
        for pkg in ["requests", "tqdm", "cloudpickle", "joblib", "pydantic"]:
            if pkg in self.installed_packages:
                categories["utilities"].append(pkg)

        return categories

    def generate_spec_file(
        self, output_path: Path, installer_mode: bool = False
    ) -> str:
        output_dir = "dist_installer" if installer_mode else "dist"
        name_suffix = "_installer" if installer_mode else ""

        # Collect all hidden imports
        all_imports = []
        for category, imports in self.required_imports.items():
            all_imports.extend(imports)

        # Add splash_screen module
        all_imports.append("splash_screen")

        # Define Splash Image Path (Ensure this file exists!)
        splash_image = "assets/splash.png"
        has_splash = os.path.exists(os.path.join(str(self.project_root), splash_image))

        spec_content = f'''# -*- mode: python ; coding: utf-8 -*-
"""
Optimized Spec File with Splash Screen
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import os
import sys

spec_root = os.path.abspath(os.getcwd())

# ============== CONFIGURATION ==============
block_cipher = None

# Explicitly exclude heavy/unused modules to improve startup speed
excluded_modules = [
    'tkinter', '_tkinter', 'turtle',
    'test', 'unittest', 'doctest', 'pydoc', 'pdb', 'bdb',
    'matplotlib.tests', 'numpy.tests', 'scipy.tests', 'pandas.tests',
    'ipython', 'IPython', 'jedi', 'jupyter', 'notebook',
    'PIL.ImageTk', 'curses',
    'distutils', 'setuptools', 'pip', 'wheel',
    'xmlrpc', 'xml.etree.cElementTree',
    'multiprocessing.dummy', 'pydoc_data',
    'pkg_resources', 'packaging'
]

# ============== DATA FILES ==============
datas = []
# Add Assets
assets_path = os.path.join(spec_root, 'assets')
if os.path.exists(assets_path):
    datas.append((assets_path, 'assets'))

# Collect essential data
datas += collect_data_files('PySide6')
datas += collect_data_files('matplotlib')

# ============== HIDDEN IMPORTS ==============
hiddenimports = {json.dumps(all_imports, indent=4)}

# Collect submodules only if necessary (heavy operation)
try:
    hiddenimports += collect_submodules('ramanspy')
    hiddenimports += collect_submodules('pybaselines')
except Exception:
    pass

# ============== BINARIES ==============
binaries = []
dll_path = os.path.join(spec_root, 'drivers')
if os.path.exists(dll_path):
    binaries += [
        (os.path.join(dll_path, 'atmcd32d.dll'), 'drivers'),
        (os.path.join(dll_path, 'atmcd64d.dll'), 'drivers'),
    ]

# ============== ANALYSIS ==============
a = Analysis(
    [os.path.join(spec_root, 'main.py')],
    # Fallback to main.py if optimized version doesn't exist
    # [os.path.join(spec_root, 'main.py')],
    pathex=[spec_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludedimports=excluded_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ============== SPLASH SCREEN ==============
# Only added if assets/splash.png exists
splash = None
'''

        if has_splash:
            spec_content += f"""
splash = Splash(
    '{splash_image}',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(10, 10),
    text_size=12,
    minify_script=True,
    always_on_top=True,
)
"""

        spec_content += f"""
# ============== EXE ==============
exe = EXE(
    pyz,
    a.scripts,
    # Exclude binaries from EXE to keep it small and fast-loading
    exclude_binaries=True,
    name='raman_app',
    debug={'all' if self.console_mode else 'False'},
    bootloader_ignore_signals=False,
    strip=False,  # Don't strip, can cause issues on Windows
    upx=True,     # Compress with UPX (reduces size by ~30%)
    upx_exclude=['vcruntime140.dll', 'python*.dll'],  # Don't compress these
    console={'True' if self.console_mode else 'False'},
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icons/app-icon.ico' if os.path.exists('assets/icons/app-icon.ico') else None,
    *([] if splash is None else [splash, splash.binaries])
)

# ============== COLLECT ==============
# Collect everything into a folder (One-Dir mode)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    *([] if splash is None else [splash.binaries]),
    strip=False,
    upx=True,
    upx_exclude=[],
    name='raman_app',
)
"""

        output_path.write_text(spec_content, encoding="utf-8")
        return spec_content

    def generate_nsis_script(self, output_path: Path) -> str:
        """Generate NSIS installer script."""

        # Read version from pyproject.toml if available
        version = "1.0.0"
        pyproject_path = self.project_root / "pyproject.toml"
        if pyproject_path.exists():
            import tomli

            try:
                with open(pyproject_path, "rb") as f:
                    pyproject = tomli.load(f)
                    version = pyproject.get("project", {}).get("version", version)
            except Exception:
                pass

        nsi_content = f"""; NSIS Installer Script for Raman Spectroscopy Analysis Application
; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
; Platform: {self.system_info['platform']}
; Python: {self.system_info['python_version']}
;
; This file was auto-generated by generate_build_configs.py

!define APP_NAME "Raman Spectroscopy Analysis"
!define APP_VERSION "{version}"
!define APP_PUBLISHER "Research Lab"
!define APP_DESCRIPTION "Real-time Raman Spectral Classifier"
!define INSTALL_DIR "$PROGRAMFILES\\RamanApp"
!define APP_EXE "raman_app.exe"

; Modern UI
!include "MUI2.nsh"

; General attributes
Name "${{APP_NAME}}"
OutFile "raman_app_installer.exe"
InstallDir "${{INSTALL_DIR}}"
InstallDirRegKey HKLM "Software\\${{APP_NAME}}" ""
RequestExecutionLevel admin

; Interface Configuration
!define MUI_ABORTWARNING
!define MUI_ICON "assets\\icons\\app-icon.ico"
!define MUI_UNICON "assets\\icons\\app-icon.ico"

; Pages
!insertmacro MUI_PAGE_LICENSE "${{NSISDIR}}\\Docs\\Modern UI\\License.txt"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; Languages
!insertmacro MUI_LANGUAGE "English"
!insertmacro MUI_LANGUAGE "Japanese"

; Installer Section
Section "Install"
    SetOutPath "$INSTDIR"
    
    ; Copy all files from staging directory
    File /r "dist_installer\\raman_app_installer_staging\\*.*"
    
    ; Create shortcuts
    CreateDirectory "$SMPROGRAMS\\${{APP_NAME}}"
    CreateShortCut "$SMPROGRAMS\\${{APP_NAME}}\\${{APP_NAME}}.lnk" "$INSTDIR\\${{APP_EXE}}"
    CreateShortCut "$SMPROGRAMS\\${{APP_NAME}}\\Uninstall.lnk" "$INSTDIR\\Uninstall.exe"
    CreateShortCut "$DESKTOP\\${{APP_NAME}}.lnk" "$INSTDIR\\${{APP_EXE}}"
    
    ; Registry keys
    WriteRegStr HKLM "Software\\${{APP_NAME}}" "" "$INSTDIR"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "DisplayName" "${{APP_NAME}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "UninstallString" "$INSTDIR\\Uninstall.exe"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "DisplayVersion" "${{APP_VERSION}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "Publisher" "${{APP_PUBLISHER}}"
    
    ; Create uninstaller
    WriteUninstaller "$INSTDIR\\Uninstall.exe"
SectionEnd

; Uninstaller Section
Section "Uninstall"
    ; Remove files and folders
    RMDir /r "$INSTDIR"
    
    ; Remove shortcuts
    Delete "$SMPROGRAMS\\${{APP_NAME}}\\*.*"
    RMDir "$SMPROGRAMS\\${{APP_NAME}}"
    Delete "$DESKTOP\\${{APP_NAME}}.lnk"
    
    ; Remove registry keys
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}"
    DeleteRegKey HKLM "Software\\${{APP_NAME}}"
SectionEnd
"""

        output_path.write_text(nsi_content, encoding="utf-8")
        return nsi_content

    def generate_report(self) -> Dict:
        """Generate comprehensive build environment report."""
        return {
            "generated_at": datetime.now().isoformat(),
            "system_info": self.system_info,
            "installed_packages": self.installed_packages,
            "required_imports": self.required_imports,
            "console_mode": self.console_mode,
            "project_root": str(self.project_root),
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate build recommendations based on environment analysis."""
        recommendations = []

        # Check Python version
        py_version = tuple(map(int, self.system_info["python_version"].split(".")[:2]))
        if py_version < (3, 10):
            recommendations.append(
                f"⚠️ Python {self.system_info['python_version']} is older than recommended. Consider upgrading to Python 3.10+"
            )

        # Check PyTorch
        if "torch" not in self.installed_packages:
            recommendations.append(
                "ℹ️ PyTorch not detected. Deep learning features will be unavailable in built executable."
            )

        # Check essential packages
        essential = ["numpy", "pandas", "scipy", "matplotlib", "PySide6"]
        missing = [pkg for pkg in essential if pkg not in self.installed_packages]
        if missing:
            recommendations.append(
                f"❌ Missing essential packages: {', '.join(missing)}"
            )

        # Check platform-specific
        if self.system_info["platform"] == "Windows":
            if self.system_info["bits"] == "32bit":
                recommendations.append(
                    "⚠️ 32-bit Python detected. 64-bit is recommended for large datasets."
                )

        if not recommendations:
            recommendations.append(
                "✅ All checks passed. Environment is optimal for building."
            )

        return recommendations


def main():
    parser = argparse.ArgumentParser(
        description="Generate build configuration files for Raman App",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_build_configs.py
  python generate_build_configs.py --console --output-dir build_scripts
  python generate_build_configs.py --analyze-only
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for generated files (default: current directory)",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Generate spec files with console window enabled (for debugging)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze environment, do not generate files",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Determine project root
    script_dir = Path(__file__).resolve().parent
    project_root = (
        script_dir.parent if script_dir.name == "build_scripts" else script_dir
    )

    print("=" * 70)
    print("Build Configuration Generator")
    print("=" * 70)
    print(f"Project Root: {project_root}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Console Mode: {args.console}")
    print("")

    # Create generator
    generator = BuildConfigGenerator(project_root, console_mode=args.console)

    # Generate report
    report = generator.generate_report()

    # Display system info
    print("System Information:")
    print(
        f"  Platform: {report['system_info']['platform']} {report['system_info']['architecture']}"
    )
    print(
        f"  Python: {report['system_info']['python_version']} ({report['system_info']['bits']})"
    )
    print(f"  Installed Packages: {len(report['installed_packages'])}")
    print("")

    # Display recommendations
    print("Recommendations:")
    for rec in report["recommendations"]:
        print(f"  {rec}")
    print("")

    # Save report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "build_config_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Report saved: {report_path}")

    if args.analyze_only:
        print("\n✓ Analysis complete (no files generated)")
        return 0

    # Generate spec files
    print("\nGenerating Configuration Files:")

    spec_portable = output_dir / "raman_app.spec"
    generator.generate_spec_file(spec_portable, installer_mode=False)
    print(f"  ✓ Generated: {spec_portable}")

    spec_installer = output_dir / "raman_app_installer.spec"
    generator.generate_spec_file(spec_installer, installer_mode=True)
    print(f"  ✓ Generated: {spec_installer}")

    nsi_script = output_dir / "raman_app_installer.nsi"
    generator.generate_nsis_script(nsi_script)
    print(f"  ✓ Generated: {nsi_script}")

    print("\n" + "=" * 70)
    print("✓ Configuration generation complete!")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Review generated spec files")
    print("  2. Build portable: .\\build_portable.ps1")
    print("  3. Build installer: .\\build_installer.ps1")
    print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
