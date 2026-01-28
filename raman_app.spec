# -*- mode: python ; coding: utf-8 -*-
"""
Optimized Spec File with Splash Screen
Created: 2026-01-07 19:43:34
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs
import importlib.util
import os
import sys

spec_root = os.path.abspath(os.getcwd())

# Build toggles controlled by build scripts
build_mode = os.environ.get('RAMAN_BUILD_MODE', 'onefile').strip().lower()
dist_name = os.environ.get('RAMAN_DIST_NAME', 'raman_app').strip()
console_enabled = os.environ.get('RAMAN_CONSOLE', '0').strip().lower() in ('1', 'true', 'yes')
no_upx = os.environ.get('RAMAN_NO_UPX', '0').strip().lower() in ('1', 'true', 'yes')

is_windows = os.name == 'nt'

# ============== CONFIGURATION ==============
block_cipher = None

# Explicitly exclude heavy/unused modules to improve startup speed
# Note: 'unittest' removed because sklearn and onnxruntime require it at runtime
# Note: 'pydoc', 'doctest' removed because seaborn/scipy/numpy/joblib/sympy require them
# Note: 'distutils', 'setuptools', 'pkg_resources', 'packaging' removed - setuptools needs them
# Note: 'wheel' removed - setuptools dependency system requires it
excluded_modules = [
    'tkinter', '_tkinter', 'turtle',
    'test', 'pdb', 'bdb',
    'matplotlib.tests', 'numpy.tests', 'scipy.tests', 'pandas.tests',
    'ipython', 'IPython', 'jedi', 'jupyter', 'notebook',
    'PIL.ImageTk', 'curses',
    'pip',
    'xmlrpc', 'xml.etree.cElementTree',
    'multiprocessing.dummy', 'pydoc_data',
    # Exclude unused PySide6 SQL drivers to suppress build warnings
    'PySide6.QtSql',
    # Exclude unused Qt3D modules
    'PySide6.Qt3DCore', 'PySide6.Qt3DRender', 'PySide6.Qt3DLogic',
    'PySide6.Qt3DInput', 'PySide6.Qt3DAnimation', 'PySide6.Qt3DExtras'
]

# ============== DATA FILES ==============
datas = []
# Add Assets
assets_path = os.path.join(spec_root, 'assets')
if os.path.exists(assets_path):
    datas.append((assets_path, 'assets'))

# Add Configs - CRITICAL for metadata fields display
configs_path = os.path.join(spec_root, 'configs')
if os.path.exists(configs_path):
    datas.append((configs_path, 'configs'))

# Collect essential data
datas += collect_data_files('PySide6')
datas += collect_data_files('matplotlib')

# ============== HIDDEN IMPORTS ==============
hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtOpenGL",
    "PySide6.QtPrintSupport",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "shiboken6",
    "numpy",
    "pandas",
    "scipy",
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
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.figure",
    "matplotlib.widgets",
    "seaborn",
    "imageio",
    "ramanspy",
    "ramanspy.preprocessing",
    "ramanspy.preprocessing.normalise",
    # ramanspy.preprocessing.noise is optional and may not exist in some ramanspy versions
    "ramanspy.preprocessing.baseline",
    "pybaselines",
    "sklearn",
    "sklearn.preprocessing",
    "sklearn.cross_decomposition",
    "sklearn.decomposition",
    "sklearn.discriminant_analysis",
    "sklearn.linear_model",
    "sklearn.covariance",
    "sklearn.ensemble",
    "sklearn.cluster",
    "sklearn.metrics",
    "sklearn.model_selection",
    "tqdm",
    "cloudpickle",
    "joblib",
    "pydantic",
    "splash_screen"
]

_maybe_noise = "ramanspy.preprocessing.noise"
if importlib.util.find_spec(_maybe_noise) is not None:
    hiddenimports.append(_maybe_noise)

# Collect submodules only if necessary (heavy operation)
try:
    hiddenimports += collect_submodules('ramanspy')
    hiddenimports += collect_submodules('pybaselines')
except:
    pass

# Optional: XGBoost (only if installed in build env)
try:
    hiddenimports += collect_submodules('xgboost')
    datas += collect_data_files('xgboost')
except:
    pass

# ============== BINARIES ==============
binaries = []

# Optional: XGBoost dynamic libraries (xgboost.dll, etc.)
try:
    binaries += collect_dynamic_libs('xgboost')
except:
    pass
if is_windows:
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
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ============== SPLASH SCREEN ==============
# Only added if assets/splash.png exists
splash = None

# ============== EXE ==============
_exe_args = [pyz, a.scripts]

# IMPORTANT:
# - onedir builds: binaries/datas are collected by COLLECT into a folder
# - onefile builds: binaries/datas MUST be embedded into the single executable
if build_mode == 'onefile':
    _exe_args += [a.binaries, a.zipfiles, a.datas]
    if splash is not None:
        _exe_args += [splash.binaries]

exe = EXE(
    *_exe_args,
    # Exclude binaries from EXE to keep it small and fast-loading (onedir only)
    exclude_binaries=(build_mode == 'onedir'),
    name=dist_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,  # Don't strip, can cause issues on Windows
    upx=(not no_upx),
    upx_exclude=['vcruntime140.dll', 'python*.dll'],  # Don't compress these
    console=console_enabled,
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icons/app-icon.ico' if os.path.exists('assets/icons/app-icon.ico') else None,
    *([] if splash is None else [splash])
)

# ============== COLLECT ==============
# onedir: Collect everything into a folder. onefile: skip COLLECT.
if build_mode == 'onedir':
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        *([] if splash is None else [splash.binaries]),
        strip=False,
        upx=(not no_upx),
        upx_exclude=[],
        name=dist_name,
    )
