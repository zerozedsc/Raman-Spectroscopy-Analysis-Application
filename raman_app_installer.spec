# -*- mode: python ; coding: utf-8 -*-
"""
Optimized Spec File with Splash Screen
Generated: 2025-11-21 13:40:31
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
    "ramanspy.preprocessing.noise",
    "ramanspy.preprocessing.baseline",
    "pybaselines",
    "sklearn",
    "sklearn.preprocessing",
    "sklearn.decomposition",
    "sklearn.linear_model",
    "sklearn.ensemble",
    "sklearn.metrics",
    "sklearn.model_selection",
    "tqdm",
    "cloudpickle",
    "joblib",
    "pydantic",
    "splash_screen"
]

# Collect submodules only if necessary (heavy operation)
try:
    hiddenimports += collect_submodules('ramanspy')
    hiddenimports += collect_submodules('pybaselines')
except:
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
    hooksconfig={},
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

# ============== EXE ==============
exe = EXE(
    pyz,
    a.scripts,
    # Exclude binaries from EXE to keep it small and fast-loading
    exclude_binaries=True,
    name='raman_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,  # Don't strip, can cause issues on Windows
    upx=True,     # Compress with UPX (reduces size by ~30%)
    upx_exclude=['vcruntime140.dll', 'python*.dll'],  # Don't compress these
    console=False,
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/app_icon.ico' if os.path.exists('assets/app_icon.ico') else None,
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
