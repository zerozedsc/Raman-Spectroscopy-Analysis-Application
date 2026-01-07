import sys
import os
import pandas as pd
import numpy as np
import ramanspy as rp
import traceback
from typing import Dict, List, Any, Optional, Tuple
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QMessageBox,
    QCheckBox,
    QSlider,
    QTextEdit,
    QScrollArea,
    QFrame,
    QSplitter,
    QProgressBar,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QTreeWidget,
    QTreeWidgetItem,
    QFormLayout,
    QTabWidget,
    QSizePolicy,
    QRadioButton,
    QButtonGroup,
)
from PySide6.QtCore import Signal, Qt, QSize, QThread, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtSvg import QSvgRenderer

# Import utils functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from components.widgets.icons import load_icon, get_icon_path
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QSize
from PySide6.QtGui import QFont, QIcon

from utils import *
from components.widgets.matplotlib_widget import MatplotlibWidget, plot_spectra
from functions.preprocess import PREPROCESSING_REGISTRY, EnhancedRamanPipeline
