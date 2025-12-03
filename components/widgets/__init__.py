"""
Enhanced Parameter Widgets Package

This package provides reusable, professional-grade parameter input widgets
optimized for scientific applications, particularly Raman spectroscopy.

Widgets included:
- CustomSpinBox: Integer input with SVG +/- buttons
- CustomDoubleSpinBox: Float input with SVG +/- buttons  
- RangeParameterWidget: Dual-range input with slider
- ParameterWidget: Base parameter input widget
- ParameterGroupWidget: Grouped parameter controls

Features:
- Professional SVG icons (decrease-circle.svg, increase-circle.svg)
- Real-time parameter validation
- Medical/scientific styling
- Extensible parameter system
- Japanese/English localization support
"""

from .parameter_widgets import (
    CustomSpinBox,
    CustomDoubleSpinBox,
    RangeParameterWidget,
    ParameterWidget,
    ParameterGroupWidget,
    DynamicParameterWidget
)
from .views_widget import GroupTreeManager

from .matplotlib_widget import MatplotlibWidget, plot_spectra
from .icons import (
    load_icon, 
    create_button_icon, 
    create_toolbar_icon,
    get_icon_path,
    list_available_icons,
    ICON_PATHS  # Import ICON_PATHS
)

__all__ = [
    'CustomSpinBox',
    'CustomDoubleSpinBox', 
    'RangeParameterWidget',
    'ParameterWidget',
    'ParameterGroupWidget',
    'DynamicParameterWidget',
    'MatplotlibWidget',
    'plot_spectra',
    'load_icon',
    'create_button_icon',
    'create_toolbar_icon',
    'get_icon_path',
    'list_available_icons',
    'GroupTreeManager',
    'ICON_PATHS'  # Add ICON_PATHS to exports
]