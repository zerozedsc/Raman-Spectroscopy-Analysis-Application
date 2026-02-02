"""
Parameter Widget Factory

This module provides functions to create parameter input widgets
dynamically based on parameter specifications.
"""

from PySide6.QtWidgets import QWidget, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox
from typing import Dict, Any


def create_parameter_widgets(param_info: Dict[str, Any]) -> QWidget:
    """
    Create an appropriate widget based on parameter type specification.

    Args:
        param_info: Parameter specification dictionary with keys:
            - type: Widget type ('spinbox', 'double_spinbox', 'combo', 'checkbox')
            - default: Default value
            - range: (min, max) for numeric widgets
            - step: Step size for numeric widgets
            - options: List of options for combo widgets
            - label: Display label (not used here)

    Returns:
        Configured QWidget for the parameter
    """
    param_type = param_info.get("type")

    if param_type == "spinbox":
        widget = QSpinBox()
        min_val, max_val = param_info.get("range", (0, 100))
        widget.setRange(min_val, max_val)
        widget.setValue(param_info.get("default", min_val))
        return widget

    elif param_type == "double_spinbox":
        widget = QDoubleSpinBox()
        min_val, max_val = param_info.get("range", (0.0, 1.0))
        widget.setRange(min_val, max_val)
        widget.setValue(param_info.get("default", min_val))

        # Set step size if provided
        step = param_info.get("step", 0.1)
        widget.setSingleStep(step)

        # Set decimals based on step size
        if step >= 1:
            widget.setDecimals(0)
        elif step >= 0.1:
            widget.setDecimals(1)
        elif step >= 0.01:
            widget.setDecimals(2)
        else:
            widget.setDecimals(3)

        return widget

    elif param_type == "combo":
        widget = QComboBox()
        options = param_info.get("options", [])

        for option in options:
            widget.addItem(str(option), option)

        # Set default value
        default = param_info.get("default")
        if default:
            index = widget.findData(default)
            if index >= 0:
                widget.setCurrentIndex(index)

        return widget

    elif param_type == "checkbox":
        widget = QCheckBox()
        widget.setChecked(param_info.get("default", False))
        return widget

    else:
        # Fallback: return a disabled widget
        widget = QWidget()
        widget.setEnabled(False)
        return widget


def get_widget_value(widget: QWidget) -> Any:
    """
    Extract the current value from a parameter widget.

    Args:
        widget: Parameter widget created by create_parameter_widgets

    Returns:
        Current value of the widget
    """
    if isinstance(widget, QSpinBox):
        return widget.value()

    elif isinstance(widget, QDoubleSpinBox):
        return widget.value()

    elif isinstance(widget, QComboBox):
        return widget.currentData() or widget.currentText()

    elif isinstance(widget, QCheckBox):
        return widget.isChecked()

    else:
        return None


def set_widget_value(widget: QWidget, value: Any):
    """
    Set the value of a parameter widget.

    Args:
        widget: Parameter widget created by create_parameter_widgets
        value: Value to set
    """
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.setValue(value)

    elif isinstance(widget, QComboBox):
        index = widget.findData(value)
        if index >= 0:
            widget.setCurrentIndex(index)
        else:
            # Try to find by text
            index = widget.findText(str(value))
            if index >= 0:
                widget.setCurrentIndex(index)

    elif isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
