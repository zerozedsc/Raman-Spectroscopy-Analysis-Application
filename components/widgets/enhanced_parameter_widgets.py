"""
Enhanced Parameter Widgets with Constraint Hints

This module extends the basic parameter widgets to include constraint validation
and user hints for preprocessing parameters.
"""

from .parameter_widgets import CustomSpinBox, CustomDoubleSpinBox
from .utils import *
from .icons import create_button_icon
import sys
import os

# Add the functions directory to path for importing constraint analyzer
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "functions"))

try:
    from preprocess.parameter_constraints import ParameterConstraints

    CONSTRAINTS_AVAILABLE = True
except ImportError:
    CONSTRAINTS_AVAILABLE = False


class ConstrainedSpinBox(QWidget):
    """Enhanced spinbox with parameter constraint validation and hints."""

    valueChanged = Signal(int)
    constraintViolated = Signal(str)  # Emitted when constraint is violated

    def __init__(self, parameter_name: str = None, parent=None):
        super().__init__(parent)
        self.parameter_name = parameter_name
        self.constraints = ParameterConstraints() if CONSTRAINTS_AVAILABLE else None
        self._value = 0
        self._setup_ui()
        self._apply_constraints()

    def _setup_ui(self):
        """Setup the main UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)

        # Spinbox container
        spinbox_container = QWidget()
        layout = QHBoxLayout(spinbox_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Minus button
        self.minus_btn = QPushButton()
        self.minus_btn.setFixedSize(24, 24)
        self.minus_btn.setIcon(create_button_icon("minus"))
        self.minus_btn.setStyleSheet(self._button_style())
        self.minus_btn.clicked.connect(self._decrease_value)

        # Value input
        self.value_input = QLineEdit()
        self.value_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_input.setFixedHeight(24)
        self.value_input.setStyleSheet(self._input_style())
        self.value_input.editingFinished.connect(self._on_text_changed)

        # Plus button
        self.plus_btn = QPushButton()
        self.plus_btn.setFixedSize(24, 24)
        self.plus_btn.setIcon(create_button_icon("plus"))
        self.plus_btn.setStyleSheet(self._button_style())
        self.plus_btn.clicked.connect(self._increase_value)

        layout.addWidget(self.minus_btn)
        layout.addWidget(self.value_input)
        layout.addWidget(self.plus_btn)

        # Hint label
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(
            """
            QLabel {
                color: #6c757d;
                font-size: 10px;
                padding: 2px 4px;
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 3px;
                margin-top: 2px;
            }
        """
        )

        main_layout.addWidget(spinbox_container)
        main_layout.addWidget(self.hint_label)

    def _apply_constraints(self):
        """Apply parameter constraints if available."""
        if not self.constraints or not self.parameter_name:
            self.hint_label.hide()
            return

        constraint_info = self.constraints.get_constraint_info(self.parameter_name)
        if not constraint_info:
            self.hint_label.hide()
            return

        # Set min/max values
        if "min" in constraint_info:
            self._minimum = int(constraint_info["min"])
        if "max" in constraint_info:
            self._maximum = int(constraint_info["max"])
        if "default" in constraint_info:
            self._value = int(constraint_info["default"])
            self.value_input.setText(str(self._value))

        # Set step size
        step = self.constraints.get_step_size(self.parameter_name)
        if isinstance(step, int):
            self._step = step

        # Update hint
        self._update_hint()

    def _update_hint(self):
        """Update the hint label with current parameter information."""
        if not self.constraints or not self.parameter_name:
            return

        hint_text = self.constraints.get_parameter_hint(
            self.parameter_name, self._value
        )

        # Add constraint-specific styling
        is_valid, error_msg = self.constraints.validate_parameter(
            self.parameter_name, self._value
        )
        if not is_valid:
            self.hint_label.setStyleSheet(
                """
                QLabel {
                    color: #dc3545;
                    font-size: 10px;
                    padding: 2px 4px;
                    background-color: #f8d7da;
                    border: 1px solid #f5c6cb;
                    border-radius: 3px;
                    margin-top: 2px;
                }
            """
            )
            self.constraintViolated.emit(error_msg)
        else:
            self.hint_label.setStyleSheet(
                """
                QLabel {
                    color: #155724;
                    font-size: 10px;
                    padding: 2px 4px;
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    border-radius: 3px;
                    margin-top: 2px;
                }
            """
            )

        self.hint_label.setText(hint_text)

    def _button_style(self) -> str:
        """Get button stylesheet."""
        return """
            QPushButton {
                background-color: #f8f9fa;
                border: 1px solid #6c757d;
                border-radius: 12px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #6c757d;
            }
            QPushButton:pressed {
                background-color: #5a6268;
            }
            QPushButton:disabled {
                background-color: #f8f9fa;
                border-color: #bdc3c7;
            }
        """

    def _input_style(self) -> str:
        """Get input field stylesheet."""
        return """
            QLineEdit {
                padding: 4px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                background-color: white;
                font-size: 12px;
                text-align: center;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """

    def _increase_value(self):
        """Increase the current value."""
        new_value = self._value + self._step
        if new_value <= self._maximum:
            self.set_value(new_value)

    def _decrease_value(self):
        """Decrease the current value."""
        new_value = self._value - self._step
        if new_value >= self._minimum:
            self.set_value(new_value)

    def _on_text_changed(self):
        """Handle manual text input."""
        try:
            value = int(self.value_input.text())
            self.set_value(value)
        except ValueError:
            # Reset to current value if invalid input
            self.value_input.setText(str(self._value))

    def set_value(self, value: int):
        """Set the spinbox value."""
        # Apply constraints
        value = max(self._minimum, min(self._maximum, value))

        if value != self._value:
            self._value = value
            self.value_input.setText(str(value))
            self._update_hint()
            self.valueChanged.emit(value)

    def value(self) -> int:
        """Get the current value."""
        return self._value

    def set_range(self, minimum: int, maximum: int):
        """Set the value range."""
        self._minimum = minimum
        self._maximum = maximum
        # Ensure current value is within range
        self.set_value(self._value)


class ParameterHintWidget(QWidget):
    """Standalone widget for displaying parameter hints and constraints."""

    def __init__(self, parameter_name: str, parent=None):
        super().__init__(parent)
        self.parameter_name = parameter_name
        self.constraints = ParameterConstraints() if CONSTRAINTS_AVAILABLE else None
        self._setup_ui()

    def _setup_ui(self):
        """Setup the hint display UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        if not self.constraints or not self.parameter_name:
            no_info_label = QLabel("No constraint information available")
            no_info_label.setStyleSheet("color: #6c757d; font-style: italic;")
            layout.addWidget(no_info_label)
            return

        constraint_info = self.constraints.get_constraint_info(self.parameter_name)
        if not constraint_info:
            no_info_label = QLabel("Parameter not found in constraint database")
            no_info_label.setStyleSheet("color: #6c757d; font-style: italic;")
            layout.addWidget(no_info_label)
            return

        # Parameter description
        if "description" in constraint_info:
            desc_label = QLabel(constraint_info["description"])
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("font-weight: bold; color: #495057;")
            layout.addWidget(desc_label)

        # Range information
        range_info = []
        if "min" in constraint_info:
            range_info.append(f"Min: {constraint_info['min']}")
        if "max" in constraint_info:
            range_info.append(f"Max: {constraint_info['max']}")
        if "typical_range" in constraint_info:
            min_val, max_val = constraint_info["typical_range"]
            range_info.append(f"Typical: {min_val} - {max_val}")

        if range_info:
            range_label = QLabel(" | ".join(range_info))
            range_label.setStyleSheet("color: #6c757d; font-size: 11px;")
            layout.addWidget(range_label)

        # Constraints
        constraints = constraint_info.get("constraints", [])
        if constraints:
            constraint_text = "Constraints: " + ", ".join(
                [c.replace("_", " ").title() for c in constraints]
            )
            constraint_label = QLabel(constraint_text)
            constraint_label.setWordWrap(True)
            constraint_label.setStyleSheet("color: #dc3545; font-size: 10px;")
            layout.addWidget(constraint_label)

        # Hint
        if "hint" in constraint_info:
            hint_label = QLabel(constraint_info["hint"])
            hint_label.setWordWrap(True)
            hint_label.setStyleSheet(
                """
                QLabel {
                    color: #155724;
                    font-size: 11px;
                    padding: 4px;
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    border-radius: 3px;
                    margin-top: 4px;
                }
            """
            )
            layout.addWidget(hint_label)
