from .__utils__ import *


class CustomSpinBox(QWidget):
    """Custom spinbox with +/- buttons instead of up/down arrows."""

    valueChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0
        self._minimum = 0
        self._maximum = 99
        self._step = 1
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Minus button with SVG icon
        self.minus_btn = QPushButton()
        self.minus_btn.setFixedSize(24, 24)
        self.minus_btn.setIcon(load_icon("minus", QSize(16, 16), None))
        self.minus_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f8f9fa;
                border: 1px solid #e74c3c;
                border-radius: 12px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #e74c3c;
            }
            QPushButton:pressed {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #f8f9fa;
                border-color: #bdc3c7;
            }
        """
        )
        self.minus_btn.clicked.connect(self._decrease_value)

        # Value display/input
        self.value_input = QLineEdit()
        self.value_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_input.setFixedHeight(24)
        self.value_input.setStyleSheet(
            """
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
        )
        self.value_input.editingFinished.connect(self._on_text_changed)

        # Plus button with SVG icon
        self.plus_btn = QPushButton()
        self.plus_btn.setFixedSize(24, 24)
        self.plus_btn.setIcon(load_icon("plus", QSize(16, 16), "#27ae60"))
        self.plus_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f8f9fa;
                border: 1px solid #27ae60;
                border-radius: 12px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #f8f9fa;
                border-color: #bdc3c7;
            }
        """
        )
        self.plus_btn.clicked.connect(self._increase_value)

        layout.addWidget(self.minus_btn)
        layout.addWidget(self.value_input)
        layout.addWidget(self.plus_btn)

        self._update_display()
        self._update_button_states()

    def _decrease_value(self):
        new_value = max(self._minimum, self._value - self._step)
        self.setValue(new_value)

    def _increase_value(self):
        new_value = min(self._maximum, self._value + self._step)
        self.setValue(new_value)

    def _on_text_changed(self):
        try:
            new_value = int(self.value_input.text())
            new_value = max(self._minimum, min(self._maximum, new_value))
            self.setValue(new_value)
        except ValueError:
            self._update_display()

    def _update_display(self):
        self.value_input.setText(str(self._value))

    def _update_button_states(self):
        self.minus_btn.setEnabled(self._value > self._minimum)
        self.plus_btn.setEnabled(self._value < self._maximum)

    def setValue(self, value: int):
        value = max(self._minimum, min(self._maximum, value))
        if value != self._value:
            self._value = value
            self._update_display()
            self._update_button_states()
            self.valueChanged.emit(self._value)

    def value(self) -> int:
        return self._value

    def setRange(self, minimum: int, maximum: int):
        self._minimum = minimum
        self._maximum = maximum
        self.setValue(max(minimum, min(maximum, self._value)))

    def setSingleStep(self, step: int):
        self._step = step

    def setStyleSheet(self, style: str):
        # Override to prevent external styling from breaking our design
        pass


class CustomDoubleSpinBox(QWidget):
    """Custom double spinbox with +/- buttons instead of up/down arrows."""

    valueChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._minimum = 0.0
        self._maximum = 99.9
        self._step = 0.1
        self._decimals = 1
        self._suffix = ""
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Minus button with SVG icon
        self.minus_btn = QPushButton()
        self.minus_btn.setFixedSize(24, 24)
        self.minus_btn.setIcon(load_icon("minus", QSize(16, 16), "#e74c3c"))
        self.minus_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f8f9fa;
                border: 1px solid #e74c3c;
                border-radius: 12px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #e74c3c;
            }
            QPushButton:pressed {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #f8f9fa;
                border-color: #bdc3c7;
            }
        """
        )
        self.minus_btn.clicked.connect(self._decrease_value)

        # Value display/input
        self.value_input = QLineEdit()
        self.value_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_input.setFixedHeight(24)
        self.value_input.setStyleSheet(
            """
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
        )
        self.value_input.editingFinished.connect(self._on_text_changed)

        # Plus button with SVG icon
        self.plus_btn = QPushButton()
        self.plus_btn.setFixedSize(24, 24)
        self.plus_btn.setIcon(load_icon("plus", QSize(16, 16), "#27ae60"))
        self.plus_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f8f9fa;
                border: 1px solid #27ae60;
                border-radius: 12px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #f8f9fa;
                border-color: #bdc3c7;
            }
        """
        )
        self.plus_btn.clicked.connect(self._increase_value)

        layout.addWidget(self.minus_btn)
        layout.addWidget(self.value_input)
        layout.addWidget(self.plus_btn)

        self._update_display()
        self._update_button_states()

    def _decrease_value(self):
        new_value = max(self._minimum, self._value - self._step)
        self.setValue(new_value)

    def _increase_value(self):
        new_value = min(self._maximum, self._value + self._step)
        self.setValue(new_value)

    def _on_text_changed(self):
        try:
            text = self.value_input.text()
            if self._suffix:
                text = text.replace(self._suffix, "").strip()
            new_value = float(text)
            new_value = max(self._minimum, min(self._maximum, new_value))
            self.setValue(new_value)
        except ValueError:
            self._update_display()

    def _update_display(self):
        text = f"{self._value:.{self._decimals}f}{self._suffix}"
        self.value_input.setText(text)

    def _update_button_states(self):
        self.minus_btn.setEnabled(self._value > self._minimum)
        self.plus_btn.setEnabled(self._value < self._maximum)

    def setValue(self, value: float):
        value = max(self._minimum, min(self._maximum, value))
        if abs(value - self._value) > 1e-9:  # Use small epsilon for float comparison
            self._value = value
            self._update_display()
            self._update_button_states()
            self.valueChanged.emit(self._value)

    def value(self) -> float:
        return self._value

    def setRange(self, minimum: float, maximum: float):
        self._minimum = minimum
        self._maximum = maximum
        self.setValue(max(minimum, min(maximum, self._value)))

    def setSingleStep(self, step: float):
        self._step = step

    def setDecimals(self, decimals: int):
        self._decimals = decimals
        self._update_display()

    def setSuffix(self, suffix: str):
        self._suffix = suffix
        self._update_display()

    def setToolTip(self, tooltip: str):
        super().setToolTip(tooltip)
        self.value_input.setToolTip(tooltip)

    def setMinimumWidth(self, width: int):
        super().setMinimumWidth(width)

    def setStyleSheet(self, style: str):
        # Override to prevent external styling from breaking our design
        pass


class RangeParameterWidget(QWidget):
    """Widget for dual input range parameters with slider (like Cropper region)."""

    # Signal emitted when parameters change
    parametersChanged = Signal()
    # Signal emitted for immediate real-time updates (no debouncing)
    realTimeUpdate = Signal()

    def __init__(
        self,
        param_name: str,
        info: Dict[str, Any],
        default_value: Any = None,
        data_range: tuple = None,
        parent=None,
    ):
        super().__init__(parent)
        self.param_name = param_name
        self.info = info
        self.setObjectName("rangeParameterWidget")

        # Get range limits - prefer data_range if provided, otherwise use info range
        if data_range is not None:
            self.range_min, self.range_max = data_range
        else:
            self.range_limits = info.get("range", [400, 4000])
            if isinstance(self.range_limits[0], (tuple, list)):
                self.range_limits = self.range_limits[0]  # Handle nested range format
            self.range_min, self.range_max = self.range_limits

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Input boxes row
        input_layout = QHBoxLayout()
        input_layout.setSpacing(8)

        # Min value input
        self.min_input = CustomDoubleSpinBox()
        self.min_input.setRange(self.range_min, self.range_max)
        self.min_input.setDecimals(1)
        self.min_input.setToolTip(
            LOCALIZE("PREPROCESS.UI.range_min_tooltip", param_name=param_name)
        )
        self.min_input.setMinimumWidth(120)

        # Range separator
        separator = QLabel(LOCALIZE("PREPROCESS.UI.range_separator"))
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        separator.setStyleSheet("font-weight: bold; color: #7f8c8d; font-size: 14px;")

        # Max value input
        self.max_input = CustomDoubleSpinBox()
        self.max_input.setRange(self.range_min, self.range_max)
        self.max_input.setDecimals(1)
        self.max_input.setToolTip(
            LOCALIZE("PREPROCESS.UI.range_max_tooltip", param_name=param_name)
        )
        self.max_input.setMinimumWidth(120)

        # Unit label
        unit_label = QLabel(LOCALIZE("PREPROCESS.UI.unit_wavenumber"))
        unit_label.setStyleSheet("font-weight: 500; color: #666; margin-left: 8px;")

        input_layout.addWidget(self.min_input)
        input_layout.addWidget(separator)
        input_layout.addWidget(self.max_input)
        input_layout.addWidget(unit_label)
        input_layout.addStretch()

        layout.addLayout(input_layout)

        # Slider section
        slider_frame = QFrame()
        slider_frame.setFrameStyle(QFrame.Shape.Box)
        slider_frame.setLineWidth(1)
        slider_frame.setStyleSheet(
            """
            QFrame {
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                background-color: #f8f9fa;
                padding: 8px;
            }
        """
        )

        slider_layout = QVBoxLayout(slider_frame)
        slider_layout.setContentsMargins(8, 8, 8, 8)
        slider_layout.setSpacing(4)

        # Slider container
        slider_row = QHBoxLayout()
        slider_row.setSpacing(4)

        # Min slider
        self.min_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_slider.setRange(int(self.range_min), int(self.range_max))
        self.min_slider.setFixedHeight(20)
        self.min_slider.setToolTip(LOCALIZE("PREPROCESS.UI.slider_min_tooltip"))

        # Max slider
        self.max_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_slider.setRange(int(self.range_min), int(self.range_max))
        self.max_slider.setFixedHeight(20)
        self.max_slider.setToolTip(LOCALIZE("PREPROCESS.UI.slider_max_tooltip"))

        slider_row.addWidget(self.min_slider)
        slider_row.addWidget(self.max_slider)

        # Range labels
        range_labels = QHBoxLayout()
        range_labels.setContentsMargins(0, 0, 0, 0)
        min_label = QLabel(f"{self.range_min:.0f}")
        min_label.setStyleSheet("font-size: 10px; color: #7f8c8d;")
        max_label = QLabel(f"{self.range_max:.0f}")
        max_label.setStyleSheet("font-size: 10px; color: #7f8c8d;")
        max_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        range_labels.addWidget(min_label)
        range_labels.addStretch()
        range_labels.addWidget(max_label)

        slider_layout.addLayout(slider_row)
        slider_layout.addLayout(range_labels)

        layout.addWidget(slider_frame)

        # Set default values - ensure they're within the valid range
        if (
            default_value is not None
            and isinstance(default_value, (tuple, list))
            and len(default_value) == 2
        ):
            # Clamp default values to the actual data range
            min_val = max(float(default_value[0]), self.range_min)
            max_val = min(float(default_value[1]), self.range_max)

            self.min_input.setValue(min_val)
            self.max_input.setValue(max_val)
            self.min_slider.setValue(int(min_val))
            self.max_slider.setValue(int(max_val))
        else:
            # Use sensible defaults within the actual data range
            mid_point = (self.range_min + self.range_max) / 2
            quarter_range = (self.range_max - self.range_min) / 4
            default_min = max(mid_point - quarter_range, self.range_min)
            default_max = min(mid_point + quarter_range, self.range_max)

            self.min_input.setValue(default_min)
            self.max_input.setValue(default_max)
            self.min_slider.setValue(int(default_min))
            self.max_slider.setValue(int(default_max))

        # Connect signals
        self.min_input.valueChanged.connect(self._on_min_input_changed)
        self.max_input.valueChanged.connect(self._on_max_input_changed)
        self.min_slider.valueChanged.connect(self._on_min_slider_changed)
        self.max_slider.valueChanged.connect(self._on_max_slider_changed)

        # Apply styling
        self._apply_styling()

    def _apply_styling(self):
        """Apply consistent styling."""
        # Custom widgets handle their own styling
        slider_style = """
            QSlider::groove:horizontal {
                border: 1px solid #bdc3c7;
                height: 6px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f6f7fa, stop:1 #dadbde);
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3498db, stop:1 #2980b9);
                border: 1px solid #2c3e50;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5dade2, stop:1 #3498db);
            }
        """

        # Custom widgets handle their own styling, only style sliders
        self.min_slider.setStyleSheet(slider_style)
        self.max_slider.setStyleSheet(slider_style)

    def _on_min_input_changed(self, value):
        """Handle min input change."""
        if value >= self.max_input.value():
            self.min_input.setValue(self.max_input.value() - 0.1)
            return
        self.min_slider.setValue(int(value))
        self.parametersChanged.emit()

    def _on_max_input_changed(self, value):
        """Handle max input change."""
        if value <= self.min_input.value():
            self.max_input.setValue(self.min_input.value() + 0.1)
            return
        self.max_slider.setValue(int(value))
        self.parametersChanged.emit()

    def _on_min_slider_changed(self, value):
        """Handle min slider change."""
        if value >= self.max_slider.value():
            self.min_slider.setValue(self.max_slider.value() - 1)
            return
        self.min_input.setValue(float(value))
        self.parametersChanged.emit()
        self.realTimeUpdate.emit()  # Immediate update for slider

    def _on_max_slider_changed(self, value):
        """Handle max slider change."""
        if value <= self.min_slider.value():
            self.max_slider.setValue(self.min_slider.value() + 1)
            return
        self.max_input.setValue(float(value))
        self.parametersChanged.emit()
        self.realTimeUpdate.emit()  # Immediate update for slider

    def get_value(self) -> tuple:
        """Get the current range as a tuple."""
        min_val = self.min_input.value()
        max_val = self.max_input.value()
        return (min_val, max_val)

    def set_value(self, value: tuple):
        """Set the range values."""
        if isinstance(value, (tuple, list)) and len(value) == 2:
            # Clamp values to valid range
            min_val = max(float(value[0]), self.range_min)
            max_val = min(float(value[1]), self.range_max)

            self.min_input.setValue(min_val)
            self.max_input.setValue(max_val)
            self.min_slider.setValue(int(min_val))
            self.max_slider.setValue(int(max_val))


class DictParameterWidget(QWidget):
    """Widget for dictionary parameters with add/remove functionality."""

    # Signal emitted when parameters change
    parametersChanged = Signal()

    def __init__(
        self,
        param_name: str,
        info: Dict[str, Any],
        default_value: Any = None,
        parent=None,
    ):
        super().__init__(parent)
        self.param_name = param_name
        self.info = info
        self.entries = []
        self.setObjectName("dictParameterWidget")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Container for entries
        self.entries_container = QWidget()
        self.entries_layout = QVBoxLayout(self.entries_container)
        self.entries_layout.setContentsMargins(0, 0, 0, 0)
        self.entries_layout.setSpacing(4)

        # Scrollable area for entries
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.entries_container)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(150)
        scroll_area.setStyleSheet(
            """
            QScrollArea {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: #f8f9fa;
            }
        """
        )

        # Add button
        self.add_button = QPushButton("➕ Add Entry")
        self.add_button.setObjectName("addEntryButton")
        self.add_button.clicked.connect(self.add_entry)
        self.add_button.setStyleSheet(
            """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """
        )

        layout.addWidget(scroll_area)
        layout.addWidget(self.add_button)

        # Initialize with default values
        if default_value and isinstance(default_value, dict):
            for key, value in default_value.items():
                self.add_entry(key, value)
        else:
            # Add one empty entry by default
            self.add_entry()

    def add_entry(self, key: str = "", value: Any = ""):
        """Add a new key-value entry."""
        entry_widget = QWidget()
        entry_layout = QHBoxLayout(entry_widget)
        entry_layout.setContentsMargins(4, 4, 4, 4)
        entry_layout.setSpacing(8)

        # Key input
        key_input = QLineEdit(str(key))
        key_input.setPlaceholderText(LOCALIZE("PREPROCESS.UI.dict_key_placeholder"))
        key_input.setStyleSheet(
            """
            QLineEdit {
                padding: 4px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                background-color: white;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """
        )

        # Separator
        separator = QLabel(LOCALIZE("PREPROCESS.UI.dict_separator"))
        separator.setStyleSheet("font-weight: bold; color: #7f8c8d;")

        # Value input
        value_input = CustomDoubleSpinBox()
        value_input.setRange(0, 10000)
        value_input.setDecimals(1)
        value_input.setValue(
            float(value) if str(value).replace(".", "").isdigit() else 520.5
        )
        value_input.setSuffix(" cm⁻¹")
        value_input.valueChanged.connect(self.parametersChanged.emit)

        # Connect key input changes to signal
        key_input.textChanged.connect(self.parametersChanged.emit)

        # Remove button with SVG icon
        remove_button = QPushButton()
        remove_button.setFixedSize(24, 24)
        remove_button.setToolTip("Remove this entry")
        remove_button.clicked.connect(lambda: self.remove_entry(entry_widget))

        # Load and set the trash icon
        icon_path = "assets/icons/trash-xmark.svg"
        if os.path.exists(icon_path):
            remove_button.setIcon(QIcon(icon_path))
            remove_button.setIconSize(QSize(16, 16))
        else:
            # Fallback to emoji if icon not found
            remove_button.setText(LOCALIZE("PREPROCESS.UI.remove_item_button"))

        remove_button.setStyleSheet(
            """
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """
        )

        entry_layout.addWidget(key_input, 1)
        entry_layout.addWidget(separator)
        entry_layout.addWidget(value_input, 1)
        entry_layout.addWidget(remove_button)

        self.entries_layout.addWidget(entry_widget)
        self.entries.append(
            {"widget": entry_widget, "key_input": key_input, "value_input": value_input}
        )

    def remove_entry(self, entry_widget: QWidget):
        """Remove an entry."""
        # Find and remove the entry
        for i, entry in enumerate(self.entries):
            if entry["widget"] == entry_widget:
                entry_widget.deleteLater()
                del self.entries[i]
                break

        # Ensure at least one entry remains
        if not self.entries:
            self.add_entry()

        # Emit signal that parameters changed
        self.parametersChanged.emit()

    def get_value(self) -> dict:
        """Get the current dictionary value."""
        result = {}
        for entry in self.entries:
            key = entry["key_input"].text().strip()
            value = entry["value_input"].value()
            if key:  # Only add entries with non-empty keys
                result[key] = value
        return result if result else None

    def set_value(self, value: dict):
        """Set the dictionary value."""
        # Clear existing entries
        for entry in self.entries[:]:
            entry["widget"].deleteLater()
        self.entries.clear()

        # Add new entries
        if value and isinstance(value, dict):
            for key, val in value.items():
                self.add_entry(key, val)
        else:
            self.add_entry()


class DynamicParameterWidget(QWidget):
    """Dynamic parameter widget that creates UI controls based on parameter info."""

    def __init__(
        self,
        method_info: Dict[str, Any],
        saved_params: Dict[str, Any] = None,
        data_range: tuple = None,
        parent=None,
    ):
        super().__init__(parent)
        self.method_info = method_info
        self.saved_params = saved_params or {}
        self.param_widgets = {}
        self.data_range = data_range  # Store actual data range for tuple parameters
        self.setObjectName("dynamicParameterWidget")
        self._setup_ui()

    def set_data_range(self, data_range: tuple):
        """Set the data range for tuple parameters and refresh UI."""
        self.data_range = data_range
        self._setup_ui()  # Refresh UI with new data range

    def _setup_ui(self):
        # Clear any existing layout
        if self.layout():
            while self.layout().count():
                child = self.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            self.layout().deleteLater()

        layout = QFormLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        param_info = self.method_info.get("param_info", {})
        default_params = self.method_info.get("default_params", {})

        if not param_info:
            # No parameters
            label = QLabel(LOCALIZE("PREPROCESS.no_parameters"))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: #666; font-style: italic;")
            layout.addRow(label)
            return

        # Add parameters with saved values taking precedence
        for param_name, info in param_info.items():
            # Use saved parameter value if available, otherwise use default
            param_value = self.saved_params.get(
                param_name, default_params.get(param_name)
            )
            widget = self._create_parameter_widget(param_name, info, param_value)
            if widget:
                label = QLabel(f"{param_name}:")
                label.setToolTip(info.get("description", ""))
                layout.addRow(label, widget)
                self.param_widgets[param_name] = widget

        # Force layout update
        self.updateGeometry()
        self.update()

    def _create_parameter_widget(
        self, param_name: str, info: Dict[str, Any], default_value: Any
    ) -> QWidget:
        """Create appropriate widget based on parameter type."""
        param_type = info.get("type", "float")

        if param_type == "int":
            widget = CustomSpinBox()
            range_info = info.get("range", [0, 100])
            widget.setRange(range_info[0], range_info[1])
            if "step" in info:
                widget.setSingleStep(info["step"])
            if default_value is not None:
                widget.setValue(int(default_value))
            return widget

        elif param_type == "float":
            widget = CustomDoubleSpinBox()
            range_info = info.get("range", [0.0, 1.0])
            widget.setRange(range_info[0], range_info[1])
            # Set default to 6 decimals for better visual appearance
            widget.setDecimals(6)
            if "step" in info:
                widget.setSingleStep(info["step"])
            if default_value is not None:
                widget.setValue(float(default_value))
            return widget

        elif param_type == "scientific":
            widget = CustomDoubleSpinBox()
            range_info = info.get("range", [1e-9, 1e12])
            widget.setRange(range_info[0], range_info[1])
            # Set default to 6 decimals for scientific parameters
            # Users can still input more precision if needed
            widget.setDecimals(6)
            if default_value is not None:
                widget.setValue(float(default_value))
            return widget
            
        elif param_type == "choice" or param_type == "combo":
            widget = QComboBox()
            choices = info.get("choices", [])
            widget.addItems(choices)
            if default_value is not None and default_value in choices:
                widget.setCurrentText(str(default_value))
            widget.setStyleSheet(
                """
                QComboBox {
                    padding: 6px;
                    border: 2px solid #bdc3c7;
                    border-radius: 4px;
                    background-color: white;
                    font-size: 12px;
                }
                QComboBox:focus {
                    border-color: #3498db;
                }
                QComboBox::drop-down {
                    border: none;
                }
                QComboBox::down-arrow {
                    image: url(down_arrow.png);
                    width: 12px;
                    height: 12px;
                }
            """
            )
            return widget
            
        elif param_type == "radio":
            # Create radio button group for mutually exclusive options
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
            
            choices = info.get("choices", [])
            descriptions = info.get("descriptions", {})
            
            button_group = QButtonGroup(widget)
            
            for idx, choice in enumerate(choices):
                radio_container = QWidget()
                radio_layout = QVBoxLayout(radio_container)
                radio_layout.setContentsMargins(0, 0, 0, 0)
                radio_layout.setSpacing(2)
                
                radio_btn = QRadioButton(choice)
                radio_btn.setProperty("choice_value", choice)
                
                # Set default selection
                if default_value is not None and str(default_value) == str(choice):
                    radio_btn.setChecked(True)
                elif idx == 0 and default_value is None:
                    radio_btn.setChecked(True)
                
                radio_btn.setStyleSheet("""
                    QRadioButton {
                        font-size: 12px;
                        font-weight: 600;
                        spacing: 8px;
                        color: #2c3e50;
                    }
                    QRadioButton::indicator {
                        width: 16px;
                        height: 16px;
                    }
                    QRadioButton::indicator:unchecked {
                        background-color: white;
                        border: 2px solid #bdc3c7;
                        border-radius: 8px;
                    }
                    QRadioButton::indicator:checked {
                        background-color: #3498db;
                        border: 2px solid #2980b9;
                        border-radius: 8px;
                    }
                """)
                
                button_group.addButton(radio_btn)
                radio_layout.addWidget(radio_btn)
                
                # Add description if available (use LOCALIZE if key is localization key)
                if choice in descriptions:
                    desc_text = descriptions[choice]
                    # Try to localize if it looks like a localization key
                    # For Vector normalization, we map keys to localization keys
                    if param_name == "norm" and choice in ["l1", "l2", "max"]:
                        # Use localization for Vector normalization descriptions
                        localization_key = f"PREPROCESS.norm_{choice}_desc"
                        try:
                            desc_text = LOCALIZE(localization_key)
                        except:
                            # Fallback to original description if localization fails
                            desc_text = descriptions[choice]
                    
                    desc_label = QLabel(desc_text)
                    desc_label.setWordWrap(True)
                    desc_label.setStyleSheet("""
                        QLabel {
                            font-size: 11px;
                            color: #6c757d;
                            padding-left: 24px;
                            line-height: 1.4;
                        }
                    """)
                    radio_layout.addWidget(desc_label)
                
                layout.addWidget(radio_container)
            
            # Store button group for later retrieval
            widget.button_group = button_group
            return widget
            
        elif param_type == "tuple":
            # Use new RangeParameterWidget for tuple parameters with actual data range
            return RangeParameterWidget(
                param_name, info, default_value, self.data_range
            )

        elif param_type == "dict":
            # Use new DictParameterWidget for dictionary parameters
            return DictParameterWidget(param_name, info, default_value)

        elif param_type == "bool":
            widget = QCheckBox()
            if default_value is not None:
                widget.setChecked(bool(default_value))
            widget.setStyleSheet(
                """
                QCheckBox {
                    font-size: 12px;
                    spacing: 8px;
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                }
                QCheckBox::indicator:unchecked {
                    background-color: white;
                    border: 2px solid #bdc3c7;
                    border-radius: 3px;
                }
                QCheckBox::indicator:checked {
                    background-color: #3498db;
                    border: 2px solid #2980b9;
                    border-radius: 3px;
                }
            """
            )
            return widget

        elif param_type == "list_int":
            widget = QLineEdit()
            if default_value is not None:
                if isinstance(default_value, list):
                    widget.setText(", ".join(map(str, default_value)))
                else:
                    widget.setText(str(default_value))
            widget.setPlaceholderText(LOCALIZE("PREPROCESS.list_int_format_hint"))
            widget.setStyleSheet(
                """
                QLineEdit {
                    padding: 6px;
                    border: 2px solid #bdc3c7;
                    border-radius: 4px;
                    background-color: white;
                    font-size: 12px;
                }
                QLineEdit:focus {
                    border-color: #3498db;
                }
            """
            )
            return widget

        elif param_type == "list_float":
            widget = QLineEdit()
            if default_value is not None:
                if isinstance(default_value, list):
                    widget.setText(", ".join(map(str, default_value)))
                else:
                    widget.setText(str(default_value))
            widget.setPlaceholderText(LOCALIZE("PREPROCESS.list_float_format_hint"))
            widget.setStyleSheet(
                """
                QLineEdit {
                    padding: 6px;
                    border: 2px solid #bdc3c7;
                    border-radius: 4px;
                    background-color: white;
                    font-size: 12px;
                }
                QLineEdit:focus {
                    border-color: #3498db;
                }
            """
            )
            return widget

        elif param_type == "optional":
            # Handle optional parameters
            widget = QLineEdit()
            if default_value is not None:
                widget.setText(str(default_value))
            widget.setPlaceholderText(
                LOCALIZE("PREPROCESS.UI.optional_field_placeholder")
            )
            widget.setStyleSheet(
                """
                QLineEdit {
                    padding: 6px;
                    border: 2px solid #bdc3c7;
                    border-radius: 4px;
                    background-color: white;
                    font-size: 12px;
                }
                QLineEdit:focus {
                    border-color: #3498db;
                }
            """
            )
            return widget

        else:
            # Default to text input
            widget = QLineEdit()
            if default_value is not None:
                widget.setText(str(default_value))
            widget.setStyleSheet(
                """
                QLineEdit {
                    padding: 6px;
                    border: 2px solid #bdc3c7;
                    border-radius: 4px;
                    background-color: white;
                    font-size: 12px;
                }
                QLineEdit:focus {
                    border-color: #3498db;
                }
            """
            )
            return widget

    def get_parameters(self) -> Dict[str, Any]:
        """Extract parameters from widgets."""
        params = {}
        param_info = self.method_info.get("param_info", {})

        for param_name, widget in self.param_widgets.items():
            info = param_info.get(param_name, {})
            param_type = info.get("type", "float")

            try:
                if param_type == "int":
                    params[param_name] = widget.value()
                elif param_type in ["float", "scientific"]:
                    params[param_name] = widget.value()
                elif param_type in ["choice", "combo"]:
                    params[param_name] = widget.currentText()
                elif param_type == "radio":
                    # Get selected radio button value
                    if hasattr(widget, 'button_group'):
                        checked_btn = widget.button_group.checkedButton()
                        if checked_btn:
                            params[param_name] = checked_btn.property("choice_value")
                elif param_type == "bool":
                    params[param_name] = widget.isChecked()
                elif param_type == "tuple":
                    if isinstance(widget, RangeParameterWidget):
                        # Use new RangeParameterWidget
                        params[param_name] = widget.get_value()
                    else:
                        # Fallback to old text-based parsing
                        text = widget.text().strip()
                        if text:
                            try:
                                values = [float(x.strip()) for x in text.split(",")]
                                if len(values) == 2:
                                    params[param_name] = tuple(values)
                                else:
                                    # Log validation error but use default or None
                                    create_logs(
                                        "DynamicParameterWidget",
                                        "parameter_validation",
                                        f"Tuple parameter {param_name} requires exactly 2 values, got {len(values)}",
                                        status="warning",
                                    )
                                    # Use default value from method info if available
                                    default_params = self.method_info.get(
                                        "default_params", {}
                                    )
                                    if param_name in default_params:
                                        params[param_name] = default_params[param_name]
                            except ValueError as ve:
                                # Log parsing error but use default
                                create_logs(
                                    "DynamicParameterWidget",
                                    "parameter_validation",
                                    f"Error parsing tuple parameter {param_name}: {ve}",
                                    status="warning",
                                )
                                # Use default value from method info if available
                                default_params = self.method_info.get(
                                    "default_params", {}
                                )
                                if param_name in default_params:
                                    params[param_name] = default_params[param_name]
                        else:
                            # Empty text - use default value if available
                            default_params = self.method_info.get("default_params", {})
                            if param_name in default_params:
                                params[param_name] = default_params[param_name]
                elif param_type == "dict":
                    if isinstance(widget, DictParameterWidget):
                        # Use new DictParameterWidget
                        params[param_name] = widget.get_value()
                    else:
                        # Fallback to old text-based parsing
                        text = widget.text().strip()
                        if text:
                            # Parse "key1:value1, key2:value2" format
                            dict_result = {}
                            for pair in text.split(","):
                                if ":" in pair:
                                    key, value = pair.split(":", 1)
                                    key = key.strip()
                                    try:
                                        # Try to convert value to float, fallback to string
                                        dict_result[key] = float(value.strip())
                                    except ValueError:
                                        dict_result[key] = value.strip()
                            params[param_name] = dict_result if dict_result else None
                elif param_type == "list_int":
                    text = widget.text().strip()
                    if text:
                        params[param_name] = [int(x.strip()) for x in text.split(",")]
                elif param_type == "list_float":
                    text = widget.text().strip()
                    if text:
                        values = [float(x.strip()) for x in text.split(",")]
                        params[param_name] = values if values else None
                elif param_type == "optional":
                    text = widget.text().strip()
                    # For optional parameters, empty text means None/default
                    params[param_name] = text if text else None
                else:
                    text = widget.text().strip()
                    if text:
                        params[param_name] = text
            except Exception as e:
                create_logs(
                    "DynamicParameterWidget",
                    "parameter_extraction",
                    f"Error extracting parameter {param_name}: {e}",
                    status="warning",
                )

        return params
