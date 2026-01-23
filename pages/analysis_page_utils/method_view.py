"""
Method View Components

This module handles the method-specific view with input forms and results display.
Includes dynamic parameter widget generation and results visualization.
"""

import traceback
import warnings
import re

from typing import Dict, Any, Callable, Optional
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QFrame,
    QScrollArea,
    QTabWidget,
    QGroupBox,
    QComboBox,
    QSplitter,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QStackedWidget,
    QButtonGroup,
    QRadioButton,
    QCheckBox,
    QLineEdit,
    QMessageBox,
)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QFont, QStandardItemModel, QStandardItem

from components.widgets import load_icon, DynamicParameterWidget, ComponentSelectorPlotPanel
from components.widgets.results_panel import ResultsPanel as SharedResultsPanel

from .tab_group_manager import TabGroupManager

from .registry import ANALYSIS_METHODS
from .group_assignment_table import GroupAssignmentTable

# Import PROJECT_MANAGER for group persistence
from utils import PROJECT_MANAGER, register_groups_changed_listener

# Import visualization helpers
from .methods.exploratory import create_spectrum_preview_figure
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from configs.configs import create_logs




# =============================================================================
# CHECKABLE COMBOBOX - Multi-select dropdown with checkboxes (max 4 components)
# =============================================================================
class CheckableComboBox(QComboBox):
    """
    A ComboBox with checkable items for multi-component selection.
    Used for selecting multiple PCA components (max 4).
    """

    def __init__(self, parent=None, max_items=4):
        super().__init__(parent)
        self.max_items = max_items
        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        self.view().pressed.connect(self._on_item_pressed)

        # Prevent popup from closing on item selection
        self.view().viewport().installEventFilter(self)

        # Style
        self.setStyleSheet(
            """
            QComboBox {
                padding: 6px 12px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                font-size: 12px;
                min-width: 140px;
            }
            QComboBox:hover { border-color: #0078d4; }
            QComboBox::drop-down { border: none; width: 24px; background-color: white; }
            QComboBox QAbstractItemView {
                border: 1px solid #ced4da;
                selection-background-color: #e3f2fd;
                background-color: white;
            }
        """
        )

    def eventFilter(self, obj, event):
        """Keep popup open when clicking items."""
        if event.type() == event.Type.MouseButtonRelease:
            return True  # Block the release event to keep popup open
        return super().eventFilter(obj, event)

    def addCheckableItem(self, text, checked=False):
        """Add a checkable item to the combo box."""
        item = QStandardItem(text)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(Qt.Checked if checked else Qt.Unchecked, Qt.CheckStateRole)
        self._model.appendRow(item)

    def _on_item_pressed(self, index):
        """Toggle check state when item is pressed."""
        item = self._model.itemFromIndex(index)
        if item.checkState() == Qt.Checked:
            item.setCheckState(Qt.Unchecked)
        else:
            # Check max selection limit
            if len(self.getCheckedItems()) >= self.max_items:
                return  # Don't allow more than max_items
            item.setCheckState(Qt.Checked)
        self._update_display_text()

    def _update_display_text(self):
        """Update the display text to show selected items."""
        checked = self.getCheckedItems()
        if not checked:
            self.setCurrentText("Select Components...")
        else:
            self.setCurrentText(", ".join(checked))

    def getCheckedItems(self):
        """Return list of checked item texts."""
        checked = []
        for i in range(self._model.rowCount()):
            item = self._model.item(i)
            if item.checkState() == Qt.Checked:
                checked.append(item.text())
        return checked

    def getCheckedIndices(self):
        """Return list of checked item indices (0-based)."""
        indices = []
        for i in range(self._model.rowCount()):
            item = self._model.item(i)
            if item.checkState() == Qt.Checked:
                indices.append(i)
        return indices


def _v1_create_method_view(
    category: str,
    method_key: str,
    dataset_names: list,
    localize_func: Callable,
    on_run_analysis: Callable,
    on_back: Callable,
) -> QWidget:
    """
    Create method-specific view with input form and results display (Image 2 reference).

    Args:
        category: Method category
        method_key: Method identifier
        dataset_names: Available dataset names list (strings)
        localize_func: Localization function
        on_run_analysis: Callback when Run Analysis is clicked
        on_back: Callback for back button

    Returns:
        Method view widget with accessible components
    """
    method_info = ANALYSIS_METHODS[category][method_key]

    method_widget = QWidget()
    method_widget.setObjectName("methodView")

    main_layout = QVBoxLayout(method_widget)
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.setSpacing(0)

    # Splitter: Left (Input Form) | Right (Results)
    splitter = QSplitter(Qt.Horizontal)
    splitter.setChildrenCollapsible(False)

    # === LEFT PANEL: Input Form ===
    left_panel = QWidget()
    left_layout = QVBoxLayout(left_panel)
    left_layout.setContentsMargins(24, 24, 24, 24)
    left_layout.setSpacing(16)

    # Method header
    method_name_label = QLabel(method_info["name"])
    method_name_label.setStyleSheet(
        """
        font-size: 20px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 8px;
    """
    )
    left_layout.addWidget(method_name_label)

    # Use localized description from locales, not hardcoded from registry
    method_desc_text = localize_func(f"ANALYSIS_PAGE.METHOD_DESC.{method_key}")
    method_desc_label = QLabel(method_desc_text)
    method_desc_label.setWordWrap(True)
    method_desc_label.setStyleSheet(
        """
        font-size: 13px;
        color: #6c757d;
        line-height: 1.5;
        margin-bottom: 16px;
    """
    )
    left_layout.addWidget(method_desc_label)

    # Dataset selection - conditional widget based on method requirements
    dataset_selection_mode = method_info.get("dataset_selection_mode", "single")
    min_datasets = method_info.get("min_datasets", 1)

    dataset_card = QFrame()
    dataset_card.setObjectName("datasetCard")
    dataset_card.setStyleSheet(
        """
        QFrame#datasetCard {
            background-color: #ffffff;
            border: 1px solid #dfe3ea;
            border-radius: 16px;
            padding: 0px;
        }
    """
    )
    dataset_card_layout = QVBoxLayout(dataset_card)
    dataset_card_layout.setContentsMargins(24, 24, 24, 24)
    dataset_card_layout.setSpacing(18)

    # Card header with icon + badges
    dataset_header = QHBoxLayout()
    dataset_header.setSpacing(12)

    dataset_title = QLabel("📂 " + localize_func("ANALYSIS_PAGE.dataset_selection"))
    dataset_title.setStyleSheet(
        """
        font-size: 18px;
        font-weight: 600;
        color: #1f2a37;
    """
    )
    dataset_header.addWidget(dataset_title)

    dataset_header.addStretch()

    selection_badge = QLabel(
        localize_func("ANALYSIS_PAGE.dataset_mode_multi")
        if dataset_selection_mode == "multi"
        else localize_func("ANALYSIS_PAGE.dataset_mode_single")
    )
    selection_badge.setStyleSheet(
        """
        background-color: #eef2ff;
        color: #4338ca;
        border-radius: 999px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 600;
    """
    )
    dataset_header.addWidget(selection_badge)

    min_badge_text = localize_func("ANALYSIS_PAGE.min_datasets").format(
        count=min_datasets
    )
    min_badge = QLabel(min_badge_text)
    min_badge.setStyleSheet(
        """
        background-color: #ecfdf5;
        color: #047857;
        border-radius: 999px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 600;
    """
    )
    dataset_header.addWidget(min_badge)

    dataset_card_layout.addLayout(dataset_header)

    dataset_subtitle = QLabel(localize_func("ANALYSIS_PAGE.dataset_selection_subtitle"))
    dataset_subtitle.setWordWrap(True)
    dataset_subtitle.setStyleSheet(
        """
        font-size: 13px;
        color: #4b5563;
        line-height: 1.5;
    """
    )
    dataset_card_layout.addWidget(dataset_subtitle)

    dataset_layout = QVBoxLayout()
    dataset_layout.setSpacing(16)
    dataset_layout.setContentsMargins(0, 0, 0, 0)
    dataset_card_layout.addLayout(dataset_layout)

    # For multi-dataset methods, add professional pill-shaped segmented control
    mode_toggle = None
    comparison_radio = None
    classification_radio = None

    if dataset_selection_mode == "multi":
        # Modern pill-shaped segmented control container
        toggle_frame = QFrame()
        toggle_frame.setStyleSheet(
            """
            QFrame {
                background-color: #e9ecef;
                border: 1px solid #dee2e6;
                border-radius: 25px;
                padding: 3px;
                max-width: 500px;
            }
        """
        )
        toggle_layout = QHBoxLayout(toggle_frame)
        toggle_layout.setContentsMargins(3, 3, 3, 3)
        toggle_layout.setSpacing(3)

        # Comparison mode button - pill shaped
        comparison_radio = QRadioButton("📊 Unsupervised")
        comparison_radio.setObjectName("comparison_radio")
        comparison_radio.setChecked(True)
        comparison_radio.setCursor(Qt.PointingHandCursor)
        comparison_radio.setStyleSheet(
            """
            QRadioButton {
                background-color: transparent;
                border-radius: 22px;
                padding: 12px 32px;
                font-weight: 600;
                font-size: 14px;
                color: #495057;
            }
            QRadioButton:hover {
                background-color: rgba(0, 120, 212, 0.1);
            }
            QRadioButton:checked {
                background-color: #0078d4;
                color: white;
            }
            QRadioButton::indicator {
                width: 0px;
                height: 0px;
            }
        """
        )
        toggle_layout.addWidget(comparison_radio)

        # Classification mode button - pill shaped
        classification_radio = QRadioButton("🔬 Grouped Classification")
        classification_radio.setObjectName("classification_radio")
        classification_radio.setCursor(Qt.PointingHandCursor)
        classification_radio.setStyleSheet(
            """
            QRadioButton {
                background-color: transparent;
                border-radius: 22px;
                padding: 12px 32px;
                font-weight: 600;
                font-size: 14px;
                color: #495057;
            }
            QRadioButton:hover {
                background-color: rgba(40, 167, 69, 0.1);
            }
            QRadioButton:checked {
                background-color: #28a745;
                color: white;
            }
            QRadioButton::indicator {
                width: 0px;
                height: 0px;
            }
        """
        )
        toggle_layout.addWidget(classification_radio)

        # Button group for mutual exclusion
        mode_toggle = QButtonGroup()
        mode_toggle.addButton(comparison_radio, 0)
        mode_toggle.addButton(classification_radio, 1)

        dataset_layout.addWidget(toggle_frame)

    # Create stacked widget for simple vs group mode
    dataset_stack = QStackedWidget()

    # === PAGE 0: Simple Selection (Comparison Mode) ===
    simple_widget = QWidget()
    simple_layout = QVBoxLayout(simple_widget)
    simple_layout.setContentsMargins(0, 0, 0, 0)

    # Add professional hint label
    if dataset_selection_mode == "multi":
        hint_label = QLabel(localize_func("ANALYSIS_PAGE.unsupervised_mode_hint"))
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet(
            """
            font-size: 12px;
            color: #495057;
            padding: 12px;
            background-color: #e7f3ff;
            border-left: 4px solid #0078d4;
            border-radius: 4px;
            margin-top: 8px;
        """
        )
        simple_layout.addWidget(hint_label)

    # Create appropriate widget based on selection mode
    dataset_widget = None
    select_all_checkbox = None

    if dataset_selection_mode == "single":
        # Single dropdown for single-dataset methods
        dataset_combo = QComboBox()
        dataset_combo.setObjectName("datasetComboBox")
        dataset_combo.setMinimumHeight(40)
        dataset_combo.addItems(dataset_names)
        # Enhanced QComboBox styling with proper dropdown list background
        dataset_combo.setStyleSheet(
            """
            QComboBox {
                border: 1px solid #ced4da;
                border-radius: 6px;
                padding: 8px 12px;
                background-color: #ffffff;
                font-size: 13px;
                color: #2c3e50;
                min-width: 200px;
            }
            QComboBox:hover {
                border-color: #0078d4;
                background-color: #f8f9fa;
            }
            QComboBox:focus {
                border-color: #0078d4;
                border-width: 2px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left: 1px solid #e0e0e0;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
                background-color: #f8f9fa;
            }
            QComboBox::down-arrow {
                width: 12px;
                height: 12px;
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #495057;
            }
            /* CRITICAL FIX: Dropdown list popup styling */
            QComboBox QAbstractItemView {
                border: 1px solid #ced4da;
                border-radius: 6px;
                background-color: #ffffff;
                selection-background-color: #e7f3ff;
                selection-color: #0078d4;
                outline: none;
                padding: 4px;
            }
            QComboBox QAbstractItemView::item {
                min-height: 32px;
                padding: 8px 12px;
                background-color: #ffffff;
                color: #2c3e50;
                border-radius: 4px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #f0f7ff;
                color: #0078d4;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #e7f3ff;
                color: #0078d4;
                font-weight: 600;
            }
        """
        )
        dataset_widget = dataset_combo
        simple_layout.addWidget(dataset_combo)
    else:
        # Create toolbar with Select All checkbox
        from PySide6.QtWidgets import QCheckBox

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(8, 8, 8, 4)
        
        def _select_all_label(all_selected: bool) -> str:
            try:
                return (
                    localize_func("ANALYSIS_PAGE.deselect_all_button")
                    if all_selected
                    else localize_func("ANALYSIS_PAGE.select_all_button")
                )
            except Exception:
                return "Deselect All" if all_selected else "Select All"

        select_all_checkbox = QCheckBox(_select_all_label(False))
        select_all_checkbox.setObjectName("selectAllCheckbox")
        select_all_checkbox.setStyleSheet(
            """
            QCheckBox {
                font-weight: 600;
                font-size: 13px;
                color: #495057;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #adb5bd;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:hover {
                border-color: #0078d4;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border-color: #0078d4;
                image: url(none);
            }
            QCheckBox::indicator:checked:after {
                content: "✓";
                color: white;
            }
        """
        )
        toolbar_layout.addWidget(select_all_checkbox)
        toolbar_layout.addStretch()
        simple_layout.addLayout(toolbar_layout)

        # Enhanced list with larger rows and hover effects
        dataset_list = QListWidget()
        dataset_list.setObjectName("datasetListWidget")
        dataset_list.setSelectionMode(QAbstractItemView.MultiSelection)
        dataset_list.setMinimumHeight(180)
        dataset_list.setMaximumHeight(300)
        dataset_list.addItems(dataset_names)
        dataset_list.setSpacing(2)
        dataset_list.setStyleSheet(
            """
            QListWidget {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 6px;
                background-color: white;
            }
            QListWidget::item {
                padding: 12px 16px;
                border-radius: 4px;
                margin: 1px 0;
                border: 1px solid transparent;
                font-size: 13px;
            }
            QListWidget::item:hover {
                background-color: #f8f9fa;
                border-color: #e9ecef;
            }
            QListWidget::item:selected {
                background-color: #e7f3ff;
                color: #0078d4;
                border-left: 3px solid #0078d4;
                font-weight: 600;
            }
        """
        )

        # Connect Select All functionality
        def toggle_select_all(checked):
            # Action-aware: if user clicks while already fully selected, treat it as deselect.
            total = dataset_list.count()
            selected = len(dataset_list.selectedItems())
            currently_all = (total > 0 and selected == total)

            if checked and not currently_all:
                dataset_list.selectAll()
            else:
                dataset_list.clearSelection()

        select_all_checkbox.toggled.connect(toggle_select_all)

        # Update Select All state when list selection changes
        def update_select_all_state():
            total = dataset_list.count()
            selected = len(dataset_list.selectedItems())
            select_all_checkbox.blockSignals(True)
            all_selected = (selected == total and total > 0)
            select_all_checkbox.setChecked(all_selected)
            select_all_checkbox.setText(_select_all_label(all_selected))
            select_all_checkbox.blockSignals(False)

        dataset_list.itemSelectionChanged.connect(update_select_all_state)

        # Initialize label
        update_select_all_state()

        dataset_widget = dataset_list
        simple_layout.addWidget(dataset_list)

    dataset_stack.addWidget(simple_widget)

    # === PAGE 1: Group Assignment (Classification Mode) ===
    group_widget = None
    if dataset_selection_mode == "multi":
        group_widget = GroupAssignmentTable(dataset_names, localize_func)
        group_widget.setMinimumHeight(400)
        dataset_stack.addWidget(group_widget)

        # Load saved groups from ProjectManager
        saved_groups = PROJECT_MANAGER.get_analysis_groups()
        if saved_groups:
            create_logs(
                "MethodView",
                "method_view",
                f"Loading {len(saved_groups)} saved groups from project",
                status="debug",
            )
            group_widget.set_groups(saved_groups)
        else:
            create_logs(
                "MethodView",
                "method_view",
                "No saved groups found in project",
                status="debug",
            )

        # Connect groups_changed signal to save groups to ProjectManager
        def on_groups_changed(groups: Dict[str, list]):
            create_logs(
                "MethodView",
                "method_view",
                f"Groups changed, saving {len(groups)} groups to ProjectManager",
                status="debug",
            )
            PROJECT_MANAGER.set_analysis_groups(groups, origin="analysis")

        group_widget.groups_changed.connect(on_groups_changed)

        # Connect mode toggle to switch between modes
        def toggle_mode(button):
            create_logs("MethodView", "method_view", "toggle_mode called", status="debug")
            create_logs("MethodView", "method_view", f"Button clicked: {button}", status="debug")
            create_logs(
                "MethodView",
                "method_view",
                f"Button objectName: {button.objectName()}",
                status="debug",
            )
            create_logs("MethodView", "method_view", f"comparison_radio: {comparison_radio}", status="debug")
            create_logs("MethodView", "method_view", f"classification_radio: {classification_radio}", status="debug")
            create_logs(
                "MethodView",
                "method_view",
                f"Current stack index BEFORE: {dataset_stack.currentIndex()}",
                status="debug",
            )

            # Check which button was clicked and switch pages
            if button == comparison_radio:
                create_logs(
                    "MethodView",
                    "method_view",
                    "Switching to Comparison Mode (page 0)",
                    status="debug",
                )
                dataset_stack.setCurrentIndex(0)  # Show simple selection
                create_logs(
                    "MethodView",
                    "method_view",
                    f"Stack index AFTER: {dataset_stack.currentIndex()}",
                    status="debug",
                )
            elif button == classification_radio:
                create_logs(
                    "MethodView",
                    "method_view",
                    "Switching to Classification Mode (page 1)",
                    status="debug",
                )
                dataset_stack.setCurrentIndex(1)  # Show group assignment table
                create_logs(
                    "MethodView",
                    "method_view",
                    f"Stack index AFTER: {dataset_stack.currentIndex()}",
                    status="debug",
                )
            else:
                create_logs(
                    "MethodView",
                    "method_view",
                    "Button not recognized in toggle_mode",
                    status="warning",
                )

            create_logs(
                "MethodView",
                "method_view",
                f"Current visible widget: {dataset_stack.currentWidget()}",
                status="debug",
            )

        # CRITICAL FIX: Connect individual button toggled signals instead of buttonClicked
        # buttonClicked sometimes fails silently, toggled is more reliable
        comparison_radio.toggled.connect(
            lambda checked: toggle_mode(comparison_radio) if checked else None
        )
        classification_radio.toggled.connect(
            lambda checked: toggle_mode(classification_radio) if checked else None
        )

        # Also try buttonClicked as backup
        mode_toggle.buttonClicked.connect(toggle_mode)

        create_logs(
            "MethodView",
            "method_view",
            "Mode toggle signals connected (toggled + buttonClicked)",
            status="debug",
        )
        create_logs(
            "MethodView",
            "method_view",
            f"comparison_radio.toggled: {comparison_radio.toggled}",
            status="debug",
        )
        create_logs(
            "MethodView",
            "method_view",
            f"classification_radio.toggled: {classification_radio.toggled}",
            status="debug",
        )
        create_logs(
            "MethodView",
            "method_view",
            f"mode_toggle.buttonClicked: {mode_toggle.buttonClicked}",
            status="debug",
        )
        create_logs(
            "MethodView",
            "method_view",
            f"Toggle function object: {toggle_mode}",
            status="debug",
        )
        create_logs(
            "MethodView",
            "method_view",
            f"Initial stack index: {dataset_stack.currentIndex()}",
            status="debug",
        )
        create_logs(
            "MethodView",
            "method_view",
            f"Initial visible widget: {dataset_stack.currentWidget()}",
            status="debug",
        )
        create_logs(
            "MethodView",
            "method_view",
            f"comparison_radio checked: {comparison_radio.isChecked()}",
            status="debug",
        )
        create_logs(
            "MethodView",
            "method_view",
            f"classification_radio checked: {classification_radio.isChecked()}",
            status="debug",
        )

        # Set initial page to Comparison Mode
        dataset_stack.setCurrentIndex(0)
        comparison_radio.setChecked(True)
        create_logs(
            "MethodView",
            "method_view",
            "Set initial mode to Comparison (index 0)",
            status="debug",
        )
        create_logs(
            "MethodView",
            "method_view",
            f"After init - comparison_radio checked: {comparison_radio.isChecked()}",
            status="debug",
        )

    dataset_layout.addWidget(dataset_stack)

    left_layout.addWidget(dataset_card)

    # Parameters section
    params_group = QGroupBox(localize_func("ANALYSIS_PAGE.parameters"))
    params_group.setStyleSheet(
        """
        QGroupBox {
            font-weight: 600;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 16px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 4px;
        }
    """
    )
    params_layout = QVBoxLayout(params_group)

    # Create dynamic parameter widget
    # Convert registry format to DynamicParameterWidget format
    # Registry uses: "spinbox"/"double_spinbox"/"combo"/"checkbox"
    # DynamicParameterWidget expects: "int"/"float"/"choice"/"bool"

    type_mapping = {
        "spinbox": "int",
        "double_spinbox": "float",
        "combo": "choice",
        "checkbox": "bool",
    }

    # Convert params from registry format to param_info format
    params_dict = method_info.get("params", {})
    param_info = {}
    default_params = {}

    for param_name, param_config in params_dict.items():
        # Map the type
        registry_type = param_config.get("type", "float")
        widget_type = type_mapping.get(registry_type, registry_type)

        # Build param_info entry
        param_info[param_name] = {
            "type": widget_type,
            "description": param_config.get("label", param_name),
        }

        # Add range if exists
        if "range" in param_config:
            param_info[param_name]["range"] = param_config["range"]

        # Add step if exists
        if "step" in param_config:
            param_info[param_name]["step"] = param_config["step"]

        # Add choices if exists (for combo/choice type)
        if "options" in param_config:
            param_info[param_name]["choices"] = param_config["options"]

        # Store default value
        if "default" in param_config:
            default_params[param_name] = param_config["default"]

    # Create widget with converted format
    param_widget = DynamicParameterWidget(
        method_info={"param_info": param_info, "default_params": default_params},
        saved_params={},
        data_range=None,
        parent=params_group,
    )
    params_layout.addWidget(param_widget)

    left_layout.addWidget(params_group)
    left_layout.addStretch()

    # Action buttons
    button_layout = QHBoxLayout()
    button_layout.setSpacing(12)

    back_btn = QPushButton("← " + localize_func("ANALYSIS_PAGE.back_button"))
    back_btn.setObjectName("secondaryButton")
    back_btn.setMinimumHeight(40)
    back_btn.clicked.connect(on_back)
    button_layout.addWidget(back_btn)

    run_btn = QPushButton(localize_func("ANALYSIS_PAGE.run_analysis"))
    run_btn.setObjectName("primaryButton")
    run_btn.setMinimumHeight(40)
    run_btn.setStyleSheet(
        """
        QPushButton#primaryButton {
            background-color: #0078d4;
            color: white;
            border: none;
            border-radius: 4px;
            font-weight: 600;
            font-size: 14px;
        }
        QPushButton#primaryButton:hover {
            background-color: #006abc;
        }
        QPushButton#primaryButton:pressed {
            background-color: #005a9e;
        }
        QPushButton#primaryButton:disabled {
            background-color: #c0c0c0;
        }
    """
    )

    # Connect run button - extract selected dataset(s) correctly
    def _get_selected_datasets():
        """
        Extract selected dataset names based on widget type and mode.

        Returns:
            For Comparison Mode:
                - Single string (single-dataset methods)
                - List of strings (multi-dataset methods)
            For Classification Mode:
                - Dict[str, List[str]] mapping group names to dataset lists
        """
        # Check if Classification Mode is active (for multi-dataset methods)
        if classification_radio and classification_radio.isChecked() and group_widget:
            # Return group assignments
            groups = group_widget.get_groups()
            if not groups:
                # No groups assigned - show warning
                return None
            return groups

        # Comparison Mode (simple selection)
        if dataset_selection_mode == "single":
            return dataset_widget.currentText()  # Single string
        else:
            # Multi-select list widget
            selected_items = dataset_widget.selectedItems()
            return [item.text() for item in selected_items]  # List of strings

    run_btn.clicked.connect(
        lambda: on_run_analysis(
            category, method_key, _get_selected_datasets(), param_widget
        )
    )
    button_layout.addWidget(run_btn)

    left_layout.addLayout(button_layout)

    # === RIGHT PANEL: Results Display ===
    right_panel = create_results_panel(localize_func)

    # Add panels to splitter
    splitter.addWidget(left_panel)
    splitter.addWidget(right_panel)
    # Give the left (settings) side a bit more space by default.
    splitter.setSizes([460, 740])  # Initial sizes

    main_layout.addWidget(splitter)

    # Store references for external access
    method_widget.dataset_widget = (
        dataset_widget  # Store the actual widget (QComboBox or QListWidget)
    )
    method_widget.group_widget = (
        group_widget  # Store group widget (GroupAssignmentTable)
    )
    method_widget.comparison_radio = comparison_radio  # Store comparison mode button
    method_widget.classification_radio = (
        classification_radio  # Store classification mode button
    )
    method_widget.dataset_selection_mode = dataset_selection_mode
    method_widget.param_widget = param_widget
    method_widget.run_btn = run_btn
    method_widget.back_btn = back_btn
    method_widget.results_panel = right_panel
    method_widget.category = category
    method_widget.method_key = method_key

    return method_widget


def _v1_create_results_panel(localize_func: Callable) -> QWidget:
    """
    Create results display panel with tabs for different result types.

    Args:
        localize_func: Localization function

    Returns:
        Results panel widget with tab_widget attribute
    """
    results_panel = QWidget()
    results_panel.setObjectName("resultsPanel")
    results_panel.setStyleSheet(
        """
        QWidget#resultsPanel {
            background-color: #ffffff;
            border-left: 1px solid #e0e0e0;
        }
    """)
    
    # Create a stacked layout to support loading overlay
    from PySide6.QtWidgets import QStackedLayout
    stacked_layout = QStackedLayout(results_panel)
    stacked_layout.setStackingMode(QStackedLayout.StackAll)
    
    # Main results widget (index 0)
    main_widget = QWidget()
    layout = QVBoxLayout(main_widget)
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    # Header with export buttons
    header_layout = QHBoxLayout()

    results_title = QLabel("📊 " + localize_func("ANALYSIS_PAGE.results_title"))
    results_title.setStyleSheet(
        """
        font-size: 18px;
        font-weight: 600;
        color: #2c3e50;
    """
    )
    header_layout.addWidget(results_title)
    header_layout.addStretch()

    # Export CSV button (hidden until results available)
    # Note: Plot export (PNG/SVG) available via matplotlib toolbar right-click
    export_data_btn = QPushButton(localize_func("ANALYSIS_PAGE.export_csv"))
    export_data_btn.setObjectName("exportButton")
    export_data_btn.setMinimumHeight(32)
    export_data_btn.setVisible(False)
    header_layout.addWidget(export_data_btn)

    layout.addLayout(header_layout)

    # Tab widget for different result views
    tab_widget = QTabWidget()
    tab_widget.setObjectName("resultsTabWidget")
    tab_widget.setStyleSheet(
        """
        QTabWidget::pane {
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            background-color: white;
        }
        QTabBar::tab {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: white;
            border-bottom-color: white;
            font-weight: 600;
        }
        QTabBar::tab:hover {
            background-color: #e7f3ff;
        }
    """
    )

    # Placeholder tabs (will be populated with actual results)
    placeholder_label = QLabel(localize_func("ANALYSIS_PAGE.no_results_yet"))
    placeholder_label.setAlignment(Qt.AlignCenter)
    placeholder_label.setStyleSheet(
        """
        font-size: 14px;
        color: #6c757d;
        padding: 40px;
    """
    )
    tab_widget.addTab(placeholder_label, localize_func("ANALYSIS_PAGE.results_tab"))

    layout.addWidget(tab_widget)
    
    # Add main widget to stacked layout
    stacked_layout.addWidget(main_widget)
    
    # Loading overlay widget (index 1) - semi-transparent gray overlay
    loading_overlay = QWidget()
    loading_overlay.setStyleSheet("""
        QWidget {
            background-color: rgba(255, 255, 255, 0.85);
        }
    """)
    loading_overlay_layout = QVBoxLayout(loading_overlay)
    loading_overlay_layout.setAlignment(Qt.AlignCenter)
    
    loading_label = QLabel("⏳ Analyzing...")
    loading_label.setStyleSheet("""
        font-size: 18px;
        font-weight: 600;
        color: #0078d4;
        background-color: white;
        padding: 20px 40px;
        border-radius: 8px;
        border: 2px solid #0078d4;
    """)
    loading_label.setAlignment(Qt.AlignCenter)
    loading_overlay_layout.addWidget(loading_label)
    
    stacked_layout.addWidget(loading_overlay)
    loading_overlay.setVisible(False)  # Hidden by default
    
    # Store references for external access
    results_panel.tab_widget = tab_widget
    results_panel.export_data_btn = export_data_btn
    results_panel.results_title = results_title
    results_panel.loading_overlay = loading_overlay
    
    # Add helper methods to show/hide loading
    def show_loading():
        loading_overlay.setVisible(True)
        loading_overlay.raise_()
    
    def hide_loading():
        loading_overlay.setVisible(False)
    
    results_panel.show_loading = show_loading
    results_panel.hide_loading = hide_loading
    
    return results_panel


class MethodParametersWidget(QWidget):
    """
    A styled 'Card' widget that holds the dynamic parameter inputs.
    Replaces the generic QGroupBox with a clean, SaaS-style look.
    """

    def __init__(self, method_info, saved_params: dict | None = None, parent=None):
        super().__init__(parent)
        self.method_info = method_info
        self._saved_params = saved_params or {}
        self.dynamic_widget = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(8)

        # 1. Header (Clean Text)
        header = QLabel("⚙️ Configuration")  # You can localize this
        header.setStyleSheet(
            """
            font-size: 13px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 4px;
        """
        )
        layout.addWidget(header)

        # 2. Card Container (White box with subtle border)
        card = QFrame()
        card.setStyleSheet(
            """
            QFrame {
                background-color: #ffffff;
                border: 1px solid #dfe3ea;
                border-radius: 8px;
            }
        """
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)

        # --- Data Conversion for DynamicParameterWidget ---
        # We map the registry types to the types your widget expects
        type_mapping = {
            "spinbox": "int",
            "double_spinbox": "float",
            "combo": "choice",
            "checkbox": "bool",
        }

        params_dict = self.method_info.get("params", {})
        param_info = {}
        default_params = {}

        for p_name, p_config in params_dict.items():
            reg_type = p_config.get("type", "float")
            w_type = type_mapping.get(reg_type, reg_type)

            # Build the info dict
            param_info[p_name] = {
                "type": w_type,
                "description": p_config.get("label", p_name),
            }

            # Transfer constraints
            if "range" in p_config:
                param_info[p_name]["range"] = p_config["range"]
            if "step" in p_config:
                param_info[p_name]["step"] = p_config["step"]
            if "options" in p_config:
                param_info[p_name]["choices"] = p_config["options"]
            if "default" in p_config:
                default_params[p_name] = p_config["default"]

        # Instantiate the Dynamic Logic
        # Note: We assume DynamicParameterWidget is available in your imports
        self.dynamic_widget = DynamicParameterWidget(
            method_info={"param_info": param_info, "default_params": default_params},
            saved_params=self._saved_params,
            data_range=None,
            parent=card,
        )

        card_layout.addWidget(self.dynamic_widget)
        layout.addWidget(card)

    def get_params(self):
        """Bridge to get data from the inner widget"""
        if self.dynamic_widget:
            return self.dynamic_widget.get_params()
        return {}


class DatasetSelectionWidget(QWidget):
    def __init__(
        self,
        dataset_names,
        mode="single",
        localize_func=None,
        parent=None,
        *,
        allow_grouped_mode: bool = True,
    ):
        super().__init__(parent)
        self.dataset_names = dataset_names
        self.mode = mode
        self.localize = localize_func
        self.allow_grouped_mode = bool(allow_grouped_mode)

        # Simple-mode helpers (search/group/sort)
        self._simple_search = None
        self._simple_group_combo = None
        self._simple_sort_combo = None
        self._simple_all_names = list(dataset_names or [])

        # Persist selection across search/group/sort rebuilds.
        self._simple_selected_names: set[str] = set()
        self._simple_is_rebuilding_list = False

        # Used to suppress persistence when rebuilding UI from external changes.
        self._suspend_group_persist = False

        # Widgets
        self.simple_input = None
        self.group_manager = None
        self.radio_group = None
        self.select_all_cb = None  # Reference for logic

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)  # Consistent spacing

        # --- 1. HEADER ROW (Label + Toggle) ---
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # Label
        label_text = (
            self.localize("ANALYSIS_PAGE.dataset_selection")
            if self.localize
            else "Select Datasets"
        )
        label = QLabel("📂 " + label_text)
        # Using color #2c3e50 from stylesheets.py
        label.setStyleSheet("font-size: 14px; font-weight: 600; color: #2c3e50;")
        header_layout.addWidget(label)

        header_layout.addStretch()

        # Toggle (Only for multi mode when grouped mode is allowed)
        if self.mode == "multi" and self.allow_grouped_mode:
            self.stack = QStackedWidget()
            self._create_toggle(header_layout)

        layout.addWidget(header_container)

        # --- 2. CONTENT AREA ---
        if self.mode == "single":
            # Single Selection (ComboBox)
            self.simple_input = QComboBox()
            self.simple_input.addItems(self.dataset_names)
            self.simple_input.setMinimumHeight(40)
            # Applying 'combo_box' style logic from stylesheets.py
            self.simple_input.setStyleSheet(
                """
                QComboBox {
                    padding: 8px;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    background-color: white;
                    font-size: 13px;
                    color: #2c3e50;
                }
                QComboBox:hover { border-color: #0078d4; }
                QComboBox::drop-down { border: none; width: 24px; background-color: white; }
                QComboBox QAbstractItemView {
                    border: 1px solid #ced4da;
                    selection-background-color: #e3f2fd;
                    selection-color: #1976d2;
                    background-color: white;
                }
            """
            )
            layout.addWidget(self.simple_input)

        else:
            # Multi Selection (Stack)

            # PAGE 0: Optimized Simple List (List + Toolbar)
            page_simple = QWidget()
            layout_s = QVBoxLayout(page_simple)
            layout_s.setContentsMargins(0, 0, 0, 0)
            layout_s.setSpacing(0)  # Toolbar sits flush with list

            # A. Toolbar (Select All)
            toolbar = QFrame()
            toolbar.setStyleSheet(
                """
                QFrame {
                    background-color: #f8f9fa;
                    border: 1px solid #ced4da;
                    border-bottom: none;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
            """
            )
            tb_v = QVBoxLayout(toolbar)
            tb_v.setContentsMargins(12, 8, 12, 8)
            tb_v.setSpacing(6)

            row1 = QHBoxLayout()
            row1.setContentsMargins(0, 0, 0, 0)
            row1.setSpacing(8)

            row2 = QHBoxLayout()
            row2.setContentsMargins(0, 0, 0, 0)
            row2.setSpacing(8)

            def _select_all_label(all_selected: bool) -> str:
                if self.localize:
                    return (
                        self.localize("ANALYSIS_PAGE.deselect_all_button")
                        if all_selected
                        else self.localize("ANALYSIS_PAGE.select_all_button")
                    )
                return "Deselect All" if all_selected else "Select All"

            self.select_all_cb = QCheckBox(_select_all_label(False))
            self.select_all_cb.setCursor(Qt.PointingHandCursor)
            self.select_all_cb.setStyleSheet(
                """
                QCheckBox { font-size: 13px; font-weight: 600; color: #0078d4; }
                QCheckBox::indicator { width: 16px; height: 16px; }
                QCheckBox::indicator:unchecked {
                    border: 1px solid #0078d4;
                    border-radius: 3px;
                    background: #ffffff;
                }
                QCheckBox::indicator:checked {
                    border: 1px solid #0078d4;
                    border-radius: 3px;
                    background: #0078d4;
                }
            """
            )
            row1.addWidget(self.select_all_cb)

            def _t(key: str, fallback: str) -> str:
                try:
                    return self.localize(key) if self.localize else fallback
                except Exception:
                    return fallback

            # Search / Group / Sort controls (simple mode)
            row1.addStretch()

            self._simple_search = QLineEdit()
            self._simple_search.setClearButtonEnabled(True)
            self._simple_search.setFixedHeight(28)
            self._simple_search.setPlaceholderText(
                _t("ANALYSIS_PAGE.dataset_search_placeholder", "Search datasets...")
            )
            self._simple_search.setStyleSheet(
                """
                QLineEdit {
                    padding: 6px 10px;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    background-color: white;
                    font-size: 12px;
                    color: #2c3e50;
                    min-width: 180px;
                }
                QLineEdit:hover { border-color: #0078d4; }
                """
            )

            combo_qss = """
                QComboBox {
                    padding: 4px 8px;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    background-color: white;
                    font-size: 12px;
                    color: #2c3e50;
                    min-width: 160px;
                }
                QComboBox:hover { border-color: #0078d4; }
                QComboBox::drop-down { border: none; width: 22px; }
                QComboBox QAbstractItemView {
                    border: 1px solid #ced4da;
                    selection-background-color: #e3f2fd;
                    selection-color: #1976d2;
                    background-color: white;
                }
            """

            self._simple_group_combo = QComboBox()
            self._simple_group_combo.setFixedHeight(28)
            self._simple_group_combo.setStyleSheet(combo_qss)
            self._simple_group_combo.addItem(_t("ANALYSIS_PAGE.group_by_none", "Group: None"), "none")
            self._simple_group_combo.addItem(
                _t("ANALYSIS_PAGE.group_by_raw_preprocessed", "Group: Raw / Preprocessed"),
                "raw_preprocessed",
            )
            self._simple_group_combo.addItem(_t("ANALYSIS_PAGE.group_by_date", "Group: Date"), "date")
            self._simple_group_combo.addItem(_t("ANALYSIS_PAGE.group_by_alphabet", "Group: Alphabet"), "alphabet")

            self._simple_sort_combo = QComboBox()
            self._simple_sort_combo.setFixedHeight(28)
            self._simple_sort_combo.setStyleSheet(combo_qss)
            self._simple_sort_combo.addItem(_t("ANALYSIS_PAGE.sort_by_name_asc", "Sort: Name (A→Z)"), "name_asc")
            self._simple_sort_combo.addItem(_t("ANALYSIS_PAGE.sort_by_name_desc", "Sort: Name (Z→A)"), "name_desc")
            self._simple_sort_combo.addItem(_t("ANALYSIS_PAGE.sort_by_date_newest", "Sort: Date (New→Old)"), "date_new")
            self._simple_sort_combo.addItem(_t("ANALYSIS_PAGE.sort_by_date_oldest", "Sort: Date (Old→New)"), "date_old")

            row1.addWidget(self._simple_search)

            row2.addStretch()
            row2.addWidget(self._simple_group_combo)
            row2.addWidget(self._simple_sort_combo)

            tb_v.addLayout(row1)
            tb_v.addLayout(row2)

            layout_s.addWidget(toolbar)

            # B. The Rich List Widget
            self.simple_input = QListWidget()
            self.simple_input.setSelectionMode(QAbstractItemView.MultiSelection)
            self.simple_input.addItems(self.dataset_names)
            self.simple_input.setMinimumHeight(200)

            # Applying 'dataset_list' style from stylesheets.py PREPROCESS_PAGE_STYLES
            self.simple_input.setStyleSheet(
                """
                QListWidget {
                    border: 1px solid #ced4da;
                    border-top: none; /* Merge with toolbar */
                    border-bottom-left-radius: 4px;
                    border-bottom-right-radius: 4px;
                    background-color: white;
                    padding: 4px;
                    outline: none;
                }
                QListWidget::item {
                    padding: 10px;
                    border-bottom: 1px solid #f1f3f4;
                    background-color: #ffffff;
                    margin-bottom: 2px;
                    border-radius: 3px;
                    color: #2c3e50;
                }
                QListWidget::item:hover {
                    background-color: #f5f5f5;
                }
                QListWidget::item:selected {
                    background-color: #e3f2fd; /* Light blue from theme */
                    color: #0078d4; /* Primary blue text */
                    border: 1px solid #0078d4;
                    font-weight: 600;
                }
                QListWidget::item:selected:hover {
                    background-color: #d0e7ff;
                }
            """
            )

            # Logic: Connect Select All
            self.select_all_cb.toggled.connect(self._toggle_select_all)
            self.simple_input.itemSelectionChanged.connect(self._on_simple_selection_changed)

            layout_s.addWidget(self.simple_input)

            # Apply search/group/sort on changes.
            try:
                self._simple_search.textChanged.connect(self._refresh_simple_dataset_list)
                self._simple_group_combo.currentIndexChanged.connect(self._refresh_simple_dataset_list)
                self._simple_sort_combo.currentIndexChanged.connect(self._refresh_simple_dataset_list)
            except Exception:
                pass

            # Initial apply
            try:
                self._refresh_simple_dataset_list()
            except Exception:
                pass

            # PAGE 1: Grouped Mode (ML-style tabs)
            if self.allow_grouped_mode:
                self.group_manager = TabGroupManager(self.dataset_names, self.localize)

            # Load persisted groups (shared with ML) so Analysis grouped mode reflects them.
            try:
                saved_groups = PROJECT_MANAGER.get_analysis_groups() if self.allow_grouped_mode else None
            except Exception:
                saved_groups = None

            try:
                saved_enabled = (
                    PROJECT_MANAGER.get_analysis_group_enabled()
                    if self.allow_grouped_mode and hasattr(PROJECT_MANAGER, "get_analysis_group_enabled")
                    else None
                )
            except Exception:
                saved_enabled = None

            if self.allow_grouped_mode and isinstance(saved_groups, dict) and saved_groups:
                try:
                    self.group_manager.set_groups(saved_groups, enabled=saved_enabled)
                except Exception:
                    pass

            # Persist edits (drag/drop, create, rename, delete) back into the project.
            if self.allow_grouped_mode:
                try:
                    self.group_manager.groups_changed.connect(self._persist_groups)
                except Exception:
                    pass

            # Runtime sync: refresh this UI when ML page updates groups.
            if self.allow_grouped_mode:
                try:
                    register_groups_changed_listener(self._on_external_groups_changed)
                except Exception:
                    pass

            if self.allow_grouped_mode:
                self.stack.addWidget(page_simple)
                self.stack.addWidget(self.group_manager)
                layout.addWidget(self.stack)
            else:
                # Simple-only multi-selection (no grouped toggle)
                layout.addWidget(page_simple)

    def _create_toggle(self, layout):
        container = QFrame()
        # Matches 'input_field' border color #ced4da
        container.setStyleSheet(
            "background-color: white; border-radius: 16px; border: 1px solid #ced4da;"
        )
        l = QHBoxLayout(container)
        l.setContentsMargins(2, 2, 2, 2)
        l.setSpacing(0)

        self.radio_group = QButtonGroup(self)
        btn_simple = self._make_pill(self.localize("ANALYSIS_PAGE.simple_mode"))
        btn_group = self._make_pill(self.localize("ANALYSIS_PAGE.grouped_mode"))

        self.radio_group.addButton(btn_simple, 0)
        self.radio_group.addButton(btn_group, 1)
        btn_simple.setChecked(True)

        self.radio_group.idToggled.connect(self.stack.setCurrentIndex)
        l.addWidget(btn_simple)
        l.addWidget(btn_group)
        layout.addWidget(container)

    def _make_pill(self, text):
        r = QRadioButton(text)
        r.setCursor(Qt.PointingHandCursor)
        # Uses primary color #0078d4 for active state
        r.setStyleSheet(
            """
            QRadioButton {
                background: transparent;
                color: #6c757d;
                padding: 6px 16px;
                font-size: 12px;
                font-weight: 600;
                border-radius: 14px;
                border: none;
            }
            QRadioButton:checked {
                background-color: #e3f2fd; /* Very light blue bg */
                color: #0078d4; /* Primary blue text */
            }
            QRadioButton:hover:!checked {
                background-color: #f8f9fa;
                color: #495057;
            }
            QRadioButton::indicator { width: 0; height: 0; }
        """
        )
        return r

    def _toggle_select_all(self, checked):
        """Selects or Deselects all items in the list."""
        if not self.simple_input:
            return

        total = self._simple_dataset_item_count()
        selected = self._simple_selected_dataset_count()
        currently_all = (total > 0 and selected == total)

        # Update persistent selection set based on currently visible dataset items.
        visible_names: set[str] = set()
        try:
            for i in range(self.simple_input.count()):
                item = self.simple_input.item(i)
                if item is None or self._simple_is_header_item(item):
                    continue
                visible_names.add(item.text())
        except Exception:
            visible_names = set()

        if checked and not currently_all:
            self._simple_selected_names |= visible_names
            self.simple_input.selectAll()
        else:
            self._simple_selected_names -= visible_names
            self.simple_input.clearSelection()

        # Ensure label matches the post-action state.
        self._update_checkbox_state()

        # Rebuild list so pinned ordering updates immediately.
        try:
            QTimer.singleShot(0, self._refresh_simple_dataset_list)
        except Exception:
            pass

    def _on_simple_selection_changed(self) -> None:
        """Keep selection persistent across list rebuilds and pin selected items to top."""
        if not isinstance(self.simple_input, QListWidget):
            return
        if self._simple_is_rebuilding_list:
            # Don't mutate persistent selection while we are programmatically rebuilding.
            self._update_checkbox_state()
            return

        # Visible dataset names (current list contents; excludes headers)
        visible: set[str] = set()
        try:
            for i in range(self.simple_input.count()):
                item = self.simple_input.item(i)
                if item is None or self._simple_is_header_item(item):
                    continue
                visible.add(item.text())
        except Exception:
            visible = set()

        selected_visible: set[str] = set()
        try:
            selected_visible = {
                i.text()
                for i in self.simple_input.selectedItems()
                if i is not None and not self._simple_is_header_item(i)
            }
        except Exception:
            selected_visible = set()

        # Persist: update only the visible subset; keep hidden selections intact.
        self._simple_selected_names = (self._simple_selected_names - visible) | selected_visible

        self._update_checkbox_state()

        # Rebuild to apply pinned ordering, but defer to avoid fighting Qt selection updates.
        try:
            QTimer.singleShot(0, self._refresh_simple_dataset_list)
        except Exception:
            pass

    def _update_checkbox_state(self):
        """Updates checkbox if user manually selects/deselects items."""
        if not self.simple_input:
            return

        count = self._simple_dataset_item_count()
        selected = self._simple_selected_dataset_count()

        def _select_all_label(all_selected: bool) -> str:
            if self.localize:
                return (
                    self.localize("ANALYSIS_PAGE.deselect_all_button")
                    if all_selected
                    else self.localize("ANALYSIS_PAGE.select_all_button")
                )
            return "Deselect All" if all_selected else "Select All"

        self.select_all_cb.blockSignals(True)
        if selected == count and count > 0:
            self.select_all_cb.setCheckState(Qt.Checked)
            self.select_all_cb.setText(_select_all_label(True))
        elif selected > 0:
            self.select_all_cb.setCheckState(Qt.PartiallyChecked)
            self.select_all_cb.setText(_select_all_label(False))
        else:
            self.select_all_cb.setCheckState(Qt.Unchecked)
            self.select_all_cb.setText(_select_all_label(False))
        self.select_all_cb.blockSignals(False)

    def _simple_is_header_item(self, item: QListWidgetItem) -> bool:
        try:
            return item is not None and item.data(Qt.UserRole) == "header"
        except Exception:
            return False

    def _simple_dataset_item_count(self) -> int:
        if not isinstance(self.simple_input, QListWidget):
            return 0
        total = 0
        for i in range(self.simple_input.count()):
            item = self.simple_input.item(i)
            if item is None:
                continue
            try:
                if (item.flags() & Qt.ItemIsSelectable) and not self._simple_is_header_item(item):
                    total += 1
            except Exception:
                pass
        return total

    def _simple_selected_dataset_count(self) -> int:
        if not isinstance(self.simple_input, QListWidget):
            return 0
        total = 0
        for item in self.simple_input.selectedItems():
            if item is None:
                continue
            if not self._simple_is_header_item(item):
                total += 1
        return total

    def _simple_get_selected_dataset_names(self) -> set[str]:
        if not isinstance(self.simple_input, QListWidget):
            return set()
        # Prefer persistent selection (survives search/group/sort filtering).
        if hasattr(self, "_simple_selected_names") and isinstance(self._simple_selected_names, set):
            return set(self._simple_selected_names)
        return {
            i.text() for i in self.simple_input.selectedItems() if not self._simple_is_header_item(i)
        }

    def _parse_date_prefix(self, name: str) -> str | None:
        """Best-effort YYYYMMDD prefix parser; returns 'YYYYMMDD' or None."""
        if not name:
            return None
        m = re.match(r"^(\d{8})", name)
        return m.group(1) if m else None

    def _is_preprocessed_name(self, name: str) -> bool:
        n = (name or "").lower()
        return ("processed" in n) and ("unprocessed" not in n)

    def _is_preprocessed_dataset(self, name: str) -> bool:
        """Metadata-driven raw/preprocessed detection.

        Rules (per user requirement):
        - metadata is None or {} -> raw
        - missing 'is_preprocessed' -> raw
        - is_preprocessed == True -> preprocessed
        """
        try:
            md = PROJECT_MANAGER.get_dataframe_metadata(str(name))
        except Exception:
            md = None

        if not md:
            return False
        try:
            return bool(md.get("is_preprocessed", False))
        except Exception:
            return False

    def _refresh_simple_dataset_list(self) -> None:
        """Rebuild the simple-mode dataset list based on search/group/sort controls."""
        if self.mode != "multi" or not isinstance(self.simple_input, QListWidget):
            return

        selected_before = self._simple_get_selected_dataset_names()

        search_text = ""
        try:
            if self._simple_search is not None:
                search_text = (self._simple_search.text() or "").strip().lower()
        except Exception:
            search_text = ""

        group_mode = "none"
        try:
            if self._simple_group_combo is not None:
                group_mode = self._simple_group_combo.currentData() or "none"
        except Exception:
            group_mode = "none"

        sort_mode = "name_asc"
        try:
            if self._simple_sort_combo is not None:
                sort_mode = self._simple_sort_combo.currentData() or "name_asc"
        except Exception:
            sort_mode = "name_asc"

        names = list(self._simple_all_names)
        if search_text:
            names = [n for n in names if search_text in (n or "").lower()]

        def name_key(n: str) -> str:
            return (n or "").lower()

        def date_key(n: str) -> str:
            # Unknown dates sort to the end by default.
            return self._parse_date_prefix(n) or "00000000"

        if sort_mode == "name_desc":
            names.sort(key=name_key, reverse=True)
        elif sort_mode == "date_new":
            names.sort(key=date_key, reverse=True)
        elif sort_mode == "date_old":
            names.sort(key=date_key, reverse=False)
        else:
            names.sort(key=name_key, reverse=False)

        # Pin selected datasets to top (in the current sorted order).
        pinned_selected = [n for n in names if n in selected_before]
        remaining_names = [n for n in names if n not in selected_before]

        def _t(key: str, fallback: str) -> str:
            try:
                return self.localize(key) if self.localize else fallback
            except Exception:
                return fallback

        def add_header(text: str) -> None:
            header_item = QListWidgetItem(text)
            header_item.setData(Qt.UserRole, "header")
            header_item.setFlags(Qt.ItemIsEnabled)
            try:
                header_item.setFont(QFont("", 10, QFont.Bold))
            except Exception:
                pass
            self.simple_input.addItem(header_item)

        def add_dataset_item(name: str) -> None:
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, "dataset")
            self.simple_input.addItem(item)

        self.simple_input.blockSignals(True)
        self._simple_is_rebuilding_list = True
        try:
            self.simple_input.clear()

            # Selected section (always on top; respects search filtering)
            if pinned_selected:
                add_header(_t("ANALYSIS_PAGE.group_header_selected", "Selected"))
                for n in pinned_selected:
                    add_dataset_item(n)

            if group_mode == "raw_preprocessed":
                raw: list[str] = []
                pre: list[str] = []
                for n in remaining_names:
                    if self._is_preprocessed_dataset(n):
                        pre.append(n)
                    else:
                        raw.append(n)
                if raw:
                    add_header(_t("ANALYSIS_PAGE.group_header_raw", "Raw"))
                    for n in raw:
                        add_dataset_item(n)
                if pre:
                    add_header(_t("ANALYSIS_PAGE.group_header_preprocessed", "Preprocessed"))
                    for n in pre:
                        add_dataset_item(n)
            elif group_mode == "date":
                buckets: dict[str, list[str]] = {}
                unknown: list[str] = []
                for n in remaining_names:
                    d = self._parse_date_prefix(n)
                    if not d:
                        unknown.append(n)
                        continue
                    buckets.setdefault(d, []).append(n)

                date_keys = list(buckets.keys())
                if sort_mode == "date_old":
                    date_keys.sort()
                else:
                    date_keys.sort(reverse=True)

                for d in date_keys:
                    add_header(d)
                    for n in buckets.get(d, []):
                        add_dataset_item(n)
                if unknown:
                    add_header(_t("ANALYSIS_PAGE.group_header_unknown_date", "Unknown date"))
                    for n in unknown:
                        add_dataset_item(n)
            elif group_mode == "alphabet":
                buckets: dict[str, list[str]] = {}
                for n in remaining_names:
                    first = (n or "")[:1].upper()
                    if not first or not first.isalpha():
                        first = "#"
                    buckets.setdefault(first, []).append(n)

                keys = sorted([k for k in buckets.keys() if k != "#"])
                if "#" in buckets:
                    keys.append("#")

                for k in keys:
                    add_header(k)
                    for n in buckets.get(k, []):
                        add_dataset_item(n)
            else:
                for n in remaining_names:
                    add_dataset_item(n)

            # Restore selection
            if selected_before:
                for i in range(self.simple_input.count()):
                    item = self.simple_input.item(i)
                    if item is None or self._simple_is_header_item(item):
                        continue
                    if item.text() in selected_before:
                        item.setSelected(True)
        finally:
            self.simple_input.blockSignals(False)
            self._simple_is_rebuilding_list = False

        self._update_checkbox_state()

    def get_selection(self):
        if self.mode == "single":
            return self.simple_input.currentText()

        if self.allow_grouped_mode and self.radio_group and self.radio_group.checkedId() == 1:
            # Grouped Mode

            # Respect per-group "Use for analysis" ticks.
            try:
                groups = self.group_manager.get_enabled_groups()
            except Exception:
                groups = self.group_manager.get_groups()

            # Drop empty groups to avoid confusing downstream methods.
            if isinstance(groups, dict):
                return {k: v for k, v in groups.items() if list(v or [])}
            return groups
        else:
            # Simple Mode
            items = self.simple_input.selectedItems()
            return [i.text() for i in items]

    def _persist_groups(self, groups: dict):
        try:
            if self._suspend_group_persist:
                return
            if isinstance(groups, dict):
                PROJECT_MANAGER.set_analysis_groups(groups, origin="analysis")

                # Persist enabled map (if supported by ProjectManager)
                try:
                    if hasattr(PROJECT_MANAGER, "set_analysis_group_enabled") and self.group_manager:
                        PROJECT_MANAGER.set_analysis_group_enabled(
                            self.group_manager.get_enabled_map(),
                            origin="analysis",
                        )
                except Exception:
                    pass
        except Exception:
            pass

    def _on_external_groups_changed(self, origin: Optional[str] = None) -> None:
        # Ignore updates originating from this view to avoid feedback loops.
        if origin == "analysis":
            return
        # Defer UI rebuild to avoid interfering with in-progress drag/drop.
        try:
            QTimer.singleShot(0, self._reload_groups_from_project)
        except Exception:
            self._reload_groups_from_project()

    def _reload_groups_from_project(self) -> None:
        if not self.group_manager:
            return
        try:
            saved_groups = PROJECT_MANAGER.get_analysis_groups()
        except Exception:
            saved_groups = None

        try:
            saved_enabled = (
                PROJECT_MANAGER.get_analysis_group_enabled()
                if hasattr(PROJECT_MANAGER, "get_analysis_group_enabled")
                else None
            )
        except Exception:
            saved_enabled = None

        self._suspend_group_persist = True
        try:
            if isinstance(saved_groups, dict) and saved_groups:
                self.group_manager.set_groups(saved_groups, enabled=saved_enabled)
            else:
                # Keep defaults when no groups are saved.
                if hasattr(self.group_manager, "reset"):
                    self.group_manager.reset(add_default=True)
        finally:
            self._suspend_group_persist = False


class ResultsPanel(SharedResultsPanel):
    """Backwards-compatible alias for the shared results panel component."""

    pass


class MethodView(QWidget):
    """
    The main controller for a specific Analysis Method View.
    Orchestrates the Input Panel (Left) and Results Panel (Right).
    """

    def __init__(
        self,
        category,
        method_key,
        dataset_names,
        localize_func,
        on_run,
        on_back,
        on_stop=None,
        saved_params: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)

        # Store Context
        self.category = category
        self.method_key = method_key
        self.dataset_names = dataset_names
        self.localize = localize_func
        self.on_run_callback = on_run
        self.on_back_callback = on_back
        self.on_stop_callback = on_stop

        self._is_running = False

        # Optional: used for history restoration.
        self._saved_params = saved_params or {}

        # Load Configuration
        self.method_info = ANALYSIS_METHODS[category][method_key]

        # UI References (to be filled during build)
        self.dataset_widget = None
        self.params_widget = None
        self.results_panel = None

        # Build the UI
        self._init_ui()

    def _init_ui(self):
        """Main build orchestrator."""
        self.setObjectName("methodView")

        # 1. Root Layout (No Margins)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 2. Splitter
        # User requirement: make the splitter static (non-draggable).
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        try:
            # Remove the handle area so users can't drag.
            self.splitter.setHandleWidth(0)
        except Exception:
            pass

        # 3. Build Panels
        left_panel = self._build_left_panel()
        self.results_panel = self._build_right_panel()

        # UX: ensure the left settings panel doesn't feel cramped.
        try:
            left_panel.setMinimumWidth(440)
        except Exception:
            pass

        # 4. Add to Splitter
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(self.results_panel)
        self.splitter.setSizes([460, 740])  # Default ratio

        try:
            # Allow both sides to grow, but keep the results a bit more dominant.
            self.splitter.setStretchFactor(0, 3)
            self.splitter.setStretchFactor(1, 5)
        except Exception:
            pass

        # Disable splitter handle interaction (belt + suspenders).
        try:
            for i in range(1, self.splitter.count()):
                h = self.splitter.handle(i)
                if h is not None:
                    h.setEnabled(False)
        except Exception:
            pass

        main_layout.addWidget(self.splitter)

    def _build_left_panel(self):
        """Constructs the Input Form (Left Side)."""
        panel = QWidget()
        panel.setStyleSheet(
            "background-color: #fcfcfc; border-right: 1px solid #dfe3ea;"
        )

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # A. Header
        self._build_header(layout)

        # B. Scroll Area for Inputs (In case parameters are long)
        # We wrap inputs in a scroll area for responsiveness
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(24)

        # --- Input Components ---
        self._build_dataset_section(content_layout)
        self._build_parameter_section(content_layout)

        content_layout.addStretch()  # Push contents up
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)

        # C. Footer Actions
        self._build_action_buttons(layout)

        return panel

    def _build_header(self, parent_layout):
        """Creates Title and Description."""
        title = QLabel(self.method_info["name"])
        title.setStyleSheet("font-size: 20px; font-weight: 600; color: #2c3e50;")

        desc_text = self.localize(f"ANALYSIS_PAGE.METHOD_DESC.{self.method_key}")
        desc = QLabel(desc_text)
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 13px; color: #6c757d; line-height: 1.5;")

        parent_layout.addWidget(title)
        parent_layout.addWidget(desc)

    def _build_dataset_section(self, parent_layout):
        """Instantiates the DatasetSelectionWidget."""
        mode = self.method_info.get("dataset_selection_mode", "single")

        allow_grouped_mode = self.method_info.get("allow_grouped_mode", True)

        self.dataset_widget = DatasetSelectionWidget(
            self.dataset_names,
            mode,
            self.localize,
            allow_grouped_mode=allow_grouped_mode,
        )
        parent_layout.addWidget(self.dataset_widget)

    def _build_parameter_section(self, parent_layout):
        """Instantiates the MethodParametersWidget."""
        self.params_widget = MethodParametersWidget(
            self.method_info,
            saved_params=self._saved_params,
        )
        parent_layout.addWidget(self.params_widget)

    def _build_action_buttons(self, parent_layout):
        """Creates Back and Run buttons."""
        row = QHBoxLayout()
        row.setSpacing(12)

        self.back_btn = QPushButton(self.localize("ANALYSIS_PAGE.back_button"))
        self.back_btn.setMinimumHeight(40)
        self.back_btn.setCursor(Qt.PointingHandCursor)
        self.back_btn.setStyleSheet(
            """
            QPushButton { 
                border: 1px solid #dfe3ea; background: white; color: #2c3e50; 
                border-radius: 6px; font-weight: 600; font-size: 13px;
            }
            QPushButton:hover { background: #f8f9fa; border-color: #adb5bd; }
        """
        )
        self.back_btn.clicked.connect(self.on_back_callback)

        self.run_btn = QPushButton(self.localize("ANALYSIS_PAGE.start_analysis_button"))
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self._apply_run_button_style(is_running=False)
        self.run_btn.clicked.connect(self._handle_run_click)

        row.addWidget(self.back_btn)
        row.addWidget(self.run_btn)
        parent_layout.addLayout(row)

    def _handle_run_click(self):
        """Collects data and triggers run/stop callbacks."""

        # When running, this button acts as a Stop button.
        if self._is_running:
            if callable(self.on_stop_callback):
                try:
                    self.on_stop_callback()
                except Exception:
                    pass
            return

        # 1. Get Datasets
        selected_data = self.dataset_widget.get_selection()

        if not selected_data:
            QMessageBox.warning(
                self,
                self.localize("ANALYSIS_PAGE.input_error_title"),
                self.localize("ANALYSIS_PAGE.input_error_no_dataset"),
            )
            return

        # 2. Get Parameters (Pass the widget itself, logic handled downstream)
        # The runner expects the widget to extract params, or we pass values
        # Currently your logic passes the widget, so we keep that pattern:
        self.on_run_callback(
            self.category,
            self.method_key,
            selected_data,
            self.params_widget.dynamic_widget,  # Pass the inner dynamic widget
        )

    def set_running_state(self, is_running: bool, *, enabled: bool = True) -> None:
        """Update the view to reflect whether analysis is running."""

        self._is_running = bool(is_running)
        try:
            self.run_btn.setEnabled(bool(enabled))
        except Exception:
            pass

        try:
            if self._is_running:
                self.run_btn.setText(self.localize("ANALYSIS_PAGE.stop_analysis_button"))
            else:
                self.run_btn.setText(self.localize("ANALYSIS_PAGE.start_analysis_button"))
        except Exception:
            # Fallback text
            try:
                self.run_btn.setText("Stop" if self._is_running else "Start Analysis")
            except Exception:
                pass

        self._apply_run_button_style(is_running=self._is_running)

        # Optional UX: avoid navigation mid-run.
        try:
            self.back_btn.setEnabled(not self._is_running)
        except Exception:
            pass

    def _apply_run_button_style(self, *, is_running: bool) -> None:
        try:
            if is_running:
                self.run_btn.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #dc3545; color: white; border: none;
                        border-radius: 6px; font-weight: 700; font-size: 14px;
                    }
                    QPushButton:hover { background-color: #c82333; }
                    QPushButton:pressed { background-color: #bd2130; }
                    QPushButton:disabled { background-color: #e9a2aa; color: white; }
                    """
                )
            else:
                self.run_btn.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #0078d4; color: white; border: none;
                        border-radius: 6px; font-weight: 600; font-size: 14px;
                    }
                    QPushButton:hover { background-color: #006abc; }
                    QPushButton:pressed { background-color: #005a9e; }
                    QPushButton:disabled { background-color: #9ac7ea; color: white; }
                    """
                )
        except Exception:
            pass

    def _build_right_panel(self):
        """Creates the Results Panel using the new Class."""
        # Instantiate the class
        self.results_panel = ResultsPanel(self.localize)

        # If you need to connect the export button signal to a handler in MethodView
        self.results_panel.export_btn.clicked.connect(self._handle_export_csv)

        return self.results_panel

    def _handle_export_csv(self):
        """Export the current analysis result table.

        The ResultsPanel button lives inside MethodView, but the source of truth for
        current results + export manager is AnalysisPage. We locate it via the parent
        chain and delegate to its export method.
        """
        try:
            parent = self.parent()
            while parent is not None:
                # Prefer the multi-format export (it provides user feedback dialogs)
                if hasattr(parent, "_export_data_multi_format"):
                    parent._export_data_multi_format()
                    return
                if hasattr(parent, "_export_csv"):
                    parent._export_csv()
                    return
                parent = parent.parent()
        except Exception:
            parent = None

        # Fallback: if we couldn't find AnalysisPage, show a warning (avoid silent no-op)
        QMessageBox.warning(
            self,
            self.localize("ANALYSIS_PAGE.export_error_title"),
            self.localize("ANALYSIS_PAGE.export_error").format("Export handler not found"),
        )


def populate_results_tabs(
    results_panel: QWidget,
    result: Any,
    localize_func: Callable,
    matplotlib_widget_class: type,
    method_name: str = "",
) -> None:
    """
    Populate results tabs with analysis output.

    Args:
        results_panel: Results panel widget from create_results_panel
        result: AnalysisResult object
        localize_func: Localization function
        matplotlib_widget_class: MatplotlibWidget class for plot rendering
        method_name: Name of the analysis method (for conditional UI behavior)
    """
    tab_widget = results_panel.tab_widget
    
    # PRESERVE current tab index before clearing tabs
    current_tab_index = tab_widget.currentIndex() if tab_widget.count() > 0 else 0
    
    # Clear existing tabs
    while tab_widget.count() > 0:
        tab_widget.removeTab(0)

    # Show CSV export button (plot export via matplotlib toolbar)
    results_panel.export_data_btn.setVisible(True)
    
    # ✅ Phase 3.5: Methods that don't need spectrum preview (user requested)
    methods_without_preview = [
        "Peak Detection and Analysis",
        "Group Mean Spectral Comparison"
    ]
    show_spectrum_preview = method_name not in methods_without_preview
    
    # === Tab 0: Spectrum Preview with Collapsible Display Options Sidebar ===
    if result.dataset_data and show_spectrum_preview:
        try:
            # ✅ CRITICAL FIX: Move imports to top of try block before any usage
            from PySide6.QtGui import QIcon
            from PySide6.QtCore import Qt, QSize
            import os
            
            # Create main container
            spectrum_container = QWidget()
            spectrum_main_layout = QVBoxLayout(spectrum_container)
            spectrum_main_layout.setContentsMargins(0, 0, 0, 0)
            spectrum_main_layout.setSpacing(0)
            
            # TOP: Toolbar with toggle button
            toolbar_widget = QWidget()
            toolbar_layout = QHBoxLayout(toolbar_widget)
            toolbar_layout.setContentsMargins(8, 4, 8, 4)
            toolbar_layout.setSpacing(8)
            toolbar_widget.setStyleSheet("background-color: #f8f9fa; border-bottom: 1px solid #e0e0e0;")
            
            toggle_btn = QPushButton(localize_func("ANALYSIS_PAGE.display_options_button"))
            toggle_btn.setCheckable(True)
            toggle_btn.setChecked(False)  # Collapsed by default
            toggle_btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 12px;
                    background-color: #0078d4;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #005a9e; }
                QPushButton:checked { background-color: #005a9e; }
            """)
            toolbar_layout.addWidget(toggle_btn)
            toolbar_layout.addStretch()
            
            spectrum_main_layout.addWidget(toolbar_widget)
            
            # BOTTOM: Horizontal layout with collapsible sidebar
            content_widget = QWidget()
            content_layout = QHBoxLayout(content_widget)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(0)
            
            # LEFT: Collapsible Sidebar
            sidebar = QFrame()
            sidebar.setFixedWidth(220)
            sidebar.setVisible(False)  # Hidden by default
            sidebar.setStyleSheet("""
                QFrame {
                    background-color: #f8f9fa;
                    border-right: 1px solid #e0e0e0;
                }
            """)
            sidebar_layout = QVBoxLayout(sidebar)
            sidebar_layout.setContentsMargins(12, 12, 12, 12)
            sidebar_layout.setSpacing(10)
            
            # Title
            title_label = QLabel("📊 " + localize_func("ANALYSIS_PAGE.display_mode").rstrip(":"))
            title_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #2c3e50; border: none;")
            sidebar_layout.addWidget(title_label)
            
            # Display mode section
            mode_label = QLabel(localize_func("ANALYSIS_PAGE.display_mode"))
            mode_label.setStyleSheet("font-size: 11px; color: #495057; border: none; font-weight: bold;")
            sidebar_layout.addWidget(mode_label)
            
            mode_combo = QComboBox()
            mode_combo.addItems([localize_func("ANALYSIS_PAGE.display_mode_mean"), 
                                localize_func("ANALYSIS_PAGE.display_mode_all")])
            mode_combo.setStyleSheet("""
                QComboBox {
                    padding: 6px 10px;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    background-color: white;
                    font-size: 11px;
                }
                QComboBox:hover { border-color: #0078d4; }
            """)
            sidebar_layout.addWidget(mode_combo)
            
            # Separator
            sep1 = QFrame()
            sep1.setFrameShape(QFrame.HLine)
            sep1.setStyleSheet("border: none; background-color: #dee2e6; max-height: 1px;")
            sidebar_layout.addWidget(sep1)
            
            # Dataset selection section
            dataset_label = QLabel(localize_func("ANALYSIS_PAGE.select_datasets_to_show"))
            dataset_label.setStyleSheet("font-size: 11px; color: #495057; border: none; font-weight: bold;")
            sidebar_layout.addWidget(dataset_label)

            # Hard cap to keep UI responsive (prevents huge redraws and cramped plots)
            MAX_PREVIEW_DATASETS = 6
            limit_hint = QLabel(f"Max {MAX_PREVIEW_DATASETS} datasets")
            limit_hint.setStyleSheet("font-size: 10px; color: #6c757d; border: none;")
            sidebar_layout.addWidget(limit_hint)

            limit_warning = QLabel(f"Max {MAX_PREVIEW_DATASETS} datasets can be shown")
            limit_warning.setVisible(False)
            limit_warning.setStyleSheet(
                "font-size: 10px; color: #b91c1c; border: none; font-weight: 600;"
            )
            sidebar_layout.addWidget(limit_warning)
            
            def _select_all_label(all_selected: bool) -> str:
                try:
                    return (
                        localize_func("ANALYSIS_PAGE.deselect_all_button")
                        if all_selected
                        else localize_func("ANALYSIS_PAGE.select_all_button")
                    )
                except Exception:
                    return "Deselect All" if all_selected else "Select All"

            # "Select All" checkbox (action label)
            # NOTE: We default to showing only 1 dataset for performance.
            select_all_cb = QCheckBox(_select_all_label(False))
            select_all_cb.setChecked(False)
            select_all_cb.setStyleSheet("""
                QCheckBox {
                    font-size: 11px;
                    color: #0078d4;
                    font-weight: bold;
                    border: none;
                    padding: 4px;
                }
                QCheckBox:checked {
                    background-color: #e3f2fd;
                    border-radius: 3px;
                }
            """)
            sidebar_layout.addWidget(select_all_cb)
            
            # Scrollable checkbox list for datasets
            dataset_scroll = QScrollArea()
            dataset_scroll.setWidgetResizable(True)
            dataset_scroll.setStyleSheet("""
                QScrollArea {
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    background-color: white;
                }
            """)
            dataset_scroll.setMaximumHeight(200)
            
            dataset_list_widget = QWidget()
            dataset_list_layout = QVBoxLayout(dataset_list_widget)
            dataset_list_layout.setContentsMargins(8, 8, 8, 8)
            dataset_list_layout.setSpacing(6)
            
            # ✅ Phase 3.4: Eye icon toggle buttons for each dataset
            # Load eye icons
            eye_open_icon = QIcon(os.path.join("assets", "icons", "eye-open.svg"))
            eye_close_icon = QIcon(os.path.join("assets", "icons", "eye-close.svg"))
            
            dataset_checkboxes = []  # Keep name for compatibility, but now contains buttons
            for ds_name, df in result.dataset_data.items():
                n_spectra = df.shape[1]
                
                # Create horizontal layout for each row
                row_layout = QHBoxLayout()
                row_layout.setSpacing(8)
                row_layout.setContentsMargins(0, 0, 0, 0)
                
                # Eye icon toggle button
                eye_btn = QPushButton()
                eye_btn.setIcon(eye_open_icon)
                eye_btn.setCheckable(True)
                # Default to OFF; we'll enable a small subset after building the list.
                eye_btn.setChecked(False)
                eye_btn.setFixedSize(28, 28)
                eye_btn.setIconSize(QSize(18, 18))
                eye_btn.setStyleSheet("""
                    QPushButton {
                        border: 1px solid #ced4da;
                        border-radius: 4px;
                        background-color: white;
                        padding: 4px;
                    }
                    QPushButton:hover {
                        border-color: #0078d4;
                        background-color: #f0f7ff;
                    }
                    QPushButton:checked {
                        background-color: #e3f2fd;
                        border-color: #0078d4;
                    }
                """)
                
                # Toggle icon on state change
                def make_toggle_handler(btn):
                    def toggle_icon(checked):
                        btn.setIcon(eye_open_icon if checked else eye_close_icon)
                    return toggle_icon
                
                eye_btn.toggled.connect(make_toggle_handler(eye_btn))
                eye_btn.dataset_name = ds_name  # Store dataset name
                # QToolButton already provides isChecked()/setChecked(); do not monkeypatch.
                
                # Dataset label
                ds_label = QLabel(f"{ds_name} ({n_spectra} spectra)")
                ds_label.setStyleSheet("""
                    QLabel {
                        font-size: 10px;
                        color: #495057;
                        border: none;
                    }
                """)
                ds_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                
                row_layout.addWidget(eye_btn)
                row_layout.addWidget(ds_label, 1)  # Stretch label
                
                # Create container widget for the row
                row_widget = QWidget()
                row_widget.setLayout(row_layout)
                row_widget.setStyleSheet("""
                    QWidget:hover {
                        background-color: #f8f9fa;
                        border-radius: 3px;
                    }
                """)
                
                dataset_checkboxes.append(eye_btn)  # Store button for compatibility
                dataset_list_layout.addWidget(row_widget)
            
            dataset_list_layout.addStretch()
            dataset_scroll.setWidget(dataset_list_widget)
            sidebar_layout.addWidget(dataset_scroll)
            
            sidebar_layout.addStretch()
            content_layout.addWidget(sidebar)
            
            # RIGHT: Plot area
            spectrum_tab = matplotlib_widget_class()
            spectrum_fig = create_spectrum_preview_figure(result.dataset_data)
            spectrum_tab.update_plot(spectrum_fig)
            content_layout.addWidget(spectrum_tab, 1)
            
            spectrum_main_layout.addWidget(content_widget, 1)
            
            # Toggle sidebar visibility
            def toggle_sidebar():
                sidebar.setVisible(toggle_btn.isChecked())
            
            toggle_btn.toggled.connect(toggle_sidebar)
            
            _select_state = {"all_selected": False}

            def _update_select_all_ui():
                total = len(dataset_checkboxes)
                checked_count = sum(1 for cb in dataset_checkboxes if cb.isChecked())
                all_selected = (total > 0 and checked_count == total)
                _select_state["all_selected"] = all_selected

                select_all_cb.blockSignals(True)
                select_all_cb.setChecked(all_selected)
                select_all_cb.setText(_select_all_label(all_selected))
                select_all_cb.blockSignals(False)

            def _flash_limit_warning():
                try:
                    limit_warning.setVisible(True)
                    QTimer.singleShot(2000, lambda: limit_warning.setVisible(False))
                except Exception:
                    pass

            def on_select_all_clicked(_checked: bool):
                total = len(dataset_checkboxes)
                if total <= 0:
                    return

                if _select_state.get("all_selected"):
                    # Deselect all, but keep at least one selected for a valid preview.
                    for cb in dataset_checkboxes:
                        cb.setChecked(False)
                    try:
                        dataset_checkboxes[0].setChecked(True)
                    except Exception:
                        pass
                else:
                    # Select up to MAX_PREVIEW_DATASETS (avoid lag / clutter)
                    for cb in dataset_checkboxes:
                        cb.setChecked(False)
                    for cb in dataset_checkboxes[: min(MAX_PREVIEW_DATASETS, total)]:
                        cb.setChecked(True)
                    if total > MAX_PREVIEW_DATASETS:
                        _flash_limit_warning()

                _update_select_all_ui()

            select_all_cb.clicked.connect(on_select_all_clicked)
            
            def _ensure_min_one_selected():
                if not dataset_checkboxes:
                    return
                checked_count = sum(1 for cb in dataset_checkboxes if cb.isChecked())
                if checked_count <= 0:
                    # Force first item selected
                    try:
                        dataset_checkboxes[0].blockSignals(True)
                        dataset_checkboxes[0].setChecked(True)
                        dataset_checkboxes[0].blockSignals(False)
                    except Exception:
                        pass

            def _enforce_max_selected(changed_cb, checked: bool) -> None:
                if not dataset_checkboxes:
                    return

                if checked:
                    checked_count = sum(1 for cb in dataset_checkboxes if cb.isChecked())
                    if checked_count > MAX_PREVIEW_DATASETS:
                        # Revert this toggle.
                        try:
                            changed_cb.blockSignals(True)
                            changed_cb.setChecked(False)
                            changed_cb.blockSignals(False)
                        except Exception:
                            pass
                        _flash_limit_warning()
                        return

                _ensure_min_one_selected()
                _update_select_all_ui()
            
            # Update plot when any selection changes
            def update_spectrum_display():
                try:
                    mode = mode_combo.currentText()
                    # Check for localized "All Spectra" text
                    show_all = (mode == localize_func("ANALYSIS_PAGE.display_mode_all") or mode == "All Spectra")
                    
                    # Get selected datasets
                    selected_datasets = {}
                    for cb in dataset_checkboxes:
                        if cb.isChecked():
                            ds_name = cb.dataset_name
                            if ds_name in result.dataset_data:
                                selected_datasets[ds_name] = result.dataset_data[ds_name]
                    
                    # Prevent unselecting last item
                    if not selected_datasets:
                        _ensure_min_one_selected()
                        return
                    
                    # Generate figure
                    fig = create_spectrum_preview_figure(selected_datasets, show_all=show_all)
                    spectrum_tab.update_plot(fig)
                    plt.close(fig)
                    
                except Exception as e:
                    create_logs(
                        "MethodView",
                        "method_view",
                        f"Failed to update spectrum preview: {e}\n{traceback.format_exc()}",
                        status="error",
                    )

            # Connect all dataset visibility toggles (these are checkable buttons)
            for cb in dataset_checkboxes:
                try:
                    if hasattr(cb, "toggled"):
                        cb.toggled.connect(lambda checked, btn=cb: _enforce_max_selected(btn, checked))
                        cb.toggled.connect(update_spectrum_display)
                    else:
                        cb.stateChanged.connect(lambda _state, btn=cb: _enforce_max_selected(btn, bool(_state)))
                        cb.stateChanged.connect(update_spectrum_display)
                except Exception:
                    pass
            mode_combo.currentIndexChanged.connect(update_spectrum_display)

            # Default selection: show only 1 dataset to keep the preview fast.
            try:
                if dataset_checkboxes:
                    dataset_checkboxes[0].setChecked(True)
                _update_select_all_ui()
                update_spectrum_display()
            except Exception:
                pass

            tab_widget.addTab(spectrum_container, "📈 " + localize_func("ANALYSIS_PAGE.spectrum_preview_tab"))
        except Exception as e:
            create_logs(
                "MethodView",
                "method_view",
                f"Failed to create spectrum preview: {e}\n{traceback.format_exc()}",
                status="error",
            )

    # === Special handling for PCA Analysis: Create 5-tab visualization ===
    is_pca = hasattr(result, "raw_results") and "pca_model" in result.raw_results

    if is_pca:
        create_logs(
            "MethodView",
            "method_view",
            "PCA Analysis detected - creating 5-tab visualization",
            status="debug",
        )

        # Extract figures from raw_results
        scree_figure = result.raw_results.get("scree_figure")
        loadings_figure = result.raw_results.get("loadings_figure")
        biplot_figure = result.raw_results.get("biplot_figure")
        cumulative_variance_figure = result.raw_results.get(
            "cumulative_variance_figure"
        )
        distributions_figure = result.raw_results.get("distributions_figure")

        create_logs("MethodView", "method_view", "PCA figures found:", status="debug")
        create_logs(
            "MethodView",
            "method_view",
            f"  scree_figure: {scree_figure is not None}",
            status="debug",
        )
        create_logs(
            "MethodView",
            "method_view",
            f"  loadings_figure: {loadings_figure is not None}",
            status="debug",
        )
        create_logs(
            "MethodView",
            "method_view",
            f"  biplot_figure: {biplot_figure is not None}",
            status="debug",
        )
        create_logs(
            "MethodView",
            "method_view",
            f"  cumulative_variance_figure: {cumulative_variance_figure is not None}",
            status="debug",
        )
        create_logs(
            "MethodView",
            "method_view",
            f"  distributions_figure: {distributions_figure is not None}",
            status="debug",
        )

        # Tab 1: Score Plot (PC1 vs PC2)
        if result.primary_figure:
            score_tab = matplotlib_widget_class()
            score_tab.update_plot(result.primary_figure)
            score_tab.setMinimumHeight(400)
            tab_widget.addTab(score_tab, "📈 " + localize_func("ANALYSIS_PAGE.score_plot_tab"))

        # Optional: PCA→LDA dedicated tab (decision regions in PC1–PC2)
        try:
            pca_lda = (result.raw_results or {}).get("pca_lda")
        except Exception:
            pca_lda = None

        try:
            if isinstance(pca_lda, dict) and pca_lda.get("enabled"):
                scores = (result.raw_results or {}).get("scores")
                labels = (result.raw_results or {}).get("labels")
                unique_labels = (result.raw_results or {}).get("unique_labels")
                label_to_color = (result.raw_results or {}).get("label_to_color")
                colors = (result.raw_results or {}).get("colors")

                if scores is not None and labels is not None and unique_labels is not None:
                    from matplotlib.colors import ListedColormap
                    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

                    X_2d = np.asarray(scores, dtype=float)[:, :2]
                    labs = np.asarray(labels)
                    uniq = list(unique_labels)

                    # Robust color map per label
                    try:
                        color_map: dict[str, object] = {}
                        if isinstance(label_to_color, dict) and label_to_color:
                            color_map = {str(k): v for k, v in label_to_color.items()}
                        else:
                            if colors is not None and len(colors) > 0:
                                for idx, lab in enumerate(uniq):
                                    if idx < len(colors):
                                        color_map[str(lab)] = colors[idx]
                        if not color_map or len(color_map) < len(uniq):
                            cmap = plt.get_cmap("tab20", max(int(len(uniq)), 10))
                            for idx, lab in enumerate(uniq):
                                color_map.setdefault(str(lab), cmap(idx))
                    except Exception:
                        cmap = plt.get_cmap("tab20", max(int(len(uniq)), 10))
                        color_map = {str(lab): cmap(i) for i, lab in enumerate(uniq)}

                    label_to_int = {lab: idx for idx, lab in enumerate(uniq)}
                    y_int = np.array([label_to_int[l] for l in labs], dtype=int)

                    lda = LinearDiscriminantAnalysis()
                    lda.fit(X_2d, y_int)

                    x_min, x_max = float(np.min(X_2d[:, 0]) - 1.0), float(np.max(X_2d[:, 0]) + 1.0)
                    y_min, y_max = float(np.min(X_2d[:, 1]) - 1.0), float(np.max(X_2d[:, 1]) + 1.0)
                    xx, yy = np.meshgrid(
                        np.linspace(x_min, x_max, 250),
                        np.linspace(y_min, y_max, 250),
                    )
                    grid = np.c_[xx.ravel(), yy.ravel()]
                    Z = lda.predict(grid).reshape(xx.shape)

                    fig_lda, ax_lda = plt.subplots(figsize=(10, 8))
                    base_colors = [tuple(np.asarray(color_map[str(lab)])[:3]) for lab in uniq]
                    cmap_regions = ListedColormap(base_colors)
                    ax_lda.contourf(xx, yy, Z, cmap=cmap_regions, alpha=0.18, antialiased=True, zorder=0)
                    if len(uniq) == 2:
                        ax_lda.contour(xx, yy, Z, levels=[0.5], colors=["#333333"], linewidths=1.8, zorder=1)
                    else:
                        ax_lda.contour(xx, yy, Z, colors=["#333333"], linewidths=1.0, alpha=0.7, zorder=1)

                    for lab in uniq:
                        mask = (labs == lab)
                        ax_lda.scatter(
                            X_2d[mask, 0],
                            X_2d[mask, 1],
                            c=[color_map[str(lab)]],
                            label=str(lab),
                            alpha=0.7,
                            s=90,
                            edgecolors="white",
                            linewidth=1.0,
                            zorder=5,
                        )

                    ax_lda.set_xlabel("PC1", fontsize=12, fontweight="bold")
                    ax_lda.set_ylabel("PC2", fontsize=12, fontweight="bold")
                    ax_lda.set_title("PCA→LDA Decision Regions (PC1–PC2)", fontsize=14, fontweight="bold")
                    ax_lda.legend(loc="best", framealpha=0.95, fontsize=10, edgecolor="#cccccc", fancybox=True)
                    ax_lda.grid(True, alpha=0.3)
                    ax_lda.axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
                    ax_lda.axvline(x=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

                    # CV annotation if available
                    try:
                        acc = pca_lda.get("cv_accuracy")
                        folds = pca_lda.get("cv_folds_used")
                        if acc is not None:
                            ax_lda.text(
                                0.02,
                                0.98,
                                f"CV accuracy: {float(acc):.3f} (folds={folds})",
                                transform=ax_lda.transAxes,
                                fontsize=10,
                                verticalalignment="top",
                                bbox=dict(
                                    boxstyle="round,pad=0.4",
                                    facecolor="white",
                                    alpha=0.92,
                                    edgecolor="#cccccc",
                                    linewidth=0.6,
                                ),
                            )
                    except Exception:
                        pass

                    lda_tab = matplotlib_widget_class()
                    lda_tab.update_plot(fig_lda)
                    lda_tab.setMinimumHeight(400)
                    tab_widget.addTab(lda_tab, "🧠 PCA→LDA")
                    plt.close(fig_lda)
        except Exception:
            pass

        # Tab 2: Scree Plot
        if scree_figure:
            scree_tab = matplotlib_widget_class()
            scree_tab.update_plot(scree_figure)
            scree_tab.setMinimumHeight(400)
            tab_widget.addTab(scree_tab, "📊 " + localize_func("ANALYSIS_PAGE.scree_plot_tab"))

        # Tab 3: Loading Plot with 2-Column Grid and Collapsible Component Selector
        if loadings_figure and "pca_model" in result.raw_results:
            # Create main container
            loading_container = QWidget()
            loading_main_layout = QVBoxLayout(loading_container)
            loading_main_layout.setContentsMargins(0, 0, 0, 0)
            loading_main_layout.setSpacing(0)

            # TOP: Toolbar with toggle button
            toolbar_widget = QWidget()
            toolbar_layout = QHBoxLayout(toolbar_widget)
            toolbar_layout.setContentsMargins(8, 4, 8, 4)
            toolbar_layout.setSpacing(8)
            toolbar_widget.setStyleSheet("background-color: #f8f9fa; border-bottom: 1px solid #e0e0e0;")
            
            toggle_loading_btn = QPushButton(localize_func("ANALYSIS_PAGE.show_components_button_loading"))
            toggle_loading_btn.setCheckable(True)
            toggle_loading_btn.setChecked(False)  # Collapsed by default
            toggle_loading_btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 12px;
                    background-color: #0078d4;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #005a9e; }
                QPushButton:checked { background-color: #005a9e; }
            """)
            toolbar_layout.addWidget(toggle_loading_btn)
            toolbar_layout.addStretch()
            
            loading_main_layout.addWidget(toolbar_widget)
            
            # BOTTOM: Horizontal layout with collapsible sidebar
            content_widget = QWidget()
            content_layout = QHBoxLayout(content_widget)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(0)
            
            # LEFT: Collapsible Sidebar
            loading_sidebar = QFrame()
            loading_sidebar.setFixedWidth(220)
            loading_sidebar.setVisible(False)  # Hidden by default
            loading_sidebar.setStyleSheet(
                """
                QFrame {
                    background-color: #f8f9fa;
                    border-right: 1px solid #e0e0e0;
                }
            """
            )
            sidebar_layout = QVBoxLayout(loading_sidebar)
            sidebar_layout.setContentsMargins(12, 12, 12, 12)
            sidebar_layout.setSpacing(10)

            # Title
            comp_title_label = QLabel(localize_func("ANALYSIS_PAGE.select_components_loading"))
            comp_title_label.setStyleSheet(
                "font-weight: bold; font-size: 12px; color: #2c3e50; border: none;"
            )
            sidebar_layout.addWidget(comp_title_label)

            # Info label
            info_label = QLabel(localize_func("ANALYSIS_PAGE.loading_plot_info"))
            info_label.setStyleSheet("font-size: 10px; color: #6c757d; border: none;")
            info_label.setWordWrap(True)
            sidebar_layout.addWidget(info_label)
            
            # Get PCA model
            pca_model = result.raw_results["pca_model"]
            # IMPORTANT: allow selecting any PC up to n_components, but cap the
            # number of simultaneously displayed plots for UI readability.
            MAX_PCA_PLOTS = 6
            n_comps = int(getattr(pca_model, "n_components_", 0) or 0)

            PCA_SELECT_ALL_CB_QSS = """
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border: 2px solid #adb5bd;
                    border-radius: 4px;
                    background-color: white;
                }
                QCheckBox::indicator:hover {
                    border-color: #0078d4;
                }
                QCheckBox::indicator:checked {
                    background-color: #0078d4;
                    border-color: #0078d4;
                    image: url(assets/icons/checkmark_white.svg);
                }
                QCheckBox {
                    spacing: 6px;
                    font-size: 11px;
                    font-weight: bold;
                    color: #0078d4;
                    border: none;
                    padding: 4px;
                }
                QCheckBox:checked {
                    background-color: #e3f2fd;
                    border-radius: 3px;
                }
                QCheckBox:disabled {
                    color: #999999;
                }
            """
            PCA_COMPONENT_CB_QSS = """
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border: 2px solid #adb5bd;
                    border-radius: 4px;
                    background-color: white;
                }
                QCheckBox::indicator:hover {
                    border-color: #0078d4;
                }
                QCheckBox::indicator:checked {
                    background-color: #0078d4;
                    border-color: #0078d4;
                    image: url(assets/icons/checkmark_white.svg);
                }
                QCheckBox {
                    spacing: 6px;
                    font-size: 10px;
                    color: #495057;
                    border: none;
                    padding: 4px;
                }
                QCheckBox:checked {
                    background-color: #e3f2fd;
                    border-radius: 3px;
                }
            """
            
            def _select_all_label(all_selected: bool) -> str:
                try:
                    return (
                        localize_func("ANALYSIS_PAGE.deselect_all_button")
                        if all_selected
                        else localize_func("ANALYSIS_PAGE.select_all_button")
                    )
                except Exception:
                    return "Deselect All" if all_selected else "Select All"

            # "Select All" checkbox (action label)
            select_all_loading_cb = QCheckBox(_select_all_label(False))
            # Only enable Select All if total components <= MAX_PCA_PLOTS
            select_all_loading_cb.setEnabled(n_comps <= MAX_PCA_PLOTS)
            if n_comps > MAX_PCA_PLOTS:
                select_all_loading_cb.setToolTip(
                    f"Cannot select all: {n_comps} components available, max {MAX_PCA_PLOTS} plots at once"
                )
            select_all_loading_cb.setStyleSheet(PCA_SELECT_ALL_CB_QSS)
            sidebar_layout.addWidget(select_all_loading_cb)
            
            # Scrollable checkbox list for components
            comp_scroll = QScrollArea()
            comp_scroll.setWidgetResizable(True)
            comp_scroll.setStyleSheet("""
                QScrollArea {
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    background-color: white;
                }
            """)
            # ~8 items visible, then scroll.
            comp_scroll.setMaximumHeight(240)
            
            comp_list_widget = QWidget()
            comp_list_layout = QVBoxLayout(comp_list_widget)
            comp_list_layout.setContentsMargins(8, 8, 8, 8)
            try:
                comp_list_layout.setSpacing(6)
            except Exception:
                pass
            
            # Create checkboxes for each component
            loading_checkboxes = []
            loading_selection_order = []  # Track selection order for FIFO
            for i in range(n_comps):
                explained_var = pca_model.explained_variance_ratio_[i] * 100
                cb = QCheckBox(f"PC{i+1} ({explained_var:.1f}%)")
                # Default: show only 1 plot (PC1) until user expands selector.
                cb.setChecked(i == 0)
                cb.setStyleSheet(PCA_COMPONENT_CB_QSS)
                cb.pc_index = i  # Store PC index
                loading_checkboxes.append(cb)
                comp_list_layout.addWidget(cb)
            
            # Initialize selection order with defaults (PC1)
            for cb in loading_checkboxes:
                if cb.isChecked():
                    loading_selection_order.append(cb)
            comp_scroll.setWidget(comp_list_widget)
            sidebar_layout.addWidget(comp_scroll)
            
            sidebar_layout.addStretch()
            content_layout.addWidget(loading_sidebar)
            
            # RIGHT: Plot area
            loading_tab = matplotlib_widget_class()
            loading_tab.update_plot(loadings_figure)
            content_layout.addWidget(loading_tab, 1)
            
            loading_main_layout.addWidget(content_widget, 1)
            
            # Toggle sidebar visibility
            def toggle_loading_sidebar():
                loading_sidebar.setVisible(toggle_loading_btn.isChecked())
            
            toggle_loading_btn.toggled.connect(toggle_loading_sidebar)
            
            _loading_select_state = {"all_selected": False}

            def _update_loading_select_all_ui() -> None:
                total = len(loading_checkboxes)
                checked_count = sum(1 for cb in loading_checkboxes if cb.isChecked())
                all_selected = (total > 0 and checked_count == total)
                _loading_select_state["all_selected"] = all_selected

                select_all_loading_cb.blockSignals(True)
                select_all_loading_cb.setChecked(all_selected)
                select_all_loading_cb.setText(_select_all_label(all_selected))
                select_all_loading_cb.blockSignals(False)

            def on_loading_select_all_clicked(_checked: bool):
                # Only select all if within limit
                if len(loading_checkboxes) > MAX_PCA_PLOTS:
                    return

                if _loading_select_state.get("all_selected"):
                    for cb in loading_checkboxes:
                        cb.setChecked(False)
                else:
                    for cb in loading_checkboxes:
                        cb.setChecked(True)

                _update_loading_select_all_ui()

            select_all_loading_cb.clicked.connect(on_loading_select_all_clicked)
            
            # Prevent unchecking the last selected checkbox + FIFO logic
            def on_loading_checkbox_changed():
                checked_count = sum(1 for cb in loading_checkboxes if cb.isChecked())
                
                # FIFO Logic: If more than 6 selected, uncheck the first one
                if checked_count > MAX_PCA_PLOTS:
                    # Find the first checked checkbox in selection order
                    while loading_selection_order and not loading_selection_order[0].isChecked():
                        loading_selection_order.pop(0)  # Remove unchecked items from order

                    if loading_selection_order:
                        first_cb = loading_selection_order.pop(0)
                        first_cb.blockSignals(True)
                        first_cb.setChecked(False)
                        first_cb.blockSignals(False)

                    # Recount after FIFO uncheck
                    checked_count = sum(1 for cb in loading_checkboxes if cb.isChecked())
                
                # If only one checkbox is checked, keep it checked
                if checked_count == 1:
                    for cb in loading_checkboxes:
                        if cb.isChecked():
                            cb.blockSignals(True)
                            cb.setChecked(True)  # Force it to stay checked
                            cb.blockSignals(False)

                _update_loading_select_all_ui()
            
            # Track selection order for FIFO
            def track_loading_selection(cb):
                def handler():
                    if cb.isChecked():
                        # Add to selection order if newly checked
                        if cb not in loading_selection_order:
                            loading_selection_order.append(cb)
                    else:
                        # Remove from selection order if unchecked
                        if cb in loading_selection_order:
                            loading_selection_order.remove(cb)
                return handler
            
            # Update plot in 2-column grid when selection changes
            def update_loadings_grid():
                try:
                    # Get selected component indices (preserve selection order)
                    selected_indices = [
                        cb.pc_index
                        for cb in loading_selection_order
                        if cb is not None and cb.isChecked()
                    ]
                    for cb in loading_checkboxes:
                        if cb.isChecked() and cb.pc_index not in selected_indices:
                            selected_indices.append(cb.pc_index)
                    
                    # Prevent unselecting last item
                    if not selected_indices:
                        on_loading_checkbox_changed()
                        return
                    
                    # Limit to max plots (defensive; FIFO logic should already enforce this)
                    if len(selected_indices) > MAX_PCA_PLOTS:
                        selected_indices = selected_indices[-MAX_PCA_PLOTS:]
                    n_selected = len(selected_indices)
                    
                    # Prefer wavenumbers from analysis raw_results (more robust across grouped modes)
                    wavenumbers = result.raw_results.get("wavenumbers")
                    if wavenumbers is None:
                        # Fallback: infer from dataset_data if available
                        if not result.dataset_data:
                            return
                        first_df = next(iter(result.dataset_data.values()))
                        wavenumbers = first_df.index.values
                    
                    # Calculate grid dimensions: 2 columns, dynamic rows
                    # Examples: 8→4x2, 6→3x2, 5→3+2, 4→2x2, 3→2+1, 2→1x2, 1→1x1
                    n_cols = 2 if n_selected > 1 else 1
                    n_rows = int(np.ceil(n_selected / n_cols))
                    
                    # Create figure with 2-column grid - adjust title size when many plots
                    fig = plt.figure(figsize=(12, 3 * n_rows))
                    
                    # Dynamic color palette
                    cmap = plt.get_cmap('tab20', max(n_selected, 10))
                    colors = [cmap(i) for i in range(n_selected)]
                    
                    # Determine title fontsize based on number of subplots
                    title_fontsize = 9 if n_selected > 4 else 11
                    
                    for subplot_idx, pc_idx in enumerate(selected_indices):
                        # Calculate subplot position (1-indexed for matplotlib)
                        ax = fig.add_subplot(n_rows, n_cols, subplot_idx + 1)
                        
                        component = pca_model.components_[pc_idx]
                        explained_var = (
                            pca_model.explained_variance_ratio_[pc_idx] * 100
                        )

                        ax.plot(
                            wavenumbers,
                            component,
                            color=colors[subplot_idx],
                            linewidth=1.5,
                        )
                        ax.set_ylabel("Loading", fontsize=10)
                        ax.set_title(
                            f"PC{pc_idx+1} ({explained_var:.1f}%)",
                            fontsize=title_fontsize,
                            fontweight="bold",
                        )
                        ax.grid(True, alpha=0.3)
                        ax.invert_xaxis()
                        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
                        
                        # Determine if this subplot is in the PHYSICAL bottom row
                        # For 2-column layout: 
                        # - If we have even number (e.g., 6): indices 4,5 are bottom (row 2)
                        # - If we have odd number (e.g., 5): indices 4 is bottom of col 0, 5 would be bottom of col 1 if it existed
                        # The key is: a subplot is at bottom if there's NO subplot below it in its column
                        
                        # Calculate which row this subplot is in (0-indexed)
                        row_idx = subplot_idx // n_cols
                        col_idx = subplot_idx % n_cols
                        
                        # Check if there's another row below this one
                        # A subplot is at bottom if: row_idx == n_rows - 1
                        # OR if it's the last subplot in an odd-numbered grid (special case)
                        is_bottom_row = (row_idx == n_rows - 1)
                        
                        # Show x-axis labels and ticks only on bottom row
                        if is_bottom_row:
                            ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
                            ax.tick_params(axis='x', labelbottom=True, bottom=True)
                        else:
                            ax.set_xlabel('')  # Clear xlabel
                            ax.tick_params(axis='x', labelbottom=False, bottom=False)
                        
                        # Annotate top 3 peaks
                        abs_loadings = np.abs(component)
                        top_indices = np.argsort(abs_loadings)[-3:]
                        for peak_idx in top_indices:
                            peak_wn = wavenumbers[peak_idx]
                            peak_val = component[peak_idx]
                            ax.plot(
                                peak_wn,
                                peak_val,
                                "o",
                                color=colors[subplot_idx],
                                markersize=5,
                            )

                    # tight_layout can emit warnings for complex axes layouts; suppress spam.
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r"This figure includes Axes that are not compatible with tight_layout.*",
                            category=UserWarning,
                        )
                        try:
                            fig.tight_layout()
                        except Exception:
                            # Fallback to a reasonable default spacing.
                            fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.92, hspace=0.35, wspace=0.25)
                    loading_tab.update_plot(fig)
                    plt.close(fig)

                except Exception as e:
                    create_logs(
                        "MethodView",
                        "method_view",
                        f"Failed to update loadings grid: {e}\n{traceback.format_exc()}",
                        status="error",
                    )
            
            # Connect all checkboxes
            for cb in loading_checkboxes:
                cb.stateChanged.connect(track_loading_selection(cb))
                cb.stateChanged.connect(on_loading_checkbox_changed)
                cb.stateChanged.connect(update_loadings_grid)

            # Initialize Select All label/state
            try:
                _update_loading_select_all_ui()
            except Exception:
                pass

            # Ensure initial view matches default selection (PC1 only), regardless of
            # what the precomputed loadings_figure contained.
            try:
                QTimer.singleShot(0, update_loadings_grid)
            except Exception:
                pass
            
            tab_widget.addTab(loading_container, "🔬 " + localize_func("ANALYSIS_PAGE.loading_plot_tab"))
        elif loadings_figure:
            # Fallback if no model
            loading_tab = matplotlib_widget_class()
            loading_tab.update_plot(loadings_figure)
            loading_tab.setMinimumHeight(400)
            tab_widget.addTab(loading_tab, "🔬 " + localize_func("ANALYSIS_PAGE.loading_plot_tab"))
        
        # Tab 4: Biplot
        if biplot_figure:
            biplot_tab = matplotlib_widget_class()
            biplot_tab.update_plot(biplot_figure)
            biplot_tab.setMinimumHeight(400)
            tab_widget.addTab(biplot_tab, "🎯 " + localize_func("ANALYSIS_PAGE.biplot_tab"))

        # Tab 5: Cumulative Variance - REMOVED (redundant with Scree Plot)
        # if cumulative_variance_figure:
        #     cumvar_tab = matplotlib_widget_class()
        #     cumvar_tab.update_plot(cumulative_variance_figure)
        #     cumvar_tab.setMinimumHeight(400)
        #     tab_widget.addTab(cumvar_tab, "📈 Cumulative Variance")

        # Tab 5: Distributions (with component selector)
        if distributions_figure and "pca_model" in result.raw_results:
            # Create main container
            dist_container = QWidget()
            dist_main_layout = QVBoxLayout(dist_container)
            dist_main_layout.setContentsMargins(0, 0, 0, 0)
            dist_main_layout.setSpacing(0)
            
            # TOP: Toolbar with toggle button
            dist_toolbar_widget = QWidget()
            dist_toolbar_layout = QHBoxLayout(dist_toolbar_widget)
            dist_toolbar_layout.setContentsMargins(8, 4, 8, 4)
            dist_toolbar_layout.setSpacing(8)
            dist_toolbar_widget.setStyleSheet("background-color: #f8f9fa; border-bottom: 1px solid #e0e0e0;")
            
            toggle_dist_btn = QPushButton(localize_func("ANALYSIS_PAGE.show_components_button_dist"))
            toggle_dist_btn.setCheckable(True)
            toggle_dist_btn.setChecked(False)  # Collapsed by default
            toggle_dist_btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 12px;
                    background-color: #0078d4;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #005a9e; }
                QPushButton:checked { background-color: #005a9e; }
            """)
            dist_toolbar_layout.addWidget(toggle_dist_btn)
            dist_toolbar_layout.addStretch()
            
            dist_main_layout.addWidget(dist_toolbar_widget)
            
            # BOTTOM: Horizontal layout with collapsible sidebar
            dist_content_widget = QWidget()
            dist_content_layout = QHBoxLayout(dist_content_widget)
            dist_content_layout.setContentsMargins(0, 0, 0, 0)
            dist_content_layout.setSpacing(0)
            
            # LEFT: Collapsible Sidebar
            dist_sidebar = QFrame()
            dist_sidebar.setFixedWidth(220)
            dist_sidebar.setVisible(False)  # Hidden by default
            dist_sidebar.setStyleSheet("""
                QFrame {
                    background-color: #f8f9fa;
                    border-right: 1px solid #e0e0e0;
                }
            """)
            dist_sidebar_layout = QVBoxLayout(dist_sidebar)
            dist_sidebar_layout.setContentsMargins(12, 12, 12, 12)
            dist_sidebar_layout.setSpacing(10)
            
            # Title
            dist_comp_title_label = QLabel(localize_func("ANALYSIS_PAGE.select_components_dist"))
            dist_comp_title_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #2c3e50; border: none;")
            dist_sidebar_layout.addWidget(dist_comp_title_label)
            
            # Info label
            dist_info_label = QLabel(localize_func("ANALYSIS_PAGE.distributions_plot_info"))
            dist_info_label.setStyleSheet("font-size: 10px; color: #6c757d; border: none;")
            dist_info_label.setWordWrap(True)
            dist_sidebar_layout.addWidget(dist_info_label)
            
            # Get PCA model
            pca_model = result.raw_results["pca_model"]
            # IMPORTANT: allow selecting any PC up to n_components, but cap the
            # number of simultaneously displayed plots for UI readability.
            MAX_PCA_PLOTS = 6
            n_comps = int(getattr(pca_model, "n_components_", 0) or 0)
            
            def _select_all_label(all_selected: bool) -> str:
                try:
                    return (
                        localize_func("ANALYSIS_PAGE.deselect_all_button")
                        if all_selected
                        else localize_func("ANALYSIS_PAGE.select_all_button")
                    )
                except Exception:
                    return "Deselect All" if all_selected else "Select All"

            # "Select All" checkbox (action label)
            select_all_dist_cb = QCheckBox(_select_all_label(False))
            # Only enable Select All if total components <= MAX_PCA_PLOTS
            select_all_dist_cb.setEnabled(n_comps <= MAX_PCA_PLOTS)
            if n_comps > MAX_PCA_PLOTS:
                select_all_dist_cb.setToolTip(
                    f"Cannot select all: {n_comps} components available, max {MAX_PCA_PLOTS} plots at once"
                )
            select_all_dist_cb.setStyleSheet(PCA_SELECT_ALL_CB_QSS)
            dist_sidebar_layout.addWidget(select_all_dist_cb)
            
            # Scrollable checkbox list for components
            dist_comp_scroll = QScrollArea()
            dist_comp_scroll.setWidgetResizable(True)
            dist_comp_scroll.setStyleSheet("""
                QScrollArea {
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    background-color: white;
                }
            """)
            # ~8 items visible, then scroll.
            dist_comp_scroll.setMaximumHeight(240)
            
            dist_comp_list_widget = QWidget()
            dist_comp_list_layout = QVBoxLayout(dist_comp_list_widget)
            dist_comp_list_layout.setContentsMargins(8, 8, 8, 8)
            try:
                dist_comp_list_layout.setSpacing(6)
            except Exception:
                pass
            
            # Create checkboxes for each component
            dist_checkboxes = []
            dist_selection_order = []  # Track selection order for FIFO
            for i in range(n_comps):
                explained_var = pca_model.explained_variance_ratio_[i] * 100
                cb = QCheckBox(f"PC{i+1} ({explained_var:.1f}%)")
                # Default: show only 1 plot (PC1) until user expands selector.
                cb.setChecked(i == 0)
                cb.setStyleSheet(PCA_COMPONENT_CB_QSS)
                cb.pc_index = i  # Store PC index
                dist_checkboxes.append(cb)
                dist_comp_list_layout.addWidget(cb)
            
            # Initialize selection order with defaults (PC1)
            for cb in dist_checkboxes:
                if cb.isChecked():
                    dist_selection_order.append(cb)
            dist_comp_scroll.setWidget(dist_comp_list_widget)
            dist_sidebar_layout.addWidget(dist_comp_scroll)
            
            dist_sidebar_layout.addStretch()
            dist_content_layout.addWidget(dist_sidebar)
            
            # RIGHT: Plot area
            dist_tab = matplotlib_widget_class()
            dist_tab.update_plot(distributions_figure)
            dist_content_layout.addWidget(dist_tab, 1)
            
            dist_main_layout.addWidget(dist_content_widget, 1)
            
            # Toggle sidebar visibility
            def toggle_dist_sidebar():
                dist_sidebar.setVisible(toggle_dist_btn.isChecked())
            
            toggle_dist_btn.toggled.connect(toggle_dist_sidebar)
            
            _dist_select_state = {"all_selected": False}

            def _update_dist_select_all_ui() -> None:
                total = len(dist_checkboxes)
                checked_count = sum(1 for cb in dist_checkboxes if cb.isChecked())
                all_selected = (total > 0 and checked_count == total)
                _dist_select_state["all_selected"] = all_selected

                select_all_dist_cb.blockSignals(True)
                select_all_dist_cb.setChecked(all_selected)
                select_all_dist_cb.setText(_select_all_label(all_selected))
                select_all_dist_cb.blockSignals(False)

            def on_dist_select_all_clicked(_checked: bool):
                if len(dist_checkboxes) > MAX_PCA_PLOTS:
                    return

                if _dist_select_state.get("all_selected"):
                    for cb in dist_checkboxes:
                        cb.setChecked(False)
                else:
                    for cb in dist_checkboxes:
                        cb.setChecked(True)

                _update_dist_select_all_ui()

            select_all_dist_cb.clicked.connect(on_dist_select_all_clicked)
            
            # Prevent unchecking the last selected checkbox + FIFO logic
            def on_dist_checkbox_changed():
                checked_count = sum(1 for cb in dist_checkboxes if cb.isChecked())
                
                # FIFO Logic: If more than 6 selected, uncheck the first one
                if checked_count > MAX_PCA_PLOTS:
                    # Find the first checked checkbox in selection order
                    while dist_selection_order and not dist_selection_order[0].isChecked():
                        dist_selection_order.pop(0)  # Remove unchecked items from order
                    
                    if dist_selection_order:
                        first_cb = dist_selection_order.pop(0)
                        first_cb.blockSignals(True)
                        first_cb.setChecked(False)
                        first_cb.blockSignals(False)
                    
                    # Recount after FIFO uncheck
                    checked_count = sum(1 for cb in dist_checkboxes if cb.isChecked())
                
                # If only one checkbox is checked, keep it checked
                if checked_count == 1:
                    for cb in dist_checkboxes:
                        if cb.isChecked():
                            cb.blockSignals(True)
                            cb.setChecked(True)  # Force it to stay checked
                            cb.blockSignals(False)

                _update_dist_select_all_ui()
            
            # Track selection order for FIFO
            def track_dist_selection(cb):
                def handler():
                    if cb.isChecked():
                        # Add to selection order if newly checked
                        if cb not in dist_selection_order:
                            dist_selection_order.append(cb)
                    else:
                        # Remove from selection order if unchecked
                        if cb in dist_selection_order:
                            dist_selection_order.remove(cb)
                return handler
            
            # Update distributions plot when selection changes
            def update_distributions():
                try:
                    # Get selected component indices (preserve selection order)
                    selected_indices = [
                        cb.pc_index
                        for cb in dist_selection_order
                        if cb is not None and cb.isChecked()
                    ]
                    for cb in dist_checkboxes:
                        if cb.isChecked() and cb.pc_index not in selected_indices:
                            selected_indices.append(cb.pc_index)
                    
                    # Prevent unselecting last item
                    if not selected_indices:
                        on_dist_checkbox_changed()
                        return
                    
                    # Limit to max plots (defensive; FIFO logic should already enforce this)
                    if len(selected_indices) > MAX_PCA_PLOTS:
                        selected_indices = selected_indices[-MAX_PCA_PLOTS:]
                    n_selected = len(selected_indices)
                    
                    # Get scores and labels from result
                    scores = result.raw_results.get("scores")
                    labels = result.raw_results.get("labels")
                    unique_labels = result.raw_results.get("unique_labels")
                    colors = result.raw_results.get("colors")
                    label_to_color = result.raw_results.get("label_to_color")
                    
                    if scores is None or labels is None or unique_labels is None:
                        return
                    
                    # Calculate grid dimensions (max 2 columns)
                    n_cols = 2 if n_selected > 1 else 1
                    n_rows = int(np.ceil(n_selected / n_cols))
                    
                    # Create figure
                    fig = plt.figure(figsize=(6*n_cols, 4*n_rows))
                    
                    labels_arr = np.asarray(labels)

                    # Robust color assignment: prefer explicit label→color mapping.
                    # Fallback: derive mapping from `colors` + `unique_labels`.
                    # Final fallback: generate a palette large enough for all labels.
                    try:
                        color_map: dict[str, object] = {}
                        if isinstance(label_to_color, dict) and label_to_color:
                            color_map = {str(k): v for k, v in label_to_color.items()}
                        else:
                            if colors is not None and len(colors) > 0:
                                for idx, lab in enumerate(list(unique_labels)):
                                    if idx < len(colors):
                                        color_map[str(lab)] = colors[idx]

                        if not color_map or len(color_map) < len(unique_labels):
                            cmap = plt.get_cmap("tab20", max(int(len(unique_labels)), 10))
                            for idx, lab in enumerate(list(unique_labels)):
                                color_map.setdefault(str(lab), cmap(idx))
                    except Exception:
                        cmap = plt.get_cmap("tab20", max(int(len(unique_labels)), 10))
                        color_map = {str(lab): cmap(i) for i, lab in enumerate(list(unique_labels))}

                    for subplot_idx, pc_idx in enumerate(selected_indices):
                        ax = fig.add_subplot(n_rows, n_cols, subplot_idx + 1)

                        # Build per-group score arrays first so we can create a shared x-range.
                        group_scores: list[tuple[str, np.ndarray]] = []
                        for dataset_label in unique_labels:
                            try:
                                mask = labels_arr == dataset_label
                            except Exception:
                                mask = np.array([l == dataset_label for l in labels_arr], dtype=bool)
                            pc_scores = np.asarray(scores[mask, pc_idx], dtype=float)
                            pc_scores = pc_scores[np.isfinite(pc_scores)]
                            group_scores.append((dataset_label, pc_scores))

                        # Shared x-range across groups for this PC (prevents 'weird' overlays).
                        all_scores = np.concatenate([s for _, s in group_scores if s.size > 0]) if group_scores else np.array([], dtype=float)
                        if all_scores.size:
                            lo = float(np.nanpercentile(all_scores, 1))
                            hi = float(np.nanpercentile(all_scores, 99))
                        else:
                            lo, hi = -1.0, 1.0
                        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                            lo, hi = float(np.nanmin(all_scores)) if all_scores.size else -1.0, float(np.nanmax(all_scores)) if all_scores.size else 1.0
                        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                            lo, hi = -1.0, 1.0

                        pad = 0.10 * (hi - lo) if hi > lo else 1.0
                        x_range = np.linspace(lo - pad, hi + pad, 240)

                        # Plot distributions for each dataset
                        for i, (dataset_label, pc_scores) in enumerate(group_scores):
                            color = color_map.get(str(dataset_label))
                            if pc_scores.size == 0:
                                continue

                            # Histogram (always)
                            ax.hist(
                                pc_scores,
                                bins=24,
                                density=True,
                                alpha=0.12,
                                color=color,
                                edgecolor="white",
                                linewidth=0.5,
                            )

                            # KDE (only when variance is non-degenerate)
                            try:
                                if pc_scores.size >= 3 and float(np.nanstd(pc_scores)) > 1e-9:
                                    kde = stats.gaussian_kde(pc_scores)
                                    kde_values = kde(x_range)
                                    ax.plot(
                                        x_range,
                                        kde_values,
                                        color=color,
                                        linewidth=2.2,
                                        label=str(dataset_label),
                                        alpha=0.95,
                                    )
                                    ax.fill_between(x_range, kde_values, alpha=0.18, color=color)
                            except Exception:
                                # Non-fatal: keep histogram.
                                pass
                        
                        explained_var = pca_model.explained_variance_ratio_[pc_idx] * 100
                        ax.set_title(f'PC{pc_idx+1} ({explained_var:.1f}%)', fontsize=11, fontweight='bold')
                        ax.set_xlabel(f'PC{pc_idx+1} Score', fontsize=10)
                        ax.set_ylabel('Density', fontsize=10)
                        ax.legend(fontsize=9)
                        ax.grid(True, alpha=0.3)
                    
                    fig.suptitle('PC Score Distributions', fontsize=14, fontweight='bold')
                    # tight_layout can emit warnings for complex axes layouts; suppress spam.
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r"This figure includes Axes that are not compatible with tight_layout.*",
                            category=UserWarning,
                        )
                        try:
                            fig.tight_layout()
                        except Exception:
                            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.90, hspace=0.35, wspace=0.25)
                    dist_tab.update_plot(fig)
                    plt.close(fig)
                    
                except Exception as e:
                    create_logs(
                        "MethodView",
                        "method_view",
                        f"Failed to update distributions: {e}\n{traceback.format_exc()}",
                        status="error",
                    )
            
            # Connect all checkboxes
            for cb in dist_checkboxes:
                cb.stateChanged.connect(track_dist_selection(cb))
                cb.stateChanged.connect(on_dist_checkbox_changed)
                cb.stateChanged.connect(update_distributions)

            # Initialize Select All label/state
            try:
                _update_dist_select_all_ui()
            except Exception:
                pass

            # Ensure initial view matches default selection (PC1 only), regardless of
            # what the precomputed distributions_figure contained.
            try:
                QTimer.singleShot(0, update_distributions)
            except Exception:
                pass
            
            tab_widget.addTab(dist_container, "📊 " + localize_func("ANALYSIS_PAGE.distributions_tab_pca"))
        elif distributions_figure:
            # Fallback if no model available
            dist_tab = matplotlib_widget_class()
            dist_tab.update_plot(distributions_figure)
            dist_tab.setMinimumHeight(400)
            tab_widget.addTab(dist_tab, "📊 " + localize_func("ANALYSIS_PAGE.distributions_tab_pca"))
    
    else:
        # === Standard analysis results (non-PCA) ===

        # === Main Plot Tab (Scores/Primary Figure) ===
        if result.primary_figure:
            plot_tab = matplotlib_widget_class()
            plot_tab.update_plot(result.primary_figure)
            plot_tab.setMinimumHeight(400)
            # PLS-DA: explicitly present as a Scores scatter tab (LV1 vs LV2)
            if getattr(result, "method_key", None) == "pls_da":
                plot_title = "📈 " + localize_func("ANALYSIS_PAGE.scores_tab")
            else:
                plot_title = (
                    "📈 " + localize_func("ANALYSIS_PAGE.scores_tab")
                    if hasattr(result, "loadings_figure")
                    else localize_func("ANALYSIS_PAGE.plot_tab")
                )
            tab_widget.addTab(
                plot_tab,
                plot_title,
            )

        def _select_all_label(all_selected: bool) -> str:
            try:
                return (
                    localize_func("ANALYSIS_PAGE.deselect_all_button")
                    if all_selected
                    else localize_func("ANALYSIS_PAGE.select_all_button")
                )
            except Exception:
                return "Deselect All" if all_selected else "Select All"

        rr = getattr(result, "raw_results", {}) or {}
        component_spectra = rr.get("component_spectra")
        wavenumbers = rr.get("wavenumbers")
        comp_prefix = str(rr.get("component_prefix") or "Comp")
        comp_ylabel = str(rr.get("component_y_label") or "Loading")

        # --- PLS-DA: implement PCA-like Loadings + Distributions tabs ---
        try:
            if result.method_key == "pls_da" and component_spectra is not None and wavenumbers is not None:
                component_spectra = np.asarray(component_spectra, dtype=float)
                wavenumbers = np.asarray(wavenumbers, dtype=float)
                n_comps = int(component_spectra.shape[1]) if component_spectra.ndim == 2 else 0

                if n_comps >= 1:
                    lv_labels = [f"LV{i+1}" for i in range(n_comps)]

                    def _update_pls_loadings(selected_indices: list[int]) -> None:
                        try:
                            if not selected_indices:
                                return
                            n_selected = len(selected_indices)
                            n_cols = 2 if n_selected > 1 else 1
                            n_rows = int(np.ceil(n_selected / n_cols))
                            fig = plt.figure(figsize=(12, 3 * n_rows))
                            cmap = plt.get_cmap("tab20", max(n_selected, 10))
                            colors = [cmap(i) for i in range(n_selected)]
                            title_fontsize = 9 if n_selected > 4 else 11

                            for subplot_idx, comp_idx in enumerate(selected_indices):
                                ax = fig.add_subplot(n_rows, n_cols, subplot_idx + 1)
                                y = component_spectra[:, comp_idx]
                                ax.plot(wavenumbers, y, color=colors[subplot_idx], linewidth=1.5)
                                ax.set_ylabel(comp_ylabel, fontsize=10)
                                ax.set_title(
                                    f"{comp_prefix}{comp_idx+1}",
                                    fontsize=title_fontsize,
                                    fontweight="bold",
                                )
                                ax.grid(True, alpha=0.3)
                                ax.invert_xaxis()
                                ax.axhline(y=0, color="k", linestyle="--", linewidth=0.6, alpha=0.45)
                                row_idx = subplot_idx // n_cols
                                if row_idx == n_rows - 1:
                                    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
                                else:
                                    ax.tick_params(axis="x", labelbottom=False, bottom=False)
                            try:
                                fig.tight_layout()
                            except Exception:
                                pass
                            pls_loadings_panel.update_figure(fig)
                            plt.close(fig)
                        except Exception as e:
                            create_logs(
                                "MethodView",
                                "method_view",
                                f"Failed to update PLS-DA loadings grid: {e}\n{traceback.format_exc()}",
                                status="error",
                            )

                    pls_loadings_panel = ComponentSelectorPlotPanel(
                        toggle_text=localize_func("ANALYSIS_PAGE.show_components_button_loading")
                        if hasattr(localize_func, "__call__")
                        else "Show components",
                        title_text="Select components",
                        info_text=f"Select components to display (max 6, 2-column grid).",
                        item_labels=lv_labels,
                        max_selected=6,
                        initial_checked=[0],
                        select_all_label_func=_select_all_label,
                        on_update=_update_pls_loadings,
                        plot_widget_factory=matplotlib_widget_class,
                    )
                    tab_widget.addTab(pls_loadings_panel, "🔬 Loadings")

                    # Distributions (LV score distributions by label)
                    x_scores = rr.get("x_scores")
                    labels = rr.get("labels")
                    if x_scores is not None and labels is not None:
                        x_scores = np.asarray(x_scores, dtype=float)
                        labels_arr = np.asarray(labels)
                        uniq = rr.get("unique_labels")
                        if uniq is None:
                            uniq = sorted(set([str(v) for v in labels_arr]))

                        # Color mapping (consistent with PCA logic)
                        label_to_color = rr.get("label_to_color")
                        colors = rr.get("colors")
                        try:
                            color_map: dict[str, object] = {}
                            if isinstance(label_to_color, dict) and label_to_color:
                                color_map = {str(k): v for k, v in label_to_color.items()}
                            else:
                                if colors is not None and len(colors) > 0:
                                    for idx, lab in enumerate(list(uniq)):
                                        if idx < len(colors):
                                            color_map[str(lab)] = colors[idx]
                            if not color_map or len(color_map) < len(uniq):
                                cmap = plt.get_cmap("tab20", max(int(len(uniq)), 10))
                                for idx, lab in enumerate(list(uniq)):
                                    color_map.setdefault(str(lab), cmap(idx))
                        except Exception:
                            cmap = plt.get_cmap("tab20", max(int(len(uniq)), 10))
                            color_map = {str(lab): cmap(i) for i, lab in enumerate(list(uniq))}

                        def _update_pls_distributions(selected_indices: list[int]) -> None:
                            try:
                                if not selected_indices:
                                    return
                                n_selected = len(selected_indices)
                                n_cols = 2 if n_selected > 1 else 1
                                n_rows = int(np.ceil(n_selected / n_cols))
                                fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))

                                for subplot_idx, comp_idx in enumerate(selected_indices):
                                    ax = fig.add_subplot(n_rows, n_cols, subplot_idx + 1)

                                    # Build per-group score arrays first for shared range.
                                    group_scores = []
                                    for lab in list(uniq):
                                        m = labels_arr == lab
                                        s = np.asarray(x_scores[m, comp_idx], dtype=float)
                                        s = s[np.isfinite(s)]
                                        group_scores.append((lab, s))

                                    all_scores = (
                                        np.concatenate([s for _, s in group_scores if s.size > 0])
                                        if group_scores
                                        else np.array([], dtype=float)
                                    )
                                    if all_scores.size:
                                        lo = float(np.nanpercentile(all_scores, 1))
                                        hi = float(np.nanpercentile(all_scores, 99))
                                    else:
                                        lo, hi = -1.0, 1.0
                                    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                                        lo = float(np.nanmin(all_scores)) if all_scores.size else -1.0
                                        hi = float(np.nanmax(all_scores)) if all_scores.size else 1.0
                                    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                                        lo, hi = -1.0, 1.0
                                    pad = 0.10 * (hi - lo) if hi > lo else 1.0
                                    x_range = np.linspace(lo - pad, hi + pad, 240)

                                    for lab, s in group_scores:
                                        c = color_map.get(str(lab))
                                        if s.size == 0:
                                            continue
                                        ax.hist(
                                            s,
                                            bins=24,
                                            density=True,
                                            alpha=0.12,
                                            color=c,
                                            edgecolor="white",
                                            linewidth=0.5,
                                        )
                                        try:
                                            if s.size >= 3 and float(np.nanstd(s)) > 1e-9:
                                                kde = stats.gaussian_kde(s)
                                                kde_values = kde(x_range)
                                                ax.plot(
                                                    x_range,
                                                    kde_values,
                                                    color=c,
                                                    linewidth=2.2,
                                                    label=str(lab),
                                                    alpha=0.95,
                                                )
                                                ax.fill_between(x_range, kde_values, alpha=0.18, color=c)
                                        except Exception:
                                            pass

                                    ax.set_title(f"{comp_prefix}{comp_idx+1}", fontsize=11, fontweight="bold")
                                    ax.set_xlabel(f"{comp_prefix}{comp_idx+1} Score", fontsize=10)
                                    ax.set_ylabel("Density", fontsize=10)
                                    ax.legend(fontsize=9)
                                    ax.grid(True, alpha=0.3)

                                fig.suptitle("LV Score Distributions (PLS-DA)", fontsize=14, fontweight="bold")
                                try:
                                    fig.tight_layout()
                                except Exception:
                                    pass
                                pls_dist_panel.update_figure(fig)
                                plt.close(fig)
                            except Exception as e:
                                create_logs(
                                    "MethodView",
                                    "method_view",
                                    f"Failed to update PLS-DA distributions: {e}\n{traceback.format_exc()}",
                                    status="error",
                                )

                        pls_dist_panel = ComponentSelectorPlotPanel(
                            toggle_text=localize_func("ANALYSIS_PAGE.show_components_button_dist")
                            if hasattr(localize_func, "__call__")
                            else "Show components",
                            title_text="Select components",
                            info_text=f"Select components to display (max 6, 2-column grid).",
                            item_labels=lv_labels,
                            max_selected=6,
                            initial_checked=[0],
                            select_all_label_func=_select_all_label,
                            on_update=_update_pls_distributions,
                            plot_widget_factory=matplotlib_widget_class,
                        )
                        tab_widget.addTab(pls_dist_panel, "📊 Distributions")
        except Exception:
            # Don't let optional PLS-DA extras crash other results.
            pass

        # --- Generic Components tab (MCR/NMF/ICA and others that provide component_spectra) ---
        try:
            if component_spectra is not None and wavenumbers is not None and result.method_key != "pls_da":
                component_spectra = np.asarray(component_spectra, dtype=float)
                wavenumbers = np.asarray(wavenumbers, dtype=float)
                if component_spectra.ndim == 2 and component_spectra.shape[1] >= 1:
                    n_comps = int(component_spectra.shape[1])
                    item_labels = [f"{comp_prefix}{i+1}" for i in range(n_comps)]

                    def _update_components(selected_indices: list[int]) -> None:
                        try:
                            if not selected_indices:
                                return
                            n_selected = len(selected_indices)
                            n_cols = 2 if n_selected > 1 else 1
                            n_rows = int(np.ceil(n_selected / n_cols))
                            fig = plt.figure(figsize=(12, 3 * n_rows))
                            cmap = plt.get_cmap("tab20", max(n_selected, 10))
                            colors = [cmap(i) for i in range(n_selected)]
                            title_fontsize = 9 if n_selected > 4 else 11

                            for subplot_idx, comp_idx in enumerate(selected_indices):
                                ax = fig.add_subplot(n_rows, n_cols, subplot_idx + 1)
                                y = component_spectra[:, comp_idx]
                                ax.plot(wavenumbers, y, color=colors[subplot_idx], linewidth=1.5)
                                ax.set_ylabel(comp_ylabel, fontsize=10)
                                ax.set_title(
                                    f"{comp_prefix}{comp_idx+1}",
                                    fontsize=title_fontsize,
                                    fontweight="bold",
                                )
                                ax.grid(True, alpha=0.3)
                                ax.invert_xaxis()
                                ax.axhline(y=0, color="k", linestyle="--", linewidth=0.6, alpha=0.45)
                                row_idx = subplot_idx // n_cols
                                if row_idx == n_rows - 1:
                                    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
                                else:
                                    ax.tick_params(axis="x", labelbottom=False, bottom=False)
                            try:
                                fig.tight_layout()
                            except Exception:
                                pass
                            comp_panel.update_figure(fig)
                            plt.close(fig)
                        except Exception as e:
                            create_logs(
                                "MethodView",
                                "method_view",
                                f"Failed to update component viewer: {e}\n{traceback.format_exc()}",
                                status="error",
                            )

                    comp_panel = ComponentSelectorPlotPanel(
                        toggle_text="Show components",
                        title_text="Select components",
                        info_text=f"Select components to display (max 6, 2-column grid).",
                        item_labels=item_labels,
                        max_selected=6,
                        initial_checked=[0],
                        select_all_label_func=_select_all_label,
                        on_update=_update_components,
                        plot_widget_factory=matplotlib_widget_class,
                    )
                    tab_widget.addTab(comp_panel, "🧩 Components")
        except Exception:
            pass

        # === Loadings Tab (for dimensionality reduction) ===
        create_logs("MethodView", "method_view", "Checking loadings_figure...", status="debug")
        create_logs(
            "MethodView",
            "method_view",
            f"  hasattr(result, 'loadings_figure'): {hasattr(result, 'loadings_figure')}",
            status="debug",
        )
        if hasattr(result, "loadings_figure"):
            create_logs(
                "MethodView",
                "method_view",
                f"  result.loadings_figure is not None: {result.loadings_figure is not None}",
                status="debug",
            )
            create_logs(
                "MethodView",
                "method_view",
                f"  result.loadings_figure type: {type(result.loadings_figure)}",
                status="debug",
            )

        if hasattr(result, "loadings_figure") and result.loadings_figure:
            create_logs("MethodView", "method_view", "Creating Loadings tab...", status="debug")
            loadings_tab = matplotlib_widget_class()
            loadings_tab.update_plot(result.loadings_figure)
            loadings_tab.setMinimumHeight(400)
            tab_widget.addTab(
                loadings_tab, "🔬 " + localize_func("ANALYSIS_PAGE.loadings_tab")
            )
            create_logs(
                "MethodView",
                "method_view",
                "Loadings tab added successfully",
                status="debug",
            )
        else:
            create_logs(
                "MethodView",
                "method_view",
                "Loadings tab NOT created (figure missing or None)",
                status="debug",
            )

        # === Distributions Tab (for classification) ===
        if hasattr(result, "distributions_figure") and result.distributions_figure:
            dist_tab = matplotlib_widget_class()
            dist_tab.update_plot(result.distributions_figure)
            dist_tab.setMinimumHeight(400)
            tab_widget.addTab(
                dist_tab, "📊 " + localize_func("ANALYSIS_PAGE.distributions_tab")
            )

        # === Legacy Secondary Figure Tab (deprecated but kept for compatibility) ===
        if hasattr(result, "secondary_figure") and result.secondary_figure:
            secondary_tab = matplotlib_widget_class()
            secondary_tab.update_plot(result.secondary_figure)
            secondary_tab.setMinimumHeight(400)
            tab_widget.addTab(
                secondary_tab, "📊 " + localize_func("ANALYSIS_PAGE.secondary_plot_tab")
            )

    # === Data Table Tab ===
    if result.data_table is not None:
        table_tab = create_data_table_tab(result.data_table)
        tab_widget.addTab(table_tab, "📋 " + localize_func("ANALYSIS_PAGE.data_tab"))

    # === Summary Tab ===
    if result.detailed_summary:
        summary_tab = create_summary_tab(result.detailed_summary)
        tab_widget.addTab(
            summary_tab, "📋 " + localize_func("ANALYSIS_PAGE.summary_tab")
        )

    # === Diagnostics Tab (if available) ===
    if hasattr(result, "diagnostics") and result.diagnostics:
        diag_tab = create_summary_tab(result.diagnostics)
        tab_widget.addTab(diag_tab, "🔍 " + localize_func("ANALYSIS_PAGE.diagnostics_tab"))
    
    # RESTORE the previous tab index to avoid jumping to first tab on analysis restart
    if current_tab_index < tab_widget.count():
        tab_widget.setCurrentIndex(current_tab_index)
    else:
        # If previous index is now out of bounds, default to first tab
        tab_widget.setCurrentIndex(0)


def create_data_table_tab(data_table) -> QWidget:
    """
    Create data table tab from pandas DataFrame or dict.

    Args:
        data_table: DataFrame or dict containing tabular data

    Returns:
        Table widget
    """
    table_widget = QTableWidget()
    table_widget.setStyleSheet(
        """
        QTableWidget {
            border: none;
            gridline-color: #e0e0e0;
            background-color: white;
        }
        QHeaderView::section {
            background-color: #f8f9fa;
            padding: 8px;
            border: none;
            border-bottom: 2px solid #e0e0e0;
            font-weight: 600;
        }
        QTableWidget::item {
            padding: 6px;
        }
    """
    )

    # Convert to DataFrame if dict
    import pandas as pd

    if isinstance(data_table, dict):
        df = pd.DataFrame(data_table)
    else:
        df = data_table

    # Reset index to ensure integer-based row access
    # This fixes TypeError when DataFrame has non-integer index (e.g., string labels)
    df = df.reset_index(drop=True)

    # Populate table
    table_widget.setRowCount(len(df))
    table_widget.setColumnCount(len(df.columns))
    table_widget.setHorizontalHeaderLabels([str(col) for col in df.columns])

    # Use enumerate to get integer row indices (fixes setItem TypeError)
    for row_idx, (_, row) in enumerate(df.iterrows()):
        for col_idx, value in enumerate(row):
            # Format numeric values for better readability
            if isinstance(value, float):
                if abs(value) < 0.001 or abs(value) > 10000:
                    formatted_value = (
                        f"{value:.3e}"  # Scientific notation for very small/large
                    )
                else:
                    formatted_value = (
                        f"{value:.4f}"  # 4 decimal places for normal floats
                    )
            else:
                formatted_value = str(value)
            item = QTableWidgetItem(formatted_value)
            table_widget.setItem(row_idx, col_idx, item)

    table_widget.resizeColumnsToContents()
    return table_widget


def create_summary_tab(summary_text: str) -> QWidget:
    """
    Create summary/diagnostics text tab.

    Args:
        summary_text: Summary text content

    Returns:
        Text display widget
    """
    text_edit = QTextEdit()
    text_edit.setReadOnly(True)
    text_edit.setPlainText(summary_text)
    text_edit.setStyleSheet(
        """
        QTextEdit {
            border: none;
            background-color: white;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            padding: 12px;
        }
    """
    )
    return text_edit
