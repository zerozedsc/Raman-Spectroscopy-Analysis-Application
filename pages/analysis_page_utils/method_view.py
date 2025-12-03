"""
Method View Components

This module handles the method-specific view with input forms and results display.
Includes dynamic parameter widget generation and results visualization.
"""

from typing import Dict, Any, Callable, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QTabWidget, QGroupBox, QComboBox,
    QSplitter, QTextEdit, QTableWidget, QTableWidgetItem, QListWidget,
    QAbstractItemView, QStackedWidget, QButtonGroup, QRadioButton, QCheckBox, QMessageBox
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont

from components.widgets import load_icon, GroupTreeManager, DynamicParameterWidget

from .registry import ANALYSIS_METHODS
from .group_assignment_table import GroupAssignmentTable

# Import PROJECT_MANAGER for group persistence
from utils import PROJECT_MANAGER

# Import visualization helpers
from .methods.exploratory import create_spectrum_preview_figure
import matplotlib.pyplot as plt
import numpy as np


def _v1_create_method_view(
    category: str,
    method_key: str,
    dataset_names: list,
    localize_func: Callable,
    on_run_analysis: Callable,
    on_back: Callable
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
    method_name_label.setStyleSheet("""
        font-size: 20px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 8px;
    """)
    left_layout.addWidget(method_name_label)
    
    # Use localized description from locales, not hardcoded from registry
    method_desc_text = localize_func(f"ANALYSIS_PAGE.METHOD_DESC.{method_key}")
    method_desc_label = QLabel(method_desc_text)
    method_desc_label.setWordWrap(True)
    method_desc_label.setStyleSheet("""
        font-size: 13px;
        color: #6c757d;
        line-height: 1.5;
        margin-bottom: 16px;
    """)
    left_layout.addWidget(method_desc_label)
    
    # Dataset selection - conditional widget based on method requirements
    dataset_selection_mode = method_info.get("dataset_selection_mode", "single")
    min_datasets = method_info.get("min_datasets", 1)
    
    dataset_card = QFrame()
    dataset_card.setObjectName("datasetCard")
    dataset_card.setStyleSheet("""
        QFrame#datasetCard {
            background-color: #ffffff;
            border: 1px solid #dfe3ea;
            border-radius: 16px;
            padding: 0px;
        }
    """)
    dataset_card_layout = QVBoxLayout(dataset_card)
    dataset_card_layout.setContentsMargins(24, 24, 24, 24)
    dataset_card_layout.setSpacing(18)

    # Card header with icon + badges
    dataset_header = QHBoxLayout()
    dataset_header.setSpacing(12)

    dataset_title = QLabel("📂 " + localize_func("ANALYSIS_PAGE.dataset_selection"))
    dataset_title.setStyleSheet("""
        font-size: 18px;
        font-weight: 600;
        color: #1f2a37;
    """)
    dataset_header.addWidget(dataset_title)

    dataset_header.addStretch()

    selection_badge = QLabel(
        localize_func("ANALYSIS_PAGE.dataset_mode_multi")
        if dataset_selection_mode == "multi"
        else localize_func("ANALYSIS_PAGE.dataset_mode_single")
    )
    selection_badge.setStyleSheet("""
        background-color: #eef2ff;
        color: #4338ca;
        border-radius: 999px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 600;
    """)
    dataset_header.addWidget(selection_badge)

    min_badge_text = localize_func("ANALYSIS_PAGE.min_datasets").format(count=min_datasets)
    min_badge = QLabel(min_badge_text)
    min_badge.setStyleSheet("""
        background-color: #ecfdf5;
        color: #047857;
        border-radius: 999px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 600;
    """)
    dataset_header.addWidget(min_badge)

    dataset_card_layout.addLayout(dataset_header)

    dataset_subtitle = QLabel(localize_func("ANALYSIS_PAGE.dataset_selection_subtitle"))
    dataset_subtitle.setWordWrap(True)
    dataset_subtitle.setStyleSheet("""
        font-size: 13px;
        color: #4b5563;
        line-height: 1.5;
    """)
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
        toggle_frame.setStyleSheet("""
            QFrame {
                background-color: #e9ecef;
                border: 1px solid #dee2e6;
                border-radius: 25px;
                padding: 3px;
                max-width: 500px;
            }
        """)
        toggle_layout = QHBoxLayout(toggle_frame)
        toggle_layout.setContentsMargins(3, 3, 3, 3)
        toggle_layout.setSpacing(3)
        
        # Comparison mode button - pill shaped
        comparison_radio = QRadioButton("📊 Unsupervised")
        comparison_radio.setObjectName("comparison_radio")
        comparison_radio.setChecked(True)
        comparison_radio.setCursor(Qt.PointingHandCursor)
        comparison_radio.setStyleSheet("""
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
        """)
        toggle_layout.addWidget(comparison_radio)
        
        # Classification mode button - pill shaped
        classification_radio = QRadioButton("🔬 Grouped Classification")
        classification_radio.setObjectName("classification_radio")
        classification_radio.setCursor(Qt.PointingHandCursor)
        classification_radio.setStyleSheet("""
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
        """)
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
        hint_label = QLabel("💡 <b>Unsupervised Mode:</b> Select multiple datasets for combined analysis. Click checkboxes or use 'Select All' below.")
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("""
            font-size: 12px;
            color: #495057;
            padding: 12px;
            background-color: #e7f3ff;
            border-left: 4px solid #0078d4;
            border-radius: 4px;
            margin-top: 8px;
        """)
        simple_layout.addWidget(hint_label)
    
    # Create appropriate widget based on selection mode
    dataset_widget = None
    select_all_checkbox = None
    
    if dataset_selection_mode == "single":
        # Single dropdown for single-dataset methods
        dataset_combo = QComboBox()
        dataset_combo.setObjectName("datasetComboBox")
        dataset_combo.setMinimumHeight(36)
        dataset_combo.addItems(dataset_names)
        dataset_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 6px 12px;
                background-color: white;
                font-size: 13px;
            }
            QComboBox:hover {
                border-color: #0078d4;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox::down-arrow {
                image: url(none);
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #6c757d;
                width: 0;
                height: 0;
            }
        """)
        dataset_widget = dataset_combo
        simple_layout.addWidget(dataset_combo)
    else:
        # Create toolbar with Select All checkbox
        from PySide6.QtWidgets import QCheckBox
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(8, 8, 8, 4)
        
        select_all_checkbox = QCheckBox("Select All")
        select_all_checkbox.setObjectName("selectAllCheckbox")
        select_all_checkbox.setStyleSheet("""
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
        """)
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
        dataset_list.setStyleSheet("""
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
        """)
        
        # Connect Select All functionality
        def toggle_select_all(checked):
            if checked:
                dataset_list.selectAll()
            else:
                dataset_list.clearSelection()
        
        select_all_checkbox.toggled.connect(toggle_select_all)
        
        # Update Select All state when list selection changes
        def update_select_all_state():
            total = dataset_list.count()
            selected = len(dataset_list.selectedItems())
            select_all_checkbox.blockSignals(True)
            select_all_checkbox.setChecked(selected == total and total > 0)
            select_all_checkbox.blockSignals(False)
        
        dataset_list.itemSelectionChanged.connect(update_select_all_state)
        
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
            print(f"[DEBUG] Loading {len(saved_groups)} saved groups from project")
            group_widget.set_groups(saved_groups)
        else:
            print("[DEBUG] No saved groups found in project")
        
        # Connect groups_changed signal to save groups to ProjectManager
        def on_groups_changed(groups: Dict[str, list]):
            print(f"[DEBUG] Groups changed, saving {len(groups)} groups to ProjectManager")
            PROJECT_MANAGER.set_analysis_groups(groups)
        
        group_widget.groups_changed.connect(on_groups_changed)
        
        # Connect mode toggle to switch between modes
        def toggle_mode(button):
            print("[DEBUG] toggle_mode called")
            print(f"[DEBUG] Button clicked: {button}")
            print(f"[DEBUG] Button objectName: {button.objectName()}")
            print(f"[DEBUG] comparison_radio: {comparison_radio}")
            print(f"[DEBUG] classification_radio: {classification_radio}")
            print(f"[DEBUG] Current stack index BEFORE: {dataset_stack.currentIndex()}")
            
            # Check which button was clicked and switch pages
            if button == comparison_radio:
                print("[DEBUG] Switching to Comparison Mode (page 0)")
                dataset_stack.setCurrentIndex(0)  # Show simple selection
                print(f"[DEBUG] Stack index AFTER: {dataset_stack.currentIndex()}")
            elif button == classification_radio:
                print("[DEBUG] Switching to Classification Mode (page 1)")
                dataset_stack.setCurrentIndex(1)  # Show group assignment table
                print(f"[DEBUG] Stack index AFTER: {dataset_stack.currentIndex()}")
            else:
                print("[DEBUG] WARNING: Button not recognized!")
            
            print(f"[DEBUG] Current visible widget: {dataset_stack.currentWidget()}")
        
        # CRITICAL FIX: Connect individual button toggled signals instead of buttonClicked
        # buttonClicked sometimes fails silently, toggled is more reliable
        comparison_radio.toggled.connect(lambda checked: toggle_mode(comparison_radio) if checked else None)
        classification_radio.toggled.connect(lambda checked: toggle_mode(classification_radio) if checked else None)
        
        # Also try buttonClicked as backup
        mode_toggle.buttonClicked.connect(toggle_mode)
        
        print("[DEBUG] Mode toggle signals connected (toggled + buttonClicked)")
        print(f"[DEBUG] comparison_radio.toggled: {comparison_radio.toggled}")
        print(f"[DEBUG] classification_radio.toggled: {classification_radio.toggled}")
        print(f"[DEBUG] mode_toggle.buttonClicked: {mode_toggle.buttonClicked}")
        print(f"[DEBUG] Toggle function object: {toggle_mode}")
        print(f"[DEBUG] Initial stack index: {dataset_stack.currentIndex()}")
        print(f"[DEBUG] Initial visible widget: {dataset_stack.currentWidget()}")
        print(f"[DEBUG] comparison_radio checked: {comparison_radio.isChecked()}")
        print(f"[DEBUG] classification_radio checked: {classification_radio.isChecked()}")
        
        # Set initial page to Comparison Mode
        dataset_stack.setCurrentIndex(0)
        comparison_radio.setChecked(True)
        print("[DEBUG] Set initial mode to Comparison (index 0)")
        print(f"[DEBUG] After init - comparison_radio checked: {comparison_radio.isChecked()}")
    
    dataset_layout.addWidget(dataset_stack)
    
    left_layout.addWidget(dataset_card)
    
    # Parameters section
    params_group = QGroupBox(localize_func("ANALYSIS_PAGE.parameters"))
    params_group.setStyleSheet("""
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
    """)
    params_layout = QVBoxLayout(params_group)
    
    # Create dynamic parameter widget
    # Convert registry format to DynamicParameterWidget format
    # Registry uses: "spinbox"/"double_spinbox"/"combo"/"checkbox"
    # DynamicParameterWidget expects: "int"/"float"/"choice"/"bool"
    
    type_mapping = {
        "spinbox": "int",
        "double_spinbox": "float",
        "combo": "choice",
        "checkbox": "bool"
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
            "description": param_config.get("label", param_name)
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
        method_info={
            "param_info": param_info,
            "default_params": default_params
        },
        saved_params={},
        data_range=None,
        parent=params_group
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
    run_btn.setStyleSheet("""
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
    """)
    
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
    
    run_btn.clicked.connect(lambda: on_run_analysis(
        category, method_key, _get_selected_datasets(), param_widget
    ))
    button_layout.addWidget(run_btn)
    
    left_layout.addLayout(button_layout)
    
    # === RIGHT PANEL: Results Display ===
    right_panel = create_results_panel(localize_func)
    
    # Add panels to splitter
    splitter.addWidget(left_panel)
    splitter.addWidget(right_panel)
    splitter.setSizes([400, 600])  # Initial sizes
    
    main_layout.addWidget(splitter)
    
    # Store references for external access
    method_widget.dataset_widget = dataset_widget  # Store the actual widget (QComboBox or QListWidget)
    method_widget.group_widget = group_widget  # Store group widget (GroupAssignmentTable)
    method_widget.comparison_radio = comparison_radio  # Store comparison mode button
    method_widget.classification_radio = classification_radio  # Store classification mode button
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
    results_panel.setStyleSheet("""
        QWidget#resultsPanel {
            background-color: #ffffff;
            border-left: 1px solid #e0e0e0;
        }
    """)
    
    layout = QVBoxLayout(results_panel)
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)
    
    # Header with export buttons
    header_layout = QHBoxLayout()
    
    results_title = QLabel("📊 " + localize_func("ANALYSIS_PAGE.results_title"))
    results_title.setStyleSheet("""
        font-size: 18px;
        font-weight: 600;
        color: #2c3e50;
    """)
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
    tab_widget.setStyleSheet("""
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
    """)
    
    # Placeholder tabs (will be populated with actual results)
    placeholder_label = QLabel(localize_func("ANALYSIS_PAGE.no_results_yet"))
    placeholder_label.setAlignment(Qt.AlignCenter)
    placeholder_label.setStyleSheet("""
        font-size: 14px;
        color: #6c757d;
        padding: 40px;
    """)
    tab_widget.addTab(placeholder_label, localize_func("ANALYSIS_PAGE.results_tab"))
    
    layout.addWidget(tab_widget)
    
    # Store references for external access
    results_panel.tab_widget = tab_widget
    results_panel.export_data_btn = export_data_btn
    results_panel.results_title = results_title
    
    return results_panel

class MethodParametersWidget(QWidget):
    """
    A styled 'Card' widget that holds the dynamic parameter inputs.
    Replaces the generic QGroupBox with a clean, SaaS-style look.
    """
    def __init__(self, method_info, parent=None):
        super().__init__(parent)
        self.method_info = method_info
        self.dynamic_widget = None 
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(8)

        # 1. Header (Clean Text)
        header = QLabel("⚙️ Configuration")  # You can localize this
        header.setStyleSheet("""
            font-size: 13px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 4px;
        """)
        layout.addWidget(header)

        # 2. Card Container (White box with subtle border)
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #dfe3ea;
                border-radius: 8px;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        
        # --- Data Conversion for DynamicParameterWidget ---
        # We map the registry types to the types your widget expects
        type_mapping = {
            "spinbox": "int", 
            "double_spinbox": "float", 
            "combo": "choice", 
            "checkbox": "bool"
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
                "description": p_config.get("label", p_name)
            }
            
            # Transfer constraints
            if "range" in p_config: param_info[p_name]["range"] = p_config["range"]
            if "step" in p_config: param_info[p_name]["step"] = p_config["step"]
            if "options" in p_config: param_info[p_name]["choices"] = p_config["options"]
            if "default" in p_config: default_params[p_name] = p_config["default"]

        # Instantiate the Dynamic Logic
        # Note: We assume DynamicParameterWidget is available in your imports
        self.dynamic_widget = DynamicParameterWidget(
            method_info={"param_info": param_info, "default_params": default_params},
            saved_params={}, 
            data_range=None, 
            parent=card
        )
        
        card_layout.addWidget(self.dynamic_widget)
        layout.addWidget(card)

    def get_params(self):
        """Bridge to get data from the inner widget"""
        if self.dynamic_widget:
            return self.dynamic_widget.get_params()
        return {}


class DatasetSelectionWidget(QWidget):
    def __init__(self, dataset_names, mode="single", localize_func=None, parent=None):
        super().__init__(parent)
        self.dataset_names = dataset_names
        self.mode = mode
        self.localize = localize_func
        
        # Widgets
        self.simple_input = None
        self.group_manager = None
        self.radio_group = None
        self.select_all_cb = None # Reference for logic
        
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12) # Consistent spacing

        # --- 1. HEADER ROW (Label + Toggle) ---
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        label_text = self.localize("ANALYSIS_PAGE.dataset_selection") if self.localize else "Select Datasets"
        label = QLabel("📂 " + label_text)
        # Using color #2c3e50 from stylesheets.py
        label.setStyleSheet("font-size: 14px; font-weight: 600; color: #2c3e50;")
        header_layout.addWidget(label)
        
        header_layout.addStretch()

        # Toggle (Only for multi mode)
        if self.mode == "multi":
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
            self.simple_input.setStyleSheet("""
                QComboBox {
                    padding: 8px;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    background-color: white;
                    font-size: 13px;
                    color: #2c3e50;
                }
                QComboBox:hover { border-color: #0078d4; }
                QComboBox::drop-down { border: none; width: 24px; }
                QComboBox QAbstractItemView {
                    border: 1px solid #ced4da;
                    selection-background-color: #e3f2fd;
                    selection-color: #1976d2;
                }
            """)
            layout.addWidget(self.simple_input)
            
        else:
            # Multi Selection (Stack)
            
            # PAGE 0: Optimized Simple List (List + Toolbar)
            page_simple = QWidget()
            layout_s = QVBoxLayout(page_simple)
            layout_s.setContentsMargins(0,0,0,0)
            layout_s.setSpacing(0) # Toolbar sits flush with list
            
            # A. Toolbar (Select All)
            toolbar = QFrame()
            toolbar.setStyleSheet("""
                QFrame {
                    background-color: #f8f9fa;
                    border: 1px solid #ced4da;
                    border-bottom: none;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
            """)
            tb_layout = QHBoxLayout(toolbar)
            tb_layout.setContentsMargins(12, 8, 12, 8)
            
            self.select_all_cb = QCheckBox("Select All")
            self.select_all_cb.setCursor(Qt.PointingHandCursor)
            self.select_all_cb.setStyleSheet("""
                QCheckBox { font-size: 13px; font-weight: 500; color: #495057; }
                QCheckBox::indicator { width: 16px; height: 16px; }
            """)
            tb_layout.addWidget(self.select_all_cb)
            tb_layout.addStretch()
            
            layout_s.addWidget(toolbar)
            
            # B. The Rich List Widget
            self.simple_input = QListWidget()
            self.simple_input.setSelectionMode(QAbstractItemView.MultiSelection)
            self.simple_input.addItems(self.dataset_names)
            self.simple_input.setMinimumHeight(200)
            
            # Applying 'dataset_list' style from stylesheets.py PREPROCESS_PAGE_STYLES
            self.simple_input.setStyleSheet("""
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
            """)
            
            # Logic: Connect Select All
            self.select_all_cb.toggled.connect(self._toggle_select_all)
            self.simple_input.itemSelectionChanged.connect(self._update_checkbox_state)
            
            layout_s.addWidget(self.simple_input)
            
            # PAGE 1: Group Tree
            self.group_manager = GroupTreeManager(self.dataset_names, self.localize)
            
            self.stack.addWidget(page_simple)
            self.stack.addWidget(self.group_manager)
            layout.addWidget(self.stack)

    def _create_toggle(self, layout):
        container = QFrame()
        # Matches 'input_field' border color #ced4da
        container.setStyleSheet("background-color: white; border-radius: 16px; border: 1px solid #ced4da;")
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
        r.setStyleSheet("""
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
        """)
        return r

    def _toggle_select_all(self, checked):
        """Selects or Deselects all items in the list."""
        if checked:
            self.simple_input.selectAll()
        else:
            self.simple_input.clearSelection()

    def _update_checkbox_state(self):
        """Updates checkbox if user manually selects/deselects items."""
        if not self.simple_input: return
        count = self.simple_input.count()
        selected = len(self.simple_input.selectedItems())
        
        self.select_all_cb.blockSignals(True)
        if selected == count and count > 0:
            self.select_all_cb.setCheckState(Qt.Checked)
        elif selected > 0:
            self.select_all_cb.setCheckState(Qt.PartiallyChecked)
        else:
            self.select_all_cb.setCheckState(Qt.Unchecked)
        self.select_all_cb.blockSignals(False)

    def get_selection(self):
        if self.mode == "single":
            return self.simple_input.currentText()
        
        if self.radio_group.checkedId() == 1:
            # Grouped Mode
            return self.group_manager.get_groups()
        else:
            # Simple Mode
            items = self.simple_input.selectedItems()
            return [i.text() for i in items]


class ResultsPanel(QWidget):
    """
    A styled container for Analysis Results.
    Manages the header, export actions, and the tabbed result views.
    """
    def __init__(self, localize_func, parent=None):
        super().__init__(parent)
        self.localize = localize_func
        self._setup_ui()

    def _setup_ui(self):
        self.setObjectName("resultsPanel")
        
        # 1. Main Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # Clean edges
        layout.setSpacing(0)

        # 2. Header Section (Title + Export)
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-bottom: 1px solid #e0e0e0;
            }
        """)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(24, 16, 24, 16)
        header_layout.setSpacing(16)

        self.title_label = QLabel("📊 " + self.localize("ANALYSIS_PAGE.results_title"))
        self.title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: 700;
            color: #2c3e50;
        """)
        
        self.export_btn = QPushButton(self.localize("ANALYSIS_PAGE.export_csv"))
        self.export_btn.setCursor(Qt.PointingHandCursor)
        self.export_btn.setVisible(False) # Hidden by default
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: #333;
                border: 1px solid #d0d0d0;
                padding: 6px 16px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #f8f9fa;
                border-color: #b0b0b0;
                color: #000;
            }
        """)

        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.export_btn)

        layout.addWidget(header_frame)

        # 3. The Tab Widget (Modern Styling)
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True) # Removes the heavy outer frame
        
        # PROFESSIONAL TAB STYLING
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: #ffffff;
                padding-top: 10px;
            }
            
            QTabWidget::tab-bar {
                alignment: left;
            }
            
            QTabBar::tab {
                background: transparent;
                color: #6c757d;
                font-size: 13px;
                font-weight: 600;
                padding: 12px 20px;
                border-bottom: 2px solid transparent;
                margin-left: 8px;
            }
            
            QTabBar::tab:hover {
                color: #0078d4;
                background-color: #f8f9fa;
                border-radius: 4px 4px 0 0;
            }
            
            QTabBar::tab:selected {
                color: #0078d4;
                border-bottom: 2px solid #0078d4; /* The modern underline indicator */
            }
        """)

        layout.addWidget(self.tab_widget)
        
        # 4. Initial Empty State
        self.show_placeholder()

    @property
    def export_data_btn(self):
        return self.export_btn
    
    def show_placeholder(self):
        """Displays the 'No Results' state."""
        self.tab_widget.clear()
        
        placeholder = QWidget()
        p_layout = QVBoxLayout(placeholder)
        p_layout.setAlignment(Qt.AlignCenter)
        
        icon_label = QLabel("📉")
        icon_label.setStyleSheet("font-size: 48px; margin-bottom: 10px;")
        
        text_label = QLabel(self.localize("ANALYSIS_PAGE.no_results_yet"))
        text_label.setStyleSheet("font-size: 16px; color: #adb5bd; font-weight: 500;")
        
        p_layout.addWidget(icon_label, 0, Qt.AlignCenter)
        p_layout.addWidget(text_label, 0, Qt.AlignCenter)
        
        self.tab_widget.addTab(placeholder, "Info")
        self.export_btn.setVisible(False)

    def add_result_tab(self, widget, title, icon=None):
        """Helper to add tabs easily."""
        index = self.tab_widget.addTab(widget, title)
        if icon:
            self.tab_widget.setTabIcon(index, icon)
        self.tab_widget.setCurrentIndex(index)
        self.export_btn.setVisible(True)

    def clear_results(self):
        """Clears all tabs and shows placeholder."""
        self.tab_widget.clear()
        self.show_placeholder()

class MethodView(QWidget):
    """
    The main controller for a specific Analysis Method View.
    Orchestrates the Input Panel (Left) and Results Panel (Right).
    """
    def __init__(self, category, method_key, dataset_names, localize_func, on_run, on_back, parent=None):
        super().__init__(parent)
        
        # Store Context
        self.category = category
        self.method_key = method_key
        self.dataset_names = dataset_names
        self.localize = localize_func
        self.on_run_callback = on_run
        self.on_back_callback = on_back
        
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
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        
        # 3. Build Panels
        left_panel = self._build_left_panel()
        self.results_panel = self._build_right_panel()
        
        # 4. Add to Splitter
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(self.results_panel)
        self.splitter.setSizes([380, 620]) # Default ratio
        
        main_layout.addWidget(self.splitter)

    def _build_left_panel(self):
        """Constructs the Input Form (Left Side)."""
        panel = QWidget()
        panel.setStyleSheet("background-color: #fcfcfc; border-right: 1px solid #dfe3ea;")
        
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
        
        content_layout.addStretch() # Push contents up
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
        
        self.dataset_widget = DatasetSelectionWidget(
            self.dataset_names, 
            mode, 
            self.localize
        )
        parent_layout.addWidget(self.dataset_widget)

    def _build_parameter_section(self, parent_layout):
        """Instantiates the MethodParametersWidget."""
        self.params_widget = MethodParametersWidget(
            self.method_info
        )
        parent_layout.addWidget(self.params_widget)

    def _build_action_buttons(self, parent_layout):
        """Creates Back and Run buttons."""
        row = QHBoxLayout()
        row.setSpacing(12)
        
        self.back_btn = QPushButton(self.localize("ANALYSIS_PAGE.back_button"))
        self.back_btn.setMinimumHeight(40)
        self.back_btn.setCursor(Qt.PointingHandCursor)
        self.back_btn.setStyleSheet("""
            QPushButton { 
                border: 1px solid #dfe3ea; background: white; color: #2c3e50; 
                border-radius: 6px; font-weight: 600; font-size: 13px;
            }
            QPushButton:hover { background: #f8f9fa; border-color: #adb5bd; }
        """)
        self.back_btn.clicked.connect(self.on_back_callback)
        
        self.run_btn = QPushButton(self.localize("ANALYSIS_PAGE.run_analysis"))
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.setStyleSheet("""
            QPushButton { 
                background-color: #0078d4; color: white; border: none; 
                border-radius: 6px; font-weight: 600; font-size: 14px; 
            }
            QPushButton:hover { background-color: #006abc; }
            QPushButton:pressed { background-color: #005a9e; }
        """)
        self.run_btn.clicked.connect(self._handle_run_click)
        
        row.addWidget(self.back_btn)
        row.addWidget(self.run_btn)
        parent_layout.addLayout(row)

    def _build_right_panel(self):
        """Creates the Results Panel."""
        # Uses your existing utility function
        return ResultsPanel(localize_func=self.localize, parent=self)

    def _handle_run_click(self):
        """Collects data and triggers the run callback."""
        # 1. Get Datasets
        selected_data = self.dataset_widget.get_selection()
        
        if not selected_data:
            QMessageBox.warning(self, "Input Error", "Please select at least one dataset.")
            return
            
        # 2. Get Parameters (Pass the widget itself, logic handled downstream)
        # The runner expects the widget to extract params, or we pass values
        # Currently your logic passes the widget, so we keep that pattern:
        self.on_run_callback(
            self.category, 
            self.method_key, 
            selected_data, 
            self.params_widget.dynamic_widget # Pass the inner dynamic widget
        )

    def _build_right_panel(self):
        """Creates the Results Panel using the new Class."""
        # Instantiate the class
        self.results_panel = ResultsPanel(self.localize)
        
        # If you need to connect the export button signal to a handler in MethodView
        self.results_panel.export_btn.clicked.connect(self._handle_export_csv)
        
        return self.results_panel

    def _handle_export_csv(self):
        # Logic to handle export
        pass



def populate_results_tabs(
    results_panel: QWidget,
    result: Any,
    localize_func: Callable,
    matplotlib_widget_class: type
) -> None:
    """
    Populate results tabs with analysis output.
    
    Args:
        results_panel: Results panel widget from create_results_panel
        result: AnalysisResult object
        localize_func: Localization function
        matplotlib_widget_class: MatplotlibWidget class for plot rendering
    """
    tab_widget = results_panel.tab_widget
    
    # Clear existing tabs
    while tab_widget.count() > 0:
        tab_widget.removeTab(0)
    
    # Show CSV export button (plot export via matplotlib toolbar)
    results_panel.export_data_btn.setVisible(True)
    
    # === Tab 0: Spectrum Preview (New Feature) ===
    if result.dataset_data:
        try:
            spectrum_fig = create_spectrum_preview_figure(result.dataset_data)
            spectrum_tab = matplotlib_widget_class()
            spectrum_tab.update_plot(spectrum_fig)
            spectrum_tab.setMinimumHeight(400)
            tab_widget.addTab(spectrum_tab, "📈 Spectrum Preview")
        except Exception as e:
            print(f"[ERROR] Failed to create spectrum preview: {e}")
    
    # === Special handling for PCA Analysis: Create 5-tab visualization ===
    is_pca = hasattr(result, "raw_results") and "pca_model" in result.raw_results
    
    if is_pca:
        print("[DEBUG] PCA Analysis detected - creating 5-tab visualization")
        
        # Extract figures from raw_results
        scree_figure = result.raw_results.get("scree_figure")
        loadings_figure = result.raw_results.get("loadings_figure")
        biplot_figure = result.raw_results.get("biplot_figure")
        cumulative_variance_figure = result.raw_results.get("cumulative_variance_figure")
        distributions_figure = result.raw_results.get("distributions_figure")
        
        print(f"[DEBUG] PCA figures found:")
        print(f"[DEBUG]   scree_figure: {scree_figure is not None}")
        print(f"[DEBUG]   loadings_figure: {loadings_figure is not None}")
        print(f"[DEBUG]   biplot_figure: {biplot_figure is not None}")
        print(f"[DEBUG]   cumulative_variance_figure: {cumulative_variance_figure is not None}")
        print(f"[DEBUG]   distributions_figure: {distributions_figure is not None}")
        
        # Tab 1: Score Plot (PC1 vs PC2)
        if result.primary_figure:
            score_tab = matplotlib_widget_class()
            score_tab.update_plot(result.primary_figure)
            score_tab.setMinimumHeight(400)
            tab_widget.addTab(score_tab, "📈 Score Plot")
        
        # Tab 2: Scree Plot
        if scree_figure:
            scree_tab = matplotlib_widget_class()
            scree_tab.update_plot(scree_figure)
            scree_tab.setMinimumHeight(400)
            tab_widget.addTab(scree_tab, "📊 Scree Plot")
        
        # Tab 3: Loading Plot with Component Selection
        if loadings_figure and "pca_model" in result.raw_results:
            # Create container
            loading_container = QWidget()
            loading_layout = QVBoxLayout(loading_container)
            loading_layout.setContentsMargins(0, 5, 0, 0)
            
            # Controls
            controls_layout = QHBoxLayout()
            controls_layout.addStretch()
            controls_layout.addWidget(QLabel("Show Component:"))
            
            comp_combo = QComboBox()
            pca_model = result.raw_results["pca_model"]
            n_comps = pca_model.n_components_
            comp_combo.addItem("All Components")
            for i in range(n_comps):
                comp_combo.addItem(f"PC{i+1}")
            
            controls_layout.addWidget(comp_combo)
            loading_layout.addLayout(controls_layout)
            
            # Plot
            loading_tab = matplotlib_widget_class()
            loading_tab.update_plot(loadings_figure)
            loading_layout.addWidget(loading_tab)
            
            # Update logic
            def update_loadings(index):
                try:
                    # Get wavenumbers from dataset_data
                    if not result.dataset_data:
                        return
                    
                    # Assume all datasets have same wavenumbers
                    first_df = next(iter(result.dataset_data.values()))
                    wavenumbers = first_df.index.values
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if index == 0: # All components
                        # Recreate the original multi-subplot figure (simplified)
                        # Actually, better to just show all in one plot or subplots
                        # For simplicity, let's plot all in one axis if "All" is selected, 
                        # OR revert to the original figure if possible.
                        # But we can't easily revert to original figure object without storing it.
                        # So let's plot all lines on one axis for "All Components" in this view
                        # OR just use the original figure if index == 0?
                        # The original figure was passed as loadings_figure.
                        # We can re-use it!
                        loading_tab.update_plot(loadings_figure)
                        return
                    else:
                        # Specific component (index-1)
                        pc_idx = index - 1
                        component = pca_model.components_[pc_idx]
                        explained_var = pca_model.explained_variance_ratio_[pc_idx] * 100
                        
                        ax.plot(wavenumbers, component, color='#1f77b4', linewidth=1.5)
                        ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
                        ax.set_ylabel('Loading Value', fontsize=12)
                        ax.set_title(f'PC{pc_idx+1} Loadings ({explained_var:.2f}%)', fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        ax.invert_xaxis()
                        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
                        
                        # Annotate peaks
                        abs_loadings = np.abs(component)
                        top_indices = np.argsort(abs_loadings)[-5:]
                        for peak_idx in top_indices:
                            peak_wn = wavenumbers[peak_idx]
                            peak_val = component[peak_idx]
                            ax.plot(peak_wn, peak_val, 'o', color='red', markersize=5)
                            ax.annotate(f'{peak_wn:.0f}', xy=(peak_wn, peak_val), xytext=(0, 10),
                                       textcoords='offset points', ha='center', fontsize=8)
                        
                        fig.tight_layout()
                        loading_tab.update_plot(fig)
                        plt.close(fig)
                        
                except Exception as e:
                    print(f"[ERROR] Failed to update loadings plot: {e}")

            comp_combo.currentIndexChanged.connect(update_loadings)
            
            tab_widget.addTab(loading_container, "🔬 Loading Plot")
        elif loadings_figure:
            # Fallback if no model
            loading_tab = matplotlib_widget_class()
            loading_tab.update_plot(loadings_figure)
            loading_tab.setMinimumHeight(400)
            tab_widget.addTab(loading_tab, "🔬 Loading Plot")
        
        # Tab 4: Biplot
        if biplot_figure:
            biplot_tab = matplotlib_widget_class()
            biplot_tab.update_plot(biplot_figure)
            biplot_tab.setMinimumHeight(400)
            tab_widget.addTab(biplot_tab, "🎯 Biplot")
        
        # Tab 5: Cumulative Variance
        if cumulative_variance_figure:
            cumvar_tab = matplotlib_widget_class()
            cumvar_tab.update_plot(cumulative_variance_figure)
            cumvar_tab.setMinimumHeight(400)
            tab_widget.addTab(cumvar_tab, "📈 Cumulative Variance")
        
        # Bonus: Distributions (if available)
        if distributions_figure:
            dist_tab = matplotlib_widget_class()
            dist_tab.update_plot(distributions_figure)
            dist_tab.setMinimumHeight(400)
            tab_widget.addTab(dist_tab, "📊 Distributions")
    
    else:
        # === Standard analysis results (non-PCA) ===
        
        # === Main Plot Tab (Scores/Primary Figure) ===
        if result.primary_figure:
            plot_tab = matplotlib_widget_class()
            plot_tab.update_plot(result.primary_figure)
            plot_tab.setMinimumHeight(400)
            tab_widget.addTab(plot_tab, "📈 " + localize_func("ANALYSIS_PAGE.scores_tab") if hasattr(result, "loadings_figure") else localize_func("ANALYSIS_PAGE.plot_tab"))
        
        # === Loadings Tab (for dimensionality reduction) ===
        print(f"[DEBUG] Checking loadings_figure...")
        print(f"[DEBUG]   hasattr(result, 'loadings_figure'): {hasattr(result, 'loadings_figure')}")
        if hasattr(result, 'loadings_figure'):
            print(f"[DEBUG]   result.loadings_figure is not None: {result.loadings_figure is not None}")
            print(f"[DEBUG]   result.loadings_figure type: {type(result.loadings_figure)}")
        
        if hasattr(result, "loadings_figure") and result.loadings_figure:
            print(f"[DEBUG] Creating Loadings tab...")
            loadings_tab = matplotlib_widget_class()
            loadings_tab.update_plot(result.loadings_figure)
            loadings_tab.setMinimumHeight(400)
            tab_widget.addTab(loadings_tab, "🔬 " + localize_func("ANALYSIS_PAGE.loadings_tab"))
            print(f"[DEBUG] Loadings tab added successfully")
        else:
            print(f"[DEBUG] Loadings tab NOT created (figure missing or None)")
        
        # === Distributions Tab (for classification) ===
        if hasattr(result, "distributions_figure") and result.distributions_figure:
            dist_tab = matplotlib_widget_class()
            dist_tab.update_plot(result.distributions_figure)
            dist_tab.setMinimumHeight(400)
            tab_widget.addTab(dist_tab, "📊 " + localize_func("ANALYSIS_PAGE.distributions_tab"))
        
        # === Legacy Secondary Figure Tab (deprecated but kept for compatibility) ===
        if hasattr(result, "secondary_figure") and result.secondary_figure:
            secondary_tab = matplotlib_widget_class()
            secondary_tab.update_plot(result.secondary_figure)
            secondary_tab.setMinimumHeight(400)
            tab_widget.addTab(secondary_tab, "📊 " + localize_func("ANALYSIS_PAGE.secondary_plot_tab"))
    
    # === Data Table Tab ===
    if result.data_table is not None:
        table_tab = create_data_table_tab(result.data_table)
        tab_widget.addTab(table_tab, "📋 " + localize_func("ANALYSIS_PAGE.data_tab"))
    
    # === Summary Tab ===
    if result.detailed_summary:
        summary_tab = create_summary_tab(result.detailed_summary)
        tab_widget.addTab(summary_tab, "📋 " + localize_func("ANALYSIS_PAGE.summary_tab"))
    
    # === Diagnostics Tab (if available) ===
    if hasattr(result, "diagnostics") and result.diagnostics:
        diag_tab = create_summary_tab(result.diagnostics)
        tab_widget.addTab(diag_tab, "🔍 " + localize_func("ANALYSIS_PAGE.diagnostics_tab"))


def create_data_table_tab(data_table) -> QWidget:
    """
    Create data table tab from pandas DataFrame or dict.
    
    Args:
        data_table: DataFrame or dict containing tabular data
    
    Returns:
        Table widget
    """
    table_widget = QTableWidget()
    table_widget.setStyleSheet("""
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
    """)
    
    # Convert to DataFrame if dict
    import pandas as pd
    if isinstance(data_table, dict):
        df = pd.DataFrame(data_table)
    else:
        df = data_table
    
    # Populate table
    table_widget.setRowCount(len(df))
    table_widget.setColumnCount(len(df.columns))
    table_widget.setHorizontalHeaderLabels([str(col) for col in df.columns])
    
    for i, row in df.iterrows():
        for j, value in enumerate(row):
            item = QTableWidgetItem(str(value))
            table_widget.setItem(i, j, item)
    
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
    text_edit.setStyleSheet("""
        QTextEdit {
            border: none;
            background-color: white;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            padding: 12px;
        }
    """)
    return text_edit
