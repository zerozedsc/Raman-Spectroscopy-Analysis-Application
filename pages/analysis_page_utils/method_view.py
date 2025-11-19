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
    QAbstractItemView, QStackedWidget, QButtonGroup, QRadioButton
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont

from components.widgets import load_icon
from components.widgets.parameter_widgets import DynamicParameterWidget
from .registry import ANALYSIS_METHODS
from .group_assignment_table import GroupAssignmentTable


def create_method_view(
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
    
    dataset_group = QGroupBox(localize_func("ANALYSIS_PAGE.dataset_selection"))
    dataset_group.setStyleSheet("""
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
    dataset_layout = QVBoxLayout(dataset_group)
    
    # For multi-dataset methods, add segmented control (toggle buttons)
    mode_toggle = None
    comparison_radio = None
    classification_radio = None
    
    if dataset_selection_mode == "multi":
        # Segmented control frame
        toggle_frame = QFrame()
        toggle_frame.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                padding: 4px;
            }
        """)
        toggle_layout = QHBoxLayout(toggle_frame)
        toggle_layout.setContentsMargins(4, 4, 4, 4)
        toggle_layout.setSpacing(4)
        
        # Comparison mode button
        comparison_radio = QRadioButton("📊 Comparison (Simple)")
        comparison_radio.setObjectName("comparison_radio")
        comparison_radio.setChecked(True)
        print("[DEBUG] Created comparison_radio button")
        comparison_radio.setStyleSheet("""
            QRadioButton {
                background-color: white;
                border: 2px solid #0078d4;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 13px;
                color: #0078d4;
            }
            QRadioButton:hover {
                background-color: #e7f3ff;
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
        
        # Add direct click handler for debugging
        def on_comparison_clicked():
            print("[DEBUG] comparison_radio.clicked() signal fired!")
            print(f"[DEBUG] comparison_radio is checked: {comparison_radio.isChecked()}")
        
        comparison_radio.clicked.connect(on_comparison_clicked)
        print("[DEBUG] Connected comparison_radio.clicked signal")
        
        # Classification mode button
        classification_radio = QRadioButton("🔬 Classification (Groups)")
        classification_radio.setObjectName("classification_radio")
        print("[DEBUG] Created classification_radio button")
        classification_radio.setStyleSheet("""
            QRadioButton {
                background-color: white;
                border: 2px solid #28a745;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 13px;
                color: #28a745;
            }
            QRadioButton:hover {
                background-color: #e8f5e9;
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
        
        # Add direct click handler for debugging
        def on_classification_clicked():
            print("[DEBUG] classification_radio.clicked() signal fired!")
            print(f"[DEBUG] classification_radio is checked: {classification_radio.isChecked()}")
        
        classification_radio.clicked.connect(on_classification_clicked)
        print("[DEBUG] Connected classification_radio.clicked signal")
        
        # Button group to ensure mutual exclusion
        mode_toggle = QButtonGroup()
        print("[DEBUG] Creating QButtonGroup for mode toggle")
        mode_toggle.addButton(comparison_radio, 0)
        print(f"[DEBUG] Added comparison_radio to button group with ID 0")
        mode_toggle.addButton(classification_radio, 1)
        print(f"[DEBUG] Added classification_radio to button group with ID 1")
        print(f"[DEBUG] Button group contains {len(mode_toggle.buttons())} buttons")
        print(f"[DEBUG] comparison_radio object: {comparison_radio}")
        print(f"[DEBUG] classification_radio object: {classification_radio}")
        
        dataset_layout.addWidget(toggle_frame)
    
    # Create stacked widget for simple vs group mode
    dataset_stack = QStackedWidget()
    
    # === PAGE 0: Simple Selection (Comparison Mode) ===
    simple_widget = QWidget()
    simple_layout = QVBoxLayout(simple_widget)
    simple_layout.setContentsMargins(0, 0, 0, 0)
    
    # Add hint label for multi-dataset methods
    if dataset_selection_mode == "multi":
        hint_label = QLabel(localize_func("ANALYSIS_PAGE.comparison_mode_hint"))
        hint_label.setStyleSheet("""
            font-size: 11px;
            color: #6c757d;
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 3px;
            margin-top: 8px;
        """)
        hint_label.setWordWrap(True)
        simple_layout.addWidget(hint_label)
    
    # Create appropriate widget based on selection mode
    dataset_widget = None
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
        # Multi-select list for multi-dataset methods
        dataset_list = QListWidget()
        dataset_list.setObjectName("datasetListWidget")
        dataset_list.setSelectionMode(QAbstractItemView.MultiSelection)
        dataset_list.setMinimumHeight(150)
        dataset_list.setMaximumHeight(250)
        dataset_list.addItems(dataset_names)
        dataset_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
            }
            QListWidget:hover {
                border-color: #0078d4;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 3px;
                margin: 2px;
            }
            QListWidget::item:hover {
                background-color: #f0f0f0;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
        """)
        dataset_widget = dataset_list
        simple_layout.addWidget(dataset_list)
    
    dataset_stack.addWidget(simple_widget)
    
    # === PAGE 1: Group Assignment (Classification Mode) ===
    group_widget = None
    if dataset_selection_mode == "multi":
        group_widget = GroupAssignmentTable(dataset_names, localize_func)
        group_widget.setMinimumHeight(400)
        dataset_stack.addWidget(group_widget)
        
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
    
    left_layout.addWidget(dataset_group)
    
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


def create_results_panel(localize_func: Callable) -> QWidget:
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
    
    # === Main Plot Tab (Scores/Primary Figure) ===
    if result.primary_figure:
        plot_tab = matplotlib_widget_class()
        plot_tab.update_plot(result.primary_figure)
        plot_tab.setMinimumHeight(400)
        tab_widget.addTab(plot_tab, "📈 " + localize_func("ANALYSIS_PAGE.scores_tab") if hasattr(result, "loadings_figure") else localize_func("ANALYSIS_PAGE.plot_tab"))
    
    # === Loadings Tab (for PCA/dimensionality reduction) ===
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
    
    # === Distributions Tab (for PCA/classification) ===
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
