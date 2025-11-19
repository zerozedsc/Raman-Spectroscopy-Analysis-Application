"""
Method View Components

This module handles the method-specific view with input forms and results display.
Includes dynamic parameter widget generation and results visualization.
"""

from typing import Dict, Any, Callable, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QTabWidget, QGroupBox, QComboBox,
    QSplitter, QTextEdit, QTableWidget, QTableWidgetItem
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont

from components.widgets import load_icon
from components.widgets.parameter_widgets import DynamicParameterWidget
from .registry import ANALYSIS_METHODS


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
    
    method_desc_label = QLabel(method_info.get("description", ""))
    method_desc_label.setWordWrap(True)
    method_desc_label.setStyleSheet("""
        font-size: 13px;
        color: #6c757d;
        line-height: 1.5;
        margin-bottom: 16px;
    """)
    left_layout.addWidget(method_desc_label)
    
    # Dataset selection
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
    
    dataset_combo = QComboBox()
    dataset_combo.setObjectName("datasetComboBox")
    dataset_combo.setMinimumHeight(36)
    dataset_combo.addItems(dataset_names)  # dataset_names is already a list of strings
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
    dataset_layout.addWidget(dataset_combo)
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
    run_btn.clicked.connect(lambda: on_run_analysis(
        category, method_key, dataset_combo.currentText(), param_widget
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
    method_widget.dataset_combo = dataset_combo
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
    
    # Export buttons (hidden until results available)
    export_png_btn = QPushButton(localize_func("ANALYSIS_PAGE.export_png"))
    export_png_btn.setObjectName("exportButton")
    export_png_btn.setMinimumHeight(32)
    export_png_btn.setVisible(False)
    header_layout.addWidget(export_png_btn)
    
    export_svg_btn = QPushButton(localize_func("ANALYSIS_PAGE.export_svg"))
    export_svg_btn.setObjectName("exportButton")
    export_svg_btn.setMinimumHeight(32)
    export_svg_btn.setVisible(False)
    header_layout.addWidget(export_svg_btn)
    
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
    results_panel.export_png_btn = export_png_btn
    results_panel.export_svg_btn = export_svg_btn
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
    
    # Show export buttons
    results_panel.export_png_btn.setVisible(True)
    results_panel.export_svg_btn.setVisible(True)
    results_panel.export_data_btn.setVisible(True)
    
    # === Plot Tab ===
    if result.figure:
        plot_tab = matplotlib_widget_class(result.figure)
        plot_tab.setMinimumHeight(400)
        tab_widget.addTab(plot_tab, "📈 " + localize_func("ANALYSIS_PAGE.plot_tab"))
    
    # === Data Table Tab ===
    if result.data_table is not None:
        table_tab = create_data_table_tab(result.data_table)
        tab_widget.addTab(table_tab, "📋 " + localize_func("ANALYSIS_PAGE.data_tab"))
    
    # === Summary Tab ===
    if result.summary:
        summary_tab = create_summary_tab(result.summary)
        tab_widget.addTab(summary_tab, "📝 " + localize_func("ANALYSIS_PAGE.summary_tab"))
    
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
