"""
Analysis Page for Raman Spectroscopy Data (Enhanced Card-Based Design)

This module provides comprehensive analysis capabilities for Raman spectroscopy data
with an intuitive card-based interface for method selection and result visualization.

Features:
- Card-based startup view showing all available analysis methods
- Category-based organization (Exploratory, Statistical, Visualization)
- Method-specific input forms with dynamic parameters
- Real-time analysis results display
- Export capabilities (PNG, SVG, CSV, full reports)
- Analysis history tracking

Design Pattern:
1. Startup View → Method Cards → Method Input View → Results View
2. Top bar with "New Analysis" button to return to startup view
3. Sidebar for analysis history navigation

Author: AI Assistant
Date: 2024-12-18
Version: 2.0 (Card-Based Design)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QLabel, QPushButton, QComboBox, QListWidget, QListWidgetItem,
    QScrollArea, QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget,
    QTextEdit, QProgressBar, QFrame, QMessageBox, QGridLayout,
    QStackedWidget, QFileDialog
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QSize
from PySide6.QtGui import QIcon

from components.widgets import MatplotlibWidget, load_icon
from configs.configs import load_config, LocalizationManager, create_logs
from utils import RAMAN_DATA, PROJECT_MANAGER

# Import analysis utilities
from .analysis_page_utils import (
    ANALYSIS_METHODS,
    AnalysisResult,
    AnalysisThread,
    create_parameter_widgets
)


@dataclass
class AnalysisHistoryItem:
    """Data class for analysis history tracking."""
    method_key: str
    method_name: str
    category: str
    timestamp: str
    dataset_names: List[str]
    n_spectra: int
    result: Optional[AnalysisResult] = None


class AnalysisPage(QWidget):
    """
    Enhanced Analysis Page with Card-Based Interface.
    
    Views:
    1. Startup View - Card grid of all analysis methods
    2. Method View - Input form + results display
    3. History sidebar - Navigate previous analyses
    
    Features:
    - Intuitive card-based method selection
    - Dynamic parameter forms per method
    - Real-time analysis with progress tracking
    - Result caching and history
    - Export capabilities (PNG, SVG, CSV, reports)
    """
    
    showNotification = Signal(str, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("analysisPage")
        
        # View state management
        self.current_view = "startup"  # "startup", "method", "results"
        self.current_method_key: Optional[str] = None
        self.current_category: Optional[str] = None
        
        # Analysis state
        self.analysis_thread: Optional[QThread] = None
        self.current_analysis_result: Optional[AnalysisResult] = None
        self.analysis_cache: Dict[str, AnalysisResult] = {}
        self.analysis_history: List[AnalysisHistoryItem] = []
        self.param_widgets: Dict[str, QWidget] = {}
        
        # Dataset state
        self.selected_datasets: List[str] = []
        self.dataset_data: Dict[str, pd.DataFrame] = {}
        
        # Localization
        self.config = load_config()
        self.locale_manager = LocalizationManager(
            locale_dir='assets/locales',
            default_lang=self.config.get('language', 'en')
        )
        
        self._setup_ui()
        self._connect_signals()
        
        # Auto-load project data
        QTimer.singleShot(100, self.load_project_data)
    
    def _setup_ui(self):
        """Setup the main UI with stacked widget architecture for different views."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top bar with navigation
        top_bar = self._create_top_bar()
        main_layout.addWidget(top_bar)
        
        # Main content area with stacked widget
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setHandleWidth(2)
        
        # History sidebar (collapsible)
        self.history_sidebar = self._create_history_sidebar()
        content_splitter.addWidget(self.history_sidebar)
        
        # Stacked widget for different views
        self.view_stack = QStackedWidget()
        self.view_stack.setObjectName("viewStack")
        
        # View 1: Startup view with method cards
        self.startup_view = self._create_startup_view()
        self.view_stack.addWidget(self.startup_view)
        
        # View 2: Method input/results view
        self.method_view = self._create_method_view()
        self.view_stack.addWidget(self.method_view)
        
        content_splitter.addWidget(self.view_stack)
        
        # Set initial splitter sizes: sidebar 250px, main content expanding
        content_splitter.setSizes([250, 1000])
        content_splitter.setCollapsible(0, True)
        content_splitter.setCollapsible(1, False)
        
    
    def _create_top_bar(self) -> QWidget:
        """Create top navigation bar with New Analysis button."""
        top_bar = QWidget()
        top_bar.setObjectName("topBar")
        top_bar.setStyleSheet("""
            QWidget#topBar {
                background-color: #ffffff;
                border-bottom: 2px solid #e0e0e0;
                padding: 8px 16px;
            }
        """)
        
        layout = QHBoxLayout(top_bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)
        
        # Title
        title_label = QLabel("📊 " + self.LOCALIZE("ANALYSIS_PAGE.title"))
        title_label.setStyleSheet("font-weight: 600; font-size: 16px; color: #2c3e50;")
        layout.addWidget(title_label)
        
        layout.addStretch()
        
        # New Analysis button (plus icon)
        self.new_analysis_btn = QPushButton()
        self.new_analysis_btn.setObjectName("newAnalysisButton")
        plus_icon = load_icon("plus", QSize(20, 20), "white")
        self.new_analysis_btn.setIcon(plus_icon)
        self.new_analysis_btn.setIconSize(QSize(20, 20))
        self.new_analysis_btn.setFixedSize(40, 40)
        self.new_analysis_btn.setToolTip(self.LOCALIZE("ANALYSIS_PAGE.new_analysis_tooltip"))
        self.new_analysis_btn.setCursor(Qt.PointingHandCursor)
        self.new_analysis_btn.setStyleSheet("""
            QPushButton#newAnalysisButton {
                background-color: #0078d4;
                border: none;
                border-radius: 20px;
            }
            QPushButton#newAnalysisButton:hover {
                background-color: #006abc;
            }
            QPushButton#newAnalysisButton:pressed {
                background-color: #005a9e;
            }
        """)
        self.new_analysis_btn.clicked.connect(self._show_startup_view)
        layout.addWidget(self.new_analysis_btn)
        
        # Hide by default (only show in method view)
        self.new_analysis_btn.setVisible(False)
        
        return top_bar
    
    def _create_left_panel(self) -> QWidget:
        """Create the left control panel with all analysis controls."""
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_panel.setMaximumWidth(450)
        left_panel.setMinimumWidth(380)
        
        layout = QVBoxLayout(left_panel)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(12)
        
        # 1. Dataset Selection Group
        dataset_group = self._create_dataset_selection_group()
        layout.addWidget(dataset_group)
        
        # 2. Analysis Method Selector Group
        method_group = self._create_analysis_method_selector()
        layout.addWidget(method_group)
        
        # 3. Dynamic Parameters Group
        params_group = self._create_parameters_group()
        layout.addWidget(params_group)
        
        # 4. Quick Stats Summary
        stats_group = self._create_quick_stats_group()
        layout.addWidget(stats_group)
        
        # 5. Run Analysis Button
        run_layout = QHBoxLayout()
        self.run_btn = QPushButton("▶ " + self.LOCALIZE("ANALYSIS_PAGE.run_button"))
        self.run_btn.setObjectName("ctaButton")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self.run_analysis)
        run_layout.addWidget(self.run_btn)
        
        self.cancel_btn = QPushButton("⏹ " + self.LOCALIZE("ANALYSIS_PAGE.cancel_button"))
        self.cancel_btn.setObjectName("secondaryButton")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        run_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(run_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        return left_panel
    
    def _create_dataset_selection_group(self) -> QGroupBox:
        """Create dataset selection group with multi-select capability."""
        group = QGroupBox()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 4, 12, 12)
        layout.setSpacing(8)
        
        # Title bar with refresh button
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)
        
        title_label = QLabel(self.LOCALIZE("ANALYSIS_PAGE.dataset_selection_title"))
        title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
        title_layout.addWidget(title_label)
        
        # Hint button
        hint_btn = QPushButton("?")
        hint_btn.setObjectName("hintButton")
        hint_btn.setFixedSize(20, 20)
        hint_btn.setToolTip(self.LOCALIZE("ANALYSIS_PAGE.dataset_hint"))
        hint_btn.setCursor(Qt.PointingHandCursor)
        hint_btn.setStyleSheet("""
            QPushButton#hintButton {
                background-color: #e7f3ff;
                color: #0078d4;
                border: 1px solid #90caf9;
                border-radius: 10px;
                font-weight: bold;
                font-size: 11px;
                padding: 0px;
            }
            QPushButton#hintButton:hover {
                background-color: #0078d4;
                color: white;
                border-color: #0078d4;
            }
        """)
        title_layout.addWidget(hint_btn)
        
        title_layout.addStretch()
        
        # Refresh button
        refresh_btn = QPushButton()
        refresh_btn.setObjectName("titleBarButton")
        refresh_icon = load_icon("reload", QSize(14, 14), "#0078d4")
        refresh_btn.setIcon(refresh_icon)
        refresh_btn.setFixedSize(24, 24)
        refresh_btn.setToolTip(self.LOCALIZE("ANALYSIS_PAGE.refresh_datasets"))
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setStyleSheet("""
            QPushButton#titleBarButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 2px;
            }
            QPushButton#titleBarButton:hover {
                background-color: #e7f3ff;
                border-color: #90caf9;
            }
            QPushButton#titleBarButton:pressed {
                background-color: #bbdefb;
            }
        """)
        refresh_btn.clicked.connect(self.load_project_data)
        title_layout.addWidget(refresh_btn)
        
        layout.addWidget(title_widget)
        
        # Dataset filter tabs
        self.dataset_tabs = QTabWidget()
        self.dataset_tabs.setObjectName("datasetTabs")
        
        # All datasets tab
        self.dataset_list_all = QListWidget()
        self.dataset_list_all.setSelectionMode(QListWidget.ExtendedSelection)
        self.dataset_list_all.setMinimumHeight(120)
        self.dataset_list_all.setMaximumHeight(180)
        self.dataset_list_all.itemSelectionChanged.connect(self._on_dataset_selection_changed)
        self.dataset_tabs.addTab(self.dataset_list_all, self.LOCALIZE("ANALYSIS_PAGE.all_datasets"))
        
        # Preprocessed only tab
        self.dataset_list_preprocessed = QListWidget()
        self.dataset_list_preprocessed.setSelectionMode(QListWidget.ExtendedSelection)
        self.dataset_list_preprocessed.setMinimumHeight(120)
        self.dataset_list_preprocessed.setMaximumHeight(180)
        self.dataset_list_preprocessed.itemSelectionChanged.connect(self._on_dataset_selection_changed)
        self.dataset_tabs.addTab(self.dataset_list_preprocessed, self.LOCALIZE("ANALYSIS_PAGE.preprocessed_only"))
        
        layout.addWidget(self.dataset_tabs)
        
        # Selection info label
        self.selection_info_label = QLabel(self.LOCALIZE("ANALYSIS_PAGE.no_datasets_selected"))
        self.selection_info_label.setStyleSheet("color: #6c757d; font-size: 11px; padding: 4px 0;")
        layout.addWidget(self.selection_info_label)
        
        return group
    
    def _create_analysis_method_selector(self) -> QGroupBox:
        """Create analysis method selector with category organization."""
        group = QGroupBox()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 4, 12, 12)
        layout.setSpacing(8)
        
        # Title bar with hint button
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)
        
        title_label = QLabel(self.LOCALIZE("ANALYSIS_PAGE.method_selection_title"))
        title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
        title_layout.addWidget(title_label)
        
        # Hint button
        hint_btn = QPushButton("?")
        hint_btn.setObjectName("hintButton")
        hint_btn.setFixedSize(20, 20)
        hint_btn.setToolTip(self.LOCALIZE("ANALYSIS_PAGE.method_selection_hint"))
        hint_btn.setCursor(Qt.PointingHandCursor)
        hint_btn.setStyleSheet("""
            QPushButton#hintButton {
                background-color: #e7f3ff;
                color: #0078d4;
                border: 1px solid #90caf9;
                border-radius: 10px;
                font-weight: bold;
                font-size: 11px;
                padding: 0px;
            }
            QPushButton#hintButton:hover {
                background-color: #0078d4;
                color: white;
                border-color: #0078d4;
            }
        """)
        title_layout.addWidget(hint_btn)
        
        title_layout.addStretch()
        
        layout.addWidget(title_widget)
        
        # Category dropdown
        cat_label = QLabel("📊 " + self.LOCALIZE("ANALYSIS_PAGE.category"))
        cat_label.setStyleSheet("font-weight: 500; color: #495057; font-size: 11px;")
        layout.addWidget(cat_label)
        
        self.category_combo = QComboBox()
        # Block signals during initialization
        self.category_combo.blockSignals(True)
        self.category_combo.addItem(self.LOCALIZE("ANALYSIS_PAGE.exploratory"), "exploratory")
        self.category_combo.addItem(self.LOCALIZE("ANALYSIS_PAGE.statistical"), "statistical")
        self.category_combo.addItem(self.LOCALIZE("ANALYSIS_PAGE.visualization"), "visualization")
        self.category_combo.currentIndexChanged.connect(self._update_method_options)
        layout.addWidget(self.category_combo)
        
        # Method dropdown (dynamic)
        method_label = QLabel("🔬 " + self.LOCALIZE("ANALYSIS_PAGE.method"))
        method_label.setStyleSheet("font-weight: 500; color: #495057; font-size: 11px;")
        layout.addWidget(method_label)
        
        self.method_combo = QComboBox()
        # Block signals during initialization
        self.method_combo.blockSignals(True)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        layout.addWidget(self.method_combo)
        
        # Initialize method options (signals still blocked)
        self._update_method_options()
        
        return group
    
    def _create_parameters_group(self) -> QGroupBox:
        """Create dynamic parameter panel based on selected method."""
        group = QGroupBox()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 4, 12, 12)
        layout.setSpacing(8)
        
        # Title bar with hint and reset buttons
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)
        
        title_label = QLabel(self.LOCALIZE("ANALYSIS_PAGE.parameters_title"))
        title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
        title_layout.addWidget(title_label)
        
        # Hint button
        hint_btn = QPushButton("?")
        hint_btn.setObjectName("hintButton")
        hint_btn.setFixedSize(20, 20)
        hint_btn.setToolTip(self.LOCALIZE("ANALYSIS_PAGE.parameters_hint"))
        hint_btn.setCursor(Qt.PointingHandCursor)
        hint_btn.setStyleSheet("""
            QPushButton#hintButton {
                background-color: #e7f3ff;
                color: #0078d4;
                border: 1px solid #90caf9;
                border-radius: 10px;
                font-weight: bold;
                font-size: 11px;
                padding: 0px;
            }
            QPushButton#hintButton:hover {
                background-color: #0078d4;
                color: white;
                border-color: #0078d4;
            }
        """)
        title_layout.addWidget(hint_btn)
        
        title_layout.addStretch()
        
        # Reset button
        reset_btn = QPushButton(self.LOCALIZE("ANALYSIS_PAGE.reset_defaults"))
        reset_btn.setObjectName("secondaryButton")
        reset_btn.setFixedHeight(28)
        reset_btn.clicked.connect(self._reset_parameters)
        title_layout.addWidget(reset_btn)
        
        layout.addWidget(title_widget)
        
        # Scrollable container for parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(180)
        scroll_area.setMaximumHeight(300)
        
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        self.params_layout.setContentsMargins(8, 8, 8, 8)
        self.params_layout.setSpacing(12)
        scroll_area.setWidget(self.params_container)
        
        layout.addWidget(scroll_area)
        
        # No parameters message (initially shown)
        self.no_params_label = QLabel(self.LOCALIZE("ANALYSIS_PAGE.select_method_first"))
        self.no_params_label.setStyleSheet("color: #6c757d; font-style: italic; padding: 20px;")
        self.no_params_label.setAlignment(Qt.AlignCenter)
        self.params_layout.addWidget(self.no_params_label)
        
        return group
    
    def _create_quick_stats_group(self) -> QGroupBox:
        """Create quick stats summary display."""
        group = QGroupBox()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 4, 12, 12)
        layout.setSpacing(8)
        
        # Title bar with hint button
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)
        
        title_label = QLabel(self.LOCALIZE("ANALYSIS_PAGE.quick_stats_title"))
        title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
        title_layout.addWidget(title_label)
        
        # Hint button
        hint_btn = QPushButton("?")
        hint_btn.setObjectName("hintButton")
        hint_btn.setFixedSize(20, 20)
        hint_btn.setToolTip(self.LOCALIZE("ANALYSIS_PAGE.quick_stats_hint"))
        hint_btn.setCursor(Qt.PointingHandCursor)
        hint_btn.setStyleSheet("""
            QPushButton#hintButton {
                background-color: #e7f3ff;
                color: #0078d4;
                border: 1px solid #90caf9;
                border-radius: 10px;
                font-weight: bold;
                font-size: 11px;
                padding: 0px;
            }
            QPushButton#hintButton:hover {
                background-color: #0078d4;
                color: white;
                border-color: #0078d4;
            }
        """)
        title_layout.addWidget(hint_btn)
        
        title_layout.addStretch()
        
        layout.addWidget(title_widget)
        
        # Stats display (console-style)
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setMaximumHeight(120)
        self.stats_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-size: 11px;
                font-family: 'Consolas', 'Monaco', monospace;
                color: #2c3e50;
            }
        """)
        self.stats_display.setPlainText(self.LOCALIZE("ANALYSIS_PAGE.no_analysis_run"))
        layout.addWidget(self.stats_display)
        
        return group
    
    def _create_right_panel(self) -> QWidget:
        """Create the right visualization panel with tabbed interface."""
        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")
        
        layout = QVBoxLayout(right_panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Main visualization tabs
        self.viz_tabs = QTabWidget()
        self.viz_tabs.setObjectName("vizTabs")
        
        # Tab 1: Primary Visualization
        self.primary_viz_widget = self._create_primary_viz_tab()
        self.viz_tabs.addTab(self.primary_viz_widget, "📊 " + self.LOCALIZE("ANALYSIS_PAGE.primary_viz"))
        
        # Tab 2: Secondary Visualization
        self.secondary_viz_widget = self._create_secondary_viz_tab()
        self.viz_tabs.addTab(self.secondary_viz_widget, "📈 " + self.LOCALIZE("ANALYSIS_PAGE.secondary_viz"))
        
        # Tab 3: Data Table
        self.data_table_widget = self._create_data_table_tab()
        self.viz_tabs.addTab(self.data_table_widget, "📋 " + self.LOCALIZE("ANALYSIS_PAGE.data_table"))
        
        layout.addWidget(self.viz_tabs, 3)
        
        # Collapsible results summary panel
        results_group = self._create_results_summary_panel()
        layout.addWidget(results_group, 1)
        
        return right_panel
    
    def _create_primary_viz_tab(self) -> QWidget:
        """Create primary visualization tab with matplotlib widget."""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Control bar
        control_bar = QWidget()
        control_layout = QHBoxLayout(control_bar)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(8)
        
        # Preview checkbox
        self.preview_checkbox = QCheckBox(self.LOCALIZE("ANALYSIS_PAGE.enable_preview"))
        self.preview_checkbox.setChecked(True)
        control_layout.addWidget(self.preview_checkbox)
        
        control_layout.addStretch()
        
        # Export buttons
        export_png_btn = QPushButton(self.LOCALIZE("ANALYSIS_PAGE.export_png"))
        export_png_btn.setObjectName("secondaryButton")
        export_png_btn.clicked.connect(lambda: self._export_plot("png"))
        control_layout.addWidget(export_png_btn)
        
        export_svg_btn = QPushButton(self.LOCALIZE("ANALYSIS_PAGE.export_svg"))
        export_svg_btn.setObjectName("secondaryButton")
        export_svg_btn.clicked.connect(lambda: self._export_plot("svg"))
        control_layout.addWidget(export_svg_btn)
        
        layout.addWidget(control_bar)
        
        # Matplotlib widget
        self.primary_plot = MatplotlibWidget()
        self.primary_plot.setMinimumHeight(400)
        layout.addWidget(self.primary_plot)
        
        return tab_widget
    
    def _create_secondary_viz_tab(self) -> QWidget:
        """Create secondary visualization tab."""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Matplotlib widget
        self.secondary_plot = MatplotlibWidget()
        self.secondary_plot.setMinimumHeight(400)
        layout.addWidget(self.secondary_plot)
        
        return tab_widget
    
    def _create_data_table_tab(self) -> QWidget:
        """Create data table tab for numerical results."""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Control bar
        control_bar = QWidget()
        control_layout = QHBoxLayout(control_bar)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(8)
        
        control_layout.addStretch()
        
        # Export CSV button
        export_csv_btn = QPushButton(self.LOCALIZE("ANALYSIS_PAGE.export_csv"))
        export_csv_btn.setObjectName("secondaryButton")
        export_csv_btn.clicked.connect(self._export_data_table)
        control_layout.addWidget(export_csv_btn)
        
        layout.addWidget(control_bar)
        
        # Data display (using QTextEdit for now, can be replaced with QTableView)
        self.data_table_display = QTextEdit()
        self.data_table_display.setReadOnly(True)
        self.data_table_display.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-size: 11px;
                font-family: 'Consolas', 'Monaco', monospace;
                color: #2c3e50;
            }
        """)
        layout.addWidget(self.data_table_display)
        
        return tab_widget
    
    def _create_results_summary_panel(self) -> QGroupBox:
        """Create collapsible results summary panel."""
        group = QGroupBox(self.LOCALIZE("ANALYSIS_PAGE.results_summary"))
        layout = QVBoxLayout(group)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)
        
        # Results text
        self.results_summary_text = QTextEdit()
        self.results_summary_text.setReadOnly(True)
        self.results_summary_text.setMaximumHeight(150)
        self.results_summary_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-size: 11px;
                color: #2c3e50;
            }
        """)
        layout.addWidget(self.results_summary_text)
        
        # Export buttons
        export_layout = QHBoxLayout()
        export_layout.setSpacing(8)
        
        export_layout.addStretch()
        
        export_report_btn = QPushButton(self.LOCALIZE("ANALYSIS_PAGE.export_report"))
        export_report_btn.setObjectName("ctaButton")
        export_report_btn.clicked.connect(self._export_full_report)
        export_layout.addWidget(export_report_btn)
        
        save_results_btn = QPushButton(self.LOCALIZE("ANALYSIS_PAGE.save_results"))
        save_results_btn.setObjectName("secondaryButton")
        save_results_btn.clicked.connect(self._save_analysis_results)
        export_layout.addWidget(save_results_btn)
        
        layout.addLayout(export_layout)
        
        return group
    
    def _connect_signals(self):
        """Connect UI signals to handlers."""
        pass
    
    def LOCALIZE(self, key: str, **kwargs) -> str:
        """Convenience method for localization."""
        return self.locale_manager.get(key, **kwargs)
    
    # Data management methods
    def load_project_data(self):
        """Load project data into dataset lists."""
        create_logs("AnalysisPage", "data_loading", "Loading project data...", status='info')
        
        # Clear existing lists
        self.dataset_list_all.clear()
        self.dataset_list_preprocessed.clear()
        
        if not RAMAN_DATA:
            self.showNotification.emit(
                self.LOCALIZE("ANALYSIS_PAGE.no_data_loaded"),
                "warning"
            )
            return
        
        # Populate dataset lists
        for dataset_name, data in RAMAN_DATA.items():
            # Add to all datasets list
            item_all = QListWidgetItem(f"📊 {dataset_name}")
            item_all.setData(Qt.UserRole, dataset_name)
            self.dataset_list_all.addItem(item_all)
            
            # Add to preprocessed list if applicable
            if "preprocessed" in dataset_name.lower() or "_processed" in dataset_name.lower():
                item_prep = QListWidgetItem(f"🔬 {dataset_name}")
                item_prep.setData(Qt.UserRole, dataset_name)
                self.dataset_list_preprocessed.addItem(item_prep)
        
        create_logs("AnalysisPage", "data_loading", 
                   f"Loaded {len(RAMAN_DATA)} datasets", status='info')
    
    def clear_project_data(self):
        """Clear all project data when switching projects."""
        self.dataset_list_all.clear()
        self.dataset_list_preprocessed.clear()
        self.selected_datasets.clear()
        self.dataset_data.clear()
        self.analysis_cache.clear()
        self.current_analysis_result = None
        
        # Reset UI
        self.stats_display.setPlainText(self.LOCALIZE("ANALYSIS_PAGE.no_analysis_run"))
        self.results_summary_text.clear()
        self.data_table_display.clear()
        
        # Clear plots
        self.primary_plot.clear_plot()
        self.secondary_plot.clear_plot()
    
    def _on_dataset_selection_changed(self):
        """Handle dataset selection changes."""
        # Get current tab
        current_tab = self.dataset_tabs.currentIndex()
        
        if current_tab == 0:
            selected_items = self.dataset_list_all.selectedItems()
        else:
            selected_items = self.dataset_list_preprocessed.selectedItems()
        
        self.selected_datasets = [item.data(Qt.UserRole) for item in selected_items]
        
        # Update selection info
        n_selected = len(self.selected_datasets)
        if n_selected == 0:
            self.selection_info_label.setText(self.LOCALIZE("ANALYSIS_PAGE.no_datasets_selected"))
        else:
            total_spectra = sum(RAMAN_DATA[name].shape[1] for name in self.selected_datasets)
            self.selection_info_label.setText(
                self.LOCALIZE("ANALYSIS_PAGE.datasets_selected", 
                            count=n_selected, spectra=total_spectra)
            )
        
        # Update dataset data
        self.dataset_data.clear()
        for name in self.selected_datasets:
            if name in RAMAN_DATA:
                self.dataset_data[name] = RAMAN_DATA[name]
    
    def _update_method_options(self):
        """Update method dropdown based on selected category."""
        self.method_combo.clear()
        
        category = self.category_combo.currentData()
        if category and category in ANALYSIS_METHODS:
            methods = ANALYSIS_METHODS[category]
            for method_key, method_info in methods.items():
                self.method_combo.addItem(method_info["name"], method_key)
    
    def _on_method_changed(self):
        """Handle method selection change."""
        category = self.category_combo.currentData()
        method_key = self.method_combo.currentData()
        
        if category and method_key:
            self._populate_parameters(category, method_key)
    
    def _populate_parameters(self, category: str, method_key: str):
        """Dynamically generate parameter widgets based on method."""
        # Clear existing widgets
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.param_widgets.clear()
        
        # Get method info
        if category not in ANALYSIS_METHODS or method_key not in ANALYSIS_METHODS[category]:
            self.no_params_label = QLabel(self.LOCALIZE("ANALYSIS_PAGE.method_not_found"))
            self.params_layout.addWidget(self.no_params_label)
            return
        
        method_info = ANALYSIS_METHODS[category][method_key]
        params = method_info.get("params", {})
        
        if not params:
            self.no_params_label = QLabel(self.LOCALIZE("ANALYSIS_PAGE.no_params_required"))
            self.no_params_label.setStyleSheet("color: #6c757d; font-style: italic; padding: 20px;")
            self.no_params_label.setAlignment(Qt.AlignCenter)
            self.params_layout.addWidget(self.no_params_label)
            return
        
        # Create parameter widgets
        for param_key, param_info in params.items():
            param_label = QLabel(self.LOCALIZE(f"ANALYSIS.PARAM.{param_key}", default=param_key.replace('_', ' ').title()) + ":")
            param_label.setStyleSheet("font-weight: 500; color: #2c3e50;")
            self.params_layout.addWidget(param_label)
            
            widget = create_parameter_widgets(param_info)
            self.params_layout.addWidget(widget)
            self.param_widgets[param_key] = widget
        
        self.params_layout.addStretch()
    
    def _reset_parameters(self):
        """Reset all parameters to default values."""
        category = self.category_combo.currentData()
        method_key = self.method_combo.currentData()
        
        if category and method_key:
            self._populate_parameters(category, method_key)
            self.showNotification.emit(
                self.LOCALIZE("ANALYSIS_PAGE.parameters_reset"),
                "info"
            )
    
    def _get_parameter_values(self) -> Dict[str, Any]:
        """Extract current parameter values from widgets."""
        params = {}
        
        for param_key, widget in self.param_widgets.items():
            if isinstance(widget, QSpinBox):
                params[param_key] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                params[param_key] = widget.value()
            elif isinstance(widget, QComboBox):
                params[param_key] = widget.currentData() or widget.currentText()
            elif isinstance(widget, QCheckBox):
                params[param_key] = widget.isChecked()
        
        return params
    
    def run_analysis(self):
        """Run the selected analysis method."""
        # Validate selection
        if not self.selected_datasets:
            self.showNotification.emit(
                self.LOCALIZE("ANALYSIS_PAGE.select_datasets_first"),
                "warning"
            )
            return
        
        category = self.category_combo.currentData()
        method_key = self.method_combo.currentData()
        
        if not category or not method_key:
            self.showNotification.emit(
                self.LOCALIZE("ANALYSIS_PAGE.select_method_first"),
                "warning"
            )
            return
        
        # Get parameters
        params = self._get_parameter_values()
        
        # Create cache key
        cache_key = self._generate_cache_key(category, method_key, params)
        
        # Check cache
        if cache_key in self.analysis_cache:
            create_logs("AnalysisPage", "run_analysis", 
                       "Using cached result", status='info')
            self._display_results(self.analysis_cache[cache_key])
            return
        
        # Run analysis in thread
        self.run_btn.setVisible(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.analysis_thread = AnalysisThread(
            category=category,
            method_key=method_key,
            params=params,
            dataset_data=self.dataset_data
        )
        
        self.analysis_thread.progress.connect(self.progress_bar.setValue)
        self.analysis_thread.finished.connect(self._on_analysis_finished)
        self.analysis_thread.error.connect(self._on_analysis_error)
        
        self.analysis_thread.start()
        
        create_logs("AnalysisPage", "run_analysis", 
                   f"Started analysis: {category}/{method_key}", status='info')
    
    def cancel_analysis(self):
        """Cancel running analysis."""
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.terminate()
            self.analysis_thread.wait()
            
            self.run_btn.setVisible(True)
            self.cancel_btn.setVisible(False)
            self.progress_bar.setVisible(False)
            
            self.showNotification.emit(
                self.LOCALIZE("ANALYSIS_PAGE.analysis_cancelled"),
                "info"
            )
    
    def _on_analysis_finished(self, result: AnalysisResult):
        """Handle analysis completion."""
        self.run_btn.setVisible(True)
        self.cancel_btn.setVisible(False)
        self.progress_bar.setVisible(False)
        
        # Cache result
        cache_key = self._generate_cache_key(
            result.category, result.method_key, result.params
        )
        self.analysis_cache[cache_key] = result
        self.current_analysis_result = result
        
        # Display results
        self._display_results(result)
        
        self.showNotification.emit(
            self.LOCALIZE("ANALYSIS_PAGE.analysis_complete"),
            "success"
        )
        
        create_logs("AnalysisPage", "analysis_finished", 
                   f"Analysis completed: {result.method_name}", status='info')
    
    def _on_analysis_error(self, error_msg: str):
        """Handle analysis error."""
        self.run_btn.setVisible(True)
        self.cancel_btn.setVisible(False)
        self.progress_bar.setVisible(False)
        
        self.showNotification.emit(
            self.LOCALIZE("ANALYSIS_PAGE.analysis_error") + f": {error_msg}",
            "error"
        )
        
        create_logs("AnalysisPage", "analysis_error", 
                   f"Analysis failed: {error_msg}", status='error')
    
    def _display_results(self, result: AnalysisResult):
        """Display analysis results in UI."""
        # Update quick stats
        stats_text = f"""📊 {self.LOCALIZE("ANALYSIS_PAGE.analysis_summary")}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Method: {result.method_name}
Category: {result.category}
Datasets: {len(result.dataset_names)} ({result.n_spectra} spectra)

{result.summary_text}

Status: ✓ Complete | Time: {result.execution_time:.2f}s
"""
        self.stats_display.setPlainText(stats_text)
        
        # Update results summary
        self.results_summary_text.setPlainText(result.detailed_summary)
        
        # Update plots
        if result.primary_figure:
            self.primary_plot.display_figure(result.primary_figure)
        
        if result.secondary_figure:
            self.secondary_plot.display_figure(result.secondary_figure)
        
        # Update data table
        if result.data_table is not None:
            self.data_table_display.setPlainText(result.data_table.to_string())
    
    def _generate_cache_key(self, category: str, method_key: str, params: Dict[str, Any]) -> str:
        """Generate cache key from analysis parameters."""
        import hashlib
        import json
        
        # Create deterministic string from parameters
        param_str = json.dumps({
            'category': category,
            'method': method_key,
            'params': params,
            'datasets': sorted(self.selected_datasets)
        }, sort_keys=True)
        
        # Generate hash
        return hashlib.md5(param_str.encode()).hexdigest()
    
    # Export methods
    def _export_plot(self, format: str):
        """Export current plot to file."""
        if not self.current_analysis_result:
            self.showNotification.emit(
                self.LOCALIZE("ANALYSIS_PAGE.no_results_to_export"),
                "warning"
            )
            return
        
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self.LOCALIZE("ANALYSIS_PAGE.save_plot"),
            f"analysis_plot.{format}",
            f"{format.upper()} Files (*.{format})"
        )
        
        if file_path:
            try:
                if self.viz_tabs.currentIndex() == 0:
                    fig = self.current_analysis_result.primary_figure
                else:
                    fig = self.current_analysis_result.secondary_figure
                
                if fig:
                    fig.savefig(file_path, dpi=300, bbox_inches='tight')
                    self.showNotification.emit(
                        self.LOCALIZE("ANALYSIS_PAGE.plot_exported"),
                        "success"
                    )
            except Exception as e:
                self.showNotification.emit(
                    f"{self.LOCALIZE('ANALYSIS.export_error')}: {str(e)}",
                    "error"
                )
    
    def _export_data_table(self):
        """Export data table to CSV."""
        if not self.current_analysis_result or self.current_analysis_result.data_table is None:
            self.showNotification.emit(
                self.LOCALIZE("ANALYSIS_PAGE.no_data_to_export"),
                "warning"
            )
            return
        
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self.LOCALIZE("ANALYSIS_PAGE.save_data"),
            "analysis_data.csv",
            "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                self.current_analysis_result.data_table.to_csv(file_path, index=True)
                self.showNotification.emit(
                    self.LOCALIZE("ANALYSIS_PAGE.data_exported"),
                    "success"
                )
            except Exception as e:
                self.showNotification.emit(
                    f"{self.LOCALIZE('ANALYSIS.export_error')}: {str(e)}",
                    "error"
                )
    
    def _export_full_report(self):
        """Export full analysis report as PDF."""
        self.showNotification.emit(
            self.LOCALIZE("ANALYSIS_PAGE.feature_coming_soon"),
            "info"
        )
    
    def _save_analysis_results(self):
        """Save analysis results to project."""
        if not self.current_analysis_result:
            self.showNotification.emit(
                self.LOCALIZE("ANALYSIS_PAGE.no_results_to_save"),
                "warning"
            )
            return
        
        # Save to project folder
        if PROJECT_MANAGER.current_project_path:
            analysis_dir = os.path.join(
                PROJECT_MANAGER.current_project_path,
                "analysis_results"
            )
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Generate filename
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.current_analysis_result.method_key}_{timestamp}"
            
            # Save figures
            if self.current_analysis_result.primary_figure:
                fig_path = os.path.join(analysis_dir, f"{filename}_primary.png")
                self.current_analysis_result.primary_figure.savefig(fig_path, dpi=300, bbox_inches='tight')
            
            if self.current_analysis_result.secondary_figure:
                fig_path = os.path.join(analysis_dir, f"{filename}_secondary.png")
                self.current_analysis_result.secondary_figure.savefig(fig_path, dpi=300, bbox_inches='tight')
            
            # Save data table
            if self.current_analysis_result.data_table is not None:
                data_path = os.path.join(analysis_dir, f"{filename}_data.csv")
                self.current_analysis_result.data_table.to_csv(data_path, index=True)
            
            # Save summary
            summary_path = os.path.join(analysis_dir, f"{filename}_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"Analysis Report\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"Method: {self.current_analysis_result.method_name}\n")
                f.write(f"Category: {self.current_analysis_result.category}\n")
                f.write(f"Execution Time: {self.current_analysis_result.execution_time:.2f}s\n\n")
                f.write(f"{self.current_analysis_result.detailed_summary}\n")
            
            self.showNotification.emit(
                self.LOCALIZE("ANALYSIS_PAGE.results_saved"),
                "success"
            )
            
            create_logs("AnalysisPage", "save_results", 
                       f"Saved analysis results to {analysis_dir}", status='info')
        else:
            self.showNotification.emit(
                self.LOCALIZE("ANALYSIS_PAGE.no_project_loaded"),
                "warning"
            )
