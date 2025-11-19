"""
Analysis Page (Version 2.0) - Card-Based Architecture

This module implements a modern card-based analysis interface for Raman spectroscopy data
with categorized method selection, dynamic parameter generation, and comprehensive results.

Architecture:
- Startup view: Card gallery organized by category (Exploratory, Statistical, Visualization)
- Method view: Split layout with input form (left) and results display (right)
- History sidebar: Session-based analysis tracking with clickable items
- Top bar: Navigation with "New Analysis" button

Key Features:
- 15+ analysis methods across 3 categories
- Dynamic parameter widgets generated from registry
- Threaded analysis execution with progress feedback
- Multi-tab results (plots, data tables, summaries, diagnostics)
- Comprehensive export (PNG, SVG, CSV, full reports)
- Full localization support (English + Japanese)
- Responsive design with proper error handling

UI Components (Modularized):
- views.py: Startup view, category sections, method cards, history sidebar, top bar
- method_view.py: Method-specific input forms and results panels
- export_utils.py: Export manager for all output formats
- thread.py: Background analysis execution
- result.py: Result data structures
- registry.py: Method definitions and parameters

Author: Enhanced by AI Assistant
Date: 2024-12-18
Version: 2.0
"""

import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QListWidgetItem, QMessageBox, QProgressBar, QFrame, QSplitter
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont

from components.widgets import load_icon
from components.widgets.matplotlib_widget import MatplotlibWidget
from configs.configs import load_config
from utils import RAMAN_DATA, PROJECT_MANAGER

# Import analysis utilities
from .analysis_page_utils import (
    ANALYSIS_METHODS,
    AnalysisResult,
    AnalysisThread
)
from .analysis_page_utils.views import (
    create_startup_view,
    create_history_sidebar,
    create_top_bar
)
from .analysis_page_utils.method_view import (
    create_method_view,
    populate_results_tabs
)
from .analysis_page_utils.export_utils import ExportManager


@dataclass
class AnalysisHistoryItem:
    """Represents a single analysis in the session history."""
    timestamp: datetime
    category: str
    method_key: str
    method_name: str
    dataset_name: str
    parameters: Dict[str, Any]
    result: Optional[AnalysisResult] = None


class AnalysisPage(QWidget):
    """
    Main analysis page with card-based interface.
    
    View States:
    - startup: Card gallery showing all available methods
    - method: Input form and results for selected method
    
    Signals:
    - analysis_started: Emitted when analysis begins
    - analysis_finished: Emitted when analysis completes
    - error_occurred: Emitted on analysis errors
    """
    
    analysis_started = Signal(str, str)  # category, method_key
    analysis_finished = Signal(str, str, object)  # category, method_key, result
    error_occurred = Signal(str)  # error message
    showNotification = Signal(str, str)  # title, message - for toast notifications
    
    def __init__(self, parent=None):
        """Initialize Analysis Page with card-based architecture."""
        super().__init__(parent)
        
        # Core references
        self.raman_data = RAMAN_DATA
        self.project_manager = PROJECT_MANAGER
        # Use global localization manager instead of creating a new one
        from utils import LOCALIZE
        self.localize = LOCALIZE  # Use global LOCALIZE function
        
        # State management
        self.current_view = "startup"  # 'startup' or 'method'
        self.current_category = None
        self.current_method_key = None
        self.current_result = None
        self.analysis_history: List[AnalysisHistoryItem] = []
        
        # Analysis thread
        self.analysis_thread: Optional[AnalysisThread] = None
        
        # Export manager
        self.export_manager = ExportManager(self, self.localize, self.project_manager)
        
        # Build UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup main UI layout with stacked views."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top bar with reduced height
        # self.top_bar = create_top_bar(self.localize, self._show_startup_view)
        # main_layout.addWidget(self.top_bar)
        
        # Main content: Sidebar + Stacked Views
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setChildrenCollapsible(False)
        
        # History sidebar
        self.history_sidebar = create_history_sidebar(self.localize)
        self.history_sidebar.history_list.itemClicked.connect(self._on_history_item_clicked)
        self.history_sidebar.clear_btn.clicked.connect(self._clear_history)
        content_splitter.addWidget(self.history_sidebar)
        
        # Stacked widget for views
        self.view_stack = QStackedWidget()
        
        # Startup view
        self.startup_view = create_startup_view(self.localize, self._show_method_view)
        self.view_stack.addWidget(self.startup_view)
        
        # Method view placeholder (created dynamically)
        self.method_view = None
        
        content_splitter.addWidget(self.view_stack)
        content_splitter.setSizes([280, 1000])  # Sidebar smaller than main content
        
        main_layout.addWidget(content_splitter)
        
        # Show startup view initially
        self._show_startup_view()
    
    def _show_startup_view(self):
        """Switch to startup view showing method cards."""
        self.current_view = "startup"
        self.view_stack.setCurrentWidget(self.startup_view)
        
        # Update top bar (if enabled)
        if hasattr(self, 'top_bar'):
            self.top_bar.new_analysis_btn.setVisible(False)
            self.top_bar.back_btn.setVisible(False)
            self.top_bar.title_label.setText("📊 " + self.localize("ANALYSIS_PAGE.title"))
    
    def _show_method_view(self, category: str, method_key: str):
        """
        Switch to method view for specific analysis method.
        
        Args:
            category: Method category (exploratory, statistical, visualization)
            method_key: Method identifier from registry
        """
        self.current_view = "method"
        self.current_category = category
        self.current_method_key = method_key
        
        # Get method info
        method_info = ANALYSIS_METHODS[category][method_key]
        
        # Get available datasets - RAMAN_DATA is a dict, get names from keys
        dataset_names = list(self.raman_data.keys()) if self.raman_data else []
        if not dataset_names:
            QMessageBox.warning(
                self,
                self.localize("ANALYSIS_PAGE.no_datasets_title"),
                self.localize("ANALYSIS_PAGE.no_datasets_message")
            )
            return
        
        # Remove old method view if exists
        if self.method_view:
            self.view_stack.removeWidget(self.method_view)
            self.method_view.deleteLater()
        
        # Create new method view
        self.method_view = create_method_view(
            category, method_key, dataset_names,
            self.localize,
            self._run_analysis,
            self._show_startup_view
        )
        
        # Connect export buttons (plot export via matplotlib toolbar)
        results_panel = self.method_view.results_panel
        results_panel.export_data_btn.clicked.connect(self._export_csv)
        
        self.view_stack.addWidget(self.method_view)
        self.view_stack.setCurrentWidget(self.method_view)
        
        # Update top bar (if enabled)
        if hasattr(self, 'top_bar'):
            self.top_bar.new_analysis_btn.setVisible(True)
            self.top_bar.back_btn.setVisible(True)
            self.top_bar.title_label.setText(f"📊 {method_info['name']}")
    
    def _run_analysis(self, category: str, method_key: str, dataset_selection, param_widget):
        """
        Execute analysis with selected parameters.
        
        Args:
            category: Method category
            method_key: Method identifier
            dataset_selection: Can be:
                - string: Single dataset name
                - list: Multiple dataset names
                - dict: Group assignments {group_label: [dataset_names]}
            param_widget: DynamicParameterWidget instance
        """
        # Get method info for validation
        method_info = ANALYSIS_METHODS[category][method_key]
        min_datasets = method_info.get("min_datasets", 1)
        max_datasets = method_info.get("max_datasets", None)
        
        # Handle group-based selection
        group_labels = None
        if isinstance(dataset_selection, dict):
            # Group mode: {group_label: [dataset_names]}
            if not dataset_selection:
                QMessageBox.warning(
                    self,
                    self.localize("ANALYSIS_PAGE.validation_error_title"),
                    self.localize("ANALYSIS_PAGE.no_groups_defined")
                )
                return
            
            # Flatten groups to get all dataset names
            selected_datasets = []
            group_labels = {}  # {dataset_name: group_label}
            for group_name, datasets in dataset_selection.items():
                for ds in datasets:
                    selected_datasets.append(ds)
                    group_labels[ds] = group_name
        
        # Normalize dataset_selection to list for consistent processing
        elif isinstance(dataset_selection, str):
            selected_datasets = [dataset_selection]
        elif isinstance(dataset_selection, list):
            selected_datasets = dataset_selection
        else:
            QMessageBox.critical(
                self,
                self.localize("ANALYSIS_PAGE.error_title"),
                "Invalid dataset selection format"
            )
            return
        
        # Validate number of datasets selected
        num_selected = len(selected_datasets)
        if num_selected < min_datasets:
            QMessageBox.warning(
                self,
                self.localize("ANALYSIS_PAGE.validation_error_title"),
                self.localize("ANALYSIS_PAGE.insufficient_datasets_message", 
                            required=min_datasets, selected=num_selected)
            )
            return
        
        if max_datasets is not None and num_selected > max_datasets:
            QMessageBox.warning(
                self,
                self.localize("ANALYSIS_PAGE.validation_error_title"),
                self.localize("ANALYSIS_PAGE.too_many_datasets_message",
                            max=max_datasets, selected=num_selected)
            )
            return
        
        # Get datasets from RAMAN_DATA dict
        dataset_data = {}
        missing_datasets = []
        for name in selected_datasets:
            dataset = self.raman_data.get(name)
            if dataset is None:
                missing_datasets.append(name)
            else:
                dataset_data[name] = dataset
        
        if missing_datasets:
            QMessageBox.critical(
                self,
                self.localize("ANALYSIS_PAGE.error_title"),
                self.localize("ANALYSIS_PAGE.datasets_not_found", 
                            datasets=", ".join(missing_datasets))
            )
            return
        
        # Extract parameters from widget using get_parameters() method
        parameters = param_widget.get_parameters() if param_widget else {}
        
        # Add group labels to parameters if present
        if group_labels:
            parameters['_group_labels'] = group_labels
        
        # Disable run button during execution
        self.method_view.run_btn.setEnabled(False)
        self.method_view.run_btn.setText(self.localize("ANALYSIS_PAGE.running"))
        
        # Create and start analysis thread
        # AnalysisThread expects dataset_data as Dict[str, pd.DataFrame]
        self.analysis_thread = AnalysisThread(
            category, method_key, parameters, dataset_data
        )
        self.analysis_thread.finished.connect(
            lambda result: self._on_analysis_finished(result, category, method_key, selected_datasets, parameters)
        )
        self.analysis_thread.error.connect(self._on_analysis_error)
        self.analysis_thread.progress.connect(self._on_analysis_progress)
        
        # Emit signal
        self.analysis_started.emit(category, method_key)
        
        # Start thread
        self.analysis_thread.start()
    
    def _on_analysis_finished(
        self,
        result: AnalysisResult,
        category: str,
        method_key: str,
        dataset_names: list,
        parameters: Dict
    ):
        """
        Handle completed analysis.
        
        Args:
            result: Analysis result object
            category: Method category
            method_key: Method identifier
            dataset_names: List of dataset names used
            parameters: Analysis parameters
        """
        # Re-enable run button
        self.method_view.run_btn.setEnabled(True)
        self.method_view.run_btn.setText(self.localize("ANALYSIS_PAGE.start_analysis_button"))
        
        # Store result
        self.current_result = result
        
        # Populate results tabs
        populate_results_tabs(
            self.method_view.results_panel,
            result,
            self.localize,
            MatplotlibWidget
        )
        
        # Add to history (use first dataset name for display, full list in result)
        method_info = ANALYSIS_METHODS[category][method_key]
        display_name = dataset_names[0] if len(dataset_names) == 1 else f"{len(dataset_names)} datasets"
        history_item = AnalysisHistoryItem(
            timestamp=datetime.now(),
            category=category,
            method_key=method_key,
            method_name=method_info["name"],
            dataset_name=display_name,
            parameters=parameters,
            result=result
        )
        self.analysis_history.append(history_item)
        self._update_history_list()
        
        # Emit signal
        self.analysis_finished.emit(category, method_key, result)
    
    def _on_analysis_error(self, error_msg: str):
        """
        Handle analysis errors.
        
        Args:
            error_msg: Error message
        """
        # Re-enable run button
        if self.method_view:
            self.method_view.run_btn.setEnabled(True)
            self.method_view.run_btn.setText(self.localize("ANALYSIS_PAGE.start_analysis_button"))
        
        # Show error dialog
        QMessageBox.critical(
            self,
            self.localize("ANALYSIS_PAGE.error_title"),
            self.localize("ANALYSIS_PAGE.analysis_error").format(error_msg)
        )
        
        # Emit signal
        self.error_occurred.emit(error_msg)
    
    def _on_analysis_progress(self, progress: int):
        """
        Update progress during analysis.
        
        Args:
            progress: Progress percentage (0-100)
        """
        # Update button text with progress
        if self.method_view:
            self.method_view.run_btn.setText(f"{self.localize('ANALYSIS_PAGE.running')}... ({progress}%)")
    
    def _update_history_list(self):
        """Update history sidebar with recent analyses."""
        history_list = self.history_sidebar.history_list
        history_list.clear()
        
        # Add items in reverse chronological order
        for idx, item in enumerate(reversed(self.analysis_history)):
            list_item = QListWidgetItem()
            
            # Format: "🕐 14:35 | PCA | Dataset 1"
            time_str = item.timestamp.strftime("%H:%M")
            text = f"🕐 {time_str}\n{item.method_name}\n📁 {item.dataset_name}"
            
            list_item.setText(text)
            list_item.setData(Qt.UserRole, len(self.analysis_history) - 1 - idx)  # Store index
            list_item.setFont(QFont("Segoe UI", 10))
            
            history_list.addItem(list_item)
    
    def _on_history_item_clicked(self, item: QListWidgetItem):
        """
        Restore analysis from history.
        
        Args:
            item: Clicked history list item
        """
        idx = item.data(Qt.UserRole)
        history_item = self.analysis_history[idx]
        
        # Show method view for this analysis
        self._show_method_view(history_item.category, history_item.method_key)
        
        # Restore parameters (if possible)
        if self.method_view and hasattr(self.method_view, 'param_widget'):
            # Set dataset - handle both simple and group modes
            dataset_widget = self.method_view.dataset_widget
            if dataset_widget and isinstance(history_item.dataset_name, str):
                # Handle both QComboBox (single mode) and QListWidget (multi mode)
                if hasattr(dataset_widget, 'findText'):  # QComboBox
                    dataset_idx = dataset_widget.findText(history_item.dataset_name)
                    if dataset_idx >= 0:
                        dataset_widget.setCurrentIndex(dataset_idx)
                elif hasattr(dataset_widget, 'findItems'):  # QListWidget
                    items = dataset_widget.findItems(history_item.dataset_name, Qt.MatchExactly)
                    if items:
                        items[0].setSelected(True)
            
            # Restore parameters by recreating the widget with saved params
            # The DynamicParameterWidget constructor accepts saved_params
            # This will be handled by _show_method_view when it creates the widget
            
            # Display cached result if available
            if history_item.result:
                populate_results_tabs(
                    self.method_view.results_panel,
                    history_item.result,
                    self.localize,
                    MatplotlibWidget
                )
                self.current_result = history_item.result
    
    def _clear_history(self):
        """Clear analysis history."""
        reply = QMessageBox.question(
            self,
            self.localize("ANALYSIS_PAGE.clear_history_title"),
            self.localize("ANALYSIS_PAGE.clear_history_confirm"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.analysis_history.clear()
            self._update_history_list()
    
    def _on_dataset_changed(self):
        """Handle dataset changes (load/remove)."""
        # If in method view, update dataset widget
        if self.method_view and hasattr(self.method_view, 'dataset_widget'):
            widget = self.method_view.dataset_widget
            
            # Handle QComboBox (single-dataset mode)
            if hasattr(widget, 'currentText'):
                current = widget.currentText()
                widget.clear()
                
                if self.raman_data:
                    # RAMAN_DATA is a dict, get dataset names from keys
                    widget.addItems(list(self.raman_data.keys()))
                    
                    # Restore selection if still available
                    idx = widget.findText(current)
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
            
            # Handle QListWidget (multi-dataset mode)
            elif hasattr(widget, 'selectedItems'):
                selected_names = [item.text() for item in widget.selectedItems()]
                widget.clear()
                
                if self.raman_data:
                    widget.addItems(list(self.raman_data.keys()))
                    
                    # Restore selections if still available
                    for i in range(widget.count()):
                        if widget.item(i).text() in selected_names:
                            widget.item(i).setSelected(True)
    
    # === Export Methods ===
    
    def _export_png(self):
        """Export current plot as PNG."""
        if not self.current_result or not self.current_result.primary_figure:
            return
        
        method_name = ANALYSIS_METHODS[self.current_category][self.current_method_key]["name"]
        filename = f"{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        self.export_manager.export_plot_png(self.current_result.primary_figure, filename)
    
    def _export_svg(self):
        """Export current plot as SVG."""
        if not self.current_result or not self.current_result.primary_figure:
            return
        
        method_name = ANALYSIS_METHODS[self.current_category][self.current_method_key]["name"]
        filename = f"{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg"
        
        self.export_manager.export_plot_svg(self.current_result.primary_figure, filename)
    
    def _export_csv(self):
        """Export current data table as CSV."""
        if not self.current_result or self.current_result.data_table is None:
            return
        
        method_name = ANALYSIS_METHODS[self.current_category][self.current_method_key]["name"]
        filename = f"{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        self.export_manager.export_data_csv(self.current_result.data_table, filename)
    
    def export_full_report(self):
        """Export complete analysis report (public method for external calls)."""
        if not self.current_result:
            QMessageBox.warning(
                self,
                self.localize("ANALYSIS_PAGE.warning_title"),
                self.localize("ANALYSIS_PAGE.no_results_to_export")
            )
            return
        
        method_name = ANALYSIS_METHODS[self.current_category][self.current_method_key]["name"]
        
        # Get dataset name from widget
        dataset_name = "Unknown"
        if self.method_view and hasattr(self.method_view, 'dataset_widget'):
            widget = self.method_view.dataset_widget
            if hasattr(widget, 'currentText'):
                dataset_name = widget.currentText()
            elif hasattr(widget, 'selectedItems'):
                selected = widget.selectedItems()
                dataset_name = selected[0].text() if selected else "Multiple"
        
        # Get current parameters
        parameters = {}
        if self.method_view and hasattr(self.method_view, 'param_widget'):
            try:
                parameters = self.method_view.param_widget.get_parameters()
            except:
                pass
        
        self.export_manager.export_full_report(
            self.current_result,
            method_name,
            parameters,
            dataset_name
        )
    
    def save_to_project(self):
        """Save current analysis to project folder (public method)."""
        if not self.current_result:
            QMessageBox.warning(
                self,
                self.localize("ANALYSIS_PAGE.warning_title"),
                self.localize("ANALYSIS_PAGE.no_results_to_save")
            )
            return
        
        method_name = ANALYSIS_METHODS[self.current_category][self.current_method_key]["name"]
        
        # Get dataset name from widget
        dataset_name = "Unknown"
        if self.method_view and hasattr(self.method_view, 'dataset_widget'):
            widget = self.method_view.dataset_widget
            if hasattr(widget, 'currentText'):
                dataset_name = widget.currentText()
            elif hasattr(widget, 'selectedItems'):
                selected = widget.selectedItems()
                dataset_name = selected[0].text() if selected else "Multiple"
        
        # Get current parameters
        parameters = {}
        if self.method_view and hasattr(self.method_view, 'param_widget'):
            try:
                parameters = self.method_view.param_widget.get_parameters()
            except:
                pass
        
        self.export_manager.save_to_project(
            self.current_result,
            method_name,
            parameters,
            dataset_name
        )
