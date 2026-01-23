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

Author: MUHAMMAD HELMI BIN ROZAIN
Date: 2024-12-18
Version: 2.0
"""

import sys
import os
import traceback
import json
import pickle
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

import matplotlib.pyplot as plt

from configs.configs import create_logs

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QListWidgetItem,
    QLabel,
    QPushButton,
    QToolButton,
    QMessageBox,
    QProgressBar,
    QFrame,
    QSplitter,
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont

from components.widgets import load_icon
from components.widgets.matplotlib_widget import MatplotlibWidget
from configs.configs import load_config
from utils import RAMAN_DATA, PROJECT_MANAGER

# Import analysis utilities
from .analysis_page_utils import ANALYSIS_METHODS, AnalysisResult, AnalysisThread
from .analysis_page_utils.views import (
    create_startup_view,
    create_history_sidebar,
    create_top_bar,
)
from .analysis_page_utils.method_view import MethodView, populate_results_tabs
from .analysis_page_utils.export_utils import ExportManager


# Maximum number of history items to prevent memory leaks from figure storage
MAX_HISTORY = 20


@dataclass
class AnalysisHistoryItem:
    """
    Represents a single analysis in the session history.

    Stores full dataset list and group labels for accurate history restoration
    (P0-3 fix: previously only stored display string which broke reproducibility).
    """

    timestamp: datetime
    category: str
    method_key: str
    method_name: str
    dataset_names: List[str]  # Full list of dataset names (P0-3 fix)
    parameters: Dict[str, Any]
    group_labels: Optional[Dict[str, str]] = (
        None  # {dataset: group} for group mode (P0-3 fix)
    )
    result: Optional[AnalysisResult] = None
    storage_id: Optional[str] = None  # stable id for disk persistence + UI selection

    @property
    def display_name(self) -> str:
        """Generate display name for UI from dataset_names."""
        if len(self.dataset_names) == 1:
            return self.dataset_names[0]
        return f"{len(self.dataset_names)} datasets"


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
        self._history_by_id: Dict[str, AnalysisHistoryItem] = {}

        # Analysis thread
        self.analysis_thread: Optional[AnalysisThread] = None
        # Cancellation / force-stop (detach) state
        self._analysis_cancel_requested: bool = False
        self._ignored_thread_ids: set[int] = set()

        # Export manager
        self.export_manager = ExportManager(self, self.localize, self.project_manager)

        # Build UI
        self._setup_ui()

    # === Persistent analysis history (per project) ===
    def load_project_data(self):
        """WorkspacePage calls this after PROJECT_MANAGER.load_project()."""
        try:
            self._load_persisted_history()
        except Exception as e:
            create_logs(
                "AnalysisPage",
                "analysis_history",
                f"Failed to load persisted analysis history: {e}",
                status="warning",
            )

    def _get_history_storage_dir(self) -> str | None:
        project_path = self.project_manager.current_project_data.get("projectFilePath")
        if not project_path:
            return None
        project_root = os.path.dirname(project_path)
        history_dir = os.path.join(project_root, "analysis_history")
        os.makedirs(history_dir, exist_ok=True)
        return history_dir

    def _history_index_path(self) -> str | None:
        d = self._get_history_storage_dir()
        if not d:
            return None
        return os.path.join(d, "index.json")

    def _load_persisted_history(self) -> None:
        """Load persisted history metadata for the current project."""
        history_dir = self._get_history_storage_dir()
        if not history_dir:
            self.analysis_history = []
            self._history_by_id = {}
            self._update_history_list()
            return

        index_path = self._history_index_path()
        entries: list[dict] = []

        if index_path and os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                entries = payload.get("entries", []) if isinstance(payload, dict) else []
            except Exception:
                entries = []

        # Fallback: scan for legacy/meta files if index is missing.
        if not entries:
            try:
                for fname in os.listdir(history_dir):
                    if fname.endswith(".meta.json"):
                        with open(os.path.join(history_dir, fname), "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        if isinstance(meta, dict):
                            entries.append(meta)
            except Exception:
                pass

        items: List[AnalysisHistoryItem] = []
        by_id: Dict[str, AnalysisHistoryItem] = {}

        for meta in entries:
            try:
                storage_id = str(meta.get("id") or "").strip()
                ts = meta.get("timestamp")
                timestamp = datetime.fromisoformat(ts) if isinstance(ts, str) else datetime.now()
                category = str(meta.get("category") or "")
                method_key = str(meta.get("method_key") or "")
                method_name = str(meta.get("method_name") or method_key)
                dataset_names = list(meta.get("dataset_names") or [])
                group_labels = meta.get("group_labels")
                if not isinstance(group_labels, dict):
                    group_labels = None

                item = AnalysisHistoryItem(
                    timestamp=timestamp,
                    category=category,
                    method_key=method_key,
                    method_name=method_name,
                    dataset_names=[str(x) for x in dataset_names],
                    parameters={},  # loaded lazily on click
                    group_labels=group_labels,
                    result=None,
                    storage_id=storage_id or None,
                )
                if item.storage_id:
                    by_id[item.storage_id] = item
                items.append(item)
            except Exception:
                continue

        # Sort newest first
        items.sort(key=lambda it: it.timestamp, reverse=True)
        self.analysis_history = items
        self._history_by_id = by_id
        self._update_history_list()

    def _persist_history_item(self, item: AnalysisHistoryItem) -> None:
        """Persist one history item's metadata + parameters for later restore."""
        history_dir = self._get_history_storage_dir()
        if not history_dir:
            return

        if not item.storage_id:
            item.storage_id = item.timestamp.strftime("%Y%m%d_%H%M%S_%f")

        meta_path = os.path.join(history_dir, f"{item.storage_id}.meta.json")
        params_path = os.path.join(history_dir, f"{item.storage_id}.params.pkl")

        meta = {
            "id": item.storage_id,
            "timestamp": item.timestamp.isoformat(),
            "category": item.category,
            "method_key": item.method_key,
            "method_name": item.method_name,
            "dataset_names": list(item.dataset_names or []),
            "group_labels": item.group_labels or None,
            "mode": "grouped" if item.group_labels else "simple",
        }

        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # Parameters can contain non-JSON types; store as pickle.
        try:
            with open(params_path, "wb") as f:
                pickle.dump(
                    {
                        "parameters": item.parameters,
                        "group_labels": item.group_labels,
                        "dataset_names": item.dataset_names,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception as e:
            create_logs(
                "AnalysisPage",
                "analysis_history",
                f"Failed to persist params for history item {item.storage_id}: {e}",
                status="warning",
            )

        # Update index
        index_path = self._history_index_path()
        if not index_path:
            return

        payload = {"version": 1, "entries": []}
        try:
            if os.path.exists(index_path):
                with open(index_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, dict) and isinstance(existing.get("entries"), list):
                    payload = existing
        except Exception:
            payload = {"version": 1, "entries": []}

        # De-dup then prepend
        entries = [e for e in payload.get("entries", []) if isinstance(e, dict) and e.get("id") != item.storage_id]
        entries.insert(0, meta)
        payload["entries"] = entries[:200]  # cap growth

        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _setup_ui(self):
        """Setup main UI layout with stacked views."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top bar with navigation (DISABLED: User requested to keep disabled for now)
        # self.top_bar = create_top_bar(self.localize, self._show_startup_view)
        # main_layout.addWidget(self.top_bar)

        # Main content: Sidebar + Stacked Views
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setChildrenCollapsible(False)

        # History sidebar
        self.history_sidebar = create_history_sidebar(self.localize)
        self.history_sidebar.history_list.itemClicked.connect(
            self._on_history_item_clicked
        )
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

    def clear_project_data(self):
        """
        Clear all analysis data and reset to startup view.
        Called when switching projects or returning to home.
        """
        create_logs(
            "clear_project_data",
            "analysis_page",
            "Clearing all analysis data and resetting to startup view",
            "info",
        )
        
        # Cancel any running analysis
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.cancel()
            self.analysis_thread.wait()
        
        # Reset state variables
        self.current_view = "startup"
        self.current_category = None
        self.current_method_key = None
        self.current_result = None
        self.analysis_history = []
        
        # Clear sidebar
        if hasattr(self, "sidebar_widget"):
            self.sidebar_widget.history_list.clear()
        
        # Remove method view if it exists
        if self.method_view:
            self.view_stack.removeWidget(self.method_view)
            self.method_view.deleteLater()
            self.method_view = None
        
        # Show startup view
        self._show_startup_view()
        
        create_logs(
            "clear_project_data",
            "analysis_page",
            "Successfully cleared analysis data",
            "info",
        )

    def _show_startup_view(self):
        """Switch to startup view showing method cards."""
        self.current_view = "startup"
        self.view_stack.setCurrentWidget(self.startup_view)

        # Update top bar (if enabled)
        if hasattr(self, "top_bar"):
            self.top_bar.new_analysis_btn.setVisible(False)
            self.top_bar.back_btn.setVisible(False)
            self.top_bar.title_label.setText(
                "📊 " + self.localize("ANALYSIS_PAGE.title")
            )

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
            create_logs(
                "_show_method_view",
                "analysis_page",
                "No datasets available for analysis.",
                "warning",
            )

            QMessageBox.warning(
                self,
                self.localize("ANALYSIS_PAGE.no_datasets_title"),
                self.localize("ANALYSIS_PAGE.no_datasets_message"),
            )
            return

        # Remove old method view if exists
        if self.method_view:
            self.view_stack.removeWidget(self.method_view)
            self.method_view.deleteLater()

        # Create new method view
        self.method_view = MethodView(
            category,
            method_key,
            dataset_names,
            self.localize,
            self._run_analysis,
            self._show_startup_view,
            on_stop=self._stop_analysis,
        )

        # Connect export buttons (plot export via matplotlib toolbar)
        results_panel = self.method_view.results_panel
        results_panel.export_data_btn.clicked.connect(
            self.method_view._handle_export_csv
        )

        self.view_stack.addWidget(self.method_view)
        self.view_stack.setCurrentWidget(self.method_view)

        # Update top bar (if enabled)
        if hasattr(self, "top_bar"):
            self.top_bar.new_analysis_btn.setVisible(True)
            self.top_bar.back_btn.setVisible(True)
            self.top_bar.title_label.setText(f"📊 {method_info['name']}")

    def _run_analysis(
        self,
        category: str,
        method_key: str,
        dataset_selection,
        param_widget,
        history_restore_id: str | None = None,
    ):
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
        # Prevent running multiple analyses concurrently (starting a second analysis while
        # another is running can overload CPU/memory and lead to confusing UI updates).
        try:
            if self.analysis_thread and self.analysis_thread.isRunning():
                QMessageBox.information(
                    self,
                    self.localize("ANALYSIS_PAGE.validation_error_title"),
                    self.localize("ANALYSIS_PAGE.running"),
                )
                return
        except Exception:
            pass

        # Reset per-run cancellation state
        self._analysis_cancel_requested = False

        # Get method info for validation
        method_info = ANALYSIS_METHODS[category][method_key]
        min_datasets = method_info.get("min_datasets", 1)
        max_datasets = method_info.get("max_datasets", None)

        # Handle group-based selection
        group_labels = None
        if isinstance(dataset_selection, dict):
            # Group mode: {group_label: [dataset_names]}
            if not dataset_selection:
                create_logs(
                    "_run_analysis",
                    "analysis_page",
                    "No groups defined in dataset selection",
                    "warning",
                )

                QMessageBox.warning(
                    self,
                    self.localize("ANALYSIS_PAGE.validation_error_title"),
                    self.localize("ANALYSIS_PAGE.no_groups_defined"),
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
            create_logs(
                "_run_analysis",
                "analysis_page",
                "Invalid dataset selection format",
                "error",
            )

            QMessageBox.critical(
                self,
                self.localize("ANALYSIS_PAGE.error_title"),
                self.localize("ANALYSIS_PAGE.invalid_dataset_selection_format"),
            )
            return

        # Validate number of datasets selected
        num_selected = len(selected_datasets)
        if num_selected < min_datasets:
            create_logs(
                "_run_analysis",
                "analysis_page",
                f"Insufficient datasets selected: {num_selected} (min {min_datasets})",
                "warning",
            )

            QMessageBox.warning(
                self,
                self.localize("ANALYSIS_PAGE.validation_error_title"),
                self.localize(
                    "ANALYSIS_PAGE.insufficient_datasets_message",
                    required=min_datasets,
                    selected=num_selected,
                ),
            )
            return

        if max_datasets is not None and num_selected > max_datasets:
            create_logs(
                "_run_analysis",
                "analysis_page",
                f"Too many datasets selected: {num_selected} (max {max_datasets})",
                "warning",
            )

            QMessageBox.warning(
                self,
                self.localize("ANALYSIS_PAGE.validation_error_title"),
                self.localize(
                    "ANALYSIS_PAGE.too_many_datasets_message",
                    max=max_datasets,
                    selected=num_selected,
                ),
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
            create_logs(
                "_run_analysis",
                "analysis_page",
                f"Datasets not found: {', '.join(missing_datasets)}",
                "error",
            )

            QMessageBox.critical(
                self,
                self.localize("ANALYSIS_PAGE.error_title"),
                self.localize(
                    "ANALYSIS_PAGE.datasets_not_found",
                    datasets=", ".join(missing_datasets),
                ),
            )
            return

        # Extract parameters from widget using get_parameters() method
        parameters = param_widget.get_parameters() if param_widget else {}

        # Add group labels to parameters if present
        if group_labels:
            parameters["_group_labels"] = group_labels

        # Switch Run → Stop during execution (and show progress on results overlay)
        if self.method_view:
            self.method_view.set_running_state(True)

        if hasattr(self.method_view.results_panel, "show_loading"):
            self.method_view.results_panel.show_loading(self.localize("ANALYSIS_PAGE.running"))
        
        # Create and start analysis thread
        # AnalysisThread expects dataset_data as Dict[str, pd.DataFrame]
        self.analysis_thread = AnalysisThread(
            category, method_key, parameters, dataset_data
        )
        self.analysis_thread.finished.connect(
            lambda result: self._on_analysis_finished(
                result,
                category,
                method_key,
                selected_datasets,
                parameters,
                history_restore_id,
            )
        )
        self.analysis_thread.error.connect(self._on_analysis_error)
        self.analysis_thread.cancelled.connect(self._on_analysis_cancelled)
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
        parameters: Dict,
        history_restore_id: str | None = None,
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
        # Ignore late results from a force-detached thread (safe "force stop").
        try:
            s = self.sender()
            if s is not None and id(s) in self._ignored_thread_ids:
                return
            if self.analysis_thread is not None and s is not None and s is not self.analysis_thread:
                return
        except Exception:
            pass

        self._analysis_cancel_requested = False

        # Restore Run state
        if self.method_view:
            self.method_view.set_running_state(False)
        
        # Hide loading overlay
        if hasattr(self.method_view.results_panel, 'hide_loading'):
            self.method_view.results_panel.hide_loading()
        
        # Store result
        self.current_result = result

        # Get method name from ANALYSIS_METHODS
        method_info = ANALYSIS_METHODS.get(category, {}).get(method_key, {})
        method_name = method_info.get("name", method_key)

        # Populate results tabs
        populate_results_tabs(
            self.method_view.results_panel, result, self.localize, MatplotlibWidget, method_name
        )

        # Extract group labels if present in parameters
        group_labels = (
            parameters.pop("_group_labels", None)
            if "_group_labels" in parameters
            else None
        )

        # If this was triggered from history restore, update the existing item instead
        # of creating a new entry.
        if history_restore_id and history_restore_id in self._history_by_id:
            hist = self._history_by_id[history_restore_id]
            hist.parameters = parameters
            hist.group_labels = group_labels
            hist.result = result
            try:
                self._persist_history_item(hist)
            except Exception:
                pass
            self._update_history_list()
        else:
            # P0-2: Limit history size to prevent memory leaks from figure storage
            if len(self.analysis_history) >= MAX_HISTORY:
                old_item = self.analysis_history.pop(0)
                self._cleanup_history_item(old_item)
                if old_item.storage_id and old_item.storage_id in self._history_by_id:
                    self._history_by_id.pop(old_item.storage_id, None)

            method_info = ANALYSIS_METHODS[category][method_key]
            history_item = AnalysisHistoryItem(
                timestamp=datetime.now(),
                category=category,
                method_key=method_key,
                method_name=method_info["name"],
                dataset_names=dataset_names,
                parameters=parameters,
                group_labels=group_labels,
                result=result,
            )
            # Persist and index
            self._persist_history_item(history_item)
            if history_item.storage_id:
                self._history_by_id[history_item.storage_id] = history_item
            self.analysis_history.append(history_item)
            self._update_history_list()

        # Emit signal
        self.analysis_finished.emit(category, method_key, result)

    def _on_analysis_error(self, error_msg: str, traceback: str = ""):
        """
        Handle analysis errors.

        Args:
            error_msg: Error message
        """
        # Ignore late errors from a force-detached thread.
        try:
            s = self.sender()
            if s is not None and id(s) in self._ignored_thread_ids:
                return
            if self.analysis_thread is not None and s is not None and s is not self.analysis_thread:
                return
        except Exception:
            pass

        self._analysis_cancel_requested = False

        # Restore Run state
        if self.method_view:
            self.method_view.set_running_state(False)
            
            # Hide loading overlay
            if hasattr(self.method_view.results_panel, 'hide_loading'):
                self.method_view.results_panel.hide_loading()
        
        # log error
        create_logs(
            "_on_analysis_error",
            "analysis_page",
            f"Analysis error: {error_msg} \n {traceback}",
            "error",
        )

        # Show error dialog
        QMessageBox.critical(
            self,
            self.localize("ANALYSIS_PAGE.error_title"),
            self.localize("ANALYSIS_PAGE.analysis_error", error=error_msg),
        )
        

        # Emit signal
        self.error_occurred.emit(error_msg)

    def _on_analysis_cancelled(self) -> None:
        """Handle cooperative cancellation (Stop button)."""

        # Ignore late cancelled signals from a force-detached thread.
        try:
            s = self.sender()
            if s is not None and id(s) in self._ignored_thread_ids:
                return
            if self.analysis_thread is not None and s is not None and s is not self.analysis_thread:
                return
        except Exception:
            pass

        self._analysis_cancel_requested = False

        try:
            if self.method_view:
                self.method_view.set_running_state(False)
                if hasattr(self.method_view.results_panel, "hide_loading"):
                    self.method_view.results_panel.hide_loading()
        except Exception:
            pass

        try:
            create_logs(
                "_on_analysis_cancelled",
                "analysis_page",
                "Analysis cancelled by user",
                status="info",
            )
        except Exception:
            pass

        # Optional: show a lightweight placeholder again.
        try:
            if self.method_view and hasattr(self.method_view.results_panel, "show_placeholder"):
                self.method_view.results_panel.show_placeholder()
        except Exception:
            pass

    def _on_analysis_progress(self, progress: int):
        """
        Update progress during analysis.

        Args:
            progress: Progress percentage (0-100)
        """
        # Ignore late progress from a force-detached thread.
        try:
            s = self.sender()
            if s is not None and id(s) in self._ignored_thread_ids:
                return
            if self.analysis_thread is not None and s is not None and s is not self.analysis_thread:
                return
        except Exception:
            pass

        # Show progress on results overlay (keep button as Stop)
        try:
            if self.method_view and hasattr(self.method_view.results_panel, "show_loading"):
                self.method_view.results_panel.show_loading(
                    f"{self.localize('ANALYSIS_PAGE.running')} ({progress}%)"
                )
        except Exception:
            pass

    def _stop_analysis(self) -> None:
        """Stop the running analysis.

        - First click: cooperative cancellation (best-effort).
        - Second click (if still running): safe "force stop" by detaching the UI
          from the worker thread so the app stays responsive. The computation may
          still finish in the background, but its signals/results are ignored.
        """

        if not getattr(self, "analysis_thread", None):
            return

        try:
            if self.analysis_thread and not self.analysis_thread.isRunning():
                return
        except Exception:
            pass

        # Second click: detach (safe force stop)
        if self._analysis_cancel_requested:
            try:
                t = self.analysis_thread
                if t is not None:
                    self._ignored_thread_ids.add(id(t))
            except Exception:
                pass

            # Restore UI immediately
            try:
                if self.method_view:
                    self.method_view.set_running_state(False)
                    if hasattr(self.method_view.results_panel, "hide_loading"):
                        self.method_view.results_panel.hide_loading()
                    if hasattr(self.method_view.results_panel, "show_placeholder"):
                        self.method_view.results_panel.show_placeholder()
            except Exception:
                pass

            try:
                create_logs(
                    "_stop_analysis",
                    "analysis_page",
                    "Force-stopped analysis UI (detached from worker thread)",
                    status="warning",
                )
            except Exception:
                pass

            # Detach thread reference so a new run can start.
            self.analysis_thread = None
            self._analysis_cancel_requested = False
            return

        self._analysis_cancel_requested = True

        # Keep Stop clickable: user can click again to force-detach if the worker is stuck
        try:
            if self.method_view:
                self.method_view.set_running_state(True, enabled=True)
        except Exception:
            pass

        try:
            if self.method_view and hasattr(self.method_view.results_panel, "show_loading"):
                self.method_view.results_panel.show_loading(self.localize("ANALYSIS_PAGE.stopping"))
        except Exception:
            pass

        try:
            self.analysis_thread.cancel()
        except Exception:
            pass

    def _update_history_list(self):
        """Update history sidebar with recent analyses."""
        history_list = self.history_sidebar.history_list
        history_list.clear()

        # Add items in reverse chronological order with day headers.
        last_day: str | None = None
        for item in sorted(self.analysis_history, key=lambda it: it.timestamp, reverse=True):
            day = item.timestamp.strftime("%Y-%m-%d")
            if day != last_day:
                header = QListWidgetItem(day)
                header.setFlags(Qt.ItemIsEnabled)
                header.setFont(QFont("Segoe UI", 9, QFont.Bold))
                header.setForeground(Qt.gray)
                history_list.addItem(header)
                last_day = day

            list_item = QListWidgetItem()
            dt_str = item.timestamp.strftime("%H:%M")

            if item.group_labels:
                group_names = sorted(set(item.group_labels.values()))
                mode_line = "🔀 " + ", ".join(group_names[:4]) + ("…" if len(group_names) > 4 else "")
            else:
                mode_line = "🔹 " + self.localize("ANALYSIS_PAGE.simple_mode")

            # Store stable id when available; otherwise fallback to index.
            list_item.setData(Qt.UserRole, item.storage_id)

            # Custom widget so we can attach a per-item delete button.
            w = QWidget()
            w_layout = QVBoxLayout(w)
            w_layout.setContentsMargins(8, 6, 8, 6)
            w_layout.setSpacing(2)

            top_row = QHBoxLayout()
            top_row.setContentsMargins(0, 0, 0, 0)
            top_row.setSpacing(6)

            time_label = QLabel(f"🕐 {dt_str}")
            time_label.setStyleSheet("font-size: 10px; color: #6c757d;")

            delete_btn = QToolButton()
            delete_btn.setObjectName("historyItemDeleteButton")
            delete_btn.setToolTip(self.localize("ANALYSIS_PAGE.delete_history_item_tooltip"))
            delete_btn.setFixedSize(26, 26)
            delete_btn.setCursor(Qt.PointingHandCursor)
            delete_btn.setFocusPolicy(Qt.NoFocus)
            try:
                delete_btn.setIcon(load_icon("trash_bin", QSize(14, 14), "#dc3545"))
                delete_btn.setIconSize(QSize(14, 14))
            except Exception:
                delete_btn.setText("✕")
            delete_btn.setStyleSheet(
                """
                QToolButton#historyItemDeleteButton {
                    border: 1px solid transparent;
                    border-radius: 6px;
                    background: transparent;
                    padding: 2px;
                }
                QToolButton#historyItemDeleteButton:hover {
                    background: #ffecec;
                    border-color: #ffb3b3;
                }
                QToolButton#historyItemDeleteButton:pressed {
                    background: #ffd6d6;
                }
                """
            )

            if item.storage_id:
                delete_btn.clicked.connect(lambda _=False, sid=item.storage_id: self._delete_history_item(sid))
            else:
                delete_btn.setEnabled(False)

            top_row.addWidget(time_label)
            top_row.addStretch()
            top_row.addWidget(delete_btn)
            w_layout.addLayout(top_row)

            method_label = QLabel(str(item.method_name))
            method_label.setStyleSheet("font-size: 12px; font-weight: 700; color: #2c3e50;")
            w_layout.addWidget(method_label)

            mode_label = QLabel(mode_line)
            mode_label.setStyleSheet("font-size: 10px; color: #495057;")
            w_layout.addWidget(mode_label)

            path_label = QLabel(f"📁 {item.display_name}")
            path_label.setStyleSheet("font-size: 10px; color: #6c757d;")
            path_label.setWordWrap(True)
            w_layout.addWidget(path_label)

            history_list.addItem(list_item)
            list_item.setSizeHint(w.sizeHint())
            history_list.setItemWidget(list_item, w)

    def _delete_history_item(self, storage_id: str) -> None:
        """Delete a single history entry (UI + persisted files)."""

        if not storage_id:
            return

        hist = self._history_by_id.get(storage_id)
        if hist is None:
            return

        # Confirm (deleting removes persisted restore files)
        try:
            ts = hist.timestamp.strftime("%Y-%m-%d %H:%M")
        except Exception:
            ts = ""
        resp = QMessageBox.question(
            self,
            self.localize("ANALYSIS_PAGE.delete_history_item_title"),
            self.localize(
                "ANALYSIS_PAGE.delete_history_item_confirm",
                method=str(hist.method_name or ""),
                time=str(ts),
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if resp != QMessageBox.Yes:
            return

        # Remove from disk
        history_dir = self._get_history_storage_dir()
        if history_dir:
            for suffix in (".meta.json", ".params.pkl"):
                p = os.path.join(history_dir, f"{storage_id}{suffix}")
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

            # Update index.json if present
            index_path = self._history_index_path()
            if index_path and os.path.exists(index_path):
                try:
                    with open(index_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
                        payload["entries"] = [
                            e
                            for e in payload.get("entries", [])
                            if not (isinstance(e, dict) and str(e.get("id")) == storage_id)
                        ]
                        with open(index_path, "w", encoding="utf-8") as f:
                            json.dump(payload, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

        # Remove from memory
        try:
            self._cleanup_history_item(hist)
        except Exception:
            pass

        self.analysis_history = [it for it in self.analysis_history if it.storage_id != storage_id]
        self._history_by_id.pop(storage_id, None)
        self._update_history_list()

    def _on_history_item_clicked(self, item: QListWidgetItem):
        """
        Restore analysis from history.

        Args:
            item: Clicked history list item
        """
        storage_id = item.data(Qt.UserRole)
        if not storage_id:
            return

        history_item = self._history_by_id.get(storage_id)
        if history_item is None:
            return

        # UX: show a loading overlay immediately when restoring.
        try:
            if self.method_view and hasattr(self.method_view.results_panel, "show_loading"):
                self.method_view.results_panel.show_loading("⏳ Restoring...")
        except Exception:
            pass

        # Load persisted parameters if needed
        if not history_item.parameters and history_item.storage_id:
            history_dir = self._get_history_storage_dir()
            if history_dir:
                params_path = os.path.join(history_dir, f"{history_item.storage_id}.params.pkl")
                try:
                    with open(params_path, "rb") as f:
                        payload = pickle.load(f)
                    if isinstance(payload, dict):
                        history_item.parameters = payload.get("parameters") or {}
                        gl = payload.get("group_labels")
                        history_item.group_labels = gl if isinstance(gl, dict) else history_item.group_labels
                        dn = payload.get("dataset_names")
                        if isinstance(dn, list) and dn:
                            history_item.dataset_names = [str(x) for x in dn]
                except Exception:
                    history_item.parameters = history_item.parameters or {}

        # Show method view for this analysis (pass saved params so widgets are pre-filled)
        self.current_view = "method"
        self.current_category = history_item.category
        self.current_method_key = history_item.method_key

        # Use existing method builder but inject saved params
        # (Keeps behavior identical for fresh analyses)
        self._show_method_view(history_item.category, history_item.method_key)
        try:
            # Rebuild params widget with saved params by recreating MethodView
            if self.method_view:
                # Replace method view with one that uses saved params
                self.view_stack.removeWidget(self.method_view)
                self.method_view.deleteLater()
                self.method_view = None

            dataset_names = list(self.raman_data.keys()) if self.raman_data else []
            self.method_view = MethodView(
                history_item.category,
                history_item.method_key,
                dataset_names,
                self.localize,
                self._run_analysis,
                self._show_startup_view,
                on_stop=self._stop_analysis,
                saved_params=history_item.parameters,
            )

            results_panel = self.method_view.results_panel
            results_panel.export_data_btn.clicked.connect(self.method_view._handle_export_csv)

            self.view_stack.addWidget(self.method_view)
            self.view_stack.setCurrentWidget(self.method_view)
        except Exception:
            pass

        # Restore dataset selection and rerun (or show cached result if we have one).
        if self.method_view and history_item.dataset_names:
            ds_widget = self.method_view.dataset_widget
            try:
                if getattr(ds_widget, "mode", None) == "single":
                    if ds_widget.simple_input and len(history_item.dataset_names) == 1:
                        idx = ds_widget.simple_input.findText(history_item.dataset_names[0])
                        if idx >= 0:
                            ds_widget.simple_input.setCurrentIndex(idx)
                else:
                    # Multi mode
                    if history_item.group_labels and getattr(ds_widget, "radio_group", None):
                        # Enable grouped mode
                        btn = ds_widget.radio_group.button(1)
                        if btn is not None:
                            btn.setChecked(True)

                        # Convert dataset->group map into group->datasets
                        groups: Dict[str, List[str]] = {}
                        for ds_name, gname in history_item.group_labels.items():
                            groups.setdefault(str(gname), []).append(str(ds_name))

                        if getattr(ds_widget, "group_manager", None) is not None:
                            ds_widget._suspend_group_persist = True
                            try:
                                ds_widget.group_manager.set_groups(groups)
                            finally:
                                ds_widget._suspend_group_persist = False
                    else:
                        # Simple multi-selection list
                        if ds_widget.simple_input:
                            ds_widget.simple_input.clearSelection()
                            for i in range(ds_widget.simple_input.count()):
                                li = ds_widget.simple_input.item(i)
                                if li and li.text() in history_item.dataset_names:
                                    li.setSelected(True)
            except Exception:
                pass

        if history_item.result:
            populate_results_tabs(
                self.method_view.results_panel,
                history_item.result,
                self.localize,
                MatplotlibWidget,
                history_item.method_name,
            )
            self.current_result = history_item.result

            # Cached result restore: hide loading immediately.
            try:
                if self.method_view and hasattr(self.method_view.results_panel, "hide_loading"):
                    self.method_view.results_panel.hide_loading()
            except Exception:
                pass
        else:
            # Re-run analysis to regenerate results.
            try:
                selection = self.method_view.dataset_widget.get_selection()
                self._run_analysis(
                    history_item.category,
                    history_item.method_key,
                    selection,
                    self.method_view.params_widget.dynamic_widget,
                    history_restore_id=history_item.storage_id,
                )
            except Exception as e:
                create_logs(
                    "AnalysisPage",
                    "analysis_history",
                    f"Failed to rerun analysis from history: {e}",
                    status="warning",
                )

    def _cleanup_history_item(self, item: AnalysisHistoryItem):
        """
        P0-2: Clean up matplotlib figures from a history item to free memory.

        This prevents memory leaks by closing figures when history items
        are removed. Called when history exceeds MAX_HISTORY limit.

        Args:
            item: History item to clean up
        """
        if item.result:
            # Close primary and secondary figures
            if item.result.primary_figure:
                try:
                    plt.close(item.result.primary_figure)
                except Exception:
                    pass  # Ignore errors if figure already closed

            if item.result.secondary_figure:
                try:
                    plt.close(item.result.secondary_figure)
                except Exception:
                    pass

            # Close extra PCA figures stored in raw_results
            if item.result.raw_results:
                extra_figure_keys = [
                    "scree_figure",
                    "loadings_figure",
                    "biplot_figure",
                    "cumulative_variance_figure",
                    "distributions_figure",
                ]
                for key in extra_figure_keys:
                    if key in item.result.raw_results:
                        try:
                            plt.close(item.result.raw_results[key])
                        except Exception:
                            pass

    def _clear_history(self):
        """Clear analysis history."""
        reply = QMessageBox.question(
            self,
            self.localize("ANALYSIS_PAGE.clear_history_title"),
            self.localize("ANALYSIS_PAGE.clear_history_confirm"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # P0-2: Clean up all figures before clearing
            for item in self.analysis_history:
                self._cleanup_history_item(item)
            self.analysis_history.clear()
            self._history_by_id = {}
            self._update_history_list()

            # Also clear persisted history for this project.
            try:
                history_dir = self._get_history_storage_dir()
                if history_dir and os.path.isdir(history_dir):
                    for fname in os.listdir(history_dir):
                        if fname.endswith((".meta.json", ".params.pkl", "index.json")):
                            try:
                                os.remove(os.path.join(history_dir, fname))
                            except Exception:
                                continue
            except Exception:
                pass

    def _on_dataset_changed(self):
        """Handle dataset changes (load/remove)."""
        # If in method view, update dataset widget
        if self.method_view and hasattr(self.method_view, "dataset_widget"):
            widget = self.method_view.dataset_widget

            # Handle QComboBox (single-dataset mode)
            if hasattr(widget, "currentText"):
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
            elif hasattr(widget, "selectedItems"):
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

        method_name = ANALYSIS_METHODS[self.current_category][self.current_method_key][
            "name"
        ]
        filename = f"{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        self.export_manager.export_plot_png(
            self.current_result.primary_figure, filename
        )

    def _export_svg(self):
        """Export current plot as SVG."""
        if not self.current_result or not self.current_result.primary_figure:
            return

        method_name = ANALYSIS_METHODS[self.current_category][self.current_method_key][
            "name"
        ]
        filename = f"{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg"

        self.export_manager.export_plot_svg(
            self.current_result.primary_figure, filename
        )

    def _export_csv(self):
        """Export current data table as CSV."""
        if not self.current_result or self.current_result.data_table is None:
            return

        method_name = ANALYSIS_METHODS[self.current_category][self.current_method_key][
            "name"
        ]
        filename = f"{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        self.export_manager.export_data_csv(self.current_result.data_table, filename)
    
    def _export_data_multi_format(self):
        """Export current data table in multiple formats (CSV, Excel, JSON, etc.)."""
        if not self.current_result or self.current_result.data_table is None:
            create_logs(
                "_export_data_multi_format",
                "analysis_page",
                "No data table available for export.",
                "warning",
            )

            QMessageBox.warning(
                self,
                self.localize("ANALYSIS_PAGE.export_error_title"),
                self.localize("ANALYSIS_PAGE.no_data_to_export")
            )
            return
        
        method_name = ANALYSIS_METHODS[self.current_category][self.current_method_key]["name"]
        filename = f"{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.export_manager.export_data_multi_format(self.current_result.data_table, filename)
    
    def export_full_report(self):
        """Export complete analysis report (public method for external calls)."""
        if not self.current_result:
            create_logs(
                "_export_full_report",
                "analysis_page",
                "No analysis results available for report export.",
                "warning",
            )
            
            QMessageBox.warning(
                self,
                self.localize("ANALYSIS_PAGE.warning_title"),
                self.localize("ANALYSIS_PAGE.no_results_to_export"),
            )
            return

        method_name = ANALYSIS_METHODS[self.current_category][self.current_method_key][
            "name"
        ]

        # Get dataset name from widget
        dataset_name = "Unknown"
        if self.method_view and hasattr(self.method_view, "dataset_widget"):
            widget = self.method_view.dataset_widget
            if hasattr(widget, "currentText"):
                dataset_name = widget.currentText()
            elif hasattr(widget, "selectedItems"):
                selected = widget.selectedItems()
                dataset_name = selected[0].text() if selected else "Multiple"

        # Get current parameters
        parameters = {}
        if self.method_view and hasattr(self.method_view, "param_widget"):
            try:
                parameters = self.method_view.param_widget.get_parameters()
            except Exception:
                parameters = {}

        self.export_manager.export_full_report(
            self.current_result, method_name, parameters, dataset_name
        )

    def save_to_project(self):
        """Save current analysis to project folder (public method)."""
        if not self.current_result:
            create_logs(
                "_save_to_project",
                "analysis_page",
                "No analysis results available to save to project.",
                "warning",
            )

            QMessageBox.warning(
                self,
                self.localize("ANALYSIS_PAGE.warning_title"),
                self.localize("ANALYSIS_PAGE.no_results_to_save"),
            )
            return

        method_name = ANALYSIS_METHODS[self.current_category][self.current_method_key][
            "name"
        ]

        # Get dataset name from widget
        dataset_name = "Unknown"
        if self.method_view and hasattr(self.method_view, "dataset_widget"):
            widget = self.method_view.dataset_widget
            if hasattr(widget, "currentText"):
                dataset_name = widget.currentText()
            elif hasattr(widget, "selectedItems"):
                selected = widget.selectedItems()
                dataset_name = selected[0].text() if selected else "Multiple"

        # Get current parameters
        parameters = {}
        if self.method_view and hasattr(self.method_view, "param_widget"):
            try:
                parameters = self.method_view.param_widget.get_parameters()
            except Exception:
                parameters = {}

        self.export_manager.save_to_project(
            self.current_result, method_name, parameters, dataset_name
        )
