import sys
import os
import json

import pandas as pd
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QGroupBox,
    QScrollArea,
    QFrame,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QTabWidget,
    QTextEdit,
    QSizePolicy,
    QAbstractItemView,
    QComboBox,
    QDialog,
    QProgressBar,
)
from PySide6.QtCore import Qt, Signal, QUrl, QSize, QThread
from PySide6.QtGui import QIcon

from functions.data_loader import load_data_from_path, load_metadata_from_json
from configs.configs import create_logs
from utils import LOCALIZE, PROJECT_MANAGER, CONFIGS, RAMAN_DATA
from components.widgets.matplotlib_widget import MatplotlibWidget, plot_spectra
from components.widgets.icons import get_icon_path, load_icon




class BatchImportProgressDialog(QDialog):
    """Progress dialog for batch dataset import operations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(LOCALIZE("DATA_PACKAGE_PAGE.batch_import_progress_title"))
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title_label = QLabel(
            LOCALIZE("DATA_PACKAGE_PAGE.batch_import_progress_message")
        )
        title_label.setStyleSheet("font-weight: 600; font-size: 13px;")
        layout.addWidget(title_label)

        # Current file label
        self.current_file_label = QLabel("")
        self.current_file_label.setStyleSheet("color: #6c757d; font-size: 11px;")
        layout.addWidget(self.current_file_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 11px; color: #28a745;")
        layout.addWidget(self.status_label)

    def set_total(self, total):
        """Set total number of datasets to import."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)

    def update_progress(self, current, total, folder_name, success_count, failed_count):
        """Update progress display."""
        self.progress_bar.setValue(current)
        self.current_file_label.setText(
            f"{LOCALIZE('DATA_PACKAGE_PAGE.processing_folder')}: {folder_name}"
        )
        self.status_label.setText(
            f"{LOCALIZE('DATA_PACKAGE_PAGE.import_status')}: {current}/{total} | "
            f"✓ {success_count} | ✗ {failed_count}"
        )
        QApplication.processEvents()  # Keep UI responsive


class DragDropLabel(QLabel):
    pathDropped = Signal(str)

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setObjectName("dragDropLabel")
        self.setMinimumHeight(80)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("border: 2px dashed #0078d4; background-color: #eaf2f8;")
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("")

    def dropEvent(self, event):
        self.setStyleSheet("")
        if urls := event.mimeData().urls():
            self.pathDropped.emit(urls[0].toLocalFile())


class DatasetItemWidget(QWidget):
    removeRequested = Signal(str)

    def __init__(self, dataset_name: str, parent=None):
        super().__init__(parent)
        self.dataset_name = dataset_name
        self.setObjectName("datasetItemWidget")

        # Main horizontal layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 5, 5, 5)
        main_layout.setSpacing(10)

        # Vertical layout for name and info
        info_vbox = QVBoxLayout()
        info_vbox.setSpacing(2)

        # Dataset name label
        name_label = QLabel(dataset_name)
        name_label.setObjectName("datasetItemLabel")
        name_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        info_vbox.addWidget(name_label)

        # Get dataset info from RAMAN_DATA
        try:
            df = RAMAN_DATA.get(dataset_name)
            if df is not None and not df.empty:
                # Create info text with smaller font
                num_spectra = df.shape[1]
                wavelength_min = df.index.min()
                wavelength_max = df.index.max()
                data_points = df.shape[0]

                info_text = f"{num_spectra} spectra | {wavelength_min:.1f}–{wavelength_max:.1f} cm⁻¹ | {data_points} pts"
                info_label = QLabel(info_text)
                info_label.setObjectName("datasetItemInfo")
                info_label.setStyleSheet("font-size: 10px; color: #7f8c8d;")
                info_vbox.addWidget(info_label)
        except Exception as e:
            # If there's an error getting info, just show the name
            pass

        main_layout.addLayout(info_vbox)
        main_layout.addStretch()

        # Modify remove button to use SVG icon
        remove_button = QPushButton()
        remove_button.setObjectName("removeListItemButton")
        remove_button.setFixedSize(26, 26)
        remove_button.setToolTip(f"Remove '{dataset_name}'")
        icon_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "icons", "trash-xmark.svg"
        )  # Adjust path if needed
        remove_button.setIcon(QIcon(icon_path))
        remove_button.setIconSize(QSize(16, 16))  # Adjust size as needed
        remove_button.clicked.connect(
            lambda: self.removeRequested.emit(self.dataset_name)
        )

        main_layout.addWidget(remove_button)


class DataPackagePage(QWidget):
    showNotification = Signal(str, str)
    datasets_changed = Signal()  # Emitted when datasets are added or removed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.preview_dataframe = None
        self.preview_metadata = {}
        self.pending_datasets = (
            {}
        )  # For multiple folder import: {folder_name: {df, metadata}}
        self.auto_preview_enabled = True  # Auto-preview feature flag
        self._setup_ui()
        self._connect_signals()
        self.load_project_data()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(
            16, 12, 16, 16
        )  # Match preprocessing page margins
        main_layout.setSpacing(16)  # Match preprocessing page spacing
        self._create_left_panel(main_layout)
        self._create_right_panel(main_layout)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 2)

    def _create_left_panel(self, parent_layout):
        left_vbox = QVBoxLayout()
        left_vbox.setSpacing(12)  # Reduced spacing

        importer_group = self._create_importer_group_modern()

        # === LOADED DATASETS GROUP with standardized title ===
        loaded_group = QGroupBox()
        loaded_group.setObjectName("modernLoadedGroup")

        loaded_layout = QVBoxLayout(loaded_group)
        loaded_layout.setContentsMargins(12, 4, 12, 12)
        loaded_layout.setSpacing(10)

        # Standardized title bar
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)

        title_label = QLabel(LOCALIZE("DATA_PACKAGE_PAGE.loaded_datasets_title"))
        title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        # Delete all button (red theme)
        self.delete_all_btn = QPushButton()
        self.delete_all_btn.setObjectName("titleBarButtonRed")
        delete_all_icon = load_icon("delete_all", QSize(14, 14), "#dc3545")
        self.delete_all_btn.setIcon(delete_all_icon)
        self.delete_all_btn.setIconSize(QSize(14, 14))
        self.delete_all_btn.setFixedSize(24, 24)
        self.delete_all_btn.setToolTip(LOCALIZE("DATA_PACKAGE_PAGE.delete_all_tooltip"))
        self.delete_all_btn.setCursor(Qt.PointingHandCursor)
        self.delete_all_btn.setStyleSheet(
            """
            QPushButton#titleBarButtonRed {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 2px;
            }
            QPushButton#titleBarButtonRed:hover {
                background-color: #f8d7da;
                border-color: #dc3545;
            }
            QPushButton#titleBarButtonRed:pressed {
                background-color: #f5c6cb;
            }
        """
        )
        self.delete_all_btn.clicked.connect(self._handle_delete_all_datasets)
        title_layout.addWidget(self.delete_all_btn)

        loaded_layout.addWidget(title_widget)

        self.loaded_data_list = QListWidget()
        self.loaded_data_list.setObjectName("loadedDataListWidget")
        self.loaded_data_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        loaded_layout.addWidget(self.loaded_data_list)

        left_vbox.addWidget(importer_group)
        left_vbox.addWidget(loaded_group, 1)
        parent_layout.addLayout(left_vbox)

        self.loaded_data_list.currentItemChanged.connect(self.display_selected_dataset)

    def _create_importer_group_modern(self) -> QGroupBox:
        """Create modern importer group with standardized title bar."""
        importer_group = QGroupBox()
        importer_group.setObjectName("modernImporterGroup")

        layout = QVBoxLayout(importer_group)
        layout.setContentsMargins(12, 4, 12, 12)
        layout.setSpacing(12)

        # === STANDARDIZED TITLE BAR ===
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)

        title_label = QLabel(LOCALIZE("DATA_PACKAGE_PAGE.importer_title"))
        title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
        title_layout.addWidget(title_label)

        # Hint button
        hint_btn = QPushButton("?")
        hint_btn.setObjectName("hintButton")
        hint_btn.setFixedSize(20, 20)
        hint_btn.setToolTip(LOCALIZE("DATA_PACKAGE_PAGE.importer_hint"))
        hint_btn.setCursor(Qt.PointingHandCursor)
        hint_btn.setStyleSheet(
            """
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
        """
        )
        title_layout.addWidget(hint_btn)
        title_layout.addStretch()

        layout.addWidget(title_widget)

        # === DATA IMPORT SECTION ===
        data_label = QLabel(
            f"<b>{LOCALIZE('DATA_PACKAGE_PAGE.data_source_label')}:</b>"
        )
        layout.addWidget(data_label)

        # Data path with browse button
        data_path_layout = QHBoxLayout()
        data_path_layout.setSpacing(8)
        self.data_path_edit = QLineEdit()
        self.data_path_edit.setReadOnly(True)
        self.data_path_edit.setPlaceholderText(
            LOCALIZE("DATA_PACKAGE_PAGE.data_path_placeholder")
        )

        browse_data_btn = QPushButton()
        browse_data_btn.setObjectName("browseButton")
        browse_data_btn.setIcon(load_icon("focus_horizontal", QSize(16, 16), "#6c757d"))
        browse_data_btn.setIconSize(QSize(16, 16))
        browse_data_btn.setFixedSize(32, 32)
        browse_data_btn.setToolTip(LOCALIZE("DATA_PACKAGE_PAGE.browse_data_button"))
        browse_data_btn.setCursor(Qt.PointingHandCursor)

        data_path_layout.addWidget(self.data_path_edit, 1)
        data_path_layout.addWidget(browse_data_btn)
        layout.addLayout(data_path_layout)

        # Drag-drop hint (smaller)
        drag_hint = QLabel(f"<i>{LOCALIZE('DATA_PACKAGE_PAGE.drag_drop_hint')}</i>")
        drag_hint.setStyleSheet("color: #6c757d; font-size: 10px; padding: 4px;")
        drag_hint.setWordWrap(True)
        layout.addWidget(drag_hint)

        # === DATASET SELECTOR (for batch import, hidden by default) ===
        self.dataset_selector_widget = QWidget()
        selector_layout = QVBoxLayout(self.dataset_selector_widget)
        selector_layout.setContentsMargins(0, 8, 0, 8)
        selector_layout.setSpacing(4)

        selector_label = QLabel(
            f"<b>{LOCALIZE('DATA_PACKAGE_PAGE.dataset_selector_label')}</b>"
        )
        selector_label.setStyleSheet("color: #2c3e50;")
        self.dataset_selector = QComboBox()
        self.dataset_selector.setObjectName("datasetSelector")
        self.dataset_selector.currentIndexChanged.connect(
            self._on_dataset_selector_changed
        )

        selector_layout.addWidget(selector_label)
        selector_layout.addWidget(self.dataset_selector)

        layout.addWidget(self.dataset_selector_widget)
        self.dataset_selector_widget.setVisible(False)  # Hidden by default

        # === METADATA IMPORT SECTION ===
        meta_label = QLabel(
            f"<b>{LOCALIZE('DATA_PACKAGE_PAGE.metadata_source_label')}:</b>"
        )
        layout.addWidget(meta_label)

        # Metadata path with browse button
        meta_path_layout = QHBoxLayout()
        meta_path_layout.setSpacing(8)
        self.meta_path_edit = QLineEdit()
        self.meta_path_edit.setReadOnly(True)
        self.meta_path_edit.setPlaceholderText(
            LOCALIZE("DATA_PACKAGE_PAGE.meta_path_placeholder")
        )

        browse_meta_btn = QPushButton()
        browse_meta_btn.setObjectName("browseButton")
        browse_meta_btn.setIcon(load_icon("focus_horizontal", QSize(16, 16), "#6c757d"))
        browse_meta_btn.setIconSize(QSize(16, 16))
        browse_meta_btn.setFixedSize(32, 32)
        browse_meta_btn.setToolTip(LOCALIZE("DATA_PACKAGE_PAGE.browse_meta_button"))
        browse_meta_btn.setCursor(Qt.PointingHandCursor)

        meta_path_layout.addWidget(self.meta_path_edit, 1)
        meta_path_layout.addWidget(browse_meta_btn)
        layout.addLayout(meta_path_layout)

        # Metadata hint
        meta_hint = QLabel(
            f"<i>{LOCALIZE('DATA_PACKAGE_PAGE.metadata_optional_hint')}</i>"
        )
        meta_hint.setStyleSheet("color: #6c757d; font-size: 10px; padding: 4px;")
        meta_hint.setWordWrap(True)
        layout.addWidget(meta_hint)

        # === ACTION BUTTONS ===
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        self.preview_button = QPushButton(LOCALIZE("DATA_PACKAGE_PAGE.preview_button"))
        self.preview_button.setObjectName("secondaryButton")

        self.add_to_project_button = QPushButton(
            LOCALIZE("DATA_PACKAGE_PAGE.add_to_project_button")
        )
        self.add_to_project_button.setObjectName("ctaButton")
        self.add_to_project_button.setEnabled(False)

        button_layout.addWidget(self.preview_button)
        button_layout.addWidget(self.add_to_project_button)
        layout.addLayout(button_layout)

        # === ENABLE DRAG-DROP ON GROUPBOX ===
        importer_group.setAcceptDrops(True)
        importer_group.dragEnterEvent = self._on_drag_enter
        importer_group.dropEvent = self._on_drop

        # Connect signals
        browse_data_btn.clicked.connect(self.browse_for_data)
        browse_meta_btn.clicked.connect(self.browse_for_metadata)

        return importer_group

    def _on_drag_enter(self, event):
        """Handle drag enter for entire importer group."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def _on_drop(self, event):
        """Handle drop for entire importer group."""
        if urls := event.mimeData().urls():
            path = urls[0].toLocalFile()
            # Detect if it's metadata JSON or data
            if (
                path.lower().endswith(".json")
                and "metadata" in os.path.basename(path).lower()
            ):
                self.meta_path_edit.setText(path)
            else:
                self._set_data_path(path)

    def _create_right_panel(self, parent_layout):
        right_vbox = QVBoxLayout()
        right_vbox.setSpacing(12)

        # Preview section with dataset selector
        preview_group = self._create_preview_group_modern()

        # Metadata editor section with standardized title
        meta_editor_group = self._create_metadata_editor_group()

        right_vbox.addWidget(
            preview_group, 3
        )  # Give much more space to preview (3:1 ratio)
        right_vbox.addWidget(meta_editor_group, 1)  # Less space for metadata
        parent_layout.addLayout(right_vbox)

    def _create_metadata_editor_group(self) -> QGroupBox:
        """Create metadata editor group with standardized title bar."""
        meta_editor_group = QGroupBox()
        meta_editor_group.setObjectName("modernMetadataGroup")

        layout = QVBoxLayout(meta_editor_group)
        layout.setContentsMargins(12, 4, 12, 12)
        layout.setSpacing(10)

        # === STANDARDIZED TITLE BAR ===
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)

        title_label = QLabel(LOCALIZE("DATA_PACKAGE_PAGE.metadata_editor_title"))
        title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        # Edit metadata button (pencil icon)
        self.edit_meta_button = QPushButton()
        self.edit_meta_button.setObjectName("titleBarButton")
        edit_icon = load_icon("edit", QSize(14, 14), "#0078d4")
        self.edit_meta_button.setIcon(edit_icon)
        self.edit_meta_button.setIconSize(QSize(14, 14))
        self.edit_meta_button.setFixedSize(24, 24)
        self.edit_meta_button.setToolTip(LOCALIZE("DATA_PACKAGE_PAGE.edit_meta_button"))
        self.edit_meta_button.setCursor(Qt.PointingHandCursor)
        self.edit_meta_button.setCheckable(True)  # Toggle button
        self.edit_meta_button.setStyleSheet(
            """
            QPushButton#titleBarButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 2px;
            }
            QPushButton#titleBarButton:hover {
                background-color: #e7f3ff;
                border-color: #0078d4;
            }
            QPushButton#titleBarButton:checked {
                background-color: #0078d4;
                border-color: #0078d4;
            }
        """
        )
        self.edit_meta_button.clicked.connect(self._toggle_metadata_editing)
        title_layout.addWidget(self.edit_meta_button)

        # Save metadata button with icon
        self.save_meta_button = QPushButton()
        self.save_meta_button.setObjectName("titleBarButtonGreen")
        save_icon = load_icon("save", QSize(14, 14), "#28a745")
        self.save_meta_button.setIcon(save_icon)
        self.save_meta_button.setIconSize(QSize(14, 14))
        self.save_meta_button.setFixedSize(24, 24)
        self.save_meta_button.setToolTip(LOCALIZE("DATA_PACKAGE_PAGE.save_meta_button"))
        self.save_meta_button.setCursor(Qt.PointingHandCursor)
        self.save_meta_button.setStyleSheet(
            """
            QPushButton#titleBarButtonGreen {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 2px;
            }
            QPushButton#titleBarButtonGreen:hover {
                background-color: #d4edda;
                border-color: #28a745;
            }
            QPushButton#titleBarButtonGreen:pressed {
                background-color: #c3e6cb;
            }
        """
        )
        self.save_meta_button.setVisible(False)  # Hidden by default
        title_layout.addWidget(self.save_meta_button)

        # Export metadata button with icon (orange theme)
        self.export_meta_button = QPushButton()
        self.export_meta_button.setObjectName("titleBarButtonOrange")
        export_icon = load_icon("export_button", QSize(14, 14), "#fd7e14")
        self.export_meta_button.setIcon(export_icon)
        self.export_meta_button.setIconSize(QSize(14, 14))
        self.export_meta_button.setFixedSize(24, 24)
        self.export_meta_button.setToolTip(
            LOCALIZE("DATA_PACKAGE_PAGE.export_meta_button")
        )
        self.export_meta_button.setCursor(Qt.PointingHandCursor)
        self.export_meta_button.setStyleSheet(
            """
            QPushButton#titleBarButtonOrange {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 2px;
            }
            QPushButton#titleBarButtonOrange:hover {
                background-color: #fff3e0;
                border-color: #fd7e14;
            }
            QPushButton#titleBarButtonOrange:pressed {
                background-color: #ffe0b2;
            }
        """
        )
        self.export_meta_button.clicked.connect(self.save_metadata_as_json)
        title_layout.addWidget(self.export_meta_button)

        layout.addWidget(title_widget)

        # Metadata tabs
        self.meta_tabs = QTabWidget()
        self._populate_metadata_fields()
        layout.addWidget(self.meta_tabs)

        self.meta_editor_group = meta_editor_group
        self.current_editing_dataset = None  # Track which dataset is being edited
        return meta_editor_group

    def _create_preview_group_modern(self) -> QGroupBox:
        """Create modern preview group with maximized graph space."""
        preview_group = QGroupBox()
        preview_group.setObjectName("modernPreviewGroup")

        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(8, 4, 8, 8)  # Reduced margins for more space
        preview_layout.setSpacing(8)  # Tighter spacing

        # === STANDARDIZED TITLE BAR ===
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)

        title_label = QLabel(LOCALIZE("DATA_PACKAGE_PAGE.preview_title"))
        title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
        title_layout.addWidget(title_label)

        # Store reference for dynamic updates
        self.preview_title_label = title_label
        self.current_preview_dataset_name = None

        # Info label (for spectrum details) - moved to title bar as subtitle
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(
            "font-size: 9px; color: #6c757d; font-weight: normal;"
        )
        self.info_label.setWordWrap(False)
        title_layout.addWidget(self.info_label)

        title_layout.addStretch()

        # Auto-preview toggle button (eye icon)
        self.auto_preview_btn = QPushButton()
        self.auto_preview_btn.setObjectName("titleBarButton")
        self.auto_preview_btn.setFixedSize(24, 24)
        self.auto_preview_btn.setToolTip(
            LOCALIZE("DATA_PACKAGE_PAGE.toggle_auto_preview_tooltip")
        )
        self.auto_preview_btn.setCursor(Qt.PointingHandCursor)
        self._update_auto_preview_icon()
        self.auto_preview_btn.clicked.connect(self._toggle_auto_preview)
        self.auto_preview_btn.setStyleSheet(
            """
            QPushButton#titleBarButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 2px;
            }
            QPushButton#titleBarButton:hover {
                background-color: #e7f3ff;
                border-color: #0078d4;
            }
            QPushButton#titleBarButton:pressed {
                background-color: #d0e7ff;
            }
        """
        )
        title_layout.addWidget(self.auto_preview_btn)

        preview_layout.addWidget(title_widget)

        # === PLOT WIDGET (maximum space with high stretch factor) ===
        # Dataset selector moved to import section
        self.plot_widget = MatplotlibWidget()
        self.plot_widget.setMinimumHeight(
            350
        )  # Increased minimum height for better visibility
        preview_layout.addWidget(
            self.plot_widget, 10
        )  # High stretch factor for maximum space

        # Info label moved to title bar (no longer overlaying graph)

        return preview_group

    def _update_auto_preview_icon(self):
        """Update auto-preview button icon based on state."""
        if self.auto_preview_enabled:
            icon = load_icon("eye_open", QSize(14, 14), "#0078d4")
            tooltip = LOCALIZE("DATA_PACKAGE_PAGE.auto_preview_enabled")
        else:
            icon = load_icon("eye_close", QSize(14, 14), "#6c757d")
            tooltip = LOCALIZE("DATA_PACKAGE_PAGE.auto_preview_disabled")
        self.auto_preview_btn.setIcon(icon)
        self.auto_preview_btn.setIconSize(QSize(14, 14))
        self.auto_preview_btn.setToolTip(tooltip)

    def _toggle_auto_preview(self):
        """Toggle auto-preview feature."""
        self.auto_preview_enabled = not self.auto_preview_enabled
        self._update_auto_preview_icon()

    def _update_preview_title(self, dataset_name: str = None):
        """Update preview title with current dataset name."""
        base_title = LOCALIZE("DATA_PACKAGE_PAGE.preview_title")
        if dataset_name:
            self.preview_title_label.setText(f"{base_title}: {dataset_name}")
            self.current_preview_dataset_name = dataset_name
        else:
            self.preview_title_label.setText(base_title)
            self.current_preview_dataset_name = None

    def _on_dataset_selector_changed(self, index):
        """Handle dataset selector change for multiple dataset preview."""
        if index < 0 or not self.pending_datasets:
            return

        dataset_name = self.dataset_selector.currentText()
        if dataset_name in self.pending_datasets:
            dataset_info = self.pending_datasets[dataset_name]
            self._update_preview_title(dataset_name)  # Update title with dataset name
            self.update_preview_display(
                dataset_info.get("df"),
                dataset_info.get("metadata", {}),
                is_preview=True,
            )

    def _populate_metadata_fields(self):
        """Populate metadata editor tabs and fields from configuration.

        Critical for metadata display functionality. If this fails silently,
        metadata section will appear empty in the UI.
        """
        self.metadata_widgets = {}
        meta_structure = CONFIGS.get("metadata_structure", {})

        # Defensive logging for debugging frozen exe issues
        if not meta_structure:
            create_logs(
                "DataPackagePage",
                "data_package_page",
                "metadata_structure is empty or missing from CONFIGS!",
                status="error",
            )
            create_logs(
                "DataPackagePage",
                "data_package_page",
                f"CONFIGS keys: {list(CONFIGS.keys())}",
                status="error",
            )
            create_logs(
                "DataPackagePage",
                "data_package_page",
                f"Frozen mode: {getattr(sys, 'frozen', False)}",
                status="error",
            )
            if getattr(sys, "frozen", False):
                create_logs(
                    "DataPackagePage",
                    "data_package_page",
                    f"_MEIPASS path: {getattr(sys, '_MEIPASS', 'NOT_SET')}",
                    status="error",
                )
            return

        create_logs(
            "DataPackagePage",
            "data_package_page",
            f"Loading metadata structure with {len(meta_structure)} tabs",
            status="info",
        )

        for tab_key, fields in meta_structure.items():
            tab_widget = QWidget()
            tab_layout = QGridLayout(tab_widget)
            tab_name = LOCALIZE(f"DATA_PACKAGE_PAGE.metadata_tab_{tab_key}")
            self.meta_tabs.addTab(tab_widget, tab_name)
            self.metadata_widgets[tab_key] = {}
            row = 0
            for field_key, field_info in fields.items():
                field_name = LOCALIZE(f"DATA_PACKAGE_PAGE.metadata_field_{field_key}")
                label = QLabel(f"{field_name}:")
                field_type = field_info.get("type", "LineEdit")
                if field_type == "TextEdit":
                    widget = QTextEdit()
                    widget.setMinimumHeight(80)
                else:
                    widget = QLineEdit()
                widget.setPlaceholderText(field_info.get("placeholder", ""))
                tab_layout.addWidget(label, row, 0)
                tab_layout.addWidget(widget, row, 1)
                self.metadata_widgets[tab_key][field_key] = widget
                row += 1
            tab_layout.setRowStretch(row, 1)

        create_logs(
            "DataPackagePage",
            "data_package_page",
            f"Successfully created {self.meta_tabs.count()} metadata tabs",
            status="info",
        )

    def _connect_signals(self):
        self.preview_button.clicked.connect(self.handle_preview_data)
        self.add_to_project_button.clicked.connect(self.handle_add_to_project)
        self.save_meta_button.clicked.connect(self.save_metadata_for_dataset)

    def _set_data_path(self, path: str):
        """Set data path and trigger auto-preview if enabled."""
        # Check if data is already loaded in preview
        if self.preview_dataframe is not None or self.pending_datasets:
            # Show warning dialog
            warning_dialog = QMessageBox(self)
            warning_dialog.setWindowTitle(
                LOCALIZE("DATA_PACKAGE_PAGE.overwrite_warning_title")
            )
            warning_dialog.setText(LOCALIZE("DATA_PACKAGE_PAGE.overwrite_warning_text"))
            warning_dialog.setIcon(QMessageBox.Icon.Warning)
            warning_dialog.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            warning_dialog.setDefaultButton(QMessageBox.StandardButton.No)

            result = warning_dialog.exec()

            if result == QMessageBox.StandardButton.No:
                return  # User cancelled, don't load new data

        self.data_path_edit.setText(path)

        # Trigger auto-preview if enabled
        if self.auto_preview_enabled and path:
            self.handle_preview_data()

    def load_project_data(self):
        current_selection_name = (
            self.loaded_data_list.currentItem().data(Qt.UserRole)
            if self.loaded_data_list.currentItem()
            else None
        )
        self.loaded_data_list.clear()
        if not RAMAN_DATA:
            placeholder_item = QListWidgetItem(
                LOCALIZE("DATA_PACKAGE_PAGE.no_datasets_loaded")
            )
            placeholder_item.setFlags(placeholder_item.flags() & ~Qt.ItemIsSelectable)
            self.loaded_data_list.addItem(placeholder_item)
            self.loaded_data_list.setEnabled(False)
            self.update_preview_display(None, {})
        else:
            self.loaded_data_list.setEnabled(True)
            for name in sorted(RAMAN_DATA.keys()):
                item_widget = DatasetItemWidget(name)
                item_widget.removeRequested.connect(self.handle_remove_dataset)
                list_item = QListWidgetItem(self.loaded_data_list)
                list_item.setSizeHint(item_widget.sizeHint())
                list_item.setData(Qt.UserRole, name)
                self.loaded_data_list.addItem(list_item)
                self.loaded_data_list.setItemWidget(list_item, item_widget)
            if current_selection_name:
                for i in range(self.loaded_data_list.count()):
                    if (
                        self.loaded_data_list.item(i).data(Qt.UserRole)
                        == current_selection_name
                    ):
                        self.loaded_data_list.setCurrentRow(i)
                        break
            elif self.loaded_data_list.count() > 0:
                self.loaded_data_list.setCurrentRow(0)

    def display_selected_dataset(
        self, current_item: QListWidgetItem, previous_item: QListWidgetItem
    ):
        if not current_item or not self.loaded_data_list.isEnabled():
            self.update_preview_display(None, {})
            self._update_preview_title(None)  # Clear title when no selection
            return
        dataset_name = current_item.data(Qt.UserRole)
        if not dataset_name:
            return
        df = RAMAN_DATA.get(dataset_name)
        try:
            metadata = PROJECT_MANAGER.get_dataframe_metadata(dataset_name)
            metadata = metadata if isinstance(metadata, dict) else {}
        except Exception:
            metadata = {}
        # Update preview title with selected dataset name
        self._update_preview_title(dataset_name)
        self.update_preview_display(df, metadata, is_preview=False)

    def browse_for_data(self):
        """Browse for data file or folder with user choice dialog."""
        # First, ask user what they want to select
        choice_dialog = QMessageBox(self)
        choice_dialog.setWindowTitle(LOCALIZE("DATA_PACKAGE_PAGE.browse_choice_title"))
        choice_dialog.setText(LOCALIZE("DATA_PACKAGE_PAGE.browse_choice_text"))
        choice_dialog.setIcon(QMessageBox.Icon.Question)

        # Create custom buttons
        files_button = choice_dialog.addButton(
            LOCALIZE("DATA_PACKAGE_PAGE.browse_choice_files"),
            QMessageBox.ButtonRole.AcceptRole,
        )
        folder_button = choice_dialog.addButton(
            LOCALIZE("DATA_PACKAGE_PAGE.browse_choice_folder"),
            QMessageBox.ButtonRole.AcceptRole,
        )
        cancel_button = choice_dialog.addButton(QMessageBox.StandardButton.Cancel)

        choice_dialog.setDefaultButton(folder_button)
        choice_dialog.exec()

        clicked_button = choice_dialog.clickedButton()

        # User cancelled
        if clicked_button == cancel_button:
            return

        # Open appropriate file dialog based on choice
        if clicked_button == files_button:
            # Select multiple files
            paths, _ = QFileDialog.getOpenFileNames(
                self,
                LOCALIZE("DATA_PACKAGE_PAGE.browse_files_dialog_title"),
                "",
                "Data Files (*.csv *.txt *.asc *.ascii *.pkl);;All Files (*.*)",
            )
            if paths:
                # For multiple files, use the first one or parent directory
                if len(paths) == 1:
                    self._set_data_path(paths[0])
                else:
                    # Multiple files - use common directory
                    common_dir = os.path.dirname(paths[0])
                    self._set_data_path(common_dir)

        elif clicked_button == folder_button:
            # Select folder
            folder_path = QFileDialog.getExistingDirectory(
                self,
                LOCALIZE("DATA_PACKAGE_PAGE.browse_folder_dialog_title"),
                "",
                QFileDialog.Option.ShowDirsOnly,
            )
            if folder_path:
                self._set_data_path(folder_path)

    def browse_for_metadata(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            LOCALIZE("DATA_PACKAGE_PAGE.browse_meta_dialog_title"),
            "",
            f"JSON Files (*.json)",
        )
        if path:
            self.meta_path_edit.setText(path)

    def handle_preview_data(self):
        """Handle data preview with support for multiple folder import."""
        data_path = self.data_path_edit.text()
        if not data_path:
            self.showNotification.emit(
                LOCALIZE("NOTIFICATIONS.data_source_missing"), "error"
            )
            return

        # Check if path is a parent folder containing multiple dataset folders
        if os.path.isdir(data_path):
            subfolders = [
                f
                for f in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, f))
            ]

            # Check if subfolders contain data files (batch import scenario)
            if subfolders and self._check_if_batch_import(data_path, subfolders):
                self._handle_batch_import(data_path, subfolders)
                return

        # Single file/folder import (original behavior)
        self._handle_single_import(data_path)

    def _check_if_batch_import(self, parent_path: str, subfolders: list) -> bool:
        """Check if this is a batch import scenario (parent folder with dataset folders)."""
        # Check first few subfolders for data files
        check_count = min(3, len(subfolders))
        folders_with_data = 0

        for folder in subfolders[:check_count]:
            folder_path = os.path.join(parent_path, folder)
            # Check for supported data files
            has_data = any(
                any(f.endswith(ext) for ext in [".txt", ".asc", ".csv", ".pkl"])
                for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
            )
            if has_data:
                folders_with_data += 1

        # If majority of checked folders have data files, treat as batch import
        return folders_with_data >= check_count * 0.5

    def _handle_batch_import(self, parent_path: str, subfolders: list):
        """Handle batch import of multiple datasets from subfolders with progress dialog."""
        self.showNotification.emit(
            LOCALIZE(
                "DATA_PACKAGE_PAGE.multiple_datasets_detected", count=len(subfolders)
            ),
            "info",
        )

        # Create and show progress dialog
        progress_dialog = BatchImportProgressDialog(self)
        progress_dialog.set_total(len(subfolders))
        progress_dialog.show()

        self.pending_datasets = {}
        success_count = 0
        failed_count = 0
        current = 0

        for folder_name in subfolders:
            current += 1
            folder_path = os.path.join(parent_path, folder_name)

            # Update progress
            progress_dialog.update_progress(
                current, len(subfolders), folder_name, success_count, failed_count
            )

            try:
                # Load data from subfolder
                df = load_data_from_path(folder_path)
                if isinstance(df, str):
                    failed_count += 1
                    continue

                # Check for metadata.json in subfolder
                metadata = {}
                metadata_path = os.path.join(folder_path, "metadata.json")
                if os.path.exists(metadata_path):
                    meta = load_metadata_from_json(metadata_path)
                    if not isinstance(meta, str):
                        metadata = meta

                # Store in pending datasets
                self.pending_datasets[folder_name] = {
                    "df": df,
                    "metadata": metadata,
                    "path": folder_path,
                }
                success_count += 1

            except Exception as e:
                failed_count += 1
                continue

        # Close progress dialog
        progress_dialog.close()

        if success_count > 0:
            # Populate dataset selector
            self.dataset_selector.clear()
            self.dataset_selector.addItems(sorted(self.pending_datasets.keys()))
            self.dataset_selector_widget.setVisible(True)

            # Show first dataset preview
            if self.dataset_selector.count() > 0:
                self.dataset_selector.setCurrentIndex(0)
                first_dataset = self.dataset_selector.currentText()
                dataset_info = self.pending_datasets[first_dataset]
                self.preview_dataframe = dataset_info["df"]
                self.preview_metadata = dataset_info.get("metadata", {})

                # Update preview title with first dataset name
                self._update_preview_title(first_dataset)

                self.update_preview_display(
                    self.preview_dataframe, self.preview_metadata, is_preview=True
                )

            self.add_to_project_button.setEnabled(True)

            # Show notification
            if failed_count > 0:
                self.showNotification.emit(
                    LOCALIZE(
                        "DATA_PACKAGE_PAGE.batch_import_partial",
                        success=success_count,
                        total=len(subfolders),
                        failed=failed_count,
                    ),
                    "warning",
                )
            else:
                self.showNotification.emit(
                    LOCALIZE(
                        "DATA_PACKAGE_PAGE.batch_import_info", count=success_count
                    ),
                    "success",
                )
        else:
            self.showNotification.emit(
                LOCALIZE(
                    "NOTIFICATIONS.data_load_error",
                    error="No valid data found in subfolders",
                ),
                "error",
            )
            self.add_to_project_button.setEnabled(False)

    def _handle_single_import(self, data_path: str):
        """Handle single file/folder import (original behavior)."""
        df = load_data_from_path(data_path)

        if isinstance(df, str):
            self.showNotification.emit(
                LOCALIZE("NOTIFICATIONS.data_load_error", error=df), "error"
            )
            self.preview_dataframe = None
            self.add_to_project_button.setEnabled(False)
            self._update_preview_title(None)  # Clear title
        else:
            self.preview_dataframe = df
            self.showNotification.emit(
                LOCALIZE("NOTIFICATIONS.data_load_success"), "success"
            )
            self.add_to_project_button.setEnabled(True)

            # Hide dataset selector for single import
            self.dataset_selector_widget.setVisible(False)
            self.pending_datasets = {}

            # Update preview title with filename
            if os.path.isdir(data_path):
                preview_name = os.path.basename(data_path)
            else:
                preview_name, _ = os.path.splitext(os.path.basename(data_path))
            self._update_preview_title(preview_name)  # Remove "Preview:" prefix

        # Handle metadata
        meta_path = self.meta_path_edit.text()
        if meta_path:
            meta = load_metadata_from_json(meta_path)
            if isinstance(meta, str):
                self.showNotification.emit(
                    LOCALIZE("NOTIFICATIONS.meta_load_error", error=meta), "error"
                )
                self.preview_metadata = {}
            else:
                self.preview_metadata = meta
                self.showNotification.emit(
                    LOCALIZE("DATA_PACKAGE_PAGE.metadata_autofilled"), "info"
                )
        else:
            # Check for metadata.json in same folder as data
            if os.path.isdir(data_path):
                auto_meta_path = os.path.join(data_path, "metadata.json")
                if os.path.exists(auto_meta_path):
                    meta = load_metadata_from_json(auto_meta_path)
                    if not isinstance(meta, str):
                        self.preview_metadata = meta
                        self.meta_path_edit.setText(auto_meta_path)
                        self.showNotification.emit(
                            LOCALIZE("DATA_PACKAGE_PAGE.metadata_autofilled"), "info"
                        )
                    else:
                        self.preview_metadata = {}
                else:
                    self.preview_metadata = {}
            else:
                self.preview_metadata = {}

        self.update_preview_display(
            self.preview_dataframe, self.preview_metadata, is_preview=True
        )

    def handle_add_to_project(self):
        """Handle adding dataset(s) to project with support for batch import."""
        # Check if this is a batch import
        if self.pending_datasets:
            self._handle_batch_add_to_project()
        else:
            self._handle_single_add_to_project()

    def _handle_batch_add_to_project(self):
        """Handle adding multiple datasets to project."""
        success_count = 0
        failed_count = 0

        for dataset_name, dataset_info in self.pending_datasets.items():
            df = dataset_info.get("df")
            metadata = dataset_info.get("metadata", {})

            if df is None:
                failed_count += 1
                continue

            # Check if dataset name already exists
            if dataset_name in RAMAN_DATA:
                # Add suffix to avoid conflict
                base_name = dataset_name
                counter = 1
                while f"{base_name}_{counter}" in RAMAN_DATA:
                    counter += 1
                dataset_name = f"{base_name}_{counter}"

            # Add to project
            success = PROJECT_MANAGER.add_dataframe_to_project(
                dataset_name, df, metadata
            )
            if success:
                success_count += 1
            else:
                failed_count += 1

        if success_count > 0:
            self.showNotification.emit(
                LOCALIZE("DATA_PACKAGE_PAGE.batch_import_success", count=success_count),
                "success",
            )
            self.load_project_data()
            self.clear_importer_fields()
            self.datasets_changed.emit()  # Notify workspace of dataset changes
        else:
            self.showNotification.emit(
                LOCALIZE("NOTIFICATIONS.dataset_add_error"), "error"
            )

    def _handle_single_add_to_project(self):
        """Handle adding single dataset to project with name prompt."""
        if self.preview_dataframe is None:
            self.showNotification.emit(
                LOCALIZE("NOTIFICATIONS.no_data_to_add"), "error"
            )
            return

        # Get suggested name from data path
        suggested_name = ""
        data_path = self.data_path_edit.text().strip()
        if data_path:
            base_name = os.path.basename(data_path)
            if os.path.isdir(data_path):
                suggested_name = base_name
            else:
                suggested_name, _ = os.path.splitext(base_name)
            # Clean up the name
            suggested_name = suggested_name.replace("_", " ").replace("-", " ").title()

        # Prompt user for dataset name
        from PySide6.QtWidgets import QInputDialog

        dataset_name, ok = QInputDialog.getText(
            self,
            LOCALIZE("DATA_PACKAGE_PAGE.dataset_name_dialog_title"),
            LOCALIZE("DATA_PACKAGE_PAGE.dataset_name_dialog_message"),
            text=suggested_name,
        )

        if not ok or not dataset_name.strip():
            return  # User cancelled or entered empty name

        dataset_name = dataset_name.strip()

        # Check if name already exists
        if dataset_name in RAMAN_DATA:
            self.showNotification.emit(
                LOCALIZE("NOTIFICATIONS.dataset_name_exists", name=dataset_name),
                "error",
            )
            return

        # Get metadata from editor if checked
        if self.meta_editor_group.isChecked():
            self.preview_metadata = self._get_metadata_from_editor()

        success = PROJECT_MANAGER.add_dataframe_to_project(
            dataset_name, self.preview_dataframe, self.preview_metadata
        )
        if success:
            self.showNotification.emit(
                LOCALIZE("NOTIFICATIONS.dataset_add_success", name=dataset_name),
                "success",
            )
            self.load_project_data()
            self.clear_importer_fields()
            self.datasets_changed.emit()  # Notify workspace of dataset change
        else:
            self.showNotification.emit(
                LOCALIZE("NOTIFICATIONS.dataset_add_error"), "error"
            )

    def handle_remove_dataset(self, name: str):
        reply = QMessageBox.question(
            self,
            LOCALIZE("DATA_PACKAGE_PAGE.remove_dataset_confirm_title"),
            LOCALIZE("DATA_PACKAGE_PAGE.remove_dataset_confirm_text", name=name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            if PROJECT_MANAGER.remove_dataframe_from_project(name):
                self.showNotification.emit(
                    LOCALIZE("NOTIFICATIONS.dataset_remove_success", name=name),
                    "success",
                )
                self.load_project_data()
                self.datasets_changed.emit()  # Notify workspace of dataset change
            else:
                self.showNotification.emit(
                    LOCALIZE("NOTIFICATIONS.dataset_remove_error", name=name), "error"
                )

    def _handle_delete_all_datasets(self):
        """Handle deleting all datasets from project with confirmation and progress dialog."""
        if not RAMAN_DATA:
            self.showNotification.emit(
                LOCALIZE("DATA_PACKAGE_PAGE.no_datasets_to_delete"), "info"
            )
            return

        dataset_count = len(RAMAN_DATA)
        reply = QMessageBox.question(
            self,
            LOCALIZE("DATA_PACKAGE_PAGE.delete_all_confirm_title"),
            LOCALIZE("DATA_PACKAGE_PAGE.delete_all_confirm_text", count=dataset_count),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Get all dataset names (copy to avoid modification during iteration)
            dataset_names = list(RAMAN_DATA.keys())

            # Create progress dialog
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle(
                LOCALIZE("DATA_PACKAGE_PAGE.delete_progress_title")
            )
            progress_dialog.setModal(True)
            progress_dialog.setMinimumWidth(500)
            progress_dialog.setWindowFlags(
                progress_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint
            )

            layout = QVBoxLayout(progress_dialog)
            layout.setSpacing(12)

            # Title
            title_label = QLabel(LOCALIZE("DATA_PACKAGE_PAGE.delete_progress_message"))
            title_label.setStyleSheet("font-weight: 600; font-size: 13px;")
            layout.addWidget(title_label)

            # Current dataset label
            current_label = QLabel("")
            current_label.setStyleSheet("color: #6c757d; font-size: 11px;")
            layout.addWidget(current_label)

            # Progress bar
            progress_bar = QProgressBar()
            progress_bar.setMinimum(0)
            progress_bar.setMaximum(dataset_count)
            progress_bar.setTextVisible(True)
            layout.addWidget(progress_bar)

            # Status label (✓ X | ✗ Y format)
            status_label = QLabel("✓ 0 | ✗ 0")
            status_label.setStyleSheet("font-size: 12px; color: #2c3e50;")
            layout.addWidget(status_label)

            # Cancel button
            cancel_btn = QPushButton(LOCALIZE("COMMON.cancel"))
            cancel_btn.clicked.connect(progress_dialog.reject)
            cancel_layout = QHBoxLayout()
            cancel_layout.addStretch()
            cancel_layout.addWidget(cancel_btn)
            layout.addLayout(cancel_layout)

            # Show dialog
            progress_dialog.show()
            QApplication.processEvents()

            # Delete datasets with progress updates
            success_count = 0
            failed_count = 0
            cancelled = False

            for i, name in enumerate(dataset_names, 1):
                # Check if cancelled
                if not progress_dialog.isVisible():
                    cancelled = True
                    break

                # Update current dataset
                current_label.setText(
                    f"{LOCALIZE('DATA_PACKAGE_PAGE.deleting_dataset')}: {name}"
                )
                progress_bar.setValue(i)
                status_label.setText(f"✓ {success_count} | ✗ {failed_count}")
                QApplication.processEvents()

                # Delete dataset
                if PROJECT_MANAGER.remove_dataframe_from_project(name):
                    success_count += 1
                else:
                    failed_count += 1

            # Close progress dialog
            progress_dialog.close()

            # Show result notification
            if cancelled:
                self.showNotification.emit(
                    LOCALIZE(
                        "DATA_PACKAGE_PAGE.delete_cancelled",
                        deleted=success_count,
                        remaining=dataset_count - success_count,
                    ),
                    "warning",
                )
            elif success_count > 0:
                self.showNotification.emit(
                    LOCALIZE(
                        "DATA_PACKAGE_PAGE.delete_all_success", count=success_count
                    ),
                    "success",
                )
                self.load_project_data()
                self.datasets_changed.emit()  # Notify workspace of dataset changes
            else:
                self.showNotification.emit(
                    LOCALIZE("DATA_PACKAGE_PAGE.delete_all_error"), "error"
                )

    def update_preview_display(
        self, df: pd.DataFrame, metadata: dict, is_preview: bool = True
    ):
        self.plot_widget.update_plot(plot_spectra(df))
        if df is not None and not df.empty:
            info_text = (
                f"<b>{LOCALIZE('DATA_PACKAGE_PAGE.info_num_spectra')}:</b> {df.shape[1]} | "
                f"<b>{LOCALIZE('DATA_PACKAGE_PAGE.info_wavenumber_range')}:</b> {df.index.min():.2f} - {df.index.max():.2f} cm⁻¹ | "
                f"<b>{LOCALIZE('DATA_PACKAGE_PAGE.info_data_points')}:</b> {df.shape[0]}"
            )
            self.info_label.setText(info_text)
        else:
            self.info_label.setText(LOCALIZE("DATA_PACKAGE_PAGE.no_data_preview"))

        # Display metadata
        self._set_metadata_in_editor(metadata)

        # For loaded datasets, enable viewing but not editing by default
        if not is_preview:
            self._set_metadata_read_only(True)
            self.edit_meta_button.setChecked(False)
            self.edit_meta_button.setVisible(True)
            self.save_meta_button.setVisible(False)
        else:
            # For previews, enable editing
            self._set_metadata_read_only(False)
            self.edit_meta_button.setVisible(False)
            self.save_meta_button.setVisible(False)

    def _get_metadata_from_editor(self) -> dict:
        metadata = {}
        for tab_key, fields in self.metadata_widgets.items():
            metadata[tab_key] = {}
            for field_key, widget in fields.items():
                metadata[tab_key][field_key] = (
                    widget.toPlainText()
                    if isinstance(widget, QTextEdit)
                    else widget.text()
                )
        return metadata

    def _set_metadata_in_editor(self, metadata: dict):
        for tab_key, fields in self.metadata_widgets.items():
            for field_key, widget in fields.items():
                value = metadata.get(tab_key, {}).get(field_key, "")
                if isinstance(widget, QTextEdit):
                    widget.setPlainText(str(value))
                else:
                    widget.setText(str(value))

    def _set_metadata_read_only(self, read_only: bool):
        """Set all metadata fields to read-only or editable."""
        for tab in self.metadata_widgets.values():
            for widget in tab.values():
                widget.setReadOnly(read_only)

    def _toggle_metadata_editing(self):
        """Toggle metadata editing mode."""
        is_editing = self.edit_meta_button.isChecked()
        self._set_metadata_read_only(not is_editing)
        self.save_meta_button.setVisible(is_editing)

        # Update button icon color based on state
        if is_editing:
            edit_icon = load_icon("edit", QSize(14, 14), "#ffffff")
            self.edit_meta_button.setToolTip(
                LOCALIZE("DATA_PACKAGE_PAGE.view_mode_button")
            )
        else:
            edit_icon = load_icon("edit", QSize(14, 14), "#0078d4")
            self.edit_meta_button.setToolTip(
                LOCALIZE("DATA_PACKAGE_PAGE.edit_meta_button")
            )
        self.edit_meta_button.setIcon(edit_icon)

    def save_metadata_for_dataset(self):
        """Save metadata for the currently selected dataset."""
        # Get the currently selected dataset
        current_item = self.loaded_data_list.currentItem()
        if not current_item:
            self.showNotification.emit(
                LOCALIZE("DATA_PACKAGE_PAGE.no_dataset_selected"), "error"
            )
            return

        dataset_name = current_item.data(Qt.UserRole)
        if not dataset_name:
            return

        # Get metadata from editor
        metadata = self._get_metadata_from_editor()

        # Update metadata in PROJECT_MANAGER
        if PROJECT_MANAGER.update_dataframe_metadata(dataset_name, metadata):
            self.showNotification.emit(
                LOCALIZE("DATA_PACKAGE_PAGE.metadata_save_success", name=dataset_name),
                "success",
            )
            # Exit edit mode
            self.edit_meta_button.setChecked(False)
            self._toggle_metadata_editing()
        else:
            self.showNotification.emit(
                LOCALIZE("DATA_PACKAGE_PAGE.metadata_save_error"), "error"
            )

    def toggle_metadata_editing(self, checked, read_only=False):
        """Legacy method for compatibility."""
        pass  # No longer needed, replaced by _toggle_metadata_editing

    def save_metadata_as_json(self):
        """Export metadata to JSON file (legacy method, kept for compatibility)."""
        # Get the currently selected dataset
        current_item = self.loaded_data_list.currentItem()
        if not current_item:
            self.showNotification.emit(
                LOCALIZE("DATA_PACKAGE_PAGE.no_dataset_selected"), "error"
            )
            return

        dataset_name = current_item.data(Qt.UserRole)
        if not dataset_name:
            return

        data_dir = PROJECT_MANAGER._get_project_data_dir()
        if not data_dir:
            self.showNotification.emit(
                LOCALIZE("NOTIFICATIONS.no_project_loaded_for_save"), "error"
            )
            return

        from components.widgets import get_export_options
        from pathlib import Path

        default_filename = f"{dataset_name.replace(' ', '_').lower()}_metadata.json"
        default_dir = str(data_dir)
        default_base = str(Path(default_filename).stem)

        opts = get_export_options(
            self,
            title=LOCALIZE("DATA_PACKAGE_PAGE.save_meta_dialog_title"),
            formats=[("json", "JSON (.json)")],
            default_directory=default_dir,
            default_filename=default_base,
            show_filename=True,
            show_format=False,
            # Labels
            format_label="Export Format:",
            location_label="Save Location:",
            filename_label="Filename:",
            browse_button_text="Browse...",
            select_location_title="Select Location",
            show_metadata_checkbox=False,
        )
        if opts is None:
            return

        base = str(opts.filename or default_base).strip() or default_base
        if not base.lower().endswith(".json"):
            base = f"{base}.json"
        path = str(Path(str(opts.directory)) / base)

        manual_meta = self._get_metadata_from_editor()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(manual_meta, f, indent=4, ensure_ascii=False)
            self.showNotification.emit(
                LOCALIZE("NOTIFICATIONS.meta_save_success"), "success"
            )
        except Exception as e:
            self.showNotification.emit(
                LOCALIZE("NOTIFICATIONS.meta_save_error", error=str(e)), "error"
            )

    def clear_importer_fields(self):
        """Clear all importer fields and reset state."""
        self.data_path_edit.clear()
        self.meta_path_edit.clear()
        self.preview_dataframe = None
        self.preview_metadata = {}
        self.pending_datasets = {}
        self.dataset_selector.clear()
        self.dataset_selector_widget.setVisible(False)
        self.add_to_project_button.setEnabled(False)
        self._set_metadata_in_editor({})
        self.meta_editor_group.setChecked(False)
        self._update_preview_title(None)  # Reset preview title
