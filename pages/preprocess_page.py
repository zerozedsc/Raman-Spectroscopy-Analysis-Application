import sys
import os

# Add components to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from components.widgets import *
from .preprocess_page_utils import *
from configs.configs import create_logs

class PreprocessPage(QWidget):
    """Enhanced preprocessing page with dynamic pipeline building and comprehensive parameter controls."""

    showNotification = Signal(str, str)
    datasets_changed = Signal()  # Emitted when preprocessing creates new datasets

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("preprocessPage")
        self.processing_thread = None
        self.pipeline_steps: List[PipelineStep] = []
        self.current_step_widget = None

        # Separate processing queue management
        self.separate_processing_queue = []
        self.separate_processing_count = 0
        self.separate_processing_total = 0
        self.current_separate_task = None  # Current task being processed

        # Real-time preview attributes
        self.preview_data = None  # Current dataset for preview
        self.original_data = None  # Original unprocessed data
        self.processed_data = None  # Current processed data
        self.preview_cache = {}  # Cache for processed steps
        self.preview_timer = QTimer()  # Debounce timer for parameter changes
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._update_preview)
        self.preview_enabled = True  # Toggle for preview functionality

        # Track dataset selection for pipeline transfer logic
        self._last_selected_was_preprocessed = False

        # Global pipeline memory to persist steps across dataset switches
        self._global_pipeline_memory: List[PipelineStep] = (
            []
        )  # Persistent pipeline steps
        self._current_dataset_name = None  # Track current dataset for state management

        self._setup_ui()
        self._connect_signals()

        # Auto-refresh data when page is shown
        QTimer.singleShot(100, self.load_project_data)

    def update_localized_text(self):
        """Update all UI text with current localization."""
        # This method should be called when language changes
        # Currently, we need to refresh the entire UI or store references to all labels
        # For now, we'll implement a simple version that updates key components
        pass  # Placeholder for future implementation

    def _setup_ui(self):
        """Setup the main UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 8, 16, 16)  # Reduced top margin to 8
        main_layout.setSpacing(12)  # Reduced spacing to 12

        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(2)

        # Left panel - Controls
        left_panel = self._create_left_panel()
        main_splitter.addWidget(left_panel)

        # Right panel - Parameters and Visualization
        right_panel = self._create_right_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setSizes([400, 600])
        main_splitter.setCollapsible(0, False)
        main_splitter.setCollapsible(1, False)

        main_layout.addWidget(main_splitter)

    def _create_left_panel(self) -> QWidget:
        """Create the left control panel."""
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_panel.setMaximumWidth(450)

        layout = QVBoxLayout(left_panel)
        layout.setContentsMargins(12, 4, 12, 12)  # Reduced top margin to 4
        layout.setSpacing(10)  # Reduced spacing to 10

        # Input datasets section
        input_group = self._create_input_datasets_group()
        layout.addWidget(input_group)

        # Pipeline building section
        pipeline_group = self._create_pipeline_building_group()
        layout.addWidget(pipeline_group)

        # Output configuration section
        output_group = self._create_output_configuration_group()
        layout.addWidget(output_group)

        layout.addStretch()
        return left_panel

    def _create_pipeline_building_group(self) -> QGroupBox:
        """Create modern pipeline building group with enhanced medical theme and optimized layout."""
        pipeline_group = QGroupBox()
        pipeline_group.setObjectName("modernPipelineGroup")

        # Create custom title widget to match Input Datasets section
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)

        title_label = QLabel(LOCALIZE("PREPROCESS.pipeline_building_title"))
        title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
        title_layout.addWidget(title_label)

        # Add hint button for pipeline building
        pipeline_hint_btn = QPushButton("?")
        pipeline_hint_btn.setObjectName("hintButton")
        pipeline_hint_btn.setFixedSize(20, 20)
        pipeline_hint_btn.setToolTip(LOCALIZE("PREPROCESS.pipeline_building_hint"))
        pipeline_hint_btn.setCursor(Qt.PointingHandCursor)
        pipeline_hint_btn.setStyleSheet(
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
        title_layout.addWidget(pipeline_hint_btn)

        title_layout.addStretch()

        # Import pipeline button - small icon button matching input dataset style
        import_btn = QPushButton()
        import_btn.setObjectName("titleBarButtonGreen")
        import_icon = load_icon("load_project", QSize(14, 14), "#28a745")
        import_btn.setIcon(import_icon)
        import_btn.setIconSize(QSize(14, 14))
        import_btn.setFixedSize(24, 24)
        import_btn.setToolTip(LOCALIZE("PREPROCESS.import_pipeline_tooltip"))
        import_btn.setCursor(Qt.PointingHandCursor)
        import_btn.setStyleSheet(
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
        import_btn.clicked.connect(self.import_pipeline)
        title_layout.addWidget(import_btn)

        # Export pipeline button - small icon button matching input dataset style
        export_btn = QPushButton()
        export_btn.setObjectName("titleBarButton")
        export_icon = load_icon("export", QSize(14, 14), "#0078d4")
        export_btn.setIcon(export_icon)
        export_btn.setIconSize(QSize(14, 14))
        export_btn.setFixedSize(24, 24)
        export_btn.setToolTip(LOCALIZE("PREPROCESS.export_pipeline_tooltip"))
        export_btn.setCursor(Qt.PointingHandCursor)
        export_btn.setStyleSheet(
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
        export_btn.clicked.connect(self.export_pipeline)
        title_layout.addWidget(export_btn)

        # Apply stylesheet
        from configs.style.stylesheets import PREPROCESS_PAGE_STYLES

        if "modern_pipeline_group" in PREPROCESS_PAGE_STYLES:
            pipeline_group.setStyleSheet(
                PREPROCESS_PAGE_STYLES["modern_pipeline_group"]
            )

        layout = QVBoxLayout(pipeline_group)
        layout.setContentsMargins(12, 4, 12, 12)
        layout.setSpacing(8)

        # Add title widget
        layout.addWidget(title_widget)

        # Compact category and method selection in a single row
        selection_row = QHBoxLayout()
        selection_row.setSpacing(8)

        # Category dropdown (compact)
        cat_container = QVBoxLayout()
        cat_container.setSpacing(4)
        cat_label = QLabel("📂 " + LOCALIZE("PREPROCESS.category"))
        cat_label.setStyleSheet("font-weight: 500; color: #495057; font-size: 11px;")
        cat_container.addWidget(cat_label)

        self.category_combo = QComboBox()
        self.category_combo.setStyleSheet(
            """
            QComboBox {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 5px 8px;
                background: white;
                font-size: 12px;
            }
            QComboBox:hover {
                border-color: #0078d4;
            }
            QComboBox:focus {
                border-color: #0078d4;
                outline: none;
            }
            QComboBox::drop-down {
                border: none;
                width: 18px;
            }
        """
        )
        categories = PREPROCESSING_REGISTRY.get_categories()
        for category in categories:
            display_name = LOCALIZE(f"PREPROCESS.CATEGORY.{category.upper()}")
            self.category_combo.addItem(display_name, category)
        cat_container.addWidget(self.category_combo)
        selection_row.addLayout(cat_container, 1)

        # Method dropdown (compact)
        method_container = QVBoxLayout()
        method_container.setSpacing(4)
        method_label = QLabel("⚙️ " + LOCALIZE("PREPROCESS.method"))
        method_label.setStyleSheet("font-weight: 500; color: #495057; font-size: 11px;")
        method_container.addWidget(method_label)

        self.method_combo = QComboBox()
        self.method_combo.setStyleSheet(
            """
            QComboBox {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 5px 8px;
                background: white;
                font-size: 12px;
            }
            QComboBox:hover {
                border-color: #0078d4;
            }
            QComboBox:focus {
                border-color: #0078d4;
                outline: none;
            }
            QComboBox::drop-down {
                border: none;
                width: 18px;
            }
        """
        )
        method_container.addWidget(self.method_combo)
        selection_row.addLayout(method_container, 1)

        # Add step button (compact, square icon button with SVG)
        add_step_btn = QPushButton()
        add_step_btn.setObjectName("addStepButton")
        add_step_btn.setFixedSize(60, 50)  # Tall enough to match the two-row height
        plus_icon = load_icon("plus", QSize(24, 24), "white")
        add_step_btn.setIcon(plus_icon)
        add_step_btn.setIconSize(QSize(24, 24))
        add_step_btn.setStyleSheet(
            """
            QPushButton#addStepButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 20px;
                font-weight: 600;
            }
            QPushButton#addStepButton:hover {
                background-color: #218838;
            }
            QPushButton#addStepButton:pressed {
                background-color: #1e7e34;
            }
        """
        )
        add_step_btn.setToolTip(LOCALIZE("PREPROCESS.add_step_button"))
        add_step_btn.setCursor(Qt.PointingHandCursor)
        add_step_btn.clicked.connect(self.add_pipeline_step)
        selection_row.addWidget(add_step_btn, 0, Qt.AlignBottom)

        layout.addLayout(selection_row)

        # Pipeline steps list label
        steps_label = QLabel("📋 " + LOCALIZE("PREPROCESS.pipeline_steps_label"))
        steps_label.setStyleSheet(
            "font-weight: 600; font-size: 12px; color: #2c3e50; margin-top: 2px;"
        )
        layout.addWidget(steps_label)

        # Pipeline steps list with modern styling (optimized for non-maximized windows)
        self.pipeline_list = QListWidget()
        self.pipeline_list.setObjectName("modernPipelineList")
        self.pipeline_list.setMinimumHeight(180)
        self.pipeline_list.setMaximumHeight(215)  # Show max 5 steps before scrolling
        self.pipeline_list.setDragDropMode(QListWidget.InternalMove)
        self.pipeline_list.setStyleSheet(
            """
            QListWidget#modernPipelineList {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 4px;
            }
            QListWidget#modernPipelineList::item {
                background-color: transparent;
                border: none;
                padding: 2px;
                margin: 2px 0px;
                border-radius: 0px;
            }
            QListWidget#modernPipelineList::item:selected {
                background-color: transparent;
                border: none;
            }
            QListWidget#modernPipelineList::item:hover {
                background-color: transparent;
                border: none;
            }
        """
        )
        layout.addWidget(self.pipeline_list)

        # Pipeline control buttons with SVG icons (more compact)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(6)

        # Remove button (more compact with SVG icon)
        remove_btn = QPushButton()
        remove_btn.setObjectName("compactButton")
        remove_btn.setFixedHeight(28)
        trash_icon = load_icon("trash_bin", QSize(14, 14), "#dc3545")
        remove_btn.setIcon(trash_icon)
        remove_btn.setIconSize(QSize(14, 14))
        remove_btn.setStyleSheet(
            """
            QPushButton#compactButton {
                background-color: #f8f9fa;
                color: #495057;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: 500;
                font-size: 14px;
            }
            QPushButton#compactButton:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
            }
            QPushButton#compactButton:pressed {
                background-color: #dee2e6;
            }
        """
        )
        remove_btn.setToolTip(LOCALIZE("PREPROCESS.remove_step"))
        remove_btn.setCursor(Qt.PointingHandCursor)
        remove_btn.clicked.connect(self.remove_pipeline_step)

        # Clear button (more compact)
        clear_btn = QPushButton("🧹")
        clear_btn.setObjectName("compactButton")
        clear_btn.setFixedHeight(28)
        clear_btn.setStyleSheet(
            """
            QPushButton#compactButton {
                background-color: #f8f9fa;
                color: #495057;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: 500;
                font-size: 14px;
            }
            QPushButton#compactButton:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
            }
            QPushButton#compactButton:pressed {
                background-color: #dee2e6;
            }
        """
        )
        clear_btn.setToolTip(LOCALIZE("PREPROCESS.clear_pipeline"))
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.clicked.connect(self.clear_pipeline)

        # Toggle all existing steps button (more compact)
        self.toggle_all_btn = QPushButton("🔄")
        self.toggle_all_btn.setObjectName("compactButton")
        self.toggle_all_btn.setFixedHeight(28)
        self.toggle_all_btn.setStyleSheet(
            """
            QPushButton#compactButton {
                background-color: #f8f9fa;
                color: #495057;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: 500;
                font-size: 14px;
            }
            QPushButton#compactButton:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
            }
            QPushButton#compactButton:pressed {
                background-color: #dee2e6;
            }
        """
        )
        self.toggle_all_btn.setToolTip(LOCALIZE("PREPROCESS.toggle_all_existing"))
        self.toggle_all_btn.setVisible(False)  # Initially hidden
        self.toggle_all_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_all_btn.clicked.connect(self.toggle_all_existing_steps)

        # Add buttons
        button_layout.addWidget(remove_btn)
        button_layout.addWidget(clear_btn)
        button_layout.addWidget(self.toggle_all_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        return pipeline_group

    def _create_input_datasets_group(self) -> QGroupBox:
        """Create input datasets selection group with tabs for filtering."""
        input_group = QGroupBox()

        # Create custom title with hint button and action buttons
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)

        title_label = QLabel(LOCALIZE("PREPROCESS.input_datasets_title"))
        title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
        title_layout.addWidget(title_label)

        # Hint button with ? icon
        hint_btn = QPushButton("?")
        hint_btn.setObjectName("hintButton")
        hint_btn.setFixedSize(20, 20)
        hint_btn.setToolTip(
            LOCALIZE("PREPROCESS.multi_select_hint")
            + "\n\n"
            + LOCALIZE("PREPROCESS.multi_dataset_hint")
        )
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

        # Add stretch to push buttons to the right
        title_layout.addStretch()

        # Select All button - tab-aware selection toggle
        self.select_all_btn = QPushButton()
        self.select_all_btn.setObjectName("titleBarButton")
        self.select_all_btn.setFixedSize(24, 24)
        self.select_all_btn.setToolTip(LOCALIZE("PREPROCESS.select_all_tooltip"))
        self.select_all_btn.setCursor(Qt.PointingHandCursor)
        # Use checkmark icon for select all
        checkmark_icon = load_icon("checkmark", QSize(14, 14), "#0078d4")
        self.select_all_btn.setIcon(checkmark_icon)
        self.select_all_btn.setIconSize(QSize(14, 14))
        self.select_all_btn.setStyleSheet(
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
        self.select_all_btn.clicked.connect(self._toggle_select_all_datasets)
        title_layout.addWidget(self.select_all_btn)

        # Refresh button - compact icon in title bar
        refresh_btn = QPushButton()
        refresh_btn.setObjectName("titleBarButton")
        reload_icon = load_icon("reload", QSize(14, 14), "#0078d4")
        refresh_btn.setIcon(reload_icon)
        refresh_btn.setIconSize(QSize(14, 14))
        refresh_btn.setFixedSize(24, 24)
        refresh_btn.setToolTip(LOCALIZE("PREPROCESS.refresh_datasets_tooltip"))
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setStyleSheet(
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
        refresh_btn.clicked.connect(self.load_project_data)
        title_layout.addWidget(refresh_btn)

        # Export button - compact icon in title bar
        export_btn = QPushButton()
        export_btn.setObjectName("titleBarButtonGreen")
        export_icon = load_icon("export", QSize(14, 14), "#28a745")
        export_btn.setIcon(export_icon)
        export_btn.setIconSize(QSize(14, 14))
        export_btn.setFixedSize(24, 24)
        export_btn.setToolTip(LOCALIZE("PREPROCESS.export_button_tooltip"))
        export_btn.setCursor(Qt.PointingHandCursor)
        export_btn.setStyleSheet(
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
            QPushButton#titleBarButtonGreen:disabled {
                opacity: 0.5;
            }
        """
        )
        export_btn.clicked.connect(self.export_dataset)
        title_layout.addWidget(export_btn)

        input_group.setStyleSheet(
            """
            QGroupBox {
                font-weight: 600;
                font-size: 13px;
                color: #2c3e50;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
            }
        """
        )
        input_group.setTitle("")  # Empty title since we use custom widget

        layout = QVBoxLayout(input_group)
        layout.setContentsMargins(
            12, 2, 12, 8
        )  # Reduced top margin from 4 to 2, bottom from 12 to 8
        layout.setSpacing(6)  # Reduced spacing from 8 to 6

        # Add title widget
        layout.addWidget(title_widget)

        # Tab widget for filtering datasets (no separate button row)
        self.dataset_tabs = QTabWidget()
        self.dataset_tabs.setObjectName("datasetTabs")

        # Style tabs for medical theme
        self.dataset_tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 2px solid #dee2e6;
                border-radius: 4px;
                background-color: white;
                top: -1px;
            }
            QTabBar::tab {
                background-color: #f8f9fa;
                color: #495057;
                border: 1px solid #dee2e6;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 12px;
                margin-right: 2px;
                min-width: 60px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #0078d4;
                font-weight: 600;
                border-color: #0078d4;
                border-bottom: 2px solid white;
            }
            QTabBar::tab:hover {
                background-color: #e9ecef;
            }
        """
        )

        # Create three tabs with separate list widgets
        self.dataset_list_all = QListWidget()
        self.dataset_list_raw = QListWidget()
        self.dataset_list_preprocessed = QListWidget()

        # Configure all list widgets - increased height to show minimum 3-4 items
        for list_widget in [
            self.dataset_list_all,
            self.dataset_list_raw,
            self.dataset_list_preprocessed,
        ]:
            list_widget.setObjectName("datasetList")
            list_widget.setSelectionMode(QListWidget.ExtendedSelection)
            list_widget.setMinimumHeight(
                140
            )  # Increased from 100 to 140 (shows ~3-4 items)
            list_widget.setMaximumHeight(
                160
            )  # Increased from 120 to 160 (shows ~4-5 items)

            # Apply custom styling
            from configs.style.stylesheets import PREPROCESS_PAGE_STYLES

            if "dataset_list" in PREPROCESS_PAGE_STYLES:
                list_widget.setStyleSheet(PREPROCESS_PAGE_STYLES["dataset_list"])

            # Connect selection changed signal for all lists
            list_widget.itemSelectionChanged.connect(self._on_dataset_selection_changed)

        # Add tabs
        self.dataset_tabs.addTab(
            self.dataset_list_all, LOCALIZE("PREPROCESS.tab_all_datasets")
        )
        self.dataset_tabs.addTab(
            self.dataset_list_raw, LOCALIZE("PREPROCESS.tab_raw_datasets")
        )
        self.dataset_tabs.addTab(
            self.dataset_list_preprocessed,
            LOCALIZE("PREPROCESS.tab_preprocessed_datasets"),
        )

        # Set default to "All" tab
        self.dataset_tabs.setCurrentIndex(0)

        # Keep reference to the active list (for backward compatibility)
        self.dataset_list = self.dataset_list_all

        # Connect tab change to update active list reference
        self.dataset_tabs.currentChanged.connect(self._on_dataset_tab_changed)

        layout.addWidget(self.dataset_tabs)

        return input_group

    def _on_dataset_tab_changed(self, index: int):
        """Update the active dataset list reference when tab changes."""
        if index == 0:
            self.dataset_list = self.dataset_list_all
        elif index == 1:
            self.dataset_list = self.dataset_list_raw
        elif index == 2:
            self.dataset_list = self.dataset_list_preprocessed

        # Update preview toggle based on tab type
        # Raw datasets (tabs 0 & 1) should have preview ON by default
        # Preprocessed datasets (tab 2) should have preview OFF by default
        if index in [0, 1]:  # All or Raw datasets
            if not self.preview_toggle_btn.isChecked():
                self.preview_toggle_btn.blockSignals(True)
                self.preview_toggle_btn.setChecked(True)
                self.preview_toggle_btn.blockSignals(False)
                self._update_preview_button_state(True)
                self.preview_enabled = True
        elif index == 2:  # Preprocessed datasets
            if self.preview_toggle_btn.isChecked():
                self.preview_toggle_btn.blockSignals(True)
                self.preview_toggle_btn.setChecked(False)
                self.preview_toggle_btn.blockSignals(False)
                self._update_preview_button_state(False)
                self.preview_enabled = False

        # Trigger selection changed event for the newly active tab
        self._on_dataset_selection_changed()

        # Keep Select All button state in sync with the active tab
        try:
            self._update_select_all_button_state()
        except Exception:
            pass

    def _update_select_all_button_state(self):
        """Update Select All button tooltip/icon based on current selection state."""
        if not hasattr(self, "select_all_btn") or self.select_all_btn is None:
            return
        if not hasattr(self, "dataset_list") or self.dataset_list is None:
            return

        total_items = self.dataset_list.count()
        if total_items <= 0:
            self.select_all_btn.setEnabled(False)
            return
        self.select_all_btn.setEnabled(True)

        selected_items = self.dataset_list.selectedItems()
        all_selected = len(selected_items) == total_items

        # Tooltip reflects the action that will happen on click.
        tooltip = (
            LOCALIZE("PREPROCESS.deselect_all_button")
            if all_selected
            else LOCALIZE("PREPROCESS.select_all_button")
        )
        self.select_all_btn.setToolTip(tooltip)

        # Icon reflects action: checkmark = select, minus = deselect
        try:
            if all_selected:
                icon = load_icon("minus", QSize(14, 14), "#dc3545")
            else:
                icon = load_icon("checkmark", QSize(14, 14), "#0078d4")
            self.select_all_btn.setIcon(icon)
        except Exception:
            pass

    def _toggle_select_all_datasets(self):
        """Toggle select all/deselect all for datasets in current tab."""
        current_list = self.dataset_list

        # Check if all items are currently selected
        total_items = current_list.count()
        if total_items == 0:
            return

        selected_items = current_list.selectedItems()
        all_selected = len(selected_items) == total_items

        if all_selected:
            # Deselect all
            current_list.clearSelection()
        else:
            # Select all items in current tab
            current_list.selectAll()

        try:
            self._update_select_all_button_state()
        except Exception:
            pass

    def _on_dataset_selection_changed(self):
        """Handle dataset selection changes across all tabs - syncs selection and updates visualization."""
        selected_items = self.dataset_list.selectedItems()

        try:
            self._update_select_all_button_state()
        except Exception:
            pass

        if not selected_items:
            self.plot_widget.clear_plot()
            self._clear_preprocessing_history()
            self._clear_default_output_name()
            self.original_data = None
            self.processed_data = None
            self.preview_data = None
            return

        # Set default output name based on first selected dataset
        first_dataset_name = self._clean_dataset_name(selected_items[0].text())
        self._set_default_output_name(first_dataset_name)

        # Handle single selection for preprocessing history
        if len(selected_items) == 1:
            dataset_name = self._clean_dataset_name(selected_items[0].text())

            try:
                metadata = PROJECT_MANAGER.get_dataframe_metadata(dataset_name)
                is_preprocessed = metadata and metadata.get("is_preprocessed", False)

                if is_preprocessed:
                    # Show preprocessing history
                    self._show_preprocessing_history(metadata)
                    # Load existing pipeline with steps DISABLED by default
                    self._load_preprocessing_pipeline(
                        metadata.get("preprocessing_pipeline", []),
                        default_disabled=True,
                        source_dataset=dataset_name,
                    )
                    # Auto-disable preview for preprocessed datasets
                    if self.preview_toggle_btn.isChecked():
                        self.preview_toggle_btn.blockSignals(True)
                        self.preview_toggle_btn.setChecked(False)
                        self.preview_toggle_btn.blockSignals(False)
                        self._update_preview_button_state(False)
                        self.preview_enabled = False
                    self._last_selected_was_preprocessed = True
                else:
                    # Check if switching from preprocessed to raw
                    if (
                        hasattr(self, "_last_selected_was_preprocessed")
                        and self._last_selected_was_preprocessed
                    ):
                        # Auto-enable preview for raw datasets
                        if not self.preview_toggle_btn.isChecked():
                            self.preview_toggle_btn.blockSignals(True)
                            self.preview_toggle_btn.setChecked(True)
                            self.preview_toggle_btn.blockSignals(False)
                            self._update_preview_button_state(True)
                            self.preview_enabled = True
                    else:
                        # First load or raw to raw: ensure preview is ON
                        if not self.preview_toggle_btn.isChecked():
                            self.preview_toggle_btn.blockSignals(True)
                            self.preview_toggle_btn.setChecked(True)
                            self.preview_toggle_btn.blockSignals(False)
                            self._update_preview_button_state(True)
                            self.preview_enabled = True

                    # For raw datasets, restore from global pipeline memory
                    self._restore_global_pipeline_memory()
                    self._clear_preprocessing_history_display_only()
                    self._last_selected_was_preprocessed = False

            except Exception as e:
                create_logs(
                    "PreprocessPage",
                    "preview_error",
                    f"Error previewing dataset {dataset_name}: {e}",
                    status="error",
                )
                self._clear_preprocessing_history()
        else:
            # Multiple datasets selected - keep global pipeline, just clear history display
            self._clear_preprocessing_history_display_only()
            # Restore global pipeline if we have it
            if not self.pipeline_steps and self._global_pipeline_memory:
                self._restore_global_pipeline_memory()

        # Show spectral data and store for preview
        all_dfs = []
        for item in selected_items:
            dataset_name = self._clean_dataset_name(item.text())
            if dataset_name in RAMAN_DATA:
                all_dfs.append(RAMAN_DATA[dataset_name])

        if all_dfs:
            try:
                combined_df = pd.concat(all_dfs, axis=1)
                self.original_data = combined_df

                # Show data with current preview mode
                if (
                    self.preview_toggle_btn.isChecked()
                    and hasattr(self, "pipeline_steps")
                    and self.pipeline_steps
                ):
                    self._schedule_preview_update()
                else:
                    fig = plot_spectra(
                        combined_df,
                        title="Original Data (Preview OFF)",
                        auto_focus=False,
                    )
                    self.plot_widget.update_plot(fig)

            except Exception as e:
                create_logs(
                    "PreprocessPage",
                    "preview_error",
                    f"Error previewing data: {e}",
                    status="error",
                )
                self.plot_widget.clear_plot()

    def _create_output_configuration_group(self) -> QGroupBox:
        """Create output configuration group."""
        output_group = QGroupBox()

        # Create custom title widget to match other sections
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)

        title_label = QLabel(LOCALIZE("PREPROCESS.output_config_title"))
        title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
        title_layout.addWidget(title_label)

        # Add hint button
        output_hint_btn = QPushButton("?")
        output_hint_btn.setObjectName("hintButton")
        output_hint_btn.setFixedSize(20, 20)
        output_hint_btn.setToolTip(LOCALIZE("PREPROCESS.output_config_hint"))
        output_hint_btn.setCursor(Qt.PointingHandCursor)
        output_hint_btn.setStyleSheet(
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
        title_layout.addWidget(output_hint_btn)

        title_layout.addStretch()

        layout = QVBoxLayout(output_group)
        layout.setContentsMargins(12, 4, 12, 12)
        layout.setSpacing(8)

        # Add title widget
        layout.addWidget(title_widget)

        # Output name input
        name_layout = QVBoxLayout()
        name_layout.addWidget(QLabel(LOCALIZE("PREPROCESS.output_name_label")))

        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText(
            LOCALIZE("PREPROCESS.output_name_placeholder")
        )
        name_layout.addWidget(self.output_name_edit)
        layout.addLayout(name_layout)

        # Progress bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label (initially hidden)
        self.status_label = QLabel()
        self.status_label.setVisible(False)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Run button with better styling
        self.run_button = QPushButton(
            LOCALIZE("PREPROCESS.UI.play_button")
            + " "
            + LOCALIZE("PREPROCESS.run_button")
        )
        self.run_button.setObjectName("ctaButton")
        self.run_button.setToolTip(LOCALIZE("PREPROCESS.run_button_tooltip"))
        layout.addWidget(self.run_button)

        # Cancel button with better styling (initially hidden)
        self.cancel_button = QPushButton(
            LOCALIZE("PREPROCESS.UI.stop_button")
            + " "
            + LOCALIZE("PREPROCESS.cancel_button")
        )
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setToolTip(LOCALIZE("PREPROCESS.cancel_button_tooltip"))
        self.cancel_button.setVisible(False)
        layout.addWidget(self.cancel_button)

        return output_group

    def _create_right_panel(self) -> QWidget:
        """Create the right panel for parameters and visualization."""
        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")

        layout = QVBoxLayout(right_panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Parameters section
        self.params_group = QGroupBox()

        # Create custom title widget to match other sections
        params_title_widget = QWidget()
        params_title_layout = QHBoxLayout(params_title_widget)
        params_title_layout.setContentsMargins(0, 0, 0, 0)
        params_title_layout.setSpacing(8)

        # Store reference to title label for dynamic updates
        self.params_title_label = QLabel(LOCALIZE("PREPROCESS.parameters_title"))
        self.params_title_label.setStyleSheet(
            "font-weight: 600; font-size: 13px; color: #2c3e50;"
        )
        params_title_layout.addWidget(self.params_title_label)

        # Add hint button
        params_hint_btn = QPushButton("?")
        params_hint_btn.setObjectName("hintButton")
        params_hint_btn.setFixedSize(20, 20)
        params_hint_btn.setToolTip(LOCALIZE("PREPROCESS.parameters_hint"))
        params_hint_btn.setCursor(Qt.PointingHandCursor)
        params_hint_btn.setStyleSheet(
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
        params_title_layout.addWidget(params_hint_btn)

        params_title_layout.addStretch()

        # Add "Ignore max limit" checkbox for ALL parameters (Gray theme - subtle)
        self.ignore_max_checkbox = QCheckBox(
            LOCALIZE("PARAMETER_WIDGETS.ignore_max_limit")
        )
        self.ignore_max_checkbox.setStyleSheet(
            """
            QCheckBox {
                font-size: 11px;
                color: #6c757d;
                font-weight: 600;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #adb5bd;
                border-radius: 4px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #6c757d;
                border-color: #6c757d;
            }
            QCheckBox::indicator:hover {
                border-color: #495057;
                background-color: #e9ecef;
            }
        """
        )
        self.ignore_max_checkbox.setToolTip(
            LOCALIZE("PARAMETER_WIDGETS.ignore_max_tooltip")
        )
        self.ignore_max_checkbox.toggled.connect(self._on_ignore_max_toggled_global)
        params_title_layout.addWidget(self.ignore_max_checkbox)

        # Add step info badge on the right side
        self.params_step_badge = QLabel("")
        self.params_step_badge.setStyleSheet(
            """
            QLabel {
                background-color: #e7f3ff;
                color: #0078d4;
                border: 1px solid #90caf9;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 600;
            }
        """
        )
        self.params_step_badge.setVisible(False)  # Hidden by default
        params_title_layout.addWidget(self.params_step_badge)

        params_layout = QVBoxLayout(self.params_group)
        params_layout.setContentsMargins(12, 4, 12, 12)
        params_layout.setSpacing(8)

        # Add title widget
        params_layout.addWidget(params_title_widget)

        # Scrollable area for parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(220)  # Reduced to give more space to visualization
        scroll_area.setMaximumHeight(320)  # Reduced to prevent taking too much space

        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)

        scroll_area.setWidget(self.params_container)
        params_layout.addWidget(scroll_area)

        layout.addWidget(self.params_group, 1)

        # Visualization section
        plot_group = QGroupBox()

        # Create custom title widget to match other sections
        viz_title_widget = QWidget()
        viz_title_layout = QHBoxLayout(viz_title_widget)
        viz_title_layout.setContentsMargins(0, 0, 0, 0)
        viz_title_layout.setSpacing(8)

        viz_title_label = QLabel(LOCALIZE("PREPROCESS.visualization_title"))
        viz_title_label.setStyleSheet(
            "font-weight: 600; font-size: 13px; color: #2c3e50;"
        )
        viz_title_layout.addWidget(viz_title_label)

        # Add hint button
        viz_hint_btn = QPushButton("?")
        viz_hint_btn.setObjectName("hintButton")
        viz_hint_btn.setFixedSize(20, 20)
        viz_hint_btn.setToolTip(LOCALIZE("PREPROCESS.visualization_hint"))
        viz_hint_btn.setCursor(Qt.PointingHandCursor)
        viz_hint_btn.setStyleSheet(
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
        viz_title_layout.addWidget(viz_hint_btn)

        viz_title_layout.addStretch()

        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(12, 4, 12, 12)
        plot_layout.setSpacing(8)

        # Add title widget
        plot_layout.addWidget(viz_title_widget)

        # Preview controls - Compact UI for non-maximized windows
        preview_controls = QHBoxLayout()
        preview_controls.setSpacing(8)
        preview_controls.setContentsMargins(0, 0, 0, 0)

        # Preview mode toggle with enhanced styling
        self.preview_toggle_btn = QPushButton()
        self.preview_toggle_btn.setCheckable(True)
        # Default state: ON for raw datasets (default tab), OFF for preprocessed
        # This will be adjusted by _on_dataset_tab_changed when tabs are created
        self.preview_toggle_btn.setChecked(True)  # Default for raw datasets
        self.preview_toggle_btn.setFixedHeight(28)  # Reduced from 32
        self.preview_toggle_btn.setMinimumWidth(110)  # Reduced from 120
        self.preview_toggle_btn.setToolTip(
            LOCALIZE("PREPROCESS.real_time_preview_tooltip")
        )

        # Load eye icons using centralized icon paths
        self.eye_open_icon = load_icon("eye_open", QSize(16, 16), "#2c3e50")
        self.eye_close_icon = load_icon("eye_close", QSize(16, 16), "#7f8c8d")

        # Set initial state
        self._update_preview_button_state(True)

        self.preview_toggle_btn.toggled.connect(self._toggle_preview_mode)
        preview_controls.addWidget(self.preview_toggle_btn)

        # Manual refresh button with SVG icon (compact)
        self.manual_refresh_btn = QPushButton()
        reload_icon = load_icon("reload", QSize(14, 14), "#7f8c8d")
        self.manual_refresh_btn.setIcon(reload_icon)
        self.manual_refresh_btn.setIconSize(QSize(14, 14))
        self.manual_refresh_btn.setFixedSize(28, 28)  # Reduced from 32x32
        self.manual_refresh_btn.setToolTip(
            LOCALIZE("PREPROCESS.manual_refresh_tooltip")
        )
        self.manual_refresh_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #ecf0f1;
                border: 2px solid #bdc3c7;
                border-radius: 14px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #d5dbdb;
                border-color: #85929e;
            }
            QPushButton:pressed {
                background-color: #aeb6bf;
            }
        """
        )
        self.manual_refresh_btn.clicked.connect(self._manual_refresh_preview)
        preview_controls.addWidget(self.manual_refresh_btn)

        # Manual focus button with SVG icon (compact)
        self.manual_focus_btn = QPushButton()
        focus_icon = load_icon("focus_horizontal", QSize(14, 14), "#7f8c8d")
        self.manual_focus_btn.setIcon(focus_icon)
        self.manual_focus_btn.setIconSize(QSize(14, 14))
        self.manual_focus_btn.setFixedSize(28, 28)  # Reduced from 32x32
        self.manual_focus_btn.setToolTip(LOCALIZE("PREPROCESS.UI.manual_focus_tooltip"))
        self.manual_focus_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #ecf0f1;
                border: 2px solid #bdc3c7;
                border-radius: 14px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #d5dbdb;
                border-color: #85929e;
            }
            QPushButton:pressed {
                background-color: #aeb6bf;
            }
        """
        )
        self.manual_focus_btn.clicked.connect(self._manual_focus)
        preview_controls.addWidget(self.manual_focus_btn)

        preview_controls.addStretch()

        # Compact status indicator
        self.preview_status = QLabel(LOCALIZE("PREPROCESS.UI.status_dot"))
        self.preview_status.setStyleSheet("color: #27ae60; font-size: 12px;")
        self.preview_status.setToolTip(LOCALIZE("PREPROCESS.preview_status_ready"))
        preview_controls.addWidget(self.preview_status)

        self.preview_status_text = QLabel(LOCALIZE("PREPROCESS.UI.ready_status"))
        self.preview_status_text.setStyleSheet(
            "font-size: 10px; color: #27ae60; font-weight: bold;"
        )
        preview_controls.addWidget(self.preview_status_text)

        plot_layout.addLayout(preview_controls)

        self.plot_widget = MatplotlibWidget()
        self.plot_widget.setMinimumHeight(
            400
        )  # Increased from 350 for better visibility
        plot_layout.addWidget(self.plot_widget)

        layout.addWidget(
            plot_group, 3
        )  # Increased stretch factor from 2 to 3 for more space

        return right_panel

    def _connect_signals(self):
        """Connect UI signals to their handlers."""
        self.category_combo.currentTextChanged.connect(self.update_method_combo)

        # Note: itemSelectionChanged is now connected in _create_input_datasets_group for all list widgets
        # This ensures all tabs (All, Raw, Preprocessed) respond to selection changes

        self.pipeline_list.currentItemChanged.connect(self.on_pipeline_step_selected)
        self.run_button.clicked.connect(self.run_preprocessing)
        self.cancel_button.clicked.connect(self.cancel_preprocessing)

        # Connect drag and drop reordering to automatic preview update
        self.pipeline_list.model().rowsMoved.connect(self._on_pipeline_reordered)

        # Initialize method combo
        self.update_method_combo()

    def update_method_combo(self):
        """Update method combo based on selected category."""
        self.method_combo.clear()

        current_category = self.category_combo.currentData()
        if current_category:
            methods = PREPROCESSING_REGISTRY.get_methods_by_category(current_category)
            for method_name, method_info in methods.items():
                self.method_combo.addItem(method_name)

    def load_project_data(self):
        """Load project data and populate the dataset lists (all tabs) with preprocessing info."""
        try:
            # Clear existing UI in all tabs
            self.dataset_list_all.clear()
            self.dataset_list_raw.clear()
            self.dataset_list_preprocessed.clear()
            self.plot_widget.clear_plot()

            create_logs(
                "PreprocessPage",
                "data_loading",
                f"Loading {len(RAMAN_DATA)} datasets from RAMAN_DATA",
                status="info",
            )

            if RAMAN_DATA:
                # Sort datasets: raw data first, then preprocessed
                all_items = []
                raw_items = []
                preprocessed_items = []

                for dataset_name in sorted(RAMAN_DATA.keys()):
                    # Check if this is preprocessed data
                    try:
                        metadata = PROJECT_MANAGER.get_dataframe_metadata(dataset_name)
                        is_preprocessed = metadata and metadata.get(
                            "is_preprocessed", False
                        )

                        # Create items for each list
                        item_all = QListWidgetItem()
                        item_specific = QListWidgetItem()

                        if is_preprocessed:
                            # Preprocessed data
                            item_all.setText(f"🔬 {dataset_name}")
                            item_all.setToolTip(
                                LOCALIZE(
                                    "PREPROCESS.preprocessed_data_tooltip",
                                    steps=len(
                                        metadata.get("preprocessing_pipeline", [])
                                    ),
                                )
                            )
                            item_specific.setText(f"🔬 {dataset_name}")
                            item_specific.setToolTip(
                                LOCALIZE(
                                    "PREPROCESS.preprocessed_data_tooltip",
                                    steps=len(
                                        metadata.get("preprocessing_pipeline", [])
                                    ),
                                )
                            )

                            all_items.append(
                                (1, item_all)
                            )  # Preprocessed (sort order 1)
                            preprocessed_items.append(item_specific)
                        else:
                            # Raw data
                            item_all.setText(f"📊 {dataset_name}")
                            item_all.setToolTip(LOCALIZE("PREPROCESS.raw_data_tooltip"))
                            item_specific.setText(f"📊 {dataset_name}")
                            item_specific.setToolTip(
                                LOCALIZE("PREPROCESS.raw_data_tooltip")
                            )

                            all_items.append((0, item_all))  # Raw (sort order 0)
                            raw_items.append(item_specific)

                    except Exception as e:
                        # Fallback for metadata access errors - treat as raw
                        item_all = QListWidgetItem(f"📊 {dataset_name}")
                        item_all.setToolTip(LOCALIZE("PREPROCESS.raw_data_tooltip"))
                        item_specific = QListWidgetItem(f"📊 {dataset_name}")
                        item_specific.setToolTip(
                            LOCALIZE("PREPROCESS.raw_data_tooltip")
                        )

                        all_items.append((0, item_all))
                        raw_items.append(item_specific)

                        create_logs(
                            "PreprocessPage",
                            "metadata_error",
                            f"Error accessing metadata for {dataset_name}: {e}",
                            status="warning",
                        )

                    # Ensure items are selectable
                    item_all.setFlags(
                        Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    )
                    item_specific.setFlags(
                        Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    )

                # Sort and populate "All" tab (raw first, then preprocessed)
                all_items.sort(key=lambda x: x[0])
                for _, item in all_items:
                    self.dataset_list_all.addItem(item)

                # Populate "Raw" tab
                for item in raw_items:
                    self.dataset_list_raw.addItem(item)

                # Populate "Preprocessed" tab
                for item in preprocessed_items:
                    self.dataset_list_preprocessed.addItem(item)

                # Enable all dataset lists
                self.dataset_list_all.setEnabled(True)
                self.dataset_list_raw.setEnabled(True)
                self.dataset_list_preprocessed.setEnabled(True)

                # Auto-select first item in "All" tab if available
                if self.dataset_list_all.count() > 0:
                    self.dataset_list_all.setCurrentRow(0)
                    self.showNotification.emit(
                        LOCALIZE("PREPROCESS.datasets_loaded", count=len(RAMAN_DATA)),
                        "info",
                    )
            else:
                # No data available - add placeholder in all tabs
                for list_widget in [
                    self.dataset_list_all,
                    self.dataset_list_raw,
                    self.dataset_list_preprocessed,
                ]:
                    item = QListWidgetItem(LOCALIZE("PREPROCESS.no_datasets_available"))
                    item.setFlags(Qt.ItemFlag.NoItemFlags)  # Make it non-selectable
                    list_widget.addItem(item)
                    list_widget.setEnabled(False)

                self.showNotification.emit(
                    LOCALIZE("PREPROCESS.no_data_warning"), "warning"
                )

        except Exception as e:
            create_logs(
                "PreprocessPage",
                "load_data_error",
                f"Critical error loading project data: {e}",
                status="error",
            )
            self.showNotification.emit(f"Error loading data: {str(e)}", "error")

    def clear_project_data(self):
        """Clear all project data and reset preprocessing page state when returning to home."""
        try:
            # Clear global RAMAN_DATA dictionary
            RAMAN_DATA.clear()

            # Clear all dataset lists
            self.dataset_list_all.clear()
            self.dataset_list_raw.clear()
            self.dataset_list_preprocessed.clear()

            # Clear pipeline
            self.pipeline_list.clear()
            self.pipeline_steps = []

            # Clear original and processed data
            self.original_data = None
            self.processed_data = None

            # Clear selected datasets list
            self.selected_datasets = []

            # Clear parameter widget
            self._clear_parameter_widget()
            self.current_step_widget = None

            # Clear output name if it exists
            if hasattr(self, "output_name_input"):
                self.output_name_input.clear()

            # Clear visualization
            self.plot_widget.clear_plot()

            # Reset preview state
            if hasattr(self, "preview_toggle_btn"):
                self.preview_toggle_btn.setChecked(False)

            # Clear global pipeline memory
            self._clear_global_memory()

            # Cancel any running processing
            if self.processing_thread and self.processing_thread.isRunning():
                self.processing_thread.cancel()
                self.processing_thread.wait(1000)  # Wait up to 1 second

            # Reset UI state
            self._reset_ui_state()

            # Disable lists until new project is loaded
            self.dataset_list_all.setEnabled(False)
            self.dataset_list_raw.setEnabled(False)
            self.dataset_list_preprocessed.setEnabled(False)

        except Exception as e:
            create_logs(
                "PreprocessPage",
                "clear_data_error",
                f"Error clearing preprocessing page data: {e}",
                status="error",
            )

    def export_dataset(self):
        """Export selected dataset(s) to file with format selection and metadata export."""
        # Check if any dataset is selected
        selected_items = self.dataset_list.selectedItems()
        if not selected_items:
            self.showNotification.emit(
                LOCALIZE("PREPROCESS.export_no_selection"), "warning"
            )
            return

        # Get selected datasets
        dataset_names = [
            self._clean_dataset_name(item.text()) for item in selected_items
        ]

        # Check if all datasets exist in RAMAN_DATA
        for dataset_name in dataset_names:
            if dataset_name not in RAMAN_DATA:
                self.showNotification.emit(
                    LOCALIZE("PREPROCESS.export_dataset_not_found", name=dataset_name),
                    "error",
                )
                return

        # Standardized export dialog (shared widget)
        from components.widgets import get_export_options

        formats = [
            ("csv", LOCALIZE("PREPROCESS.export_format_csv")),
            ("txt", LOCALIZE("PREPROCESS.export_format_txt")),
            ("asc", LOCALIZE("PREPROCESS.export_format_asc")),
            ("pkl", LOCALIZE("PREPROCESS.export_format_pickle")),
        ]

        last_export_path = getattr(self, "_last_export_location", None)
        if not (last_export_path and os.path.exists(last_export_path)):
            last_export_path = ""

        opts = get_export_options(
            self,
            title=LOCALIZE("PREPROCESS.export_dialog_title"),
            formats=formats,
            default_directory=last_export_path,
            default_filename=dataset_names[0] if len(dataset_names) == 1 else "",
            show_filename=(len(dataset_names) == 1),
            show_format=True,
            multiple_info_text=(
                LOCALIZE("PREPROCESS.export_multiple_info", count=len(dataset_names))
                if len(dataset_names) > 1
                else None
            ),
            multiple_names_hint=(
                LOCALIZE("PREPROCESS.export_multiple_names_info")
                if len(dataset_names) > 1
                else None
            ),
            format_label=LOCALIZE("PREPROCESS.export_format_label"),
            location_label=LOCALIZE("PREPROCESS.export_location_label"),
            filename_label=LOCALIZE("PREPROCESS.export_filename_label"),
            browse_button_text=LOCALIZE("PREPROCESS.export_browse_button"),
            select_location_title=LOCALIZE("PREPROCESS.export_select_location"),
            show_metadata_checkbox=True,
            metadata_checkbox_text=LOCALIZE("PREPROCESS.export_metadata_checkbox"),
            metadata_tooltip=LOCALIZE("PREPROCESS.export_metadata_tooltip"),
            default_export_metadata=True,
        )

        if opts is not None:
            try:
                export_format = opts.format_key
                export_path = opts.directory
                export_metadata = bool(opts.export_metadata)

                # Validate location
                if not export_path:
                    from PySide6.QtWidgets import QMessageBox
                    create_logs(
                        "export_no_location",
                        "PreprocessPage",
                        "No export location specified by user.",
                        status="warning",
                    )

                    QMessageBox.warning(
                        self,
                        LOCALIZE("PREPROCESS.export_warning_title"),
                        LOCALIZE("PREPROCESS.export_no_location_warning"),
                        QMessageBox.Ok,
                    )
                    return

                # Validate location exists
                if not os.path.exists(export_path):
                    self.showNotification.emit(
                        LOCALIZE("PREPROCESS.export_invalid_location"), "error"
                    )
                    return

                # Store last export location
                self._last_export_location = export_path

                # Export single or multiple datasets
                if len(dataset_names) == 1:
                    # Single export
                    filename = opts.filename if len(dataset_names) == 1 else dataset_names[0]

                    if not filename:
                        self.showNotification.emit(
                            LOCALIZE("PREPROCESS.export_no_filename_warning"), "warning"
                        )
                        return

                    success = self._export_single_dataset(
                        dataset_names[0],
                        export_path,
                        filename,
                        export_format,
                        export_metadata,
                    )

                    if success:
                        self.showNotification.emit(
                            LOCALIZE(
                                "PREPROCESS.export_success",
                                name=dataset_names[0],
                                path=os.path.join(
                                    export_path, f"{filename}.{export_format}"
                                ),
                            ),
                            "success",
                        )
                else:
                    # Multiple exports
                    success_count = 0
                    failed_count = 0

                    for dataset_name in dataset_names:
                        success = self._export_single_dataset(
                            dataset_name,
                            export_path,
                            dataset_name,
                            export_format,
                            export_metadata,
                        )
                        if success:
                            success_count += 1
                        else:
                            failed_count += 1

                    # Show summary
                    if failed_count == 0:
                        self.showNotification.emit(
                            LOCALIZE(
                                "PREPROCESS.export_multiple_success",
                                count=success_count,
                                path=export_path,
                            ),
                            "success",
                        )
                    else:
                        self.showNotification.emit(
                            LOCALIZE(
                                "PREPROCESS.export_multiple_partial",
                                success=success_count,
                                failed=failed_count,
                                total=len(dataset_names),
                            ),
                            "warning",
                        )

            except Exception as e:
                create_logs(
                    "PreprocessPage",
                    "export_error",
                    f"Error exporting dataset: {e}",
                    status="error",
                )
                self.showNotification.emit(
                    LOCALIZE("PREPROCESS.export_error", error=str(e)), "error"
                )

    def _export_single_dataset(
        self,
        dataset_name: str,
        export_path: str,
        filename: str,
        export_format: str,
        export_metadata: bool = True,
    ) -> bool:
        """
        Export a single dataset with optional metadata.

        Args:
            dataset_name: Name of the dataset in RAMAN_DATA
            export_path: Directory path for export
            filename: Base filename (without extension)
            export_format: File format (csv, txt, asc, pkl)
            export_metadata: Whether to export metadata JSON

        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            # Build full file path
            full_path = os.path.join(export_path, f"{filename}.{export_format}")

            # Get the data
            df = RAMAN_DATA[dataset_name]

            # Export based on format
            if export_format == "csv":
                df.to_csv(full_path)
            elif export_format == "txt":
                # Tab-separated format
                df.to_csv(full_path, sep="\t")
            elif export_format == "asc":
                # ASCII format (similar to txt)
                df.to_csv(full_path, sep="\t")
            elif export_format == "pkl":
                df.to_pickle(full_path)

            # Export metadata if requested
            if export_metadata:
                # Get metadata from project for this specific dataset
                metadata = PROJECT_MANAGER.get_dataframe_metadata(dataset_name)
                if metadata is None:
                    # If no metadata found, create empty metadata dictionary
                    metadata = {}

                self._export_metadata_json(
                    metadata, export_path, filename, df.shape, dataset_name
                )

            return True

        except Exception as e:
            create_logs(
                "PreprocessPage",
                "export_single_error",
                f"Error exporting {dataset_name}: {e}",
                status="error",
            )
            return False

    def _export_metadata_json(
        self,
        metadata: Dict,
        export_path: str,
        filename: str,
        data_shape: tuple,
        dataset_name: str = None,
    ):
        """
        Export metadata to JSON file.

        Args:
            metadata: Metadata dictionary from PROJECT_MANAGER
            export_path: Directory path for export
            filename: Base filename (without extension)
            data_shape: Shape of the exported dataframe
            dataset_name: Original dataset name in RAMAN_DATA
        """
        try:
            import json
            from datetime import datetime

            # Get additional metadata from the project's dataPackage structure
            project_metadata = metadata if metadata else {}

            # Build metadata export dictionary
            export_meta = {
                "export_info": {
                    "export_date": datetime.now().isoformat(),
                    "dataset_name": filename,
                    "original_name": dataset_name if dataset_name else filename,
                    "data_shape": {"rows": data_shape[0], "columns": data_shape[1]},
                },
                "sample": project_metadata.get("sample", {}),
                "instrument": project_metadata.get("instrument", {}),
                "measurement": project_metadata.get("measurement", {}),
                "notes": project_metadata.get("notes", {}),
                "preprocessing": {
                    "is_preprocessed": project_metadata.get("is_preprocessed", False),
                    "processing_date": project_metadata.get("processing_date", None),
                    "source_datasets": project_metadata.get("source_datasets", []),
                    "pipeline": project_metadata.get("preprocessing_pipeline", []),
                    "pipeline_summary": project_metadata.get("pipeline_summary", {}),
                    "successful_steps": project_metadata.get("successful_pipeline", []),
                    "failed_steps": project_metadata.get("failed_pipeline", []),
                },
                "spectral_info": {
                    "num_spectra": project_metadata.get("num_spectra", data_shape[1]),
                    "spectral_axis_start": (
                        project_metadata.get("spectral_axis", [None])[0]
                        if project_metadata.get("spectral_axis")
                        else None
                    ),
                    "spectral_axis_end": (
                        project_metadata.get("spectral_axis", [None, None])[-1]
                        if project_metadata.get("spectral_axis")
                        else None
                    ),
                    "spectral_points": (
                        len(project_metadata.get("spectral_axis", []))
                        if project_metadata.get("spectral_axis")
                        else data_shape[0]
                    ),
                },
            }

            # Write to JSON file (match dataset name)
            metadata_path = os.path.join(export_path, f"{filename}.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(export_meta, f, indent=2, ensure_ascii=False)

            create_logs(
                "PreprocessPage",
                "metadata_export",
                f"Exported metadata to {metadata_path}",
                status="info",
            )

        except Exception as e:
            create_logs(
                "PreprocessPage",
                "metadata_export_error",
                f"Error exporting metadata: {e}",
                status="error",
            )

    def add_pipeline_step(self):
        """Add a new step to the preprocessing pipeline."""
        current_category = self.category_combo.currentData()
        current_method = self.method_combo.currentText()

        if not current_category or not current_method:
            self.showNotification.emit(
                LOCALIZE("PREPROCESS.select_category_method"), "warning"
            )
            return

        # Save current step parameters before adding new step
        if self.current_step_widget and self.pipeline_list.currentItem():
            current_step_index = self.pipeline_list.currentItem().data(
                Qt.ItemDataRole.UserRole
            )
            if current_step_index is not None and 0 <= current_step_index < len(
                self.pipeline_steps
            ):
                current_step = self.pipeline_steps[current_step_index]
                current_step.params = self.current_step_widget.get_parameters()

        # Create pipeline step (new steps are not marked as existing)
        step = PipelineStep(current_category, current_method)
        step.is_existing = False  # New step
        self.pipeline_steps.append(step)

        # Add to UI list with custom widget
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, len(self.pipeline_steps) - 1)
        self.pipeline_list.addItem(item)

        # Create and set custom widget
        step_widget = PipelineStepWidget(step, len(self.pipeline_steps) - 1)
        step_widget.toggled.connect(self.on_step_toggled)
        item.setSizeHint(step_widget.sizeHint())
        self.pipeline_list.setItemWidget(item, step_widget)

        # Clear parameter widget before selecting new item
        self._clear_parameter_widget()

        # Set current item and force parameter widget update
        self.pipeline_list.setCurrentItem(item)
        QTimer.singleShot(100, lambda: self._show_parameter_widget(step))

        # Save to global memory after adding step
        self._save_to_global_memory()

        # Trigger automatic preview update
        self._schedule_preview_update()

        self.showNotification.emit(
            LOCALIZE("PREPROCESS.step_added", step=current_method), "info"
        )

    def remove_pipeline_step(self):
        """Remove the selected step from the preprocessing pipeline."""
        current_row = self.pipeline_list.currentRow()
        if current_row >= 0:
            # Remove from pipeline steps
            del self.pipeline_steps[current_row]

            # Remove from UI
            self.pipeline_list.takeItem(current_row)

            # Update indices in remaining items and their widgets
            for i in range(self.pipeline_list.count()):
                item = self.pipeline_list.item(i)
                item.setData(Qt.ItemDataRole.UserRole, i)

                # Update step widget index
                step_widget = self.pipeline_list.itemWidget(item)
                if isinstance(step_widget, PipelineStepWidget):
                    step_widget.step_index = i

            # Clear parameter widget if no items left
            if self.pipeline_list.count() == 0:
                self._clear_parameter_widget()
                self.toggle_all_btn.setVisible(False)
            else:
                # Select the next item or previous if at end
                new_row = min(current_row, self.pipeline_list.count() - 1)
                self.pipeline_list.setCurrentRow(new_row)

            # Save to global memory after removing step
            self._save_to_global_memory()

            # Trigger automatic preview update
            self._schedule_preview_update()

            self.showNotification.emit(LOCALIZE("PREPROCESS.step_removed"), "info")

    def clear_pipeline(self):
        """Clear all pipeline steps."""
        self.pipeline_steps.clear()
        self.pipeline_list.clear()
        self._clear_parameter_widget()
        self.toggle_all_btn.setVisible(False)

        # Clear global memory as well
        self._clear_global_memory()

        # Trigger automatic preview update to show original data
        if self.preview_enabled:
            self._schedule_preview_update()

        self.showNotification.emit(LOCALIZE("PREPROCESS.pipeline_cleared"), "info")

    def on_step_toggled(self, step_index: int, enabled: bool):
        """Handle step toggle state change."""
        # Find the actual step index by searching through the list widget
        actual_step_index = None
        sender_widget = self.sender()

        # Search for the widget that sent the signal
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            widget = self.pipeline_list.itemWidget(item)
            if widget == sender_widget:
                actual_step_index = i
                break

        # Fallback to the provided step_index if we couldn't find the widget
        if actual_step_index is None:
            actual_step_index = step_index

        # Validate the index
        if not (0 <= actual_step_index < len(self.pipeline_steps)):
            create_logs(
                "PreprocessPage",
                "step_toggle_error",
                f"Invalid step index {actual_step_index}, pipeline has {len(self.pipeline_steps)} steps",
                status="error",
            )
            return

        step = self.pipeline_steps[actual_step_index]
        step.enabled = enabled

        # Save to global memory after state change
        self._save_to_global_memory()

        # Trigger automatic preview update
        self._schedule_preview_update()

    def toggle_all_existing_steps(self):
        """Toggle all existing steps on/off."""
        # Check current state of existing steps
        existing_enabled_count = 0
        existing_total_count = 0

        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            step_widget = self.pipeline_list.itemWidget(item)

            if isinstance(step_widget, PipelineStepWidget):
                step = step_widget.step
                if hasattr(step, "is_existing") and step.is_existing:
                    existing_total_count += 1
                    if step_widget.is_enabled():
                        existing_enabled_count += 1

        # If more than half are enabled, disable all; otherwise enable all
        new_state = existing_enabled_count < (existing_total_count / 2)

        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            step_widget = self.pipeline_list.itemWidget(item)

            if isinstance(step_widget, PipelineStepWidget):
                step = step_widget.step
                if hasattr(step, "is_existing") and step.is_existing:
                    # Update the step's enabled state
                    step.enabled = new_state
                    # Update the checkbox state
                    step_widget.set_enabled(new_state)
                    # Update the eye button to reflect the new state
                    step_widget._update_enable_button()
                    # Update the visual appearance
                    step_widget._update_appearance()

        # Update button text
        if new_state:
            self.toggle_all_btn.setText(LOCALIZE("PREPROCESS.disable_all_existing"))
        else:
            self.toggle_all_btn.setText(LOCALIZE("PREPROCESS.enable_all_existing"))

    def export_pipeline(self):
        """Export current pipeline configuration to JSON file."""
        # Check if pipeline has steps
        if not self.pipeline_steps:
            self.showNotification.emit(
                LOCALIZE("PREPROCESS.DIALOGS.export_pipeline_no_steps"), "warning"
            )
            return

        # Create export dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(LOCALIZE("PREPROCESS.DIALOGS.export_pipeline_title"))
        dialog.setModal(True)
        dialog.setMinimumWidth(500)

        # Apply dialog styling
        dialog.setStyleSheet(
            """
            QDialog {
                background-color: #ffffff;
            }
            QLabel {
                color: #2c3e50;
            }
            QLineEdit, QTextEdit {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: #ffffff;
                color: #2c3e50;
            }
            QLineEdit:focus, QTextEdit:focus {
                border-color: #0078d4;
            }
            QPushButton {
                padding: 8px 16px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: #f8f9fa;
                color: #2c3e50;
            }
            QPushButton:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
            }
            QPushButton#ctaButton {
                background-color: #0078d4;
                color: white;
                border: none;
            }
            QPushButton#ctaButton:hover {
                background-color: #006abc;
            }
        """
        )

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Pipeline name input
        name_label = QLabel(LOCALIZE("PREPROCESS.DIALOGS.export_pipeline_name_label"))
        name_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(name_label)

        name_edit = QLineEdit()
        name_edit.setPlaceholderText(
            LOCALIZE("PREPROCESS.DIALOGS.export_pipeline_name_placeholder")
        )
        layout.addWidget(name_edit)

        # Description input
        desc_label = QLabel(
            LOCALIZE("PREPROCESS.DIALOGS.export_pipeline_description_label")
        )
        desc_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(desc_label)

        desc_edit = QTextEdit()
        desc_edit.setPlaceholderText(
            LOCALIZE("PREPROCESS.DIALOGS.export_pipeline_description_placeholder")
        )
        desc_edit.setMaximumHeight(100)
        layout.addWidget(desc_edit)

        # Pipeline info
        info_label = QLabel(f"📊 {len(self.pipeline_steps)} steps in current pipeline")
        info_label.setStyleSheet(
            "color: #6c757d; font-size: 12px; padding: 8px; background-color: #f8f9fa; border-radius: 4px;"
        )
        layout.addWidget(info_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton(LOCALIZE("COMMON.cancel"))
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)

        export_btn = QPushButton(LOCALIZE("PREPROCESS.export_button"))
        export_btn.setObjectName("ctaButton")
        export_btn.clicked.connect(dialog.accept)
        export_btn.setDefault(True)
        button_layout.addWidget(export_btn)

        layout.addLayout(button_layout)

        # Show dialog
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        # Get input values
        pipeline_name = name_edit.text().strip()
        pipeline_description = desc_edit.toPlainText().strip()

        if not pipeline_name:
            self.showNotification.emit(
                LOCALIZE("PREPROCESS.DIALOGS.export_pipeline_no_name"), "warning"
            )
            return

        try:
            # Get project directory
            if not PROJECT_MANAGER.current_project_data:
                self.showNotification.emit("No project loaded", "error")
                return

            project_name = (
                PROJECT_MANAGER.current_project_data.get("projectName", "")
                .replace(" ", "_")
                .lower()
            )
            project_root = os.path.join(PROJECT_MANAGER.projects_dir, project_name)
            pipelines_dir = os.path.join(project_root, "pipelines")
            os.makedirs(pipelines_dir, exist_ok=True)

            # Create pipeline data
            pipeline_data = {
                "name": pipeline_name,
                "description": pipeline_description,
                "created_date": datetime.datetime.now().isoformat(),
                "step_count": len(self.pipeline_steps),
                "steps": [],
            }

            # Add steps to pipeline data
            for step in self.pipeline_steps:
                step_data = {
                    "category": step.category,
                    "method": step.method,
                    "params": step.params,
                    "enabled": step.enabled,
                }
                pipeline_data["steps"].append(step_data)

            # Save to file
            filename = f"{pipeline_name.replace(' ', '_').lower()}.json"
            filepath = os.path.join(pipelines_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(pipeline_data, f, indent=2, ensure_ascii=False)

            create_logs(
                "PreprocessPage",
                "export_pipeline",
                f"Exported pipeline '{pipeline_name}' to {filepath}",
                status="info",
            )

            self.showNotification.emit(
                LOCALIZE(
                    "PREPROCESS.DIALOGS.export_pipeline_success", name=pipeline_name
                ),
                "success",
            )

        except Exception as e:
            create_logs(
                "PreprocessPage",
                "export_pipeline_error",
                f"Error exporting pipeline: {e}",
                status="error",
            )
            self.showNotification.emit(
                LOCALIZE("PREPROCESS.DIALOGS.export_pipeline_error", error=str(e)),
                "error",
            )

    def import_pipeline(self):
        """Import pipeline configuration from saved file."""
        try:
            # Get project directory
            if not PROJECT_MANAGER.current_project_data:
                self.showNotification.emit("No project loaded", "error")
                return

            project_name = (
                PROJECT_MANAGER.current_project_data.get("projectName", "")
                .replace(" ", "_")
                .lower()
            )
            project_root = os.path.join(PROJECT_MANAGER.projects_dir, project_name)
            pipelines_dir = os.path.join(project_root, "pipelines")

            # Get list of saved pipelines
            saved_pipelines = []
            if os.path.exists(pipelines_dir):
                for filename in os.listdir(pipelines_dir):
                    if filename.endswith(".json"):
                        filepath = os.path.join(pipelines_dir, filename)
                        try:
                            with open(filepath, "r", encoding="utf-8") as f:
                                pipeline_data = json.load(f)
                                saved_pipelines.append(
                                    {
                                        "name": pipeline_data.get("name", filename),
                                        "description": pipeline_data.get(
                                            "description", ""
                                        ),
                                        "step_count": pipeline_data.get(
                                            "step_count", 0
                                        ),
                                        "created_date": pipeline_data.get(
                                            "created_date", ""
                                        ),
                                        "filepath": filepath,
                                        "data": pipeline_data,
                                    }
                                )
                        except Exception as e:
                            create_logs(
                                "PreprocessPage",
                                "import_pipeline_read_error",
                                f"Error reading {filename}: {e}",
                                status="warning",
                            )

            # Create import dialog
            dialog = QDialog(self)
            dialog.setWindowTitle(LOCALIZE("PREPROCESS.DIALOGS.import_pipeline_title"))
            dialog.setModal(True)
            dialog.setMinimumWidth(650)  # Increased from 600 for better text visibility
            dialog.setMinimumHeight(450)  # Increased from 400

            # Apply dialog styling
            dialog.setStyleSheet(
                """
                QDialog {
                    background-color: #ffffff;
                }
                QLabel {
                    color: #2c3e50;
                }
                QListWidget {
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    background-color: #ffffff;
                }
                QListWidget::item {
                    padding: 4px;
                    border-bottom: 1px solid #e9ecef;
                }
                QListWidget::item:selected {
                    background-color: #e7f3ff;
                    border-left: 3px solid #0078d4;
                }
                QListWidget::item:hover {
                    background-color: #f8f9fa;
                }
                QPushButton {
                    padding: 8px 16px;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    background-color: #f8f9fa;
                    color: #2c3e50;
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                    border-color: #adb5bd;
                }
                QPushButton#ctaButton {
                    background-color: #0078d4;
                    color: white;
                    border: none;
                }
                QPushButton#ctaButton:hover {
                    background-color: #006abc;
                }
            """
            )

            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(16)

            # Saved pipelines list
            list_label = QLabel(
                LOCALIZE("PREPROCESS.DIALOGS.import_pipeline_saved_label")
            )
            list_label.setStyleSheet("font-weight: 600;")
            layout.addWidget(list_label)

            pipeline_list = QListWidget()
            pipeline_list.setMinimumHeight(
                280
            )  # Increased from 250 for better visibility

            if saved_pipelines:
                for pipeline in saved_pipelines:
                    item = QListWidgetItem()
                    widget = QWidget()
                    widget_layout = QVBoxLayout(widget)
                    widget_layout.setContentsMargins(
                        12, 10, 12, 10
                    )  # Increased vertical padding
                    widget_layout.setSpacing(6)  # Increased spacing

                    name_label = QLabel(f"<b>{pipeline['name']}</b>")
                    name_label.setStyleSheet(
                        "font-size: 13px; color: #2c3e50;"
                    )  # Larger font
                    name_label.setWordWrap(True)  # Enable word wrap for long names
                    widget_layout.addWidget(name_label)

                    info_label = QLabel(
                        f"📊 {pipeline['step_count']} steps | 📅 {pipeline['created_date'][:10]}"
                    )
                    info_label.setStyleSheet(
                        "color: #6c757d; font-size: 12px;"
                    )  # Increased from 11px
                    widget_layout.addWidget(info_label)

                    if pipeline["description"]:
                        desc_label = QLabel(pipeline["description"][:100])
                        desc_label.setStyleSheet(
                            "color: #495057; font-size: 12px; font-style: italic;"
                        )  # Increased from 11px
                        desc_label.setWordWrap(True)
                        widget_layout.addWidget(desc_label)

                    # Ensure adequate height for item
                    widget.adjustSize()
                    item.setSizeHint(
                        QSize(
                            widget.sizeHint().width(),
                            max(widget.sizeHint().height(), 80),
                        )
                    )  # Minimum 80px height
                    item.setData(Qt.ItemDataRole.UserRole, pipeline)
                    pipeline_list.addItem(item)
                    pipeline_list.setItemWidget(item, widget)
            else:
                no_pipelines_label = QLabel(
                    LOCALIZE("PREPROCESS.DIALOGS.import_pipeline_no_pipelines")
                )
                no_pipelines_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                no_pipelines_label.setStyleSheet("color: #6c757d; padding: 20px;")
                layout.addWidget(no_pipelines_label)

            layout.addWidget(pipeline_list)

            # External file button
            external_btn = QPushButton(
                LOCALIZE("PREPROCESS.DIALOGS.import_pipeline_external_button")
            )
            external_btn.clicked.connect(lambda: self._import_external_pipeline(dialog))
            layout.addWidget(external_btn)

            # Buttons
            button_layout = QHBoxLayout()
            button_layout.addStretch()

            cancel_btn = QPushButton(LOCALIZE("COMMON.cancel"))
            cancel_btn.clicked.connect(dialog.reject)
            button_layout.addWidget(cancel_btn)

            import_btn = QPushButton(LOCALIZE("PREPROCESS.import_pipeline_button"))
            import_btn.setObjectName("ctaButton")
            import_btn.clicked.connect(dialog.accept)
            import_btn.setDefault(True)
            button_layout.addWidget(import_btn)

            layout.addLayout(button_layout)

            # Show dialog
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            # Get selected pipeline
            current_item = pipeline_list.currentItem()
            if not current_item:
                return

            selected_pipeline = current_item.data(Qt.ItemDataRole.UserRole)

            # Confirm replacement if current pipeline exists
            if self.pipeline_steps:
                confirm = QMessageBox.question(
                    self,
                    LOCALIZE(
                        "PREPROCESS.DIALOGS.import_pipeline_confirm_replace_title"
                    ),
                    LOCALIZE(
                        "PREPROCESS.DIALOGS.import_pipeline_confirm_replace_message",
                        count=len(self.pipeline_steps),
                    ),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if confirm != QMessageBox.StandardButton.Yes:
                    return

            # Load pipeline
            self._load_pipeline_from_data(selected_pipeline["data"])

            self.showNotification.emit(
                LOCALIZE(
                    "PREPROCESS.DIALOGS.import_pipeline_success",
                    name=selected_pipeline["name"],
                    steps=selected_pipeline["step_count"],
                ),
                "success",
            )

        except Exception as e:
            create_logs(
                "PreprocessPage",
                "import_pipeline_error",
                f"Error importing pipeline: {e}",
                status="error",
            )
            self.showNotification.emit(
                LOCALIZE("PREPROCESS.DIALOGS.import_pipeline_error", error=str(e)),
                "error",
            )

    def _import_external_pipeline(self, parent_dialog):
        """Import pipeline from external file."""
        from PySide6.QtWidgets import QFileDialog

        filepath, _ = QFileDialog.getOpenFileName(
            parent_dialog,
            LOCALIZE("PREPROCESS.DIALOGS.import_pipeline_select_file"),
            "",
            "JSON Files (*.json)",
        )

        if not filepath:
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                pipeline_data = json.load(f)

            # Validate pipeline data
            if "steps" not in pipeline_data:
                raise ValueError("Invalid pipeline file: missing 'steps' field")

            # Confirm replacement if current pipeline exists
            if self.pipeline_steps:
                confirm = QMessageBox.question(
                    parent_dialog,
                    LOCALIZE(
                        "PREPROCESS.DIALOGS.import_pipeline_confirm_replace_title"
                    ),
                    LOCALIZE(
                        "PREPROCESS.DIALOGS.import_pipeline_confirm_replace_message",
                        count=len(self.pipeline_steps),
                    ),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if confirm != QMessageBox.StandardButton.Yes:
                    return

            # Load pipeline
            self._load_pipeline_from_data(pipeline_data)

            parent_dialog.accept()  # Close the import dialog

            self.showNotification.emit(
                LOCALIZE(
                    "PREPROCESS.DIALOGS.import_pipeline_success",
                    name=pipeline_data.get("name", "External Pipeline"),
                    steps=len(pipeline_data["steps"]),
                ),
                "success",
            )

        except Exception as e:
            create_logs(
                "PreprocessPage",
                "import_external_error",
                f"Error importing external pipeline: {e}",
                status="error",
            )
            self.showNotification.emit(
                LOCALIZE("PREPROCESS.DIALOGS.import_pipeline_error", error=str(e)),
                "error",
            )

    def _load_pipeline_from_data(self, pipeline_data):
        """Load pipeline steps from pipeline data dictionary."""
        # Clear current pipeline
        self.clear_pipeline()

        # Load steps
        for step_data in pipeline_data.get("steps", []):
            step = PipelineStep(
                step_data["category"], step_data["method"], step_data.get("params", {})
            )
            step.enabled = step_data.get("enabled", True)
            step.is_existing = False

            self.pipeline_steps.append(step)

            # Add to UI
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, len(self.pipeline_steps) - 1)
            self.pipeline_list.addItem(item)

            step_widget = PipelineStepWidget(step, len(self.pipeline_steps) - 1)
            step_widget.toggled.connect(self.on_step_toggled)
            item.setSizeHint(step_widget.sizeHint())
            self.pipeline_list.setItemWidget(item, step_widget)

        # Update global memory
        self._save_to_global_memory()

        # Trigger preview update
        self._schedule_preview_update()

    def on_pipeline_step_selected(self, current, previous):
        """Handle pipeline step selection to show appropriate parameters."""
        # Update visual selection state for all widgets
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            widget = self.pipeline_list.itemWidget(item)
            if widget and hasattr(widget, "set_selected"):
                widget.set_selected(item == current)

        # Save parameters from the previously selected step
        if previous and self.current_step_widget:
            prev_step_index = previous.data(Qt.ItemDataRole.UserRole)
            if prev_step_index is not None and 0 <= prev_step_index < len(
                self.pipeline_steps
            ):
                prev_step = self.pipeline_steps[prev_step_index]
                # Save current parameters to the step
                prev_step.params = self.current_step_widget.get_parameters()

        if not current:
            self._clear_parameter_widget()
            return

        step_index = current.data(Qt.ItemDataRole.UserRole)
        if step_index is not None and 0 <= step_index < len(self.pipeline_steps):
            step = self.pipeline_steps[step_index]
            # Show parameter widget with saved parameters
            QTimer.singleShot(50, lambda: self._show_parameter_widget(step))

    def _get_data_wavenumber_range(self) -> tuple:
        """Get the wavenumber range from the currently loaded data."""
        if self.original_data is not None and not self.original_data.empty:
            wavenumbers = self.original_data.index.values
            min_wn = float(wavenumbers.min())
            max_wn = float(wavenumbers.max())
            return (min_wn, max_wn)
        else:
            # Fallback to default range if no data is loaded
            return (400.0, 4000.0)

    def _show_parameter_widget(self, step: PipelineStep):
        """Show parameter widget for the selected step."""
        # Clear existing widget completely
        self._clear_parameter_widget()

        # Get method info
        method_info = PREPROCESSING_REGISTRY.get_method_info(step.category, step.method)
        if not method_info:
            return

        # Get actual data range for tuple parameters
        data_range = self._get_data_wavenumber_range()

        # Create parameter widget with saved parameters and data range
        self.current_step_widget = DynamicParameterWidget(
            method_info, step.params, data_range
        )
        self.params_layout.addWidget(self.current_step_widget)

        # Connect parameter signals for automatic preview updates
        self._connect_parameter_signals(self.current_step_widget)

        # Apply global ignore max limit state to new widgets
        if hasattr(self, "ignore_max_checkbox"):
            self._on_ignore_max_toggled_global(self.ignore_max_checkbox.isChecked())

        # Update title label with category and method name
        category_display = step.category.replace("_", " ").title()

        # Update and show step badge
        if hasattr(self, "params_step_badge"):
            self.params_step_badge.setText(
                f"{LOCALIZE(f'PREPROCESS.CATEGORY.{category_display.upper()}')}: {step.method}"
            )
            self.params_step_badge.setVisible(True)

    def _clear_parameter_widget(self):
        """Clear the parameter widget completely."""
        # Remove all widgets from the layout
        while self.params_layout.count():
            child = self.params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Clean up current step widget reference
        if self.current_step_widget:
            self.current_step_widget.deleteLater()
            self.current_step_widget = None

        # Reset title label to default
        self.params_title_label.setText(LOCALIZE("PREPROCESS.parameters_title"))

        # Hide step badge
        if hasattr(self, "params_step_badge"):
            self.params_step_badge.setVisible(False)

    def _connect_parameter_signals(self, param_widget):
        """Connect parameter widget signals for automatic preview updates."""
        if not param_widget:
            return

        # Connect all parameter widget signals to trigger preview updates
        for widget in param_widget.param_widgets.values():
            # Connect custom parameter widgets with parametersChanged signal
            if hasattr(widget, "parametersChanged"):
                widget.parametersChanged.connect(
                    lambda: self._schedule_preview_update()
                )
                # Connect real-time updates for sliders with immediate response
                if hasattr(widget, 'realTimeUpdate'):
                    widget.realTimeUpdate.connect(lambda: self._schedule_preview_update(delay_ms=50))
            # Connect radio button groups (for "radio" parameter type)
            elif hasattr(widget, 'button_group') and widget.button_group is not None:
                widget.button_group.buttonClicked.connect(lambda: self._schedule_preview_update())
            # Connect standard Qt widgets
            elif hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(lambda: self._schedule_preview_update())
            elif hasattr(widget, "textChanged"):
                widget.textChanged.connect(lambda: self._schedule_preview_update())
            elif hasattr(widget, "currentTextChanged"):
                widget.currentTextChanged.connect(
                    lambda: self._schedule_preview_update()
                )
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(lambda: self._schedule_preview_update())

    def _update_step_parameters(self):
        """Update parameters for the currently selected step."""
        current_item = self.pipeline_list.currentItem()
        if not current_item or not self.current_step_widget:
            return

        step_index = current_item.data(Qt.ItemDataRole.UserRole)
        if step_index is not None and 0 <= step_index < len(self.pipeline_steps):
            step = self.pipeline_steps[step_index]
            step.params = self.current_step_widget.get_parameters()

    def _on_ignore_max_toggled_global(self, checked: bool):
        """Handle global ignore max limit checkbox toggle for ALL parameters."""
        if not self.current_step_widget:
            return

        # Apply to ALL parameter widgets in current step
        for param_name, widget in self.current_step_widget.param_widgets.items():
            # Check if widget has ignore_max_limit capability
            if hasattr(widget, "_ignore_max_limit"):
                # Set the flag
                widget._ignore_max_limit = checked

                # Update button states to show orange warning if beyond max
                if hasattr(widget, "_update_button_states"):
                    widget._update_button_states()

                # Update tooltip warning
                if hasattr(widget, "value_input"):
                    if checked:
                        warning_text = LOCALIZE("PARAMETER_WIDGETS.beyond_max_warning")
                        widget.value_input.setToolTip(warning_text)
                    else:
                        widget.value_input.setToolTip("")

            # Also check for individual checkboxes (hide them when global is used)
            if hasattr(widget, "ignore_max_checkbox"):
                # Sync individual checkbox with global
                widget.ignore_max_checkbox.blockSignals(True)
                widget.ignore_max_checkbox.setChecked(checked)
                widget.ignore_max_checkbox.blockSignals(False)
                # Hide individual checkbox since we have global one
                widget.ignore_max_checkbox.setVisible(False)

    def run_preprocessing(self):
        """Execute the preprocessing pipeline."""
        # Validate inputs
        selected_items = self.dataset_list.selectedItems()
        if not selected_items:
            self.showNotification.emit(
                LOCALIZE("PREPROCESS.no_datasets_selected"), "error"
            )
            return

        if not self.pipeline_steps:
            self.showNotification.emit(
                LOCALIZE("PREPROCESS.no_pipeline_steps"), "error"
            )
            return

        # Check if any steps are enabled
        enabled_steps = [step for step in self.pipeline_steps if step.enabled]
        if not enabled_steps:
            self.showNotification.emit(LOCALIZE("PREPROCESS.no_enabled_steps"), "error")
            return

        output_name = self.output_name_edit.text().strip()
        if not output_name:
            self.showNotification.emit(LOCALIZE("PREPROCESS.no_output_name"), "error")
            return

        # Update step enabled states from UI
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            step_widget = self.pipeline_list.itemWidget(item)

            if isinstance(step_widget, PipelineStepWidget) and i < len(
                self.pipeline_steps
            ):
                self.pipeline_steps[i].enabled = step_widget.is_enabled()

        # Save current step parameters before processing
        if self.current_step_widget and self.pipeline_list.currentItem():
            current_step_index = self.pipeline_list.currentItem().data(
                Qt.ItemDataRole.UserRole
            )
            if current_step_index is not None and 0 <= current_step_index < len(
                self.pipeline_steps
            ):
                current_step = self.pipeline_steps[current_step_index]
                current_step.params = self.current_step_widget.get_parameters()

        # Check if output name already exists
        if output_name in RAMAN_DATA:
            reply = QMessageBox.question(
                self,
                LOCALIZE("PREPROCESS.overwrite_title"),
                LOCALIZE("PREPROCESS.overwrite_message", name=output_name),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Collect input DataFrames - FIX THE DATASET NAME EXTRACTION
        input_dfs = []
        selected_datasets = []
        for item in selected_items:
            # Remove the emoji prefixes to get the actual dataset name
            dataset_name = item.text().replace("🔬 ", "").replace("📊 ", "")
            if dataset_name in RAMAN_DATA:
                input_dfs.append(RAMAN_DATA[dataset_name])
                selected_datasets.append(dataset_name)

        if not input_dfs:
            self.showNotification.emit(
                LOCALIZE("PREPROCESS.no_valid_datasets"), "error"
            )
            create_logs(
                "PreprocessPage",
                "validation_error",
                f"No valid datasets found. Selected items: {[item.text() for item in selected_items]}",
                status="error",
            )
            return

        # Show confirmation dialog with only enabled steps
        try:
            enabled_pipeline_steps = [
                step for step in self.pipeline_steps if step.enabled
            ]
            dialog = PipelineConfirmationDialog(
                enabled_pipeline_steps, output_name, selected_datasets, self
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            # Get the actual selected datasets from dialog (user may have unchecked some)
            confirmed_datasets = dialog.get_selected_datasets()
            if not confirmed_datasets:
                self.showNotification.emit(
                    LOCALIZE("PREPROCESS.no_datasets_selected"), "error"
                )
                return

            # Get output mode (combined, separate, or single)
            output_mode = dialog.get_output_mode()

            # Filter input_dfs to only include confirmed datasets
            confirmed_input_dfs = []
            for dataset_name in confirmed_datasets:
                if dataset_name in RAMAN_DATA:
                    confirmed_input_dfs.append(RAMAN_DATA[dataset_name])

            if not confirmed_input_dfs:
                self.showNotification.emit(
                    LOCALIZE("PREPROCESS.no_valid_datasets"), "error"
                )
                return

        except Exception as e:
            create_logs(
                "PreprocessPage",
                "dialog_error",
                f"Error showing confirmation dialog: {e}",
                status="error",
            )
            self.showNotification.emit(f"Dialog error: {str(e)}", "error")
            return

        # Start processing with only enabled steps
        self._start_processing_ui()

        try:
            # Filter to only enabled steps for processing
            enabled_steps = [step for step in self.pipeline_steps if step.enabled]

            # Handle different output modes
            if output_mode == "separate":
                # Initialize separate processing state
                self.separate_processing_queue = []
                self.separate_processing_count = 0
                self.separate_processing_total = len(confirmed_datasets)

                # Prepare all processing tasks
                for dataset_name, df in zip(confirmed_datasets, confirmed_input_dfs):
                    separate_output_name = f"{dataset_name}_processed"

                    # Check if output name already exists
                    if separate_output_name in RAMAN_DATA:
                        reply = QMessageBox.question(
                            self,
                            LOCALIZE("PREPROCESS.overwrite_title"),
                            LOCALIZE(
                                "PREPROCESS.overwrite_message",
                                name=separate_output_name,
                            ),
                            QMessageBox.StandardButton.Yes
                            | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No,
                        )
                        if reply == QMessageBox.StandardButton.No:
                            continue

                    # Add task to queue with pipeline steps for metadata
                    self.separate_processing_queue.append(
                        {
                            "dataset_name": dataset_name,
                            "df": df,
                            "output_name": separate_output_name,
                            "enabled_steps": enabled_steps,
                            "pipeline_steps": self.pipeline_steps.copy(),  # Copy pipeline for metadata
                        }
                    )

                # Start processing first dataset
                if self.separate_processing_queue:
                    self._process_next_separate_dataset()
                    self.showNotification.emit(
                        LOCALIZE("PREPROCESS.processing_started")
                        + f" ({self.separate_processing_total} datasets separately)",
                        "info",
                    )
                else:
                    self._reset_ui_state()
                    self.showNotification.emit(
                        LOCALIZE("PREPROCESS.no_valid_datasets"), "error"
                    )

            else:
                # Combined or single mode - use original logic
                # Create and start processing thread
                self.processing_thread = PreprocessingThread(
                    enabled_steps,  # Pass only enabled steps
                    confirmed_input_dfs,  # Use confirmed datasets
                    output_name,
                    self,
                )

                # Connect signals with enhanced error handling
                self.processing_thread.progress_updated.connect(
                    self.on_progress_updated
                )
                self.processing_thread.status_updated.connect(self.on_status_updated)
                self.processing_thread.step_completed.connect(self.on_step_completed)
                self.processing_thread.step_failed.connect(self.on_step_failed)
                self.processing_thread.processing_completed.connect(
                    self.on_processing_completed
                )
                self.processing_thread.processing_error.connect(
                    self.on_processing_error
                )
                self.processing_thread.finished.connect(self._on_thread_finished)

                # Start the thread
                self.processing_thread.start()

                create_logs(
                    "PreprocessPage",
                    "processing_started",
                    f"Started preprocessing with {len(confirmed_input_dfs)} datasets and {len(enabled_steps)} enabled steps",
                    status="info",
                )

                self.showNotification.emit(
                    LOCALIZE("PREPROCESS.processing_started"), "info"
                )

        except Exception as e:
            create_logs(
                "PreprocessPage",
                "thread_creation_error",
                f"Error creating processing thread: {e}",
                status="error",
            )
            self.showNotification.emit(f"Processing error: {str(e)}", "error")
            self._reset_ui_state()

    def _process_next_separate_dataset(self):
        """Process the next dataset in the separate processing queue."""
        try:
            create_logs(
                "PreprocessPage",
                "_process_next_called",
                f"Processing next dataset. Queue size: {len(self.separate_processing_queue)}, Completed: {self.separate_processing_count}/{self.separate_processing_total}",
                status="info",
            )

            if not self.separate_processing_queue:
                # All datasets processed - show completion notification
                create_logs(
                    "PreprocessPage",
                    "separate_processing_complete",
                    f"All {self.separate_processing_count}/{self.separate_processing_total} datasets processed successfully",
                    status="info",
                )

                # Reset counters
                self.separate_processing_count = 0
                self.separate_processing_total = 0

                self._reset_ui_state()
                self.showNotification.emit(
                    f"Successfully processed all datasets separately", "success"
                )
                return

            # Check if thread is still running (safety check)
            if self.processing_thread and self.processing_thread.isRunning():
                create_logs(
                    "PreprocessPage",
                    "thread_still_running",
                    "WARNING: Previous thread still running, waiting...",
                    status="warning",
                )
                # Retry after delay
                QTimer.singleShot(200, self._process_next_separate_dataset)
                return

            # Get next task
            task = self.separate_processing_queue.pop(0)
            self.separate_processing_count += 1

            # Store current task for handler to access
            self.current_separate_task = task

            create_logs(
                "PreprocessPage",
                "starting_dataset",
                f"Starting dataset {self.separate_processing_count}/{self.separate_processing_total}: {task['dataset_name']}",
                status="info",
            )

            # Create thread for this dataset
            self.processing_thread = PreprocessingThread(
                task["enabled_steps"], [task["df"]], task["output_name"], self
            )

            # Connect signals
            self.processing_thread.progress_updated.connect(self.on_progress_updated)
            self.processing_thread.status_updated.connect(self.on_status_updated)
            self.processing_thread.step_completed.connect(self.on_step_completed)
            self.processing_thread.step_failed.connect(self.on_step_failed)
            self.processing_thread.processing_completed.connect(
                self._on_separate_processing_completed
            )
            self.processing_thread.processing_error.connect(
                self._on_separate_processing_error
            )
            self.processing_thread.finished.connect(self._on_separate_thread_finished)

            # Start processing
            self.processing_thread.start()

            create_logs(
                "PreprocessPage",
                "separate_processing_started",
                f"Thread started for {task['dataset_name']} -> {task['output_name']}",
                status="info",
            )

        except Exception as e:
            create_logs(
                "PreprocessPage",
                "_process_next_error",
                f"Error processing next dataset: {e}\n{traceback.format_exc()}",
                status="error",
            )
            self.showNotification.emit(f"Error: {str(e)}", "error")
            # Try to continue with next dataset
            QTimer.singleShot(100, self._process_next_separate_dataset)

    def _on_separate_processing_completed(self, result_data):
        """Handle completion of one dataset in separate processing mode."""
        try:
            create_logs(
                "PreprocessPage",
                "separate_processing_result",
                f"Dataset {self.separate_processing_count}/{self.separate_processing_total} completed successfully",
                status="info",
            )

            # Process result using standard handler
            self.on_processing_completed(result_data)

        except Exception as e:
            create_logs(
                "PreprocessPage",
                "separate_processing_result_error",
                f"Error handling result: {e}\n{traceback.format_exc()}",
                status="error",
            )
            self.showNotification.emit(f"Error processing result: {str(e)}", "error")

    def _on_separate_processing_error(self, error_msg):
        """Handle error during separate processing."""
        try:
            create_logs(
                "PreprocessPage",
                "separate_processing_error",
                f"Dataset {self.separate_processing_count}/{self.separate_processing_total} failed: {error_msg}",
                status="error",
            )
            self.showNotification.emit(error_msg, "error")

        except Exception as e:
            create_logs(
                "PreprocessPage",
                "separate_error_handler_error",
                f"Error in error handler: {e}\n{traceback.format_exc()}",
                status="error",
            )

    def _on_separate_thread_finished(self):
        """Clean up thread after separate processing (don't reset UI yet)."""
        try:
            create_logs(
                "PreprocessPage",
                "separate_thread_finished",
                f"Thread finished for dataset {self.separate_processing_count}/{self.separate_processing_total}",
                status="info",
            )

            if self.processing_thread:
                # Disconnect all signals
                try:
                    self.processing_thread.progress_updated.disconnect()
                    self.processing_thread.status_updated.disconnect()
                    self.processing_thread.step_completed.disconnect()
                    self.processing_thread.step_failed.disconnect()
                    self.processing_thread.processing_completed.disconnect()
                    self.processing_thread.processing_error.disconnect()
                    self.processing_thread.finished.disconnect()

                    create_logs(
                        "PreprocessPage",
                        "signals_disconnected",
                        "All signals disconnected successfully",
                        status="info",
                    )
                except Exception as e:
                    create_logs(
                        "PreprocessPage",
                        "signal_disconnect_error",
                        f"Error disconnecting signals: {e}\n{traceback.format_exc()}",
                        status="warning",
                    )

                # Wait for thread to fully finish
                if self.processing_thread.isRunning():
                    create_logs(
                        "PreprocessPage",
                        "waiting_for_thread",
                        "Thread still running, waiting...",
                        status="info",
                    )
                    self.processing_thread.wait(1000)  # Wait up to 1 second

                # Clean up thread reference
                self.processing_thread.deleteLater()
                self.processing_thread = None

                create_logs(
                    "PreprocessPage",
                    "thread_cleanup_complete",
                    "Thread cleanup completed",
                    status="info",
                )

            # Use QTimer to delay next processing (allows thread cleanup to complete)
            QTimer.singleShot(100, self._process_next_separate_dataset)

        except Exception as e:
            create_logs(
                "PreprocessPage",
                "separate_thread_cleanup_error",
                f"Error during separate thread cleanup: {e}\n{traceback.format_exc()}",
                status="error",
            )
            # Still try to continue with next dataset
            QTimer.singleShot(100, self._process_next_separate_dataset)

    def _on_thread_finished(self):
        """Handle thread completion (success or failure) with proper cleanup."""
        try:
            # Ensure thread is properly cleaned up
            if self.processing_thread:
                # Disconnect all signals to prevent memory leaks
                try:
                    self.processing_thread.progress_updated.disconnect()
                    self.processing_thread.status_updated.disconnect()
                    self.processing_thread.step_completed.disconnect()
                    self.processing_thread.step_failed.disconnect()
                    self.processing_thread.processing_completed.disconnect()
                    self.processing_thread.processing_error.disconnect()
                    self.processing_thread.finished.disconnect()
                except (TypeError, RuntimeError):
                    pass  # Ignore if signals are already disconnected

                # Wait for thread to finish if still running
                if self.processing_thread.isRunning():
                    self.processing_thread.wait(3000)  # Wait up to 3 seconds

                # Clean up thread reference
                self.processing_thread.deleteLater()
                self.processing_thread = None

            # Only reset UI if it wasn't already reset by completion handlers
            # This handles cases where the thread finished due to cancellation or error
            if self.progress_bar.isVisible():
                self._reset_ui_state()

        except Exception as e:
            create_logs(
                "PreprocessPage",
                "thread_cleanup_error",
                f"Error during thread cleanup: {e}",
                status="error",
            )
            # Force UI reset even if cleanup failed
            self._reset_ui_state()

    def on_step_completed(self, step_name: str, step_number: int):
        """Handle step completion updates."""
        total_steps = len(self.pipeline_steps)
        self.status_label.setText(
            LOCALIZE(
                "PREPROCESS.STATUS.step_completed",
                step=step_name,
                number=step_number,
                total=total_steps,
            )
        )

    def on_step_failed(self, step_name: str, error_message: str):
        """Handle individual step failures with logging only."""
        create_logs(
            "PreprocessPage",
            "step_failure",
            f"Step '{step_name}' failed: {error_message}",
            status="warning",
        )

    def cancel_preprocessing(self):
        """Cancel the current preprocessing operation."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.cancel()
            self.processing_thread.wait(2000)
            self._reset_ui_state()
            self.showNotification.emit(
                LOCALIZE("PREPROCESS.processing_cancelled"), "warning"
            )

    def _start_processing_ui(self):
        """Set UI to processing state."""
        self.run_button.setEnabled(False)
        self.run_button.setText("⏳ " + LOCALIZE("PREPROCESS.processing"))
        self.cancel_button.setVisible(True)
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.progress_bar.setValue(0)

    def _reset_ui_state(self):
        """Reset UI to normal state after processing."""
        self.run_button.setEnabled(True)
        self.run_button.setText("▶ " + LOCALIZE("PREPROCESS.run_button"))
        self.cancel_button.setVisible(False)
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.progress_bar.setValue(0)

    def on_progress_updated(self, value):
        """Handle progress updates from processing thread."""
        self.progress_bar.setValue(value)

    def on_status_updated(self, status):
        """Handle status updates from processing thread."""
        self.status_label.setText(status)

    def on_processing_completed(self, result_data):
        """Handle completion of preprocessing with detailed feedback."""
        try:
            create_logs(
                "PreprocessPage",
                "on_processing_completed_called",
                f"Handler called with result_data keys: {result_data.keys()}",
                status="info",
            )

            processed_df = result_data["processed_df"]
            successful_steps = result_data["successful_steps"]
            failed_steps = result_data["failed_steps"]
            total_steps = result_data["total_steps"]
            success_rate = result_data["success_rate"]

            # Use output_name from result_data (important for separate processing mode)
            output_name = result_data.get(
                "output_name", self.output_name_edit.text().strip()
            )

            create_logs(
                "PreprocessPage",
                "processing_completed_data",
                f"Processing completed for '{output_name}': {len(successful_steps)}/{total_steps} steps successful",
                status="info",
            )

            # Show failed steps dialog if there are failures
            if failed_steps:
                failed_dialog = FailedStepsDialog(failed_steps, successful_steps, self)
                failed_dialog.exec()

            # Create comprehensive metadata including preprocessing history
            selected_items = self.dataset_list.selectedItems()

            # For separate processing, use pipeline from current task
            # For combined/single mode, use self.pipeline_steps
            if hasattr(self, "current_separate_task") and self.current_separate_task:
                pipeline_steps_for_metadata = self.current_separate_task.get(
                    "pipeline_steps", self.pipeline_steps
                )
                create_logs(
                    "PreprocessPage",
                    "using_task_pipeline",
                    f"Using pipeline from separate task with {len(pipeline_steps_for_metadata)} steps",
                    status="info",
                )
            else:
                pipeline_steps_for_metadata = self.pipeline_steps

            # Save pipeline steps data for future reference
            pipeline_data = []
            for step in pipeline_steps_for_metadata:
                pipeline_data.append(
                    {
                        "category": step.category,
                        "method": step.method,
                        "params": step.params,
                        "enabled": step.enabled,
                    }
                )

            metadata = {
                "source_datasets": [item.text() for item in selected_items],
                "preprocessing_pipeline": pipeline_data,  # Save the complete pipeline
                "pipeline_summary": {
                    "total_steps": total_steps,
                    "successful_steps": len(successful_steps),
                    "failed_steps": len(failed_steps),
                    "success_rate": success_rate,
                },
                "successful_pipeline": successful_steps,
                "failed_pipeline": failed_steps,
                "processing_date": pd.Timestamp.now().isoformat(),
                "processing_type": "spectral_preprocessing",
                "spectral_axis": processed_df.index.tolist(),
                "num_spectra": processed_df.shape[1],
                "data_shape": processed_df.shape,
                "is_preprocessed": True,  # Flag to identify preprocessed data
            }

            # Save to project
            create_logs(
                "PreprocessPage",
                "saving_to_project",
                f"Attempting to save '{output_name}' with shape {processed_df.shape}",
                status="info",
            )
            success = PROJECT_MANAGER.add_dataframe_to_project(
                output_name, processed_df, metadata
            )

            if success:
                create_logs(
                    "PreprocessPage",
                    "save_success",
                    f"Successfully saved '{output_name}' to project",
                    status="info",
                )

                # Show success message with summary
                if failed_steps:
                    message = LOCALIZE(
                        "PREPROCESS.processing_success_with_failures",
                        name=output_name,
                        successful=len(successful_steps),
                        failed=len(failed_steps),
                        total=total_steps,
                    )
                    notification_type = "warning"
                else:
                    message = LOCALIZE(
                        "PREPROCESS.processing_success_complete",
                        name=output_name,
                        steps=len(successful_steps),
                    )
                    notification_type = "success"

                self.showNotification.emit(message, notification_type)
                self.load_project_data()
                self.datasets_changed.emit()  # Notify workspace of new dataset

                # Select the new dataset
                for i in range(self.dataset_list.count()):
                    if self.dataset_list.item(i).text() == output_name:
                        self.dataset_list.setCurrentRow(i)
                        break

                # Visualize results
                fig = plot_spectra(
                    processed_df,
                    title=LOCALIZE(
                        "PREPROCESS.processed_spectra_title", name=output_name
                    ),
                )
                self.plot_widget.update_plot(fig)

                # Clear form only if not in separate processing mode
                if (
                    not self.separate_processing_queue
                    and self.separate_processing_count == 0
                ):
                    self.clear_pipeline()
                    self.output_name_edit.clear()

            else:
                self.showNotification.emit(LOCALIZE("PREPROCESS.save_error"), "error")

        except Exception as e:
            error_msg = LOCALIZE("PREPROCESS.result_error", error=str(e))
            self.showNotification.emit(error_msg, "error")
            create_logs(
                "PreprocessPage",
                "result_error",
                f"Error processing results: {e}\n{traceback.format_exc()}",
                status="error",
            )
        finally:
            # Only reset UI if not in separate processing mode
            if (
                not self.separate_processing_queue
                and self.separate_processing_count == 0
            ):
                create_logs(
                    "PreprocessPage",
                    "resetting_ui_single_mode",
                    "Single mode complete, resetting UI",
                    status="info",
                )
                self._reset_ui_state()
            else:
                create_logs(
                    "PreprocessPage",
                    "separate_mode_continue",
                    f"Separate mode: {len(self.separate_processing_queue)} remaining, not resetting UI",
                    status="info",
                )

    def on_processing_error(self, error_msg):
        """Handle processing errors."""
        self.showNotification.emit(error_msg, "error")
        self._reset_ui_state()

    def _show_preprocessing_history(self, metadata: Dict):
        """Show preprocessing history information."""
        pipeline_summary = metadata.get("pipeline_summary", {})
        processing_date = metadata.get("processing_date", "")

        if processing_date:
            try:
                date_obj = pd.to_datetime(processing_date)
                formatted_date = date_obj.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                formatted_date = processing_date
        else:
            formatted_date = LOCALIZE("COMMON.unknown")

        history_text = LOCALIZE(
            "PREPROCESS.history_info",
            date=formatted_date,
            total_steps=pipeline_summary.get("total_steps", 0),
            successful_steps=pipeline_summary.get("successful_steps", 0),
            failed_steps=pipeline_summary.get("failed_steps", 0),
        )

        # Show history in parameters area temporarily
        history_label = QLabel(history_text)
        history_label.setObjectName("preprocessingHistory")
        history_label.setWordWrap(True)
        history_label.setStyleSheet(
            """
            QLabel#preprocessingHistory {
                background-color: #e8f5e8;
                border: 1px solid #4caf50;
                border-radius: 6px;
                padding: 12px;
                color: #2e7d32;
            }
        """
        )

        # Clear existing parameter widget and show history
        self._clear_parameter_widget()
        self.params_layout.addWidget(history_label)

    def _pipelines_differ(
        self, current_steps: List, target_pipeline: List[Dict]
    ) -> bool:
        """Compare current pipeline with target pipeline to check if they differ."""
        try:
            # If different number of steps, they differ
            if len(current_steps) != len(target_pipeline):
                return True

            # Compare each step
            for i, (current_step, target_step) in enumerate(
                zip(current_steps, target_pipeline)
            ):
                # Compare category and method
                if current_step.category != target_step.get(
                    "category"
                ) or current_step.method != target_step.get("method"):
                    return True

                # Compare parameters (basic comparison)
                current_params = current_step.params or {}
                target_params = target_step.get("params", {})

                # Compare keys first
                if set(current_params.keys()) != set(target_params.keys()):
                    return True

                # Compare values
                for key in current_params.keys():
                    if current_params[key] != target_params[key]:
                        return True

            return False

        except Exception as e:
            # If comparison fails, assume they differ to be safe
            return True

    def _show_pipeline_transfer_dialog(
        self, dataset_name: str, current_pipeline_backup: list = None
    ):
        """Show dialog to ask user if they want to replace current pipeline with new preprocessed dataset's pipeline."""

        from PySide6.QtWidgets import QMessageBox

        # Create custom message box
        dialog = QMessageBox(self)
        dialog.setWindowTitle(LOCALIZE("DIALOGS.pipeline_difference_title"))
        dialog.setIcon(QMessageBox.Icon.Question)

        # Main message for preprocessed-to-preprocessed transfer
        main_text = LOCALIZE("DIALOGS.pipeline_difference_message")
        dialog.setText(main_text)

        # Show current pipeline steps
        steps_text = "Current pipeline steps:\n"
        for i, step in enumerate(self.pipeline_steps, 1):
            status = "✓" if step.enabled else "○"
            steps_text += f"{i}. {status} {step.get_display_name()}\n"
        dialog.setDetailedText(steps_text)

        # Add buttons with localized text
        use_current_btn = dialog.addButton(
            LOCALIZE("DIALOGS.use_current_pipeline"), QMessageBox.ButtonRole.RejectRole
        )
        use_dataset_btn = dialog.addButton(
            LOCALIZE("DIALOGS.use_dataset_pipeline"), QMessageBox.ButtonRole.AcceptRole
        )
        cancel_btn = dialog.addButton(
            LOCALIZE("DIALOGS.cancel_switch"), QMessageBox.ButtonRole.DestructiveRole
        )

        # Show dialog
        dialog.exec()

        # Handle response
        clicked_button = dialog.clickedButton()
        if clicked_button == use_dataset_btn:
            # Get metadata and load the new pipeline with steps disabled by default
            metadata = PROJECT_MANAGER.get_dataframe_metadata(dataset_name)
            if metadata:
                self._show_preprocessing_history(metadata)
                self._load_preprocessing_pipeline(
                    metadata.get("preprocessing_pipeline", []),
                    default_disabled=True,
                    source_dataset=dataset_name,
                )
        elif clicked_button == cancel_btn:
            # User cancelled the switch - keep current dataset and pipeline
            # This is acceptable behavior: user stays on current dataset
            return
        else:  # use_current_btn
            # Keep existing pipeline steps - just restore the backup if available
            if current_pipeline_backup:
                # The pipeline steps are already in place, no need to restore
                pass

    def _clear_preprocessing_history(self):
        """Clear preprocessing history display."""
        self._clear_parameter_widget()
        # Also clear the pipeline steps to prevent them from being applied in preview
        self.pipeline_steps.clear()
        self.pipeline_list.clear()
        # Hide toggle all button
        self.toggle_all_btn.setVisible(False)

    def _clear_preprocessing_history_display_only(self):
        """Clear only the preprocessing history display but keep pipeline steps."""
        self._clear_parameter_widget()
        # Don't clear pipeline_steps - keep them for raw dataset processing
        # Hide toggle all button since there's no preprocessing history to show
        self.toggle_all_btn.setVisible(False)

    def _load_preprocessing_pipeline(
        self,
        pipeline_data: List[Dict],
        default_disabled: bool = False,
        source_dataset: str = None,
    ):
        """Load existing preprocessing pipeline for editing/extension."""
        create_logs(
            "PreprocessPage",
            "_load_preprocessing_pipeline_called",
            f"Loading pipeline: {len(pipeline_data)} steps, default_disabled={default_disabled}, source={source_dataset}",
            status="info",
        )

        self.pipeline_steps.clear()
        self.pipeline_list.clear()

        has_existing_steps = False

        for i, step_data in enumerate(pipeline_data):
            create_logs(
                "PreprocessPage",
                "loading_step",
                f"Step {i+1}: {step_data['category']}.{step_data['method']}",
                status="info",
            )

            step = PipelineStep(
                step_data["category"],
                step_data["method"],
                step_data.get("params", {}),
                source_dataset,
            )
            # Set enabled state - for preprocessed datasets, default to disabled unless specified
            if default_disabled:
                step.enabled = False
            else:
                step.enabled = step_data.get("enabled", True)

            # Mark as existing step (will be skipped by default unless toggled)
            step.is_existing = True
            has_existing_steps = True

            self.pipeline_steps.append(step)

            # Add to UI list with custom widget
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.pipeline_list.addItem(item)

            # Create and set custom widget
            step_widget = PipelineStepWidget(step, i)
            step_widget.toggled.connect(self.on_step_toggled)
            item.setSizeHint(step_widget.sizeHint())
            self.pipeline_list.setItemWidget(item, step_widget)

        create_logs(
            "PreprocessPage",
            "_load_preprocessing_pipeline_complete",
            f"Loaded {len(self.pipeline_steps)} steps successfully",
            status="info",
        )

        # Show toggle all button if there are existing steps
        if has_existing_steps:
            self.toggle_all_btn.setVisible(True)
            if default_disabled:
                self.toggle_all_btn.setText(LOCALIZE("PREPROCESS.enable_all_existing"))
            else:
                self.toggle_all_btn.setText(LOCALIZE("PREPROCESS.disable_all_existing"))

    def _set_default_output_name(self, dataset_name: str):
        """Set default output name based on selected dataset name with '-pp' suffix."""
        # Only set default if the output name field is empty or contains a previous default
        current_text = self.output_name_edit.text().strip()

        # Check if current text is empty or appears to be a previous default (ends with -pp)
        if not current_text or current_text.endswith("-pp"):
            default_name = f"{dataset_name}-pp"
            self.output_name_edit.setText(default_name)
            # Set placeholder to show the default behavior
            self.output_name_edit.setPlaceholderText(
                LOCALIZE("PREPROCESS.output_name_placeholder")
            )

    def _clear_default_output_name(self):
        """Clear the default output name when no datasets are selected."""
        current_text = self.output_name_edit.text().strip()

        # Only clear if it appears to be a default name (ends with -pp)
        if current_text.endswith("-pp"):
            self.output_name_edit.clear()

    # ============== AUTOMATIC PREVIEW SYSTEM ==============

    def _toggle_preview_mode(self, enabled: bool):
        """Toggle real-time preview mode."""

        # Check if we're enabling preview on a preprocessed dataset
        if (
            enabled
            and hasattr(self, "_last_selected_was_preprocessed")
            and self._last_selected_was_preprocessed
        ):

            # Show warning dialog about double preprocessing
            from PySide6.QtWidgets import QMessageBox

            dialog = QMessageBox(self)
            dialog.setWindowTitle(LOCALIZE("DIALOGS.preview_toggle_warning_title"))
            dialog.setIcon(QMessageBox.Icon.Warning)

            dialog.setText(LOCALIZE("DIALOGS.preview_on_preprocessed_warning"))

            dialog.addButton("Continue", QMessageBox.ButtonRole.AcceptRole)
            cancel_btn = dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

            dialog.exec()
            clicked_button = dialog.clickedButton()

            if clicked_button == cancel_btn:
                # Reset the toggle button to OFF without triggering this method again
                self.preview_toggle_btn.blockSignals(True)
                self.preview_toggle_btn.setChecked(False)
                self.preview_toggle_btn.blockSignals(False)
                return

        # Check if we're disabling preview on a raw dataset (and there are pipeline steps)
        elif (
            not enabled
            and hasattr(self, "_last_selected_was_preprocessed")
            and not self._last_selected_was_preprocessed
            and hasattr(self, "pipeline_steps")
            and len(self.pipeline_steps) > 0
        ):

            # Show warning dialog about hiding processing effects
            from PySide6.QtWidgets import QMessageBox

            dialog = QMessageBox(self)
            dialog.setWindowTitle(LOCALIZE("DIALOGS.preview_toggle_warning_title"))
            dialog.setIcon(QMessageBox.Icon.Warning)

            dialog.setText(LOCALIZE("DIALOGS.preview_off_raw_warning"))

            dialog.addButton("Continue", QMessageBox.ButtonRole.AcceptRole)
            cancel_btn = dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

            dialog.exec()
            clicked_button = dialog.clickedButton()

            if clicked_button == cancel_btn:
                # Reset the toggle button to ON without triggering this method again
                self.preview_toggle_btn.blockSignals(True)
                self.preview_toggle_btn.setChecked(True)
                self.preview_toggle_btn.blockSignals(False)
                return

        self.preview_enabled = enabled
        self._update_preview_button_state(enabled)

        if enabled:
            self._update_preview_status("ready", "green")
            # Trigger preview update if data is available
            if self.original_data is not None:
                self._schedule_preview_update()
        else:
            self._update_preview_status("disabled", "gray")
            # Show original data only
            self._show_original_data()

    def _update_preview_status(self, status: str, color: str):
        """Update the preview status indicator."""
        color_map = {
            "green": "#27ae60",
            "processing": "#666666",  # Dark gray for processing state
            "red": "#e74c3c",
            "gray": "#95a5a6",
        }

        status_text_map = {
            "ready": LOCALIZE("PREPROCESS.UI.ready_status"),
            "processing": LOCALIZE("PREPROCESS.UI.processing_status"),
            "error": LOCALIZE("PREPROCESS.UI.error_status"),
            "disabled": LOCALIZE("PREPROCESS.UI.disabled_status"),
        }

        mapped_color = color_map.get(color, color)
        status_text = status_text_map.get(status, status.title())

        self.preview_status.setStyleSheet(f"color: {mapped_color}; font-size: 14px;")
        self.preview_status_text.setStyleSheet(
            f"font-size: 11px; color: {mapped_color}; font-weight: bold;"
        )
        self.preview_status_text.setText(status_text)

        status_messages = {
            "ready": LOCALIZE("PREPROCESS.preview_status_ready"),
            "processing": LOCALIZE("PREPROCESS.preview_status_processing"),
            "error": LOCALIZE("PREPROCESS.preview_status_error"),
            "disabled": LOCALIZE("PREPROCESS.preview_status_disabled"),
        }
        tooltip = status_messages.get(status, status)
        self.preview_status.setToolTip(tooltip)
        self.preview_status_text.setToolTip(tooltip)

    def _manual_refresh_preview(self):
        """Manually trigger a preview refresh."""
        if self.original_data is not None:
            self._update_preview_status("processing", "processing")
            # Use immediate update instead of scheduled
            QTimer.singleShot(
                100, self._update_preview
            )  # Small delay for UI responsiveness
        else:
            self.showNotification.emit(
                LOCALIZE("PREPROCESS.no_data_for_preview"), "warning"
            )

    def _schedule_preview_update(self, delay_ms: int = 300):
        """Schedule a preview update with debouncing."""
        if not self.preview_enabled:
            return

        # Stop any existing timer
        self.preview_timer.stop()
        self._update_preview_status("processing", "processing")

        # Save parameters to global memory when they change
        self._save_to_global_memory()

        # Start new timer with delay for debouncing
        self.preview_timer.start(delay_ms)

    def _apply_full_pipeline(self, data, steps):
        """Apply preprocessing steps to data without sampling - for manual focus and final processing."""
        try:
            # Return original data if no steps
            if not steps:
                return data

            # Work directly with DataFrame
            processed_data = data.copy()

            # Update current step parameters from widget before applying pipeline
            current_row = self.pipeline_list.currentRow()
            if (
                current_row >= 0
                and self.current_step_widget
                and current_row < len(self.pipeline_steps)
            ):
                # Get the actual step from the full pipeline list (not the filtered enabled steps)
                current_step = self.pipeline_steps[current_row]
                # Only update if this step is in the steps list being processed
                if current_step in steps:
                    current_params = self.current_step_widget.get_parameters()
                    if current_params:
                        current_step.params = current_params

            # Apply each enabled step
            for i, step in enumerate(steps):
                if not step.enabled:
                    continue

                try:
                    # Get method instance
                    method_instance = step.create_instance()

                    # Apply to full data (no sampling)
                    if hasattr(method_instance, "apply"):
                        # Convert DataFrame to SpectralContainer format
                        wavenumbers = processed_data.index.values
                        intensities = (
                            processed_data.values.T
                        )  # Shape: (n_spectra, n_wavenumbers)

                        # NO SAMPLING for full pipeline

                        # Create SpectralContainer for this step
                        import ramanspy as rp

                        temp_spectral = rp.SpectralContainer(intensities, wavenumbers)

                        # Apply processing
                        result = method_instance.apply(temp_spectral)

                        # Convert back to DataFrame
                        new_wavenumbers = result.spectral_axis
                        new_intensities = (
                            result.intensities.T
                        )  # Shape: (n_wavenumbers, n_spectra)

                        # Create new DataFrame with preserved column names
                        if new_intensities.shape[1] == processed_data.shape[1]:
                            # Same number of spectra, preserve column names
                            processed_data = pd.DataFrame(
                                new_intensities,
                                index=new_wavenumbers,
                                columns=processed_data.columns,
                            )
                        else:
                            # Different number of spectra, create new column names
                            new_columns = [
                                f"spectrum_{j}" for j in range(new_intensities.shape[1])
                            ]
                            processed_data = pd.DataFrame(
                                new_intensities,
                                index=new_wavenumbers,
                                columns=new_columns,
                            )

                except Exception as step_error:
                    continue

            return processed_data

        except Exception as e:
            return data  # Return original data on error

    def _apply_preview_pipeline(self, data, steps):
        """Apply preprocessing steps to data for preview - DataFrame-first approach."""
        try:
            # Return original data if no steps
            if not steps:
                return data

            # Work directly with DataFrame (original approach)
            processed_data = data.copy()

            # DO NOT update parameters from current widget - each step has its own params
            # The current_step_widget might be showing different step's parameters

            # Apply each enabled step
            for step in steps:
                # Skip disabled steps
                if not step.enabled:
                    continue

                try:

                    # Get preprocessing method instance with step's own parameters
                    method_info = PREPROCESSING_REGISTRY.get_method_info(
                        step.category, step.method
                    )

                    if not method_info:
                        create_logs(
                            "preview_method_error",
                            "PreprocessPage",
                            f"Method {step.category}.{step.method} not found in registry",
                            status="warning",
                        )
                        continue

                    # Use step.params directly - don't contaminate with current widget params
                    method_instance = PREPROCESSING_REGISTRY.create_method_instance(
                        step.category, step.method, step.params
                    )

                    # Handle different method types
                    if step.method in ["WavenumberCalibration", "IntensityCalibration"]:
                        # Skip calibration methods in preview
                        create_logs(
                            "preview_skip",
                            "PreprocessPage",
                            f"Skipping {step.method} in preview (requires special parameters)",
                            status="info",
                        )
                        continue

                    elif hasattr(method_instance, "apply"):
                        # Convert DataFrame to SpectralContainer for this step only
                        wavenumbers = processed_data.index.values
                        intensities = (
                            processed_data.values.T
                        )  # Shape: (n_spectra, n_wavenumbers)

                        # Use sample for faster preview
                        if intensities.shape[0] > 10:
                            sample_indices = list(
                                range(
                                    0,
                                    intensities.shape[0],
                                    max(1, intensities.shape[0] // 5),
                                )
                            )
                            intensities = intensities[sample_indices]

                        # Create SpectralContainer for this step only
                        import ramanspy as rp

                        temp_spectral = rp.SpectralContainer(intensities, wavenumbers)

                        # Apply processing
                        result = method_instance.apply(temp_spectral)

                        # Convert result back to DataFrame immediately
                        if hasattr(result, "spectral_data") and hasattr(
                            result, "spectral_axis"
                        ):
                            import pandas as pd

                            processed_data = pd.DataFrame(
                                result.spectral_data.T,  # Transpose back to (wavenumbers, spectra)
                                index=result.spectral_axis,
                                columns=[
                                    f"preview_{i}"
                                    for i in range(result.spectral_data.shape[0])
                                ],
                            )
                            processed_data.index.name = "wavenumber"

                    else:
                        create_logs(
                            "preview_skip",
                            "PreprocessPage",
                            f"Method {step.method} does not have apply method",
                            status="warning",
                        )

                except Exception as e:
                    # Log the error but continue with next step
                    error_msg = str(e)
                    create_logs(
                        "preview_method_error",
                        "PreprocessPage",
                        f"[{step.category}] Error applying {step.method} step_method: {error_msg}",
                        status="error",
                    )

                    # Show error dialog to user
                    self._show_parameter_error_dialog(
                        step.method, step.category, error_msg
                    )
                    continue

            return processed_data

        except Exception as e:
            # Log the error and return original data if pipeline fails
            create_logs(
                "preview_pipeline_error",
                "PreprocessPage",
                f"Pipeline failed: {str(e)}",
                status="error",
            )
            return data

    def _should_auto_focus(self) -> bool:
        """
        Check if auto-focus should be enabled based on pipeline contents.
        Auto-focus is only enabled when there are range-limiting preprocessing steps.
        """
        if not self.pipeline_steps:
            return False

        # Check for range-limiting steps that benefit from auto-focus
        range_limiting_steps = {
            "Cropper",  # Primary range-limiting step
            "Range Selector",
            "Spectral Cropper",
            "Wavenumber Filter",
        }

        enabled_steps = [step for step in self.pipeline_steps if step.enabled]

        for step in enabled_steps:
            step_name = getattr(step, "method_name", "")
            if any(
                range_step.lower() in step_name.lower()
                for range_step in range_limiting_steps
            ):
                return True

        return False

    def _extract_crop_bounds(self):
        """Extract crop bounds from pipeline steps for proper focus padding."""
        try:
            for i, step in enumerate(self.pipeline_steps):
                # Check for Cropper method specifically
                if step.enabled and step.method in [
                    "Cropper",
                    "cropper",
                    "Crop",
                    "crop",
                ]:
                    if hasattr(step, "params") and "region" in step.params:
                        region = step.params["region"]
                        if isinstance(region, (tuple, list)) and len(region) == 2:
                            return region[0], region[1]
            return None
        except Exception as e:
            return None

    def _show_parameter_error_dialog(
        self, method_name: str, category: str, error_message: str
    ):
        """Show error dialog when preprocessing fails due to parameter issues."""
        from PySide6.QtWidgets import QMessageBox

        # Get localized title (but show error in English for clarity)
        error_title = LOCALIZE("PARAMETER_WIDGETS.parameter_error_title")

        # Build error message without localization template (show raw error in English)
        header_text = "An error occurred during preprocessing:"
        error_detail = f"[{category}] {method_name}\n\n{error_message}"
        suggestion_text = "\nPlease adjust parameter values or check your settings."

        full_message = f"{header_text}\n\n{error_detail}{suggestion_text}"

        # Create and show the error dialog with modern app theme
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle(error_title)
        msg_box.setText(full_message)
        msg_box.setStandardButtons(QMessageBox.Ok)

        # Modern styling matching app theme (blue/white, not yellow)
        msg_box.setStyleSheet(
            """
            QMessageBox {
                background-color: #ffffff;
                border: 2px solid #0078d4;
            }
            QLabel {
                color: #2c3e50;
                font-size: 13px;
                padding: 10px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 24px;
                border-radius: 4px;
                font-size: 13px;
                font-weight: 600;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:pressed {
                background-color: #004578;
            }
        """
        )

        msg_box.exec()

    def _show_original_data(self):
        """Show original unprocessed data without any pipeline effects."""
        if self.original_data is not None:
            # For preview OFF mode, show original data without any pipeline influence
            # Always disable auto-focus to show the complete original spectrum
            crop_bounds = self._extract_crop_bounds()
            self.plot_widget.plot_spectra(
                self.original_data,
                title="Original Data (Preview OFF)",
                auto_focus=False,
                crop_bounds=crop_bounds,
            )
        else:
            self.plot_widget.clear_plot()

    def _show_preview_data(self, processed_data):
        """Show processed data with original data overlay."""
        if self.original_data is not None and processed_data is not None:
            try:
                # Convert processed_data from SpectralContainer if needed
                if hasattr(processed_data, "spectral_data"):
                    # SpectralContainer format - extract data
                    processed_array = processed_data.spectral_data
                    wavenumbers = processed_data.spectral_axis
                else:
                    # Already numpy array
                    processed_array = processed_data
                    wavenumbers = None

                # Get original data for comparison
                if hasattr(self.original_data, "values"):
                    # DataFrame format
                    original_array = (
                        self.original_data.values.T
                    )  # Transpose to match format
                    original_wavenumbers = self.original_data.index.values
                else:
                    original_array = self.original_data
                    original_wavenumbers = None

                # Sample data for visualization if too large
                if original_array.shape[0] > 10:
                    sample_indices = np.arange(
                        0,
                        original_array.shape[0],
                        max(1, original_array.shape[0] // 10),
                    )
                    sample_original = original_array[sample_indices]
                else:
                    sample_original = original_array

                # Plot both original and processed data with conditional auto-focus
                auto_focus = self._should_auto_focus()

                if wavenumbers is not None and original_wavenumbers is not None:
                    # Both have wavenumber info
                    crop_bounds = self._extract_crop_bounds()
                    self.plot_widget.plot_comparison_spectra_with_wavenumbers(
                        sample_original,
                        processed_array,
                        original_wavenumbers,
                        wavenumbers,
                        titles=["Original", "Processed"],
                        colors=["lightblue", "darkblue"],
                        auto_focus=auto_focus,
                        focus_padding=50,  # Add 50 unit padding for auto-focus
                        crop_bounds=crop_bounds,
                    )
                else:
                    # Fallback to original method
                    self.plot_widget.plot_comparison_spectra(
                        sample_original,
                        processed_array,
                        titles=["Original", "Processed"],
                        colors=["lightblue", "darkblue"],
                    )

            except Exception as e:
                create_logs(
                    "preview_display_error",
                    "PreprocessPage",
                    f"Error displaying preview data: {str(e)}",
                    status="error",
                )
                # Fallback to showing original data
                self._show_original_data()

    def _on_pipeline_reordered(self):
        """Handle pipeline reordering via drag and drop."""
        # Update pipeline_steps order to match UI order
        new_order = []
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            old_index = item.data(Qt.ItemDataRole.UserRole)
            if old_index is not None and old_index < len(self.pipeline_steps):
                new_order.append(self.pipeline_steps[old_index])

        self.pipeline_steps = new_order

        # Update item data to reflect new order
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            item.setData(Qt.ItemDataRole.UserRole, i)

        # Trigger preview update
        self._schedule_preview_update()

    def _get_data_wavenumber_range(self):
        """Get the wavenumber range from the currently loaded data."""
        if self.original_data is not None and not self.original_data.empty:
            wavenumbers = self.original_data.index.values
            min_wn = float(wavenumbers.min())
            max_wn = float(wavenumbers.max())
            return (min_wn, max_wn)
        else:
            # Fallback to default range if no data is loaded
            return (400.0, 4000.0)

    def _should_auto_focus(self) -> bool:
        """Check if auto-focus should be enabled based on pipeline contents."""
        range_limiting_steps = [
            "Cropper",
            "Range Selector",
            "Spectral Window",
            "Baseline Range",
        ]
        enabled_steps = [step for step in self.pipeline_steps if step.enabled]
        return any(step.method in range_limiting_steps for step in enabled_steps)

    def _update_preview(self):
        """Update the preview plot with current pipeline."""
        try:
            # Get current selected datasets
            selected_items = self.dataset_list.selectedItems()
            if not selected_items:
                self.plot_widget.clear_plot()
                return

            # If preview is disabled, show original data only
            if not self.preview_enabled:
                first_item = selected_items[0]
                dataset_name = self._clean_dataset_name(first_item.text())

                if dataset_name in RAMAN_DATA:
                    self.original_data = RAMAN_DATA[dataset_name]
                    self._show_original_data()
                else:
                    self.plot_widget.clear_plot()
                return

            # Use first selected dataset for preview
            first_item = selected_items[0]
            dataset_name = self._clean_dataset_name(first_item.text())

            if dataset_name not in RAMAN_DATA:
                self.plot_widget.clear_plot()
                return

            original_data = RAMAN_DATA[dataset_name]
            self.original_data = original_data  # Ensure original_data is set

            # Apply current pipeline to get processed data
            enabled_steps = [step for step in self.pipeline_steps if step.enabled]
            processed_data = self._apply_preview_pipeline(original_data, enabled_steps)

            # Determine if auto-focus should be used
            auto_focus = self._should_auto_focus()

            # Plot comparison with conditional auto-focus
            if processed_data is not None:
                # Check if data has been modified by comparing shapes and values
                data_modified = False
                try:
                    # Compare processed DataFrame with original DataFrame
                    if hasattr(processed_data, "shape") and hasattr(
                        original_data, "shape"
                    ):
                        # Both are DataFrames - do shape and content comparison
                        if processed_data.shape != original_data.shape:
                            data_modified = True
                        elif hasattr(processed_data, "equals") and hasattr(
                            original_data, "equals"
                        ):
                            # DataFrame comparison
                            if not processed_data.equals(original_data):
                                data_modified = True
                        else:
                            # Fallback to numpy array comparison
                            import numpy as np

                            if not np.array_equal(
                                processed_data.values, original_data.values
                            ):
                                data_modified = True
                    else:
                        # Fallback - assume modified if we can't compare properly
                        data_modified = True
                except Exception as e:
                    # If comparison fails, assume data is modified
                    data_modified = True

                if data_modified:
                    # Both original_data and processed_data are now DataFrames
                    # Use plot_spectra for both as it handles DataFrames correctly
                    # instead of trying to extract arrays manually

                    # Create a simple comparison by showing both datasets
                    crop_bounds = self._extract_crop_bounds()
                    self.plot_widget.plot_spectra(
                        processed_data,
                        title="Preprocessed Data (Preview)",
                        auto_focus=auto_focus,
                        focus_padding=50,  # Add 50 unit padding for auto-focus
                        crop_bounds=crop_bounds,
                    )
                else:
                    # Show original data only
                    crop_bounds = self._extract_crop_bounds()
                    self.plot_widget.plot_spectra(
                        original_data,
                        title="Original Data",
                        auto_focus=False,  # No auto-focus for original data view
                        crop_bounds=crop_bounds,
                    )
            else:
                # Show original data only
                crop_bounds = self._extract_crop_bounds()
                self.plot_widget.plot_spectra(
                    original_data,
                    title="Original Data",
                    auto_focus=False,  # No auto-focus for original data view
                    crop_bounds=crop_bounds,
                )

            # Update status
            self.preview_status.setStyleSheet("color: #27ae60; font-size: 14px;")
            self.preview_status_text.setText(LOCALIZE("PREPROCESS.UI.ready_status"))
            self.preview_status_text.setStyleSheet(
                "font-size: 11px; color: #27ae60; font-weight: bold;"
            )

        except Exception as e:
            pass  # Silently handle preview errors to avoid log spam
            self.preview_status.setStyleSheet("color: #e74c3c; font-size: 14px;")
            self.preview_status_text.setText(LOCALIZE("PREPROCESS.UI.error_status"))
            self.preview_status_text.setStyleSheet(
                "font-size: 11px; color: #e74c3c; font-weight: bold;"
            )

    def _clean_dataset_name(self, item_text: str) -> str:
        """Clean dataset name by removing UI prefixes like emojis."""
        return item_text.replace("📊 ", "").replace("🔬 ", "").strip()

    def _manual_focus(self):
        """Manually apply focus to the current plot with padding."""
        try:
            import numpy as np

            # Get current selected datasets
            selected_items = self.dataset_list.selectedItems()
            if not selected_items:
                return

            # Use first selected dataset
            first_item = selected_items[0]
            dataset_name = self._clean_dataset_name(first_item.text())

            if dataset_name not in RAMAN_DATA:
                return

            original_data = RAMAN_DATA[dataset_name]
            enabled_steps = [step for step in self.pipeline_steps if step.enabled]
            processed_data = self._apply_full_pipeline(original_data, enabled_steps)

            # Force auto-focus with padding
            if processed_data is not None:
                # Check if data has been modified
                data_modified = False
                try:
                    # Compare processed DataFrame with original DataFrame
                    if hasattr(processed_data, "shape") and hasattr(
                        original_data, "shape"
                    ):
                        # Both are DataFrames - do shape and content comparison
                        if processed_data.shape != original_data.shape:
                            data_modified = True
                        elif hasattr(processed_data, "equals") and hasattr(
                            original_data, "equals"
                        ):
                            # DataFrame comparison
                            if not processed_data.equals(original_data):
                                data_modified = True
                        else:
                            # Fallback to numpy array comparison
                            if not np.array_equal(
                                processed_data.values, original_data.values
                            ):
                                data_modified = True
                    else:
                        # Fallback - assume modified if we can't compare properly
                        data_modified = True
                except Exception as comp_e:
                    # If comparison fails, assume data is modified
                    data_modified = True

                if data_modified:
                    # Both original_data and processed_data are now DataFrames
                    # Extract data for comparison plot
                    original_array = original_data.values
                    original_wavenumbers = original_data.index.values
                    processed_array = processed_data.values
                    processed_wavenumbers = processed_data.index.values

                    # Show comparison plot with forced auto-focus and padding
                    crop_bounds = self._extract_crop_bounds()
                    self.plot_widget.plot_comparison_spectra_with_wavenumbers(
                        original_array,
                        processed_array,
                        original_wavenumbers,
                        processed_wavenumbers,
                        titles=["Original", "Processed"],
                        auto_focus=True,  # Force auto-focus
                        focus_padding=50,  # Add 50 unit padding on each side
                        crop_bounds=crop_bounds,
                    )
                else:
                    # Show original data with forced auto-focus and padding
                    crop_bounds = self._extract_crop_bounds()
                    self.plot_widget.plot_spectra(
                        original_data,
                        title="Original Data (Focused)",
                        auto_focus=True,  # Force auto-focus
                        focus_padding=50,  # Add 50 unit padding on each side
                        crop_bounds=crop_bounds,
                    )
            else:
                # Show original data with forced auto-focus and padding
                crop_bounds = self._extract_crop_bounds()
                self.plot_widget.plot_spectra(
                    original_data,
                    title="Original Data (Focused)",
                    auto_focus=True,  # Force auto-focus
                    focus_padding=50,  # Add 50 unit padding on each side
                    crop_bounds=crop_bounds,
                )

        except Exception as e:
            create_logs(
                "PreprocessPage",
                "_manual_focus",
                f"Error in manual focus: {e}",
                status="error",
            )

    def _toggle_preview_mode(self, enabled):
        """Toggle preview mode on/off."""
        self.preview_enabled = enabled
        self._update_preview_button_state(enabled)

        if enabled:
            # Ensure we have data loaded when turning preview back on
            if self.original_data is None or self.original_data.empty:
                self.preview_raw_data()  # This will load data and trigger preview
            else:
                self._update_preview()
        else:
            self.plot_widget.clear_plot()

    def _update_preview_button_state(self, enabled):
        """Update preview button appearance based on state."""
        if enabled:
            self.preview_toggle_btn.setIcon(self.eye_open_icon)
            text = LOCALIZE("PREPROCESS.UI.preview_on")
            self.preview_toggle_btn.setText(text)
            self.preview_toggle_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #e8f5e8;
                    border: 2px solid #27ae60;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: bold;
                    color: #27ae60;
                }
                QPushButton:hover {
                    background-color: #d5e8d4;
                }
            """
            )
        else:
            self.preview_toggle_btn.setIcon(self.eye_close_icon)
            text = LOCALIZE("PREPROCESS.UI.preview_off")
            self.preview_toggle_btn.setText(text)
            self.preview_toggle_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #f8f8f8;
                    border: 2px solid #7f8c8d;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: bold;
                    color: #7f8c8d;
                }
                QPushButton:hover {
                    background-color: #ecf0f1;
                }
            """
            )

        # Calculate dynamic width based on text length
        self._adjust_button_width_to_text()

    def _adjust_button_width_to_text(self):
        """Adjust button width dynamically based on text content."""
        # Get current text and font metrics
        text = self.preview_toggle_btn.text()
        font = self.preview_toggle_btn.font()

        # Calculate text width using font metrics
        from PySide6.QtGui import QFontMetrics

        font_metrics = QFontMetrics(font)
        text_width = font_metrics.horizontalAdvance(text)

        # Add padding for icon (16px) + spacing (8px) + left/right padding (16px) + border (4px)
        icon_width = 16
        spacing = 8 if text.strip() else 0  # No spacing if no text
        padding = 16  # 8px left + 8px right from CSS padding
        border = 4  # 2px left + 2px right from CSS border

        total_width = text_width + icon_width + spacing + padding + border

        # Set minimum width to prevent button from being too small
        min_width = 80
        dynamic_width = max(min_width, total_width)

        self.preview_toggle_btn.setFixedWidth(dynamic_width)

    # ============== GLOBAL PIPELINE MEMORY MANAGEMENT ==============

    def _save_to_global_memory(self):
        """Save current pipeline steps to global memory."""
        if hasattr(self, "pipeline_steps"):
            # Update current step parameters before saving
            self._update_current_step_parameters()

            # Deep copy to preserve step state
            self._global_pipeline_memory = []
            for step in self.pipeline_steps:
                # Create a copy of the step with preserved state
                memory_step = PipelineStep(
                    category=step.category,
                    method=step.method,
                    params=step.params.copy() if step.params else {},
                )
                memory_step.enabled = step.enabled  # Preserve enabled state
                self._global_pipeline_memory.append(memory_step)

    def _restore_global_pipeline_memory(self):
        """Restore pipeline steps from global memory."""
        if hasattr(self, "_global_pipeline_memory") and self._global_pipeline_memory:

            # Restore pipeline steps from memory
            self.pipeline_steps = []
            for memory_step in self._global_pipeline_memory:
                # Create a copy of the memory step
                restored_step = PipelineStep(
                    category=memory_step.category,
                    method=memory_step.method,
                    params=memory_step.params.copy() if memory_step.params else {},
                )
                restored_step.enabled = memory_step.enabled  # Restore enabled state
                self.pipeline_steps.append(restored_step)

            # Rebuild the UI with restored steps
            self._rebuild_pipeline_ui()

    def _update_current_step_parameters(self):
        """Update current step parameters from the parameter widget before saving."""
        try:
            if self.current_step_widget and hasattr(
                self.current_step_widget, "get_parameters"
            ):
                current_row = self.pipeline_list.currentRow()
                if 0 <= current_row < len(self.pipeline_steps):
                    updated_params = self.current_step_widget.get_parameters()
                    self.pipeline_steps[current_row].params = updated_params

        except Exception as e:
            pass

    def _clear_global_memory(self):
        """Clear the global pipeline memory."""
        self._global_pipeline_memory = []

    def _rebuild_pipeline_ui(self):
        """Rebuild the pipeline UI with current steps."""
        self.pipeline_list.clear()
        for i, step in enumerate(self.pipeline_steps):
            # Create list item
            step_item = QListWidgetItem()
            step_item.setData(Qt.ItemDataRole.UserRole, i)
            self.pipeline_list.addItem(step_item)

            # Create and set custom widget
            step_widget = PipelineStepWidget(step, i)
            step_widget.toggled.connect(self.on_step_toggled)
            step_item.setSizeHint(step_widget.sizeHint())
            self.pipeline_list.setItemWidget(step_item, step_widget)
