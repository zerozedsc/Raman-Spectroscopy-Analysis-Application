from .__utils__ import *


class PipelineStep:
    """Represents a step in the preprocessing pipeline."""

    def __init__(
        self,
        category: str,
        method: str,
        params: Dict[str, Any] = None,
        source_dataset: str = None,
    ):
        self.category = category
        self.method = method
        self.params = params or {}
        self.enabled = True
        self.source_dataset = source_dataset  # Track which dataset this step came from

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "category": self.category,
            "method": self.method,
            "params": self.params,
            "enabled": self.enabled,
        }
        if self.source_dataset:
            result["source_dataset"] = self.source_dataset
        return result

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], source_dataset: str = None
    ) -> "PipelineStep":
        step = cls(
            data["category"],
            data["method"],
            data.get("params", {}),
            source_dataset or data.get("source_dataset"),
        )
        step.enabled = data.get("enabled", True)
        return step

    def create_instance(self) -> Any:
        """Create the actual preprocessing instance."""
        return PREPROCESSING_REGISTRY.create_method_instance(
            self.category, self.method, self.params
        )

    def get_display_name(self) -> str:
        """Get display name for the step."""
        category_name = LOCALIZE(f"PREPROCESS.CATEGORY.{self.category.upper()}")
        return f"{category_name} - {self.method}"


class PipelineConfirmationDialog(QDialog):
    """Redesigned compact confirmation dialog with medical theme."""

    def __init__(
        self,
        pipeline_steps: List[PipelineStep],
        output_name: str,
        selected_datasets: List[str],
        parent=None,
    ):
        super().__init__(parent)
        self.pipeline_steps = pipeline_steps
        self.output_name = output_name
        self.selected_datasets = selected_datasets
        self.setModal(True)
        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        self.setWindowTitle(LOCALIZE("PREPROCESS.CONFIRM_DIALOG.title"))
        self.setMinimumSize(700, 600)
        self.setMaximumSize(900, 800)
        self.resize(750, 650)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(20)

        # === COMPACT HEADER SECTION ===
        header_frame = QFrame()
        header_frame.setObjectName("headerCard")
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(
            20, 12, 20, 12
        )  # Further reduced vertical padding
        header_layout.setSpacing(10)  # Reduced spacing

        # Title with icon (single row, no metrics)
        title_row = QHBoxLayout()
        title_row.setSpacing(10)

        icon_label = QLabel("🔬")
        icon_label.setStyleSheet("font-size: 22px;")
        title_row.addWidget(icon_label)

        title_label = QLabel(LOCALIZE("PREPROCESS.CONFIRM_DIALOG.header"))
        title_label.setObjectName("titleLabel")
        title_row.addWidget(title_label)
        title_row.addStretch()

        # Add counts inline in title
        counts_label = QLabel(
            f"{len(self.selected_datasets)} datasets • {len(self.pipeline_steps)} steps"
        )
        counts_label.setStyleSheet(
            "font-size: 12px; color: #7f8c8d; font-weight: normal;"
        )
        title_row.addWidget(counts_label)

        header_layout.addLayout(title_row)

        # === COMPACT OUTPUT NAME DISPLAY ===
        output_frame = QFrame()
        output_frame.setObjectName("outputNameFrame")
        output_layout = QHBoxLayout(output_frame)
        output_layout.setContentsMargins(
            12, 8, 12, 8
        )  # Further reduced vertical padding
        output_layout.setSpacing(10)

        output_icon = QLabel("💾")
        output_icon.setStyleSheet("font-size: 16px;")  # Smaller icon
        output_layout.addWidget(output_icon)

        output_label_text = QLabel(
            LOCALIZE("PREPROCESS.CONFIRM_DIALOG.output_label") + ":"
        )
        output_label_text.setObjectName("outputLabelText")
        output_layout.addWidget(output_label_text)

        output_value = QLabel(self.output_name)
        output_value.setObjectName("outputValue")
        output_value.setWordWrap(True)
        output_layout.addWidget(output_value, 1)

        header_layout.addWidget(output_frame)
        content_layout.addWidget(header_frame)

        # === INPUT DATASETS SECTION ===
        input_section = self._create_compact_section(
            "📊 " + LOCALIZE("PREPROCESS.CONFIRM_DIALOG.input_section_title"),
            self._create_datasets_content(),
        )
        content_layout.addWidget(input_section)

        # === PIPELINE SECTION ===
        pipeline_section = self._create_compact_section(
            "⚙️ " + LOCALIZE("PREPROCESS.CONFIRM_DIALOG.pipeline_section_title"),
            self._create_pipeline_content(),
        )
        content_layout.addWidget(pipeline_section)

        content_layout.addStretch()
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        # === BUTTON BAR ===
        button_bar = QFrame()
        button_bar.setObjectName("buttonBar")
        button_layout = QHBoxLayout(button_bar)
        button_layout.setContentsMargins(24, 16, 24, 16)
        button_layout.setSpacing(12)

        button_layout.addStretch()

        cancel_btn = QPushButton(LOCALIZE("COMMON.cancel"))
        cancel_btn.setObjectName("cancelButton")
        cancel_btn.setMinimumWidth(120)
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)

        start_btn = QPushButton(
            "▶ " + LOCALIZE("PREPROCESS.CONFIRM_DIALOG.start_processing")
        )
        start_btn.setObjectName("startButton")
        start_btn.setMinimumWidth(160)
        start_btn.setCursor(Qt.PointingHandCursor)
        start_btn.clicked.connect(self.accept)

        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(start_btn)

        main_layout.addWidget(button_bar)

    def _create_metric_item(self, icon: str, value: str, label: str) -> QFrame:
        """Create a compact metric display item with icon, value, and label."""
        metric = QFrame()
        metric.setObjectName("metricItem")
        metric_layout = QVBoxLayout(metric)
        metric_layout.setContentsMargins(12, 10, 12, 10)  # Reduced from 16, 12, 16, 12
        metric_layout.setSpacing(4)  # Reduced from 6

        # Icon row
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 18px;")  # Reduced from 20px
        icon_label.setAlignment(Qt.AlignCenter)
        metric_layout.addWidget(icon_label)

        # Value (large and prominent)
        value_label = QLabel(value)
        value_label.setObjectName("metricValue")
        value_label.setAlignment(Qt.AlignCenter)
        metric_layout.addWidget(value_label)

        # Label (small description)
        label_widget = QLabel(label)
        label_widget.setObjectName("metricLabel")
        label_widget.setAlignment(Qt.AlignCenter)
        label_widget.setWordWrap(True)
        metric_layout.addWidget(label_widget)

        return metric

    def _create_compact_section(self, title: str, content_widget: QWidget) -> QFrame:
        """Create a compact section with title and content."""
        section = QFrame()
        section.setObjectName("contentCard")
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(16, 14, 16, 14)
        section_layout.setSpacing(12)

        # Title
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        section_layout.addWidget(title_label)

        # Content
        section_layout.addWidget(content_widget)

        return section

    def _create_datasets_content(self) -> QWidget:
        """Create datasets list with checkboxes and output options."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Add output grouping options (only if multiple datasets)
        if len(self.selected_datasets) > 1:
            options_frame = QFrame()
            options_frame.setObjectName("optionsFrame")
            options_layout = QVBoxLayout(options_frame)
            options_layout.setContentsMargins(12, 10, 12, 10)
            options_layout.setSpacing(10)

            options_label = QLabel(
                "📦 " + LOCALIZE("PREPROCESS.CONFIRM_DIALOG.output_options_label")
            )
            options_label.setStyleSheet(
                "font-size: 13px; font-weight: 600; color: #2c3e50;"
            )
            options_layout.addWidget(options_label)

            # Store output mode selection
            self.output_mode_group = QButtonGroup(content)

            # Option 1: Combined output (default)
            self.combined_radio = QRadioButton(
                LOCALIZE("PREPROCESS.CONFIRM_DIALOG.output_combined")
            )
            self.combined_radio.setChecked(True)
            self.combined_radio.setStyleSheet("font-size: 12px; color: #495057;")
            self.output_mode_group.addButton(self.combined_radio, 0)
            options_layout.addWidget(self.combined_radio)

            hint_combined = QLabel(
                LOCALIZE("PREPROCESS.CONFIRM_DIALOG.output_combined_hint")
            )
            hint_combined.setStyleSheet(
                "font-size: 11px; color: #6c757d; padding-left: 24px; font-style: italic;"
            )
            hint_combined.setWordWrap(True)
            options_layout.addWidget(hint_combined)

            # Option 2: Separate outputs
            self.separate_radio = QRadioButton(
                LOCALIZE("PREPROCESS.CONFIRM_DIALOG.output_separate")
            )
            self.separate_radio.setStyleSheet("font-size: 12px; color: #495057;")
            self.output_mode_group.addButton(self.separate_radio, 1)
            options_layout.addWidget(self.separate_radio)

            hint_separate = QLabel(
                LOCALIZE("PREPROCESS.CONFIRM_DIALOG.output_separate_hint")
            )
            hint_separate.setStyleSheet(
                "font-size: 11px; color: #6c757d; padding-left: 24px; font-style: italic;"
            )
            hint_separate.setWordWrap(True)
            options_layout.addWidget(hint_separate)

            layout.addWidget(options_frame)

            # Add spacing
            layout.addSpacing(8)

        # Dataset list with checkboxes (all checked by default, read-only for now)
        datasets_label = QLabel(
            "📂 " + LOCALIZE("PREPROCESS.CONFIRM_DIALOG.selected_datasets_label")
        )
        datasets_label.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: #2c3e50;"
        )
        layout.addWidget(datasets_label)

        # Show all datasets with checkboxes
        self.dataset_checkboxes = []
        for i, dataset in enumerate(self.selected_datasets):
            item_frame = QFrame()
            item_frame.setObjectName("listItem")
            item_layout = QHBoxLayout(item_frame)
            item_layout.setContentsMargins(10, 8, 10, 8)
            item_layout.setSpacing(10)

            # Checkbox (checked and enabled)
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox.setStyleSheet("font-size: 13px;")
            self.dataset_checkboxes.append(checkbox)
            item_layout.addWidget(checkbox)

            # Dataset name with number
            item_label = QLabel(f"{i+1}. {dataset}")
            item_label.setStyleSheet("font-size: 13px; color: #2c3e50;")
            item_layout.addWidget(item_label, 1)

            layout.addWidget(item_frame)

        return content

    def get_selected_datasets(self) -> List[str]:
        """Get list of datasets that are checked."""
        return [
            dataset
            for i, dataset in enumerate(self.selected_datasets)
            if self.dataset_checkboxes[i].isChecked()
        ]

    def get_output_mode(self) -> str:
        """Get the selected output mode: 'combined' or 'separate'."""
        if len(self.selected_datasets) == 1:
            return "single"

        if hasattr(self, "output_mode_group"):
            if self.output_mode_group.checkedId() == 0:
                return "combined"
            elif self.output_mode_group.checkedId() == 1:
                return "separate"

        return "combined"  # Default

    def _create_pipeline_content(self) -> QWidget:
        """Create compact pipeline steps list."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        for i, step in enumerate(self.pipeline_steps):
            step_frame = self._create_pipeline_step_item(i + 1, step)
            layout.addWidget(step_frame)

        return content

    def _create_pipeline_step_item(
        self, step_number: int, step: PipelineStep
    ) -> QFrame:
        """Create a single pipeline step item with visual flow."""
        item = QFrame()
        item.setObjectName("pipelineItem")
        item_layout = QVBoxLayout(item)
        item_layout.setContentsMargins(12, 10, 12, 10)
        item_layout.setSpacing(6)

        # Step header with number and category
        header_layout = QHBoxLayout()

        # Step number badge
        number_label = QLabel(str(step_number))
        number_label.setObjectName("stepNumber")
        number_label.setFixedSize(28, 28)
        number_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(number_label)

        # Category and method
        category_display = LOCALIZE(f"PREPROCESS.CATEGORY.{step.category.upper()}")
        step_info_layout = QVBoxLayout()
        step_info_layout.setSpacing(2)

        method_label = QLabel(f"<b>{step.method}</b>")
        method_label.setStyleSheet("font-size: 13px; color: #1976d2;")
        step_info_layout.addWidget(method_label)

        category_label = QLabel(category_display)
        category_label.setStyleSheet("font-size: 11px; color: #6c757d;")
        step_info_layout.addWidget(category_label)

        header_layout.addLayout(step_info_layout)
        header_layout.addStretch()

        item_layout.addLayout(header_layout)

        # Parameters (if any)
        if step.params:
            params_text = self._format_parameters_compact(step.params)
            params_label = QLabel(f"<i>{params_text}</i>")
            params_label.setStyleSheet(
                "font-size: 11px; color: #757575; padding-left: 36px;"
            )
            params_label.setWordWrap(True)
            item_layout.addWidget(params_label)

        return item

    def _format_parameters_compact(self, params: Dict[str, Any]) -> str:
        """Format parameters in a compact way."""
        if not params:
            return ""

        formatted = []
        for key, value in list(params.items())[:4]:  # Show max 4 params
            if isinstance(value, float):
                formatted.append(f"{key}={value:.2f}")
            elif isinstance(value, (tuple, list)):
                formatted.append(
                    f"{key}=({', '.join(map(str, value[:2]))}...)"
                    if len(value) > 2
                    else f"{key}={value}"
                )
            elif isinstance(value, str) and len(str(value)) > 20:
                formatted.append(f"{key}={str(value)[:20]}...")
            else:
                formatted.append(f"{key}={value}")

        result = " · ".join(formatted)
        if len(params) > 4:
            result += f" · +{len(params) - 4} more"
        return result

    def _apply_styles(self):
        """Apply modern medical theme styling."""
        self.setStyleSheet(
            """
            QDialog {
                background-color: #f0f4f8;
            }
            
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            
            /* Header Card */
            #headerCard {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f8fbff);
                border: 1px solid #d0dae6;
                border-radius: 10px;
            }
            
            #titleLabel {
                font-size: 17px;  /* Reduced from 20px for more compact header */
                font-weight: 600;
                color: #1a365d;
                letter-spacing: -0.5px;
            }
            
            /* Metric Items */
            #metricItem {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #fafcfe);
                border: 1px solid #e1e8ed;
                border-radius: 8px;
                min-width: 140px;
                max-width: 200px;
            }
            
            #metricItem:hover {
                border-color: #90caf9;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fbff, stop:1 #e3f2fd);
            }
            
            #metricValue {
                font-size: 24px;
                font-weight: 700;
                color: #0078d4;
                letter-spacing: -0.5px;
            }
            
            #metricLabel {
                font-size: 11px;
                font-weight: 500;
                color: #6c757d;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            /* Output Name Frame - Prominent Display */
            #outputNameFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e8f5e9, stop:1 #c8e6c9);
                border: 2px solid #4caf50;
                border-radius: 8px;
                padding: 12px 16px;
            }
            
            #outputLabelText {
                font-size: 13px;
                font-weight: 600;
                color: #2e7d32;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            #outputValue {
                font-size: 16px;
                font-weight: 700;
                color: #1b5e20;
                letter-spacing: -0.3px;
            }
            
            /* Options Frame */
            #optionsFrame {
                background-color: #fff3e0;
                border: 1px solid #ffb74d;
                border-left: 4px solid #ff9800;
                border-radius: 6px;
                padding: 10px 12px;
            }
            
            QRadioButton {
                color: #495057;
                font-size: 12px;
                spacing: 8px;
            }
            
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid #ced4da;
                background-color: white;
            }
            
            QRadioButton::indicator:hover {
                border-color: #ff9800;
            }
            
            QRadioButton::indicator:checked {
                background-color: #ff9800;
                border-color: #ff9800;
            }
            
            QRadioButton::indicator:checked::before {
                content: "";
                width: 8px;
                height: 8px;
                border-radius: 4px;
                background-color: white;
            }
            
            QCheckBox {
                color: #2c3e50;
                font-size: 13px;
                spacing: 8px;
            }
            
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #ced4da;
                border-radius: 4px;
                background-color: white;
            }
            
            QCheckBox::indicator:hover {
                border-color: #0078d4;
            }
            
            QCheckBox::indicator:checked {
                background-color: #28a745;
                border-color: #28a745;
                image: url(assets/icons/checkmark.svg);
            }
            
            /* Content Cards */
            #contentCard {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 10px;
                padding: 14px 16px;
            }
            
            #sectionTitle {
                font-size: 15px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 4px;
            }
            
            /* List Items */
            #listItem {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 6px;
                padding: 8px 10px;
            }
            
            #listItem:hover {
                background-color: #e9ecef;
                border-color: #dee2e6;
            }
            
            /* Pipeline Items */
            #pipelineItem {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f8f9fa);
                border: 1px solid #e3f2fd;
                border-left: 4px solid #1976d2;
                border-radius: 6px;
                padding: 10px 12px;
            }
            
            #pipelineItem:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f0f8ff, stop:1 #e3f2fd);
                border-left-color: #0d47a1;
            }
            
            #stepNumber {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1976d2, stop:1 #1565c0);
                color: white;
                border-radius: 14px;
                font-weight: 700;
                font-size: 13px;
            }
            
            /* Button Bar */
            #buttonBar {
                background-color: white;
                border: none;
                border-top: 2px solid #e9ecef;
                border-radius: 0px;
            }
            
            #startButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4caf50, stop:1 #388e3c);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 14px;
            }
            
            #startButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #66bb6a, stop:1 #4caf50);
            }
            
            #startButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #388e3c, stop:1 #2e7d32);
            }
            
            #cancelButton {
                background-color: #f5f5f5;
                color: #424242;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 500;
                font-size: 14px;
            }
            
            #cancelButton:hover {
                background-color: #e9ecef;
                border-color: #ced4da;
            }
            
            #cancelButton:pressed {
                background-color: #dee2e6;
            }
            
            /* Scrollbar */
            QScrollBar:vertical {
                background-color: #f8f9fa;
                width: 10px;
                border-radius: 5px;
                margin: 0px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #ced4da;
                border-radius: 5px;
                min-height: 30px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #adb5bd;
            }
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """
        )

    def _generate_benefits_content(self) -> str:
        """Generate HTML content describing expected benefits."""
        categories_used = set(step.category for step in self.pipeline_steps)

        html_content = "<ul style='margin-left: 20px;'>"
        for category in categories_used:
            title = LOCALIZE(f"PREPROCESS.BENEFITS.{category.upper()}.title")
            benefits = LOCALIZE(f"PREPROCESS.BENEFITS.{category.upper()}.benefits")

            html_content += f"<li><b>{title}</b><ul>"
            for benefit in benefits:
                html_content += f"<li>{benefit}</li>"
            html_content += "</ul></li>"

        html_content += "</ul>"

        # Add overall benefits
        overall_benefits = LOCALIZE("PREPROCESS.BENEFITS.overall")
        html_content += f"<br><b>{LOCALIZE('PREPROCESS.BENEFITS.overall_title')}:</b><ul style='margin-left: 20px;'>"
        for benefit in overall_benefits:
            html_content += f"<li>{benefit}</li>"
        html_content += "</ul>"

        return html_content

    def _calculate_quality_metrics(self) -> List[str]:
        """Calculate expected quality improvements."""
        metrics = []
        categories_used = set(step.category for step in self.pipeline_steps)

        for category in categories_used:
            metric = LOCALIZE(f"PREPROCESS.METRICS.{category.upper()}")
            if metric:
                metrics.append(metric)

        # Overall metrics
        metrics.extend(
            [
                LOCALIZE(
                    "PREPROCESS.METRICS.total_steps", steps=len(self.pipeline_steps)
                ),
                LOCALIZE("PREPROCESS.METRICS.data_integrity"),
                LOCALIZE("PREPROCESS.METRICS.reproducibility"),
            ]
        )

        return metrics


class FailedStepsDialog(QDialog):
    """Dialog to show detailed information about failed preprocessing steps."""

    def __init__(
        self, failed_steps: List[Dict], successful_steps: List[Dict], parent=None
    ):
        super().__init__(parent)
        self.failed_steps = failed_steps
        self.successful_steps = successful_steps
        self.setModal(True)
        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        self.setWindowTitle(LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.title"))
        self.setMinimumSize(700, 500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Header
        header_label = QLabel(LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.header"))
        header_label.setObjectName("failedDialogHeader")
        layout.addWidget(header_label)

        # Summary
        summary_text = LOCALIZE(
            "PREPROCESS.FAILED_STEPS_DIALOG.summary",
            failed=len(self.failed_steps),
            successful=len(self.successful_steps),
            total=len(self.failed_steps) + len(self.successful_steps),
        )
        summary_label = QLabel(summary_text)
        summary_label.setObjectName("failedDialogSummary")
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        # Failed steps details
        if self.failed_steps:
            failed_group = QGroupBox(
                LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.failed_steps_title")
            )
            failed_layout = QVBoxLayout(failed_group)

            failed_tree = QTreeWidget()
            failed_tree.setHeaderLabels(
                [
                    LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.step"),
                    LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.category"),
                    LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.error_type"),
                    LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.error_message"),
                ]
            )

            for step in self.failed_steps:
                item = QTreeWidgetItem(
                    [
                        f"{step['step_index']}. {step['step_name']}",
                        LOCALIZE(f"PREPROCESS.CATEGORY.{step['category'].upper()}"),
                        step["error_type"],
                        (
                            step["error"][:100] + "..."
                            if len(step["error"]) > 100
                            else step["error"]
                        ),
                    ]
                )
                item.setToolTip(3, step["error"])  # Full error message in tooltip
                failed_tree.addTopLevelItem(item)

            failed_tree.expandAll()
            for i in range(4):
                failed_tree.resizeColumnToContents(i)

            failed_layout.addWidget(failed_tree)
            layout.addWidget(failed_group)

        # Successful steps summary
        if self.successful_steps:
            success_group = QGroupBox(
                LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.successful_steps_title")
            )
            success_layout = QVBoxLayout(success_group)

            success_list = QListWidget()
            success_list.setMaximumHeight(150)

            for step in self.successful_steps:
                category_key = f"PREPROCESS.CATEGORY.{step['category'].upper()}"
                category_display = LOCALIZE(category_key)
                item_text = (
                    f"✓ {step['step_index']}. {step['step_name']} ({category_display})"
                )
                item = QListWidgetItem(item_text)
                success_list.addItem(item)

            success_layout.addWidget(success_list)
            layout.addWidget(success_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_btn = QPushButton(LOCALIZE("COMMON.close"))
        close_btn.setObjectName("closeButton")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addWidget(QWidget())  # Spacer
        layout.addLayout(button_layout)

    def _apply_styles(self):
        """Apply styling to the failed steps dialog."""
        self.setStyleSheet(
            """
            QDialog {
                background-color: #f8f9fa;
            }
            
            #failedDialogHeader {
                font-size: 18px;
                font-weight: 700;
                color: #d32f2f;
                margin-bottom: 8px;
            }
            
            #failedDialogSummary {
                font-size: 14px;
                color: #424242;
                background-color: #fff3e0;
                border: 1px solid #ffcc02;
                border-radius: 6px;
                padding: 12px;
                margin-bottom: 16px;
            }
            
            QGroupBox {
                font-size: 14px;
                font-weight: 600;
                color: #424242;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 12px;
                background-color: white;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                background-color: white;
            }
            
            QTreeWidget, QListWidget {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background-color: white;
            }
            
            QTreeWidget::item, QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f0f0f0;
            }
            
            #closeButton {
                background-color: #1976d2;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: 600;
                min-width: 100px;
            }
            
            #closeButton:hover {
                background-color: #1565c0;
            }
        """
        )


class PipelineStepWidget(QWidget):
    """
    Custom widget for displaying pipeline steps with modern theme integration.

    Provides visual representation of preprocessing pipeline steps with enable/disable
    toggle functionality, status indicators, and theme-consistent styling that matches
    the application's design system.

    Features:
    - Eye toggle button for enable/disable
    - Status indicators for new vs existing steps
    - Checkbox for existing step reuse control
    - Hover effects and visual feedback
    - Color coding based on step state and source

    Args:
        step (PipelineStep): The pipeline step data object
        step_index (int): Index position in the pipeline
        parent (QWidget, optional): Parent widget. Defaults to None.

    Signals:
        toggled (int, bool): Emitted when step state changes (step_index, enabled)

    Use in:
        - pages/preprocess_page.py: PreprocessPage.add_pipeline_step()
        - pages/preprocess_page_utils/pipeline.py: Pipeline management

    Example:
        >>> step = PipelineStep("Noise Removal", "SavGol")
        >>> widget = PipelineStepWidget(step, 0)
        >>> widget.toggled.connect(self.on_step_toggled)

    Note:
        Widget automatically updates appearance based on step state.
        Follows app theme colors: #0078d4 (primary), #28a745 (success), #dc3545 (danger)
    """

    toggled = Signal(int, bool)  # step_index, enabled

    def __init__(self, step: "PipelineStep", step_index: int, parent=None):
        super().__init__(parent)
        self.step = step
        self.step_index = step_index
        self.is_selected = False  # Track selection state
        self._setup_ui()

    def set_selected(self, selected: bool):
        """Set the selection state and update appearance."""
        self.is_selected = selected
        self._update_appearance()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        # Add enable/disable toggle button with eye icon
        from components.widgets.icons import load_icon

        self.enable_toggle_btn = QPushButton()
        self.enable_toggle_btn.setFixedSize(24, 24)
        self.enable_toggle_btn.setIconSize(QSize(16, 16))
        self.enable_toggle_btn.setFlat(True)
        self.enable_toggle_btn.clicked.connect(self._toggle_enabled)
        self._update_enable_button()
        layout.addWidget(self.enable_toggle_btn)

        # Toggle checkbox for existing steps
        if hasattr(self.step, "is_existing") and self.step.is_existing:
            self.toggle_checkbox = QCheckBox()
            self.toggle_checkbox.setChecked(
                False
            )  # Default: don't reuse existing steps
            self.toggle_checkbox.setToolTip(
                LOCALIZE("PREPROCESS.toggle_existing_step_tooltip")
            )
            self.toggle_checkbox.toggled.connect(self._on_toggled)
            self.toggle_checkbox.setStyleSheet(
                """
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                    border: 2px solid #ced4da;
                    border-radius: 3px;
                    background-color: white;
                }
                QCheckBox::indicator:checked {
                    background-color: #0078d4;
                    border-color: #0078d4;
                }
                QCheckBox::indicator:checked:hover {
                    background-color: #106ebe;
                    border-color: #106ebe;
                }
            """
            )
            layout.addWidget(self.toggle_checkbox)

            # Visual indicator for existing steps (modern icon)
            status_label = QLabel("⚙")
            status_label.setStyleSheet("font-size: 14px; color: #6c757d;")
            status_label.setToolTip(LOCALIZE("PREPROCESS.existing_step_indicator"))
            layout.addWidget(status_label)
        else:
            # Visual indicator for new steps (modern icon)
            status_label = QLabel("✚")
            status_label.setStyleSheet(
                "font-size: 14px; color: #28a745; font-weight: bold;"
            )
            status_label.setToolTip(LOCALIZE("PREPROCESS.new_step_indicator"))
            layout.addWidget(status_label)

        # Step name label
        display_name = self.step.get_display_name()
        self.name_label = QLabel(display_name)
        self.name_label.setWordWrap(False)
        self.name_label.setTextFormat(Qt.PlainText)
        self.name_label.setMinimumWidth(200)
        self.name_label.setStyleSheet(
            """
            QLabel {
                font-size: 13px; 
                color: #2c3e50; 
                font-weight: 500;
                padding: 2px 0px;
            }
        """
        )
        layout.addWidget(self.name_label, 1)

        # Set widget styling to match app theme
        self.setMinimumHeight(40)
        # Note: Main styling is applied in _update_appearance() to avoid conflicts

        # Update appearance based on step status
        self._update_appearance()

    def _on_toggled(self, checked: bool):
        """Handle toggle state change."""
        self.toggled.emit(self.step_index, checked)
        self._update_appearance()

    def _toggle_enabled(self):
        """Toggle the enabled state of the preprocessing step."""
        self.step.enabled = not self.step.enabled
        self._update_enable_button()
        self._update_appearance()
        # Emit signal to notify parent that step state changed
        self.toggled.emit(self.step_index, self.step.enabled)

    def _update_enable_button(self):
        """Update the enable/disable button icon and tooltip."""
        from components.widgets.icons import load_icon

        if self.step.enabled:
            # Step is enabled, show eye_open icon (indicating it's visible/enabled)
            icon = load_icon("eye_open", "button")
            tooltip = LOCALIZE("PREPROCESS.disable_step_tooltip")
            button_style = """
                QPushButton {
                    background-color: #e8f5e8;
                    border: 1px solid #28a745;
                    border-radius: 12px;
                }
                QPushButton:hover {
                    background-color: #28a745;
                }
            """
        else:
            # Step is disabled, show eye_close icon (indicating it's hidden/disabled)
            icon = load_icon("eye_close", "button")
            tooltip = LOCALIZE("PREPROCESS.enable_step_tooltip")
            button_style = """
                QPushButton {
                    background-color: #f8d7da;
                    border: 1px solid #dc3545;
                    border-radius: 12px;
                }
                QPushButton:hover {
                    background-color: #dc3545;
                }
            """

        self.enable_toggle_btn.setIcon(icon)
        self.enable_toggle_btn.setToolTip(tooltip)
        self.enable_toggle_btn.setStyleSheet(button_style)

    def _update_appearance(self):
        """Update visual appearance based on step state."""
        widget_style = ""
        text_style = ""

        # Base widget styling
        base_widget_style = """
            QWidget {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                margin: 1px;
            }
            QWidget:hover {
                background-color: #f8f9fa;
                border-color: #adb5bd;
            }
        """

        # Apply enabled/disabled styling
        if not self.step.enabled:
            # Disabled step - muted appearance
            widget_style = """
                QWidget {
                    background-color: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 6px;
                    margin: 1px;
                }
                QWidget:hover {
                    background-color: #e9ecef;
                    border-color: #ced4da;
                }
            """
            text_style = """
                QLabel {
                    font-size: 13px;
                    color: #6c757d;
                    font-weight: 400;
                    font-style: italic;
                    padding: 2px 0px;
                }
            """
            tooltip_suffix = f" ({LOCALIZE('PREPROCESS.step_disabled')})"
        else:
            tooltip_suffix = f" ({LOCALIZE('PREPROCESS.step_enabled')})"

        # Add source dataset info to tooltip if available
        if hasattr(self.step, "source_dataset") and self.step.source_dataset:
            tooltip_suffix += f" (From: {self.step.source_dataset})"

        # Apply specific styling based on step type
        if hasattr(self.step, "is_existing") and self.step.is_existing:
            if hasattr(self, "toggle_checkbox") and self.toggle_checkbox.isChecked():
                # Existing step enabled for reuse
                if self.step.enabled:
                    # Check if this step is from another dataset
                    if (
                        hasattr(self.step, "source_dataset")
                        and self.step.source_dataset
                    ):
                        # Step from another dataset - blue accent
                        widget_style = """
                            QWidget {
                                background-color: #e3f2fd;
                                border: 1px solid #1976d2;
                                border-radius: 6px;
                                margin: 1px;
                            }
                            QWidget:hover {
                                background-color: #e1f5fe;
                                border-color: #1565c0;
                            }
                        """
                        text_style = """
                            QLabel {
                                font-size: 13px;
                                color: #1976d2;
                                font-weight: 600;
                                padding: 2px 0px;
                            }
                        """
                    else:
                        # Step from current dataset - green accent
                        widget_style = """
                            QWidget {
                                background-color: #e8f5e8;
                                border: 1px solid #28a745;
                                border-radius: 6px;
                                margin: 1px;
                            }
                            QWidget:hover {
                                background-color: #d4edda;
                                border-color: #1e7e34;
                            }
                        """
                        text_style = """
                            QLabel {
                                font-size: 13px;
                                color: #155724;
                                font-weight: 600;
                                padding: 2px 0px;
                            }
                        """
                else:
                    # Use disabled styles if not enabled
                    pass
                self.setToolTip(
                    LOCALIZE("PREPROCESS.existing_step_enabled_tooltip")
                    + tooltip_suffix
                )
            else:
                # Existing step disabled (default) - muted existing step style
                widget_style = """
                    QWidget {
                        background-color: #f8f9fa;
                        border: 1px solid #e9ecef;
                        border-radius: 6px;
                        margin: 1px;
                        border-style: dashed;
                    }
                    QWidget:hover {
                        background-color: #e9ecef;
                        border-color: #ced4da;
                    }
                """
                if hasattr(self.step, "source_dataset") and self.step.source_dataset:
                    # Step from another dataset - muted blue
                    text_style = """
                        QLabel {
                            font-size: 13px;
                            color: #64b5f6;
                            font-weight: 400;
                            font-style: italic;
                            padding: 2px 0px;
                        }
                    """
                else:
                    # Step from current dataset - muted gray
                    text_style = """
                        QLabel {
                            font-size: 13px;
                            color: #6c757d;
                            font-weight: 400;
                            font-style: italic;
                            padding: 2px 0px;
                        }
                    """
                self.setToolTip(
                    LOCALIZE("PREPROCESS.existing_step_disabled_tooltip")
                    + tooltip_suffix
                )
        else:
            # New step - primary theme colors
            if self.step.enabled:
                widget_style = """
                    QWidget {
                        background-color: white;
                        border: 1px solid #0078d4;
                        border-radius: 6px;
                        margin: 1px;
                    }
                    QWidget:hover {
                        background-color: #f0f8ff;
                        border-color: #005a9e;
                    }
                """
                text_style = """
                    QLabel {
                        font-size: 13px;
                        color: #0078d4;
                        font-weight: 600;
                        padding: 2px 0px;
                    }
                """
            else:
                # Use base disabled styles
                pass
            self.setToolTip(LOCALIZE("PREPROCESS.new_step_tooltip") + tooltip_suffix)

        # Apply the final styles
        if widget_style:
            self.setStyleSheet(widget_style)
        else:
            self.setStyleSheet(base_widget_style)

        # Override with selection styling if selected - subtle gray border
        if self.is_selected:
            # Selected state - keep current background but add prominent gray border
            # Determine base background color from step state
            if not self.step.enabled:
                bg_color = "#f8f9fa"  # Disabled background
            elif hasattr(self.step, "is_existing") and self.step.is_existing:
                if (
                    hasattr(self, "toggle_checkbox")
                    and self.toggle_checkbox.isChecked()
                ):
                    if (
                        hasattr(self.step, "source_dataset")
                        and self.step.source_dataset
                    ):
                        bg_color = "#e3f2fd"  # Blue for external
                    else:
                        bg_color = "#e8f5e8"  # Green for current
                else:
                    bg_color = "#f8f9fa"  # Muted
            else:
                bg_color = "white"  # New step

            selected_style = f"""
                QWidget {{
                    background-color: {bg_color};
                    border: 2px solid #6c757d;
                    border-radius: 6px;
                    margin: 1px;
                }}
                QWidget:hover {{
                    background-color: {bg_color};
                    border-color: #495057;
                }}
            """
            self.setStyleSheet(selected_style)

        # Apply text styling
        if text_style:
            self.name_label.setStyleSheet(text_style)
        else:
            # Default enabled style
            self.name_label.setStyleSheet(
                """
                QLabel {
                    font-size: 13px; 
                    color: #2c3e50; 
                    font-weight: 500;
                    padding: 2px 0px;
                }
            """
            )

    def is_enabled(self) -> bool:
        """Check if step is enabled for processing."""
        if hasattr(self.step, "is_existing") and self.step.is_existing:
            return hasattr(self, "toggle_checkbox") and self.toggle_checkbox.isChecked()
        return True  # New steps are always enabled

    def set_enabled(self, enabled: bool):
        """Set the enabled state of the step."""
        if hasattr(self, "toggle_checkbox"):
            self.toggle_checkbox.setChecked(enabled)
