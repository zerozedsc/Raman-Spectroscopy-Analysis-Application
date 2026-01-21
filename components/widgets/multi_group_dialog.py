"""components.widgets.multi_group_dialog

Reusable dialog for creating multiple groups at once with include/exclude keyword patterns.

This was originally implemented under `pages/analysis_page_utils/multi_group_dialog.py` and moved
here so it can be reused by other pages (e.g. Machine Learning page).

Keyword matching semantics (fixed):
- Include keywords are AND logic: dataset name must contain *all* include keywords.
- Exclude keywords are NOT-ANY logic: dataset name must contain *none* of the exclude keywords.

Keywords are treated as case-insensitive substrings.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from configs.style.stylesheets import BASE_STYLES, PREPROCESS_PAGE_STYLES_2


class MultiGroupCreationDialog(QDialog):
    """Dialog for creating multiple groups at once with keyword matching."""

    def __init__(
        self,
        available_datasets: List[str],
        localize_func: Callable,
        parent=None,
        *,
        initial_rows: int = 1,
        default_auto_assign: bool = True,
        initial_groups: List[Dict] | None = None,
    ):
        super().__init__(parent)
        self.available_datasets = available_datasets
        self.localize_func = localize_func
        self.group_configs: List[Dict] = []
        self._initial_rows = max(0, int(initial_rows))
        self._default_auto_assign = bool(default_auto_assign)
        self._initial_groups = list(initial_groups) if initial_groups else []

        self.setWindowTitle(localize_func("ANALYSIS_PAGE.multi_group_dialog_title"))
        self.setModal(True)
        self.resize(900, 600)

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        # Ensure this dialog inherits the app theme even if created before the global stylesheet
        # is applied (or if OS/theme settings interfere).
        self.setStyleSheet(
            "\n".join(
                [
                    "QDialog, QWidget { background-color: #f8f9fa; color: #2c3e50; }",
                    "\n".join(BASE_STYLES.values()),
                    PREPROCESS_PAGE_STYLES_2,
                ]
            )
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        splitter = QSplitter(Qt.Horizontal)
        left_widget = self._create_group_table_widget()
        splitter.addWidget(left_widget)

        right_widget = self._create_preview_widget()
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, 1)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        preview_btn = QPushButton(self.localize_func("ANALYSIS_PAGE.preview_assignments_button"))
        preview_btn.setObjectName("ctaButton")
        preview_btn.setMinimumHeight(36)
        preview_btn.clicked.connect(self._preview_assignments)
        button_layout.addWidget(preview_btn)

        apply_btn = QPushButton(self.localize_func("ANALYSIS_PAGE.apply_groups_button"))
        apply_btn.setMinimumHeight(36)
        apply_btn.setStyleSheet(BASE_STYLES["success_button"])
        apply_btn.clicked.connect(self._apply_groups)
        button_layout.addWidget(apply_btn)

        cancel_btn = QPushButton(self.localize_func("ANALYSIS_PAGE.cancel_button"))
        cancel_btn.setObjectName("cancelButton")
        cancel_btn.setMinimumHeight(36)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        main_layout.addLayout(button_layout)

    def _create_group_table_widget(self) -> QWidget:
        group_box = QGroupBox(self.localize_func("ANALYSIS_PAGE.group_definitions_title"))
        group_box.setStyleSheet(BASE_STYLES["group_box"])
        layout = QVBoxLayout(group_box)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            [
                self.localize_func("ANALYSIS_PAGE.group_name_column"),
                self.localize_func("ANALYSIS_PAGE.include_keywords_column"),
                self.localize_func("ANALYSIS_PAGE.exclude_keywords_column"),
                self.localize_func("ANALYSIS_PAGE.auto_assign_column"),
            ]
        )

        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)

        self.table.setColumnWidth(0, 150)
        self.table.setColumnWidth(3, 100)

        self.table.setStyleSheet(BASE_STYLES.get("table_widget", ""))
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)

        self.table.setFont(QFont("Segoe UI", 10))
        self.table.horizontalHeader().setMinimumSectionSize(90)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()

        add_row_btn = QPushButton(self.localize_func("ANALYSIS_PAGE.add_row_button"))
        add_row_btn.setStyleSheet(BASE_STYLES["secondary_button"])
        add_row_btn.setMinimumHeight(34)
        add_row_btn.clicked.connect(self._add_row)
        btn_layout.addWidget(add_row_btn)

        remove_row_btn = QPushButton(self.localize_func("ANALYSIS_PAGE.remove_row_button"))
        remove_row_btn.setStyleSheet(BASE_STYLES["secondary_button"])
        remove_row_btn.setMinimumHeight(34)
        remove_row_btn.clicked.connect(self._remove_row)
        btn_layout.addWidget(remove_row_btn)

        btn_layout.addStretch(1)

        layout.addLayout(btn_layout)

        # Seed rows
        seed_count = max(self._initial_rows, len(self._initial_groups))
        for _ in range(seed_count):
            self._add_row()

        # Prefill if provided
        for row, g in enumerate(self._initial_groups):
            if row >= self.table.rowCount():
                break
            try:
                name = str(g.get("name", "") or "")
                include = g.get("include", "")
                exclude = g.get("exclude", "")
                auto_assign = bool(g.get("auto_assign", self._default_auto_assign))

                name_w = self.table.cellWidget(row, 0)
                if isinstance(name_w, QLineEdit):
                    name_w.setText(name)

                inc_w = self.table.cellWidget(row, 1)
                if isinstance(inc_w, QLineEdit):
                    if isinstance(include, list):
                        inc_w.setText(", ".join([str(x) for x in include if str(x).strip()]))
                    else:
                        inc_w.setText(str(include) if include is not None else "")

                exc_w = self.table.cellWidget(row, 2)
                if isinstance(exc_w, QLineEdit):
                    if isinstance(exclude, list):
                        exc_w.setText(", ".join([str(x) for x in exclude if str(x).strip()]))
                    else:
                        exc_w.setText(str(exclude) if exclude is not None else "")

                cb_container = self.table.cellWidget(row, 3)
                if cb_container is not None:
                    cb = cb_container.findChild(QCheckBox)
                    if cb is not None:
                        cb.setChecked(auto_assign)
            except Exception:
                # Prefill should never break dialog creation.
                pass
        return group_box

    def _create_preview_widget(self) -> QWidget:
        group_box = QGroupBox(self.localize_func("ANALYSIS_PAGE.preview_title"))
        group_box.setStyleSheet(BASE_STYLES["group_box"])
        layout = QVBoxLayout(group_box)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self.preview_tree = QTreeWidget()
        self.preview_tree.setHeaderLabels(["Group", "Count"])
        self.preview_tree.setColumnWidth(0, 300)
        self.preview_tree.setStyleSheet(BASE_STYLES.get("tree_widget", ""))
        self.preview_tree.setFont(QFont("Segoe UI", 10))
        layout.addWidget(self.preview_tree)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 12px; color: #6c757d; margin-top: 6px;")
        layout.addWidget(self.status_label)

        return group_box

    def _add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)

        try:
            self.table.setRowHeight(row, 42)
        except Exception:
            pass

        name_edit = QLineEdit()
        name_edit.setPlaceholderText(self.localize_func("ANALYSIS_PAGE.group_name_placeholder"))
        name_edit.setStyleSheet(BASE_STYLES["input_field"])
        name_edit.setMinimumHeight(34)
        self.table.setCellWidget(row, 0, name_edit)

        include_edit = QLineEdit()
        include_edit.setPlaceholderText(self.localize_func("ANALYSIS_PAGE.include_keywords_placeholder"))
        include_edit.setStyleSheet(BASE_STYLES["input_field"])
        include_edit.setMinimumHeight(34)
        self.table.setCellWidget(row, 1, include_edit)

        exclude_edit = QLineEdit()
        exclude_edit.setPlaceholderText(self.localize_func("ANALYSIS_PAGE.exclude_keywords_placeholder"))
        exclude_edit.setStyleSheet(BASE_STYLES["input_field"])
        exclude_edit.setMinimumHeight(34)
        self.table.setCellWidget(row, 2, exclude_edit)

        checkbox = QCheckBox()
        checkbox.setChecked(self._default_auto_assign)
        checkbox_container = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_container)
        checkbox_layout.addWidget(checkbox)
        checkbox_layout.setAlignment(Qt.AlignCenter)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.table.setCellWidget(row, 3, checkbox_container)

    def _remove_row(self):
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)

    def _preview_assignments(self):
        is_valid, error_msg, configs = self._validate_and_get_configs()
        if not is_valid:
            QMessageBox.warning(self, self.localize_func("ANALYSIS_PAGE.validation_error_title"), error_msg)
            return

        assignments = self._match_datasets_to_groups(configs)

        self.preview_tree.clear()
        total_assigned = 0

        for group_name, datasets in sorted(assignments.items()):
            if group_name == "Unassigned":
                continue

            group_item = QTreeWidgetItem([f"🧪 {group_name}", str(len(datasets))])
            font = QFont()
            font.setBold(True)
            group_item.setFont(0, font)
            group_item.setForeground(0, Qt.darkBlue)

            max_per_group = 50
            datasets_to_show = sorted(datasets)[:max_per_group]
            for dataset in datasets_to_show:
                dataset_item = QTreeWidgetItem([f"  📊 {dataset}", ""])
                group_item.addChild(dataset_item)

            if len(datasets) > max_per_group:
                summary_item = QTreeWidgetItem(
                    [
                        "  "
                        + self.localize_func("ANALYSIS_PAGE.and_more_datasets").format(
                            count=len(datasets) - max_per_group
                        ),
                        "",
                    ]
                )
                summary_item.setForeground(0, Qt.gray)
                group_item.addChild(summary_item)

            self.preview_tree.addTopLevelItem(group_item)
            total_assigned += len(datasets)

        if "Unassigned" in assignments and assignments["Unassigned"]:
            unassigned_count = len(assignments["Unassigned"])
            unassigned_item = QTreeWidgetItem(
                [self.localize_func("ANALYSIS_PAGE.unassigned_group"), str(unassigned_count)]
            )
            font = QFont()
            font.setBold(True)
            unassigned_item.setFont(0, font)
            unassigned_item.setForeground(0, Qt.darkRed)

            max_display = 20
            datasets_to_show = sorted(assignments["Unassigned"])[:max_display]
            for dataset in datasets_to_show:
                dataset_item = QTreeWidgetItem([f"  ⚠️ {dataset}", ""])
                unassigned_item.addChild(dataset_item)

            if unassigned_count > max_display:
                summary_item = QTreeWidgetItem(
                    [
                        "  "
                        + self.localize_func("ANALYSIS_PAGE.and_more_datasets").format(
                            count=unassigned_count - max_display
                        ),
                        "",
                    ]
                )
                summary_item.setForeground(0, Qt.gray)
                unassigned_item.addChild(summary_item)

            self.preview_tree.addTopLevelItem(unassigned_item)

        for i in range(self.preview_tree.topLevelItemCount()):
            item = self.preview_tree.topLevelItem(i)
            if item.text(0) != self.localize_func("ANALYSIS_PAGE.unassigned_group"):
                item.setExpanded(True)

        unassigned_count = len(assignments.get("Unassigned", []))
        if unassigned_count == 0:
            self.status_label.setText(
                self.localize_func("ANALYSIS_PAGE.all_datasets_assigned_status").format(
                    total=len(self.available_datasets),
                    groups=len(assignments),
                )
            )
            self.status_label.setStyleSheet("font-size: 11px; color: #28a745; margin-top: 8px;")
        else:
            self.status_label.setText(
                self.localize_func("ANALYSIS_PAGE.some_unassigned_status").format(
                    unassigned=unassigned_count,
                    assigned=total_assigned,
                    total=len(self.available_datasets),
                    groups=len(assignments) - 1,
                )
            )
            self.status_label.setStyleSheet("font-size: 11px; color: #ff9800; margin-top: 8px;")

    def _match_datasets_to_groups(self, group_configs: List[Dict]) -> Dict[str, List[str]]:
        """Match datasets to groups based on include/exclude keywords."""
        assignments: Dict[str, List[str]] = {}
        assigned_datasets = set()

        for config in group_configs:
            if not config.get("auto_assign"):
                continue

            group_name = config.get("name")
            include_keywords: List[str] = list(config.get("include") or [])
            exclude_keywords: List[str] = list(config.get("exclude") or [])

            assignments[group_name] = []

            include_lc = [kw.lower() for kw in include_keywords if kw.strip()]
            exclude_lc = [kw.lower() for kw in exclude_keywords if kw.strip()]

            for dataset in self.available_datasets:
                if dataset in assigned_datasets:
                    continue

                dataset_lower = str(dataset).lower()

                # Exclude: NOT-ANY semantics
                if any(ex_kw in dataset_lower for ex_kw in exclude_lc):
                    continue

                # Include: AND semantics (must contain ALL include keywords)
                if include_lc and not all(in_kw in dataset_lower for in_kw in include_lc):
                    continue

                assignments[group_name].append(dataset)
                assigned_datasets.add(dataset)

        unassigned = [ds for ds in self.available_datasets if ds not in assigned_datasets]
        if unassigned:
            assignments["Unassigned"] = unassigned

        return assignments

    def _validate_and_get_configs(self) -> Tuple[bool, str, List[Dict]]:
        configs: List[Dict] = []
        group_names = set()

        for row in range(self.table.rowCount()):
            name_widget = self.table.cellWidget(row, 0)
            if not name_widget:
                continue

            group_name = name_widget.text().strip() if isinstance(name_widget, QLineEdit) else ""
            if not group_name:
                continue

            if group_name in group_names:
                return (
                    False,
                    self.localize_func("ANALYSIS_PAGE.duplicate_name_error").format(name=group_name),
                    [],
                )

            group_names.add(group_name)

            include_widget = self.table.cellWidget(row, 1)
            include_keywords: List[str] = []
            if include_widget and isinstance(include_widget, QLineEdit):
                include_text = include_widget.text().strip()
                if include_text:
                    include_keywords = [kw.strip() for kw in include_text.split(",") if kw.strip()]

            exclude_widget = self.table.cellWidget(row, 2)
            exclude_keywords: List[str] = []
            if exclude_widget and isinstance(exclude_widget, QLineEdit):
                exclude_text = exclude_widget.text().strip()
                if exclude_text:
                    exclude_keywords = [kw.strip() for kw in exclude_text.split(",") if kw.strip()]

            checkbox_widget = self.table.cellWidget(row, 3)
            auto_assign = False
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    auto_assign = checkbox.isChecked()

            # UX safety: if the user typed include keywords, treat it as auto-assign intent.
            # This prevents the common failure mode where preview shows "0 groups" assigned
            # because the checkbox was left unchecked.
            if include_keywords and not auto_assign:
                auto_assign = True
                if checkbox_widget:
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if checkbox:
                        checkbox.setChecked(True)

            if auto_assign and not include_keywords:
                return (
                    False,
                    self.localize_func("ANALYSIS_PAGE.validation_error_no_include_keywords").format(
                        group_name=group_name
                    ),
                    [],
                )

            configs.append(
                {
                    "name": group_name,
                    "include": include_keywords,
                    "exclude": exclude_keywords,
                    "auto_assign": auto_assign,
                }
            )

        if not configs:
            return False, self.localize_func("ANALYSIS_PAGE.validation_error_no_groups"), []

        return True, "", configs

    def _apply_groups(self):
        is_valid, error_msg, configs = self._validate_and_get_configs()
        if not is_valid:
            QMessageBox.warning(self, self.localize_func("ANALYSIS_PAGE.validation_error_title"), error_msg)
            return

        assignments = self._match_datasets_to_groups(configs)

        if "Unassigned" in assignments:
            unassigned_count = len(assignments["Unassigned"])
            del assignments["Unassigned"]

            if unassigned_count > 0:
                reply = QMessageBox.question(
                    self,
                    self.localize_func("ANALYSIS_PAGE.unassigned_datasets_dialog_title"),
                    self.localize_func("ANALYSIS_PAGE.unassigned_datasets_dialog_message").format(
                        count=unassigned_count
                    ),
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.No:
                    return

        self.group_configs = configs
        self.final_assignments = assignments
        self.accept()

    def get_assignments(self) -> Dict[str, List[str]]:
        return self.final_assignments if hasattr(self, "final_assignments") else {}

    def get_group_configs(self) -> List[Dict]:
        """Get the final group definitions (names/keywords/auto_assign flags)."""
        return list(self.group_configs) if self.group_configs else []
