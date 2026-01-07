"""
Group Assignment Widget for Multi-Dataset Analysis

This module provides a flexible UI for assigning datasets to labeled groups.
Supports scenarios like:
- Single group with multiple datasets (e.g., analyzing "Control" with 5 replicates)
- Multiple groups (e.g., "MM" vs "MGUS" vs "NORMAL")
- Custom group labels for classification tasks
"""

from typing import Dict, List, Callable, Optional
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QListWidget,
    QGroupBox,
    QAbstractItemView,
    QComboBox,
    QScrollArea,
    QFrame,
    QSplitter,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont


class GroupAssignmentWidget(QWidget):
    """
    Widget for assigning datasets to labeled groups.

    Signals:
        groups_changed: Emitted when group assignments change
    """

    groups_changed = Signal(dict)  # {group_label: [dataset_names]}

    def __init__(self, dataset_names: List[str], localize_func: Callable, parent=None):
        """
        Initialize group assignment widget.

        Args:
            dataset_names: List of available dataset names
            localize_func: Localization function
            parent: Parent widget
        """
        super().__init__(parent)
        self.dataset_names = dataset_names
        self.localize_func = localize_func
        self.groups: Dict[str, List[str]] = {}  # {group_label: [dataset_names]}

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(12)

        # Header with instructions
        header_label = QLabel(
            self.localize_func("ANALYSIS_PAGE.group_assignment_header")
        )
        header_label.setWordWrap(True)
        header_label.setStyleSheet(
            """
            font-size: 12px;
            color: #6c757d;
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 4px;
            border-left: 3px solid #0078d4;
        """
        )
        main_layout.addWidget(header_label)

        # Splitter: Available Datasets | Group Configuration
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # === LEFT: Available Datasets ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        available_label = QLabel(self.localize_func("ANALYSIS_PAGE.available_datasets"))
        available_label.setStyleSheet("font-weight: 600; font-size: 13px;")
        left_layout.addWidget(available_label)

        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.available_list.addItems(self.dataset_names)
        self.available_list.setStyleSheet(
            """
            QListWidget {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
            }
            QListWidget::item {
                padding: 6px 8px;
                border-radius: 3px;
                margin: 1px;
            }
            QListWidget::item:hover {
                background-color: #f0f0f0;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
        """
        )
        left_layout.addWidget(self.available_list)

        splitter.addWidget(left_widget)

        # === RIGHT: Group Configuration ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        groups_label = QLabel(self.localize_func("ANALYSIS_PAGE.groups"))
        groups_label.setStyleSheet("font-weight: 600; font-size: 13px;")
        right_layout.addWidget(groups_label)

        # Add group button and label input
        add_group_layout = QHBoxLayout()
        self.group_name_input = QLineEdit()
        self.group_name_input.setPlaceholderText(
            self.localize_func("ANALYSIS_PAGE.group_name_placeholder")
        )
        self.group_name_input.setMinimumHeight(32)
        self.group_name_input.setStyleSheet(
            """
            QLineEdit {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QLineEdit:focus {
                border-color: #0078d4;
            }
        """
        )
        add_group_layout.addWidget(self.group_name_input)

        add_group_btn = QPushButton(self.localize_func("ANALYSIS_PAGE.add_group"))
        add_group_btn.setMinimumHeight(32)
        add_group_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:pressed {
                background-color: #004578;
            }
        """
        )
        add_group_btn.clicked.connect(self._add_group)
        add_group_layout.addWidget(add_group_btn)

        right_layout.addLayout(add_group_layout)

        # Groups scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        self.groups_container = QWidget()
        self.groups_layout = QVBoxLayout(self.groups_container)
        self.groups_layout.setContentsMargins(0, 0, 0, 0)
        self.groups_layout.setSpacing(8)
        self.groups_layout.addStretch()

        scroll_area.setWidget(self.groups_container)
        right_layout.addWidget(scroll_area)

        splitter.addWidget(right_widget)
        splitter.setSizes([300, 400])

        main_layout.addWidget(splitter)

        # Summary label
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet(
            """
            font-size: 11px;
            color: #6c757d;
            padding: 6px;
            background-color: #f8f9fa;
            border-radius: 3px;
        """
        )
        self._update_summary()
        main_layout.addWidget(self.summary_label)

    def _add_group(self):
        """Add a new group with selected datasets."""
        group_name = self.group_name_input.text().strip()

        if not group_name:
            return

        if group_name in self.groups:
            # Group already exists, just add selected datasets
            pass

        # Get selected datasets
        selected_items = self.available_list.selectedItems()
        if not selected_items:
            return

        selected_datasets = [item.text() for item in selected_items]

        # Add or update group
        if group_name in self.groups:
            # Append new datasets (avoid duplicates)
            existing = set(self.groups[group_name])
            for ds in selected_datasets:
                if ds not in existing:
                    self.groups[group_name].append(ds)
        else:
            self.groups[group_name] = selected_datasets
            self._add_group_widget(group_name)

        # Clear selection and input
        self.available_list.clearSelection()
        self.group_name_input.clear()

        self._update_summary()
        self.groups_changed.emit(self.groups)

    def _add_group_widget(self, group_name: str):
        """Add a visual group widget to the UI."""
        group_box = QGroupBox(group_name)
        group_box.setObjectName(f"group_{group_name}")
        group_box.setStyleSheet(
            """
            QGroupBox {
                font-weight: 600;
                border: 2px solid #0078d4;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 16px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                background-color: white;
            }
        """
        )

        group_layout = QVBoxLayout(group_box)

        # Dataset list for this group
        dataset_list = QListWidget()
        dataset_list.setMaximumHeight(100)
        dataset_list.setStyleSheet(
            """
            QListWidget {
                border: 1px solid #d0d0d0;
                border-radius: 3px;
                background-color: white;
            }
            QListWidget::item {
                padding: 4px 6px;
            }
        """
        )
        group_layout.addWidget(dataset_list)

        # Remove button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        remove_btn = QPushButton(self.localize_func("ANALYSIS_PAGE.remove_group"))
        remove_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 4px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """
        )
        remove_btn.clicked.connect(lambda: self._remove_group(group_name))
        button_layout.addWidget(remove_btn)

        group_layout.addLayout(button_layout)

        # Insert before stretch
        self.groups_layout.insertWidget(self.groups_layout.count() - 1, group_box)

        # Update dataset list
        self._update_group_widget(group_name)

    def _update_group_widget(self, group_name: str):
        """Update the dataset list in a group widget."""
        group_box = self.groups_container.findChild(QGroupBox, f"group_{group_name}")
        if group_box:
            dataset_list = group_box.findChild(QListWidget)
            if dataset_list:
                dataset_list.clear()
                dataset_list.addItems(self.groups[group_name])

    def _remove_group(self, group_name: str):
        """Remove a group."""
        if group_name in self.groups:
            del self.groups[group_name]

        # Remove widget
        group_box = self.groups_container.findChild(QGroupBox, f"group_{group_name}")
        if group_box:
            self.groups_layout.removeWidget(group_box)
            group_box.deleteLater()

        self._update_summary()
        self.groups_changed.emit(self.groups)

    def _update_summary(self):
        """Update the summary label."""
        if not self.groups:
            summary_text = self.localize_func("ANALYSIS_PAGE.no_groups_defined")
        else:
            group_count = len(self.groups)
            total_datasets = sum(len(datasets) for datasets in self.groups.values())
            summary_text = self.localize_func(
                "ANALYSIS_PAGE.group_summary",
                group_count=group_count,
                dataset_count=total_datasets,
            )

        self.summary_label.setText(summary_text)

    def get_groups(self) -> Dict[str, List[str]]:
        """
        Get the current group assignments.

        Returns:
            Dictionary mapping group labels to lists of dataset names
        """
        return self.groups.copy()

    def set_groups(self, groups: Dict[str, List[str]]):
        """
        Set group assignments programmatically.

        Args:
            groups: Dictionary mapping group labels to dataset lists
        """
        # Clear existing groups
        for group_name in list(self.groups.keys()):
            self._remove_group(group_name)

        # Add new groups
        self.groups = groups.copy()
        for group_name in self.groups:
            self._add_group_widget(group_name)
            self._update_group_widget(group_name)

        self._update_summary()

    def clear_groups(self):
        """Clear all group assignments."""
        self.set_groups({})
