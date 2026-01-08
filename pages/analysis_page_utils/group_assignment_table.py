"""
Professional Group Assignment Widget for Raman Spectroscopy Classification

This module implements a table-based group assignment interface specifically
designed for scientists performing classification tasks (e.g., MM vs MGUS).

Key Features:
- Table widget with [Dataset Name | Group Label] columns
- Dropdown selectors for group labels with custom entries
- Pattern-based auto-assignment (e.g., "Control_01" → "Control")
- Simple, intuitive workflow without mental mapping overhead
"""

import traceback

from configs.configs import create_logs
from typing import Dict, List, Callable, Optional
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QHeaderView,
    QMessageBox,
    QLineEdit,
    QDialog,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
import re


class GroupAssignmentTable(QWidget):
    """
    Professional table-based group assignment widget.

    Designed for scientists:
    - Clear visual representation of dataset → group mapping
    - Dropdown selectors for common group names
    - Auto-assign feature using intelligent pattern matching
    - Minimal cognitive overhead

    Signals:
        groups_changed: Emitted when group assignments change {group_name: [datasets]}
    """

    groups_changed = Signal(dict)

    def __init__(self, dataset_names: List[str], localize_func: Callable, parent=None):
        """
        Initialize group assignment table.

        Args:
            dataset_names: List of available dataset names
            localize_func: Localization function
            parent: Parent widget
        """
        super().__init__(parent)
        create_logs(__name__, __file__, "GroupAssignmentTable __init__ called", status="debug")
        create_logs(__name__, __file__, f"Number of datasets: {len(dataset_names)}", status="debug")
        create_logs(__name__, __file__, f"Dataset names: {dataset_names}", status="debug")

        self.dataset_names = dataset_names
        self.localize_func = localize_func
        self.common_groups = [
            "Control",
            "Disease",
            "Treatment A",
            "Treatment B",
            "MM",
            "MGUS",
            "Normal",
        ]

        create_logs(__name__, __file__, "Calling _init_ui()", status="debug")
        self._init_ui()
        create_logs(__name__, __file__, "GroupAssignmentTable initialization complete", status="debug")

    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(12)

        # Instructions
        instructions = QLabel(self.localize_func("ANALYSIS_PAGE.group_assignment_instructions"))
        instructions.setWordWrap(True)
        instructions.setStyleSheet(
            """
            padding: 12px;
            background-color: #f0f8ff;
            border-left: 4px solid #0078d4;
            border-radius: 4px;
            font-size: 12px;
        """
        )
        main_layout.addWidget(instructions)

        # Action toolbar - modern toolbar design
        from PySide6.QtWidgets import QToolBar, QWidget as QW

        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setStyleSheet(
            """
            QToolBar {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 4px 8px;
                spacing: 8px;
            }
            QToolButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: 600;
                font-size: 12px;
                color: #495057;
            }
            QToolButton:hover {
                background-color: #e9ecef;
                border-color: #dee2e6;
            }
            QToolButton:pressed {
                background-color: #dee2e6;
            }
        """
        )

        # Auto-assign action
        auto_assign_action = toolbar.addAction(self.localize_func("ANALYSIS_PAGE.auto_assign_button"))
        auto_assign_action.setToolTip(self.localize_func("ANALYSIS_PAGE.auto_assign_tooltip"))
        auto_assign_action.triggered.connect(self._auto_assign_groups)

        # Reset action
        reset_action = toolbar.addAction(self.localize_func("ANALYSIS_PAGE.reset_all_button"))
        reset_action.setToolTip(self.localize_func("ANALYSIS_PAGE.reset_all_tooltip"))
        reset_action.triggered.connect(self._reset_all)

        toolbar.addSeparator()
        
        # Multi-group creation action (NEW - replaces old single-input dialog)
        create_logs(__name__, __file__, "========== INITIALIZING CREATE GROUPS BUTTON ==========", status="debug")
        multi_group_action = toolbar.addAction(self.localize_func("ANALYSIS_PAGE.create_groups_button"))
        multi_group_action.setToolTip(self.localize_func("ANALYSIS_PAGE.create_groups_tooltip"))
        create_logs(__name__, __file__, f"Button created: {multi_group_action}", status="debug")
        create_logs(__name__, __file__, f"Button text: {multi_group_action.text()}", status="debug")
        create_logs(__name__, __file__, "Connecting to method: _on_create_groups_clicked", status="debug")
        multi_group_action.triggered.connect(self._on_create_groups_clicked)
        create_logs(__name__, __file__, "Connection successful!", status="debug")
        create_logs(__name__, __file__, "==========================================================", status="debug")
        
        main_layout.addWidget(toolbar)

        # Table widget
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels([
            self.localize_func("ANALYSIS_PAGE.dataset_name_column"),
            self.localize_func("ANALYSIS_PAGE.group_label_column")
        ])
        self.table.setRowCount(len(self.dataset_names))

        # Professional table styling with comfortable spacing and minimal gridlines
        self.table.setStyleSheet(
            """
            QTableWidget {
                border: 1px solid #dee2e6;
                border-radius: 6px;
                background-color: white;
                gridline-color: transparent;
                font-size: 13px;
            }
            QTableWidget::item {
                padding: 16px 12px;
                border-bottom: 1px solid #f1f3f5;
            }
            QTableWidget::item:hover {
                background-color: #f8f9fa;
            }
            QTableWidget::item:selected {
                background-color: #e7f3ff;
                color: #212529;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 14px 12px;
                border: none;
                border-bottom: 2px solid #dee2e6;
                font-weight: 700;
                font-size: 12px;
                color: #495057;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
        """
        )

        # Configure columns
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        self.table.setColumnWidth(1, 200)

        # Populate table
        create_logs(__name__, __file__, f"Populating table with {len(self.dataset_names)} datasets", status="debug")
        for row, dataset_name in enumerate(self.dataset_names):
            create_logs(
                __name__,
                __file__,
                f"Adding dataset {row + 1}/{len(self.dataset_names)}: {dataset_name}",
                status="debug",
            )
            # Dataset name (read-only)
            name_item = QTableWidgetItem(dataset_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            name_item.setFont(QFont("Segoe UI", 10))
            self.table.setItem(row, 0, name_item)

            # Group selector (dropdown)
            group_combo = QComboBox()
            group_combo.addItem(self.localize_func("ANALYSIS_PAGE.select_group_placeholder"))
            group_combo.addItems(self.common_groups)
            group_combo.addItem(self.localize_func("ANALYSIS_PAGE.add_custom_group_option"))

            # Set larger font for better visibility
            combo_font = QFont("Segoe UI", 11)
            combo_font.setWeight(QFont.Medium)
            group_combo.setFont(combo_font)

            group_combo.setStyleSheet(
                """
                QComboBox {
                    border: 1px solid #d0d0d0;
                    border-radius: 3px;
                    padding: 8px 12px;
                    padding-right: 25px;
                    background-color: white;
                    color: #2c3e50;
                    font-size: 13px;
                    font-weight: 500;
                    min-height: 28px;
                }
                QComboBox:hover {
                    border-color: #0078d4;
                    background-color: #f8f9fa;
                }
                QComboBox:focus {
                    border-color: #0078d4;
                    border-width: 2px;
                }
                QComboBox::drop-down {
                    subcontrol-origin: padding;
                    subcontrol-position: right center;
                    width: 20px;
                    border: none;
                }
                QComboBox::down-arrow {
                    image: none;
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 6px solid #6c757d;
                    width: 0;
                    height: 0;
                }
                QComboBox QAbstractItemView {
                    background-color: white;
                    border: 1px solid #d0d0d0;
                    selection-background-color: #0078d4;
                    selection-color: white;
                    padding: 4px;
                    font-size: 13px;
                }
                QComboBox QAbstractItemView::item {
                    padding: 8px 12px;
                    min-height: 28px;
                }
                QComboBox QAbstractItemView::item:hover {
                    background-color: #e7f3ff;
                }
            """
            )
            group_combo.currentTextChanged.connect(
                lambda text, r=row: self._on_group_changed(r, text)
            )

            create_logs(__name__, __file__, f"Created group combo for row {row} with font size 11", status="debug")
            self.table.setCellWidget(row, 1, group_combo)

        main_layout.addWidget(self.table)

        # Summary label
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet(
            """
            font-size: 11px;
            color: #6c757d;
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 3px;
        """
        )
        self._update_summary()
        main_layout.addWidget(self.summary_label)

    def _on_group_changed(self, row: int, text: str):
        """Handle group selection change."""
        if text == "+ Add Custom Group...":
            # Show custom group dialog
            custom_group = self._prompt_custom_group()
            combo = self.table.cellWidget(row, 1)
            if custom_group:
                # Add to common groups if not already there
                if custom_group not in self.common_groups:
                    self.common_groups.append(custom_group)
                    # Update all combos
                    for r in range(self.table.rowCount()):
                        cb = self.table.cellWidget(r, 1)
                        if cb.findText(custom_group) == -1:
                            cb.insertItem(cb.count() - 1, custom_group)
                # Set the new group
                combo.setCurrentText(custom_group)
            else:
                # User cancelled, reset to first item
                combo.setCurrentIndex(0)

        self._update_summary()
        self.groups_changed.emit(self.get_groups())

    def _prompt_custom_group(self) -> Optional[str]:
        """Prompt user to enter a custom group name."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Custom Group")
        dialog.setMinimumWidth(300)

        layout = QVBoxLayout(dialog)

        label = QLabel("Enter custom group name:")
        layout.addWidget(label)

        input_field = QLineEdit()
        input_field.setPlaceholderText("e.g., Treatment C, Benign, Stage 1...")
        layout.addWidget(input_field)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.Accepted:
            group_name = input_field.text().strip()
            if group_name:
                return group_name

        return None

    def _auto_assign_groups(self):
        """
        Automatically assign groups based on dataset name patterns.

        Enhanced Algorithm:
        1. Try date prefix pattern (YYYYMMDD_*)
        2. Try keyword pattern (MM, MGUS, Control, etc.)
        3. Try underscore/hyphen prefix pattern (Prefix_*)
        4. Try mixed patterns with multiple segments
        """
        create_logs(__name__, __file__, "Auto-assign groups called (enhanced version)", status="debug")
        create_logs(__name__, __file__, f"Analyzing {len(self.dataset_names)} dataset names", status="debug")

        # Pattern extraction with multiple strategies
        patterns = {}

        for idx, dataset_name in enumerate(self.dataset_names):
            create_logs(__name__, __file__, f"Analyzing pattern for: {dataset_name}", status="debug")
            group_assigned = False

            # === Strategy 1: Date prefix pattern (YYYYMMDD_*) ===
            # Example: "20220314_MgusO1_B" → extract "Mgus" or "MgusO1"
            date_match = re.match(r"^\d{8}_(.+)", dataset_name)
            if date_match:
                remainder = date_match.group(1)
                create_logs(__name__, __file__, f"Date prefix detected, remainder: {remainder}", status="debug")

                # Extract alphanumeric prefix before next separator
                parts = re.split(r"[_\-\s]", remainder)
                if parts:
                    # Try to find keyword in first part
                    first_part = parts[0]
                    create_logs(__name__, __file__, f"First part after date: {first_part}", status="debug")

                    # Check for known keywords in first part
                    for keyword in [
                        "MM",
                        "MGUS",
                        "Control",
                        "Disease",
                        "Treatment",
                        "Normal",
                        "Healthy",
                        "Cancer",
                    ]:
                        if keyword.lower() in first_part.lower():
                            group_name = (
                                keyword.upper()
                                if len(keyword) <= 4
                                else keyword.capitalize()
                            )
                            create_logs(
                                __name__,
                                __file__,
                                f"Keyword '{keyword}' found in '{first_part}' -> Group '{group_name}'",
                                status="debug",
                            )

                            if group_name not in patterns:
                                patterns[group_name] = []
                            patterns[group_name].append(idx)
                            group_assigned = True
                            break

                    # If no keyword found, use first part as group (e.g., "MgusO1" → "Mgus")
                    if not group_assigned:
                        # Extract alpha prefix from alphanumeric string
                        alpha_prefix = re.match(r"^([A-Za-z]+)", first_part)
                        if alpha_prefix:
                            group_name = alpha_prefix.group(1).capitalize()
                            create_logs(
                                __name__,
                                __file__,
                                f"Using alpha prefix '{group_name}' from '{first_part}'",
                                status="debug",
                            )

                            if group_name not in patterns:
                                patterns[group_name] = []
                            patterns[group_name].append(idx)
                            group_assigned = True

            # === Strategy 2: Direct keyword pattern (no date prefix) ===
            if not group_assigned:
                # Extract alphanumeric words
                words = re.findall(r"[A-Za-z]+", dataset_name)
                create_logs(__name__, __file__, f"Extracted words: {words}", status="debug")

                # Look for common group keywords
                for word in words:
                    word_lower = word.lower()
                    if word_lower in [
                        "control",
                        "disease",
                        "treatment",
                        "mm",
                        "mgus",
                        "normal",
                        "healthy",
                        "cancer",
                    ]:
                        # Capitalize first letter
                        group_name = word.capitalize()
                        if word_lower in ["mm", "mgus"]:
                            group_name = word.upper()

                        create_logs(
                            __name__,
                            __file__,
                            f"Keyword match: '{word}' -> Group '{group_name}'",
                            status="debug",
                        )

                        if group_name not in patterns:
                            patterns[group_name] = []
                        patterns[group_name].append(idx)
                        group_assigned = True
                        break

            # === Strategy 3: Prefix pattern (Prefix_* or Prefix-*) ===
            if not group_assigned:
                # Split by underscore or hyphen and use first segment
                prefix_match = re.match(r"^([A-Za-z]+)[\-_]", dataset_name)
                if prefix_match:
                    group_name = prefix_match.group(1).capitalize()
                    create_logs(
                        __name__,
                        __file__,
                        f"Prefix pattern: '{dataset_name}' -> Group '{group_name}'",
                        status="debug",
                    )

                    if group_name not in patterns:
                        patterns[group_name] = []
                    patterns[group_name].append(idx)
                    group_assigned = True

            # === Strategy 4: Fallback - use first alpha sequence ===
            if not group_assigned:
                alpha_match = re.match(r"^([A-Za-z]+)", dataset_name)
                if alpha_match:
                    group_name = alpha_match.group(1).capitalize()
                    create_logs(
                        __name__,
                        __file__,
                        f"Fallback alpha pattern: '{dataset_name}' -> Group '{group_name}'",
                        status="debug",
                    )

                    if group_name not in patterns:
                        patterns[group_name] = []
                    patterns[group_name].append(idx)
                    group_assigned = True

            if not group_assigned:
                create_logs(__name__, __file__, f"No pattern detected for '{dataset_name}'", status="debug")

        # Apply assignments
        create_logs(__name__, __file__, f"Patterns found: {patterns}", status="debug")
        if patterns:
            assigned_count = 0
            for group_name, indices in patterns.items():
                create_logs(
                    __name__,
                    __file__,
                    f"Processing group '{group_name}' with {len(indices)} datasets",
                    status="debug",
                )
                # Ensure group exists
                if group_name not in self.common_groups:
                    create_logs(
                        __name__,
                        __file__,
                        f"Adding new group '{group_name}' to common groups",
                        status="debug",
                    )
                    self.common_groups.append(group_name)
                    # Add to all combos
                    for r in range(self.table.rowCount()):
                        cb = self.table.cellWidget(r, 1)
                        if cb.findText(group_name) == -1:
                            cb.insertItem(cb.count() - 1, group_name)

                # Assign to datasets
                for idx in indices:
                    combo = self.table.cellWidget(idx, 1)
                    combo.setCurrentText(group_name)
                    assigned_count += 1

            # Show detailed result message
            pattern_summary = "\n".join(
                [
                    f"• {name}: {len(indices)} dataset(s)"
                    for name, indices in patterns.items()
                ]
            )
            QMessageBox.information(
                self,
                self.localize_func("ANALYSIS_PAGE.auto_assign_complete_title"),
                self.localize_func(
                    "ANALYSIS_PAGE.auto_assign_complete_message",
                    assigned_count=assigned_count,
                    group_count=len(patterns),
                    pattern_summary=pattern_summary,
                ),
            )
        else:
            create_logs(
                "auto_assign_no_patterns",
                "GroupAssignmentTable",
                "No patterns detected during auto-assignment.",
                status="warning",
            )

            QMessageBox.warning(
                self,
                self.localize_func("ANALYSIS_PAGE.auto_assign_no_patterns_title"),
                self.localize_func("ANALYSIS_PAGE.auto_assign_no_patterns_message"),
            )

        self._update_summary()
        self.groups_changed.emit(self.get_groups())

    def _reset_all(self):
        """Reset all group assignments."""
        reply = QMessageBox.question(
            self,
            self.localize_func("ANALYSIS_PAGE.reset_assignments_title"),
            self.localize_func("ANALYSIS_PAGE.reset_assignments_message"),
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            for row in range(self.table.rowCount()):
                combo = self.table.cellWidget(row, 1)
                combo.setCurrentIndex(0)  # "-- Select Group --"

            self._update_summary()
            self.groups_changed.emit(self.get_groups())

    def _add_custom_group(self):
        """
        Legacy method - now redirects to multi-group dialog.
        Kept for backward compatibility in case called from elsewhere.
        """
        create_logs(
            __name__,
            __file__,
            "_add_custom_group called - redirecting to multi-group dialog",
            status="debug",
        )
        self._on_create_groups_clicked()
    
    def _on_create_groups_clicked(self):
        """Handler for Create Groups button click."""
        create_logs(__name__, __file__, "=" * 70, status="debug")
        create_logs(__name__, __file__, "CREATE GROUPS BUTTON CLICKED!", status="debug")
        create_logs(
            __name__,
            __file__,
            f"Timestamp: {__import__('datetime').datetime.now()}",
            status="debug",
        )
        create_logs(__name__, __file__, "Method: _on_create_groups_clicked", status="debug")
        create_logs(__name__, __file__, f"Instance: {self}", status="debug")
        create_logs(__name__, __file__, f"Instance class: {self.__class__.__name__}", status="debug")
        create_logs(__name__, __file__, "=" * 70, status="debug")
        self._open_multi_group_dialog()
    
    def _open_multi_group_dialog(self):
        """Open the multi-group creation dialog."""
        create_logs(__name__, __file__, "=" * 70, status="debug")
        create_logs(__name__, __file__, "OPENING MULTI-GROUP DIALOG", status="debug")
        create_logs(__name__, __file__, "=" * 70, status="debug")
        create_logs(
            __name__,
            __file__,
            "Step 1: Attempting to import MultiGroupCreationDialog...",
            status="debug",
        )
        create_logs(__name__, __file__, "Import path: .multi_group_dialog", status="debug")
        try:
            from .multi_group_dialog import MultiGroupCreationDialog
            create_logs(__name__, __file__, "MultiGroupCreationDialog imported successfully!", status="debug")
            create_logs(__name__, __file__, f"Class: {MultiGroupCreationDialog}", status="debug")
        except Exception as e:
            create_logs(__name__, __file__, f"Failed to import MultiGroupCreationDialog: {e}", status="error")
            create_logs(__name__, __file__, f"Exception type: {type(e).__name__}", status="error")
            create_logs(__name__, __file__, "Full traceback:", status="error")
            create_logs(__name__, __file__, traceback.format_exc(), status="error")
            QMessageBox.critical(
                self,
                self.localize_func("ANALYSIS_PAGE.group_assignment_import_error_title"),
                self.localize_func(
                    "ANALYSIS_PAGE.group_assignment_import_error_message",
                    error=str(e),
                ),
            )
            return
        
        try:
            create_logs(__name__, __file__, "Step 2: Creating dialog instance...", status="debug")
            create_logs(__name__, __file__, f"Dataset count: {len(self.dataset_names)}", status="debug")
            create_logs(__name__, __file__, f"Localize function: {self.localize_func}", status="debug")
            dialog = MultiGroupCreationDialog(
                self.dataset_names,
                self.localize_func,
                self
            )
            create_logs(__name__, __file__, "Dialog created successfully!", status="debug")
            create_logs(__name__, __file__, f"Dialog title: {dialog.windowTitle()}", status="debug")
            create_logs(__name__, __file__, "Step 3: Calling dialog.exec()...", status="debug")
            
            result = dialog.exec()
            create_logs(__name__, __file__, "Step 4: Dialog closed", status="debug")
            create_logs(__name__, __file__, f"Result: {result} (Accepted={QDialog.Accepted})", status="debug")
            
            if result == QDialog.Accepted:
                create_logs(__name__, __file__, "Dialog was ACCEPTED", status="debug")
                assignments = dialog.get_assignments()
                create_logs(__name__, __file__, f"Assignments: {assignments}", status="debug")
                
                if assignments:
                    # Apply assignments using set_groups
                    self.set_groups(assignments)
                    
                    # Show success message
                    total_assigned = sum(len(datasets) for datasets in assignments.values())
                    group_summary = "\n".join(
                        [
                            f"• {name}: {len(datasets)} dataset(s)"
                            for name, datasets in assignments.items()
                        ]
                    )
                    QMessageBox.information(
                        self,
                        self.localize_func("ANALYSIS_PAGE.groups_created_title"),
                        self.localize_func(
                            "ANALYSIS_PAGE.groups_created_message",
                            group_count=len(assignments),
                            dataset_count=total_assigned,
                            group_summary=group_summary,
                        ),
                    )
                    
                    self._update_summary()
                    self.groups_changed.emit(self.get_groups())
                else:
                    create_logs(__name__, __file__, "No assignments returned", status="debug")
            else:
                create_logs(__name__, __file__, "Dialog was CANCELLED/REJECTED", status="debug")
                
        except Exception as e:
            create_logs(__name__, __file__, f"Exception in _open_multi_group_dialog: {e}", status="error")
            create_logs(__name__, __file__, f"Exception type: {type(e).__name__}", status="error")
            create_logs(__name__, __file__, "Full traceback:", status="error")
            create_logs(__name__, __file__, traceback.format_exc(), status="error")
            QMessageBox.critical(
                self,
                self.localize_func("COMMON.error"),
                self.localize_func(
                    "ANALYSIS_PAGE.group_assignment_error_message",
                    error=str(e),
                ),
            )
        finally:
            create_logs(__name__, __file__, "=" * 70, status="debug")
            create_logs(__name__, __file__, "_open_multi_group_dialog completed", status="debug")
            create_logs(__name__, __file__, "=" * 70, status="debug")
    
    def _old_add_custom_group_DISABLED(self):
        """OLD METHOD - DISABLED - Use _on_create_groups_clicked instead."""
        from PySide6.QtWidgets import QInputDialog

        text, ok = QInputDialog.getText(
            self, "Create Custom Group", "Enter new group label:", text="Group"
        )

        if ok and text.strip():
            new_group = text.strip()

            # Add to all combo boxes if not already present
            for row in range(self.table.rowCount()):
                combo = self.table.cellWidget(row, 1)
                # Check if already exists
                existing_items = [combo.itemText(i) for i in range(combo.count())]
                if new_group not in existing_items:
                    combo.addItem(new_group)

            QMessageBox.information(
                self,
                self.localize_func("ANALYSIS_PAGE.group_created_title"),
                self.localize_func(
                    "ANALYSIS_PAGE.group_created_message",
                    group_name=new_group,
                ),
            )

    def _update_summary(self):
        """Update the summary label."""
        groups = self.get_groups()

        if not groups:
            self.summary_label.setText(
                "ℹ️ No groups assigned yet. Use the dropdowns or 'Auto-Assign' button."
            )
        else:
            group_count = len(groups)
            total_datasets = sum(len(datasets) for datasets in groups.values())
            unassigned = len(self.dataset_names) - total_datasets

            summary = f"✓ {group_count} group(s) defined • {total_datasets} dataset(s) assigned"
            if unassigned > 0:
                summary += f" • {unassigned} unassigned"

            self.summary_label.setText(summary)

    def get_groups(self) -> Dict[str, List[str]]:
        """
        Get current group assignments.

        Returns:
            Dictionary mapping group labels to lists of dataset names
        """
        create_logs(__name__, __file__, "get_groups() called", status="debug")
        groups = {}

        for row in range(self.table.rowCount()):
            dataset_name = self.table.item(row, 0).text()
            combo = self.table.cellWidget(row, 1)
            group_name = combo.currentText()

            create_logs(
                __name__,
                __file__,
                f"Row {row}: Dataset='{dataset_name}', Group='{group_name}'",
                status="debug",
            )

            # Skip unassigned
            if (
                group_name == "-- Select Group --"
                or group_name == "+ Add Custom Group..."
            ):
                continue

            if group_name not in groups:
                groups[group_name] = []

            groups[group_name].append(dataset_name)

        create_logs(__name__, __file__, f"Final groups: {groups}", status="debug")
        return groups

    def set_groups(self, groups: Dict[str, List[str]]):
        """
        Set group assignments programmatically.

        Args:
            groups: Dictionary mapping group labels to dataset lists
        """
        # Reset all first
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 1)
            combo.setCurrentIndex(0)

        # Apply assignments
        for group_name, datasets in groups.items():
            # Ensure group exists
            if group_name not in self.common_groups:
                self.common_groups.append(group_name)
                # Add to all combos
                for r in range(self.table.rowCount()):
                    cb = self.table.cellWidget(r, 1)
                    if cb.findText(group_name) == -1:
                        cb.insertItem(cb.count() - 1, group_name)

            # Assign datasets
            for dataset_name in datasets:
                for row in range(self.table.rowCount()):
                    if self.table.item(row, 0).text() == dataset_name:
                        combo = self.table.cellWidget(row, 1)
                        combo.setCurrentText(group_name)
                        break

        self._update_summary()
