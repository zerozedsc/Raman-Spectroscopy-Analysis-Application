"""
Multi-Group Creation Dialog

Allows creating multiple groups at once with include/exclude keyword patterns
and smart auto-assignment functionality.
"""

from typing import Dict, List, Tuple, Callable, Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QLineEdit, QCheckBox, QHeaderView,
    QMessageBox, QTreeWidget, QTreeWidgetItem, QSplitter,
    QWidget, QGroupBox, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
import re


class MultiGroupCreationDialog(QDialog):
    """
    Dialog for creating multiple groups at once with keyword matching.
    
    Features:
    - Table with Group Name | Include Keywords | Exclude Keywords | Auto-Assign
    - Add/Remove rows dynamically
    - Preview assignments before applying
    - Validation (no duplicates, no empty names, no conflicts)
    """
    
    def __init__(self, available_datasets: List[str], localize_func: Callable, parent=None):
        """
        Initialize multi-group creation dialog.
        
        Args:
            available_datasets: List of available dataset names
            localize_func: Localization function
            parent: Parent widget
        """
        super().__init__(parent)
        self.available_datasets = available_datasets
        self.localize_func = localize_func
        self.group_configs = []
        
        self.setWindowTitle(localize_func("ANALYSIS_PAGE.multi_group_dialog_title"))
        self.setModal(True)
        self.resize(900, 600)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(16)
        
        # Create splitter for table and preview
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Group definition table
        left_widget = self._create_group_table_widget()
        splitter.addWidget(left_widget)
        
        # Right side: Preview pane
        right_widget = self._create_preview_widget()
        splitter.addWidget(right_widget)
        
        splitter.setStretchFactor(0, 2)  # Table takes 2/3
        splitter.setStretchFactor(1, 1)  # Preview takes 1/3
        
        main_layout.addWidget(splitter, 1)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        preview_btn = QPushButton(self.localize_func("ANALYSIS_PAGE.preview_assignments_button"))
        preview_btn.setMinimumHeight(36)
        preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
        """)
        preview_btn.clicked.connect(self._preview_assignments)
        button_layout.addWidget(preview_btn)
        
        apply_btn = QPushButton(self.localize_func("ANALYSIS_PAGE.apply_groups_button"))
        apply_btn.setMinimumHeight(36)
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        apply_btn.clicked.connect(self._apply_groups)
        button_layout.addWidget(apply_btn)
        
        cancel_btn = QPushButton(self.localize_func("ANALYSIS_PAGE.cancel_button"))
        cancel_btn.setMinimumHeight(36)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        main_layout.addLayout(button_layout)
    
    def _create_group_table_widget(self) -> QWidget:
        """Create the group definition table widget."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title = QLabel("<b>Group Definitions</b>")
        title.setStyleSheet("font-size: 13px; color: #2c3e50; margin-bottom: 8px;")
        layout.addWidget(title)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            self.localize_func("ANALYSIS_PAGE.group_name_column"),
            self.localize_func("ANALYSIS_PAGE.include_keywords_column"),
            self.localize_func("ANALYSIS_PAGE.exclude_keywords_column"),
            self.localize_func("ANALYSIS_PAGE.auto_assign_column")
        ])
        
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        
        self.table.setColumnWidth(0, 150)
        self.table.setColumnWidth(3, 100)
        
        self.table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: white;
                gridline-color: #dee2e6;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 8px;
                border: none;
                border-bottom: 2px solid #dee2e6;
                font-weight: 600;
                color: #495057;
            }
        """)
        
        layout.addWidget(self.table)
        
        # Add/Remove row buttons
        btn_layout = QHBoxLayout()
        
        add_row_btn = QPushButton("➕ " + self.localize_func("ANALYSIS_PAGE.add_row_button"))
        add_row_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        add_row_btn.clicked.connect(self._add_row)
        btn_layout.addWidget(add_row_btn)
        
        remove_row_btn = QPushButton("➖ " + self.localize_func("ANALYSIS_PAGE.remove_row_button"))
        remove_row_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        remove_row_btn.clicked.connect(self._remove_row)
        btn_layout.addWidget(remove_row_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Hint
        hint = QLabel("💡 " + self.localize_func("ANALYSIS_PAGE.keywords_hint"))
        hint.setStyleSheet("font-size: 11px; color: #6c757d; margin-top: 4px;")
        layout.addWidget(hint)
        
        # Add 1 default row (user can add more as needed)
        self._add_row()
        
        return container
    
    def _create_preview_widget(self) -> QWidget:
        """Create the assignment preview widget."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title = QLabel("<b>" + self.localize_func("ANALYSIS_PAGE.preview_title") + "</b>")
        title.setStyleSheet("font-size: 13px; color: #2c3e50; margin-bottom: 8px;")
        layout.addWidget(title)
        
        # Tree widget for preview
        self.preview_tree = QTreeWidget()
        self.preview_tree.setHeaderLabels(["Group", "Count"])
        self.preview_tree.setColumnWidth(0, 300)
        self.preview_tree.setStyleSheet("""
            QTreeWidget {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: white;
                font-size: 12px;
            }
            QTreeWidget::item {
                padding: 6px;
                border-bottom: 1px solid #f1f3f5;
            }
            QTreeWidget::item:hover {
                background-color: #f8f9fa;
            }
            QTreeWidget::item:selected {
                background-color: #e7f3ff;
                color: #212529;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 8px;
                border: none;
                border-bottom: 2px solid #dee2e6;
                font-weight: 600;
                color: #495057;
            }
        """)
        layout.addWidget(self.preview_tree)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 11px; color: #6c757d; margin-top: 8px;")
        layout.addWidget(self.status_label)
        
        return container
    
    def _add_row(self):
        """Add a new row to the table."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        # Group name (QLineEdit for better UX)
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("e.g., Control, Disease, MM, MGUS")
        name_edit.setStyleSheet("""
            QLineEdit {
                border: none;
                padding: 4px 8px;
                background-color: transparent;
                font-size: 13px;
            }
            QLineEdit:focus {
                background-color: #fff9e6;
                border: 1px solid #ffc107;
                border-radius: 2px;
            }
        """)
        self.table.setCellWidget(row, 0, name_edit)
        
        # Include keywords (QLineEdit with placeholder)
        include_edit = QLineEdit()
        include_edit.setPlaceholderText("e.g., ctrl, control, con")
        include_edit.setStyleSheet("""
            QLineEdit {
                border: none;
                padding: 4px 8px;
                background-color: transparent;
                font-size: 13px;
            }
            QLineEdit:focus {
                background-color: #e8f5e9;
                border: 1px solid #4caf50;
                border-radius: 2px;
            }
        """)
        self.table.setCellWidget(row, 1, include_edit)
        
        # Exclude keywords (QLineEdit with placeholder)
        exclude_edit = QLineEdit()
        exclude_edit.setPlaceholderText("e.g., treatment, test")
        exclude_edit.setStyleSheet("""
            QLineEdit {
                border: none;
                padding: 4px 8px;
                background-color: transparent;
                font-size: 13px;
            }
            QLineEdit:focus {
                background-color: #ffebee;
                border: 1px solid #f44336;
                border-radius: 2px;
            }
        """)
        self.table.setCellWidget(row, 2, exclude_edit)
        
        # Auto-assign checkbox
        checkbox = QCheckBox()
        checkbox.setChecked(True)
        checkbox_container = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_container)
        checkbox_layout.addWidget(checkbox)
        checkbox_layout.setAlignment(Qt.AlignCenter)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.table.setCellWidget(row, 3, checkbox_container)
    
    def _remove_row(self):
        """Remove the selected row from the table."""
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)
    
    def _preview_assignments(self):
        """Preview which datasets will be assigned to which groups."""
        # Get group configs
        is_valid, error_msg, configs = self._validate_and_get_configs()
        
        if not is_valid:
            QMessageBox.warning(self, "Validation Error", error_msg)
            return
        
        # Calculate assignments
        assignments = self._match_datasets_to_groups(configs)
        
        # Update preview tree
        self.preview_tree.clear()
        
        total_assigned = 0
        for group_name, datasets in sorted(assignments.items()):
            if group_name == "Unassigned":
                # Add unassigned at the end
                continue
            
            group_item = QTreeWidgetItem([f"🧪 {group_name}", str(len(datasets))])
            font = QFont()
            font.setBold(True)
            group_item.setFont(0, font)
            group_item.setForeground(0, Qt.darkBlue)
            
            # Limit display to first 50 datasets per group
            MAX_PER_GROUP = 50
            datasets_to_show = sorted(datasets)[:MAX_PER_GROUP]
            
            for dataset in datasets_to_show:
                dataset_item = QTreeWidgetItem([f"  📊 {dataset}", ""])
                group_item.addChild(dataset_item)
            
            # Add summary if there are more
            if len(datasets) > MAX_PER_GROUP:
                summary_item = QTreeWidgetItem([
                    f"  ... and {len(datasets) - MAX_PER_GROUP} more",
                    ""
                ])
                summary_item.setForeground(0, Qt.gray)
                group_item.addChild(summary_item)
            
            self.preview_tree.addTopLevelItem(group_item)
            total_assigned += len(datasets)
        
        # Add unassigned
        if "Unassigned" in assignments and assignments["Unassigned"]:
            unassigned_count = len(assignments["Unassigned"])
            unassigned_item = QTreeWidgetItem([
                self.localize_func("ANALYSIS_PAGE.unassigned_group"),
                str(unassigned_count)
            ])
            font = QFont()
            font.setBold(True)
            unassigned_item.setFont(0, font)
            unassigned_item.setForeground(0, Qt.darkRed)
            
            # Limit display to first 20 datasets for performance
            MAX_DISPLAY = 20
            datasets_to_show = sorted(assignments["Unassigned"])[:MAX_DISPLAY]
            
            for dataset in datasets_to_show:
                dataset_item = QTreeWidgetItem([f"  ⚠️ {dataset}", ""])
                unassigned_item.addChild(dataset_item)
            
            # Add summary if there are more
            if unassigned_count > MAX_DISPLAY:
                summary_item = QTreeWidgetItem([
                    f"  ... and {unassigned_count - MAX_DISPLAY} more",
                    ""
                ])
                summary_item.setForeground(0, Qt.gray)
                unassigned_item.addChild(summary_item)
            
            self.preview_tree.addTopLevelItem(unassigned_item)
        
        # Expand only assigned groups (keep unassigned collapsed for large lists)
        for i in range(self.preview_tree.topLevelItemCount()):
            item = self.preview_tree.topLevelItem(i)
            # Expand all except Unassigned (to avoid showing 244 items at once)
            if item.text(0) != self.localize_func("ANALYSIS_PAGE.unassigned_group"):
                item.setExpanded(True)
        
        # Update status
        unassigned_count = len(assignments.get("Unassigned", []))
        if unassigned_count == 0:
            self.status_label.setText(
                f"✅ All {len(self.available_datasets)} datasets assigned to {len(assignments)} group(s)."
            )
            self.status_label.setStyleSheet("font-size: 11px; color: #28a745; margin-top: 8px;")
        else:
            self.status_label.setText(
                f"⚠️ {unassigned_count} dataset(s) unassigned. "
                f"{total_assigned}/{len(self.available_datasets)} assigned to {len(assignments)-1} group(s)."
            )
            self.status_label.setStyleSheet("font-size: 11px; color: #ff9800; margin-top: 8px;")
    
    def _match_datasets_to_groups(self, group_configs: List[Dict]) -> Dict[str, List[str]]:
        """
        Match datasets to groups based on include/exclude keywords.
        
        Args:
            group_configs: List of {name, include, exclude, auto_assign}
        
        Returns:
            {group_name: [matched_datasets]}
        """
        assignments = {}
        assigned_datasets = set()
        
        for config in group_configs:
            if not config["auto_assign"]:
                continue
            
            group_name = config["name"]
            include_keywords = config["include"]
            exclude_keywords = config["exclude"]
            
            assignments[group_name] = []
            
            for dataset in self.available_datasets:
                if dataset in assigned_datasets:
                    continue  # Skip already assigned
                
                dataset_lower = dataset.lower()
                
                # Check exclude keywords first
                excluded = False
                for exclude_kw in exclude_keywords:
                    if exclude_kw.lower() in dataset_lower:
                        excluded = True
                        break
                
                if excluded:
                    continue
                
                # Check include keywords
                matched = False
                for include_kw in include_keywords:
                    if include_kw.lower() in dataset_lower:
                        matched = True
                        break
                
                if matched:
                    assignments[group_name].append(dataset)
                    assigned_datasets.add(dataset)
        
        # Add unassigned datasets
        unassigned = [ds for ds in self.available_datasets if ds not in assigned_datasets]
        if unassigned:
            assignments["Unassigned"] = unassigned
        
        return assignments
    
    def _validate_and_get_configs(self) -> Tuple[bool, str, List[Dict]]:
        """
        Validate table contents and extract group configs.
        
        Returns:
            (is_valid, error_message, configs)
        """
        configs = []
        group_names = set()
        
        for row in range(self.table.rowCount()):
            # Get group name from QLineEdit widget
            name_widget = self.table.cellWidget(row, 0)
            if not name_widget:
                continue
            
            group_name = name_widget.text().strip() if isinstance(name_widget, QLineEdit) else ""
            if not group_name:
                continue  # Skip empty rows
            
            # Check for duplicate names
            if group_name in group_names:
                return False, self.localize_func(
                    "ANALYSIS_PAGE.duplicate_name_error"
                ).format(name=group_name), []
            
            group_names.add(group_name)
            
            # Get include keywords from QLineEdit widget
            include_widget = self.table.cellWidget(row, 1)
            include_keywords = []
            if include_widget and isinstance(include_widget, QLineEdit):
                include_text = include_widget.text().strip()
                if include_text:
                    include_keywords = [kw.strip() for kw in include_text.split(',') if kw.strip()]
            
            # Get exclude keywords from QLineEdit widget
            exclude_widget = self.table.cellWidget(row, 2)
            exclude_keywords = []
            if exclude_widget and isinstance(exclude_widget, QLineEdit):
                exclude_text = exclude_widget.text().strip()
                if exclude_text:
                    exclude_keywords = [kw.strip() for kw in exclude_text.split(',') if kw.strip()]
            
            # Get auto-assign checkbox
            checkbox_widget = self.table.cellWidget(row, 3)
            auto_assign = False
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    auto_assign = checkbox.isChecked()
            
            # Validate: must have at least one include keyword if auto-assign is checked
            if auto_assign and not include_keywords:
                return False, f"Group '{group_name}' has auto-assign enabled but no include keywords.", []
            
            configs.append({
                "name": group_name,
                "include": include_keywords,
                "exclude": exclude_keywords,
                "auto_assign": auto_assign
            })
        
        if not configs:
            return False, "No groups defined. Please add at least one group.", []
        
        return True, "", configs
    
    def _apply_groups(self):
        """Apply the group configurations and close dialog."""
        # Validate
        is_valid, error_msg, configs = self._validate_and_get_configs()
        
        if not is_valid:
            QMessageBox.warning(self, "Validation Error", error_msg)
            return
        
        # Calculate final assignments
        assignments = self._match_datasets_to_groups(configs)
        
        # Remove "Unassigned" from assignments (we'll handle these separately)
        if "Unassigned" in assignments:
            unassigned_count = len(assignments["Unassigned"])
            del assignments["Unassigned"]
            
            if unassigned_count > 0:
                reply = QMessageBox.question(
                    self,
                    "Unassigned Datasets",
                    f"{unassigned_count} dataset(s) will remain unassigned. Continue?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply == QMessageBox.No:
                    return
        
        # Store configs for retrieval
        self.group_configs = configs
        self.final_assignments = assignments
        
        self.accept()
    
    def get_assignments(self) -> Dict[str, List[str]]:
        """
        Get the final group assignments.
        
        Returns:
            Dictionary mapping group names to lists of dataset names
        """
        return self.final_assignments if hasattr(self, 'final_assignments') else {}
