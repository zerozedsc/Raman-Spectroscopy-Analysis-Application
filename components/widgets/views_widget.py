from configs.configs import create_logs
from typing import Dict, Any, Callable, Optional
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QScrollArea,
    QTabWidget,
    QGroupBox,
    QComboBox,
    QSplitter,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QListWidget,
    QAbstractItemView,
    QStackedWidget,
    QButtonGroup,
    QRadioButton,
    QCheckBox,
    QTreeView,
    QTreeWidget,
    QTreeWidgetItem,
    QInputDialog,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QIcon


class GroupTreeManager(QWidget):
    """
    A professional Tree-based widget for managing dataset groups.
    Features: Drag & Drop, Auto-Assign, Context Menus.
    """

    def __init__(self, dataset_names, localize_func, parent=None):
        super().__init__(parent)
        self.dataset_names = dataset_names
        self.localize = localize_func
        self._setup_ui()
        self.reset()  # Initialize with all items in Unassigned

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 1. Unified Toolbar (User Request #1)
        toolbar = QFrame()
        toolbar.setStyleSheet(
            """
            QFrame { 
                background-color: #f9f9f9; 
                border: 1px solid #ccc; 
                border-bottom: none;
                border-top-left-radius: 4px; 
                border-top-right-radius: 4px; 
            }
        """
        )
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 4, 8, 4)
        tb_layout.setSpacing(10)

        # Toolbar Buttons
        btn_style = """
            QPushButton { 
                border: none; color: #333; font-size: 11px; font-weight: 600; 
                background: transparent; padding: 5px 10px; border-radius: 3px; 
            }
            QPushButton:hover { background-color: #e0e0e0; color: #000; }
        """

        self.btn_create = QPushButton("➕ Create Group")
        self.btn_create.setStyleSheet(btn_style)
        self.btn_create.clicked.connect(self.create_group_dialog)

        self.btn_auto = QPushButton("✨ Auto-Assign")
        self.btn_auto.setStyleSheet(btn_style)
        self.btn_auto.clicked.connect(self.auto_assign)

        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.setStyleSheet(btn_style)
        self.btn_reset.clicked.connect(self.reset)

        tb_layout.addWidget(self.btn_create)
        tb_layout.addWidget(self.btn_auto)
        tb_layout.addWidget(self.btn_reset)
        tb_layout.addStretch()

        layout.addWidget(toolbar)

        # 2. The Tree Gadget (User Request #2)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDropIndicatorShown(True)
        self.tree.setDragDropMode(QAbstractItemView.InternalMove)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # Styling the Tree to look professional
        self.tree.setStyleSheet(
            """
            QTreeWidget {
                border: 1px solid #ccc;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
                font-size: 13px;
            }
            QTreeWidget::item { 
                height: 28px; 
                padding-left: 4px;
            }
            QTreeWidget::item:hover { background-color: #f0f0f0; }
            QTreeWidget::item:selected { background-color: #e7f3ff; color: #0078d4; }
        """
        )

        layout.addWidget(self.tree)

    def reset(self):
        """Reset tree: Clear all groups, put everything in 'Unassigned'."""
        self.tree.clear()

        # Create immutable "Unassigned" group
        self.unassigned_root = QTreeWidgetItem(self.tree)
        self.unassigned_root.setText(0, "📂 Unassigned Datasets")
        self.unassigned_root.setFlags(
            Qt.ItemIsEnabled | Qt.ItemIsDropEnabled
        )  # Not selectable/draggable itself, just a container
        self.unassigned_root.setForeground(0, Qt.darkGray)

        # Add datasets
        icon_dataset = QIcon()  # You can add a file icon here if available
        for name in self.dataset_names:
            item = QTreeWidgetItem(self.unassigned_root)
            item.setText(0, name)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)

        self.tree.expandAll()

    def create_group_dialog(self):
        """Prompt user for group name and create it - NOW USING MULTI-GROUP DIALOG."""
        create_logs(__name__, __file__, "=" * 70, status="debug")
        create_logs(__name__, __file__, "CREATE GROUP BUTTON CLICKED (GroupTreeManager)", status="debug")
        create_logs(
            __name__,
            __file__,
            f"Timestamp: {__import__('datetime').datetime.now()}",
            status="debug",
        )
        create_logs(__name__, __file__, "Method: create_group_dialog", status="debug")
        create_logs(__name__, __file__, f"Instance: {self}", status="debug")
        create_logs(__name__, __file__, "=" * 70, status="debug")
        
        self._open_multi_group_dialog()
    
    def _open_multi_group_dialog(self):
        """Open the multi-group creation dialog."""
        import traceback

        create_logs(__name__, __file__, "=" * 70, status="debug")
        create_logs(__name__, __file__, "OPENING MULTI-GROUP DIALOG (GroupTreeManager)", status="debug")
        create_logs(__name__, __file__, "=" * 70, status="debug")
        create_logs(
            __name__,
            __file__,
            "Step 1: Attempting to import MultiGroupCreationDialog...",
            status="debug",
        )
        
        try:
            from pages.analysis_page_utils.multi_group_dialog import MultiGroupCreationDialog
            create_logs(__name__, __file__, "MultiGroupCreationDialog imported successfully!", status="debug")
            create_logs(__name__, __file__, f"Class: {MultiGroupCreationDialog}", status="debug")
        except Exception as e:
            create_logs(__name__, __file__, f"Failed to import MultiGroupCreationDialog: {e}", status="error")
            create_logs(__name__, __file__, f"Exception type: {type(e).__name__}", status="error")
            create_logs(__name__, __file__, "Full traceback:", status="error")
            create_logs(__name__, __file__, traceback.format_exc(), status="error")
            
            from PySide6.QtWidgets import QMessageBox

            create_logs(
                "GroupTreeManager-open_multi_group_dialog",
                "views_widget",
                f"Failed to import multi-group dialog: {e}\n\n{traceback.format_exc()}",
                "error",
            )

            QMessageBox.critical(
                self,
                self.localize("ANALYSIS_PAGE.group_assignment_import_error_title"),
                self.localize(
                    "ANALYSIS_PAGE.group_assignment_import_error_message",
                    error=str(e),
                ),
            )
            return
        
        try:
            create_logs(__name__, __file__, "Step 2: Creating dialog instance...", status="debug")
            create_logs(__name__, __file__, f"Dataset count: {len(self.dataset_names)}", status="debug")
            create_logs(__name__, __file__, f"Localize function: {self.localize}", status="debug")
            
            dialog = MultiGroupCreationDialog(
                self.dataset_names,
                self.localize,
                self
            )
            create_logs(__name__, __file__, "Dialog created successfully!", status="debug")
            create_logs(__name__, __file__, f"Dialog title: {dialog.windowTitle()}", status="debug")
            create_logs(__name__, __file__, "Step 3: Calling dialog.exec()...", status="debug")
            
            from PySide6.QtWidgets import QDialog
            result = dialog.exec()
            create_logs(__name__, __file__, "Step 4: Dialog closed", status="debug")
            create_logs(__name__, __file__, f"Result: {result} (Accepted={QDialog.Accepted})", status="debug")
            
            if result == QDialog.Accepted:
                create_logs(__name__, __file__, "Dialog was ACCEPTED", status="debug")
                assignments = dialog.get_assignments()
                create_logs(__name__, __file__, f"Assignments: {assignments}", status="debug")
                
                if assignments:
                    # Clear existing groups first (keep Unassigned)
                    root = self.tree.invisibleRootItem()
                    items_to_remove = []
                    for i in range(root.childCount()):
                        item = root.child(i)
                        if item != self.unassigned_root:
                            items_to_remove.append(item)
                    
                    for item in items_to_remove:
                        root.removeChild(item)
                    
                    # Create groups and assign datasets
                    for group_name, dataset_list in assignments.items():
                        create_logs(
                            __name__,
                            __file__,
                            f"Creating group '{group_name}' with {len(dataset_list)} datasets",
                            status="debug",
                        )
                        
                        # Create group
                        group_item = self.add_group(group_name)
                        
                        # Move datasets from Unassigned to this group
                        for dataset_name in dataset_list:
                            # Find dataset in Unassigned
                            for i in range(self.unassigned_root.childCount()):
                                item = self.unassigned_root.child(i)
                                if item.text(0) == dataset_name:
                                    # Move to group
                                    cloned_item = item.clone()
                                    self.unassigned_root.removeChild(item)
                                    group_item.addChild(cloned_item)
                                    break
                    
                    # Show success message
                    from PySide6.QtWidgets import QMessageBox
                    total_assigned = sum(len(datasets) for datasets in assignments.values())
                    group_summary = "\n".join(
                        [
                            f"• {name}: {len(datasets)} dataset(s)"
                            for name, datasets in assignments.items()
                        ]
                    )
                    QMessageBox.information(
                        self,
                        self.localize("ANALYSIS_PAGE.groups_created_title"),
                        self.localize(
                            "ANALYSIS_PAGE.groups_created_message",
                            group_count=len(assignments),
                            dataset_count=total_assigned,
                            group_summary=group_summary,
                        ),
                    )
                else:
                    create_logs(__name__, __file__, "No assignments returned", status="debug")
            else:
                create_logs(__name__, __file__, "Dialog was CANCELLED/REJECTED", status="debug")
                
        except Exception as e:
            create_logs(__name__, __file__, f"Exception in _open_multi_group_dialog: {e}", status="error")
            create_logs(__name__, __file__, f"Exception type: {type(e).__name__}", status="error")
            create_logs(__name__, __file__, traceback.format_exc(), status="error")
            
            from PySide6.QtWidgets import QMessageBox
            create_logs(
                "GroupTreeManager-open_multi_group_dialog",
                "views_widget",
                f"An error occurred while creating groups: {e}",
                "error",
            )

            QMessageBox.critical(
                self,
                self.localize("COMMON.error"),
                self.localize(
                    "ANALYSIS_PAGE.group_assignment_error_message",
                    error=str(e),
                ),
            )
        finally:
            create_logs(__name__, __file__, "=" * 70, status="debug")
            create_logs(__name__, __file__, "_open_multi_group_dialog completed (GroupTreeManager)", status="debug")
            create_logs(__name__, __file__, "=" * 70, status="debug")
    
    def _old_create_group_dialog_DISABLED(self):
        """OLD METHOD - DISABLED - Use _open_multi_group_dialog instead."""
        dialog = QInputDialog(self)
        dialog.setWindowTitle(
            self.localize("ANALYSIS_PAGE.create_group_title")
            if self.localize
            else "Create Group"
        )
        dialog.setLabelText(
            self.localize("ANALYSIS_PAGE.group_name_label")
            if self.localize
            else "Group Name:"
        )
        dialog.setStyleSheet(
            """
            QInputDialog {
                background-color: #ffffff;
            }
            QLabel {
                color: #2c3e50;
                font-size: 13px;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #dfe3ea;
                border-radius: 4px;
                background-color: #ffffff;
                color: #2c3e50;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: #0078d4;
            }
            QPushButton {
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 13px;
                min-width: 80px;
            }
            QPushButton[text="OK"] {
                background-color: #0078d4;
                color: white;
                border: none;
            }
            QPushButton[text="OK"]:hover {
                background-color: #006abc;
            }
            QPushButton[text="Cancel"] {
                background-color: white;
                color: #2c3e50;
                border: 1px solid #dfe3ea;
            }
            QPushButton[text="Cancel"]:hover {
                background-color: #f8f9fa;
            }
        """
        )

        ok = dialog.exec()
        text = dialog.textValue()

        if ok and text:
            self.add_group(text)

    def add_group(self, name):
        """Add a new group folder to the tree."""
        group_root = QTreeWidgetItem(self.tree)
        group_root.setText(0, f"🧪 {name}")
        group_root.setFlags(
            Qt.ItemIsEnabled
            | Qt.ItemIsSelectable
            | Qt.ItemIsDropEnabled
            | Qt.ItemIsEditable
        )
        # Insert before 'Unassigned' (which is usually last index or 0 depending on logic, let's just append)
        self.tree.addTopLevelItem(group_root)
        group_root.setExpanded(True)
        return group_root

    def auto_assign(self):
        """
        Auto-assign datasets to existing groups based on name matching.

        Logic:
        1. Check if groups exist (excluding Unassigned)
        2. If no groups, show dialog prompting user to create groups first
        3. For each unassigned dataset, check if group name appears in dataset name
        4. Case-insensitive matching with partial string search
        """
        from PySide6.QtWidgets import QMessageBox
        import re

        # 1. Get all existing groups (exclude Unassigned)
        existing_groups = []
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            group_item = root.child(i)
            if group_item != self.unassigned_root:
                # Remove emoji prefix if present
                group_name = group_item.text(0).replace("🧪 ", "")
                existing_groups.append((group_name, group_item))

        # 2. Check if any groups exist
        if not existing_groups:
            msg = QMessageBox(self)
            msg.setWindowTitle(
                self.localize("ANALYSIS_PAGE.auto_assign_no_groups_title")
                if self.localize
                else "No Groups Available"
            )
            msg.setText(
                self.localize("ANALYSIS_PAGE.auto_assign_no_groups_message")
                if self.localize
                else "Please create at least one group before using Auto-Assign.\n\nClick 'Create Group' to get started."
            )
            msg.setIcon(QMessageBox.Information)
            msg.setStyleSheet(
                """
                QMessageBox {
                    background-color: white;
                }
                QLabel {
                    color: #2c3e50;
                    font-size: 13px;
                }
                QPushButton {
                    background-color: #0078d4;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 6px 16px;
                    font-weight: 600;
                    font-size: 13px;
                    min-width: 60px;
                }
                QPushButton:hover {
                    background-color: #006abc;
                }
            """
            )
            msg.exec()
            return

        # 3. Gather unassigned items
        items_to_process = []
        for i in range(self.unassigned_root.childCount()):
            items_to_process.append(self.unassigned_root.child(i))

        if not items_to_process:
            return  # Nothing to assign

        # 4. Match datasets to groups
        assigned_count = 0
        for item in items_to_process:
            dataset_name = item.text(0)
            dataset_name_lower = dataset_name.lower()

            # Try to match with existing groups
            best_match = None
            best_match_length = 0

            for group_name, group_item in existing_groups:
                group_name_lower = group_name.lower()

                # Check if group name appears in dataset name (case-insensitive)
                if group_name_lower in dataset_name_lower:
                    # Prefer longer matches (e.g., "MGUS" over "MM" if both match)
                    if len(group_name) > best_match_length:
                        best_match = group_item
                        best_match_length = len(group_name)

            # Assign to best matching group
            if best_match:
                cloned_item = item.clone()
                self.unassigned_root.removeChild(item)
                best_match.addChild(cloned_item)
                assigned_count += 1

        # Optional: Show summary message if useful
        # print(f"Auto-assigned {assigned_count} dataset(s) to existing groups")

    def get_groups(self):
        """
        Returns dictionary of { "GroupName": ["Dataset1", "Dataset2"] }
        Ignores the 'Unassigned' folder.
        """
        result = {}
        root = self.tree.invisibleRootItem()

        for i in range(root.childCount()):
            group_item = root.child(i)

            # Skip the Unassigned folder
            if group_item == self.unassigned_root:
                continue

            group_name = group_item.text(0).replace("🧪 ", "")  # Remove emoji
            datasets = []

            for j in range(group_item.childCount()):
                datasets.append(group_item.child(j).text(0))

            if datasets:
                result[group_name] = datasets

        return result
