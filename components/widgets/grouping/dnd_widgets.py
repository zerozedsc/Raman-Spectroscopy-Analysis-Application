from __future__ import annotations

from PySide6.QtCore import QMimeData, Qt, QSize
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QWidget,
)

from configs.configs import create_logs
from components.widgets.icons import load_icon


_MIME_TYPE = "application/x-raman-dataset-name"


class DatasetSourceList(QListWidget):
    """Drag source list for dataset names (copy semantics)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setDragEnabled(True)
        self.setDefaultDropAction(Qt.CopyAction)
        self.setDragDropMode(QListWidget.DragOnly)

    def startDrag(self, supportedActions):  # noqa: N802 (Qt override)
        item = self.currentItem()
        if item is None:
            return

        dataset_name = item.data(Qt.UserRole) or item.text()
        mime = QMimeData()
        mime.setData(_MIME_TYPE, str(dataset_name).encode("utf-8"))
        mime.setText(str(dataset_name))

        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)


class GroupDropList(QListWidget):
    """Drop target list for datasets in a group."""

    def __init__(self, *, group_id: str, localize_func=None, parent=None):
        super().__init__(parent)
        self.group_id = group_id
        self.localize = localize_func
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDragEnabled(False)
        self.setDragDropMode(QListWidget.DropOnly)
        self.setDefaultDropAction(Qt.CopyAction)

    def _notify_parent_changed(self):
        """Notify the owning widget to persist changes.

        We walk up the parent chain and call a best-effort hook.
        """
        try:
            parent = self.parent()
            # The QListWidget is often embedded inside a tab page -> walk up a bit.
            while parent is not None:
                if hasattr(parent, "_on_groups_changed"):
                    parent._on_groups_changed()
                    break
                if hasattr(parent, "_save_groups_to_project"):
                    parent._save_groups_to_project()
                    break
                parent = parent.parent()
        except Exception:
            pass

    def add_dataset(self, name: str) -> bool:
        """Add a dataset entry with an inline delete button.

        Returns:
            True if the dataset was added; False if it already existed or was invalid.
        """
        name = (name or "").strip()
        if not name:
            return False

        # Avoid duplicates
        for i in range(self.count()):
            it = self.item(i)
            if (it.data(Qt.UserRole) or it.text()) == name:
                return False

        item = QListWidgetItem(str(name))
        item.setData(Qt.UserRole, str(name))
        self.addItem(item)

        # Build an item widget with a delete button
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(8, 2, 6, 2)
        row.setSpacing(8)
        lbl = QLabel(str(name))
        lbl.setStyleSheet("font-size: 12px;")
        row.addWidget(lbl, 1)

        btn = QPushButton()
        btn.setFixedSize(28, 28)
        try:
            btn.setToolTip(
                self.localize("ML_PAGE.delete_dataset_tooltip", name=name)
                if self.localize
                else f"Delete '{name}'"
            )
        except Exception:
            btn.setToolTip(f"Delete '{name}'")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            """
            QPushButton {
                background-color: #fff5f5;
                border: 1px solid #dc3545;
                border-radius: 4px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #f8d7da;
                border-color: #b02a37;
            }
            QPushButton:pressed {
                background-color: #f5c6cb;
            }
            """
        )
        try:
            btn.setIcon(load_icon("trash", QSize(16, 16), "#dc3545"))
            btn.setIconSize(QSize(16, 16))
        except Exception:
            btn.setText("✕")

        def _do_remove():
            row_idx = self.row(item)
            if row_idx >= 0:
                self.takeItem(row_idx)
                self._notify_parent_changed()

        btn.clicked.connect(_do_remove)
        row.addWidget(btn, 0)
        w.setLayout(row)
        item.setSizeHint(w.sizeHint())
        self.setItemWidget(item, w)

        self._notify_parent_changed()
        return True

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasFormat(_MIME_TYPE) or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802
        if event.mimeData().hasFormat(_MIME_TYPE) or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):  # noqa: N802
        try:
            name = None
            if event.mimeData().hasFormat(_MIME_TYPE):
                name = bytes(event.mimeData().data(_MIME_TYPE)).decode("utf-8")
            elif event.mimeData().hasText():
                name = event.mimeData().text()

            name = (name or "").strip()
            if not name:
                event.ignore()
                return

            # Add (handles duplicate checks + UI widget)
            _ = self.add_dataset(name)
            create_logs(
                "GroupDnD",
                "group_drop",
                f"[DEBUG] Dropped dataset '{name}' into group '{self.group_id}'",
                status="debug",
            )
            event.acceptProposedAction()
        except Exception as e:
            create_logs(
                "GroupDnD",
                "group_drop_error",
                f"Failed dropEvent: {e}",
                status="warning",
            )
            event.ignore()

    def keyPressEvent(self, event):  # noqa: N802
        """Allow removing selected items with Delete/Backspace."""
        key = event.key()
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            for item in list(self.selectedItems()):
                row = self.row(item)
                if row >= 0:
                    self.takeItem(row)
            self._notify_parent_changed()
            return
        super().keyPressEvent(event)
