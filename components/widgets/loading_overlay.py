from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class LoadingOverlay(QWidget):
    """A simple translucent overlay to indicate blocking work.

    Intended usage:
        overlay = LoadingOverlay(parent_widget)
        overlay.show_loading("Loading...")
        ... do work ...
        overlay.hide_loading()

    Notes:
        - The overlay automatically tracks the size of its parent widget.
        - The overlay blocks mouse/keyboard interaction with the underlying UI.
    """

    def __init__(self, target: QWidget, *, parent: Optional[QWidget] = None):
        super().__init__(parent or target)
        # NOTE: Qt can still dispatch events to an eventFilter during teardown.
        # If the Python object is partially GC'd, instance attributes may be gone.
        # Keep access to the target defensive throughout this class.
        self._target = target

        self.setObjectName("loadingOverlay")
        self.setVisible(False)

        # Track geometry of the target widget.
        self._install_filter()

        self.setStyleSheet(
            """
            QWidget#loadingOverlay {
                background-color: rgba(255, 255, 255, 0.85);
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)

        self._label = QLabel("⏳ Loading...")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(
            """
            QLabel {
                font-size: 16px;
                font-weight: 700;
                color: #0078d4;
                background-color: white;
                padding: 18px 32px;
                border-radius: 10px;
                border: 2px solid #0078d4;
            }
            """
        )
        layout.addWidget(self._label)

        self._sync_to_target()

    def _install_filter(self) -> None:
        try:
            target = getattr(self, "_target", None)
            if target is not None:
                target.installEventFilter(self)
        except Exception:
            pass

    def _remove_filter(self) -> None:
        try:
            target = getattr(self, "_target", None)
            if target is not None:
                target.removeEventFilter(self)
        except Exception:
            pass

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        target = getattr(self, "_target", None)
        if target is not None and watched is target and event.type() in (QEvent.Resize, QEvent.Move, QEvent.Show):
            self._sync_to_target()
        return super().eventFilter(watched, event)

    def _sync_to_target(self) -> None:
        try:
            target = getattr(self, "_target", None)
            if target is None:
                return
            self.setGeometry(target.rect())
            self.raise_()
        except Exception:
            pass

    def detach(self) -> None:
        """Detach the overlay from its target.

        Call this when the target is about to be destroyed to avoid eventFilter
        callbacks referencing a partially-torn-down Python object.
        """
        self._remove_filter()
        try:
            self._target = None
        except Exception:
            pass

    def show_loading(self, text: str | None = None) -> None:
        try:
            if text:
                self._label.setText(str(text))
        except Exception:
            pass

        self._sync_to_target()
        self.setVisible(True)
        self.raise_()

    def hide_loading(self) -> None:
        try:
            self.setVisible(False)
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        try:
            self.detach()
        except Exception:
            pass
        super().closeEvent(event)
