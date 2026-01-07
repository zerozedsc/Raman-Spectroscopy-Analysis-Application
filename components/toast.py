import sys
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PySide6.QtGui import QFont


class Toast(QWidget):
    """
    A non-blocking toast/bubble notification widget that overlays its parent.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self.layout = QVBoxLayout(self)
        self.label = QLabel(self)
        self.label.setObjectName("toastLabel")
        self.layout.addWidget(self.label)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._hide_animation)

        self.animation = QPropertyAnimation(self, b"windowOpacity", self)
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.hide()

    def show_message(self, message: str, level: str = "info", duration: int = 3000):
        """
        Shows the toast notification.

        Args:
            message (str): The message to display.
            level (str): 'info', 'success', or 'error' for different styling.
            duration (int): How long the message stays visible in milliseconds.
        """
        if not self.parent():
            return

        self.label.setText(message)
        self.label.setProperty("level", level)
        self.style().polish(self.label)  # Apply new property style

        self.adjustSize()
        self._reposition()

        self.timer.start(duration)
        self._show_animation()

    def _reposition(self):
        """Positions the toast at the bottom-center of the parent widget."""
        parent_geo = self.parent().geometry()
        pos = parent_geo.bottomLeft() - QPoint(0, self.height() + 20)
        pos.setX(parent_geo.x() + (parent_geo.width() - self.width()) // 2)
        self.move(pos)

    def _show_animation(self):
        self.setWindowOpacity(0.0)
        self.show()
        self.animation.setDirection(QPropertyAnimation.Direction.Forward)
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.start()

    def _hide_animation(self):
        self.animation.setDirection(QPropertyAnimation.Direction.Backward)
        self.animation.setStartValue(1.0)
        self.animation.setEndValue(0.0)
        self.animation.finished.connect(self.hide)
        self.animation.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.parent():
            self._reposition()
