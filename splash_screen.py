"""
Splash Screen with Progress Bar for Raman App
Shows loading progress during application startup
"""

from PySide6.QtWidgets import QSplashScreen, QProgressBar, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont
import os


class SplashScreen(QSplashScreen):
    """Custom splash screen with progress bar and status messages."""

    def __init__(self):
        # Try to load splash image, fallback to colored background
        splash_path = os.path.join(os.path.dirname(__file__), "assets", "splash.png")

        if os.path.exists(splash_path):
            pixmap = QPixmap(splash_path)
        else:
            # Create a simple gradient background if no splash image
            pixmap = QPixmap(600, 400)
            pixmap.fill(QColor(45, 52, 70))  # Dark blue-gray

            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Draw gradient overlay
            from PySide6.QtGui import QLinearGradient

            gradient = QLinearGradient(0, 0, 0, 400)
            gradient.setColorAt(0.0, QColor(45, 52, 70, 255))
            gradient.setColorAt(1.0, QColor(30, 35, 50, 255))
            painter.fillRect(pixmap.rect(), gradient)

            # Draw app title
            painter.setPen(QColor(255, 255, 255))
            title_font = QFont("Segoe UI", 28, QFont.Weight.Bold)
            painter.setFont(title_font)
            painter.drawText(
                pixmap.rect().adjusted(0, -100, 0, 0),
                Qt.AlignmentFlag.AlignCenter,
                "Raman Spectroscopy",
            )

            subtitle_font = QFont("Segoe UI", 16)
            painter.setFont(subtitle_font)
            painter.drawText(
                pixmap.rect().adjusted(0, -40, 0, 0),
                Qt.AlignmentFlag.AlignCenter,
                "Analysis Application",
            )

            painter.end()

        super().__init__(pixmap, Qt.WindowType.WindowStaysOnTopHint)

        self.progress_value = 0
        self.status_message = "Initializing..."

        # Set message alignment
        self.setFont(QFont("Segoe UI", 10))

    def show_progress(self, progress: int, message: str = ""):
        """Update progress bar and status message.

        Args:
            progress: Progress percentage (0-100)
            message: Status message to display
        """
        self.progress_value = progress
        if message:
            self.status_message = message

        self.showMessage(
            f"{self.status_message}\n{self.progress_value}%",
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            QColor(255, 255, 255),
        )

        # Force repaint
        self.repaint()

    def drawContents(self, painter):
        """Override to draw custom progress bar."""
        super().drawContents(painter)

        # Draw progress bar
        bar_width = 500
        bar_height = 8
        bar_x = (self.width() - bar_width) // 2
        bar_y = self.height() - 60

        # Background
        painter.fillRect(
            bar_x, bar_y, bar_width, bar_height, QColor(100, 100, 100, 180)
        )

        # Progress fill
        progress_width = int((bar_width * self.progress_value) / 100)
        painter.fillRect(
            bar_x, bar_y, progress_width, bar_height, QColor(76, 175, 80, 255)
        )  # Green


def create_splash() -> SplashScreen:
    """Create and show splash screen."""
    splash = SplashScreen()
    splash.show()
    return splash
