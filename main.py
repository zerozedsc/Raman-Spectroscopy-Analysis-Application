"""
Optimized main.py with lazy imports and splash screen
Dramatically improves startup time for packaged executable
"""

import sys
import os
import argparse

# Early imports - only absolute essentials
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFontDatabase, QIcon

# Import splash screen (lightweight)
from splash_screen import create_splash


def get_app_icon() -> QIcon:
    """Resolve the main application icon (SVG source-of-truth)."""
    try:
        if getattr(sys, "frozen", False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))

        icon_path = os.path.join(base_path, "assets", "icons", "app-icon.svg")
        if os.path.exists(icon_path):
            return QIcon(icon_path)
    except Exception:
        pass
    return QIcon()


def parse_arguments():
    """Parse command-line arguments for application configuration."""
    parser = argparse.ArgumentParser(
        description="Raman Spectroscopy Analysis Application"
    )
    
    # ✅ PRODUCTION FIX: Default to False for portable/installer builds
    # Developers can still use --debug true when running from source
    # Users of portable/installer get clean production mode by default
    parser.add_argument(
        "--debug",
        type=lambda x: x.lower() == "true",
        default=False,  # Changed from True to False for production mode
        help="Enable debug mode (default: False). Use --debug true for development."
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code (en, ja). Default: en"
    )
    return parser.parse_args()


def lazy_import_modules(splash):
    """Lazy import heavy modules with progress updates."""

    # Stage 1: Core utilities (10%)
    splash.show_progress(10, "Loading core utilities...")
    QApplication.processEvents()
    from utils import LOCALIZE, PROJECT_MANAGER, load_application_fonts

    # Stage 2: Stylesheets (20%)
    splash.show_progress(20, "Loading stylesheets...")
    QApplication.processEvents()
    from configs.style.stylesheets import get_main_stylesheet

    # Stage 3: Components (40%)
    splash.show_progress(40, "Loading UI components...")
    QApplication.processEvents()
    from components.toast import Toast

    # Stage 4: Pages (60%)
    splash.show_progress(60, "Loading application pages...")
    QApplication.processEvents()
    from pages.home_page import HomePage

    # Stage 5: Workspace (80%)
    splash.show_progress(80, "Loading workspace...")
    QApplication.processEvents()
    from pages.workspace_page import WorkspacePage

    # Stage 6: Final setup (90%)
    splash.show_progress(90, "Initializing application...")
    QApplication.processEvents()

    from PySide6.QtWidgets import QMainWindow, QStackedWidget

    return {
        "LOCALIZE": LOCALIZE,
        "PROJECT_MANAGER": PROJECT_MANAGER,
        "load_application_fonts": load_application_fonts,
        "get_main_stylesheet": get_main_stylesheet,
        "Toast": Toast,
        "HomePage": HomePage,
        "WorkspacePage": WorkspacePage,
        "QMainWindow": QMainWindow,
        "QStackedWidget": QStackedWidget,
    }


class MainWindow:
    """Main window class - constructed after all imports."""

    def __init__(self, modules):
        """Initialize with pre-imported modules."""
        QMainWindow = modules["QMainWindow"]
        QStackedWidget = modules["QStackedWidget"]
        LOCALIZE = modules["LOCALIZE"]
        PROJECT_MANAGER = modules["PROJECT_MANAGER"]
        HomePage = modules["HomePage"]
        WorkspacePage = modules["WorkspacePage"]
        Toast = modules["Toast"]

        # Create actual window instance
        self.window = QMainWindow()
        self.window.setWindowIcon(get_app_icon())
        self.window.setWindowTitle(LOCALIZE("MAIN_WINDOW.title"))
        self.window.resize(1440, 900)
        self.window.setMinimumHeight(600)
        self.window.setMinimumWidth(1000)

        # Central stacked widget
        self.central_stack = QStackedWidget()
        self.window.setCentralWidget(self.central_stack)

        # Create Pages
        self.home_page = HomePage()
        self.workspace_page = WorkspacePage()

        self.central_stack.addWidget(self.home_page)
        self.central_stack.addWidget(self.workspace_page)

        # Create Toast
        self.toast = Toast(self.window)

        # Connect signals
        self.home_page.projectOpened.connect(
            lambda path: self.open_project_workspace(path, LOCALIZE, PROJECT_MANAGER)
        )
        self.home_page.newProjectCreated.connect(
            lambda path: self.open_project_workspace(path, LOCALIZE, PROJECT_MANAGER)
        )
        self.workspace_page.data_page.showNotification.connect(self.toast.show_message)
        self.workspace_page.analysis_page.showNotification.connect(
            self.toast.show_message
        )

        # Start on home page
        self.central_stack.setCurrentWidget(self.home_page)

        # Override resize event
        original_resize = self.window.resizeEvent

        def custom_resize(event):
            original_resize(event)
            if self.toast.isVisible():
                self.toast.hide()

        self.window.resizeEvent = custom_resize

    def open_project_workspace(self, project_path: str, LOCALIZE, PROJECT_MANAGER):
        """Load project and switch to workspace."""
        if PROJECT_MANAGER.load_project(project_path):
            self.workspace_page.load_project(project_path)
            self.central_stack.setCurrentWidget(self.workspace_page)
            project_name = os.path.basename(project_path)
            self.toast.show_message(
                LOCALIZE("NOTIFICATIONS.project_loaded_success", name=project_name),
                "success",
            )
        else:
            self.toast.show_message(
                LOCALIZE("NOTIFICATIONS.project_loaded_error"), "error"
            )

    def show(self):
        """Show the main window."""
        self.window.show()


def main():
    """Main application entry point with optimized loading."""
    
    # Parse command-line arguments FIRST (before any other imports)
    args = parse_arguments()
    
    # Set global DEBUG mode early (before creating logs)
    import configs.configs as cfg
    cfg.set_debug_mode(args.debug)

    # Create QApplication first (required for splash)
    app = QApplication(sys.argv)

    # Set application icon early (taskbar/window icon)
    app.setWindowIcon(get_app_icon())

    # Show splash screen immediately
    splash = create_splash()
    splash.show_progress(5, "Starting application...")
    app.processEvents()

    # Lazy load all modules with progress updates
    modules = lazy_import_modules(splash)

    # Load fonts
    splash.show_progress(92, "Loading fonts...")
    app.processEvents()
    modules["load_application_fonts"]()

    # Apply stylesheet
    splash.show_progress(95, "Applying styles...")
    app.processEvents()
    font_family = modules["LOCALIZE"]("APP_CONFIG.font_family")
    dynamic_stylesheet = modules["get_main_stylesheet"](font_family)
    app.setStyleSheet(dynamic_stylesheet)

    # Create main window
    splash.show_progress(98, "Creating main window...")
    app.processEvents()
    window = MainWindow(modules)

    # Show window and close splash
    splash.show_progress(100, "Ready!")
    app.processEvents()

    # Delay splash close slightly for smooth transition
    QTimer.singleShot(500, splash.close)
    QTimer.singleShot(300, window.show)
    
    # Focus and raise the window to bring it to front (after showing)
    QTimer.singleShot(600, lambda: window.window.raise_())
    QTimer.singleShot(600, lambda: window.window.activateWindow())

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
