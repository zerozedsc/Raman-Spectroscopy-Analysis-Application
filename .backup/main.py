import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PySide6.QtGui import QFontDatabase

# --- Import project modules ---
from utils import *
from configs.style.stylesheets import get_main_stylesheet
from pages.home_page import HomePage
from pages.workspace_page import WorkspacePage
from components.toast import Toast

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(LOCALIZE("MAIN_WINDOW.title"))
        self.resize(1440, 900)
        self.setMinimumHeight(600)  # Minimum height for non-maximized windows
        self.setMinimumWidth(1000)   # Minimum width to maintain layout

        # --- Central stacked widget to manage main views (Home vs Workspace) ---
        self.central_stack = QStackedWidget()
        self.setCentralWidget(self.central_stack)

        # --- Create Pages ---
        self.home_page = HomePage()
        self.workspace_page = WorkspacePage()

        self.central_stack.addWidget(self.home_page)
        self.central_stack.addWidget(self.workspace_page)

        # --- Create Toast Notification Widget, parented to the main window ---
        self.toast = Toast(self)

        # --- Connect signals ---
        self.home_page.projectOpened.connect(self.open_project_workspace)
        self.home_page.newProjectCreated.connect(self.open_project_workspace)
        # Connect the notification signal from the data page to the toast's slot
        self.workspace_page.data_page.showNotification.connect(self.toast.show_message)
        # Connect the notification signal from the analysis page to the toast's slot
        self.workspace_page.analysis_page.showNotification.connect(self.toast.show_message)

        # Start on the home page
        self.central_stack.setCurrentWidget(self.home_page)

    def open_project_workspace(self, project_path: str):
        """
        Loads the project data using the ProjectManager and switches
        to the main workspace view.
        """
        if PROJECT_MANAGER.load_project(project_path):
            self.workspace_page.load_project(project_path)
            self.central_stack.setCurrentWidget(self.workspace_page)
            project_name = os.path.basename(project_path)
            self.toast.show_message(LOCALIZE("NOTIFICATIONS.project_loaded_success", name=project_name), "success")
        else:
            self.toast.show_message(LOCALIZE("NOTIFICATIONS.project_loaded_error"), "error")

    def resizeEvent(self, event):
        # Ensure toast repositions correctly on window resize
        super().resizeEvent(event)
        if self.toast.isVisible():
            self.toast.hide() # Hide to prevent visual artifacts, will show again on next message


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Load bundled fonts into the application database
    load_application_fonts()
    
    # Language is already set from command-line args in utils.py
    # Only use config language if no command-line arg was provided
    # Note: args.lang is parsed in utils.py and used to initialize LOCALIZEMANAGER
    # The language is already active, so we don't need to call set_language() again
    
    # Generate and apply the dynamic, language-aware stylesheet
    font_family = LOCALIZE("APP_CONFIG.font_family")
    dynamic_stylesheet = get_main_stylesheet(font_family)
    app.setStyleSheet(dynamic_stylesheet)
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
