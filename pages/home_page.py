from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFrame,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QSizePolicy,
    QGroupBox,
    QScrollArea,
    QDialog,
    QComboBox,
    QFormLayout,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt, QSize, Signal, QPropertyAnimation, QEasingCurve, QTimer
from PySide6.QtGui import QIcon, QPixmap, QPainter, QMouseEvent, QFont
from PySide6.QtSvg import QSvgRenderer
from configs.configs import create_logs
from components.widgets import *
from utils import *
import sys
import os
import json
from configs.user_settings import save_user_settings


class SettingsDialog(QDialog):
    """Settings dialog for language and theme selection."""
    
    settingsChanged = Signal(dict)  # Emits dict with 'language' and 'theme' keys
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(LOCALIZE("HOME_PAGE.settings_dialog_title"))
        self.setMinimumWidth(400)
        self.setModal(True)
        
        self._setup_ui()
        self._load_current_settings()
    
    def _setup_ui(self):
        """Setup the settings dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Form layout for settings
        form_layout = QFormLayout()
        form_layout.setSpacing(16)
        
        # Language selection
        self.language_combo = QComboBox()
        self.language_combo.addItem("English", "en")
        self.language_combo.addItem("日本語", "ja")
        form_layout.addRow(
            LOCALIZE("HOME_PAGE.settings_language_label"),
            self.language_combo
        )
        
        # Theme selection
        self.theme_combo = QComboBox()
        self.theme_combo.addItem(
            LOCALIZE("HOME_PAGE.settings_theme_light"),
            "light"
        )
        self.theme_combo.addItem(
            LOCALIZE("HOME_PAGE.settings_theme_dark"),
            "dark"
        )
        self.theme_combo.addItem(
            LOCALIZE("HOME_PAGE.settings_theme_system"),
            "system"
        )
        form_layout.addRow(
            LOCALIZE("HOME_PAGE.settings_theme_label"),
            self.theme_combo
        )

        # User requested: keep theme code, but hide/disable theme selection UI for now.
        try:
            theme_label = form_layout.labelForField(self.theme_combo)
            if theme_label is not None:
                theme_label.setVisible(False)
            self.theme_combo.setVisible(False)
            self.theme_combo.setEnabled(False)
        except Exception:
            pass
        
        layout.addLayout(form_layout)
        
        # Info label
        info_label = QLabel(LOCALIZE("HOME_PAGE.settings_restart_info"))
        info_label.setObjectName("settingsInfoLabel")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _load_current_settings(self):
        """Load current language and theme settings."""
        # Load current language (from LOCALIZEMANAGER)
        current_lang = LOCALIZEMANAGER.current_lang
        lang_index = self.language_combo.findData(current_lang)
        if lang_index >= 0:
            self.language_combo.setCurrentIndex(lang_index)
        
        # Load current theme from config
        current_theme = CONFIGS.get("theme", "system")
        theme_index = self.theme_combo.findData(current_theme)
        if theme_index >= 0:
            self.theme_combo.setCurrentIndex(theme_index)
    
    def get_settings(self) -> dict:
        """Get selected settings."""
        return {
            "language": self.language_combo.currentData(),
            "theme": self.theme_combo.currentData()
        }


class ActionCard(QWidget):
    """A modern, animated clickable card widget for primary actions."""

    clicked = Signal()

    def __init__(self, icon: QIcon, title: str, text: str, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("actionCard")
        self.setMinimumSize(260, 140)
        self.setMaximumSize(340, 180)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        # Enable hover tracking
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._setup_ui(icon, title, text)

    def _setup_ui(self, icon: QIcon, title: str, text: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Icon container
        icon_container = QWidget()
        icon_container.setFixedHeight(40)
        icon_layout = QHBoxLayout(icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)

        icon_label = QLabel()
        icon_label.setPixmap(icon.pixmap(QSize(32, 32)))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        icon_layout.addWidget(icon_label)
        icon_layout.addStretch()

        # Title
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        title_label.setWordWrap(True)

        # Description
        desc_label = QLabel(text)
        desc_label.setObjectName("cardDescription")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout.addWidget(icon_container)
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addStretch()

    def enterEvent(self, event):
        self.setProperty("hover", True)
        self.style().unpolish(self)
        self.style().polish(self)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setProperty("hover", False)
        self.style().unpolish(self)
        self.style().polish(self)
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class RecentProjectItemWidget(QWidget):
    """Enhanced widget for displaying recent projects with better visual hierarchy."""

    def __init__(self, project_name: str, last_modified: str, parent=None):
        super().__init__(parent)
        self.setObjectName("recentProjectItem")
        self.setMinimumHeight(72)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Track selection/click feedback via dynamic properties (stylesheet-driven)
        self.setProperty("selected", False)
        self.setProperty("clickedFlash", False)

        # Enable hover tracking
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._setup_ui(project_name, last_modified)

    def set_selected(self, selected: bool) -> None:
        try:
            self.setProperty("selected", bool(selected))
            self.style().unpolish(self)
            self.style().polish(self)
            self.update()
        except Exception:
            pass

    def flash_clicked(self, ms: int = 180) -> None:
        """Brief visual feedback for 'click succeeded' (without opening)."""
        try:
            self.setProperty("clickedFlash", True)
            self.style().unpolish(self)
            self.style().polish(self)
            self.update()

            def _clear() -> None:
                try:
                    self.setProperty("clickedFlash", False)
                    self.style().unpolish(self)
                    self.style().polish(self)
                    self.update()
                except Exception:
                    pass

            QTimer.singleShot(int(ms), _clear)
        except Exception:
            pass

    def _setup_ui(self, project_name: str, last_modified: str):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        # Project icon
        icon_label = QLabel()
        project_icon = load_icon("recent_projects", QSize(24, 24), "#0078d4")
        icon_label.setPixmap(project_icon.pixmap(QSize(24, 24)))
        icon_label.setFixedSize(24, 24)

        # Project info
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(4)

        name_label = QLabel(project_name)
        name_label.setObjectName("projectName")
        name_label.setWordWrap(True)

        time_label = QLabel(
            LOCALIZE("HOME_PAGE.last_modified_label", date=last_modified)
        )
        time_label.setObjectName("projectTime")

        info_layout.addWidget(name_label)
        info_layout.addWidget(time_label)

        layout.addWidget(icon_label)
        layout.addLayout(info_layout)
        layout.addStretch()


class HomePage(QWidget):
    """Modern, scientific-themed landing page with enhanced visual design and responsiveness."""

    newProjectCreated = Signal(str)
    projectOpened = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("homePage")
        self._setup_ui()
        self.populate_recent_projects()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create sidebar and content area
        sidebar = self._create_sidebar()
        content_area = self._create_content_area()

        # Add widgets with proper stretch factors
        main_layout.addWidget(sidebar, 0)  # No stretch - takes minimum space
        main_layout.addWidget(content_area, 1)  # Stretch - takes remaining space

    def _create_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("homeSidebar")

        # Set responsive width constraints
        sidebar.setMinimumWidth(320)
        sidebar.setMaximumWidth(420)
        sidebar.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(40, 48, 40, 48)
        layout.setSpacing(48)

        # Title section
        title_container = self._create_title_section()

        # Action cards section
        actions_container = self._create_actions_section()

        layout.addWidget(title_container)
        layout.addWidget(actions_container)
        layout.addStretch()  # Push content to top

        # Bottom-left version label (user requested)
        try:
            version_text = str(CONFIGS.get("version") or "").strip()
        except Exception:
            version_text = ""
        if version_text:
            version_label = QLabel(version_text)
            version_label.setObjectName("homeVersion")
            version_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
            version_label.setWordWrap(False)
            layout.addWidget(version_label)

        return sidebar

    def _create_title_section(self) -> QWidget:
        title_container = QWidget()
        title_container.setObjectName("homeTitleContainer")
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(10)

        # Title row: app icon + title (cleaner, more "product" feel)
        title_row = QWidget()
        title_row.setObjectName("homeTitleRow")
        title_row_layout = QHBoxLayout(title_row)
        title_row_layout.setContentsMargins(0, 0, 0, 0)
        title_row_layout.setSpacing(12)

        app_icon_label = QLabel()
        app_icon_label.setObjectName("homeAppIcon")
        try:
            ico = load_icon("app_icon", QSize(28, 28), "#0078d4")
            app_icon_label.setPixmap(ico.pixmap(QSize(28, 28)))
        except Exception:
            pass
        app_icon_label.setFixedSize(36, 36)
        app_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel(LOCALIZE("MAIN_WINDOW.title"))
        title_label.setObjectName("homeTitle")
        title_label.setWordWrap(True)
        title_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        title_row_layout.addWidget(app_icon_label)
        title_row_layout.addWidget(title_label, 1)

        subtitle_label = QLabel(LOCALIZE("HOME_PAGE.subtitle"))
        subtitle_label.setObjectName("homeSubtitle")
        subtitle_label.setWordWrap(True)
        subtitle_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        divider = QFrame()
        divider.setObjectName("homeTitleDivider")
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Plain)
        divider.setFixedHeight(1)

        title_layout.addWidget(title_row)
        title_layout.addWidget(subtitle_label)
        title_layout.addWidget(divider)

        return title_container

    def _create_actions_section(self) -> QWidget:
        actions_container = QWidget()
        actions_layout = QVBoxLayout(actions_container)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(20)

        # New project card
        new_icon = load_icon("new_project", QSize(32, 32), "#0078d4")
        new_card = ActionCard(
            new_icon,
            LOCALIZE("HOME_PAGE.new_project_button"),
            LOCALIZE("HOME_PAGE.new_project_desc"),
        )
        new_card.clicked.connect(self.handle_new_project)

        # Open project card
        open_icon = load_icon("open_project", QSize(32, 32), "#0078d4")
        open_card = ActionCard(
            open_icon,
            LOCALIZE("HOME_PAGE.open_project_button"),
            LOCALIZE("HOME_PAGE.open_project_desc"),
        )
        open_card.clicked.connect(self.handle_open_project)

        actions_layout.addWidget(new_card)
        actions_layout.addWidget(open_card)

        return actions_container

    def _create_content_area(self) -> QWidget:
        content_area = QWidget()
        content_area.setObjectName("homeContentArea")
        content_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        layout = QVBoxLayout(content_area)
        layout.setContentsMargins(48, 48, 48, 48)
        layout.setSpacing(32)

        # Header section
        header_container = self._create_header_section()

        # Recent projects list
        self.recent_projects_list = QListWidget()
        self.recent_projects_list.setObjectName("recentProjectsList")
        self.recent_projects_list.setFrameShape(QFrame.Shape.NoFrame)
        self.recent_projects_list.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.recent_projects_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.recent_projects_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.recent_projects_list.setSpacing(8)
        try:
            # Single click = highlight/feedback; double click = open.
            self.recent_projects_list.itemClicked.connect(self._handle_recent_item_clicked)
        except Exception:
            pass
        self.recent_projects_list.itemDoubleClicked.connect(
            self.handle_recent_item_opened
        )

        layout.addWidget(header_container)
        layout.addWidget(self.recent_projects_list, 1)

        return content_area

    def _handle_recent_item_clicked(self, item: QListWidgetItem) -> None:
        """Give clear feedback that the click was registered (selection + brief flash)."""
        try:
            for i in range(self.recent_projects_list.count()):
                it = self.recent_projects_list.item(i)
                w = self.recent_projects_list.itemWidget(it)
                if isinstance(w, RecentProjectItemWidget):
                    w.set_selected(it is item)
        except Exception:
            pass

        try:
            w = self.recent_projects_list.itemWidget(item)
            if isinstance(w, RecentProjectItemWidget):
                w.flash_clicked()
        except Exception:
            pass

    def _create_header_section(self) -> QWidget:
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(16)

        # Recent projects icon
        recent_icon = load_icon("recent_projects", QSize(24, 24), "#0078d4")
        icon_label = QLabel()
        icon_label.setPixmap(recent_icon.pixmap(QSize(24, 24)))
        icon_label.setFixedSize(24, 24)

        # Header label
        header_label = QLabel(LOCALIZE("HOME_PAGE.recent_projects_title"))
        header_label.setObjectName("recentProjectsHeader")
        header_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        # Settings button
        settings_button = QPushButton(LOCALIZE("HOME_PAGE.settings_button"))
        settings_button.setObjectName("settingsButton")
        settings_button.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_button.clicked.connect(self.handle_settings)

        header_layout.addWidget(icon_label)
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(settings_button)

        return header_container

    def populate_recent_projects(self):
        """Populate the recent projects list with enhanced styling."""
        self.recent_projects_list.clear()
        recent_projects = PROJECT_MANAGER.get_recent_projects()

        if not recent_projects:
            # Create empty state item
            empty_item = QListWidgetItem()
            # empty_item.setObjectName("emptyStateLabel")
            empty_item.setText(LOCALIZE("HOME_PAGE.no_recent_projects"))
            empty_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_item.setFlags(empty_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            empty_item.setSizeHint(QSize(0, 80))
            self.recent_projects_list.addItem(empty_item)
        else:
            for project in recent_projects:
                list_item = QListWidgetItem(self.recent_projects_list)
                list_item.setData(Qt.ItemDataRole.UserRole, project["path"])

                # Create custom widget for project item
                item_widget = RecentProjectItemWidget(
                    project["name"], project["last_modified"]
                )
                list_item.setSizeHint(item_widget.sizeHint())

                self.recent_projects_list.addItem(list_item)
                self.recent_projects_list.setItemWidget(list_item, item_widget)

    def handle_new_project(self):
        """Handle new project creation with improved error handling."""
        project_name, ok = QInputDialog.getText(
            self,
            LOCALIZE("HOME_PAGE.new_project_dialog_title"),
            LOCALIZE("HOME_PAGE.new_project_dialog_label"),
        )
        if ok and project_name.strip():
            project_path = PROJECT_MANAGER.create_new_project(project_name.strip())
            if project_path:
                self.populate_recent_projects()
                self.newProjectCreated.emit(project_path)
            else:
                create_logs(
                    "new_project_error",
                    "HomePage",
                    f"Project creation failed: {project_name.strip()} already exists.",
                    status="warning"
                )

                QMessageBox.warning(
                    self,
                    LOCALIZE("COMMON.error"),
                    LOCALIZE(
                        "HOME_PAGE.project_exists_error", name=project_name.strip()
                    ),
                )

    def handle_open_project(self):
        """Handle project opening with proper file filtering and error handling."""
        try:
            projects_dir = PROJECT_MANAGER.projects_dir
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                LOCALIZE("HOME_PAGE.open_project_dialog_title"),
                projects_dir,
                LOCALIZE("HOME_PAGE.project_file_filter"),
            )
            if file_path and file_path.strip():
                self.projectOpened.emit(file_path)
                create_logs(
                    "HomePage",
                    "open_project",
                    f"Opening project: {file_path}",
                    status="info",
                )
        except Exception as e:
            create_logs(
                "open_project_error",
                "HomePage",
                f"Error in open project dialog: {e}",
                status="error",
            )
            QMessageBox.critical(
                self, LOCALIZE("COMMON.error"), LOCALIZE("HOME_PAGE.open_project_error")
            )

    def handle_settings(self):
        """Handle settings dialog."""
        try:
            dialog = SettingsDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                settings = dialog.get_settings()
                
                # Save settings to config file
                self._save_settings(settings)
                
                # Restart immediately so language changes apply everywhere.
                QMessageBox.information(
                    self,
                    LOCALIZE("HOME_PAGE.settings_dialog_title"),
                    LOCALIZE("HOME_PAGE.settings_saved_message"),
                )

                try:
                    ok = restart_application(reason="settings_changed")
                    if not ok:
                        raise RuntimeError("restart_application returned False")
                except Exception as e:
                    create_logs(
                        "HomePage",
                        "settings",
                        f"Failed to restart application after settings change: {e}",
                        status="error",
                    )
                
                create_logs(
                    "HomePage",
                    "settings",
                    f"Settings updated: {settings}",
                    status="info"
                )
        except Exception as e:
            create_logs(
                "settings_error",
                "HomePage",
                f"Error in settings dialog: {e}",
                status="error"
            )
            QMessageBox.critical(
                self,
                LOCALIZE("COMMON.error"),
                LOCALIZE("HOME_PAGE.settings_error")
            )
    
    def _save_settings(self, settings: dict):
        """Save settings to app config file."""
        try:
            ini_path = save_user_settings(
                language=settings["language"],
                theme=settings["theme"],
            )

            # Keep in-memory config consistent for the remainder of this session.
            # (Some UI strings still require restart to fully refresh.)
            CONFIGS["language"] = settings["language"]
            CONFIGS["theme"] = settings["theme"]

            create_logs(
                "HomePage",
                "settings",
                f"Saved user settings to {ini_path}",
                status="info",
            )
                
        except Exception as e:
            create_logs(
                "HomePage",
                "settings_save_error",
                f"Error saving settings: {e}",
                status="error"
            )

    def handle_recent_item_opened(self, item: QListWidgetItem):
        """Handle recent project item selection with proper error handling."""
        try:
            project_path = item.data(Qt.ItemDataRole.UserRole)
            if project_path and project_path.strip():
                # Verify project file exists before emitting signal
                if os.path.exists(project_path):
                    self.projectOpened.emit(project_path)
                    create_logs(
                        "open_recent",
                        "HomePage",
                        f"Opening recent project: {project_path}",
                        status="info",
                    )
                else:
                    QMessageBox.warning(
                        self,
                        LOCALIZE("COMMON.error"),
                        LOCALIZE(
                            "HOME_PAGE.project_not_found_error", path=project_path
                        ),
                    )
                    # Refresh recent projects list to remove invalid entries
                    self.populate_recent_projects()
        except Exception as e:
            create_logs(
                "open_recent_error",
                "HomePage",
                f"Error opening recent project: {e}",
                status="error",
            )
            QMessageBox.critical(
                self, LOCALIZE("COMMON.error"), LOCALIZE("HOME_PAGE.open_project_error")
            )
