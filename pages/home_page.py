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
)
from PySide6.QtCore import Qt, QSize, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QIcon, QPixmap, QPainter, QMouseEvent, QFont
from PySide6.QtSvg import QSvgRenderer
from components.widgets import *
from utils import *


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

        # Enable hover tracking
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._setup_ui(project_name, last_modified)

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

        return sidebar

    def _create_title_section(self) -> QWidget:
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(12)

        title_label = QLabel(LOCALIZE("MAIN_WINDOW.title"))
        title_label.setObjectName("homeTitle")
        title_label.setWordWrap(True)
        title_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        subtitle_label = QLabel(LOCALIZE("HOME_PAGE.subtitle"))
        subtitle_label.setObjectName("homeSubtitle")
        subtitle_label.setWordWrap(True)
        subtitle_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)

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
        self.recent_projects_list.itemDoubleClicked.connect(
            self.handle_recent_item_opened
        )

        layout.addWidget(header_container)
        layout.addWidget(self.recent_projects_list, 1)

        return content_area

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

        header_layout.addWidget(icon_label)
        header_layout.addWidget(header_label)
        header_layout.addStretch()

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
                "HomePage",
                "open_project_error",
                f"Error in open project dialog: {e}",
                status="error",
            )
            QMessageBox.critical(
                self, LOCALIZE("COMMON.error"), LOCALIZE("HOME_PAGE.open_project_error")
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
                        "HomePage",
                        "open_recent",
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
                "HomePage",
                "open_recent_error",
                f"Error opening recent project: {e}",
                status="error",
            )
            QMessageBox.critical(
                self, LOCALIZE("COMMON.error"), LOCALIZE("HOME_PAGE.open_project_error")
            )
