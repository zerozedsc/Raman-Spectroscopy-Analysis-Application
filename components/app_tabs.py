import sys
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
    QPushButton,
    QButtonGroup,
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon

from utils import LOCALIZE


class AppTabBar(QWidget):
    """
    A custom tab bar component for main application navigation.
    Uses styled QPushButtons to act as tabs.
    """

    # Signal emitted when a tab is changed, sending the index of the tab
    tabChanged = Signal(int)
    # Signal emitted when home tab is clicked
    homeRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("appTabBar")

        # --- Layout and Button Group ---
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)  # Only one button can be checked at a time

        # --- Define Tabs ---
        # Note: Home tab is separate and handled differently
        # Store tab definitions for dynamic text updates
        self.tab_definitions = [
            ("data", "TABS.data"),
            ("preprocessing", "TABS.preprocessing"),
            ("analysis", "TABS.analysis"),
            ("machine_learning", "TABS.machine_learning"),
        ]

        # --- Create Home Button (separate from main tabs) ---
        self.home_button = QPushButton(LOCALIZE("TABS.home"))
        self.home_button.setObjectName("home")
        self.home_button.clicked.connect(self.homeRequested.emit)
        layout.addWidget(self.home_button)

        # Add separator or spacing
        layout.addSpacing(10)

        # --- Create and Add Tab Buttons ---
        for index, (object_name, loc_key) in enumerate(self.tab_definitions):
            button = QPushButton(LOCALIZE(loc_key))
            button.setObjectName(object_name)  # For specific styling if needed
            button.setCheckable(True)
            button.setProperty("loc_key", loc_key)  # Store localization key for updates
            layout.addWidget(button)
            self.button_group.addButton(button, index)

        # Connect the button group's signal to our custom signal
        self.button_group.idClicked.connect(self.tabChanged.emit)

        # Set the first tab as active by default
        if self.button_group.button(0):
            self.button_group.button(0).setChecked(True)

    def update_localized_text(self):
        """Update all button texts with current localization."""
        # Update home button
        self.home_button.setText(LOCALIZE("TABS.home"))

        # Update all tab buttons
        for index in range(len(self.tab_definitions)):
            button = self.button_group.button(index)
            if button:
                loc_key = button.property("loc_key")
                if loc_key:
                    button.setText(LOCALIZE(loc_key))

    def setActiveTab(self, index: int):
        """Programmatically sets the active tab."""
        button = self.button_group.button(index)
        if button:
            button.setChecked(True)
            self.tabChanged.emit(index)

    def clearActiveTab(self):
        """Clear all active tabs (used when on home page)."""
        self.button_group.setExclusive(False)
        for button in self.button_group.buttons():
            button.setChecked(False)
        self.button_group.setExclusive(True)
