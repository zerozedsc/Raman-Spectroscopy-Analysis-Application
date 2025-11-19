"""
Analysis Page View Components

This module contains UI view creation functions for the card-based analysis page:
- Startup view with method cards
- Method view with input forms and results
- History sidebar
- Category sections
"""

from typing import Dict, Any, Callable
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QGridLayout, QListWidget, QListWidgetItem,
    QGroupBox, QTextEdit
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont

from components.widgets import load_icon
from .registry import ANALYSIS_METHODS


def create_startup_view(localize_func: Callable, on_method_selected: Callable) -> QWidget:
    """
    Create startup view with categorized method cards in a cleaner layout.
    
    Args:
        localize_func: Localization function
        on_method_selected: Callback when method card is clicked (category, method_key)
    
    Returns:
        Startup view widget
    """
    startup_widget = QWidget()
    startup_widget.setObjectName("startupView")
    startup_widget.setStyleSheet("""
        QWidget#startupView {
            background-color: #f5f7fa;
        }
    """)
    
    layout = QVBoxLayout(startup_widget)
    layout.setContentsMargins(32, 16, 32, 32)
    layout.setSpacing(20)
    
    # Welcome header with better styling
    header_container = QWidget()
    header_layout = QVBoxLayout(header_container)
    header_layout.setContentsMargins(0, 0, 0, 8)
    header_layout.setSpacing(6)
    
    welcome_label = QLabel(localize_func("ANALYSIS_PAGE.welcome_title"))
    welcome_label.setStyleSheet("""
        font-size: 24px;
        font-weight: 700;
        color: #1a1a1a;
    """)
    header_layout.addWidget(welcome_label)
    
    subtitle_label = QLabel(localize_func("ANALYSIS_PAGE.welcome_subtitle"))
    subtitle_label.setStyleSheet("""
        font-size: 14px;
        color: #666666;
        margin-bottom: 4px;
    """)
    header_layout.addWidget(subtitle_label)
    
    layout.addWidget(header_container)
    
    # Scroll area for cards
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setFrameShape(QFrame.NoFrame)
    scroll_area.setStyleSheet("""
        QScrollArea {
            background-color: transparent;
            border: none;
        }
    """)
    
    cards_container = QWidget()
    cards_container.setStyleSheet("background-color: transparent;")
    cards_layout = QVBoxLayout(cards_container)
    cards_layout.setSpacing(32)
    cards_layout.setContentsMargins(0, 0, 0, 0)
    
    # Create category sections with method cards
    for category_key in ["exploratory", "statistical", "visualization"]:
        category_section = create_category_section(
            category_key, localize_func, on_method_selected
        )
        cards_layout.addWidget(category_section)
    
    cards_layout.addStretch()
    scroll_area.setWidget(cards_container)
    layout.addWidget(scroll_area)
    
    return startup_widget


def create_category_section(
    category_key: str, 
    localize_func: Callable, 
    on_method_selected: Callable
) -> QWidget:
    """
    Create a category section with method cards in a better layout.
    
    Args:
        category_key: Category identifier
        localize_func: Localization function
        on_method_selected: Callback for card clicks
    
    Returns:
        Category section widget
    """
    section = QFrame()
    section.setObjectName("categorySection")
    section.setStyleSheet("""
        QFrame#categorySection {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 0px;
        }
    """)
    
    layout = QVBoxLayout(section)
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(20)
    
    # Category header with icon
    category_icons = {
        "exploratory": "🔍",
        "statistical": "📊", 
        "visualization": "📈"
    }
    
    icon = category_icons.get(category_key, "📊")
    # Use localization for category names
    category_name = localize_func(f"ANALYSIS_PAGE.CATEGORIES.{category_key}")
    
    header_label = QLabel(f"{icon} {category_name}")
    header_label.setStyleSheet("""
        font-size: 20px;
        font-weight: 700;
        color: #2c3e50;
        padding-bottom: 4px;
        border-bottom: 3px solid #0078d4;
    """)
    layout.addWidget(header_label)
    
    # Grid layout for method cards (responsive 3-column)
    grid_widget = QWidget()
    grid_layout = QGridLayout(grid_widget)
    grid_layout.setSpacing(16)
    grid_layout.setContentsMargins(0, 0, 0, 0)
    
    # Get methods for this category
    methods = ANALYSIS_METHODS.get(category_key, {})
    
    # Add method cards to grid (3 columns)
    row = 0
    col = 0
    for method_key, method_info in methods.items():
        card = create_method_card(
            category_key, method_key, method_info,
            localize_func, on_method_selected
        )
        grid_layout.addWidget(card, row, col)
        
        col += 1
        if col >= 3:  # 3 cards per row
            col = 0
            row += 1
    
    layout.addWidget(grid_widget)
    
    return section


def create_method_card(
    category: str,
    method_key: str,
    method_info: Dict,
    localize_func: Callable,
    on_method_selected: Callable
) -> QFrame:
    """
    Create an individual method card (Image 1 reference).
    
    Args:
        category: Method category
        method_key: Method identifier
        method_info: Method configuration
        localize_func: Localization function
        on_method_selected: Callback for card clicks
    
    Returns:
        Method card widget
    """
    card = QFrame()
    card.setObjectName("methodCard")
    card.setStyleSheet("""
        QFrame#methodCard {
            background-color: #fafbfc;
            border: 2px solid #e1e4e8;
            border-radius: 8px;
            padding: 20px;
        }
        QFrame#methodCard:hover {
            border-color: #0078d4;
            background-color: #f0f6ff;
        }
    """)
    card.setMinimumWidth(280)
    card.setMinimumHeight(200)
    card.setCursor(Qt.PointingHandCursor)
    
    layout = QVBoxLayout(card)
    layout.setSpacing(12)
    layout.setContentsMargins(0, 0, 0, 0)
    
    # Method name - use localization
    method_name = localize_func(f"ANALYSIS_PAGE.METHODS.{method_key}")
    name_label = QLabel(method_name)
    name_label.setStyleSheet("""
        font-size: 16px;
        font-weight: 700;
        color: #24292e;
        margin-bottom: 4px;
    """)
    name_label.setWordWrap(True)
    layout.addWidget(name_label)
    
    # Description - use localization
    desc_text = localize_func(f"ANALYSIS_PAGE.METHOD_DESC.{method_key}")
    desc_label = QLabel(desc_text)
    desc_label.setStyleSheet("""
        font-size: 13px;
        color: #586069;
        line-height: 1.5;
    """)
    desc_label.setWordWrap(True)
    desc_label.setMinimumHeight(60)
    layout.addWidget(desc_label)
    
    layout.addStretch()
    
    # Start button
    start_btn = QPushButton(localize_func("ANALYSIS_PAGE.start_analysis_button"))
    start_btn.setObjectName("cardStartButton")
    start_btn.setMinimumHeight(40)
    start_btn.setStyleSheet("""
        QPushButton#cardStartButton {
            background-color: #0078d4;
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            font-size: 14px;
            padding: 10px 20px;
        }
        QPushButton#cardStartButton:hover {
            background-color: #106ebe;
        }
        QPushButton#cardStartButton:pressed {
            background-color: #005a9e;
        }
    """)
    start_btn.clicked.connect(lambda: on_method_selected(category, method_key))
    layout.addWidget(start_btn)
    
    # Make entire card clickable
    card.mousePressEvent = lambda event: on_method_selected(category, method_key)
    
    return card


def create_history_sidebar(localize_func: Callable) -> QWidget:
    """
    Create collapsible history sidebar for analysis session tracking.
    
    Args:
        localize_func: Localization function
    
    Returns:
        History sidebar widget
    """
    sidebar = QWidget()
    sidebar.setObjectName("historySidebar")
    sidebar.setStyleSheet("""
        QWidget#historySidebar {
            background-color: #f8f9fa;
            border-right: 1px solid #e0e0e0;
        }
    """)
    sidebar.setMaximumWidth(280)
    sidebar.setMinimumWidth(200)
    
    layout = QVBoxLayout(sidebar)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(12)
    
    # Header
    header_label = QLabel("📜 " + localize_func("ANALYSIS_PAGE.history_title"))
    header_label.setStyleSheet("""
        font-size: 14px;
        font-weight: 600;
        color: #2c3e50;
        padding: 8px 0;
    """)
    layout.addWidget(header_label)
    
    # History list
    history_list = QListWidget()
    history_list.setObjectName("historyList")
    history_list.setStyleSheet("""
        QListWidget#historyList {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
        }
        QListWidget#historyList::item {
            padding: 8px;
            border-bottom: 1px solid #f0f0f0;
        }
        QListWidget#historyList::item:selected {
            background-color: #e7f3ff;
            color: #2c3e50;
        }
        QListWidget#historyList::item:hover {
            background-color: #f0f0f0;
        }
    """)
    layout.addWidget(history_list)
    
    # Clear history button
    clear_btn = QPushButton(localize_func("ANALYSIS_PAGE.clear_history"))
    clear_btn.setObjectName("secondaryButton")
    clear_btn.setMinimumHeight(32)
    layout.addWidget(clear_btn)
    
    # Store reference for external access
    sidebar.history_list = history_list
    sidebar.clear_btn = clear_btn
    
    return sidebar


def create_top_bar(localize_func: Callable, on_new_analysis: Callable) -> QWidget:
    """
    Create top navigation bar with New Analysis button.
    
    Args:
        localize_func: Localization function
        on_new_analysis: Callback for new analysis button
    
    Returns:
        Top bar widget with new_analysis_btn attribute
    """
    top_bar = QWidget()
    top_bar.setObjectName("topBar")
    top_bar.setStyleSheet("""
        QWidget#topBar {
            background-color: #ffffff;
            border-bottom: 2px solid #e0e0e0;
            padding: 8px 16px;
        }
    """)
    
    layout = QHBoxLayout(top_bar)
    layout.setContentsMargins(12, 8, 12, 8)
    layout.setSpacing(12)
    
    # Title with back button
    back_btn = QPushButton("←")
    back_btn.setObjectName("backButton")
    back_btn.setFixedSize(32, 32)
    back_btn.setStyleSheet("""
        QPushButton#backButton {
            background-color: transparent;
            border: 1px solid #e0e0e0;
            border-radius: 16px;
            font-size: 18px;
            color: #2c3e50;
        }
        QPushButton#backButton:hover {
            background-color: #f0f0f0;
            border-color: #0078d4;
        }
    """)
    back_btn.setCursor(Qt.PointingHandCursor)
    back_btn.clicked.connect(on_new_analysis)
    back_btn.setVisible(False)  # Hidden by default
    layout.addWidget(back_btn)
    
    title_label = QLabel("📊 " + localize_func("ANALYSIS_PAGE.title"))
    title_label.setStyleSheet("font-weight: 600; font-size: 16px; color: #2c3e50;")
    layout.addWidget(title_label)
    
    layout.addStretch()
    
    # New Analysis button (plus icon)
    new_analysis_btn = QPushButton()
    new_analysis_btn.setObjectName("newAnalysisButton")
    plus_icon = load_icon("plus", QSize(20, 20), "white")
    new_analysis_btn.setIcon(plus_icon)
    new_analysis_btn.setIconSize(QSize(20, 20))
    new_analysis_btn.setFixedSize(40, 40)
    new_analysis_btn.setToolTip(localize_func("ANALYSIS_PAGE.new_analysis_tooltip"))
    new_analysis_btn.setCursor(Qt.PointingHandCursor)
    new_analysis_btn.setStyleSheet("""
        QPushButton#newAnalysisButton {
            background-color: #0078d4;
            border: none;
            border-radius: 20px;
        }
        QPushButton#newAnalysisButton:hover {
            background-color: #006abc;
        }
        QPushButton#newAnalysisButton:pressed {
            background-color: #005a9e;
        }
    """)
    new_analysis_btn.clicked.connect(on_new_analysis)
    new_analysis_btn.setVisible(False)  # Hidden in startup view
    layout.addWidget(new_analysis_btn)
    
    # Store references for external access
    top_bar.new_analysis_btn = new_analysis_btn
    top_bar.back_btn = back_btn
    top_bar.title_label = title_label
    
    return top_bar
