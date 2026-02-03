"""
Analysis Page View Components

This module contains UI view creation functions for the card-based analysis page:
- Startup view with method cards
- Method view with input forms and results
- History sidebar
- Category sections
"""

from typing import Dict, Any, Callable
import os
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QFrame,
    QScrollArea,
    QGridLayout,
    QListWidget,
    QListWidgetItem,
    QListView,
    QGroupBox,
    QTextEdit,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QPixmap

from components.widgets import load_icon
from .registry import ANALYSIS_METHODS


# Method key to image file mapping
METHOD_IMAGES = {
    # Exploratory methods
    "pca": "pca_analysis.png",
    "umap": "umap.png",
    "tsne": "t-sne.png",
    "hierarchical_clustering": "hierarchical_clustering.png",
    "kmeans": "k-means.png",  # Fixed: was "kmeans_clustering"
    "pls_da": "PLS-DA.png",
    "lda": "lda.png",
    "mcr_als": "MCR-ALS.png",
    "nmf": "NMF.png",
    "ica": "ica.png",
    "outlier_detection": "outlier_detection.png",
    # Statistical methods
    "spectral_comparison": "spectral_comparison.png",
    "peak_analysis": "peak_analysis.png",
    "correlation_analysis": "correlation_analysis.png",
    "anova_test": "ANOVA.png",  # Fixed: was "anova"
    "pairwise_tests": "pairwise-statistical.png",
    "band_ratio": "band_ratio.png",
    # Visualization methods
    "heatmap": "spectral_heatmap.png",  # Fixed: was "spectral_heatmap"
    "mean_spectra_overlay": "mean_spectra_overlay.png",
    "waterfall_plot": "waterfall.png",
    "correlation_heatmap": "correlation_heatmap.png",
    "peak_intensity_scatter": "peak_intensity_scatter.png",
    "derivative_spectra": "derivative_spectra.png",
}


def create_startup_view(
    localize_func: Callable, on_method_selected: Callable
) -> QWidget:
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
    startup_widget.setStyleSheet(
        """
        QWidget#startupView {
            background-color: #f8f9fa;
        }
    """
    )

    layout = QVBoxLayout(startup_widget)
    layout.setContentsMargins(20, 8, 20, 16)  # Minimal top margin: 8px
    layout.setSpacing(8)  # Minimal spacing

    # Compact header - single line only
    header_label = QLabel(
        localize_func("ANALYSIS_PAGE.welcome_subtitle")
    )  # Use subtitle as main text
    header_label.setStyleSheet(
        """
        font-size: 14px;
        font-weight: 500;
        color: #495057;
        padding: 4px 0px;
    """
    )
    layout.addWidget(header_label)

    # Scroll area for cards
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setFrameShape(QFrame.NoFrame)
    scroll_area.setStyleSheet(
        """
        QScrollArea {
            background-color: transparent;
            border: none;
        }
    """
    )

    cards_container = QWidget()
    cards_container.setStyleSheet("background-color: transparent;")
    cards_layout = QVBoxLayout(cards_container)
    cards_layout.setSpacing(16)  # Compact section spacing
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
    category_key: str, localize_func: Callable, on_method_selected: Callable
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
    section.setStyleSheet(
        """
        QFrame#categorySection {
            background-color: transparent;
            border: none;
        }
    """
    )

    layout = QVBoxLayout(section)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)  # Minimal spacing between header and grid

    # Category header with icon
    category_icons = {"exploratory": "🔍", "statistical": "📊", "visualization": "📈"}

    icon = category_icons.get(category_key, "📊")
    # Use localization for category names
    category_name = localize_func(f"ANALYSIS_PAGE.CATEGORIES.{category_key}")

    header_label = QLabel(f"{icon} {category_name}")
    header_label.setStyleSheet(
        """
        font-size: 15px;
        font-weight: 600;
        color: #2c3e50;
        padding: 4px 0px;
    """
    )
    layout.addWidget(header_label)

    # Grid layout for method cards (responsive 3-column)
    grid_widget = QWidget()
    grid_layout = QGridLayout(grid_widget)
    grid_layout.setSpacing(12)  # Reduced from 16
    grid_layout.setContentsMargins(0, 0, 0, 0)

    # Get methods for this category
    methods = ANALYSIS_METHODS.get(category_key, {})

    # Add method cards to grid (3 columns)
    row = 0
    col = 0
    for method_key, method_info in methods.items():
        card = create_method_card(
            category_key, method_key, method_info, localize_func, on_method_selected
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
    on_method_selected: Callable,
) -> QFrame:
    """
    Create modern method card with hover effects.

    Styling matches technical guide specifications:
    - Background: #ffffff
    - Border: 1px solid #e0e0e0
    - Border Radius: 8px
    - Hover: Border #0078d4 + shadow

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
    card.setStyleSheet(
        """
        QFrame#methodCard {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 12px;
        }
        QFrame#methodCard:hover {
            border-color: #0078d4;
            background-color: #f8fcff;
        }
    """
    )
    card.setMinimumWidth(260)  # Slightly smaller
    card.setMaximumWidth(380)  # Slightly smaller
    card.setMinimumHeight(200)  # Increased to fit image
    card.setCursor(Qt.PointingHandCursor)

    layout = QVBoxLayout(card)
    layout.setSpacing(8)  # Tighter spacing
    layout.setContentsMargins(0, 0, 0, 0)

    # Add method image if available
    image_filename = METHOD_IMAGES.get(method_key, "")
    if image_filename:
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)

        # Get image path
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        image_path = os.path.join(base_dir, "assets", "image", image_filename)

        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # Scale to fit card width while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    240,
                    120,  # Max width 240px, height 120px
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                image_label.setPixmap(scaled_pixmap)
                image_label.setStyleSheet(
                    """
                    QLabel {
                        background-color: #f8f9fa;
                        border-radius: 4px;
                        padding: 4px;
                    }
                """
                )
                layout.addWidget(image_label)

    # Method name - use localization
    method_name = localize_func(f"ANALYSIS_PAGE.METHODS.{method_key}")
    name_label = QLabel(method_name)
    name_label.setStyleSheet(
        """
        font-size: 14px;
        font-weight: 600;
        color: #2c3e50;
    """
    )
    name_label.setWordWrap(True)
    layout.addWidget(name_label)

    # Description - use localization
    desc_text = localize_func(f"ANALYSIS_PAGE.METHOD_DESC.{method_key}")
    desc_label = QLabel(desc_text)
    desc_label.setStyleSheet(
        """
        font-size: 11px;
        color: #6c757d;
        line-height: 1.3;
    """
    )
    desc_label.setWordWrap(True)
    desc_label.setMinimumHeight(40)  # Reduced from 50
    layout.addWidget(desc_label)

    layout.addStretch()

    # Start button
    start_btn = QPushButton(localize_func("ANALYSIS_PAGE.start_analysis_button"))
    start_btn.setObjectName("cardStartButton")
    start_btn.setMinimumHeight(32)  # Reduced from 36
    start_btn.setStyleSheet(
        """
        QPushButton#cardStartButton {
            background-color: #0078d4;
            color: white;
            border: none;
            border-radius: 4px;
            font-weight: 600;
            font-size: 12px;
            padding: 6px 12px;
        }
        QPushButton#cardStartButton:hover {
            background-color: #006abc;
        }
        QPushButton#cardStartButton:pressed {
            background-color: #005a9e;
        }
    """
    )
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
    sidebar.setStyleSheet(
        """
        QWidget#historySidebar {
            background-color: #f8f9fa;
            border-right: 1px solid #e0e0e0;
        }
    """
    )
    # Expanded width range (used by horizontal collapse)
    sidebar._expanded_min_width = 200
    # Allow user resizing via splitter; keep a sensible upper bound.
    sidebar._expanded_max_width = 520
    sidebar._collapsed_width = 44

    sidebar.setMaximumWidth(sidebar._expanded_max_width)
    sidebar.setMinimumWidth(sidebar._expanded_min_width)
    # IMPORTANT: do not use Fixed, otherwise splitter dragging won't resize the sidebar.
    sidebar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

    layout = QVBoxLayout(sidebar)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(12)

    # Header row (title + collapse toggle)
    header_row = QHBoxLayout()
    header_row.setContentsMargins(0, 0, 0, 0)
    header_row.setSpacing(8)

    header_label = QLabel("📜 " + localize_func("ANALYSIS_PAGE.history_title"))
    header_label.setStyleSheet(
        """
        font-size: 14px;
        font-weight: 600;
        color: #2c3e50;
        padding: 8px 0;
    """
    )
    header_row.addWidget(header_label)
    header_row.addStretch(1)

    toggle_btn = QToolButton()
    toggle_btn.setObjectName("historyCollapseButton")
    toggle_btn.setCursor(Qt.PointingHandCursor)
    toggle_btn.setText("▾")
    toggle_btn.setToolTip(
        localize_func("ANALYSIS_PAGE.history_collapse_tooltip")
        if callable(localize_func)
        else "Collapse"
    )
    toggle_btn.setStyleSheet(
        """
        QToolButton#historyCollapseButton {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 2px 8px;
            background: white;
            color: #2c3e50;
            font-weight: 700;
        }
        QToolButton#historyCollapseButton:hover {
            background: #e7f3ff;
            border-color: #0078d4;
        }
        """
    )
    header_row.addWidget(toggle_btn)
    layout.addLayout(header_row)

    # Collapsible content container (hidden when sidebar is horizontally collapsed)
    content = QWidget()
    content_layout = QVBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(12)

    # History list
    history_list = QListWidget()
    history_list.setObjectName("historyList")
    history_list.setStyleSheet(
        """
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
    """
    )
    # Improve dynamic item widget relayout (important when collapsing/expanding the sidebar)
    try:
        history_list.setUniformItemSizes(False)
    except Exception:
        pass
    try:
        history_list.setResizeMode(QListView.Adjust)
    except Exception:
        pass
    content_layout.addWidget(history_list)

    # Clear history button
    clear_btn = QPushButton(localize_func("ANALYSIS_PAGE.clear_history"))
    clear_btn.setObjectName("secondaryButton")
    clear_btn.setMinimumHeight(32)
    content_layout.addWidget(clear_btn)

    layout.addWidget(content)

    # Toggle behavior
    sidebar._history_collapsed = False

    def _apply_width(w: int) -> None:
        # For splitter sidebars, using fixed min/max is more reliable than setFixedWidth.
        sidebar.setMinimumWidth(int(w))
        sidebar.setMaximumWidth(int(w))
        sidebar.updateGeometry()

    def _set_collapsed(collapsed: bool) -> None:
        # Horizontal (right-to-left) collapse to free space for the main view.
        sidebar._history_collapsed = bool(collapsed)

        if sidebar._history_collapsed:
            content.setVisible(False)
            header_label.setVisible(False)
            toggle_btn.setText("▸")
            try:
                toggle_btn.setToolTip(localize_func("ANALYSIS_PAGE.history_expand_tooltip"))
            except Exception:
                toggle_btn.setToolTip("Expand")
            # Tighten margins for collapsed strip
            layout.setContentsMargins(6, 8, 6, 8)
            layout.setSpacing(8)
            _apply_width(getattr(sidebar, "_collapsed_width", 44))
        else:
            content.setVisible(True)
            header_label.setVisible(True)
            toggle_btn.setText("◂")
            try:
                toggle_btn.setToolTip(localize_func("ANALYSIS_PAGE.history_collapse_tooltip"))
            except Exception:
                toggle_btn.setToolTip("Collapse")
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(12)
            sidebar.setMinimumWidth(getattr(sidebar, "_expanded_min_width", 200))
            sidebar.setMaximumWidth(getattr(sidebar, "_expanded_max_width", 520))
            sidebar.updateGeometry()

            # Force the list to re-layout embedded item widgets after a collapse/expand cycle.
            # Without this, some child widgets (e.g., per-item delete buttons) can appear clipped.
            try:
                from PySide6.QtCore import QTimer

                def _refresh_list() -> None:
                    try:
                        history_list.doItemsLayout()
                    except Exception:
                        pass
                    # Recompute item size hints based on the current width (word-wrapped labels
                    # change height, and stale size hints can clip child widgets).
                    try:
                        vw = int(history_list.viewport().width())
                        for i in range(history_list.count()):
                            it = history_list.item(i)
                            w = history_list.itemWidget(it)
                            if w is not None:
                                try:
                                    if vw > 0:
                                        w.setFixedWidth(max(140, vw - 4))
                                except Exception:
                                    pass
                                it.setSizeHint(w.sizeHint())
                    except Exception:
                        pass
                    try:
                        history_list.updateGeometry()
                    except Exception:
                        pass
                    try:
                        history_list.viewport().update()
                    except Exception:
                        pass

                QTimer.singleShot(0, _refresh_list)
                QTimer.singleShot(50, _refresh_list)
            except Exception:
                pass

    # When the sidebar is resized (e.g., splitter drag), recompute list item size hints
    # so word-wrapped labels don't get clipped.
    try:
        from PySide6.QtCore import QTimer

        _orig_resize_event = sidebar.resizeEvent

        def _resize_event(ev):
            try:
                _orig_resize_event(ev)
            except Exception:
                pass
            try:
                # Defer to allow layout to settle.
                def _refresh() -> None:
                    try:
                        history_list.doItemsLayout()
                    except Exception:
                        pass
                    try:
                        vw = int(history_list.viewport().width())
                        for i in range(history_list.count()):
                            it = history_list.item(i)
                            w = history_list.itemWidget(it)
                            if w is not None and vw > 0:
                                try:
                                    w.setFixedWidth(max(140, vw - 4))
                                except Exception:
                                    pass
                                try:
                                    it.setSizeHint(w.sizeHint())
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    try:
                        history_list.updateGeometry()
                    except Exception:
                        pass
                    try:
                        history_list.viewport().update()
                    except Exception:
                        pass

                QTimer.singleShot(0, _refresh)
                QTimer.singleShot(40, _refresh)
            except Exception:
                pass

        sidebar.resizeEvent = _resize_event
    except Exception:
        pass

    def _toggle() -> None:
        _set_collapsed(not bool(getattr(sidebar, "_history_collapsed", False)))

    toggle_btn.clicked.connect(_toggle)

    # Store reference for external access
    sidebar.history_list = history_list
    sidebar.clear_btn = clear_btn
    sidebar.toggle_btn = toggle_btn
    sidebar.set_history_collapsed = _set_collapsed

    # Default to expanded state, with a "collapse left" affordance.
    try:
        toggle_btn.setText("◂")
    except Exception:
        pass

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
    top_bar.setStyleSheet(
        """
        QWidget#topBar {
            background-color: #ffffff;
            border-bottom: 1px solid #dee2e6;
        }
    """
    )

    layout = QHBoxLayout(top_bar)
    layout.setContentsMargins(12, 4, 12, 4)  # Reduced from 6 to 4 - very tight
    layout.setSpacing(2)  # Reduced from 10 to 8

    # Title with back button
    back_btn = QPushButton("←")
    back_btn.setObjectName("backButton")
    back_btn.setFixedSize(26, 26)  # Reduced from 28x28
    back_btn.setStyleSheet(
        """
        QPushButton#backButton {
            background-color: transparent;
            border: 1px solid #e0e0e0;
            border-radius: 13px;
            font-size: 14px;
            color: #2c3e50;
        }
        QPushButton#backButton:hover {
            background-color: #e7f3ff;
            border-color: #0078d4;
        }
    """
    )
    back_btn.setCursor(Qt.PointingHandCursor)
    back_btn.clicked.connect(on_new_analysis)
    back_btn.setVisible(False)  # Hidden by default
    layout.addWidget(back_btn)

    title_label = QLabel("📊 " + localize_func("ANALYSIS_PAGE.title"))
    title_label.setStyleSheet(
        "font-weight: 600; font-size: 13px; color: #2c3e50;"
    )  # Reduced from 14px
    layout.addWidget(title_label)

    layout.addStretch()

    # New Analysis button (plus icon)
    new_analysis_btn = QPushButton()
    new_analysis_btn.setObjectName("newAnalysisButton")
    plus_icon = load_icon("plus", QSize(16, 16), "white")  # Reduced from 18x18
    new_analysis_btn.setIcon(plus_icon)
    new_analysis_btn.setIconSize(QSize(16, 16))
    new_analysis_btn.setFixedSize(32, 32)  # Reduced from 36x36
    new_analysis_btn.setToolTip(localize_func("ANALYSIS_PAGE.new_analysis_tooltip"))
    new_analysis_btn.setCursor(Qt.PointingHandCursor)
    new_analysis_btn.setStyleSheet(
        """
        QPushButton#newAnalysisButton {
            background-color: #0078d4;
            border: none;
            border-radius: 16px;
        }
        QPushButton#newAnalysisButton:hover {
            background-color: #006abc;
        }
        QPushButton#newAnalysisButton:pressed {
            background-color: #005a9e;
        }
    """
    )
    new_analysis_btn.clicked.connect(on_new_analysis)
    new_analysis_btn.setVisible(False)  # Hidden in startup view
    layout.addWidget(new_analysis_btn)

    # Store references for external access
    top_bar.new_analysis_btn = new_analysis_btn
    top_bar.back_btn = back_btn
    top_bar.title_label = title_label

    return top_bar
