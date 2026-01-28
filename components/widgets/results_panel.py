from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedLayout,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


def apply_modern_tab_style(tab_widget: QTabWidget, object_name: Optional[str] = None) -> None:
    """Apply the app's modern tab styling (including scroll arrow buttons).

    Notes:
        - QTabBar uses internal QToolButtons for scroll arrows when there are too many tabs.
        - Styling these consistently avoids the "random tiny arrows" look across pages.
    """
    if object_name:
        tab_widget.setObjectName(object_name)

    # If an objectName is provided, scope the stylesheet to that widget to reduce side effects.
    # Otherwise, apply a generic QTabWidget style (still fairly safe).
    selector = f"QTabWidget#{object_name}" if object_name else "QTabWidget"

    tab_widget.setDocumentMode(True)

    # Use our own left/right buttons at opposite edges; Qt's built-in scroll buttons
    # appear together on the right, which makes the layout feel cramped.
    try:
        tab_widget.setUsesScrollButtons(False)
    except Exception:
        pass

    # Idempotent: avoid recreating corner widgets on repeated calls.
    if not tab_widget.property("_modern_tabs_has_corner_arrows"):
        left_btn = QToolButton(tab_widget)
        left_btn.setObjectName("modernTabScrollLeft")
        left_btn.setText("◀")
        left_btn.setCursor(Qt.PointingHandCursor)

        right_btn = QToolButton(tab_widget)
        right_btn.setObjectName("modernTabScrollRight")
        right_btn.setText("▶")
        right_btn.setCursor(Qt.PointingHandCursor)

        tab_widget.setCornerWidget(left_btn, Qt.TopLeftCorner)
        tab_widget.setCornerWidget(right_btn, Qt.TopRightCorner)
        tab_widget.setProperty("_modern_tabs_has_corner_arrows", True)

        def _update_btn_state() -> None:
            try:
                count = tab_widget.count()
                cur = tab_widget.currentIndex()
                left_btn.setEnabled(cur > 0)
                right_btn.setEnabled(0 <= cur < count - 1)
            except Exception:
                left_btn.setEnabled(False)
                right_btn.setEnabled(False)

        def _move(delta: int) -> None:
            try:
                cur = tab_widget.currentIndex()
                nxt = max(0, min(tab_widget.count() - 1, cur + delta))
                tab_widget.setCurrentIndex(nxt)
            finally:
                _update_btn_state()

        left_btn.clicked.connect(lambda: _move(-1))
        right_btn.clicked.connect(lambda: _move(+1))
        tab_widget.currentChanged.connect(lambda _idx: _update_btn_state())
        _update_btn_state()
    tab_widget.setStyleSheet(
        f"""
        {selector}::pane {{
            border: none;
            background: #ffffff;
            padding-top: 10px;
        }}

        {selector}::tab-bar {{
            alignment: left;
        }}

        {selector} QTabBar {{
            background: #ffffff;
            border-bottom: 1px solid #dee2e6;
        }}

        {selector} QTabBar::tab {{
            background: transparent;
            color: #6c757d;
            font-size: 13px;
            font-weight: 600;
            padding: 12px 20px;
            border-bottom: 2px solid transparent;
            margin-left: 8px;
        }}

        {selector} QTabBar::tab:hover {{
            color: #0078d4;
            background-color: #f8f9fa;
            border-radius: 4px 4px 0 0;
        }}

        {selector} QTabBar::tab:selected {{
            color: #0078d4;
            border-bottom: 2px solid #0078d4;
        }}

        /* Corner scroll arrows (custom) */
        {selector} QToolButton#modernTabScrollLeft,
        {selector} QToolButton#modernTabScrollRight {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin: 6px;
            padding: 2px 6px;
            min-width: 26px;
            min-height: 26px;
            color: #2c3e50;
            font-weight: 700;
        }}

        {selector} QToolButton#modernTabScrollLeft:hover,
        {selector} QToolButton#modernTabScrollRight:hover {{
            background-color: #e3f2fd;
            border-color: #0078d4;
        }}

        {selector} QToolButton#modernTabScrollLeft:disabled,
        {selector} QToolButton#modernTabScrollRight:disabled {{
            color: #adb5bd;
            background-color: #f8f9fa;
            border-color: #e9ecef;
        }}
        """
    )


class ResultsPanel(QWidget):
    """Reusable results panel with a header and tabbed content.

    This is shared between pages to keep consistent styling and behavior.

    API compatibility notes:
        - `tab_widget` attribute exists (used by Analysis populate code).
        - `export_btn` is provided; `export_data_btn` property returns it.
    """

    def __init__(self, localize_func: Callable[[str], str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.localize = localize_func
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setObjectName("resultsPanel")

        # 1. Stacked Layout (main UI + optional overlay)
        stacked = QStackedLayout(self)
        stacked.setStackingMode(QStackedLayout.StackAll)

        main = QWidget(self)
        layout = QVBoxLayout(main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 2. Header Section (Title + Export)
        header_frame = QFrame()
        header_frame.setStyleSheet(
            """
            QFrame {
                background-color: #ffffff;
                border-bottom: 1px solid #e0e0e0;
            }
            """
        )
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(24, 16, 24, 16)
        header_layout.setSpacing(16)

        self.title_label = QLabel("📊 " + self.localize("ANALYSIS_PAGE.results_title"))
        self.title_label.setStyleSheet(
            """
            font-size: 16px;
            font-weight: 700;
            color: #2c3e50;
            """
        )

        # Phase 3: user requested the Analysis export button be renamed from "Export CSV" to "Export"
        try:
            self.export_btn = QPushButton(self.localize("ANALYSIS_PAGE.export"))
        except Exception:
            self.export_btn = QPushButton(self.localize("ANALYSIS_PAGE.export_csv"))
        self.export_btn.setCursor(Qt.PointingHandCursor)
        self.export_btn.setVisible(False)
        self.export_btn.setStyleSheet(
            """
            QPushButton {
                background-color: white;
                color: #333;
                border: 1px solid #d0d0d0;
                padding: 6px 16px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #f8f9fa;
                border-color: #b0b0b0;
                color: #000;
            }
            """
        )

        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.export_btn)

        layout.addWidget(header_frame)

        # 3. Tab Widget
        self.tab_widget = QTabWidget()
        apply_modern_tab_style(self.tab_widget, object_name="analysisResultsTabs")
        layout.addWidget(self.tab_widget)

        # 4. Initial Empty State
        self.show_placeholder()

        # Add main to stack
        stacked.addWidget(main)

        # Overlay
        overlay = QWidget(self)
        overlay.setObjectName("resultsPanelLoadingOverlay")
        overlay.setStyleSheet(
            """
            QWidget#resultsPanelLoadingOverlay {
                background-color: rgba(255, 255, 255, 0.85);
            }
            """
        )
        o_layout = QVBoxLayout(overlay)
        o_layout.setContentsMargins(0, 0, 0, 0)
        o_layout.setAlignment(Qt.AlignCenter)

        self._loading_label = QLabel("⏳ Loading...")
        self._loading_label.setAlignment(Qt.AlignCenter)
        self._loading_label.setStyleSheet(
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
        o_layout.addWidget(self._loading_label)

        stacked.addWidget(overlay)
        overlay.setVisible(False)

        # Expose overlay for callers that previously used the legacy results panel.
        self.loading_overlay = overlay
        self._stacked_layout = stacked

    def show_loading(self, text: str | None = None) -> None:
        """Show a lightweight overlay while results are being computed/restored."""

        try:
            if text:
                self._loading_label.setText(str(text))
        except Exception:
            pass
        try:
            self.loading_overlay.setVisible(True)
            self.loading_overlay.raise_()
        except Exception:
            pass

    def hide_loading(self) -> None:
        """Hide the loading overlay."""

        try:
            self.loading_overlay.setVisible(False)
        except Exception:
            pass

    @property
    def export_data_btn(self) -> QPushButton:
        return self.export_btn

    def show_placeholder(self) -> None:
        """Display the "No Results" state."""
        self.tab_widget.clear()

        placeholder = QWidget()
        p_layout = QVBoxLayout(placeholder)
        p_layout.setAlignment(Qt.AlignCenter)

        icon_label = QLabel("📉")
        icon_label.setStyleSheet("font-size: 48px; margin-bottom: 10px;")

        text_label = QLabel(self.localize("ANALYSIS_PAGE.no_results_yet"))
        text_label.setStyleSheet("font-size: 16px; color: #adb5bd; font-weight: 500;")

        p_layout.addWidget(icon_label, 0, Qt.AlignCenter)
        p_layout.addWidget(text_label, 0, Qt.AlignCenter)

        self.tab_widget.addTab(placeholder, "Info")
        self.export_btn.setVisible(False)

    def add_result_tab(self, widget: QWidget, title: str, icon=None) -> None:
        """Helper to add tabs easily."""
        index = self.tab_widget.addTab(widget, title)
        if icon:
            self.tab_widget.setTabIcon(index, icon)
        self.tab_widget.setCurrentIndex(index)
        self.export_btn.setVisible(True)

    def clear_results(self) -> None:
        """Clear all tabs and show placeholder."""
        self.tab_widget.clear()
        self.show_placeholder()
