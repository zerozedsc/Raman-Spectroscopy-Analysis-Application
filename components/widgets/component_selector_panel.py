"""Reusable component selector + plot panel.

This widget provides a PCA-like collapsible "Show components" sidebar with:
- Select-all (action label)
- FIFO enforcement when selecting more than `max_selected`
- A scrollable checkbox list
- A MatplotlibWidget plot area updated via a callback

Intended to be reused across analysis methods (PCA/PLS-DA/MCR/NMF/ICA/etc.).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .matplotlib_widget import MatplotlibWidget


PANEL_TOGGLE_BUTTON_QSS = """
QPushButton {
    padding: 6px 12px;
    background-color: #0078d4;
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 11px;
    font-weight: bold;
}
QPushButton:hover { background-color: #005a9e; }
QPushButton:checked { background-color: #005a9e; }
"""

SELECT_ALL_CB_QSS = """
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #adb5bd;
    border-radius: 4px;
    background-color: white;
}
QCheckBox::indicator:hover {
    border-color: #0078d4;
}
QCheckBox::indicator:checked {
    background-color: #0078d4;
    border-color: #0078d4;
    image: url(assets/icons/checkmark_white.svg);
}
QCheckBox {
    spacing: 6px;
    font-size: 11px;
    font-weight: bold;
    color: #0078d4;
    border: none;
    padding: 4px;
}
QCheckBox:checked {
    background-color: #e3f2fd;
    border-radius: 3px;
}
QCheckBox:disabled {
    color: #999999;
}
"""

ITEM_CB_QSS = """
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #adb5bd;
    border-radius: 4px;
    background-color: white;
}
QCheckBox::indicator:hover {
    border-color: #0078d4;
}
QCheckBox::indicator:checked {
    background-color: #0078d4;
    border-color: #0078d4;
    image: url(assets/icons/checkmark_white.svg);
}
QCheckBox {
    spacing: 6px;
    font-size: 10px;
    color: #495057;
    border: none;
    padding: 4px;
}
QCheckBox:checked {
    background-color: #e3f2fd;
    border-radius: 3px;
}
"""


class ComponentSelectorPlotPanel(QWidget):
    """A collapsible component selector paired with a Matplotlib plot area."""

    def __init__(
        self,
        *,
        toggle_text: str,
        title_text: str,
        info_text: str,
        item_labels: Sequence[str],
        max_selected: int = 6,
        initial_checked: Optional[Sequence[int]] = None,
        select_all_label_func: Optional[Callable[[bool], str]] = None,
        on_update: Optional[Callable[[list[int]], None]] = None,
        parent: Optional[QWidget] = None,
        default_sidebar_visible: bool = False,
        plot_min_height: int = 400,
        plot_widget_factory: Optional[Callable[[], QWidget]] = None,
    ):
        super().__init__(parent)

        self._item_labels = list(item_labels)
        self._max_selected = int(max_selected)
        self._on_update = on_update
        self._select_all_label_func = select_all_label_func

        if initial_checked is None:
            initial_checked = [0] if self._item_labels else []
        self._initial_checked = list(initial_checked)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Toolbar
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        toolbar_layout.setSpacing(8)
        toolbar.setStyleSheet("background-color: #f8f9fa; border-bottom: 1px solid #e0e0e0;")

        self.toggle_btn = QPushButton(toggle_text)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(bool(default_sidebar_visible))
        self.toggle_btn.setStyleSheet(PANEL_TOGGLE_BUTTON_QSS)
        toolbar_layout.addWidget(self.toggle_btn)
        toolbar_layout.addStretch()
        main_layout.addWidget(toolbar)

        # Content
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Sidebar
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(220)
        self.sidebar.setVisible(bool(default_sidebar_visible))
        self.sidebar.setStyleSheet(
            """
            QFrame {
                background-color: #f8f9fa;
                border-right: 1px solid #e0e0e0;
            }
            """
        )
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(10)

        title = QLabel(title_text)
        title.setStyleSheet("font-weight: bold; font-size: 12px; color: #2c3e50; border: none;")
        sidebar_layout.addWidget(title)

        info = QLabel(info_text)
        info.setStyleSheet("font-size: 10px; color: #6c757d; border: none;")
        info.setWordWrap(True)
        sidebar_layout.addWidget(info)

        self._select_state = {"all_selected": False}

        def _default_select_all_label(all_selected: bool) -> str:
            return "Deselect All" if all_selected else "Select All"

        self._select_all_label = select_all_label_func or _default_select_all_label

        self.select_all_cb = QCheckBox(self._select_all_label(False))
        n_items = len(self._item_labels)
        self.select_all_cb.setEnabled(n_items <= self._max_selected)
        if n_items > self._max_selected:
            self.select_all_cb.setToolTip(
                f"Cannot select all: {n_items} components available, max {self._max_selected} plots at once"
            )
        self.select_all_cb.setStyleSheet(SELECT_ALL_CB_QSS)
        sidebar_layout.addWidget(self.select_all_cb)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            """
            QScrollArea {
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
            }
            """
        )
        scroll.setMaximumHeight(360)

        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(8, 8, 8, 8)
        list_layout.setSpacing(6)

        self.checkboxes: list[QCheckBox] = []
        self.selection_order: list[QCheckBox] = []

        for i, label in enumerate(self._item_labels):
            cb = QCheckBox(str(label))
            cb.item_index = i
            cb.setStyleSheet(ITEM_CB_QSS)
            cb.setChecked(i in self._initial_checked)
            self.checkboxes.append(cb)
            list_layout.addWidget(cb)

        list_layout.addStretch()
        scroll.setWidget(list_widget)
        sidebar_layout.addWidget(scroll)
        sidebar_layout.addStretch()

        content_layout.addWidget(self.sidebar)

        # Plot area
        plot_factory = plot_widget_factory or MatplotlibWidget
        self.plot = plot_factory()
        try:
            self.plot.setMinimumHeight(int(plot_min_height))
        except Exception:
            pass
        content_layout.addWidget(self.plot, 1)

        main_layout.addWidget(content, 1)

        # Wiring
        self.toggle_btn.toggled.connect(self.sidebar.setVisible)

        for cb in self.checkboxes:
            if cb.isChecked():
                self.selection_order.append(cb)

        def _update_select_all_ui() -> None:
            total = len(self.checkboxes)
            checked_count = sum(1 for cb in self.checkboxes if cb.isChecked())
            all_selected = (total > 0 and checked_count == total)
            self._select_state["all_selected"] = all_selected

            self.select_all_cb.blockSignals(True)
            self.select_all_cb.setChecked(all_selected)
            self.select_all_cb.setText(self._select_all_label(all_selected))
            self.select_all_cb.blockSignals(False)

        def _track_selection(cb: QCheckBox):
            def handler():
                if cb.isChecked():
                    if cb not in self.selection_order:
                        self.selection_order.append(cb)
                else:
                    if cb in self.selection_order:
                        self.selection_order.remove(cb)
            return handler

        def _enforce_limits():
            checked_count = sum(1 for cb in self.checkboxes if cb.isChecked())
            if checked_count > self._max_selected:
                while self.selection_order and not self.selection_order[0].isChecked():
                    self.selection_order.pop(0)
                if self.selection_order:
                    first_cb = self.selection_order.pop(0)
                    first_cb.blockSignals(True)
                    first_cb.setChecked(False)
                    first_cb.blockSignals(False)

            checked_count = sum(1 for cb in self.checkboxes if cb.isChecked())
            if checked_count == 0 and self.checkboxes:
                self.checkboxes[0].blockSignals(True)
                self.checkboxes[0].setChecked(True)
                self.checkboxes[0].blockSignals(False)

        def selected_indices() -> list[int]:
            indices = [
                cb.item_index
                for cb in self.selection_order
                if cb is not None and cb.isChecked()
            ]
            for cb in self.checkboxes:
                if cb.isChecked() and cb.item_index not in indices:
                    indices.append(cb.item_index)
            if len(indices) > self._max_selected:
                indices = indices[-self._max_selected :]
            return indices

        def update_plot() -> None:
            _enforce_limits()
            _update_select_all_ui()
            if self._on_update is not None:
                self._on_update(selected_indices())

        def on_select_all_clicked(_checked: bool):
            if self._select_state.get("all_selected"):
                for cb in self.checkboxes:
                    cb.blockSignals(True)
                    cb.setChecked(False)
                    cb.blockSignals(False)
                if self.checkboxes:
                    self.checkboxes[0].setChecked(True)
            else:
                if len(self.checkboxes) <= self._max_selected:
                    for cb in self.checkboxes:
                        cb.setChecked(True)
            update_plot()

        self.select_all_cb.clicked.connect(on_select_all_clicked)

        for cb in self.checkboxes:
            cb.stateChanged.connect(_track_selection(cb))
            cb.stateChanged.connect(update_plot)

        _update_select_all_ui()
        QTimer.singleShot(0, update_plot)

    def update_figure(self, fig) -> None:
        self.plot.update_plot(fig)

    def set_on_update(self, on_update: Callable[[list[int]], None]) -> None:
        self._on_update = on_update

    def get_selected_indices(self) -> list[int]:
        indices = [cb.item_index for cb in self.selection_order if cb is not None and cb.isChecked()]
        for cb in self.checkboxes:
            if cb.isChecked() and cb.item_index not in indices:
                indices.append(cb.item_index)
        if len(indices) > self._max_selected:
            indices = indices[-self._max_selected :]
        return indices
