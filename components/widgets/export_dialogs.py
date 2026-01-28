from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


_EXPORT_DIALOG_STYLESHEET = """
QDialog {
    background-color: #f8f9fa;
}
QLabel {
    color: #2c3e50;
    font-size: 13px;
}
QLabel#infoLabel {
    color: #0078d4;
    font-weight: 600;
    background-color: #e3f2fd;
    border-left: 4px solid #0078d4;
    padding: 10px;
    border-radius: 4px;
}
QLabel#hintLabel {
    color: #6c757d;
    font-size: 11px;
    font-style: italic;
}
QLineEdit {
    padding: 10px;
    border: 2px solid #ced4da;
    border-radius: 6px;
    font-size: 13px;
    background-color: white;
}
QLineEdit:focus {
    border-color: #0078d4;
    background-color: #f0f8ff;
}
QLineEdit:read-only {
    background-color: #e9ecef;
}
QComboBox {
    padding: 10px;
    border: 2px solid #ced4da;
    border-radius: 6px;
    background-color: white;
    font-size: 13px;
}
QComboBox:focus {
    border-color: #0078d4;
}
QComboBox::drop-down {
    border: none;
    width: 30px;
}
QComboBox::down-arrow {
    image: url(assets/icons/chevron-down.svg);
    width: 12px;
    height: 12px;
}
QPushButton {
    background-color: #0078d4;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    font-weight: 500;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #106ebe;
}
QPushButton:pressed {
    background-color: #005a9e;
}
QCheckBox {
    color: #2c3e50;
    font-size: 13px;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border: 2px solid #ced4da;
    border-radius: 4px;
    background-color: white;
}
QCheckBox::indicator:hover {
    border-color: #0078d4;
}
QCheckBox::indicator:checked {
    background-color: #28a745;
    border-color: #28a745;
    image: url(assets/icons/checkmark.svg);
}
QDialogButtonBox QPushButton {
    min-width: 80px;
    padding: 10px 16px;
}
"""


_WINDOWS_RESERVED_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


@dataclass(frozen=True)
class ExportOptions:
    directory: str
    filename: str
    format_key: str
    export_metadata: bool = False


@dataclass(frozen=True)
class ExportBundleOptions:
    """Options for exporting a bundle of artifacts into multiple files."""

    directory: str
    base_name: str
    image_format: str
    export_report_csv: bool = True
    export_report_json: bool = False
    export_confusion_matrix: bool = True
    export_pca_boundary: bool = True
    export_roc_curve: bool = True
    export_prediction_distribution: bool = False
    export_feature_importance: bool = False


@dataclass(frozen=True)
class ExportShapBundleOptions:
    """Options for exporting SHAP explanation artifacts into multiple files."""

    directory: str
    base_name: str
    image_format: str
    export_spectrum_plot: bool = True
    export_shap_plot: bool = True
    export_contributors_csv: bool = True
    export_raw_json: bool = True
    export_metadata_json: bool = False


@dataclass(frozen=True)
class ExportAnalysisBundleOptions:
    """Options for exporting Analysis artifacts into multiple files."""

    directory: str
    base_name: str
    image_format: str
    export_data_csv: bool = True
    export_data_json: bool = False
    export_primary_plot: bool = True
    export_secondary_plot: bool = False


def _safe_str(x: object) -> str:
    return str(x or "").strip()


def _sanitize_filename_component(value: object, *, fallback: str = "export") -> str:
    """Sanitize user-provided filename/base-name components.

    Security/portability goals:
    - Prevent directory traversal (e.g. "../../foo")
    - Avoid Windows reserved basenames (CON, PRN, ...)
    - Keep filenames cross-platform (portable builds)

    Note: Callers typically add extensions themselves.
    """

    raw = _safe_str(value)
    # Drop any path segments (defense-in-depth).
    raw = os.path.basename(raw)

    # Allow dot for extensions (callers can decide whether to keep/append).
    cleaned = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".", " ")) else "_" for ch in raw)
    cleaned = cleaned.strip(" ._")
    if not cleaned:
        cleaned = fallback

    if cleaned.upper() in _WINDOWS_RESERVED_BASENAMES:
        cleaned = f"_{cleaned}"
    return cleaned


def get_export_options(
    parent: QWidget,
    *,
    title: str,
    formats: Iterable[tuple[str, str]],
    default_directory: str | None = None,
    default_filename: str | None = None,
    show_filename: bool = True,
    show_format: bool = True,
    multiple_info_text: str | None = None,
    multiple_names_hint: str | None = None,
    # Labels (callers should pass localized strings)
    format_label: str = "Export Format:",
    location_label: str = "Save Location:",
    filename_label: str = "Filename:",
    browse_button_text: str = "Browse...",
    select_location_title: str = "Select Location",
    # Optional metadata
    show_metadata_checkbox: bool = False,
    metadata_checkbox_text: str = "Export metadata (.json)",
    metadata_tooltip: str = "Export metadata alongside the exported file",
    default_export_metadata: bool = True,
) -> Optional[ExportOptions]:
    """Show an export options dialog and return user choices.

    This is the shared export UX used across pages.
    """

    dialog = QDialog(parent)
    dialog.setWindowTitle(_safe_str(title) or "Export")
    dialog.setMinimumWidth(550)
    dialog.setStyleSheet(_EXPORT_DIALOG_STYLESHEET)

    layout = QVBoxLayout(dialog)
    layout.setSpacing(16)
    layout.setContentsMargins(24, 24, 24, 24)

    if multiple_info_text:
        info_label = QLabel(_safe_str(multiple_info_text))
        info_label.setObjectName("infoLabel")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

    # Format selection
    format_combo: Optional[QComboBox] = None
    if show_format:
        row = QHBoxLayout()
        row.addWidget(QLabel(_safe_str(format_label)))
        format_combo = QComboBox()
        for fmt_key, fmt_label in formats:
            format_combo.addItem(_safe_str(fmt_label), _safe_str(fmt_key))
        row.addWidget(format_combo)
        layout.addLayout(row)

    # Directory selection
    location_layout = QHBoxLayout()
    location_layout.addWidget(QLabel(_safe_str(location_label)))

    location_edit = QLineEdit(_safe_str(default_directory) or "")
    location_edit.setReadOnly(True)

    browse_btn = QLabel()  # placeholder type to satisfy type checker
    from PySide6.QtWidgets import QPushButton

    browse_btn = QPushButton(_safe_str(browse_button_text))

    def _browse_location() -> None:
        start_path = location_edit.text() if location_edit.text() else os.getcwd()
        picked = QFileDialog.getExistingDirectory(dialog, _safe_str(select_location_title), start_path)
        if picked:
            location_edit.setText(str(picked))

    browse_btn.clicked.connect(_browse_location)

    location_layout.addWidget(location_edit, 1)
    location_layout.addWidget(browse_btn)
    layout.addLayout(location_layout)

    # Filename
    filename_edit: Optional[QLineEdit] = None
    if show_filename:
        row = QHBoxLayout()
        row.addWidget(QLabel(_safe_str(filename_label)))
        filename_edit = QLineEdit(_safe_str(default_filename) or "")
        row.addWidget(filename_edit)
        layout.addLayout(row)
    else:
        if multiple_names_hint:
            hint = QLabel(_safe_str(multiple_names_hint))
            hint.setObjectName("hintLabel")
            hint.setWordWrap(True)
            layout.addWidget(hint)

    # Optional metadata export
    metadata_checkbox: Optional[QCheckBox] = None
    if show_metadata_checkbox:
        metadata_checkbox = QCheckBox(_safe_str(metadata_checkbox_text))
        metadata_checkbox.setChecked(bool(default_export_metadata))
        metadata_checkbox.setToolTip(_safe_str(metadata_tooltip))
        layout.addWidget(metadata_checkbox)

    # Dialog buttons
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    if dialog.exec() != QDialog.Accepted:
        return None

    directory = _safe_str(location_edit.text())
    if not directory:
        return None

    filename = _safe_str(filename_edit.text()) if filename_edit is not None else ""
    if filename:
        filename = _sanitize_filename_component(filename, fallback="export")
    fmt_key = ""
    if format_combo is not None:
        fmt_key = _safe_str(format_combo.currentData())

    export_metadata = bool(metadata_checkbox.isChecked()) if metadata_checkbox is not None else False

    return ExportOptions(
        directory=directory,
        filename=filename,
        format_key=fmt_key,
        export_metadata=export_metadata,
    )


def get_export_bundle_options(
    parent: QWidget,
    *,
    title: str,
    default_directory: str | None = None,
    default_base_name: str | None = None,
    default_image_format: str = "png",
    # Which artifacts are offered
    show_report: bool = True,
    show_confusion_matrix: bool = True,
    show_pca_boundary: bool = True,
    show_roc_curve: bool = True,
    show_prediction_distribution: bool = True,
    show_feature_importance: bool = True,
    # Default selection
    default_export_report_csv: bool = True,
    default_export_report_json: bool = False,
    default_export_confusion_matrix: bool = True,
    default_export_pca_boundary: bool = True,
    default_export_roc_curve: bool = True,
    default_export_prediction_distribution: bool = False,
    default_export_feature_importance: bool = False,
    # Labels (callers should pass localized strings)
    location_label: str = "Save Location:",
    base_name_label: str = "Base Name:",
    browse_button_text: str = "Browse...",
    select_location_title: str = "Select Location",
    image_format_label: str = "Image Format:",
    section_label: str = "Include:",
    report_csv_label: str = "Classification report (CSV)",
    report_json_label: str = "Classification report (JSON)",
    confusion_label: str = "Confusion matrix (image)",
    pca_boundary_label: str = "PCA / decision boundary (image)",
    roc_label: str = "ROC curve (image)",
    pred_dist_label: str = "Prediction distribution (image)",
    feat_imp_label: str = "Feature importance (image)",
) -> Optional[ExportBundleOptions]:
    """Shared export dialog for exporting multiple ML result artifacts.

    Returns None if the user cancels.
    """

    dialog = QDialog(parent)
    dialog.setWindowTitle(_safe_str(title) or "Export")
    dialog.setMinimumWidth(560)
    dialog.setStyleSheet(_EXPORT_DIALOG_STYLESHEET)

    layout = QVBoxLayout(dialog)
    layout.setSpacing(16)
    layout.setContentsMargins(24, 24, 24, 24)

    # Directory selection
    location_layout = QHBoxLayout()
    location_layout.addWidget(QLabel(_safe_str(location_label)))

    location_edit = QLineEdit(_safe_str(default_directory) or "")
    location_edit.setReadOnly(True)

    from PySide6.QtWidgets import QPushButton

    browse_btn = QPushButton(_safe_str(browse_button_text))

    def _browse_location() -> None:
        start_path = location_edit.text() if location_edit.text() else os.getcwd()
        picked = QFileDialog.getExistingDirectory(dialog, _safe_str(select_location_title), start_path)
        if picked:
            location_edit.setText(str(picked))

    browse_btn.clicked.connect(_browse_location)

    location_layout.addWidget(location_edit, 1)
    location_layout.addWidget(browse_btn)
    layout.addLayout(location_layout)

    # Base name
    base_row = QHBoxLayout()
    base_row.addWidget(QLabel(_safe_str(base_name_label)))
    base_edit = QLineEdit(_safe_str(default_base_name) or "")
    base_row.addWidget(base_edit)
    layout.addLayout(base_row)

    # Image format
    fmt_row = QHBoxLayout()
    fmt_row.addWidget(QLabel(_safe_str(image_format_label)))
    img_fmt_combo = QComboBox()
    img_fmt_combo.addItem("PNG (.png)", "png")
    img_fmt_combo.addItem("SVG (.svg)", "svg")
    try:
        idx = img_fmt_combo.findData(_safe_str(default_image_format).lower())
        if idx >= 0:
            img_fmt_combo.setCurrentIndex(idx)
    except Exception:
        pass
    fmt_row.addWidget(img_fmt_combo)
    layout.addLayout(fmt_row)

    # Section label
    section = QLabel(_safe_str(section_label))
    section.setStyleSheet("font-weight: 700;")
    layout.addWidget(section)

    # Checkboxes
    cb_report_csv = QCheckBox(_safe_str(report_csv_label))
    cb_report_csv.setChecked(bool(default_export_report_csv))
    cb_report_json = QCheckBox(_safe_str(report_json_label))
    cb_report_json.setChecked(bool(default_export_report_json))

    cb_conf = QCheckBox(_safe_str(confusion_label))
    cb_conf.setChecked(bool(default_export_confusion_matrix))

    cb_pca = QCheckBox(_safe_str(pca_boundary_label))
    cb_pca.setChecked(bool(default_export_pca_boundary))

    cb_roc = QCheckBox(_safe_str(roc_label))
    cb_roc.setChecked(bool(default_export_roc_curve))

    cb_dist = QCheckBox(_safe_str(pred_dist_label))
    cb_dist.setChecked(bool(default_export_prediction_distribution))

    cb_fi = QCheckBox(_safe_str(feat_imp_label))
    cb_fi.setChecked(bool(default_export_feature_importance))

    if show_report:
        layout.addWidget(cb_report_csv)
        layout.addWidget(cb_report_json)
    if show_confusion_matrix:
        layout.addWidget(cb_conf)
    if show_pca_boundary:
        layout.addWidget(cb_pca)
    if show_roc_curve:
        layout.addWidget(cb_roc)
    if show_prediction_distribution:
        layout.addWidget(cb_dist)
    if show_feature_importance:
        layout.addWidget(cb_fi)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    if dialog.exec() != QDialog.Accepted:
        return None

    directory = _safe_str(location_edit.text())
    if not directory:
        return None

    base_name = _safe_str(base_edit.text())
    if not base_name:
        return None

    base_name = _sanitize_filename_component(base_name, fallback="export")

    image_format = _safe_str(img_fmt_combo.currentData()) or "png"
    image_format = image_format.lower()
    if image_format not in ("png", "svg"):
        image_format = "png"

    return ExportBundleOptions(
        directory=directory,
        base_name=base_name,
        image_format=image_format,
        export_report_csv=bool(cb_report_csv.isChecked()) if show_report else False,
        export_report_json=bool(cb_report_json.isChecked()) if show_report else False,
        export_confusion_matrix=bool(cb_conf.isChecked()) if show_confusion_matrix else False,
        export_pca_boundary=bool(cb_pca.isChecked()) if show_pca_boundary else False,
        export_roc_curve=bool(cb_roc.isChecked()) if show_roc_curve else False,
        export_prediction_distribution=bool(cb_dist.isChecked()) if show_prediction_distribution else False,
        export_feature_importance=bool(cb_fi.isChecked()) if show_feature_importance else False,
    )


def get_export_analysis_bundle_options(
    parent: QWidget,
    *,
    title: str,
    default_directory: str | None = None,
    default_base_name: str | None = None,
    default_image_format: str = "png",
    # Defaults
    default_export_data_csv: bool = True,
    default_export_data_json: bool = False,
    default_export_primary_plot: bool = True,
    default_export_secondary_plot: bool = False,
    # Labels
    location_label: str = "Save Location:",
    base_name_label: str = "Base Name:",
    browse_button_text: str = "Browse...",
    select_location_title: str = "Select Location",
    image_format_label: str = "Image Format:",
    section_label: str = "Include:",
    data_csv_label: str = "Data table (CSV)",
    data_json_label: str = "Data table (JSON)",
    primary_plot_label: str = "Primary plot (image)",
    secondary_plot_label: str = "Secondary plot (image)",
) -> Optional[ExportAnalysisBundleOptions]:
    """Shared export dialog for exporting Analysis result artifacts.

    Returns None if the user cancels.
    """

    dialog = QDialog(parent)
    dialog.setWindowTitle(_safe_str(title) or "Export")
    dialog.setMinimumWidth(560)
    dialog.setStyleSheet(_EXPORT_DIALOG_STYLESHEET)

    layout = QVBoxLayout(dialog)
    layout.setSpacing(16)
    layout.setContentsMargins(24, 24, 24, 24)

    # Directory selection
    location_layout = QHBoxLayout()
    location_layout.addWidget(QLabel(_safe_str(location_label)))

    location_edit = QLineEdit(_safe_str(default_directory) or "")
    location_edit.setReadOnly(True)

    browse_btn = QPushButton(_safe_str(browse_button_text))

    def _browse_location() -> None:
        start_path = location_edit.text() if location_edit.text() else os.getcwd()
        picked = QFileDialog.getExistingDirectory(dialog, _safe_str(select_location_title), start_path)
        if picked:
            location_edit.setText(str(picked))

    browse_btn.clicked.connect(_browse_location)

    location_layout.addWidget(location_edit, 1)
    location_layout.addWidget(browse_btn)
    layout.addLayout(location_layout)

    # Base name
    base_row = QHBoxLayout()
    base_row.addWidget(QLabel(_safe_str(base_name_label)))
    base_edit = QLineEdit(_safe_str(default_base_name) or "")
    base_row.addWidget(base_edit)
    layout.addLayout(base_row)

    # Image format
    fmt_row = QHBoxLayout()
    fmt_row.addWidget(QLabel(_safe_str(image_format_label)))
    img_fmt_combo = QComboBox()
    img_fmt_combo.addItem("PNG (.png)", "png")
    img_fmt_combo.addItem("SVG (.svg)", "svg")
    try:
        idx = img_fmt_combo.findData(_safe_str(default_image_format).lower())
        if idx >= 0:
            img_fmt_combo.setCurrentIndex(idx)
    except Exception:
        pass
    fmt_row.addWidget(img_fmt_combo)
    layout.addLayout(fmt_row)

    # Section label
    section = QLabel(_safe_str(section_label))
    section.setStyleSheet("font-weight: 700;")
    layout.addWidget(section)

    cb_data_csv = QCheckBox(_safe_str(data_csv_label))
    cb_data_csv.setChecked(bool(default_export_data_csv))
    cb_data_json = QCheckBox(_safe_str(data_json_label))
    cb_data_json.setChecked(bool(default_export_data_json))

    cb_primary = QCheckBox(_safe_str(primary_plot_label))
    cb_primary.setChecked(bool(default_export_primary_plot))
    cb_secondary = QCheckBox(_safe_str(secondary_plot_label))
    cb_secondary.setChecked(bool(default_export_secondary_plot))

    layout.addWidget(cb_data_csv)
    layout.addWidget(cb_data_json)
    layout.addWidget(cb_primary)
    layout.addWidget(cb_secondary)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    if dialog.exec() != QDialog.Accepted:
        return None

    directory = _safe_str(location_edit.text())
    if not directory:
        return None

    base_name = _safe_str(base_edit.text())
    if not base_name:
        return None

    base_name = _sanitize_filename_component(base_name, fallback="analysis")

    image_format = _safe_str(img_fmt_combo.currentData()) or "png"
    image_format = image_format.lower()
    if image_format not in ("png", "svg"):
        image_format = "png"

    return ExportAnalysisBundleOptions(
        directory=directory,
        base_name=base_name,
        image_format=image_format,
        export_data_csv=bool(cb_data_csv.isChecked()),
        export_data_json=bool(cb_data_json.isChecked()),
        export_primary_plot=bool(cb_primary.isChecked()),
        export_secondary_plot=bool(cb_secondary.isChecked()),
    )


def get_export_shap_bundle_options(
    parent: QWidget,
    *,
    title: str,
    default_directory: str | None = None,
    default_base_name: str | None = None,
    default_image_format: str = "png",
    # Default selection
    default_export_spectrum_plot: bool = True,
    default_export_shap_plot: bool = True,
    default_export_contributors_csv: bool = True,
    default_export_raw_json: bool = True,
    default_export_metadata_json: bool = False,
    # Labels (callers should pass localized strings)
    location_label: str = "Save Location:",
    base_name_label: str = "Base Name:",
    browse_button_text: str = "Browse...",
    select_location_title: str = "Select Location",
    image_format_label: str = "Image Format:",
    section_label: str = "Include:",
    spectrum_label: str = "Spectrum plot (image)",
    shap_label: str = "SHAP plot (image)",
    table_label: str = "Contributors table (CSV)",
    raw_json_label: str = "Raw values (JSON)",
    meta_json_label: str = "Metadata (JSON)",
) -> Optional[ExportShapBundleOptions]:
    """Shared export dialog for SHAP explanation artifacts.

    Returns None if the user cancels.
    """

    dialog = QDialog(parent)
    dialog.setWindowTitle(_safe_str(title) or "Export")
    dialog.setMinimumWidth(560)
    dialog.setStyleSheet(_EXPORT_DIALOG_STYLESHEET)

    layout = QVBoxLayout(dialog)
    layout.setSpacing(16)
    layout.setContentsMargins(24, 24, 24, 24)

    # Directory selection
    location_layout = QHBoxLayout()
    location_layout.addWidget(QLabel(_safe_str(location_label)))

    location_edit = QLineEdit(_safe_str(default_directory) or "")
    location_edit.setReadOnly(True)

    from PySide6.QtWidgets import QPushButton

    browse_btn = QPushButton(_safe_str(browse_button_text))

    def _browse_location() -> None:
        start_path = location_edit.text() if location_edit.text() else os.getcwd()
        picked = QFileDialog.getExistingDirectory(dialog, _safe_str(select_location_title), start_path)
        if picked:
            location_edit.setText(str(picked))

    browse_btn.clicked.connect(_browse_location)

    location_layout.addWidget(location_edit, 1)
    location_layout.addWidget(browse_btn)
    layout.addLayout(location_layout)

    # Base name
    base_row = QHBoxLayout()
    base_row.addWidget(QLabel(_safe_str(base_name_label)))
    base_edit = QLineEdit(_safe_str(default_base_name) or "")
    base_row.addWidget(base_edit)
    layout.addLayout(base_row)

    # Image format
    fmt_row = QHBoxLayout()
    fmt_row.addWidget(QLabel(_safe_str(image_format_label)))
    img_fmt_combo = QComboBox()
    img_fmt_combo.addItem("PNG (.png)", "png")
    img_fmt_combo.addItem("SVG (.svg)", "svg")
    try:
        idx = img_fmt_combo.findData(_safe_str(default_image_format).lower())
        if idx >= 0:
            img_fmt_combo.setCurrentIndex(idx)
    except Exception:
        pass
    fmt_row.addWidget(img_fmt_combo)
    layout.addLayout(fmt_row)

    # Section label
    section = QLabel(_safe_str(section_label))
    section.setStyleSheet("font-weight: 700;")
    layout.addWidget(section)

    cb_spec = QCheckBox(_safe_str(spectrum_label))
    cb_spec.setChecked(bool(default_export_spectrum_plot))
    cb_shap = QCheckBox(_safe_str(shap_label))
    cb_shap.setChecked(bool(default_export_shap_plot))
    cb_table = QCheckBox(_safe_str(table_label))
    cb_table.setChecked(bool(default_export_contributors_csv))
    cb_raw = QCheckBox(_safe_str(raw_json_label))
    cb_raw.setChecked(bool(default_export_raw_json))
    cb_meta = QCheckBox(_safe_str(meta_json_label))
    cb_meta.setChecked(bool(default_export_metadata_json))

    layout.addWidget(cb_spec)
    layout.addWidget(cb_shap)
    layout.addWidget(cb_table)
    layout.addWidget(cb_raw)
    layout.addWidget(cb_meta)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    if dialog.exec() != QDialog.Accepted:
        return None

    directory = _safe_str(location_edit.text())
    if not directory:
        return None

    base_name = _safe_str(base_edit.text())
    if not base_name:
        return None

    base_name = _sanitize_filename_component(base_name, fallback="shap")

    image_format = _safe_str(img_fmt_combo.currentData()) or "png"
    image_format = image_format.lower()
    if image_format not in ("png", "svg"):
        image_format = "png"

    return ExportShapBundleOptions(
        directory=directory,
        base_name=base_name,
        image_format=image_format,
        export_spectrum_plot=bool(cb_spec.isChecked()),
        export_shap_plot=bool(cb_shap.isChecked()),
        export_contributors_csv=bool(cb_table.isChecked()),
        export_raw_json=bool(cb_raw.isChecked()),
        export_metadata_json=bool(cb_meta.isChecked()),
    )
