from __future__ import annotations

import datetime
import csv
import json
import os
import re
import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
	QCheckBox,
	QComboBox,
	QFormLayout,
	QGridLayout,
	QFrame,
	QGroupBox,
	QHBoxLayout,
	QDialog,
	QLabel,
	QLineEdit,
	QMessageBox,
	QProgressBar,
	QPushButton,
	QScrollArea,
	QSplitter,
	QTabWidget,
	QTextEdit,
	QVBoxLayout,
	QWidget,
	QFileDialog,
	QListWidgetItem,
	QTableWidget,
	QTableWidgetItem,
)

from components.widgets.matplotlib_widget import MatplotlibWidget
from components.widgets.results_panel import apply_modern_tab_style
from components.widgets.parameter_widgets import CustomDoubleSpinBox, CustomSpinBox
from components.widgets.multi_group_dialog import MultiGroupCreationDialog
from components.widgets.external_evaluation_dialog import ExternalEvaluationDialog
from configs.configs import create_logs
from configs.style.stylesheets import combine_styles, get_base_style, get_page_style
from functions.ML import prepare_features_for_dataset
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	brier_score_loss,
	confusion_matrix,
	f1_score,
	roc_auc_score,
)
from functions.visualization.model_evaluation import (
	create_confusion_matrix_figure,
	create_feature_importance_figure,
	create_prediction_distribution_figure,
	create_roc_curve_figure,
)
from functions.visualization.classification_report import create_classification_report_table
from functions.visualization.pca_decision_boundary import (
	create_pca_decision_boundary_figure,
	create_pca_scatter_figure,
)
from components.widgets.grouping.dnd_widgets import DatasetSourceList, GroupDropList
from pages.machine_learning_page_utils.thread import MLTrainingOutput, MLTrainingThread
from utils import LOCALIZE, PROJECT_MANAGER, RAMAN_DATA, register_groups_changed_listener

from components.widgets.icons import load_icon


class CardFrame(QFrame):
	def __init__(self, title: str, parent=None):
		super().__init__(parent)
		self.setObjectName("cardFrame")
		self.setStyleSheet(
			"""
			QFrame#cardFrame {
				background-color: #ffffff;
				border: 1px solid #dfe3ea;
				border-radius: 12px;
			}
			"""
		)
		self.main_layout = QVBoxLayout(self)
		self.main_layout.setContentsMargins(16, 16, 16, 16)
		self.main_layout.setSpacing(12)

		header_layout = QHBoxLayout()
		self.title_label = QLabel(title)
		self.title_label.setStyleSheet(
			"font-size: 16px; font-weight: 600; color: #1f2a37;"
		)
		header_layout.addWidget(self.title_label)
		header_layout.addStretch()
		self.main_layout.addLayout(header_layout)

		# Content placeholder
		self.content_layout = QVBoxLayout()
		self.content_layout.setSpacing(10)
		self.main_layout.addLayout(self.content_layout)

	def add_widget(self, widget: QWidget):
		self.content_layout.addWidget(widget)

	def add_layout(self, layout):
		self.content_layout.addLayout(layout)

	def set_title(self, title: str):
		self.title_label.setText(title)


def _sanitize_filename_component(text: str) -> str:
	# Windows-safe filename component (keep it simple and predictable)
	s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
	s = re.sub(r"_+", "_", s).strip("_")
	return s or "untitled"


class _DestinationDialog(QDialog):
	"""Reusable destination dialog (folder + filename) with optional checkbox."""

	def __init__(
		self,
		*,
		title: str,
		default_dir: str,
		default_filename: str,
		ok_text_key: str = "ML_PAGE.dialog_save",
		cancel_text_key: str = "COMMON.cancel",
		show_save_params: bool = False,
		default_save_params: bool = True,
		parent=None,
	):
		super().__init__(parent)
		self.setWindowTitle(title)
		self.setModal(True)

		self._dir = str(default_dir or os.getcwd())
		self._filename = str(default_filename or "")
		self._save_params = bool(default_save_params)

		root = QVBoxLayout(self)
		root.setContentsMargins(16, 16, 16, 16)
		root.setSpacing(12)

		form = QFormLayout()
		form.setVerticalSpacing(10)
		form.setHorizontalSpacing(12)
		root.addLayout(form)

		# Destination folder
		dir_row = QHBoxLayout()
		dir_row.setContentsMargins(0, 0, 0, 0)
		self.dir_edit = QLineEdit(self._dir)
		self.dir_edit.setReadOnly(True)
		browse_btn = QPushButton(LOCALIZE("ML_PAGE.dialog_browse"))
		browse_btn.setCursor(Qt.PointingHandCursor)
		browse_btn.setStyleSheet(get_base_style("secondary_button"))
		browse_btn.clicked.connect(self._browse_dir)
		dir_row.addWidget(self.dir_edit, 1)
		dir_row.addWidget(browse_btn, 0)
		form.addRow(QLabel(LOCALIZE("ML_PAGE.dialog_destination")), dir_row)

		# File name
		self.name_edit = QLineEdit(self._filename)


@dataclass(frozen=True)
class _SHAPOptions:
	source: str  # "train" or "test"
	sample_index: int
	background_samples: int
	max_evals: int  # 0 = auto
	top_k: int
	random_state: int


class _SHAPOptionsDialog(QDialog):
	def __init__(
		self,
		*,
		train_count: int = 0,
		test_count: int = 0,
		default_source: str = "test",
		default_sample_index: int = 0,
		default_background_samples: int = 30,
		default_max_evals: int = 0,
		default_top_k: int = 12,
		default_random_state: int = 0,
		parent=None,
	):
		super().__init__(parent)
		self.setWindowTitle(LOCALIZE("ML_PAGE.shap_dialog_title"))
		self.setModal(True)
		self._train_count = max(0, int(train_count))
		self._test_count = max(0, int(test_count))

		self._opts = _SHAPOptions(
			source=str(default_source or "test"),
			sample_index=int(default_sample_index),
			background_samples=int(default_background_samples),
			max_evals=int(default_max_evals),
			top_k=int(default_top_k),
			random_state=int(default_random_state),
		)

		root = QVBoxLayout(self)
		root.setContentsMargins(16, 16, 16, 16)
		root.setSpacing(12)

		form = QFormLayout()
		form.setVerticalSpacing(10)
		form.setHorizontalSpacing(12)
		root.addLayout(form)

		self.source_combo = QComboBox()
		self.source_combo.addItem(LOCALIZE("ML_PAGE.shap_source_test"), userData="test")
		self.source_combo.addItem(LOCALIZE("ML_PAGE.shap_source_train"), userData="train")
		# Default selection
		if str(default_source) == "train":
			self.source_combo.setCurrentIndex(1)
		form.addRow(QLabel(LOCALIZE("ML_PAGE.shap_source_label")), self.source_combo)

		self.sample_index_spin = CustomSpinBox()
		self.sample_index_spin.setRange(0, 0)
		self.sample_index_spin.setValue(int(default_sample_index))
		form.addRow(QLabel(LOCALIZE("ML_PAGE.shap_sample_index")), self.sample_index_spin)

		def _sync_index_range() -> None:
			src = str(self.source_combo.currentData() or self.source_combo.currentText()).strip() or "test"
			count = self._train_count if src == "train" else self._test_count
			max_idx = max(0, int(count) - 1)
			try:
				self.sample_index_spin.setRange(0, max_idx)
			except Exception:
				pass
			# Clamp current value
			try:
				v = int(self.sample_index_spin.value())
				self.sample_index_spin.setValue(int(max(0, min(v, max_idx))))
			except Exception:
				pass

		_sync_index_range()
		self.source_combo.currentIndexChanged.connect(_sync_index_range)

		self.bg_samples_spin = CustomSpinBox()
		self.bg_samples_spin.setRange(1, 1000)
		self.bg_samples_spin.setValue(int(default_background_samples))
		form.addRow(QLabel(LOCALIZE("ML_PAGE.shap_background_samples")), self.bg_samples_spin)

		# Advanced: max_evals for permutation explainer.
		# 0 means "auto" (the worker will set it to >= 2*num_features+1).
		self.max_evals_spin = CustomSpinBox()
		self.max_evals_spin.setRange(0, 200000)
		self.max_evals_spin.setValue(int(default_max_evals))
		form.addRow(QLabel(LOCALIZE("ML_PAGE.shap_max_evals")), self.max_evals_spin)

		self.top_k_spin = CustomSpinBox()
		self.top_k_spin.setRange(5, 200)
		self.top_k_spin.setValue(int(default_top_k))
		form.addRow(QLabel(LOCALIZE("ML_PAGE.shap_top_k")), self.top_k_spin)

		self.seed_spin = CustomSpinBox()
		self.seed_spin.setRange(0, 999999)
		self.seed_spin.setValue(int(default_random_state))
		form.addRow(QLabel(LOCALIZE("ML_PAGE.shap_seed")), self.seed_spin)

		btn_row = QHBoxLayout()
		btn_row.addStretch(1)
		cancel_btn = QPushButton(LOCALIZE("COMMON.cancel"))
		cancel_btn.setCursor(Qt.PointingHandCursor)
		cancel_btn.setStyleSheet(get_base_style("secondary_button"))
		cancel_btn.clicked.connect(self.reject)
		run_btn = QPushButton(LOCALIZE("ML_PAGE.shap_run"))
		run_btn.setCursor(Qt.PointingHandCursor)
		run_btn.setStyleSheet(get_base_style("primary_button"))
		run_btn.clicked.connect(self._on_run)
		btn_row.addWidget(cancel_btn)
		btn_row.addWidget(run_btn)
		root.addLayout(btn_row)

	def _on_run(self):
		src = str(self.source_combo.currentData() or self.source_combo.currentText()).strip() or "test"
		self._opts = _SHAPOptions(
			source=src,
			sample_index=int(self.sample_index_spin.value()),
			background_samples=int(self.bg_samples_spin.value()),
			max_evals=int(self.max_evals_spin.value()),
			top_k=int(self.top_k_spin.value()),
			random_state=int(self.seed_spin.value()),
		)
		self.accept()

	@property
	def options(self) -> _SHAPOptions:
		return self._opts
		form.addRow(QLabel(LOCALIZE("ML_PAGE.dialog_filename")), self.name_edit)

		# Optional checkbox
		self.save_params_cb = None
		if show_save_params:
			self.save_params_cb = QCheckBox(LOCALIZE("ML_PAGE.dialog_save_params"))
			self.save_params_cb.setChecked(bool(default_save_params))
			root.addWidget(self.save_params_cb)

		# Buttons
		btn_row = QHBoxLayout()
		btn_row.addStretch(1)
		try:
			cancel_text = LOCALIZE(cancel_text_key)
		except Exception:
			cancel_text = "Cancel"
		cancel_btn = QPushButton(cancel_text)
		cancel_btn.setCursor(Qt.PointingHandCursor)
		cancel_btn.setStyleSheet(get_base_style("secondary_button"))
		cancel_btn.clicked.connect(self.reject)
		try:
			ok_text = LOCALIZE(ok_text_key)
		except Exception:
			ok_text = "OK"
		ok_btn = QPushButton(ok_text)
		ok_btn.setCursor(Qt.PointingHandCursor)
		ok_btn.setStyleSheet(get_base_style("primary_button"))
		ok_btn.clicked.connect(self.accept)
		btn_row.addWidget(cancel_btn)
		btn_row.addWidget(ok_btn)
		root.addLayout(btn_row)

	def _browse_dir(self):
		picked = QFileDialog.getExistingDirectory(self, LOCALIZE("ML_PAGE.dialog_destination"), self._dir)
		if picked:
			self._dir = str(picked)
			self.dir_edit.setText(self._dir)

	def selected_dir(self) -> str:
		return str(self.dir_edit.text() or self._dir or os.getcwd())

	def filename(self) -> str:
		return str(self.name_edit.text() or "").strip()

	def save_params(self) -> bool:
		if self.save_params_cb is None:
			return False
		return bool(self.save_params_cb.isChecked())


@dataclass
class _GroupUi:
	group_id: str
	container: QWidget
	name_label: QLabel
	name_edit: QLineEdit
	include_checkbox: QCheckBox
	list_widget: GroupDropList
	remove_button: Optional[QPushButton] = None


class MachineLearningPage(QWidget):
	"""Functional Machine Learning page.

	Workflow:
	- Drag datasets into 2+ groups (editable class labels)
	- Choose split strategy + ratio
	- Choose model + parameters
	- Train in background thread
	- View results on the right (tabs)
	- Save/load model (.pkl) and evaluate on another dataset
	"""

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setObjectName("machineLearningPage")
		self._groups: List[_GroupUi] = []
		self._group_counter = 0
		# Used to suppress auto-saving during bulk UI reconstruction (project load / dialog apply)
		self._suspend_group_persist = False

		self._training_thread: Optional[MLTrainingThread] = None
		self._trained_model = None
		self._trained_axis: Optional[np.ndarray] = None
		self._trained_class_labels: List[str] = []
		self._trained_model_key: Optional[str] = None
		self._trained_model_params: Dict = {}
		self._last_train_context: Dict[str, object] = {}
		self._last_report_rows: List[Dict[str, object]] = []

		self._xgboost_available = bool(importlib.util.find_spec("xgboost") is not None)

		self._setup_ui()
		self._connect_signals()
		self.update_localized_text()
		self.load_project_data()

		# Runtime sync: refresh this page's groups when Analysis updates shared groups.
		try:
			register_groups_changed_listener(self._on_external_groups_changed)
		except Exception:
			pass

		# Add page-specific styles for better visibility
		self.setStyleSheet("""
			QCheckBox::indicator {
				width: 20px;
				height: 20px;
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
				spacing: 8px;
				font-size: 13px;
				color: #333;
			}

			/* Fix: Windows theme can render combo popup dark; force light popup like Analysis page */
			QComboBox {
				background-color: #ffffff;
				color: #212529;
				border: 1px solid #ced4da;
				border-radius: 6px;
				padding: 4px 8px;
				min-height: 28px;
			}
			QComboBox:hover { border-color: #86b7fe; }
			QComboBox:focus { border-color: #0078d4; }
			QComboBox::drop-down {
				border: none;
				width: 26px;
				background-color: #ffffff;
			}
			QComboBox QAbstractItemView {
				background-color: #ffffff;
				color: #212529;
				selection-background-color: #0078d4;
				selection-color: #ffffff;
				outline: 0;
				border: 1px solid #ced4da;
				/* Some Windows styles can render the popup with a transparent base unless explicit */
				alternate-background-color: #ffffff;
			}
			QComboBox QAbstractItemView::item {
				background-color: #ffffff;
				padding: 6px 10px;
			}
			QComboBox QAbstractItemView::item:selected {
				background-color: #0078d4;
				color: #ffffff;
			}
			QComboBox QAbstractItemView::item:hover {
				background-color: #e7f1ff;
				color: #212529;
			}
		""")

	def _setup_ui(self):
		root = QVBoxLayout(self)
		root.setContentsMargins(0, 0, 0, 0)
		root.setSpacing(0)

		# Static two-panel layout (no draggable splitter)
		panels = QHBoxLayout()
		panels.setContentsMargins(0, 0, 0, 0)
		panels.setSpacing(0)
		root.addLayout(panels, 1)

		# === LEFT PANEL ===
		self.left_scroll = QScrollArea()
		self.left_scroll.setWidgetResizable(True)
		self.left_scroll.setFrameShape(QFrame.NoFrame)
		self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		# Keep left panel width stable (user requested static sizing)
		self.left_scroll.setFixedWidth(600)
		
		self.left_panel_content = QWidget()
		self.left_panel_content.setStyleSheet("background-color: #f8f9fa;")
		left_layout = QVBoxLayout(self.left_panel_content)
		left_layout.setContentsMargins(24, 24, 24, 24)
		left_layout.setSpacing(20)

		self.left_scroll.setWidget(self.left_panel_content)
		panels.addWidget(self.left_scroll, 0)

		# Page Title
		self.title_label = QLabel()
		self.title_label.setStyleSheet("font-size: 24px; font-weight: 700; color: #1a1a1a; margin-bottom: 8px;")
		left_layout.addWidget(self.title_label)

		# 1. Available Datasets Card
		self.datasets_card = CardFrame("")
		left_layout.addWidget(self.datasets_card)
		
		# Refresh button row
		self.datasets_refresh_button = QPushButton()
		self.datasets_refresh_button.setObjectName("mlRefreshDatasetsButton")
		self.datasets_refresh_button.setCursor(Qt.PointingHandCursor)
		# Style refresh button as secondary/icon
		self.datasets_refresh_button.setStyleSheet(get_base_style("secondary_button"))
		
		# List
		self.dataset_source_list = DatasetSourceList()
		self.dataset_source_list.setMinimumHeight(150)
		
		self.datasets_card.add_layout(self._create_header_with_widget(self.datasets_refresh_button))
		self.datasets_card.add_widget(self.dataset_source_list)

		# 2. Groups Card
		self.groups_card = CardFrame("")
		left_layout.addWidget(self.groups_card)
		self.groups_title_label = QLabel()  # Subtitle description
		self.groups_title_label.setWordWrap(True)
		self.groups_title_label.setStyleSheet(
			"color: #6c757d; font-size: 13px; margin-bottom: 8px;"
		)
		self.groups_card.add_widget(self.groups_title_label)

		# Groups container (Tabs)
		self.groups_tabs = QTabWidget()
		self.groups_tabs.setTabsClosable(True)
		self.groups_tabs.setMovable(False)
		self.groups_tabs.setMinimumHeight(250)
		self.groups_tabs.setStyleSheet(
			"""
			QTabWidget::pane { border: 1px solid #dee2e6; border-radius: 4px; top: -1px; }
			QTabBar::tab { background: #f8f9fa; border: 1px solid #dee2e6; padding: 6px 12px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; color: #495057; }
			QTabBar::tab:selected { background: #ffffff; border-bottom-color: #ffffff; font-weight: 500; color: #0078d4; }
			QTabBar::close-button { subcontrol-position: right; }
		"""
		)
		self.groups_tabs.tabCloseRequested.connect(self._on_group_tab_closed)
		self.groups_card.add_widget(self.groups_tabs)

		# Group actions
		groups_btn_row = QHBoxLayout()
		self.add_group_button = QPushButton()
		self.add_group_button.setCursor(Qt.PointingHandCursor)
		self.add_group_button.setStyleSheet(get_base_style("secondary_button"))

		self.clear_groups_button = QPushButton()
		self.clear_groups_button.setCursor(Qt.PointingHandCursor)
		self.clear_groups_button.setStyleSheet(get_base_style("danger_button"))

		groups_btn_row.addWidget(self.add_group_button)
		groups_btn_row.addWidget(self.clear_groups_button)
		groups_btn_row.addStretch(1)
		self.groups_card.add_layout(groups_btn_row)

		# 3. Split Settings Card
		self.split_card = CardFrame("")
		left_layout.addWidget(self.split_card)

		# Use Grid Layout for better space utilization
		split_layout = QGridLayout()
		split_layout.setVerticalSpacing(12)
		split_layout.setHorizontalSpacing(16)

		# Row 1: Train Ratio & Random State
		self.train_ratio_label = QLabel()
		self.train_ratio_spin = CustomDoubleSpinBox()
		self.train_ratio_spin.setRange(0.05, 0.95)
		self.train_ratio_spin.setSingleStep(0.05)
		self.train_ratio_spin.setDecimals(2)
		self.train_ratio_spin.setValue(0.8)
		self.train_ratio_spin.setMinimumWidth(100)

		self.random_state_label = QLabel()
		self.random_state_spin = CustomSpinBox()
		self.random_state_spin.setRange(0, 10_000_000)
		self.random_state_spin.setSingleStep(1)
		self.random_state_spin.setValue(42)
		self.random_state_spin.setMinimumWidth(100)

		split_layout.addWidget(self.train_ratio_label, 0, 0)
		split_layout.addWidget(self.train_ratio_spin, 0, 1)
		split_layout.addWidget(self.random_state_label, 0, 2)
		split_layout.addWidget(self.random_state_spin, 0, 3)

		# Row 2: Split Mode (Full width span for better combo visibility)
		self.split_mode_label = QLabel()
		self.split_mode_combo = QComboBox()
		self.split_mode_combo.addItem("by_spectra", userData="by_spectra")
		self.split_mode_combo.addItem("by_dataset", userData="by_dataset")
		self.split_mode_combo.setMinimumWidth(150)

		# Label on left, Combo spans rest
		split_layout.addWidget(self.split_mode_label, 1, 0)
		split_layout.addWidget(self.split_mode_combo, 1, 1, 1, 3)

		# Set column stretches to keep inputs compact but flexible
		split_layout.setColumnStretch(1, 1)
		split_layout.setColumnStretch(3, 1)

		self.split_card.add_layout(split_layout)

		# 4. Model Settings Card
		self.model_card = CardFrame("")
		left_layout.addWidget(self.model_card)

		model_outer = QVBoxLayout()
		model_outer.setSpacing(12)
		
		# Model Selector
		model_sel_row = QHBoxLayout()
		self.model_method_label = QLabel()
		self.model_method_label.setStyleSheet("font-weight: 500;")
		self.model_combo = QComboBox()
		self.model_combo.addItem("linear_regression", userData="linear_regression")
		self.model_combo.addItem("logistic_regression", userData="logistic_regression")
		self.model_combo.addItem("svm", userData="svm")
		self.model_combo.addItem("random_forest", userData="random_forest")
		if self._xgboost_available:
			self.model_combo.addItem("xgboost", userData="xgboost")
		self.model_combo.setMinimumWidth(200)
		model_sel_row.addWidget(self.model_method_label)
		model_sel_row.addWidget(self.model_combo)
		model_sel_row.addStretch()
		model_outer.addLayout(model_sel_row)

		# Params Stack
		self.model_params_stack = QTabWidget()
		self.model_params_stack.setDocumentMode(True)
		self.model_params_stack.tabBar().setVisible(False)
		self.model_params_stack.setStyleSheet("background: transparent;")
		self._build_model_param_tabs()
		model_outer.addWidget(self.model_params_stack)
		
		self.model_card.add_layout(model_outer)

		# 5. Actions / Progress
		actions_row = QHBoxLayout()
		actions_row.setSpacing(12)
		
		self.start_training_button = QPushButton()
		self.start_training_button.setStyleSheet(get_base_style("primary_button") + "QPushButton { font-size: 14px; font-weight: 600; }")
		self.start_training_button.setCursor(Qt.PointingHandCursor)
		self.start_training_button.setMinimumHeight(45)
		
		self.save_model_button = QPushButton()
		self.save_model_button.setStyleSheet(get_base_style("secondary_button"))
		self.load_model_button = QPushButton()
		self.load_model_button.setStyleSheet(get_base_style("secondary_button"))
		
		actions_row.addWidget(self.start_training_button, 2)
		actions_row.addWidget(self.save_model_button, 1)
		actions_row.addWidget(self.load_model_button, 1)
		left_layout.addLayout(actions_row)

		self.progress = QProgressBar()
		self.progress.setRange(0, 100)
		self.progress.setValue(0)
		self.progress.setTextVisible(False)
		self.progress.setFixedHeight(6) # Thinner progress bar
		left_layout.addWidget(self.progress)

		self.status_label = QLabel()
		self.status_label.setAlignment(Qt.AlignCenter)
		self.status_label.setStyleSheet("color: #666; font-size: 13px;")
		left_layout.addWidget(self.status_label)

		# 6. Evaluation Card
		self.eval_card = CardFrame("")
		left_layout.addWidget(self.eval_card)

		eval_layout = QGridLayout()
		eval_layout.setVerticalSpacing(12)
		eval_layout.setHorizontalSpacing(16)

		self.eval_dataset_label = QLabel()
		self.eval_dataset_combo = QComboBox()

		self.eval_true_label_label = QLabel()
		self.eval_true_label_combo = QComboBox()
		# Note: localized in update_localized_text via setItemText(0, ...)
		self.eval_true_label_combo.addItem("(unknown)")

		# Row 1: Target Dataset
		eval_layout.addWidget(self.eval_dataset_label, 0, 0)
		eval_layout.addWidget(self.eval_dataset_combo, 0, 1)

		# Row 2: True Label
		eval_layout.addWidget(self.eval_true_label_label, 1, 0)
		eval_layout.addWidget(self.eval_true_label_combo, 1, 1)

		self.evaluate_button = QPushButton()
		self.evaluate_button.setStyleSheet(
			get_base_style("primary_button") + "QPushButton { margin-top: 8px; }"
		)
		self.evaluate_button.setCursor(Qt.PointingHandCursor)
		# Button spans both columns
		eval_layout.addWidget(self.evaluate_button, 2, 0, 1, 2)

		self.eval_card.add_layout(eval_layout)

		# 7. Log Card
		self.log_card = CardFrame("")
		left_layout.addWidget(self.log_card)
		
		self.log_text = QTextEdit()
		self.log_text.setReadOnly(True)
		self.log_text.setMaximumHeight(150)
		self.log_text.setStyleSheet("border: 1px solid #dee2e6; border-radius: 4px; padding: 4px; font-family: monospace; font-size: 11px;")
		self.log_card.add_widget(self.log_text)

		left_layout.addStretch()

		# === RIGHT PANEL (RESULTS) ===
		self.right_panel = QFrame()
		self.right_panel.setObjectName("rightPanel")
		self.right_panel.setStyleSheet("background-color: #ffffff;")
		right_layout = QVBoxLayout(self.right_panel)
		right_layout.setContentsMargins(16, 16, 16, 16)
		right_layout.setSpacing(8)

		# Results actions
		right_actions = QHBoxLayout()
		right_actions.setContentsMargins(0, 0, 0, 0)
		right_actions.addStretch(1)
		self.pca_view_combo = QComboBox()
		self.pca_view_combo.setObjectName("mlPcaViewCombo")
		self.pca_view_combo.setMinimumWidth(190)
		self.pca_view_combo.addItem("PCA + Decision Boundary", userData="boundary")
		self.pca_view_combo.addItem("PCA only", userData="pca_only")
		right_actions.addWidget(self.pca_view_combo)
		self.shap_button = QPushButton()
		self.shap_button.setObjectName("mlShapButton")
		self.shap_button.setCursor(Qt.PointingHandCursor)
		self.shap_button.setStyleSheet(get_base_style("secondary_button"))
		right_actions.addWidget(self.shap_button)
		self.export_report_button = QPushButton()
		self.export_report_button.setObjectName("mlExportReportButton")
		self.export_report_button.setCursor(Qt.PointingHandCursor)
		self.export_report_button.setStyleSheet(get_base_style("secondary_button"))
		right_actions.addWidget(self.export_report_button)
		right_layout.addLayout(right_actions)

		self.results_tabs = QTabWidget()
		apply_modern_tab_style(self.results_tabs, object_name="mlResultsTabs")

		self.cm_plot = MatplotlibWidget()
		self.pca_plot = MatplotlibWidget()
		self.roc_plot = MatplotlibWidget()
		self.fi_plot = MatplotlibWidget()
		self.pd_plot = MatplotlibWidget()
		self.report_table = QTableWidget()
		self.report_table.setColumnCount(5)
		self.report_table.setHorizontalHeaderLabels([
			"class",
			"precision",
			"recall",
			"f1-score",
			"support",
		])
		self.report_table.setEditTriggers(QTableWidget.NoEditTriggers)
		self.report_table.setAlternatingRowColors(True)
		self.report_table.horizontalHeader().setStretchLastSection(True)
		self.report_table.verticalHeader().setVisible(False)
		self.report_table.setStyleSheet("QTableWidget { border: 1px solid #dee2e6; }")

		self.eval_summary_text = QTextEdit()
		self.eval_summary_text.setReadOnly(True)
		self.eval_summary_text.setStyleSheet(
			"border: 1px solid #dee2e6; border-radius: 4px; padding: 8px; font-family: monospace; font-size: 12px;"
		)

		self.results_tabs.addTab(self.cm_plot, "")
		self.results_tabs.addTab(self.report_table, "")
		self.results_tabs.addTab(self.eval_summary_text, "")
		self.results_tabs.addTab(self.pca_plot, "")
		self.results_tabs.addTab(self.roc_plot, "")
		self.results_tabs.addTab(self.fi_plot, "")
		self.results_tabs.addTab(self.pd_plot, "")
		# Ensure the Matplotlib navigation toolbars are visible and not collapsed.
		for w in (self.cm_plot, self.pca_plot, self.roc_plot, self.fi_plot, self.pd_plot):
			try:
				w.toolbar.setVisible(True)
				w.toolbar.show()
				w.toolbar.setMinimumHeight(30)
			except Exception:
				pass

		right_layout.addWidget(self.results_tabs, 1)
		panels.addWidget(self.right_panel, 1)

		# Default: no groups. If the project has saved ML groups, they are restored.

	def _create_header_with_widget(self, widget: QWidget) -> QHBoxLayout:
		layout = QHBoxLayout()
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addWidget(widget)
		layout.addStretch()
		return layout

	def _connect_signals(self):
		self.datasets_refresh_button.clicked.connect(self.load_project_data)
		self.clear_groups_button.clicked.connect(self._clear_groups)
		# "Add group" opens the multi-group dialog (create multiple groups at once)
		self.add_group_button.clicked.connect(self._open_group_creation_dialog)
		self.model_combo.currentIndexChanged.connect(self._sync_model_param_tab)
		self.start_training_button.clicked.connect(self._start_training)
		self.save_model_button.clicked.connect(self._save_model)
		self.load_model_button.clicked.connect(self._load_model)
		self.evaluate_button.clicked.connect(self._evaluate_external_dataset)
		self.export_report_button.clicked.connect(self._export_classification_report)
		self.pca_view_combo.currentIndexChanged.connect(self._render_pca_plot)
		self.shap_button.clicked.connect(self._run_shap)

	def _build_model_param_tabs(self):
		# Helper for tab inner layout
		def make_param_tab(layout_func):
			w = QWidget()
			l = layout_func(w)
			l.setContentsMargins(0, 8, 0, 0)
			l.setVerticalSpacing(12)
			l.setHorizontalSpacing(16)
			l.setLabelAlignment(Qt.AlignLeft)
			return w, l

		def mk_label(key: str, fallback: str) -> QLabel:
			lbl = QLabel(LOCALIZE(key) if key else fallback)
			# Keep references so update_localized_text can refresh.
			self._ml_param_labels[key] = lbl
			return lbl

		# Localized labels used by model parameter forms
		self._ml_param_labels = {}

		# Logistic Regression
		# Linear Regression (classification via label-encoding)
		lin_tab, lin_form = make_param_tab(QFormLayout)
		self.lin_fit_intercept = QCheckBox("")
		self.lin_fit_intercept.setChecked(True)
		self.lin_positive = QCheckBox("")
		self.lin_positive.setChecked(False)
		lin_form.addRow(mk_label("ML_PAGE.params_fit_intercept", "Fit Intercept"), self.lin_fit_intercept)
		lin_form.addRow(mk_label("ML_PAGE.params_positive_coefs", "Positive Coefs"), self.lin_positive)
		self.model_params_stack.addTab(lin_tab, "linear_regression")

		# Logistic Regression
		log_tab, log_form = make_param_tab(QFormLayout)
		self.lr_C = CustomDoubleSpinBox()
		self.lr_C.setRange(1e-6, 1e6)
		self.lr_C.setDecimals(6)
		self.lr_C.setValue(1.0)
		self.lr_max_iter = CustomSpinBox()
		self.lr_max_iter.setRange(50, 5000)
		self.lr_max_iter.setValue(200)
		self.lr_solver = QComboBox()
		self.lr_solver.addItems(["lbfgs", "liblinear", "saga", "newton-cg"])
		log_form.addRow(mk_label("ML_PAGE.params_lr_c", "C (Regularization)"), self.lr_C)
		log_form.addRow(mk_label("ML_PAGE.params_lr_max_iter", "Max Iterations"), self.lr_max_iter)
		log_form.addRow(mk_label("ML_PAGE.params_lr_solver", "Solver"), self.lr_solver)
		self.model_params_stack.addTab(log_tab, "logistic_regression")

		# SVM
		svm_tab, svm_form = make_param_tab(QFormLayout)
		self.svm_kernel = QComboBox()
		self.svm_kernel.addItem("rbf", userData="rbf")
		self.svm_kernel.addItem("linear", userData="linear")
		self.svm_kernel.addItem("poly", userData="poly")
		self.svm_kernel.addItem("sigmoid", userData="sigmoid")
		self.svm_C = CustomDoubleSpinBox()
		self.svm_C.setDecimals(6)
		self.svm_C.setDecimals(6)
		self.svm_C.setValue(1.0)
		self.svm_gamma_mode = QComboBox()
		self.svm_gamma_mode.addItem("scale", userData="scale")
		self.svm_gamma_mode.addItem("auto", userData="auto")
		self.svm_gamma_mode.addItem("custom", userData="custom")
		self.svm_gamma_value = CustomDoubleSpinBox()
		self.svm_gamma_value.setRange(1e-9, 1e3)
		self.svm_gamma_value.setDecimals(9)
		self.svm_degree = CustomSpinBox()
		self.svm_degree.setRange(2, 8)
		self.svm_degree.setValue(3)
		svm_form.addRow(mk_label("ML_PAGE.params_svm_kernel", "Kernel"), self.svm_kernel)
		svm_form.addRow(mk_label("ML_PAGE.params_svm_c", "C"), self.svm_C)
		svm_form.addRow(mk_label("ML_PAGE.params_svm_gamma_mode", "Gamma Mode"), self.svm_gamma_mode)
		svm_form.addRow(mk_label("ML_PAGE.params_svm_gamma_value", "Gamma Value"), self.svm_gamma_value)
		svm_form.addRow(mk_label("ML_PAGE.params_svm_degree", "Degree (Poly)"), self.svm_degree)
		self.model_params_stack.addTab(svm_tab, "svm")

		def _sync_gamma_enabled():
			self.svm_gamma_value.setEnabled(self.svm_gamma_mode.currentData() == "custom")
		_sync_gamma_enabled()
		self.svm_gamma_mode.currentIndexChanged.connect(_sync_gamma_enabled)

		# Random Forest
		rf_tab, rf_form = make_param_tab(QFormLayout)
		self.rf_n_estimators = CustomSpinBox()
		self.rf_n_estimators.setRange(10, 5000)
		self.rf_n_estimators.setValue(200)
		self.rf_max_depth = CustomSpinBox()
		self.rf_max_depth.setRange(0, 256)
		self.rf_max_depth.setValue(0)
		self.rf_max_depth.setToolTip(LOCALIZE("ML_PAGE.rf_max_depth_tooltip"))
		self.rf_random_state = CustomSpinBox()
		self.rf_random_state.setRange(0, 10_000_000)
		self.rf_random_state.setValue(42)
		rf_form.addRow(mk_label("ML_PAGE.params_rf_trees", "Trees (n_estimators)"), self.rf_n_estimators)
		rf_form.addRow(mk_label("ML_PAGE.params_rf_max_depth", "Max Depth"), self.rf_max_depth)
		rf_form.addRow(mk_label("ML_PAGE.params_rf_random_state", "Random State"), self.rf_random_state)
		self.model_params_stack.addTab(rf_tab, "random_forest")

		# XGBoost (only if available)
		if self._xgboost_available:
			xgb_tab, xgb_form = make_param_tab(QFormLayout)
			self.xgb_n_estimators = CustomSpinBox()
			self.xgb_n_estimators.setRange(10, 5000)
			self.xgb_n_estimators.setValue(300)
			self.xgb_max_depth = CustomSpinBox()
			self.xgb_max_depth.setRange(1, 32)
			self.xgb_max_depth.setValue(6)
			self.xgb_learning_rate = CustomDoubleSpinBox()
			self.xgb_learning_rate.setRange(1e-4, 1.0)
			self.xgb_learning_rate.setDecimals(6)
			self.xgb_learning_rate.setValue(0.1)
			self.xgb_subsample = CustomDoubleSpinBox()
			self.xgb_subsample.setRange(0.1, 1.0)
			self.xgb_subsample.setDecimals(3)
			self.xgb_subsample.setValue(0.8)
			self.xgb_colsample_bytree = CustomDoubleSpinBox()
			self.xgb_colsample_bytree.setRange(0.1, 1.0)
			self.xgb_colsample_bytree.setDecimals(3)
			self.xgb_colsample_bytree.setValue(0.8)
			self.xgb_reg_lambda = CustomDoubleSpinBox()
			self.xgb_reg_lambda.setRange(0.0, 1000.0)
			self.xgb_reg_lambda.setDecimals(6)
			self.xgb_reg_lambda.setValue(1.0)

			xgb_form.addRow(mk_label("ML_PAGE.params_xgb_trees", "Trees (n_estimators)"), self.xgb_n_estimators)
			xgb_form.addRow(mk_label("ML_PAGE.params_xgb_max_depth", "Max depth"), self.xgb_max_depth)
			xgb_form.addRow(mk_label("ML_PAGE.params_xgb_learning_rate", "Learning rate"), self.xgb_learning_rate)
			xgb_form.addRow(mk_label("ML_PAGE.params_xgb_subsample", "Subsample"), self.xgb_subsample)
			xgb_form.addRow(mk_label("ML_PAGE.params_xgb_colsample_bytree", "Colsample bytree"), self.xgb_colsample_bytree)
			xgb_form.addRow(mk_label("ML_PAGE.params_xgb_reg_lambda", "L2 (lambda)"), self.xgb_reg_lambda)
			self.model_params_stack.addTab(xgb_tab, "xgboost")

		self._sync_model_param_tab()

	def _sync_model_param_tab(self):
		key = self.model_combo.currentData() or self.model_combo.currentText()
		for i in range(self.model_params_stack.count()):
			if self.model_params_stack.tabText(i) == key:
				self.model_params_stack.setCurrentIndex(i)
				break

	def update_localized_text(self):
		# Update window/root text
		self.title_label.setText(LOCALIZE("ML_PAGE.title"))
		
		# Update Cards
		self.datasets_card.set_title(LOCALIZE("ML_PAGE.available_datasets"))
		self.groups_card.set_title(LOCALIZE("ML_PAGE.groups"))
		self.split_card.set_title(LOCALIZE("ML_PAGE.split_settings"))
		self.model_card.set_title(LOCALIZE("ML_PAGE.model_settings"))
		self.eval_card.set_title(LOCALIZE("ML_PAGE.evaluation_settings"))
		self.log_card.set_title(LOCALIZE("ML_PAGE.log"))

		# Buttons & Labels
		self.datasets_refresh_button.setText(LOCALIZE("ML_PAGE.refresh"))
		self.groups_title_label.setText(LOCALIZE("ML_PAGE.groups_title"))
		self.add_group_button.setText(LOCALIZE("ML_PAGE.add_group"))
		self.clear_groups_button.setText(LOCALIZE("ML_PAGE.clear_groups"))
		self._set_training_button_state(bool(self._training_thread and self._training_thread.isRunning()))
		self.save_model_button.setText(LOCALIZE("ML_PAGE.save_model"))
		self.load_model_button.setText(LOCALIZE("ML_PAGE.load_model"))
		self.export_report_button.setText(LOCALIZE("ML_PAGE.export_report"))
		try:
			self.export_report_button.setToolTip(LOCALIZE("ML_PAGE.export_report_tooltip"))
		except Exception:
			pass
		self.shap_button.setText(LOCALIZE("ML_PAGE.shap_button"))
		try:
			self.shap_button.setToolTip(LOCALIZE("ML_PAGE.shap_button_tooltip"))
		except Exception:
			pass
		try:
			self.pca_view_combo.setItemText(0, LOCALIZE("ML_PAGE.pca_view_boundary"))
			self.pca_view_combo.setItemText(1, LOCALIZE("ML_PAGE.pca_view_pca_only"))
			self.pca_view_combo.setToolTip(LOCALIZE("ML_PAGE.pca_view_tooltip"))
		except Exception:
			# Fallback: keep default English labels.
			pass
		self.evaluate_button.setText(LOCALIZE("ML_PAGE.evaluate_dataset"))
		self.status_label.setText(LOCALIZE("ML_PAGE.status_ready"))

		# Split controls labels
		self.train_ratio_label.setText(LOCALIZE("ML_PAGE.train_ratio"))
		self.split_mode_label.setText(LOCALIZE("ML_PAGE.split_mode"))
		self.random_state_label.setText(LOCALIZE("ML_PAGE.random_state"))

		# Model controls labels
		self.model_method_label.setText(LOCALIZE("ML_PAGE.model_method"))

		# Model parameter form labels
		for key, lbl in getattr(self, "_ml_param_labels", {}).items():
			try:
				lbl.setText(LOCALIZE(key))
			except Exception:
				pass
		try:
			self.rf_max_depth.setToolTip(LOCALIZE("ML_PAGE.rf_max_depth_tooltip"))
		except Exception:
			pass

		# Evaluation controls labels
		self.eval_dataset_label.setText(LOCALIZE("ML_PAGE.evaluation_dataset"))
		self.eval_true_label_label.setText(LOCALIZE("ML_PAGE.evaluation_true_label"))
		self.eval_true_label_combo.setItemText(0, LOCALIZE("ML_PAGE.unknown_label"))

		# Split mode display text
		self.split_mode_combo.setItemText(0, LOCALIZE("ML_PAGE.split_by_spectra"))
		self.split_mode_combo.setItemText(1, LOCALIZE("ML_PAGE.split_by_dataset"))

		# Model method display text (robust to optional methods)
		def _set_model_text(model_key: str, text: str) -> None:
			for i in range(self.model_combo.count()):
				if str(self.model_combo.itemData(i) or "") == model_key:
					self.model_combo.setItemText(i, text)
					return

		_set_model_text("linear_regression", LOCALIZE("ML_PAGE.model_linear_regression"))
		_set_model_text("logistic_regression", LOCALIZE("ML_PAGE.model_logistic_regression"))
		_set_model_text("svm", LOCALIZE("ML_PAGE.model_svm"))
		_set_model_text("random_forest", LOCALIZE("ML_PAGE.model_random_forest"))
		if self._xgboost_available:
			_set_model_text("xgboost", LOCALIZE("ML_PAGE.model_xgboost"))

		# SVM parameter combo display text (keep userData stable for sklearn)
		try:
			self.svm_kernel.setItemText(0, LOCALIZE("ML_PAGE.svm_kernel_rbf"))
			self.svm_kernel.setItemText(1, LOCALIZE("ML_PAGE.svm_kernel_linear"))
			self.svm_kernel.setItemText(2, LOCALIZE("ML_PAGE.svm_kernel_poly"))
			self.svm_kernel.setItemText(3, LOCALIZE("ML_PAGE.svm_kernel_sigmoid"))
		except Exception:
			pass
		try:
			self.svm_gamma_mode.setItemText(0, LOCALIZE("ML_PAGE.svm_gamma_scale"))
			self.svm_gamma_mode.setItemText(1, LOCALIZE("ML_PAGE.svm_gamma_auto"))
			self.svm_gamma_mode.setItemText(2, LOCALIZE("ML_PAGE.svm_gamma_custom"))
		except Exception:
			pass

		# Update existing group widgets
		for ui in self._groups:
			ui.name_label.setText(LOCALIZE("ML_PAGE.group_name"))
			ui.name_edit.setPlaceholderText(LOCALIZE("ML_PAGE.group_name_placeholder"))
			ui.include_checkbox.setText(LOCALIZE("ML_PAGE.include_in_training"))
			if ui.remove_button:
				ui.remove_button.setText(LOCALIZE("ML_PAGE.remove_group"))

		# Results tabs
		self.results_tabs.setTabText(0, LOCALIZE("ML_PAGE.results_confusion"))
		self.results_tabs.setTabText(1, LOCALIZE("ML_PAGE.results_report"))
		self.results_tabs.setTabText(2, LOCALIZE("ML_PAGE.results_eval_summary"))
		self.results_tabs.setTabText(3, LOCALIZE("ML_PAGE.results_pca_boundary"))
		self.results_tabs.setTabText(4, LOCALIZE("ML_PAGE.results_roc"))
		self.results_tabs.setTabText(5, LOCALIZE("ML_PAGE.results_importance"))
		self.results_tabs.setTabText(6, LOCALIZE("ML_PAGE.results_distribution"))

		# Classification report table headers
		try:
			self.report_table.setHorizontalHeaderLabels(
				[
					LOCALIZE("ML_PAGE.report_header_class"),
					LOCALIZE("ML_PAGE.report_header_precision"),
					LOCALIZE("ML_PAGE.report_header_recall"),
					LOCALIZE("ML_PAGE.report_header_f1"),
					LOCALIZE("ML_PAGE.report_header_support"),
				]
			)
		except Exception:
			pass

	def load_project_data(self):
		"""Refresh available datasets from the current project context."""
		try:
			self.dataset_source_list.clear() # This works (DatasetSourceList preserved)
			self.eval_dataset_combo.clear()

			if isinstance(RAMAN_DATA, dict) and len(RAMAN_DATA) > 0:
				for name in sorted(RAMAN_DATA.keys()):
					self.dataset_source_list.addItem(str(name))
					self.eval_dataset_combo.addItem(str(name))
			else:
				self.dataset_source_list.addItem(LOCALIZE("ML_PAGE.no_datasets_loaded"))

			# Restore ML groups from project (if any) only when UI has no groups yet.
			# This prevents overwriting in-progress user edits on manual refresh.
			if not self._groups:
				self._load_groups_from_project()
			self._append_log("[DEBUG] Refreshed dataset list")
		except Exception as e:
			create_logs(
				"MachineLearningPage",
				"load_project_data_error",
				f"Failed to load dataset list: {e}",
				status="warning",
			)
			
	def clear_project_data(self):
		"""Clear UI state when leaving/closing a project."""
		self.progress.setValue(0)
		self.status_label.setText(LOCALIZE("ML_PAGE.status_ready"))
		self.log_text.clear()
		try:
			self.eval_summary_text.clear()
		except Exception:
			pass
		self.dataset_source_list.clear()
		self.eval_dataset_combo.clear()
		self._trained_model = None
		self._trained_axis = None
		self._trained_class_labels = []
		self._trained_model_key = None
		self._trained_model_params = {}
		# Do not wipe persisted groups here; user might be switching projects.
		self._clear_groups(add_default=False, persist=False)

	def _on_groups_changed(self):
		"""Central hook for changes that should update persistence/UI.

		Called by GroupDropList when datasets are added/removed via drag/drop or inline delete.
		"""
		if self._suspend_group_persist:
			return
		self._refresh_eval_label_choices()
		self._save_groups_to_project()

	def _append_log(self, msg: str):
		ts = datetime.datetime.now().strftime("%H:%M:%S")
		self.log_text.append(f"[{ts}] {msg}")
		# Keep the log bounded to avoid UI slowdowns / memory growth during long sessions.
		try:
			max_lines = 2000
			doc = self.log_text.document()
			excess = doc.blockCount() - max_lines
			if excess > 0:
				cursor = QTextCursor(doc)
				cursor.movePosition(QTextCursor.Start)
				for _ in range(excess):
					cursor.select(QTextCursor.LineUnderCursor)
					cursor.removeSelectedText()
					cursor.deleteChar()  # remove the newline
		except Exception:
			pass

	def _add_group(self, group_name: Optional[str] = None) -> _GroupUi:
		self._group_counter += 1
		group_id = f"group_{self._group_counter}"
		default_name = group_name or LOCALIZE("ML_PAGE.default_group_name", index=self._group_counter)

		# Tab page widget
		page = QWidget()
		layout = QVBoxLayout(page)
		layout.setContentsMargins(16, 16, 16, 16)
		layout.setSpacing(12)

		name_row = QHBoxLayout()
		name_row.setSpacing(8)
		name_lbl = QLabel(LOCALIZE("ML_PAGE.group_name"))
		name_edit = QLineEdit()
		name_edit.setPlaceholderText(LOCALIZE("ML_PAGE.group_name_placeholder"))
		name_edit.setText(default_name)
		
		name_row.addWidget(name_lbl)
		name_row.addWidget(name_edit, 1)
		layout.addLayout(name_row)

		include_checkbox = QCheckBox(LOCALIZE("ML_PAGE.include_in_training"))
		include_checkbox.setChecked(True)
		include_checkbox.setCursor(Qt.PointingHandCursor)
		include_checkbox.setStyleSheet(
			"""
			QCheckBox {
				font-size: 13px;
				font-weight: 600;
				color: #2c3e50;
				padding: 6px 0;
			}
			QCheckBox::indicator {
				width: 18px;
				height: 18px;
			}
			"""
		)
		include_checkbox.toggled.connect(lambda _checked: self._on_groups_changed())
		layout.addWidget(include_checkbox)

		list_widget = GroupDropList(group_id=group_id, localize_func=LOCALIZE)
		# Style list widget to match clean tab content
		list_widget.setStyleSheet(get_base_style("list_widget") + """
			QListWidget {
				background-color: #ffffff;
				border: 1px solid #ced4da;
			}
		""")
		layout.addWidget(list_widget)

		ui = _GroupUi(
			group_id=group_id,
			container=page, # The tab page is the container
			name_label=name_lbl,
			name_edit=name_edit,
			include_checkbox=include_checkbox,
			list_widget=list_widget,
			remove_button=None,
		)
		
		# Update tab title when name changes
		def update_title(text):
			idx = self.groups_tabs.indexOf(page)
			if idx >= 0:
				self.groups_tabs.setTabText(idx, text or LOCALIZE("ML_PAGE.unnamed_group"))
			self._on_groups_changed()

		name_edit.textChanged.connect(update_title)
		self._groups.append(ui)

		# Add to tabs
		self.groups_tabs.addTab(page, default_name)
		self.groups_tabs.setCurrentWidget(page) # Switch to new tab

		# self._append_log(f"[DEBUG] Added group {group_id}")
		self._refresh_eval_label_choices()
		return ui

	def _on_group_tab_closed(self, index: int):
		widget = self.groups_tabs.widget(index)
		for ui in list(self._groups):
			if ui.container == widget:
				self._remove_group(ui, persist=True)
				break

	def _remove_group(self, ui: _GroupUi, *, persist: bool = True):
		if ui not in self._groups:
			return
		
		idx = self.groups_tabs.indexOf(ui.container)
		if idx >= 0:
			self.groups_tabs.removeTab(idx)
			
		self._groups.remove(ui)
		ui.container.deleteLater()
		# self._append_log(f"[DEBUG] Removed group {ui.group_id}")
		if persist:
			self._on_groups_changed()
		else:
			self._refresh_eval_label_choices()

	def _clear_groups(self, *, add_default: bool = False, persist: bool = True):
		for ui in list(self._groups):
			self._remove_group(ui, persist=persist)
			
		self._group_counter = 0 
		if add_default:
			# Add default 2
			self._add_group()
			self._add_group()
		if persist:
			self._on_groups_changed()

	def _find_group_ui_by_id(self, group_id: str) -> Optional[_GroupUi]:
		for ui in self._groups:
			if ui.group_id == group_id:
				return ui
		return None

	def _remove_dataset_from_group(self, ui: _GroupUi, dataset_name: str):
		"""Remove a dataset item from a group's list widget by name."""
		dataset_name = str(dataset_name or "").strip()
		if not dataset_name:
			return
		for i in range(ui.list_widget.count() - 1, -1, -1):
			item = ui.list_widget.item(i)
			name = (item.data(Qt.UserRole) or item.text() or "").strip()
			if name == dataset_name:
				ui.list_widget.takeItem(i)
				break
		self._on_groups_changed()

	def _open_group_creation_dialog(self):
		# Gather dataset names from current project context
		available = []
		if isinstance(RAMAN_DATA, dict) and len(RAMAN_DATA) > 0:
			available = sorted([str(k) for k in RAMAN_DATA.keys()])
		else:
			QMessageBox.warning(self, LOCALIZE("COMMON.warning"), LOCALIZE("ML_PAGE.no_datasets_loaded"))
			return

		# Seed dialog: prefer saved configs (so user doesn't have to re-enter keywords).
		saved_cfg = PROJECT_MANAGER.get_ml_group_configs() if PROJECT_MANAGER.current_project_data else []
		# If no groups exist yet, show 2 default group rows.
		if self._groups:
			seed_groups = [
				{"name": (ui.name_edit.text() or "").strip(), "include": "", "exclude": "", "auto_assign": False}
				for ui in self._groups
			]
			seed_groups = [g for g in seed_groups if g.get("name")]
			# Overlay saved keyword settings when names match
			if saved_cfg:
				by_name = {str(c.get("name") or "").strip(): c for c in saved_cfg if isinstance(c, dict)}
				for g in seed_groups:
					c = by_name.get(str(g.get("name") or "").strip())
					if c:
						# Keep list[str] for the dialog; it will render as comma-separated text.
						g["include"] = c.get("include") or []
						g["exclude"] = c.get("exclude") or []
						g["auto_assign"] = bool(c.get("auto_assign", False))
		else:
			seed_groups = saved_cfg if saved_cfg else [
				{"name": "Class 1", "include": "", "exclude": "", "auto_assign": False},
				{"name": "Class 2", "include": "", "exclude": "", "auto_assign": False},
			]

		dialog = MultiGroupCreationDialog(
			available,
			LOCALIZE,
			self,
			initial_rows=2,
			default_auto_assign=False,
			initial_groups=seed_groups,
		)
		result = dialog.exec()
		if result != QDialog.Accepted:
			return

		configs = dialog.get_group_configs() or []
		assignments = dialog.get_assignments() or {}
		if not configs:
			return

		# If there are existing groups, confirm replacement.
		if self._groups:
			reply = QMessageBox.question(
				self,
				LOCALIZE("COMMON.confirm"),
				LOCALIZE("ML_PAGE.auto_assign_replace_groups_confirm"),
				QMessageBox.Yes | QMessageBox.No,
			)
			if reply != QMessageBox.Yes:
				return

		# Replace existing groups
		self._suspend_group_persist = True
		self._clear_groups(add_default=False, persist=False)
		for cfg in configs:
			group_name = str(cfg.get("name") or "").strip()
			if not group_name:
				continue
			ui = self._add_group(group_name=group_name)
			# Populate datasets if auto-assign produced results for this group
			for ds in assignments.get(group_name, []) or []:
				name = str(ds).strip()
				if not name:
					continue
				ui.list_widget.add_dataset(name)
		self._refresh_eval_label_choices()
		self._suspend_group_persist = False

		# Persist group definitions and assignments to project JSON
		try:
			PROJECT_MANAGER.set_ml_group_configs(configs)
			self._on_groups_changed()
		except Exception as e:
			self._append_log(f"[DEBUG] Failed to save ML groups to project: {e}")

	def _save_groups_to_project(self):
		"""Persist current group assignments into the project JSON."""
		if self._suspend_group_persist:
			return
		if not PROJECT_MANAGER.current_project_data:
			return
		groups: Dict[str, List[str]] = {}
		enabled: Dict[str, bool] = {}
		for ui in self._groups:
			name = (ui.name_edit.text() or "").strip()
			if not name:
				continue
			ds_list: List[str] = []
			for i in range(ui.list_widget.count()):
				item = ui.list_widget.item(i)
				ds_name = (item.data(Qt.UserRole) or item.text() or "").strip()
				if ds_name:
					ds_list.append(ds_name)
			groups[name] = ds_list
			enabled[name] = bool(ui.include_checkbox.isChecked())
		# Save groups + enabled flags in one project write
		PROJECT_MANAGER.set_ml_groups_and_enabled(groups, enabled, origin="ml")

	def _load_groups_from_project(self, *, persist_reconcile: bool = True):
		"""Load group assignments from the project JSON and apply to the UI."""
		if not PROJECT_MANAGER.current_project_data:
			return
		saved = PROJECT_MANAGER.get_ml_groups() or {}
		if not saved:
			return
		enabled_map = PROJECT_MANAGER.get_ml_group_enabled_map() if hasattr(PROJECT_MANAGER, "get_ml_group_enabled_map") else {}
		# Reconcile with available datasets.
		# IMPORTANT: On project-open, RAMAN_DATA may not be populated yet; reconciling against it
		# would incorrectly drop everything and overwrite the saved groups with empty lists.
		available: set[str] = set()
		try:
			data_packages = PROJECT_MANAGER.current_project_data.get("dataPackages", {})
			if isinstance(data_packages, dict) and data_packages:
				available = set(map(str, data_packages.keys()))
		except Exception:
			available = set()
		if not available and isinstance(RAMAN_DATA, dict) and RAMAN_DATA:
			available = set(map(str, RAMAN_DATA.keys()))
		clean: Dict[str, List[str]] = {}
		for gname, ds_list in saved.items():
			if not isinstance(ds_list, list):
				ds_list = []
			if available:
				filtered = [str(ds) for ds in ds_list if str(ds) in available]
			else:
				# No reliable availability signal yet; keep as-is and let ProjectManager reconcile later.
				filtered = [str(ds) for ds in ds_list]
			clean[str(gname)] = filtered

		# Replace current UI groups with saved ones (without persisting empties mid-rebuild)
		self._suspend_group_persist = True
		self._clear_groups(add_default=False, persist=False)
		for gname, ds_list in clean.items():
			ui = self._add_group(group_name=str(gname))
			try:
				ui.include_checkbox.setChecked(bool(enabled_map.get(str(gname), True)))
			except Exception:
				pass
			for ds in ds_list:
				ui.list_widget.add_dataset(str(ds))
		self._refresh_eval_label_choices()
		self._suspend_group_persist = False
		# Persist only if we actually changed something (and had a meaningful basis to reconcile).
		# NOTE: Skip persistence during external refresh to prevent feedback loops.
		if persist_reconcile:
			try:
				if clean != saved and available:
					PROJECT_MANAGER.set_ml_groups_and_enabled(clean, enabled_map, origin="ml")
			except Exception:
				pass

	def _on_external_groups_changed(self, origin: Optional[str] = None) -> None:
		"""Refresh groups when another page updates the shared group map."""
		if origin == "ml":
			return
		# Defer to the event loop so we don't rebuild tabs mid-drag.
		try:
			QTimer.singleShot(0, lambda: self._load_groups_from_project(persist_reconcile=False))
		except Exception:
			self._load_groups_from_project(persist_reconcile=False)

	def _build_group_assignments(self) -> Dict[str, str]:
		assignments: Dict[str, str] = {}
		for ui in self._groups:
			if not bool(ui.include_checkbox.isChecked()):
				continue
			label = (ui.name_edit.text() or "").strip()
			if not label:
				continue
			for i in range(ui.list_widget.count()):
				item = ui.list_widget.item(i)
				ds_name = (item.data(Qt.UserRole) or item.text() or "").strip()
				if ds_name:
					assignments[ds_name] = label
		return assignments

	def _collect_model_params(self) -> Dict:
		key = self.model_combo.currentData() or self.model_combo.currentText()
		if key == "linear_regression":
			return {
				"fit_intercept": bool(self.lin_fit_intercept.isChecked()),
				"positive": bool(self.lin_positive.isChecked()),
			}
		if key == "logistic_regression":
			return {
				"C": float(self.lr_C.value()),
				"max_iter": int(self.lr_max_iter.value()),
				"solver": str(self.lr_solver.currentText()),
			}
		if key == "svm":
			gamma_mode = str(self.svm_gamma_mode.currentData() or self.svm_gamma_mode.currentText())
			gamma = gamma_mode
			if gamma_mode == "custom":
				gamma = float(self.svm_gamma_value.value())
			return {
				"kernel": str(self.svm_kernel.currentData() or self.svm_kernel.currentText()),
				"C": float(self.svm_C.value()),
				"gamma": gamma,
				"degree": int(self.svm_degree.value()),
				"probability": True,
			}
		if key == "random_forest":
			max_depth = int(self.rf_max_depth.value())
			return {
				"n_estimators": int(self.rf_n_estimators.value()),
				"max_depth": None if max_depth <= 0 else max_depth,
				"random_state": int(self.rf_random_state.value()),
			}
		if key == "xgboost":
			# NOTE: XGBoost UI is optional; guard against missing widgets.
			if not hasattr(self, "xgb_n_estimators"):
				return {}
			return {
				"n_estimators": int(self.xgb_n_estimators.value()),
				"max_depth": int(self.xgb_max_depth.value()),
				"learning_rate": float(self.xgb_learning_rate.value()),
				"subsample": float(self.xgb_subsample.value()),
				"colsample_bytree": float(self.xgb_colsample_bytree.value()),
				"reg_lambda": float(self.xgb_reg_lambda.value()),
			}
		return {}

	def _model_display_name(self, model_key: str | None) -> str:
		key = str(model_key or "").strip()
		if not key:
			return LOCALIZE("COMMON.unknown")
		# Prefer localized labels when available
		mapping = {
			"linear_regression": LOCALIZE("ML_PAGE.model_linear_regression"),
			"logistic_regression": LOCALIZE("ML_PAGE.model_logistic_regression"),
			"svm": LOCALIZE("ML_PAGE.model_svm"),
			"random_forest": LOCALIZE("ML_PAGE.model_random_forest"),
			"xgboost": LOCALIZE("ML_PAGE.model_xgboost"),
		}
		return mapping.get(key, key.replace("_", " ").title())

	def _model_display_name_en(self, model_key: str | None) -> str:
		"""English model name for plot text.

		We intentionally keep Matplotlib result plots English-only to avoid glyph issues
		when the UI language is Japanese.
		"""
		key = str(model_key or "").strip()
		if not key:
			return "Unknown"
		mapping = {
			"linear_regression": "Linear Regression",
			"logistic_regression": "Logistic Regression",
			"svm": "SVM",
			"random_forest": "Random Forest",
			"xgboost": "XGBoost",
		}
		return mapping.get(key, key.replace("_", " ").title())

	def _build_plot_label_display_map(self, class_labels: list) -> Dict[str, str]:
		"""Build a stable display-label map for ML plots.

		We keep training/evaluation labels as-is (can be Japanese), but for the ML Results
		plots we prefer the original labels when they are ASCII-safe. Otherwise, we fall
		back to English-only labels (Class 1, Class 2, ...) to avoid Matplotlib glyph
		issues when the UI language is Japanese.
		"""
		labels = [str(l) for l in (class_labels or [])]
		out: Dict[str, str] = {}
		for i, lab in enumerate(labels):
			# ASCII-safe labels are OK to show directly.
			try:
				lab.encode("ascii")
			except Exception:
				out[lab] = f"Class {i + 1}"
			else:
				out[lab] = lab
		return out

	def _current_pca_view_mode(self) -> str:
		"""Return the current PCA view mode.

		Returns:
			"boundary" (default) or "pca_only".
		"""
		btn = getattr(self, "pca_view_combo", None)
		if btn is None:
			return "boundary"
		try:
			val = str(btn.currentData() or btn.currentText()).strip()
		except Exception:
			return "boundary"
		return val or "boundary"

	def _render_pca_plot(self):
		"""Re-render the PCA plot from the last training output without retraining."""
		if not getattr(self, "_last_training_output", None):
			return
		if not hasattr(self, "pca_plot"):
			return

		try:
			out = self._last_training_output
			label_display = self._build_plot_label_display_map(list(getattr(out.split, "class_labels", []) or []))
			model_name_en = self._model_display_name_en(getattr(out, "model_key", None))
			model_key = str(getattr(out, "model_key", "") or "")
			mode = self._current_pca_view_mode()

			if mode == "pca_only":
				title = f"PCA (2D) (Model: {model_name_en})"
				fig = create_pca_scatter_figure(
					X=np.asarray(out.split.X_train, dtype=float),
					y=np.asarray(out.split.y_train, dtype=object),
					title=title,
					label_display_map=label_display,
				)
			else:
				# Note: decision boundary is computed by a visualization classifier in PCA space.
				# For unsupported model_key values (e.g., xgboost), the visualization falls back
				# to logistic regression. Avoid claiming the boundary corresponds to the true model.
				if model_key in {"logistic_regression", "svm", "random_forest"}:
					title = f"PCA + Decision Boundary (Visualization: {model_name_en})"
				else:
					title = "PCA + Decision Boundary (Visualization)"
				fig = create_pca_decision_boundary_figure(
					X=np.asarray(out.split.X_train, dtype=float),
					y=np.asarray(out.split.y_train, dtype=object),
					model_key=model_key,
					model_params=dict(getattr(out, "model_params", {}) or {}),
					title=title,
					label_display_map=label_display,
				)

			# Update the Matplotlib widget.
			if hasattr(self.pca_plot, "update_plot_with_config"):
				# Background is image-based; don't add colorbar (it tends to clutter this view).
				self.pca_plot.update_plot_with_config(
					fig,
					{
						"colorbar": False,
						"subplots_adjust": {"left": 0.08, "right": 0.985, "bottom": 0.12, "top": 0.90},
						"figure": {
							"tight_layout": False,
							"constrained_layout": False,
							"tight_layout_on_resize": False,
						},
					},
				)
			else:
				self.pca_plot.figure.clear()
				ax = self.pca_plot.figure.add_subplot(111)
				# Best effort: draw the figure into existing canvas.
				try:
					ax.remove()
				except Exception:
					pass
				self.pca_plot.figure = fig
				self.pca_plot.canvas.draw()
		except Exception as e:
			self._append_log(f"[DEBUG] PCA plot render failed: {e}")

	def _set_controls_enabled(self, enabled: bool, *, keep_training_button_enabled: bool = False):
		# During training we keep the training button enabled so it can act as "Stop".
		if not keep_training_button_enabled:
			self.start_training_button.setEnabled(enabled)
		self.add_group_button.setEnabled(enabled)
		self.clear_groups_button.setEnabled(enabled)
		self.datasets_refresh_button.setEnabled(enabled)
		self.model_combo.setEnabled(enabled)
		self.model_params_stack.setEnabled(enabled)
		self.train_ratio_spin.setEnabled(enabled)
		self.split_mode_combo.setEnabled(enabled)
		self.random_state_spin.setEnabled(enabled)
		self.save_model_button.setEnabled(enabled)
		self.load_model_button.setEnabled(enabled)
		self.export_report_button.setEnabled(enabled)
		self.shap_button.setEnabled(enabled)
		self.pca_view_combo.setEnabled(enabled)
		self.evaluate_button.setEnabled(enabled)
		self.eval_dataset_combo.setEnabled(enabled)
		self.eval_true_label_combo.setEnabled(enabled)

	def _set_training_button_state(self, running: bool) -> None:
		"""Toggle Start/Stop training button UI."""
		try:
			if running:
				self.start_training_button.setText(LOCALIZE("ML_PAGE.stop_training"))
				# Use danger styling so it reads as an interrupt action.
				self.start_training_button.setStyleSheet(
					get_base_style("danger_button")
					+ "QPushButton { font-size: 14px; font-weight: 600; }"
				)
			else:
				self.start_training_button.setText(LOCALIZE("ML_PAGE.start_training"))
				self.start_training_button.setStyleSheet(
					get_base_style("primary_button")
					+ "QPushButton { font-size: 14px; font-weight: 600; }"
				)
			self.start_training_button.setEnabled(True)
		except Exception:
			pass

	def _stop_training(self) -> None:
		"""Best-effort cancellation of in-flight training.

		We do a cooperative cancel first; if the worker doesn't stop quickly (e.g., stuck
		inside a long sklearn fit), we fall back to a hard QThread termination.
		"""
		t = self._training_thread
		if t is None or (not t.isRunning()):
			return

		self._append_log("[DEBUG] Stop training requested")
		self.status_label.setText(LOCALIZE("ML_PAGE.status_stopping"))
		# Detach UI from worker signals to avoid late updates after stop.
		try:
			t.progress_updated.disconnect()
		except Exception:
			pass
		try:
			t.status_updated.disconnect()
		except Exception:
			pass
		try:
			t.training_error.disconnect()
		except Exception:
			pass
		try:
			t.training_completed.disconnect()
		except Exception:
			pass

		try:
			t.requestInterruption()
		except Exception:
			pass

		# Give the worker a brief chance to exit cleanly.
		try:
			t.wait(200)
		except Exception:
			pass
		if t.isRunning():
			try:
				t.terminate()
				t.wait(1000)
			except Exception:
				pass

		self._training_thread = None
		self.progress.setValue(0)
		self.status_label.setText(LOCALIZE("ML_PAGE.status_cancelled"))
		self._set_training_button_state(False)
		self._set_controls_enabled(True)

	def _start_training(self):
		# Toggle: while running, this button becomes "Stop training".
		if self._training_thread and self._training_thread.isRunning():
			self._stop_training()
			return

		assignments = self._build_group_assignments()
		class_labels = sorted(set(assignments.values()))
		if len(class_labels) < 2:
			QMessageBox.warning(self, LOCALIZE("COMMON.warning"), LOCALIZE("ML_PAGE.error_no_group_data"))
			return

		model_key = self.model_combo.currentData() or self.model_combo.currentText()
		model_params = self._collect_model_params()
		train_ratio = float(self.train_ratio_spin.value())
		split_mode = self.split_mode_combo.currentData() or (
			"by_spectra" if self.split_mode_combo.currentIndex() == 0 else "by_dataset"
		)
		random_state = int(self.random_state_spin.value())
		# Capture context for later export / reproducibility
		try:
			group_names = sorted({(ui.name_edit.text() or "").strip() for ui in self._groups if (ui.name_edit.text() or "").strip()})
		except Exception:
			group_names = []
		self._last_train_context = {
			"model_key": str(model_key),
			"model_params": dict(model_params),
			"group_assignments": dict(assignments),
			"group_names": group_names,
			"class_labels": list(class_labels),
			"train_ratio": float(train_ratio),
			"split_mode": str(split_mode),
			"random_state": int(random_state),
			"started_at": datetime.datetime.now().isoformat(),
		}

		self._append_log(
			f"[DEBUG] Start training: model={model_key} classes={class_labels} train_ratio={train_ratio} split_mode={split_mode}"
		)
		self.progress.setValue(0)
		self.status_label.setText(LOCALIZE("ML_PAGE.status_training"))
		self._set_training_button_state(True)
		self._set_controls_enabled(False, keep_training_button_enabled=True)

		self._training_thread = MLTrainingThread(
			raman_data=RAMAN_DATA,
			group_assignments=assignments,
			train_ratio=train_ratio,
			split_mode=split_mode,
			random_state=random_state,
			model_key=model_key,
			model_params=model_params,
			parent=self,
		)
		self._training_thread.progress_updated.connect(self.progress.setValue)
		self._training_thread.status_updated.connect(self.status_label.setText)
		self._training_thread.training_error.connect(self._on_training_error)
		self._training_thread.training_completed.connect(self._on_training_completed)
		self._training_thread.start()

	def _on_training_error(self, msg: str):
		self._append_log(f"[DEBUG] Training error: {msg}")
		self.status_label.setText(LOCALIZE("COMMON.error"))
		self.progress.setValue(0)
		self._set_training_button_state(False)
		self._set_controls_enabled(True)
		QMessageBox.critical(self, LOCALIZE("COMMON.error"), msg)

	def _bootstrap_ci(
		self,
		*,
		y_true: np.ndarray,
		y_pred: np.ndarray,
		metric: str,
		labels: List[str],
		n_boot: int = 300,
		alpha: float = 0.05,
		random_state: int = 0,
	) -> Optional[Tuple[float, float]]:
		"""Bootstrap CI over test samples (spectra-level).

		Returns:
			(lo, hi) or None if not computable.
		"""
		y_true = np.asarray(y_true, dtype=object)
		y_pred = np.asarray(y_pred, dtype=object)
		if y_true.shape[0] == 0 or y_pred.shape[0] != y_true.shape[0]:
			return None

		metric_key = str(metric or "").strip().lower()
		if metric_key not in {"accuracy", "balanced_accuracy", "macro_f1"}:
			return None

		rng = np.random.RandomState(int(random_state))
		idx = np.arange(y_true.shape[0])
		vals: List[float] = []
		for _ in range(int(n_boot)):
			s = rng.choice(idx, size=idx.shape[0], replace=True)
			yt = y_true[s]
			yp = y_pred[s]
			try:
				if metric_key == "accuracy":
					v = float(accuracy_score(yt, yp))
				elif metric_key == "balanced_accuracy":
					v = float(balanced_accuracy_score(yt, yp))
				else:
					v = float(f1_score(yt, yp, labels=list(labels), average="macro", zero_division=0))
				vals.append(v)
			except Exception:
				continue

		if len(vals) < max(20, int(n_boot) // 5):
			return None
		arr = np.asarray(vals, dtype=float)
		lo = float(np.quantile(arr, alpha / 2.0))
		hi = float(np.quantile(arr, 1.0 - alpha / 2.0))
		return lo, hi

	def _aggregate_dataset_majority_vote(
		self,
		*,
		y_true: np.ndarray,
		y_pred: np.ndarray,
		ds_names: np.ndarray,
	) -> Tuple[np.ndarray, np.ndarray]:
		"""Aggregate spectrum predictions to dataset-level via majority vote."""
		y_true = np.asarray(y_true, dtype=object)
		y_pred = np.asarray(y_pred, dtype=object)
		ds_names = np.asarray(ds_names, dtype=object)
		if y_true.shape[0] == 0:
			return np.asarray([], dtype=object), np.asarray([], dtype=object)
		if not (y_true.shape[0] == y_pred.shape[0] == ds_names.shape[0]):
			return np.asarray([], dtype=object), np.asarray([], dtype=object)

		def _majority(arr: np.ndarray) -> object:
			arr = np.asarray(arr, dtype=object)
			vals, counts = np.unique(arr, return_counts=True)
			if vals.shape[0] == 0:
				return ""
			return vals[int(np.argmax(counts))]

		uniq_ds = sorted({str(x) for x in ds_names.tolist() if str(x)})
		y_true_ds: List[object] = []
		y_pred_ds: List[object] = []
		for ds in uniq_ds:
			mask = np.asarray([str(x) == ds for x in ds_names], dtype=bool)
			if not np.any(mask):
				continue
			y_true_ds.append(_majority(y_true[mask]))
			y_pred_ds.append(_majority(y_pred[mask]))
		return np.asarray(y_true_ds, dtype=object), np.asarray(y_pred_ds, dtype=object)

	def _build_eval_summary_text(self, out: MLTrainingOutput) -> str:
		"""Build a research-friendly evaluation summary (human readable).

		Focus:
		- Split design and leakage risks
		- Spectrum-level metrics (what we currently optimize)
		- Dataset-level metrics (closer to "patient-level" when multiple spectra per dataset)
		- Simple uncertainty via bootstrap CI (spectra-level)
		"""
		labels = list(out.split.class_labels or [])
		split_mode = str((self._last_train_context or {}).get("split_mode") or "")
		train_ratio = (self._last_train_context or {}).get("train_ratio")
		random_state = int((self._last_train_context or {}).get("random_state") or 0)

		y_true = np.asarray(out.split.y_test, dtype=object)
		y_pred = np.asarray(out.y_pred, dtype=object)
		ds_train = np.asarray(out.split.sample_dataset_names_train, dtype=object)
		ds_test = np.asarray(out.split.sample_dataset_names_test, dtype=object)

		train_ds = {str(x) for x in ds_train.tolist() if str(x)}
		test_ds = {str(x) for x in ds_test.tolist() if str(x)}
		overlap = sorted(train_ds.intersection(test_ds))

		lines: List[str] = []
		lines.append(f"{LOCALIZE('ML_PAGE.eval_summary_model')}: {self._model_display_name(out.model_key)}")
		if split_mode:
			tr = f"{float(train_ratio):.2f}" if isinstance(train_ratio, (int, float)) else str(train_ratio)
			lines.append(f"{LOCALIZE('ML_PAGE.eval_summary_split')}: {split_mode} (train_ratio={tr}, seed={random_state})")
		lines.append(
			f"{LOCALIZE('ML_PAGE.eval_summary_counts')}: "
			f"train={int(np.asarray(out.split.y_train).shape[0])} spectra / {len(train_ds)} datasets, "
			f"test={int(y_true.shape[0])} spectra / {len(test_ds)} datasets"
		)
		if labels:
			lines.append(f"{LOCALIZE('ML_PAGE.eval_summary_classes')}: {', '.join(map(str, labels))}")

		if split_mode == "by_spectra":
			lines.append("")
			lines.append(f"{LOCALIZE('ML_PAGE.eval_summary_warning')}: {LOCALIZE('ML_PAGE.eval_summary_warning_by_spectra')}")
		if overlap:
			lines.append("")
			lines.append(
				f"{LOCALIZE('ML_PAGE.eval_summary_warning')}: "
				+ LOCALIZE("ML_PAGE.eval_summary_warning_dataset_overlap").format(count=len(overlap))
			)

		# ---- Spectrum-level metrics ----
		lines.append("")
		lines.append(f"[{LOCALIZE('ML_PAGE.eval_summary_spectrum_level')}]" )
		try:
			acc = float(accuracy_score(y_true, y_pred))
		except Exception:
			acc = float("nan")
		try:
			bal = float(balanced_accuracy_score(y_true, y_pred))
		except Exception:
			bal = float("nan")
		try:
			macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
			micro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0))
		except Exception:
			macro_f1 = float("nan")
			micro_f1 = float("nan")

		lines.append(f"accuracy={acc:.4f}  balanced_accuracy={bal:.4f}  macro_f1={macro_f1:.4f}  micro_f1={micro_f1:.4f}")

		ci_acc = self._bootstrap_ci(
			y_true=y_true,
			y_pred=y_pred,
			metric="accuracy",
			labels=labels,
			n_boot=300,
			random_state=random_state,
		)
		ci_bal = self._bootstrap_ci(
			y_true=y_true,
			y_pred=y_pred,
			metric="balanced_accuracy",
			labels=labels,
			n_boot=300,
			random_state=random_state,
		)
		ci_f1 = self._bootstrap_ci(
			y_true=y_true,
			y_pred=y_pred,
			metric="macro_f1",
			labels=labels,
			n_boot=300,
			random_state=random_state,
		)
		ci_parts: List[str] = []
		if ci_acc:
			ci_parts.append(f"accuracy_CI95=[{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]")
		if ci_bal:
			ci_parts.append(f"balanced_accuracy_CI95=[{ci_bal[0]:.4f}, {ci_bal[1]:.4f}]")
		if ci_f1:
			ci_parts.append(f"macro_f1_CI95=[{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]")
		if ci_parts:
			lines.append("  " + "  ".join(ci_parts))
			lines.append(f"  {LOCALIZE('ML_PAGE.eval_summary_ci_note')}")

		# Class-wise sensitivity/specificity (OvR)
		try:
			cm = confusion_matrix(y_true, y_pred, labels=labels)
			cm = np.asarray(cm, dtype=float)
			total = float(np.sum(cm))
			lines.append("")
			lines.append(f"{LOCALIZE('ML_PAGE.eval_summary_per_class')}:" )
			for i, lab in enumerate(labels):
				tp = cm[i, i]
				fn = float(np.sum(cm[i, :]) - tp)
				fp = float(np.sum(cm[:, i]) - tp)
				tn = total - tp - fn - fp
				sens = (tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
				spec = (tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
				lines.append(f"  {lab}: sensitivity(recall)={sens:.4f}  specificity={spec:.4f}")
		except Exception:
			pass

		# Probabilistic metrics (if available)
		if out.proba is not None:
			proba = np.asarray(out.proba, dtype=float)
			if proba.ndim == 2 and proba.shape[0] == y_true.shape[0] and labels:
				try:
					if len(labels) == 2:
						pos_idx = 1
						pos_label = labels[pos_idx]
						y_bin = np.asarray([1 if str(v) == str(pos_label) else 0 for v in y_true], dtype=int)
						score = proba[:, pos_idx]
						auc = float(roc_auc_score(y_bin, score))
						brier = float(brier_score_loss(y_bin, score))
						lines.append("")
						lines.append(f"auc(ROC)={auc:.4f}  brier={brier:.4f}")
					else:
						auc = float(
							roc_auc_score(
								y_true,
								proba,
								labels=labels,
								multi_class="ovr",
								average="macro",
							)
						)
						lines.append("")
						lines.append(f"auc_ovr_macro={auc:.4f}")
				except Exception:
					pass

		# ---- Dataset-level metrics (majority vote) ----
		y_true_ds, y_pred_ds = self._aggregate_dataset_majority_vote(
			y_true=y_true,
			y_pred=y_pred,
			ds_names=ds_test,
		)
		if y_true_ds.shape[0] > 0:
			lines.append("")
			lines.append(f"[{LOCALIZE('ML_PAGE.eval_summary_dataset_level')}]" )
			try:
				acc_ds = float(accuracy_score(y_true_ds, y_pred_ds))
			except Exception:
				acc_ds = float("nan")
			try:
				bal_ds = float(balanced_accuracy_score(y_true_ds, y_pred_ds))
			except Exception:
				bal_ds = float("nan")
			try:
				macro_f1_ds = float(f1_score(y_true_ds, y_pred_ds, labels=labels, average="macro", zero_division=0))
			except Exception:
				macro_f1_ds = float("nan")
			lines.append(
				f"datasets={int(y_true_ds.shape[0])}  "
				f"accuracy={acc_ds:.4f}  balanced_accuracy={bal_ds:.4f}  macro_f1={macro_f1_ds:.4f}"
			)
			lines.append(f"  {LOCALIZE('ML_PAGE.eval_summary_dataset_note')}")

		return "\n".join(lines).strip() + "\n"

	def _on_training_completed(self, out: MLTrainingOutput):
		# Training completed successfully; restore Start/Stop button state.
		self._set_training_button_state(False)
		self._trained_model = out.model
		self._trained_axis = np.asarray(out.split.common_axis, dtype=float)
		self._trained_class_labels = list(out.split.class_labels)
		self._trained_model_key = out.model_key
		self._trained_model_params = dict(out.model_params)
		# Cache full training output so plots can be re-rendered without retraining.
		self._last_training_output = out
		try:
			self._last_train_context["completed_at"] = datetime.datetime.now().isoformat()
		except Exception:
			pass
		self._refresh_eval_label_choices()

		self._append_log(f"[DEBUG] Accuracy: {out.accuracy:.4f}")
		# Classification report table
		try:
			rows = create_classification_report_table(
				y_true=out.split.y_test,
				y_pred=out.y_pred,
				labels=list(out.split.class_labels),
			)
			self._populate_report_table(rows)
		except Exception as e:
			self._append_log(f"[DEBUG] Report table failed: {e}")

		# Evaluation summary (metrics + leakage warnings)
		try:
			self.eval_summary_text.setPlainText(self._build_eval_summary_text(out))
		except Exception as e:
			self._append_log(f"[DEBUG] Eval summary build failed: {e}")

		# Confusion matrix
		try:
			label_display_map = self._build_plot_label_display_map(list(out.split.class_labels))
			cm_title = f"Confusion Matrix (Model: {self._model_display_name_en(out.model_key)})"
			cm_fig = create_confusion_matrix_figure(
				y_true=out.split.y_test,
				y_pred=out.y_pred,
				class_labels=out.split.class_labels,
				title=cm_title,
				label_display_map=label_display_map,
			)
			self.cm_plot.update_plot_with_config(cm_fig, {"colorbar": True})
		except Exception as e:
			self._append_log(f"[DEBUG] Confusion matrix plot failed: {e}")

		# ROC
		try:
			if out.proba is not None and len(out.split.class_labels) >= 2:
				roc_fig = create_roc_curve_figure(
					y_true=out.split.y_test,
					y_score=out.proba,
					class_labels=out.split.class_labels,
					title="ROC Curve",
					label_display_map=label_display_map,
				)
				self.roc_plot.update_plot_with_config(roc_fig)
			else:
				self.roc_plot.figure.clear()
				self.roc_plot.canvas.draw()
		except Exception as e:
			self._append_log(f"[DEBUG] ROC plot failed: {e}")

		# Feature importance
		try:
			if out.feature_importances is not None:
				fi_fig = create_feature_importance_figure(
					feature_importances=out.feature_importances,
					wavenumbers=out.split.common_axis,
					title="Feature Importance",
				)
				self.fi_plot.update_plot_with_config(fi_fig)
			else:
				self.fi_plot.figure.clear()
				self.fi_plot.canvas.draw()
		except Exception as e:
			self._append_log(f"[DEBUG] Feature importance plot failed: {e}")

		# Prediction distribution
		try:
			if out.proba is not None:
				pd_fig = create_prediction_distribution_figure(
					y_true=out.split.y_test,
					y_score=out.proba,
					class_labels=out.split.class_labels,
					title="Prediction Distribution",
					label_display_map=label_display_map,
				)
				self.pd_plot.update_plot_with_config(pd_fig)
			else:
				self.pd_plot.figure.clear()
				self.pd_plot.canvas.draw()
		except Exception as e:
			self._append_log(f"[DEBUG] Prediction distribution plot failed: {e}")

		# PCA view (boundary vs PCA-only) - visualization-only, no retraining
		try:
			self._render_pca_plot()
		except Exception as e:
			self._append_log(f"[DEBUG] PCA plot failed: {e}")

		self.status_label.setText(LOCALIZE("ML_PAGE.status_done"))
		self._set_controls_enabled(True)
		# Enable SHAP once a model exists.
		try:
			self.shap_button.setEnabled(True)
		except Exception:
			pass

	def _run_shap(self):
		"""Compute SHAP per-spectrum explanation using the already-trained model."""
		out = getattr(self, "_last_training_output", None)
		if out is None:
			QMessageBox.warning(self, LOCALIZE("COMMON.warning"), LOCALIZE("ML_PAGE.error_no_model"))
			return

		# Build explain candidates from all samples that participated in training (train+test).
		try:
			x_train = np.asarray(out.split.X_train, dtype=float)
			x_test = np.asarray(out.split.X_test, dtype=float)
			y_train = np.asarray(out.split.y_train, dtype=object)
			y_test = np.asarray(out.split.y_test, dtype=object)

			feature_axis = np.asarray(out.split.common_axis, dtype=float)

			raw_train_names = getattr(out.split, "sample_dataset_names_train", None)
			raw_test_names = getattr(out.split, "sample_dataset_names_test", None)
			train_names = [] if raw_train_names is None else list(raw_train_names)
			test_names = [] if raw_test_names is None else list(raw_test_names)

			# Combine train + test to allow selecting a dataset and spectrum directly.
			x_explain = x_train if x_test.size == 0 else np.concatenate([x_train, x_test], axis=0)
			y_explain = y_train if y_test.size == 0 else np.concatenate([y_train, y_test], axis=0)
			dataset_names = list(train_names) + list(test_names)
		except Exception as e:
			QMessageBox.critical(self, LOCALIZE("COMMON.error"), str(e))
			return

		# Parameter dialog (dataset + spectrum index + preview + metadata)
		default_seed = 0
		try:
			default_seed = int(self._last_train_context.get("random_state") or 0)
		except Exception:
			default_seed = 0
		if default_seed == 0:
			try:
				default_seed = int(getattr(self, "random_state_spin", None).value())
			except Exception:
				default_seed = 0

		try:
			from pages.machine_learning_page_utils.shap_parameter_dialog import SHAPParameterDialog
		except Exception as e:
			QMessageBox.critical(self, LOCALIZE("COMMON.error"), str(e))
			return

		dlg = SHAPParameterDialog(
			source="all",
			x_explain=x_explain,
			y_explain=y_explain,
			dataset_names=dataset_names,
			feature_axis=feature_axis,
			default_random_state=int(default_seed),
			parent=self,
		)
		if dlg.exec() != QDialog.Accepted:
			return
		sel = dlg.selection()

		idx = int(sel.global_index)
		self._append_log(
			f"[DEBUG] SHAP settings: source={sel.source} dataset={sel.dataset_name} within={sel.index_within_dataset} global={idx} bg={sel.background_samples} max_evals={sel.max_evals} top_k={sel.top_k} seed={sel.random_state}"
		)

		try:
			from pages.machine_learning_page_utils.shap_explain_dialog import SHAPExplainDialog
		except Exception as e:
			QMessageBox.critical(self, LOCALIZE("COMMON.error"), str(e))
			return

		default_dir = self._default_report_export_dir()
		base = self._default_model_basename()
		dataset_part = _sanitize_filename_component(sel.dataset_name)
		dialog = SHAPExplainDialog(
			model=out.model,
			x_train=x_train,
			x_explain=x_explain,
			y_explain=y_explain,
			feature_axis=feature_axis,
			dataset_names=dataset_names,
			class_labels=list(out.split.class_labels),
			source=str(sel.source),
			selected_index=idx,
			background_samples=int(sel.background_samples),
			max_evals=int(sel.max_evals),
			top_k=int(sel.top_k),
			random_state=int(sel.random_state),
			default_export_dir=default_dir,
			default_base_name=f"{base}_shap_{dataset_part}_idx{int(sel.index_within_dataset)}_gi{idx}",
			parent=self,
		)
		dialog.exec()

	def _populate_report_table(self, rows: List[Dict[str, object]]):
		"""Populate the classification report table."""
		self._last_report_rows = list(rows or [])
		self.report_table.setRowCount(0)
		if not rows:
			return
		self.report_table.setRowCount(len(rows))
		for r, row in enumerate(rows):
			values = [
				row.get("class", ""),
				row.get("precision", ""),
				row.get("recall", ""),
				row.get("f1-score", ""),
				row.get("support", ""),
			]
			for c, v in enumerate(values):
				it = QTableWidgetItem(str(v))
				self.report_table.setItem(r, c, it)

	def _default_report_export_dir(self) -> str:
		project_path = PROJECT_MANAGER.current_project_data.get("projectFilePath")
		if project_path:
			root_dir = os.path.dirname(project_path)
			out_dir = os.path.join(root_dir, "reports")
			os.makedirs(out_dir, exist_ok=True)
			return out_dir
		return os.getcwd()

	def _default_model_basename(self) -> str:
		model_key = str(self._trained_model_key or (self.model_combo.currentData() or self.model_combo.currentText()) or "model")
		model_part = _sanitize_filename_component(self._model_display_name(model_key))
		groups = []
		try:
			groups = list(self._last_train_context.get("group_names") or [])
		except Exception:
			groups = []
		if not groups:
			try:
				groups = [
					(ui.name_edit.text() or "").strip()
					for ui in self._groups
					if (ui.name_edit.text() or "").strip() and bool(ui.include_checkbox.isChecked())
				]
			except Exception:
				groups = []
		group_part = "-".join(_sanitize_filename_component(g) for g in groups[:4])
		ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		base = f"{model_part}"
		if group_part:
			base += f"_{group_part}"
		base += f"_{ts}"
		return base

	def _default_model_save_dir(self) -> str:
		project_path = PROJECT_MANAGER.current_project_data.get("projectFilePath")
		if project_path:
			root_dir = os.path.dirname(project_path)
			out_dir = os.path.join(root_dir, "models")
			os.makedirs(out_dir, exist_ok=True)
			return out_dir
		return os.getcwd()

	def _save_model(self):
		if self._trained_model is None or self._trained_axis is None:
			QMessageBox.warning(self, LOCALIZE("COMMON.warning"), LOCALIZE("ML_PAGE.error_no_model"))
			return

		default_dir = self._default_model_save_dir()
		default_base = self._default_model_basename()
		dlg = _DestinationDialog(
			title=LOCALIZE("ML_PAGE.save_dialog_title"),
			default_dir=default_dir,
			default_filename=f"{default_base}.pkl",
			ok_text_key="ML_PAGE.dialog_save",
			show_save_params=True,
			default_save_params=True,
			parent=self,
		)
		if dlg.exec() != QDialog.Accepted:
			return
		out_dir = dlg.selected_dir()
		filename = dlg.filename()
		if not filename:
			return
		if not filename.lower().endswith(".pkl"):
			filename = f"{filename}.pkl"
		path = os.path.join(out_dir, filename)
		save_params_json = dlg.save_params()

		payload = {
			"model": self._trained_model,
			"common_axis": self._trained_axis,
			"class_labels": self._trained_class_labels,
			"model_key": self._trained_model_key,
			"model_params": self._trained_model_params,
			"train_context": dict(self._last_train_context or {}),
			"trained_at": datetime.datetime.now().isoformat(),
		}
		joblib.dump(payload, path)
		self._append_log(f"[DEBUG] Saved model to {path}")

		if save_params_json:
			try:
				params_path = os.path.splitext(path)[0] + "_params.json"
				params_payload = {
					"trained_at": payload.get("trained_at"),
					"model_key": self._trained_model_key,
					"model_display": self._model_display_name(self._trained_model_key),
					"model_params": dict(self._trained_model_params or {}),
					"train_context": dict(self._last_train_context or {}),
					"class_labels": list(self._trained_class_labels or []),
				}
				with open(params_path, "w", encoding="utf-8") as f:
					json.dump(params_payload, f, ensure_ascii=False, indent=2)
				self._append_log(f"[DEBUG] Saved params to {params_path}")
			except Exception as e:
				self._append_log(f"[DEBUG] Params JSON save failed: {e}")

	def _export_classification_report(self):
		# Prefer stored report rows, but fall back to reading the table.
		rows: List[Dict[str, object]] = list(self._last_report_rows or [])
		if not rows and self.report_table.rowCount() > 0:
			for r in range(self.report_table.rowCount()):
				row = {
					"class": (self.report_table.item(r, 0).text() if self.report_table.item(r, 0) else ""),
					"precision": (self.report_table.item(r, 1).text() if self.report_table.item(r, 1) else ""),
					"recall": (self.report_table.item(r, 2).text() if self.report_table.item(r, 2) else ""),
					"f1-score": (self.report_table.item(r, 3).text() if self.report_table.item(r, 3) else ""),
					"support": (self.report_table.item(r, 4).text() if self.report_table.item(r, 4) else ""),
				}
				rows.append(row)
		if not rows:
			QMessageBox.warning(self, LOCALIZE("COMMON.warning"), LOCALIZE("ML_PAGE.error_no_report"))
			return

		from pathlib import Path
		from components.widgets import get_export_bundle_options

		default_dir = self._default_report_export_dir()
		base = self._default_model_basename()

		opts = get_export_bundle_options(
			self,
			title=LOCALIZE("ML_PAGE.export_dialog_title"),
			default_directory=default_dir,
			default_base_name=f"{base}",
			default_image_format="png",
			# Labels (keep simple if localization keys are missing)
			location_label="Save Location:",
			base_name_label="Base Name:",
			browse_button_text="Browse...",
			select_location_title="Select Location",
			image_format_label="Image Format:",
			section_label="Include:",
			report_csv_label="Classification report (CSV)",
			report_json_label="Classification report (JSON)",
			confusion_label="Confusion matrix (image)",
			pca_boundary_label="PCA / decision boundary (image)",
			roc_label="ROC curve (image)",
			pred_dist_label="Prediction distribution (image)",
			feat_imp_label="Feature importance (image)",
		)
		if opts is None:
			return

		out_dir = Path(opts.directory)
		img_ext = ".png" if opts.image_format == "png" else ".svg"
		created: List[str] = []
		missing: List[str] = []

		# 1) Classification report
		headers = ["class", "precision", "recall", "f1-score", "support"]
		try:
			if opts.export_report_csv:
				path = out_dir / f"classification_report_{opts.base_name}.csv"
				with open(path, "w", newline="", encoding="utf-8") as f:
					writer = csv.DictWriter(f, fieldnames=headers)
					writer.writeheader()
					for row in rows:
						writer.writerow({h: row.get(h, "") for h in headers})
				created.append(str(path))

			if opts.export_report_json:
				path = out_dir / f"classification_report_{opts.base_name}.json"
				with open(path, "w", encoding="utf-8") as f:
					json.dump(rows, f, ensure_ascii=False, indent=2)
				created.append(str(path))
		except Exception as e:
			QMessageBox.critical(self, LOCALIZE("COMMON.error"), str(e))
			return

		# Helper to save a matplotlib figure
		def _save_fig(fig, stem: str) -> None:
			if fig is None:
				missing.append(stem)
				return
			try:
				path = out_dir / f"{stem}_{opts.base_name}{img_ext}"
				fig.savefig(str(path), dpi=300, bbox_inches="tight", format=opts.image_format)
				created.append(str(path))
			except Exception:
				missing.append(stem)

		# 2) Plots
		if opts.export_confusion_matrix:
			_save_fig(getattr(self.cm_plot, "figure", None), "confusion_matrix")
		if opts.export_pca_boundary:
			_save_fig(getattr(self.pca_plot, "figure", None), "pca_boundary")
		if opts.export_roc_curve:
			_save_fig(getattr(self.roc_plot, "figure", None), "roc_curve")
		if opts.export_prediction_distribution:
			_save_fig(getattr(self.pd_plot, "figure", None), "prediction_distribution")
		if opts.export_feature_importance:
			_save_fig(getattr(self.fi_plot, "figure", None), "feature_importance")

		if not created:
			QMessageBox.warning(self, LOCALIZE("COMMON.warning"), LOCALIZE("ML_PAGE.error_no_report"))
			return

		self._append_log(f"[DEBUG] Exported ML report bundle ({len(created)} files)")
		msg = "\n".join(created)
		if missing:
			msg += "\n\nMissing / not available:\n" + "\n".join(sorted(set(missing)))
		QMessageBox.information(self, LOCALIZE("COMMON.info"), msg)

	def _load_model(self):
		default_dir = self._default_model_save_dir()
		path, _ = QFileDialog.getOpenFileName(
			self,
			LOCALIZE("ML_PAGE.load_model"),
			default_dir,
			"Pickle (*.pkl)",
		)
		if not path:
			return

		# Security warning: pickle/joblib files can execute arbitrary code when loaded.
		try:
			resp = QMessageBox.warning(
				self,
				LOCALIZE("ML_PAGE.load_model_security_warning_title"),
				LOCALIZE("ML_PAGE.load_model_security_warning_text", path=path),
				QMessageBox.Yes | QMessageBox.No,
				QMessageBox.No,
			)
		except Exception:
			resp = QMessageBox.Yes
		if resp != QMessageBox.Yes:
			return

		try:
			payload = joblib.load(path)
		except Exception as e:
			QMessageBox.critical(
				self,
				LOCALIZE("COMMON.error"),
				LOCALIZE("ML_PAGE.load_model_failed").format(error=str(e)),
			)
			return
		self._trained_model = payload.get("model")
		self._trained_axis = payload.get("common_axis")
		self._trained_class_labels = list(payload.get("class_labels") or [])
		self._trained_model_key = payload.get("model_key")
		self._trained_model_params = dict(payload.get("model_params") or {})
		try:
			self._last_train_context = dict(payload.get("train_context") or {})
		except Exception:
			self._last_train_context = {}
		self._append_log(f"[DEBUG] Loaded model from {path}")
		self._refresh_eval_label_choices()

	def _refresh_eval_label_choices(self):
		# Populate from current groups + trained model labels
		labels = set()
		for ui in self._groups:
			label = (ui.name_edit.text() or "").strip()
			if label:
				labels.add(label)

		for lab in self._trained_class_labels:
			labels.add(str(lab))

		current = self.eval_true_label_combo.currentText()
		self.eval_true_label_combo.blockSignals(True)
		self.eval_true_label_combo.clear()
		self.eval_true_label_combo.addItem(LOCALIZE("ML_PAGE.unknown_label"))
		for lab in sorted(labels):
			self.eval_true_label_combo.addItem(str(lab))
		if current:
			idx = self.eval_true_label_combo.findText(current)
			if idx >= 0:
				self.eval_true_label_combo.setCurrentIndex(idx)
		self.eval_true_label_combo.blockSignals(False)

	def _evaluate_external_dataset(self):
		if self._trained_model is None or self._trained_axis is None:
			QMessageBox.warning(self, LOCALIZE("COMMON.warning"), LOCALIZE("ML_PAGE.error_no_model"))
			return

		ds_name = self.eval_dataset_combo.currentText()
		if not ds_name or ds_name not in RAMAN_DATA:
			return

		true_label = self.eval_true_label_combo.currentText()
		if true_label == LOCALIZE("ML_PAGE.unknown_label"):
			true_label = ""

		try:
			df = RAMAN_DATA[ds_name]
			X = prepare_features_for_dataset(df=df, target_axis=self._trained_axis)
			y_pred = self._trained_model.predict(X)
			proba = None
			if hasattr(self._trained_model, "predict_proba"):
				proba = self._trained_model.predict_proba(X)

			self._append_log(f"[DEBUG] Evaluated model on dataset '{ds_name}' (n={X.shape[0]})")
			preview_vals = [str(v) for v in np.asarray(y_pred).ravel()[:50]]
			heading = LOCALIZE("ML_PAGE.predictions_preview_heading").format(count=50)
			preview_text = heading + "\n\n" + ", ".join(preview_vals)

			# Counts
			counts: Dict[str, int] = {}
			for v in np.asarray(y_pred).ravel():
				k = str(v)
				counts[k] = counts.get(k, 0) + 1

			class_labels = self._trained_class_labels or sorted({str(v) for v in np.asarray(y_pred).ravel()})
			label_display_map = self._build_plot_label_display_map(list(class_labels))
			y_true = None
			if true_label:
				y_true = np.asarray([true_label] * len(y_pred), dtype=object)

			report_rows = []
			cm_fig = None
			roc_fig = None
			pd_fig = None

			if y_true is not None:
				try:
					report_rows = create_classification_report_table(
						y_true=y_true,
						y_pred=y_pred,
						labels=list(class_labels),
					)
				except Exception:
					report_rows = []
				try:
					cm_title = f"Confusion Matrix (Model: {self._model_display_name_en(self._trained_model_key)})"
					cm_fig = create_confusion_matrix_figure(
						y_true=y_true,
						y_pred=y_pred,
						class_labels=class_labels,
						title=cm_title,
						label_display_map=label_display_map,
					)
				except Exception:
					cm_fig = None

			if proba is not None and y_true is not None and len(class_labels) >= 2:
				try:
					roc_fig = create_roc_curve_figure(
						y_true=y_true,
						y_score=proba,
						class_labels=class_labels,
						title="ROC Curve",
						label_display_map=label_display_map,
					)
				except Exception:
					roc_fig = None

			if proba is not None:
				try:
					pd_fig = create_prediction_distribution_figure(
						y_true=y_true,
						y_score=proba,
						class_labels=class_labels,
						title="Prediction Distribution",
						label_display_map=label_display_map,
					)
				except Exception:
					pd_fig = None

			dlg = ExternalEvaluationDialog(
				dataset_name=str(ds_name),
				n_samples=int(X.shape[0]),
				true_label=str(true_label or ""),
				prediction_counts=counts,
				preview_text=preview_text,
				report_rows=report_rows,
				confusion_fig=cm_fig,
				roc_fig=roc_fig,
				dist_fig=pd_fig,
				parent=self,
			)
			dlg.exec()

		except Exception as e:
			create_logs(
				"MachineLearningPage",
				"evaluate_error",
				f"External evaluation failed: {e}",
				status="error",
			)
			QMessageBox.critical(self, LOCALIZE("COMMON.error"), str(e))
