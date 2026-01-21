from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

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
from components.widgets.parameter_widgets import CustomDoubleSpinBox, CustomSpinBox
from components.widgets.multi_group_dialog import MultiGroupCreationDialog
from components.widgets.external_evaluation_dialog import ExternalEvaluationDialog
from configs.configs import create_logs
from configs.style.stylesheets import combine_styles, get_base_style, get_page_style
from functions.ML import prepare_features_for_dataset
from functions.visualization.model_evaluation import (
	create_confusion_matrix_figure,
	create_feature_importance_figure,
	create_prediction_distribution_figure,
	create_roc_curve_figure,
)
from functions.visualization.classification_report import create_classification_report_table
from functions.visualization.pca_decision_boundary import create_pca_decision_boundary_figure
from pages.machine_learning_page_utils.dnd_widgets import DatasetSourceList, GroupDropList
from pages.machine_learning_page_utils.thread import MLTrainingOutput, MLTrainingThread
from utils import LOCALIZE, PROJECT_MANAGER, RAMAN_DATA

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

		self._setup_ui()
		self._connect_signals()
		self.update_localized_text()
		self.load_project_data()

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

		self.results_tabs = QTabWidget()
		self.results_tabs.setObjectName("mlResultsTabs")
		self.results_tabs.setDocumentMode(True)
		# Match Analysis page tab styling (improves tab-bar visibility/consistency)
		self.results_tabs.setStyleSheet(
			"""
			QTabWidget#mlResultsTabs::pane {
				border: none;
				background: #ffffff;
				padding-top: 10px;
			}
			QTabWidget#mlResultsTabs::tab-bar {
				alignment: left;
			}
			QTabWidget#mlResultsTabs QTabBar {
				background: #ffffff;
				border-bottom: 1px solid #dee2e6;
			}
			QTabWidget#mlResultsTabs QTabBar::tab {
				background: transparent;
				color: #6c757d;
				font-size: 13px;
				font-weight: 600;
				padding: 12px 20px;
				border-bottom: 2px solid transparent;
				margin-left: 8px;
			}
			QTabWidget#mlResultsTabs QTabBar::tab:hover {
				color: #0078d4;
				background-color: #f8f9fa;
				border-radius: 4px 4px 0 0;
			}
			QTabWidget#mlResultsTabs QTabBar::tab:selected {
				color: #0078d4;
				border-bottom: 2px solid #0078d4;
			}
			"""
		)

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

		self.results_tabs.addTab(self.cm_plot, "")
		self.results_tabs.addTab(self.report_table, "")
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
		self.start_training_button.setText(LOCALIZE("ML_PAGE.start_training"))
		self.save_model_button.setText(LOCALIZE("ML_PAGE.save_model"))
		self.load_model_button.setText(LOCALIZE("ML_PAGE.load_model"))
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

		# Model method display text
		self.model_combo.setItemText(0, LOCALIZE("ML_PAGE.model_linear_regression"))
		self.model_combo.setItemText(1, LOCALIZE("ML_PAGE.model_logistic_regression"))
		self.model_combo.setItemText(2, LOCALIZE("ML_PAGE.model_svm"))
		self.model_combo.setItemText(3, LOCALIZE("ML_PAGE.model_random_forest"))

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
		self.results_tabs.setTabText(2, LOCALIZE("ML_PAGE.results_pca_boundary"))
		self.results_tabs.setTabText(3, LOCALIZE("ML_PAGE.results_roc"))
		self.results_tabs.setTabText(4, LOCALIZE("ML_PAGE.results_importance"))
		self.results_tabs.setTabText(5, LOCALIZE("ML_PAGE.results_distribution"))

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
		PROJECT_MANAGER.set_ml_groups_and_enabled(groups, enabled)

	def _load_groups_from_project(self):
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
		try:
			if clean != saved and available:
				PROJECT_MANAGER.set_ml_groups(clean)
		except Exception:
			pass

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
		}
		return mapping.get(key, key.replace("_", " ").title())

	def _set_controls_enabled(self, enabled: bool):
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
		self.evaluate_button.setEnabled(enabled)
		self.eval_dataset_combo.setEnabled(enabled)
		self.eval_true_label_combo.setEnabled(enabled)

	def _start_training(self):
		if self._training_thread and self._training_thread.isRunning():
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

		self._append_log(
			f"[DEBUG] Start training: model={model_key} classes={class_labels} train_ratio={train_ratio} split_mode={split_mode}"
		)
		self.progress.setValue(0)
		self.status_label.setText(LOCALIZE("ML_PAGE.status_training"))
		self._set_controls_enabled(False)

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
		self._set_controls_enabled(True)
		QMessageBox.critical(self, LOCALIZE("COMMON.error"), msg)

	def _on_training_completed(self, out: MLTrainingOutput):
		self._trained_model = out.model
		self._trained_axis = np.asarray(out.split.common_axis, dtype=float)
		self._trained_class_labels = list(out.split.class_labels)
		self._trained_model_key = out.model_key
		self._trained_model_params = dict(out.model_params)
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

		# Confusion matrix
		try:
			cm_title = f"{LOCALIZE('ML_PAGE.results_confusion')} (Model: {self._model_display_name(out.model_key)})"
			cm_fig = create_confusion_matrix_figure(
				y_true=out.split.y_test,
				y_pred=out.y_pred,
				class_labels=out.split.class_labels,
				title=cm_title,
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
					title=LOCALIZE("ML_PAGE.results_roc"),
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
					title=LOCALIZE("ML_PAGE.results_importance"),
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
					title=LOCALIZE("ML_PAGE.results_distribution"),
				)
				self.pd_plot.update_plot_with_config(pd_fig)
			else:
				self.pd_plot.figure.clear()
				self.pd_plot.canvas.draw()
		except Exception as e:
			self._append_log(f"[DEBUG] Prediction distribution plot failed: {e}")

		# PCA + decision boundary (visualization-only)
		try:
			pca_fig = create_pca_decision_boundary_figure(
				X=np.asarray(out.split.X_train, dtype=float),
				y=np.asarray(out.split.y_train, dtype=object),
				model_key=str(out.model_key),
				model_params=dict(out.model_params),
				title=LOCALIZE("ML_PAGE.results_pca_boundary"),
			)
			# Background is image-based; don't add colorbar (it tends to clutter this view).
			self.pca_plot.update_plot_with_config(pca_fig, {"colorbar": False})
		except Exception as e:
			self._append_log(f"[DEBUG] PCA boundary plot failed: {e}")

		self.status_label.setText(LOCALIZE("ML_PAGE.status_done"))
		self._set_controls_enabled(True)

	def _populate_report_table(self, rows: List[Dict[str, object]]):
		"""Populate the classification report table."""
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
		default_name = f"ml_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
		path, _ = QFileDialog.getSaveFileName(
			self,
			LOCALIZE("ML_PAGE.save_model"),
			os.path.join(default_dir, default_name),
			"Pickle (*.pkl)",
		)
		if not path:
			return

		payload = {
			"model": self._trained_model,
			"common_axis": self._trained_axis,
			"class_labels": self._trained_class_labels,
			"model_key": self._trained_model_key,
			"model_params": self._trained_model_params,
			"trained_at": datetime.datetime.now().isoformat(),
		}
		joblib.dump(payload, path)
		self._append_log(f"[DEBUG] Saved model to {path}")

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

		payload = joblib.load(path)
		self._trained_model = payload.get("model")
		self._trained_axis = payload.get("common_axis")
		self._trained_class_labels = list(payload.get("class_labels") or [])
		self._trained_model_key = payload.get("model_key")
		self._trained_model_params = dict(payload.get("model_params") or {})
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
					cm_title = f"{LOCALIZE('ML_PAGE.results_confusion')} (Model: {self._model_display_name(self._trained_model_key)})"
					cm_fig = create_confusion_matrix_figure(
						y_true=y_true,
						y_pred=y_pred,
						class_labels=class_labels,
						title=cm_title,
					)
				except Exception:
					cm_fig = None

			if proba is not None and y_true is not None and len(class_labels) >= 2:
				try:
					roc_fig = create_roc_curve_figure(
						y_true=y_true,
						y_score=proba,
						class_labels=class_labels,
						title=LOCALIZE("ML_PAGE.results_roc"),
					)
				except Exception:
					roc_fig = None

			if proba is not None:
				try:
					pd_fig = create_prediction_distribution_figure(
						y_true=y_true,
						y_score=proba,
						class_labels=class_labels,
						title=LOCALIZE("ML_PAGE.results_distribution"),
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
