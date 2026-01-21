from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
	QDialog,
	QHBoxLayout,
	QLabel,
	QTableWidget,
	QTableWidgetItem,
	QTabWidget,
	QTextEdit,
	QVBoxLayout,
	QWidget,
)

from components.widgets.matplotlib_widget import MatplotlibWidget
from utils import LOCALIZE


class ExternalEvaluationDialog(QDialog):
	"""Modal dialog to present external evaluation results.

	This keeps the main ML results tabs (train/test) intact, while allowing the user
	to inspect predictions and (optionally) metrics for a selected evaluation dataset.
	"""

	def __init__(
		self,
		*,
		dataset_name: str,
		n_samples: int,
		true_label: str = "",
		prediction_counts: Dict[str, int] | None = None,
		preview_text: str = "",
		report_rows: List[Dict[str, object]] | None = None,
		confusion_fig=None,
		roc_fig=None,
		dist_fig=None,
		parent=None,
	):
		super().__init__(parent)
		self.setModal(True)
		self.setWindowTitle(
			LOCALIZE("ML_PAGE.external_eval_window_title", dataset_name=dataset_name)
		)
		self.resize(900, 650)

		main = QVBoxLayout(self)
		main.setContentsMargins(16, 16, 16, 16)
		main.setSpacing(10)

		header = QHBoxLayout()
		title = QLabel(
			LOCALIZE(
				"ML_PAGE.external_eval_header_title",
				dataset_name=dataset_name,
				n=int(n_samples),
			)
		)
		title.setStyleSheet("font-size: 16px; font-weight: 700;")
		header.addWidget(title)
		header.addStretch(1)
		if true_label:
			lab = QLabel(
				LOCALIZE("ML_PAGE.external_eval_true_label", true_label=true_label)
			)
			lab.setStyleSheet("color: #0078d4; font-weight: 600;")
			header.addWidget(lab)
		main.addLayout(header)

		self.tabs = QTabWidget()
		self.tabs.setDocumentMode(True)
		self.tabs.setStyleSheet(
			"""
			QTabWidget::pane { border: none; background: #ffffff; padding-top: 8px; }
			QTabBar::tab { padding: 10px 18px; font-weight: 600; color: #6c757d; }
			QTabBar::tab:selected { color: #0078d4; border-bottom: 2px solid #0078d4; }
			"""
		)
		main.addWidget(self.tabs, 1)

		# --- Summary tab ---
		summary = QWidget()
		sum_layout = QVBoxLayout(summary)
		sum_layout.setContentsMargins(0, 0, 0, 0)
		sum_layout.setSpacing(8)

		if prediction_counts:
			lines = [f"{k}: {v}" for k, v in sorted(prediction_counts.items(), key=lambda x: (-x[1], x[0]))]
			counts_text = "\n".join(lines)
		else:
			counts_text = LOCALIZE("ML_PAGE.external_eval_no_predictions")

		counts_box = QTextEdit()
		counts_box.setReadOnly(True)
		counts_box.setMaximumHeight(140)
		counts_box.setText(
			LOCALIZE("ML_PAGE.external_eval_prediction_counts") + "\n\n" + counts_text
		)
		sum_layout.addWidget(counts_box)

		preview_box = QTextEdit()
		preview_box.setReadOnly(True)
		preview_box.setText(preview_text or "")
		sum_layout.addWidget(preview_box, 1)

		self.tabs.addTab(summary, LOCALIZE("ML_PAGE.external_eval_tab_summary"))

		# --- Report tab ---
		self.report_table = QTableWidget()
		self.report_table.setColumnCount(5)
		self.report_table.setHorizontalHeaderLabels(
			[
				LOCALIZE("ML_PAGE.report_header_class"),
				LOCALIZE("ML_PAGE.report_header_precision"),
				LOCALIZE("ML_PAGE.report_header_recall"),
				LOCALIZE("ML_PAGE.report_header_f1"),
				LOCALIZE("ML_PAGE.report_header_support"),
			]
		)
		self.report_table.setEditTriggers(QTableWidget.NoEditTriggers)
		self.report_table.setAlternatingRowColors(True)
		self.report_table.horizontalHeader().setStretchLastSection(True)
		self.report_table.verticalHeader().setVisible(False)

		self._populate_report_table(report_rows or [])
		self.tabs.addTab(self.report_table, LOCALIZE("ML_PAGE.external_eval_tab_report"))

		# --- Confusion matrix tab ---
		cm_widget = MatplotlibWidget()
		if confusion_fig is not None:
			cm_widget.update_plot_with_config(confusion_fig, {"colorbar": True})
		else:
			# No true label -> no confusion matrix
			cm_widget.figure.clear()
			ax = cm_widget.figure.add_subplot(111)
			ax.axis("off")
			ax.text(
				0.5,
				0.5,
				LOCALIZE("ML_PAGE.external_eval_requires_true_label_confusion"),
				ha="center",
				va="center",
			)
			cm_widget.canvas.draw()
		self.tabs.addTab(cm_widget, LOCALIZE("ML_PAGE.external_eval_tab_confusion"))

		# --- ROC tab ---
		roc_widget = MatplotlibWidget()
		if roc_fig is not None:
			roc_widget.update_plot_with_config(roc_fig)
		else:
			roc_widget.figure.clear()
			ax = roc_widget.figure.add_subplot(111)
			ax.axis("off")
			ax.text(
				0.5,
				0.5,
				LOCALIZE("ML_PAGE.external_eval_requires_true_label_roc"),
				ha="center",
				va="center",
			)
			roc_widget.canvas.draw()
		self.tabs.addTab(roc_widget, LOCALIZE("ML_PAGE.external_eval_tab_roc"))

		# --- Distribution tab ---
		dist_widget = MatplotlibWidget()
		if dist_fig is not None:
			dist_widget.update_plot_with_config(dist_fig)
		else:
			dist_widget.figure.clear()
			ax = dist_widget.figure.add_subplot(111)
			ax.axis("off")
			ax.text(
				0.5,
				0.5,
				LOCALIZE("ML_PAGE.external_eval_requires_prob_outputs"),
				ha="center",
				va="center",
			)
			dist_widget.canvas.draw()
		self.tabs.addTab(dist_widget, LOCALIZE("ML_PAGE.external_eval_tab_distribution"))

	def _populate_report_table(self, rows: List[Dict[str, object]]):
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
				it.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
				self.report_table.setItem(r, c, it)
