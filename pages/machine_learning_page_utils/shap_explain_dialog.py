from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
	QDialog,
	QHBoxLayout,
	QLabel,
	QMessageBox,
	QProgressBar,
	QPushButton,
	QTableWidget,
	QTableWidgetItem,
	QTabWidget,
	QTextEdit,
	QVBoxLayout,
	QWidget,
)

from components.widgets import LoadingOverlay, MatplotlibWidget, get_export_shap_bundle_options
from configs.configs import create_logs
from utils import LOCALIZE

from .shap_thread import SHAPExplainThread, SHAPResult


def _safe_filename_component(s: str) -> str:
	out = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in (s or ""))
	out = out.strip("_ ")
	return out or "shap"


class SHAPExplainDialog(QDialog):
	"""Dialog UI for SHAP per-spectrum explanation.

	This dialog owns a SHAPExplainThread and shows:
	- progress/status (UI-visible)
	- spectrum plot and SHAP contribution plot
	- a contributor table (with optional peak assignments)
	- export button for the displayed results
	"""

	def __init__(
		self,
		*,
		model: Any,
		x_train: np.ndarray,
		x_explain: np.ndarray,
		y_explain: np.ndarray,
		feature_axis: Optional[np.ndarray] = None,
		dataset_names: Optional[List[str]] = None,
		class_labels: Optional[List[str]] = None,
		source: str = "test",
		selected_index: int = 0,
		background_samples: int = 30,
		max_evals: int = 0,
		top_k: int = 12,
		random_state: int = 0,
		default_export_dir: str = "",
		default_base_name: str = "",
		parent=None,
	):
		super().__init__(parent)
		self.setModal(True)
		# Result dialog (parameter setting happens in a separate dialog)
		self.setWindowTitle(LOCALIZE("ML_PAGE.shap_result_title"))
		self.resize(980, 720)

		self._model = model
		self._x_train = np.asarray(x_train, dtype=float)
		self._x_explain = np.asarray(x_explain, dtype=float)
		self._y_explain = np.asarray(y_explain, dtype=object)
		self._feature_axis = None if feature_axis is None else np.asarray(feature_axis, dtype=float)
		self._dataset_names = list(dataset_names) if dataset_names is not None else None
		self._class_labels = list(class_labels) if class_labels is not None else None

		self._source = str(source or "test")
		self._selected_index = int(selected_index)
		self._background_samples = int(background_samples)
		self._max_evals = int(max_evals)
		self._top_k = int(top_k)
		self._random_state = int(random_state)

		self._default_export_dir = str(default_export_dir or "")
		self._default_base_name = str(default_base_name or "")

		self._thread: Optional[SHAPExplainThread] = None
		self._result: Optional[SHAPResult] = None

		main = QVBoxLayout(self)
		main.setContentsMargins(16, 16, 16, 16)
		main.setSpacing(10)

		# Header: title + status + export
		header = QHBoxLayout()
		title = QLabel(LOCALIZE("ML_PAGE.shap_dialog_header"))
		title.setStyleSheet("font-size: 16px; font-weight: 700;")
		header.addWidget(title)
		header.addStretch(1)

		self._status_label = QLabel(LOCALIZE("ML_PAGE.shap_status_preparing"))
		self._status_label.setStyleSheet("color: #6c757d;")
		header.addWidget(self._status_label)

		self._progress = QProgressBar()
		self._progress.setRange(0, 100)
		self._progress.setValue(0)
		self._progress.setFixedWidth(220)
		header.addWidget(self._progress)

		self._stop_btn = QPushButton(LOCALIZE("ML_PAGE.stop"))
		self._stop_btn.setEnabled(False)
		self._stop_btn.clicked.connect(self._stop)
		header.addWidget(self._stop_btn)

		self._export_btn = QPushButton(LOCALIZE("ML_PAGE.shap_export"))
		self._export_btn.setEnabled(False)
		self._export_btn.clicked.connect(self._export_results)
		header.addWidget(self._export_btn)

		close_btn = QPushButton(LOCALIZE("COMMON.close"))
		close_btn.clicked.connect(self.accept)
		header.addWidget(close_btn)

		main.addLayout(header)

		# Content area (tabs) + scoped loading overlay.
		# IMPORTANT: Do not overlay the header; users must be able to press Stop/Close.
		content = QWidget()
		content_layout = QVBoxLayout(content)
		content_layout.setContentsMargins(0, 0, 0, 0)
		content_layout.setSpacing(0)
		main.addWidget(content, 1)

		self.tabs = QTabWidget()
		self.tabs.setDocumentMode(True)
		self.tabs.setStyleSheet(
			"""
			QTabWidget::pane { border: none; background: #ffffff; padding-top: 8px; }
			QTabBar::tab { padding: 10px 18px; font-weight: 600; color: #6c757d; }
			QTabBar::tab:selected { color: #0078d4; border-bottom: 2px solid #0078d4; }
			"""
		)
		content_layout.addWidget(self.tabs, 1)

		# Summary tab
		summary = QWidget()
		sum_layout = QVBoxLayout(summary)
		sum_layout.setContentsMargins(0, 0, 0, 0)
		sum_layout.setSpacing(8)

		self._meta_text = QTextEdit()
		self._meta_text.setReadOnly(True)
		self._meta_text.setMaximumHeight(150)
		self._meta_text.setStyleSheet(
			"border: 1px solid #dee2e6; border-radius: 4px; padding: 8px; font-family: monospace; font-size: 12px;"
		)
		sum_layout.addWidget(self._meta_text)

		self._table = QTableWidget()
		self._table.setColumnCount(6)
		self._table.setHorizontalHeaderLabels(
			[
				LOCALIZE("ML_PAGE.shap_table_wavenumber"),
				LOCALIZE("ML_PAGE.shap_table_value"),
				LOCALIZE("ML_PAGE.shap_table_abs"),
				LOCALIZE("ML_PAGE.shap_table_sign"),
				LOCALIZE("ML_PAGE.shap_table_assignment"),
				LOCALIZE("ML_PAGE.shap_table_reference"),
			]
		)
		self._table.setEditTriggers(QTableWidget.NoEditTriggers)
		self._table.setAlternatingRowColors(True)
		try:
			self._table.setSortingEnabled(True)
			self._table.horizontalHeader().setSectionsClickable(True)
			self._table.horizontalHeader().setSortIndicatorShown(True)
		except Exception:
			pass
		self._table.horizontalHeader().setStretchLastSection(True)
		self._table.verticalHeader().setVisible(False)
		self._table.setStyleSheet("QTableWidget { border: 1px solid #dee2e6; }")
		sum_layout.addWidget(self._table, 1)

		self.tabs.addTab(summary, LOCALIZE("ML_PAGE.shap_tab_summary"))

		# Report tab (enhanced summary / peak assignment style)
		report = QWidget()
		report_layout = QVBoxLayout(report)
		report_layout.setContentsMargins(0, 0, 0, 0)
		report_layout.setSpacing(8)
		self._report_text = QTextEdit()
		self._report_text.setReadOnly(True)
		self._report_text.setStyleSheet(
			"border: 1px solid #dee2e6; border-radius: 6px; padding: 10px; font-family: monospace; font-size: 12px;"
		)
		report_layout.addWidget(self._report_text, 1)
		self.tabs.addTab(report, LOCALIZE("ML_PAGE.shap_tab_report"))

		# Spectrum tab
		self._spectrum_plot = MatplotlibWidget()
		try:
			self._spectrum_plot.toolbar.setVisible(True)
			self._spectrum_plot.toolbar.show()
		except Exception:
			pass
		self.tabs.addTab(self._spectrum_plot, LOCALIZE("ML_PAGE.shap_tab_spectrum"))

		# SHAP tab
		self._shap_plot = MatplotlibWidget()
		try:
			self._shap_plot.toolbar.setVisible(True)
			self._shap_plot.toolbar.show()
		except Exception:
			pass
		self.tabs.addTab(self._shap_plot, LOCALIZE("ML_PAGE.shap_tab_shap"))

		# Optional provenance tab (simple)
		prov = QWidget()
		prov_layout = QVBoxLayout(prov)
		prov_layout.setContentsMargins(0, 0, 0, 0)
		prov_layout.setSpacing(8)
		self._prov_text = QTextEdit()
		self._prov_text.setReadOnly(True)
		self._prov_text.setStyleSheet(
			"border: 1px solid #dee2e6; border-radius: 4px; padding: 8px; font-family: monospace; font-size: 12px;"
		)
		prov_layout.addWidget(self._prov_text, 1)
		self.tabs.addTab(prov, LOCALIZE("ML_PAGE.shap_tab_provenance"))

		self._loading_overlay = LoadingOverlay(content)

		# Start compute after the dialog is shown (so overlay can paint)
		QTimer.singleShot(0, self._start)

	def _release_result_figures(self) -> None:
		"""Best-effort cleanup for Matplotlib figures held by the previous result.

		Why:
		- SHAP runs can be repeated in the same dialog.
		- Each run creates new Matplotlib Figure objects.
		- If we keep old results referenced, memory usage can grow (especially in portable builds).
		"""
		res = self._result
		if res is None:
			return
		self._result = None

		try:
			import matplotlib.pyplot as plt
		except Exception:
			plt = None

		for fig in (getattr(res, "spectrum_figure", None), getattr(res, "shap_figure", None)):
			if fig is None:
				continue
			try:
				# Clear artists so large arrays can be released earlier.
				fig.clf()
			except Exception:
				pass
			try:
				if plt is not None:
					plt.close(fig)
			except Exception:
				pass

	def _start(self) -> None:
		if self._thread is not None and self._thread.isRunning():
			return
		# If this dialog is re-run, release previous figures to avoid growing memory.
		try:
			self._release_result_figures()
		except Exception:
			pass
		self._export_btn.setEnabled(False)
		self._meta_text.setPlainText("")
		self._prov_text.setPlainText("")
		self._table.setRowCount(0)

		self._loading_overlay.show_loading(LOCALIZE("ML_PAGE.shap_status_preparing"))
		self._status_label.setText(LOCALIZE("ML_PAGE.shap_status_preparing"))
		self._progress.setValue(0)
		self._stop_btn.setEnabled(True)

		self._thread = SHAPExplainThread(
			model=self._model,
			x_train=self._x_train,
			x_explain=self._x_explain,
			y_explain=self._y_explain,
			dataset_names=self._dataset_names,
			class_labels=self._class_labels,
			source=self._source,
			selected_index=self._selected_index,
			feature_axis=self._feature_axis,
			background_samples=self._background_samples,
			max_evals=self._max_evals,
			top_k=self._top_k,
			random_state=self._random_state,
			parent=self,
		)
		self._thread.status_updated.connect(self._on_status)
		self._thread.progress_updated.connect(self._progress.setValue)
		self._thread.completed.connect(self._on_done)
		self._thread.error.connect(self._on_error)
		self._thread.start()

	def _stop(self, *, hard: bool = False) -> None:
		"""Best-effort cancellation of an in-flight SHAP computation.

		We avoid hard termination by default because QThread.terminate() can leave Qt
		objects in a bad state and crash the app. The worker performs cooperative
		cancellation via requestInterruption().
		"""
		t = self._thread
		if t is None or (not t.isRunning()):
			return
		self._stop_btn.setEnabled(False)
		self._export_btn.setEnabled(False)
		self._status_label.setText(LOCALIZE("ML_PAGE.status_stopping"))
		try:
			self._loading_overlay.show_loading(LOCALIZE("ML_PAGE.status_stopping"))
		except Exception:
			pass

		# Detach signals to avoid UI updates after cancellation.
		try:
			t.status_updated.disconnect()
		except Exception:
			pass
		try:
			t.progress_updated.disconnect()
		except Exception:
			pass
		try:
			t.completed.disconnect()
		except Exception:
			pass
		try:
			t.error.disconnect()
		except Exception:
			pass

		try:
			t.requestInterruption()
		except Exception:
			pass

		# Give the worker time to stop on its own.
		try:
			t.wait(300)
		except Exception:
			pass

		# Optional last-resort: only when explicitly requested.
		if hard and t.isRunning():
			try:
				t.terminate()
				t.wait(1000)
			except Exception:
				pass

		self._thread = None
		try:
			self._loading_overlay.hide_loading()
		except Exception:
			pass
		self._status_label.setText(LOCALIZE("ML_PAGE.status_cancelled"))
		self._progress.setValue(0)

	def closeEvent(self, event: QCloseEvent) -> None:
		# If user closes while running, request cancellation to avoid background updates.
		try:
			self._stop(hard=False)
		except Exception:
			pass
		try:
			self._release_result_figures()
		except Exception:
			pass
		try:
			self._loading_overlay.detach()
		except Exception:
			pass
		super().closeEvent(event)

	def _on_status(self, s: str) -> None:
		msg = str(s or "")
		if msg:
			self._status_label.setText(msg)
			try:
				self._loading_overlay.show_loading(msg)
			except Exception:
				pass

	def _on_done(self, res: SHAPResult) -> None:
		self._result = res
		self._loading_overlay.hide_loading()
		self._status_label.setText(LOCALIZE("ML_PAGE.shap_status_done"))
		self._progress.setValue(100)
		self._export_btn.setEnabled(True)
		self._stop_btn.setEnabled(False)

		try:
			self._spectrum_plot.update_plot_with_config(
				res.spectrum_figure,
				{"colorbar": False, "figure": {"tight_layout": False, "constrained_layout": False}},
			)
			self._shap_plot.update_plot_with_config(
				res.shap_figure,
				{"colorbar": False, "figure": {"tight_layout": False, "constrained_layout": False}},
			)
		except Exception as e:
			create_logs("SHAPExplainDialog", "render_error", str(e), status="warning")

		self._meta_text.setPlainText(self._build_meta_text(res))
		self._prov_text.setPlainText(self._build_provenance_text(res))
		self._populate_table(res.contributor_rows)
		try:
			self._report_text.setPlainText(self._build_report_text(res))
		except Exception:
			self._report_text.setPlainText("")

		# Bring user to the most relevant plot
		try:
			# Prefer the SHAP plot tab.
			self.tabs.setCurrentIndex(3)
		except Exception:
			pass

	def _build_report_text(self, res: SHAPResult) -> str:
		"""Build an 'enhanced prediction analysis' style report (text).

		This is designed to match the user's requested summary view:
		- headline math breakdown
		- top positive/negative contributors with peak assignments
		- simple decision logic hints
		"""
		lines: List[str] = []
		lines.append("ENHANCED PREDICTION ANALYSIS WITH PEAK ASSIGNMENTS")
		lines.append("")
		lines.append("MATHEMATICAL PREDICTION BREAKDOWN")
		lines.append(f"  Base model expectation: {res.base_value:.6f}")
		lines.append(f"  Total SHAP contribution: {float(np.sum(res.shap_values)):.6f}")
		lines.append(f"  Final prediction score: {res.approx_output:.6f}")
		lines.append(f"  Predicted class: {res.pred_label} (p={res.pred_proba:.4f})")
		lines.append("")

		# Separate contributors by sign.
		rows = list(res.contributor_rows or [])
		pos = [r for r in rows if str(r.get("sign")) == "+" or float(r.get("shap_value", 0.0)) >= 0]
		neg = [r for r in rows if str(r.get("sign")) == "-" or float(r.get("shap_value", 0.0)) < 0]
		pos.sort(key=lambda r: float(r.get("abs_shap", 0.0)), reverse=True)
		neg.sort(key=lambda r: float(r.get("abs_shap", 0.0)), reverse=True)

		# Determine opposite label for binary models (best-effort).
		opp = ""
		try:
			if self._class_labels and res.pred_label in self._class_labels and len(self._class_labels) == 2:
				opp = [c for c in self._class_labels if c != res.pred_label][0]
		except Exception:
			opp = ""

		lines.append(f"TOP {res.pred_label} CONTRIBUTORS")
		for i, r in enumerate(pos[:5], start=1):
			lines.append(
				f"  #{i}  {float(r.get('wavenumber', 0.0)):.4f} cm⁻¹ | SHAP {float(r.get('shap_value', 0.0)):.6f} | {str(r.get('assignment', '')).strip()}"
			)
		lines.append("")
		lines.append(f"TOP {opp or 'OPPOSING'} CONTRIBUTORS")
		for i, r in enumerate(neg[:5], start=1):
			lines.append(
				f"  #{i}  {float(r.get('wavenumber', 0.0)):.4f} cm⁻¹ | SHAP {float(r.get('shap_value', 0.0)):.6f} | {str(r.get('assignment', '')).strip()}"
			)
		lines.append("")

		lines.append("DECISION LOGIC & PEAK ANALYSIS")
		net = float(np.sum(res.shap_values))
		net_dir = f"→ {res.pred_label}" if net >= 0 else f"→ {opp or 'opposing'}"
		lines.append(f"  Net SHAP direction: {net_dir} (sum={net:.6f})")
		if res.pred_proba >= 0.90:
			conf = "High"
		elif res.pred_proba >= 0.70:
			conf = "Medium"
		else:
			conf = "Low"
		lines.append(f"  Confidence level: {conf} (p={res.pred_proba:.4f})")
		try:
			key = rows[0] if rows else None
			if key:
				lines.append(
					f"  Key decision factor: {float(key.get('wavenumber', 0.0)):.4f} cm⁻¹ | SHAP {float(key.get('shap_value', 0.0)):.6f} | {str(key.get('assignment', '')).strip()}"
				)
		except Exception:
			pass

		return "\n".join(lines)

	def _on_error(self, msg: str) -> None:
		self._loading_overlay.hide_loading()
		self._status_label.setText(LOCALIZE("COMMON.error"))
		self._progress.setValue(0)
		QMessageBox.critical(
			self,
			LOCALIZE("COMMON.error"),
			LOCALIZE("ML_PAGE.shap_error_prefix") + "\n" + str(msg),
		)

	def _build_meta_text(self, res: SHAPResult) -> str:
		lines = [
			f"source: {res.source}",
			f"index: {res.selected_index}",
		]
		if res.dataset_name:
			lines.append(f"dataset: {res.dataset_name}")
		if res.true_label:
			lines.append(f"true label: {res.true_label}")
		lines.extend(
			[
				f"predicted: {res.pred_label} (p={res.pred_proba:.4f})",
				f"background samples: {res.used_background}",
				f"max_evals used: {res.max_evals_used}",
				f"base + sum(shap) ≈ {res.base_value:.6f} + {float(np.sum(res.shap_values)):.6f} = {res.approx_output:.6f}",
			]
		)
		return "\n".join(lines)

	def _build_provenance_text(self, res: SHAPResult) -> str:
		# Minimal (optional) provenance tab
		lines = [
			LOCALIZE("ML_PAGE.shap_provenance_hint"),
			"",
			f"source: {res.source}",
			f"dataset: {res.dataset_name or '(unknown)'}",
			f"true: {res.true_label or '(unknown)'}",
			f"pred: {res.pred_label}",
		]
		return "\n".join(lines)

	def _populate_table(self, rows: List[Dict[str, object]]) -> None:
		self._table.setRowCount(0)
		if not rows:
			return
		self._table.setRowCount(len(rows))
		for r, row in enumerate(rows):
			wv = row.get("wavenumber", "")
			val = row.get("shap_value", "")
			abs_val = row.get("abs_shap", "")
			sign = row.get("sign", "")
			assign = row.get("assignment", "")
			ref = row.get("reference", "")
			vals = [wv, val, abs_val, sign, assign, ref]
			for c, v in enumerate(vals):
				it = QTableWidgetItem(str(v))
				# Make numeric columns sort numerically.
				try:
					from PySide6.QtCore import Qt
					if c in (0, 1, 2):
						it.setData(Qt.EditRole, float(v))
				except Exception:
					pass
				self._table.setItem(r, c, it)

	def _export_results(self) -> None:
		res = self._result
		if res is None:
			return

		base = self._default_base_name
		if not base:
			bits = ["shap", res.source, f"idx{res.selected_index}"]
			if res.dataset_name:
				bits.append(_safe_filename_component(res.dataset_name)[:24])
			base = "_".join(bits)

		opts = get_export_shap_bundle_options(
			self,
			title=LOCALIZE("ML_PAGE.shap_export_title"),
			default_directory=self._default_export_dir or os.getcwd(),
			default_base_name=base,
			default_image_format="png",
		)
		if opts is None:
			return

		out_dir = opts.directory
		os.makedirs(out_dir, exist_ok=True)
		base_name = _safe_filename_component(opts.base_name)

		try:
			written: List[str] = []

			if opts.export_spectrum_plot:
				p = os.path.join(out_dir, f"{base_name}_spectrum.{opts.image_format}")
				res.spectrum_figure.savefig(p)
				written.append(p)
			if opts.export_shap_plot:
				p = os.path.join(out_dir, f"{base_name}_shap.{opts.image_format}")
				res.shap_figure.savefig(p)
				written.append(p)

			if opts.export_contributors_csv:
				p = os.path.join(out_dir, f"{base_name}_contributors.csv")
				with open(p, "w", newline="", encoding="utf-8") as f:
					w = csv.writer(f)
					w.writerow(["rank", "wavenumber", "shap_value", "abs_shap", "sign", "assignment", "reference"])
					for i, row in enumerate(res.contributor_rows, start=1):
						w.writerow(
							[
								i,
								row.get("wavenumber", ""),
								row.get("shap_value", ""),
								row.get("abs_shap", ""),
								row.get("sign", ""),
								row.get("assignment", ""),
								row.get("reference", ""),
							]
						)
				written.append(p)

			if opts.export_raw_json:
				p = os.path.join(out_dir, f"{base_name}_raw.json")
				payload = {
					"source": res.source,
					"selected_index": res.selected_index,
					"dataset_name": res.dataset_name,
					"true_label": res.true_label,
					"pred_label": res.pred_label,
					"pred_proba": res.pred_proba,
					"used_background": res.used_background,
					"max_evals_used": res.max_evals_used,
					"base_value": res.base_value,
					"approx_output": res.approx_output,
					"axis": [float(x) for x in np.asarray(res.axis).reshape(-1)],
					"spectrum": [float(x) for x in np.asarray(res.spectrum).reshape(-1)],
					"shap_values": [float(x) for x in np.asarray(res.shap_values).reshape(-1)],
					"contributors": list(res.contributor_rows or []),
				}
				with open(p, "w", encoding="utf-8") as f:
					json.dump(payload, f, ensure_ascii=False, indent=2)
				written.append(p)

			if opts.export_metadata_json:
				p = os.path.join(out_dir, f"{base_name}_meta.json")
				payload = {
					"source": res.source,
					"selected_index": res.selected_index,
					"dataset_name": res.dataset_name,
					"true_label": res.true_label,
					"pred_label": res.pred_label,
					"pred_proba": res.pred_proba,
					"used_background": res.used_background,
					"max_evals_used": res.max_evals_used,
				}
				with open(p, "w", encoding="utf-8") as f:
					json.dump(payload, f, ensure_ascii=False, indent=2)
				written.append(p)

			create_logs(
				"SHAPExplainDialog",
				"export",
				"Exported SHAP results:\n" + "\n".join(written),
				status="info",
			)
			QMessageBox.information(
				self,
				LOCALIZE("COMMON.success"),
				LOCALIZE("ML_PAGE.shap_export_done") + "\n\n" + "\n".join(written),
			)
		except Exception as e:
			create_logs("SHAPExplainDialog", "export_error", str(e), status="error")
			QMessageBox.critical(self, LOCALIZE("COMMON.error"), str(e))