from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
	QComboBox,
	QDialog,
	QFormLayout,
	QHBoxLayout,
	QLabel,
	QSplitter,
	QTextEdit,
	QVBoxLayout,
	QWidget,
	QPushButton,
)

from components.widgets.parameter_widgets import CustomSpinBox
from components.widgets.matplotlib_widget import MatplotlibWidget
from configs.style.stylesheets import get_base_style
from utils import LOCALIZE, PROJECT_MANAGER


@dataclass(frozen=True)
class SHAPParameterSelection:
	"""Resolved selection from the parameter dialog."""

	source: str  # currently only "test" (reserved for future)
	dataset_name: str
	index_within_dataset: int
	global_index: int
	background_samples: int
	max_evals: int
	top_k: int
	random_state: int


class SHAPParameterDialog(QDialog):
	"""Parameter-setting dialog for SHAP explainability.

	Design goals:
	- User selects dataset (from the trained model's included datasets) and spectrum index.
	- Right side shows live spectrum preview + dataset metadata.
	- Keeps advanced knobs (background_samples, max_evals, top_k, random_state).
	"""

	def __init__(
		self,
		*,
		source: str,
		x_explain: np.ndarray,
		y_explain: np.ndarray,
		dataset_names: Sequence[str] | None,
		feature_axis: np.ndarray | None,
		default_background_samples: int = 30,
		default_max_evals: int = 0,
		default_top_k: int = 12,
		default_random_state: int = 0,
		parent=None,
	):
		super().__init__(parent)
		self.setModal(True)
		self.setWindowTitle(LOCALIZE("ML_PAGE.shap_settings_title"))

		self._source = str(source or "test")
		self._x_explain = np.asarray(x_explain, dtype=float)
		self._y_explain = np.asarray(y_explain, dtype=object)
		self._dataset_names = list(dataset_names) if dataset_names is not None else []
		self._axis = None if feature_axis is None else np.asarray(feature_axis, dtype=float).reshape(-1)

		# Build dataset -> global indices mapping.
		self._dataset_to_indices: Dict[str, List[int]] = {}
		for i in range(int(self._x_explain.shape[0] if self._x_explain.ndim == 2 else 0)):
			name = ""
			try:
				if self._dataset_names and i < len(self._dataset_names):
					name = str(self._dataset_names[i])
			except Exception:
				name = ""
			name = name or "(unknown)"
			self._dataset_to_indices.setdefault(name, []).append(i)

		# UI layout
		root = QVBoxLayout(self)
		root.setContentsMargins(16, 16, 16, 16)
		root.setSpacing(12)

		splitter = QSplitter(Qt.Horizontal)
		root.addWidget(splitter, 1)

		# Left panel: parameters
		left = QWidget()
		left_layout = QVBoxLayout(left)
		left_layout.setContentsMargins(0, 0, 0, 0)
		left_layout.setSpacing(10)
		splitter.addWidget(left)

		form = QFormLayout()
		form.setVerticalSpacing(10)
		form.setHorizontalSpacing(12)
		left_layout.addLayout(form)

		self.dataset_combo = QComboBox()
		# Make long dataset names readable.
		self.dataset_combo.setMinimumWidth(360)
		for name in sorted(self._dataset_to_indices.keys()):
			self.dataset_combo.addItem(str(name))
		form.addRow(QLabel(LOCALIZE("ML_PAGE.shap_dataset_label")), self.dataset_combo)

		self.index_spin = CustomSpinBox()
		self.index_spin.setRange(0, 0)
		self.index_spin.setValue(0)
		form.addRow(QLabel(LOCALIZE("ML_PAGE.shap_spectrum_index")), self.index_spin)

		self.count_hint = QLabel("")
		self.count_hint.setStyleSheet("color: #6c757d; font-size: 11px;")
		left_layout.addWidget(self.count_hint)

		self.bg_samples_spin = CustomSpinBox()
		self.bg_samples_spin.setRange(1, 1000)
		self.bg_samples_spin.setValue(int(default_background_samples))
		form.addRow(QLabel(LOCALIZE("ML_PAGE.shap_background_samples")), self.bg_samples_spin)

		self.max_evals_spin = CustomSpinBox()
		self.max_evals_spin.setRange(0, 200000)
		self.max_evals_spin.setValue(int(default_max_evals))
		form.addRow(QLabel(LOCALIZE("ML_PAGE.shap_max_evals")), self.max_evals_spin)

		self.top_k_spin = CustomSpinBox()
		self.top_k_spin.setRange(5, 200)
		self.top_k_spin.setValue(int(default_top_k))
		form.addRow(QLabel(LOCALIZE("ML_PAGE.shap_top_k")), self.top_k_spin)

		self.seed_spin = CustomSpinBox()
		self.seed_spin.setRange(0, 9999999)
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
		run_btn.clicked.connect(self.accept)
		btn_row.addWidget(cancel_btn)
		btn_row.addWidget(run_btn)
		left_layout.addLayout(btn_row)

		# Right panel: preview + metadata
		right = QWidget()
		right_layout = QVBoxLayout(right)
		right_layout.setContentsMargins(0, 0, 0, 0)
		right_layout.setSpacing(10)
		splitter.addWidget(right)

		self.preview_plot = MatplotlibWidget()
		try:
			self.preview_plot.toolbar.setVisible(True)
			self.preview_plot.toolbar.show()
		except Exception:
			pass
		right_layout.addWidget(self.preview_plot, 2)

		self.meta_text = QTextEdit()
		self.meta_text.setReadOnly(True)
		self.meta_text.setStyleSheet(
			"border: 1px solid #dee2e6; border-radius: 6px; padding: 8px; font-family: monospace; font-size: 12px;"
		)
		right_layout.addWidget(self.meta_text, 1)

		# Default sizing: this dialog often needs extra width for long dataset names.
		try:
			self.setMinimumWidth(980)
			self.resize(1120, 680)
		except Exception:
			pass
		try:
			splitter.setStretchFactor(0, 0)
			splitter.setStretchFactor(1, 1)
			splitter.setSizes([380, 740])
		except Exception:
			pass

		# Signals
		self.dataset_combo.currentIndexChanged.connect(self._sync_index_range)
		self.index_spin.valueChanged.connect(self._refresh_preview)

		# Init
		self._sync_index_range()
		self._refresh_preview()

	def _current_dataset(self) -> str:
		name = str(self.dataset_combo.currentText() or "")
		return name or "(unknown)"

	def _indices_for_dataset(self, name: str) -> List[int]:
		return list(self._dataset_to_indices.get(name, []))

	def _sync_index_range(self) -> None:
		name = self._current_dataset()
		indices = self._indices_for_dataset(name)
		count = len(indices)
		max_idx = max(0, count - 1)
		try:
			self.index_spin.setRange(0, max_idx)
		except Exception:
			pass
		try:
			self.count_hint.setText(LOCALIZE("ML_PAGE.shap_dataset_count", name=name, count=count))
		except Exception:
			self.count_hint.setText(f"{name}: {count} spectra")
		# Clamp
		try:
			v = int(self.index_spin.value())
			self.index_spin.setValue(int(max(0, min(v, max_idx))))
		except Exception:
			pass

		self._refresh_preview()

	def _resolve_global_index(self) -> int:
		name = self._current_dataset()
		indices = self._indices_for_dataset(name)
		if not indices:
			return 0
		within = int(self.index_spin.value())
		within = int(max(0, min(within, len(indices) - 1)))
		return int(indices[within])

	def _refresh_preview(self) -> None:
		# Preview plot
		try:
			import matplotlib.pyplot as plt

			gi = self._resolve_global_index()
			x = np.asarray(self._x_explain[gi], dtype=float).reshape(-1)
			axis = self._axis
			if axis is None or axis.shape[0] != x.shape[0]:
				axis = np.arange(x.shape[0], dtype=float)

			fig = plt.Figure(figsize=(7, 3.0))
			ax = fig.add_subplot(111)
			ax.plot(axis, x, color="#1f77b4", lw=1.2, label="Spectrum")
			ax.set_xlabel("Raman shift")
			ax.set_ylabel("Intensity")

			name = self._current_dataset()
			true_label = ""
			try:
				true_label = str(self._y_explain[gi])
			except Exception:
				true_label = ""
			ax.set_title(f"Preview | dataset={name} | idx={int(self.index_spin.value())} | true={true_label}")
			ax.grid(True, alpha=0.2)
			try:
				ax.legend(loc="best", fontsize=9, framealpha=0.85)
			except Exception:
				pass
			fig.tight_layout()

			self.preview_plot.update_plot_with_config(fig, {"figure": {"tight_layout": True}})
		except Exception:
			# Best-effort: keep dialog usable even if preview fails.
			try:
				self.preview_plot.figure.clear()
				self.preview_plot.canvas.draw()
			except Exception:
				pass

		# Metadata
		try:
			name = self._current_dataset()
			meta = PROJECT_MANAGER.get_dataframe_metadata(name) if hasattr(PROJECT_MANAGER, "get_dataframe_metadata") else None
			meta = meta if isinstance(meta, dict) else {}
			self.meta_text.setPlainText(json.dumps(meta, ensure_ascii=False, indent=2) if meta else "")
		except Exception:
			self.meta_text.setPlainText("")

	def selection(self) -> SHAPParameterSelection:
		name = self._current_dataset()
		indices = self._indices_for_dataset(name)
		within = int(self.index_spin.value())
		within = int(max(0, min(within, len(indices) - 1))) if indices else 0
		global_index = self._resolve_global_index()
		return SHAPParameterSelection(
			source=str(self._source),
			dataset_name=str(name),
			index_within_dataset=int(within),
			global_index=int(global_index),
			background_samples=int(self.bg_samples_spin.value()),
			max_evals=int(self.max_evals_spin.value()),
			top_k=int(self.top_k_spin.value()),
			random_state=int(self.seed_spin.value()),
		)
