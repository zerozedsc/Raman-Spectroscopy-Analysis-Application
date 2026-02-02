from __future__ import annotations

import traceback
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PySide6.QtCore import QThread, Signal

from configs.configs import create_logs


def _load_peak_assignments() -> Dict[int, Dict[str, object]]:
	"""Load peak assignment map from assets/data/raman_peaks.json.

	Returns a dict keyed by integer wavenumber.
	"""
	try:
		root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
		path = os.path.join(root, "assets", "data", "raman_peaks.json")
		with open(path, "r", encoding="utf-8") as f:
			raw = json.load(f)
		out: Dict[int, Dict[str, object]] = {}
		if isinstance(raw, dict):
			for k, v in raw.items():
				try:
					ki = int(float(k))
				except Exception:
					continue
				if isinstance(v, dict):
					out[ki] = v
		return out
	except Exception:
		return {}


_PEAK_ASSIGNMENTS_CACHE: Optional[Dict[int, Dict[str, object]]] = None


def _get_peak_assignment(wavenumber: float) -> tuple[str, str]:
	"""Return (assignment, reference_number) for a wavenumber, if known."""
	global _PEAK_ASSIGNMENTS_CACHE
	if _PEAK_ASSIGNMENTS_CACHE is None:
		_PEAK_ASSIGNMENTS_CACHE = _load_peak_assignments()
	lookup = _PEAK_ASSIGNMENTS_CACHE
	if not lookup:
		return "", ""
	try:
		w = int(round(float(wavenumber)))
	except Exception:
		return "", ""
	# Try exact then nearby (axis might be non-integer)
	for cand in (w, w - 1, w + 1, w - 2, w + 2):
		v = lookup.get(cand)
		if isinstance(v, dict):
			assignment = str(v.get("assignment") or "")
			reference = str(v.get("reference_number") or "")
			return assignment, reference
	return "", ""


@dataclass(frozen=True)
class SHAPResult:
	"""Per-spectrum SHAP explanation result used by the dialog UI."""

	spectrum_figure: Any
	shap_figure: Any
	contributor_rows: List[Dict[str, object]]
	axis: np.ndarray
	spectrum: np.ndarray
	shap_values: np.ndarray
	used_background: int
	selected_index: int
	source: str
	true_label: str
	pred_label: str
	pred_proba: float
	base_value: float
	approx_output: float
	max_evals_used: int
	dataset_name: str = ""


class SHAPExplainThread(QThread):
	"""Background worker for SHAP per-spectrum explanation.

	This runs on the *already-trained* model and computes SHAP values for a
	*single selected spectrum*, returning plots + a contributor table.
	"""

	progress_updated = Signal(int)
	status_updated = Signal(str)
	completed = Signal(object)  # SHAPResult
	error = Signal(str)

	def __init__(
		self,
		*,
		model: Any,
		x_train: np.ndarray,
		x_explain: np.ndarray,
		y_explain: np.ndarray,
		dataset_names: Optional[Sequence[str]] = None,
		class_labels: Optional[Sequence[str]] = None,
		source: str = "test",
		selected_index: int = 0,
		feature_axis: Optional[np.ndarray] = None,
		background_samples: int = 30,
		max_evals: int = 0,
		top_k: int = 12,
		random_state: int = 0,
		parent=None,
	):
		super().__init__(parent)
		self._model = model
		self._x_train = np.asarray(x_train, dtype=float)
		self._x_explain = np.asarray(x_explain, dtype=float)
		self._y_explain = np.asarray(y_explain, dtype=object)
		self._dataset_names = list(dataset_names) if dataset_names is not None else None
		self._class_labels = list(class_labels) if class_labels is not None else None
		self._source = str(source or "test")
		self._selected_index = int(selected_index)
		self._feature_axis = None if feature_axis is None else np.asarray(feature_axis)
		self._background_samples = int(background_samples)
		self._max_evals = int(max_evals)
		self._top_k = int(top_k)
		self._random_state = int(random_state)

	def run(self):
		try:
			def _check_cancel() -> bool:
				if self.isInterruptionRequested():
					try:
						self.status_updated.emit("Cancelled")
					except Exception:
						pass
					return True
				return False

			self.status_updated.emit("Preparing SHAP...")
			self.progress_updated.emit(5)
			if _check_cancel():
				return

			model = self._model
			if not hasattr(model, "predict_proba"):
				raise ValueError("Model does not support predict_proba; SHAP needs probabilistic output.")

			rng = np.random.default_rng(self._random_state)

			x_explain_all = self._x_explain
			if x_explain_all.ndim != 2:
				raise ValueError("x_explain must be 2D")
			if x_explain_all.shape[0] < 1:
				raise ValueError("x_explain is empty")
			idx_sel = int(np.clip(self._selected_index, 0, x_explain_all.shape[0] - 1))
			x_sample = np.asarray(x_explain_all[idx_sel], dtype=float)
			if x_sample.ndim != 1:
				raise ValueError("Selected sample must be 1D")
			if _check_cancel():
				return

			# Sample background from training set
			x_bg = self._x_train
			if x_bg.ndim != 2:
				raise ValueError("x_train must be 2D")
			bg_n = max(1, min(int(self._background_samples), x_bg.shape[0]))
			if x_bg.shape[0] > bg_n:
				idx_bg = rng.choice(x_bg.shape[0], bg_n, replace=False)
				x_bg = x_bg[idx_bg]
			if _check_cancel():
				return

			self.status_updated.emit("Loading SHAP library...")
			self.progress_updated.emit(12)
			if _check_cancel():
				return

			# Import SHAP lazily (can be heavy)
			import shap  # type: ignore
			if _check_cancel():
				return

			def predict_fn(X: np.ndarray) -> np.ndarray:
				X = np.asarray(X, dtype=float)
				return np.asarray(model.predict_proba(X), dtype=float)

			self.status_updated.emit("Building explainer (may take time)...")
			self.progress_updated.emit(20)
			if _check_cancel():
				return

			# Use the unified Explainer interface; it chooses a suitable method.
			explainer = shap.Explainer(predict_fn, x_bg)
			if _check_cancel():
				return

			# NOTE:
			# When SHAP selects the permutation explainer (common for generic predict_fn),
			# it enforces: max_evals >= 2 * num_features + 1.
			# Otherwise it errors with: "max_evals=500 is too low...".
			n_features_for_evals = int(x_sample.shape[0])
			min_required_evals = 2 * n_features_for_evals + 1
			requested = int(self._max_evals)
			if requested > 0:
				max_evals = max(requested, min_required_evals)
			else:
				# Keep prior default behavior (500) but raise to SHAP's required minimum.
				max_evals = max(500, min_required_evals)

			create_logs(
				"SHAPExplainThread",
				"config",
				f"[DEBUG] SHAP max_evals resolved: requested={requested} min_required={min_required_evals} used={max_evals}",
				status="debug",
			)

			self.status_updated.emit("Computing SHAP values...")
			self.progress_updated.emit(55)
			if _check_cancel():
				return
			X_one = x_sample.reshape(1, -1)
			try:
				shap_values = explainer(X_one, max_evals=max_evals)
			except TypeError:
				# Some explainer types may not accept max_evals; fall back.
				shap_values = explainer(X_one)
			if _check_cancel():
				return

			values = getattr(shap_values, "values", None)
			if values is None:
				raise ValueError("SHAP returned no values")
			values = np.asarray(values)

			# Resolve predicted class / labels
			proba = np.asarray(model.predict_proba(X_one), dtype=float).reshape(-1)
			pred_idx = int(np.argmax(proba)) if proba.size else 0
			pred_proba = float(proba[pred_idx]) if pred_idx < proba.size else 0.0

			classes = getattr(model, "classes_", None)
			# Prefer configured display labels when available.
			if self._class_labels is not None and len(self._class_labels) == proba.size:
				pred_label = str(self._class_labels[pred_idx])
			elif classes is not None and len(classes) == proba.size:
				pred_label = str(classes[pred_idx])
			else:
				pred_label = str(pred_idx)

			def _map_label(value: object) -> str:
				"""Map a raw label (e.g., 0/1) to the configured display label when possible."""
				try:
					v_str = str(value)
				except Exception:
					v_str = ""

				# Best case: explicit classes_ alignment.
				try:
					if (
						classes is not None
						and self._class_labels is not None
						and len(classes) == len(self._class_labels)
						and len(classes) > 0
					):
						mapping = {str(classes[i]): str(self._class_labels[i]) for i in range(len(classes))}
						if v_str in mapping:
							return mapping[v_str]
				except Exception:
					pass

				# Fallback: treat raw label as index into class_labels.
				try:
					if self._class_labels is not None:
						vi = int(value)  # type: ignore[arg-type]
						if 0 <= vi < len(self._class_labels):
							return str(self._class_labels[vi])
				except Exception:
					pass

				return v_str

			true_label = ""
			try:
				if 0 <= idx_sel < self._y_explain.shape[0]:
					true_label = _map_label(self._y_explain[idx_sel])
			except Exception:
				true_label = ""

			dataset_name = ""
			try:
				if self._dataset_names is not None and 0 <= idx_sel < len(self._dataset_names):
					dataset_name = str(self._dataset_names[idx_sel])
			except Exception:
				dataset_name = ""

			# values can be:
			# - (1, n_features)
			# - (1, n_features, n_outputs)
			if values.ndim == 2:
				shap_vec = np.asarray(values[0], dtype=float)
				base_vals = getattr(shap_values, "base_values", None)
				base_value = float(np.asarray(base_vals).reshape(-1)[0]) if base_vals is not None else 0.0
			elif values.ndim == 3:
				shap_vec = np.asarray(values[0, :, pred_idx], dtype=float)
				base_vals = getattr(shap_values, "base_values", None)
				base_flat = np.asarray(base_vals).reshape(-1) if base_vals is not None else np.asarray([0.0])
				base_value = float(base_flat[pred_idx]) if pred_idx < base_flat.size else float(base_flat[0])
			else:
				raise ValueError(f"Unexpected SHAP values shape: {values.shape}")

			if shap_vec.ndim != 1:
				raise ValueError("SHAP vector is not 1D")

			approx_output = float(base_value + float(np.sum(shap_vec)))

			axis = self._feature_axis
			if axis is None or axis.shape[0] != shap_vec.shape[0]:
				axis = np.arange(shap_vec.shape[0], dtype=float)

			k = max(1, min(int(self._top_k), int(shap_vec.shape[0])))
			idx_pos = np.argsort(-shap_vec)[:k]
			idx_neg = np.argsort(shap_vec)[:k]
			idx_top = np.unique(np.concatenate([idx_pos, idx_neg]))

			rows: List[Dict[str, object]] = []
			for i in idx_top.tolist():
				wv = float(axis[int(i)])
				val = float(shap_vec[int(i)])
				assignment, ref = _get_peak_assignment(wv)
				rows.append(
					{
						"index": int(i),
						"wavenumber": wv,
						"shap_value": val,
						"abs_shap": abs(val),
						"sign": "+" if val >= 0 else "-",
						"assignment": assignment,
						"reference": ref,
					}
				)
			rows.sort(key=lambda r: float(r.get("abs_shap", 0.0)), reverse=True)
			if _check_cancel():
				return

			self.status_updated.emit("Rendering plots...")
			self.progress_updated.emit(85)
			if _check_cancel():
				return

			import matplotlib.pyplot as plt

			spectrum_fig = plt.Figure(figsize=(9, 3.2))
			ax0 = spectrum_fig.add_subplot(111)
			ax0.plot(axis, x_sample, color="#1f77b4", lw=1.3, label="Spectrum")
			ax0.set_xlabel("Raman shift")
			ax0.set_ylabel("Intensity")
			title_bits = [f"SHAP explanation ({self._source})", f"idx={idx_sel}"]
			if dataset_name:
				title_bits.append(f"dataset={dataset_name}")
			if true_label:
				title_bits.append(f"true={true_label}")
			ax0.set_title(" | ".join(title_bits))
			# Mark a few strongest contributors (and provide a legend)
			pos_labeled = False
			neg_labeled = False
			for r in rows[: min(10, len(rows))]:
				wv = float(r.get("wavenumber", 0.0))
				val = float(r.get("shap_value", 0.0))
				if val >= 0:
					lbl = "Positive contribution" if not pos_labeled else None
					pos_labeled = True
					ax0.axvline(wv, color="#d62728", alpha=0.25, lw=1.0, label=lbl)
				else:
					lbl = "Negative contribution" if not neg_labeled else None
					neg_labeled = True
					ax0.axvline(wv, color="#1f77b4", alpha=0.25, lw=1.0, label=lbl)
			ax0.grid(True, alpha=0.2)
			try:
				ax0.legend(loc="best", fontsize=9, framealpha=0.85)
			except Exception:
				pass
			spectrum_fig.tight_layout()

			shap_fig = plt.Figure(figsize=(9, 3.2))
			ax1 = shap_fig.add_subplot(111)
			# Bar plot across axis (color by sign)
			try:
				step = float(np.median(np.diff(axis))) if axis.size > 1 else 1.0
			except Exception:
				step = 1.0
			# SHAP convention: red = positive contribution, blue = negative contribution.
			colors = np.where(shap_vec >= 0, "#d62728", "#1f77b4")
			ax1.bar(axis, shap_vec, width=step * 0.85, color=colors, alpha=0.75)
			ax1.axhline(0, color="#6c757d", lw=1.0)
			ax1.set_xlabel("Raman shift")
			ax1.set_ylabel("SHAP value")
			ax1.set_title(
				f"Contributions to predicted '{pred_label}' (p={pred_proba:.3f}) | base={base_value:.3f} + sum={float(np.sum(shap_vec)):.3f} ≈ {approx_output:.3f}"
			)
			ax1.grid(True, alpha=0.2)
			# Add a small legend for color meaning (proxy artists).
			try:
				from matplotlib.patches import Patch
				ax1.legend(
					handles=[
						Patch(facecolor="#d62728", alpha=0.75, label="Positive"),
						Patch(facecolor="#1f77b4", alpha=0.75, label="Negative"),
					],
					loc="best",
					fontsize=9,
					framealpha=0.85,
				)
			except Exception:
				pass
			shap_fig.tight_layout()

			res = SHAPResult(
				spectrum_figure=spectrum_fig,
				shap_figure=shap_fig,
				contributor_rows=rows,
				axis=np.asarray(axis, dtype=float),
				spectrum=np.asarray(x_sample, dtype=float),
				shap_values=np.asarray(shap_vec, dtype=float),
				used_background=int(x_bg.shape[0]),
				selected_index=int(idx_sel),
				source=str(self._source),
				true_label=true_label,
				pred_label=pred_label,
				pred_proba=float(pred_proba),
				base_value=float(base_value),
				approx_output=float(approx_output),
				max_evals_used=int(max_evals),
				dataset_name=str(dataset_name),
			)

			self.progress_updated.emit(100)
			self.status_updated.emit("Done")
			self.completed.emit(res)

			create_logs(
				"SHAPExplainThread",
				"done",
				f"[DEBUG] SHAP computed: source={res.source} idx={res.selected_index} bg={res.used_background} top_k={len(res.contributor_rows)} max_evals={res.max_evals_used}",
				status="debug",
			)

		except Exception as e:
			tb = traceback.format_exc()
			create_logs(
				"SHAPExplainThread",
				"error",
				f"SHAP computation failed: {e}\n{tb}",
				status="error",
			)
			self.error.emit(str(e))
