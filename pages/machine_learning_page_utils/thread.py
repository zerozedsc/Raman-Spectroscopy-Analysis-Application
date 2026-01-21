from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal

from configs.configs import create_logs
from functions.ML import prepare_train_test_split
from functions.ML.linear_regression import train_linear_regression_classifier
from functions.ML.logistic_regression import train_logistic_regression
from functions.ML.random_forest import train_random_forest_classifier
from functions.ML.svm import train_svm_classifier


@dataclass(frozen=True)
class MLTrainingOutput:
	model: Any
	split: Any
	model_key: str
	model_params: Dict[str, Any]
	accuracy: float
	report: str
	y_pred: np.ndarray
	proba: Optional[np.ndarray]
	feature_importances: Optional[np.ndarray]


class MLTrainingThread(QThread):
	"""Background worker for ML training + evaluation (train/test)."""

	progress_updated = Signal(int)
	status_updated = Signal(str)
	training_completed = Signal(object)  # MLTrainingOutput
	training_error = Signal(str)

	def __init__(
		self,
		*,
		raman_data: Mapping[str, pd.DataFrame],
		group_assignments: Mapping[str, str],
		train_ratio: float,
		split_mode: str,
		random_state: int,
		model_key: str,
		model_params: Dict[str, Any],
		parent=None,
	):
		super().__init__(parent)
		self._raman_data = raman_data
		self._group_assignments = dict(group_assignments)
		self._train_ratio = float(train_ratio)
		self._split_mode = str(split_mode)
		self._random_state = int(random_state)
		self._model_key = str(model_key)
		self._model_params = dict(model_params)

	def run(self):
		try:
			create_logs(
				"MLTrainingThread",
				"start",
				f"[DEBUG] Starting training model={self._model_key} split_mode={self._split_mode} train_ratio={self._train_ratio}",
				status="debug",
			)
			self.status_updated.emit("Preparing data...")
			self.progress_updated.emit(10)

			split = prepare_train_test_split(
				raman_data=self._raman_data,
				group_assignments=self._group_assignments,
				train_ratio=self._train_ratio,
				split_mode=self._split_mode,
				random_state=self._random_state,
			)

			self.status_updated.emit("Training model...")
			self.progress_updated.emit(55)

			if self._model_key == "logistic_regression":
				res = train_logistic_regression(
					split.X_train,
					split.y_train,
					split.X_test,
					split.y_test,
					**self._model_params,
				)
				feature_importances = _extract_feature_importances_from_pipeline(res.model)

			elif self._model_key == "linear_regression":
				res = train_linear_regression_classifier(
					split.X_train,
					split.y_train,
					split.X_test,
					split.y_test,
					**self._model_params,
				)
				feature_importances = _extract_feature_importances_from_pipeline(res.model)

			elif self._model_key == "svm":
				res = train_svm_classifier(
					split.X_train,
					split.y_train,
					split.X_test,
					split.y_test,
					**self._model_params,
				)
				feature_importances = _extract_feature_importances_from_pipeline(res.model)

			elif self._model_key == "random_forest":
				res = train_random_forest_classifier(
					split.X_train,
					split.y_train,
					split.X_test,
					split.y_test,
					**self._model_params,
				)
				feature_importances = getattr(res.model, "feature_importances_", None)

			else:
				raise ValueError(f"Unknown model_key: {self._model_key}")

			self.status_updated.emit("Finalizing results...")
			self.progress_updated.emit(90)

			out = MLTrainingOutput(
				model=res.model,
				split=split,
				model_key=self._model_key,
				model_params=self._model_params,
				accuracy=float(res.accuracy),
				report=str(res.report),
				y_pred=np.asarray(res.y_pred),
				proba=np.asarray(res.proba) if res.proba is not None else None,
				feature_importances=np.asarray(feature_importances)
				if feature_importances is not None
				else None,
			)

			self.progress_updated.emit(100)
			self.status_updated.emit("Done")
			self.training_completed.emit(out)

			create_logs(
				"MLTrainingThread",
				"done",
				f"[DEBUG] Training completed acc={out.accuracy:.4f}",
				status="debug",
			)

		except Exception as e:
			tb = traceback.format_exc()
			create_logs(
				"MLTrainingThread",
				"error",
				f"ML training failed: {e}\n{tb}",
				status="error",
			)
			self.training_error.emit(str(e))


def _extract_feature_importances_from_pipeline(model: Any) -> Optional[np.ndarray]:
	"""Try to derive a 1D importance vector from sklearn Pipeline models.

	- LogisticRegression: uses mean absolute coefficient across classes
	- Linear SVM: uses mean absolute coefficient across classes
	"""
	try:
		clf = getattr(model, "named_steps", {}).get("clf")
		if clf is None:
			clf = model

		coef = getattr(clf, "coef_", None)
		if coef is None:
			return None

		coef = np.asarray(coef, dtype=float)
		if coef.ndim == 1:
			return np.abs(coef)
		if coef.ndim == 2:
			return np.mean(np.abs(coef), axis=0)
		return None
	except Exception:
		return None
