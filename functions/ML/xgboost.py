from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


@dataclass(frozen=True)
class ClassificationResult:
	model: Any
	accuracy: float
	confusion_matrix: np.ndarray
	report: str
	y_pred: np.ndarray
	proba: np.ndarray | None


class XGBLabelModel:
	"""A small wrapper that keeps string labels for XGBoost models.

	XGBoost's sklearn wrapper typically expects encoded labels. This wrapper:
	- encodes labels for training
	- decodes predictions back to the original labels
	- forwards predict_proba and feature_importances_
	
	This keeps the rest of the app (evaluation dialogs, reports) consistent.
	"""

	def __init__(self, *, model: Any, encoder: LabelEncoder):
		self._model = model
		self._encoder = encoder

	def predict(self, X: np.ndarray) -> np.ndarray:
		pred = self._model.predict(X)
		pred = np.asarray(pred)
		try:
			pred_int = pred.astype(int)
			y = self._encoder.inverse_transform(pred_int)
			return np.asarray(y, dtype=object)
		except Exception:
			# If the underlying model already returns strings/objects, keep as-is.
			return np.asarray(pred, dtype=object)

	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		return np.asarray(self._model.predict_proba(X), dtype=float)

	def fit(self, X: np.ndarray, y: np.ndarray):
		return self._model.fit(X, y)

	def __getattr__(self, name: str):
		# Delegate everything else.
		return getattr(self._model, name)

	@property
	def feature_importances_(self) -> Optional[np.ndarray]:
		return getattr(self._model, "feature_importances_", None)


def train_xgboost_classifier(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray | None = None,
	y_test: np.ndarray | None = None,
	*,
	n_estimators: int = 300,
	max_depth: int = 6,
	learning_rate: float = 0.1,
	subsample: float = 0.8,
	colsample_bytree: float = 0.8,
	reg_lambda: float = 1.0,
	random_state: int | None = 0,
) -> ClassificationResult:
	"""Train an XGBoost classifier.

	If X_test/y_test are provided, metrics are computed on the test set.
	Otherwise, metrics are computed on the training set (backward-compatible).
	
	Note: xgboost is imported lazily so the app can still start if the dependency
	is missing (the UI will hide this method when unavailable).
	"""
	try:
		from xgboost import XGBClassifier  # type: ignore
	except Exception as e:
		raise ImportError("xgboost is not installed") from e

	X_train = np.asarray(X_train)
	y_train = np.asarray(y_train)
	X_eval = np.asarray(X_test) if X_test is not None else X_train
	y_eval = np.asarray(y_test) if y_test is not None else y_train

	# Encode labels so XGBoost sees numeric classes, but keep string labels outward.
	enc = LabelEncoder()
	enc.fit(np.asarray(list(y_train) + list(y_eval), dtype=object))
	y_train_enc = enc.transform(np.asarray(y_train, dtype=object))

	# Choose objective based on number of classes
	n_classes = int(len(enc.classes_))
	if n_classes <= 2:
		objective = "binary:logistic"
		model = XGBClassifier(
			n_estimators=int(n_estimators),
			max_depth=int(max_depth),
			learning_rate=float(learning_rate),
			subsample=float(subsample),
			colsample_bytree=float(colsample_bytree),
			reg_lambda=float(reg_lambda),
			objective=objective,
			random_state=None if random_state is None else int(random_state),
			eval_metric="logloss",
		)
	else:
		objective = "multi:softprob"
		model = XGBClassifier(
			n_estimators=int(n_estimators),
			max_depth=int(max_depth),
			learning_rate=float(learning_rate),
			subsample=float(subsample),
			colsample_bytree=float(colsample_bytree),
			reg_lambda=float(reg_lambda),
			objective=objective,
			num_class=n_classes,
			random_state=None if random_state is None else int(random_state),
			eval_metric="mlogloss",
		)

	model.fit(X_train, y_train_enc)
	wrapped = XGBLabelModel(model=model, encoder=enc)

	y_pred = wrapped.predict(X_eval)
	proba = wrapped.predict_proba(X_eval) if hasattr(wrapped, "predict_proba") else None

	acc = float(accuracy_score(y_eval, y_pred))
	cm = confusion_matrix(y_eval, y_pred, labels=list(enc.classes_))
	report = classification_report(y_eval, y_pred)

	return ClassificationResult(
		model=wrapped,
		accuracy=acc,
		confusion_matrix=cm,
		report=report,
		y_pred=np.asarray(y_pred),
		proba=np.asarray(proba) if proba is not None else None,
	)
