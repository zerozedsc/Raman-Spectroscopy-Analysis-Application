from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass(frozen=True)
class ClassificationResult:
	model: Any
	accuracy: float
	confusion_matrix: np.ndarray
	report: str
	y_pred: np.ndarray
	proba: np.ndarray | None


def train_random_forest_classifier(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray | None = None,
	y_test: np.ndarray | None = None,
	*,
	n_estimators: int = 200,
	max_depth: int | None = None,
	random_state: int | None = 0,
	class_weight: str | dict | None = "balanced",
) -> ClassificationResult:
	"""Train a RandomForest classifier.

	If X_test/y_test are provided, metrics are computed on the test set.
	Otherwise, metrics are computed on the training set (backward-compatible).
	"""
	X_train = np.asarray(X_train)
	y_train = np.asarray(y_train)
	X_eval = np.asarray(X_test) if X_test is not None else X_train
	y_eval = np.asarray(y_test) if y_test is not None else y_train

	model = RandomForestClassifier(
		n_estimators=n_estimators,
		max_depth=max_depth,
		random_state=random_state,
		class_weight=class_weight,
	)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_eval)
	proba = model.predict_proba(X_eval) if hasattr(model, "predict_proba") else None

	acc = float(accuracy_score(y_eval, y_pred))
	cm = confusion_matrix(y_eval, y_pred)
	report = classification_report(y_eval, y_pred)

	return ClassificationResult(
		model=model,
		accuracy=acc,
		confusion_matrix=cm,
		report=report,
		y_pred=y_pred,
		proba=proba,
	)
