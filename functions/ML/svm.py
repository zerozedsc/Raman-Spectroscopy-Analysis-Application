from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class ClassificationResult:
	model: Any
	accuracy: float
	confusion_matrix: np.ndarray
	report: str
	y_pred: np.ndarray
	proba: np.ndarray | None


def train_svm_classifier(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray | None = None,
	y_test: np.ndarray | None = None,
	*,
	C: float = 1.0,
	kernel: str = "rbf",
	gamma: str | float = "scale",
	degree: int = 3,
	probability: bool = True,
	class_weight: str | dict | None = "balanced",
) -> ClassificationResult:
	"""Train an SVM classifier (SVC) with standard scaling.

	If X_test/y_test are provided, metrics are computed on the test set.
	Otherwise, metrics are computed on the training set (backward-compatible).
	"""
	X_train = np.asarray(X_train)
	y_train = np.asarray(y_train)
	X_eval = np.asarray(X_test) if X_test is not None else X_train
	y_eval = np.asarray(y_test) if y_test is not None else y_train

	pipe = Pipeline(
		[
			("scaler", StandardScaler()),
			(
				"clf",
				SVC(
					C=C,
					kernel=kernel,
					gamma=gamma,
					degree=degree,
					probability=probability,
					class_weight=class_weight,
				),
			),
		]
	)
	pipe.fit(X_train, y_train)
	y_pred = pipe.predict(X_eval)
	proba = (
		pipe.predict_proba(X_eval)
		if probability and hasattr(pipe, "predict_proba")
		else None
	)

	acc = float(accuracy_score(y_eval, y_pred))
	cm = confusion_matrix(y_eval, y_pred)
	report = classification_report(y_eval, y_pred)

	return ClassificationResult(
		model=pipe,
		accuracy=acc,
		confusion_matrix=cm,
		report=report,
		y_pred=y_pred,
		proba=proba,
	)
