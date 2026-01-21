from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClassificationResult:
	model: Any
	accuracy: float
	confusion_matrix: np.ndarray
	report: str
	y_pred: np.ndarray
	proba: np.ndarray | None


def train_logistic_regression(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray | None = None,
	y_test: np.ndarray | None = None,
	*,
	C: float = 1.0,
	max_iter: int = 200,
	class_weight: str | dict | None = "balanced",
	solver: str = "lbfgs",
	n_jobs: int | None = None,
	predict_proba: bool = True,
) -> ClassificationResult:
	"""Train a logistic regression classifier with standard scaling.

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
				LogisticRegression(
					C=C,
					max_iter=max_iter,
					class_weight=class_weight,
					solver=solver,
					n_jobs=n_jobs,
				),
			),
		]
	)
	pipe.fit(X_train, y_train)
	y_pred = pipe.predict(X_eval)
	proba = (
		pipe.predict_proba(X_eval)
		if predict_proba and hasattr(pipe, "predict_proba")
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
