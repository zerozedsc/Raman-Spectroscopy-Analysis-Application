from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	mean_squared_error,
	r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .logistic_regression import ClassificationResult


@dataclass(frozen=True)
class RegressionResult:
	model: Any
	r2: float
	rmse: float
	y_pred: np.ndarray


def train_linear_regression(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray | None = None,
	y_test: np.ndarray | None = None,
	*,
	fit_intercept: bool = True,
	n_jobs: int | None = None,
	positive: bool = False,
) -> RegressionResult:
	"""Train a simple linear regression model.

	Args:
		X: shape (n_samples, n_features)
		y: shape (n_samples,) or (n_samples, n_targets)
	"""
	X_train = np.asarray(X_train)
	y_train = np.asarray(y_train)
	X_eval = np.asarray(X_test) if X_test is not None else X_train
	y_eval = np.asarray(y_test) if y_test is not None else y_train

	model = LinearRegression(
		fit_intercept=fit_intercept,
		n_jobs=n_jobs,
		positive=positive,
	)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_eval)

	# Metrics (if multi-target, sklearn returns arrays; we collapse to float when possible)
	r2 = float(np.mean(r2_score(y_eval, y_pred, multioutput="uniform_average")))
	rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))

	return RegressionResult(model=model, r2=r2, rmse=rmse, y_pred=y_pred)


def train_linear_regression_classifier(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray | None = None,
	y_test: np.ndarray | None = None,
	*,
	fit_intercept: bool = True,
	n_jobs: int | None = None,
	positive: bool = False,
) -> ClassificationResult:
	"""Train Linear Regression as a *classifier* by label-encoding targets.

	This supports the app's current ML workflow (group-based classification)
	while exposing "Linear Regression" in the method chooser.

	Mechanism:
	- Encode labels to integers $0..K-1$
	- Regress the encoded values
	- Round predictions to nearest integer and clip to valid range
	- Decode back to label strings
	"""
	X_train = np.asarray(X_train)
	y_train = np.asarray(y_train, dtype=object)
	X_eval = np.asarray(X_test) if X_test is not None else X_train
	y_eval = np.asarray(y_test, dtype=object) if y_test is not None else y_train

	le = LabelEncoder()
	y_train_enc = le.fit_transform(y_train)
	y_eval_enc = le.transform(y_eval)

	pipe = Pipeline(
		[
			("scaler", StandardScaler()),
			(
				"clf",
				LinearRegression(
					fit_intercept=fit_intercept,
					n_jobs=n_jobs,
					positive=positive,
				),
			),
		]
	)
	pipe.fit(X_train, y_train_enc)
	y_pred_float = np.asarray(pipe.predict(X_eval), dtype=float)

	# Convert regression output -> class indices
	y_pred_enc = np.rint(y_pred_float).astype(int)
	y_pred_enc = np.clip(y_pred_enc, 0, len(le.classes_) - 1)
	y_pred = le.inverse_transform(y_pred_enc)

	acc = float(accuracy_score(y_eval_enc, y_pred_enc))
	cm = confusion_matrix(y_eval_enc, y_pred_enc)
	report = classification_report(y_eval_enc, y_pred_enc, target_names=[str(c) for c in le.classes_])

	return ClassificationResult(
		model=pipe,
		accuracy=acc,
		confusion_matrix=cm,
		report=report,
		y_pred=np.asarray(y_pred, dtype=object),
		proba=None,
	)
