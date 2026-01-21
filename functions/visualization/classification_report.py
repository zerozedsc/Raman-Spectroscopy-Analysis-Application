from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import classification_report


def create_classification_report_table(
	*,
	y_true: np.ndarray,
	y_pred: np.ndarray,
	labels: List[str] | None = None,
	zero_division: int = 0,
) -> List[Dict[str, object]]:
	"""Return a table-friendly representation of sklearn's classification report.

	The ML page renders this as a QTableWidget.

	Returns:
		List of rows like:
		[{"class": "MM", "precision": 0.9, "recall": 0.8, "f1-score": 0.85, "support": 120}, ...]
		Includes summary rows: accuracy, macro avg, weighted avg.
	"""
	rep = classification_report(
		y_true,
		y_pred,
		labels=labels,
		output_dict=True,
		zero_division=zero_division,
	)

	rows: List[Dict[str, object]] = []

	# class-specific rows
	for key, metrics in rep.items():
		if key in {"accuracy", "macro avg", "weighted avg"}:
			continue
		if not isinstance(metrics, dict):
			continue
		rows.append(
			{
				"class": str(key),
				"precision": round(float(metrics.get("precision", 0.0)), 4),
				"recall": round(float(metrics.get("recall", 0.0)), 4),
				"f1-score": round(float(metrics.get("f1-score", 0.0)), 4),
				"support": int(metrics.get("support", 0) or 0),
			}
		)

	# summary rows
	acc = rep.get("accuracy", None)
	if acc is not None:
		rows.append(
			{
				"class": "accuracy",
				"precision": "",
				"recall": "",
				"f1-score": round(float(acc), 4),
				"support": int(sum(int(r.get("support", 0) or 0) for r in rows if str(r.get("class")) not in {"accuracy", "macro avg", "weighted avg"})),
			}
		)

	for key in ("macro avg", "weighted avg"):
		metrics = rep.get(key)
		if isinstance(metrics, dict):
			rows.append(
				{
					"class": key,
					"precision": round(float(metrics.get("precision", 0.0)), 4),
					"recall": round(float(metrics.get("recall", 0.0)), 4),
					"f1-score": round(float(metrics.get("f1-score", 0.0)), 4),
					"support": int(metrics.get("support", 0) or 0),
				}
			)

	return rows
