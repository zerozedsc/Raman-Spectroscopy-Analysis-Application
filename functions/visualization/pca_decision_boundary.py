from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def _build_classifier(*, model_key: str, model_params: Dict) -> object:
	model_key = str(model_key)
	params = dict(model_params or {})

	if model_key == "logistic_regression":
		# Use params compatible with sklearn LogisticRegression
		return LogisticRegression(
			C=float(params.get("C", 1.0)),
			max_iter=int(params.get("max_iter", 200)),
			solver=str(params.get("solver", "lbfgs")),
			class_weight=params.get("class_weight", "balanced"),
		)
	if model_key == "svm":
		gamma = params.get("gamma", "scale")
		return SVC(
			C=float(params.get("C", 1.0)),
			kernel=str(params.get("kernel", "rbf")),
			gamma=gamma,
			degree=int(params.get("degree", 3)),
			probability=False,
			class_weight=params.get("class_weight", "balanced"),
		)
	if model_key == "random_forest":
		return RandomForestClassifier(
			n_estimators=int(params.get("n_estimators", 200)),
			max_depth=params.get("max_depth", None),
			random_state=params.get("random_state", 42),
			class_weight=params.get("class_weight", "balanced"),
		)

	# Fallback: logistic regression
	return LogisticRegression(max_iter=200)


def create_pca_decision_boundary_figure(
	*,
	X: np.ndarray,
	y: np.ndarray,
	model_key: str,
	model_params: Dict,
	title: str = "PCA + Decision Boundary",
	figsize: Tuple[int, int] = (8, 5),
	grid_step: float = 0.2,
	alpha_region: float = 0.25,
) -> plt.Figure:
	"""Create a PCA scatter plot with a decision boundary background.

	Important: The boundary is computed using a *visualization* classifier trained
	in PCA(2) space. This does not change the main trained model; it's for
	interpretability.
	"""
	X = np.asarray(X, dtype=float)
	y = np.asarray(y, dtype=object)
	if X.ndim != 2:
		raise ValueError("X must be 2D")
	if X.shape[0] != y.shape[0]:
		raise ValueError("X and y must have the same number of samples")

	# Encode labels to integers for contouring
	labels = [str(v) for v in sorted(set(map(str, y)))]
	label_to_int = {lab: i for i, lab in enumerate(labels)}
	y_int = np.asarray([label_to_int.get(str(v), -1) for v in y], dtype=int)
	mask = y_int >= 0
	X = X[mask]
	y_int = y_int[mask]

	pipe = Pipeline(
		[
			("scaler", StandardScaler()),
			("pca", PCA(n_components=2, random_state=0)),
			("clf", _build_classifier(model_key=model_key, model_params=model_params)),
		]
	)
	pipe.fit(X, y_int)

	# Transform for scatter
	X2 = pipe.named_steps["pca"].transform(pipe.named_steps["scaler"].transform(X))
	pc1 = X2[:, 0]
	pc2 = X2[:, 1]

	# Grid for decision regions
	x_min, x_max = pc1.min() - 1.0, pc1.max() + 1.0
	y_min, y_max = pc2.min() - 1.0, pc2.max() + 1.0
	xx, yy = np.meshgrid(
		np.arange(x_min, x_max, grid_step),
		np.arange(y_min, y_max, grid_step),
	)
	grid = np.c_[xx.ravel(), yy.ravel()]
	# Predict in PCA space: use clf directly
	Z = pipe.named_steps["clf"].predict(grid)
	Z = Z.reshape(xx.shape)

	# Colors
	base = plt.cm.get_cmap("tab10", max(2, len(labels)))
	region_cmap = ListedColormap([base(i) for i in range(len(labels))])

	fig = plt.Figure(figsize=figsize)
	ax = fig.add_subplot(111)

	# IMPORTANT:
	# MatplotlibWidget's copy logic reliably supports AxesImage (imshow).
	# Using imshow here yields a smoother/cleaner boundary than a giant scatter.
	ax.imshow(
		Z,
		origin="lower",
		extent=(x_min, x_max, y_min, y_max),
		cmap=region_cmap,
		alpha=alpha_region,
		interpolation="nearest",
		aspect="auto",
	)

	# Plot points
	for i, lab in enumerate(labels):
		m = y_int == i
		ax.scatter(
			pc1[m],
			pc2[m],
			s=18,
			alpha=0.85,
			label=lab,
			edgecolors="none",
		)

	# Centroids
	for i, lab in enumerate(labels):
		m = y_int == i
		if not np.any(m):
			continue
		cx = float(np.mean(pc1[m]))
		cy = float(np.mean(pc2[m]))
		ax.scatter([cx], [cy], s=70, marker="x", color="black")
		ax.annotate(f"{lab} centroid", (cx, cy), xytext=(5, 5), textcoords="offset points", fontsize=9)

	ax.set_title(title)
	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	ax.grid(True, alpha=0.25)
	ax.legend(loc="best", fontsize=9)
	fig.tight_layout()
	return fig
