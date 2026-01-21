"""Legacy ML module.

This file preserves the original content of `functions/ML.py` so the codebase can
migrate to a package at `functions/ML/` (required to host multiple modules) without
breaking functionality.

NOTE: This file is auto-generated during the migration. Do not edit manually unless
you know what you're doing.
"""

# === BEGIN COPIED CONTENT FROM functions/ML.py ===
from typing import Any, List, Sequence, Tuple, Union
from functions.configs import *

import json
import os
import traceback

from sklearn.svm import SVC
from sklearn.model_selection import (
	StratifiedKFold,
	train_test_split,
	GridSearchCV,
	cross_val_score,
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from collections import Counter
from datetime import datetime

import numpy as np
import ramanspy as rp
import onnxruntime as ort
import matplotlib.pyplot as plt
import skl2onnx
import onnx
import shap
import time
import sklearn
import pickle


def detect_model_type(model):
	"""
	Detect the actual model type, handling CalibratedClassifierCV wrappers.

	Returns:
		tuple: (is_calibrated, base_model_type, base_model)
	"""
	from sklearn.calibration import CalibratedClassifierCV

	is_calibrated = isinstance(model, CalibratedClassifierCV)

	if is_calibrated:
		# Get the base estimator
		if hasattr(model, "estimators_") and len(model.estimators_) > 0:
			base_model = model.estimators_[0]
		elif hasattr(model, "base_estimator"):
			base_model = model.base_estimator
		else:
			base_model = None

		base_model_type = type(base_model).__name__ if base_model else "Unknown"
	else:
		base_model = model
		base_model_type = type(model).__name__

	return is_calibrated, base_model_type, base_model


def get_model_features(self, model):
	"""
	Get number of features from any model type, handling calibrated models.
	"""
	is_calibrated, base_model_type, base_model = detect_model_type(model)

	# Try different methods to get n_features
	if hasattr(model, "n_features_in_"):
		return model.n_features_in_
	elif base_model and hasattr(base_model, "support_vectors_"):
		return base_model.support_vectors_.shape[1]
	elif base_model and hasattr(base_model, "feature_importances_"):
		return base_model.feature_importances_.shape[0]
	elif hasattr(self, "n_features_in") and self.n_features_in is not None:
		return self.n_features_in
	else:
		raise ValueError("Cannot determine number of features")


def get_unified_predict_function(model):
	"""
	Create a unified prediction function for any model type.
	"""
	is_calibrated, base_model_type, base_model = detect_model_type(model)

	if is_calibrated:
		# Calibrated models always have predict_proba and it's properly calibrated
		def predict_fn(X):
			if X.ndim == 1:
				X = X.reshape(1, -1)
			return model.predict_proba(X)

		return predict_fn
	else:
		# Handle non-calibrated models
		if base_model_type == "RandomForestClassifier":

			def predict_fn(X):
				if X.ndim == 1:
					X = X.reshape(1, -1)
				return model.predict_proba(X)

			return predict_fn
		elif base_model_type == "SVC":
			if hasattr(model, "predict_proba") and model.probability:

				def predict_fn(X):
					if X.ndim == 1:
						X = X.reshape(1, -1)
					return model.predict_proba(X)

				return predict_fn
			else:
				# SVC without probability
				def predict_fn(X):
					if X.ndim == 1:
						X = X.reshape(1, -1)
					decision = model.decision_function(X)
					from scipy.special import expit

					if decision.ndim == 1:
						prob_pos = expit(decision)
						return np.column_stack([1 - prob_pos, prob_pos])
					else:
						exp_scores = np.exp(
							decision - np.max(decision, axis=1, keepdims=True)
						)
						return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

				return predict_fn
		else:
			# Generic fallback
			def predict_fn(X):
				if X.ndim == 1:
					X = X.reshape(1, -1)
				if hasattr(model, "predict_proba"):
					return model.predict_proba(X)
				else:
					predictions = model.predict(X)
					n_classes = len(np.unique(predictions))
					return np.eye(n_classes)[predictions]

			return predict_fn


class RamanML:
	"""
	A class to handle the machine learning pipeline for Raman spectral data.

	Attributes:
	-----------
	region : tuple[int, int]
		The wavenumber region used for cropping spectral data (for reference only).
	"""

	def __init__(self, region: Tuple[int, int] = (1050, 1700)):
		self.region = region
		self._model = None
		self.common_axis = None
		self.n_features_in = None
		self.X_train = None
		self.y_train = None
		self.X_test = None
		self.y_test = None
		self.y_pred = None
		self.y_true = None
		self.x_decision_function = None
		self.y_decision_function = None
		self.z_decision_function = None

	def SVCMODEL(
		self,
		C=1.0,
		kernel="rbf",
		degree=3,
		gamma="scale",
		coef0=0.0,
		shrinking=True,
		probability=False,
		tol=0.001,
		cache_size=200,
		class_weight="balanced",
		verbose=False,
		max_iter=-1,
		decision_function_shape="ovr",
		break_ties=False,
		random_state=None,
	) -> SVC:
		"""
		Set the SVC model to be used for training.

		Parameters:
			All parameters are passed to sklearn.svm.SVC.

		Returns:
			SVC
		"""
		return SVC(
			C=C,
			kernel=kernel,
			degree=degree,
			gamma=gamma,
			coef0=coef0,
			shrinking=shrinking,
			probability=probability,
			tol=tol,
			cache_size=cache_size,
			class_weight=class_weight,
			verbose=verbose,
			max_iter=max_iter,
			decision_function_shape=decision_function_shape,
			break_ties=break_ties,
			random_state=random_state,
		)

	def train_svc(
		self,
		normal_data: Tuple[List[rp.SpectralContainer], str],
		disease_data: Tuple[List[rp.SpectralContainer], str],
		test_size: float = 0.3,
		param_search: bool = True,
		random_state: int = 42,
		SVC_model: SVC = SVC(
			kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"
		),
		calibrate: dict = None,
	) -> dict:
		"""
		Train a Support Vector Machine (SVM) classifier to distinguish between normal and disease Raman spectra.

		Args:
			normal_data (Tuple[List[rp.SpectralContainer], str]):
				Tuple containing a list of normal spectra containers and the corresponding label (e.g., 'normal').
			disease_data (Tuple[List[rp.SpectralContainer], str]):
				Tuple containing a list of disease spectra containers and the corresponding label (e.g., 'disease').
			test_size (float, optional):
				Proportion of the dataset to include in the test split. Defaults to 0.3.
			param_search (bool, optional):
				If True, performs hyperparameter tuning using GridSearchCV. Defaults to True.
			random_state (int, optional):
				Random seed for reproducibility. Defaults to 42.
			SVC_model (SVC, optional):
				Predefined SVC model to use when param_search is False. Defaults to SVC with RBF kernel and balanced class weights.
			calibrate (dict, optional):
				Dictionary containing CalibratedClassifierCV parameters. If None or empty, no calibration is performed.
				Example: {'method': 'sigmoid', 'cv': 3, 'ensemble': True}

		Returns:
			dict: Dictionary containing:
				- "confusion_matrix": Confusion matrix of predictions on the test set.
				- "classification_report": Classification report as a string.
				- "model": Trained SVM model (calibrated if calibrate is provided).
				- "cross_val_score": Cross-validation scores on the training set.
				- "decision_function_score": Decision function scores on the test set (None if calibrated).
				- "x_train": Training feature matrix.
				- "y_train": Training labels.
				- "x_test": Test feature matrix.
				- "y_test": Test labels.
				- "is_calibrated": Boolean indicating if the model was calibrated.

		Notes:
			- All spectra are interpolated to a common wavenumber axis before training.
			- If param_search is True, hyperparameters are tuned using 10-fold stratified cross-validation.
			- If calibrate is provided, the model will be wrapped with CalibratedClassifierCV for probability calibration.
		"""
		try:
			start_time = time.time()
			normal_spectra, normal_label = normal_data
			disease_spectra, disease_label = disease_data

			# Merge all containers for each class
			def merge_containers(containers):
				if len(containers) == 1:
					return containers[0]
				all_data = np.vstack([c.spectral_data for c in containers])
				return rp.SpectralContainer(all_data, containers[0].spectral_axis)

			normal_merged = merge_containers(normal_spectra)
			disease_merged = merge_containers(disease_spectra)

			# 1. Find the union of all wavenumbers
			normal_axis = normal_merged.spectral_axis
			disease_axis = disease_merged.spectral_axis
			common_axis = np.union1d(normal_axis, disease_axis)
			self.common_axis = common_axis
			self.n_features_in = len(common_axis)

			# 2. Interpolate all spectra to the common axis (vectorized)
			def interp_all(spectral_data, from_axis, to_axis):
				return np.array(
					[np.interp(to_axis, from_axis, s) for s in spectral_data]
				)

			X_normal = interp_all(
				normal_merged.spectral_data, normal_merged.spectral_axis, common_axis
			)
			X_disease = interp_all(
				disease_merged.spectral_data, disease_merged.spectral_axis, common_axis
			)
			console_log(
				f"Interpolated ({len(X_normal)}) {normal_label} and ({len(X_disease)}) {disease_label} spectra to common axis."
			)

			X = np.vstack([X_normal, X_disease])
			y = np.array(
				[normal_label] * X_normal.shape[0]
				+ [disease_label] * X_disease.shape[0]
			)

			# 3. Train/test split and SVM
			X_train, X_test, y_train, y_test = train_test_split(
				X, y, test_size=test_size, stratify=y, random_state=random_state
			)
			console_log(
				f"Training SVM with {len(X_train)} training samples and {len(X_test)} test samples."
			)

			self.X_train = X_train
			self.y_train = y_train
			self.X_test = X_test
			self.y_test = y_test

			crossValScore = 0
			decisionFunctionScore = 0
			if param_search:
				console_log("Performing hyperparameter tuning with GridSearchCV...")
				param_grid = {
					"C": [0.01, 0.1, 1, 10, 100],
					"gamma": ["scale", 0.01, 0.1, 1, 10],
					"kernel": ["rbf", "linear"],
				}
				cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
				grid = GridSearchCV(SVC(class_weight="balanced"), param_grid, cv=cv)
				grid.fit(X_train, y_train)
				clf = grid.best_estimator_
				console_log(f"Best SVM parameters: {grid.best_params_}")
			else:
				clf = SVC_model
				cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
				clf.fit(X_train, y_train)

			# 4. Apply calibration if requested
			is_calibrated = False
			if calibrate and isinstance(calibrate, dict) and calibrate:
				console_log("Applying probability calibration...")

				# Set default calibration parameters
				cal_params = {
					"method": calibrate.get("method", "sigmoid"),
					"cv": calibrate.get("cv", 3),
					"ensemble": calibrate.get("ensemble", True),
				}

				# Create calibrated classifier with correct parameter name
				try:
					# Try with 'estimator' parameter (newer scikit-learn versions)
					clf_calibrated = CalibratedClassifierCV(estimator=clf, **cal_params)
				except TypeError:
					# Fallback to 'base_estimator' for older versions
					clf_calibrated = CalibratedClassifierCV(
						base_estimator=clf, **cal_params
					)

				# Fit the calibrated classifier
				clf_calibrated.fit(X_train, y_train)
				clf = clf_calibrated
				is_calibrated = True
				console_log(f"Model calibrated with parameters: {cal_params}")

			end_time = time.time() - start_time

			# 5. Cross-validation and decision function
			crossValScore = cross_val_score(clf, X_train, y_train, cv=cv)

			# Decision function is only available for non-calibrated SVM
			if not is_calibrated and hasattr(clf, "decision_function"):
				decisionFunctionScore = clf.decision_function(X_test)
			else:
				decisionFunctionScore = None
				if is_calibrated:
					console_log(
						"Decision function not available for calibrated models."
					)

			self._model = clf

			y_pred = clf.predict(X_test)
			conf_matrix = confusion_matrix(y_test, y_pred)
			report = classification_report(
				y_test, y_pred, target_names=[normal_label, disease_label]
			)

			return {
				"success": True,
				"msg": "train_svc_success",
				"confusion_matrix": conf_matrix,
				"classification_report": report,
				"model": clf,
				"cross_val_score": crossValScore,
				"decision_function_score": decisionFunctionScore,
				"x_train": X_train,
				"y_train": y_train,
				"x_test": X_test,
				"y_test": y_test,
				"training_time": end_time,
				"is_calibrated": is_calibrated,
			}

		except Exception as e:
			create_logs(
				"train_svc",
				"ML",
				f"Error training SVC model: {e} \n {traceback.format_exc()}",
				status="error",
			)
			return {
				"success": False,
				"msg": "train_svc_error",
				"detail": f"{e} \n {traceback.format_exc()}",
			}


	def RFMODEL(
		self,
		n_estimators=100,
		criterion="gini",
		max_depth=None,
		min_samples_split=2,
		min_samples_leaf=1,
		min_weight_fraction_leaf=0.0,
		max_features="sqrt",
		max_leaf_nodes=None,
		min_impurity_decrease=0.0,
		bootstrap=True,
		oob_score=False,
		n_jobs=None,
		random_state=None,
		verbose=0,
		warm_start=False,
		class_weight="balanced",
		ccp_alpha=0.0,
		max_samples=None,
	) -> RandomForestClassifier:
		"""
		Set the Random Forest model to be used for training.

		Parameters:
			All parameters are passed to sklearn.ensemble.RandomForestClassifier.
			n_estimators: int, default=100
				The number of trees in the forest.
			criterion: {"gini", "entropy"}, default="gini"
				The function to measure the quality of a split.
			max_depth: int, default=None
				The maximum depth of the tree.
			min_samples_split: int or float, default=2
				The minimum number of samples required to split an internal node.
			min_samples_leaf: int or float, default=1
				The minimum number of samples required to be at a leaf node.
			min_weight_fraction_leaf: float, default=0.0
				The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.
			max_features: {"auto", "sqrt", "log2"}, default="sqrt"
				The number of features to consider when looking for the best split.
			max_leaf_nodes: int, default=None
				Grow trees with max_leaf_nodes in best-first fashion.
			min_impurity_decrease: float, default=0.0
				A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
			bootstrap: bool, default=True
				Whether bootstrap samples are used when building trees.
			oob_score: bool, default=False
				Whether to use out-of-bag samples to estimate the generalization accuracy.
			n_jobs: int, default=None
				The number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context.
			random_state: int, default=None
				Controls the randomness of the estimator.
			verbose: int, default=0
				Controls the verbosity when fitting and predicting.
			warm_start: bool, default=False
				When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble.
			class_weight: dict, list of dicts, "balanced", or None, default="balanced"
				Weights associated with classes in the form {class_label: weight}.
			ccp_alpha: non-negative float, default=0.0
				Complexity parameter used for Minimal Cost-Complexity Pruning.
			max_samples: int or float, default=None
				If bootstrap is True, the number of samples to draw to train each base estimator.
				If float, it should be in (0.0, 1.0] and represent the proportion of the training set to sample.
				If int, it represents the absolute number of samples.
				If None, then draw X.shape[0] samples.
				If max_samples is not None, then bootstrap will be True.


		Returns:
			RandomForestClassifier
		"""
		return RandomForestClassifier(
			n_estimators=n_estimators,
			criterion=criterion,
			max_depth=max_depth,
			min_samples_split=min_samples_split,
			min_samples_leaf=min_samples_leaf,
			min_weight_fraction_leaf=min_weight_fraction_leaf,
			max_features=max_features,
			max_leaf_nodes=max_leaf_nodes,
			min_impurity_decrease=min_impurity_decrease,
			bootstrap=bootstrap,
			oob_score=oob_score,
			n_jobs=n_jobs,
			random_state=random_state,
			verbose=verbose,
			warm_start=warm_start,
			class_weight=class_weight,
			ccp_alpha=ccp_alpha,
			max_samples=max_samples,
		)

	def train_rf(
		self,
		normal_data: Tuple[List[rp.SpectralContainer], str],
		disease_data: Tuple[List[rp.SpectralContainer], str],
		test_size: float = 0.3,
		param_search: bool = True,
		random_state: int = 42,
		RF_model: RandomForestClassifier = RandomForestClassifier(
			class_weight="balanced", random_state=42
		),
	) -> dict:
		"""
		Train a Random Forest model on Raman data (normal vs. disease).

		Attributes:
			normal_data (Tuple[List[rp.SpectralContainer], str]): List of normal spectra and label string (e.g., 'normal')
			disease_data (Tuple[List[rp.SpectralContainer], str]): List of disease spectra and label string (e.g., 'disease')
			test_size (float): Train/test split ratio
			param_search (bool): Whether to use GridSearchCV to tune RF hyperparameters
			random_state (int): Random state for reproducibility
			RF_model (RandomForestClassifier): A predefined RandomForestClassifier model to use when param_search is False

		Returns:
			dict: Dictionary containing:
				- "confusion_matrix": Confusion matrix of predictions on the test set.
				- "classification_report": Classification report as a string.
				- "model": Trained Random Forest model.
				- "cross_val_score": Cross-validation scores on the training set.
				- "feature_importances": Feature importances from the trained model.
				- "x_train": Training feature matrix.
				- "y_train": Training labels.
				- "x_test": Test feature matrix.
				- "y_test": Test labels.
				- "training_time": Time taken in seconds to train the model.
		"""

		try:
			start_time = time.time()
			normal_spectra, normal_label = normal_data
			disease_spectra, disease_label = disease_data

			def merge_containers(containers):
				if len(containers) == 1:
					return containers[0]
				all_data = np.vstack([c.spectral_data for c in containers])
				return rp.SpectralContainer(all_data, containers[0].spectral_axis)

			normal_merged = merge_containers(normal_spectra)
			disease_merged = merge_containers(disease_spectra)

			normal_axis = normal_merged.spectral_axis
			disease_axis = disease_merged.spectral_axis
			common_axis = np.union1d(normal_axis, disease_axis)
			self.common_axis = common_axis
			self.n_features_in = len(common_axis)

			def interp_all(spectral_data, from_axis, to_axis):
				return np.array(
					[np.interp(to_axis, from_axis, s) for s in spectral_data]
				)

			X_normal = interp_all(
				normal_merged.spectral_data, normal_merged.spectral_axis, common_axis
			)
			X_disease = interp_all(
				disease_merged.spectral_data, disease_merged.spectral_axis, common_axis
			)
			console_log(
				f"Interpolated ({len(X_normal)}) {normal_label} and ({len(X_disease)}) {disease_label} spectra to common axis."
			)

			X = np.vstack([X_normal, X_disease])
			y = np.array(
				[normal_label] * X_normal.shape[0]
				+ [disease_label] * X_disease.shape[0]
			)

			X_train, X_test, y_train, y_test = train_test_split(
				X, y, test_size=test_size, stratify=y, random_state=random_state
			)

			self.X_train = X_train
			self.y_train = y_train
			self.X_test = X_test
			self.y_test = y_test

			console_log(
				f"Training Random Forest with {len(X_train)} training samples and {len(X_test)} test samples."
			)
			crossValScore = 0
			featureImportances = 0
			if param_search:
				console_log("Performing hyperparameter tuning with GridSearchCV...")
				param_grid = {
					"n_estimators": [50, 100, 200],
					"max_depth": [None, 5, 10, 20],
					"min_samples_split": [2, 5, 10],
				}
				cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
				grid = GridSearchCV(
					RandomForestClassifier(class_weight="balanced", random_state=42),
					param_grid,
					cv=cv,
				)
				grid.fit(X_train, y_train)
				clf = grid.best_estimator_
				console_log(f"Best RF parameters: {grid.best_params_}")
			else:
				clf = RF_model
				cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
				clf.fit(X_train, y_train)

			end_time = time.time() - start_time
			crossValScore = cross_val_score(clf, X_train, y_train, cv=cv)
			featureImportances = clf.feature_importances_
			self._model = clf

			y_pred = clf.predict(X_test)
			conf_matrix = confusion_matrix(y_test, y_pred)
			report = classification_report(
				y_test, y_pred, target_names=[normal_label, disease_label]
			)

			return {
				"success": True,
				"msg": "train_rf_success",
				"confusion_matrix": conf_matrix,
				"classification_report": report,
				"model": clf,
				"cross_val_score": crossValScore,
				"feature_importances": featureImportances,
				"x_train": X_train,
				"y_train": y_train,
				"x_test": X_test,
				"y_test": y_test,
				"training_time": end_time,
			}

		except Exception as e:
			create_logs(
				"train_rf",
				"ML",
				f"Error training RF model: {e} \n {traceback.format_exc()}",
				status="error",
			)
			return {
				"success": False,
				"msg": "train_rf_error",
				"detail": f"{e} \n {traceback.format_exc()}",
			}

	def KNNMODEL(
		self,
		n_neighbors=5,
		weights="uniform",
		algorithm="auto",
		leaf_size=30,
		p=2,
		metric="minkowski",
		metric_params=None,
		n_jobs=None,
	) -> KNeighborsClassifier:
		"""
		Set the KNN model to be used for training.

		Parameters:
			n_neighbors: int, default=5
				Number of neighbors to use
			weights: {'uniform', 'distance'}, default='uniform'
				Weight function used in prediction
			algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
				Algorithm used to compute the nearest neighbors
			p: int, default=2
				Power parameter for the Minkowski metric (1=Manhattan, 2=Euclidean)
			metric: str, default='minkowski'
				Distance metric to use
		"""
		return KNeighborsClassifier(
			n_neighbors=n_neighbors,
			weights=weights,
			algorithm=algorithm,
			leaf_size=leaf_size,
			p=p,
			metric=metric,
			metric_params=metric_params,
			n_jobs=n_jobs,
		)

	def train_knn(
		self,
		normal_data: Tuple[List[rp.SpectralContainer], str],
		disease_data: Tuple[List[rp.SpectralContainer], str],
		test_size: float = 0.3,
		param_search: bool = True,
		random_state: int = 42,
		KNN_model: KNeighborsClassifier = None,
		use_pca: bool = True,
		n_components: int = 50,
		use_scaling: bool = True,
	) -> dict:
		"""
		Train a K-Nearest Neighbors classifier for Raman spectral data.

		Args:
			normal_data: Tuple of normal spectra and label
			disease_data: Tuple of disease spectra and label
			test_size: Train/test split ratio
			param_search: Whether to use GridSearchCV for hyperparameter tuning
			random_state: Random seed
			KNN_model: Predefined KNN model (if param_search=False)
			use_pca: Whether to apply PCA for dimensionality reduction
			n_components: Number of PCA components to keep
			use_scaling: Whether to standardize features
		"""
		try:
			start_time = time.time()
			normal_spectra, normal_label = normal_data
			disease_spectra, disease_label = disease_data

			# Data preparation (same as other methods)
			def merge_containers(containers):
				if len(containers) == 1:
					return containers[0]
				all_data = np.vstack([c.spectral_data for c in containers])
				return rp.SpectralContainer(all_data, containers[0].spectral_axis)

			normal_merged = merge_containers(normal_spectra)
			disease_merged = merge_containers(disease_spectra)

			# Find common axis and interpolate
			normal_axis = normal_merged.spectral_axis
			disease_axis = disease_merged.spectral_axis
			common_axis = np.union1d(normal_axis, disease_axis)
			self.common_axis = common_axis
			self.n_features_in = len(common_axis)

			def interp_all(spectral_data, from_axis, to_axis):
				return np.array(
					[np.interp(to_axis, from_axis, s) for s in spectral_data]
				)

			X_normal = interp_all(
				normal_merged.spectral_data, normal_merged.spectral_axis, common_axis
			)
			X_disease = interp_all(
				disease_merged.spectral_data, disease_merged.spectral_axis, common_axis
			)

			console_log(
				f"Interpolated ({len(X_normal)}) {normal_label} and ({len(X_disease)}) {disease_label} spectra to common axis."
			)

			X = np.vstack([X_normal, X_disease])
			y = np.array(
				[normal_label] * X_normal.shape[0]
				+ [disease_label] * X_disease.shape[0]
			)

			# Train/test split
			X_train, X_test, y_train, y_test = train_test_split(
				X, y, test_size=test_size, stratify=y, random_state=random_state
			)

			self.X_train = X_train
			self.y_train = y_train
			self.X_test = X_test
			self.y_test = y_test

			console_log(
				f"Training KNN with {len(X_train)} training samples and {len(X_test)} test samples."
			)

			# Build preprocessing pipeline
			preprocessing_steps = []

			if use_scaling:
				preprocessing_steps.append(("scaler", StandardScaler()))

			if use_pca:
				preprocessing_steps.append(
					("pca", PCA(n_components=n_components, random_state=random_state))
				)

			# Create model pipeline
			if KNN_model is None:
				KNN_model = KNeighborsClassifier(n_neighbors=5, weights="distance")

			if preprocessing_steps:
				pipeline_steps = preprocessing_steps + [("knn", KNN_model)]
				clf = Pipeline(pipeline_steps)
			else:
				clf = KNN_model

			# Hyperparameter tuning
			cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

			if param_search:
				console_log("Performing hyperparameter tuning with GridSearchCV...")

				param_grid = {
					"knn__n_neighbors": [3, 5, 7, 9, 11],
					"knn__weights": ["uniform", "distance"],
					"knn__metric": ["euclidean", "manhattan", "cosine"],
				}

				if use_pca:
					param_grid["pca__n_components"] = [20, 50, 100, 200]

				grid = GridSearchCV(
					clf, param_grid, cv=cv, scoring="accuracy", n_jobs=-1
				)
				grid.fit(X_train, y_train)
				clf = grid.best_estimator_
				console_log(f"Best KNN parameters: {grid.best_params_}")
			else:
				clf.fit(X_train, y_train)

			end_time = time.time() - start_time

			# Cross-validation
			crossValScore = cross_val_score(clf, X_train, y_train, cv=cv)
			self._model = clf

			# Predictions and evaluation
			y_pred = clf.predict(X_test)
			conf_matrix = confusion_matrix(y_test, y_pred)
			report = classification_report(
				y_test, y_pred, target_names=[normal_label, disease_label]
			)

			return {
				"success": True,
				"msg": "train_knn_success",
				"confusion_matrix": conf_matrix,
				"classification_report": report,
				"model": clf,
				"cross_val_score": crossValScore,
				"x_train": X_train,
				"y_train": y_train,
				"x_test": X_test,
				"y_test": y_test,
				"training_time": end_time,
			}

		except Exception as e:
			create_logs(
				"train_knn",
				"ML",
				f"Error training KNN model: {e} \n {traceback.format_exc()}",
				status="error",
			)
			return {
				"success": False,
				"msg": "train_knn_error",
				"detail": f"{e} \n {traceback.format_exc()}",
			}

	def predict(
		self,
		test_spectra: List[rp.SpectralContainer],
		model: Union[SVC, RandomForestClassifier, None] = None,
		common_axis: np.ndarray = None,
		sample_indices: List[int] = None,
		true_labels: List[str] = None,
		use_threshold: bool = False,
		positive_label: str = None,
		threshold: float = 0.7,
		predict_proba: bool = False,
		# New parameters for PCA decision boundary
		calculate_pca_boundary: bool = False,
		boundary_resolution: int = 100,
		pca_components: int = 2,
	) -> dict:
		"""
		Enhanced predict function that can also calculate PCA decision boundary in one go.

		Args:
			test_spectra (List[rp.SpectralContainer]):
				List of spectral containers to predict on.
			model (Union[SVC, RandomForestClassifier, None]):
				Pre-trained model to use for prediction. If None, uses the stored model.
			common_axis (np.union1d):
				Common wavenumber axis for interpolation. If None, uses the stored common axis.
			sample_indices (List[int], optional):
				Specific indices of samples to predict. If None, predicts all samples.
			true_labels (List[str], optional):
				True labels for the test samples, used for evaluation if provided.
			use_threshold (bool):
				Whether to apply threshold-based prediction for RandomForestClassifier.
			positive_label (str, optional):
				The label considered as positive in threshold-based prediction.
			threshold (float):
				Threshold value for positive class prediction.
			predict_proba (bool):
				If True, return probabilities instead of discrete class predictions.
			calculate_pca_boundary (bool):
				If True, calculate PCA decision boundary data for visualization.
			boundary_resolution (int):
				Resolution for PCA decision boundary grid.
			pca_components (int):
				Number of PCA components to use (default 2 for 2D visualization).

		Returns:
			dict: Dictionary containing prediction results and optionally PCA boundary data.
		"""
		# Determine which model to use
		clf = model

		if clf is None:
			if hasattr(self, "_model") and self._model is not None:
				clf = self._model
			else:
				raise ValueError(
					"No trained model found. Train a model first or provide one."
				)

		# Validate threshold-based prediction requirements
		if use_threshold:
			if not isinstance(clf, RandomForestClassifier):
				raise ValueError(
					"Threshold-based prediction is only supported for RandomForestClassifier models. "
					f"Current model type: {type(clf).__name__}"
				)
			if positive_label is None:
				raise ValueError(
					"positive_label must be specified when use_threshold=True"
				)

		if (
			common_axis is None
			and hasattr(self, "common_axis")
			and self.common_axis is not None
		):
			common_axis = getattr(self, "common_axis", None)
		else:
			raise ValueError("No common axis found for the model.")

		# Determine the reference axis
		n_features = get_model_features(self, clf)

		# Prepare test data - collect ALL spectra first
		X_test_all = []
		for s in test_spectra:
			if s.spectral_data.shape[1] != n_features:
				for spec in s.spectral_data:
					interp = np.interp(common_axis, s.spectral_axis, spec)
					X_test_all.append(interp)
			else:
				for spec in s.spectral_data:
					X_test_all.append(spec)

		X_test_all = np.array(X_test_all)

		# Apply sample selection if specified
		if sample_indices is not None:
			# Validate indices
			max_idx = len(X_test_all) - 1
			invalid_indices = [
				idx for idx in sample_indices if idx < 0 or idx > max_idx
			]
			if invalid_indices:
				raise ValueError(
					f"Invalid sample indices: {invalid_indices}. Valid range: 0-{max_idx}"
				)

			# Select specified samples
			X_test = X_test_all[sample_indices]
			selected_indices = sample_indices

			# Handle true labels if provided
			if true_labels is not None:
				# Check if true_labels corresponds to all samples or just selected samples
				if len(true_labels) == len(X_test_all):
					# true_labels corresponds to all samples - subset them
					y_true = np.array([true_labels[i] for i in sample_indices])
				elif len(true_labels) == len(sample_indices):
					# true_labels already corresponds to selected samples
					y_true = np.array(true_labels)
					console_log(f"Using provided true labels for selected samples")
				else:
					raise ValueError(
						f"Length of true_labels ({len(true_labels)}) must match either "
						f"total samples ({len(X_test_all)}) or selected samples ({len(sample_indices)})"
					)
			else:
				# Try to use stored test labels if available
				if (
					hasattr(self, "y_test")
					and self.y_test is not None
					and len(self.y_test) == len(X_test_all)
				):
					y_true = self.y_test[sample_indices]
					console_log(
						f"Using stored test labels for selected samples {sample_indices}"
					)
				else:
					y_true = None

			console_log(
				f"Selected {len(sample_indices)} samples from {len(X_test_all)} total samples: {sample_indices}"
			)
		else:
			# Use all samples
			X_test = X_test_all
			selected_indices = list(range(len(X_test_all)))

			if true_labels is not None:
				if len(true_labels) != len(X_test_all):
					raise ValueError(
						f"Length of true_labels ({len(true_labels)}) must match total number of samples ({len(X_test_all)})"
					)
				y_true = np.array(true_labels)
			else:
				# Try to use stored test labels if available
				if hasattr(self, "y_test") and self.y_test is not None:
					y_true = self.y_test
					console_log("Using stored test labels for all samples")
				else:
					y_true = None

			console_log(f"Predicting all {len(X_test)} samples")

		self.y_true = y_true

		console_log(
			f"Predicting {len(X_test)} test samples with {n_features} features."
		)

		# Get unified prediction function and make predictions
		unified_predict = get_unified_predict_function(clf)

		# Get probabilities using unified function
		proba = unified_predict(X_test)

		# Get class labels for probability processing
		is_calibrated, base_model_type, base_model = detect_model_type(clf)
		if is_calibrated:
			# For calibrated models, get classes from the estimators
			if hasattr(clf, "estimators_") and len(clf.estimators_) > 0:
				class_labels = clf.estimators_[0].classes_
			elif hasattr(clf, "base_estimator") and hasattr(
				clf.base_estimator, "classes_"
			):
				class_labels = clf.base_estimator.classes_
			else:
				class_labels = np.array(["class_0", "class_1"])  # Fallback
		else:
			# For non-calibrated models
			if hasattr(clf, "classes_"):
				class_labels = clf.classes_
			else:
				class_labels = np.array(["class_0", "class_1"])  # Fallback

		# Handle predict_proba mode
		if predict_proba:
			console_log("Using probability prediction mode")
			# Return probabilities instead of discrete predictions
			y_pred = proba
			prediction_method = "probability"
		else:
			# Get discrete predictions - this works for all models
			y_pred = clf.predict(X_test)
			prediction_method = "standard"

		self.y_pred = y_pred

		# === NEW: Calculate PCA decision boundary if requested ===
		if calculate_pca_boundary and len(np.unique(class_labels)) == 2:
			try:
				console_log("Calculating PCA decision boundary...")
				from sklearn.decomposition import PCA

				# 1. Fit PCA on the test data
				pca = PCA(n_components=pca_components)
				X_pca = pca.fit_transform(X_test)

				# 2. Create meshgrid in PCA space
				x_min, x_max = X_pca[:, 0].min() - 2, X_pca[:, 0].max() + 2
				y_min, y_max = X_pca[:, 1].min() - 2, X_pca[:, 1].max() + 2
				xx, yy = np.meshgrid(
					np.linspace(x_min, x_max, boundary_resolution),
					np.linspace(y_min, y_max, boundary_resolution),
				)
				grid_points = np.c_[xx.ravel(), yy.ravel()]

				# 3. Inverse transform to original feature space
				console_log(
					f"Inverse transforming {grid_points.shape[0]} grid points..."
				)
				X_grid_original = pca.inverse_transform(grid_points)

				# 4. Get model predictions for grid
				console_log("Getting model predictions for decision boundary...")
				if hasattr(clf, "predict_proba"):
					grid_proba = clf.predict_proba(X_grid_original)
					Z = (
						grid_proba[:, 1]
						if grid_proba.shape[1] == 2
						else np.max(grid_proba, axis=1)
					)
				elif hasattr(clf, "decision_function"):
					decisions = clf.decision_function(X_grid_original)
					from scipy.special import expit

					Z = expit(decisions)
				else:
					grid_preds = clf.predict(X_grid_original)
					# Convert to numeric
					label_to_int = {label: i for i, label in enumerate(class_labels)}
					Z = np.array([label_to_int.get(pred, 0) for pred in grid_preds])

				# 5. Store decision boundary data
				self.pca_boundary_data = {
					"pca": pca,
					"X_pca": X_pca,
					"xx": xx,
					"yy": yy,
					"Z": Z.reshape(xx.shape),
					"explained_variance_ratio": pca.explained_variance_ratio_,
					"class_labels": (
						class_labels.tolist()
						if hasattr(class_labels, "tolist")
						else list(class_labels)
					),
					"y_true_for_boundary": y_true,
					"selected_indices": selected_indices,
				}

				console_log(
					f"PCA decision boundary calculated with resolution {boundary_resolution}x{boundary_resolution}"
				)

			except Exception as e:
				console_log(f"Error calculating PCA decision boundary: {e}")
				import traceback

				traceback.print_exc()
				self.pca_boundary_data = None

		# Process probabilities and confidences
		confidences = []
		all_probabilities = []

		if predict_proba:
			# For probability mode, confidence is the max probability
			for idx in range(len(y_pred)):
				if proba is not None:
					# Create probability dictionary
					prob_dict = {
						str(class_labels[i]): float(proba[idx][i])
						for i in range(len(class_labels))
					}

					# Confidence is the max probability
					confidence = float(np.max(proba[idx]))
					all_probabilities.append(prob_dict)
				else:
					confidence = None
					all_probabilities.append(None)

				confidences.append(confidence)

			# For probability mode, calculate label percentages based on highest probability
			predicted_labels = [class_labels[np.argmax(prob)] for prob in y_pred]
			label_counts = Counter(predicted_labels)
			total = len(predicted_labels)
			label_percentages = {
				label: count / total for label, count in label_counts.items()
			}
			most_common_label = max(label_counts, key=label_counts.get)

		else:
			# Standard discrete prediction mode
			for idx, pred in enumerate(y_pred):
				if proba is not None:
					# Create probability dictionary
					prob_dict = {
						str(class_labels[i]): float(proba[idx][i])
						for i in range(len(class_labels))
					}

					# Get confidence for this prediction
					pred_idx = np.where(class_labels == pred)[0]
					if len(pred_idx) > 0:
						confidence = float(proba[idx][pred_idx[0]])
					else:
						confidence = float(np.max(proba[idx]))

					all_probabilities.append(prob_dict)
				else:
					confidence = None
					all_probabilities.append(None)

				confidences.append(confidence)

			# Handle threshold-based prediction for RandomForest
			if use_threshold and isinstance(clf, RandomForestClassifier):
				console_log(
					f"Applying threshold-based prediction with threshold={threshold} for positive_label='{positive_label}'"
				)

				# Find the positive class index
				pos_class_idx = np.where(class_labels == positive_label)[0]
				if len(pos_class_idx) == 0:
					raise ValueError(
						f"positive_label '{positive_label}' not found in model classes: {class_labels}"
					)

				pos_class_idx = pos_class_idx[0]

				# Apply threshold to probabilities
				y_pred_thresh = []
				for i in range(len(proba)):
					if proba[i][pos_class_idx] >= threshold:
						y_pred_thresh.append(positive_label)
					else:
						# Use the other class (assuming binary classification)
						other_classes = [
							cls for cls in class_labels if cls != positive_label
						]
						y_pred_thresh.append(
							other_classes[0] if other_classes else class_labels[0]
						)

				y_pred = np.array(y_pred_thresh)
				self.y_pred = y_pred
				prediction_method = "threshold_based"

			# Calculate label statistics for discrete predictions
			labels = list(y_pred)
			label_counts = Counter(labels)
			total = len(labels)
			label_percentages = {
				label: count / total for label, count in label_counts.items()
			}
			most_common_label = max(label_counts, key=label_counts.get)

		# Prepare return dictionary
		return_dict = {
			"label_percentages": label_percentages,
			"most_common_label": most_common_label,
			"y_pred": y_pred,
			"confidences": confidences,
			"all_probabilities": all_probabilities,
			"probabilities": (
				proba.tolist() if proba is not None else None
			),  # For LIME compatibility
			"selected_indices": selected_indices,
			"total_samples": len(X_test),
			"y_true": y_true,
			"prediction_method": prediction_method,
		}

		# Add PCA boundary data to return dict if calculated
		if (
			calculate_pca_boundary
			and hasattr(self, "pca_boundary_data")
			and self.pca_boundary_data
		):
			return_dict["pca_boundary_data"] = self.pca_boundary_data
			console_log("Added PCA boundary data to prediction results")

		# Legacy decision function support (keep for backward compatibility)
		return_dict["decision_function"] = {
			"X": (
				self.x_decision_function
				if hasattr(self, "x_decision_function")
				else None
			),
			"Y": (
				self.y_decision_function
				if hasattr(self, "y_decision_function")
				else None
			),
			"Z": (
				self.z_decision_function
				if hasattr(self, "z_decision_function")
				else None
			),
		}

		def analyze_prediction_results(
			y_pred, y_true, selected_indices, confidences, predict_proba=False
		):
			"""
			Analyze prediction results and organize by correctness and class labels.
			"""
			results = {}

			if predict_proba:
				# For probability predictions, convert to discrete labels for analysis
				predicted_labels = [class_labels[np.argmax(prob)] for prob in y_pred]
			else:
				predicted_labels = y_pred

			# Get unique labels from both true and predicted
			unique_labels = set(list(y_true) + list(predicted_labels))

			# Initialize result dictionaries for each label
			for label in unique_labels:
				# Correctly predicted as this label
				results[f"true_{label}"] = []
				# Incorrectly predicted as this label
				results[f"false_{label}"] = []

			# Process each prediction
			for i in range(len(predicted_labels)):
				sample_idx = selected_indices[i]
				true_label = y_true[i]
				pred_label = predicted_labels[i]
				confidence = confidences[i] if confidences[i] is not None else 0.0

				if true_label == pred_label:
					# Correct prediction - add to true_[label]
					results[f"true_{pred_label}"].append((sample_idx, confidence))
				else:
					# Incorrect prediction - add to false_[predicted_label]
					results[f"false_{pred_label}"].append((sample_idx, confidence))

			# Add summary statistics
			summary = {"prediction_breakdown": results, "summary": {}}

			# Calculate summary for each label
			for label in unique_labels:
				true_count = len(results[f"true_{label}"])
				false_count = len(results[f"false_{label}"])

				summary["summary"][label] = {
					"correctly_predicted": true_count,
					"incorrectly_predicted": false_count,
					"precision": (
						true_count / (true_count + false_count)
						if (true_count + false_count) > 0
						else 0.0
					),
				}

			return summary

		if y_true is not None:
			prediction_results = analyze_prediction_results(
				y_pred, y_true, selected_indices, confidences, predict_proba
			)
			return_dict.update(prediction_results)

		# Add threshold-specific information if used
		if use_threshold:
			return_dict.update(
				{
					"threshold_used": threshold,
					"positive_label": positive_label,
				}
			)

		# Add predict_proba specific information
		if predict_proba:
			return_dict.update(
				{
					"predict_proba": True,
					"class_labels": (
						class_labels.tolist()
						if hasattr(class_labels, "tolist")
						else list(class_labels)
					),
				}
			)

		return return_dict


class MLModel:
	def __init__(
		self,
		onnx_path: str = None,
		meta_path: str = None,
		pickle_path: str = None,
		sess_options: Any | None = None,
		providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
		provider_options: Sequence[dict[Any, Any]] | None = None,
		**kwargs: Any,
	):
		self.session = None
		self.metadata = None
		self.common_axis = None
		self.n_features_in = None
		self.onnx_model = None
		self.sklearn_model = None
		self.load_msg = None
		self.load_success = False
		self._name = None
		self._type = None
		self._version = None
		load_data = {}

		self.region = kwargs.get("region", None)
		self.X_train = None
		self.y_train = None
		self.X_test = None
		self.y_test = None
		self.y_pred = None
		self.x_decision_function = None
		self.y_decision_function = None
		self.z_decision_function = None

		if onnx_path and meta_path:
			load_data = self.load(
				onnx_path,
				meta_path,
				sess_options,
				providers,
				provider_options,
				**kwargs,
			)
			self.load_msg = load_data.get("msg", None)
			self.load_success = load_data.get("success", False)

		if pickle_path:
			pickle_data = self.pickle_load(pickle_path, meta_path)
			if pickle_data.get("success", False):
				self.sklearn_model = pickle_data.get("model", None)
				self.metadata = pickle_data.get("metadata", None)
				self.common_axis = np.array(self.metadata["common_axis"])
				self.n_features_in = int(self.metadata["n_features_in"])
			else:
				create_logs(
					"pickle_load",
					"ML",
					f"Error loading pickle model: {pickle_data.get('detail', '')}",
					status="error",
				)

	def save(
		self,
		model: Union[SVC, RandomForestClassifier],
		labels: list[str],
		filename: str,
		common_axis: np.ndarray,
		n_features_in: int,
		save_pickle: bool = True,
		meta: dict = {
			"model_short": "",
			"model_type": "",
			"model_name": "",
			"model_version": "",
			"model_description": "",
			"model_author": "",
			"model_date": "",
			"model_license": "",
			"model_source": "",
		},
		other_meta: dict = {},
	) -> dict:
		"""
		Save the trained model to ONNX format with enhanced calibration information.
		Automatically handles filename conflicts by appending numbers.

		Parameters:
			model: The trained sklearn model (possibly calibrated)
			labels: List of class labels
			filename: Name for the saved files (will be auto-incremented if exists)
			common_axis: Common wavenumber axis
			n_features_in: Number of input features
			save_pickle: Whether to save pickle version
			meta: Basic metadata dictionary
			other_meta: Additional metadata
		"""

		def get_unique_filename(
			base_filename: str, model_dir: str
		) -> tuple[int, str, str]:
			"""
			Generate a unique filename by appending numbers if the file already exists.

			Args:
				base_filename: Original filename without extension
				model_dir: Directory where files will be saved

			Returns:
				tuple: (counter, unique_filename, model_specific_dir)
			"""
			# Create model-specific subdirectory based on model_short
			model_short = meta.get("model_short", "UNKNOWN_MODEL")
			model_specific_dir = os.path.join(model_dir, model_short)
			os.makedirs(model_specific_dir, exist_ok=True)

			# Check for existing files
			counter = 0
			current_filename = base_filename

			while True:
				onnx_path = os.path.join(
					model_specific_dir, f"{current_filename}.onnx"
				)
				meta_path = os.path.join(
					model_specific_dir, f"{current_filename}.json"
				)
				pickle_path = os.path.join(
					model_specific_dir, f"{current_filename}.pkl"
				)

				# Check if any of the files exist
				files_exist = [
					os.path.exists(onnx_path),
					os.path.exists(meta_path),
					save_pickle and os.path.exists(pickle_path),
				]

				if not any(files_exist):
					# No conflicts, use this filename
					break

				# Files exist, increment counter and try again
				counter += 1
				current_filename = f"{base_filename}_v{counter}"

				# Safety check to prevent infinite loops
				if counter > 1000:
					raise ValueError(
						f"Unable to generate unique filename after {counter} attempts"
					)

			if counter > 0:
				console_log(
					f"⚠️  Filename conflict detected. Using '{current_filename}' instead of '{base_filename}'"
				)

			return counter, current_filename, model_specific_dir

		try:
			# Handle calibrated models for n_features
			actual_model = model
			if hasattr(model, "estimators_") and len(model.estimators_) > 0:
				actual_model = model.estimators_[0]
			elif hasattr(model, "base_estimator"):
				actual_model = model.base_estimator

			n_features = actual_model.n_features_in_

			# Ensure the models directory exists
			model_dir = os.path.join(CURRENT_DIR, "models")
			os.makedirs(model_dir, exist_ok=True)

			# Generate safe base filename
			filename_safe = safe_filename(filename)

			# Get unique filename and model-specific directory
			counter, unique_filename, model_specific_dir = get_unique_filename(
				filename_safe, model_dir
			)

			# Create final file paths
			onnx_path = os.path.join(model_specific_dir, f"{unique_filename}.onnx")
			meta_path = os.path.join(model_specific_dir, f"{unique_filename}.json")
			pickle_path = os.path.join(model_specific_dir, f"{unique_filename}.pkl")

			console_log(f"💾 Saving model to: {model_specific_dir}")
			console_log(f"   • ONNX: {unique_filename}.onnx")
			console_log(f"   • Metadata: {unique_filename}.json")
			if save_pickle:
				console_log(f"   • Pickle: {unique_filename}.pkl")

			initial_type = [("float_input", FloatTensorType([None, n_features]))]
			onnx_model = convert_sklearn(model, initial_types=initial_type)

			# Save the model to ONNX format
			with open(onnx_path, "wb") as f:
				f.write(onnx_model.SerializeToString())

			# Save the model to a pickle file
			if save_pickle:
				with open(pickle_path, "wb") as f:
					pickle.dump(model, f)

			# === ENHANCED CALIBRATION INFORMATION EXTRACTION ===
			def extract_calibration_info(model):
				"""Extract comprehensive calibration information"""
				from sklearn.calibration import CalibratedClassifierCV

				calibration_info = {
					"is_calibrated": False,
					"calibration_method": None,
					"cv_folds": None,
					"n_calibrated_classifiers": 0,
					"base_estimator_type": None,
					"base_estimator_params": {},
					"sigmoid_parameters": [],
					"isotonic_parameters": [],
				}

				if isinstance(model, CalibratedClassifierCV):
					calibration_info.update(
						{
							"is_calibrated": True,
							"calibration_method": model.method,
							"cv_folds": (
								model.cv if hasattr(model, "cv") else "unknown"
							),
							"n_calibrated_classifiers": len(
								model.calibrated_classifiers_
							),
						}
					)

					# FIXED: Get base estimator info properly for CCCV
					base_est = None
					if (
						hasattr(model, "base_estimator")
						and model.base_estimator is not None
					):
						# Newer sklearn versions use base_estimator
						base_est = model.base_estimator
					elif hasattr(model, "estimator") and model.estimator is not None:
						# Some versions use estimator
						base_est = model.estimator
					elif hasattr(model, "estimators_") and len(model.estimators_) > 0:
						# Fallback to first estimator from fitted calibrated classifiers
						first_cal_clf = model.calibrated_classifiers_[0]
						if hasattr(first_cal_clf, "base_estimator"):
							base_est = first_cal_clf.base_estimator
						elif hasattr(first_cal_clf, "estimator"):
							base_est = first_cal_clf.estimator

					if base_est is not None:
						calibration_info["base_estimator_type"] = (
							base_est.__class__.__name__
						)
						calibration_info["base_estimator_params"] = (
							extract_serializable_params(base_est)
						)
						console_log(
							f"✅ Extracted base estimator params for {base_est.__class__.__name__}"
						)
					else:
						console_log(
							"⚠️  Could not find base estimator for calibrated model"
						)
						calibration_info["base_estimator_type"] = "Unknown"
						calibration_info["base_estimator_params"] = {}

					# Extract calibration function parameters
					for i, cal_clf in enumerate(model.calibrated_classifiers_):
						if hasattr(cal_clf, "calibrators"):
							for j, calibrator in enumerate(cal_clf.calibrators):
								calibrator_info = {
									"classifier_index": i,
									"calibrator_index": j,
									"calibrator_type": calibrator.__class__.__name__,
								}

								# Sigmoid calibration parameters
								if hasattr(calibrator, "a_") and hasattr(
									calibrator, "b_"
								):
									calibrator_info.update(
										{
											"a": float(calibrator.a_),
											"b": float(calibrator.b_),
										}
									)
									calibration_info[
										"sigmoid_parameters"
									].append(calibrator_info)

								# Isotonic calibration parameters
								elif hasattr(calibrator, "X_min_") and hasattr(
									calibrator, "X_max_"
								):
									calibrator_info.update(
										{
											"X_min": float(calibrator.X_min_),
											"X_max": float(calibrator.X_max_),
											"n_features_in": (
												int(calibrator.n_features_in_)
												if hasattr(calibrator, "n_features_in_")
												else None
											),
										}
									)
									calibration_info[
										"isotonic_parameters"
									].append(calibrator_info)
				else:
					# Non-calibrated model
					calibration_info["base_estimator_type"] = model.__class__.__name__
					calibration_info["base_estimator_params"] = (
						extract_serializable_params(model)
					)

				return calibration_info

			def extract_serializable_params(model):
				"""Extract only JSON-serializable parameters from a model"""
				if not hasattr(model, "get_params"):
					console_log(
						f"⚠️  Model {type(model).__name__} has no get_params method"
					)
					return {}

				try:
					params = model.get_params()
					serializable_params = {}

					for key, value in params.items():
						try:
							# Check if the value is JSON serializable
							if value is None:
								serializable_params[key] = None
							elif isinstance(value, (bool, int, float, str)):
								serializable_params[key] = value
							elif isinstance(value, (list, tuple)):
								# Check if all elements are serializable
								if all(
									isinstance(v, (bool, int, float, str, type(None)))
									for v in value
								):
									serializable_params[key] = list(value)
								else:
									serializable_params[key] = str(value)
							elif isinstance(value, dict):
								# Recursively handle dict (though sklearn params usually don't have nested dicts)
								serializable_params[key] = {
									k: v
									for k, v in value.items()
									if isinstance(
										v, (bool, int, float, str, type(None))
									)
								}
							elif hasattr(value, "__name__"):
								# For function objects, store the name
								serializable_params[key] = value.__name__
							elif isinstance(value, np.ndarray):
								# Handle numpy arrays
								if value.size < 100:  # Only for small arrays
									serializable_params[key] = value.tolist()
								else:
									serializable_params[key] = (
										f"numpy.ndarray(shape={value.shape}, dtype={value.dtype})"
									)
							else:
								# For complex objects, store string representation
								serializable_params[key] = str(value)
						except (TypeError, ValueError) as e:
							# If serialization fails, store as string
							console_log(
								f"⚠️  Could not serialize parameter {key}: {e}"
							)
							serializable_params[key] = str(value)

					console_log(
						f"✅ Extracted {len(serializable_params)} parameters from {type(model).__name__}"
					)
					return serializable_params

				except Exception as e:
					console_log(
						f"❌ Error extracting parameters from {type(model).__name__}: {e}"
					)
					return {}

			# Extract calibration information
			calibration_info = extract_calibration_info(model)

			# FIXED: Proper version handling
			base_version = "1.0.0"
			if counter > 0:
				# If counter > 0, it means we had filename conflicts
				version_parts = base_version.split(".")
				version_parts[1] = str(counter)  # Increment minor version
				final_version = ".".join(version_parts)
			else:
				final_version = base_version

			# Override with user-provided version if specified
			if meta.get("model_version"):
				final_version = meta.get("model_version")

			# Save metadata
			meta = meta or {}
			metadata = {
				"model_short": meta.get("model_short", ""),
				"model_type": meta.get("model_type", type(model).__name__),
				"model_name": meta.get("model_name", ""),
				"model_version": final_version,  # FIXED: Use proper version
				"model_description": meta.get("model_description", ""),
				"model_author": meta.get("model_author", ""),
				"model_date": meta.get(
					"model_date", datetime.now().strftime("%Y-%m-%d")
				),
				"model_license": meta.get("model_license", ""),
				"model_source": meta.get("model_source", ""),
				"labels": labels,
				"common_axis": common_axis.tolist(),
				"n_features_in": int(n_features_in),
				"calibration_info": calibration_info,  # Add comprehensive calibration info
				"file_info": {  # Add file information
					"original_filename": filename,
					"actual_filename": unique_filename,
					"model_directory": model_specific_dir,
					"filename_was_modified": unique_filename != filename_safe,
					"version_incremented": counter > 0,
					"filename_counter": counter,
				},
			}

			# Add library versions
			try:
				metadata["library"] = {
					"sklearn": sklearn.__version__,
					"onnx": skl2onnx.__version__,
					"onnxruntime": ort.__version__,
					"numpy": np.__version__,
					"skl2onnx": skl2onnx.__version__,
				}
			except Exception as e:
				console_log(f"⚠️  Could not extract library versions: {e}")
				metadata["library"] = {}

			# Merge other_meta, ensuring it's serializable
			def make_serializable(obj):
				"""Recursively make an object JSON serializable"""
				if obj is None or isinstance(obj, (bool, int, float, str)):
					return obj
				elif isinstance(obj, (list, tuple)):
					return [make_serializable(item) for item in obj]
				elif isinstance(obj, dict):
					return {
						str(key): make_serializable(value)
						for key, value in obj.items()
					}
				elif isinstance(obj, np.ndarray):
					if obj.size < 1000:  # Only serialize small arrays
						return obj.tolist()
					else:
						return f"numpy.ndarray(shape={obj.shape}, dtype={obj.dtype})"
				elif hasattr(obj, "tolist"):  # numpy scalars
					return obj.tolist()
				elif hasattr(obj, "item"):  # numpy scalars
					return obj.item()
				else:
					return str(obj)

			# Make other_meta serializable and merge
			try:
				serializable_other_meta = make_serializable(other_meta)
				metadata.update(serializable_other_meta)
			except Exception as e:
				console_log(f"⚠️  Error making other_meta serializable: {e}")

			# Save metadata to JSON
			try:
				with open(meta_path, "w") as f:
					json.dump(metadata, f, indent=2)
			except Exception as e:
				console_log(f"❌ Error saving metadata: {e}")
				raise

			save_str = f"✅ Model successfully saved!"
			save_str += f"\n   📁 Directory: {model_specific_dir}"
			save_str += f"\n   📄 ONNX: {unique_filename}.onnx"
			save_str += f"\n   📋 Metadata: {unique_filename}.json"
			save_str += f"\n   🔢 Version: {final_version}"
			if save_pickle:
				save_str += f"\n   🥒 Pickle: {unique_filename}.pkl"

			console_log(save_str)

			create_logs(
				"save_onnx",
				"ML",
				f"Model saved to {onnx_path} and metadata to {meta_path}",
				status="info",
			)

			# Enhanced calibration summary
			if calibration_info["is_calibrated"]:
				console_log("📊 Calibration Information Saved:")
				console_log(
					f"  - Method: {calibration_info['calibration_method']}"
				)
				console_log(f"  - CV Folds: {calibration_info['cv_folds']}")
				console_log(
					f"  - Base Estimator: {calibration_info['base_estimator_type']}"
				)
				console_log(
					f"  - Base Params Extracted: {len(calibration_info['base_estimator_params'])} parameters"
				)

				if calibration_info["sigmoid_parameters"]:
					console_log(
						f"  - Sigmoid Parameters: {len(calibration_info['sigmoid_parameters'])} calibrators"
					)
				if calibration_info["isotonic_parameters"]:
					console_log(
						f"  - Isotonic Parameters: {len(calibration_info['isotonic_parameters'])} calibrators"
					)

			self.onnx_model = onnx_model
			self.metadata = metadata
			self.common_axis = common_axis
			self.n_features_in = int(n_features_in)
			self.session = ort.InferenceSession(
				onnx_path, providers=["CPUExecutionProvider"]
			)

			return {
				"onnx_model": onnx_model,
				"onnx_path": onnx_path,
				"meta_path": meta_path,
				"pickle_path": pickle_path if save_pickle else None,
				"metadata": metadata,
				"success": True,
				"calibration_info": calibration_info,  # Include in return for immediate access
				"filename_info": {
					"original_filename": filename,
					"actual_filename": unique_filename,
					"was_modified": unique_filename != filename_safe,
					"model_directory": model_specific_dir,
					"version": final_version,
					"counter": counter,
				},
			}

		except Exception as e:
			create_logs(
				"save_onnx",
				"ML",
				f"Error saving model to ONNX: {e}\n{traceback.format_exc()}",
				status="error",
			)
			return {
				"success": False,
				"msg": "save_onnx_error",
				"detail": f"{e} \n {traceback.format_exc()}",
			}

	def load(
		self,
		onnx_path: str = None,
		meta_path: str = None,
		sess_options: Any | None = None,
		providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
		provider_options: Sequence[dict[Any, Any]] | None = None,
		**kwargs: Any,
	) -> dict:
		"""
		Load an ONNX model and (optionally) its metadata, and create an ONNX Runtime inference session.

		Parameters:
			onnx_path (str): Path to the ONNX model file.
			meta_path (str, optional): Path to the metadata JSON file.

		Returns:
			dict: {
				"onnx_session": ONNX Runtime inference session,
				"metadata": dict (if meta_path is provided),
				"onnx_model": loaded ONNX model,
				"success": True/False,
				"msg": status message,
			}
		"""
		try:
			# Load ONNX model
			onnx_model = onnx.load(onnx_path)
			# Create ONNX Runtime inference session
			session = ort.InferenceSession(
				onnx_path,
				sess_options=sess_options,
				providers=providers,
				provider_options=provider_options,
				**kwargs,
			)

			# Load metadata if provided
			metadata = None
			if meta_path and os.path.exists(meta_path):
				with open(meta_path, "r") as f:
					metadata = json.load(f)

			create_logs(
				"load_onnx",
				"ML",
				f"Model loaded from {onnx_path} and metadata from {meta_path}",
				status="info",
			)

			self.onnx_model = onnx_model
			self.metadata = metadata
			self.common_axis = np.array(metadata["common_axis"])
			self.n_features_in = int(metadata["n_features_in"])
			self._type = metadata.get("model_type", None)
			self._name = metadata.get("model_name", None)
			self._version = metadata.get("model_version", None)
			self.session = session
			self.load_msg = "load_onnx_success"
			self.load_success = True
			self.region = (
				Tuple(metadata["region"]) if self.region is None else self.region
			)
			if self.region is None:
				console_log(
					"Region not included together, please include region before predict"
				)

			return {
				"onnx_session": session,
				"metadata": metadata,
				"onnx_model": onnx_model,
				"success": True,
				"msg": "load_onnx_success",
			}
		except Exception as e:
			create_logs(
				"load_onnx",
				"ML",
				f"Error loading ONNX model: {e} \n {traceback.format_exc()}",
				status="error",
			)
			self.load_msg = "load_onnx_error"
			self.load_success = False
			return {
				"success": False,
				"msg": "load_onnx_error",
				"detail": f"{e} \n {traceback.format_exc()}",
			}

	def pickle_load(self, pickle_path: str, meta_path: str) -> dict:
		"""
		Load a pickled model and its metadata.

		Args:
			pickle_path (str): Path to the pickled model file.
			meta_path (str): Path to the metadata JSON file.

		Returns:
			dict: {
				"model": loaded model,
				"metadata": dict,
				"success": True/False,
				"msg": status message,
			}
		"""
		try:
			model = None
			with open(pickle_path, "rb") as f:
				model = pickle.load(f)

			self.sklearn_model = model

			with open(meta_path, "r") as f:
				metadata = json.load(f)

			self.metadata = metadata
			self.common_axis = np.array(metadata["common_axis"])
			self.n_features_in = int(metadata["n_features_in"])

			create_logs(
				"pickle_load",
				"ML",
				f"Model loaded from {pickle_path} and metadata from {meta_path}",
				status="info",
			)

			return {
				"model": model,
				"metadata": metadata,
				"success": True,
				"msg": "pickle_load_success",
			}
		except Exception as e:
			create_logs(
				"pickle_load",
				"ML",
				f"Error loading pickled model: {e} \n {traceback.format_exc()}",
				status="error",
			)
			return {
				"success": False,
				"msg": "pickle_load_error",
				"detail": f"{e} \n {traceback.format_exc()}",
			}

	def predict(
		self,
		test_spectra: Union[List[rp.SpectralContainer], rp.SpectralContainer, np.ndarray],
		true_labels: List[str] = None,
		use_onnx: bool = True,
		session: ort.InferenceSession = None,
		n_features: int = None,
		common_axis: np.ndarray = None,
		class_labels: list = None,
		sample_indices: List[int] = None,
		use_threshold: bool = False,
		positive_label: str = None,
		threshold: float = 0.7,
		predict_proba: bool = False,
		calculate_pca_boundary: bool = False,
		boundary_resolution: int = 100,
		pca_components: int = 2,
	) -> dict:
		"""
		Predict class labels using either ONNX Runtime or sklearn model.

		Args:
			test_spectra: Input data - can be:
				- List of SpectralContainers
				- Single SpectralContainer
				- numpy array (shape: [n_samples, n_features])
			true_labels: Ground truth labels for evaluation
			use_onnx: If True, use ONNX model; if False, use sklearn model
			session: ONNX Runtime session (optional)
			n_features: Number of features (auto-detected if None)
			common_axis: Common wavenumber axis (auto-detected if None)
			class_labels: List of class labels (auto-detected if None)
			sample_indices: Specific sample indices to predict
			use_threshold: Apply threshold-based prediction
			positive_label: Label considered positive for threshold
			threshold: Threshold value for positive class
			predict_proba: Return probabilities instead of discrete predictions
			calculate_pca_boundary: Calculate PCA decision boundary
			boundary_resolution: Resolution for PCA boundary grid
			pca_components: Number of PCA components

		Returns:
			dict: Prediction results
		"""

		def _validate_inputs():
			"""Validate input parameters and model availability."""
			if test_spectra is None:
				raise ValueError("test_spectra must be provided")

			# Validate test_spectra type
			if not isinstance(test_spectra, (list, rp.SpectralContainer, np.ndarray)):
				raise ValueError(
					f"test_spectra must be a list, rp.SpectralContainer, or np.ndarray, "
					f"got {type(test_spectra)}"
				)

			if use_onnx and not self.session and not session:
				raise ValueError(
					"No ONNX session found. Load a model first or provide one."
				)

			if not use_onnx and not self.sklearn_model:
				raise ValueError("No sklearn model found. Load a model first.")

			if use_threshold and positive_label is None:
				raise ValueError(
					"positive_label must be specified when use_threshold=True"
				)

		def _get_model_parameters():
			"""Get model parameters from various sources."""
			nonlocal n_features, common_axis, class_labels

			# Get n_features
			if n_features is None:
				if hasattr(self, "n_features_in") and self.n_features_in is not None:
					n_features = self.n_features_in
				elif self.metadata and "n_features_in" in self.metadata:
					n_features = self.metadata["n_features_in"]
				else:
					raise ValueError("n_features not found")

			# Get common_axis
			if common_axis is None:
				if hasattr(self, "common_axis") and self.common_axis is not None:
					common_axis = self.common_axis
				elif self.metadata and "common_axis" in self.metadata:
					common_axis = np.array(self.metadata["common_axis"])
				else:
					raise ValueError("common_axis not found")

			# Get class_labels
			if class_labels is None and self.metadata and "labels" in self.metadata:
				class_labels = self.metadata["labels"]

		def _prepare_test_data():
			"""Prepare and preprocess test data."""
			# Handle different input types for test_spectra
			if isinstance(test_spectra, np.ndarray):
				# Direct numpy array input
				console_log(
					f"Using direct numpy array input with shape {test_spectra.shape}"
				)
				return test_spectra.astype(np.float32)

			elif isinstance(test_spectra, rp.SpectralContainer):
				# Single SpectralContainer - convert to list for uniform processing
				containers_list = [test_spectra]

			elif isinstance(test_spectra, list):
				# List of SpectralContainers
				containers_list = test_spectra

			else:
				raise ValueError(f"Unsupported test_spectra type: {type(test_spectra)}")

			# Process SpectralContainer(s)
			X_test_all = []
			for s in containers_list:
				if not isinstance(s, rp.SpectralContainer):
					raise ValueError(f"Expected SpectralContainer, got {type(s)}")

				if s.spectral_data.shape[1] != n_features:
					# Need interpolation
					console_log(
						f"Interpolating spectra from {s.spectral_data.shape[1]} to {n_features} features"
					)
					for spec in s.spectral_data:
						interp = np.interp(common_axis, s.spectral_axis, spec)
						X_test_all.append(interp)
				else:
					# No interpolation needed
					for spec in s.spectral_data:
						X_test_all.append(spec)

			return np.array(X_test_all, dtype=np.float32)

		def _select_samples(X_test_all):
			"""Select specific samples based on indices."""
			if sample_indices is not None:
				# Validate indices
				max_idx = len(X_test_all) - 1
				invalid_indices = [
					idx for idx in sample_indices if idx < 0 or idx > max_idx
				]
				if invalid_indices:
					raise ValueError(f"Invalid sample indices: {invalid_indices}")

				X_test = X_test_all[sample_indices]
				selected_indices = sample_indices

				# Handle true labels
				y_true = None
				if true_labels is not None:
					if len(true_labels) == len(X_test_all):
						y_true = np.array([true_labels[i] for i in sample_indices])
					elif len(true_labels) == len(sample_indices):
						y_true = np.array(true_labels)
					else:
						raise ValueError("Length mismatch in true_labels")

				console_log(
					f"Selected {len(sample_indices)} samples from {len(X_test_all)} total"
				)
			else:
				X_test = X_test_all
				selected_indices = list(range(len(X_test_all)))

				y_true = None
				if true_labels is not None:
					if len(true_labels) != len(X_test_all):
						raise ValueError("Length mismatch in true_labels")
					y_true = np.array(true_labels)

				console_log(f"Predicting all {len(X_test)} samples")

			return X_test, selected_indices, y_true

		def _predict_onnx(X_test):
			"""Make predictions using ONNX model."""
			current_session = session if session else self.session
			input_name = current_session.get_inputs()[0].name
			outputs = current_session.run(None, {input_name: X_test})

			# Parse outputs
			if len(outputs) == 1:
				y_pred = outputs[0]
				proba = None
			elif len(outputs) == 2:
				if outputs[0].ndim > 1 and outputs[0].shape[1] > 1:
					proba, y_pred = outputs[0], outputs[1]
				else:
					y_pred, proba = outputs[0], outputs[1]
			else:
				y_pred = outputs[0]
				proba = None

			# Convert bytes to string if needed
			if hasattr(y_pred, "dtype") and y_pred.dtype.kind in {"S", "O"}:
				y_pred = np.array(
					[x.decode() if isinstance(x, bytes) else x for x in y_pred]
				)

			# Ensure proba is numeric
			if proba is not None and not np.issubdtype(proba.dtype, np.number):
				try:
					proba = proba.astype(np.float32)
				except Exception:
					proba = None

			return y_pred, proba

		def _predict_sklearn(X_test):
			"""Make predictions using sklearn model."""
			model = self.sklearn_model

			# Get class labels
			model_class_labels = getattr(model, "classes_", class_labels)
			if model_class_labels is None:
				model_class_labels = ["class_0", "class_1"]

			# Get unified prediction function
			unified_predict = get_unified_predict_function(model)
			proba = unified_predict(X_test)

			# Get discrete predictions
			if predict_proba:
				y_pred = proba
			else:
				y_pred = model.predict(X_test)

			return y_pred, proba, model_class_labels

		def _apply_threshold(y_pred, proba, model_class_labels):
			"""Apply threshold-based prediction for classification."""
			if not use_threshold:
				return y_pred

			# Find positive class index
			pos_class_idx = np.where(model_class_labels == positive_label)[0]
			if len(pos_class_idx) == 0:
				raise ValueError(
					f"positive_label '{positive_label}' not found in classes"
				)

			pos_class_idx = pos_class_idx[0]

			# Apply threshold
			y_pred_thresh = []
			for i in range(len(proba)):
				if proba[i][pos_class_idx] >= threshold:
					y_pred_thresh.append(positive_label)
				else:
					other_classes = [
						cls
						for cls in model_class_labels
						if cls != positive_label
					]
					y_pred_thresh.append(
						other_classes[0] if other_classes else model_class_labels[0]
					)

			return np.array(y_pred_thresh)

		def _calculate_pca_boundary(X_test, model_class_labels):
			"""Calculate PCA decision boundary for visualization."""
			if not calculate_pca_boundary or len(np.unique(model_class_labels)) != 2:
				return None

			try:
				from sklearn.decomposition import PCA

				# Fit PCA
				pca = PCA(n_components=pca_components)
				X_pca = pca.fit_transform(X_test)

				# Create meshgrid
				x_min, x_max = X_pca[:, 0].min() - 2, X_pca[:, 0].max() + 2
				y_min, y_max = X_pca[:, 1].min() - 2, X_pca[:, 1].max() + 2
				xx, yy = np.meshgrid(
					np.linspace(x_min, x_max, boundary_resolution),
					np.linspace(y_min, y_max, boundary_resolution),
				)
				grid_points = np.c_[xx.ravel(), yy.ravel()]

				# Inverse transform to original space
				X_grid_original = pca.inverse_transform(grid_points)

				# Get predictions for grid
				if use_onnx:
					Z_pred, _ = _predict_onnx(X_grid_original.astype(np.float32))
					# Convert predictions to numeric for boundary
					if hasattr(Z_pred, "dtype") and Z_pred.dtype.kind in {
						"U",
						"S",
						"O",
					}:
						unique_labels = np.unique(Z_pred)
						label_to_int = {
							label: i for i, label in enumerate(unique_labels)
						}
						Z = np.array([label_to_int[pred] for pred in Z_pred])
					else:
						Z = Z_pred
				else:
					model = self.sklearn_model
					if hasattr(model, "predict_proba"):
						grid_proba = model.predict_proba(X_grid_original)
						Z = (
							grid_proba[:, 1]
							if grid_proba.shape[1] == 2
							else np.max(grid_proba, axis=1)
						)
					elif hasattr(model, "decision_function"):
						decisions = model.decision_function(X_grid_original)
						from scipy.special import expit

						Z = expit(decisions)
					else:
						Z_pred = model.predict(X_grid_original)
						label_to_int = {
							label: i
							for i, label in enumerate(model_class_labels)
						}
						Z = np.array(
							[label_to_int.get(pred, 0) for pred in Z_pred]
						)

				return {
					"pca": pca,
					"X_pca": X_pca,
					"xx": xx,
					"yy": yy,
					"Z": Z.reshape(xx.shape),
					"explained_variance_ratio": pca.explained_variance_ratio_,
					"class_labels": (
						model_class_labels.tolist()
						if hasattr(model_class_labels, "tolist")
						else list(model_class_labels)
					),
				}

			except Exception as e:
				console_log(f"Error calculating PCA boundary: {e}")
				return None

		def _process_results(
			y_pred, proba, model_class_labels, selected_indices, y_true
		):
			"""Process prediction results and calculate statistics."""
			confidences = []
			all_probabilities = []

			if predict_proba:
				# Probability mode
				for idx in range(len(y_pred)):
					if proba is not None:
						prob_dict = {
							str(model_class_labels[i]): float(proba[idx][i])
							for i in range(len(model_class_labels))
						}
						confidence = float(np.max(proba[idx]))
						all_probabilities.append(prob_dict)
					else:
						confidence = None
						all_probabilities.append(None)
					confidences.append(confidence)

				# Calculate label percentages from highest probabilities
				predicted_labels = [
					model_class_labels[np.argmax(prob)] for prob in y_pred
				]
				label_counts = Counter(predicted_labels)

			else:
				# Discrete prediction mode
				for idx, pred in enumerate(y_pred):
					if proba is not None:
						prob_dict = {
							str(model_class_labels[i]): float(proba[idx][i])
							for i in range(len(model_class_labels))
						}

						# Get confidence for this prediction
						pred_idx = np.where(model_class_labels == pred)[0]
						confidence = (
							float(proba[idx][pred_idx[0]])
							if len(pred_idx) > 0
							else float(np.max(proba[idx]))
						)
						all_probabilities.append(prob_dict)
					else:
						confidence = None
						all_probabilities.append(None)
					confidences.append(confidence)

				label_counts = Counter(y_pred)

			# Calculate statistics
			total = len(y_pred) if not predict_proba else len(predicted_labels)
			label_percentages = {
				label: count / total for label, count in label_counts.items()
			}
			most_common_label = max(label_counts, key=label_counts.get)

			return {
				"label_percentages": label_percentages,
				"most_common_label": most_common_label,
				"y_pred": y_pred,
				"confidences": confidences,
				"all_probabilities": all_probabilities,
				"probabilities": proba.tolist() if proba is not None else None,
				"selected_indices": selected_indices,
				"total_samples": len(y_pred),
				"y_true": y_true,
				"prediction_method": "onnx" if use_onnx else "sklearn",
			}

		# === MAIN EXECUTION ===
		try:
			# 1. Validate inputs
			_validate_inputs()

			# 2. Get model parameters
			_get_model_parameters()

			# 3. Prepare test data
			X_test_all = _prepare_test_data()

			# 4. Select samples
			X_test, selected_indices, y_true = _select_samples(X_test_all)

			console_log(
				f"Predicting {len(X_test)} samples with {n_features} features using {'ONNX' if use_onnx else 'sklearn'}"
			)

			# 5. Make predictions
			if use_onnx:
				y_pred, proba = _predict_onnx(X_test)
				model_class_labels = (
					class_labels if class_labels else ["class_0", "class_1"]
				)
			else:
				y_pred, proba, model_class_labels = _predict_sklearn(X_test)

			# 6. Apply threshold if needed
			if use_threshold and not predict_proba:
				y_pred = _apply_threshold(y_pred, proba, model_class_labels)

			# 7. Calculate PCA boundary if requested
			pca_boundary_data = _calculate_pca_boundary(X_test, model_class_labels)

			# 8. Process results
			results = _process_results(
				y_pred, proba, model_class_labels, selected_indices, y_true
			)

			# ===== Store results in class attributes =====
			self.X_test = X_test  # The actual test features used for prediction
			self.y_test = y_true  # The true labels for test data
			self.y_pred = y_pred  # The predicted labels
			self.y_true = y_true  # Keep for backward compatibility
			self.X_test_all = X_test_all  # All test data before sample selection
			self.selected_indices = selected_indices  # Which samples were used

			# 9. Add additional information
			if pca_boundary_data:
				results["pca_boundary_data"] = pca_boundary_data

			if use_threshold:
				results.update(
					{
						"threshold_used": threshold,
						"positive_label": positive_label,
					}
				)

			if predict_proba:
				results.update(
					{
						"predict_proba": True,
						"class_labels": (
							model_class_labels.tolist()
							if hasattr(model_class_labels, "tolist")
							else list(model_class_labels)
						),
					}
				)

			return results

		except Exception as e:
			create_logs(
				"mlmodel_predict",
				"ML",
				f"Error in MLModel.predict: {e}\n{traceback.format_exc()}",
				status="error",
			)
			raise
