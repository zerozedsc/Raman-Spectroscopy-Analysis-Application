from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from configs.configs import create_logs


@dataclass(frozen=True)
class PreparedSplit:
	X_train: np.ndarray
	X_test: np.ndarray
	y_train: np.ndarray
	y_test: np.ndarray
	common_axis: np.ndarray
	sample_dataset_names_train: np.ndarray
	sample_dataset_names_test: np.ndarray
	class_labels: List[str]


def _interpolate_matrix_to_axis(
	*,
	source_axis: np.ndarray,
	spectra_matrix: np.ndarray,
	target_axis: np.ndarray,
) -> np.ndarray:
	"""Interpolate spectra (n_spectra, n_points) from source axis to target axis."""
	# Assumes source_axis and target_axis are 1D increasing.
	if spectra_matrix.ndim != 2:
		raise ValueError("spectra_matrix must be 2D (n_spectra, n_points)")

	# Fast-path if axes match.
	if source_axis.shape == target_axis.shape and np.allclose(source_axis, target_axis):
		return spectra_matrix

	out = np.empty((spectra_matrix.shape[0], target_axis.shape[0]), dtype=float)
	for i in range(spectra_matrix.shape[0]):
		out[i, :] = np.interp(target_axis, source_axis, spectra_matrix[i, :])
	return out


def _ensure_1d_numeric_axis(index: pd.Index) -> np.ndarray:
	try:
		axis = np.asarray(index, dtype=float)
	except Exception as e:
		raise ValueError(f"Wavenumber axis is not numeric: {e}")
	if axis.ndim != 1:
		raise ValueError("Wavenumber axis must be 1D")
	# Ensure increasing
	if axis.shape[0] >= 2 and axis[0] > axis[-1]:
		axis = axis[::-1]
	return axis


def _dataset_to_samples(
	*,
	df: pd.DataFrame,
	dataset_name: str,
	label: str,
	common_axis: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Convert one dataset DataFrame into samples.

	Returns:
		X: (n_samples, n_features)
		y: (n_samples,)
		axis_used: (n_features,)
		dataset_names: (n_samples,)
	"""
	if df is None or df.empty:
		raise ValueError(f"Dataset '{dataset_name}' is empty")

	axis = _ensure_1d_numeric_axis(df.index)
	# Ensure values matrix is aligned with axis increasing
	values = df.values
	if axis.shape[0] != values.shape[0]:
		raise ValueError(
			f"Dataset '{dataset_name}' axis length mismatch: index={axis.shape[0]} rows={values.shape[0]}"
		)
	if df.index.shape[0] >= 2 and df.index[0] > df.index[-1]:
		values = values[::-1, :]

	# columns are spectra -> transpose to (n_spectra, n_points)
	X = values.T
	if common_axis is not None:
		X = _interpolate_matrix_to_axis(source_axis=axis, spectra_matrix=X, target_axis=common_axis)
		axis_used = common_axis
	else:
		axis_used = axis

	y = np.asarray([label] * X.shape[0], dtype=object)
	ds_names = np.asarray([dataset_name] * X.shape[0], dtype=object)
	return X, y, axis_used, ds_names


def prepare_features_for_dataset(
	*,
	df: pd.DataFrame,
	target_axis: np.ndarray,
) -> np.ndarray:
	"""Convert a dataset DataFrame into an (n_spectra, n_points) feature matrix.

	This is primarily intended for external evaluation when reusing a trained model.
	The output is always interpolated to `target_axis`.
	"""
	if df is None or df.empty:
		raise ValueError("Dataset is empty")

	source_axis = _ensure_1d_numeric_axis(df.index)
	values = df.values
	if source_axis.shape[0] != values.shape[0]:
		raise ValueError(
			f"Axis length mismatch: index={source_axis.shape[0]} rows={values.shape[0]}"
		)
	if df.index.shape[0] >= 2 and df.index[0] > df.index[-1]:
		values = values[::-1, :]

	X = values.T
	target_axis = np.asarray(target_axis, dtype=float)
	return _interpolate_matrix_to_axis(
		source_axis=source_axis,
		spectra_matrix=X,
		target_axis=target_axis,
	)


def prepare_train_test_split(
	*,
	raman_data: Mapping[str, pd.DataFrame],
	group_assignments: Mapping[str, str],
	train_ratio: float = 0.8,
	split_mode: str = "by_spectra",
	random_state: int = 42,
) -> PreparedSplit:
	"""Prepare X/y for ML with either dataset-level or spectra-level splitting.

	Parameters:
		raman_data: global RAMAN_DATA mapping {dataset_name: DataFrame(index=wavenumber, columns=spectra)}
		group_assignments: {dataset_name: group_label}. Only these datasets are used.
		train_ratio: fraction for train set.
		split_mode: "by_dataset" (patient-level) or "by_spectra" (random spectra split)
		random_state: random seed.
	"""
	if not group_assignments:
		raise ValueError("No datasets assigned to any group")
	if not (0.05 <= train_ratio <= 0.95):
		raise ValueError("train_ratio must be between 0.05 and 0.95")
	if split_mode not in {"by_dataset", "by_spectra"}:
		raise ValueError("split_mode must be 'by_dataset' or 'by_spectra'")

	# Determine class labels in stable order
	class_labels = sorted({str(v) for v in group_assignments.values()})

	# Choose a common axis from the first valid dataset
	common_axis: np.ndarray | None = None
	for ds_name in group_assignments.keys():
		df = raman_data.get(ds_name)
		if df is not None and not df.empty:
			common_axis = _ensure_1d_numeric_axis(df.index)
			break
	if common_axis is None:
		raise ValueError("No valid datasets found for ML preparation")

	# Build per-dataset sample matrices (interpolated to common_axis)
	per_dataset_X: Dict[str, np.ndarray] = {}
	per_dataset_y: Dict[str, np.ndarray] = {}
	per_dataset_ds_names: Dict[str, np.ndarray] = {}

	for ds_name, label in group_assignments.items():
		df = raman_data.get(ds_name)
		if df is None:
			raise ValueError(f"Dataset '{ds_name}' not found")
		X_ds, y_ds, _, ds_names = _dataset_to_samples(
			df=df,
			dataset_name=ds_name,
			label=str(label),
			common_axis=common_axis,
		)
		per_dataset_X[ds_name] = X_ds
		per_dataset_y[ds_name] = y_ds
		per_dataset_ds_names[ds_name] = ds_names

	create_logs(
		"MLDataPreparation",
		"ml_data_preparation",
		f"Prepared {len(per_dataset_X)} datasets for ML (split_mode={split_mode}, train_ratio={train_ratio})",
		status="debug",
	)

	# Split
	if split_mode == "by_spectra":
		X_all = np.vstack([per_dataset_X[k] for k in per_dataset_X.keys()])
		y_all = np.concatenate([per_dataset_y[k] for k in per_dataset_y.keys()])
		ds_all = np.concatenate([per_dataset_ds_names[k] for k in per_dataset_ds_names.keys()])

		test_size = 1.0 - float(train_ratio)
		try:
			X_train, X_test, y_train, y_test, ds_train, ds_test = train_test_split(
				X_all,
				y_all,
				ds_all,
				test_size=test_size,
				random_state=random_state,
				stratify=y_all,
			)
		except ValueError:
			# Fallback when stratify is not possible (too few samples)
			X_train, X_test, y_train, y_test, ds_train, ds_test = train_test_split(
				X_all,
				y_all,
				ds_all,
				test_size=test_size,
				random_state=random_state,
				stratify=None,
			)
	else:
		# Dataset-level split: split dataset names within each class label
		train_datasets: List[str] = []
		test_datasets: List[str] = []
		rng = np.random.RandomState(random_state)

		for label in class_labels:
			label_datasets = [ds for ds, lab in group_assignments.items() if str(lab) == label]
			if len(label_datasets) == 0:
				continue
			label_datasets = sorted(label_datasets)
			rng.shuffle(label_datasets)
			n_train = max(1, int(round(len(label_datasets) * float(train_ratio))))
			n_train = min(n_train, len(label_datasets) - 1) if len(label_datasets) > 1 else len(label_datasets)
			train_datasets.extend(label_datasets[:n_train])
			test_datasets.extend(label_datasets[n_train:])

		if not test_datasets:
			# If only one dataset per class, force a spectra-level split instead of failing.
			create_logs(
				"MLDataPreparation",
				"ml_data_preparation",
				"Dataset-level split produced empty test set; falling back to spectra-level split",
				status="warning",
			)
			return prepare_train_test_split(
				raman_data=raman_data,
				group_assignments=group_assignments,
				train_ratio=train_ratio,
				split_mode="by_spectra",
				random_state=random_state,
			)

		X_train = np.vstack([per_dataset_X[ds] for ds in train_datasets])
		y_train = np.concatenate([per_dataset_y[ds] for ds in train_datasets])
		ds_train = np.concatenate([per_dataset_ds_names[ds] for ds in train_datasets])

		X_test = np.vstack([per_dataset_X[ds] for ds in test_datasets])
		y_test = np.concatenate([per_dataset_y[ds] for ds in test_datasets])
		ds_test = np.concatenate([per_dataset_ds_names[ds] for ds in test_datasets])

	return PreparedSplit(
		X_train=np.asarray(X_train, dtype=float),
		X_test=np.asarray(X_test, dtype=float),
		y_train=np.asarray(y_train, dtype=object),
		y_test=np.asarray(y_test, dtype=object),
		common_axis=np.asarray(common_axis, dtype=float),
		sample_dataset_names_train=np.asarray(ds_train, dtype=object),
		sample_dataset_names_test=np.asarray(ds_test, dtype=object),
		class_labels=class_labels,
	)
