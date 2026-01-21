"""Machine Learning package.

This package is the new home for ML utilities and algorithms.

Compatibility note:
- Legacy code used to live in `functions/ML.py`.
- During migration, the full legacy implementation is preserved in
  `functions/ML_legacy.py`.
- To avoid breaking existing imports, we re-export the legacy public API here.
"""

from __future__ import annotations

# Backwards-compatible re-exports
from functions.ML_legacy import (  # noqa: F401
	MLModel,
	RamanML,
	detect_model_type,
	get_model_features,
	get_unified_predict_function,
)

# Data preparation helpers used by the ML page
from .data_preparation import (  # noqa: F401
	PreparedSplit,
	prepare_features_for_dataset,
	prepare_train_test_split,
)

# New algorithm modules
from .linear_regression import train_linear_regression, train_linear_regression_classifier  # noqa: F401
from .logistic_regression import train_logistic_regression  # noqa: F401
from .svm import train_svm_classifier  # noqa: F401
from .random_forest import train_random_forest_classifier  # noqa: F401


__all__ = [
	# Legacy-compatible exports
	"RamanML",
	"MLModel",
	"detect_model_type",
	"get_model_features",
	"get_unified_predict_function",
	# New helpers
	"PreparedSplit",
	"prepare_features_for_dataset",
	"prepare_train_test_split",
	"train_linear_regression",
	"train_linear_regression_classifier",
	"train_logistic_regression",
	"train_svm_classifier",
	"train_random_forest_classifier",
]
