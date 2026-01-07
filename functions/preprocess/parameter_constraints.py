"""
Parameter Constraint Analysis for Preprocessing Functions

This module analyzes all preprocessing function parameters to provide
user guidance on constraints, limits, and requirements.
"""

from typing import Dict, Any, List, Tuple, Optional
import math


class ParameterConstraints:
    """Analyzes and provides constraints for preprocessing parameters."""

    def __init__(self):
        """Initialize constraint analyzer."""
        self.constraints = self._build_constraint_database()

    def _build_constraint_database(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive database of parameter constraints."""
        return {
            # BASELINE CORRECTION PARAMETERS
            "baseline_asls_lam": {
                "type": "float",
                "min": 1e3,
                "max": 1e10,
                "default": 1e6,
                "step": "log_scale",
                "description": "Smoothness parameter. Higher values = smoother baseline",
                "constraints": ["must_be_positive", "log_scale_preferred"],
                "typical_range": (1e4, 1e8),
                "units": "dimensionless",
                "hint": "Start with 1e6. Increase for smoother baselines, decrease for more flexible fitting. Research supports 1e3-1e10 range.",
            },
            "baseline_asls_p": {
                "type": "float",
                "min": 0.001,
                "max": 0.1,
                "default": 0.01,
                "step": 0.001,
                "description": "Asymmetry parameter. Lower values = more asymmetric correction",
                "constraints": ["must_be_positive", "between_0_and_1"],
                "typical_range": (0.001, 0.05),
                "units": "fraction",
                "hint": "Lower values (0.001-0.01) for strong fluorescence. Higher values (0.05-0.1) for mild baselines.",
            },
            "baseline_poly_order": {
                "type": "int",
                "min": 1,
                "max": 10,
                "default": 3,
                "step": 1,
                "description": "Polynomial order for baseline fitting",
                "constraints": [
                    "must_be_positive",
                    "integer_only",
                    "reasonable_upper_limit",
                ],
                "typical_range": (2, 6),
                "units": "order",
                "hint": "Order 3-4 for most applications. Higher orders may overfit.",
            },
            "baseline_iasls_lam_1": {
                "type": "float",
                "min": 1e2,
                "max": 1e7,
                "default": 1e4,
                "step": "log_scale",
                "description": "Secondary smoothness parameter for IASLS",
                "constraints": [
                    "must_be_positive",
                    "log_scale_preferred",
                    "less_than_main_lam",
                ],
                "typical_range": (1e3, 1e6),
                "units": "dimensionless",
                "hint": "Usually 10-100x smaller than main lambda parameter.",
            },
            # SPIKE REMOVAL PARAMETERS
            "spike_gaussian_kernel": {
                "type": "int",
                "min": 1,
                "max": 101,
                "default": 5,
                "step": 2,
                "description": "Gaussian kernel size for spike detection",
                "constraints": ["must_be_positive", "odd_numbers_only", "integer_only"],
                "typical_range": (3, 21),
                "units": "pixels",
                "hint": "Must be odd number. Larger values = more smoothing. Start with 5. Research suggests 3-21 for most Raman applications.",
            },
            "spike_gaussian_threshold": {
                "type": "float",
                "min": 1.0,
                "max": 10.0,
                "default": 3.0,
                "step": 0.1,
                "description": "Threshold for spike detection (in standard deviations)",
                "constraints": ["must_be_positive", "reasonable_upper_limit"],
                "typical_range": (2.0, 5.0),
                "units": "standard deviations",
                "hint": "Lower values = more sensitive. Higher values = less sensitive to noise.",
            },
            "spike_median_kernel_size": {
                "type": "int",
                "min": 3,
                "max": 51,
                "default": 5,
                "step": 2,
                "description": "Median filter kernel size",
                "constraints": ["must_be_positive", "odd_numbers_only", "integer_only"],
                "typical_range": (3, 15),
                "units": "pixels",
                "hint": "Must be odd number. Larger kernels remove broader spikes. Research shows 3-15 optimal for cosmic ray removal.",
            },
            # DERIVATIVE PARAMETERS
            "derivative_order": {
                "type": "int",
                "min": 1,
                "max": 2,
                "default": 1,
                "step": 1,
                "description": "Derivative order",
                "constraints": ["must_be_positive", "integer_only", "limited_range"],
                "typical_range": (1, 2),
                "units": "order",
                "hint": "1st derivative for baseline removal, 2nd derivative for peak enhancement.",
            },
            "derivative_window_length": {
                "type": "int",
                "min": 3,
                "max": 101,
                "default": 5,
                "step": 2,
                "description": "Savitzky-Golay window length",
                "constraints": [
                    "must_be_positive",
                    "odd_numbers_only",
                    "integer_only",
                    "greater_than_polyorder",
                ],
                "typical_range": (5, 25),
                "units": "pixels",
                "hint": "Must be odd and > polynomial order. Larger windows = more smoothing. Research suggests 5-25 for most applications.",
            },
            "derivative_polyorder": {
                "type": "int",
                "min": 1,
                "max": 5,
                "default": 2,
                "step": 1,
                "description": "Polynomial order for Savitzky-Golay filter",
                "constraints": [
                    "must_be_positive",
                    "integer_only",
                    "less_than_window_length",
                ],
                "typical_range": (2, 4),
                "units": "order",
                "hint": "Must be < window length. Order 2-3 recommended for most cases.",
            },
            # NORMALIZATION PARAMETERS
            "normalization_vector_norm": {
                "type": "string",
                "options": ["l1", "l2", "max"],
                "default": "l2",
                "description": "Type of vector normalization",
                "constraints": ["must_be_from_options"],
                "hint": "L2 for general use, L1 for sparse data, max for peak normalization.",
            },
            "normalization_minmax_range_min": {
                "type": "float",
                "min": -10.0,
                "max": 10.0,
                "default": 0.0,
                "step": 0.1,
                "description": "Minimum value for MinMax normalization",
                "constraints": ["less_than_max_value"],
                "typical_range": (-1.0, 1.0),
                "units": "dimensionless",
                "hint": "Usually 0 or -1. Must be less than maximum value.",
            },
            "normalization_minmax_range_max": {
                "type": "float",
                "min": -10.0,
                "max": 10.0,
                "default": 1.0,
                "step": 0.1,
                "description": "Maximum value for MinMax normalization",
                "constraints": ["greater_than_min_value"],
                "typical_range": (-1.0, 1.0),
                "units": "dimensionless",
                "hint": "Usually 1. Must be greater than minimum value.",
            },
            # CALIBRATION PARAMETERS
            "calibration_shift": {
                "type": "float",
                "min": -500.0,
                "max": 500.0,
                "default": 0.0,
                "step": 0.1,
                "description": "Wavenumber shift for calibration",
                "constraints": ["reasonable_range"],
                "typical_range": (-50.0, 50.0),
                "units": "cm⁻¹",
                "hint": "Positive values shift to higher wavenumbers. Use reference peaks for calibration.",
            },
            "calibration_stretch": {
                "type": "float",
                "min": 0.8,
                "max": 1.2,
                "default": 1.0,
                "step": 0.001,
                "description": "Wavenumber stretch factor for calibration",
                "constraints": ["must_be_positive", "near_unity"],
                "typical_range": (0.95, 1.05),
                "units": "factor",
                "hint": "Values near 1.0. >1 stretches spectrum, <1 compresses it.",
            },
        }

    def get_constraint_info(self, parameter_name: str) -> Dict[str, Any]:
        """Get constraint information for a specific parameter."""
        return self.constraints.get(parameter_name, {})

    def validate_parameter(self, parameter_name: str, value: Any) -> Tuple[bool, str]:
        """
        Validate a parameter value against its constraints.

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if parameter_name not in self.constraints:
            return True, "Parameter not found in constraint database"

        constraint = self.constraints[parameter_name]

        # Type validation
        if constraint["type"] == "int":
            if not isinstance(value, int):
                return False, f"Must be an integer"
        elif constraint["type"] == "float":
            if not isinstance(value, (int, float)):
                return False, f"Must be a number"
        elif constraint["type"] == "string":
            if not isinstance(value, str):
                return False, f"Must be a string"
            if "options" in constraint and value not in constraint["options"]:
                return False, f"Must be one of: {', '.join(constraint['options'])}"

        # Range validation
        if "min" in constraint and value < constraint["min"]:
            return False, f"Must be >= {constraint['min']}"
        if "max" in constraint and value > constraint["max"]:
            return False, f"Must be <= {constraint['max']}"

        # Constraint-specific validation
        constraints = constraint.get("constraints", [])

        if "must_be_positive" in constraints and value <= 0:
            return False, "Must be positive"

        if "odd_numbers_only" in constraints and value % 2 == 0:
            return False, "Must be an odd number"

        if "between_0_and_1" in constraints and not (0 <= value <= 1):
            return False, "Must be between 0 and 1"

        return True, ""

    def validate_interdependent_parameters(
        self, parameter_dict: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validate relationships between parameters based on research best practices.

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Savitzky-Golay filter validation
        if (
            "derivative_window_length" in parameter_dict
            and "derivative_polyorder" in parameter_dict
        ):
            window_length = parameter_dict["derivative_window_length"]
            poly_order = parameter_dict["derivative_polyorder"]

            if window_length <= poly_order:
                return (
                    False,
                    "Window length must be greater than polynomial order for Savitzky-Golay filter",
                )

            # Research-based optimal ratio check
            if window_length < poly_order + 2:
                return (
                    False,
                    "Window length should be at least poly_order + 2 for stable filtering",
                )

        # Baseline correction parameter relationships
        if (
            "baseline_iasls_lam_1" in parameter_dict
            and "baseline_asls_lam" in parameter_dict
        ):
            lam_1 = parameter_dict["baseline_iasls_lam_1"]
            main_lam = parameter_dict["baseline_asls_lam"]

            if lam_1 >= main_lam:
                return (
                    False,
                    "Secondary lambda (lam_1) should be smaller than main lambda for IASLS",
                )

        # MinMax normalization range validation
        if (
            "normalization_minmax_range_min" in parameter_dict
            and "normalization_minmax_range_max" in parameter_dict
        ):
            min_val = parameter_dict["normalization_minmax_range_min"]
            max_val = parameter_dict["normalization_minmax_range_max"]

            if min_val >= max_val:
                return False, "MinMax normalization minimum must be less than maximum"

        return True, ""

    def get_parameter_hint(self, parameter_name: str, current_value: Any = None) -> str:
        """Get user-friendly hint for parameter."""
        if parameter_name not in self.constraints:
            return "No constraint information available"

        constraint = self.constraints[parameter_name]
        hint = constraint.get("hint", "")

        # Add validation status if current value provided
        if current_value is not None:
            is_valid, error_msg = self.validate_parameter(parameter_name, current_value)
            if not is_valid:
                hint += f" ❌ Current value invalid: {error_msg}"
            else:
                hint += f" ✅ Current value valid"

        # Add range information
        if "typical_range" in constraint:
            min_val, max_val = constraint["typical_range"]
            hint += f" (Typical range: {min_val} - {max_val})"

        return hint

    def get_step_size(self, parameter_name: str) -> Any:
        """Get appropriate step size for parameter."""
        if parameter_name not in self.constraints:
            return 1

        constraint = self.constraints[parameter_name]
        step = constraint.get("step", 1)

        if step == "log_scale":
            # For log scale parameters, use factors of 10
            return "log"

        return step

    def get_parameter_categories(self) -> Dict[str, List[str]]:
        """Get parameters organized by preprocessing category."""
        categories = {
            "Baseline Correction": [],
            "Spike Removal": [],
            "Derivatives": [],
            "Normalization": [],
            "Calibration": [],
        }

        for param_name in self.constraints.keys():
            if param_name.startswith("baseline_"):
                categories["Baseline Correction"].append(param_name)
            elif param_name.startswith("spike_"):
                categories["Spike Removal"].append(param_name)
            elif param_name.startswith("derivative_"):
                categories["Derivatives"].append(param_name)
            elif param_name.startswith("normalization_"):
                categories["Normalization"].append(param_name)
            elif param_name.startswith("calibration_"):
                categories["Calibration"].append(param_name)

        return categories

    def suggest_parameter_value(
        self, parameter_name: str, use_case: str = "general"
    ) -> Any:
        """Suggest parameter value based on use case and research best practices."""
        if parameter_name not in self.constraints:
            return None

        constraint = self.constraints[parameter_name]

        # Research-based use case specific suggestions
        if use_case == "biological":
            if "baseline_asls_lam" in parameter_name:
                return 1e7  # Higher smoothness for biological fluorescence
            elif parameter_name == "baseline_asls_p":
                return 0.001  # More asymmetric for strong fluorescence backgrounds
            elif parameter_name == "spike_gaussian_threshold":
                return 2.5  # More sensitive for biological samples

        elif use_case == "material_science":
            if "baseline_asls_lam" in parameter_name:
                return 1e5  # Lower smoothness for crystalline materials
            elif parameter_name == "derivative_window_length":
                return 7  # Moderate smoothing for sharp peaks
            elif parameter_name == "derivative_polyorder":
                return 3  # Higher order for better peak preservation

        elif use_case == "sensitive_detection":
            if "spike_gaussian_threshold" in parameter_name:
                return 2.0  # Lower threshold for sensitive spike detection
            elif "spike_gaussian_kernel" in parameter_name:
                return 3  # Smaller kernel for precise detection

        elif use_case == "noisy_data":
            if "derivative_window_length" in parameter_name:
                return 11  # Larger window for noise reduction
            elif "spike_gaussian_threshold" in parameter_name:
                return 4.0  # Higher threshold to avoid false positives
            elif "baseline_asls_lam" in parameter_name:
                return 1e6  # Balanced smoothness for noisy spectra

        # Default suggestion based on research
        return constraint.get("default")
