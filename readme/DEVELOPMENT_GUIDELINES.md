# Development Guidelines and Standards

## Testing Standards (Added October 14, 2025)

### Test Script Organization
**MANDATORY**: All test scripts MUST be saved in `test_script/` folder structure:
```
test_script/
├── test_preprocessing_comprehensive.py  # Comprehensive preprocessing tests
├── test_results_20251014_220536.txt     # Text output
├── test_results_20251014_220536.json    # JSON output
└── README.md                             # Test documentation
```

### Environment and Execution
1. **Use UV Environment**: This project uses UV package manager
   ```bash
   cd test_script
   uv run python test_<name>.py
   ```

2. **Never use pip/conda directly** for running tests

3. **Output Requirements**:
   - Generate timestamped output files
   - Save both text (.txt) and JSON (.json) reports
   - Include summary statistics and detailed results

### Test Comprehensiveness Requirements
1. **Complete Coverage**: Test ALL implemented features, not subsets
   - Example: If 40 preprocessing methods exist, test all 40
2. **Deep Analysis**: Validate multiple aspects:
   - Parameter definitions consistency
   - Method instantiation
   - Range validation
   - Type conversion
   - Error handling
3. **Detailed Reporting**:
   - Pass/fail status per item
   - Warning counts
   - Error descriptions with context
   - Category breakdowns

### Test Script Template
```python
"""
Test Script Name
================
Purpose: <what this tests>
Coverage: <number of items tested>
Environment: UV Python environment
Output: test_script/test_results_TIMESTAMP.txt
"""

import sys
import os
from datetime import datetime

# Setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Test logic here

# Save results
output_dir = os.path.dirname(__file__)
output_file = os.path.join(output_dir, f'test_results_{timestamp}.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("Results here...")
```

## Code Quality Standards

## Date Format Standard
**Always use ISO 8601 date format**: `yyyy-mm-dd` (e.g., 2025-10-14)
- Used in all documentation
- Used in file naming
- Used in metadata and JSON exports
- Used in git commit messages
- Used in changelog entries

## Documentation Organization Standard
**Documentation Structure**: All `.docs` content must be organized by category
- Check existing folder structure in `.docs/`
- Save documentation in relevant subdirectory:
  - `pages/` - Page-specific documentation
  - `functions/` - Function library documentation
  - `widgets/` - Widget system documentation
  - `components/` - Component documentation
  - `testing/` - Test results and plans
  - `reports_summary/` - Implementation reports
- Create new folders if needed for new categories
- Never save documentation in `.docs/` root unless it's a general index file

## Prerequisites

Before running any scripts in this project, ensure the following setup steps are completed to maintain a consistent and isolated development environment.

### Environment Setup

To ensure a consistent development environment across platforms, follow these platform-specific instructions. Always run scripts from the project root directory.

#### On Linux/macOS:
```bash
# Check for pyproject.toml and use uv if available
if [ -f "pyproject.toml" ]; then
    # Install uv if not present
    pip install uv
    
    # Create virtual environment and install dependencies
    uv venv
    uv pip install -e .
    
    # Run scripts using uv
    uv run python your_script.py
else
    # Fallback to venv
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        python -m venv venv
        source venv/bin/activate
        [ -f "requirements.txt" ] && pip install -r requirements.txt
    fi
    
    python your_script.py
fi
```

#### On Windows:
```batch
REM Check for pyproject.toml and use uv if available
if exist "pyproject.toml" (
    REM Install uv if not present
    pip install uv
    
    REM Create virtual environment and install dependencies
    uv venv
    uv pip install -e .
    
    REM Run scripts using uv
    uv run python your_script.py
) else (
    REM Fallback to venv
    if exist "venv" (
        venv\Scripts\activate
    ) else (
        python -m venv venv
        venv\Scripts\activate
        if exist "requirements.txt" pip install -r requirements.txt
    )
    
    python your_script.py
)
```

**Note**: After activating the virtual environment, verify it's active by checking `which python` (Linux/macOS) or `where python` (Windows) points to the venv's Python executable.

### Key Guidelines
- **Dependency Management**: Always use `uv` for projects with `pyproject.toml` to ensure reproducible builds and latest stable package versions.
- **Virtual Environment**: Prefer venv over global installations to avoid conflicts. Activate the environment before running any scripts.
- **Script Execution**: Run all Python scripts from the project root directory (user base dir) to ensure correct path resolution and imports.
- **Validation**: Verify the environment is active by checking `which python` points to the venv's Python executable.

### 1. Python Code Style
Follow PEP 8 with these specific guidelines:

#### Naming Conventions
```python
# Classes: PascalCase
class ParameterWidget(QWidget):
    pass

# Functions and methods: snake_case
def calculate_signal_range(data):
    pass

# Constants: UPPER_CASE
DEFAULT_CACHE_SIZE = 100
PREPROCESSING_REGISTRY = MethodRegistry()

# Private methods: leading underscore
def _setup_ui(self):
    pass

# Protected methods: single leading underscore
def _validate_parameters(self):
    pass
```

#### File Organization
```python
# File header with module docstring
"""
Raman Spectroscopy Parameter Widgets

This module provides specialized input widgets for spectroscopic parameters
with real-time validation and visual feedback.
"""

# Standard library imports
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
from PySide6.QtCore import Signal, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout

# Local imports
from configs.configs import get_config
from functions.utils import create_logs
```

#### Method Documentation
```python
def detect_signal_range(self, wavenumbers: np.ndarray, intensities: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Detect the most significant spectral range using variance analysis.
    
    This method analyzes the spectral data to identify regions with the highest
    signal variation, which typically correspond to the most informative parts
    of a Raman spectrum.
    
    Args:
        wavenumbers: Array of wavenumber values (cm⁻¹)
        intensities: Array of intensity values corresponding to wavenumbers
    
    Returns:
        Tuple of (start_wavenumber, end_wavenumber) for the detected range,
        or None if no significant range could be detected.
    
    Example:
        >>> widget = MatplotlibWidget()
        >>> wavenumbers = np.linspace(400, 4000, 1000)
        >>> intensities = np.random.random(1000)
        >>> range_tuple = widget.detect_signal_range(wavenumbers, intensities)
        >>> print(f"Detected range: {range_tuple}")
    """
```

### 2. Error Handling Standards

#### Exception Handling Pattern
```python
def safe_process_data(data, method_params):
    """Process data with comprehensive error handling."""
    try:
        # Validate inputs
        if data is None or data.empty:
            raise ValueError("Input data is empty or None")
        
        if not isinstance(method_params, dict):
            raise TypeError("Method parameters must be a dictionary")
        
        # Main processing logic
        result = process_data_implementation(data, method_params)
        
        # Validate output
        if result is None:
            raise RuntimeError("Processing failed to produce valid result")
        
        return result
        
    except ValueError as e:
        create_logs("data_processing", "validation_error", 
                   f"Input validation failed: {e}", status='error')
        return None
        
    except TypeError as e:
        create_logs("data_processing", "type_error", 
                   f"Type error in processing: {e}", status='error')
        return None
        
    except Exception as e:
        create_logs("data_processing", "unexpected_error", 
                   f"Unexpected error: {e}", status='error')
        # Re-raise for critical errors that should stop execution
        if isinstance(e, (MemoryError, KeyboardInterrupt)):
            raise
        return None
```

#### Logging Standards
```python
# Use structured logging with consistent categories
create_logs(
    category="preprocessing",           # Component category
    function_name="apply_pipeline",     # Specific function
    message="Pipeline execution completed successfully",  # Descriptive message
    status='info'                      # Status level: 'info', 'warning', 'error'
)

# Include context in error messages
create_logs(
    category="parameter_validation",
    function_name="validate_range",
    message=f"Range validation failed for parameter '{param_name}': "
            f"value {value} outside allowed range {min_val}-{max_val}",
    status='warning'
)
```

### 3. UI Development Standards

#### Widget Creation Pattern
```python
class BaseParameterWidget(QWidget):
    """Base class for all parameter input widgets."""
    
    # Define signals at class level
    value_changed = Signal(object)  # Value change notification
    validation_changed = Signal(bool, str)  # Validation state change
    
    def __init__(self, param_name: str, config: dict, parent=None):
        super().__init__(parent)
        
        # Store configuration
        self.param_name = param_name
        self.config = config
        self.parent_widget = parent
        
        # Initialize state
        self._current_value = None
        self._is_valid = True
        self._validation_message = ""
        
        # Setup UI
        self._setup_ui()
        self._apply_styling()
        self._connect_signals()
        
        # Set initial value if provided
        if 'default' in config:
            self.set_value(config['default'])
    
    def _setup_ui(self):
        """Setup widget UI - implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _setup_ui")
    
    def _apply_styling(self):
        """Apply consistent styling to widget."""
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.config.get('bg_color', 'white')};
                border: 1px solid {self.config.get('border_color', '#ddd')};
                border-radius: 4px;
                padding: 4px;
            }}
        """)
    
    def _connect_signals(self):
        """Connect internal signals - implemented by subclasses."""
        pass
    
    def get_value(self):
        """Get current widget value."""
        return self._current_value
    
    def set_value(self, value):
        """Set widget value with validation."""
        if self._validate_value(value):
            self._current_value = value
            self._update_ui_value(value)
            self.value_changed.emit(value)
    
    def _validate_value(self, value) -> bool:
        """Validate value against widget constraints."""
        # Implement validation logic
        return True
    
    def _update_ui_value(self, value):
        """Update UI elements with new value."""
        # Implement UI update logic
        pass
```

#### Signal Connection Pattern
```python
def _connect_signals(self):
    """Connect widget signals to handlers."""
    # Connect value change signals
    self.min_input.valueChanged.connect(self._on_min_changed)
    self.max_input.valueChanged.connect(self._on_max_changed)
    
    # Connect validation signals
    self.min_input.editingFinished.connect(self._validate_range)
    self.max_input.editingFinished.connect(self._validate_range)
    
    # Connect to parent update signals if available
    if hasattr(self.parent_widget, 'parameter_changed'):
        self.value_changed.connect(
            lambda: self.parent_widget.parameter_changed.emit(self.param_name)
        )

def _on_min_changed(self, value):
    """Handle minimum value change."""
    # Ensure max >= min
    if value > self.max_input.value():
        self.max_input.setValue(value)
    
    # Trigger validation and update
    self._validate_range()
    self.value_changed.emit(self.get_value())
```

### 4. Performance Standards

#### Data Handling Guidelines
```python
# Use appropriate data types for large datasets
def process_large_spectral_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process large spectral datasets efficiently."""
    
    # Convert to appropriate dtypes for memory efficiency
    if data.shape[0] > 10000:  # Large dataset
        # Use float32 instead of float64 for intensity data when possible
        intensity_columns = data.select_dtypes(include=['float64']).columns
        for col in intensity_columns:
            data[col] = data[col].astype('float32')
    
    # Use chunked processing for very large datasets
    if data.shape[0] > 100000:
        chunk_size = 10000
        processed_chunks = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i+chunk_size]
            processed_chunk = process_chunk(chunk)
            processed_chunks.append(processed_chunk)
        
        return pd.concat(processed_chunks, ignore_index=True)
    
    # Standard processing for smaller datasets
    return process_standard(data)
```

#### Caching Implementation
```python
from functools import lru_cache
import hashlib

class PreviewCache:
    """Cache for preprocessing preview results."""
    
    def __init__(self, max_size: int = 50):
        self._cache = {}
        self._max_size = max_size
        self._access_times = {}
    
    def get_cache_key(self, data_hash: str, pipeline_config: dict) -> str:
        """Generate cache key from data and pipeline."""
        pipeline_str = json.dumps(pipeline_config, sort_keys=True)
        combined = f"{data_hash}_{pipeline_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, key: str):
        """Get cached result if available."""
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None
    
    def set(self, key: str, value):
        """Store result in cache with LRU eviction."""
        # Remove oldest entries if cache is full
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._access_times, key=self._access_times.get)
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
        
        self._cache[key] = value
        self._access_times[key] = time.time()
```

### 5. Testing Standards

#### Unit Test Structure
```python
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from components.widgets.matplotlib_widget import MatplotlibWidget

class TestMatplotlibWidget(unittest.TestCase):
    """Test suite for MatplotlibWidget functionality."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.widget = MatplotlibWidget()
        
        # Create test data
        self.wavenumbers = np.linspace(400, 4000, 1000)
        self.intensities = np.random.random(1000) * 1000
        
        # Add some signal peaks for testing auto-focus
        self.intensities[300:350] += 5000  # Strong peak at ~1400 cm⁻¹
        self.intensities[600:650] += 3000  # Medium peak at ~2800 cm⁻¹
    
    def test_detect_signal_range_valid_data(self):
        """Test signal range detection with valid spectral data."""
        result = self.widget.detect_signal_range(self.wavenumbers, self.intensities)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        start, end = result
        self.assertLess(start, end)
        self.assertGreaterEqual(start, self.wavenumbers.min())
        self.assertLessEqual(end, self.wavenumbers.max())
    
    def test_detect_signal_range_empty_data(self):
        """Test signal range detection with empty data."""
        empty_wavenumbers = np.array([])
        empty_intensities = np.array([])
        
        result = self.widget.detect_signal_range(empty_wavenumbers, empty_intensities)
        self.assertIsNone(result)
    
    def test_plot_spectra_with_auto_focus(self):
        """Test plotting with auto-focus enabled."""
        with patch.object(self.widget, 'detect_signal_range') as mock_detect:
            mock_detect.return_value = (1000, 3000)
            
            self.widget.plot_spectra(
                self.wavenumbers, 
                self.intensities, 
                auto_focus=True
            )
            
            mock_detect.assert_called_once()
    
    def test_plot_spectra_without_auto_focus(self):
        """Test plotting with auto-focus disabled."""
        with patch.object(self.widget, 'detect_signal_range') as mock_detect:
            self.widget.plot_spectra(
                self.wavenumbers, 
                self.intensities, 
                auto_focus=False
            )
            
            mock_detect.assert_not_called()

if __name__ == '__main__':
    unittest.main()
```

#### Integration Test Pattern
```python
class TestPreprocessingIntegration(unittest.TestCase):
    """Integration tests for preprocessing workflow."""
    
    def setUp(self):
        """Setup integration test environment."""
        self.test_data = pd.DataFrame({
            'wavenumber': np.linspace(400, 4000, 1000),
            'intensity': np.random.random(1000) * 1000
        }).set_index('wavenumber')
        
        self.preprocess_page = PreprocessPage()
        self.preprocess_page.set_data(self.test_data)
    
    def test_conditional_auto_focus_workflow(self):
        """Test complete conditional auto-focus workflow."""
        # Initially, auto-focus should be disabled
        self.assertFalse(self.preprocess_page._should_auto_focus())
        
        # Add a cropper step
        cropper_step = PipelineStep(
            category='range',
            method='Cropper',
            params={'range': (1000, 3000)},
            enabled=True
        )
        self.preprocess_page.add_pipeline_step(cropper_step)
        
        # Now auto-focus should be enabled
        self.assertTrue(self.preprocess_page._should_auto_focus())
        
        # Verify preview generation works with auto-focus
        self.preprocess_page.update_preview()
        
        # Check that matplotlib widget received auto_focus=True
        # (This would require mocking the matplotlib widget)
```

#### GUI Testing Guidelines
When testing GUI components, ensure the following for AI agent execution:
- Activate the appropriate virtual environment (uv or venv) before running Python scripts.
- Run GUI tests using the environment's Python executable.
- Wait for the GUI application to fully close before processing the next response, allowing the agent to capture all terminal output and messages.

### 6. Documentation Standards

#### Inline Documentation
```python
class ParameterWidget(QWidget):
    """
    Base parameter input widget for spectroscopic applications.
    
    This widget provides a standardized interface for parameter input
    with real-time validation, visual feedback, and integration with
    the preprocessing pipeline system.
    
    Attributes:
        param_name: Name of the parameter this widget controls
        config: Configuration dictionary with widget settings
        value_changed: Signal emitted when parameter value changes
        validation_changed: Signal emitted when validation state changes
    
    Example:
        >>> config = {
        ...     'type': 'float',
        ...     'min': 0.0,
        ...     'max': 100.0,
        ...     'default': 50.0,
        ...     'units': 'cm⁻¹'
        ... }
        >>> widget = ParameterWidget('range_min', config)
        >>> widget.value_changed.connect(on_parameter_changed)
    """
```

#### API Documentation
```python
def process_spectral_data(
    data: pd.DataFrame, 
    method: str, 
    parameters: Dict[str, Any],
    validate: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Process spectral data using specified method and parameters.
    
    Args:
        data: Spectral data with wavenumber index and intensity columns
        method: Name of preprocessing method to apply
        parameters: Method-specific parameters
        validate: Whether to validate inputs before processing
    
    Returns:
        Tuple containing:
            - Processed spectral data
            - Processing metadata including execution time and parameters used
    
    Raises:
        ValueError: If data format is invalid or method is unknown
        TypeError: If parameters are of wrong type
        ProcessingError: If processing fails
    
    Example:
        >>> data = load_spectral_data('sample.csv')
        >>> result, metadata = process_spectral_data(
        ...     data, 
        ...     'baseline_correction', 
        ...     {'method': 'polynomial', 'order': 3}
        ... )
    """
```

These development guidelines ensure consistent, maintainable, and high-quality code throughout the Raman spectroscopy application. Following these standards will make the codebase easier to understand, extend, and debug.

### 1. Python Code Style
Follow PEP 8 with these specific guidelines:

#### Naming Conventions
```python
# Classes: PascalCase
class ParameterWidget(QWidget):
    pass

# Functions and methods: snake_case
def calculate_signal_range(data):
    pass

# Constants: UPPER_CASE
DEFAULT_CACHE_SIZE = 100
PREPROCESSING_REGISTRY = MethodRegistry()

# Private methods: leading underscore
def _setup_ui(self):
    pass

# Protected methods: single leading underscore
def _validate_parameters(self):
    pass
```

#### File Organization
```python
# File header with module docstring
"""
Raman Spectroscopy Parameter Widgets

This module provides specialized input widgets for spectroscopic parameters
with real-time validation and visual feedback.
"""

# Standard library imports
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
from PySide6.QtCore import Signal, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout

# Local imports
from configs.configs import get_config
from functions.utils import create_logs
```

#### Method Documentation
```python
def detect_signal_range(self, wavenumbers: np.ndarray, intensities: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Detect the most significant spectral range using variance analysis.
    
    This method analyzes the spectral data to identify regions with the highest
    signal variation, which typically correspond to the most informative parts
    of a Raman spectrum.
    
    Args:
        wavenumbers: Array of wavenumber values (cm⁻¹)
        intensities: Array of intensity values corresponding to wavenumbers
    
    Returns:
        Tuple of (start_wavenumber, end_wavenumber) for the detected range,
        or None if no significant range could be detected.
    
    Example:
        >>> widget = MatplotlibWidget()
        >>> wavenumbers = np.linspace(400, 4000, 1000)
        >>> intensities = np.random.random(1000)
        >>> range_tuple = widget.detect_signal_range(wavenumbers, intensities)
        >>> print(f"Detected range: {range_tuple}")
    """
```

### 2. Error Handling Standards

#### Exception Handling Pattern
```python
def safe_process_data(data, method_params):
    """Process data with comprehensive error handling."""
    try:
        # Validate inputs
        if data is None or data.empty:
            raise ValueError("Input data is empty or None")
        
        if not isinstance(method_params, dict):
            raise TypeError("Method parameters must be a dictionary")
        
        # Main processing logic
        result = process_data_implementation(data, method_params)
        
        # Validate output
        if result is None:
            raise RuntimeError("Processing failed to produce valid result")
        
        return result
        
    except ValueError as e:
        create_logs("data_processing", "validation_error", 
                   f"Input validation failed: {e}", status='error')
        return None
        
    except TypeError as e:
        create_logs("data_processing", "type_error", 
                   f"Type error in processing: {e}", status='error')
        return None
        
    except Exception as e:
        create_logs("data_processing", "unexpected_error", 
                   f"Unexpected error: {e}", status='error')
        # Re-raise for critical errors that should stop execution
        if isinstance(e, (MemoryError, KeyboardInterrupt)):
            raise
        return None
```

#### Logging Standards
```python
# Use structured logging with consistent categories
create_logs(
    category="preprocessing",           # Component category
    function_name="apply_pipeline",     # Specific function
    message="Pipeline execution completed successfully",  # Descriptive message
    status='info'                      # Status level: 'info', 'warning', 'error'
)

# Include context in error messages
create_logs(
    category="parameter_validation",
    function_name="validate_range",
    message=f"Range validation failed for parameter '{param_name}': "
            f"value {value} outside allowed range {min_val}-{max_val}",
    status='warning'
)
```

### 3. UI Development Standards

#### Widget Creation Pattern
```python
class BaseParameterWidget(QWidget):
    """Base class for all parameter input widgets."""
    
    # Define signals at class level
    value_changed = Signal(object)  # Value change notification
    validation_changed = Signal(bool, str)  # Validation state change
    
    def __init__(self, param_name: str, config: dict, parent=None):
        super().__init__(parent)
        
        # Store configuration
        self.param_name = param_name
        self.config = config
        self.parent_widget = parent
        
        # Initialize state
        self._current_value = None
        self._is_valid = True
        self._validation_message = ""
        
        # Setup UI
        self._setup_ui()
        self._apply_styling()
        self._connect_signals()
        
        # Set initial value if provided
        if 'default' in config:
            self.set_value(config['default'])
    
    def _setup_ui(self):
        """Setup widget UI - implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _setup_ui")
    
    def _apply_styling(self):
        """Apply consistent styling to widget."""
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.config.get('bg_color', 'white')};
                border: 1px solid {self.config.get('border_color', '#ddd')};
                border-radius: 4px;
                padding: 4px;
            }}
        """)
    
    def _connect_signals(self):
        """Connect internal signals - implemented by subclasses."""
        pass
    
    def get_value(self):
        """Get current widget value."""
        return self._current_value
    
    def set_value(self, value):
        """Set widget value with validation."""
        if self._validate_value(value):
            self._current_value = value
            self._update_ui_value(value)
            self.value_changed.emit(value)
    
    def _validate_value(self, value) -> bool:
        """Validate value against widget constraints."""
        # Implement validation logic
        return True
    
    def _update_ui_value(self, value):
        """Update UI elements with new value."""
        # Implement UI update logic
        pass
```

#### Signal Connection Pattern
```python
def _connect_signals(self):
    """Connect widget signals to handlers."""
    # Connect value change signals
    self.min_input.valueChanged.connect(self._on_min_changed)
    self.max_input.valueChanged.connect(self._on_max_changed)
    
    # Connect validation signals
    self.min_input.editingFinished.connect(self._validate_range)
    self.max_input.editingFinished.connect(self._validate_range)
    
    # Connect to parent update signals if available
    if hasattr(self.parent_widget, 'parameter_changed'):
        self.value_changed.connect(
            lambda: self.parent_widget.parameter_changed.emit(self.param_name)
        )

def _on_min_changed(self, value):
    """Handle minimum value change."""
    # Ensure max >= min
    if value > self.max_input.value():
        self.max_input.setValue(value)
    
    # Trigger validation and update
    self._validate_range()
    self.value_changed.emit(self.get_value())
```

### 4. Performance Standards

#### Data Handling Guidelines
```python
# Use appropriate data types for large datasets
def process_large_spectral_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process large spectral datasets efficiently."""
    
    # Convert to appropriate dtypes for memory efficiency
    if data.shape[0] > 10000:  # Large dataset
        # Use float32 instead of float64 for intensity data when possible
        intensity_columns = data.select_dtypes(include=['float64']).columns
        for col in intensity_columns:
            data[col] = data[col].astype('float32')
    
    # Use chunked processing for very large datasets
    if data.shape[0] > 100000:
        chunk_size = 10000
        processed_chunks = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i+chunk_size]
            processed_chunk = process_chunk(chunk)
            processed_chunks.append(processed_chunk)
        
        return pd.concat(processed_chunks, ignore_index=True)
    
    # Standard processing for smaller datasets
    return process_standard(data)
```

#### Caching Implementation
```python
from functools import lru_cache
import hashlib

class PreviewCache:
    """Cache for preprocessing preview results."""
    
    def __init__(self, max_size: int = 50):
        self._cache = {}
        self._max_size = max_size
        self._access_times = {}
    
    def get_cache_key(self, data_hash: str, pipeline_config: dict) -> str:
        """Generate cache key from data and pipeline."""
        pipeline_str = json.dumps(pipeline_config, sort_keys=True)
        combined = f"{data_hash}_{pipeline_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, key: str):
        """Get cached result if available."""
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None
    
    def set(self, key: str, value):
        """Store result in cache with LRU eviction."""
        # Remove oldest entries if cache is full
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._access_times, key=self._access_times.get)
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
        
        self._cache[key] = value
        self._access_times[key] = time.time()
```

### 5. Testing Standards

#### Unit Test Structure
```python
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from components.widgets.matplotlib_widget import MatplotlibWidget

class TestMatplotlibWidget(unittest.TestCase):
    """Test suite for MatplotlibWidget functionality."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.widget = MatplotlibWidget()
        
        # Create test data
        self.wavenumbers = np.linspace(400, 4000, 1000)
        self.intensities = np.random.random(1000) * 1000
        
        # Add some signal peaks for testing auto-focus
        self.intensities[300:350] += 5000  # Strong peak at ~1400 cm⁻¹
        self.intensities[600:650] += 3000  # Medium peak at ~2800 cm⁻¹
    
    def test_detect_signal_range_valid_data(self):
        """Test signal range detection with valid spectral data."""
        result = self.widget.detect_signal_range(self.wavenumbers, self.intensities)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        start, end = result
        self.assertLess(start, end)
        self.assertGreaterEqual(start, self.wavenumbers.min())
        self.assertLessEqual(end, self.wavenumbers.max())
    
    def test_detect_signal_range_empty_data(self):
        """Test signal range detection with empty data."""
        empty_wavenumbers = np.array([])
        empty_intensities = np.array([])
        
        result = self.widget.detect_signal_range(empty_wavenumbers, empty_intensities)
        self.assertIsNone(result)
    
    def test_plot_spectra_with_auto_focus(self):
        """Test plotting with auto-focus enabled."""
        with patch.object(self.widget, 'detect_signal_range') as mock_detect:
            mock_detect.return_value = (1000, 3000)
            
            self.widget.plot_spectra(
                self.wavenumbers, 
                self.intensities, 
                auto_focus=True
            )
            
            mock_detect.assert_called_once()
    
    def test_plot_spectra_without_auto_focus(self):
        """Test plotting with auto-focus disabled."""
        with patch.object(self.widget, 'detect_signal_range') as mock_detect:
            self.widget.plot_spectra(
                self.wavenumbers, 
                self.intensities, 
                auto_focus=False
            )
            
            mock_detect.assert_not_called()

if __name__ == '__main__':
    unittest.main()
```

#### Integration Test Pattern
```python
class TestPreprocessingIntegration(unittest.TestCase):
    """Integration tests for preprocessing workflow."""
    
    def setUp(self):
        """Setup integration test environment."""
        self.test_data = pd.DataFrame({
            'wavenumber': np.linspace(400, 4000, 1000),
            'intensity': np.random.random(1000) * 1000
        }).set_index('wavenumber')
        
        self.preprocess_page = PreprocessPage()
        self.preprocess_page.set_data(self.test_data)
    
    def test_conditional_auto_focus_workflow(self):
        """Test complete conditional auto-focus workflow."""
        # Initially, auto-focus should be disabled
        self.assertFalse(self.preprocess_page._should_auto_focus())
        
        # Add a cropper step
        cropper_step = PipelineStep(
            category='range',
            method='Cropper',
            params={'range': (1000, 3000)},
            enabled=True
        )
        self.preprocess_page.add_pipeline_step(cropper_step)
        
        # Now auto-focus should be enabled
        self.assertTrue(self.preprocess_page._should_auto_focus())
        
        # Verify preview generation works with auto-focus
        self.preprocess_page.update_preview()
        
        # Check that matplotlib widget received auto_focus=True
        # (This would require mocking the matplotlib widget)
```

#### GUI Testing Guidelines
When testing GUI components, ensure the following for AI agent execution:
- Activate the appropriate virtual environment (uv or venv) before running Python scripts.
- Run GUI tests using the environment's Python executable.
- Wait for the GUI application to fully close before processing the next response, allowing the agent to capture all terminal output and messages.

### 6. Documentation Standards

#### Inline Documentation
```python
class ParameterWidget(QWidget):
    """
    Base parameter input widget for spectroscopic applications.
    
    This widget provides a standardized interface for parameter input
    with real-time validation, visual feedback, and integration with
    the preprocessing pipeline system.
    
    Attributes:
        param_name: Name of the parameter this widget controls
        config: Configuration dictionary with widget settings
        value_changed: Signal emitted when parameter value changes
        validation_changed: Signal emitted when validation state changes
    
    Example:
        >>> config = {
        ...     'type': 'float',
        ...     'min': 0.0,
        ...     'max': 100.0,
        ...     'default': 50.0,
        ...     'units': 'cm⁻¹'
        ... }
        >>> widget = ParameterWidget('range_min', config)
        >>> widget.value_changed.connect(on_parameter_changed)
    """
```

#### API Documentation
```python
def process_spectral_data(
    data: pd.DataFrame, 
    method: str, 
    parameters: Dict[str, Any],
    validate: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Process spectral data using specified method and parameters.
    
    Args:
        data: Spectral data with wavenumber index and intensity columns
        method: Name of preprocessing method to apply
        parameters: Method-specific parameters
        validate: Whether to validate inputs before processing
    
    Returns:
        Tuple containing:
            - Processed spectral data
            - Processing metadata including execution time and parameters used
    
    Raises:
        ValueError: If data format is invalid or method is unknown
        TypeError: If parameters are of wrong type
        ProcessingError: If processing fails
    
    Example:
        >>> data = load_spectral_data('sample.csv')
        >>> result, metadata = process_spectral_data(
        ...     data, 
        ...     'baseline_correction', 
        ...     {'method': 'polynomial', 'order': 3}
        ... )
    """
```

These development guidelines ensure consistent, maintainable, and high-quality code throughout the Raman spectroscopy application. Following these standards will make the codebase easier to understand, extend, and debug.