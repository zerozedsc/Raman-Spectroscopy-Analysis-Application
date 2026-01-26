# Contributing Guide

Guidelines for contributing to the Raman Spectroscopy Analysis Application.

## Minimal contribution workflow (current)
 1. Fork the repository.
 2. Create a branch for your change.
 3. Install from source (editable install): `pip install -e .`
 4. Run the smoke tests: `python smoke_tests.py`
 5. Open a pull request with a clear description and screenshots/logs if applicable.

---

<!--
The content below is temporarily hidden from the rendered documentation because it contains
placeholders and workflows that do not match the current repository state.
-->

<!--

## Getting Started

### Prerequisites

**Required**:
- Python 3.10 or higher
- Git
- UV package manager

**Optional**:
- PyQt6 development tools
- Qt Designer (for UI design)
- Sphinx (for documentation building)

### Initial Setup

#### 1. Fork and Clone

```bash
# Fork repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/raman-app.git
cd raman-app

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_ORG/raman-app.git
```

#### 2. Install UV Package Manager

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Verify installation
uv --version
```

#### 3. Create Virtual Environment

```bash
# Create virtual environment
uv venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux/macOS:
source .venv/bin/activate
```

#### 4. Install Dependencies

```bash
# Install core dependencies
uv pip install -r requirements.txt

# Install development dependencies
uv pip install -r requirements-dev.txt

# Install package in editable mode
uv pip install -e .
```

#### 5. Verify Installation

```bash
# Run application in development mode
python dev_runner.py

# Run tests
pytest

# Check code formatting
black --check .
ruff check .
```

#### 6. Install Pre-commit Hooks

```bash
# Install pre-commit
uv pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

---

## Development Workflow

### Branch Strategy

We use **Git Flow** branching model:

```
main            # Production-ready code
  ├── develop   # Integration branch
  ├── feature/* # Feature branches
  ├── hotfix/*  # Hotfix branches
  └── release/* # Release branches
```

### Creating a Feature Branch

```bash
# Update develop branch
git checkout develop
git pull upstream develop

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add your feature"

# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

### Commit Message Convention

We follow **Conventional Commits** specification:

**Format**:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring (no feature change)
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Other changes (dependencies, tools)

**Examples**:

```bash
# Feature
git commit -m "feat(preprocess): add wavelet denoising method"

# Bug fix
git commit -m "fix(analysis): correct PCA variance calculation"

# Documentation
git commit -m "docs(api): update preprocessing function signatures"

# Breaking change
git commit -m "feat(ml)!: change model export format

BREAKING CHANGE: Model export now uses JSON instead of pickle"
```

### Keeping Your Branch Updated

```bash
# Fetch upstream changes
git fetch upstream

# Rebase your branch on develop
git checkout feature/your-feature-name
git rebase upstream/develop

# If conflicts, resolve and continue
git add .
git rebase --continue

# Force push (only to your fork!)
git push origin feature/your-feature-name --force-with-lease
```

---

## Code Standards

### Python Style Guide

We follow **PEP 8** with these modifications:

- **Line length**: 100 characters (not 79)
- **String quotes**: Single quotes `'` (except docstrings use `"""`)
- **Indentation**: 4 spaces (no tabs)

### Code Formatting

We use **Black** for automatic formatting:

```bash
# Format all files
black .

# Format specific file
black path/to/file.py

# Check without modifying
black --check .
```

**Configuration** (in `pyproject.toml`):

```toml
[tool.black]
line-length = 100
target-version = ['py310']
include = '\.pyi?$'
extend-exclude = '''
/(
  | .venv
  | build
  | dist
)/
'''
```

### Linting

We use **Ruff** for fast linting:

```bash
# Lint all files
ruff check .

# Lint with auto-fix
ruff check --fix .

# Lint specific file
ruff check path/to/file.py
```

**Configuration** (in `pyproject.toml`):

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by black)
    "B008",  # function calls in argument defaults
]
```

### Type Hints

Use type hints for function signatures:

```python
from typing import Optional, List, Dict, Tuple, Union
import numpy as np
import numpy.typing as npt

def process_spectrum(
    spectrum: npt.NDArray[np.float64],
    method: str,
    params: Optional[Dict[str, Union[int, float, str]]] = None
) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
    """
    Process spectrum with given method.
    
    Args:
        spectrum: Input spectrum array
        method: Processing method name
        params: Method parameters (optional)
    
    Returns:
        Tuple of (processed spectrum, metadata)
    """
    if params is None:
        params = {}
    
    # Process
    result = apply_method(spectrum, method, **params)
    metadata = {'method': method, 'success': True}
    
    return result, metadata
```

### Naming Conventions

| Type     | Convention            | Example              |
| -------- | --------------------- | -------------------- |
| Module   | `snake_case`          | `data_loader.py`     |
| Class    | `PascalCase`          | `SpectrumProcessor`  |
| Function | `snake_case`          | `apply_baseline()`   |
| Variable | `snake_case`          | `spectrum_data`      |
| Constant | `UPPER_CASE`          | `MAX_ITERATIONS`     |
| Private  | `_leading_underscore` | `_internal_method()` |

### Docstrings

Use **Google style** docstrings:

```python
def apply_asls(
    spectrum: npt.NDArray[np.float64],
    lambda_: float = 1e5,
    p: float = 0.01,
    max_iter: int = 10
) -> npt.NDArray[np.float64]:
    """
    Apply Asymmetric Least Squares (AsLS) baseline correction.
    
    AsLS is an iterative baseline correction method that fits a smooth
    baseline with asymmetric weighting, penalizing positive residuals
    more heavily to avoid fitting spectral peaks.
    
    Args:
        spectrum: Input spectrum (1D array)
        lambda_: Smoothness parameter. Higher values create smoother
            baselines. Typical range: 1e2 to 1e9.
        p: Asymmetry parameter. Controls asymmetric weighting. Lower
            values increase asymmetry. Typical range: 0.001 to 0.1.
        max_iter: Maximum number of iterations. Default 10.
    
    Returns:
        Baseline-corrected spectrum (same shape as input)
    
    Raises:
        ValueError: If spectrum is empty or not 1D
        ValueError: If lambda_ <= 0 or p not in (0, 1)
    
    Examples:
        >>> spectrum = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        >>> corrected = apply_asls(spectrum, lambda_=1e5, p=0.01)
        >>> print(corrected.shape)
        (5,)
    
    References:
        Eilers, P. H. C., & Boelens, H. F. M. (2005). Baseline correction
        with asymmetric least squares smoothing. Leiden University Medical
        Centre Report, 1(1), 5.
    """
    # Implementation
    pass
```

### Import Organization

Organize imports in three groups:

```python
# Standard library
import os
import sys
from pathlib import Path
from typing import Optional, Dict

# Third-party packages
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from sklearn.decomposition import PCA

# Local imports
from functions.preprocess import apply_asls
from configs import AppConfig
from utils import validate_spectra_data
```

Use **isort** for automatic organization:

```bash
# Format imports
isort .

# Check without modifying
isort --check .
```

**Configuration** (in `pyproject.toml`):

```toml
[tool.isort]
profile = "black"
line_length = 100
known_first_party = ["functions", "pages", "components", "configs"]
```

---

## Adding Features

### Adding a Preprocessing Method

#### 1. Create Method Function

**File**: `functions/preprocess/your_method.py`

```python
import numpy as np
import numpy.typing as npt

def apply_your_method(
    spectrum: npt.NDArray[np.float64],
    param1: float = 1.0,
    param2: int = 10
) -> npt.NDArray[np.float64]:
    """
    Apply your custom preprocessing method.
    
    Detailed description of what the method does, algorithm overview,
    and when to use it.
    
    Args:
        spectrum: Input spectrum (1D array)
        param1: Description of parameter 1. Range: [min, max]
        param2: Description of parameter 2. Must be positive integer
    
    Returns:
        Processed spectrum (same shape as input)
    
    Raises:
        ValueError: If spectrum is invalid
        ValueError: If parameters are out of range
    
    Examples:
        >>> spectrum = np.array([1.0, 2.0, 3.0])
        >>> result = apply_your_method(spectrum, param1=2.0)
    
    References:
        Author et al. (Year). Paper title. Journal.
    """
    # Validate inputs
    if spectrum.ndim != 1:
        raise ValueError("Spectrum must be 1D array")
    if param1 <= 0:
        raise ValueError("param1 must be positive")
    if param2 <= 0:
        raise ValueError("param2 must be positive integer")
    
    # Algorithm implementation
    result = np.zeros_like(spectrum)
    # ... your algorithm ...
    
    return result
```

#### 2. Register Method

**File**: `functions/preprocess/registry.py`

```python
from functions.preprocess.your_method import apply_your_method

# Register in method registry
PREPROCESSING_METHODS = {
    'your_method': {
        'function': apply_your_method,
        'name': 'Your Method',
        'category': 'Baseline Correction',  # or 'Smoothing', 'Normalization', etc.
        'description': 'Brief description of your method',
        'params': {
            'param1': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 10.0,
                'label': 'Parameter 1',
                'tooltip': 'Detailed description for UI'
            },
            'param2': {
                'type': 'int',
                'default': 10,
                'min': 1,
                'max': 100,
                'label': 'Parameter 2',
                'tooltip': 'Another parameter description'
            }
        }
    }
}
```

#### 3. Add Parameter Constraints

**File**: `functions/preprocess/parameter_constraints.py`

```python
CONSTRAINTS = {
    'your_method': {
        'param1': {
            'min': 0.1,
            'max': 10.0,
            'default': 1.0,
            'recommended_range': (0.5, 5.0),
            'description': 'Controls the strength of the effect'
        },
        'param2': {
            'min': 1,
            'max': 100,
            'default': 10,
            'must_be_odd': False,
            'description': 'Number of iterations'
        }
    }
}
```

#### 4. Write Tests

**File**: `tests/unit/test_preprocess.py`

```python
import numpy as np
import pytest
from functions.preprocess.your_method import apply_your_method

class TestYourMethod:
    def test_basic_functionality(self):
        """Test basic operation"""
        spectrum = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        result = apply_your_method(spectrum)
        
        assert result.shape == spectrum.shape
        assert np.isfinite(result).all()
    
    def test_with_parameters(self):
        """Test with custom parameters"""
        spectrum = np.array([1.0, 2.0, 3.0])
        result = apply_your_method(spectrum, param1=2.0, param2=20)
        
        assert result.shape == spectrum.shape
    
    def test_invalid_input(self):
        """Test error handling"""
        with pytest.raises(ValueError, match="must be 1D"):
            apply_your_method(np.array([[1, 2], [3, 4]]))
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        spectrum = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="must be positive"):
            apply_your_method(spectrum, param1=-1.0)
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Single point
        result = apply_your_method(np.array([1.0]))
        assert result.shape == (1,)
        
        # All zeros
        result = apply_your_method(np.zeros(10))
        assert np.isfinite(result).all()
    
    @pytest.mark.parametrize("param1,param2", [
        (0.1, 1),
        (1.0, 10),
        (10.0, 100)
    ])
    def test_parameter_ranges(self, param1, param2):
        """Test various parameter combinations"""
        spectrum = np.random.randn(100)
        result = apply_your_method(spectrum, param1=param1, param2=param2)
        assert result.shape == spectrum.shape
```

#### 5. Document Method

**File**: `docs/analysis-methods/preprocessing.md`

Add section for your method:

````markdown
### Your Method

#### Overview

Detailed description of your method, when to use it, advantages/disadvantages.

#### Theory

Mathematical background, algorithm description, formulas.

#### Parameters

| Parameter | Type  | Range    | Default | Description |
| --------- | ----- | -------- | ------- | ----------- |
| param1    | float | 0.1-10.0 | 1.0     | Description |
| param2    | int   | 1-100    | 10      | Description |

#### Usage

```python
from functions.preprocess import apply_your_method

# Basic usage
corrected = apply_your_method(spectrum)

# With custom parameters
corrected = apply_your_method(
    spectrum,
    param1=2.0,
    param2=20
)
```

#### When to Use

- Use case 1
- Use case 2

#### Troubleshooting

Common issues and solutions.
````

#### 6. Update Changelog

**File**: `CHANGELOG.md`

```markdown
## [Unreleased]

### Added
- Added new preprocessing method: Your Method (#123)
```

### Adding an Analysis Method

#### 1. Create Analysis Function

**File**: `functions/analysis/your_analysis.py`

```python
import numpy as np
from typing import Dict, Optional

def apply_your_analysis(
    spectra: np.ndarray,
    param1: float = 1.0
) -> Dict:
    """
    Apply your custom analysis method.
    
    Args:
        spectra: Input spectra (2D array: samples × features)
        param1: Analysis parameter
    
    Returns:
        Dictionary with analysis results:
        - 'result': Main result array
        - 'metadata': Additional information
    """
    # Validate
    if spectra.ndim != 2:
        raise ValueError("Spectra must be 2D array")
    
    # Analyze
    result = compute_analysis(spectra, param1)
    
    return {
        'result': result,
        'metadata': {
            'param1': param1,
            'n_samples': spectra.shape[0],
            'n_features': spectra.shape[1]
        }
    }
```

#### 2. Integrate with AnalysisPage

**File**: `pages/analysis_page.py`

```python
def apply_your_analysis(self):
    """Apply your analysis method"""
    # Get parameters from UI
    param1 = self.param1_widget.get_value()
    
    # Apply analysis
    try:
        result = apply_your_analysis(
            self.data['spectra'],
            param1=param1
        )
        
        # Display results
        self.results_panel.set_analysis_results(result)
        
        # Show success message
        Toast.success("Analysis completed successfully")
        
    except Exception as e:
        logger.exception("Analysis failed")
        Toast.error(f"Analysis failed: {str(e)}")
```

#### 3. Add Tests

```python
def test_your_analysis():
    """Test your analysis method"""
    spectra = np.random.randn(100, 500)
    result = apply_your_analysis(spectra, param1=1.0)
    
    assert 'result' in result
    assert 'metadata' in result
    assert result['result'].shape[0] == 100
```

### Adding a Machine Learning Algorithm

#### 1. Create Training Function

**File**: `functions/ML/your_algorithm.py`

```python
from sklearn.base import BaseEstimator
from typing import Dict, Optional
import numpy as np

def train_your_algorithm(
    X: np.ndarray,
    y: np.ndarray,
    param1: float = 1.0,
    param2: int = 10,
    cv: int = 5,
    random_state: Optional[int] = None
) -> Dict:
    """
    Train your custom ML algorithm.
    
    Args:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,)
        param1: Algorithm hyperparameter
        param2: Another hyperparameter
        cv: Cross-validation folds
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with:
        - 'model': Trained model object
        - 'cv_scores': Cross-validation scores
        - 'params': Final parameters used
    """
    from sklearn.model_selection import cross_val_score
    from your_algorithm_package import YourAlgorithm
    
    # Create model
    model = YourAlgorithm(
        param1=param1,
        param2=param2,
        random_state=random_state
    )
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    # Train on full data
    model.fit(X, y)
    
    return {
        'model': model,
        'cv_scores': cv_scores,
        'params': {
            'param1': param1,
            'param2': param2
        }
    }
```

#### 2. Register in ML Page

**File**: `pages/machine_learning_page.py`

```python
# Add to algorithm registry
self.algorithms = {
    'svm': train_svm,
    'random_forest': train_random_forest,
    'xgboost': train_xgboost,
    'your_algorithm': train_your_algorithm,  # Add here
}

# Add parameter specifications
self.param_specs = {
    'your_algorithm': {
        'param1': {
            'type': 'float',
            'default': 1.0,
            'min': 0.1,
            'max': 10.0,
            'label': 'Parameter 1'
        },
        'param2': {
            'type': 'int',
            'default': 10,
            'min': 1,
            'max': 100,
            'label': 'Parameter 2'
        }
    }
}
```

### Adding a New Page

#### 1. Create Page Class

**File**: `pages/your_page.py`

```python
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt6.QtCore import pyqtSignal
from pages.base_page import BasePage
from components.toast import Toast

class YourPage(BasePage):
    """
    Your custom page for specific functionality.
    
    Signals:
        data_updated: Emitted when data is updated
    """
    
    data_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.setup_ui()
        self.connect_signals()
    
    def setup_ui(self):
        """Set up user interface"""
        layout = QVBoxLayout(self)
        
        # Add your widgets
        self.button = QPushButton("Do Something")
        layout.addWidget(self.button)
    
    def connect_signals(self):
        """Connect signals and slots"""
        self.button.clicked.connect(self.on_button_clicked)
    
    def on_button_clicked(self):
        """Handle button click"""
        try:
            # Do something
            result = self.process_data()
            
            # Emit signal
            self.data_updated.emit(result)
            
            # Show success
            Toast.success("Operation completed")
            
        except Exception as e:
            Toast.error(f"Operation failed: {str(e)}")
    
    def process_data(self):
        """Process data"""
        # Your processing logic
        return {'result': 'success'}
    
    def get_state(self):
        """Get page state for persistence"""
        return {
            'data': self.data
        }
    
    def set_state(self, state):
        """Restore page state"""
        self.data = state.get('data')
```

#### 2. Register Page

**File**: `components/page_registry.py`

```python
from pages.your_page import YourPage

# Register
PageRegistry.register_page(
    page_id='your_page',
    page_class=YourPage,
    metadata={
        'title': 'Your Page',
        'category': 'Analysis',
        'icon': 'icons/your_icon.png',
        'order': 50,
        'description': 'Description of your page'
    }
)
```

#### 3. Add to Main Application

**File**: `main.py`

```python
from pages.your_page import YourPage

class MainApplication(QMainWindow):
    def setup_pages(self):
        # ... existing pages ...
        
        # Add your page
        self.your_page = YourPage()
        self.tabs.add_page('your_page', self.your_page, "Your Page")
```

### Adding a Custom Widget

#### 1. Create Widget Class

**File**: `components/widgets/your_widget.py`

```python
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import pyqtSignal

class YourWidget(QWidget):
    """
    Your custom widget.
    
    Signals:
        value_changed: Emitted when value changes
    """
    
    value_changed = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = None
        self.setup_ui()
    
    def setup_ui(self):
        """Set up user interface"""
        layout = QVBoxLayout(self)
        
        self.label = QLabel("Your Widget")
        layout.addWidget(self.label)
    
    def get_value(self):
        """Get current value"""
        return self.value
    
    def set_value(self, value):
        """Set value"""
        if value != self.value:
            self.value = value
            self.update_ui()
            self.value_changed.emit(value)
    
    def update_ui(self):
        """Update UI to reflect current value"""
        self.label.setText(f"Value: {self.value}")
```

#### 2. Write Widget Tests

**File**: `tests/ui/test_your_widget.py`

```python
import pytest
from PyQt6.QtWidgets import QApplication
from components.widgets.your_widget import YourWidget

@pytest.fixture
def widget(qtbot):
    """Create widget for testing"""
    w = YourWidget()
    qtbot.addWidget(w)
    return w

def test_initial_state(widget):
    """Test initial widget state"""
    assert widget.get_value() is None

def test_set_value(widget, qtbot):
    """Test setting value"""
    with qtbot.waitSignal(widget.value_changed):
        widget.set_value(42)
    
    assert widget.get_value() == 42

def test_signal_emission(widget, qtbot):
    """Test signal emission"""
    with qtbot.waitSignal(widget.value_changed) as blocker:
        widget.set_value(123)
    
    assert blocker.args == [123]
```

---

## Testing Requirements

### Test Coverage Goals

- **Functions**: ≥ 80% coverage
- **Pages**: ≥ 70% coverage
- **Widgets**: ≥ 70% coverage
- **Overall**: ≥ 75% coverage

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html --cov-report=term

# Run specific test file
pytest tests/unit/test_preprocess.py

# Run specific test
pytest tests/unit/test_preprocess.py::TestAsLS::test_basic_functionality

# Run tests matching pattern
pytest -k "test_baseline"

# Run with verbose output
pytest -v

# Run with print statements
pytest -s
```

### Test Structure

```
tests/
├── unit/                   # Unit tests
│   ├── test_preprocess.py
│   ├── test_analysis.py
│   └── test_ml.py
├── integration/            # Integration tests
│   ├── test_pipeline.py
│   └── test_workflows.py
├── ui/                     # UI tests
│   ├── test_widgets.py
│   ├── test_pages.py
│   └── test_dialogs.py
└── fixtures/               # Shared fixtures
    └── conftest.py
```

### Writing Unit Tests

```python
import pytest
import numpy as np
from functions.preprocess import apply_asls

def test_basic_functionality():
    """Test basic operation"""
    spectrum = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    result = apply_asls(spectrum)
    
    assert result.shape == spectrum.shape
    assert np.isfinite(result).all()

@pytest.mark.parametrize("lambda_,p", [
    (1e3, 0.01),
    (1e5, 0.01),
    (1e7, 0.001)
])
def test_parameter_combinations(lambda_, p):
    """Test various parameter combinations"""
    spectrum = np.random.randn(100)
    result = apply_asls(spectrum, lambda_=lambda_, p=p)
    assert result.shape == spectrum.shape
```

### Writing Integration Tests

```python
def test_preprocessing_pipeline():
    """Test full preprocessing pipeline"""
    # Load data
    data = load_spectra_from_csv('test_data.csv')
    
    # Create pipeline
    pipeline = [
        {'method': 'asls', 'params': {'lambda_': 1e5}},
        {'method': 'savgol', 'params': {'window_length': 11}},
        {'method': 'vector_norm', 'params': {}}
    ]
    
    # Execute pipeline
    result = execute_preprocessing_pipeline(
        data['spectra'],
        pipeline
    )
    
    # Verify
    assert result.shape == data['spectra'].shape
    assert np.isfinite(result).all()
```

### Writing UI Tests

```python
def test_parameter_widget(qtbot):
    """Test parameter widget"""
    widget = FloatParameterWidget(
        label="Test",
        default_value=1.0,
        min_value=0.0,
        max_value=10.0
    )
    qtbot.addWidget(widget)
    
    # Test initial state
    assert widget.get_value() == 1.0
    
    # Test setting value
    with qtbot.waitSignal(widget.value_changed):
        widget.set_value(5.0)
    
    assert widget.get_value() == 5.0
    
    # Test validation
    widget.set_value(15.0)  # Out of range
    assert widget.is_valid() == False
```

---

## Documentation Standards

### Code Documentation

All public functions, classes, and methods must have docstrings following Google style.

### User Documentation

Update user guide when adding user-facing features:

**File**: `docs/user-guide/your-feature.md`

````markdown
# Your Feature

## Overview

What the feature does.

## How to Use

Step-by-step instructions with screenshots.

## Parameters

Description of all parameters.

## Examples

Practical examples.

## Troubleshooting

Common issues and solutions.
````

### API Documentation

Update API reference when adding public APIs:

**File**: `docs/api/your-module.md`

See existing API documentation for format.

### Building Documentation

```bash
# Install Sphinx
uv pip install sphinx sphinx-rtd-theme myst-parser

# Build HTML documentation
cd docs
make html

# View documentation
# Open docs/_build/html/index.html in browser

# Build PDF (requires LaTeX)
make latexpdf
```

---

## Pull Request Process

### Before Creating PR

**Checklist**:
- [ ] Code follows style guide
- [ ] All tests pass
- [ ] Added tests for new features
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Commit messages follow convention
- [ ] Branch is up-to-date with develop

### Creating PR

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature
   ```

2. **Create PR on GitHub**:
   - Base branch: `develop`
   - Compare branch: `feature/your-feature`
   - Fill out PR template

3. **PR Title**: Follow conventional commits format
   ```
   feat(preprocess): add wavelet denoising method
   ```

4. **PR Description Template**:
   ```markdown
   ## Description
   Brief description of changes.
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   Describe testing done.
   
   ## Checklist
   - [ ] Code follows style guide
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] Changelog updated
   
   ## Screenshots (if applicable)
   
   ## Related Issues
   Closes #123
   ```

### PR Review Process

1. **Automated Checks**:
   - Code formatting (Black, Ruff)
   - Tests (pytest)
   - Coverage (pytest-cov)
   - Type checking (mypy)

2. **Manual Review**:
   - Code quality
   - Design decisions
   - Test coverage
   - Documentation

3. **Addressing Feedback**:
   ```bash
   # Make changes
   git add .
   git commit -m "fix: address review comments"
   git push origin feature/your-feature
   ```

4. **Approval and Merge**:
   - Requires 1-2 approvals
   - All checks must pass
   - No conflicts with base branch
   - Squash and merge to develop

---

## Code Review Guidelines

### For Authors

**Before Requesting Review**:
- Self-review your code
- Run all tests locally
- Update documentation
- Keep PR focused and small

**During Review**:
- Respond to all comments
- Ask questions if unclear
- Be open to feedback
- Update PR based on feedback

### For Reviewers

**What to Look For**:
- **Correctness**: Does code work as intended?
- **Tests**: Are there adequate tests?
- **Style**: Does code follow style guide?
- **Design**: Is the approach sound?
- **Documentation**: Is code well-documented?
- **Performance**: Any performance concerns?
- **Security**: Any security issues?

**Review Etiquette**:
- Be constructive and respectful
- Explain reasoning for suggestions
- Distinguish between blockers and suggestions
- Acknowledge good work

**Comment Format**:
```
**Blocking**: Critical issue that must be fixed

**Suggestion**: Optional improvement

**Question**: Clarification needed

**Nit**: Minor style issue (non-blocking)
```

---

## Release Process

### Versioning

We use **Semantic Versioning** (SemVer):

```
MAJOR.MINOR.PATCH

1.2.3
│ │ │
│ │ └─ Patch: Bug fixes
│ └─── Minor: New features (backward compatible)
└───── Major: Breaking changes
```

### Release Workflow

#### 1. Create Release Branch

```bash
# From develop branch
git checkout develop
git pull upstream develop

# Create release branch
git checkout -b release/v1.2.0

# Update version in pyproject.toml
# Update CHANGELOG.md
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 1.2.0"
git push origin release/v1.2.0
```

#### 2. Test Release

```bash
# Run full test suite
pytest

# Build executable
python build_scripts/test_build_executable.py

# Test executable manually
```

#### 3. Finalize Release

```bash
# Merge to main
git checkout main
git merge --no-ff release/v1.2.0
git tag -a v1.2.0 -m "Release version 1.2.0"
git push upstream main --tags

# Merge back to develop
git checkout develop
git merge --no-ff release/v1.2.0
git push upstream develop

# Delete release branch
git branch -d release/v1.2.0
git push origin --delete release/v1.2.0
```

#### 4. Create GitHub Release

1. Go to GitHub Releases
2. Click "Draft a new release"
3. Select tag: `v1.2.0`
4. Release title: `Version 1.2.0`
5. Copy changelog content to description
6. Upload executables/installers
7. Publish release

### Hotfix Process

For urgent bug fixes:

```bash
# Create hotfix branch from main
git checkout main
git checkout -b hotfix/v1.2.1

# Fix bug
# Update version
# Update changelog

# Test
pytest

# Merge to main
git checkout main
git merge --no-ff hotfix/v1.2.1
git tag -a v1.2.1 -m "Hotfix version 1.2.1"

# Merge to develop
git checkout develop
git merge --no-ff hotfix/v1.2.1

# Push
git push upstream main develop --tags
```

---

## Getting Help

### Resources

- **Documentation**: https://raman-spectroscopy-analysis-application.readthedocs.io/en/latest/
- **Issues**: https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues
- **Discussions**: https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/discussions

### Asking Questions

When asking for help:

1. **Search existing issues** first
2. **Provide context**: What are you trying to do?
3. **Include details**: Code snippets, error messages, logs
4. **Describe what you've tried**
5. **Use proper formatting** (Markdown code blocks)

### Reporting Bugs

Use the bug report template:

```markdown
**Describe the bug**
Clear description of the bug.

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What should happen.

**Screenshots**
If applicable.

**Environment**
- OS: [Windows 10/macOS 13/Ubuntu 22.04]
- Python version: [3.10/3.11/3.12]
- App version: [1.2.0]

**Additional context**
Any other relevant information.
```

---

## See Also

- [Architecture Guide](architecture.md) - System design
- [Build System](build-system.md) - Building and packaging
- [Testing Guide](testing.md) - Testing strategy
- [API Reference](../api/index.md) - Complete API documentation

---

**Last Updated**: 2026-01-26

-->
