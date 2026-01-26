# Testing Guide

This page is being revised.

## Smoke tests (current)
 The file `smoke_tests.py` is designed to be fast and minimal. It checks that key plotting / widget update paths are importable and non-crashing.

---

<!--
The content below is temporarily hidden from the rendered documentation because it contains
placeholders and a test-suite structure that does not match the current repository state.
-->

<!--

## Testing Strategy

### Test Pyramid

```
      ╱╲
     ╱  ╲    UI Tests (E2E)
    ╱    ╲   - Test complete workflows
   ╱──────╲  - User interactions
  ╱        ╲
 ╱          ╲ Integration Tests
╱            ╲ - Test component interactions
──────────────── - Test data flow
                 
══════════════════ Unit Tests
                   - Test individual functions
                   - Test edge cases
                   - Fast and isolated
```

### Testing Goals

- **Fast**: Tests should run quickly
- **Isolated**: Tests should be independent
- **Deterministic**: Same input → same output
- **Comprehensive**: Cover critical paths and edge cases

### Coverage Targets

| Component Type | Coverage Goal |
| -------------- | ------------- |
| Core Functions | ≥ 80%         |
| Business Logic | ≥ 80%         |
| Pages          | ≥ 70%         |
| Widgets        | ≥ 70%         |
| Overall        | ≥ 75%         |

---

## Test Structure

### Directory Organization

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures
│
├── unit/                       # Unit tests
│   ├── __init__.py
│   ├── test_preprocess.py      # Preprocessing functions
│   ├── test_analysis.py        # Analysis functions
│   ├── test_ml.py              # ML functions
│   ├── test_data_loader.py     # Data I/O
│   └── test_utils.py           # Utility functions
│
├── integration/                # Integration tests
│   ├── __init__.py
│   ├── test_pipeline.py        # Preprocessing pipelines
│   ├── test_workflows.py       # Complete workflows
│   └── test_page_interactions.py
│
├── ui/                         # UI tests
│   ├── __init__.py
│   ├── test_widgets.py         # Widget tests
│   ├── test_pages.py           # Page tests
│   └── test_dialogs.py         # Dialog tests
│
└── fixtures/                   # Test data and fixtures
    ├── sample_spectra.csv
    ├── sample_labels.txt
    └── test_models/
```

### Naming Conventions

- **Test files**: `test_*.py`
- **Test classes**: `Test*` or `*Tests`
- **Test functions**: `test_*`
- **Fixtures**: Descriptive names (e.g., `sample_spectrum`, `mock_data`)

---

## Unit Testing

Unit tests verify individual functions in isolation.

### Basic Test Structure

```python
import numpy as np
import pytest
from functions.preprocess import apply_asls

class TestAsLS:
    """Test AsLS baseline correction"""
    
    def test_basic_functionality(self):
        """Test basic operation"""
        spectrum = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        result = apply_asls(spectrum)
        
        # Verify output shape
        assert result.shape == spectrum.shape
        
        # Verify no NaN/Inf values
        assert np.isfinite(result).all()
    
    def test_with_custom_parameters(self):
        """Test with custom parameters"""
        spectrum = np.random.randn(100)
        result = apply_asls(
            spectrum,
            lambda_=1e6,
            p=0.001,
            max_iter=15
        )
        
        assert result.shape == spectrum.shape
        assert np.isfinite(result).all()
    
    def test_invalid_input_shape(self):
        """Test error handling for invalid input"""
        # 2D array should raise error
        with pytest.raises(ValueError, match="must be 1D"):
            apply_asls(np.array([[1, 2], [3, 4]]))
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        spectrum = np.array([1.0, 2.0, 3.0])
        
        # Negative lambda should raise error
        with pytest.raises(ValueError, match="lambda.*positive"):
            apply_asls(spectrum, lambda_=-1.0)
        
        # Invalid p (not in 0-1)
        with pytest.raises(ValueError, match="p.*between 0 and 1"):
            apply_asls(spectrum, p=1.5)
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Single point
        result = apply_asls(np.array([1.0]))
        assert result.shape == (1,)
        
        # All zeros
        result = apply_asls(np.zeros(10))
        assert np.allclose(result, 0.0)
        
        # Constant spectrum
        result = apply_asls(np.ones(10))
        assert np.isfinite(result).all()
    
    @pytest.mark.parametrize("lambda_,p", [
        (1e3, 0.01),
        (1e5, 0.01),
        (1e7, 0.001),
        (1e9, 0.1)
    ])
    def test_parameter_ranges(self, lambda_, p):
        """Test various parameter combinations"""
        spectrum = np.random.randn(100)
        result = apply_asls(spectrum, lambda_=lambda_, p=p)
        
        assert result.shape == spectrum.shape
        assert np.isfinite(result).all()
```

### Testing Preprocessing Functions

**File**: `tests/unit/test_preprocess.py`

```python
import numpy as np
import pytest
from functions.preprocess import (
    apply_asls,
    apply_savgol,
    apply_snv,
    apply_vector_norm
)

class TestBaselineCorrection:
    """Test baseline correction methods"""
    
    @pytest.fixture
    def spectrum_with_baseline(self):
        """Create spectrum with known baseline"""
        x = np.linspace(0, 100, 1000)
        baseline = 5 + 0.01 * x
        peaks = np.exp(-((x - 30) ** 2) / 50)
        return baseline + peaks
    
    def test_removes_baseline(self, spectrum_with_baseline):
        """Test that baseline is removed"""
        corrected = apply_asls(spectrum_with_baseline, lambda_=1e6)
        
        # Corrected spectrum should have lower mean
        assert corrected.mean() < spectrum_with_baseline.mean()
        
        # Should preserve peak shape
        peak_idx = np.argmax(spectrum_with_baseline)
        assert np.argmax(corrected) == peak_idx

class TestSmoothing:
    """Test smoothing methods"""
    
    @pytest.fixture
    def noisy_spectrum(self):
        """Create noisy spectrum"""
        x = np.linspace(0, 10, 100)
        signal = np.sin(x)
        noise = np.random.normal(0, 0.1, 100)
        return signal + noise
    
    def test_savgol_smoothing(self, noisy_spectrum):
        """Test Savitzky-Golay smoothing"""
        smoothed = apply_savgol(
            noisy_spectrum,
            window_length=11,
            polyorder=3
        )
        
        # Smoothed should have lower variance
        assert np.var(smoothed) < np.var(noisy_spectrum)
        
        # Should preserve shape
        assert smoothed.shape == noisy_spectrum.shape

class TestNormalization:
    """Test normalization methods"""
    
    def test_vector_norm(self):
        """Test vector normalization"""
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = apply_vector_norm(spectrum)
        
        # L2 norm should be 1
        assert np.isclose(np.linalg.norm(normalized), 1.0)
    
    def test_snv(self):
        """Test Standard Normal Variate"""
        spectrum = np.random.randn(100) * 5 + 10
        normalized = apply_snv(spectrum)
        
        # Mean should be ~0, std should be ~1
        assert np.isclose(normalized.mean(), 0.0, atol=1e-10)
        assert np.isclose(normalized.std(), 1.0, atol=1e-10)
```

### Testing Analysis Functions

**File**: `tests/unit/test_analysis.py`

```python
import numpy as np
import pytest
from functions.ML import apply_pca, apply_kmeans

class TestPCA:
    """Test PCA analysis"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        return np.random.randn(100, 50)
    
    def test_pca_dimensionality_reduction(self, sample_data):
        """Test PCA reduces dimensions"""
        result = apply_pca(sample_data, n_components=5)
        
        assert 'scores' in result
        assert result['scores'].shape == (100, 5)
        assert 'loadings' in result
        assert result['loadings'].shape == (50, 5)
    
    def test_pca_variance_explained(self, sample_data):
        """Test variance explained"""
        result = apply_pca(sample_data, n_components=0.95)
        
        assert 'explained_variance' in result
        assert result['explained_variance'].sum() >= 0.95
    
    def test_pca_whitening(self, sample_data):
        """Test whitening"""
        result = apply_pca(sample_data, n_components=5, whiten=True)
        scores = result['scores']
        
        # Whitened scores should have unit variance
        assert np.allclose(scores.std(axis=0), 1.0, atol=0.1)

class TestClustering:
    """Test clustering methods"""
    
    @pytest.fixture
    def clustered_data(self):
        """Create data with known clusters"""
        np.random.seed(42)
        cluster1 = np.random.randn(50, 10) + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cluster2 = np.random.randn(50, 10) + [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        return np.vstack([cluster1, cluster2])
    
    def test_kmeans_finds_clusters(self, clustered_data):
        """Test K-means finds correct number of clusters"""
        result = apply_kmeans(clustered_data, n_clusters=2)
        
        assert 'labels' in result
        assert len(np.unique(result['labels'])) == 2
        assert len(result['labels']) == 100
```

### Testing Machine Learning

**File**: `tests/unit/test_ml.py`

```python
import numpy as np
import pytest
from sklearn.datasets import make_classification
from functions.ML import train_svm, train_random_forest

class TestSVM:
    """Test SVM training"""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification dataset"""
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    def test_svm_training(self, classification_data):
        """Test SVM trains successfully"""
        X, y = classification_data
        result = train_svm(X, y, kernel='rbf', C=1.0)
        
        assert 'model' in result
        assert 'cv_scores' in result
        assert len(result['cv_scores']) == 5  # Default 5-fold CV
    
    def test_svm_predictions(self, classification_data):
        """Test SVM makes predictions"""
        X, y = classification_data
        result = train_svm(X, y)
        model = result['model']
        
        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert set(predictions) == set(y)

class TestRandomForest:
    """Test Random Forest training"""
    
    def test_feature_importance(self):
        """Test feature importance calculation"""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            random_state=42
        )
        
        result = train_random_forest(X, y, n_estimators=50)
        importance = result['feature_importance']
        
        assert len(importance) == 10
        assert np.isclose(importance.sum(), 1.0)
```

---

## Integration Testing

Integration tests verify that multiple components work together correctly.

### Testing Preprocessing Pipelines

**File**: `tests/integration/test_pipeline.py`

```python
import numpy as np
import pytest
from functions.preprocess.pipeline import execute_preprocessing_pipeline
from functions.data_loader import load_spectra_from_csv

class TestPreprocessingPipeline:
    """Test preprocessing pipeline execution"""
    
    @pytest.fixture
    def sample_spectra(self):
        """Create sample spectra"""
        return np.random.randn(10, 100)
    
    def test_single_step_pipeline(self, sample_spectra):
        """Test pipeline with single step"""
        pipeline = [
            {
                'method': 'asls',
                'params': {'lambda_': 1e5, 'p': 0.01}
            }
        ]
        
        result = execute_preprocessing_pipeline(
            sample_spectra,
            pipeline
        )
        
        assert result.shape == sample_spectra.shape
        assert np.isfinite(result).all()
    
    def test_multi_step_pipeline(self, sample_spectra):
        """Test pipeline with multiple steps"""
        pipeline = [
            {'method': 'asls', 'params': {'lambda_': 1e5}},
            {'method': 'savgol', 'params': {'window_length': 11, 'polyorder': 3}},
            {'method': 'vector_norm', 'params': {}}
        ]
        
        result = execute_preprocessing_pipeline(
            sample_spectra,
            pipeline
        )
        
        assert result.shape == sample_spectra.shape
        
        # Check normalization worked
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_pipeline_with_progress_callback(self, sample_spectra):
        """Test progress reporting"""
        pipeline = [
            {'method': 'asls', 'params': {}},
            {'method': 'savgol', 'params': {}},
        ]
        
        progress_calls = []
        
        def progress_callback(current, total, message):
            progress_calls.append((current, total, message))
        
        execute_preprocessing_pipeline(
            sample_spectra,
            pipeline,
            progress_callback=progress_callback
        )
        
        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == progress_calls[-1][1]
    
    def test_pipeline_error_handling(self, sample_spectra):
        """Test error handling in pipeline"""
        pipeline = [
            {'method': 'invalid_method', 'params': {}}
        ]
        
        with pytest.raises(ValueError, match="Unknown method"):
            execute_preprocessing_pipeline(sample_spectra, pipeline)
```

### Testing Workflows

**File**: `tests/integration/test_workflows.py`

```python
import numpy as np
import pytest
import tempfile
from pathlib import Path
from functions.data_loader import load_spectra_from_csv, save_spectra_to_csv
from functions.preprocess.pipeline import execute_preprocessing_pipeline
from functions.ML import apply_pca, train_svm

class TestCompleteWorkflow:
    """Test complete analysis workflow"""
    
    @pytest.fixture
    def temp_csv(self):
        """Create temporary CSV file"""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False
        ) as f:
            # Write header
            f.write('Wavenumber,' + ','.join([f'Sample{i}' for i in range(5)]) + '\n')
            
            # Write data
            for wn in range(100):
                values = np.random.randn(5)
                f.write(f'{wn},' + ','.join(map(str, values)) + '\n')
            
            filepath = f.name
        
        yield filepath
        
        # Cleanup
        Path(filepath).unlink()
    
    def test_load_preprocess_analyze_workflow(self, temp_csv):
        """Test complete workflow: Load → Preprocess → Analyze"""
        # Step 1: Load data
        data = load_spectra_from_csv(temp_csv)
        assert 'spectra' in data
        assert 'wavenumbers' in data
        
        # Step 2: Preprocess
        pipeline = [
            {'method': 'asls', 'params': {'lambda_': 1e5}},
            {'method': 'vector_norm', 'params': {}}
        ]
        preprocessed = execute_preprocessing_pipeline(
            data['spectra'],
            pipeline
        )
        
        # Step 3: Analyze with PCA
        pca_result = apply_pca(preprocessed, n_components=2)
        
        assert 'scores' in pca_result
        assert pca_result['scores'].shape[0] == 5  # 5 samples
        assert pca_result['scores'].shape[1] == 2  # 2 components
    
    def test_train_evaluate_workflow(self, temp_csv):
        """Test ML workflow: Load → Preprocess → Train → Evaluate"""
        # Load and preprocess
        data = load_spectra_from_csv(temp_csv)
        preprocessed = execute_preprocessing_pipeline(
            data['spectra'],
            [{'method': 'vector_norm', 'params': {}}]
        )
        
        # Create labels
        labels = np.array([0, 0, 1, 1, 1])
        
        # Train model
        result = train_svm(preprocessed, labels, cv=2)
        
        assert 'model' in result
        assert 'cv_scores' in result
        assert len(result['cv_scores']) == 2
```

### Testing Page Interactions

**File**: `tests/integration/test_page_interactions.py`

```python
import pytest
from PyQt6.QtCore import Qt
from pages.data_package_page import DataPackagePage
from pages.preprocess_page import PreprocessPage

class TestPageInteractions:
    """Test interactions between pages"""
    
    @pytest.fixture
    def data_page(self, qtbot):
        """Create DataPackagePage"""
        page = DataPackagePage()
        qtbot.addWidget(page)
        return page
    
    @pytest.fixture
    def preprocess_page(self, qtbot):
        """Create PreprocessPage"""
        page = PreprocessPage()
        qtbot.addWidget(page)
        return page
    
    def test_data_flow_between_pages(self, data_page, preprocess_page, qtbot):
        """Test data flows from DataPackage to Preprocess page"""
        # Connect signal
        data_page.data_loaded.connect(preprocess_page.load_data)
        
        # Load data in DataPackagePage
        test_data = {
            'spectra': np.random.randn(10, 100),
            'wavenumbers': np.arange(100),
            'labels': [f'Sample{i}' for i in range(10)]
        }
        
        # Emit signal
        with qtbot.waitSignal(data_page.data_loaded):
            data_page.data_loaded.emit(test_data)
        
        # Verify PreprocessPage received data
        assert preprocess_page.data is not None
        assert np.array_equal(
            preprocess_page.data['spectra'],
            test_data['spectra']
        )
```

---

## UI Testing

UI tests verify Qt widgets and user interactions using pytest-qt.

### Testing Widgets

**File**: `tests/ui/test_widgets.py`

```python
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from components.widgets import (
    FloatParameterWidget,
    ChoiceParameterWidget,
    PipelineBuilder
)

class TestFloatParameterWidget:
    """Test float parameter widget"""
    
    @pytest.fixture
    def widget(self, qtbot):
        """Create widget"""
        w = FloatParameterWidget(
            label="Test Parameter",
            default_value=1.0,
            min_value=0.0,
            max_value=10.0
        )
        qtbot.addWidget(w)
        return w
    
    def test_initial_value(self, widget):
        """Test initial value is set correctly"""
        assert widget.get_value() == 1.0
    
    def test_set_valid_value(self, widget, qtbot):
        """Test setting valid value"""
        with qtbot.waitSignal(widget.value_changed):
            widget.set_value(5.0)
        
        assert widget.get_value() == 5.0
        assert widget.is_valid()
    
    def test_set_invalid_value(self, widget):
        """Test setting invalid value"""
        widget.set_value(15.0)  # Out of range
        
        assert not widget.is_valid()
    
    def test_value_changed_signal(self, widget, qtbot):
        """Test value_changed signal is emitted"""
        with qtbot.waitSignal(widget.value_changed) as blocker:
            widget.set_value(7.5)
        
        assert blocker.args == [7.5]
    
    def test_reset_to_default(self, widget, qtbot):
        """Test reset to default"""
        widget.set_value(8.0)
        
        with qtbot.waitSignal(widget.value_changed):
            widget.reset_to_default()
        
        assert widget.get_value() == 1.0

class TestChoiceParameterWidget:
    """Test choice parameter widget"""
    
    @pytest.fixture
    def widget(self, qtbot):
        """Create widget"""
        w = ChoiceParameterWidget(
            label="Method",
            choices=['Option A', 'Option B', 'Option C'],
            default_value='Option A'
        )
        qtbot.addWidget(w)
        return w
    
    def test_initial_selection(self, widget):
        """Test initial selection"""
        assert widget.get_value() == 'Option A'
    
    def test_change_selection(self, widget, qtbot):
        """Test changing selection"""
        with qtbot.waitSignal(widget.value_changed):
            widget.set_value('Option B')
        
        assert widget.get_value() == 'Option B'
    
    def test_add_choice(self, widget):
        """Test adding new choice"""
        widget.add_choice('Option D')
        
        choices = widget.get_choices()
        assert 'Option D' in choices

class TestPipelineBuilder:
    """Test pipeline builder widget"""
    
    @pytest.fixture
    def widget(self, qtbot):
        """Create widget"""
        w = PipelineBuilder()
        qtbot.addWidget(w)
        return w
    
    def test_add_step(self, widget, qtbot):
        """Test adding preprocessing step"""
        with qtbot.waitSignal(widget.step_added):
            step_id = widget.add_step(
                'asls',
                {'lambda_': 1e5, 'p': 0.01}
            )
        
        assert step_id is not None
        
        pipeline = widget.get_pipeline()
        assert len(pipeline) == 1
        assert pipeline[0]['method'] == 'asls'
    
    def test_remove_step(self, widget, qtbot):
        """Test removing step"""
        step_id = widget.add_step('asls', {})
        
        with qtbot.waitSignal(widget.step_removed):
            widget.remove_step(step_id)
        
        pipeline = widget.get_pipeline()
        assert len(pipeline) == 0
    
    def test_reorder_steps(self, widget):
        """Test reordering steps"""
        step1 = widget.add_step('asls', {})
        step2 = widget.add_step('savgol', {})
        
        widget.move_step_up(step2)
        
        pipeline = widget.get_pipeline()
        assert pipeline[0]['method'] == 'savgol'
        assert pipeline[1]['method'] == 'asls'
    
    def test_pipeline_validation(self, widget):
        """Test pipeline validation"""
        widget.add_step('asls', {})
        widget.add_step('vector_norm', {})
        
        is_valid, errors = widget.validate_pipeline()
        assert is_valid
        assert len(errors) == 0
```

### Testing Pages

**File**: `tests/ui/test_pages.py`

```python
import pytest
import numpy as np
from PyQt6.QtCore import Qt
from pages.preprocess_page import PreprocessPage

class TestPreprocessPage:
    """Test preprocessing page"""
    
    @pytest.fixture
    def page(self, qtbot):
        """Create page"""
        p = PreprocessPage()
        qtbot.addWidget(p)
        return p
    
    def test_page_initialization(self, page):
        """Test page initializes correctly"""
        assert page.data is None
        assert page.pipeline_builder is not None
    
    def test_load_data(self, page, qtbot):
        """Test loading data"""
        test_data = {
            'spectra': np.random.randn(10, 100),
            'wavenumbers': np.arange(100)
        }
        
        page.load_data(test_data)
        
        assert page.data is not None
        assert np.array_equal(page.data['spectra'], test_data['spectra'])
    
    def test_apply_pipeline(self, page, qtbot):
        """Test applying preprocessing pipeline"""
        # Load data
        page.load_data({
            'spectra': np.random.randn(10, 100),
            'wavenumbers': np.arange(100)
        })
        
        # Add pipeline steps
        page.pipeline_builder.add_step('asls', {'lambda_': 1e5})
        page.pipeline_builder.add_step('vector_norm', {})
        
        # Apply pipeline
        with qtbot.waitSignal(page.data_processed, timeout=5000):
            page.apply_pipeline()
    
    def test_preview_update(self, page, qtbot):
        """Test preview updates when parameters change"""
        page.load_data({
            'spectra': np.random.randn(10, 100),
            'wavenumbers': np.arange(100)
        })
        
        page.pipeline_builder.add_step('asls', {})
        
        # Trigger preview update
        page.update_preview()
        
        # Check preview was updated
        assert page.preview_spectrum is not None
```

### Testing Dialogs

**File**: `tests/ui/test_dialogs.py`

```python
import pytest
from PyQt6.QtCore import Qt
from components.widgets import (
    ParameterEditorDialog,
    MultiGroupDialog
)

class TestParameterEditorDialog:
    """Test parameter editor dialog"""
    
    @pytest.fixture
    def dialog(self, qtbot):
        """Create dialog"""
        d = ParameterEditorDialog(
            method='asls',
            current_params={'lambda_': 1e5, 'p': 0.01}
        )
        qtbot.addWidget(d)
        return d
    
    def test_initial_values(self, dialog):
        """Test initial parameter values"""
        params = dialog.get_parameters()
        
        assert params['lambda_'] == 1e5
        assert params['p'] == 0.01
    
    def test_accept_dialog(self, dialog, qtbot):
        """Test accepting dialog with valid parameters"""
        # Modify parameters
        dialog.set_parameter('lambda_', 1e6)
        
        # Accept
        qtbot.mouseClick(dialog.ok_button, Qt.MouseButton.LeftButton)
        
        assert dialog.result() == dialog.DialogCode.Accepted
        
        params = dialog.get_parameters()
        assert params['lambda_'] == 1e6
    
    def test_reject_invalid_parameters(self, dialog, qtbot):
        """Test rejection of invalid parameters"""
        # Set invalid value
        dialog.set_parameter('p', 1.5)  # Out of range
        
        # Try to accept - should show error
        qtbot.mouseClick(dialog.ok_button, Qt.MouseButton.LeftButton)
        
        # Dialog should not close
        assert dialog.isVisible()

class TestMultiGroupDialog:
    """Test multi-group comparison dialog"""
    
    @pytest.fixture
    def dialog(self, qtbot):
        """Create dialog"""
        groups = {
            'Group A': np.random.randn(20, 100),
            'Group B': np.random.randn(20, 100),
            'Group C': np.random.randn(20, 100)
        }
        
        d = MultiGroupDialog(groups)
        qtbot.addWidget(d)
        return d
    
    def test_group_selection(self, dialog):
        """Test selecting groups for comparison"""
        dialog.select_group('Group A')
        dialog.select_group('Group B')
        
        selected = dialog.get_selected_groups()
        assert 'Group A' in selected
        assert 'Group B' in selected
    
    def test_test_method_selection(self, dialog):
        """Test selecting statistical test"""
        dialog.set_test_method('ANOVA')
        
        config = dialog.get_configuration()
        assert config['test_method'] == 'ANOVA'
```

---

## Test Fixtures

Fixtures provide reusable test data and setup.

### Common Fixtures

**File**: `tests/conftest.py`

```python
import pytest
import numpy as np
import tempfile
from pathlib import Path
from PyQt6.QtWidgets import QApplication

@pytest.fixture(scope='session')
def qapp():
    """Create QApplication for UI tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app

@pytest.fixture
def sample_spectrum():
    """Create single spectrum"""
    return np.random.randn(100)

@pytest.fixture
def sample_spectra():
    """Create multiple spectra"""
    return np.random.randn(10, 100)

@pytest.fixture
def labeled_spectra():
    """Create spectra with labels"""
    spectra = np.random.randn(20, 100)
    labels = np.array([0] * 10 + [1] * 10)
    return spectra, labels

@pytest.fixture
def spectrum_with_peaks():
    """Create spectrum with Gaussian peaks"""
    x = np.linspace(0, 100, 1000)
    spectrum = np.zeros_like(x)
    
    # Add peaks
    peaks = [30, 50, 70]
    for peak in peaks:
        spectrum += np.exp(-((x - peak) ** 2) / 10)
    
    # Add baseline
    spectrum += 5 + 0.01 * x
    
    # Add noise
    spectrum += np.random.normal(0, 0.1, len(x))
    
    return x, spectrum

@pytest.fixture
def temp_csv_file():
    """Create temporary CSV file with test data"""
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.csv',
        delete=False
    ) as f:
        # Write header
        f.write('Wavenumber,Sample1,Sample2,Sample3\n')
        
        # Write data
        for i in range(100):
            f.write(f'{i},{i*0.1},{i*0.2},{i*0.3}\n')
        
        filepath = f.name
    
    yield filepath
    
    # Cleanup
    Path(filepath).unlink()

@pytest.fixture
def mock_config():
    """Create mock configuration"""
    return {
        'WINDOW_TITLE': 'Test App',
        'WINDOW_WIDTH': 800,
        'WINDOW_HEIGHT': 600,
        'THEME': 'light',
        'DEFAULT_LOCALE': 'en'
    }
```

### Parametrized Fixtures

```python
@pytest.fixture(params=[10, 50, 100])
def various_spectrum_sizes(request):
    """Create spectra of various sizes"""
    n_features = request.param
    return np.random.randn(5, n_features)

@pytest.fixture(params=['linear', 'rbf', 'poly'])
def svm_kernels(request):
    """Provide different SVM kernels"""
    return request.param
```

---

## Code Coverage

### Running with Coverage

```bash
# Run tests with coverage
pytest --cov=. --cov-report=html --cov-report=term

# Run specific test file with coverage
pytest tests/unit/test_preprocess.py --cov=functions.preprocess

# Show missing lines
pytest --cov=. --cov-report=term-missing
```

### Coverage Configuration

**File**: `pyproject.toml`

```toml
[tool.coverage.run]
source = ["."]
omit = [
    "*/tests/*",
    "*/build/*",
    "*/dist/*",
    "*/.venv/*",
    "setup.py",
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false

exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]
```

### Coverage Reports

```bash
# Generate HTML report
pytest --cov=. --cov-report=html

# Open in browser
# Open htmlcov/index.html

# Generate XML report (for CI)
pytest --cov=. --cov-report=xml

# Generate JSON report
pytest --cov=. --cov-report=json
```

---

## Continuous Testing

### Pre-commit Hooks

**File**: `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.10
  
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.285
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
  
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

**Install hooks**:
```bash
pre-commit install
```

### GitHub Actions

**File**: `.github/workflows/test.yml`

```yaml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install uv
          uv pip install -r requirements.txt
          uv pip install -r requirements-dev.txt
      
      - name: Run tests
        run: pytest --cov --cov-report=xml --cov-report=term
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true
```

### Local Continuous Testing

Use **pytest-watch** for automatic test running:

```bash
# Install
pip install pytest-watch

# Run (auto-reruns tests on file changes)
ptw

# With coverage
ptw -- --cov=.
```

---

## Best Practices

### 1. Test Isolation

**DO**:
```python
def test_function():
    """Each test is independent"""
    data = create_test_data()  # Fresh data
    result = function(data)
    assert result == expected
```

**DON'T**:
```python
shared_data = create_test_data()  # BAD: Shared state

def test_function1():
    result = function(shared_data)  # Modifies shared_data
    assert result == expected

def test_function2():
    result = another_function(shared_data)  # Depends on test1
    assert result == expected
```

### 2. Clear Test Names

**DO**:
```python
def test_asls_removes_baseline_from_spectrum():
    """Test that AsLS successfully removes baseline"""
    pass

def test_asls_raises_error_for_negative_lambda():
    """Test parameter validation"""
    pass
```

**DON'T**:
```python
def test1():  # BAD: Unclear
    pass

def test_asls():  # BAD: Too vague
    pass
```

### 3. One Assertion Per Test (When Reasonable)

**DO**:
```python
def test_output_shape():
    result = function(input)
    assert result.shape == expected_shape

def test_output_values():
    result = function(input)
    assert np.allclose(result, expected)
```

**ACCEPTABLE**:
```python
def test_output():
    """Related assertions can be grouped"""
    result = function(input)
    assert result.shape == expected_shape  # Related
    assert np.isfinite(result).all()       # Related
```

### 4. Use Fixtures for Common Setup

**DO**:
```python
@pytest.fixture
def preprocessed_spectrum():
    spectrum = np.random.randn(100)
    return apply_asls(spectrum)

def test_smoothing(preprocessed_spectrum):
    result = apply_savgol(preprocessed_spectrum)
    assert result.shape == preprocessed_spectrum.shape
```

### 5. Test Edge Cases

```python
def test_edge_cases():
    """Test boundary conditions"""
    # Empty input
    assert function(np.array([])).size == 0
    
    # Single element
    assert function(np.array([1.0])).size == 1
    
    # All zeros
    result = function(np.zeros(10))
    assert np.isfinite(result).all()
    
    # Large values
    result = function(np.ones(10) * 1e10)
    assert np.isfinite(result).all()
```

### 6. Mock External Dependencies

```python
def test_with_mock(mocker):
    """Mock external API call"""
    mock_api = mocker.patch('module.external_api_call')
    mock_api.return_value = {'status': 'success'}
    
    result = function_that_calls_api()
    
    mock_api.assert_called_once()
    assert result['status'] == 'success'
```

---

## Troubleshooting

### Common Issues

#### Issue: Tests fail randomly

**Cause**: Non-deterministic behavior (random seeds, timing)

**Solution**:
```python
# Set random seed
np.random.seed(42)
random.seed(42)

# Use pytest-randomly to detect order dependencies
pytest --randomly-seed=42
```

#### Issue: UI tests hang

**Cause**: QApplication not properly initialized

**Solution**:
```python
@pytest.fixture(scope='session')
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
```

#### Issue: Slow tests

**Solution**:
```python
# Mark slow tests
@pytest.mark.slow
def test_long_computation():
    pass

# Skip slow tests
pytest -m "not slow"

# Run tests in parallel
pytest -n auto  # Requires pytest-xdist
```

#### Issue: Coverage not including all files

**Solution**: Configure coverage to include all source files:

```toml
[tool.coverage.run]
source = ["."]
```

---

## See Also

- [Architecture Guide](architecture.md) - System design
- [Contributing Guide](contributing.md) - Development workflow
- [Build System](build-system.md) - Building and deployment
- [API Reference](../api/index.md) - Complete API documentation

---

**Last Updated**: 2026-01-26

-->
