# Architecture

Comprehensive overview of the application architecture, design patterns, and system organization.

## Table of Contents
- [System Overview](#system-overview)
- [Architectural Layers](#architectural-layers)
- [Design Patterns](#design-patterns)
- [Module Organization](#module-organization)
- [Data Flow](#data-flow)
- [State Management](#state-management)
- [Plugin Architecture](#plugin-architecture)
- [Extension Points](#extension-points)
- [Performance Considerations](#performance-considerations)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Presentation Layer                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   Home   │  │   Data   │  │Preprocess│  │ Analysis │   │
│  │   Page   │  │ Package  │  │   Page   │  │   Page   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│         ▲              ▲              ▲              ▲        │
├─────────┼──────────────┼──────────────┼──────────────┼───────┤
│         │              │              │              │        │
│  ┌──────┴──────────────┴──────────────┴──────────────┴────┐ │
│  │              Component Layer                             │ │
│  │  - AppTabs         - ResultsPanel   - Toast            │ │
│  │  - SpectrumViewer  - DataTable      - Dialogs          │ │
│  │  - PipelineBuilder - ParameterPanel - Widgets          │ │
│  └──────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                     Business Logic Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Preprocessing│  │   Analysis   │  │   Machine    │     │
│  │   Functions  │  │   Functions  │  │   Learning   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
├─────────────────────────────────────────────────────────────┤
│                      Data Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Data Loader  │  │ Config Mgmt  │  │ Model Storage│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **UI Framework**: PyQt6
- **Scientific Computing**: NumPy, SciPy
- **Machine Learning**: scikit-learn, XGBoost
- **Visualization**: Matplotlib
- **Data Handling**: Pandas
- **Configuration**: JSON
- **Localization**: JSON-based i18n
- **Build**: PyInstaller, UV

---

## Architectural Layers

### 1. Presentation Layer

**Responsibility**: User interface and user interaction

**Components**:
- **Pages**: Full-screen application views (`pages/`)
- **Widgets**: Reusable UI components (`components/widgets/`)
- **Dialogs**: Modal windows for specific tasks
- **Styling**: Theme management (`configs/style/`)

**Key Principles**:
- Separation of UI and business logic
- Signal-slot communication pattern
- Reactive UI updates
- Consistent styling across components

**Example**:

```python
class PreprocessPage(BasePage):
    """Preprocessing page - Presentation layer"""
    
    # Signals for communication
    data_processed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.connect_signals()
    
    def on_apply_clicked(self):
        """UI event handler"""
        # Get UI state
        pipeline = self.pipeline_builder.get_pipeline()
        
        # Call business logic (next layer)
        result = execute_preprocessing_pipeline(
            self.data['spectra'],
            pipeline
        )
        
        # Emit result
        self.data_processed.emit(result)
```

### 2. Component Layer

**Responsibility**: Reusable UI building blocks

**Components**:
- **App-level**: `AppTabs`, `PageRegistry`, `Toast`
- **Data Display**: `SpectrumViewer`, `DataTable`, `ResultsPanel`
- **Input**: `ParameterPanel`, `PipelineBuilder`
- **Custom Widgets**: Parameter inputs, plot widgets, dialogs

**Design Pattern**: Composite pattern

**Example**:

```python
class PipelineBuilder(QWidget):
    """Composite component for pipeline construction"""
    
    pipeline_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.steps = []  # Composite: list of step widgets
        self.setup_ui()
    
    def add_step(self, method, params):
        """Add child component"""
        step = PipelineStepWidget(method, params)
        self.steps.append(step)
        step.changed.connect(self.pipeline_changed)
        self.layout.addWidget(step)
    
    def get_pipeline(self):
        """Get state from all children"""
        return [step.get_config() for step in self.steps]
```

### 3. Business Logic Layer

**Responsibility**: Core application logic and algorithms

**Modules**:
- **Preprocessing**: `functions/preprocess/`
- **Analysis**: `functions/ML/`, statistical functions
- **Utilities**: `functions/utils.py`, `functions/data_loader.py`

**Key Principles**:
- Pure functions where possible
- Testable without UI
- Framework-agnostic
- Well-documented algorithms

**Example**:

```python
def apply_asls(
    spectrum: np.ndarray,
    lambda_: float = 1e5,
    p: float = 0.01,
    max_iter: int = 10
) -> np.ndarray:
    """
    Apply AsLS baseline correction (Business logic layer).
    
    Pure function - no UI dependencies, fully testable.
    
    Args:
        spectrum: Input spectrum
        lambda_: Smoothness parameter
        p: Asymmetry parameter
        max_iter: Maximum iterations
    
    Returns:
        Baseline-corrected spectrum
    """
    # Algorithm implementation
    # No PyQt dependencies
    # No side effects
    return corrected_spectrum
```

### 4. Data Layer

**Responsibility**: Data persistence and I/O

**Components**:
- **Data Loader**: CSV/TXT import/export
- **Configuration**: App config, user settings
- **Model Storage**: ML model serialization
- **Project Management**: Project file structure

**Example**:

```python
class UserSettings:
    """Data layer for user preferences"""
    
    def __init__(self):
        self.settings_file = self._get_settings_path()
        self.settings = self.load()
    
    def load(self) -> dict:
        """Load from disk"""
        if os.path.exists(self.settings_file):
            with open(self.settings_file) as f:
                return json.load(f)
        return self._defaults()
    
    def save(self):
        """Persist to disk"""
        os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=2)
```

---

## Design Patterns

### 1. Model-View Pattern (MVP Variant)

**Structure**:
- **Model**: Data and business logic (`functions/`)
- **View**: UI components (`pages/`, `components/`)
- **Presenter**: Implicit in page controllers

**Benefits**:
- Testable business logic
- Reusable UI components
- Clear separation of concerns

**Example**:

```python
# Model (Business Logic)
class PreprocessingModel:
    def apply_pipeline(self, data, pipeline):
        """Pure business logic"""
        return execute_preprocessing_pipeline(data, pipeline)

# View (UI)
class PreprocessView(QWidget):
    apply_requested = pyqtSignal(list)  # pipeline
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.apply_button.clicked.connect(
            lambda: self.apply_requested.emit(self.get_pipeline())
        )

# Controller/Presenter (Page)
class PreprocessPage(BasePage):
    def __init__(self):
        super().__init__()
        self.model = PreprocessingModel()
        self.view = PreprocessView()
        self.view.apply_requested.connect(self.on_apply)
    
    def on_apply(self, pipeline):
        """Coordinate between model and view"""
        result = self.model.apply_pipeline(self.data, pipeline)
        self.view.show_result(result)
```

### 2. Observer Pattern (Signals & Slots)

**Implementation**: Qt's signal-slot mechanism

**Usage**:
- Component communication
- Event propagation
- Loose coupling

**Example**:

```python
class DataPackagePage(BasePage):
    # Observable: defines signals
    data_loaded = pyqtSignal(dict)
    groups_changed = pyqtSignal(dict)

class PreprocessPage(BasePage):
    # Observer: connects to signals
    def __init__(self):
        super().__init__()
        data_page.data_loaded.connect(self.load_data)
        data_page.groups_changed.connect(self.update_groups)
    
    def load_data(self, data):
        """React to data_loaded signal"""
        self.data = data
        self.update_ui()
```

### 3. Registry Pattern

**Implementation**: `PageRegistry`, Method Registry

**Purpose**:
- Dynamic registration
- Plugin support
- Loose coupling

**Example**:

```python
# Registry
class MethodRegistry:
    _methods = {}
    
    @classmethod
    def register(cls, name, func, params_spec):
        cls._methods[name] = {
            'function': func,
            'params': params_spec
        }
    
    @classmethod
    def get(cls, name):
        return cls._methods.get(name)

# Registration
MethodRegistry.register(
    'asls',
    apply_asls,
    {
        'lambda': {'type': 'float', 'min': 1e2, 'max': 1e9},
        'p': {'type': 'float', 'min': 0.001, 'max': 0.1}
    }
)

# Usage
method_info = MethodRegistry.get('asls')
result = method_info['function'](spectrum, lambda_=1e5, p=0.01)
```

### 4. Factory Pattern

**Implementation**: Widget factories, Method factories

**Example**:

```python
class ParameterWidgetFactory:
    """Create appropriate widget based on parameter type"""
    
    @staticmethod
    def create(param_spec):
        param_type = param_spec['type']
        
        if param_type == 'float':
            return FloatParameterWidget(
                min_value=param_spec.get('min'),
                max_value=param_spec.get('max'),
                default_value=param_spec.get('default')
            )
        elif param_type == 'int':
            return IntParameterWidget(...)
        elif param_type == 'choice':
            return ChoiceParameterWidget(
                choices=param_spec.get('choices')
            )
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

# Usage
widget = ParameterWidgetFactory.create({
    'type': 'float',
    'min': 0.001,
    'max': 0.1,
    'default': 0.01
})
```

### 5. Strategy Pattern

**Implementation**: Preprocessing methods, ML algorithms

**Example**:

```python
class BaselineCorrection:
    """Strategy interface"""
    
    def correct(self, spectrum: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class AsLSStrategy(BaselineCorrection):
    """Concrete strategy"""
    
    def __init__(self, lambda_=1e5, p=0.01):
        self.lambda_ = lambda_
        self.p = p
    
    def correct(self, spectrum):
        return apply_asls(spectrum, self.lambda_, self.p)

class AirPLSStrategy(BaselineCorrection):
    """Another concrete strategy"""
    
    def __init__(self, lambda_=100):
        self.lambda_ = lambda_
    
    def correct(self, spectrum):
        return apply_airpls(spectrum, self.lambda_)

# Context
class SpectrumProcessor:
    def __init__(self, strategy: BaselineCorrection):
        self.strategy = strategy
    
    def process(self, spectrum):
        return self.strategy.correct(spectrum)

# Usage
processor = SpectrumProcessor(AsLSStrategy(lambda_=1e5))
result = processor.process(spectrum)

# Switch strategy
processor.strategy = AirPLSStrategy(lambda_=100)
result = processor.process(spectrum)
```

### 6. Command Pattern

**Implementation**: Undo/Redo functionality

**Example**:

```python
class Command:
    """Command interface"""
    
    def execute(self):
        raise NotImplementedError
    
    def undo(self):
        raise NotImplementedError

class AddPreprocessingStepCommand(Command):
    """Concrete command"""
    
    def __init__(self, pipeline_builder, method, params):
        self.builder = pipeline_builder
        self.method = method
        self.params = params
        self.step_id = None
    
    def execute(self):
        self.step_id = self.builder.add_step(self.method, self.params)
    
    def undo(self):
        self.builder.remove_step(self.step_id)

class CommandManager:
    """Invoker"""
    
    def __init__(self):
        self.history = []
        self.current = -1
    
    def execute(self, command):
        command.execute()
        self.history = self.history[:self.current + 1]
        self.history.append(command)
        self.current += 1
    
    def undo(self):
        if self.current >= 0:
            self.history[self.current].undo()
            self.current -= 1
    
    def redo(self):
        if self.current < len(self.history) - 1:
            self.current += 1
            self.history[self.current].execute()
```

---

## Module Organization

### Core Modules

#### 1. `main.py`

**Purpose**: Application entry point

**Responsibilities**:
- Initialize Qt application
- Load configuration
- Create main window
- Set up exception handling
- Start event loop

```python
def main():
    app = QApplication(sys.argv)
    
    # Load config
    config = AppConfig.load()
    
    # Apply theme
    apply_theme(config.THEME, app)
    
    # Create main window
    window = MainApplication(config)
    window.show()
    
    # Run
    sys.exit(app.exec())
```

#### 2. `components/`

**Purpose**: Reusable UI components

**Structure**:
```
components/
├── app_tabs.py          # Tab container
├── page_registry.py     # Page registration
├── toast.py             # Notifications
└── widgets/             # Custom widgets
    ├── __init__.py
    ├── matplotlib_widget.py
    ├── parameter_widgets.py
    ├── results_panel.py
    └── ...
```

#### 3. `pages/`

**Purpose**: Application pages/views

**Structure**:
```
pages/
├── home_page.py
├── data_package_page.py
├── preprocess_page.py
├── analysis_page.py
├── machine_learning_page.py
├── workspace_page.py
└── <page>_utils/        # Page-specific utilities
```

#### 4. `functions/`

**Purpose**: Business logic and algorithms

**Structure**:
```
functions/
├── data_loader.py       # I/O functions
├── utils.py             # General utilities
├── preprocess/          # Preprocessing methods
│   ├── baseline.py
│   ├── normalization.py
│   ├── derivatives.py
│   ├── pipeline.py
│   └── registry.py
├── ML/                  # Machine learning
│   ├── svm.py
│   ├── random_forest.py
│   ├── xgboost.py
│   └── ...
└── visualization/       # Plotting utilities
```

#### 5. `configs/`

**Purpose**: Configuration management

**Structure**:
```
configs/
├── __init__.py
├── app_configs.json     # Default config
├── configs.py           # AppConfig class
├── user_settings.py     # UserSettings class
└── style/               # Themes and stylesheets
    └── stylesheets.py
```

---

## Data Flow

### 1. Data Loading Flow

```
User Action (Open File)
        ↓
DataPackagePage.import_files()
        ↓
functions/data_loader.load_spectra_from_csv()
        ↓
Validate data (validate_spectra_data)
        ↓
Store in page state (self.data)
        ↓
Emit signal (data_loaded)
        ↓
Other pages receive data
```

### 2. Preprocessing Flow

```
User configures pipeline
        ↓
PipelineBuilder stores steps
        ↓
User clicks "Apply"
        ↓
PreprocessPage.apply_pipeline()
        ↓
execute_preprocessing_pipeline()
        ↓
For each step:
    - Get method from registry
    - Apply to all spectra
    - Update progress
        ↓
Return processed data
        ↓
Update preview
        ↓
Emit data_processed signal
```

### 3. Analysis Flow

```
Preprocessed data loaded
        ↓
User selects analysis method
        ↓
User configures parameters
        ↓
User clicks "Run"
        ↓
AnalysisPage.apply_<method>()
        ↓
Call function (e.g., apply_pca)
        ↓
Display results in ResultsPanel
        ↓
User can export results
```

### 4. Machine Learning Flow

```
Preprocessed data + labels loaded
        ↓
User selects algorithm
        ↓
User configures hyperparameters
        ↓
User clicks "Train"
        ↓
MLPage.train_model()
        ↓
Split data (train/test)
        ↓
Train model with cross-validation
        ↓
Evaluate on test set
        ↓
Display results (confusion matrix, metrics)
        ↓
User can save model
```

---

## State Management

### Application State

**Centralized in**: `MainApplication` class

**Components**:
- Current project directory
- Loaded data
- Active page
- Configuration
- User settings

**Example**:

```python
class MainApplication(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.project_dir = None
        self.data = None
        self.current_page = None
```

### Page State

**Managed by**: Individual page classes

**Components**:
- Page-specific data
- UI state (selections, parameters)
- Processing results

**Persistence**:
- Save on project save
- Restore on project load

**Example**:

```python
class PreprocessPage(BasePage):
    def get_state(self):
        """Get state for persistence"""
        return {
            'pipeline': self.pipeline_builder.get_pipeline(),
            'preview_index': self.preview_index,
            'last_result': self.last_result
        }
    
    def set_state(self, state):
        """Restore state"""
        self.pipeline_builder.load_pipeline(state['pipeline'])
        self.preview_index = state['preview_index']
        self.last_result = state['last_result']
```

### Signal-Based State Propagation

**Pattern**: When state changes in one component, emit signal for others to update

```python
# State change in DataPackagePage
def create_group(self, name):
    self.groups[name] = []
    self.groups_changed.emit(self.groups)  # Notify observers

# Other pages react
class AnalysisPage(BasePage):
    def __init__(self):
        super().__init__()
        data_page.groups_changed.connect(self.update_group_selector)
    
    def update_group_selector(self, groups):
        self.group_combo.clear()
        self.group_combo.addItems(groups.keys())
```

---

## Plugin Architecture

### Extensibility Points

1. **Custom Preprocessing Methods**
2. **Custom Analysis Methods**
3. **Custom Pages**
4. **Custom Widgets**
5. **Custom Export Formats**

### Method Plugin Example

```python
# plugins/my_custom_baseline.py

from functions.preprocess.registry import MethodRegistry
import numpy as np

def my_custom_baseline(
    spectrum: np.ndarray,
    smoothness: float = 0.5
) -> np.ndarray:
    """
    Custom baseline correction method.
    
    Args:
        spectrum: Input spectrum
        smoothness: Custom smoothness parameter (0-1)
    
    Returns:
        Baseline-corrected spectrum
    """
    # Your algorithm here
    corrected = spectrum - calculate_baseline(spectrum, smoothness)
    return corrected

# Register method
MethodRegistry.register(
    name='my_custom_baseline',
    function=my_custom_baseline,
    category='Baseline Correction',
    params_spec={
        'smoothness': {
            'type': 'float',
            'min': 0.0,
            'max': 1.0,
            'default': 0.5,
            'label': 'Smoothness',
            'tooltip': 'Controls baseline smoothness'
        }
    },
    description='My custom baseline correction algorithm'
)
```

### Page Plugin Example

```python
# plugins/my_custom_page.py

from pages.base_page import BasePage
from components.page_registry import PageRegistry

class MyCustomPage(BasePage):
    """Custom analysis page"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        # Build custom UI
        pass

# Register page
PageRegistry.register_page(
    page_id='my_custom_analysis',
    page_class=MyCustomPage,
    metadata={
        'title': 'Custom Analysis',
        'category': 'Analysis',
        'icon': 'icons/custom.png',
        'order': 100
    }
)
```

---

## Extension Points

### 1. Custom Export Formats

**Interface**:

```python
class ExportFormat:
    """Base class for export formats"""
    
    @property
    def name(self) -> str:
        """Format name"""
        raise NotImplementedError
    
    @property
    def extensions(self) -> List[str]:
        """File extensions"""
        raise NotImplementedError
    
    def export(self, data: dict, filepath: str):
        """Export data to file"""
        raise NotImplementedError

# Custom implementation
class JSONExporter(ExportFormat):
    @property
    def name(self):
        return "JSON"
    
    @property
    def extensions(self):
        return ['.json']
    
    def export(self, data, filepath):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

# Register
export_registry.register(JSONExporter())
```

### 2. Custom Visualizations

**Interface**:

```python
class VisualizationPlugin:
    """Base class for visualization plugins"""
    
    def plot(self, data: dict, ax: matplotlib.axes.Axes):
        """Create plot on given axes"""
        raise NotImplementedError
    
    def get_config_widget(self) -> QWidget:
        """Return configuration widget"""
        raise NotImplementedError

# Example
class WaterfallPlot(VisualizationPlugin):
    def plot(self, data, ax):
        spectra = data['spectra']
        for i, spectrum in enumerate(spectra):
            ax.plot(data['wavenumbers'], spectrum + i * 0.1)
    
    def get_config_widget(self):
        widget = QWidget()
        # Build config UI
        return widget
```

---

## Performance Considerations

### 1. Lazy Loading

**Strategy**: Load data only when needed

```python
class DataTable(QTableWidget):
    def __init__(self, data):
        super().__init__()
        self.full_data = data
        self.load_visible_rows()
    
    def scrolled(self):
        """Load more rows as user scrolls"""
        visible_range = self.get_visible_range()
        self.load_rows(visible_range)
```

### 2. Batch Updates

**Strategy**: Disable repainting during bulk operations

```python
def add_many_items(self, items):
    self.setUpdatesEnabled(False)  # Disable repainting
    try:
        for item in items:
            self.add_item(item)
    finally:
        self.setUpdatesEnabled(True)  # Re-enable and repaint once
```

### 3. Background Processing

**Strategy**: Use threads for long operations

```python
class WorkerThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(Exception)
    progress = pyqtSignal(int, str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(e)

# Usage
def long_operation():
    # Do work
    pass

thread = WorkerThread(long_operation)
thread.finished.connect(self.on_finished)
thread.start()
```

### 4. Caching

**Strategy**: Cache expensive computations

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(data_hash, param1, param2):
    """Cached computation"""
    # Expensive operation
    return result

# Usage
data_hash = hash(data.tobytes())
result = expensive_computation(data_hash, param1, param2)
```

### 5. Vectorization

**Strategy**: Use NumPy vectorized operations

```python
# Bad: Loop over spectra
def process_spectra_slow(spectra):
    results = []
    for spectrum in spectra:
        result = process_single(spectrum)
        results.append(result)
    return np.array(results)

# Good: Vectorized
def process_spectra_fast(spectra):
    # Process all spectra at once
    return vectorized_process(spectra)
```

---

## Best Practices

### 1. Separation of Concerns

- Keep UI code separate from business logic
- Pure functions for algorithms
- Stateless where possible

### 2. Error Handling

```python
class PreprocessingError(Exception):
    """Custom exception for preprocessing errors"""
    pass

def safe_operation():
    try:
        result = risky_operation()
    except PreprocessingError as e:
        logger.error(f"Preprocessing failed: {e}")
        show_error_dialog(str(e))
        return None
    except Exception as e:
        logger.exception("Unexpected error")
        show_error_dialog("An unexpected error occurred")
        return None
```

### 3. Logging

```python
import logging

logger = logging.getLogger(__name__)

def important_operation():
    logger.info("Starting operation")
    try:
        result = do_work()
        logger.info("Operation completed successfully")
        return result
    except Exception as e:
        logger.exception("Operation failed")
        raise
```

### 4. Testing

- Write unit tests for business logic
- Integration tests for workflows
- UI tests for critical interactions

---

## See Also

- [Contributing Guide](contributing.md) - Development workflow
- [Build System](build-system.md) - Building and deployment
- [Testing Guide](testing.md) - Testing strategy
- [API Reference](../api/index.md) - Complete API documentation

---

**Last Updated**: 2026-01-24
