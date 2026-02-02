# Core Application API

Reference documentation for core application modules and utilities.

## Table of Contents
- [Application Entry Points](#application-entry-points)
- [Configuration System](#configuration-system)
- [Localization](#localization)
- [Utilities](#utilities)
- [Splash Screen](#splash-screen)

---

## Application Entry Points

### main.py

Main application entry point with GUI initialization.

#### MainApplication Class

```python
class MainApplication(QMainWindow):
    """
    Main application window managing the entire GUI.
    
    Attributes:
        config (dict): Application configuration
        locale (str): Current language locale
        project_dir (str): Current project directory
        data (dict): Loaded spectral data
    """
```

**Key Methods**:

##### `__init__(self)`

Initialize the main application window.

```python
app = MainApplication()
app.show()
```

**Responsibilities**:
- Load application configuration
- Initialize UI components
- Set up menu bar and toolbar
- Create tab system for pages
- Connect signals and slots

##### `load_config(self)`

Load application configuration from JSON.

```python
self.load_config()
```

**Returns**: None

**Side Effects**:
- Loads `configs/app_configs.json`
- Sets default values if file missing
- Updates `self.config` attribute

##### `setup_ui(self)`

Set up the main user interface.

```python
self.setup_ui()
```

**Creates**:
- Menu bar (File, Edit, View, Tools, Help)
- Toolbar with common actions
- Central widget with tab system
- Status bar with indicators

##### `change_language(self, locale: str)`

Change application language.

```python
# Change to Japanese
app.change_language('ja')

# Change to English
app.change_language('en')
```

**Parameters**:
- `locale` (str): Language code ('en', 'ja', 'ms')

**Side Effects**:
- Reloads UI with new translations
- Updates all visible text
- Saves preference to config

##### `open_project(self, directory: str = None)`

Open a project directory.

```python
# With dialog
app.open_project()

# Directly
app.open_project('/path/to/project')
```

**Parameters**:
- `directory` (str, optional): Project path. If None, shows file dialog.

**Returns**: bool - Success status

**Side Effects**:
- Loads project data
- Updates workspace page
- Sets window title

##### `save_project(self)`

Save current project state.

```python
app.save_project()
```

**Returns**: bool - Success status

**Saves**:
- Preprocessing pipelines
- Analysis results
- ML models
- Project settings

---

### dev_runner.py

Development mode runner with hot-reload.

#### DevRunner Class

```python
class DevRunner:
    """
    Development runner with auto-reload on file changes.
    
    Useful for rapid development without restarting application.
    """
```

**Usage**:

```bash
# Run in development mode
python dev_runner.py

# With specific file watching
python dev_runner.py --watch pages/ components/
```

**Features**:
- Auto-reload on .py file changes
- Preserves application state
- Hot-swaps modified modules
- Console logging for debugging

**Key Methods**:

##### `watch_files(self, paths: List[str])`

Watch specified paths for changes.

```python
runner = DevRunner()
runner.watch_files(['pages/', 'components/', 'functions/'])
runner.start()
```

##### `reload_module(self, module_name: str)`

Hot-reload a Python module.

```python
runner.reload_module('pages.exploratory_analysis_page')
```

**Parameters**:
- `module_name` (str): Full module path

**Returns**: bool - Success status

---

## Configuration System

### configs/configs.py

Application configuration management.

#### AppConfig Class

```python
from configs.configs import AppConfig

config = AppConfig()
```

**Attributes**:

```python
config.WINDOW_TITLE       # str: Application title
config.WINDOW_WIDTH       # int: Default window width
config.WINDOW_HEIGHT      # int: Default window height
config.DEFAULT_LOCALE     # str: Default language
config.THEME              # str: UI theme name
config.DATA_DIR           # str: Data directory path
config.PROJECTS_DIR       # str: Projects directory path
config.MODELS_DIR         # str: Saved models directory
```

**Methods**:

##### `load(filename: str = 'app_configs.json')`

Load configuration from file.

```python
config = AppConfig.load('custom_config.json')
```

**Parameters**:
- `filename` (str): Configuration file path

**Returns**: AppConfig instance

##### `save(filename: str = 'app_configs.json')`

Save configuration to file.

```python
config.THEME = 'dark'
config.save()
```

##### `reset_to_defaults()`

Reset all settings to default values.

```python
config.reset_to_defaults()
config.save()
```

**Resets**:
- Window dimensions
- UI theme
- Language
- Directory paths

##### `get(key: str, default=None)`

Get configuration value with fallback.

```python
theme = config.get('THEME', 'light')
max_threads = config.get('MAX_THREADS', 4)
```

**Parameters**:
- `key` (str): Configuration key
- `default` (any): Fallback value if key missing

**Returns**: Configuration value or default

---

### configs/user_settings.py

User-specific settings and preferences.

#### UserSettings Class

```python
from configs.user_settings import UserSettings

settings = UserSettings()
```

**Persistent Settings**:

```python
# Recent files
settings.recent_projects        # List[str]
settings.recent_files           # List[str]

# UI preferences
settings.window_geometry        # QByteArray: Window size/position
settings.splitter_state         # QByteArray: Splitter positions
settings.last_directory         # str: Last used directory

# Analysis preferences
settings.default_preprocessing  # dict: Default pipeline
settings.default_ml_params      # dict: Default ML parameters
settings.favorite_methods       # List[str]: Bookmarked methods
```

**Methods**:

##### `load()`

Load user settings from persistent storage.

```python
settings = UserSettings()
settings.load()
```

**Storage Location**:
- Windows: `%APPDATA%/RamanAnalysis/settings.json`
- macOS: `~/Library/Application Support/RamanAnalysis/settings.json`
- Linux: `~/.config/RamanAnalysis/settings.json`

##### `save()`

Save current settings to persistent storage.

```python
settings.window_geometry = mainwindow.saveGeometry()
settings.save()
```

##### `add_recent_file(filepath: str, max_count: int = 10)`

Add file to recent files list.

```python
settings.add_recent_file('/path/to/project.json')
settings.save()
```

**Parameters**:
- `filepath` (str): File path to add
- `max_count` (int): Maximum recent files to keep

##### `clear_recent()`

Clear recent files history.

```python
settings.clear_recent()
settings.save()
```

---

## Localization

### Locale System

Multi-language support using JSON translation files.

#### Translation Files

Located in `assets/locales/`:
- `en.json` - English (default)
- `ja.json` - Japanese
- `ms.json` - Malay

**Structure**:

```json
{
  "menu": {
    "file": "File",
    "edit": "Edit",
    "view": "View"
  },
  "toolbar": {
    "open": "Open",
    "save": "Save",
    "undo": "Undo"
  },
  "pages": {
    "home": "Home",
    "preprocessing": "Preprocessing",
    "analysis": "Analysis"
  },
  "messages": {
    "success": "Operation completed successfully",
    "error": "An error occurred"
  }
}
```

#### LocaleManager Class

```python
from utils import LocaleManager

locale_manager = LocaleManager()
```

**Methods**:

##### `load_locale(locale_code: str)`

Load translation file.

```python
locale_manager.load_locale('ja')
```

**Parameters**:
- `locale_code` (str): Language code ('en', 'ja', 'ms')

**Returns**: dict - Translation dictionary

##### `translate(key: str, default: str = None)`

Get translated string.

```python
# Simple translation
text = locale_manager.translate('menu.file')  # "File"

# With fallback
text = locale_manager.translate('unknown.key', 'Default Text')

# Nested keys
text = locale_manager.translate('pages.preprocessing.title')
```

**Parameters**:
- `key` (str): Dot-separated translation key
- `default` (str, optional): Fallback text if key not found

**Returns**: str - Translated text

##### `translate_format(key: str, **kwargs)`

Get translated string with formatting.

```python
# Translation: "Loaded {count} spectra from {filename}"
text = locale_manager.translate_format(
    'messages.loaded_spectra',
    count=150,
    filename='data.csv'
)
# Output: "Loaded 150 spectra from data.csv"
```

**Parameters**:
- `key` (str): Translation key
- `**kwargs`: Format arguments

**Returns**: str - Formatted translated text

##### `get_available_locales()`

Get list of available language codes.

```python
locales = locale_manager.get_available_locales()
# Returns: ['en', 'ja', 'ms']
```

**Returns**: List[str] - Available locale codes

---

## Utilities

### utils.py

Common utility functions used across the application.

#### File I/O Functions

##### `load_spectra_from_csv(filepath: str)`

Load Raman spectra from CSV file.

```python
from utils import load_spectra_from_csv

data = load_spectra_from_csv('data/sample.csv')

# Returns:
# {
#     'wavenumbers': np.array([...]),
#     'spectra': np.array([[...], [...], ...]),
#     'labels': ['Sample1', 'Sample2', ...],
#     'groups': [...],
#     'metadata': {...}
# }
```

**Parameters**:
- `filepath` (str): Path to CSV file

**Returns**: dict with keys:
- `wavenumbers` (np.ndarray): Wavenumber axis
- `spectra` (np.ndarray): Spectral intensities (n_samples, n_features)
- `labels` (List[str]): Sample names
- `groups` (List[str]): Group assignments
- `metadata` (dict): Additional metadata

**CSV Format**:
```text
Wavenumber,Sample1,Sample2,Sample3,...
400,1234.5,2345.6,3456.7,...
401,1235.0,2346.1,3457.2,...
...
```

**Raises**:
- `FileNotFoundError`: File doesn't exist
- `ValueError`: Invalid CSV format

##### `save_spectra_to_csv(data: dict, filepath: str)`

Save spectra to CSV file.

```python
from utils import save_spectra_to_csv

save_spectra_to_csv(data, 'output/processed.csv')
```

**Parameters**:
- `data` (dict): Data dictionary with 'wavenumbers' and 'spectra'
- `filepath` (str): Output file path

##### `export_results_to_excel(results: dict, filepath: str)`

Export analysis results to Excel file.

```python
from utils import export_results_to_excel

results = {
    'PCA Scores': pca_scores_df,
    'Feature Importance': importance_df,
    'Classification Report': report_df
}

export_results_to_excel(results, 'results.xlsx')
```

**Parameters**:
- `results` (dict): Dictionary of DataFrames (sheet_name: DataFrame)
- `filepath` (str): Output Excel file path

**Creates**: Multi-sheet Excel file with formatted tables

#### Data Validation

##### `validate_spectra_data(data: dict)`

Validate spectral data structure.

```python
from utils import validate_spectra_data

try:
    validate_spectra_data(data)
    print("Data is valid")
except ValueError as e:
    print(f"Invalid data: {e}")
```

**Parameters**:
- `data` (dict): Data dictionary to validate

**Checks**:
- Required keys present ('wavenumbers', 'spectra')
- Correct data types (numpy arrays)
- Matching dimensions
- No NaN or infinite values

**Raises**: ValueError with descriptive message

##### `validate_preprocessing_params(method: str, params: dict)`

Validate preprocessing method parameters.

```python
from utils import validate_preprocessing_params

# Valid parameters
params = {'window_length': 11, 'polyorder': 3}
validate_preprocessing_params('savgol', params)  # OK

# Invalid parameters
params = {'window_length': 4, 'polyorder': 3}
validate_preprocessing_params('savgol', params)  # Raises ValueError
```

**Parameters**:
- `method` (str): Preprocessing method name
- `params` (dict): Parameters to validate

**Raises**: ValueError if parameters invalid

#### Progress Tracking

##### `ProgressReporter`

Context manager for progress reporting.

```python
from utils import ProgressReporter

with ProgressReporter("Processing spectra", total=1000) as progress:
    for i, spectrum in enumerate(spectra):
        # Process spectrum
        result = preprocess(spectrum)
        
        # Update progress
        progress.update(i + 1)
```

**Parameters**:
- `description` (str): Progress description
- `total` (int): Total number of items

**Methods**:
- `update(n: int)`: Update progress to n
- `increment()`: Increment progress by 1
- `set_description(desc: str)`: Change description

---

## Splash Screen

### splash_screen.py

Animated splash screen for application startup.

#### SplashScreen Class

```python
from splash_screen import SplashScreen

splash = SplashScreen()
splash.show()
splash.showMessage("Loading modules...")
```

**Features**:
- Animated logo
- Progress messages
- Smooth fade-in/fade-out
- Minimum display time (prevents flicker)

**Methods**:

##### `showMessage(message: str, alignment=Qt.AlignBottom, color=Qt.white)`

Display status message on splash screen.

```python
splash.showMessage("Loading configuration...")
time.sleep(0.5)

splash.showMessage("Initializing UI...")
time.sleep(0.5)

splash.showMessage("Ready!")
```

**Parameters**:
- `message` (str): Message to display
- `alignment` (Qt.Alignment): Text alignment
- `color` (QColor): Text color

##### `finish(widget: QWidget)`

Close splash screen and show main widget.

```python
splash.finish(main_window)
```

**Parameters**:
- `widget` (QWidget): Main window to activate

**Effect**: Fades out splash, shows main window

---

## Error Handling

### Custom Exceptions

#### DataError

Raised for data-related errors.

```python
from utils import DataError

if data['spectra'].shape[0] == 0:
    raise DataError("No spectra found in dataset")
```

#### PreprocessingError

Raised for preprocessing failures.

```python
from utils import PreprocessingError

try:
    result = apply_baseline_correction(spectrum, lambda_param=-1)
except ValueError:
    raise PreprocessingError("Invalid lambda parameter")
```

#### ModelError

Raised for ML model errors.

```python
from utils import ModelError

if not model.is_fitted:
    raise ModelError("Model must be trained before prediction")
```

### Error Logging

```python
from utils import setup_logger

# Create logger
logger = setup_logger('mymodule')

# Log messages
logger.debug("Debug information")
logger.info("Processing started")
logger.warning("Unusual value detected")
logger.error("Operation failed")
logger.critical("Critical error occurred")
```

**Log File Location**: `logs/application.log`

**Log Format**:
```
2026-01-24 10:30:45 - mymodule - INFO - Processing started
2026-01-24 10:30:46 - mymodule - WARNING - Unusual value detected
```

---

## Threading and Async Operations

### WorkerThread Class

Background processing without freezing UI.

```python
from utils import WorkerThread

def long_running_task(data, param1, param2):
    # Process data
    result = expensive_computation(data, param1, param2)
    return result

# Create worker
worker = WorkerThread(
    target=long_running_task,
    args=(data,),
    kwargs={'param1': value1, 'param2': value2}
)

# Connect signals
worker.signals.finished.connect(on_finished)
worker.signals.error.connect(on_error)
worker.signals.progress.connect(on_progress)

# Start processing
worker.start()
```

**Signals**:
- `finished(result)`: Task completed successfully
- `error(exc_type, exc_value, exc_traceback)`: Error occurred
- `progress(value, message)`: Progress update

**Example with Progress**:

```python
def process_with_progress(spectra):
    for i, spectrum in enumerate(spectra):
        # Process
        result = preprocess(spectrum)
        
        # Emit progress
        progress = int((i + 1) / len(spectra) * 100)
        WorkerThread.current().signals.progress.emit(
            progress,
            f"Processing spectrum {i+1}/{len(spectra)}"
        )
    
    return results

worker = WorkerThread(target=process_with_progress, args=(spectra,))
worker.signals.progress.connect(lambda p, msg: print(f"{p}% - {msg}"))
worker.start()
```

---

## Performance Utilities

### Caching

```python
from utils import cache_result

@cache_result(maxsize=128)
def expensive_computation(data, param):
    # Expensive operation
    result = complex_processing(data, param)
    return result

# First call: computed
result1 = expensive_computation(data, param=10)

# Second call with same args: cached
result2 = expensive_computation(data, param=10)  # Fast!
```

### Profiling

```python
from utils import profile_function

@profile_function
def my_function(data):
    # Function code
    return result

# Automatically logs execution time
result = my_function(data)
# Output: "my_function took 2.345 seconds"
```

---

## Testing Utilities

### Mock Data Generation

```python
from utils import generate_mock_spectra

# Generate synthetic Raman spectra
mock_data = generate_mock_spectra(
    n_samples=100,
    n_features=1000,
    n_groups=3,
    noise_level=0.1,
    random_state=42
)

# Returns same structure as real data
assert 'wavenumbers' in mock_data
assert 'spectra' in mock_data
assert mock_data['spectra'].shape == (100, 1000)
```

**Parameters**:
- `n_samples` (int): Number of spectra
- `n_features` (int): Number of wavenumbers
- `n_groups` (int): Number of groups
- `noise_level` (float): Gaussian noise standard deviation
- `random_state` (int): Random seed

---

## Best Practices

### Configuration Management

**DO**:
```python
# Use centralized configuration
from configs.configs import AppConfig
config = AppConfig()
max_threads = config.MAX_THREADS

# Save user preferences
from configs.user_settings import UserSettings
settings = UserSettings()
settings.last_directory = path
settings.save()
```

**DON'T**:
```python
# Hard-coded values
max_threads = 4  # Bad

# Scattered configuration
self.max_threads = 4
self.data_dir = "data"
```

### Error Handling

**DO**:
```python
from utils import DataError, logger

try:
    data = load_spectra(filepath)
    validate_spectra_data(data)
except FileNotFoundError:
    logger.error(f"File not found: {filepath}")
    raise
except DataError as e:
    logger.error(f"Invalid data: {e}")
    show_error_dialog(str(e))
```

**DON'T**:
```python
# Silent failures
try:
    data = load_spectra(filepath)
except:
    pass  # Bad!

# Generic exceptions
except Exception as e:
    print(e)  # Not informative
```

### Threading

**DO**:
```python
# Use WorkerThread for long operations
worker = WorkerThread(target=expensive_task, args=(data,))
worker.signals.finished.connect(self.on_complete)
worker.start()
```

**DON'T**:
```python
# Blocking UI thread
result = expensive_task(data)  # Freezes UI!
```

---

## See Also

- [Pages API](pages.md) - Page modules
- [Components API](components.md) - UI components
- [Functions API](functions.md) - Processing functions
- [Development Guide](../dev-guide/architecture.md) - Architecture overview

---

**Last Updated**: 2026-01-24
