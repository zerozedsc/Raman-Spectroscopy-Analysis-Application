# Development Guide

Welcome to the Raman Spectroscopy Analysis Application development guide. This section provides comprehensive information for developers who want to contribute to, extend, or understand the application architecture.

## 📚 Guide Contents

### [Architecture](architecture.md)
System design, component relationships, and technical architecture.

**Topics**:
- Application structure and layers
- Design patterns and principles
- Module organization
- Data flow and state management
- Plugin architecture
- Extension points

### [Contributing Guide](contributing.md)
Guidelines for contributing to the project.

**Topics**:
- Development workflow
- Git branching strategy
- Code style and conventions
- Pull request process
- Issue reporting
- Code review guidelines

### [Build System](build-system.md)
Building, packaging, and deploying the application.

**Topics**:
- Development environment setup
- Dependency management with UV
- Building executables with PyInstaller
- Creating installers (NSIS)
- Cross-platform considerations
- CI/CD pipelines

### [Testing Guide](testing.md)
Testing strategy, frameworks, and best practices.

**Topics**:
- Test structure and organization
- Unit testing with pytest
- Integration testing
- UI testing with pytest-qt
- Test fixtures and mocks
- Coverage requirements
- Continuous testing

---

## 🚀 Quick Start for Developers

### Prerequisites

- Python 3.12 (see `pyproject.toml`)
- (Optional) UV package manager (used by `dev_runner.py`)
- Git
- (Optional) PyQt6 development tools

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application.git
cd Raman-Spectroscopy-Analysis-Application

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e .[dev]
```

### Running the Application

```bash
# Development mode (auto-restart on .py changes)
python dev_runner.py

# Standard mode
python main.py

# Smoke tests (lightweight)
python smoke_tests.py
```

---

## 📂 Project Structure

```
raman-app/
├── main.py                 # Application entry point
├── dev_runner.py          # Development mode with hot-reload
├── pyproject.toml         # Project metadata and dependencies
├── README.md              # Project overview
├── CHANGELOG.md           # Version history
├── LICENSE                # License information
│
├── assets/                # Static resources
│   ├── data/             # Reference data (Raman peaks)
│   ├── fonts/            # Custom fonts
│   ├── icons/            # Application icons
│   ├── image/            # Images and graphics
│   └── locales/          # Translation files
│
├── components/            # Reusable UI components
│   ├── app_tabs.py       # Tab container
│   ├── page_registry.py  # Dynamic page registration
│   ├── toast.py          # Notification system
│   └── widgets/          # Custom widgets
│
├── configs/               # Configuration management
│   ├── __init__.py
│   ├── app_configs.json  # Default application config
│   ├── configs.py        # Configuration classes
│   ├── user_settings.py  # User preferences
│   └── style/            # Stylesheets and themes
│
├── functions/             # Core processing functions
│   ├── data_loader.py    # Data import/export
│   ├── utils.py          # Utility functions
│   ├── andorsdk/         # Andor camera SDK wrapper
│   ├── ML/               # Machine learning algorithms
│   ├── preprocess/       # Preprocessing methods
│   └── visualization/    # Plotting utilities
│
├── pages/                 # Application pages
│   ├── home_page.py
│   ├── data_package_page.py
│   ├── preprocess_page.py
│   ├── analysis_page.py
│   ├── machine_learning_page.py
│   └── workspace_page.py
│
├── build_scripts/         # Build and deployment
│   ├── build_portable.sh
│   ├── build_installer.ps1
│   ├── generate_app_icon.py
│   └── test_build_executable.py
│
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── fixtures/         # Test fixtures
│
├── docs/                  # Documentation (Sphinx)
│   ├── conf.py           # Sphinx configuration
│   ├── index.md          # Documentation home
│   ├── getting-started/
│   ├── user-guide/
│   ├── analysis-methods/
│   ├── api/
│   └── dev-guide/        # This section
│
└── logs/                  # Application logs
```

---

## 🛠️ Development Tools

### Code Quality

- **Linting**: `ruff` for Python linting
- **Formatting**: `black` for code formatting
- **Type Checking**: `mypy` for static type analysis
- **Pre-commit**: Git hooks for automatic checks

### Testing

- **pytest**: Test framework
- **pytest-qt**: Qt application testing
- **pytest-cov**: Code coverage
- **pytest-mock**: Mocking utilities

### Documentation

- **Sphinx**: Documentation generator
- **MyST Parser**: Markdown support in Sphinx
- **sphinx-rtd-theme**: ReadTheDocs theme
- **sphinx-autodoc**: Automatic API documentation

### Build Tools

- **UV**: Fast Python package manager
- **PyInstaller**: Executable builder
- **NSIS**: Windows installer creator

---

## 🔧 Common Development Tasks

### Adding a New Preprocessing Method

1. **Create method function** in `functions/preprocess/`
2. **Register in registry** (`functions/preprocess/registry.py`)
3. **Add parameter constraints** (`functions/preprocess/parameter_constraints.py`)
4. **Write unit tests** in `tests/unit/test_preprocess.py`
5. **Document method** in `docs/analysis-methods/preprocessing.md`
6. **Update changelog** in `CHANGELOG.md`

### Adding a New Page

1. **Create page class** in `pages/` inheriting from `BasePage`
2. **Register page** in page registry
3. **Add to main tabs** in `components/app_tabs.py`
4. **Create page tests** in `tests/integration/`
5. **Document page** in `docs/api/pages.md`

### Adding a New Widget

1. **Create widget class** in `components/widgets/`
2. **Implement signals and slots**
3. **Add styling support**
4. **Write widget tests**
5. **Document in API reference** (`docs/api/widgets.md`)

### Updating Translations

1. **Edit translation file** in `assets/locales/` (e.g., `en.json`, `ja.json`)
2. **Use nested keys** for organization
3. **Test in application** with language switcher
4. **Submit for review** by native speakers

---

## 📖 Coding Standards

### Python Style

Follow [PEP 8](https://pep8.org/) with these specifics:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Single quotes for strings, double for docstrings
- **Imports**: Organized (stdlib, third-party, local)
- **Naming**:
  - `snake_case` for functions, variables, modules
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

### Documentation

- **Docstrings**: Google style
- **Type hints**: Use for function signatures
- **Comments**: Explain "why", not "what"
- **TODOs**: Format as `# TODO(author): description`

### Example

```python
from typing import Optional
import numpy as np
from PyQt6.QtWidgets import QWidget


class SpectrumProcessor(QWidget):
    """
    Spectrum processing widget with interactive controls.
    
    Provides UI for applying preprocessing methods to Raman spectra
    with real-time preview and parameter adjustment.
    
    Attributes:
        spectrum (np.ndarray): Current spectrum data
        parameters (dict): Current method parameters
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize spectrum processor.
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.spectrum: Optional[np.ndarray] = None
        self.parameters: dict = {}
        self.setup_ui()
    
    def apply_preprocessing(
        self,
        method: str,
        params: dict
    ) -> np.ndarray:
        """
        Apply preprocessing method to spectrum.
        
        Args:
            method: Preprocessing method name
            params: Method-specific parameters
        
        Returns:
            Processed spectrum array
        
        Raises:
            ValueError: If method is unknown
            PreprocessingError: If processing fails
        """
        if self.spectrum is None:
            raise ValueError("No spectrum loaded")
        
        # Process spectrum
        result = self._process(method, params)
        return result
```

---

## 🤝 Getting Help

- **Documentation**: Browse this guide and API reference
- **Issues**: https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/issues
- **Discussions**: https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/discussions

---

## 📝 License

This project is licensed under the MIT License. See
[LICENSE](https://github.com/zerozedsc/Raman-Spectroscopy-Analysis-Application/blob/main/LICENSE) for details.

---

## 🔗 See Also

- [User Guide](../user-guide/index.md) - For end users
- [API Reference](../api/index.md) - Complete API documentation
- [Analysis Methods](../analysis-methods/index.md) - Method documentation
- [FAQ](../faq.md) - Frequently asked questions

---

**Last Updated**: 2026-01-24
