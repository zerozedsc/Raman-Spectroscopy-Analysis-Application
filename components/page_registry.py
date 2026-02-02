"""components.page_registry

Central place to wire up application pages.

Rationale:
- Keeps `pages/*.py` from importing other pages directly.
- Lets the workspace container depend only on shared modules.

This module is intentionally small and should remain UI-only.
"""

from __future__ import annotations

from typing import Dict

from PySide6.QtWidgets import QWidget


def create_workspace_pages() -> Dict[str, QWidget]:
    """Create the top-level pages used in `WorkspacePage`.

    Returned keys are stable and used by `pages/workspace_page.py`.
    """

    # Local imports to avoid accidental import cycles during application startup.
    from pages.data_package_page import DataPackagePage
    from pages.preprocess_page import PreprocessPage
    from pages.exploratory_analysis_page import AnalysisPage
    from pages.home_page import HomePage
    from pages.modeling_classification_page import MachineLearningPage

    return {
        "home": HomePage(),
        "data": DataPackagePage(),
        "preprocess": PreprocessPage(),
        "analysis": AnalysisPage(),
        "ml": MachineLearningPage(),
    }
