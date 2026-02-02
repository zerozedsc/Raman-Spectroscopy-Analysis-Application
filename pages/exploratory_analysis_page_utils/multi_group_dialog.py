"""pages.exploratory_analysis_page_utils.multi_group_dialog

Backwards-compatible shim.

The dialog was moved to `components.widgets.multi_group_dialog` so it can be reused
across multiple pages (including Machine Learning page).
"""

from components.widgets.multi_group_dialog import MultiGroupCreationDialog

__all__ = ["MultiGroupCreationDialog"]

# The legacy implementation was removed from this location after being moved to
# `components.widgets.multi_group_dialog`. Keep this file as a tiny shim so any
# existing imports from `pages.analysis_page_utils.multi_group_dialog` continue
# to work.
