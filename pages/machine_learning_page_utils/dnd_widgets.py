"""pages.machine_learning_page_utils.dnd_widgets

Backwards-compatible shim.

These widgets are shared across multiple pages, so the implementation now lives in
`components.widgets.grouping.dnd_widgets`.
"""

from components.widgets.grouping.dnd_widgets import DatasetSourceList, GroupDropList

__all__ = ["DatasetSourceList", "GroupDropList"]
