from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QMessageBox, QTabBar
from PySide6.QtWidgets import (
	QCheckBox,
	QHBoxLayout,
	QLineEdit,
	QPushButton,
	QTabWidget,
	QVBoxLayout,
	QWidget,
)

from configs.configs import create_logs
from components.widgets.grouping.dnd_widgets import DatasetSourceList, GroupDropList
from components.widgets.multi_group_dialog import MultiGroupCreationDialog
from utils import PROJECT_MANAGER


class _UniqueGroupDropList(GroupDropList):
	"""GroupDropList variant that enforces a dataset can only be in one group."""

	def add_dataset(self, name: str) -> bool:  # type: ignore[override]
		name = (name or "").strip()
		if not name:
			return False
		try:
			parent = self.parent()
			while parent is not None:
				if hasattr(parent, "_remove_dataset_from_other_groups"):
					parent._remove_dataset_from_other_groups(
						name, except_group_id=self.group_id
					)
					break
				parent = parent.parent()
		except Exception:
			pass
		return super().add_dataset(name)


@dataclass
class _GroupTab:
	group_id: str
	container: QWidget
	name_edit: QLineEdit
	enabled_checkbox: QCheckBox
	list_widget: GroupDropList


class TabGroupManager(QWidget):
	"""Tab-based group assignment UI (ML-style), for Analysis grouped mode.

	Emits groups as: {group_name: [dataset1, dataset2, ...]}
	"""

	groups_changed = Signal(dict)

	def __init__(self, dataset_names: List[str], localize_func=None, parent=None):
		super().__init__(parent)
		self.dataset_names = list(dataset_names or [])
		self.localize = localize_func
		self._group_counter = 0
		self._tabs: List[_GroupTab] = []
		self._suspend_emit = False
		self._unassigned_tab_index: int | None = None

		self._setup_ui()
		self.reset(add_default=True)

	def _setup_ui(self):
		root = QVBoxLayout(self)
		root.setContentsMargins(0, 0, 0, 0)
		root.setSpacing(10)

		# Tabs (includes Unassigned + group tabs)
		self.groups_tabs = QTabWidget()
		self.groups_tabs.setTabsClosable(True)
		self.groups_tabs.setMovable(False)
		self.groups_tabs.setMinimumHeight(240)
		self.groups_tabs.setStyleSheet(
			"""
			QTabWidget::pane { border: 1px solid #dee2e6; border-radius: 4px; top: -1px; }
			QTabBar::tab { background: #f8f9fa; border: 1px solid #dee2e6; padding: 6px 12px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; color: #495057; }
			QTabBar::tab:selected { background: #ffffff; border-bottom-color: #ffffff; font-weight: 600; color: #0078d4; }
			QTabBar::close-button { subcontrol-position: right; }
			"""
		)
		self.groups_tabs.tabCloseRequested.connect(self._on_tab_close_requested)
		root.addWidget(self.groups_tabs, 1)

		self._ensure_unassigned_tab()

		actions = QHBoxLayout()
		actions.setContentsMargins(0, 0, 0, 0)
		actions.setSpacing(10)

		self.add_group_btn = QPushButton(
			"+ "
			+ (
				self.localize("ANALYSIS_PAGE.create_groups_button")
				if self.localize
				else "Add Group"
			)
		)
		self.add_group_btn.setCursor(Qt.PointingHandCursor)
		self.add_group_btn.setStyleSheet(
			"""
			QPushButton { padding: 6px 10px; border: 1px solid #ced4da; border-radius: 6px; background: #ffffff; font-weight: 600; }
			QPushButton:hover { border-color: #0078d4; color: #0078d4; }
			"""
		)
		self.add_group_btn.setToolTip(
			self.localize("ANALYSIS_PAGE.create_groups_tooltip") if self.localize else "Create groups"
		)
		self.add_group_btn.clicked.connect(self._open_group_creation_dialog)

		self.reset_btn = QPushButton(
			self.localize("ANALYSIS_PAGE.reset_all_button") if self.localize else "Reset"
		)
		self.reset_btn.setCursor(Qt.PointingHandCursor)
		self.reset_btn.setStyleSheet(
			"""
			QPushButton { padding: 6px 10px; border: 1px solid #dc3545; border-radius: 6px; background: #fff5f5; font-weight: 700; color: #b02a37; }
			QPushButton:hover { background: #f8d7da; }
			"""
		)
		self.reset_btn.clicked.connect(lambda: self.reset(add_default=True))

		actions.addWidget(self.add_group_btn)
		actions.addWidget(self.reset_btn)
		actions.addStretch(1)
		root.addLayout(actions)

		self._refresh_unassigned_source()

	def _ensure_unassigned_tab(self) -> None:
		"""Create the special 'Unassigned' tab used as the drag source.

		This reduces cramped UI by showing the source list only when needed.
		"""
		if self._unassigned_tab_index is not None:
			return
		page = QWidget()
		page.setObjectName("analysisUnassignedTab")
		layout = QVBoxLayout(page)
		layout.setContentsMargins(10, 10, 10, 10)
		layout.setSpacing(10)

		self.dataset_source = DatasetSourceList()
		self.dataset_source.setMinimumHeight(220)
		self.dataset_source.setStyleSheet(
			"""
			QListWidget {
				border: 1px solid #ced4da;
				border-radius: 6px;
				background-color: #ffffff;
				padding: 4px;
			}
			QListWidget::item {
				padding: 8px;
				border-bottom: 1px solid #f1f3f4;
			}
			QListWidget::item:selected {
				background-color: #e3f2fd;
				color: #0078d4;
				font-weight: 600;
			}
			"""
		)
		layout.addWidget(self.dataset_source, 1)

		label = (
			self.localize("ANALYSIS_PAGE.unassigned_group") if self.localize else "Unassigned"
		)
		idx = self.groups_tabs.insertTab(0, page, label)
		self._unassigned_tab_index = idx
		# Hide close button for this tab
		try:
			bar = self.groups_tabs.tabBar()
			bar.setTabButton(idx, QTabBar.RightSide, None)
			bar.setTabButton(idx, QTabBar.LeftSide, None)
		except Exception:
			pass

	def set_dataset_names(self, dataset_names: List[str]) -> None:
		self.dataset_names = list(dataset_names or [])
		self._refresh_unassigned_source()

	def _refresh_unassigned_source(self) -> None:
		"""Refresh Unassigned list to include all datasets not assigned to any group."""
		if not hasattr(self, "dataset_source"):
			return
		assigned: set[str] = set()
		for ui in self._tabs:
			lst = ui.list_widget
			for i in range(lst.count()):
				item = lst.item(i)
				ds_name = (item.data(Qt.UserRole) or item.text() or "").strip()
				if ds_name:
					assigned.add(ds_name)

		unassigned = [str(n) for n in self.dataset_names if str(n) not in assigned]
		try:
			self.dataset_source.clear()
			for name in sorted(unassigned):
				self.dataset_source.addItem(str(name))
		except Exception:
			pass

	def _refresh_dataset_source(self):
		# Backward-compatible alias
		self._refresh_unassigned_source()

	def _new_group_id(self) -> str:
		self._group_counter += 1
		return f"group_{self._group_counter}"

	def _create_group_tab(self, group_name: str) -> _GroupTab:
		group_id = self._new_group_id()
		page = QWidget()
		page.setObjectName("analysisGroupTab")
		layout = QVBoxLayout(page)
		layout.setContentsMargins(10, 10, 10, 10)
		layout.setSpacing(10)

		header = QHBoxLayout()
		header.setContentsMargins(0, 0, 0, 0)
		header.setSpacing(10)

		name_edit = QLineEdit(str(group_name or "").strip())
		name_edit.setPlaceholderText(
			self.localize("ANALYSIS_PAGE.group_name_label") if self.localize else "Group name"
		)
		name_edit.setStyleSheet(
			"""
			QLineEdit { padding: 8px; border: 1px solid #ced4da; border-radius: 6px; background: #ffffff; font-size: 12px; }
			QLineEdit:focus { border-color: #0078d4; }
			"""
		)
		header.addWidget(name_edit, 1)

		enabled_checkbox = QCheckBox(
			self.localize("ANALYSIS_PAGE.include_in_analysis")
			if self.localize
			else "Use for analysis"
		)
		enabled_checkbox.setChecked(True)
		enabled_checkbox.setCursor(Qt.PointingHandCursor)
		enabled_checkbox.setStyleSheet(
			"""
			QCheckBox { spacing: 8px; font-size: 13px; font-weight: 600; color: #333; }
			QCheckBox::indicator {
				width: 20px;
				height: 20px;
				border: 2px solid #adb5bd;
				border-radius: 4px;
				background-color: white;
			}
			QCheckBox::indicator:hover { border-color: #0078d4; }
			QCheckBox::indicator:checked {
				background-color: #0078d4;
				border-color: #0078d4;
				image: url(assets/icons/checkmark_white.svg);
			}
			"""
		)
		enabled_checkbox.toggled.connect(lambda _checked: self._emit_groups_changed())
		header.addWidget(enabled_checkbox, 0)

		layout.addLayout(header)

		lst = _UniqueGroupDropList(group_id=group_id, localize_func=self.localize, parent=page)
		lst.setMinimumHeight(180)
		layout.addWidget(lst, 1)

		# Add tab
		tab_title = name_edit.text().strip() or (
			self.localize("ANALYSIS_PAGE.group_name_label") if self.localize else "Group"
		)
		idx = self.groups_tabs.addTab(page, tab_title)
		self.groups_tabs.setCurrentIndex(idx)

		ui = _GroupTab(
			group_id=group_id,
			container=page,
			name_edit=name_edit,
			enabled_checkbox=enabled_checkbox,
			list_widget=lst,
		)
		self._tabs.append(ui)

		def _on_name_changed(_txt: str):
			self._sync_tab_titles()
			self._emit_groups_changed()

		name_edit.textChanged.connect(_on_name_changed)
		return ui

	def _sync_tab_titles(self):
		for i, ui in enumerate(self._tabs):
			try:
				title = ui.name_edit.text().strip() or f"Group {i + 1}"
				# +1 because tab 0 is Unassigned
				self.groups_tabs.setTabText(i + 1, title)
			except Exception:
				pass

	def _find_ui_by_index(self, index: int) -> Optional[_GroupTab]:
		# tab index 0 is Unassigned
		idx = index - 1
		if 0 <= idx < len(self._tabs):
			return self._tabs[idx]
		return None

	def _on_tab_close_requested(self, index: int):
		# Never close Unassigned
		if index == 0:
			return
		ui = self._find_ui_by_index(index)
		if ui is None:
			return
		self.groups_tabs.removeTab(index)
		try:
			self._tabs.pop(index - 1)
		except Exception:
			pass
		self._emit_groups_changed()

	def _open_group_creation_dialog(self):
		"""Open MultiGroupCreationDialog (same UX as ML page)."""
		available = sorted([str(x) for x in (self.dataset_names or [])])
		if not available:
			QMessageBox.warning(
				self,
				self.localize("COMMON.warning") if self.localize else "Warning",
				self.localize("ML_PAGE.no_datasets_loaded")
				if self.localize
				else "No datasets loaded.",
			)
			return

		# Seed dialog rows: current group names + optional saved configs
		saved_cfg = []
		try:
			if PROJECT_MANAGER.current_project_data and hasattr(PROJECT_MANAGER, "get_analysis_group_configs"):
				saved_cfg = PROJECT_MANAGER.get_analysis_group_configs() or []
		except Exception:
			saved_cfg = []

		if self._tabs:
			seed_groups = [
				{
					"name": (ui.name_edit.text() or "").strip(),
					"include": "",
					"exclude": "",
					"auto_assign": False,
				}
				for ui in self._tabs
			]
			seed_groups = [g for g in seed_groups if g.get("name")]
			if saved_cfg:
				by_name = {str(c.get("name") or "").strip(): c for c in saved_cfg if isinstance(c, dict)}
				for g in seed_groups:
					c = by_name.get(str(g.get("name") or "").strip())
					if c:
						g["include"] = c.get("include") or []
						g["exclude"] = c.get("exclude") or []
						g["auto_assign"] = bool(c.get("auto_assign", False))
		else:
			seed_groups = saved_cfg if saved_cfg else [
				{"name": "Group 1", "include": "", "exclude": "", "auto_assign": False},
				{"name": "Group 2", "include": "", "exclude": "", "auto_assign": False},
			]

		dialog = MultiGroupCreationDialog(
			available,
			self.localize,
			self,
			initial_rows=2,
			default_auto_assign=False,
			initial_groups=seed_groups,
		)
		result = dialog.exec()
		# In Qt/PySide, the accepted code is defined on QDialog, not on the dialog instance.
		if result != QDialog.Accepted:
			return

		configs = dialog.get_group_configs() or []
		assignments = dialog.get_assignments() or {}
		if not configs:
			return

		# Confirm replacement if groups already exist
		if self._tabs:
			reply = QMessageBox.question(
				self,
				self.localize("COMMON.confirm") if self.localize else "Confirm",
				self.localize("ML_PAGE.auto_assign_replace_groups_confirm")
				if self.localize
				else "Replace existing groups with the dialog results?",
				QMessageBox.Yes | QMessageBox.No,
			)
			if reply != QMessageBox.Yes:
				return

		# Apply dialog groups
		prev = self._suspend_emit
		self._suspend_emit = True
		self.reset(add_default=False)
		for cfg in configs:
			gname = str(cfg.get("name") or "").strip()
			if not gname:
				continue
			ui = self._create_group_tab(gname)
			for ds in assignments.get(gname, []) or []:
				name = str(ds).strip()
				if name:
					ui.list_widget.add_dataset(name)
		self._suspend_emit = prev
		self._sync_tab_titles()
		self._emit_groups_changed()

		# Persist configs (so the dialog can be prefilled next time)
		try:
			if PROJECT_MANAGER.current_project_data and hasattr(PROJECT_MANAGER, "set_analysis_group_configs"):
				PROJECT_MANAGER.set_analysis_group_configs(configs)
		except Exception:
			pass

	def _remove_dataset_from_other_groups(self, dataset_name: str, *, except_group_id: str):
		dataset_name = (dataset_name or "").strip()
		if not dataset_name:
			return
		for ui in self._tabs:
			if ui.group_id == except_group_id:
				continue
			lst = ui.list_widget
			for i in range(lst.count() - 1, -1, -1):
				item = lst.item(i)
				name = (item.data(Qt.UserRole) or item.text() or "").strip()
				if name == dataset_name:
					lst.takeItem(i)

	def _on_groups_changed(self):
		"""Hook for GroupDropList to call via parent-walk."""
		self._emit_groups_changed()

	def _emit_groups_changed(self):
		if self._suspend_emit:
			return
		# Keep Unassigned list consistent
		try:
			self._refresh_unassigned_source()
		except Exception:
			pass
		try:
			self.groups_changed.emit(self.get_groups())
		except Exception as e:
			create_logs(__name__, __file__, f"Failed to emit groups_changed: {e}", status="debug")

	def reset(self, *, add_default: bool = True):
		prev = self._suspend_emit
		self._suspend_emit = True
		self.groups_tabs.clear()
		self._tabs.clear()
		self._group_counter = 0
		self._unassigned_tab_index = None
		self._ensure_unassigned_tab()

		if add_default:
			self._create_group_tab("Group 1")
			self._create_group_tab("Group 2")

		self._suspend_emit = prev
		self._sync_tab_titles()
		self._emit_groups_changed()

	def set_groups(self, groups: Dict[str, List[str]], enabled: Dict[str, bool] | None = None):
		if not isinstance(groups, dict):
			return
		prev = self._suspend_emit
		self._suspend_emit = True
		self.groups_tabs.clear()
		self._tabs.clear()
		self._group_counter = 0
		self._unassigned_tab_index = None
		self._ensure_unassigned_tab()

		for group_name, ds_list in groups.items():
			gname = str(group_name or "").strip()
			if not gname:
				continue
			ui = self._create_group_tab(gname)
			try:
				ui.enabled_checkbox.setChecked(bool((enabled or {}).get(gname, True)))
			except Exception:
				pass
			for ds in list(ds_list or []):
				name = str(ds or "").strip()
				if name:
					ui.list_widget.add_dataset(name)

		self._suspend_emit = prev
		self._sync_tab_titles()
		self._emit_groups_changed()

	def get_groups(self) -> Dict[str, List[str]]:
		out: Dict[str, List[str]] = {}
		for ui in self._tabs:
			name = (ui.name_edit.text() or "").strip()
			if not name:
				continue
			ds_list: List[str] = []
			for i in range(ui.list_widget.count()):
				item = ui.list_widget.item(i)
				ds_name = (item.data(Qt.UserRole) or item.text() or "").strip()
				if ds_name:
					ds_list.append(ds_name)
			out[name] = ds_list
		return out

	def get_enabled_map(self) -> Dict[str, bool]:
		out: Dict[str, bool] = {}
		for ui in self._tabs:
			name = (ui.name_edit.text() or "").strip()
			if not name:
				continue
			out[name] = bool(ui.enabled_checkbox.isChecked())
		return out

	def get_enabled_groups(self) -> Dict[str, List[str]]:
		"""Return group mapping filtered to only groups checked for analysis."""
		groups = self.get_groups()
		enabled = self.get_enabled_map()
		return {k: v for k, v in groups.items() if bool(enabled.get(k, True))}
