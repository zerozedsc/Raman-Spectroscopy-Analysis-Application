import os
import json
import ast
import datetime
import weakref
import tempfile
from typing import List, Dict, Any, Callable, Optional
import pandas as pd
import argparse
from configs.configs import *

# --- Argument Parser for Command-Line Options ---
# Load persisted config first so CLI defaults can respect it
CONFIGS = load_config()

# --- Argument Parser for Command-Line Options ---
parser = argparse.ArgumentParser(
    description="Raman Spectroscopy Data Analysis Application"
)
parser.add_argument(
    "--lang",
    type=str,
    choices=["en", "ja"],
    default=None,
    help="Set the application language (overrides saved settings)",
)

# Use parse_known_args() to safely handle PyInstaller's import scanning
args, unknown = parser.parse_known_args()
# --- Global In-Memory Data Store ---
# This dictionary will hold the currently loaded DataFrames for the active project.
# The keys are the user-defined dataset names.
RAMAN_DATA: Dict[str, pd.DataFrame] = {}


# --- Lightweight in-process notifications (UI sync) ---
# We avoid making ProjectManager a QObject to keep changes small and safe.
# Listeners are stored as weakrefs so pages can be GC'd without explicit unregister.
_GROUPS_CHANGED_LISTENERS: List[object] = []


def register_groups_changed_listener(cb: Callable[[Optional[str]], None]) -> None:
    """Register a callback invoked when shared groups change.

    Callback signature: cb(origin: str | None) -> None
    """
    try:
        ref: object
        if hasattr(cb, "__self__") and cb.__self__ is not None:
            ref = weakref.WeakMethod(cb)  # type: ignore[arg-type]
        else:
            ref = weakref.ref(cb)  # type: ignore[arg-type]
        _GROUPS_CHANGED_LISTENERS.append(ref)
    except Exception:
        # Best-effort; failing to register should not crash the app.
        pass


def _emit_groups_changed(origin: Optional[str] = None) -> None:
    """Invoke registered listeners and prune dead weakrefs."""
    dead: List[object] = []
    for ref in list(_GROUPS_CHANGED_LISTENERS):
        try:
            cb = ref() if callable(ref) else None
            if cb is None:
                dead.append(ref)
                continue
            cb(origin)
        except Exception:
            # Listener errors must never crash core app.
            continue
    for ref in dead:
        try:
            _GROUPS_CHANGED_LISTENERS.remove(ref)
        except Exception:
            pass


class ProjectManager:
    """Handles all project-related file operations, including managing multiple datasets."""

    def __init__(self, projects_dir: str = "projects"):
        self.projects_dir = os.path.abspath(projects_dir)
        self.current_project_data: Dict[str, Any] = {}
        self._ensure_projects_dir_exists()

    def _ensure_projects_dir_exists(self):
        """Creates the base projects directory if it doesn't exist."""
        os.makedirs(self.projects_dir, exist_ok=True)

    def _get_project_data_dir(self) -> str | None:
        """Returns the path to the dedicated 'data' subdirectory for the current project."""
        if not self.current_project_data:
            create_logs(
                "ProjectManager",
                "projects",
                "Cannot get data dir, no project loaded.",
                status="error",
            )
            return None
        project_name = (
            self.current_project_data.get("projectName", "").replace(" ", "_").lower()
        )
        if not project_name:
            create_logs(
                "ProjectManager",
                "projects",
                "Cannot get data dir, project has no name.",
                status="error",
            )
            return None
        project_root_dir = os.path.join(self.projects_dir, project_name)
        data_dir = os.path.join(project_root_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        return data_dir

    # --- Dataset metadata sidecar handling ---
    def _metadata_sidecar_candidates(self, pickle_path: str) -> List[str]:
        """Return candidate metadata sidecar paths for a dataset pickle.

        Preferred naming (requested): <dataset>_metadata.json alongside the dataset file.
        A secondary legacy-friendly fallback is also supported.
        """

        base_dir = os.path.dirname(pickle_path)
        base_name = os.path.splitext(os.path.basename(pickle_path))[0]
        return [
            os.path.join(base_dir, f"{base_name}_metadata.json"),
            os.path.join(base_dir, f"{base_name}.metadata.json"),
        ]

    def _read_json_dict(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_json_dict_atomic(self, path: str, data: Dict[str, Any]) -> bool:
        """Atomically write a JSON dict (best-effort).

        Ensures readers never see a partially-written file.
        """

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", delete=False, dir=os.path.dirname(path), suffix=".tmp"
            ) as tmp:
                json.dump(data if isinstance(data, dict) else {}, tmp, indent=4, ensure_ascii=False)
                tmp.flush()
                tmp_path = tmp.name
            os.replace(tmp_path, path)
            return True
        except Exception as e:
            create_logs(
                "ProjectManager",
                "projects",
                f"Failed to write metadata sidecar {path}: {e}",
                status="warning",
            )
            try:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return False

    def _metadata_path_from_package_info(self, pickle_path: str, package_info: Dict[str, Any]) -> str | None:
        """Resolve the metadata sidecar path from project entry, falling back to candidates."""

        # Prefer explicitly stored metadataFile (relative filename), if present.
        mf = package_info.get("metadataFile")
        if isinstance(mf, str) and mf.strip():
            candidate = os.path.join(os.path.dirname(pickle_path), mf)
            return candidate

        # Otherwise, use the first existing candidate, else the preferred candidate.
        cands = self._metadata_sidecar_candidates(pickle_path)
        for c in cands:
            if os.path.exists(c):
                return c
        return cands[0] if cands else None

    def _migrate_dataset_metadata_to_sidecar(self, dataset_name: str, package_info: Dict[str, Any]) -> bool:
        """Migrate legacy inline metadata (project JSON) into a sidecar JSON file.

        Returns True if project data was modified (needs save).
        """

        changed = False
        pickle_path = package_info.get("path")
        if not isinstance(pickle_path, str) or not pickle_path:
            return False

        # If metadataFile isn't set but a sidecar exists, bind to it.
        if not package_info.get("metadataFile"):
            for c in self._metadata_sidecar_candidates(pickle_path):
                if os.path.exists(c):
                    package_info["metadataFile"] = os.path.basename(c)
                    changed = True
                    break

        legacy_meta = package_info.get("metadata")
        legacy_meta = legacy_meta if isinstance(legacy_meta, dict) else {}

        # If we already have a sidecar, leave it as source of truth. Optionally merge legacy into it.
        meta_path = self._metadata_path_from_package_info(pickle_path, package_info)
        if not meta_path:
            return changed

        # If sidecar exists, prefer it. If legacy has extra keys and sidecar is empty, write it.
        if os.path.exists(meta_path):
            sidecar_meta = self._read_json_dict(meta_path)
            if (not sidecar_meta) and legacy_meta:
                if self._write_json_dict_atomic(meta_path, legacy_meta):
                    create_logs(
                        "ProjectManager",
                        "projects",
                        f"Migrated legacy inline metadata into existing sidecar for '{dataset_name}': {meta_path}",
                        status="info",
                    )
            # Clear inline metadata to avoid duplication.
            if package_info.get("metadata") not in (None, {}):
                package_info["metadata"] = {}
                changed = True
            return changed

        # Sidecar doesn't exist yet; if legacy metadata exists, create it now.
        if legacy_meta:
            if self._write_json_dict_atomic(meta_path, legacy_meta):
                package_info["metadataFile"] = os.path.basename(meta_path)
                package_info["metadata"] = {}
                changed = True
                create_logs(
                    "ProjectManager",
                    "projects",
                    f"Created metadata sidecar for '{dataset_name}': {meta_path}",
                    status="info",
                )
        else:
            # Ensure inline metadata is normalized.
            if package_info.get("metadata") is None:
                package_info["metadata"] = {}
                changed = True

        return changed

    def _ensure_all_metadata_sidecars(self) -> bool:
        """Ensure all datasets have sidecar metadata written when available.

        Returns True if project entries were modified.
        """

        changed = False
        data_packages = self.current_project_data.get("dataPackages")
        if not isinstance(data_packages, dict):
            return False
        for name, pkg in data_packages.items():
            if isinstance(pkg, dict):
                try:
                    if self._migrate_dataset_metadata_to_sidecar(str(name), pkg):
                        changed = True
                except Exception:
                    continue
        return changed
    
    def _resolve_data_path(self, old_path: str, project_dir: str, project_name: str, dataset_name: str) -> str | None:
        """
        Resolve data file path when project directory has been moved or renamed.
        
        Strategy:
        1. Try relative to current project directory
        2. Try standard project structure (project_dir/data/filename)
        3. Try searching in all subdirectories
        
        Args:
            old_path: The stored (invalid) path
            project_dir: Current project directory
            project_name: Project name
            dataset_name: Dataset name for filename construction
            
        Returns:
            Resolved path or None if not found
        """
        # Extract just the filename
        filename = os.path.basename(old_path)
        
        # Strategy 1: Check in standard data directory
        standard_data_dir = os.path.join(project_dir, "data")
        standard_path = os.path.join(standard_data_dir, filename)
        if os.path.exists(standard_path):
            return standard_path
        
        # Strategy 2: Try with reconstructed filename from dataset name
        sanitized_name = dataset_name.replace(" ", "_").lower()
        constructed_filename = f"{sanitized_name}.pkl"
        constructed_path = os.path.join(standard_data_dir, constructed_filename)
        if os.path.exists(constructed_path):
            return constructed_path
        
        # Strategy 3: Search in project directory and subdirectories
        for root, dirs, files in os.walk(project_dir):
            if filename in files:
                return os.path.join(root, filename)
            if constructed_filename in files:
                return os.path.join(root, constructed_filename)
        
        # Not found
        return None

    def get_recent_projects(self) -> List[Dict[str, Any]]:
        """Scans the projects directory and returns a sorted list of projects by modification date."""
        projects = []
        project_folders = [
            d
            for d in os.listdir(self.projects_dir)
            if os.path.isdir(os.path.join(self.projects_dir, d))
        ]

        for project_name in project_folders:
            project_path = os.path.join(
                self.projects_dir, project_name, f"{project_name}.json"
            )
            if os.path.exists(project_path):
                try:
                    with open(project_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    last_modified_timestamp = os.path.getmtime(project_path)
                    projects.append(
                        {
                            "name": data.get("projectName", project_name),
                            "path": project_path,
                            "last_modified": datetime.datetime.fromtimestamp(
                                last_modified_timestamp
                            ).strftime("%Y-%m-%d %H:%M"),
                            "timestamp": last_modified_timestamp,
                        }
                    )
                except (json.JSONDecodeError, FileNotFoundError):
                    create_logs(
                        "ProjectManager",
                        "projects",
                        f"Could not read project file: {project_path}",
                        status="warning",
                    )
        return sorted(projects, key=lambda p: p["timestamp"], reverse=True)

    def create_new_project(self, project_name: str) -> str | None:
        """Creates a new project JSON file and its associated data directory."""
        sanitized_name = project_name.replace(" ", "_").lower()
        project_root_dir = os.path.join(self.projects_dir, sanitized_name)
        project_path = os.path.join(project_root_dir, f"{sanitized_name}.json")

        if os.path.exists(project_path):
            return None

        project_data = {
            "projectName": project_name,
            "creationDate": datetime.datetime.now().isoformat(),
            "schemaVersion": "1.2",  # Updated schema for nested metadata
            "dataPackages": {},
        }
        try:
            os.makedirs(os.path.join(project_root_dir, "data"), exist_ok=True)
            with open(project_path, "w", encoding="utf-8") as f:
                json.dump(project_data, f, indent=4)
            return project_path
        except IOError as e:
            create_logs(
                "ProjectManager",
                "projects",
                f"Failed to create project file: {e}",
                status="error",
            )
            return None

    def add_dataframe_to_project(
        self, dataset_name: str, df: pd.DataFrame, metadata: dict
    ) -> bool:
        """Saves a DataFrame as a pickle, and updates the project config."""
        data_dir = self._get_project_data_dir()
        if not data_dir:
            return False

        pickle_filename = f"{dataset_name.replace(' ', '_').lower()}.pkl"
        pickle_path = os.path.join(data_dir, pickle_filename)
        try:
            df.to_pickle(pickle_path)
        except IOError as e:
            create_logs(
                "ProjectManager",
                "projects",
                f"Failed to save temporary data file: {e}",
                status="error",
            )
            return False

        if "dataPackages" not in self.current_project_data:
            self.current_project_data["dataPackages"] = {}

        # Write dataset metadata to sidecar file alongside the dataset.
        cands = self._metadata_sidecar_candidates(pickle_path)
        meta_path = cands[0] if cands else None
        if meta_path:
            self._write_json_dict_atomic(meta_path, metadata if isinstance(metadata, dict) else {})

        self.current_project_data["dataPackages"][dataset_name] = {
            "path": pickle_path,
            "metadataFile": os.path.basename(meta_path) if meta_path else "",
            "metadata": {},  # legacy field kept for backward compatibility; no longer used
            "addedDate": datetime.datetime.now().isoformat(),
        }

        self.save_current_project()
        RAMAN_DATA[dataset_name] = df
        return True

    def remove_dataframe_from_project(self, dataset_name: str) -> bool:
        """Removes a dataset from the project, including its pickle file."""
        if (
            "dataPackages" not in self.current_project_data
            or dataset_name not in self.current_project_data["dataPackages"]
        ):
            create_logs(
                "ProjectManager",
                "projects",
                f"Dataset '{dataset_name}' not found in project config.",
                status="warning",
            )
            return False

        package_info = self.current_project_data["dataPackages"].pop(dataset_name, None)
        if package_info:
            pickle_path = package_info.get("path")
            if pickle_path and os.path.exists(pickle_path):
                try:
                    os.remove(pickle_path)
                    create_logs(
                        "ProjectManager",
                        "projects",
                        f"Successfully deleted data file: {pickle_path}",
                        status="info",
                    )
                except OSError as e:
                    create_logs(
                        "ProjectManager",
                        "projects",
                        f"Error deleting data file {pickle_path}: {e}",
                        status="error",
                    )
                    # Don't stop, still try to save the project file

            # Remove metadata sidecar files (best-effort).
            try:
                if isinstance(pickle_path, str) and pickle_path:
                    for p in self._metadata_sidecar_candidates(pickle_path):
                        if os.path.exists(p):
                            os.remove(p)
                    mf = package_info.get("metadataFile")
                    if isinstance(mf, str) and mf.strip():
                        p = os.path.join(os.path.dirname(pickle_path), mf)
                        if os.path.exists(p):
                            os.remove(p)
            except Exception:
                pass

        # Also remove from the in-memory store
        RAMAN_DATA.pop(dataset_name, None)

        # Keep group assignments consistent when datasets are removed.
        # This ensures Analysis/ML pages won't reference deleted datasets.
        try:
            self._reconcile_saved_groups_with_data_packages()
        except Exception as e:
            create_logs(
                "ProjectManager",
                "projects",
                f"Failed to reconcile groups after dataset removal: {e}",
                status="warning",
            )

        self.save_current_project()
        create_logs(
            "ProjectManager",
            "projects",
            f"Successfully removed dataset '{dataset_name}' from project.",
            status="info",
        )
        return True

    def load_project(self, project_path: str) -> bool:
        """Loads project JSON and all associated data packages from their pickle files."""
        global RAMAN_DATA
        try:
            with open(project_path, "r", encoding="utf-8") as f:
                self.current_project_data = json.load(f)
            # Store the full path to the project file itself for saving later
            self.current_project_data["projectFilePath"] = project_path

            RAMAN_DATA.clear()
            data_packages = self.current_project_data.get("dataPackages", {})
            
            # Get project directory for relative path resolution
            project_dir = os.path.dirname(project_path)
            project_name = self.current_project_data.get("projectName", "").replace(" ", "_").lower()
            
            paths_fixed = False
            metadata_migrated = False
            for name, package_info in data_packages.items():
                pickle_path = package_info.get("path")
                if not pickle_path:
                    continue
                    
                # Try to load the data file
                if os.path.exists(pickle_path):
                    # Path is valid, load normally
                    try:
                        RAMAN_DATA[name] = pd.read_pickle(pickle_path)
                    except Exception as e:
                        create_logs(
                            "ProjectManager",
                            "projects",
                            f"Failed to load data package '{name}' from {pickle_path}: {e}",
                            status="warning",
                        )

                    # Migrate legacy inline metadata to sidecar (best-effort).
                    try:
                        if isinstance(package_info, dict) and self._migrate_dataset_metadata_to_sidecar(name, package_info):
                            metadata_migrated = True
                    except Exception:
                        pass
                else:
                    # Path is invalid, try to fix it
                    fixed_path = self._resolve_data_path(pickle_path, project_dir, project_name, name)
                    
                    if fixed_path and os.path.exists(fixed_path):
                        # Path was successfully fixed
                        create_logs(
                            "ProjectManager",
                            "projects",
                            f"Fixed path for '{name}': {pickle_path} -> {fixed_path}",
                            status="info",
                        )
                        
                        # Update the path in project data
                        package_info["path"] = fixed_path
                        paths_fixed = True
                        
                        # Load the data
                        try:
                            RAMAN_DATA[name] = pd.read_pickle(fixed_path)
                        except Exception as e:
                            create_logs(
                                "ProjectManager",
                                "projects",
                                f"Failed to load data package '{name}' from fixed path {fixed_path}: {e}",
                                status="warning",
                            )

                        # Migrate legacy inline metadata to sidecar (best-effort).
                        try:
                            if isinstance(package_info, dict) and self._migrate_dataset_metadata_to_sidecar(name, package_info):
                                metadata_migrated = True
                        except Exception:
                            pass
                    else:
                        create_logs(
                            "ProjectManager",
                            "projects",
                            f"Data file for '{name}' not found at {pickle_path}",
                            status="warning",
                        )
            
            # Save project if any paths were fixed or metadata was migrated.
            if paths_fixed or metadata_migrated:
                self.save_current_project()
                create_logs(
                    "ProjectManager",
                    "projects",
                    "Project paths/metadata were automatically updated and saved",
                    status="info",
                )

            # Initialize analysisGroups if not present (for backward compatibility)
            if "analysisGroups" not in self.current_project_data:
                self.current_project_data["analysisGroups"] = {}

            # Initialize ML groups if not present (for backward compatibility)
            if "mlGroups" not in self.current_project_data:
                self.current_project_data["mlGroups"] = {}
            if "mlGroupConfigs" not in self.current_project_data:
                self.current_project_data["mlGroupConfigs"] = []

            # Reconcile saved groups with actual datasets (auto-fix and persist).
            try:
                changed = self._reconcile_saved_groups_with_data_packages()
                if changed:
                    self.save_current_project()
                    create_logs(
                        "ProjectManager",
                        "projects",
                        "Reconciled saved group assignments with current datasets",
                        status="info",
                    )
            except Exception as e:
                create_logs(
                    "ProjectManager",
                    "projects",
                    f"Failed to reconcile saved groups: {e}",
                    status="warning",
                )

            create_logs(
                "ProjectManager",
                "projects",
                f"Successfully loaded project: {self.current_project_data.get('projectName')}",
                status="info",
            )
            return True
        except (FileNotFoundError, json.JSONDecodeError) as e:
            create_logs(
                "ProjectManager",
                "projects",
                f"Failed to load project {project_path}: {e}",
                status="error",
            )
            self.current_project_data = {}
            RAMAN_DATA.clear()
            return False

    def save_current_project(self):
        """Saves the current project data back to its JSON file."""
        project_path = self.current_project_data.get("projectFilePath")
        if not project_path:
            create_logs(
                "ProjectManager",
                "projects",
                "Cannot save project: No project file path is set.",
                status="error",
            )
            return

        try:
            # Best-effort: ensure metadata sidecars are written before we persist.
            try:
                changed = self._ensure_all_metadata_sidecars()
                if changed:
                    create_logs(
                        "ProjectManager",
                        "projects",
                        "Ensured dataset metadata sidecars are up to date",
                        status="info",
                    )
            except Exception:
                pass

            # Create a copy to avoid saving the file path into the JSON itself
            data_to_save = self.current_project_data.copy()
            data_to_save.pop("projectFilePath", None)
            with open(project_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=4)
        except IOError as e:
            create_logs(
                "ProjectManager",
                "projects",
                f"Failed to save project {project_path}: {e}",
                status="error",
            )

    def get_dataframe_metadata(self, dataset_name: str) -> Dict[str, Any] | None:
        """Retrieve metadata for a specific dataset."""
        if "dataPackages" not in self.current_project_data:
            return None

        package_info = self.current_project_data["dataPackages"].get(dataset_name)
        if isinstance(package_info, dict):
            pickle_path = package_info.get("path")
            if isinstance(pickle_path, str) and pickle_path:
                meta_path = self._metadata_path_from_package_info(pickle_path, package_info)
                if meta_path and os.path.exists(meta_path):
                    return self._read_json_dict(meta_path)

                # Legacy fallback: if inline metadata exists, return it and create sidecar lazily.
                legacy = package_info.get("metadata")
                legacy = legacy if isinstance(legacy, dict) else {}
                if legacy and meta_path:
                    try:
                        if self._write_json_dict_atomic(meta_path, legacy):
                            package_info["metadataFile"] = os.path.basename(meta_path)
                            package_info["metadata"] = {}
                            self.save_current_project()
                    except Exception:
                        pass
                return legacy
        return None

    def update_dataframe_metadata(
        self, dataset_name: str, metadata: Dict[str, Any]
    ) -> bool:
        """Update metadata for an existing dataset."""
        if "dataPackages" not in self.current_project_data:
            return False

        if dataset_name in self.current_project_data["dataPackages"]:
            pkg = self.current_project_data["dataPackages"][dataset_name]
            if not isinstance(pkg, dict):
                return False

            pickle_path = pkg.get("path")
            if not isinstance(pickle_path, str) or not pickle_path:
                return False

            meta_path = self._metadata_path_from_package_info(pickle_path, pkg)
            if not meta_path:
                return False

            if not self._write_json_dict_atomic(meta_path, metadata if isinstance(metadata, dict) else {}):
                return False

            pkg["metadataFile"] = os.path.basename(meta_path)
            pkg["metadata"] = {}  # legacy field
            self.save_current_project()
            return True
        return False

    def get_all_dataframe_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all datasets in the project."""
        if "dataPackages" not in self.current_project_data:
            return {}

        metadata_dict = {}
        for name, package_info in self.current_project_data["dataPackages"].items():
            try:
                meta = self.get_dataframe_metadata(name)
                metadata_dict[name] = meta if isinstance(meta, dict) else {}
            except Exception:
                metadata_dict[name] = {}
        return metadata_dict

    def get_analysis_groups(self) -> Dict[str, List[str]]:
        """
        Get saved group assignments for analysis methods.

        Returns:
            Dictionary mapping group names to lists of dataset names
            Example: {"MM": ["dataset1", "dataset2"], "MGUS": ["dataset3"]}
        """
        # Analysis and ML pages share the same conceptual grouping.
        # For backward compatibility we keep both keys in the project JSON, but
        # always read whichever one is populated.
        ag = self.current_project_data.get("analysisGroups", {})
        if isinstance(ag, dict) and ag:
            return ag
        mg = self.current_project_data.get("mlGroups", {})
        return mg if isinstance(mg, dict) else {}

    def set_analysis_groups(self, groups: Dict[str, List[str]], *, origin: str | None = None):
        """
        Save group assignments for analysis methods.

        Args:
            groups: Dictionary mapping group names to dataset lists
        """
        # Keep analysis and ML groups synchronized.
        self.current_project_data["analysisGroups"] = groups
        self.current_project_data["mlGroups"] = groups

        # Clean analysis enabled map to known groups (prevents stale names lingering).
        try:
            enabled_existing = self.current_project_data.get("analysisGroupEnabled", {})
            enabled_clean: Dict[str, bool] = {}
            if isinstance(enabled_existing, dict):
                for gname in groups.keys():
                    enabled_clean[str(gname)] = bool(enabled_existing.get(gname, True))
            self.current_project_data["analysisGroupEnabled"] = enabled_clean
        except Exception:
            pass
        self.save_current_project()
        try:
            _emit_groups_changed(origin)
        except Exception:
            pass
        create_logs(
            "ProjectManager",
            "projects",
            f"Saved {len(groups)} shared groups (analysis + ML)",
            status="info",
        )

    def get_analysis_group_enabled(self) -> Dict[str, bool]:
        """Get Analysis group enabled/disabled state by group name."""
        m = self.current_project_data.get("analysisGroupEnabled", {})
        if not isinstance(m, dict):
            return {}
        out: Dict[str, bool] = {}
        for k, v in m.items():
            if not k:
                continue
            out[str(k)] = bool(v)
        return out

    def set_analysis_group_enabled(self, enabled: Dict[str, bool], *, origin: str | None = None):
        """Persist Analysis group enabled flags.

        Stored separately from the shared group mapping so ML can keep its own enabled map.
        """
        groups = self.get_analysis_groups() or {}
        enabled_clean: Dict[str, bool] = {}
        if isinstance(groups, dict) and isinstance(enabled, dict):
            for gname in groups.keys():
                enabled_clean[str(gname)] = bool(enabled.get(gname, True))
        self.current_project_data["analysisGroupEnabled"] = enabled_clean
        self.save_current_project()
        try:
            _emit_groups_changed(origin)
        except Exception:
            pass

    def get_analysis_group_configs(self) -> List[Dict[str, Any]]:
        """Get Analysis group creation configs (include/exclude keywords, auto-assign flags)."""
        cfg = self.current_project_data.get("analysisGroupConfigs", [])
        return cfg if isinstance(cfg, list) else []

    def set_analysis_group_configs(self, configs: List[Dict[str, Any]]):
        """Save Analysis group configs (from MultiGroupCreationDialog)."""
        self.current_project_data["analysisGroupConfigs"] = configs
        self.save_current_project()
        create_logs(
            "ProjectManager",
            "projects",
            f"Saved {len(configs)} Analysis group configs",
            status="info",
        )

    def get_ml_groups(self) -> Dict[str, List[str]]:
        """Get saved group assignments for the Machine Learning page.

        Returns:
            Dictionary mapping group names to lists of dataset names.
        """
        mg = self.current_project_data.get("mlGroups", {})
        if isinstance(mg, dict) and mg:
            return mg
        ag = self.current_project_data.get("analysisGroups", {})
        return ag if isinstance(ag, dict) else {}

    def get_ml_group_enabled_map(self) -> Dict[str, bool]:
        """Get ML group enabled/disabled state by group name.

        Stored separately so the core group format stays compatible with the Analysis page.
        """
        m = self.current_project_data.get("mlGroupEnabled", {})
        if not isinstance(m, dict):
            return {}
        out: Dict[str, bool] = {}
        for k, v in m.items():
            if not k:
                continue
            out[str(k)] = bool(v)
        return out

    def set_ml_groups_and_enabled(self, groups: Dict[str, List[str]], enabled: Dict[str, bool], *, origin: str | None = None):
        """Save ML groups + enabled flags in a single project write."""
        # Keep analysis and ML groups synchronized.
        self.current_project_data["analysisGroups"] = groups
        self.current_project_data["mlGroups"] = groups
        # Filter enabled map to known groups (prevents stale names lingering)
        enabled_clean: Dict[str, bool] = {}
        if isinstance(enabled, dict):
            for gname in groups.keys():
                enabled_clean[str(gname)] = bool(enabled.get(gname, True))
        self.current_project_data["mlGroupEnabled"] = enabled_clean
        self.save_current_project()
        try:
            _emit_groups_changed(origin)
        except Exception:
            pass
        create_logs(
            "ProjectManager",
            "projects",
            f"Saved {len(groups)} shared groups (analysis + ML) with enabled map",
            status="info",
        )

    def set_ml_groups(self, groups: Dict[str, List[str]]):
        """Save group assignments for the Machine Learning page.

        Backward compatible wrapper: preserves current enabled-map if present.
        """
        self.set_ml_groups_and_enabled(groups, self.get_ml_group_enabled_map())

    def get_ml_group_configs(self) -> List[Dict[str, Any]]:
        """Get ML group creation configs (include/exclude keywords, auto-assign flags)."""
        cfg = self.current_project_data.get("mlGroupConfigs", [])
        return cfg if isinstance(cfg, list) else []

    def set_ml_group_configs(self, configs: List[Dict[str, Any]]):
        """Save ML group configs (from MultiGroupCreationDialog)."""
        self.current_project_data["mlGroupConfigs"] = configs
        self.save_current_project()
        create_logs(
            "ProjectManager",
            "projects",
            f"Saved {len(configs)} ML group configs",
            status="info",
        )

    def _reconcile_saved_groups_with_data_packages(self) -> bool:
        """Remove references to datasets that no longer exist in the project.

        This keeps saved group assignments stable when datasets are deleted,
        moved, or otherwise removed from the project.

        Returns:
            True if any changes were made.
        """
        changed = False
        data_packages = self.current_project_data.get("dataPackages", {})
        valid_datasets = set(map(str, data_packages.keys()))

        def reconcile_group_map(key: str) -> bool:
            nonlocal changed
            groups = self.current_project_data.get(key, {})
            if not isinstance(groups, dict):
                self.current_project_data[key] = {}
                changed = True
                return True
            new_groups: Dict[str, List[str]] = {}
            for gname, ds_list in groups.items():
                if not gname:
                    continue
                if not isinstance(ds_list, list):
                    ds_list = []
                filtered = [str(ds) for ds in ds_list if str(ds) in valid_datasets]
                if filtered != ds_list:
                    changed = True
                new_groups[str(gname)] = filtered
            if new_groups != groups:
                self.current_project_data[key] = new_groups
                changed = True
                return True
            return False

        reconcile_group_map("analysisGroups")
        reconcile_group_map("mlGroups")

        # Reconcile mlGroupConfigs dataset-independent fields (best-effort):
        # ensure list-of-dicts, and drop configs with empty/invalid names.
        cfg = self.current_project_data.get("mlGroupConfigs", [])
        if not isinstance(cfg, list):
            self.current_project_data["mlGroupConfigs"] = []
            changed = True
        else:
            def _normalize_keywords(value) -> List[str]:
                """Normalize include/exclude keywords to a clean list of strings.

                Supports legacy persisted formats:
                - list[str]
                - comma-separated string
                - Python list literal string like "['MM', 'foo']"
                """
                items: List[str] = []
                try:
                    if value is None:
                        items = []
                    elif isinstance(value, (list, tuple)):
                        items = [str(x).strip() for x in value]
                    elif isinstance(value, str):
                        s = value.strip()
                        if not s:
                            items = []
                        elif s.startswith("[") and s.endswith("]"):
                            # Legacy: stored as a stringified Python list.
                            try:
                                parsed = ast.literal_eval(s)
                                if isinstance(parsed, (list, tuple)):
                                    items = [str(x).strip() for x in parsed]
                                else:
                                    items = [s]
                            except Exception:
                                items = [kw.strip() for kw in s.strip("[]").split(",")]
                        else:
                            items = [kw.strip() for kw in s.split(",")]
                    else:
                        items = [str(value).strip()]
                except Exception:
                    items = []

                # Drop empties + dedupe (keep order)
                out: List[str] = []
                seen = set()
                for kw in items:
                    kw = str(kw or "").strip()
                    if not kw:
                        continue
                    k = kw.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(kw)
                return out

            new_cfg: List[Dict[str, Any]] = []
            for item in cfg:
                if not isinstance(item, dict):
                    changed = True
                    continue
                name = str(item.get("name") or "").strip()
                if not name:
                    changed = True
                    continue
                # Keep only known keys to avoid bloat.
                new_cfg.append(
                    {
                        "name": name,
                        "include": _normalize_keywords(item.get("include")),
                        "exclude": _normalize_keywords(item.get("exclude")),
                        "auto_assign": bool(item.get("auto_assign", False)),
                    }
                )
            if new_cfg != cfg:
                self.current_project_data["mlGroupConfigs"] = new_cfg
                changed = True

        return changed


arg_lang = args.lang

# Final language selection: CLI overrides persisted config; fallback to English
selected_lang = (arg_lang or CONFIGS.get("language") or "en").strip()
if selected_lang not in ("en", "ja"):
    selected_lang = "en"

# Keep CONFIGS consistent with what we actually use
CONFIGS["language"] = selected_lang

# Localization: always fallback to English, start with selected language
LOCALIZEMANAGER = LocalizationManager(default_lang="en", initial_lang=selected_lang)
PROJECT_MANAGER = ProjectManager()


# --- Shorthand Function ---
def LOCALIZE(key, **kwargs):
    return LOCALIZEMANAGER.get(key, **kwargs)


def restart_application(*, reason: str | None = None) -> bool:
    """Restart the current application instance.

    Works for:
    - dev runs (python/uv): relaunches the current script with the same args
    - frozen builds (portable/installed): relaunches the current executable

    Returns:
        True if a new process was spawned (best-effort), False otherwise.
    """

    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        import subprocess
        import sys
        import os

        is_frozen = bool(getattr(sys, "frozen", False))

        if is_frozen:
            cmd = [sys.executable] + sys.argv[1:]
        else:
            script = os.path.abspath(sys.argv[0])
            cmd = [sys.executable, script] + sys.argv[1:]

        create_logs(
            "restart_application",
            "app",
            f"Restarting app (frozen={is_frozen}, reason={reason}): {cmd}",
            status="info",
        )

        # For PyInstaller onefile builds, restarting too quickly from an extracted
        # _MEI... directory can cause flaky imports (notably numpy). Make the new
        # process start from the executable's directory and reset the PyInstaller
        # environment so it extracts cleanly.
        env = os.environ.copy()
        if is_frozen:
            env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
            for k in list(env.keys()):
                if k == "_MEIPASS2" or k.startswith("_PYI_"):
                    env.pop(k, None)
            safe_cwd = os.path.dirname(sys.executable)
        else:
            safe_cwd = os.getcwd()

        creationflags = 0
        try:
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        except Exception:
            creationflags = 0

        subprocess.Popen(cmd, cwd=safe_cwd, env=env, creationflags=creationflags)

        app = QApplication.instance()
        if app is not None:
            QTimer.singleShot(0, app.quit)
        return True
    except Exception as e:
        try:
            create_logs(
                "restart_application",
                "app",
                f"Failed to restart application: {e}",
                status="error",
            )
        except Exception:
            pass
        return False
