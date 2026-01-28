"""
Export Utilities for Analysis Results

This module handles exporting analysis results to various formats:
- PNG/SVG image export
- CSV data export
- Full report generation
- Project folder integration
"""

import os
from configs.configs import create_logs
from datetime import datetime
from typing import Optional, Any
from pathlib import Path

from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QObject


class ExportManager(QObject):
    """Manages export operations for analysis results."""

    def __init__(self, parent_widget, localize_func, project_manager):
        """
        Initialize export manager.

        Args:
            parent_widget: Parent QWidget for dialogs
            localize_func: Localization function
            project_manager: ProjectManager instance
        """
        super().__init__(parent_widget)
        self.parent = parent_widget
        self.localize = localize_func
        self.project_manager = project_manager

    def export_plot_png(
        self, figure, default_filename: str = "analysis_plot.png"
    ) -> bool:
        """
        Export matplotlib figure to PNG.

        Args:
            figure: Matplotlib figure object
            default_filename: Default filename

        Returns:
            True if export succeeded
        """
        if figure is None:
            self._show_error(self.localize("ANALYSIS_PAGE.no_plot_to_export"))
            return False

        from components.widgets import get_export_options

        try:
            default_path = Path(self._get_default_export_path(default_filename))
            default_dir = str(default_path.parent)
            default_base = default_path.stem
        except Exception:
            default_dir = ""
            default_base = "analysis_plot"

        opts = get_export_options(
            self.parent,
            title=self.localize("ANALYSIS_PAGE.export_png_title"),
            formats=[("png", "PNG (.png)")],
            default_directory=default_dir,
            default_filename=default_base,
            show_filename=True,
            show_format=False,
            # Labels
            format_label="Export Format:",
            location_label="Save Location:",
            filename_label="Filename:",
            browse_button_text="Browse...",
            select_location_title="Select Location",
            show_metadata_checkbox=False,
        )
        if opts is None:
            return False

        base = str(opts.filename or default_base).strip() or default_base
        if not base.lower().endswith(".png"):
            base = f"{base}.png"
        file_path = str(Path(str(opts.directory)) / base)

        try:
            figure.savefig(file_path, dpi=300, bbox_inches="tight", format="png")
            self._show_success(
                self.localize("ANALYSIS_PAGE.export_success").format(file_path)
            )
            return True
        except Exception as e:
            self._show_error(self.localize("ANALYSIS_PAGE.export_error").format(error=str(e)))
            return False

    def export_plot_svg(
        self, figure, default_filename: str = "analysis_plot.svg"
    ) -> bool:
        """
        Export matplotlib figure to SVG.

        Args:
            figure: Matplotlib figure object
            default_filename: Default filename

        Returns:
            True if export succeeded
        """
        if figure is None:
            self._show_error(self.localize("ANALYSIS_PAGE.no_plot_to_export"))
            return False

        from components.widgets import get_export_options

        try:
            default_path = Path(self._get_default_export_path(default_filename))
            default_dir = str(default_path.parent)
            default_base = default_path.stem
        except Exception:
            default_dir = ""
            default_base = "analysis_plot"

        opts = get_export_options(
            self.parent,
            title=self.localize("ANALYSIS_PAGE.export_svg_title"),
            formats=[("svg", "SVG (.svg)")],
            default_directory=default_dir,
            default_filename=default_base,
            show_filename=True,
            show_format=False,
            # Labels
            format_label="Export Format:",
            location_label="Save Location:",
            filename_label="Filename:",
            browse_button_text="Browse...",
            select_location_title="Select Location",
            show_metadata_checkbox=False,
        )
        if opts is None:
            return False

        base = str(opts.filename or default_base).strip() or default_base
        if not base.lower().endswith(".svg"):
            base = f"{base}.svg"
        file_path = str(Path(str(opts.directory)) / base)

        try:
            figure.savefig(file_path, format="svg", bbox_inches="tight")
            self._show_success(
                self.localize("ANALYSIS_PAGE.export_success").format(file_path)
            )
            return True
        except Exception as e:
            self._show_error(self.localize("ANALYSIS_PAGE.export_error").format(error=str(e)))
            return False

    def export_data_csv(
        self, data_table, default_filename: str = "analysis_data.csv"
    ) -> bool:
        """
        Export data table to CSV.

        Args:
            data_table: Pandas DataFrame or dict
            default_filename: Default filename

        Returns:
            True if export succeeded
        """
        if data_table is None:
            self._show_error(self.localize("ANALYSIS_PAGE.no_data_to_export"))
            return False

        from components.widgets import get_export_options

        try:
            default_path = Path(self._get_default_export_path(default_filename))
            default_dir = str(default_path.parent)
            default_base = default_path.stem
        except Exception:
            default_dir = ""
            default_base = "analysis_data"

        opts = get_export_options(
            self.parent,
            title=self.localize("ANALYSIS_PAGE.export_csv_title"),
            formats=[("csv", "CSV (.csv)")],
            default_directory=default_dir,
            default_filename=default_base,
            show_filename=True,
            show_format=False,
            # Labels
            format_label="Export Format:",
            location_label="Save Location:",
            filename_label="Filename:",
            browse_button_text="Browse...",
            select_location_title="Select Location",
            show_metadata_checkbox=False,
        )
        if opts is None:
            return False

        base = str(opts.filename or default_base).strip() or default_base
        if not base.lower().endswith(".csv"):
            base = f"{base}.csv"
        file_path = str(Path(str(opts.directory)) / base)

        try:
            import pandas as pd

            # Convert to DataFrame if dict
            if isinstance(data_table, dict):
                df = pd.DataFrame(data_table)
            else:
                df = data_table

            df.to_csv(file_path, index=False)
            self._show_success(
                self.localize("ANALYSIS_PAGE.export_success").format(file_path)
            )
            return True
        except Exception as e:
            self._show_error(self.localize("ANALYSIS_PAGE.export_error").format(error=str(e)))
            return False
    
    def export_data_multi_format(self, data_table, default_filename: str = "analysis_data") -> bool:
        """
        Export data table to multiple formats (CSV, Excel, JSON, TXT, Pickle).
        Shows format selection dialog similar to preprocess_page.
        
        Args:
            data_table: Pandas DataFrame or dict
            default_filename: Default filename (without extension)
        
        Returns:
            True if export succeeded
        """
        if data_table is None:
            self._show_error(self.localize("ANALYSIS_PAGE.no_data_to_export"))
            return False
        
        from pathlib import Path

        from components.widgets import get_export_options

        export_dir_default = ""
        try:
            export_dir_default = str(Path(self._get_default_export_path("x")).parent)
        except Exception:
            export_dir_default = ""
        formats = [
            ("csv", "CSV (.csv)"),
            ("xlsx", "Excel (.xlsx)"),
            ("json", "JSON (.json)"),
            ("txt", "Text (.txt)"),
            ("pkl", "Pickle (.pkl)"),
        ]

        opts = get_export_options(
            self.parent,
            title=self.localize("ANALYSIS_PAGE.export_button"),
            formats=formats,
            default_directory=export_dir_default,
            default_filename=default_filename,
            show_filename=True,
            show_format=True,
            # Labels
            format_label="Export Format:",
            location_label="Save Location:",
            filename_label="Filename:",
            browse_button_text="Browse...",
            select_location_title="Select Location",
            show_metadata_checkbox=False,
        )

        if opts is None:
            return False

        selected_format = str(opts.format_key or "").strip().lower()
        out_dir = str(opts.directory or "").strip()
        base_name = str(opts.filename or "").strip() or default_filename

        if not out_dir:
            return False

        ext = f".{selected_format}" if selected_format else ""
        if ext and not base_name.lower().endswith(ext):
            base_name = f"{base_name}{ext}"
        file_path = str(Path(out_dir) / base_name)
        
        try:
            import pandas as pd
            
            # Convert to DataFrame if dict
            if isinstance(data_table, dict):
                df = pd.DataFrame(data_table)
            else:
                df = data_table
            
            # Export based on format
            if selected_format == "csv":
                df.to_csv(file_path, index=False)
            elif selected_format == "xlsx":
                df.to_excel(file_path, index=False, engine='openpyxl')
            elif selected_format == "json":
                df.to_json(file_path, orient='records', indent=2, force_ascii=False)
            elif selected_format == "txt":
                df.to_csv(file_path, sep='\\t', index=False)
            elif selected_format == "pkl":
                df.to_pickle(file_path)
            
            self._show_success(
                self.localize("ANALYSIS_PAGE.export_success").format(file_path)
            )
            return True
        except Exception as e:
            self._show_error(self.localize("ANALYSIS_PAGE.export_error").format(error=str(e)))
            return False

    def get_default_export_dir(self) -> str:
        """Best-effort default export directory (project exports folder if available)."""
        try:
            from pathlib import Path

            return str(Path(self._get_default_export_path("x")).parent)
        except Exception:
            return ""
    
    def export_full_report(
        self, result: Any, method_name: str, parameters: dict, dataset_name: str
    ) -> bool:
        """
        Export complete analysis report with all components.

        Args:
            result: AnalysisResult object
            method_name: Analysis method name
            parameters: Analysis parameters used
            dataset_name: Dataset name

        Returns:
            True if export succeeded
        """
        from components.widgets import get_export_options

        default_dir = ""
        try:
            default_dir = str(Path(self._get_default_export_path("x")).parent)
        except Exception:
            default_dir = ""

        opts = get_export_options(
            self.parent,
            title=self.localize("ANALYSIS_PAGE.export_report_title"),
            formats=[("folder", "Folder")],
            default_directory=default_dir,
            default_filename="",
            show_filename=False,
            show_format=False,
            multiple_info_text=self.localize("ANALYSIS_PAGE.export_report_title"),
            # Labels
            format_label="Export Format:",
            location_label="Save Location:",
            filename_label="Filename:",
            browse_button_text="Browse...",
            select_location_title="Select Location",
            show_metadata_checkbox=False,
        )
        if opts is None:
            return False

        folder_path = str(opts.directory or "").strip()
        if not folder_path:
            return False

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_folder = Path(folder_path) / f"{method_name}_{timestamp}"
            report_folder.mkdir(parents=True, exist_ok=True)

            # Export plot
            if result.primary_figure:
                plot_path = report_folder / "plot.png"
                result.primary_figure.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Export data table
            if result.data_table is not None:
                import pandas as pd

                df = (
                    pd.DataFrame(result.data_table)
                    if isinstance(result.data_table, dict)
                    else result.data_table
                )
                data_path = report_folder / "data.csv"
                df.to_csv(data_path, index=False)

            # Create report text file
            report_path = report_folder / "report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"Analysis Report: {method_name}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write("\n" + "=" * 60 + "\n\n")

                f.write("Parameters:\n")
                for key, value in parameters.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n" + "=" * 60 + "\n\n")

                if result.detailed_summary:
                    f.write("Summary:\n")
                    f.write(result.detailed_summary)
                    f.write("\n" + "=" * 60 + "\n")

                if hasattr(result, "diagnostics") and result.diagnostics:
                    f.write("Diagnostics:\n")
                    f.write(result.diagnostics)

            self._show_success(
                self.localize("ANALYSIS_PAGE.export_report_success").format(
                    str(report_folder)
                )
            )
            return True

        except Exception as e:
            self._show_error(self.localize("ANALYSIS_PAGE.export_error").format(error=str(e)))
            return False

    def save_to_project(
        self, result: Any, method_name: str, parameters: dict, dataset_name: str
    ) -> bool:
        """
        Save analysis results to current project folder.

        Args:
            result: AnalysisResult object
            method_name: Analysis method name
            parameters: Analysis parameters
            dataset_name: Dataset name

        Returns:
            True if save succeeded
        """
        if not self.project_manager or not self.project_manager.current_project_data:
            self._show_error(self.localize("ANALYSIS_PAGE.no_project_open"))
            return False

        try:
            # Get project path from current_project_data dict
            project_path = Path(
                self.project_manager.current_project_data.get("projectPath", "")
            )
            if not project_path or not project_path.exists():
                self._show_error(self.localize("ANALYSIS_PAGE.invalid_project_path"))
                return False
            analysis_folder = project_path / "analyses"
            analysis_folder.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_folder = analysis_folder / f"{method_name}_{timestamp}"
            result_folder.mkdir(parents=True, exist_ok=True)

            # Save plot
            if result.primary_figure:
                plot_path = result_folder / "plot.png"
                result.primary_figure.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Save data
            if result.data_table is not None:
                import pandas as pd

                df = (
                    pd.DataFrame(result.data_table)
                    if isinstance(result.data_table, dict)
                    else result.data_table
                )
                df.to_csv(result_folder / "data.csv", index=False)

            # Save metadata
            import json

            metadata = {
                "method": method_name,
                "dataset": dataset_name,
                "parameters": parameters,
                "timestamp": timestamp,
                "summary": result.detailed_summary if result.detailed_summary else "",
            }
            with open(result_folder / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            self._show_success(
                self.localize("ANALYSIS_PAGE.save_project_success").format(
                    str(result_folder)
                )
            )
            return True

        except Exception as e:
            self._show_error(
                self.localize("ANALYSIS_PAGE.save_project_error").format(str(e))
            )
            return False

    def _get_default_export_path(self, filename: str) -> str:
        """
        Get default export path (project folder if available, else home).

        Args:
            filename: Default filename

        Returns:
            Full path string
        """
        if self.project_manager and self.project_manager.current_project_data:
            # Get project path from current_project_data dict
            project_path_str = self.project_manager.current_project_data.get(
                "projectPath", ""
            )
            if project_path_str:
                project_path = Path(project_path_str)
                # Keep consistent with ML page bundle export UX.
                export_folder = project_path / "reports"
                export_folder.mkdir(exist_ok=True)
                return str(export_folder / filename)

        # Fallback to home directory
        return str(Path.home() / filename)

    def _show_success(self, message: str):
        """Show success message box."""
        QMessageBox.information(
            self.parent, self.localize("ANALYSIS_PAGE.export_success_title"), message
        )

    def _show_error(self, message: str):
        """Show error message box."""
        create_logs(
            "export_error",
            "ExportManager",
            message,
            status="error",
        )

        QMessageBox.critical(
            self.parent, self.localize("ANALYSIS_PAGE.export_error_title"), message
        )
