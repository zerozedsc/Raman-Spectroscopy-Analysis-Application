"""
Figure management utilities for Raman spectroscopy visualization.

This module provides the FigureManager class for managing, displaying, and
exporting matplotlib figures generated during preprocessing steps.
"""

import os
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from functions.configs import console_log


class FigureManager:
    """
    A class to manage and interact with saved matplotlib figures from preprocessing steps.

    This class provides utilities to display, compare, save, and manage figures
    that are stored in the plot_data dictionary from RamanPipeline preprocessing.

    Attributes:
        None
    """

    def __init__(self):
        """Initialize the FigureManager."""
        pass

    def show_saved_figure(self, plot_data: Dict[str, Any], step_key: str):
        """
        Display a saved matplotlib figure.

        Args:
            plot_data (Dict[str, Any]): The plot_data dictionary from preprocess method
            step_key (str): Key for the specific step to display (e.g., 'raw', 'step_1_Cropper')

        Raises:
            ValueError: If step_key not found or figure doesn't exist
        """
        if step_key not in plot_data:
            available_keys = list(plot_data.keys())
            raise ValueError(
                f"Step key '{step_key}' not found. Available keys: {available_keys}"
            )

        step_data = plot_data[step_key]

        if "figure" not in step_data:
            raise ValueError(f"No figure found for step '{step_key}'")

        fig = step_data["figure"]
        fig.show()

    def show_combined_figure(self, plot_data: Dict[str, Any]):
        """
        Display the combined preprocessing figure.

        Args:
            plot_data (Dict[str, Any]): The plot_data dictionary from preprocess method

        Raises:
            ValueError: If combined figure not found
        """
        if "combined_figure" not in plot_data:
            raise ValueError("No combined figure found in plot_data")

        fig = plot_data["combined_figure"]
        fig.show()

    def list_available_figures(self, plot_data: Dict[str, Any]) -> None:
        """
        List all available figures in the plot data.

        Args:
            plot_data (Dict[str, Any]): The plot_data dictionary from preprocess method
        """
        console_log("Available preprocessing figures:")
        console_log("=" * 40)

        for key, data in plot_data.items():
            if key == "combined_figure":
                console_log(f"Key: {key} | Combined preprocessing steps figure")
            elif "figure" in data:
                step_info = f"Key: {key}"
                if "step_index" in data:
                    step_info += f" | Step {data['step_index']}"
                if "step_name" in data:
                    step_info += f" | {data['step_name']}"
                if "title" in data:
                    step_info += f" | {data['title']}"

                console_log(step_info)

    def create_figure_comparison(
        self,
        plot_data: Dict[str, Any],
        step_keys: List[str],
        figsize: Tuple[int, int] = (15, 10),
    ) -> plt.Figure:
        """
        Create a new figure comparing multiple saved figures vertically.

        Args:
            plot_data (Dict[str, Any]): The plot_data dictionary from preprocess method
            step_keys (List[str]): List of step keys to compare
            figsize (Tuple[int, int]): Figure size for the comparison

        Returns:
            plt.Figure: New comparison figure

        Raises:
            ValueError: If no valid figures found for comparison
        """
        valid_keys = [
            key for key in step_keys if key in plot_data and "figure" in plot_data[key]
        ]

        if not valid_keys:
            raise ValueError("No valid figures found for comparison")

        n_plots = len(valid_keys)
        comparison_fig, comparison_axes = plt.subplots(
            n_plots, 1, figsize=figsize, sharex=True
        )

        if n_plots == 1:
            comparison_axes = [comparison_axes]

        for i, key in enumerate(valid_keys):
            original_fig = plot_data[key]["figure"]
            original_ax = original_fig.axes[0]

            # Copy the plot data to new axis
            for line in original_ax.get_lines():
                comparison_axes[i].plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    alpha=line.get_alpha() or 1.0,
                    color=line.get_color(),
                    linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth(),
                )

            comparison_axes[i].set_title(plot_data[key]["title"])
            comparison_axes[i].set_ylabel("Intensity")

            for text in original_ax.texts:
                comparison_axes[i].text(
                    text.get_position()[0],
                    text.get_position()[1],
                    text.get_text(),
                    transform=text.get_transform(),
                    **text.get_properties(),
                )

        comparison_axes[-1].set_xlabel("Wavenumber (cm⁻¹)")
        comparison_fig.tight_layout()

        return comparison_fig

    def save_figure_to_file(
        self,
        plot_data: Dict[str, Any],
        step_key: str,
        filepath: str,
        dpi: int = 300,
        **kwargs,
    ):
        """
        Save a specific figure to file.

        Args:
            plot_data (Dict[str, Any]): The plot_data dictionary from preprocess method
            step_key (str): Key for the specific step to save
            filepath (str): Path where to save the figure
            dpi (int): Resolution for saving (default: 300)
            **kwargs: Additional arguments for plt.savefig()

        Raises:
            ValueError: If figure not found for step_key
        """
        if step_key not in plot_data or "figure" not in plot_data[step_key]:
            raise ValueError(f"Figure not found for step '{step_key}'")

        fig = plot_data[step_key]["figure"]
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight", **kwargs)
        console_log(f"Figure saved to: {filepath}")

    def save_all_figures(
        self,
        plot_data: Dict[str, Any],
        output_dir: str,
        prefix: str = "preprocessing",
        dpi: int = 300,
        **kwargs,
    ):
        """
        Save all figures to individual files.

        Args:
            plot_data (Dict[str, Any]): The plot_data dictionary from preprocess method
            output_dir (str): Directory to save figures
            prefix (str): Prefix for saved file names (default: "preprocessing")
            dpi (int): Resolution for saving (default: 300)
            **kwargs: Additional arguments for plt.savefig()

        Returns:
            List[str]: List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        saved_files = []

        for key, data in plot_data.items():
            if "figure" in data:
                if key == "combined_figure":
                    filename = f"{prefix}_combined.png"
                else:
                    step_index = data.get("step_index", 0)
                    step_name = data.get("step_name", key)
                    filename = f"{prefix}_{step_index:02d}_{step_name}.png"

                filepath = os.path.join(output_dir, filename)
                self.save_figure_to_file(plot_data, key, filepath, dpi=dpi, **kwargs)
                saved_files.append(filepath)

        console_log(f"Saved {len(saved_files)} figures to {output_dir}")
        return saved_files

    def close_all_figures(self, plot_data: Dict[str, Any]):
        """
        Close all figures to free memory.

        Args:
            plot_data (Dict[str, Any]): The plot_data dictionary from preprocess method
        """
        closed_count = 0

        for key, data in plot_data.items():
            if key == "combined_figure":
                plt.close(data)
                closed_count += 1
            elif "figure" in data:
                plt.close(data["figure"])
                closed_count += 1

        console_log(f"Closed {closed_count} figures.")

    def get_figure_info(
        self, plot_data: Dict[str, Any], step_key: str
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific figure.

        Args:
            plot_data (Dict[str, Any]): The plot_data dictionary from preprocess method
            step_key (str): Key for the specific step

        Returns:
            Dict[str, Any]: Figure information including size, title, parameters, etc.

        Raises:
            ValueError: If step_key not found or no figure exists
        """
        if step_key not in plot_data:
            raise ValueError(f"Step key '{step_key}' not found")

        step_data = plot_data[step_key]

        if "figure" not in step_data:
            raise ValueError(f"No figure found for step '{step_key}'")

        fig = step_data["figure"]

        info = {
            "step_key": step_key,
            "title": step_data.get("title", "Unknown"),
            "step_index": step_data.get("step_index", None),
            "step_name": step_data.get("step_name", "Unknown"),
            "figure_size": fig.get_size_inches(),
            "num_axes": len(fig.axes),
            "parameters": step_data.get("parameters", {}),
            "figure_object": fig,
        }

        return info

    def create_summary_table(self, plot_data: Dict[str, Any]) -> None:
        """
        Create a summary table of all available figures.

        Args:
            plot_data (Dict[str, Any]): The plot_data dictionary from preprocess method
        """
        import pandas as pd

        summary_data = []

        for key, data in plot_data.items():
            if "figure" in data or key == "combined_figure":
                row = {
                    "Key": key,
                    "Step Index": data.get("step_index", "N/A"),
                    "Step Name": data.get(
                        "step_name",
                        "Combined" if key == "combined_figure" else "Unknown",
                    ),
                    "Title": data.get("title", "N/A"),
                    "Has Parameters": "Yes" if data.get("parameters") else "No",
                }
                summary_data.append(row)

        if summary_data:
            df = pd.DataFrame(summary_data)
            console_log("Figure Summary:")
            console_log("=" * 60)
            console_log(df.to_string(index=False))
        else:
            console_log("No figures found in plot_data")

    def export_figure_metadata(self, plot_data: Dict[str, Any], output_file: str):
        """
        Export figure metadata to JSON file.

        Args:
            plot_data (Dict[str, Any]): The plot_data dictionary from preprocess method
            output_file (str): Path to output JSON file
        """
        import json

        metadata = {}

        for key, data in plot_data.items():
            if "figure" in data or key == "combined_figure":
                metadata[key] = {
                    "title": data.get("title", ""),
                    "step_index": data.get("step_index"),
                    "step_name": data.get("step_name", ""),
                    "parameters": data.get("parameters", {}),
                    "has_figure": True,
                }

        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        console_log(f"Metadata exported to: {output_file}")

    def create_side_by_side_comparison(
        self,
        plot_data: Dict[str, Any],
        step_keys: List[str],
        figsize: Tuple[int, int] = (20, 6),
    ) -> plt.Figure:
        """
        Create a side-by-side comparison of selected figures horizontally.

        Args:
            plot_data (Dict[str, Any]): The plot_data dictionary from preprocess method
            step_keys (List[str]): List of step keys to compare side by side
            figsize (Tuple[int, int]): Figure size for the comparison

        Returns:
            plt.Figure: New side-by-side comparison figure

        Raises:
            ValueError: If no valid figures found for comparison
        """
        valid_keys = [
            key for key in step_keys if key in plot_data and "figure" in plot_data[key]
        ]

        if not valid_keys:
            raise ValueError("No valid figures found for comparison")

        n_plots = len(valid_keys)
        comparison_fig, comparison_axes = plt.subplots(
            1, n_plots, figsize=figsize, sharey=True
        )

        if n_plots == 1:
            comparison_axes = [comparison_axes]

        for i, key in enumerate(valid_keys):
            original_fig = plot_data[key]["figure"]
            original_ax = original_fig.axes[0]

            for line in original_ax.get_lines():
                comparison_axes[i].plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    alpha=line.get_alpha() or 1.0,
                    color=line.get_color(),
                    linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth(),
                )

            comparison_axes[i].set_title(plot_data[key]["title"])
            comparison_axes[i].set_xlabel("Wavenumber (cm⁻¹)")

            if i == 0:
                comparison_axes[i].set_ylabel("Intensity")

            for text in original_ax.texts:
                comparison_axes[i].text(
                    text.get_position()[0],
                    text.get_position()[1],
                    text.get_text(),
                    transform=text.get_transform(),
                    **text.get_properties(),
                )

        comparison_fig.tight_layout()

        return comparison_fig


def add_figure_manager_to_raman_pipeline():
    """
    Helper function to add FigureManager methods to RamanPipeline class.

    Returns:
        function: get_figure_manager method that returns FigureManager instance
    """

    def get_figure_manager(self):
        """Get a FigureManager instance for working with saved figures."""
        return FigureManager()

    return get_figure_manager
