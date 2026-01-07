import sys
import os
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Dict, Any, List, Tuple


logger = logging.getLogger(__name__)


def detect_signal_range(
    wavenumbers,
    intensities,
    noise_threshold_percentile=20,
    signal_threshold_factor=1.2,
    focus_padding=None,
    crop_bounds=None,
):
    """
    Automatically detect the range of wavenumbers where there is meaningful signal.
    Optimized for Raman spectroscopy data.

    Args:
        wavenumbers: Array of wavenumber values
        intensities: Array or 2D array of intensity values
        noise_threshold_percentile: Percentile to use for noise floor estimation
        signal_threshold_factor: Factor above noise floor to consider as signal
        focus_padding: Additional padding in wavenumber units (default: None for percentage-based padding)
        crop_bounds: Tuple of (min_wn, max_wn) to use as base range with padding instead of auto-detection

    Returns:
        tuple: (min_wavenumber, max_wavenumber) for the focused range
    """
    try:
        # Handle 2D data by taking mean across spectra
        if len(intensities.shape) == 2:
            mean_intensity = np.mean(intensities, axis=0)
        else:
            mean_intensity = intensities

        # If crop_bounds are provided, use them as base range with padding
        if crop_bounds is not None:
            min_crop, max_crop = crop_bounds

            # Apply focus_padding to the crop bounds
            if focus_padding is not None:
                padded_min = min_crop - focus_padding
                padded_max = max_crop + focus_padding
            else:
                # Default fixed padding of 50 wavenumber units
                padded_min = min_crop - 50
                padded_max = max_crop + 50

            # Ensure bounds are within data range
            data_min = np.min(wavenumbers)
            data_max = np.max(wavenumbers)
            final_min = max(data_min, padded_min)
            final_max = min(data_max, padded_max)

            return final_min, final_max

        # For Raman spectroscopy, try a different approach
        # Look for regions with significant variance (indicating peaks)
        window_size = max(10, len(mean_intensity) // 50)  # Adaptive window size

        # Calculate local variance to find peak regions
        variance_signal = np.zeros_like(mean_intensity)
        for i in range(len(mean_intensity)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(mean_intensity), i + window_size // 2)
            local_data = mean_intensity[start_idx:end_idx]
            variance_signal[i] = np.var(local_data)

        # Find regions with high variance (peaks)
        variance_threshold = np.percentile(
            variance_signal, 70
        )  # Top 30% variance regions
        high_variance_mask = variance_signal > variance_threshold

        # Also look for regions above intensity threshold
        intensity_threshold = np.percentile(
            mean_intensity, 70
        )  # Top 30% intensity regions
        high_intensity_mask = mean_intensity > intensity_threshold

        # Combine both criteria
        signal_mask = high_variance_mask | high_intensity_mask
        signal_indices = np.where(signal_mask)[0]

        if len(signal_indices) == 0:
            # Fallback: focus on middle 60% of spectrum (typical Raman range)
            start_idx = int(len(wavenumbers) * 0.2)
            end_idx = int(len(wavenumbers) * 0.8)
            return wavenumbers[start_idx], wavenumbers[end_idx]

        # Find contiguous regions of signal
        signal_start = signal_indices[0]
        signal_end = signal_indices[-1]

        # Add padding based on focus_padding parameter or default percentage
        if focus_padding is not None:
            # Convert focus_padding (wavenumber units) to indices
            wn_per_index = (wavenumbers[-1] - wavenumbers[0]) / len(wavenumbers)
            padding_indices = int(focus_padding / wn_per_index)
        else:
            # Default: 15% on each side
            padding_indices = int(len(wavenumbers) * 0.1)

        start_idx = max(0, signal_start - padding_indices)
        end_idx = min(len(wavenumbers) - 1, signal_end + padding_indices)

        # Ensure reasonable range (at least 25% of total, at most 80%)
        min_range = (wavenumbers[-1] - wavenumbers[0]) * 0.25
        max_range = (wavenumbers[-1] - wavenumbers[0]) * 0.8
        current_range = wavenumbers[end_idx] - wavenumbers[start_idx]

        if current_range < min_range:
            # Expand to minimum range
            center_idx = (start_idx + end_idx) // 2
            half_min_indices = int(len(wavenumbers) * 0.125)  # 12.5% on each side
            start_idx = max(0, center_idx - half_min_indices)
            end_idx = min(len(wavenumbers) - 1, center_idx + half_min_indices)
        elif current_range > max_range:
            # Contract to maximum range
            center_idx = (start_idx + end_idx) // 2
            half_max_indices = int(len(wavenumbers) * 0.4)  # 40% on each side
            start_idx = max(0, center_idx - half_max_indices)
            end_idx = min(len(wavenumbers) - 1, center_idx + half_max_indices)

        return wavenumbers[start_idx], wavenumbers[end_idx]

    except Exception as e:
        # Fallback to middle 60% of range (common Raman region)
        start_idx = int(len(wavenumbers) * 0.2)
        end_idx = int(len(wavenumbers) * 0.8)
        return wavenumbers[start_idx], wavenumbers[end_idx]


class MatplotlibWidget(QWidget):
    """
    A custom widget to embed a Matplotlib plot into a PySide6 application.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("matplotlibWidget")

        # --- Create a Figure and a Canvas ---
        self.figure = Figure(figsize=(5, 4), dpi=100, facecolor="whitesmoke")
        self.canvas = FigureCanvas(self.figure)

        # --- Create a Toolbar ---
        self.toolbar = NavigationToolbar(self.canvas, self)

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def add_custom_toolbar(self, widget: QWidget):
        """
        Add a custom widget to the toolbar area.

        Args:
            widget: The QWidget to add to the toolbar layout
        """
        # Find the toolbar layout (it's the second item in the main layout, index 1)
        # Layout structure: [Canvas, Toolbar_Layout]
        if self.layout().count() >= 2:
            # The toolbar is usually in a VBox or HBox.
            # Based on __init__, we have self.layout() as QVBoxLayout.
            # It contains self.canvas and self.toolbar.
            # We want to add the custom widget next to the toolbar or above/below it.

            # Let's add it to the main layout, right after the toolbar
            self.layout().addWidget(widget)
        else:
            # Fallback
            self.layout().addWidget(widget)

    def resizeEvent(self, event):
        """
        Override resize event to reapply tight_layout dynamically.

        ✅ FIX #4 (P0): Automatic layout recalculation on window resize
        Implements recommendation from all 6 AI analyses.
        """
        from PySide6.QtGui import QResizeEvent

        super().resizeEvent(event)
        try:
            self.figure.tight_layout(pad=1.2)
            self.canvas.draw_idle()  # Non-blocking draw
        except Exception:
            # Silent fail on resize - layout incompatibility is acceptable here
            logger.debug("Matplotlib tight_layout/draw_idle failed during resize: %s", e)


    def create_subplot_layout(self, num_plots: int, max_cols: int = 2, max_rows: int = None) -> Tuple[int, int]:
        """
        Calculate optimal subplot grid layout (rows, cols) for given number of plots.
        
        This method provides a robust, dynamic system for determining subplot arrangement
        that works for both single and multiple graph displays.
        
        Args:
            num_plots: Number of plots/graphs to display
            max_cols: Maximum number of columns (default: 2 for side-by-side layout)
            max_rows: Maximum number of rows (default: None for automatic calculation)
        
        Returns:
            Tuple[int, int]: (num_rows, num_cols) for subplot grid
        
        Examples:
            >>> create_subplot_layout(1)  # Single plot
            (1, 1)
            
            >>> create_subplot_layout(2)  # Two plots side-by-side
            (1, 2)
            
            >>> create_subplot_layout(3)  # Three plots: 2 cols x 2 rows (last row has 1 plot)
            (2, 2)
            
            >>> create_subplot_layout(4)  # Four plots: 2x2 grid
            (2, 2)
            
            >>> create_subplot_layout(6)  # Six plots: 3x2 grid
            (3, 2)
        """
        if num_plots <= 0:
            return (1, 1)
        
        if num_plots == 1:
            return (1, 1)
        
        # Calculate number of columns (capped by max_cols)
        num_cols = min(num_plots, max_cols)
        
        # Calculate required rows
        num_rows = int(np.ceil(num_plots / num_cols))
        
        # Apply max_rows constraint if specified
        if max_rows is not None and num_rows > max_rows:
            num_rows = max_rows
            # Recalculate cols to fit within max_rows constraint
            num_cols = int(np.ceil(num_plots / num_rows))
        
        return (num_rows, num_cols)

    def is_last_row(self, plot_index: int, total_plots: int, num_rows: int, num_cols: int) -> bool:
        """
        Determine if a subplot is in the last row of the grid.
        
        This is critical for showing x-axis labels only on bottom row plots
        when displaying multiple graphs with shared x-axis ranges.
        
        Args:
            plot_index: 0-based index of current plot
            total_plots: Total number of plots in grid
            num_rows: Number of rows in grid
            num_cols: Number of columns in grid
        
        Returns:
            bool: True if plot is in last row (should show x-axis), False otherwise
        
        Examples:
            >>> # For 4 plots in 2x2 grid:
            >>> is_last_row(0, 4, 2, 2)  # Top-left: False
            False
            >>> is_last_row(2, 4, 2, 2)  # Bottom-left: True
            True
            >>> is_last_row(3, 4, 2, 2)  # Bottom-right: True
            True
        """
        # Calculate which row this plot is in (0-based)
        current_row = plot_index // num_cols
        
        # Last row is (num_rows - 1)
        last_row = num_rows - 1
        
        return current_row == last_row

    def calculate_dynamic_title_size(self, num_plots: int, base_size: int = 14) -> int:
        """
        Calculate appropriate title font size based on number of plots.
        
        More plots = smaller titles to avoid overcrowding.
        
        Args:
            num_plots: Number of plots to display
            base_size: Base font size for single plot (default: 14)
        
        Returns:
            int: Recommended title font size
        
        Examples:
            >>> calculate_dynamic_title_size(1)  # Single plot
            14
            >>> calculate_dynamic_title_size(2)  # Two plots
            12
            >>> calculate_dynamic_title_size(4)  # Four plots
            11
            >>> calculate_dynamic_title_size(6)  # Six or more
            10
        """
        if num_plots == 1:
            return base_size
        elif num_plots == 2:
            return base_size - 2  # 12pt
        elif num_plots <= 4:
            return base_size - 3  # 11pt
        else:
            return base_size - 4  # 10pt for 5+ plots
        except Exception as e:
            # Best-effort: layout can fail transiently while the figure is being rebuilt.
            logger.debug("Matplotlib tight_layout/draw_idle failed during resize: %s", e)

    def update_plot(self, new_figure: Figure):
        """
        Clears the current figure and replaces it with a new one.
        """
        self.figure.clear()
        # This is a way to "copy" the contents of the new figure
        # to the existing figure managed by the canvas.
        axes_list = new_figure.get_axes()

        if not axes_list:
            # No axes to copy
            self.canvas.draw()
            return

        # ✅ FIX: Preserve complex subplot layouts (GridSpec, dendrograms with heatmaps)
        # Instead of creating simple 1xN subplots, preserve the original axes positions
        for i, ax in enumerate(axes_list):
            # Get the original axes position in figure coordinates
            try:
                # Get the position of the original axes (left, bottom, width, height)
                pos = ax.get_position()

                # Create new axes at the same position
                new_ax = self.figure.add_axes([pos.x0, pos.y0, pos.width, pos.height])
                logger.debug(
                    "Created axis %s at position: (%.2f, %.2f, %.2f, %.2f)",
                    i,
                    pos.x0,
                    pos.y0,
                    pos.width,
                    pos.height,
                )
            except Exception as e:
                # Fallback to simple layout
                logger.debug("Failed to preserve axes position; using fallback: %s", e)
                if len(axes_list) == 1:
                    new_ax = self.figure.add_subplot(111)
                else:
                    new_ax = self.figure.add_subplot(len(axes_list), 1, i + 1)

            # Copy all line plots from the original axes
            for line in ax.get_lines():
                new_ax.plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    label=line.get_label(),
                    color=line.get_color(),
                    linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth(),
                    marker=line.get_marker(),
                    markersize=line.get_markersize(),
                )

            # ✅ FIX: Copy AxesImage objects (imshow/heatmaps) - CRITICAL FOR HEATMAPS
            # This was missing and caused blank heatmaps in Correlation Heatmap and Spectral Heatmap
            from matplotlib.image import AxesImage

            images = ax.get_images()
            if images:
                logger.debug(
                    "Found %s AxesImage objects (heatmaps/imshow)",
                    len(images),
                )
                for img in images:
                    try:
                        # Get image data and properties
                        img_data = img.get_array()
                        extent = img.get_extent()
                        cmap = img.get_cmap()
                        alpha = img.get_alpha()
                        interpolation = img.get_interpolation()

                        # Get clim (color limits)
                        clim = img.get_clim()

                        # Recreate imshow on new axis
                        new_img = new_ax.imshow(
                            img_data,
                            extent=extent,
                            cmap=cmap,
                            alpha=alpha if alpha is not None else 1.0,
                            interpolation=interpolation,
                            aspect="auto",
                            vmin=clim[0],
                            vmax=clim[1],
                        )

                        # Copy colorbar if present
                        # Note: Colorbars are handled separately after all axes are copied
                        logger.debug(
                            "Successfully copied AxesImage: shape=%s, cmap=%s, clim=%s",
                            getattr(img_data, "shape", None),
                            getattr(cmap, "name", None),
                            clim,
                        )
                    except Exception as e:
                        logger.debug("Failed to copy AxesImage: %s", e)

            # Copy scatter plots (PathCollections) and LineCollections from the original axes
            from matplotlib.collections import LineCollection, PathCollection

            for collection in ax.collections:
                # Handle LineCollection (used in dendrograms, heatmaps, cluster plots)
                if isinstance(collection, LineCollection):
                    logger.debug("Copying LineCollection (dendrogram/cluster lines)")
                    # Copy line segments directly
                    segments = collection.get_segments()
                    colors = collection.get_colors()
                    linewidths = collection.get_linewidths()
                    linestyles = collection.get_linestyles()

                    new_collection = LineCollection(
                        segments,
                        colors=colors,
                        linewidths=linewidths,
                        linestyles=linestyles,
                    )
                    new_ax.add_collection(new_collection)
                    continue

                # Handle PathCollection (scatter plots)
                if isinstance(collection, PathCollection):
                    offsets = collection.get_offsets()
                    if len(offsets) > 0:
                        # Get collection properties
                        facecolors = collection.get_facecolors()
                        edgecolors = collection.get_edgecolors()
                        sizes = (
                            collection.get_sizes()
                            if hasattr(collection, "get_sizes")
                            else [50]
                        )
                        label = collection.get_label()

                        # Create scatter plot
                        new_ax.scatter(
                            offsets[:, 0],
                            offsets[:, 1],
                            c=facecolors if len(facecolors) > 0 else None,
                            s=sizes[0] if len(sizes) > 0 else 50,
                            edgecolors=edgecolors if len(edgecolors) > 0 else None,
                            label=(
                                label if label and not label.startswith("_") else None
                            ),
                            alpha=collection.get_alpha() or 1.0,
                        )

            # Recreate patches (ellipses, rectangles, arrows) on new axis
            # Patches can't be transferred between figures (RuntimeError), so we recreate them
            # Skip if too many patches (likely heatmap/correlation plot with many cells)
            num_patches = len(ax.patches)
            print(f"[DEBUG] Found {num_patches} patches on axis")

            if num_patches > 100:
                print(
                    f"[DEBUG] Too many patches ({num_patches}), skipping recreation (likely heatmap)"
                )
                print(
                    f"[DEBUG] Heatmap patches are handled by matplotlib's internal rendering"
                )
            else:
                print(f"[DEBUG] Recreating {num_patches} patches on new axis")

                from matplotlib.patches import (
                    Ellipse,
                    Rectangle,
                    Polygon,
                    FancyArrow,
                    FancyArrowPatch,
                )

                for patch in ax.patches:
                    # Get patch properties
                    if isinstance(patch, Ellipse):
                        # ✅ FIX #3 (P0): DUAL-LAYER ELLIPSE PATTERN
                        # Consensus from 6 AI analyses: α=0.08 fill + α=0.85 edge prevents dark overlaps
                        # This preserves the dual-layer pattern created in add_confidence_ellipse()

                        original_alpha = patch.get_alpha()
                        original_facecolor = patch.get_facecolor()
                        original_edgecolor = patch.get_edgecolor()

                        # Check if this is a fill-only layer (very low alpha, no edge)
                        is_fill_layer = (
                            original_alpha is not None
                            and original_alpha <= 0.15
                            and (
                                original_edgecolor is None
                                or (
                                    hasattr(original_edgecolor, "__len__")
                                    and len(original_edgecolor) == 4
                                    and original_edgecolor[3] == 0
                                )
                                or str(original_edgecolor) == "none"
                            )
                        )

                        # Check if this is an edge-only layer (high alpha, no fill)
                        is_edge_layer = (
                            original_alpha is not None
                            and original_alpha >= 0.7
                            and (
                                original_facecolor is None
                                or (
                                    hasattr(original_facecolor, "__len__")
                                    and len(original_facecolor) == 4
                                    and original_facecolor[3] == 0
                                )
                                or str(original_facecolor) == "none"
                            )
                        )

                        # Recreate ellipse preserving dual-layer properties
                        new_ellipse = Ellipse(
                            xy=patch.center,
                            width=patch.width,
                            height=patch.height,
                            angle=patch.angle,
                            facecolor=original_facecolor,
                            edgecolor=original_edgecolor,
                            linestyle=patch.get_linestyle(),
                            linewidth=(
                                patch.get_linewidth() if patch.get_linewidth() else 2.5
                            ),
                            alpha=(
                                original_alpha if original_alpha is not None else 0.08
                            ),
                            label=(
                                patch.get_label()
                                if not patch.get_label().startswith("_")
                                else None
                            ),
                            zorder=(
                                patch.get_zorder()
                                if hasattr(patch, "get_zorder")
                                else 10
                            ),
                        )
                        new_ax.add_patch(new_ellipse)

                        layer_type = (
                            "fill"
                            if is_fill_layer
                            else ("edge" if is_edge_layer else "standard")
                        )
                        print(
                            f"[DEBUG] Recreated ellipse ({layer_type} layer) at {patch.center}, α={original_alpha}"
                        )

                    elif isinstance(patch, Rectangle):
                        # Recreate rectangle (for bar plots)
                        new_rect = Rectangle(
                            xy=(patch.get_x(), patch.get_y()),
                            width=patch.get_width(),
                            height=patch.get_height(),
                            facecolor=patch.get_facecolor(),
                            edgecolor=patch.get_edgecolor(),
                            linewidth=patch.get_linewidth(),
                            alpha=patch.get_alpha(),
                        )
                        new_ax.add_patch(new_rect)
                        print(
                            f"[DEBUG] Recreated rectangle at ({patch.get_x()}, {patch.get_y()}) on new axis"
                        )

                    elif isinstance(patch, FancyArrow):
                        #
                        # Recreate FancyArrow (used in Biplots)
                        print(f"[DEBUG] Recreating FancyArrow on new axis")

                        # FancyArrow stores properties as attributes, not via get_ methods
                        new_arrow = FancyArrow(
                            x=patch._x,  # Changed from get_x()
                            y=patch._y,  # Changed from get_y()
                            dx=patch._dx,  # Changed from get_width()
                            dy=patch._dy,  # Changed from get_height()
                            width=getattr(
                                patch, "_width", 0.01
                            ),  # Keep using internal attribs if getters miss
                            head_width=getattr(patch, "_head_width", 0.03),
                            head_length=getattr(patch, "_head_length", 0.05),
                            length_includes_head=getattr(
                                patch, "_length_includes_head", False
                            ),
                            shape=getattr(patch, "_shape", "full"),
                            overhang=getattr(patch, "_overhang", 0),
                            head_starts_at_zero=getattr(
                                patch, "_head_starts_at_zero", False
                            ),
                            facecolor=patch.get_facecolor(),
                            edgecolor=patch.get_edgecolor(),
                            linewidth=patch.get_linewidth(),
                            alpha=patch.get_alpha(),
                        )
                        new_ax.add_patch(new_arrow)
                        print(f"[DEBUG] Successfully recreated FancyArrow")

                    elif isinstance(patch, FancyArrowPatch):
                        # Recreate FancyArrowPatch (more common arrow type)
                        print(f"[DEBUG] Recreating FancyArrowPatch on new axis")
                        posA = patch.get_path().vertices[0]
                        posB = patch.get_path().vertices[-1]
                        new_arrow_patch = FancyArrowPatch(
                            posA=posA,
                            posB=posB,
                            arrowstyle=patch.get_arrowstyle(),
                            mutation_scale=patch.get_mutation_scale(),
                            facecolor=patch.get_facecolor(),
                            edgecolor=patch.get_edgecolor(),
                            linewidth=patch.get_linewidth(),
                            alpha=patch.get_alpha(),
                        )
                        new_ax.add_patch(new_arrow_patch)
                        print(f"[DEBUG] Successfully recreated FancyArrowPatch")

                    else:
                        # For other patch types, log and skip
                        print(
                            f"[DEBUG] Skipping unsupported patch type: {type(patch).__name__}"
                        )

            # Copy annotations (text with arrows) - CRITICAL FOR PEAK LABELS
            annotations = [
                artist
                for artist in ax.get_children()
                if hasattr(artist, "arrow_patch")
                or (
                    hasattr(artist, "__class__")
                    and artist.__class__.__name__ == "Annotation"
                )
            ]
            num_annotations = len(annotations)
            print(f"[DEBUG] Found {num_annotations} annotations on axis")

            if num_annotations > 0:
                print(f"[DEBUG] Copying {num_annotations} annotations to new axis")
                for artist in annotations:
                    try:
                        # Get annotation properties
                        text = artist.get_text()
                        xy = artist.xy  # Point being annotated
                        xytext = artist.xyann  # Text position (tuple)

                        # Get text properties
                        fontsize = artist.get_fontsize()
                        fontweight = artist.get_fontweight()
                        color = artist.get_color()
                        ha = artist.get_ha()
                        va = artist.get_va()

                        # Get bbox properties
                        bbox = artist.get_bbox_patch()
                        bbox_props = None
                        if bbox:
                            bbox_props = dict(
                                boxstyle=bbox.get_boxstyle(),
                                facecolor=bbox.get_facecolor(),
                                edgecolor=bbox.get_edgecolor(),
                                alpha=bbox.get_alpha(),
                            )

                        # Get arrow properties
                        arrow_patch = artist.arrow_patch
                        arrowprops = None
                        if arrow_patch:
                            arrowprops = dict(
                                arrowstyle=getattr(arrow_patch, "arrowstyle", "->"),
                                connectionstyle=getattr(
                                    arrow_patch, "connectionstyle", "arc3,rad=0"
                                ),
                                color=(
                                    arrow_patch.get_edgecolor()[0:3]
                                    if hasattr(arrow_patch, "get_edgecolor")
                                    else "red"
                                ),
                                lw=(
                                    arrow_patch.get_linewidth()
                                    if hasattr(arrow_patch, "get_linewidth")
                                    else 1
                                ),
                            )

                        # Create new annotation on new axis
                        new_ax.annotate(
                            text,
                            xy=xy,
                            xytext=xytext,
                            textcoords="offset points",
                            fontsize=fontsize,
                            fontweight=fontweight,
                            color=color,
                            ha=ha,
                            va=va,
                            bbox=bbox_props,
                            arrowprops=arrowprops,
                            zorder=10,
                        )
                        logger.debug("Copied annotation: '%s...' at %s", text[:20], xy)
                    except Exception as e:
                        logger.debug("Failed to copy annotation: %s", e)

            # Copy axes properties
            new_ax.set_title(ax.get_title())
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())
            new_ax.set_xlim(ax.get_xlim())
            new_ax.set_ylim(ax.get_ylim())

            # ✅ FIX: Add colorbar for heatmaps/imshow if images were copied
            # This ensures correlation heatmaps and spectral heatmaps have proper colorbars
            if images:
                # Get the last image to use for colorbar
                last_img = new_ax.get_images()
                if last_img:
                    try:
                        cbar = self.figure.colorbar(last_img[-1], ax=new_ax)
                        # Try to copy colorbar label from original
                        original_cbar_axes = [
                            child
                            for child in new_figure.get_axes()
                            if hasattr(child, "_colorbar")
                        ]
                        logger.debug("Added colorbar for heatmap")
                    except Exception as e:
                        logger.debug("Failed to add colorbar: %s", e)

            # Copy legend if it exists and has valid artists
            legend = ax.get_legend()
            if legend and legend.get_texts():
                # Check if there are any labeled artists
                handles, labels = ax.get_legend_handles_labels()
                if handles and labels:
                    new_ax.legend(
                        handles,
                        labels,
                        loc=legend._loc if hasattr(legend, "_loc") else "best",
                    )

            # Add grid (but not for heatmaps - they look bad with grid)
            if not images:
                new_ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # ✅ FIX #1 (P0): Apply tight_layout BEFORE draw (not after)
        # Consensus from 6 AI analyses: tight_layout must be called after ALL artists added
        try:
            self.figure.tight_layout(pad=1.2, rect=[0, 0.03, 1, 0.95])
        except (ValueError, UserWarning, RuntimeWarning) as e:
            # Known incompatibility with certain subplot types (e.g., colorbar axes)
            # Try constrained_layout as fallback
        except Exception as e:
            logger.debug("tight_layout failed (%s); using constrained_layout", e)
            try:
                self.figure.set_constrained_layout(True)
            except Exception:
                # If both fail, continue without layout adjustment
                pass
        except Exception as e:
            # Catch any other unexpected errors
            print(f"[DEBUG] Layout adjustment failed: {type(e).__name__}")
            pass
        
            except Exception as e2:
                logger.debug("set_constrained_layout(True) failed: %s", e2)

        self.canvas.draw()

        # ✅ FIX #2 (P0): Close source figure to prevent memory leak
        # Validated by matplotlib docs: prevents memory growth in repeated analysis
        plt.close(new_figure)

    def update_plot_with_config(
        self, new_figure: Figure, config: Optional[Dict[str, Any]] = None
    ):
        """
        Enhanced update_plot with robust configuration options.

        Args:
            new_figure: Matplotlib Figure to display
            config: Optional configuration dictionary with keys:
                - subplot_spacing: tuple (hspace, wspace) for spacing between subplots
                - grid: dict with {enabled: bool, alpha: float, linestyle: str, linewidth: float}
                - legend: dict with {loc: str, fontsize: int, framealpha: float}
                - title: dict with {fontsize: int, fontweight: str, pad: float}
                - axes: dict with {xlabel_fontsize: int, ylabel_fontsize: int, tick_labelsize: int}
                - figure: dict with {tight_layout: bool, constrained_layout: bool}
        """
        if config is None:
            config = {}

        # Clear and copy figure (same as update_plot)
        self.figure.clear()
        axes_list = new_figure.get_axes()

        if not axes_list:
            self.canvas.draw()
            return

        # Apply subplot spacing if specified
        if "subplot_spacing" in config:
            hspace, wspace = config["subplot_spacing"]
            self.figure.subplots_adjust(hspace=hspace, wspace=wspace)

        # Copy axes with enhanced configuration
        for i, ax in enumerate(axes_list):
            # Determine subplot layout
            if len(axes_list) == 1:
                new_ax = self.figure.add_subplot(111)
            else:
                # Calculate grid layout (prefer square-ish layouts)
                n_plots = len(axes_list)
                n_cols = int(np.ceil(np.sqrt(n_plots)))
                n_rows = int(np.ceil(n_plots / n_cols))
                new_ax = self.figure.add_subplot(n_rows, n_cols, i + 1)

            # Copy plot elements (lines, collections, patches, images)
            has_images = self._copy_plot_elements(ax, new_ax)

            # Copy axes properties
            new_ax.set_title(ax.get_title())
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())
            new_ax.set_xlim(ax.get_xlim())
            new_ax.set_ylim(ax.get_ylim())

            # ✅ FIX: Add colorbar for heatmaps if images were copied
            if has_images:
                img_list = new_ax.get_images()
                if img_list:
                    try:
                        self.figure.colorbar(img_list[-1], ax=new_ax)
                        logger.debug("Added colorbar in update_plot_with_config")
                    except Exception as e:
                        logger.debug("Failed to add colorbar: %s", e)

            # Apply grid configuration (skip for heatmaps)
            if not has_images:
                if "grid" in config:
                    grid_cfg = config["grid"]
                    new_ax.grid(
                        grid_cfg.get("enabled", True),
                        which="both",
                        linestyle=grid_cfg.get("linestyle", "--"),
                        linewidth=grid_cfg.get("linewidth", 0.5),
                        alpha=grid_cfg.get("alpha", 0.3),
                    )
                else:
                    new_ax.grid(True, which="both", linestyle="--", linewidth=0.5)

            # Apply legend configuration
            legend = ax.get_legend()
            if legend and legend.get_texts():
                handles, labels = ax.get_legend_handles_labels()
                if handles and labels:
                    if "legend" in config:
                        leg_cfg = config["legend"]
                        new_ax.legend(
                            handles,
                            labels,
                            loc=leg_cfg.get("loc", "best"),
                            fontsize=leg_cfg.get("fontsize", 9),
                            framealpha=leg_cfg.get("framealpha", 0.8),
                        )
                    else:
                        new_ax.legend(handles, labels, loc="best")

            # Apply title configuration
            if "title" in config and ax.get_title():
                title_cfg = config["title"]
                new_ax.set_title(
                    ax.get_title(),
                    fontsize=title_cfg.get("fontsize", 12),
                    fontweight=title_cfg.get("fontweight", "bold"),
                    pad=title_cfg.get("pad", 10),
                )

            # Apply axes configuration
            if "axes" in config:
                axes_cfg = config["axes"]
                new_ax.set_xlabel(
                    ax.get_xlabel(), fontsize=axes_cfg.get("xlabel_fontsize", 11)
                )
                new_ax.set_ylabel(
                    ax.get_ylabel(), fontsize=axes_cfg.get("ylabel_fontsize", 11)
                )
                new_ax.tick_params(
                    axis="both", labelsize=axes_cfg.get("tick_labelsize", 9)
                )

        # Apply figure-level configuration
        if "figure" in config:
            fig_cfg = config["figure"]
            if fig_cfg.get("constrained_layout", False):
                self.figure.set_constrained_layout(True)
            elif fig_cfg.get("tight_layout", True):
                try:
                    self.figure.tight_layout(pad=1.2, rect=[0, 0.03, 1, 0.95])
                except Exception as e:
                    logger.debug("tight_layout failed (%s); using constrained_layout", e)
                    self.figure.set_constrained_layout(True)
        else:
            # ✅ DEFAULT: Always apply tight_layout when no config provided
            try:
                self.figure.tight_layout(pad=1.2, rect=[0, 0.03, 1, 0.95])
            except Exception as e:
                logger.debug("tight_layout failed (%s); using constrained_layout", e)
                try:
                    self.figure.set_constrained_layout(True)
                except Exception as e2:
                    logger.debug("set_constrained_layout(True) failed: %s", e2)

        self.canvas.draw()

        # ✅ CRITICAL: Close source figure to prevent memory leak
        plt.close(new_figure)

    def _copy_plot_elements(self, source_ax, target_ax):
        """
        Helper method to copy plot elements from source to target axis.
        Handles lines, collections (scatter, line collections), patches, and images.
        """
        # Copy lines
        for line in source_ax.get_lines():
            target_ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                label=line.get_label(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                marker=line.get_marker(),
                markersize=line.get_markersize(),
            )

        # ✅ FIX: Copy AxesImage objects (imshow/heatmaps) - CRITICAL FOR HEATMAPS
        from matplotlib.image import AxesImage

        images = source_ax.get_images()
        has_images = False
        if images:
            has_images = True
            logger.debug("_copy_plot_elements: Found %s AxesImage objects", len(images))
            for img in images:
                try:
                    img_data = img.get_array()
                    extent = img.get_extent()
                    cmap = img.get_cmap()
                    alpha = img.get_alpha()
                    interpolation = img.get_interpolation()
                    clim = img.get_clim()

                    new_img = target_ax.imshow(
                        img_data,
                        extent=extent,
                        cmap=cmap,
                        alpha=alpha if alpha is not None else 1.0,
                        interpolation=interpolation,
                        aspect="auto",
                        vmin=clim[0],
                        vmax=clim[1],
                    )
                    logger.debug(
                        "Copied AxesImage in helper: shape=%s",
                        getattr(img_data, "shape", None),
                    )
                except Exception as e:
                    logger.debug("Failed to copy AxesImage in helper: %s", e)

        # Copy collections
        from matplotlib.collections import LineCollection, PathCollection

        for collection in source_ax.collections:
            if isinstance(collection, LineCollection):
                segments = collection.get_segments()
                colors = collection.get_colors()
                linewidths = collection.get_linewidths()
                linestyles = collection.get_linestyles()
                new_collection = LineCollection(
                    segments,
                    colors=colors,
                    linewidths=linewidths,
                    linestyles=linestyles,
                )
                target_ax.add_collection(new_collection)
            elif isinstance(collection, PathCollection):
                offsets = collection.get_offsets()
                if len(offsets) > 0:
                    target_ax.scatter(
                        offsets[:, 0],
                        offsets[:, 1],
                        c=collection.get_facecolors(),
                        s=(
                            collection.get_sizes()[0]
                            if hasattr(collection, "get_sizes")
                            and len(collection.get_sizes()) > 0
                            else 50
                        ),
                        edgecolors=collection.get_edgecolors(),
                        label=(
                            collection.get_label()
                            if not collection.get_label().startswith("_")
                            else None
                        ),
                        alpha=collection.get_alpha() or 1.0,
                    )

        # Copy patches (if not too many)
        if len(source_ax.patches) <= 100:
            from matplotlib.patches import (
                Ellipse,
                Rectangle,
                FancyArrow,
                FancyArrowPatch,
            )

            for patch in source_ax.patches:
                if isinstance(patch, Ellipse):
                    # ✅ FIX #3 (P0): DUAL-LAYER ELLIPSE PATTERN in helper method
                    # Preserve alpha and layer type from source ellipse
                    original_alpha = patch.get_alpha()
                    new_patch = Ellipse(
                        xy=patch.center,
                        width=patch.width,
                        height=patch.height,
                        angle=patch.angle,
                        facecolor=patch.get_facecolor(),
                        edgecolor=patch.get_edgecolor(),
                        linestyle=patch.get_linestyle(),
                        linewidth=(
                            patch.get_linewidth() if patch.get_linewidth() else 2.5
                        ),
                        alpha=original_alpha if original_alpha is not None else 0.08,
                        label=(
                            patch.get_label()
                            if hasattr(patch, "get_label")
                            and not patch.get_label().startswith("_")
                            else None
                        ),
                        zorder=(
                            patch.get_zorder() if hasattr(patch, "get_zorder") else 10
                        ),
                    )
                    target_ax.add_patch(new_patch)
                elif isinstance(patch, Rectangle):
                    new_patch = Rectangle(
                        xy=(patch.get_x(), patch.get_y()),
                        width=patch.get_width(),
                        height=patch.get_height(),
                        facecolor=patch.get_facecolor(),
                        edgecolor=patch.get_edgecolor(),
                        linewidth=patch.get_linewidth(),
                        alpha=patch.get_alpha(),
                    )
                    target_ax.add_patch(new_patch)
                elif isinstance(patch, FancyArrow):
                    new_patch = FancyArrow(
                        x=patch.get_x(),
                        y=patch.get_y(),
                        dx=patch.get_width(),
                        dy=patch.get_height(),
                        width=getattr(patch, "_width", 0.01),
                        head_width=getattr(patch, "_head_width", 0.03),
                        head_length=getattr(patch, "_head_length", 0.05),
                        facecolor=patch.get_facecolor(),
                        edgecolor=patch.get_edgecolor(),
                        linewidth=patch.get_linewidth(),
                        alpha=patch.get_alpha(),
                    )
                    target_ax.add_patch(new_patch)
                elif isinstance(patch, FancyArrowPatch):
                    posA = patch.get_path().vertices[0]
                    posB = patch.get_path().vertices[-1]
                    new_patch = FancyArrowPatch(
                        posA=posA,
                        posB=posB,
                        arrowstyle=patch.get_arrowstyle(),
                        mutation_scale=patch.get_mutation_scale(),
                        facecolor=patch.get_facecolor(),
                        edgecolor=patch.get_edgecolor(),
                        linewidth=patch.get_linewidth(),
                        alpha=patch.get_alpha(),
                    )
                    target_ax.add_patch(new_patch)

        # Return whether images were copied (for colorbar handling)
        return has_images

    def plot_3d(
        self,
        data: Dict[str, np.ndarray],
        plot_type: str = "scatter",
        title: str = "3D Visualization",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Create 3D plot for dimensionality reduction or other 3D data.

        Args:
            data: Dictionary with keys:
                - 'x', 'y', 'z': Coordinate arrays
                - 'labels' (optional): Labels for coloring
                - 'colors' (optional): Color array
            plot_type: Type of 3D plot ('scatter', 'surface', 'wireframe')
            title: Plot title
            config: Optional configuration for axes, grid, etc.
        """
        if config is None:
            config = {}

        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")

        x = data.get("x", np.array([]))
        y = data.get("y", np.array([]))
        z = data.get("z", np.array([]))

        if len(x) == 0 or len(y) == 0 or len(z) == 0:
            ax.text2D(
                0.5,
                0.5,
                "No 3D data to display",
                ha="center",
                va="center",
                fontsize=14,
                color="gray",
                transform=ax.transAxes,
            )
            self.canvas.draw()
            return

        if plot_type == "scatter":
            # Scatter plot
            colors = data.get("colors", None)
            labels = data.get("labels", None)

            if colors is not None:
                scatter = ax.scatter(
                    x,
                    y,
                    z,
                    c=colors,
                    cmap="viridis",
                    s=config.get("marker_size", 50),
                    alpha=config.get("alpha", 0.6),
                )
                self.figure.colorbar(scatter, ax=ax, label="Value")
            elif labels is not None:
                # Color by labels
                unique_labels = np.unique(labels)
                colors_map = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(
                        x[mask],
                        y[mask],
                        z[mask],
                        c=[colors_map[i]],
                        label=f"Class {label}",
                        s=config.get("marker_size", 50),
                        alpha=config.get("alpha", 0.6),
                    )
                ax.legend()
            else:
                ax.scatter(
                    x,
                    y,
                    z,
                    c="blue",
                    s=config.get("marker_size", 50),
                    alpha=config.get("alpha", 0.6),
                )

        elif plot_type == "surface":
            # Surface plot (requires gridded data)
            if len(x.shape) == 1:
                # Create meshgrid if 1D arrays provided
                X, Y = np.meshgrid(x, y)
                Z = z.reshape(len(y), len(x))
            else:
                X, Y, Z = x, y, z

            ax.plot_surface(
                X,
                Y,
                Z,
                cmap="viridis",
                alpha=config.get("alpha", 0.8),
                antialiased=True,
            )

        elif plot_type == "wireframe":
            # Wireframe plot
            if len(x.shape) == 1:
                X, Y = np.meshgrid(x, y)
                Z = z.reshape(len(y), len(x))
            else:
                X, Y, Z = x, y, z

            ax.plot_wireframe(
                X,
                Y,
                Z,
                color=config.get("color", "blue"),
                alpha=config.get("alpha", 0.6),
            )

        # Set labels and title
        ax.set_xlabel(config.get("xlabel", "X"), fontsize=11)
        ax.set_ylabel(config.get("ylabel", "Y"), fontsize=11)
        ax.set_zlabel(config.get("zlabel", "Z"), fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")

        # Set viewing angle
        ax.view_init(elev=config.get("elev", 20), azim=config.get("azim", 45))

        # Grid
        ax.grid(config.get("grid", True), alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_dendrogram(
        self,
        dendrogram_data: Dict[str, Any],
        title: str = "Hierarchical Clustering Dendrogram",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Create dendrogram plot for hierarchical clustering.

        Args:
            dendrogram_data: Dictionary returned by scipy.cluster.hierarchy.dendrogram
                            or linkage matrix to compute dendrogram from
            title: Plot title
            config: Optional configuration
        """
        if config is None:
            config = {}

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Check if we have linkage matrix or dendrogram data
        from scipy.cluster.hierarchy import dendrogram, linkage

        if isinstance(dendrogram_data, dict) and "icoord" in dendrogram_data:
            # Already computed dendrogram data - plot directly
            # Manually recreate dendrogram from data
            icoord = dendrogram_data["icoord"]
            dcoord = dendrogram_data["dcoord"]
            colors = dendrogram_data.get("color_list", ["C0"] * len(icoord))

            for i, (xi, yi) in enumerate(zip(icoord, dcoord)):
                ax.plot(xi, yi, color=colors[i], linewidth=config.get("linewidth", 1.5))

        elif isinstance(dendrogram_data, np.ndarray):
            # Linkage matrix - compute and plot dendrogram
            dend = dendrogram(
                dendrogram_data,
                ax=ax,
                orientation=config.get("orientation", "top"),
                color_threshold=config.get("color_threshold", None),
                above_threshold_color=config.get("above_threshold_color", "grey"),
                leaf_font_size=config.get("leaf_font_size", 8),
            )

        else:
            ax.text(
                0.5,
                0.5,
                "Invalid dendrogram data",
                ha="center",
                va="center",
                fontsize=14,
                color="gray",
                transform=ax.transAxes,
            )
            self.canvas.draw()
            return

        # Set labels and title
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(config.get("xlabel", "Sample Index"), fontsize=11)
        ax.set_ylabel(config.get("ylabel", "Distance"), fontsize=11)
        ax.grid(config.get("grid", True), axis="y", alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def clear_plot(self):
        """Clears the plot area."""
        self.figure.clear()
        self.canvas.draw()

    def plot_spectra(
        self,
        data,
        title="Spectra",
        auto_focus=False,
        focus_padding=None,
        crop_bounds=None,
    ):
        """Plot spectra data directly."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if data is None:
            ax.text(
                0.5,
                0.5,
                "No data to display",
                ha="center",
                va="center",
                fontsize=14,
                color="gray",
                transform=ax.transAxes,
            )
            self.canvas.draw()
            return

        # Handle different data types
        if hasattr(data, "columns"):
            # DataFrame - handle this first before checking for shape
            num_spectra = min(data.shape[1], 10)
            for i, column in enumerate(data.columns[:num_spectra]):
                ax.plot(data.index, data[column], label=column)
        elif hasattr(data, "shape") and len(data.shape) == 2:
            # Numpy array or similar
            num_spectra = min(
                data.shape[1] if data.shape[1] < data.shape[0] else data.shape[0], 10
            )
            for i in range(num_spectra):
                spectrum = data[:, i] if data.shape[1] < data.shape[0] else data[i, :]
                ax.plot(spectrum, label=f"Spectrum {i+1}")

        ax.set_title(title)
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Apply auto-focus only if requested and data has wavenumber index
        if auto_focus:
            try:
                if hasattr(data, "index") and hasattr(data, "values"):
                    # DataFrame with wavenumber index
                    wavenumbers = data.index.values
                    intensities = data.values
                    min_wn, max_wn = detect_signal_range(
                        wavenumbers,
                        intensities.T,
                        focus_padding=focus_padding,
                        crop_bounds=crop_bounds,
                    )  # Transpose for proper shape
                    ax.set_xlim(min_wn, max_wn)
            except Exception as e:
                pass  # Silently fall back to full range

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_comparison_spectra(
        self, original_data, processed_data, titles=None, colors=None
    ):
        """Plot comparison between original and processed data."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if titles is None:
            titles = ["Original", "Processed"]
        if colors is None:
            colors = ["lightblue", "darkblue"]

        # Plot original data (sample)
        if original_data is not None:
            num_original = min(
                5,
                (
                    original_data.shape[1]
                    if hasattr(original_data, "shape") and len(original_data.shape) == 2
                    else len(original_data)
                ),
            )
            for i in range(num_original):
                if hasattr(original_data, "shape") and len(original_data.shape) == 2:
                    spectrum = (
                        original_data[:, i]
                        if original_data.shape[1] < original_data.shape[0]
                        else original_data[i, :]
                    )
                    x_data = range(len(spectrum))
                else:
                    spectrum = original_data
                    x_data = range(len(spectrum))

                ax.plot(
                    x_data,
                    spectrum,
                    color=colors[0],
                    alpha=0.6,
                    linewidth=1,
                    label=titles[0] if i == 0 else "",
                )

        # Plot processed data
        if processed_data is not None:
            num_processed = min(
                5,
                (
                    processed_data.shape[1]
                    if hasattr(processed_data, "shape")
                    and len(processed_data.shape) == 2
                    else len(processed_data)
                ),
            )
            for i in range(num_processed):
                if hasattr(processed_data, "shape") and len(processed_data.shape) == 2:
                    spectrum = (
                        processed_data[:, i]
                        if processed_data.shape[1] < processed_data.shape[0]
                        else processed_data[i, :]
                    )
                    x_data = range(len(spectrum))
                else:
                    spectrum = processed_data
                    x_data = range(len(spectrum))

                ax.plot(
                    x_data,
                    spectrum,
                    color=colors[1],
                    alpha=0.8,
                    linewidth=1.5,
                    label=titles[1] if i == 0 else "",
                )

        ax.set_title("Preprocessing Preview")
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_comparison_spectra_with_wavenumbers(
        self,
        original_data,
        processed_data,
        original_wavenumbers,
        processed_wavenumbers,
        titles=None,
        colors=None,
        auto_focus=True,
        focus_padding=None,
        crop_bounds=None,
    ):
        """Plot comparison between original and processed data with proper wavenumber axes."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if titles is None:
            titles = ["Original", "Processed"]
        if colors is None:
            colors = ["lightblue", "darkblue"]

        # Plot original data (sample) with actual wavenumbers
        if original_data is not None and original_wavenumbers is not None:
            num_original = min(
                5,
                (
                    original_data.shape[0]
                    if hasattr(original_data, "shape") and len(original_data.shape) == 2
                    else 1
                ),
            )
            for i in range(num_original):
                if hasattr(original_data, "shape") and len(original_data.shape) == 2:
                    spectrum = original_data[i, :]
                else:
                    spectrum = original_data

                ax.plot(
                    original_wavenumbers,
                    spectrum,
                    color=colors[0],
                    alpha=0.6,
                    linewidth=1,
                    label=titles[0] if i == 0 else "",
                )

        # Plot processed data with actual wavenumbers
        if processed_data is not None and processed_wavenumbers is not None:
            num_processed = min(
                5,
                (
                    processed_data.shape[0]
                    if hasattr(processed_data, "shape")
                    and len(processed_data.shape) == 2
                    else 1
                ),
            )
            for i in range(num_processed):
                if hasattr(processed_data, "shape") and len(processed_data.shape) == 2:
                    spectrum = processed_data[i, :]
                else:
                    spectrum = processed_data

                ax.plot(
                    processed_wavenumbers,
                    spectrum,
                    color=colors[1],
                    alpha=0.8,
                    linewidth=1.5,
                    label=titles[1] if i == 0 else "",
                )

        ax.set_title("Preprocessing Preview")
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Apply auto-focus only if requested
        if auto_focus:
            try:
                # Determine the best range from the available data
                focus_wavenumbers = (
                    processed_wavenumbers
                    if processed_wavenumbers is not None
                    else original_wavenumbers
                )

                if focus_wavenumbers is not None:
                    # Get intensity data for range detection
                    if processed_data is not None:
                        focus_intensities = processed_data
                    elif original_data is not None:
                        focus_intensities = original_data
                    else:
                        focus_intensities = None

                    if focus_intensities is not None:
                        min_wn, max_wn = detect_signal_range(
                            focus_wavenumbers,
                            focus_intensities,
                            focus_padding=focus_padding,
                            crop_bounds=crop_bounds,
                        )
                        ax.set_xlim(min_wn, max_wn)

            except Exception as e:
                pass  # Silently fall back to full range

        self.figure.tight_layout()
        self.canvas.draw()


def plot_spectra(df: pd.DataFrame, title: str = "", auto_focus: bool = False) -> Figure:
    """
    Generates a matplotlib Figure object containing a plot of the spectra.
    Plots a maximum of 10 spectra for clarity and applies themed styling.

    Args:
        df: DataFrame with wavenumber index and intensity columns
        title: Plot title
        auto_focus: Whether to automatically focus on signal regions
    """

    fig = Figure(figsize=(8, 6), dpi=100, facecolor="#eaf2f8")  # Themed background
    ax = fig.add_subplot(111, facecolor="#eaf2f8")  # Themed background

    # --- Robustness Check ---
    if df is None or df.empty:
        ax.text(
            0.5,
            0.5,
            "No data to display.",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig

    # --- Plotting Logic ---
    num_spectra = df.shape[1]
    plot_title = "Loaded Raman Spectra"

    # Limit the number of plotted spectra for clarity
    if num_spectra > 10:
        df_to_plot = df.iloc[:, :10]
        plot_title += f" (showing first 10 of {num_spectra})"
    else:
        df_to_plot = df

    # Plot each spectrum
    for i, column in enumerate(df_to_plot.columns):
        ax.plot(df_to_plot.index, df_to_plot[column], label=column)

    ax.set_title(plot_title, fontsize=14, weight="bold")
    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="#d1dbe5")

    # Customize legend
    legend = ax.legend(facecolor="#ffffff", framealpha=0.7)
    for text in legend.get_texts():
        text.set_color("#34495e")

    # Customize tick colors
    ax.tick_params(axis="x", colors="#34495e", labelsize=10)
    ax.tick_params(axis="y", colors="#34495e", labelsize=10)

    # Customize spine colors
    for spine in ax.spines.values():
        spine.set_edgecolor("#34495e")

    # Apply auto-focus only if requested
    if auto_focus:
        try:
            wavenumbers = df_to_plot.index.values
            intensities = df_to_plot.values
            min_wn, max_wn = detect_signal_range(
                wavenumbers, intensities.T
            )  # Transpose for proper shape
            ax.set_xlim(min_wn, max_wn)
        except Exception as e:
            pass  # Silently fall back to full range

    # Adjust layout with explicit padding to ensure y-axis labels are visible
    fig.tight_layout(pad=1.5)  # Increased padding for better visibility
    fig.subplots_adjust(
        left=0.12, right=0.95, top=0.93, bottom=0.10
    )  # Explicit margins for y-axis labels
    return fig
