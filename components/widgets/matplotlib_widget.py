import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import pandas as pd
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


def detect_signal_range(wavenumbers, intensities, noise_threshold_percentile=20, signal_threshold_factor=1.2, focus_padding=None, crop_bounds=None):
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
        variance_threshold = np.percentile(variance_signal, 70)  # Top 30% variance regions
        high_variance_mask = variance_signal > variance_threshold
        
        # Also look for regions above intensity threshold
        intensity_threshold = np.percentile(mean_intensity, 70)  # Top 30% intensity regions
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
        self.figure = Figure(figsize=(5, 4), dpi=100, facecolor='whitesmoke')
        self.canvas = FigureCanvas(self.figure)
        
        # --- Create a Toolbar ---
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

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
            
        for i, ax in enumerate(axes_list):
            # Create a new subplot in the same position
            # For simple cases, we can use add_subplot(111) for single plots
            if len(axes_list) == 1:
                new_ax = self.figure.add_subplot(111)
            else:
                # For multiple subplots, try to preserve layout
                new_ax = self.figure.add_subplot(len(axes_list), 1, i+1)
            
            # Copy all line plots from the original axes
            for line in ax.get_lines():
                new_ax.plot(line.get_xdata(), line.get_ydata(), 
                           label=line.get_label(), 
                           color=line.get_color(),
                           linestyle=line.get_linestyle(),
                           linewidth=line.get_linewidth(),
                           marker=line.get_marker(),
                           markersize=line.get_markersize())
            
            # Copy scatter plots (PathCollections) from the original axes
            from matplotlib.collections import LineCollection
            
            for collection in ax.collections:
                # Skip LineCollection (used in dendrograms, heatmaps) - they don't have get_sizes()
                if isinstance(collection, LineCollection):
                    print(f"[DEBUG] Skipping LineCollection (no get_sizes method)")
                    continue
                
                offsets = collection.get_offsets()
                if len(offsets) > 0:
                    # Get collection properties
                    facecolors = collection.get_facecolors()
                    edgecolors = collection.get_edgecolors()
                    sizes = collection.get_sizes() if hasattr(collection, 'get_sizes') else [50]
                    label = collection.get_label()
                    
                    # Create scatter plot
                    new_ax.scatter(offsets[:, 0], offsets[:, 1],
                                 c=facecolors if len(facecolors) > 0 else None,
                                 s=sizes[0] if len(sizes) > 0 else 50,
                                 edgecolors=edgecolors if len(edgecolors) > 0 else None,
                                 label=label if label and not label.startswith('_') else None,
                                 alpha=collection.get_alpha() or 1.0)
            
            # Recreate patches (ellipses, rectangles) on new axis
            # Patches can't be transferred between figures (RuntimeError), so we recreate them
            # Skip if too many patches (likely heatmap/correlation plot with many cells)
            num_patches = len(ax.patches)
            print(f"[DEBUG] Found {num_patches} patches on axis")
            
            if num_patches > 100:
                print(f"[DEBUG] Too many patches ({num_patches}), skipping recreation (likely heatmap)")
                print(f"[DEBUG] Heatmap patches are handled by matplotlib's internal rendering")
            else:
                print(f"[DEBUG] Recreating {num_patches} patches on new axis")
                
                from matplotlib.patches import Ellipse, Rectangle, Polygon
                
                for patch in ax.patches:
                    # Get patch properties
                    if isinstance(patch, Ellipse):
                        # Recreate ellipse with same properties
                        new_ellipse = Ellipse(
                            xy=patch.center,
                            width=patch.width,
                            height=patch.height,
                            angle=patch.angle,
                            facecolor=patch.get_facecolor(),
                            edgecolor=patch.get_edgecolor(),
                            linestyle=patch.get_linestyle(),
                            linewidth=patch.get_linewidth(),
                            alpha=patch.get_alpha(),
                            label=patch.get_label() if not patch.get_label().startswith('_') else None
                        )
                        new_ax.add_patch(new_ellipse)
                        print(f"[DEBUG] Recreated ellipse at {patch.center} on new axis")
                    
                    elif isinstance(patch, Rectangle):
                        # Recreate rectangle (for bar plots)
                        new_rect = Rectangle(
                            xy=(patch.get_x(), patch.get_y()),
                            width=patch.get_width(),
                            height=patch.get_height(),
                            facecolor=patch.get_facecolor(),
                            edgecolor=patch.get_edgecolor(),
                            linewidth=patch.get_linewidth(),
                            alpha=patch.get_alpha()
                        )
                        new_ax.add_patch(new_rect)
                        print(f"[DEBUG] Recreated rectangle at ({patch.get_x()}, {patch.get_y()}) on new axis")
                    
                    else:
                        # For other patch types, log and skip
                        print(f"[DEBUG] Skipping unsupported patch type: {type(patch).__name__}")
            
            # Copy axes properties
            new_ax.set_title(ax.get_title())
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())
            new_ax.set_xlim(ax.get_xlim())
            new_ax.set_ylim(ax.get_ylim())
            
            # Copy legend if it exists and has valid artists
            legend = ax.get_legend()
            if legend and legend.get_texts():
                # Check if there are any labeled artists
                handles, labels = ax.get_legend_handles_labels()
                if handles and labels:
                    new_ax.legend(handles, labels, loc=legend._loc if hasattr(legend, '_loc') else 'best')
            
            # Add grid
            new_ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        self.figure.tight_layout()
        self.canvas.draw()

    def clear_plot(self):
        """Clears the plot area."""
        self.figure.clear()
        self.canvas.draw()
    
    def plot_spectra(self, data, title="Spectra", auto_focus=False, focus_padding=None, crop_bounds=None):
        """Plot spectra data directly."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if data is None:
            ax.text(0.5, 0.5, "No data to display", ha='center', va='center', 
                   fontsize=14, color='gray', transform=ax.transAxes)
            self.canvas.draw()
            return
        
        # Handle different data types
        if hasattr(data, 'columns'):
            # DataFrame - handle this first before checking for shape
            num_spectra = min(data.shape[1], 10)
            for i, column in enumerate(data.columns[:num_spectra]):
                ax.plot(data.index, data[column], label=column)
        elif hasattr(data, 'shape') and len(data.shape) == 2:
            # Numpy array or similar
            num_spectra = min(data.shape[1] if data.shape[1] < data.shape[0] else data.shape[0], 10)
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
                if hasattr(data, 'index') and hasattr(data, 'values'):
                    # DataFrame with wavenumber index
                    wavenumbers = data.index.values
                    intensities = data.values
                    min_wn, max_wn = detect_signal_range(wavenumbers, intensities.T, focus_padding=focus_padding, crop_bounds=crop_bounds)  # Transpose for proper shape
                    ax.set_xlim(min_wn, max_wn)
            except Exception as e:
                pass  # Silently fall back to full range
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_comparison_spectra(self, original_data, processed_data, titles=None, colors=None):
        """Plot comparison between original and processed data."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if titles is None:
            titles = ["Original", "Processed"]
        if colors is None:
            colors = ["lightblue", "darkblue"]
        
        # Plot original data (sample)
        if original_data is not None:
            num_original = min(5, original_data.shape[1] if hasattr(original_data, 'shape') and len(original_data.shape) == 2 else len(original_data))
            for i in range(num_original):
                if hasattr(original_data, 'shape') and len(original_data.shape) == 2:
                    spectrum = original_data[:, i] if original_data.shape[1] < original_data.shape[0] else original_data[i, :]
                    x_data = range(len(spectrum))
                else:
                    spectrum = original_data
                    x_data = range(len(spectrum))
                
                ax.plot(x_data, spectrum, color=colors[0], alpha=0.6, linewidth=1, 
                       label=titles[0] if i == 0 else "")
        
        # Plot processed data
        if processed_data is not None:
            num_processed = min(5, processed_data.shape[1] if hasattr(processed_data, 'shape') and len(processed_data.shape) == 2 else len(processed_data))
            for i in range(num_processed):
                if hasattr(processed_data, 'shape') and len(processed_data.shape) == 2:
                    spectrum = processed_data[:, i] if processed_data.shape[1] < processed_data.shape[0] else processed_data[i, :]
                    x_data = range(len(spectrum))
                else:
                    spectrum = processed_data
                    x_data = range(len(spectrum))
                
                ax.plot(x_data, spectrum, color=colors[1], alpha=0.8, linewidth=1.5,
                       label=titles[1] if i == 0 else "")
        
        ax.set_title("Preprocessing Preview")
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_comparison_spectra_with_wavenumbers(self, original_data, processed_data, 
                                               original_wavenumbers, processed_wavenumbers,
                                               titles=None, colors=None, auto_focus=True, focus_padding=None, crop_bounds=None):
        """Plot comparison between original and processed data with proper wavenumber axes."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if titles is None:
            titles = ["Original", "Processed"]
        if colors is None:
            colors = ["lightblue", "darkblue"]
        
        # Plot original data (sample) with actual wavenumbers
        if original_data is not None and original_wavenumbers is not None:
            num_original = min(5, original_data.shape[0] if hasattr(original_data, 'shape') and len(original_data.shape) == 2 else 1)
            for i in range(num_original):
                if hasattr(original_data, 'shape') and len(original_data.shape) == 2:
                    spectrum = original_data[i, :]
                else:
                    spectrum = original_data
                
                ax.plot(original_wavenumbers, spectrum, color=colors[0], alpha=0.6, linewidth=1, 
                       label=titles[0] if i == 0 else "")
        
        # Plot processed data with actual wavenumbers
        if processed_data is not None and processed_wavenumbers is not None:
            num_processed = min(5, processed_data.shape[0] if hasattr(processed_data, 'shape') and len(processed_data.shape) == 2 else 1)
            for i in range(num_processed):
                if hasattr(processed_data, 'shape') and len(processed_data.shape) == 2:
                    spectrum = processed_data[i, :]
                else:
                    spectrum = processed_data
                
                ax.plot(processed_wavenumbers, spectrum, color=colors[1], alpha=0.8, linewidth=1.5,
                       label=titles[1] if i == 0 else "")
        
        ax.set_title("Preprocessing Preview")
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Apply auto-focus only if requested
        if auto_focus:
            try:
                # Determine the best range from the available data
                focus_wavenumbers = processed_wavenumbers if processed_wavenumbers is not None else original_wavenumbers
                
                if focus_wavenumbers is not None:
                    # Get intensity data for range detection
                    if processed_data is not None:
                        focus_intensities = processed_data
                    elif original_data is not None:
                        focus_intensities = original_data
                    else:
                        focus_intensities = None
                    
                    if focus_intensities is not None:
                        min_wn, max_wn = detect_signal_range(focus_wavenumbers, focus_intensities, focus_padding=focus_padding, crop_bounds=crop_bounds)
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
    
    fig = Figure(figsize=(8, 6), dpi=100, facecolor='#eaf2f8') # Themed background
    ax = fig.add_subplot(111, facecolor='#eaf2f8') # Themed background

    # --- Robustness Check ---
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data to display.", ha='center', va='center', fontsize=14, color='gray')
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
    
    ax.set_title(plot_title, fontsize=14, weight='bold')
    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#d1dbe5')
    
    # Customize legend
    legend = ax.legend(facecolor='#ffffff', framealpha=0.7)
    for text in legend.get_texts():
        text.set_color('#34495e')

    # Customize tick colors
    ax.tick_params(axis='x', colors='#34495e', labelsize=10)
    ax.tick_params(axis='y', colors='#34495e', labelsize=10)

    # Customize spine colors
    for spine in ax.spines.values():
        spine.set_edgecolor('#34495e')

    # Apply auto-focus only if requested
    if auto_focus:
        try:
            wavenumbers = df_to_plot.index.values
            intensities = df_to_plot.values
            min_wn, max_wn = detect_signal_range(wavenumbers, intensities.T)  # Transpose for proper shape
            ax.set_xlim(min_wn, max_wn)
        except Exception as e:
            pass  # Silently fall back to full range

    # Adjust layout with explicit padding to ensure y-axis labels are visible
    fig.tight_layout(pad=1.5)  # Increased padding for better visibility
    fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.10)  # Explicit margins for y-axis labels
    return fig
