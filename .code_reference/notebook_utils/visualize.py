# Enhanced Interactive Raman Spectrum Visualization with Advanced Features
import plotly.graph_objects as go
from ipywidgets import widgets, VBox, HBox, Output, Layout, HTML, Box
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import umap
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu

from typing import Tuple, Dict, Any, Callable, Union, List, Optional
from IPython.display import display, Markdown

class BasicRamanPlot:
    """
    A class for interactive Raman spectrum visualization with layering capabilities.
    
    Expected data format:
    raman_data = {
        'data_type_1': {
            'sample_1': [{'dataframe': df, 'metadata': meta}, ...],
            'sample_2': [...],
            ...
        },
        'data_type_2': {...},
        ...
    }

    Usage example:
    analyzer = BasicRamanPlot(raman_data)
    interface = analyzer.initialize()
    display(interface)
    """
    
    def __init__(self, raman_data, legend_metadata_keys: Optional[List[str]] = None):
        """
        Initialize the Basic Raman Spectrum Plot.
        
        Args:
            raman_data (dict): Dictionary containing Raman data organized by data types and samples
            legend_metadata_keys (List[str], optional): List of metadata keys to include in the legend.
                If a key doesn't exist for a spectrum, it's skipped. Defaults to None (no metadata in legend).
        """
        self.raman_data = raman_data
        self.legend_metadata_keys = legend_metadata_keys or []
        self.color_palette = [
            'blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'
        ]
        
        # Create consistent color mapping for data types
        self.data_type_colors = {}
        for idx, data_type in enumerate(sorted(self.raman_data.keys())):
            self.data_type_colors[data_type] = self.color_palette[idx % len(self.color_palette)]
        
        # Initialize widget containers
        self.sample_dropdowns_container = VBox([])
        self.sample_selections = {}
        self.layer_buttons = {}
        self.output = Output()
        self.presentation_mode = False
        
        # Create main widgets
        self._create_widgets()
        self._setup_observers()
        
        # Initialize the interface
        self.interface = self._create_interface()  
    
    def _create_widgets(self):
        """Create all the necessary widgets."""
        # Data type selection
        self.data_type_select = self.LimitedSelectMultiple(
            options=self._get_data_type_options(),
            value=[self._get_data_type_options()[0]],
            description='Data Types (max 5):',
            style={'description_width': 'initial'},
            layout=Layout(height='120px', width='400px'),
            max_selections=5
        )
        
        # Spectrum controls
        self.num_spectra_input = widgets.BoundedIntText(
            value=1, min=1, max=50, step=1,
            description='Num Spectra:', 
            style={'description_width': 'initial'},
            layout=Layout(width='150px')
        )
        
        self.num_spectra_slider = widgets.IntSlider(
            value=1, min=1, max=10, step=1, description='',
            layout=Layout(width='200px')
        )
        
        self.start_index_input = widgets.BoundedIntText(
            value=0, min=0, max=50, step=1,
            description='Start Index:',
            style={'description_width': 'initial'},
            layout=Layout(width='150px')
        )
        
        self.start_index_slider = widgets.IntSlider(
            value=0, min=0, max=50, step=1, description='',
            layout=Layout(width='200px')
        )
        
        # Link input boxes and sliders
        widgets.jslink((self.num_spectra_input, 'value'), (self.num_spectra_slider, 'value'))
        widgets.jslink((self.start_index_input, 'value'), (self.start_index_slider, 'value'))
        
        # Presentation mode checkbox
        self.presentation_checkbox = widgets.Checkbox(
            value=False,
            description='Presentation Mode',
            tooltip='Toggle to show only the plot for presentations',
            layout=Layout(margin='0 0 10px 0')
        )
    
    class LimitedSelectMultiple(widgets.SelectMultiple):
        """Custom SelectMultiple with max selection limit."""
        def __init__(self, max_selections=5, **kwargs):
            super().__init__(**kwargs)
            self.max_selections = max_selections
            self.observe(self._limit_selections, names='value')
        
        def _limit_selections(self, change):
            if len(change['new']) > self.max_selections:
                self.value = change['new'][:self.max_selections]
    
    def _get_data_type_options(self):
        """Get data type options with sample counts."""
        options = []
        for data_type in self.raman_data.keys():
            sample_count = len(self.raman_data[data_type])
            options.append(f"{data_type} ({sample_count} samples)")
        return options
    
    def _get_data_type_value(self, option):
        """Extract data type name from option string."""
        return option.split(' (')[0]
    
    def _get_layer_opacities(self, num_layers):
        """Calculate opacities for layering effect."""
        if num_layers == 1:
            return [1.0]
        elif num_layers == 2:
            return [1.0, 0.5]
        else:
            opacities = [1.0]
            for i in range(1, num_layers):
                opacity = max(0.1, 1.0 - (i * 0.2))
                opacities.append(opacity)
            return opacities
    
    def get_sample_color(self, data_type, sample_idx):
        """
        Get color for a sample within a data type.
        First sample uses the base data type color for layering consistency.
        Subsequent samples use next colors from the palette.
        """
        base_color = self.data_type_colors.get(data_type, self.color_palette[0])
        if sample_idx == 0:
            return base_color
        else:
            # Use next color in palette for additional samples
            data_type_idx = list(self.data_type_colors.keys()).index(data_type) if data_type in self.data_type_colors else 0
            color_idx = (data_type_idx + sample_idx) % len(self.color_palette)
            return self.color_palette[color_idx]

    def _create_multi_spectrum_plot(self, selected_data_types, sample_selections, num_spectra, start_index):
        """Create an interactive Raman spectrum plot with layering."""
        fig = go.Figure()
        layer_opacities = self._get_layer_opacities(len(selected_data_types))
        
        trace_count = 0
        
        for i, data_type in enumerate(selected_data_types):
            if data_type not in self.raman_data:
                continue
                
            sample_selection = sample_selections.get(data_type, [])
            # Handle both single sample (string) and multiple samples (list)
            if isinstance(sample_selection, str):
                sample_names = [sample_selection]
            elif isinstance(sample_selection, (list, tuple)):
                sample_names = list(sample_selection)
            else:
                sample_names = [list(self.raman_data[data_type].keys())[0]]
            
            opacity = layer_opacities[i]
            
            # Process each selected sample
            for sample_idx, sample_name in enumerate(sample_names):
                if sample_name not in self.raman_data[data_type]:
                    continue
                    
                # Get color for this sample
                color = self.get_sample_color(data_type, sample_idx)
                
                spectra_list = self.raman_data[data_type][sample_name]
                
                for spectrum_idx in range(start_index, min(start_index + num_spectra, len(spectra_list))):
                    if spectrum_idx >= len(spectra_list):
                        continue
                        
                    sample_data = spectra_list[spectrum_idx]
                    df = sample_data['dataframe']
                    metadata = sample_data['metadata']
                    
                    # Build name with metadata
                    name_parts = [data_type, sample_name, f"Spectrum {spectrum_idx}"]
                    hover_parts = [
                        f'<b>{data_type} - {sample_name}</b>',
                        '<b>Wavelength:</b> %{x} cm⁻¹',  # Fixed: No f-string for Plotly placeholders
                        '<b>Intensity:</b> %{y:.4f}',   # Fixed: No f-string for Plotly placeholders
                        f'<b>Opacity:</b> {opacity:.1f}'
                    ]
                    
                    for key in self.legend_metadata_keys:
                        if key in metadata:
                            value = metadata[key]
                            name_parts.append(f"{key}: {value}")
                            hover_parts.append(f'<b>{key}:</b> {value}')
                    
                    name = " - ".join(name_parts)
                    hovertemplate = "<br>".join(hover_parts) + '<extra></extra>'  # This is correct as-is
                    
                    fig.add_trace(go.Scatter(
                        x=df['wavelength'],
                        y=df['intensity'],
                        mode='lines',
                        name=name,
                        line=dict(color=color, width=2),
                        opacity=opacity,
                        hovertemplate=hovertemplate,
                        legendgroup=data_type,
                        showlegend=True
                    ))
                    
                    trace_count += 1
        
        if trace_count > 0:
            layer_info = []
            for i, (data_type, opacity) in enumerate(zip(selected_data_types, layer_opacities)):
                layer_info.append(f"Layer {i+1}: {data_type} (α={opacity:.1f})")
            
            fig.update_layout(
                title=f'Raman Spectra Comparison<br><sub>{" | ".join(layer_info)}<br>Spectra: {num_spectra} | Start: {start_index}</sub>',
                xaxis_title='Wavelength (cm⁻¹)',
                yaxis_title='Intensity (a.u.)',
                template='plotly_white',
                hovermode='x unified',
                height=700,
                margin=dict(t=120),  # Add top margin for legend
                xaxis=dict(range=[600, 1800], autorange=False),
                yaxis=dict(autorange=True),
                legend=dict(
                    title="Data Types & Spectra",
                    orientation="h",  # Horizontal legend
                    yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='rgba(0, 0, 0, 0.2)', borderwidth=1
                )
            )
        else:
            fig.update_layout(
                title="No Valid Data Selected",
                xaxis_title="Wavelength (cm⁻¹)",
                yaxis_title="Intensity",
                height=700
            )
        
        return fig 
    
    def _move_layer_up(self, data_type):
        """Move a data type up in the layer order."""
        current_order = list(self.data_type_select.value)
        data_type_option = None
        for option in current_order:
            if self._get_data_type_value(option) == data_type:
                data_type_option = option
                break
        
        if data_type_option:
            idx = current_order.index(data_type_option)
            if idx > 0:
                current_order[idx], current_order[idx-1] = current_order[idx-1], current_order[idx]
                self.data_type_select.value = tuple(current_order)
                self._update_sample_dropdowns()
    
    def _move_layer_down(self, data_type):
        """Move a data type down in the layer order."""
        current_order = list(self.data_type_select.value)
        data_type_option = None
        for option in current_order:
            if self._get_data_type_value(option) == data_type:
                data_type_option = option
                break
        
        if data_type_option:
            idx = current_order.index(data_type_option)
            if idx < len(current_order) - 1:
                current_order[idx], current_order[idx+1] = current_order[idx+1], current_order[idx]
                self.data_type_select.value = tuple(current_order)
                self._update_sample_dropdowns()
    
    def _toggle_presentation_mode(self, change):
        """Toggle between full interface and presentation mode."""
        self.presentation_mode = change['new']
        
        if self.presentation_mode:
            # Hide control sections, show only plot with checkbox
            # Ensure plot section maintains full width in presentation mode
            self.plot_section.layout = Layout(padding='10px', border='1px solid #ddd', margin='5px', width='100%')
            
            self.interface.children = [
                HTML("<h3 style='text-align: center; color: #2E86AB; margin-bottom: 10px;'>🔬 Basic Raman Spectrum Plot (Presentation Mode)</h3>"),
                HBox([
                    self.presentation_checkbox,
                    HTML("<div style='flex: 1;'></div>")  # Spacer to push checkbox to left
                ], layout=Layout(margin='0 0 10px 0', width='100%')),
                self.plot_section
            ]
            # Update interface layout to ensure full width
            self.interface.layout = Layout(padding='15px', border='2px solid #2E86AB', border_radius='10px', width='100%')
        else:
            # Show full interface
            # Reset plot section to original layout
            self.plot_section.layout = Layout(padding='10px', border='1px solid #ddd', margin='5px')
            
            self.interface.children = [
                HTML("<h3 style='text-align: center; color: #2E86AB; margin-bottom: 10px;'>🔬 Basic Raman Spectrum Plot</h3>"),
                self.presentation_checkbox,
                HBox([self.data_section, self.spectrum_section], layout=Layout(align_items='flex-start')),
                self.plot_section
            ]
            # Reset interface layout to original
            self.interface.layout = Layout(padding='15px', border='2px solid #2E86AB', border_radius='10px')
    
    def _create_sample_dropdowns(self, selected_data_types):
        """Create sample dropdown widgets for selected data types."""
        dropdowns = []
        self.sample_selections.clear()
        self.layer_buttons.clear()
        
        for position, data_type_option in enumerate(selected_data_types):
            data_type = self._get_data_type_value(data_type_option)
            if data_type in self.raman_data:
                sample_options = []
                for sample_name in self.raman_data[data_type].keys():
                    spectrum_count = len(self.raman_data[data_type][sample_name])
                    sample_options.append(f"{sample_name} ({spectrum_count} spectra)")
                
                # Use SelectMultiple if more than 1 sample, otherwise use Dropdown
                if len(sample_options) > 1:
                    sample_widget = widgets.SelectMultiple(
                        options=sample_options,
                        value=[sample_options[0]] if sample_options else [],
                        description=f'{data_type}:',
                        style={'description_width': 'initial'},
                        layout=Layout(width='300px', height='100px')
                    )
                else:
                    sample_widget = widgets.Dropdown(
                        options=sample_options,
                        value=sample_options[0] if sample_options else None,
                        description=f'{data_type}:',
                        style={'description_width': 'initial'},
                        layout=Layout(width='300px')
                    )
                self.sample_selections[data_type] = sample_widget
                
                # Create layer control buttons based on position
                buttons = []
                
                # Only show up button if not first
                if position > 0:
                    up_button = widgets.Button(
                        description='↑', button_style='info',
                        tooltip=f'Move {data_type} layer up',
                        layout=Layout(width='40px')
                    )
                    up_button.on_click(lambda b, dt=data_type: self._move_layer_up(dt))
                    buttons.append(up_button)
                
                # Only show down button if not last
                if position < len(selected_data_types) - 1:
                    down_button = widgets.Button(
                        description='↓', button_style='info',
                        tooltip=f'Move {data_type} layer down',
                        layout=Layout(width='40px')
                    )
                    down_button.on_click(lambda b, dt=data_type: self._move_layer_down(dt))
                    buttons.append(down_button)
                
                self.layer_buttons[data_type] = tuple(buttons)
                
                # Create row with dropdown and buttons
                if buttons:
                    row = HBox([
                        sample_widget,
                        VBox(buttons, layout=Layout(margin='0 0 0 10px'))
                    ], layout=Layout(margin='5px 0'))
                else:
                    row = sample_widget
                
                dropdowns.append(row)
        
        return dropdowns
    
    def _update_sample_dropdowns(self, change=None):
        """Update sample dropdowns based on selected data types."""
        selected_data_types = list(self.data_type_select.value)
        dropdowns = self._create_sample_dropdowns(selected_data_types)
        self.sample_dropdowns_container.children = dropdowns
        self._update_slider_max_values()
        self._observe_sample_changes()
    
    def _update_slider_max_values(self, change=None):
        """Update maximum values for sliders based on current selections."""
        selected_data_types = [self._get_data_type_value(dt) for dt in self.data_type_select.value]
        max_spectra = 1
        max_start = 0
        
        for data_type in selected_data_types:
            sample_dropdown = self.sample_selections.get(data_type)
            if sample_dropdown and hasattr(sample_dropdown, 'value') and sample_dropdown.value:
                selected_samples = sample_dropdown.value
                if isinstance(selected_samples, (list, tuple)):
                    # Handle multiple selections (SelectMultiple widget)
                    for sample_option in selected_samples:
                        sample_name = sample_option.split(' (')[0]
                        if data_type in self.raman_data and sample_name in self.raman_data[data_type]:
                            num_spectra_in_sample = len(self.raman_data[data_type][sample_name])
                            max_spectra = max(max_spectra, num_spectra_in_sample)
                            max_start = max(max_start, num_spectra_in_sample - 1)
                else:
                    # Handle single selection (Dropdown widget)
                    sample_name = selected_samples.split(' (')[0]
                    if data_type in self.raman_data and sample_name in self.raman_data[data_type]:
                        num_spectra_in_sample = len(self.raman_data[data_type][sample_name])
                        max_spectra = max(max_spectra, num_spectra_in_sample)
                        max_start = max(max_start, num_spectra_in_sample - 1)
        
        self.num_spectra_input.max = max_spectra
        self.num_spectra_slider.max = max_spectra
        self.start_index_input.max = max_start
        self.start_index_slider.max = max_start
        
        if self.num_spectra_input.value > max_spectra:
            self.num_spectra_input.value = max_spectra
        if self.start_index_input.value > max_start:
            self.start_index_input.value = max_start

    def _update_plot(self, change=None):
        """Update the spectrum plot based on current selections."""
        with self.output:
            self.output.clear_output(wait=True)
            selected_data_types = [self._get_data_type_value(dt) for dt in self.data_type_select.value]
            
            current_sample_selections = {}
            for data_type in selected_data_types:
                if data_type in self.sample_selections:
                    sample_value = self.sample_selections[data_type].value
                    if sample_value:
                        if isinstance(sample_value, (list, tuple)):
                            # Handle multiple selections (SelectMultiple widget)
                            current_sample_selections[data_type] = [s.split(' (')[0] for s in sample_value]
                        else:
                            # Handle single selection (Dropdown widget)
                            current_sample_selections[data_type] = sample_value.split(' (')[0]
            
            fig = self._create_multi_spectrum_plot(
                selected_data_types, current_sample_selections, 
                self.num_spectra_input.value, self.start_index_input.value
            )
            fig.show()
    
    def _observe_sample_changes(self):
        """Set up observers for all sample dropdowns."""
        for dropdown in self.sample_selections.values():
            if hasattr(dropdown, 'observe'):
                dropdown.observe(self._update_slider_max_values, names='value')
                dropdown.observe(self._update_plot, names='value')
    
    def _setup_observers(self):
        """Set up all observers for the widgets."""
        # Data type selection observers
        self.data_type_select.observe(self._update_sample_dropdowns, names='value')
        self.data_type_select.observe(self._update_plot, names='value')
        self.data_type_select.observe(self._update_slider_max_values, names='value')
        
        # Spectrum control observers
        self.num_spectra_input.observe(self._update_plot, names='value')
        self.num_spectra_slider.observe(self._update_plot, names='value')
        self.start_index_input.observe(self._update_plot, names='value')
        self.start_index_slider.observe(self._update_plot, names='value')
        
        # Presentation mode observer
        self.presentation_checkbox.observe(self._toggle_presentation_mode, names='value')
    
    def _create_interface(self):
        """Create the main interface layout."""
        # Create sections
        self.data_section = VBox([
            HTML("<h4 style='margin: 0; color: #2E86AB;'>📊 Data Selection</h4>"),
            self.data_type_select,
            HTML("<h5 style='margin: 5px 0; color: #A23B72;'>Sample Selection & Layer Order</h5>"),
            self.sample_dropdowns_container
        ], layout=Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        
        self.spectrum_section = VBox([
            HTML("<h4 style='margin: 0; color: #2E86AB;'>🎛️ Spectrum Controls</h4>"),
            HBox([
                VBox([self.num_spectra_input, self.num_spectra_slider], layout=Layout(margin='0 10px 0 0')),
                VBox([self.start_index_input, self.start_index_slider])
            ], layout=Layout(padding='10px'))
        ], layout=Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        
        self.plot_section = VBox([
            HTML("<h4 style='margin: 0; color: #2E86AB;'>📈 Interactive Plot</h4>"),
            self.output
        ], layout=Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        
        # Main interface
        interface = VBox([
            HTML("<h3 style='text-align: center; color: #2E86AB; margin-bottom: 10px;'>🔬 Basic Raman Spectrum Plot</h3>"),
            self.presentation_checkbox,
            HBox([self.data_section, self.spectrum_section], layout=Layout(align_items='flex-start')),
            self.plot_section
        ], layout=Layout(padding='15px', border='2px solid #2E86AB', border_radius='10px'))
        
        return interface
    
    def initialize(self):
        """Initialize the interface and display it."""
        self._update_sample_dropdowns()
        self._update_slider_max_values()
        self._update_plot()
        self._observe_sample_changes()
        return self.interface
    
    def get_interface(self):
        """Get the main interface widget."""
        return self.interface
    
    def update_data(self, new_raman_data):
        """Update the Raman data and refresh the interface."""
        self.raman_data = new_raman_data
        # Update color mapping for new data types
        self.data_type_colors = {}
        for idx, data_type in enumerate(sorted(self.raman_data.keys())):
            self.data_type_colors[data_type] = self.color_palette[idx % len(self.color_palette)]
        
        self.data_type_select.options = self._get_data_type_options()
        self.data_type_select.value = [self._get_data_type_options()[0]] if self._get_data_type_options() else []
        self._update_sample_dropdowns()
        self._update_plot()



class RamanSpectrumAdvanceVisualize:
    """
    Advanced Raman Spectrum Visualization Class
    
    This comprehensive class provides multiple visualization methods for Raman spectroscopy data
    including interactive plotting, statistical analysis, dimensionality reduction, and machine learning
    model evaluation visualizations.
    
    Expected data format (same as BasicRamanPlot):
    raman_data = {
        'data_type_1': {
            'sample_1': [{'dataframe': df, 'metadata': meta}, ...],
            'sample_2': [...],
            ...
        },
        'data_type_2': {...},
        ...
    }
    
    Available Visualization Methods:
    ================================
    
    1. INTERACTIVE SPECTRUM PLOTTING:
       - interactive_spectrum_plot(): Interactive multi-spectrum comparison with layering
       - basic_plot: Access to BasicRamanPlot class for simple spectrum visualization
    
    2. DATA EXTRACTION & FILTERING:
       - extract_spectra(): Convert nested dict to ML-ready arrays with filtering
       - create_metadata_filter(): Factory for common metadata filters
       - analyze_metadata_distribution(): Analyze metadata value distributions
    
    3. DIMENSIONALITY REDUCTION:
       - plot_raman_pca(): PCA visualization for single data type
       - plot_raman_umap(): UMAP visualization for multiple data types
       - plot_disease_progression_umap(): Disease progression continuum on UMAP
    
    4. MACHINE LEARNING EVALUATION:
       - show_model_report(): Comprehensive model performance report
       - plot_pca_results(): PCA with train/test split visualization
       - plot_sample_distribution(): Sample distribution by class
       - plot_confusion_matrices(): Confusion matrix (counts and normalized)
       - plot_roc_pr(): ROC and Precision-Recall curves
       - plot_feature_importance(): Feature importance bar plot
       - plot_cv_distribution(): Cross-validation score distribution
    
    Usage Examples:
    ===============
    
    # Initialize
    visualizer = RamanSpectrumAdvanceVisualize(raman_data)
    
    # Interactive spectrum plotting
    interface = visualizer.interactive_spectrum_plot()
    display(interface)
    
    # Extract data for ML
    X, y, sample_ids, type_labels, label_map = visualizer.extract_spectra(
        type_keys=('MGUS', 'MM'),
        metadata_filters={'Hikkoshi': 'A'}
    )
    
    # Dimensionality reduction
    visualizer.plot_raman_umap(['MGUS', 'MM', 'NL'], group_by='Hikkoshi')
    
    # Model evaluation (after training)
    visualizer.show_model_report(model_results, X_original=X, y_original=y)
    
    # Metadata analysis
    distribution_df = visualizer.analyze_metadata_distribution(
        type_keys=['MGUS', 'MM'], 
        metadata_key='Hikkoshi'
    )
    """
    
    def __init__(self, raman_data):
        """
        Initialize the Advanced Raman Spectrum Visualizer.
        
        Args:
            raman_data (dict): Raman data dictionary following the standard format
        """
        self.raman_data = raman_data
        self.basic_plot = None  # Placeholder for BasicRamanPlot instance
    
    ## INTERACTIVE SPECTRUM PLOTTING SECTION ##
    
    def interactive_spectrum_plot(self, legend_metadata_keys: List[str]=["Hikkoshi", "Date"]):
        """
        Get the interactive spectrum plotting interface from BasicRamanPlot.

        Args:
            legend_metadata_keys: List of metadata keys to include in the legend.
                If a key doesn't exist for a spectrum, it's skipped. Defaults to ["Hikkoshi", "Date"].
        Returns:
            ipywidgets.VBox: Interactive interface for spectrum visualization
        """
        self.basic_plot = BasicRamanPlot(self.raman_data, legend_metadata_keys=legend_metadata_keys)
        return self.basic_plot.initialize()
    
    ## DATA EXTRACTION & FILTERING SECTION ##
    
    def extract_spectra(
        self,
        type_keys: tuple = ("MGUS", "MM"),
        label_map: dict | None = None,
        wavelength_sort: bool = True,
        filter_condition: Callable[[dict], bool] | None = None,
        metadata_filters: dict | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Convert nested raman_dict into (X, y, sample_ids, type_labels) with optional filtering.

        Args:
            type_keys: Tuple of type names to extract
            label_map: Optional explicit mapping {type_name: int}. If omitted, auto-assign
            wavelength_sort: Whether to sort by wavelength
            filter_condition: Custom function that takes metadata dict and returns bool
            metadata_filters: Dict of {metadata_key: value} or {metadata_key: list_of_values}

        Returns:
            Tuple of (X, y, sample_ids, type_labels, label_map)
        """
        spectra = []
        y = []
        sample_ids = []
        type_labels = []
        
        if label_map is None:
            label_map = {t: i for i, t in enumerate(type_keys)}

        def _passes_filter(metadata: dict) -> bool:
            """Check if metadata passes all filter conditions."""
            # Apply custom filter condition
            if filter_condition is not None:
                if not filter_condition(metadata):
                    return False
            
            # Apply metadata filters
            if metadata_filters is not None:
                for key, expected_value in metadata_filters.items():
                    actual_value = metadata.get(key)
                    
                    # Handle list of acceptable values
                    if isinstance(expected_value, (list, tuple, set)):
                        if actual_value not in expected_value:
                            return False
                    # Handle single value
                    else:
                        if actual_value != expected_value:
                            return False
            
            return True

        for t in type_keys:
            if t not in self.raman_data:
                continue
                
            for sample_id, entries in self.raman_data[t].items():
                for e in entries:
                    metadata = e.get("metadata", {})
                    
                    # Apply filters
                    if not _passes_filter(metadata):
                        continue
                    
                    df = e["dataframe"]
                    if wavelength_sort:
                        df = df.sort_values("wavelength")
                        
                    spectra.append(df["intensity"].values)
                    y.append(label_map[t])
                    sample_ids.append(sample_id)
                    type_labels.append(t)

        X = np.asarray(spectra, dtype=float)
        y = np.asarray(y)
        sample_ids = np.asarray(sample_ids)
        type_labels = np.asarray(type_labels)
        
        return X, y, sample_ids, type_labels, label_map

    def create_metadata_filter(
        self,
        hikkoshi: str = None,
        date_range: Tuple[str, str] = None,
        exclude_samples: list = None,
        custom_conditions: dict = None
    ) -> Callable[[dict], bool]:
        """
        Factory function to create commonly used filter conditions.
        
        Args:
            hikkoshi: Filter by Hikkoshi value ('A', 'B', etc.)
            date_range: Tuple of (start_date, end_date) in string format
            exclude_samples: List of sample IDs to exclude
            custom_conditions: Dict of {metadata_key: value} conditions
        
        Returns:
            Filter function that takes metadata dict and returns bool
        """
        def filter_func(metadata: dict) -> bool:
            # Hikkoshi filter
            if hikkoshi is not None:
                if metadata.get("Hikkoshi") != hikkoshi:
                    return False
            
            # Date range filter
            if date_range is not None:
                sample_date = metadata.get("Date")
                if sample_date is not None:
                    try:
                        if not (date_range[0] <= str(sample_date) <= date_range[1]):
                            return False
                    except (TypeError, ValueError):
                        return False
            
            # Exclude specific samples
            if exclude_samples is not None:
                sample_no = metadata.get("SampleNo")
                if sample_no in exclude_samples:
                    return False
            
            # Custom conditions
            if custom_conditions is not None:
                for key, expected_value in custom_conditions.items():
                    if metadata.get(key) != expected_value:
                        return False
            
            return True
        
        return filter_func

    def analyze_metadata_distribution(
        self,
        type_keys: tuple = None,
        metadata_key: str = "Hikkoshi"
    ) -> pd.DataFrame:
        """
        Analyze distribution of metadata values across types.
        
        Args:
            type_keys: Types to analyze (if None, analyze all)
            metadata_key: Metadata field to analyze
        
        Returns:
            DataFrame with distribution counts
        """
        if type_keys is None:
            type_keys = list(self.raman_data.keys())
        
        distribution_data = []
        
        for typ in type_keys:
            if typ not in self.raman_data:
                continue
                
            value_counts = {}
            total_count = 0
            
            for sample_id, entries in self.raman_data[typ].items():
                for entry in entries:
                    metadata = entry.get("metadata", {})
                    value = metadata.get(metadata_key, "Unknown")
                    value_counts[value] = value_counts.get(value, 0) + 1
                    total_count += 1
            
            for value, count in value_counts.items():
                distribution_data.append({
                    "Type": typ,
                    metadata_key: value,
                    "Count": count,
                    "Percentage": f"{count/total_count*100:.1f}%" if total_count > 0 else "0%"
                })
        
        return pd.DataFrame(distribution_data)
    
    ## PCA and UMAP ##
    def _finite_impute(self, X: np.ndarray) -> np.ndarray:
        """Replace inf with nan and impute per-feature means for any remaining nans."""
        if X.size == 0:
            return X
        
        Xs = np.array(X, dtype=float, copy=True)
        
        # Handle the case where X has no features
        if Xs.shape[1] == 0:
            return Xs
        
        Xs[~np.isfinite(Xs)] = np.nan
        if np.isnan(Xs).any():
            imp = SimpleImputer(strategy="mean")
            Xs = imp.fit_transform(Xs)
        return Xs

    def plot_raman_pca(
        self,
        type_key: str = "MGUS",
        group_by: str = "Hikkoshi",
        n_components: int = 2,
        title: str = None,
        show_figure = True
    ) -> dict:
        """
        Plot PCA for a given type, colored by group_by metadata.
        """
        # Gather all spectra and metadata for the given type
        spectra = []
        groups = []
        for sample_list in self.raman_data.get(type_key, {}).values():
            for entry in sample_list:
                df = entry["dataframe"]
                # Ensure sorted by wavelength
                df_sorted = df.sort_values("wavelength")
                spectra.append(df_sorted["intensity"].values)
                groups.append(entry["metadata"].get(group_by, "Unknown"))
        if not spectra:
            print(f"No data found for type '{type_key}'.")
            return
        X = np.vstack(spectra)
        # PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Plot with distinct colors
        fig, ax = plt.subplots(figsize=(8, 6))
        unique_groups = sorted(set(groups))
        
        # Use highly distinct colors for better visual separation
        distinct_colors = [
            '#FF0000',  # Red
            '#0000FF',  # Blue  
            '#00FF00',  # Green
            '#FF8000',  # Orange
            '#8000FF',  # Purple
            '#FF0080',  # Magenta
            '#00FFFF',  # Cyan
            '#FFFF00',  # Yellow
            '#800000',  # Maroon
            '#000080',  # Navy
            '#008000',  # Dark Green
            '#808000',  # Olive
            '#800080',  # Purple
            '#008080',  # Teal
            '#C0C0C0',  # Silver
            '#808080'   # Gray
        ]
        
        # Cycle through colors if we have more groups than colors
        for i, group in enumerate(unique_groups):
            color = distinct_colors[i % len(distinct_colors)]
            idx = [j for j, g in enumerate(groups) if g == group]
            ax.scatter(X_pca[idx, 0], X_pca[idx, 1], label=str(group), color=color, 
                      alpha=0.7, s=50, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.set_title(title or f"PCA of {type_key} colored by {group_by}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if show_figure:
            plt.show()

        return {
            "pca": pca,
            "X_pca": X_pca,
            "groups": groups,
            "figure": fig,
            "axes": ax
        }

    def plot_raman_umap(
        self,
        type_keys: List[str],
        group_by: str = "Hikkoshi",
        size_by: Optional[str] = None,  # New: Optional metadata key for marker size (area)
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        title: str = None,
        random_state: int = 42,
        use_density_plot: bool = False,  # New: Use density plots for large datasets
        max_points_scatter: int = 5000,  # New: Threshold for switching to density plot
        subsample_ratio: float = None,  # New: Optional subsampling for very large datasets
        enable_datashader: bool = False,  # New: Use datashader for very large datasets
        **kwargs
    ) -> dict:
        """
        Plot UMAP for selected type_keys, colored by type, shaped by group_by metadata,
        and optionally sized by size_by metadata for additional differentiation.
        Optimized for large datasets with density plotting and subsampling options.
        
        Args:
            type_keys: List of data types to include.
            group_by: Metadata key for marker shapes.
            size_by: Optional metadata key for marker sizes (area). If provided, sizes will be scaled.
            n_neighbors: UMAP parameter for neighborhood size.
            min_dist: UMAP parameter for minimum distance.
            metric: UMAP distance metric.
            title: Optional plot title.
            random_state: Random state for reproducibility.
            use_density_plot: Use density plots instead of scatter for large datasets.
            max_points_scatter: Maximum points before switching to density plot automatically.
            subsample_ratio: If provided (0-1), randomly subsample data for visualization.
            enable_datashader: Use datashader for very large datasets (requires datashader package).
            **kwargs: Additional UMAP parameters.
        
        Returns:
            Dict containing UMAP results and plot elements.
        """
        spectra = []
        groups = []
        types = []
        sizes = []  # New: List to store sizes if size_by is provided
        
        for typ in type_keys:
            for sample_list in self.raman_data.get(typ, {}).values():
                for entry in sample_list:
                    df = entry["dataframe"]
                    df_sorted = df.sort_values("wavelength")
                    spectra.append(df_sorted["intensity"].values)
                    groups.append(entry["metadata"].get(group_by, "Unknown"))
                    types.append(typ)
                    # New: Compute size if size_by is provided
                    if size_by:
                        size_value = entry["metadata"].get(size_by, 0)  # Default to 0 if missing
                        sizes.append(float(size_value) if isinstance(size_value, (int, float)) else 0)
                    else:
                        sizes.append(60)  # Default size if not using size_by
        
        if not spectra:
            print(f"No data found for types: {type_keys}")
            return None

        X = np.vstack(spectra)
        original_size = len(X)
        
        # Handle subsampling for very large datasets
        indices = np.arange(len(X))
        if subsample_ratio and 0 < subsample_ratio < 1:
            n_samples = int(len(X) * subsample_ratio)
            np.random.seed(random_state)
            subsample_indices = np.random.choice(len(X), size=n_samples, replace=False)
            X = X[subsample_indices]
            groups = [groups[i] for i in subsample_indices]
            types = [types[i] for i in subsample_indices]
            sizes = [sizes[i] for i in subsample_indices]
            indices = subsample_indices
            print(f"Subsampled {original_size} points to {len(X)} points ({subsample_ratio*100:.1f}%)")
        
        # Determine plotting strategy based on data size
        n_points = len(X)
        use_density = use_density_plot or (n_points > max_points_scatter and not enable_datashader)
        
        if n_points > max_points_scatter:
            print(f"Large dataset detected ({n_points} points). ", end="")
            if enable_datashader:
                print("Using datashader for visualization.")
            elif use_density:
                print("Switching to density plot for better performance.")
            else:
                print("Consider using subsample_ratio or enable_datashader for better performance.")
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            **kwargs
        )
        X_umap = reducer.fit_transform(X)

        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define highly distinct colors for types and varied markers for groups
        unique_types = sorted(set(types))
        unique_groups = sorted(set(groups))
        
        # Highly distinct colors for better visual separation
        distinct_colors = [
            '#FF0000',  # Red
            '#0000FF',  # Blue  
            '#00FF00',  # Green
            '#FF8000',  # Orange
            '#8000FF',  # Purple
            '#FF0080',  # Magenta
            '#00FFFF',  # Cyan
            '#FFFF00',  # Yellow
            '#800000',  # Maroon
            '#000080',  # Navy
            '#008000',  # Dark Green
            '#808000',  # Olive
            '#800080',  # Purple
            '#008080',  # Teal
            '#C0C0C0',  # Silver
            '#808080'   # Gray
        ]
        
        # Varied markers for groups
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x', '8', 'P', 'X', 'H']
        
        # Create color and marker mappings
        type_color_map = {typ: distinct_colors[i % len(distinct_colors)] for i, typ in enumerate(unique_types)}
        group_marker_map = {group: markers[i % len(markers)] for i, group in enumerate(unique_groups)}
        
        # New: Normalize sizes for better visualization if size_by is used
        if size_by and sizes:
            sizes = np.array(sizes)
            if sizes.max() > sizes.min():  # Avoid division by zero
                sizes = 30 + (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 60  # Scale to 30-90 for visibility
            else:
                sizes = np.full_like(sizes, 60)  # Uniform size if all values are the same
        
        # Choose plotting method based on data size and settings
        if enable_datashader and n_points > 1000:
            self._plot_with_datashader(fig, ax, X_umap, types, groups, type_color_map, group_marker_map)
        elif use_density and n_points > max_points_scatter:
            self._plot_with_density(fig, ax, X_umap, types, groups, type_color_map, unique_types)
        else:
            # Standard scatter plot for smaller datasets
            for i, (group, typ) in enumerate(zip(groups, types)):
                ax.scatter(
                    X_umap[i, 0], X_umap[i, 1], 
                    color=type_color_map[typ],
                    marker=group_marker_map[group],
                    s=sizes[i] if size_by else 60,  # Use dynamic size if size_by is provided
                    alpha=0.7, 
                    edgecolor='black', 
                    linewidth=0.5
                )
        
        # Create legend for types (colors)
        type_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=type_color_map[typ], 
                       markersize=10, alpha=0.7, markeredgecolor='black', linewidth=0.5, label=f"Type: {typ}")
            for typ in unique_types
        ]
        
        # Create legend for groups (markers)
        group_handles = [
            plt.Line2D([0], [0], marker=group_marker_map[group], color='gray', 
                       markersize=10, alpha=0.7, markeredgecolor='black', linewidth=0.5, label=str(group))
            for group in unique_groups
        ]
        
        # Combine legends with better positioning
        first_legend = ax.legend(handles=type_handles, loc='upper left', title="Types", 
                                bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
        ax.add_artist(first_legend)
        ax.legend(handles=group_handles, loc='upper right', title=group_by,
                 bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
        
        # New: Add a note to the title if size_by is used
        dynamic_title = title or f"UMAP of {', '.join(type_keys)} colored by type, shaped by {group_by}"
        if size_by:
            dynamic_title += f", sized by {size_by}"
        ax.set_title(dynamic_title)
        
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

        return {
            "umap": reducer,
            "X_umap": X_umap,
            "groups": groups,
            "types": types,
            "sizes": sizes if size_by else None,  # New: Include sizes in return dict
            "figure": fig,
            "axes": ax,
            "type_color_map": type_color_map,
            "group_marker_map": group_marker_map,
            "original_size": original_size,
            "used_size": len(X),
            "indices": indices
        }

    def _plot_with_density(self, fig, ax, X_umap, types, groups, type_color_map, unique_types):
        """
        Plot using density plots for large datasets to improve performance.
        """
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
        
        # Create density plots for each type
        for typ in unique_types:
            typ_mask = np.array(types) == typ
            if np.sum(typ_mask) == 0:
                continue
                
            typ_points = X_umap[typ_mask]
            if len(typ_points) < 2:
                # Fallback to scatter for very few points
                ax.scatter(typ_points[:, 0], typ_points[:, 1], 
                          color=type_color_map[typ], alpha=0.7, s=30, label=f"Type: {typ}")
                continue
            
            # Create 2D histogram/density plot
            try:
                # Use hexbin for density visualization
                hb = ax.hexbin(typ_points[:, 0], typ_points[:, 1], 
                              gridsize=30, alpha=0.6, cmap='Blues', 
                              mincnt=1, label=f"Type: {typ}")
                
                # Add contour lines for better visualization
                x_min, x_max = typ_points[:, 0].min(), typ_points[:, 0].max()
                y_min, y_max = typ_points[:, 1].min(), typ_points[:, 1].max()
                
                if x_max > x_min and y_max > y_min:
                    xx, yy = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    
                    if len(typ_points) > 10:  # Need sufficient points for KDE
                        kernel = gaussian_kde(typ_points.T)
                        f = np.reshape(kernel(positions).T, xx.shape)
                        ax.contour(xx, yy, f, colors=type_color_map[typ], alpha=0.8, linewidths=1)
                        
            except Exception as e:
                print(f"Density plot failed for {typ}, falling back to scatter: {e}")
                # Fallback to downsampled scatter
                downsample = max(1, len(typ_points) // 1000)
                sampled_points = typ_points[::downsample]
                ax.scatter(sampled_points[:, 0], sampled_points[:, 1], 
                          color=type_color_map[typ], alpha=0.5, s=20, label=f"Type: {typ}")

    def _plot_with_datashader(self, fig, ax, X_umap, types, groups, type_color_map, group_marker_map):
        """
        Plot using datashader for very large datasets (requires datashader package).
        """
        try:
            import datashader as ds
            import datashader.transfer_functions as tf
            import pandas as pd
            from datashader.utils import export_image
            
            # Create DataFrame for datashader
            df = pd.DataFrame({
                'x': X_umap[:, 0],
                'y': X_umap[:, 1], 
                'type': types,
                'group': groups
            })
            
            # Set up canvas
            canvas = ds.Canvas(plot_width=800, plot_height=600,
                             x_range=(X_umap[:, 0].min(), X_umap[:, 0].max()),
                             y_range=(X_umap[:, 1].min(), X_umap[:, 1].max()))
            
            # Aggregate by type
            agg = canvas.points(df, 'x', 'y', ds.count_cat('type'))
            
            # Create color mapping for datashader
            color_key = {typ: type_color_map[typ] for typ in set(types)}
            
            # Render image
            img = tf.shade(agg, color_key=color_key, how='eq_hist')
            
            # Display in matplotlib
            extent = [X_umap[:, 0].min(), X_umap[:, 0].max(), 
                     X_umap[:, 1].min(), X_umap[:, 1].max()]
            ax.imshow(img.to_pil(), extent=extent, origin='lower', aspect='auto')
            
            print("Successfully rendered with datashader")
            
        except ImportError:
            print("Datashader not available, falling back to density plot")
            unique_types = sorted(set(types))
            self._plot_with_density(fig, ax, X_umap, types, groups, type_color_map, unique_types)
        except Exception as e:
            print(f"Datashader failed: {e}, falling back to density plot")
            unique_types = sorted(set(types))
            self._plot_with_density(fig, ax, X_umap, types, groups, type_color_map, unique_types)

    def plot_disease_progression_umap(
        self,
        umap_result: dict,
        type_progression_map: dict = None,
        cmap: str = "coolwarm",
        title: str = "Disease Progression Continuum (UMAP)"
    ) -> dict:
        """
        Plot disease progression continuum using UMAP results.
        """
        X_umap = umap_result["X_umap"]
        types = umap_result["types"]

        if type_progression_map is None:
            type_progression_map = {
                "NL": 0.0,
                "MGUS": 0.5,
                "MGUSnew": 0.5,
                "MM": 1.0,
                "MMnew": 1.0
            }

        # Map progression score
        progression_scores = [type_progression_map.get(t, np.nan) for t in types]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            X_umap[:, 0], X_umap[:, 1],
            c=progression_scores,
            cmap=cmap,
            alpha=0.8,
            edgecolor='k',
            s=50,
            linewidth=0.3
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Progression Score (0=Normal, 1=Cancer)")

        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title(title)

        fig.tight_layout()
        plt.show()

        return {
            "figure": fig,
            "axes": ax,
            "progression_scores": progression_scores,
            "type_progression_map": type_progression_map
        }

    def plot_pca_score_distributions(
        self,
        type_keys=("MGUS", "MM"),
        pca_number=6,
        max_points_per_class=None,
        standardize=False,
        random_state=42,
        fdr_correct=True,
        stats_on_full=True,
        title=None,
        **kwargs
    ):
        """
        Overlay KDE/hist plots of PCA scores for two classes with Mann–Whitney U tests.
        - type_keys: tuple of two labels to compare.
        - pca_number: number of PCs to show (kept for backward compatibility).
        - max_points_per_class: optional integer to subsample each class for plotting.
        - standardize: if True, z-score features before PCA; default is mean-centering only.
        - stats_on_full: if True, compute U-test and effect size on all samples even when plotting a subsample.
        - fdr_correct: apply Benjamini–Hochberg correction across PCs.
        Returns: dict with 'p_values', 'p_values_bh', 'cliffs_delta', 'explained_variance', 'pca', 'scores'.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA

        # 1) Extract spectra (uses existing helper)
        X, y, sample_ids, type_labels, _ = self.extract_spectra(type_keys=type_keys, wavelength_sort=True)
        if X is None or len(X) == 0:
            print("No data for the requested types.")
            return {}

        # 2) Center (and optionally standardize) then PCA
        if standardize:
            scaler = StandardScaler(with_mean=True, with_std=True)
        else:
            scaler = StandardScaler(with_mean=True, with_std=False)
        Xc = scaler.fit_transform(X)

        pca = PCA(n_components=min(pca_number, Xc.shape[1]), svd_solver=kwargs.get("svd_solver", "full"), random_state=random_state, **kwargs)
        scores = pca.fit_transform(Xc)  # shape (n_samples, n_components)

        # 3) Prepare class masks and (optional) subsample for plotting
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("This plot currently supports exactly two classes; got {}".format(classes))
        c0, c1 = int(classes[0]), int(classes[1])
        mask0, mask1 = (y == c0), (y == c1)

        def subsample_mask(mask, k):
            idx = np.flatnonzero(mask)
            if k is None or len(idx) <= k:
                return idx
            rng = np.random.default_rng(random_state)
            return rng.choice(idx, size=k, replace=False)

        plot_idx0 = subsample_mask(mask0, max_points_per_class)
        plot_idx1 = subsample_mask(mask1, max_points_per_class)

        # 4) Stats (U test + Cliff's delta); use all or plotted subset
        def cliffs_delta(a, b):
            # Nonparametric effect size; O(n*m) but acceptable for few thousand points
            a = np.asarray(a); b = np.asarray(b)
            gt = np.sum(a[:, None] > b[None, :])
            lt = np.sum(a[:, None] < b[None, :])
            return (gt - lt) / (a.size * b.size + 1e-12)

        p_raw, deltas = [], []
        for k in range(pca.n_components_):
            s0_all = scores[mask0, k]; s1_all = scores[mask1, k]
            s0 = s0_all if stats_on_full else scores[plot_idx0, k]
            s1 = s1_all if stats_on_full else scores[plot_idx1, k]
            try:
                stat, p = mannwhitneyu(s0, s1, alternative="two-sided", **kwargs)
            except ValueError:
                p = 1.0
            p_raw.append(p)
            deltas.append(cliffs_delta(s0, s1))

        # Benjamini–Hochberg FDR
        def bh_adjust(pvals):
            pvals = np.asarray(pvals)
            m = len(pvals)
            order = np.argsort(pvals)
            adj = np.empty(m, dtype=float)
            prev = 1.0
            for rank, idx in enumerate(order[::-1], start=1):
                r = m - rank + 1
                val = min(prev, pvals[idx] * m / r)
                adj[idx] = val
                prev = val
            return np.clip(adj, 0, 1)

        p_bh = bh_adjust(p_raw) if fdr_correct else np.array(p_raw)

        # 5) Plot grid
        ncols = 3
        nrows = int(math.ceil(pca.n_components_ / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
        axes = np.array(axes).reshape(-1)

        name0 = str(type_keys[0]); name1 = str(type_keys[1])
        for k in range(pca.n_components_):
            ax = axes[k]
            s0_plot = scores[plot_idx0, k]
            s1_plot = scores[plot_idx1, k]
            # Density with light hist overlay
            sns.kdeplot(s0_plot, ax=ax, color="#1f77b4", fill=True, alpha=0.35, label=name0)
            sns.kdeplot(s1_plot, ax=ax, color="#ff7f0e", fill=True, alpha=0.35, label=name1)
            ax.hist(s0_plot, bins=40, density=True, color="#1f77b4", alpha=0.15)
            ax.hist(s1_plot, bins=40, density=True, color="#ff7f0e", alpha=0.15)
            ax.set_xlabel(f"PC{k+1} score")
            ax.set_ylabel("Density")
            ax.set_title(f"PC{k+1}: {name0} vs {name1}\nMann–Whitney U p={p_raw[k]:.2e}  (BH={p_bh[k]:.2e}),  δ={deltas[k]:.2f}")
            if k == 0:
                ax.legend(loc="upper right", frameon=True)
            ax.grid(True, alpha=0.25)

        for j in range(pca.n_components_, len(axes)):
            axes[j].axis("off")

        maintitle = title or f"PCA score distributions ({name0} vs {name1})"
        fig.suptitle(maintitle, y=1.02, fontsize=13)
        fig.tight_layout()

        return {
            "figure": fig,
            "axes": axes,
            "p_values": np.array(p_raw),
            "p_values_bh": p_bh,
            "cliffs_delta": np.array(deltas),
            "explained_variance": pca.explained_variance_ratio_,
            "scores": scores,
            "pca": pca,
            "scaler": scaler,
            "plot_indices": (plot_idx0, plot_idx1),
        }

    def _pca_score_density_core(
        self,
        X: np.ndarray,
        y: np.ndarray,
        type_keys=("MGUS", "MM"),
        pca_number=6,
        max_points_per_class=None,
        standardize=False,
        random_state=42,
        fdr_correct=True,
        stats_on_full=True,
        title=None,
        kde_kwargs=None,
        pca_kwargs=None,
        axes_override=None,        # NEW: draw onto provided axes when not None
        xlabel_suffix="",          # NEW: annotate xlabels (e.g., "(BEFORE)" / "(AFTER)")
        legend_first_panel=True,   # NEW: control legend placement
    ):
        """
        Core renderer used by single and comparison plots. If axes_override is a
        flat list of Matplotlib axes with length >= pca_number, the panels are drawn
        directly onto those axes (no artist moving/copying). Otherwise, a new figure
        and axes grid are created.

        Uses scikit-learn PCA for scores and Mann–Whitney U with BH-FDR correction
        to annotate per-PC separation [web:193][web:400][web:216].
        """
        import numpy as np
        import math
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import mannwhitneyu
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        kde_kwargs = kde_kwargs or {}
        pca_kwargs = pca_kwargs or {}

        # 1) Center (and optionally standardize), PCA
        X_safe = self._finite_impute(X)
        scaler = StandardScaler(with_mean=True, with_std=bool(standardize))
        Xc = scaler.fit_transform(X_safe)
        pca = PCA(
            n_components=min(pca_number, Xc.shape[1]),
            random_state=random_state,
            svd_solver=pca_kwargs.get("svd_solver", "full"),
        )
        scores = pca.fit_transform(Xc)

        # 2) Filter to the two classes requested
        y = np.asarray(y)
        keep = (y == type_keys[0]) | (y == type_keys[1])
        scores, y = scores[keep], y[keep]
        if len(np.unique(y)) != 2:
            raise ValueError(f"Exactly two classes required in type_keys, got {np.unique(y)}")

        m0, m1 = (y == type_keys[0]), (y == type_keys[1])

        # Optional subsampling for plotting
        def subsample(mask, n):
            idx = np.flatnonzero(mask)
            if n is None or len(idx) <= n:
                return idx
            rng = np.random.default_rng(random_state)
            return rng.choice(idx, size=n, replace=False)

        id0 = subsample(m0, max_points_per_class)
        id1 = subsample(m1, max_points_per_class)

        # 3) Statistics: Mann–Whitney U + Cliff’s delta + BH
        def cliffs_delta(a, b):
            a = np.asarray(a); b = np.asarray(b)
            gt = np.sum(a[:, None] > b[None, :])
            lt = np.sum(a[:, None] < b[None, :])
            return (gt - lt) / (a.size * b.size + 1e-12)

        pvals, deltas = [], []
        for k in range(pca.n_components_):
            a_all, b_all = scores[m0, k], scores[m1, k]
            a = a_all if stats_on_full else scores[id0, k]
            b = b_all if stats_on_full else scores[id1, k]
            try:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                p = 1.0
            pvals.append(p)
            deltas.append(cliffs_delta(a, b))

        def bh_adjust(p):
            p = np.asarray(p)
            m = len(p)
            order = np.argsort(p)
            adj = np.empty(m, float)
            prev = 1.0
            for rank, idx in enumerate(order[::-1], start=1):
                r = m - rank + 1
                val = min(prev, p[idx] * m / r)
                adj[idx] = val
                prev = val
            return np.clip(adj, 0, 1)

        p_bh = bh_adjust(pvals) if fdr_correct else np.array(pvals)

        # 4) Make / use axes
        if axes_override is None:
            ncols = 3
            nrows = int(math.ceil(pca.n_components_ / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
            axes = np.array(axes).reshape(-1)
            owns_fig = True
        else:
            axes = axes_override
            if len(axes) < pca.n_components_:
                raise ValueError("axes_override not long enough for requested pca_number")
            fig = None
            owns_fig = False

        # 5) Draw directly onto target axes (no artist removal)
        import seaborn as sns
        for k in range(pca.n_components_):
            ax = axes[k]
            s0 = scores[id0, k]
            s1 = scores[id1, k]
            sns.kdeplot(s0, ax=ax, color="#1f77b4", fill=True, alpha=0.35, label=str(type_keys[0]), **kde_kwargs)
            sns.kdeplot(s1, ax=ax, color="#ff7f0e", fill=True, alpha=0.35, label=str(type_keys[1]), **kde_kwargs)
            ax.hist(s0, bins=40, density=True, color="#1f77b4", alpha=0.15)
            ax.hist(s1, bins=40, density=True, color="#ff7f0e", alpha=0.15)
            ax.set_xlabel(f"PC{k+1} score {xlabel_suffix}".strip())
            ax.set_ylabel("Density")
            ax.set_title(f"PC{k+1}: U p={pvals[k]:.2e} (BH={p_bh[k]:.2e}),  δ={deltas[k]:.2f}")
            ax.grid(True, alpha=0.25)
            if legend_first_panel and k == 0:
                ax.legend(loc="upper right", frameon=True)

        # Clear any unused axes
        for j in range(pca.n_components_, len(axes)):
            axes[j].axis("off")

        if owns_fig and title:
            fig.suptitle(title, y=1.02, fontsize=13)
            fig.tight_layout()

        return {
            "figure": fig,
            "axes": axes,
            "p_values": np.array(pvals),
            "p_values_bh": np.array(p_bh),
            "cliffs_delta": np.array(deltas),
            "explained_variance": pca.explained_variance_ratio_,
            "scores": scores,
            "pca": pca,
        }

    def plot_harmonization_comparison(self, 
                                    data_split: Dict[str, Any], 
                                    comparison_type: str = 'both',
                                    figsize: Tuple[int, int] = (15, 10),
                                    save_path: Optional[str] = None,
                                    **kwargs) -> Tuple[plt.Figure, np.ndarray]:
        """
        Comprehensive before/after harmonization comparison using PCA and UMAP.
        
        Args:
            data_split: Dictionary containing processed data from RamanDataPreparer
            comparison_type: 'pca', 'umap', or 'both' 
            figsize: Figure size for the plots
            save_path: Optional path to save the figure
            **kwargs: Additional parameters for PCA/UMAP
            
        Returns:
            Tuple of (figure, axes) objects
        """
        print("=== Creating Harmonization Comparison Visualization ===")
        
        # Extract data
        X_original = data_split.get('X_train_original')
        X_harmonized = data_split.get('X_train')
        y_train = data_split.get('y_train', [])
        batch_labels = data_split.get('batch_train', [])
        
        if X_original is None or X_harmonized is None:
            raise ValueError("Original and harmonized data must be available in data_split")
        
        # Determine plot layout
        if comparison_type == 'both':
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.flatten()
            pca_axes = [axes[0], axes[1]]
            umap_axes = [axes[2], axes[3]]
        elif comparison_type == 'pca':
            fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]//2))
            pca_axes = axes
            umap_axes = None
        elif comparison_type == 'umap':
            fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]//2))
            umap_axes = axes
            pca_axes = None
        else:
            raise ValueError("comparison_type must be 'pca', 'umap', or 'both'")
        
        # Apply PCA comparison if requested
        if pca_axes is not None:
            self._plot_pca_comparison(
                X_original, X_harmonized, y_train, batch_labels, pca_axes, **kwargs
            )
        
        # Apply UMAP comparison if requested
        if umap_axes is not None:
            self._plot_umap_comparison(
                X_original, X_harmonized, y_train, batch_labels, umap_axes, **kwargs
            )
        
        # Add main title and adjust layout
        fig.suptitle('Batch Effect Harmonization: Before vs After Comparison', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        return fig, axes

    def _plot_pca_comparison(self, 
                            X_original: np.ndarray, 
                            X_harmonized: np.ndarray, 
                            y_labels: List[str], 
                            batch_labels: List[str], 
                            axes: List[plt.Axes],
                            n_components: int = 2,
                            **kwargs) -> None:
        """Plot PCA comparison before and after harmonization."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Standardize data
        scaler_orig = StandardScaler()
        scaler_harm = StandardScaler()
        
        X_orig_scaled = scaler_orig.fit_transform(X_original)
        X_harm_scaled = scaler_harm.fit_transform(X_harmonized)
        
        # Apply PCA
        pca_orig = PCA(n_components=n_components)
        pca_harm = PCA(n_components=n_components)
        
        X_orig_pca = pca_orig.fit_transform(X_orig_scaled)
        X_harm_pca = pca_harm.fit_transform(X_harm_scaled)
        
        # Create color maps
        unique_types = np.unique(y_labels)
        unique_batches = np.unique(batch_labels)
        
        type_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_types)))
        batch_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # Plot original data
        ax1 = axes[0]
        for i, data_type in enumerate(unique_types):
            for j, batch in enumerate(unique_batches):
                mask = (np.array(y_labels) == data_type) & (np.array(batch_labels) == batch)
                if np.any(mask):
                    ax1.scatter(X_orig_pca[mask, 0], X_orig_pca[mask, 1], 
                            c=[type_colors[i]], marker=batch_markers[j % len(batch_markers)],
                            s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                            label=f'{data_type} (Batch {batch})' if i == 0 or j == 0 else "")
        
        ax1.set_title('Before Harmonization (Original Data)', fontweight='bold')
        ax1.set_xlabel(f'PC1 ({pca_orig.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca_orig.explained_variance_ratio_[1]:.1%} variance)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot harmonized data
        ax2 = axes[1]
        for i, data_type in enumerate(unique_types):
            for j, batch in enumerate(unique_batches):
                mask = (np.array(y_labels) == data_type) & (np.array(batch_labels) == batch)
                if np.any(mask):
                    ax2.scatter(X_harm_pca[mask, 0], X_harm_pca[mask, 1], 
                            c=[type_colors[i]], marker=batch_markers[j % len(batch_markers)],
                            s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                            label=f'{data_type} (Batch {batch})' if i == 0 or j == 0 else "")
        
        ax2.set_title('After Harmonization (Batch-Corrected)', fontweight='bold')
        ax2.set_xlabel(f'PC1 ({pca_harm.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca_harm.explained_variance_ratio_[1]:.1%} variance)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

    def _plot_umap_comparison(self, 
                            X_original: np.ndarray, 
                            X_harmonized: np.ndarray, 
                            y_labels: List[str], 
                            batch_labels: List[str], 
                            axes: List[plt.Axes],
                            n_components: int = 2,
                            n_neighbors: int = 15,
                            min_dist: float = 0.1,
                            **kwargs) -> None:
        """Plot UMAP comparison before and after harmonization."""
        import umap.umap_ as umap
        from sklearn.preprocessing import StandardScaler
        
        # Standardize data
        scaler_orig = StandardScaler()
        scaler_harm = StandardScaler()
        
        X_orig_scaled = scaler_orig.fit_transform(X_original)
        X_harm_scaled = scaler_harm.fit_transform(X_harmonized)
        
        # Apply UMAP
        umap_orig = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                            min_dist=min_dist, random_state=42)
        umap_harm = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                            min_dist=min_dist, random_state=42)
        
        print("Computing UMAP embeddings...")
        X_orig_umap = umap_orig.fit_transform(X_orig_scaled)
        X_harm_umap = umap_harm.fit_transform(X_harm_scaled)
        
        # Create color maps
        unique_types = np.unique(y_labels)
        unique_batches = np.unique(batch_labels)
        
        type_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_types)))
        batch_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # Plot original data
        ax1 = axes[0]
        for i, data_type in enumerate(unique_types):
            for j, batch in enumerate(unique_batches):
                mask = (np.array(y_labels) == data_type) & (np.array(batch_labels) == batch)
                if np.any(mask):
                    ax1.scatter(X_orig_umap[mask, 0], X_orig_umap[mask, 1], 
                            c=[type_colors[i]], marker=batch_markers[j % len(batch_markers)],
                            s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                            label=f'{data_type} (Batch {batch})' if i == 0 or j == 0 else "")
        
        ax1.set_title('Before Harmonization (Original Data)', fontweight='bold')
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot harmonized data
        ax2 = axes[1]
        for i, data_type in enumerate(unique_types):
            for j, batch in enumerate(unique_batches):
                mask = (np.array(y_labels) == data_type) & (np.array(batch_labels) == batch)
                if np.any(mask):
                    ax2.scatter(X_harm_umap[mask, 0], X_harm_umap[mask, 1], 
                            c=[type_colors[i]], marker=batch_markers[j % len(batch_markers)],
                            s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                            label=f'{data_type} (Batch {batch})' if i == 0 or j == 0 else "")
        
        ax2.set_title('After Harmonization (Batch-Corrected)', fontweight='bold')
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

    def plot_batch_effect_metrics(self, 
                                data_split: Dict[str, Any],
                                figsize: Tuple[int, int] = (12, 8),
                                save_path: Optional[str] = None) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot quantitative batch effect metrics before and after harmonization.
        
        Args:
            data_split: Dictionary containing processed data from RamanDataPreparer
            figsize: Figure size for the plots
            save_path: Optional path to save the figure
            
        Returns:
            Tuple of (figure, axes) objects
        """
        from scipy.stats import f_oneway
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import LabelEncoder
        
        print("=== Computing Batch Effect Metrics ===")
        
        X_original = data_split.get('X_train_original')
        X_harmonized = data_split.get('X_train')
        y_train = data_split.get('y_train', [])
        batch_labels = data_split.get('batch_train', [])
        
        if X_original is None or X_harmonized is None:
            raise ValueError("Original and harmonized data must be available")
        
        # Encode labels for metrics calculation
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_train)
        batch_encoded = LabelEncoder().fit_transform(batch_labels)
        
        # Calculate metrics
        metrics_data = []
        
        # 1. Silhouette Score (higher is better for disease types)
        sil_original_disease = silhouette_score(X_original, y_encoded)
        sil_harmonized_disease = silhouette_score(X_harmonized, y_encoded)
        sil_original_batch = silhouette_score(X_original, batch_encoded)
        sil_harmonized_batch = silhouette_score(X_harmonized, batch_encoded)
        
        metrics_data.append({
            'metric': 'Disease Separation\n(Silhouette Score)',
            'original': sil_original_disease,
            'harmonized': sil_harmonized_disease,
            'better_direction': 'higher'
        })
        
        metrics_data.append({
            'metric': 'Batch Separation\n(Silhouette Score)',
            'original': sil_original_batch,
            'harmonized': sil_harmonized_batch,
            'better_direction': 'lower'
        })
        
        # 2. F-statistic for batch effect (lower is better)
        unique_batches = np.unique(batch_labels)
        if len(unique_batches) > 1:
            # Calculate F-statistic for each feature, then take mean
            f_stats_original = []
            f_stats_harmonized = []
            
            for feature_idx in range(min(100, X_original.shape[1])):  # Sample features
                groups_orig = [X_original[np.array(batch_labels) == batch, feature_idx] 
                            for batch in unique_batches]
                groups_harm = [X_harmonized[np.array(batch_labels) == batch, feature_idx] 
                            for batch in unique_batches]
                
                f_stat_orig, _ = f_oneway(*groups_orig)
                f_stat_harm, _ = f_oneway(*groups_harm)
                
                if not np.isnan(f_stat_orig):
                    f_stats_original.append(f_stat_orig)
                if not np.isnan(f_stat_harm):
                    f_stats_harmonized.append(f_stat_harm)
            
            mean_f_orig = np.mean(f_stats_original) if f_stats_original else 0
            mean_f_harm = np.mean(f_stats_harmonized) if f_stats_harmonized else 0
            
            metrics_data.append({
                'metric': 'Batch Effect\n(F-statistic)',
                'original': mean_f_orig,
                'harmonized': mean_f_harm,
                'better_direction': 'lower'
            })
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot metrics comparison
        ax1 = axes[0]
        metric_names = [m['metric'] for m in metrics_data]
        original_values = [m['original'] for m in metrics_data]
        harmonized_values = [m['harmonized'] for m in metrics_data]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, original_values, width, label='Original', alpha=0.8, color='lightcoral')
        bars2 = ax1.bar(x + width/2, harmonized_values, width, label='Harmonized', alpha=0.8, color='lightblue')
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Batch Effect Correction Metrics')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        
        # Plot improvement percentages
        ax2 = axes[1]
        improvements = []
        colors = []
        
        for m in metrics_data:
            if m['original'] != 0:
                if m['better_direction'] == 'higher':
                    improvement = ((m['harmonized'] - m['original']) / abs(m['original'])) * 100
                else:  # lower is better
                    improvement = ((m['original'] - m['harmonized']) / abs(m['original'])) * 100
                
                improvements.append(improvement)
                colors.append('green' if improvement > 0 else 'red')
        
        bars = ax2.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Harmonization Improvement')
        ax2.set_xticks(range(len(metric_names)))
        ax2.set_xticklabels(metric_names, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, improvement) in enumerate(zip(bars, improvements)):
            height = bar.get_height()
            ax2.annotate(f'{improvement:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height > 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height > 0 else 'top', 
                        fontsize=9, fontweight='bold')
        
        # Plot variance explained by batch (before/after)
        ax3 = axes[2]
        self._plot_variance_explained_by_batch(X_original, X_harmonized, batch_labels, ax3)
        
        # Plot mean intensity distributions by batch
        ax4 = axes[3]
        self._plot_batch_intensity_distributions(X_original, X_harmonized, batch_labels, ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics figure saved to: {save_path}")
        
        plt.show()
        return fig, axes

    def _plot_variance_explained_by_batch(self, X_original, X_harmonized, batch_labels, ax):
        """Plot variance explained by batch factor."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        # Standardize data
        scaler_orig = StandardScaler()
        scaler_harm = StandardScaler()
        
        X_orig_scaled = scaler_orig.fit_transform(X_original)
        X_harm_scaled = scaler_harm.fit_transform(X_harmonized)
        
        # Encode batch labels
        batch_encoder = LabelEncoder()
        batch_encoded = batch_encoder.fit_transform(batch_labels)
        
        # Calculate variance explained by first 10 PCs
        pca_orig = PCA(n_components=min(10, X_original.shape[1]))
        pca_harm = PCA(n_components=min(10, X_harmonized.shape[1]))
        
        pca_orig.fit(X_orig_scaled)
        pca_harm.fit(X_harm_scaled)
        
        pc_range = range(1, len(pca_orig.explained_variance_ratio_) + 1)
        
        ax.plot(pc_range, np.cumsum(pca_orig.explained_variance_ratio_), 
                'o-', label='Original', linewidth=2, markersize=6)
        ax.plot(pc_range, np.cumsum(pca_harm.explained_variance_ratio_), 
                's-', label='Harmonized', linewidth=2, markersize=6)
        
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Cumulative Variance Explained')
        ax.set_title('Cumulative Variance Explained')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    def _plot_batch_intensity_distributions(self, X_original, X_harmonized, batch_labels, ax):
        """Plot mean intensity distributions by batch."""
        unique_batches = np.unique(batch_labels)
        
        # Calculate mean intensities per batch
        means_original = []
        means_harmonized = []
        
        for batch in unique_batches:
            mask = np.array(batch_labels) == batch
            means_original.append(np.mean(X_original[mask]))
            means_harmonized.append(np.mean(X_harmonized[mask]))
        
        x = np.arange(len(unique_batches))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, means_original, width, label='Original', 
                    alpha=0.8, color='lightcoral')
        bars2 = ax.bar(x + width/2, means_harmonized, width, label='Harmonized', 
                    alpha=0.8, color='lightblue')
        
        ax.set_xlabel('Batch')
        ax.set_ylabel('Mean Intensity')
        ax.set_title('Mean Intensity by Batch')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Batch {b}' for b in unique_batches])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)    

    ## MACHINE LEARNING EVALUATION SECTION ##
    
    def build_metrics_summary(self, res: dict) -> pd.DataFrame:
        """
        Create a compact table with key metrics including data split information.
        """
        m = res["metrics"]
        
        # Calculate data split counts
        train_count = len(res["sample_ids_train"])
        test_count = len(res["sample_ids_test"])
        total_count = train_count + test_count
        train_ratio = train_count / total_count if total_count > 0 else 0
        test_ratio = test_count / total_count if total_count > 0 else 0
        
        rows = [{
            "Train Samples": train_count,
            "Test Samples": test_count,
            "Train Ratio": f"{train_ratio:.1%}",
            "Test Ratio": f"{test_ratio:.1%}",
            "Accuracy": m["accuracy"],
            "Balanced Acc": m["balanced_accuracy"],
            "Macro F1": m["macro_f1"],
            "Weighted F1": m["weighted_f1"],
            "Macro Precision": m["macro_precision"],
            "Macro Recall": m["macro_recall"],
            "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
            "MCC": m["matthews_corrcoef"],
            "Log Loss": m["log_loss"],
            "CV Mean": res["cv_scores_mean"],
            "CV Std": res["cv_scores_std"]
        }]
        return pd.DataFrame(rows)

    def plot_pca_results(self, res: dict, X_original: np.ndarray, y_original: np.ndarray, 
                        sample_ids_original: np.ndarray = None, n_components: int = 2, 
                        title_prefix: str = "Classification Results"):
        """
        Plot PCA visualization of classification results showing train/test split and predictions.
        """
        from sklearn.decomposition import PCA
        
        # Get indices for train and test sets
        train_indices = res["train_indices"]
        test_indices = res["test_indices"]
        
        # Fit PCA on all data
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_original)
        
        # Get class names
        class_names = res.get("class_names", ["Class 0", "Class 1"])
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Train/Test Split Visualization
        ax1 = axes[0]
        train_colors = ['lightblue', 'lightcoral']
        test_colors = ['darkblue', 'darkred']
        
        # Plot training and test data
        for i, class_name in enumerate(class_names):
            train_mask = (y_original == i) & np.isin(np.arange(len(X_original)), train_indices)
            test_mask = (y_original == i) & np.isin(np.arange(len(X_original)), test_indices)
            
            if np.any(train_mask):
                ax1.scatter(X_pca[train_mask, 0], X_pca[train_mask, 1], 
                           c=train_colors[i], marker='o', s=30, alpha=0.7,
                           label=f'{class_name} (Train)')
            
            if np.any(test_mask):
                ax1.scatter(X_pca[test_mask, 0], X_pca[test_mask, 1], 
                           c=test_colors[i], marker='s', s=40, alpha=0.8,
                           label=f'{class_name} (Test)')
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.set_title(f'{title_prefix}: Train/Test Split')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Results (Test Set Only)
        ax2 = axes[1]
        
        # Get test set data
        X_test_pca = X_pca[test_indices]
        y_test = res["y_test"]
        y_pred = res["y_pred"]
        
        # Plot correct and incorrect predictions
        correct_mask = y_test == y_pred
        incorrect_mask = ~correct_mask
        
        if np.any(correct_mask):
            for i, class_name in enumerate(class_names):
                class_correct = correct_mask & (y_test == i)
                if np.any(class_correct):
                    ax2.scatter(X_test_pca[class_correct, 0], X_test_pca[class_correct, 1],
                               c=test_colors[i], marker='o', s=50, alpha=0.8,
                               label=f'{class_name} (Correct)')
        
        if np.any(incorrect_mask):
            ax2.scatter(X_test_pca[incorrect_mask, 0], X_test_pca[incorrect_mask, 1],
                       c='red', marker='x', s=60, alpha=0.9,
                       label='Misclassified')
        
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax2.set_title(f'{title_prefix}: Test Set Predictions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes, pca

    def plot_sample_distribution(self, res: dict, X_original: np.ndarray = None, y_original: np.ndarray = None, class_names: list = None):
        """
        Plot accurate sample distribution by class for train/test sets using original data.
        """
        train_indices = res["train_indices"]
        test_indices = res["test_indices"]
        
        if class_names is None:
            class_names = res.get("class_names", ["Class 0", "Class 1"])
        
        # Calculate exact counts if original data provided
        if X_original is not None and y_original is not None:
            train_counts = []
            test_counts = []
            
            for i, class_name in enumerate(class_names):
                train_class_count = np.sum((y_original == i) & np.isin(np.arange(len(X_original)), train_indices))
                test_class_count = np.sum((y_original == i) & np.isin(np.arange(len(X_original)), test_indices))
                train_counts.append(train_class_count)
                test_counts.append(test_class_count)
        else:
            # Fallback: approximate counts based on total samples
            total_train = len(res["sample_ids_train"])
            total_test = len(res["sample_ids_test"])
            num_classes = len(class_names)
            
            train_counts = [total_train // num_classes] * num_classes
            test_counts = [total_test // num_classes] * num_classes
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(class_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_counts, width, label='Training Set', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, test_counts, width, label='Test Set', alpha=0.8, color='orange')
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Sample Distribution by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        plt.tight_layout()
        return fig, ax

    def plot_confusion_matrices(self, res: dict, cmap="Blues"):
        """Plot confusion matrix (raw counts and normalized)."""
        cm = res["metrics"]["confusion_matrix"]
        cm_norm = res["metrics"]["confusion_matrix_normalized"]
        class_names = res.get("class_names", ["0", "1"])
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=axes[0],
                    xticklabels=class_names, yticklabels=class_names)
        axes[0].set_title("Confusion Matrix (Counts)")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap=cmap, ax=axes[1],
                    xticklabels=class_names, yticklabels=class_names)
        axes[1].set_title("Confusion Matrix (Normalized)")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")
        
        plt.tight_layout()
        return fig, axes

    def plot_roc_pr(self, res: dict):
        """Plot ROC and Precision-Recall curves."""
        y_test = res["y_test"]
        y_prob = res["y_prob"]
        class_names = res.get("class_names")
        
        if y_prob is None:
            return None, None
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        if y_prob.ndim == 1:  # Binary classification
            RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0], name="Positive Class")
            PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[1], name="Positive Class")
        else:  # Multi-class
            classes = np.unique(y_test)
            for i, c in enumerate(classes):
                name = class_names[c] if class_names and c < len(class_names) else f"Class {c}"
                RocCurveDisplay.from_predictions((y_test == c).astype(int), y_prob[:, i], ax=axes[0], name=name)
                PrecisionRecallDisplay.from_predictions((y_test == c).astype(int), y_prob[:, i], ax=axes[1], name=name)
            axes[0].plot([0, 1], [0, 1], "k--", lw=0.6, label="Random")
        
        axes[0].set_title("ROC Curve")
        axes[1].set_title("Precision-Recall Curve")
        plt.tight_layout()
        return fig, axes

    def plot_feature_importance(self, res: dict, wavelengths: np.ndarray = None, top: int = 20):
        """
        Bar plot of top feature importances with wavelength labels if provided.
        """
        fi = res.get("feature_importance")
        if fi is None or fi.empty:
            print("No feature importance data available.")
            return None, None
        
        df = fi.head(top).copy()
        labels = df["feature_index"].astype(str)
        
        # Use wavelengths as labels if provided
        if wavelengths is not None and len(wavelengths) > df["feature_index"].max():
            labels = [f"{wavelengths[idx]:.0f} nm" for idx in df["feature_index"]]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(df)), df["abs_weight"][::-1], color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(labels[::-1])
        ax.set_title(f"Top {len(df)} Feature Importances")
        ax.set_xlabel("Absolute Weight")
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig, ax

    def plot_cv_distribution(self, res: dict):
        """Plot cross-validation score distribution."""
        scores = res.get("cv_scores")
        if scores is None:
            print("No cross-validation scores available.")
            return None, None
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=scores, ax=ax, orient="h", color='lightblue')
        sns.stripplot(x=scores, color="darkblue", size=6, ax=ax, orient="h", alpha=0.7)
        
        ax.set_title(f"Cross-Validation Scores (Mean: {scores.mean():.3f} ± {scores.std():.3f})")
        ax.set_xlabel("Score")
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig, ax

    def show_model_report(
        self,
        res: dict,
        X_original: np.ndarray = None,
        y_original: np.ndarray = None,
        sample_ids_original: np.ndarray = None,
        wavelengths: np.ndarray = None,
        top_features: int = 20,
        show_text: bool = True,
        show_pca: bool = True,
        show_confusion_matrix: bool = True,
        show_roc_pr: bool = True,
        show_feature_importance: bool = True,
        show_cv_distribution: bool = True
    ):
        """
        Comprehensive model performance report with visualizations.
        """
        if show_text:
            summary_df = self.build_metrics_summary(res)
            display(Markdown("### Model Training Summary"))
            display(summary_df)

            display(Markdown("### Per-Class Performance"))
            display(res["metrics"]["per_class_table"])
        
        # Sample distribution plot
        self.plot_sample_distribution(res, X_original, y_original, res.get("class_names"))
        
        # Confusion matrices
        if show_confusion_matrix:
            self.plot_confusion_matrices(res)
        
        # ROC and PR curves
        if show_roc_pr:
            self.plot_roc_pr(res)

        # Feature importance
        if show_feature_importance:
            self.plot_feature_importance(res, wavelengths=wavelengths, top=top_features)

        # Cross-validation scores
        if show_cv_distribution:
            self.plot_cv_distribution(res)

        # PCA visualization
        if show_pca and X_original is not None and y_original is not None:
            sample_ids_to_use = sample_ids_original if sample_ids_original is not None else np.arange(len(X_original))
            self.plot_pca_results(res, X_original, y_original, sample_ids_to_use)
        
        if show_text:
            print("\n=== DETAILED CLASSIFICATION REPORT ===")
            print(res["metrics"]["classification_report_text"])

    def log_detail_model_result(self, res, X=None, y=None, sample_ids=None, wavelengths=None, **kwargs):
        """
        Enhanced model result logger with comprehensive visualizations.
        """
        self.show_model_report(
            res, 
            X_original=X, 
            y_original=y, 
            sample_ids_original=sample_ids, 
            wavelengths=wavelengths, 
            show_text=True, 
            show_pca=True,
            show_confusion_matrix=kwargs.get("show_confusion_matrix", True),
            show_roc_pr=kwargs.get("show_roc_pr", True),
            show_feature_importance=kwargs.get("show_feature_importance", True),
            show_cv_distribution=kwargs.get("show_cv_distribution", True)
        )



class MLVisualize:
    """
    A dynamic and robust visualization class for machine learning models.
    
    This class provides visualization methods that can be used with any trained ML model,
    including decision boundary plots, confusion matrices, and other model evaluation graphics.
    It is designed to be model-agnostic and flexible for different types of models and data.
    """
    
    def __init__(self, model, data_split: Dict[str, Any], label_encoder=None):
        """
        Initialize the MLVisualize class.
        
        Args:
            model: Trained machine learning model (must have predict method)
            data_split (dict): Data dictionary containing X_train, y_train, etc.
            label_encoder: Label encoder for decoding predictions (optional)
        """
        self.model = model
        self.data_split = data_split
        self.label_encoder = label_encoder
        
        # Validate required keys in data_split
        required_keys = ['X_train', 'y_train']
        missing_keys = [key for key in required_keys if key not in self.data_split]
        if missing_keys:
            raise ValueError(f"Missing required keys in data_split: {missing_keys}")
    
    def plot_decision_boundary_2d(self, X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None, 
                             X_new: Optional[np.ndarray] = None, y_new: Optional[np.ndarray] = None, 
                             show_plot: bool = True, n_components: int = 2, 
                             dim_reduction: str = 'pca', batch_size: int = 512, **kwargs) -> Dict[str, Any]:
        """
        Plot the decision boundary of the model in 2D using dimensionality reduction.
        
        This method is model-agnostic and works with any ML model that has a `predict` method,
        including KNN, Random Forest, SVM, Neural Networks, etc. For classification models,
        it uses `predict_proba` if available for better boundary visualization.
        
        Supports PCA and UMAP for dimensionality reduction. t-SNE is also supported but skips
        decision boundary visualization since it doesn't have inverse transform.
        
        Args:
            X_test (np.ndarray, optional): Test feature matrix. If None, uses data_split['X_test'] if available.
            y_test (np.ndarray, optional): Test labels. If None, uses data_split['y_test'] if available.
            X_new (np.ndarray, optional): New feature matrix for prediction
            y_new (np.ndarray, optional): New labels
            show_plot (bool): Whether to display the plot
            n_components (int): Number of components for dimensionality reduction (default 2)
            dim_reduction (str): Dimensionality reduction method ('pca', 'umap', 'tsne') (default 'pca')
            batch_size (int): Batch size for memory-efficient prediction (default 512)
            **kwargs: Additional keyword arguments passed to the dimensionality reduction method
        
        Returns:
            Dict[str, Any]: Dictionary containing plot data, transformed data, and other relevant information
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import umap
        
        # Use test data from data_split if not provided
        if X_test is None:
            X_test = self.data_split.get('X_test')
        if y_test is None:
            y_test = self.data_split.get('y_test')
        
        # Choose dimensionality reduction method
        if dim_reduction.lower() == 'pca':
            reducer = PCA(n_components=n_components, **kwargs)
            X_train_2d = reducer.fit_transform(self.data_split['X_train'])
            explained_variance = reducer.explained_variance_ratio_
            has_inverse = True
            has_transform = True
        elif dim_reduction.lower() == 'umap':
            reducer = umap.UMAP(n_components=n_components, **kwargs)
            X_train_2d = reducer.fit_transform(self.data_split['X_train'])
            explained_variance = None
            has_inverse = hasattr(reducer, 'inverse_transform')
            has_transform = True
        elif dim_reduction.lower() == 'tsne':
            # t-SNE doesn't support inverse transform or transform, so fit on combined data
            tsne_kwargs = kwargs.copy()
            tsne_kwargs.setdefault('random_state', 42)
            
            # Combine all data for t-SNE fitting
            data_list = [self.data_split['X_train']]
            indices = {'train': (0, len(self.data_split['X_train']))}
            if X_test is not None:
                data_list.append(X_test)
                indices['test'] = (len(data_list[-2]), len(data_list[-2]) + len(X_test))
            if X_new is not None:
                data_list.append(X_new)
                indices['new'] = (len(data_list[-2]), len(data_list[-2]) + len(X_new))
            
            X_combined = np.vstack(data_list)
            reducer = TSNE(n_components=n_components, **tsne_kwargs)
            X_combined_2d = reducer.fit_transform(X_combined)
            
            # Split back into train, test, new
            X_train_2d = X_combined_2d[indices['train'][0]:indices['train'][1]]
            X_test_2d = X_combined_2d[indices['test'][0]:indices['test'][1]] if 'test' in indices else None
            X_new_2d = X_combined_2d[indices['new'][0]:indices['new'][1]] if 'new' in indices else None
            
            explained_variance = None
            has_inverse = False
            has_transform = False
        else:
            raise ValueError("dim_reduction must be 'pca', 'umap', or 'tsne'")
        
        # Encode labels if label_encoder is available
        if self.label_encoder:
            y_train_encoded = self.label_encoder.fit_transform(self.data_split['y_train'])
            class_names = self.label_encoder.classes_
        else:
            y_train_encoded = self.data_split['y_train']
            class_names = [f'Class {i}' for i in np.unique(y_train_encoded)]
        
        # Create a mesh grid for the decision boundary (only if invertible)
        if has_inverse:
            x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
            y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                np.arange(y_min, y_max, 0.1))
            
            # Transform mesh back to high-dimensional space
            mesh_2d = np.c_[xx.ravel(), yy.ravel()]
            if n_components > 2:
                # Pad with zeros for additional components if needed
                mesh_2d = np.hstack([mesh_2d, np.zeros((mesh_2d.shape[0], n_components - 2))])
            mesh_hd = reducer.inverse_transform(mesh_2d)
            
            # Memory-efficient prediction in batches
            Z_list = []
            print(f"Processing mesh grid in batches (size: {batch_size}) for memory efficiency...")
            for i in range(0, len(mesh_hd), batch_size):
                batch_mesh = mesh_hd[i:i+batch_size]
                
                # Check if model has batched prediction method (like DANN)
                if hasattr(self.model, 'predict') and hasattr(self.model, 'device'):
                    # For PyTorch models like DANN
                    if hasattr(self.model, 'predict_labels'):
                        # Use the memory-efficient predict method
                        batch_pred = self.model.predict(batch_mesh, batch_size=batch_size)
                    else:
                        batch_pred = self.model.predict(batch_mesh)
                else:
                    # Standard sklearn-style prediction
                    batch_pred = self.model.predict(batch_mesh)
                
                Z_list.append(batch_pred)
                
                # Clear GPU cache if CUDA is available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            
            Z = np.concatenate(Z_list)
            
            # Handle different model types
            if hasattr(self.model, 'predict_proba') and self.label_encoder:
                # For models with predict_proba, get probabilities in batches
                Z_prob_list = []
                for i in range(0, len(mesh_hd), batch_size):
                    batch_mesh = mesh_hd[i:i+batch_size]
                    batch_prob = self.model.predict_proba(batch_mesh)
                    Z_prob_list.append(batch_prob)
                Z_prob = np.vstack(Z_prob_list)
                
                if Z_prob.shape[1] == 2:
                    Z = Z_prob[:, 1]  # Probability of positive class
                else:
                    Z = np.argmax(Z_prob, axis=1).astype(float)
            elif hasattr(self.model, 'classes_') and len(self.model.classes_) > 2:
                # Multi-class classification
                Z = Z.astype(float)
            else:
                # Regression or binary classification without probabilities
                Z = Z.astype(float)
            
            Z = Z.reshape(xx.shape)
        else:
            # For non-invertible methods, skip boundary and mesh
            xx, yy, Z = None, None, None
            print(f"Warning: {dim_reduction.upper()} does not support inverse transform. Skipping decision boundary visualization.")
        
        # Plot the decision boundary and regions
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if has_inverse:
            # Shade decision regions
            if len(np.unique(y_train_encoded)) == 2:
                ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, cmap='RdYlBu')
                ax.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='red', alpha=0.8)
            else:
                # For multi-class, shade regions
                levels = np.linspace(Z.min(), Z.max(), len(class_names) + 1)
                ax.contourf(xx, yy, Z, levels=levels, alpha=0.3, cmap='plasma')
        
        # Use distinct colors for classes
        colors = ['blue', 'orange'] if len(class_names) == 2 else ['blue', 'orange', 'green', 'red', 'purple']
        
        # Plot training data with distinct colors and shapes
        for i, class_name in enumerate(class_names):
            mask = y_train_encoded == i
            ax.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1], 
                    c=colors[i % len(colors)], marker='o', s=50, alpha=0.7, 
                    label=f'{class_name} (Train)', edgecolor='k')
        
        # Prepare data for return
        result = {
            'reducer': reducer,
            'X_train_2d': X_train_2d,
            'y_train_encoded': y_train_encoded,
            'explained_variance': explained_variance,
            'mesh_grid': {'xx': xx, 'yy': yy, 'Z': Z} if has_inverse else None,
            'figure': fig,
            'axes': ax,
            'has_inverse': has_inverse,
            'class_names': class_names
        }
        
        # Plot test data if available
        if X_test is not None and y_test is not None:
            if has_transform and dim_reduction.lower() != 'tsne':
                X_test_2d = reducer.transform(X_test)
            # For t-SNE, X_test_2d is already computed above
            if self.label_encoder:
                y_test_encoded = self.label_encoder.transform(y_test)
            else:
                y_test_encoded = y_test
            for i, class_name in enumerate(class_names):
                mask = y_test_encoded == i
                ax.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1], 
                        c=colors[i % len(colors)], marker='s', s=50, alpha=0.7, 
                        label=f'{class_name} (Test)', edgecolor='k')
            result['X_test_2d'] = X_test_2d
            result['y_test_encoded'] = y_test_encoded
        
        # Plot new data if provided
        if X_new is not None and y_new is not None:
            if has_transform and dim_reduction.lower() != 'tsne':
                X_new_2d = reducer.transform(X_new)
            # For t-SNE, X_new_2d is already computed above
            if self.label_encoder:
                y_new_encoded = self.label_encoder.transform(y_new)
            else:
                y_new_encoded = y_new
            for i, class_name in enumerate(class_names):
                mask = y_new_encoded == i
                ax.scatter(X_new_2d[mask, 0], X_new_2d[mask, 1], 
                        c=colors[i % len(colors)], marker='^', s=50, alpha=0.7, 
                        label=f'{class_name} (New)', edgecolor='k')
            result['X_new_2d'] = X_new_2d
            result['y_new_encoded'] = y_new_encoded
        
        # Add colorbar only if more than 2 classes and label_encoder is available
        if self.label_encoder and len(np.unique(y_train_encoded)) > 2:
            cbar = plt.colorbar(ax.collections[0] if ax.collections else None, ax=ax)
            cbar.set_label('Predicted Class')
            cbar.set_ticks(range(len(class_names)))
            cbar.set_ticklabels(class_names)
            result['colorbar'] = cbar
        
        xlabel = f'{dim_reduction.upper()}-1'
        ylabel = f'{dim_reduction.upper()}-2'
        if explained_variance is not None and len(explained_variance) >= 2:
            xlabel += f' ({explained_variance[0]:.1%} variance)'
            ylabel += f' ({explained_variance[1]:.1%} variance)'
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Model Decision Boundary (2D {dim_reduction.upper()} Projection)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        if show_plot:
            plt.show()
        
        print(f"{dim_reduction.upper()} projection completed.")
        if explained_variance is not None:
            print(f"Explained variance ratio: {explained_variance}")
        if has_inverse:
            print("Decision boundary and regions plotted using original model.")
        else:
            print("Data points plotted without decision boundary (non-invertible method).")
        
        return result
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             title: str = "Confusion Matrix", show_plot: bool = True) -> np.ndarray:
        """
        Plot confusion matrix for model predictions.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            title (str): Plot title
            show_plot (bool): Whether to display the plot
        
        Returns:
            np.ndarray: Confusion matrix
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_ if self.label_encoder else None,
                   yticklabels=self.label_encoder.classes_ if self.label_encoder else None)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
        return cm
    
    def plot_feature_importance(self, feature_names: Optional[List[str]] = None, 
                               top_n: int = 20, title: str = "Feature Importance", 
                               show_plot: bool = True) -> Optional[plt.Figure]:
        """
        Plot feature importance if the model supports it.
        
        Args:
            feature_names (List[str], optional): Names for features
            top_n (int): Number of top features to show
            title (str): Plot title
            show_plot (bool): Whether to display the plot
        
        Returns:
            plt.Figure or None: The figure if model supports feature importance
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0]) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_)
        else:
            print("Model does not support feature importance.")
            return None
        
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), 
                  [feature_names[i] if feature_names else f'Feature {i}' for i in indices])
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
        return plt.gcf()


# ============================================================================
# MULTI-NETWORK ENSEMBLE SYSTEM VISUALIZATIONS
# ============================================================================

class MultiNetworkVisualize:
    '''
    Visualization utilities for Multi-Network Ensemble System.
    
    Based on:
    Kothari et al., Scientific Reports (2021)
    
    Provides comprehensive visualization methods for:
    - Bayesian probability distributions
    - Variance analysis (VRA, VER)
    - Network agreement/disagreement
    - Preprocessing pipeline
    - Feature distributions
    - Training convergence
    '''
    
    @staticmethod
    def plot_preprocessing_pipeline(wavenumbers, intensity_raw, intensity_filtered,
                                    intensity_baseline, baseline, intensity_smooth,
                                    intensity_centered, intensity_normalized,
                                    wavenumbers_filtered=None):
        '''6-panel figure showing preprocessing steps'''
        if wavenumbers_filtered is None:
            wavenumbers_filtered = wavenumbers
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        axes[0].plot(wavenumbers, intensity_raw, 'b-', linewidth=1.5)
        axes[0].set_title('Step 1: Raw Spectrum', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Intensity (a.u.)', fontsize=10)
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(wavenumbers_filtered, intensity_filtered, 'g-', linewidth=1.5)
        axes[1].axvline(600, color='red', linestyle='--', linewidth=2, label='Cutoff')
        axes[1].set_title('Step 2: Range Filtered', fontsize=11, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        axes[2].plot(wavenumbers_filtered, intensity_baseline, 'b-', linewidth=1.5, label='Corrected')
        axes[2].plot(wavenumbers_filtered, baseline, 'r--', linewidth=2, label='Baseline')
        axes[2].set_title('Step 3: Baseline Corrected', fontsize=11, fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        axes[3].plot(wavenumbers_filtered, intensity_smooth, 'purple', linewidth=1.5)
        axes[3].set_title('Step 4: Smoothed', fontsize=11, fontweight='bold')
        axes[3].grid(alpha=0.3)
        
        axes[4].plot(wavenumbers_filtered, intensity_centered, 'orange', linewidth=1.5)
        axes[4].axhline(0, color='k', linestyle='-', linewidth=0.5)
        axes[4].set_title('Step 5: Mean-Centered', fontsize=11, fontweight='bold')
        axes[4].set_xlabel('Wavenumber (cm)')
        axes[4].grid(alpha=0.3)
        
        axes[5].plot(wavenumbers_filtered, intensity_normalized, 'darkgreen', linewidth=1.5)
        axes[5].set_title('Step 6: Vector Normalized (Final)', fontsize=11, fontweight='bold')
        axes[5].set_xlabel('Wavenumber (cm)')
        axes[5].grid(alpha=0.3)
        
        plt.suptitle('Preprocessing Pipeline', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_variance_scatter(vra, ver, boundary_flags, vra_threshold=1.0, ver_threshold=1.0):
        '''Scatter plot of VRA vs VER with boundary detection'''
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = {
            'high_confidence': 'green',
            'boundary': 'red',
            'low_confidence': 'orange',
            'ambiguous': 'purple'
        }
        
        for regime, color in colors.items():
            mask = boundary_flags[regime]
            count = np.sum(mask)
            ax.scatter(ver[mask], vra[mask], c=color, alpha=0.6, s=50, 
                      label=f'{regime.replace("_", " ").title()} (n={count})')
        
        ax.axvline(ver_threshold, color='gray', linestyle='--', linewidth=2)
        ax.axhline(vra_threshold, color='gray', linestyle=':', linewidth=2)
        
        ax.set_xlabel('VER (Inter-Network Variance)', fontsize=12, fontweight='bold')
        ax.set_ylabel('VRA (Intra-Network Variance)', fontsize=12, fontweight='bold')
        ax.set_title('Variance Analysis: VRA vs VER', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_network_agreement(p_FPHW, p_FP, p_HW):
        '''Compare probabilities between three networks'''
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].scatter(p_FPHW, p_FP, alpha=0.5, s=30, c='blue')
        axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect agreement')
        axes[0].set_xlabel('P(Diseased) - NN_FPHW')
        axes[0].set_ylabel('P(Diseased) - NN_FP')
        axes[0].set_title('NN_FPHW vs NN_FP')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].scatter(p_FPHW, p_HW, alpha=0.5, s=30, c='green')
        axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect agreement')
        axes[1].set_xlabel('P(Diseased) - NN_FPHW')
        axes[1].set_ylabel('P(Diseased) - NN_HW')
        axes[1].set_title('NN_FPHW vs NN_HW')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        axes[2].scatter(p_FP, p_HW, alpha=0.5, s=30, c='purple')
        axes[2].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect agreement')
        axes[2].set_xlabel('P(Diseased) - NN_FP')
        axes[2].set_ylabel('P(Diseased) - NN_HW')
        axes[2].set_title('NN_FP vs NN_HW')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.suptitle('Network Agreement Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

