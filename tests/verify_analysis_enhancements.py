
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from components.widgets.matplotlib_widget import MatplotlibWidget
from pages.analysis_page_utils.result import AnalysisResult
from pages.analysis_page_utils.methods.exploratory import perform_pca_analysis, create_spectrum_preview_figure

def verify_matplotlib_widget():
    print("Verifying MatplotlibWidget...")
    try:
        # We can't instantiate QWidget without QApplication, but we can check class attributes
        if hasattr(MatplotlibWidget, 'add_custom_toolbar'):
            print("PASS: add_custom_toolbar method exists.")
        else:
            print("FAIL: add_custom_toolbar method missing.")
    except Exception as e:
        print(f"FAIL: MatplotlibWidget check failed: {e}")

def verify_analysis_result():
    print("\nVerifying AnalysisResult...")
    try:
        # Check if dataset_data field exists
        from dataclasses import fields
        field_names = [f.name for f in fields(AnalysisResult)]
        if 'dataset_data' in field_names:
            print("PASS: dataset_data field exists in AnalysisResult.")
        else:
            print("FAIL: dataset_data field missing in AnalysisResult.")
    except Exception as e:
        print(f"FAIL: AnalysisResult check failed: {e}")

def verify_pca_enhancements():
    print("\nVerifying PCA Enhancements...")
    try:
        # Create dummy data
        n_samples = 20
        n_features = 100
        wavenumbers = np.linspace(400, 1800, n_features)
        
        # Dataset 1
        data1 = np.random.rand(n_features, n_samples)
        df1 = pd.DataFrame(data1, index=wavenumbers, columns=[f"S1_{i}" for i in range(n_samples)])
        
        # Dataset 2
        data2 = np.random.rand(n_features, n_samples) + 0.5 # Shifted
        df2 = pd.DataFrame(data2, index=wavenumbers, columns=[f"S2_{i}" for i in range(n_samples)])
        
        dataset_data = {"Group A": df1, "Group B": df2}
        
        # Run PCA
        params = {
            "n_components": 3,
            "show_loadings": True,
            "show_distributions": True,
            "show_scree": True,
            "max_loadings_components": 3,
            "n_distribution_components": 3
        }
        
        results = perform_pca_analysis(dataset_data, params)
        
        # Check results
        if results.get("biplot_figure"):
            print("PASS: Biplot figure created.")
        else:
            print("FAIL: Biplot figure missing.")
            
        if results.get("loadings_figure"):
            print("PASS: Loadings figure created.")
            # Check x-tick labels (hard to check on figure object without rendering, but we can check if it runs)
        else:
            print("FAIL: Loadings figure missing.")
            
        if results.get("distributions_figure"):
            print("PASS: Distributions figure created.")
        else:
            print("FAIL: Distributions figure missing.")
            
        # Verify spectrum preview
        fig = create_spectrum_preview_figure(dataset_data)
        if fig:
            print("PASS: Spectrum preview figure created.")
        else:
            print("FAIL: Spectrum preview figure missing.")
            
        plt.close('all')
        
    except Exception as e:
        print(f"FAIL: PCA verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_matplotlib_widget()
    verify_analysis_result()
    verify_pca_enhancements()
