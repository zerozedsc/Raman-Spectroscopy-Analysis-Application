import numpy as np
import pandas as pd
from collections import Counter
import sys, shutil, subprocess, pip, os, pickle

print("Current Dir:", os.getcwd())
print("Python:", sys.version)
print("Kernel executable:", sys.executable)
print("pip (module) version:", pip.__version__)
print("pip (module) file:", pip.__file__)
print("pip (exe) path via PATH:", shutil.which("pip"))
print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))
print("UV_CACHE_DIR:", os.environ.get("UV_CACHE_DIR"))

# MGUS → Monoclonal Gammopathy of Undetermined Significance (pre-cancerous condition, may progress to MM)
# MM → Multiple Myeloma (malignant cancer)
# NL → Normal / healthy control
# MMnew → Newly diagnosed Multiple Myeloma (probably treatment-naive)
# MGUSnew → Newly diagnosed MGUS (pre-cancer stage)
# A -> Not Hikkoshi
# B -> Hikkoshi

def take_sensei_data_clean(
    filepath: str = "MM全データ_20250724.csv",
    type_col: str = "type",
    sample_col: str = "SampleNo",
    wavelength_start: int = 600,
    wavelength_end: int = 1800
) -> dict:
    """
    Load Raman CSV and organize as dict[type][SampleNo][row_idx] = {'metadata': dict, 'dataframe': DataFrame}
    """
    df = pd.read_csv(filepath)
    # Identify wavelength columns
    wavelength_cols = [col for col in df.columns if col.isdigit()]
    wavelength_cols = [int(col) for col in wavelength_cols]
    wavelength_cols = [w for w in wavelength_cols if wavelength_start <= w <= wavelength_end]
    wavelength_cols_str = [str(w) for w in wavelength_cols]
    # Identify metadata columns
    meta_cols = [c for c in df.columns if c not in wavelength_cols_str]
    result = {}
    for _, row in df.iterrows():
        typ = row[type_col]
        sample = row[sample_col]
        # Prepare metadata (excluding wavelength columns)
        metadata = {col: row[col] for col in meta_cols}
        # Prepare spectrum dataframe
        spectrum = pd.DataFrame({
            "wavelength": wavelength_cols,
            "intensity": row[wavelength_cols_str].astype(float).values
        })
        # Insert into nested dict
        if typ not in result:
            result[typ] = {}
        if sample not in result[typ]:
            result[typ][sample] = []
        result[typ][sample].append({
            "metadata": metadata,
            "dataframe": spectrum
        })
    return result

