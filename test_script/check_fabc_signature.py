import inspect
from pybaselines import api
import numpy as np

# Create baseline fitter
fitter = api.Baseline(x_data=np.linspace(400, 1800, 100))

# Get FABC signature
sig = inspect.signature(fitter.fabc)

print("="*80)
print("pybaselines FABC Parameter Types")
print("="*80)

for name, param in sig.parameters.items():
    print(f"{name:20s}: {param.annotation if param.annotation != inspect.Parameter.empty else 'no annotation':30s} | default={param.default}")
