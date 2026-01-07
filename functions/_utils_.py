import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ramanspy as rp

import pickle
import traceback
import random


try:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    device = "cpu"
    print("PyTorch not installed, defaulting to CPU.")


def console_log(message):
    print(f"[DEBUG CONSOLE]: {message}")


def load_pickle(path):
    if not os.path.exists(path):
        console_log(f"File not found: {path}")
        return None

    data = None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
