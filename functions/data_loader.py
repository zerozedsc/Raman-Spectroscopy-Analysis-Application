import os
import glob
import json
from typing import Tuple, Any, Union, List
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from configs.configs import create_logs

SEPERATOR = [",", " ", ";", "\t"]


def load_data_from_path(path: str) -> Union[pd.DataFrame, str]:
    """
    Loads spectral data from a given path, which can be a single file
    (.csv, .pkl, .txt, .asc/.ascii) or a directory of .txt files.
    """
    if not os.path.exists(path):
        error_msg = f"Path does not exist: {path}"
        create_logs("DataLoader", "data_loading", error_msg, status="error")
        return error_msg

    if os.path.isdir(path):
        # Check for .txt, .asc, or .ascii files in the directory
        txt_files = glob.glob(os.path.join(path, "*.txt"))
        asc_files = glob.glob(os.path.join(path, "*.asc")) + glob.glob(
            os.path.join(path, "*.ascii")
        )
        if txt_files and not asc_files:
            return _load_from_txt(path)
        elif asc_files and not txt_files:
            return _load_from_asc(path)
        elif txt_files and asc_files:
            error_msg = f"Directory contains both .txt and .asc/.ascii files: {path}"
            create_logs("DataLoader", "data_loading", error_msg, status="error")
            return error_msg
        else:
            error_msg = f"No supported spectral files (.txt, .asc, .ascii) found in directory: {path}"
            create_logs("DataLoader", "data_loading", error_msg, status="error")
            return error_msg

    _, extension = os.path.splitext(path)
    if extension == ".csv":
        return _load_from_csv(path)
    elif extension == ".pkl":
        return _load_from_pkl(path)
    elif extension == ".txt":
        return _load_from_txt(path)
    elif extension == ".ascii" or extension == ".asc":
        return _load_from_asc(path)
    else:
        error_msg = f"Unsupported file type: {extension}"
        create_logs("DataLoader", "data_loading", error_msg, status="error")
        return error_msg


## [FIX:160725] - Load spectral data from a directory of .txt files or a single .txt file
def _load_from_txt(path: str) -> Union[pd.DataFrame, str]:
    """Loads spectral data from a single .txt file or all .txt files from a directory."""
    try:
        if os.path.isdir(path):
            # Process directory of .txt files
            file_paths = glob.glob(os.path.join(path, "*.txt"))
            if not file_paths:
                raise ValueError(f"No .txt files found in {path}")
        else:
            # Process single .txt file
            file_paths = [path]

        data_dict = {}
        wavenumbers = None

        set_separator = None
        for file_path in file_paths:
            file_name = os.path.basename(file_path).split(".")[0]

            # Try different separators
            if set_separator is None:
                for sep in SEPERATOR:
                    try:
                        df = pd.read_csv(
                            file_path,
                            sep=sep,
                            header=None,
                            names=["wavenumber", "intensity"],
                        )
                        if df.shape[1] == 2 and not df.isnull().any().any():
                            set_separator = sep
                            break
                    except Exception:
                        continue
            else:
                df = pd.read_csv(
                    file_path,
                    sep=set_separator,
                    header=None,
                    names=["wavenumber", "intensity"],
                )

            if df.empty:
                create_logs(
                    "DataLoader",
                    "data_loading",
                    f"Skipping empty file: {file_name}",
                    status="warning",
                )
                continue

            if df.shape[1] != 2 or df.isnull().any().any():
                create_logs(
                    "DataLoader",
                    "data_loading",
                    f"Skipping invalid file: {file_name}",
                    status="warning",
                )
                continue

            if wavenumbers is None:
                wavenumbers = df["wavenumber"].values
            elif not np.allclose(wavenumbers, df["wavenumber"].values):
                create_logs(
                    "DataLoader",
                    "data_loading",
                    f"Wavenumber mismatch in {file_name}, skipping.",
                    status="warning",
                )
                continue

            data_dict[file_name] = df["intensity"].values

        if not data_dict:
            raise ValueError("No valid .txt files could be loaded.")

        combined_df = pd.DataFrame(data_dict)
        combined_df.index = wavenumbers
        combined_df.index.name = "wavenumber"

        create_logs(
            "DataLoader",
            "data_loading",
            f"Successfully loaded {len(data_dict)} spectra from {'directory' if os.path.isdir(path) else 'file'}: {path}",
            status="info",
        )
        return combined_df

    except Exception as e:
        create_logs(
            "DataLoader",
            "data_loading",
            f"Error loading from {path}: {e}",
            status="error",
        )
        return str(e)


def _load_from_csv(csv_filepath: str) -> Union[pd.DataFrame, str]:
    """Loads spectral data from a CSV file."""
    try:
        df = pd.read_csv(csv_filepath, header=0, index_col=0)
        df.index.name = "wavenumber"
        create_logs(
            "DataLoader",
            "data_loading",
            f"Successfully loaded CSV: {csv_filepath}",
            status="info",
        )
        return df
    except Exception as e:
        create_logs(
            "DataLoader",
            "data_loading",
            f"Error loading CSV {csv_filepath}: {e}",
            status="error",
        )
        return str(e)


def _load_from_pkl(pk_filepath: str) -> Union[pd.DataFrame, str]:
    """Loads spectral data from a pickle file."""
    try:
        df = pd.read_pickle(pk_filepath)
        df.index.name = "wavenumber"
        create_logs(
            "DataLoader",
            "data_loading",
            f"Successfully loaded PKL: {pk_filepath}",
            status="info",
        )
        return df
    except Exception as e:
        create_logs(
            "DataLoader",
            "data_loading",
            f"Error loading PKL {pk_filepath}: {e}",
            status="error",
        )
        return str(e)


## [NEW:160725] - Load spectral data from an ASCII file or directory
def _load_from_asc(path: str) -> Union[pd.DataFrame, str]:
    """Loads spectral data from ASCII file(s), supporting multiple separators and both single files and directories."""
    try:
        if os.path.isdir(path):
            # Process directory of .asc/.ascii files
            file_paths = glob.glob(os.path.join(path, "*.asc")) + glob.glob(
                os.path.join(path, "*.ascii")
            )
            if not file_paths:
                raise ValueError(f"No ASCII files found in {path}")
        else:
            # Process single ASCII file
            file_paths = [path]

        data_dict = {}
        wavenumbers = None
        set_separator = None

        for file_path in file_paths:
            file_name = os.path.basename(file_path).split(".")[0]

            # Try different separators if not determined yet
            if set_separator is None:
                for sep in SEPERATOR:
                    try:
                        df = pd.read_csv(
                            file_path,
                            sep=sep,
                            header=None,
                            names=["wavenumber", "intensity"],
                        )
                        if df.shape[1] == 2 and not df.isnull().any().any():
                            set_separator = sep
                            break
                    except Exception:
                        continue

                # If no separator worked, try whitespace
                if set_separator is None:
                    try:
                        df = pd.read_csv(
                            file_path,
                            delim_whitespace=True,
                            header=None,
                            names=["wavenumber", "intensity"],
                        )
                        if df.shape[1] == 2 and not df.isnull().any().any():
                            set_separator = "whitespace"
                    except Exception:
                        pass
            else:
                # Use the previously determined separator
                if set_separator == "whitespace":
                    df = pd.read_csv(
                        file_path,
                        delim_whitespace=True,
                        header=None,
                        names=["wavenumber", "intensity"],
                    )
                else:
                    df = pd.read_csv(
                        file_path,
                        sep=set_separator,
                        header=None,
                        names=["wavenumber", "intensity"],
                    )

            if df.empty:
                create_logs(
                    "DataLoader",
                    "data_loading",
                    f"Skipping empty file: {file_name}",
                    status="warning",
                )
                continue

            if df.shape[1] != 2 or df.isnull().any().any():
                create_logs(
                    "DataLoader",
                    "data_loading",
                    f"Skipping invalid file: {file_name}",
                    status="warning",
                )
                continue

            if wavenumbers is None:
                wavenumbers = df["wavenumber"].values
            elif not np.allclose(wavenumbers, df["wavenumber"].values):
                create_logs(
                    "DataLoader",
                    "data_loading",
                    f"Wavenumber mismatch in {file_name}, skipping.",
                    status="warning",
                )
                continue

            data_dict[file_name] = df["intensity"].values

        if not data_dict:
            raise ValueError("No valid ASCII files could be loaded.")

        # Create the combined DataFrame
        combined_df = pd.DataFrame(data_dict)
        combined_df.index = wavenumbers
        combined_df.index.name = "wavenumber"

        create_logs(
            "DataLoader",
            "data_loading",
            f"Successfully loaded {len(data_dict)} spectra from {'directory' if os.path.isdir(path) else 'file'}: {path}",
            status="info",
        )
        return combined_df

    except Exception as e:
        create_logs(
            "DataLoader",
            "data_loading",
            f"Error loading ASCII from {path}: {e}",
            status="error",
        )
        return str(e)


def load_metadata_from_json(json_path: str) -> Union[dict, str]:
    """Loads metadata from a specified JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        create_logs(
            "DataLoader",
            "metadata_loading",
            f"Successfully loaded metadata: {json_path}",
            status="info",
        )
        return metadata
    except Exception as e:
        create_logs(
            "DataLoader",
            "metadata_loading",
            f"Error loading metadata {json_path}: {e}",
            status="error",
        )
        return str(e)
