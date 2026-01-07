"""
Peak Assignment Module for Raman Spectroscopy

This module provides functionality for querying the Raman peak assignment database
(raman_peaks.json) to identify molecular assignments for specific wavenumbers.

Functions:
    get_peak_assignment: Look up assignment for a single peak
    get_multiple_peak_assignments: Batch lookup for multiple peaks
    find_peaks_in_range: Find all peaks within a wavenumber range

Author: MUHAMMAD HELMI BIN ROZAIN
Created: 2025-10-01 (Extracted from core.py during Phase 1 refactoring)
"""

import json
import os
from typing import Union, List, Dict
from functions.configs import create_logs


# Module-level cache for peak database (shared across all calls)
_raman_peaks_cache = None


def get_peak_assignment(
    peak: Union[int, float, str],
    pick_near: bool = False,
    tolerance: int = 5,
    json_file_path: str = "assets/data/raman_peaks.json",
) -> dict:
    """
    Get the assignment meaning of a Raman peak based on the raman_peaks.json database.

    Parameters:
    -----------
    peak : Union[int, float, str]
        The wavenumber peak to look up. If float, will be rounded to int.
    pick_near : bool, optional
        If True, will search for the nearest peak within tolerance if exact match not found.
        Default is False.
    tolerance : int, optional
        Maximum distance (in cm⁻¹) to search for nearby peaks when pick_near=True.
        Default is 5.
    json_file_path : str, optional
        Path to the raman_peaks.json file. Default is "assets/data/raman_peaks.json".

    Returns:
    --------
    dict
        Dictionary containing peak assignment information:
        - If found: {"peak": peak, "assignment": assignment, "reference_number": ref_num}
        - If not found and pick_near=False: {"assignment": "Not Found"}
        - If not found and pick_near=True: nearest match or "Not Found"

    Examples:
    ---------
    >>> result = get_peak_assignment(1004)
    >>> print(result["assignment"])
    'Phenylalanine (protein)'

    >>> result = get_peak_assignment(1006, pick_near=True, tolerance=5)
    >>> print(result)
    {'peak': 1004, 'assignment': 'Phenylalanine (protein)', 'distance': 2, 'original_peak': 1006}
    """
    global _raman_peaks_cache

    try:
        # Convert peak to integer (round if float)
        if isinstance(peak, (float, str)):
            try:
                peak_int = int(round(float(peak)))
            except (ValueError, TypeError):
                return {"assignment": "Invalid peak value"}
        else:
            peak_int = int(peak)

        # Load the JSON data with caching
        try:
            if _raman_peaks_cache is not None:
                # Use cached data if available
                raman_data = _raman_peaks_cache
            else:
                # Load and cache the data
                # Handle relative path from current working directory
                if not os.path.isabs(json_file_path):
                    # Try relative to current working directory first
                    if os.path.exists(json_file_path):
                        full_path = json_file_path
                    else:
                        # Try relative to the script directory
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        full_path = os.path.join(script_dir, "..", "..", json_file_path)
                else:
                    full_path = json_file_path

                with open(full_path, "r", encoding="utf-8") as file:
                    raman_data = json.load(file)

                # Cache the data for future use
                _raman_peaks_cache = raman_data

        except FileNotFoundError:
            create_logs(
                "get_peak_assignment",
                "peak_assignment",
                f"Raman peaks file not found: {json_file_path}",
                status="error",
            )
            return {"assignment": "Database file not found"}
        except json.JSONDecodeError as e:
            create_logs(
                "get_peak_assignment",
                "peak_assignment",
                f"Error parsing JSON file: {e}",
                status="error",
            )
            return {"assignment": "Database file corrupted"}

        # Convert peak to string for lookup (JSON keys are strings)
        peak_str = str(peak_int)

        # Direct lookup first
        if peak_str in raman_data:
            result = raman_data[peak_str].copy()
            result["peak"] = peak_int
            return result

        # If not found and pick_near is False, return "Not Found"
        if not pick_near:
            return {"assignment": "Not Found"}

        # Find nearest peak within tolerance
        nearest_peak = None
        min_distance = float("inf")

        for db_peak_str in raman_data.keys():
            try:
                db_peak_int = int(db_peak_str)
                distance = abs(peak_int - db_peak_int)

                if distance <= tolerance and distance < min_distance:
                    min_distance = distance
                    nearest_peak = db_peak_str
            except ValueError:
                # Skip invalid peak keys
                continue

        # Return nearest peak if found within tolerance
        if nearest_peak is not None:
            result = raman_data[nearest_peak].copy()
            result["peak"] = int(nearest_peak)
            result["distance"] = min_distance
            result["original_peak"] = peak_int
            return result
        else:
            return {"assignment": "Not Found"}

    except Exception as e:
        create_logs(
            "get_peak_assignment",
            "peak_assignment",
            f"Error in get_peak_assignment: {e}",
            status="error",
        )
        return {"assignment": "Error occurred during lookup"}


def get_multiple_peak_assignments(
    peaks: List[Union[int, float, str]],
    pick_near: bool = False,
    tolerance: int = 5,
    json_file_path: str = "assets/data/raman_peaks.json",
) -> List[dict]:
    """
    Get assignments for multiple peaks at once.

    Parameters:
    -----------
    peaks : List[Union[int, float, str]]
        List of wavenumber peaks to look up.
    pick_near : bool, optional
        If True, will search for the nearest peak within tolerance if exact match not found.
    tolerance : int, optional
        Maximum distance (in cm⁻¹) to search for nearby peaks when pick_near=True.
    json_file_path : str, optional
        Path to the raman_peaks.json file.

    Returns:
    --------
    List[dict]
        List of dictionaries containing peak assignment information for each input peak.

    Examples:
    ---------
    >>> peaks = [1004, 1445, 1655]
    >>> results = get_multiple_peak_assignments(peaks)
    >>> for result in results:
    ...     print(f"{result.get('peak', 'N/A')}: {result['assignment']}")
    """
    results = []
    for peak in peaks:
        result = get_peak_assignment(peak, pick_near, tolerance, json_file_path)
        results.append(result)
    return results


def find_peaks_in_range(
    min_wavenumber: Union[int, float],
    max_wavenumber: Union[int, float],
    json_file_path: str = "assets/data/raman_peaks.json",
) -> List[dict]:
    """
    Find all peaks within a specified wavenumber range.

    Parameters:
    -----------
    min_wavenumber : Union[int, float]
        Minimum wavenumber of the range.
    max_wavenumber : Union[int, float]
        Maximum wavenumber of the range.
    json_file_path : str, optional
        Path to the raman_peaks.json file.

    Returns:
    --------
    List[dict]
        List of all peaks within the specified range, sorted by wavenumber.

    Examples:
    ---------
    >>> peaks = find_peaks_in_range(1000, 1100)
    >>> print(f"Found {len(peaks)} peaks in range 1000-1100 cm⁻¹")
    >>> for peak in peaks[:3]:
    ...     print(f"{peak['peak']}: {peak['assignment']}")
    """
    global _raman_peaks_cache

    try:
        # Load data using the existing function (ensures cache is populated)
        dummy_result = get_peak_assignment(1000, json_file_path=json_file_path)
        if "Database file" in str(dummy_result.get("assignment", "")):
            return []

        # Get cached data
        raman_data = _raman_peaks_cache

        if raman_data is None:
            return []

        results = []
        for peak_str, data in raman_data.items():
            try:
                peak_int = int(peak_str)
                if min_wavenumber <= peak_int <= max_wavenumber:
                    result = data.copy()
                    result["peak"] = peak_int
                    results.append(result)
            except ValueError:
                continue

        # Sort by wavenumber
        results.sort(key=lambda x: x["peak"])
        return results

    except Exception as e:
        create_logs(
            "find_peaks_in_range",
            "peak_assignment",
            f"Error in find_peaks_in_range: {e}",
            status="error",
        )
        return []


def clear_cache():
    """
    Clear the peak database cache.

    Useful for testing or when the database file has been updated.
    """
    global _raman_peaks_cache
    _raman_peaks_cache = None
