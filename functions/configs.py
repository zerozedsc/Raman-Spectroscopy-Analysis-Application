from datetime import datetime
from pprint import pprint
from typing import Any, List, Tuple, Union
from glob import glob
from pathlib import Path

import time
import logging
import json
import os
import sys
import uuid
import ast
import traceback
import re
import random

DEBUG = True

# Global Constants
CURRENT_DIR = os.getcwd()
DATETIME_FT = "%d-%m-%Y %H:%M:%S"
DATE_FT = "%d-%m-%Y"
LOCAL_DATA = {}


# logs function
def create_logs(log_name, filename, log_message, status="info"):
    """
    Create log entries with configurable log level filtering.

    Log level can be controlled via RAMAN_LOG_LEVEL environment variable.
    Supported values: DEBUG, INFO, WARNING, ERROR
    Default: WARNING (logs warnings and errors only)
    """
    # Get log level from environment variable (default to WARNING)
    env_log_level = os.getenv("RAMAN_LOG_LEVEL", "WARNING").upper()
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    min_level = log_level_map.get(env_log_level, logging.WARNING)

    # Map status to logging level
    status_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    message_level = status_level_map.get(status.lower(), logging.INFO)

    # Skip if message level is below minimum level
    if message_level < min_level:
        return

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    foldername = "logs"
    full_filename = os.path.join(foldername, filename + ".log")

    # Ensure the logs directory exists
    os.makedirs(foldername, exist_ok=True)

    # Create a logger
    logger = logging.getLogger(log_name)
    # Set logger to debug level to catch all messages
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Check if logger already has handlers to prevent duplicate messages
    if not logger.handlers:
        # Create file handler (always log everything to file)
        file_handler = logging.FileHandler(full_filename)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)  # File gets all logs

        # Create stream handler for terminal output (respects min_level)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(min_level)  # Console respects environment setting

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    # Log the message
    if status == "info":
        logger.info(log_message)
    elif status == "error":
        logger.error(log_message)
    elif status == "warning":
        logger.warning(log_message)
    elif status == "debug":
        logger.debug(log_message)


def console_log(
    log_message, status="info", show_status=False, show_time=False, show=True, **kwargs
) -> None:
    """
    Log messages to the console with different severity levels.

    Args:
        log_message (str): The message to log.
        status (str): The severity level of the log message ('info', 'error', 'warning', 'debug').
        show_status (bool): Whether to show the status prefix in the log message.
        show_time (bool): Whether to include the current time in the log message.
        show (bool): Whether to actually print the log message.
        **kwargs: Additional keyword arguments, e.g., 'indent' for pprint indentation.

    """
    if not show:
        return

    st_dict = {
        "info": "[INFO] ",
        "error": "[ERROR] ",
        "warning": "[WARNING] ",
        "debug": "[DEBUG] ",
    }
    time_str = ""

    if show_time:
        time_now = datetime.now().strftime(DATETIME_FT)
        time_str = f"{time_now} - "

    status_str = ""
    if show_status:
        status_str = st_dict.get(status, "")

    indent = kwargs.get("indent", 1)
    if indent <= 1:
        return_str = f"{time_str}{status_str}{log_message}"
    else:
        return_str = f"{time_str}{status_str}"


def generate_id(i, n, id_type="uuid") -> str:
    """Generate ID based on the type and length provided."""

    if id_type == "numeric":
        # Generate n digits number ID with leading zeros
        return str(i).zfill(n)
    elif id_type == "uuid":
        # Generate random UUID
        return str(uuid.uuid4())
    elif id_type == "alphanumeric":
        # Generate number and letter as ID
        prefix = chr(65 + (i // (10**n)))  # 65 is ASCII for 'A'
        number = i % (10**n)
        return f"{prefix}{str(number).zfill(n)}"
    else:
        return None


def ts_id() -> int:
    """Return current timestamp as integer/ID."""

    time_now = datetime.now().strftime(DATETIME_FT)
    dt_object = datetime.strptime(time_now, DATETIME_FT)
    timestamp = datetime.timestamp(dt_object)
    return int(timestamp)


# global functions
def datetime_now() -> str:
    return datetime.now().strftime(DATETIME_FT)


def str2dt(s, ft) -> datetime:
    """Convert string to datetime object using DATETIME_FT format."""
    return datetime.strptime(s, ft)


def timestamp_id2dt(timestamp) -> str:
    dt_object = datetime.fromtimestamp(timestamp)
    return dt_object.strftime(DATETIME_FT)


def split_date(date_input, date_format):
    """
    Splits a date into a list of [dd, mm, yyyy] based on the provided format.

    :param date_input: The date string or datetime object to split.
    :param date_format: The format of the date string, e.g., '%d-%m-%Y', '%d/%m/%Y', etc.
                        If date_input is already a datetime object, this parameter is ignored.
    :return: A list of [dd, mm, yyyy] as integers.
    """
    # If date_input is a string, parse it into a datetime object based on the provided format
    if isinstance(date_input, str):
        date_obj = datetime.strptime(date_input, date_format)
    elif isinstance(date_input, datetime):
        date_obj = date_input
    else:
        raise ValueError("date_input must be a string or a datetime object")

    # Extract and return the day, month, and year components
    return [date_obj.day, date_obj.month, date_obj.year]


def numeric(value) -> any:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def compare_datetime(datetime_str1, datetime_str2) -> bool:
    """
    Compare two datetime strings and return the latest.

    :param datetime_str1: The first datetime string.
    :param datetime_str2: The second datetime string.
    :return: The latest datetime string.
    """
    # Parse the datetime strings into actual datetime objects
    datetime_obj1 = str2dt(datetime_str1, DATETIME_FT)
    datetime_obj2 = str2dt(datetime_str2, DATETIME_FT)

    # Compare the datetime objects
    if datetime_obj1 > datetime_obj2:
        return True
    else:
        return False


def json_str_process(value) -> dict:
    try:
        if type(value) is dict:
            return value
        elif type(value) is str:
            return json.loads(value)
        else:
            return {"error": "Invalid data type"}
    except json.JSONDecodeError as e:
        create_logs(
            "from_json",
            "app",
            f"convert to json.JSONDecodeError error: {e}",
            status="error",
        )
        return {}

    except Exception as e:
        create_logs("from_json", "app", f"convert to json error: {e}", status="error")
        return {}


def slash_escape(value) -> str:
    if "/" in value:
        return value.replace("/", "'")
    if "'" in value:
        return value.replace("'", "/")


def rmspace(value) -> str:
    return value.replace(" ", "")


def safe_filename(value) -> str:
    """
    Remove special characters from a string to make it a safe filename.
    """
    # Define a regex pattern to match invalid characters
    pattern = r'[<>:"/\\|?*]'

    # Replace invalid characters with an underscore
    safe_value = re.sub(pattern, "_", value)

    return safe_value
