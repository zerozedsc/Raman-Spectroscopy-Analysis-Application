"""User settings persistence (INI)

This module provides a single, robust location for reading/writing user settings
(language/theme) across three run modes:

- Development (uv / python): stored in user config directory
- Installed (Program Files, typically non-writable): stored in user config directory
- Portable (folder next to the .exe is writable): stored next to the executable

Design goals:
- No third-party dependencies (avoid appdirs)
- Safe defaults and graceful failure
- Explicit, testable path resolution
"""

from __future__ import annotations

import configparser
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional


APP_DIR_NAME = "RamanApp"
SETTINGS_FILENAME = "raman_app.ini"
SETTINGS_SECTION = "app"


@dataclass(frozen=True)
class UserSettings:
    language: str = "en"
    theme: str = "system"

    def to_dict(self) -> Dict[str, str]:
        return {"language": self.language, "theme": self.theme}


def _is_writable_dir(dir_path: str) -> bool:
    """Best-effort check whether a directory is writable."""
    try:
        os.makedirs(dir_path, exist_ok=True)
        test_path = os.path.join(dir_path, ".__raman_write_test__")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
        return True
    except Exception:
        return False


def get_user_config_dir() -> str:
    """Return an OS-appropriate user config directory (cross-platform)."""
    # Windows
    appdata = os.environ.get("APPDATA")
    if appdata:
        return os.path.join(appdata, APP_DIR_NAME)

    # macOS
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Application Support", APP_DIR_NAME)

    # Linux / other
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return os.path.join(xdg, APP_DIR_NAME)

    return os.path.join(home, ".config", APP_DIR_NAME)


def get_portable_settings_path() -> Optional[str]:
    """Return the portable settings path (next to .exe) if in frozen mode."""
    if not getattr(sys, "frozen", False):
        return None

    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    return os.path.join(exe_dir, SETTINGS_FILENAME)


def resolve_settings_path() -> str:
    """Resolve where the settings INI should live for this run."""
    portable_path = get_portable_settings_path()
    user_path = os.path.join(get_user_config_dir(), SETTINGS_FILENAME)

    if portable_path:
        # If the portable INI already exists, we consider the app "portable".
        if os.path.exists(portable_path):
            return portable_path

        # Otherwise: if the exe directory is writable, prefer portable behavior.
        exe_dir = os.path.dirname(portable_path)
        if _is_writable_dir(exe_dir):
            return portable_path

    # Fallback: installed/dev uses per-user config directory.
    return user_path


def load_user_settings(
    *,
    defaults: Optional[Dict[str, Any]] = None,
    ini_path: Optional[str] = None,
) -> UserSettings:
    """Load user settings from INI, overlaying onto provided defaults."""
    base = defaults or {}
    language = str(base.get("language", "en") or "en")
    theme = str(base.get("theme", "system") or "system")

    path = ini_path or resolve_settings_path()
    if not os.path.exists(path):
        return UserSettings(language=language, theme=theme)

    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
        section = parser[SETTINGS_SECTION] if SETTINGS_SECTION in parser else {}
        language = str(section.get("language", language) or language)
        theme = str(section.get("theme", theme) or theme)
        return UserSettings(language=language, theme=theme)
    except Exception:
        # Fail closed to defaults
        return UserSettings(language=language, theme=theme)


def save_user_settings(
    *,
    language: str,
    theme: str,
    ini_path: Optional[str] = None,
) -> str:
    """Persist settings to the resolved INI path. Returns the path written."""
    path = ini_path or resolve_settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    parser = configparser.ConfigParser()
    parser[SETTINGS_SECTION] = {
        "language": str(language),
        "theme": str(theme),
    }

    with open(path, "w", encoding="utf-8") as f:
        parser.write(f)

    return path
