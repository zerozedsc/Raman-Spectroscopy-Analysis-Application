import os, sys
import logging, platform, json
from typing import Dict, List, Any

CURRENT_DIR = os.getcwd()
DATETIME_FT = "%d-%m-%Y %H:%M:%S"

# Global DEBUG flag - controlled by command-line --debug argument
# Default: True (development mode with verbose logging)
# Production: False (set via --debug false for minimal logging)
DEBUG = True


def set_debug_mode(enabled: bool):
    """
    Set global DEBUG mode. Called from main.py with --debug argument.
    
    Args:
        enabled (bool): True for development (verbose logging, console output)
                       False for production (minimal logging, no console spam)
    """
    global DEBUG
    DEBUG = enabled


class LocalizationManager:
    """
    A class to manage loading and retrieving internationalization (i18n) strings
    from JSON files for a PySide6 application.

    It uses a dot-notation key (e.g., 'MAIN_WINDOW.title') to access nested
    JSON values and caches loaded language files to avoid redundant disk I/O.
    """

    def __init__(
        self,
        locale_dir: str = "assets/locales",
        default_lang: str = "en",
        initial_lang: str = None,
    ):
        """
        Initializes the LocalizationManager.

        Args:
            locale_dir (str): The directory where locale JSON files (e.g., 'en.json') are stored.
            default_lang (str): The fallback language to use if a key is not found in the current language.
            initial_lang (str): The initial language to set. Defaults to `default_lang`.
        """
        # Support both frozen (.exe) and development modes
        if getattr(sys, "frozen", False):
            # Running in PyInstaller bundle
            base_path = sys._MEIPASS
            self.locale_dir = os.path.join(base_path, locale_dir)
        else:
            # Running in normal Python
            self.locale_dir = locale_dir
            if not os.path.isdir(self.locale_dir):
                # Try to resolve path relative to the current file
                base_dir = os.path.dirname(os.path.abspath(__file__))
                self.locale_dir = os.path.join(base_dir, "..", locale_dir)

        if not os.path.isdir(self.locale_dir):
            create_logs(
                "LocalizationManager",
                "localization",
                f"Locale directory not found: {self.locale_dir}",
                status="error",
            )
            raise FileNotFoundError(f"Locale directory not found: {self.locale_dir}")

        self.translations: Dict[str, Dict[str, Any]] = {}
        self.default_lang = default_lang
        self.current_lang = initial_lang if initial_lang else default_lang

        # Pre-load the default and initial languages
        self._load_language(self.default_lang)
        if self.current_lang != self.default_lang:
            self._load_language(self.current_lang)

    def _load_language(self, lang_code: str) -> bool:
        """
        Loads a language file from the locale directory.

        Args:
            lang_code (str): The language code (e.g., 'en', 'ja').

        Returns:
            bool: True if the language was loaded successfully, False otherwise.
        """
        if lang_code in self.translations:
            return True

        file_path = os.path.join(self.locale_dir, f"{lang_code}.json")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.translations[lang_code] = json.load(f)
                create_logs(
                    "LocalizationManager",
                    "localization",
                    f"Successfully loaded language: {lang_code}",
                    status="info",
                )
                return True
        except FileNotFoundError:
            create_logs(
                "LocalizationManager",
                "localization",
                f"Language file not found: {file_path}",
                status="warning",
            )
            return False
        except json.JSONDecodeError as e:
            create_logs(
                "LocalizationManager",
                "localization",
                f"Error decoding JSON from {file_path}: {e}",
                status="error",
            )
            return False

    def set_language(self, lang_code: str):
        """
        Sets the current language for translations.

        Args:
            lang_code (str): The language code to switch to.
        """
        if self._load_language(lang_code):
            self.current_lang = lang_code
        else:
            create_logs(
                "LocalizationManager",
                "localization",
                f"Failed to set language to '{lang_code}'. It may not exist.",
                status="error",
            )

    def get(self, key: str, **kwargs) -> str:
        """
        Retrieves a translated string using a dot-separated key.

        It first tries the current language, then falls back to the default language,
        and finally returns the key itself if not found.

        Args:
            key (str): The dot-separated key (e.g., 'PAGE_NAME.option').
            **kwargs: Values to format into the translated string.

        Returns:
            str: The translated and formatted string.
        """
        keys = key.split(".")

        # Try to get from the current language
        translation = self._find_translation(self.current_lang, keys)

        # If not found, try the default language
        if translation is None and self.current_lang != self.default_lang:
            translation = self._find_translation(self.default_lang, keys)

        # If still not found, return the key itself as a fallback
        if translation is None:
            create_logs(
                "LocalizationManager",
                "localization",
                f"Translation key not found: '{key}'",
                status="warning",
            )
            return keys[1].replace("_", " ")

        # Format the string with any provided keyword arguments
        try:
            return translation.format(**kwargs)
        except KeyError as e:
            create_logs(
                "LocalizationManager",
                "localization",
                f"Missing format argument {e} for key '{key}'",
                status="warning",
            )
            return translation  # Return unformatted string

    def _find_translation(self, lang_code: str, keys: List[str]) -> str | None:
        """
        A helper to traverse the dictionary for a given language.
        """
        if lang_code not in self.translations:
            return None

        data = self.translations.get(lang_code)
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
            else:
                return None

        return data if isinstance(data, str) else None

    def available_languages(self) -> List[str]:
        """
        Scans the locale directory and returns a list of available languages.

        Returns:
            List[str]: A list of language codes found in the directory.
        """
        try:
            files = os.listdir(self.locale_dir)
            # Return the filename without the .json extension
            return [f.split(".")[0] for f in files if f.endswith(".json")]
        except FileNotFoundError:
            return []

    def tr(self, key: str, **kwargs) -> str:
        """
        A convenient alias for the get() method.
        """
        return self.get(key, **kwargs)


def load_config(file_path: str = "configs/app_configs.json") -> dict:
    """
    Load a JSON configuration file and return its contents as a dictionary.
    Supports both frozen (.exe) and development modes.

    Args:
        file_path (str): Path to the JSON configuration file.

    Returns:
        dict: Contents of the JSON file.
    """
    """Load app configuration.

    Source of truth:
    - Defaults: bundled JSON (read-only in frozen builds)
    - User overrides: per-user/portable INI (language/theme)

    This fixes a long-standing issue where frozen builds read configs from
    `sys._MEIPASS` (non-writable) while the Settings dialog wrote elsewhere.
    """

    # Resolve default JSON location (supports frozen mode)
    default_json_path = file_path
    if getattr(sys, "frozen", False):
        default_json_path = os.path.join(sys._MEIPASS, file_path)

    config: dict = {}

    if os.path.exists(default_json_path):
        try:
            with open(default_json_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                create_logs(
                    "ConfigLoader",
                    "config",
                    f"Loaded default configuration from {default_json_path}",
                    status="info",
                )
        except json.JSONDecodeError as e:
            create_logs(
                "ConfigLoader",
                "config",
                f"Error decoding JSON from {default_json_path}: {e}",
                status="error",
            )
            config = {}
        except Exception as e:
            create_logs(
                "ConfigLoader",
                "config",
                f"Error reading default configuration from {default_json_path}: {e}",
                status="error",
            )
            config = {}
    else:
        # Defaults missing should not crash the app; return a minimal baseline.
        create_logs(
            "ConfigLoader",
            "config",
            f"Default configuration file not found: {default_json_path}",
            status="warning",
        )
        config = {}

    # Overlay user settings from INI (portable/user dir)
    try:
        from configs.user_settings import load_user_settings

        user = load_user_settings(defaults=config)
        config["language"] = user.language
        config["theme"] = user.theme
    except Exception as e:
        # Keep JSON defaults if INI load fails
        create_logs(
            "ConfigLoader",
            "config",
            f"Failed to load INI user settings: {e}",
            status="warning",
        )

    return config


def create_logs(
    log_name="LOGS", filename="logs", log_message="", status="info", show_console=None
):
    """
    Create a log file with the specified name and message.
    
    Args:
        log_name (str): Name of the logger.
        filename (str): Name of the log file (without extension).
        log_message (str): Message to log.
        status (str): Log level ('info', 'error', 'warning', 'debug', 'console').
        show_console (bool|None): Override global DEBUG for console output.
                                   None = respect global DEBUG flag.
    """
    global DEBUG
    
    # Determine whether to show console output
    should_show_console = DEBUG if show_console is None else show_console
    
    # Handle console-only output with UTF-8 encoding to prevent UnicodeEncodeError
    if status == "console":
        if should_show_console:
            try:
                # Use UTF-8 encoding for console output
                sys.stdout.buffer.write((log_message + '\n').encode('utf-8'))
                sys.stdout.flush()
            except (AttributeError, IOError):
                # Fallback if buffer is not available
                try:
                    print(log_message)
                except UnicodeEncodeError:
                    # Last resort: replace non-ASCII characters
                    print(log_message.encode('ascii', 'replace').decode('ascii'))
        return  # Exit early for console-only messages
    
    # Skip debug-level logs in production mode (when DEBUG=False)
    if status == "debug" and not DEBUG:
        return

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    foldername = "logs"
    if not os.path.exists(foldername):
        foldername = os.path.join(os.getcwd(), foldername)
        if not os.path.exists(foldername):
            os.makedirs(foldername)
    else:
        foldername = os.path.abspath(foldername)

    filename = filename if filename.endswith(".log") else filename + ".log"
    full_filename = os.path.join(foldername, filename)

    # Check if logs directory exists, if not create it
    try:
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        elif not os.path.isdir(foldername):
            # If 'logs' exists but is not a directory (e.g., it's a file)
            raise OSError(f"'{foldername}' exists but is not a directory")
    except OSError as e:
        return  # Exit function if we can't create the directory

    # Create a logger
    logger = logging.getLogger(log_name)
    # Set logger to debug level to catch all messages
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Attach handlers in an idempotent way.
    # (Previous behavior only attached a StreamHandler if the *first* call happened
    # with console enabled; and it fixed the console level based on the first call's
    # status. That made later INFO/DEBUG appear to be "missing" from the terminal.)
    try:
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(full_filename)
            for h in logger.handlers
        )

        if not has_file_handler:
            file_handler = logging.FileHandler(full_filename, encoding="utf-8")
            file_handler.setFormatter(formatter)
            # Always capture everything to file
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

        # Stream handler for terminal output (only when enabled)
        has_stdout_stream = any(
            isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
            for h in logger.handlers
        )

        if should_show_console and not has_stdout_stream:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)

            # Best-effort UTF-8 console
            if hasattr(stream_handler.stream, "reconfigure"):
                try:
                    stream_handler.stream.reconfigure(encoding="utf-8")
                except Exception:
                    pass

            # In dev mode we want to see everything; the caller still controls verbosity
            # by choosing which statuses to emit.
            stream_handler.setLevel(logging.DEBUG)
            logger.addHandler(stream_handler)
    except Exception:
        return

    # Log the message
    try:
        if status == "info":
            logger.info(log_message)
        elif status == "error":
            logger.error(log_message)
        elif status == "warning":
            logger.warning(log_message)
        elif status == "debug":
            logger.debug(log_message)
    except Exception as e:
        pass  # Silently ignore logging errors


def load_application_fonts():
    """
    Loads all .ttf fonts from the assets/fonts directory into the application.
    This makes them available for use in stylesheets without system installation.
    Supports both frozen (.exe) and development modes.
    """
    from PySide6.QtGui import QFontDatabase

    # Support frozen mode
    if getattr(sys, "frozen", False):
        # Running in PyInstaller bundle
        fonts_dir = os.path.join(sys._MEIPASS, "assets", "fonts")
    else:
        # Try to resolve fonts directory relative to the project root if not found in current dir
        fonts_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "assets", "fonts"
        )
        if not os.path.exists(fonts_dir):
            # Fallback: try relative to this file (legacy location)
            fonts_dir = os.path.join(os.path.dirname(__file__), "assets", "fonts")

    if not os.path.exists(fonts_dir):
        return

    for font_file in os.listdir(fonts_dir):
        if font_file.endswith(".ttf"):
            font_path = os.path.join(fonts_dir, font_file)
            font_id = QFontDatabase.addApplicationFont(font_path)
            if font_id == -1:
                pass  # Font loading failed
            else:
                # You can optionally print the loaded font families
                families = QFontDatabase.applicationFontFamilies(font_id)
                # print(f"Successfully loaded font: {families[0]}")

    # After loading, it's good practice to log all available families to debug
    # print("All available font families:", QFontDatabase().families())
