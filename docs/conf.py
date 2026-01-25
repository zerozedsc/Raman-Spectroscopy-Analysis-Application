# Configuration file for the Sphinx documentation builder.
#
# This repository contains multiple languages under docs/en and docs/ja.
# Read the Docs builds *one language per project* (parent + translation projects).
# We use READTHEDOCS_LANGUAGE to select which subtree to build.

from __future__ import annotations

import importlib.util
import os


def _load_base_conf():
    here = os.path.abspath(os.path.dirname(__file__))
    base_conf_path = os.path.join(here, "en", "conf.py")

    spec = importlib.util.spec_from_file_location("base_conf", base_conf_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base Sphinx config at {base_conf_path}")

    base_conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_conf)
    return base_conf


_base_conf = _load_base_conf()

# Import all settings from English configuration as defaults.
for _attr in dir(_base_conf):
    if not _attr.startswith("_"):
        globals()[_attr] = getattr(_base_conf, _attr)


def _detect_language() -> str:
    # READTHEDOCS_LANGUAGE is set by Read the Docs and matches the project's language
    # code (lowercase, dash separator), e.g. "en", "ja", "pt-br".
    lang = (os.environ.get("READTHEDOCS_LANGUAGE") or os.environ.get("DOCS_LANGUAGE") or "en").strip()
    return lang.lower()


_detected_lang = _detect_language()

# Normalize to a supported docs subtree.
if _detected_lang.startswith("ja"):
    language = "ja"
else:
    language = "en"

# Root doc lives under docs/<lang>/index.md
root_doc = master_doc = f"{language}/index"

# Ensure we only build one language tree at a time.
# Also exclude legacy top-level docs/ content that duplicates the language trees.
_exclude = list(exclude_patterns) if "exclude_patterns" in globals() else []
_exclude += [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Legacy/duplicate top-level docs (kept for history but not built)
    "index.md",
    "changelog.md",
    "dev-guide/**",
    "api/**",
    "analysis-methods/**",
    "user-guide/**",
]

if language == "en":
    _exclude.append("ja/**")
else:
    _exclude.append("en/**")

# Keep ordering stable while removing duplicates.
exclude_patterns = list(dict.fromkeys(_exclude))

# Japanese-specific metadata.
if language == "ja":
    project = "ラマン分光分析アプリケーション"
    html_title = "ラマン分光分析アプリケーション ドキュメント"
    html_search_language = "ja"

# Optional gettext settings: only enable locale_dirs if it exists.
_here = os.path.abspath(os.path.dirname(__file__))
_locale_dir = os.path.join(_here, "locale")
if os.path.isdir(_locale_dir):
    locale_dirs = ["locale"]
    gettext_compact = False
    # Recommended by Read the Docs for stable gettext catalogs.
    gettext_uuid = True
