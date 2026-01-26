# Configuration file for the Sphinx documentation builder.
#
# This repository contains multiple languages under docs/en and docs/ja.
# Read the Docs builds *one language per project* (parent + translation projects).
# We use READTHEDOCS_LANGUAGE to select which subtree to build.

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


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


# Read the Docs invokes Sphinx with source dir at the repository root (".") and
# this config file at docs/conf.py. Our actual documentation lives under
# docs/en/** and docs/ja/**.
#
# RTD additionally requires an index.html at the HTML output root.
# We satisfy that requirement by generating a tiny redirecting index.html via
# Sphinx's html_additional_pages, while keeping Sphinx's master_doc
# language-specific so navigation/search are correct.


def _normalize_docs_language(value: str | None) -> str:
    lang = (value or "").strip().lower()
    if lang.startswith("ja"):
        return "ja"
    return "en"


def _cli_language_override() -> str | None:
    """Read `-D language=<lang>` from Sphinx CLI arguments (RTD uses this)."""

    for i, arg in enumerate(sys.argv):
        if arg == "-D" and i + 1 < len(sys.argv):
            keyval = sys.argv[i + 1]
            if keyval.startswith("language="):
                return keyval.split("=", 1)[1]
        elif arg.startswith("-Dlanguage="):
            return arg.split("=", 1)[1]
    return None


def _cli_builder() -> str | None:
    """Read the selected builder from `-b <builder>` (RTD uses this for PDF/EPUB)."""

    for i, arg in enumerate(sys.argv):
        if arg == "-b" and i + 1 < len(sys.argv):
            return sys.argv[i + 1].strip()
        # Uncommon, but handle combined forms like "-blatex".
        if arg.startswith("-b") and len(arg) > 2:
            return arg[2:].strip()
    return None


def _detect_sphinx_srcdir() -> Path:
    """Best-effort detection of Sphinx's *source directory*.

    Sphinx is typically invoked as:

        sphinx-build [options] <sourcedir> <outputdir>

    Read the Docs and local invocations may also call `python -m sphinx`.

    We intentionally detect based on the sourcedir argument rather than
    `os.getcwd()`, because some Sphinx frontends may change the working
    directory during startup.
    """

    if len(sys.argv) >= 3:
        candidate = Path(sys.argv[-2])
        if not candidate.is_absolute():
            candidate = Path(os.getcwd()) / candidate
        return candidate.resolve()

    return Path(os.getcwd()).resolve()


def _detect_docs_prefix(srcdir: Path) -> str:
    """Detect whether docs live under <srcdir>/docs/ or directly under <srcdir>.

    - If <srcdir>/en exists, language roots are direct children → "".
    - If <srcdir>/docs/en exists, language roots are under docs/ → "docs/".
    """

    if (srcdir / "en").is_dir() and (srcdir / "conf.py").is_file():
        return ""

    if (srcdir / "docs" / "en").is_dir() and (srcdir / "docs" / "conf.py").is_file():
        return "docs/"

    # Fallback to the common layout in this repo.
    return "docs/"


# If the environment explicitly selects a language (RTD translation projects,
# or local builds via DOCS_LANGUAGE), set it here so it's available during
# document parsing (e.g., for `ifconfig`).
_env_lang = os.environ.get("READTHEDOCS_LANGUAGE") or os.environ.get("DOCS_LANGUAGE")
if _env_lang:
    language = _normalize_docs_language(_env_lang)

# Documentation sources can be under either:
# - <srcdir>/docs/<lang>/...  (srcdir is repo root)
# - <srcdir>/<lang>/...       (srcdir is docs/)
_sphinx_srcdir = _detect_sphinx_srcdir()
_docs_prefix = _detect_docs_prefix(_sphinx_srcdir)

language = _normalize_docs_language(
    _cli_language_override()
    or os.environ.get("READTHEDOCS_LANGUAGE")
    or os.environ.get("DOCS_LANGUAGE")
    or globals().get("language")
    or "en"
)

# Read the Docs builds extra formats by invoking Sphinx with different builders
# (e.g. `-b latex` for PDF and `-b epub`). Those builders would require the
# external Mermaid CLI (`mmdc`) to render diagrams. To keep builds robust even
# when `mmdc` isn't available, treat Mermaid fences as plain code blocks and
# disable the Mermaid extension for non-HTML builders.
_builder = (_cli_builder() or "").lower()
if _builder and _builder != "html":
    extensions = [e for e in (globals().get("extensions") or []) if e != "sphinxcontrib.mermaid"]

    _myst_fence = globals().get("myst_fence_as_directive")
    if isinstance(_myst_fence, set):
        myst_fence_as_directive = {x for x in _myst_fence if x != "mermaid"}
    elif isinstance(_myst_fence, (list, tuple)):
        myst_fence_as_directive = {x for x in _myst_fence if x != "mermaid"}
    elif isinstance(_myst_fence, dict):
        myst_fence_as_directive = {
            k: v for k, v in _myst_fence.items() if str(k).strip().lower() != "mermaid"
        }

root_doc = master_doc = f"{_docs_prefix}{language}/index"

# Avoid pulling unrelated Markdown files (README.md, CHANGELOG.md, etc.).
# When srcdir is repo root, restrict to docs/**. When srcdir is docs/, include
# everything under docs/ (i.e. "**").
if _docs_prefix:
    include_patterns = [f"{_docs_prefix}**"]
else:
    include_patterns = ["**"]

# Exclude legacy/duplicate top-level docs and the *other* language tree.
exclude_patterns = list(globals().get("exclude_patterns") or [])
exclude_patterns += [
    # Defensive excludes (in case the builder source dir changes or include_patterns
    # becomes broad). These folders should never be treated as documentation
    # sources.
    ".venv",
    ".venv/**",
    ".rtd-venv",
    ".rtd-venv/**",
    ".AGI-BANKS",
    ".AGI-BANKS/**",
    ".docs",
    ".docs/**",
    "node_modules",
    "node_modules/**",
    f"{_docs_prefix}_build",
    f"{_docs_prefix}_build/**",
    "Thumbs.db",
    ".DS_Store",
    # Legacy/duplicate top-level docs (kept for history but not built)
    f"{_docs_prefix}changelog.md",
    f"{_docs_prefix}dev-guide/**",
    f"{_docs_prefix}api/**",
    f"{_docs_prefix}analysis-methods/**",
    f"{_docs_prefix}user-guide/**",
    # The old RTD entrypoint we no longer use as master_doc.
    f"{_docs_prefix}index.md",
]

if language == "en":
    exclude_patterns.append(f"{_docs_prefix}ja/**")
else:
    exclude_patterns.append(f"{_docs_prefix}en/**")

# De-dupe while preserving order.
exclude_patterns = list(dict.fromkeys(exclude_patterns))

if language == "ja":
    project = "ラマン分光分析アプリケーション"
    html_title = "ラマン分光分析アプリケーション ドキュメント"
    html_search_language = "ja"

# Ensure templates are available (relative to the config dir).
# This config file lives at docs/conf.py, so template/static paths should be
# relative to the docs/ directory (the configuration directory), not the
# language subfolders.
templates_path = ["_templates"]

# Shared static assets live in docs/_static.
html_static_path = ["_static"]

# Provide a root output index.html (RTD requirement). This produces
# "$OUTDIR/index.html" even though the documentation root is
# "docs/<lang>/index" when building from repo root.
html_context = dict(globals().get("html_context") or {})
html_context["rtd_root_lang"] = language
html_context["rtd_root_redirect_target"] = f"{_docs_prefix}{language}/index.html"
html_additional_pages = dict(globals().get("html_additional_pages") or {})
html_additional_pages["index"] = "rtd_root_index.html"

# Optional gettext settings: only enable locale_dirs if it exists.
_here = os.path.abspath(os.path.dirname(__file__))
_locale_dir = os.path.join(_here, "locale")
if os.path.isdir(_locale_dir):
    locale_dirs = ["locale"]
    gettext_compact = False
    # Recommended by Read the Docs for stable gettext catalogs.
    gettext_uuid = True
