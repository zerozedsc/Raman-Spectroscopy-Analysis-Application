"""Generate a Windows .ico from the SVG source-of-truth.

Windows executables and NSIS installers typically require a .ico.
We keep `assets/icons/app-icon.svg` as the main design source and generate:
- `assets/icons/app-icon.ico`

This script is intended to be called by build scripts before PyInstaller/NSIS.
"""

from __future__ import annotations

import os
import sys


def _project_root() -> str:
    # build_scripts/ is directly under the repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    project_root = _project_root()

    svg_path = os.path.join(project_root, "assets", "icons", "app-icon.svg")
    out_ico_path = os.path.join(project_root, "assets", "icons", "app-icon.ico")

    if not os.path.exists(svg_path):
        print(f"ERROR: SVG not found: {svg_path}")
        return 2

    os.makedirs(os.path.dirname(out_ico_path), exist_ok=True)

    # Import PySide6 lazily to keep import side-effects contained.
    try:
        # Some environments may be headless; attempt to be conservative.
        os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false")

        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtSvg import QSvgRenderer
    except Exception as e:
        print(f"ERROR: PySide6 is required to generate ICO: {e}")
        return 3

    try:
        # Render at a high resolution (Windows will scale down as needed).
        size = 256
        renderer = QSvgRenderer(svg_path)
        if not renderer.isValid():
            print(f"ERROR: Invalid SVG (renderer not valid): {svg_path}")
            return 4

        image = QImage(size, size, QImage.Format.Format_ARGB32)
        image.fill(0x00000000)

        painter = QPainter(image)
        renderer.render(painter)
        painter.end()

        # QImage can write ICO on Windows builds of Qt.
        ok = image.save(out_ico_path)
        if not ok:
            # Fallback: try explicit format
            ok = image.save(out_ico_path, "ICO")

        if not ok:
            print(f"ERROR: Failed to write ICO: {out_ico_path}")
            return 5

        print(f"OK: Wrote {out_ico_path}")
        return 0

    except Exception as e:
        print(f"ERROR: Exception generating ICO: {e}")
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
