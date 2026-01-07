#!/usr/bin/env bash
set -euo pipefail

# Portable build script for Linux/macOS (PyInstaller)
#
# Default behavior:
#   - mode: onefile
#   - console: off (windowed)
#
# Examples:
#   ./build_scripts/build_portable.sh
#   ./build_scripts/build_portable.sh --mode onedir
#   ./build_scripts/build_portable.sh --console
#   ./build_scripts/build_portable.sh --clean

MODE="onefile"
CONSOLE=0
NO_UPX=0
LOG_LEVEL="WARNING"
OUTPUT_DIR="dist"
WORK_DIR="build"
DIST_NAME="raman_app"
CLEAN=0

usage() {
  cat <<'USAGE'
Usage: build_portable.sh [options]

Options:
  --mode onefile|onedir   Build mode (default: onefile)
  --console               Enable console window/logging (default: off)
  --no-upx                Disable UPX compression (default: off)
  --log-level LEVEL       DEBUG|INFO|WARNING|ERROR (default: WARNING)
  --output-dir DIR        Dist output directory (default: dist)
  --work-dir DIR          PyInstaller work directory (default: build)
  --dist-name NAME        Executable base name (default: raman_app)
  --clean                 Move existing build/dist to build_backups/
  -h, --help              Show help

Notes:
  - This script drives behavior via environment variables consumed by raman_app.spec.
  - If .venv/bin/python exists, it is preferred.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"; shift 2 ;;
    --console)
      CONSOLE=1; shift ;;
    --no-upx)
      NO_UPX=1; shift ;;
    --log-level)
      LOG_LEVEL="${2:-}"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"; shift 2 ;;
    --work-dir)
      WORK_DIR="${2:-}"; shift 2 ;;
    --dist-name)
      DIST_NAME="${2:-}"; shift 2 ;;
    --clean)
      CLEAN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "onefile" && "$MODE" != "onedir" ]]; then
  echo "Invalid --mode: $MODE (expected onefile|onedir)" >&2
  exit 2
fi

case "$LOG_LEVEL" in
  DEBUG|INFO|WARNING|ERROR) ;;
  *)
    echo "Invalid --log-level: $LOG_LEVEL (expected DEBUG|INFO|WARNING|ERROR)" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="python3"
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif [[ -x "$PROJECT_ROOT/.venv/Scripts/python.exe" ]]; then
  # In case someone runs this on Git Bash on Windows
  PYTHON="$PROJECT_ROOT/.venv/Scripts/python.exe"
fi

if [[ ! -f "$PROJECT_ROOT/raman_app.spec" ]]; then
  echo "ERROR: raman_app.spec not found in project root" >&2
  exit 1
fi

if [[ $CLEAN -eq 1 ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  backup_dir="$PROJECT_ROOT/build_backups/backup_${ts}"
  mkdir -p "$backup_dir"

  if [[ -d "$PROJECT_ROOT/$WORK_DIR" ]]; then
    mv "$PROJECT_ROOT/$WORK_DIR" "$backup_dir/" || true
  fi
  if [[ -d "$PROJECT_ROOT/$OUTPUT_DIR" ]]; then
    mv "$PROJECT_ROOT/$OUTPUT_DIR" "$backup_dir/" || true
  fi

  echo "Previous builds moved to: $backup_dir"
fi

export RAMAN_BUILD_MODE="$MODE"
export RAMAN_DIST_NAME="$DIST_NAME"
export RAMAN_CONSOLE="$CONSOLE"
export RAMAN_NO_UPX="$NO_UPX"
export RAMAN_LOG_LEVEL="$LOG_LEVEL"

set +e
"$PYTHON" -m PyInstaller --version >/dev/null 2>&1
pyi_rc=$?
set -e
if [[ $pyi_rc -ne 0 ]]; then
  echo "ERROR: PyInstaller is not available in the selected Python environment ($PYTHON)" >&2
  echo "Hint: install dependencies into your venv, e.g. python -m pip install pyinstaller pyinstaller-hooks-contrib" >&2
  exit 1
fi

echo "Building Raman App portable executable"
echo "  python:     $PYTHON"
echo "  mode:       $MODE"
echo "  console:    $CONSOLE"
echo "  dist name:  $DIST_NAME"
echo "  output dir: $OUTPUT_DIR"
echo "  work dir:   $WORK_DIR"

"$PYTHON" -m PyInstaller --distpath "$OUTPUT_DIR" --workpath "$WORK_DIR" raman_app.spec

if [[ "$MODE" == "onefile" ]]; then
  exe_path="$OUTPUT_DIR/${DIST_NAME}"
else
  exe_path="$OUTPUT_DIR/${DIST_NAME}/${DIST_NAME}"
fi

echo "Build output expected at: $exe_path"
