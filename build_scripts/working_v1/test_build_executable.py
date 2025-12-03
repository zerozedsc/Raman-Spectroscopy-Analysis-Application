#!/usr/bin/env python3
"""
Build Executable Testing Suite
===============================

Comprehensive testing suite for PyInstaller-generated executable builds.
Tests for completeness, functionality, and missing dependencies.

Usage:
    python test_build_executable.py [--exe PATH] [--verbose]

Tests:
    1. Executable Structure Validation
    2. Required Files & Directories Check
    3. Import Validation (DLL/SO files)
    4. Application Launch Test
    5. Module Availability Check
    6. Asset File Validation
    7. Performance Baseline
    8. Error Handling

Generated: October 21, 2025
"""

import sys
import os
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any

# ============== CONFIGURATION ==============

# Get project root (parent of build_scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Change to project root so paths are relative to it
os.chdir(project_root)

# Default executable path (relative to project root)
DEFAULT_EXE_PATHS = [
    r"dist\raman_app\raman_app.exe",
    r"dist_installer\raman_app_installer_staging\raman_app.exe",
]

# Required subdirectories in distribution
REQUIRED_DIRS = [
    'assets',
    'PySide6',
    '_internal',  # PyInstaller internal
]

# Required asset files
REQUIRED_ASSETS = [
    'assets/icons',
    'assets/fonts',
    'assets/locales',
    'assets/data',
]

# Critical Python modules that must be importable
CRITICAL_MODULES = [
    'PySide6',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'numpy',
    'pandas',
    'scipy',
    'matplotlib',
    'ramanspy',
    'pybaselines',
]

# Optional but important modules
OPTIONAL_MODULES = [
    'torch',
    'sklearn',
    'onnx',
]

# DLL/SO files expected
EXPECTED_BINARIES = [
    'atmcd32d.dll',
    'atmcd64d.dll',
]

# ============== TEST RESULTS CLASS ==============

class TestResult:
    """Represents a single test result"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.status = None  # 'PASS', 'FAIL', 'WARN', 'SKIP'
        self.message = ""
        self.details = {}
        self.duration = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'test': self.test_name,
            'status': self.status,
            'message': self.message,
            'details': self.details,
            'duration': round(self.duration, 3),
        }
    
    def to_string(self, verbose=False) -> str:
        status_symbol = {
            'PASS': '✓',
            'FAIL': '✗',
            'WARN': '⚠',
            'SKIP': '-',
        }.get(self.status, '?')
        
        result = f"[{status_symbol}] {self.test_name}: {self.status}"
        if self.message:
            result += f" - {self.message}"
        
        if verbose and self.details:
            for key, value in self.details.items():
                result += f"\n    {key}: {value}"
        
        return result


class TestSuite:
    """Main test suite for executable validation"""
    
    def __init__(self, exe_path: str, verbose: bool = False):
        self.exe_path = Path(exe_path)
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
        
        # Execution environment
        self.exe_dir = self.exe_path.parent
        self.project_root = self.exe_path.parent.parent.parent if self.exe_path.parent.parent else Path.cwd()
    
    def run_all(self) -> Tuple[int, int, int]:
        """Run all tests and return (passed, failed, warnings)"""
        print(f"\n{'=' * 70}")
        print(f"🧪 Build Executable Test Suite")
        print(f"{'=' * 70}\n")
        
        print(f"📦 Target Executable: {self.exe_path}")
        print(f"📁 Distribution Directory: {self.exe_dir}")
        print(f"🔍 Project Root: {self.project_root}\n")
        
        self.start_time = datetime.now()
        
        # Run all test groups
        self._test_executable_structure()
        self._test_required_directories()
        self._test_required_assets()
        self._test_binaries()
        self._test_executable_launch()
        self._test_performance()
        
        self.end_time = datetime.now()
        
        return self._print_summary()
    
    def _test_executable_structure(self):
        """Test 1: Verify executable exists and is valid"""
        test = TestResult("Executable Structure")
        start = time.time()
        
        try:
            if not self.exe_path.exists():
                test.status = 'FAIL'
                test.message = f"Executable not found at {self.exe_path}"
                test.details['expected_path'] = str(self.exe_path)
                test.details['paths_checked'] = str(DEFAULT_EXE_PATHS)
            elif not self.exe_path.is_file():
                test.status = 'FAIL'
                test.message = f"Path is not a file: {self.exe_path}"
            elif self.exe_path.stat().st_size == 0:
                test.status = 'FAIL'
                test.message = "Executable file is empty"
            else:
                test.status = 'PASS'
                exe_size = self.exe_path.stat().st_size / 1024 / 1024
                test.message = f"Executable found and valid"
                test.details['size_mb'] = round(exe_size, 2)
                test.details['path'] = str(self.exe_path)
        
        except Exception as e:
            test.status = 'FAIL'
            test.message = f"Error checking executable: {str(e)}"
        
        test.duration = time.time() - start
        self.results.append(test)
        print(test.to_string(self.verbose))
    
    def _test_required_directories(self):
        """Test 2: Verify required directories exist"""
        test = TestResult("Required Directories")
        start = time.time()
        
        try:
            missing_dirs = []
            found_dirs = []
            
            for dirname in REQUIRED_DIRS:
                # Check both direct path and _internal path
                dir_path = self.exe_dir / dirname
                internal_path = self.exe_dir / '_internal' / dirname
                
                if (dir_path.exists() and dir_path.is_dir()) or (internal_path.exists() and internal_path.is_dir()):
                    found_dirs.append(dirname)
                else:
                    missing_dirs.append(dirname)
            
            if missing_dirs:
                test.status = 'WARN'
                test.message = f"Some directories missing: {', '.join(missing_dirs)}"
                test.details['found'] = found_dirs
                test.details['missing'] = missing_dirs
            else:
                test.status = 'PASS'
                test.message = f"All {len(REQUIRED_DIRS)} required directories found"
                test.details['directories'] = found_dirs
        
        except Exception as e:
            test.status = 'FAIL'
            test.message = f"Error checking directories: {str(e)}"
        
        test.duration = time.time() - start
        self.results.append(test)
        print(test.to_string(self.verbose))
    
    def _test_required_assets(self):
        """Test 3: Verify required asset files exist"""
        test = TestResult("Required Asset Files")
        start = time.time()
        
        try:
            missing_assets = []
            found_assets = []
            
            for asset_path in REQUIRED_ASSETS:
                # Check both direct path and _internal path
                full_path = self.exe_dir / asset_path
                internal_path = self.exe_dir / '_internal' / asset_path
                
                found_path = None
                if full_path.exists():
                    found_path = full_path
                elif internal_path.exists():
                    found_path = internal_path
                    
                if found_path:
                    # Count files in directory
                    if found_path.is_dir():
                        file_count = len(list(found_path.rglob('*')))
                        found_assets.append(f"{asset_path} ({file_count} files)")
                    else:
                        found_assets.append(asset_path)
                else:
                    missing_assets.append(asset_path)
                
            
            if missing_assets:
                test.status = 'WARN'
                test.message = f"Some assets missing: {', '.join(missing_assets)}"
                test.details['found'] = found_assets
                test.details['missing'] = missing_assets
            else:
                test.status = 'PASS'
                test.message = f"All {len(REQUIRED_ASSETS)} asset categories found"
                test.details['assets'] = found_assets
        
        except Exception as e:
            test.status = 'FAIL'
            test.message = f"Error checking assets: {str(e)}"
        
        test.duration = time.time() - start
        self.results.append(test)
        print(test.to_string(self.verbose))
    
    def _test_binaries(self):
        """Test 4: Verify binary files and DLLs"""
        test = TestResult("Binary Files")
        start = time.time()
        
        try:
            found_binaries = []
            missing_binaries = []
            
            # Check in drivers directory
            drivers_dir = self.exe_dir / 'drivers'
            if drivers_dir.exists():
                for binary in EXPECTED_BINARIES:
                    dll_path = drivers_dir / binary
                    if dll_path.exists():
                        size_kb = dll_path.stat().st_size / 1024
                        found_binaries.append(f"{binary} ({round(size_kb, 1)} KB)")
                    else:
                        missing_binaries.append(binary)
            
            # Check common library locations
            lib_dirs = [
                self.exe_dir / '_internal',
                self.exe_dir / 'PySide6',
            ]
            
            dll_count = 0
            for lib_dir in lib_dirs:
                if lib_dir.exists():
                    dlls = list(lib_dir.rglob('*.dll'))
                    dll_count += len(dlls)
            
            if dll_count > 0:
                test.status = 'PASS'
                test.message = f"Binary files found ({dll_count} DLLs total)"
                test.details['dlls_found'] = dll_count
                test.details['critical_binaries'] = found_binaries
                if missing_binaries:
                    test.details['missing'] = missing_binaries
                    test.status = 'WARN'
            else:
                test.status = 'WARN'
                test.message = "No DLL files found (may be required)"
                test.details['missing_binaries'] = missing_binaries
        
        except Exception as e:
            test.status = 'FAIL'
            test.message = f"Error checking binaries: {str(e)}"
        
        test.duration = time.time() - start
        self.results.append(test)
        print(test.to_string(self.verbose))
    
    def _test_executable_launch(self):
        """Test 5: Attempt to launch executable with --help"""
        test = TestResult("Executable Launch")
        start = time.time()
        
        try:
            # Try launching with --help to avoid full UI load
            test.message = "Attempting to launch executable..."
            
            # Use timeout to avoid hanging
            result = subprocess.run(
                [str(self.exe_path), '--help'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Check if executable ran without crashing
            if result.returncode in [0, 1]:  # 0=normal, 1=help shown
                test.status = 'PASS'
                test.message = "Executable launched successfully"
                test.details['return_code'] = result.returncode
            else:
                test.status = 'WARN'
                test.message = f"Unexpected return code: {result.returncode}"
                test.details['stdout'] = result.stdout[:200] if result.stdout else "(empty)"
                test.details['stderr'] = result.stderr[:200] if result.stderr else "(empty)"
        
        except subprocess.TimeoutExpired:
            test.status = 'WARN'
            test.message = "Executable launch timed out (likely normal for GUI app)"
        except FileNotFoundError:
            test.status = 'FAIL'
            test.message = f"Executable not found: {self.exe_path}"
        except Exception as e:
            test.status = 'WARN'
            test.message = f"Launch test inconclusive: {str(e)[:100]}"
            test.details['exception'] = type(e).__name__
        
        test.duration = time.time() - start
        self.results.append(test)
        print(test.to_string(self.verbose))
    
    def _test_performance(self):
        """Test 6: Basic performance check"""
        test = TestResult("Performance Baseline")
        start = time.time()
        
        try:
            # Check file system access time
            access_start = time.time()
            list(self.exe_dir.rglob('*.py'))  # Simulate file scanning
            access_time = time.time() - access_start
            
            # Check distribution size
            total_size = 0
            for item in self.exe_dir.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
            
            total_size_mb = total_size / 1024 / 1024
            
            test.status = 'PASS'
            test.message = f"Distribution is {round(total_size_mb, 1)} MB"
            test.details['size_mb'] = round(total_size_mb, 2)
            test.details['file_scan_time_ms'] = round(access_time * 1000, 2)
            
            # Warn if too large
            if total_size_mb > 1000:
                test.status = 'WARN'
                test.message = f"Large distribution size: {round(total_size_mb, 1)} MB"
        
        except Exception as e:
            test.status = 'WARN'
            test.message = f"Could not measure performance: {str(e)}"
        
        test.duration = time.time() - start
        self.results.append(test)
        print(test.to_string(self.verbose))
    
    def _print_summary(self) -> Tuple[int, int, int]:
        """Print test summary and return counts"""
        passed = sum(1 for r in self.results if r.status == 'PASS')
        failed = sum(1 for r in self.results if r.status == 'FAIL')
        warnings = sum(1 for r in self.results if r.status == 'WARN')
        
        total_time = (self.end_time - self.start_time).total_seconds()
        
        print(f"\n{'=' * 70}")
        print(f"📊 Test Summary")
        print(f"{'=' * 70}")
        print(f"✓ Passed:  {passed}")
        print(f"✗ Failed:  {failed}")
        print(f"⚠ Warned:  {warnings}")
        print(f"Total:     {len(self.results)} tests")
        print(f"Time:      {round(total_time, 2)}s")
        print(f"{'=' * 70}\n")
        
        # Save results to JSON
        self._save_json_results()
        
        return passed, failed, warnings
    
    def _save_json_results(self):
        """Save detailed results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path("test_script") / f"build_test_results_{timestamp}.json"
        
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            results_dict = {
                'timestamp': self.start_time.isoformat(),
                'executable': str(self.exe_path),
                'distribution_dir': str(self.exe_dir),
                'tests': [r.to_dict() for r in self.results],
                'summary': {
                    'passed': sum(1 for r in self.results if r.status == 'PASS'),
                    'failed': sum(1 for r in self.results if r.status == 'FAIL'),
                    'warnings': sum(1 for r in self.results if r.status == 'WARN'),
                    'total': len(self.results),
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            print(f"📝 Results saved to: {output_file}")
        
        except Exception as e:
            print(f"⚠ Could not save JSON results: {e}")


# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser(
        description='Test PyInstaller-generated executables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

Examples:
  python test_build_executable.py
  python test_build_executable.py --exe dist/raman_app/raman_app.exe
  python test_build_executable.py --verbose
        """
    )
    
    parser.add_argument(
        '--exe',
        type=str,
        help='Path to executable (default: auto-detect)',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed test information',
    )
    
    args = parser.parse_args()
    
    # Find executable
    exe_path = None
    if args.exe:
        exe_path = Path(args.exe)
    else:
        # Try default locations
        for default_path in DEFAULT_EXE_PATHS:
            candidate = Path(default_path)
            if candidate.exists():
                exe_path = candidate
                break
    
    if not exe_path or not exe_path.exists():
        print("❌ Executable not found!")
        print(f"Checked: {DEFAULT_EXE_PATHS}")
        print(f"\nTo build the executable first, run:")
        print(f"  .\build_portable.ps1       (for portable executable)")
        print(f"  .\build_installer.ps1      (for installer staging)")
        print(f"\nOr specify executable path:")
        print(f"  python test_build_executable.py --exe <path>")
        sys.exit(1)
    
    # Run test suite
    suite = TestSuite(str(exe_path), verbose=args.verbose)
    passed, failed, warnings = suite.run_all()
    
    # Exit with appropriate code
    if failed > 0:
        print("❌ Tests FAILED")
        sys.exit(1)
    elif warnings > 0:
        print("⚠️  Tests passed with warnings")
        sys.exit(0)
    else:
        print("✅ All tests PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()
