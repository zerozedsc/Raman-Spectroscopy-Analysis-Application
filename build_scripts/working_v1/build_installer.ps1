# Build script for Raman App Windows Installer (NSIS)
# Usage: .\build_installer.ps1
# This script builds the executable staging files for NSIS installer

param(
    [switch]$Clean = $false,
    [switch]$Debug = $false,
    [switch]$BuildOnly = $false,
    [switch]$Console = $false,
    [ValidateSet('DEBUG', 'INFO', 'WARNING', 'ERROR')]
    [string]$LogLevel = 'WARNING'
)

# Colors for output
$Colors = @{
    Success = 'Green'
    Error   = 'Red'
    Warning = 'Yellow'
    Info    = 'Cyan'
    Section = 'Magenta'
}

function Write-Status {
    param([string]$Message, [string]$Type = 'Info')
    $Color = $Colors[$Type]
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $Message" -ForegroundColor $Color
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor $Colors.Section
    Write-Host $Title -ForegroundColor $Colors.Section
    Write-Host ("=" * 70) -ForegroundColor $Colors.Section
}

try {
    Write-Section "Raman App Installer Build (NSIS)"
    
    # Get project root directory (parent of build_scripts)
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $ProjectRoot = Split-Path -Parent $ScriptDir
    
    # Change to project root so spec file can find relative paths
    Push-Location $ProjectRoot
    Write-Status "Project root: $ProjectRoot" 'Info'
    Write-Status "Working directory: $(Get-Location)" 'Info'
    
    # ============== ENVIRONMENT CHECK ==============
    Write-Section "Environment Check"
    
    # Check Python
    Write-Status "Checking Python environment..." 'Info'
    $PythonVersion = python --version 2>&1
    Write-Status "Python: $PythonVersion" 'Success'
    
    # Check PyInstaller
    Write-Status "Checking PyInstaller installation..." 'Info'
    $PyInstallerVersion = pyinstaller --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Status "PyInstaller: $PyInstallerVersion" 'Success'
    }
    else {
        Write-Status "ERROR: PyInstaller not found!" 'Error'
        exit 1
    }
    
    # Check for NSIS (optional but recommended)
    $NSISPath = "C:\Program Files (x86)\NSIS\makensis.exe"
    if (Test-Path $NSISPath) {
        Write-Status "NSIS found at: $NSISPath" 'Success'
        $HasNSIS = $true
    }
    else {
        Write-Status "NSIS not found (installer creation will be skipped)" 'Warning'
        Write-Status "Download from: https://nsis.sourceforge.io/" 'Info'
        $HasNSIS = $false
    }
    
    # Check spec file
    if (-not (Test-Path "raman_app_installer.spec")) {
        Write-Status "ERROR: raman_app_installer.spec not found!" 'Error'
        exit 1
    }
    Write-Status "Spec file found: raman_app_installer.spec" 'Success'
    
    # ============== CLEANUP WITH BACKUP ==============
    if ($Clean) {
        Write-Section "Cleaning Previous Builds"
        
        # Create backup timestamp
        $BackupTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $BackupDir = "build_backups\backup_installer_$BackupTimestamp"
        $HasBackup = $false
        
        # Check if there's anything to backup
        $DirsToClean = @('build', 'build_installer', 'dist_installer')
        foreach ($Dir in $DirsToClean) {
            if (Test-Path $Dir) {
                # Create backup directory if needed
                if (-not $HasBackup) {
                    New-Item -Path $BackupDir -ItemType Directory -Force | Out-Null
                    Write-Status "Created backup directory: $BackupDir" 'Info'
                    $HasBackup = $true
                }
                
                # Move to backup instead of deleting
                $BackupTarget = Join-Path $BackupDir $Dir
                Write-Status "Backing up $Dir/ to $BackupTarget" 'Info'
                Move-Item $Dir $BackupTarget -Force -ErrorAction SilentlyContinue
            }
        }
        
        if ($HasBackup) {
            Write-Status "Previous builds backed up to: $BackupDir" 'Success'
        }
        else {
            Write-Status "No previous builds to clean" 'Info'
        }
        Write-Status "Cleanup complete" 'Success'
    }
    
    # ============== PRE-BUILD CHECKS ==============
    Write-Section "Pre-Build Validation"
    
    # Check pyproject.toml
    Write-Status "Checking project files..." 'Info'
    if ((Test-Path "pyproject.toml") -and (Test-Path "main.py")) {
        Write-Status "Project files found" 'Success'
    }
    else {
        Write-Status "ERROR: Required files missing!" 'Error'
        exit 1
    }
    
    # ============== BUILD EXECUTABLE FOR INSTALLER ==============
    Write-Section "Building Executable (Installer Staging)"
    
    $BuildArgs = @(
        '--distpath', 'dist_installer',
        '--workpath', 'build_installer'
    )
    
    if ($Debug) {
        $BuildArgs += '--debug'
        $BuildArgs += 'all'
        Write-Status "Debug mode enabled" 'Warning'
    }
    
    # Append spec file last so PyInstaller applies options correctly
    $BuildArgs += 'raman_app_installer.spec'

    Write-Status "Building executable for installer..." 'Info'
    
    $StartTime = Get-Date
    & pyinstaller @BuildArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Status "ERROR: PyInstaller build failed!" 'Error'
        exit 1
    }
    
    $EndTime = Get-Date
    $BuildTime = ($EndTime - $StartTime).TotalSeconds
    Write-Status "Build completed in $([Math]::Round($BuildTime, 2)) seconds" 'Success'
    
    # ============== VERIFY BUILD OUTPUT ==============
    Write-Section "Build Output Verification"
    
    $StagingDir = "dist_installer\raman_app_installer_staging"
    $ExePath = Join-Path $StagingDir "raman_app.exe"
    
    if (Test-Path $ExePath) {
        Write-Status "Executable created successfully" 'Success'
        
        $ExeSize = (Get-Item $ExePath).Length / 1MB
        Write-Status "Executable size: $([Math]::Round($ExeSize, 2)) MB" 'Info'
        
        $DirSize = (Get-ChildItem -Path $StagingDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
        Write-Status "Total staging size: $([Math]::Round($DirSize, 2)) MB" 'Info'
    }
    else {
        Write-Status "ERROR: Executable not created!" 'Error'
        exit 1
    }
    
    if ($BuildOnly) {
        Write-Status "Build-only mode: Stopping before NSIS processing" 'Info'
    }
    else {
        # ============== NSIS INSTALLER CREATION ==============
        if ($HasNSIS) {
            Write-Section "Creating NSIS Installer"
            
            # Check for NSIS script
            $NSISScript = "raman_app_installer.nsi"
            if (-not (Test-Path $NSISScript)) {
                Write-Status "NSIS script not found: $NSISScript" 'Warning'
                Write-Status "Installer creation skipped. Generate one using: makensis /HELP" 'Info'
                Write-Status "Or create template: $NSISScript" 'Info'
            }
            else {
                Write-Status "Building installer from: $NSISScript" 'Info'
                
                # Run NSIS compiler
                $NSISArgs = @(
                    "/DPROJECT_ROOT=$ProjectRoot",
                    "/DSTAGING_DIR=$StagingDir",
                    $NSISScript
                )
                
                Write-Status "Running NSIS compiler..." 'Info'
                & $NSISPath @NSISArgs
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Status "Installer created successfully" 'Success'
                    
                    # Find installer output
                    $Installers = Get-ChildItem -Path "." -Filter "raman_app_installer*.exe" -ErrorAction SilentlyContinue
                    foreach ($Installer in $Installers) {
                        $Size = ($Installer.Length / 1MB)
                        $SizeRounded = [Math]::Round($Size, 2)
                        Write-Status "Output: $($Installer.Name) ($SizeRounded MB)" 'Success'
                    }
                }
                else {
                    Write-Status "NSIS compilation failed (return code: $LASTEXITCODE)" 'Warning'
                }
            }
        }
        else {
            Write-Status "NSIS not found - skipping installer creation" 'Warning'
            Write-Status "To create installer, install NSIS and re-run with NSIS script" 'Info'
        }
    }
    
    # ============== BUILD SUMMARY ==============
    Write-Section "Build Summary"
    
    Write-Status "Build type: Installer (Staging)" 'Info'
    Write-Status "Staging directory: $StagingDir\" 'Info'
    Write-Status "Build time: $([Math]::Round($BuildTime, 2))s" 'Info'
    Write-Status "Total size: $([Math]::Round($DirSize, 2)) MB" 'Info'
    
    if ($HasNSIS) {
        if (Test-Path $NSISScript) {
            Write-Status "NSIS installer: Ready" 'Success'
        }
        else {
            Write-Status "NSIS script: Not found (raman_app_installer.nsi)" 'Warning'
        }
    }
    
    # ============== NEXT STEPS ==============
    Write-Section "Next Steps"
    
    Write-Host ""
    Write-Host "Portable Build (Recommended for Testing):" -ForegroundColor $Colors.Info
    Write-Host "  1. Run: .\build_portable.ps1" -ForegroundColor $Colors.Info
    Write-Host "  2. Test: .\dist\raman_app\raman_app.exe" -ForegroundColor $Colors.Info
    Write-Host ""
    
    Write-Host "Installer Requirements:" -ForegroundColor $Colors.Info
    Write-Host "  1. Install NSIS from: https://nsis.sourceforge.io/" -ForegroundColor $Colors.Info
    Write-Host "  2. Create raman_app_installer.nsi script" -ForegroundColor $Colors.Info
    Write-Host "  3. Re-run this script" -ForegroundColor $Colors.Info
    Write-Host ""
    
    Write-Host "Testing:" -ForegroundColor $Colors.Info
    Write-Host "  python test_build_executable.py --exe dist_installer/raman_app_installer_staging/raman_app.exe" -ForegroundColor $Colors.Info
    Write-Host ""
    
    Write-Status "Installer staging build complete!" 'Success'
    
    # Restore original directory
    Pop-Location
}
catch {
    # Restore original directory on error
    Pop-Location -ErrorAction SilentlyContinue
    Write-Status "FATAL ERROR: $_" 'Error'
    Write-Status $_.ScriptStackTrace 'Error'
    exit 1
}
