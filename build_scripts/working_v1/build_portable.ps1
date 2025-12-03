# Build script for portable Raman App executable
# Usage: .\build_portable.ps1
# This script builds a standalone portable executable for Windows

param(
    [switch]$Clean = $false,
    [switch]$Debug = $false,
    [string]$OutputDir = "dist",
    [switch]$NoCompress = $false,
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
    Write-Section "Raman App Portable Executable Build"
    
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
        Write-Status "PyInstaller not found. Installing..." 'Warning'
        pip install pyinstaller
    }
    
    # Check spec file
    if (-not (Test-Path "raman_app.spec")) {
        Write-Status "ERROR: raman_app.spec not found!" 'Error'
        exit 1
    }
    Write-Status "Spec file found: raman_app.spec" 'Success'
    
    # ============== CLEANUP WITH BACKUP ==============
    if ($Clean) {
        Write-Section "Cleaning Previous Builds"
        
        # Create backup timestamp
        $BackupTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $BackupDir = "build_backups\backup_$BackupTimestamp"
        $HasBackup = $false
        
        # Check if there's anything to backup
        $DirsToClean = @('build', 'dist')
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
    Write-Status "Checking pyproject.toml..." 'Info'
    if (Test-Path "pyproject.toml") {
        Write-Status "pyproject.toml found" 'Success'
        
        # Extract version
        $VersionLine = Select-String -Path "pyproject.toml" -Pattern 'version\s*=' | Select-Object -First 1
        Write-Status "Version: $VersionLine" 'Info'
    }
    
    # Check main entry point
    if (Test-Path "main.py") {
        Write-Status "main.py found" 'Success'
    }
    else {
        Write-Status "ERROR: main.py not found!" 'Error'
        exit 1
    }
    
    # Check assets directory
    if (Test-Path "assets") {
        Write-Status "assets directory found" 'Success'
        $AssetCount = (Get-ChildItem -Path "assets" -Recurse | Measure-Object).Count
        Write-Status "Assets count: $AssetCount files" 'Info'
    }
    
    # ============== BUILD EXECUTABLE ==============
    Write-Section "Building Portable Executable"
    
    # Set environment variable for log level (used at runtime)
    $env:RAMAN_LOG_LEVEL = $LogLevel
    Write-Status "Log level set to: $LogLevel" 'Info'
    
    # Modify spec file for console mode if requested
    if ($Console) {
        Write-Status "Console mode enabled - updating spec file temporarily" 'Warning'
        $SpecContent = Get-Content "raman_app.spec" -Raw
        $SpecContent = $SpecContent -replace "console=False", "console=True"
        $SpecContent | Set-Content "raman_app_temp.spec"
        $SpecFile = 'raman_app_temp.spec'
    }
    else {
        $SpecFile = 'raman_app.spec'
    }
    
    $BuildArgs = @(
        '--distpath', $OutputDir,
        '--workpath', 'build'
    )
    
    if ($Debug) {
        $BuildArgs += '--debug'
        $BuildArgs += 'all'
        Write-Status "Debug mode enabled" 'Warning'
    }
    
    if ($NoCompress) {
        Write-Status "Compression disabled" 'Warning'
    }

    # Append spec file as final argument so PyInstaller applies options correctly
    $BuildArgs += $SpecFile
    
    Write-Status "Building with PyInstaller..." 'Info'
    Write-Status "Command: pyinstaller $($BuildArgs -join ' ')" 'Info'
    
    $StartTime = Get-Date
    
    & pyinstaller @BuildArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Status "ERROR: PyInstaller build failed!" 'Error'
        exit 1
    }
    
    $EndTime = Get-Date
    $BuildTime = ($EndTime - $StartTime).TotalSeconds
    Write-Status "Build completed in $([Math]::Round($BuildTime, 2)) seconds" 'Success'
    
    # ============== POST-BUILD VALIDATION ==============
    Write-Section "Post-Build Validation"
    
    $ExePath = Join-Path $OutputDir "raman_app" "raman_app.exe"
    
    if (Test-Path $ExePath) {
        Write-Status "Executable created: $ExePath" 'Success'
        
        $ExeSize = (Get-Item -LiteralPath $ExePath).Length
        $ExeSizeMB = [Math]::Round($ExeSize / 1MB, 2)
        Write-Status "Executable size: $ExeSizeMB MB" 'Info'
        
        $DirPath = Join-Path $OutputDir "raman_app"
        $DirSize = (Get-ChildItem -LiteralPath $DirPath -Recurse | Measure-Object -Property Length -Sum).Sum
        $DirSizeMB = [Math]::Round($DirSize / 1MB, 2)
        Write-Status "Total distribution size: $DirSizeMB MB" 'Info'
    }
    else {
        Write-Status "ERROR: Executable not created at expected location!" 'Error'
        exit 1
    }
    
    # Check for required directories/files in output
    $RequiredItems = @('assets', 'PySide6', 'drivers')
    $MissingItems = @()
    
    foreach ($Item in $RequiredItems) {
        $ItemPath = Join-Path $OutputDir "raman_app" $Item
        if (Test-Path $ItemPath) {
            Write-Status "Found: $Item" 'Success'
        }
        else {
            Write-Status "Missing: $Item (may be required)" 'Warning'
            $MissingItems += $Item
        }
    }
    
    # ============== BUILD SUMMARY ==============
    Write-Section "Build Summary"
    
    Write-Status "Build type: Portable Executable" 'Info'
    Write-Status "Output location: $OutputDir\raman_app\" 'Info'
    Write-Status "Executable: raman_app.exe" 'Success'
    Write-Status "Build time: $([Math]::Round($BuildTime, 2))s" 'Info'
    Write-Status "Total size: $DirSizeMB MB" 'Info'
    
    if ($MissingItems.Count -gt 0) {
        Write-Status "Warning: Some items were not included ($($MissingItems -join ', '))" 'Warning'
    }
    else {
        Write-Status "All required components included" 'Success'
    }
    
    # ============== NEXT STEPS ==============
    Write-Section "Next Steps"
    
    Write-Host ""
    Write-Host "1. Test the executable:" -ForegroundColor $Colors.Info
    Write-Host "   .\$OutputDir\raman_app\raman_app.exe" -ForegroundColor $Colors.Info
    Write-Host ""
    Write-Host "2. Run test suite:" -ForegroundColor $Colors.Info
    Write-Host "   python test_build_executable.py --exe dist/raman_app/raman_app.exe" -ForegroundColor $Colors.Info
    Write-Host ""
    Write-Host "3. For installer build:" -ForegroundColor $Colors.Info
    Write-Host "   .\build_installer.ps1" -ForegroundColor $Colors.Info
    Write-Host ""
    
    Write-Status "Portable build complete!" 'Success'
    
    # Cleanup temporary spec file if created
    if ($Console -and (Test-Path "raman_app_temp.spec")) {
        Remove-Item "raman_app_temp.spec" -Force
        Write-Status "Cleaned up temporary spec file" 'Info'
    }
    
    # Restore original directory
    Pop-Location
}
catch {
    # Cleanup temporary spec file on error
    if ($Console -and (Test-Path "raman_app_temp.spec")) {
        Remove-Item "raman_app_temp.spec" -Force -ErrorAction SilentlyContinue
    }
    
    # Restore original directory on error
    Pop-Location -ErrorAction SilentlyContinue
    Write-Status "FATAL ERROR: $_" 'Error'
    Write-Status $_.ScriptStackTrace 'Error'
    exit 1
}
