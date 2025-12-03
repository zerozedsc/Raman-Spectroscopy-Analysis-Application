# Quick Build Script for Optimized Portable Executable
# Regenerates configs and builds with all optimizations

param(
    [switch]$SkipRegenerate = $false,
    [switch]$Console = $false,
    [switch]$Clean = $true
)

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
    Write-Host ("=" * 80) -ForegroundColor $Colors.Section
    Write-Host $Title -ForegroundColor $Colors.Section
    Write-Host ("=" * 80) -ForegroundColor $Colors.Section
}

try {
    Write-Section "Optimized Build - Quick Start"
    
    # Navigate to project root
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $ProjectRoot = Split-Path -Parent $ScriptDir
    Push-Location $ProjectRoot
    
    Write-Status "Project root: $ProjectRoot" 'Info'
    
    # Check for required files
    Write-Section "Pre-Build Verification"
    
    # Critical files that must exist
    $RequiredFiles = @(
        'main.py'
    )
    
    # Optional files with fallbacks
    $OptionalFiles = @{
        'assets/splash.png' = 'Will use auto-generated gradient background'
    }
    
    # Check required files
    $MissingFiles = @()
    foreach ($File in $RequiredFiles) {
        if (Test-Path $File) {
            Write-Status "[OK] Found: $File" 'Success'
        }
        else {
            Write-Status "[MISSING] Missing: $File" 'Error'
            $MissingFiles += $File
        }
    }
    
    # Check optional files
    foreach ($File in $OptionalFiles.Keys) {
        if (Test-Path $File) {
            Write-Status "[OK] Found: $File" 'Success'
        }
        else {
            Write-Status "[OPTIONAL] Missing: $File" 'Warning'
            Write-Status "  Fallback: $($OptionalFiles[$File])" 'Info'
        }
    }
    
    if ($MissingFiles.Count -gt 0) {
        Write-Status "CRITICAL: Missing required files!" 'Error'
        Write-Host ""
        Write-Host "Missing files:" -ForegroundColor Red
        foreach ($File in $MissingFiles) {
            Write-Host "  - $File" -ForegroundColor Red
        }
        exit 1
    }
    
    # Regenerate build configs
    if (-not $SkipRegenerate) {
        Write-Section "Regenerating Build Configurations"
        Write-Status "Running generate_build_configs.py..." 'Info'
        
        python build_scripts/generate_build_configs.py
        
        if ($LASTEXITCODE -ne 0) {
            Write-Status "Config generation failed!" 'Error'
            exit 1
        }
        
        Write-Status "Build configs regenerated successfully" 'Success'
    }
    else {
        # Use single quotes to prevent '--' being interpreted as operator
        Write-Status 'Skipping config regeneration (--SkipRegenerate)' 'Warning'
    }
    
    # Build executable
    Write-Section "Building Optimized Executable"
    
    $BuildScript = ".\build_scripts\build_portable.ps1"
    Write-Status "Calling: $BuildScript" 'Info'
    
    # Call the build script using explicit parameter binding to prevent errors
    # We force OutputDir to 'dist' to prevent -Clean from shifting positions
    & $BuildScript -Clean:$Clean -Console:$Console -OutputDir "dist"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Status "Build failed!" 'Error'
        exit 1
    }
    
    # Success summary
    Write-Section "Build Complete - Performance Summary"
    
    $ExePath = "dist\raman_app\raman_app.exe"
    if (Test-Path $ExePath) {
        $ExeSize = (Get-Item $ExePath).Length / 1MB
        $DirSize = (Get-ChildItem -Path "dist\raman_app" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
        
        Write-Host ""
        Write-Status "[OK] Executable: $ExePath" 'Success'
        Write-Status "[OK] Exe size: $([Math]::Round($ExeSize, 2)) MB" 'Info'
        Write-Status "[OK] Total size: $([Math]::Round($DirSize, 2)) MB" 'Info'
        Write-Host ""
        
        Write-Host "Optimizations Applied:" -ForegroundColor Cyan
        Write-Host "  [+] Splash screen integration" -ForegroundColor Green
        Write-Host "  [+] Excluded unused modules (watchdog, tkinter, etc)" -ForegroundColor Green
        Write-Host "  [+] UPX compression enabled" -ForegroundColor Green
        Write-Host "  [+] One-Dir mode for faster extraction" -ForegroundColor Green
        Write-Host ""
    }
    
    # Next steps
    Write-Section "Next Steps"
    Write-Host ""
    Write-Host "1. Test the executable:" -ForegroundColor Cyan
    Write-Host "   .\dist\raman_app\raman_app.exe" -ForegroundColor White
    Write-Host ""
    Write-Host "2. Verify splash screen appears quickly" -ForegroundColor Cyan
    Write-Host "   (Should see splash within 1-2 seconds)" -ForegroundColor White
    Write-Host ""
    
    Write-Status "All done!" 'Success'
    
    Pop-Location
}
catch {
    Pop-Location -ErrorAction SilentlyContinue
    Write-Status "FATAL ERROR: $_" 'Error'
    Write-Status $_.ScriptStackTrace 'Error'
    exit 1
}