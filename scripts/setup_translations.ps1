# Documentation Translation Quick Setup
# Run this script from the project root directory

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Sphinx Gettext Translation Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "docs/en/conf.py")) {
    Write-Host "Error: Please run this script from the project root directory" -ForegroundColor Red
    Write-Host "Expected to find: docs/en/conf.py" -ForegroundColor Red
    exit 1
}

Write-Host "[1/7] Checking Python environment..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "  Found: $pythonVersion" -ForegroundColor Green

Write-Host ""
Write-Host "[2/7] Installing required packages..." -ForegroundColor Yellow
Write-Host "  Installing sphinx-intl..." -ForegroundColor Gray
pip install --quiet sphinx-intl
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ sphinx-intl installed" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to install sphinx-intl" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[3/7] Creating directory structure..." -ForegroundColor Yellow

# Create locale directory
if (-not (Test-Path "docs/locale")) {
    New-Item -ItemType Directory -Path "docs/locale" | Out-Null
    Write-Host "  ✓ Created docs/locale/" -ForegroundColor Green
} else {
    Write-Host "  → docs/locale/ already exists" -ForegroundColor Gray
}

# Create _build directory
if (-not (Test-Path "docs/_build")) {
    New-Item -ItemType Directory -Path "docs/_build" | Out-Null
    Write-Host "  ✓ Created docs/_build/" -ForegroundColor Green
} else {
    Write-Host "  → docs/_build/ already exists" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[4/7] Backing up current Japanese docs..." -ForegroundColor Yellow
if (Test-Path "docs/ja") {
    if (-not (Test-Path "docs/ja_archive")) {
        Write-Host "  Moving docs/ja/ → docs/ja_archive/" -ForegroundColor Gray
        Move-Item "docs/ja" "docs/ja_archive"
        Write-Host "  ✓ Japanese docs archived for reference" -ForegroundColor Green
    } else {
        Write-Host "  → docs/ja_archive/ already exists, skipping backup" -ForegroundColor Yellow
    }
} else {
    Write-Host "  → No docs/ja/ to archive" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[5/7] Generating translatable strings (.pot files)..." -ForegroundColor Yellow
Write-Host "  Running: sphinx-build -b gettext docs/en docs/_build/gettext" -ForegroundColor Gray

Push-Location
Set-Location "docs"
$output = sphinx-build -b gettext en _build/gettext 2>&1
Pop-Location

if ($LASTEXITCODE -eq 0) {
    $potFiles = Get-ChildItem -Path "docs/_build/gettext" -Filter "*.pot" -Recurse
    Write-Host "  ✓ Generated $($potFiles.Count) .pot files" -ForegroundColor Green
    
    # Show first few files
    $potFiles | Select-Object -First 3 | ForEach-Object {
        Write-Host "    - $($_.Name)" -ForegroundColor Gray
    }
    if ($potFiles.Count -gt 3) {
        Write-Host "    ... and $($potFiles.Count - 3) more" -ForegroundColor Gray
    }
} else {
    Write-Host "  ✗ Failed to generate .pot files" -ForegroundColor Red
    Write-Host $output -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[6/7] Creating Japanese translation files (.po)..." -ForegroundColor Yellow
Write-Host "  Running: sphinx-intl update -p docs/_build/gettext -l ja" -ForegroundColor Gray

Push-Location
Set-Location "docs"
$output = sphinx-intl update -p _build/gettext -l ja 2>&1
Pop-Location

if ($LASTEXITCODE -eq 0) {
    $poFiles = Get-ChildItem -Path "docs/locale/ja/LC_MESSAGES" -Filter "*.po" -Recurse
    Write-Host "  ✓ Created $($poFiles.Count) .po files" -ForegroundColor Green
    
    # Show first few files
    $poFiles | Select-Object -First 3 | ForEach-Object {
        Write-Host "    - $($_.Name)" -ForegroundColor Gray
    }
    if ($poFiles.Count -gt 3) {
        Write-Host "    ... and $($poFiles.Count - 3) more" -ForegroundColor Gray
    }
} else {
    Write-Host "  ✗ Failed to create .po files" -ForegroundColor Red
    Write-Host $output -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[7/7] Creating helper scripts..." -ForegroundColor Yellow

# Create build script
$buildScript = @'
# Build Documentation (All Languages)
# Usage: .\docs\build_docs.ps1

$ErrorActionPreference = "Stop"

Write-Host "Building Raman Spectroscopy Documentation..." -ForegroundColor Cyan
Write-Host ""

# Build English
Write-Host "[1/2] Building English documentation..." -ForegroundColor Yellow
Set-Location docs
sphinx-build -b html en _build/html
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ English build complete" -ForegroundColor Green
} else {
    Write-Host "  ✗ English build failed" -ForegroundColor Red
    exit 1
}

# Build Japanese
Write-Host ""
Write-Host "[2/2] Building Japanese documentation..." -ForegroundColor Yellow
sphinx-build -b html -D language=ja en _build/html/ja
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Japanese build complete" -ForegroundColor Green
} else {
    Write-Host "  ✗ Japanese build failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Documentation Build Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Output locations:" -ForegroundColor Cyan
Write-Host "  English:  docs\_build\html\index.html" -ForegroundColor White
Write-Host "  Japanese: docs\_build\html\ja\index.html" -ForegroundColor White
Write-Host ""
Write-Host "Open in browser?" -ForegroundColor Yellow
$open = Read-Host "  [Y/n]"
if ($open -ne "n") {
    Start-Process "_build\html\index.html"
}

Set-Location ..
'@
Set-Content -Path "docs/build_docs.ps1" -Value $buildScript -Encoding UTF8
Write-Host "  ✓ Created docs/build_docs.ps1" -ForegroundColor Green

# Create update translations script
$updateScript = @'
# Update Translation Files
# Run this after modifying English documentation
# Usage: .\docs\update_translations.ps1

$ErrorActionPreference = "Stop"

Write-Host "Updating Translation Files..." -ForegroundColor Cyan
Write-Host ""

Set-Location docs

# Regenerate .pot files
Write-Host "[1/2] Extracting translatable strings..." -ForegroundColor Yellow
sphinx-build -b gettext en _build/gettext
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ .pot files updated" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to generate .pot files" -ForegroundColor Red
    Set-Location ..
    exit 1
}

# Update .po files
Write-Host ""
Write-Host "[2/2] Updating .po files..." -ForegroundColor Yellow
sphinx-intl update -p _build/gettext -l ja
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ .po files updated" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to update .po files" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Translations Updated!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Translate new/modified strings in:" -ForegroundColor White
Write-Host "     docs\locale\ja\LC_MESSAGES\*.po" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Rebuild documentation:" -ForegroundColor White
Write-Host "     .\docs\build_docs.ps1" -ForegroundColor Gray
Write-Host ""

Set-Location ..
'@
Set-Content -Path "docs/update_translations.ps1" -Value $updateScript -Encoding UTF8
Write-Host "  ✓ Created docs/update_translations.ps1" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Translation infrastructure is ready!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Directory structure created:" -ForegroundColor Yellow
Write-Host "  docs/" -ForegroundColor White
Write-Host "  ├── _build/gettext/       # Generated .pot templates" -ForegroundColor Gray
Write-Host "  ├── locale/ja/            # Japanese translations (.po)" -ForegroundColor Gray
Write-Host "  ├── ja_archive/           # Original JA docs (reference)" -ForegroundColor Gray
Write-Host "  ├── build_docs.ps1        # Build script" -ForegroundColor Gray
Write-Host "  └── update_translations.ps1  # Update script" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Translate strings in .po files:" -ForegroundColor Yellow
Write-Host "   - Edit: docs\locale\ja\LC_MESSAGES\*.po" -ForegroundColor White
Write-Host "   - Use text editor or Poedit" -ForegroundColor Gray
Write-Host "   - Fill msgstr with Japanese translations" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Build documentation:" -ForegroundColor Yellow
Write-Host "   .\docs\build_docs.ps1" -ForegroundColor White
Write-Host ""
Write-Host "3. View in browser:" -ForegroundColor Yellow
Write-Host "   - English:  docs\_build\html\index.html" -ForegroundColor White
Write-Host "   - Japanese: docs\_build\html\ja\index.html" -ForegroundColor White
Write-Host ""
Write-Host "For detailed guidance, see:" -ForegroundColor Cyan
Write-Host "  .docs\TRANSLATION_SETUP.md" -ForegroundColor White
Write-Host ""
