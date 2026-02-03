# VoiceShield - Install dependencies (PowerShell)
# Run from ml-service: .\install_deps.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Checking for Python..." -ForegroundColor Cyan
$pythonCmd = $null
foreach ($cmd in @("python", "py", "python3")) {
    try {
        $v = & $cmd --version 2>&1
        if ($v -match "Python 3\.\d+") {
            $pythonCmd = $cmd
            Write-Host "Found: $cmd - $v" -ForegroundColor Green
            break
        }
    } catch {
        # ignore
    }
}

if (-not $pythonCmd) {
    Write-Host ""
    Write-Host "Python not found or only the Store stub is on PATH." -ForegroundColor Red
    Write-Host "  - Install from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "  - In the installer, CHECK 'Add python.exe to PATH'" -ForegroundColor Yellow
    Write-Host "  - Then close and reopen this terminal and run this script again." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "Installing dependencies with: $pythonCmd -m pip install -r requirements.txt" -ForegroundColor Cyan
& $pythonCmd -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Install failed. Try: $pythonCmd -m pip install --upgrade pip" -ForegroundColor Yellow
    exit 1
}
Write-Host ""
Write-Host "Done. Next: run training then web app (see SETUP_WINDOWS.md)." -ForegroundColor Green
