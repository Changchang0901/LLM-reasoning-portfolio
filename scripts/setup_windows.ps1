[CmdletBinding()]
param(
    [string]$PythonCommand = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$venvPython = if (Test-Path -LiteralPath ".venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { $null }

if (-not $venvPython) {
    if ($PythonCommand) {
        & $PythonCommand -m venv .venv
    }
    elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv .venv
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonPlatform = & python -c "import sysconfig; print(sysconfig.get_platform())"
        if (-not $pythonPlatform.StartsWith("win")) {
            throw "The python command is not a standard Windows Python installation. Install 64-bit CPython 3.11 from python.org."
        }
        & python -m venv .venv
    }
    else {
        throw "Python was not found. Install 64-bit Python 3.11 or pass -PythonCommand with its full path."
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create .venv. Install 64-bit Python 3.11, or pass -PythonCommand with its full path."
    }
    $venvPython = if (Test-Path -LiteralPath ".venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else {
        throw "The virtual environment was created, but it is not a native Windows environment. Install 64-bit CPython 3.11 from python.org."
    }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

& $venvPython -m pip install --upgrade -r requirements-windows.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }

Write-Host "Windows API environment is ready: $projectRoot\.venv" -ForegroundColor Green
