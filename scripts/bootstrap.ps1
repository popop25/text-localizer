[CmdletBinding()]
param(
    [switch]$SkipDevDependencies
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Resolve-ProjectPython {
    if ($env:PYTHON_BIN -and (Test-Path -LiteralPath $env:PYTHON_BIN)) {
        return $env:PYTHON_BIN
    }

    $KnownPython = 'C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe'
    if (Test-Path -LiteralPath $KnownPython) {
        return $KnownPython
    }

    $Command = Get-Command python -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }

    throw 'Python 3.12 was not found. Set PYTHON_BIN to the Windows python.exe path.'
}

$Python = Resolve-ProjectPython
Write-Host "Using Python: $Python"

Push-Location $ProjectRoot
try {
    & $Python -m venv .venv
    $VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    & $VenvPython -m pip install --upgrade pip
    if ($SkipDevDependencies) {
        & $VenvPython -m pip install --editable .
    }
    else {
        & $VenvPython -m pip install --editable '.[dev]'
    }
    & $VenvPython -m text_localizer doctor --json
}
finally {
    Pop-Location
}
