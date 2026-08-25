[CmdletBinding()]
param(
    [ValidateSet('ocr', 'translation', 'all')]
    [string]$Model = 'all'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Project virtual environment is missing. Run scripts\bootstrap.ps1 first.'
}

Push-Location $ProjectRoot
try {
    if ($Model -eq 'ocr') {
        & $Python -m pip install --editable '.[ocr]'
        & $Python -m text_localizer setup-models --ocr
    }
    elseif ($Model -eq 'translation') {
        & $Python -m pip install --editable '.[translation]'
        & $Python -m text_localizer setup-models --translation
    }
    else {
        & $Python -m pip install --editable '.[ocr,translation]'
        & $Python -m text_localizer setup-models --ocr --translation
    }
}
finally {
    Pop-Location
}
