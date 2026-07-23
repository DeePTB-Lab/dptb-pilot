$ErrorActionPreference = "Stop"

$pluginRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$repoRoot = Resolve-Path (Join-Path $pluginRoot "..\..")
$server = Join-Path $PSScriptRoot "server.py"

function Get-ConfigPath {
    if ($env:DPTB_PILOT_CONFIG) {
        return $env:DPTB_PILOT_CONFIG
    }
    return (Join-Path $HOME ".dptb-pilot-codex\config.json")
}

function Get-ConfiguredPython {
    if ($env:DPTB_PILOT_PYTHON -and (Test-Path -LiteralPath $env:DPTB_PILOT_PYTHON)) {
        return $env:DPTB_PILOT_PYTHON
    }

    $configPath = Get-ConfigPath
    if (Test-Path -LiteralPath $configPath) {
        try {
            $config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
            if ($config.python_path -and (Test-Path -LiteralPath $config.python_path)) {
                return $config.python_path
            }
        }
        catch {
            Write-Warning "Unable to read DPTB Pilot config at $configPath. Falling back to Python discovery."
        }
    }

    $candidates = @(
        (Join-Path $repoRoot ".venv\Scripts\python.exe"),
        (Join-Path $pluginRoot ".venv\Scripts\python.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    return "python"
}

$python = Get-ConfiguredPython
& $python $server
exit $LASTEXITCODE
