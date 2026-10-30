param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("user-prompt", "pre-tool")]
    [string]$Mode
)

$ErrorActionPreference = "Stop"
$pluginRoot = if ($env:PLUGIN_ROOT) { $env:PLUGIN_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
$policy = Join-Path $pluginRoot "hooks\strict_science_policy.py"

$python = $env:DPTB_PILOT_PYTHON
if (-not $python -or -not (Test-Path -LiteralPath $python)) {
    $configPath = if ($env:DPTB_PILOT_CONFIG) { $env:DPTB_PILOT_CONFIG } else { Join-Path ([Environment]::GetFolderPath("UserProfile")) ".dptb-pilot-codex\config.json" }
    if (Test-Path -LiteralPath $configPath) {
        try {
            $config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
            if ($config.python_path -and (Test-Path -LiteralPath $config.python_path)) {
                $python = $config.python_path
            }
        }
        catch {
            $python = $null
        }
    }
    if (-not $python) {
        $repoPython = Join-Path $pluginRoot "..\..\.venv\Scripts\python.exe"
        if (Test-Path -LiteralPath $repoPython) {
            $python = $repoPython
        }
        else {
            $python = "python"
        }
    }
}

$payload = [Console]::In.ReadToEnd()
$payload | & $python $policy $Mode
exit $LASTEXITCODE
