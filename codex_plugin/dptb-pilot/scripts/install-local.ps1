[CmdletBinding()]
param(
    [string]$ServerUrl = "http://localhost:50003",
    [ValidatePattern("^.{32}$")]
    [string]$SessionId,
    [int]$TimeoutMs = 30000,
    [string]$PythonPath,
    [switch]$SkipConfig,
    [switch]$SkipMarketplace,
    [switch]$NoCopy,
    [switch]$Force,
    [switch]$OpenDeeplink
)

$ErrorActionPreference = "Stop"

function Set-JsonProperty {
    param(
        [Parameter(Mandatory = $true)] $Object,
        [Parameter(Mandatory = $true)] [string] $Name,
        [Parameter(Mandatory = $true)] $Value
    )

    if ($Object.PSObject.Properties.Name -contains $Name) {
        $Object.$Name = $Value
    }
    else {
        $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
    }
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)] [string] $Path,
        [Parameter(Mandatory = $true)] $Object
    )

    $json = $Object | ConvertTo-Json -Depth 10
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $json, $utf8NoBom)
}

$scriptDir = $PSScriptRoot
$pluginRoot = (Resolve-Path -LiteralPath (Join-Path $scriptDir "..")).Path
$manifestPath = Join-Path $pluginRoot ".codex-plugin\plugin.json"
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$pluginName = $manifest.name

$homeDir = [Environment]::GetFolderPath("UserProfile")
$pluginsRoot = Join-Path $homeDir "plugins"
$installPath = Join-Path $pluginsRoot $pluginName
$marketplaceDir = Join-Path $homeDir ".agents\plugins"
$marketplacePath = Join-Path $marketplaceDir "marketplace.json"
$configDir = Join-Path $homeDir ".dptb-pilot-codex"
$configPath = Join-Path $configDir "config.json"
$marketplaceName = "personal"

if (-not $NoCopy) {
    New-Item -ItemType Directory -Force -Path $pluginsRoot | Out-Null

    $sourceIsInstallPath = $false
    if (Test-Path -LiteralPath $installPath) {
        $sourceIsInstallPath = $pluginRoot.Equals((Resolve-Path -LiteralPath $installPath).Path, [System.StringComparison]::OrdinalIgnoreCase)
    }

    if (-not $sourceIsInstallPath) {
        if (Test-Path -LiteralPath $installPath) {
            if (-not $Force) {
                throw "Plugin already exists at $installPath. Re-run with -Force to replace it from the repo copy."
            }

            $resolvedInstallPath = (Resolve-Path -LiteralPath $installPath).Path
            $resolvedPluginsRoot = (Resolve-Path -LiteralPath $pluginsRoot).Path
            if (-not $resolvedInstallPath.StartsWith($resolvedPluginsRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing to replace a path outside the plugin install root: $resolvedInstallPath"
            }

            Remove-Item -LiteralPath $installPath -Recurse -Force
        }

        Copy-Item -LiteralPath $pluginRoot -Destination $installPath -Recurse -Force
        Get-ChildItem -LiteralPath $installPath -Recurse -Directory -Filter "__pycache__" |
            Remove-Item -Recurse -Force
    }
}

if (-not $SkipConfig) {
    New-Item -ItemType Directory -Force -Path $configDir | Out-Null

    $existing = $null
    if (Test-Path -LiteralPath $configPath) {
        try {
            $existing = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
        }
        catch {
            Write-Warning "Existing config could not be parsed and will be replaced with v2 settings."
        }
    }

    if (-not $SessionId) {
        if ($existing -and $existing.version -eq 2 -and $existing.connections -and $existing.connections.default) {
            $SessionId = $existing.connections.default.session_id
        }
        elseif ($existing -and $existing.session_id) {
            $SessionId = $existing.session_id
        }
        else {
            $SessionId = [guid]::NewGuid().ToString("N")
        }
    }

    if ($SessionId.Length -ne 32) {
        throw "DPTB Pilot session id must be exactly 32 characters."
    }

    $connections = [ordered]@{}
    if ($existing -and $existing.version -eq 2 -and $existing.connections) {
        foreach ($property in $existing.connections.PSObject.Properties) {
            $connections[$property.Name] = $property.Value
        }
    }
    $connections["default"] = [ordered]@{
        server_url = $ServerUrl.TrimEnd("/")
        session_id = $SessionId
        timeout_ms = $TimeoutMs
    }

    $config = [ordered]@{
        version = 2
        active_connection = if ($existing -and $existing.version -eq 2 -and $existing.active_connection) { $existing.active_connection } else { "default" }
        connections = $connections
        active_chat = if ($existing -and $existing.version -eq 2) { $existing.active_chat } else { $null }
        workflow_mode = if ($existing -and $existing.workflow_mode) { $existing.workflow_mode } else { "auto" }
        strict_scientific_routing = $true
        execution = if ($existing -and $existing.version -eq 2 -and $existing.execution) { $existing.execution } else { [ordered]@{ mode = "Local"; profile_id = $null; parallel_mode = "auto"; max_parallelism = 4 } }
        execution_profiles = if ($existing -and $existing.version -eq 2 -and $existing.execution_profiles) { $existing.execution_profiles } else { [ordered]@{} }
        download_directory = if ($existing -and $existing.download_directory) { $existing.download_directory } else { ".dptb-pilot-downloads" }
        matterviz_uri = if ($existing -and $existing.matterviz_uri) { $existing.matterviz_uri } else { "matterviz://open?path={path}" }
    }

    if ($PythonPath) {
        $config.python_path = (Resolve-Path -LiteralPath $PythonPath).Path
    }
    elseif ($existing -and $existing.python_path) {
        $config.python_path = $existing.python_path
    }

    Write-JsonFile -Path $configPath -Object $config
}

if (-not $SkipMarketplace) {
    New-Item -ItemType Directory -Force -Path $marketplaceDir | Out-Null

    if (Test-Path -LiteralPath $marketplacePath) {
        $marketplace = Get-Content -Raw -LiteralPath $marketplacePath | ConvertFrom-Json
    }
    else {
        $marketplace = [pscustomobject]@{
            name = "personal"
            interface = [pscustomobject]@{
                displayName = "Personal"
            }
            plugins = @()
        }
    }

    if (-not $marketplace.name) {
        Set-JsonProperty -Object $marketplace -Name "name" -Value "personal"
    }
    $marketplaceName = $marketplace.name
    if (-not ($marketplace.PSObject.Properties.Name -contains "interface") -or -not $marketplace.interface) {
        Set-JsonProperty -Object $marketplace -Name "interface" -Value ([pscustomobject]@{})
    }
    if (-not ($marketplace.interface.PSObject.Properties.Name -contains "displayName") -or -not $marketplace.interface.displayName) {
        Set-JsonProperty -Object $marketplace.interface -Name "displayName" -Value "Personal"
    }

    $entry = [pscustomobject]@{
        name = $pluginName
        source = [pscustomobject]@{
            source = "local"
            path = "./plugins/$pluginName"
        }
        policy = [pscustomobject]@{
            installation = "AVAILABLE"
            authentication = "ON_INSTALL"
        }
        category = "Engineering"
    }

    $plugins = @()
    if ($marketplace.plugins) {
        $plugins = @($marketplace.plugins | Where-Object { $_.name -ne $pluginName })
    }
    $plugins += $entry
    Set-JsonProperty -Object $marketplace -Name "plugins" -Value $plugins

    Write-JsonFile -Path $marketplacePath -Object $marketplace
}

# The Codex desktop app does not ship a `codex` CLI, so `codex plugin add ...`
# is not recognized there. Use the codex:// deeplink instead. The personal
# marketplace is discovered implicitly from ~/.agents/plugins/marketplace.json.
$deeplink = "codex://plugins/$pluginName`?marketplacePath=$([uri]::EscapeDataString($marketplacePath))"

Write-Host "Installed source: $installPath"
Write-Host "Config file:      $configPath"
Write-Host "Marketplace:      $marketplacePath"
Write-Host ""
Write-Host "Finish in the Codex desktop app by opening this deeplink:"
Write-Host "  $deeplink"

if ($OpenDeeplink) {
    Start-Process $deeplink
}
