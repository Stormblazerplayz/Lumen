# Creates the isolated BeamNG test profile used while building the robotaxi mod.
# Storm's real profile (%LOCALAPPDATA%\BeamNG\BeamNG.drive) is never touched.
#   powershell -NoProfile -ExecutionPolicy Bypass -File setup-profile.ps1
param(
    [string]$ProfileRoot = "$env:LOCALAPPDATA\BeamNGTest\robotaxi"
)
$ErrorActionPreference = 'Stop'

# The game may treat -userpath as the folder that holds 'current\', or as the user
# folder itself. Write the same minimal settings into both places; it reads one.
$targets = @($ProfileRoot, (Join-Path $ProfileRoot 'current'))

# Window mode so the test instance never takes over Storm's screen; quality levels
# copied from Storm's own settings (Ultra mesh/lighting, Normal textures, High shadows).
# enableMcp turns on the game's built-in MCP server (127.0.0.1 only).
$local = [ordered]@{
    GraphicDisplayModes       = 'Window'
    GraphicDisplayResolutions = '1600 900'
    GraphicBorderless         = $false
    GraphicMeshQuality        = 'Ultra'
    GraphicTextureQuality     = 'Normal'
    GraphicLightingQuality    = 'Ultra'
    GraphicShadowsQuality     = 'High'
    enableMcp                 = $true
    trafficAmount             = 0
    trafficParkedAmount       = 0
}
# 'disable' for both consent settings: the most privacy-preserving answer, and it
# means the first-run consent wizard has nothing to ask (nothing is accepted).
$cloud = [ordered]@{
    onlineFeatures = 'disable'
    telemetry      = 'disable'
}

# UTF-8 without a BOM: PowerShell 5.1's Out-File adds one, which JSON readers can reject.
$utf8 = New-Object System.Text.UTF8Encoding($false)
foreach ($t in $targets) {
    $settingsDir = Join-Path $t 'settings'
    $cloudDir = Join-Path $settingsDir 'cloud'
    New-Item -ItemType Directory -Force -Path $cloudDir | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $t 'mods\unpacked') | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $settingsDir 'settings.json'), ($local | ConvertTo-Json), $utf8)
    [System.IO.File]::WriteAllText((Join-Path $cloudDir 'settings.json'), ($cloud | ConvertTo-Json), $utf8)
}
Write-Host "Test profile ready at $ProfileRoot"
