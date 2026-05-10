$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

Set-Location $RepoRoot
python -m pip install -r "$ScriptDir\requirements_pic_rgb_gui.txt"

python -m PyInstaller `
  --noconfirm `
  --windowed `
  --name PicRGBRoiTool `
  --paths "$RepoRoot\pic_RGB" `
  --hidden-import rawpy._rawpy `
  --collect-submodules rawpy `
  --distpath "$ScriptDir\dist" `
  --workpath "$ScriptDir\build" `
  --specpath "$ScriptDir" `
  "$ScriptDir\pic_rgb_gui.py"

Write-Host "Built: $ScriptDir\dist\PicRGBRoiTool"
