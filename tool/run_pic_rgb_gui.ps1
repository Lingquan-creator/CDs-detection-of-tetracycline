$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

Set-Location $RepoRoot
python -m pip install -r "$ScriptDir\requirements_pic_rgb_gui.txt"
python "$ScriptDir\pic_rgb_gui.py"
