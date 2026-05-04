$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

pyinstaller `
  --noconfirm `
  --windowed `
  --name ConcentrationPredictor `
  --add-data "models;models" `
  --add-data "model_registry.json;." `
  app.py

Write-Host "Built: $PSScriptRoot\dist\ConcentrationPredictor"
