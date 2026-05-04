# Concentration Predictor

This folder contains a local desktop program for manual ROI/BG selection and concentration prediction.

## Run From Source

```bash
cd /home/jzh/CDs
./.venv/bin/python application/app.py
```

Workflow:

1. Open a `.dng` image.
2. Drag the active box on the image. Select `ROI`, `BG inner`, or `BG outer` in the side panel.
3. Choose the phone model and prediction model.
4. Click `Predict`.

Bundled phone model folders:

- `mate20x`
- `mate3_2`
- `xiaomi`
- `vivo100x`

Available model types:

- `Regression`: Ridge regression model.
- `GS posterior`: amortized posterior model, shown as posterior median.
- `EXP`: exponential inversion model using the best saved channel in `exp_fit_params.csv`.

## Build Later

Linux:

```bash
cd application
bash build_linux.sh
```

Windows PowerShell:

```powershell
cd application
.\build_windows.ps1
```
