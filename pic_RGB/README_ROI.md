# ROI GUI Selection & Feature Extraction

Manual rectangle selection for each DNG frame using an OpenCV GUI, with the existing ROI statistics and visualization pipeline retained.

## What's New
- Auto/advanced ROI modes removed; every image now requires a GUI rectangle selection.
- Background ring handling and feature exports (JSON/CSV/NPY/PNG) stay compatible with previous outputs.
- Configuration now focuses on paths, background ring thickness, and preview scaling for the GUI.

## Workflow
1. Load RAW DNG with `rawpy`, split RGGB channels, and apply black-level correction.
2. Build an intensity map for preview, then open an OpenCV window for rectangle drawing (Enter/Space confirms, Esc/C cancels).
3. Create ROI and concentric background masks; compute per-channel stats and background-corrected values.
4. Save visualization plus JSON/CSV/NPY feature outputs.

## Quick Start
1. Install dependencies (from `pic_RGB`):  
   `pip install -r requirements.txt`
2. Place DNG files in `pic_RGB/data_pic/` (adjust paths in `roi_config.py` if needed).
3. Run batch processing (GUI pops up for each image):  
   `python roi_processor.py`
4. Guided test options (single/batch) with the same GUI workflow:  
   `python test_roi.py`
5. Outputs are written to `pic_RGB/roi_results/` by default:
   - `*_roi.png` visualizations
   - `roi_results.json`
   - `roi_features.csv`
   - `feature_matrix.npy`
   - `roi_detailed_statistics.csv` (when detailed stats are generated)

## Configuration Highlights (`roi_config.py`)
- `DATA_DIR`, `OUTPUT_DIR`: input and output paths.
- `bg_ring_inner`, `bg_ring_outer`: background ring thickness around the selected ROI.
- `preview_max_dim`, `preview_percentile`: GUI preview scaling/contrast.
- `SAVE_VISUALIZATION`, `VISUALIZATION_DPI`, `VISUALIZATION_FIGSIZE`: output controls.

## GUI Tips
- Drag a rectangle in the first window; the second window lets you fine-tune **angle/size/position** (trackbars for angle/scale, WASD/arrow keys to nudge, +/- to resize base).
- Press Enter/Space to confirm, Esc/C to cancel.
- Batch mode will prompt once per image.
- The preview auto-scales to `preview_max_dim` to fit common screens.

## Outputs
- **Visualization**: PNG overlays of the selected ROI and background ring.
- **JSON/CSV/NPY**: per-image ROI stats and background-corrected channel values.
- **Detailed CSV**: optional full statistics for ROI/background (see `test_detailed_stats.py`).
