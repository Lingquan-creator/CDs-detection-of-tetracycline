# Quickstart (GUI ROI)

1. Install deps from `pic_RGB`  
   `pip install -r requirements.txt`
2. Put your DNG files in `pic_RGB/data_pic/` (update `roi_config.py` if paths differ).
3. Run batch processing (OpenCV window per image; draw rectangle, then adjust angle/size/position in the follow-up window; Enter/Space to confirm):  
   `python roi_processor.py`
4. Try the guided script (`python test_roi.py`) for single/batch runs with the same GUI workflow.
5. Results land in `pic_RGB/roi_results/`:
   - Visuals: `*_roi.png`
   - Data: `roi_results.json`, `roi_features.csv`, `feature_matrix.npy`, `roi_detailed_statistics.csv`

Config knobs (`roi_config.py`):
- `DATA_DIR` / `OUTPUT_DIR`
- `bg_ring_inner`, `bg_ring_outer`
- `preview_max_dim`, `preview_percentile` (preview sizing/contrast for the GUI)
- `SAVE_VISUALIZATION`, `VISUALIZATION_DPI`, `VISUALIZATION_FIGSIZE`
