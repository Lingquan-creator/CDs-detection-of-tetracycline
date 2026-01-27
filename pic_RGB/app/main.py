from __future__ import annotations

"""
High-level entry points supporting single-image and batch processing.
"""

from pathlib import Path
from typing import List, Optional

# Allow running this file directly (python path/to/app/main.py)
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pic_RGB.config import app_config
from pic_RGB.config.roi_config import get_settings, SAVE_VISUALIZATION
from pic_RGB.features import build_feature_vectors, build_rgb_vectors
from pic_RGB.gui import launch_roi_gui
from pic_RGB.kmeans import KMeansAlgorithm
from pic_RGB.persistence import save_selection
from pic_RGB.pipeline import ROIWorkflow
from pic_RGB.roi_init import initialize_selections


def _process_one(filepath: Path, interactive: bool = True):
    settings = get_settings()
    workflow = ROIWorkflow(settings=settings, output_dir=app_config.OUTPUT_DIR)
    algo = KMeansAlgorithm()

    raw_channels, metadata = workflow.load_raw(filepath)
    roi_init, bg_init, _ = initialize_selections(
        raw_channels,
        settings,
        algorithm=algo,
        filename=filepath.name,
        output_dir=app_config.OUTPUT_DIR,
    )
    roi_sel, bg_sel = launch_roi_gui(
        raw_channels=raw_channels,
        settings=settings,
        initial_roi=roi_init,
        initial_bg=bg_init,
        interactive=interactive,
    )

    result = workflow.process_loaded(
        filepath=filepath,
        raw_channels=raw_channels,
        metadata=metadata,
        roi_sel=roi_sel,
        bg_sel=bg_sel,
        save_visualization=SAVE_VISUALIZATION,
    )
    save_selection(app_config.SELECTION_CACHE_PATH, filepath.name, roi_sel, bg_sel, overwrite=app_config.OVERWRITE_SELECTION)
    return result


def run_single(image_path: Optional[Path] = None, interactive: bool = True):
    if image_path is None:
        candidates = sorted(app_config.DATA_DIR.glob(app_config.IMAGE_EXTENSION))
        if not candidates:
            raise FileNotFoundError(f"No images found under {app_config.DATA_DIR}")
        image_path = candidates[0]
    result = _process_one(image_path, interactive=interactive)
    print(f"Processed single image: {image_path.name}")
    print(f"  ROI center=({result.center_x:.1f},{result.center_y:.1f}) size={result.width:.1f}x{result.height:.1f}")
    return result


def run_batch(image_dir: Optional[Path] = None, interactive: bool = True):
    base_dir = image_dir or app_config.DATA_DIR
    files: List[Path] = sorted(base_dir.glob(app_config.IMAGE_EXTENSION))
    if not files:
        raise FileNotFoundError(f"No images found under {base_dir}")

    results = []
    for fp in files:
        print(f"=== {fp.name} ===")
        results.append(_process_one(fp, interactive=interactive))

    workflow = ROIWorkflow(settings=get_settings(), output_dir=app_config.OUTPUT_DIR)
    workflow.save_results(results)

    feature_matrix = build_feature_vectors(results)
    rgb_matrix = build_rgb_vectors(results)
    print(f"Batch done: {len(results)} images")
    print(f"  Feature matrix shape: {feature_matrix.shape}")
    print(f"  RGB matrix shape: {rgb_matrix.shape}")
    return results


if __name__ == "__main__":
    run_single()
