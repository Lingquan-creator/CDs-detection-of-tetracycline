from __future__ import annotations

"""
Batch ROI processing entry point (merged from former pic_RGB.app.batch_process).
"""

from pathlib import Path
import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from pic_RGB.config import settings as config
from pic_RGB.pipeline import ROIProcessor


def batch_process(
    image_dir: Path | None = None,
    output_dir: Path | None = None,
    save_visualization: bool | None = None,
    export_csv: bool | None = None,
):
    """Run batch ROI extraction using defaults unless overridden."""
    config.print_config()
    processor = ROIProcessor()
    print("\n寮€濮嬫壒閲忓鐞?ROI ...")
    save_png = config.EXPORT_PNG if save_visualization is None else save_visualization
    save_csv = config.EXPORT_CSV if export_csv is None else export_csv
    results = processor.batch_process(
        image_dir=image_dir or config.DATA_DIR,
        output_dir=output_dir or config.OUTPUT_DIR,
        save_visualization=save_png,
        export_csv=save_csv,
    )

    if results:
        feature_matrix = np.array([r.get_feature_vector() for r in results])
        print("\n=== 缁熻姒傝 ===")
        print(f"鐗瑰緛鐭╅樀缁村害: {feature_matrix.shape}")
        print(f"R 鑼冨洿: [{feature_matrix[:,0].min():.1f}, {feature_matrix[:,0].max():.1f}]")
        print(f"G1 鑼冨洿: [{feature_matrix[:,1].min():.1f}, {feature_matrix[:,1].max():.1f}]")
        print(f"G2 鑼冨洿: [{feature_matrix[:,2].min():.1f}, {feature_matrix[:,2].max():.1f}]")
        print(f"B 鑼冨洿: [{feature_matrix[:,3].min():.1f}, {feature_matrix[:,3].max():.1f}]")
    return results


def run(image_dir: Path | None = None):
    return batch_process(image_dir=image_dir)


def _main():
    image_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    batch_process(image_dir=image_dir)


if __name__ == "__main__":
    _main()
