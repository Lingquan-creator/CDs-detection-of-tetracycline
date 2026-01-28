from __future__ import annotations

"""Reusable ROI workflow built atop ROIProcessor."""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Allow running this file directly (python path/to/pipeline/processor.py)
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from pic_RGB.config import settings as config
from pic_RGB.gui.roi_gui import BackgroundSelection, ROISelection
from pic_RGB.pipeline.roi_processor import ROIProcessor, ROIResult


class ROIWorkflow:
    """Orchestrates loading, processing, and saving ROI/BG results."""

    def __init__(self, settings: Optional[config.ROISettings] = None, output_dir: Optional[Path] = None):
        self.settings = settings or config.get_settings()
        self.output_dir = output_dir or config.OUTPUT_DIR
        self.processor = ROIProcessor(settings=self.settings)

    def load_raw(self, filepath: Path):
        return self.processor.load_raw_image(filepath)

    def process_loaded(
        self,
        filepath: Path,
        raw_channels: np.ndarray,
        metadata: dict,
        roi_sel: ROISelection,
        bg_sel: BackgroundSelection,
        save_visualization: bool = False,
    ) -> ROIResult:
        return self.processor._process_loaded_image(
            filepath=filepath,
            raw_channels=raw_channels,
            metadata=metadata,
            roi_sel=roi_sel,
            bg_sel=bg_sel,
            save_visualization=save_visualization,
            output_dir=self.output_dir,
        )

    def save_results(self, results: Iterable[ROIResult]):
        results_list: List[ROIResult] = list(results)
        if not results_list:
            return
        self.processor._save_results(results_list, self.output_dir)
        self.processor._save_detailed_statistics(
            results_list,
            self.output_dir,
            export_csv=config.EXPORT_CSV,
        )
