from __future__ import annotations

"""Serialize ROI/BG selections to JSON for reuse."""

import json
from pathlib import Path
from typing import Dict

from pic_RGB.extraction.roi_gui import BackgroundSelection, ROISelection


def selection_to_dict(filename: str, roi: ROISelection, bg: BackgroundSelection) -> Dict:
    return {
        "filename": filename,
        "roi": roi.to_dict(),
        "bg": bg.to_dict(),
    }


def save_selection(path: Path, filename: str, roi: ROISelection, bg: BackgroundSelection, overwrite: bool = True):
    """Persist a single selection record keyed by filename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                existing = data
        except Exception:
            existing = {}

    if not overwrite and filename in existing:
        return

    existing[filename] = {
        "roi": roi.to_dict(),
        "bg": bg.to_dict(),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
