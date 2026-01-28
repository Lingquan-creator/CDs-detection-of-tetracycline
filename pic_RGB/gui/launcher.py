from __future__ import annotations

"""Wrapper to launch the ROI GUI with optional defaults."""

from typing import Optional, Tuple

import numpy as np

from pic_RGB.config import settings as config
from pic_RGB.gui.roi_gui import BackgroundSelection, ROISelection, select_roi_with_gui


def launch_roi_gui(
    raw_channels: np.ndarray,
    settings: config.ROISettings,
    initial_roi: Optional[ROISelection] = None,
    initial_bg: Optional[BackgroundSelection] = None,
    interactive: bool = True,
) -> Tuple[ROISelection, BackgroundSelection]:
    """Launch GUI if interactive is True; otherwise return provided selections."""
    if not interactive:
        return initial_roi or ROISelection(
            center=(raw_channels.shape[1] / 2, raw_channels.shape[0] / 2),
            width=settings.fixed_roi_width,
            height=settings.fixed_roi_height,
            angle_deg=settings.roi_angle_deg,
        ), initial_bg or BackgroundSelection(
            center=(raw_channels.shape[1] / 2, raw_channels.shape[0] / 2),
            inner_x=settings.bg_ring_inner_x,
            inner_y=settings.bg_ring_inner_y,
            outer_x=settings.bg_ring_outer_x,
            outer_y=settings.bg_ring_outer_y,
            angle_deg=settings.roi_angle_deg,
        )

    return select_roi_with_gui(
        raw_channels=raw_channels,
        settings=settings,
        initial_roi=initial_roi,
        initial_bg=initial_bg,
    )
