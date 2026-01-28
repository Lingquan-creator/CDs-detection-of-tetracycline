"""Processing pipeline that ties together selection, ROI computation, and saving."""

from .processor import ROIWorkflow
from .roi_processor import ROIProcessor, ROIResult

__all__ = ["ROIWorkflow", "ROIProcessor", "ROIResult"]
