"""Configuration for cross-device RGB color space mapping.

Defaults now resolve relative to the repository, so the code can run from
any working directory without hard-coded drive letters."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

# Root of the multi_device_rgb package (…/multi_device_rgb)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PACKAGE_ROOT / "device_mapping" / "results"


@dataclass
class MappingConfig:
    # Data locations
    pic1_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / "pic_1")
    pic2_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / "pic_2")
    roi_filename: str = "roi_results.json"

    # Matching strategy
    matching_concentrations: Optional[Tuple[float, ...]] = None  # None => auto intersect
    use_four_channels: bool = True  # True: R,G1,G2,B; False: collapse to RGB with G = (G1+G2)/2

    # Model hyper-parameters (Polynomial Ridge Regression per methodology)
    polynomial_degree: int = 2
    alpha_grid: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    min_clip: float = 0.1  # avoid negative corrected values after bg subtraction

    # Output
    results_dir: Path = field(default_factory=lambda: DEFAULT_RESULTS_DIR)

    def ensure_results_dir(self) -> Path:
        """Create the results directory if needed and return it."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self.results_dir
