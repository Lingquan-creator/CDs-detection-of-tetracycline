from __future__ import annotations

import json
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rawpy

from roi_config import (
    DATA_DIR,
    OUTPUT_DIR,
    SAVE_VISUALIZATION,
    VISUALIZATION_DPI,
    VISUALIZATION_FIGSIZE,
    ROISettings,
    get_settings,
    print_config,
    REUSE_LAST_SELECTION,
    SELECTION_CACHE_PATH,
)
from roi_gui import BackgroundSelection, ROISelection, select_roi_with_gui
from roi_manual import ManualROITools
RAW_CHANNELS = ("R", "G1", "G2", "B")

# 字体设置：尽量兼容中英文
matplotlib.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


@dataclass
class ROIResult:
    """ROI 提取结果。"""

    filename: str
    concentration: Optional[float]
    center_x: float
    center_y: float
    width: float
    height: float
    angle_deg: float
    measurement_center_x: float
    measurement_center_y: float
    measurement_width: float
    measurement_height: float
    measurement_angle_deg: float
    bg_center_x: float
    bg_center_y: float
    bg_inner_center_x: float
    bg_inner_center_y: float
    bg_outer_center_x: float
    bg_outer_center_y: float
    bg_inner_x: float
    bg_inner_y: float
    bg_outer_x: float
    bg_outer_y: float
    bg_angle_deg: float
    bbox: Tuple[int, int, int, int]
    area: int
    R_mean: float
    G1_mean: float
    G2_mean: float
    B_mean: float
    R_bg: float
    G1_bg: float
    G2_bg: float
    B_bg: float
    R_corrected: float
    G1_corrected: float
    G2_corrected: float
    B_corrected: float
    R_display: float
    G_display: float
    B_display: float
    exposure_time: Optional[float] = None
    iso: Optional[int] = None
    source_path: Optional[str] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, (np.integer, np.floating)):
                data[key] = value.item()
            elif isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif isinstance(value, (tuple, list)):
                # 处理包含numpy类型的元组/列表
                data[key] = [
                    v.item() if isinstance(v, (np.integer, np.floating)) else v
                    for v in value
                ]
        return data

    def get_feature_vector(self) -> np.ndarray:
        """基础特征向量：背景校正后的 R, G1, G2, B 均值（4 维）。"""
        return np.array(
            [
                self.R_corrected,
                self.G1_corrected,
                self.G2_corrected,
                self.B_corrected,
            ]
        )


class ROIProcessor:
    """封装基于 GUI 的 ROI 处理流程。"""

    def __init__(
        self,
        settings: Optional[ROISettings] = None,
        roi_params: Optional[Dict[str, float]] = None,
        reuse_last_selection: Optional[bool] = None,
        selection_cache_path: Optional[Path] = None,
    ):
        base_settings = settings or get_settings()
        if roi_params:
            base_settings = base_settings.with_overrides(roi_params)
        self.settings = base_settings

        self.reuse_last_selection = (
            REUSE_LAST_SELECTION if reuse_last_selection is None else reuse_last_selection
        )
        self.selection_cache_path = selection_cache_path or SELECTION_CACHE_PATH
        self.selection_cache: Dict[str, Tuple[ROISelection, BackgroundSelection]] = (
            self._load_selection_cache()
        )
        self.directory_metadata_cache: Dict[Path, Dict[str, Any]] = {}

        self.manual = ManualROITools(self.settings)

    @staticmethod
    def _parse_scalar(value: str) -> Any:
        text = value.strip()
        if text == "":
            return text
        lower = text.lower()
        if lower in {"true", "false"}:
            return lower == "true"
        try:
            if any(ch in text for ch in (".", "e", "E")):
                return float(text)
            return int(text)
        except ValueError:
            return text

    def _load_directory_metadata(self, directory: Path) -> Dict[str, Any]:
        resolved = directory.resolve()
        if resolved in self.directory_metadata_cache:
            return self.directory_metadata_cache[resolved]

        metadata_file = directory / "metadata.txt"
        parsed: Dict[str, Any] = {}
        if metadata_file.exists():
            try:
                for line in metadata_file.read_text(encoding="utf-8").splitlines():
                    text = line.strip()
                    if not text or text.startswith("#") or "=" not in text:
                        continue
                    key, value = text.split("=", maxsplit=1)
                    key = key.strip()
                    if not key:
                        continue
                    parsed[key] = self._parse_scalar(value)
            except OSError as exc:
                print(f"  警告: 无法读取目录元数据 {metadata_file}: {exc}")

        self.directory_metadata_cache[resolved] = parsed
        if parsed:
            print(f"  已加载目录元数据: {metadata_file.name} ({', '.join(sorted(parsed.keys()))})")
        return parsed

    @staticmethod
    def _parse_filename_metadata(filepath: Path) -> Dict[str, Any]:
        stem = filepath.stem
        candidates: List[Tuple[str, str]] = []
        if "_" in stem:
            candidates.append(stem.rsplit("_", maxsplit=1))
        if "." in stem:
            candidates.append(stem.rsplit(".", maxsplit=1))

        for concentration_text, exposure_text in candidates:
            if not concentration_text or not exposure_text:
                continue

            metadata: Dict[str, Any] = {}
            try:
                metadata["concentration"] = float(concentration_text)
            except ValueError:
                continue
            try:
                exposure_value = float(exposure_text)
            except ValueError:
                continue
            if "_" not in stem and exposure_value < 100:
                continue
            metadata["exposure_time"] = exposure_value
            return metadata

        return {}

    @staticmethod
    def _lookup_metadata_value(metadata: Dict[str, Any], *keys: str) -> Any:
        lowered = {str(k).lower(): v for k, v in metadata.items()}
        for key in keys:
            if key in metadata:
                return metadata[key]
            lowered_key = key.lower()
            if lowered_key in lowered:
                return lowered[lowered_key]
        return None

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _convert_roi_csv_to_json(csv_path: Path, json_path: Path) -> bool:
        """Convert roi_detailed_statistics.csv into roi_results.json for selection reuse."""
        if not csv_path.exists():
            return False
        try:
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                rows = [dict(row) for row in csv.DictReader(f) if row.get("filename")]
        except Exception as exc:
            print(f"  警告: 无法读取 ROI CSV 复用文件 {csv_path}: {exc}")
            return False

        if not rows:
            print(f"  警告: ROI CSV 复用文件为空: {csv_path}")
            return False

        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
            print(f"  未找到 {json_path.name}，已由 {csv_path.name} 转换生成")
            return True
        except Exception as exc:
            print(f"  警告: 无法写入由 CSV 转换的 ROI JSON {json_path}: {exc}")
            return False

    def _load_selection_cache(self) -> Dict[str, Tuple[ROISelection, BackgroundSelection]]:
        """Load previous selections from roi_results.json and optional cache file."""
        cache: Dict[str, Tuple[ROISelection, BackgroundSelection]] = {}

        def load_from_path(path: Path, label: str):
            if not path or not path.exists():
                return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                print(f"  警告: 无法读取历史选择 ({label}): {exc}")
                return

            # roi_results.json is a list of ROIResult dicts
            if isinstance(data, list):
                for entry in data:
                    fname = entry.get("filename")
                    if not fname:
                        continue
                    try:
                        roi_sel = ROISelection(
                            center=(float(entry["center_x"]), float(entry["center_y"])),
                            width=float(entry["width"]),
                            height=float(entry["height"]),
                            angle_deg=float(entry.get("angle_deg", 0.0)),
                        )
                        bg_sel = BackgroundSelection(
                            center=(
                                float(entry.get("bg_outer_center_x", entry.get("bg_center_x", roi_sel.center[0]))),
                                float(entry.get("bg_outer_center_y", entry.get("bg_center_y", roi_sel.center[1]))),
                            ),
                            inner_center=(
                                float(entry.get("bg_inner_center_x", entry.get("bg_center_x", roi_sel.center[0]))),
                                float(entry.get("bg_inner_center_y", entry.get("bg_center_y", roi_sel.center[1]))),
                            ),
                            outer_center=(
                                float(entry.get("bg_outer_center_x", entry.get("bg_center_x", roi_sel.center[0]))),
                                float(entry.get("bg_outer_center_y", entry.get("bg_center_y", roi_sel.center[1]))),
                            ),
                            inner_x=(
                                float(entry.get("bg_inner_x", 0))
                                if "bg_inner_center_x" in entry
                                else roi_sel.width + 2.0 * float(entry.get("bg_inner_x", 0))
                            ),
                            inner_y=(
                                float(entry.get("bg_inner_y", 0))
                                if "bg_inner_center_y" in entry
                                else roi_sel.height + 2.0 * float(entry.get("bg_inner_y", 0))
                            ),
                            outer_x=(
                                float(entry.get("bg_outer_x", 0))
                                if "bg_outer_center_x" in entry
                                else roi_sel.width + 2.0 * float(entry.get("bg_outer_x", 0))
                            ),
                            outer_y=(
                                float(entry.get("bg_outer_y", 0))
                                if "bg_outer_center_y" in entry
                                else roi_sel.height + 2.0 * float(entry.get("bg_outer_y", 0))
                            ),
                            angle_deg=float(entry.get("bg_angle_deg", roi_sel.angle_deg)),
                        )
                        cache[fname] = (roi_sel, bg_sel)
                    except Exception:
                        continue

            # selection cache file stored as {filename: {"roi": {...}, "bg": {...}}}
            if isinstance(data, dict):
                for fname, pair in data.items():
                    try:
                        roi_sel = ROISelection.from_dict(pair.get("roi", {}))
                        bg_sel = BackgroundSelection.from_dict(pair.get("bg", {}), roi=roi_sel)
                        cache[fname] = (roi_sel, bg_sel)
                    except Exception:
                        continue

        # 1) load from previous roi_results.json as baseline. If the JSON is missing,
        # fall back to roi_detailed_statistics.csv and convert it once.
        roi_results_json = OUTPUT_DIR / "roi_results.json"
        if not roi_results_json.exists():
            self._convert_roi_csv_to_json(OUTPUT_DIR / "roi_detailed_statistics.csv", roi_results_json)
        load_from_path(roi_results_json, "roi_results.json")
        # 2) overlay with dedicated selection cache (latest GUI choices in单张处理)
        if self.selection_cache_path:
            load_from_path(self.selection_cache_path, str(self.selection_cache_path))

        return cache

    def _persist_selection(self, filename: str, roi_sel: ROISelection, bg_sel: BackgroundSelection):
        if not self.selection_cache_path:
            return
        self.selection_cache[filename] = (roi_sel, bg_sel)
        try:
            self.selection_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.selection_cache_path, "w", encoding="utf-8") as f:
                serializable = {
                    fname: {"roi": r.to_dict(), "bg": b.to_dict()}
                    for fname, (r, b) in self.selection_cache.items()
                }
                json.dump(serializable, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"  警告: 无法写入 ROI/BG 选择缓存: {exc}")

    def _get_cached_selection(
        self, filename: str
    ) -> Optional[Tuple[ROISelection, BackgroundSelection]]:
        if filename not in self.selection_cache:
            return None
        return self.selection_cache.get(filename)

    def load_raw_image(self, filepath: Path) -> Tuple[np.ndarray, Dict]:
        """加载 RAW 图像为 4 通道矩阵，并返回元信息。"""
        with rawpy.imread(str(filepath)) as raw:
            metadata = {
                "exposure_time": None,
                "iso": None,
                "concentration": None,
                "black_level": None,
                "white_level": None,
            }

            # 优先支持新命名规则：浓度_曝光时长.dng；兼容旧规则：浓度.曝光时长.dng
            filename_meta = self._parse_filename_metadata(filepath)
            if "concentration" in filename_meta:
                metadata["concentration"] = filename_meta["concentration"]
            if "exposure_time" in filename_meta:
                metadata["exposure_time"] = filename_meta["exposure_time"]

            # 同目录 metadata.txt（如存在）作为附加元数据输入
            directory_meta = self._load_directory_metadata(filepath.parent)
            if directory_meta:
                metadata["directory_metadata"] = dict(directory_meta)
                if metadata["exposure_time"] is None:
                    exp_from_txt = self._lookup_metadata_value(directory_meta, "exposure_time", "exposure", "exp")
                    if isinstance(exp_from_txt, (int, float)):
                        metadata["exposure_time"] = float(exp_from_txt)
                iso_from_txt = self._lookup_metadata_value(directory_meta, "iso", "iso_speed")
                if isinstance(iso_from_txt, (int, float)):
                    metadata["iso"] = int(iso_from_txt)

            try:
                raw_meta = getattr(raw, "metadata", None)
                if raw_meta is not None:
                    if metadata["exposure_time"] is None and hasattr(raw_meta, "shutter"):
                        shutter = getattr(raw_meta, "shutter")
                        if isinstance(shutter, (int, float)) and shutter > 0:
                            metadata["exposure_time"] = float(shutter)
                    if metadata["iso"] is None and hasattr(raw_meta, "iso_speed"):
                        iso_speed = getattr(raw_meta, "iso_speed")
                        if isinstance(iso_speed, (int, float)) and iso_speed > 0:
                            metadata["iso"] = int(iso_speed)
            except Exception:
                pass

            try:
                if hasattr(raw, "white_level"):
                    metadata["white_level"] = raw.white_level
                elif hasattr(raw, "camera_white_level_per_channel"):
                    metadata["white_level"] = raw.camera_white_level_per_channel
            except Exception:
                pass

            raw_data = raw.raw_image.copy().astype(np.float32)

            black_levels = [0, 0, 0, 0]
            try:
                if (
                    hasattr(raw, "black_level_per_channel")
                    and raw.black_level_per_channel is not None
                ):
                    bl = raw.black_level_per_channel
                    if isinstance(bl, (list, tuple, np.ndarray)) and len(bl) >= 4:
                        black_levels = list(bl)[:4]
                    elif isinstance(bl, (int, float)):
                        black_levels = [bl] * 4
                    metadata["black_level"] = black_levels
                elif (
                    hasattr(raw, "camera_black_level_per_channel")
                    and raw.camera_black_level_per_channel is not None
                ):
                    bl = raw.camera_black_level_per_channel
                    if isinstance(bl, (list, tuple, np.ndarray)) and len(bl) >= 4:
                        black_levels = list(bl)[:4]
                    elif isinstance(bl, (int, float)):
                        black_levels = [bl] * 4
                    metadata["black_level"] = black_levels
            except Exception as exc:
                print(f"  警告: 无法获取黑电平，使用默认值0: {exc}")

            R = raw_data[0::2, 0::2] - black_levels[0]
            G1 = raw_data[0::2, 1::2] - black_levels[1]
            G2 = raw_data[1::2, 0::2] - black_levels[2]
            B = raw_data[1::2, 1::2] - black_levels[3]

            R = np.maximum(R, 0)
            G1 = np.maximum(G1, 0)
            G2 = np.maximum(G2, 0)
            B = np.maximum(B, 0)

            raw_channels = np.stack([R, G1, G2, B], axis=-1)
            return raw_channels, metadata

    def _trimmed_mean(self, values: np.ndarray) -> float:
        trim_fraction = float(np.clip(self.settings.trim_fraction, 0.0, 0.49))
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            return 0.0
        if trim_fraction <= 0.0:
            return float(np.mean(finite_values))

        low, high = np.quantile(finite_values, [trim_fraction, 1.0 - trim_fraction])
        trimmed = finite_values[(finite_values >= low) & (finite_values <= high)]
        if trimmed.size == 0:
            trimmed = finite_values
        return float(np.mean(trimmed))

    def compute_roi_statistics(self, raw_channels: np.ndarray, roi_mask: np.ndarray, metadata: Optional[Dict] = None) -> Dict[str, float]:
        """计算当前输入向量所需的四通道截尾均值。"""
        stats: Dict[str, float] = {}
        for i, name in enumerate(RAW_CHANNELS):
            channel_data = raw_channels[:, :, i]
            roi_values = channel_data[roi_mask > 0]
            if len(roi_values) > 0:
                stats[f"{name}_mean"] = self._trimmed_mean(roi_values)
            else:
                stats[f"{name}_mean"] = 0.0

        return stats

    def process_image(
        self,
        filepath: Path,
        save_visualization: bool = True,
        output_dir: Optional[Path] = None,
        roi_selection: Optional[ROISelection] = None,
        bg_selection: Optional[BackgroundSelection] = None,
        interactive: bool = True,
    ) -> ROIResult:
        """处理单张图像，输出 ROIResult。"""
        if output_dir is None:
            output_dir = OUTPUT_DIR

        print(f"\n处理图像: {filepath.name}")

        raw_channels, metadata = self.load_raw_image(filepath)
        H, W = raw_channels.shape[:2]
        print(f"  图像尺寸: {W}x{H}")

        roi_sel, bg_sel = self._resolve_selections(
            raw_channels=raw_channels,
            roi_selection=roi_selection,
            bg_selection=bg_selection,
            interactive=interactive,
            filename=filepath.name,
        )

        print(
            f"  ROI: center=({roi_sel.center[0]:.1f}, {roi_sel.center[1]:.1f}), size={roi_sel.width:.1f}x{roi_sel.height:.1f}, angle={roi_sel.angle_deg:.1f}°"
        )
        print(
            f"  BG : inner center=({bg_sel.inner_center[0]:.1f}, {bg_sel.inner_center[1]:.1f}), size={bg_sel.inner_x:.1f}x{bg_sel.inner_y:.1f}; "
            f"outer center=({bg_sel.outer_center[0]:.1f}, {bg_sel.outer_center[1]:.1f}), size={bg_sel.outer_x:.1f}x{bg_sel.outer_y:.1f}"
        )

        result = self._process_loaded_image(
            filepath=filepath,
            raw_channels=raw_channels,
            metadata=metadata,
            roi_sel=roi_sel,
            bg_sel=bg_sel,
            save_visualization=save_visualization,
            output_dir=output_dir,
        )
        self._persist_selection(filepath.name, roi_sel, bg_sel)
        return result

    def _resolve_selections(
        self,
        raw_channels: np.ndarray,
        roi_selection: Optional[ROISelection],
        bg_selection: Optional[BackgroundSelection],
        interactive: bool,
        filename: str,
    ) -> Tuple[ROISelection, BackgroundSelection]:
        H, W = raw_channels.shape[:2]
        roi_sel = roi_selection or ROISelection(
            center=(W / 2, H / 2),
            width=float(self.settings.fixed_roi_width),
            height=float(self.settings.fixed_roi_height),
            angle_deg=float(self.settings.roi_angle_deg),
        )
        bg_sel = bg_selection or BackgroundSelection(
            center=roi_sel.center,
            inner_center=roi_sel.center,
            outer_center=roi_sel.center,
            inner_x=float(roi_sel.width + 2 * self.settings.bg_ring_inner_x),
            inner_y=float(roi_sel.height + 2 * self.settings.bg_ring_inner_y),
            outer_x=float(roi_sel.width + 2 * self.settings.bg_ring_outer_x),
            outer_y=float(roi_sel.height + 2 * self.settings.bg_ring_outer_y),
            angle_deg=float(roi_sel.angle_deg),
        )

        cached_sel: Optional[Tuple[ROISelection, BackgroundSelection]] = None
        if roi_selection is None and bg_selection is None:
            cached_sel = self._get_cached_selection(filename)

        should_launch_gui = interactive
        if cached_sel:
            roi_sel, bg_sel = cached_sel
            if self.reuse_last_selection:
                should_launch_gui = False
                print("  复用上次保存的 ROI/BG 选择")
            else:
                print("  已载入上次的 ROI/BG 作为初始值（本次仍可重新选择）")

        if bg_sel.outer_x <= bg_sel.inner_x:
            bg_sel = BackgroundSelection(
                center=bg_sel.outer_center,
                inner_center=bg_sel.inner_center,
                outer_center=bg_sel.outer_center,
                inner_x=bg_sel.inner_x,
                inner_y=bg_sel.inner_y,
                outer_x=bg_sel.inner_x + 1,
                outer_y=bg_sel.outer_y,
                angle_deg=bg_sel.angle_deg,
            )
        if bg_sel.outer_y <= bg_sel.inner_y:
            bg_sel = BackgroundSelection(
                center=bg_sel.outer_center,
                inner_center=bg_sel.inner_center,
                outer_center=bg_sel.outer_center,
                inner_x=bg_sel.inner_x,
                inner_y=bg_sel.inner_y,
                outer_x=bg_sel.outer_x,
                outer_y=bg_sel.inner_y + 1,
                angle_deg=bg_sel.angle_deg,
            )

        if should_launch_gui:
            roi_sel, bg_sel = select_roi_with_gui(
                raw_channels=raw_channels,
                settings=self.settings,
                initial_roi=roi_sel,
                initial_bg=bg_sel,
            )
        return roi_sel, bg_sel

    def _measurement_roi_geometry(self, roi_sel: ROISelection) -> Tuple[Tuple[float, float], float, float, float]:
        """Return the central measurement ROI inside the user-selected luminous region."""
        width_fraction = float(np.clip(self.settings.measurement_roi_width_fraction, 0.0, 1.0))
        height_fraction = float(np.clip(self.settings.measurement_roi_height_fraction, 0.0, 1.0))
        measurement_width = max(1.0, float(roi_sel.width) * width_fraction)
        measurement_height = max(1.0, float(roi_sel.height) * height_fraction)
        return roi_sel.center, measurement_width, measurement_height, float(roi_sel.angle_deg)

    def _process_loaded_image(
        self,
        filepath: Path,
        raw_channels: np.ndarray,
        metadata: Dict,
        roi_sel: ROISelection,
        bg_sel: BackgroundSelection,
        save_visualization: bool,
        output_dir: Optional[Path],
    ) -> ROIResult:
        H, W = raw_channels.shape[:2]
        selected_roi_mask = self.manual.create_fixed_roi(
            roi_sel.center, (H, W), roi_sel.width, roi_sel.height, roi_sel.angle_deg
        )
        measurement_center, measurement_width, measurement_height, measurement_angle = (
            self._measurement_roi_geometry(roi_sel)
        )
        roi_mask = self.manual.create_fixed_roi(
            measurement_center,
            (H, W),
            measurement_width,
            measurement_height,
            measurement_angle,
        )
        bg_mask = self.manual.create_background_roi(
            bg_sel.outer_center,
            (H, W),
            roi_angle=bg_sel.angle_deg,
            inner_center=bg_sel.inner_center,
            outer_center=bg_sel.outer_center,
            inner_width=bg_sel.inner_x,
            inner_height=bg_sel.inner_y,
            outer_width=bg_sel.outer_x,
            outer_height=bg_sel.outer_y,
            exclude_mask=selected_roi_mask,
        )

        roi_stats = self.compute_roi_statistics(raw_channels, roi_mask, metadata)
        bg_stats = self.compute_roi_statistics(raw_channels, bg_mask, metadata)

        R_corrected = roi_stats["R_mean"] - bg_stats["R_mean"]
        G1_corrected = roi_stats["G1_mean"] - bg_stats["G1_mean"]
        G2_corrected = roi_stats["G2_mean"] - bg_stats["G2_mean"]
        B_corrected = roi_stats["B_mean"] - bg_stats["B_mean"]
        G_corrected = (G1_corrected + G2_corrected) / 2

        directory_meta = metadata.get("directory_metadata", {})
        if not isinstance(directory_meta, dict):
            directory_meta = {}
        r_gain = max(0.0, self._to_float(self._lookup_metadata_value(directory_meta, "R_gain"), 1.0))
        g_gain = max(0.0, self._to_float(self._lookup_metadata_value(directory_meta, "G_gain"), 1.0))
        b_gain = max(0.0, self._to_float(self._lookup_metadata_value(directory_meta, "B_gain"), 1.0))
        gamma = self._to_float(self._lookup_metadata_value(directory_meta, "gama", "gamma"), 1.0)
        saturability = self._to_float(self._lookup_metadata_value(directory_meta, "saturability"), 0.0)

        R_display = max(0.0, R_corrected) * r_gain
        G_display = max(0.0, G_corrected) * g_gain
        B_display = max(0.0, B_corrected) * b_gain

        if gamma > 0 and gamma != 1.0:
            inv_gamma = 1.0 / gamma
            R_display = R_display ** inv_gamma
            G_display = G_display ** inv_gamma
            B_display = B_display ** inv_gamma

        if saturability > 0:
            R_display = min(R_display, saturability)
            G_display = min(G_display, saturability)
            B_display = min(B_display, saturability)

        area = int(np.sum(roi_mask))
        measurement_box = self.manual.get_rotated_box(
            measurement_center, measurement_width, measurement_height, measurement_angle
        )
        bbox = (
            int(np.clip(np.min(measurement_box[:, 0]), 0, W - 1)),
            int(np.clip(np.min(measurement_box[:, 1]), 0, H - 1)),
            int(np.clip(np.max(measurement_box[:, 0]), 0, W - 1)),
            int(np.clip(np.max(measurement_box[:, 1]), 0, H - 1)),
        )

        result = ROIResult(
            filename=filepath.name,
            concentration=metadata.get("concentration"),
            center_x=roi_sel.center[0],
            center_y=roi_sel.center[1],
            width=float(roi_sel.width),
            height=float(roi_sel.height),
            angle_deg=float(roi_sel.angle_deg),
            measurement_center_x=measurement_center[0],
            measurement_center_y=measurement_center[1],
            measurement_width=float(measurement_width),
            measurement_height=float(measurement_height),
            measurement_angle_deg=float(measurement_angle),
            bg_center_x=bg_sel.outer_center[0],
            bg_center_y=bg_sel.outer_center[1],
            bg_inner_center_x=bg_sel.inner_center[0],
            bg_inner_center_y=bg_sel.inner_center[1],
            bg_outer_center_x=bg_sel.outer_center[0],
            bg_outer_center_y=bg_sel.outer_center[1],
            bg_inner_x=float(bg_sel.inner_x),
            bg_inner_y=float(bg_sel.inner_y),
            bg_outer_x=float(bg_sel.outer_x),
            bg_outer_y=float(bg_sel.outer_y),
            bg_angle_deg=float(bg_sel.angle_deg),
            bbox=bbox,
            area=area,
            R_mean=roi_stats["R_mean"],
            G1_mean=roi_stats["G1_mean"],
            G2_mean=roi_stats["G2_mean"],
            B_mean=roi_stats["B_mean"],
            R_bg=bg_stats["R_mean"],
            G1_bg=bg_stats["G1_mean"],
            G2_bg=bg_stats["G2_mean"],
            B_bg=bg_stats["B_mean"],
            R_corrected=max(0, R_corrected),
            G1_corrected=max(0, G1_corrected),
            G2_corrected=max(0, G2_corrected),
            B_corrected=max(0, B_corrected),
            R_display=max(0, R_display),
            G_display=max(0, G_display),
            B_display=max(0, B_display),
            exposure_time=metadata.get("exposure_time"),
            iso=metadata.get("iso"),
            source_path=str(filepath),
        )

        print(
            f"  特征向量: R={R_corrected:.1f}, G1={G1_corrected:.1f}, G2={G2_corrected:.1f}, B={B_corrected:.1f}"
        )
        print(
            "  统计ROI: "
            f"center=({measurement_center[0]:.1f}, {measurement_center[1]:.1f}), "
            f"size={measurement_width:.1f}x{measurement_height:.1f}, "
            f"angle={measurement_angle:.1f}°"
        )

        if save_visualization and output_dir is not None:
            self._save_visualization(
                filepath, raw_channels, roi_mask, bg_mask, result, output_dir
            )

        return result

    def _save_visualization(
        self,
        filepath: Path,
        raw_channels: np.ndarray,
        roi_mask: np.ndarray,
        bg_mask: np.ndarray,
        result: ROIResult,
        output_dir: Path,
    ):
        """保存可视化结果。"""
        Y = np.max(raw_channels, axis=-1)
        fig, axes = plt.subplots(2, 3, figsize=VISUALIZATION_FIGSIZE)

        ax = axes[0, 0]
        im = ax.imshow(Y, cmap="gray")
        ax.set_title("强度图 Y = max(R,G1,G2,B)")
        ax.axis("off")
        plt.colorbar(im, ax=ax)

        ax = axes[0, 1]
        ax.imshow(roi_mask, cmap="Reds")
        ax.set_title("统计 ROI mask")
        ax.axis("off")

        ax = axes[0, 2]
        ax.imshow(bg_mask, cmap="Blues")
        ax.set_title("背景环 mask")
        ax.axis("off")

        ax = axes[1, 0]
        rgb_image = np.stack(
            [
                raw_channels[:, :, 0],
                (raw_channels[:, :, 1] + raw_channels[:, :, 2]) / 2,
                raw_channels[:, :, 3],
            ],
            axis=-1,
        )
        rgb_image = np.clip(rgb_image / np.percentile(rgb_image, 99.5), 0, 1)
        ax.imshow(rgb_image)
        selected_roi_box = self.manual.get_rotated_box(
            (result.center_x, result.center_y), result.width, result.height, result.angle_deg
        )
        measurement_roi_box = self.manual.get_rotated_box(
            (result.measurement_center_x, result.measurement_center_y),
            result.measurement_width,
            result.measurement_height,
            result.measurement_angle_deg,
        )
        bg_outer_box = self.manual.get_rotated_box(
            (result.bg_outer_center_x, result.bg_outer_center_y),
            result.bg_outer_x,
            result.bg_outer_y,
            result.bg_angle_deg,
        )
        bg_inner_box = self.manual.get_rotated_box(
            (result.bg_inner_center_x, result.bg_inner_center_y),
            result.bg_inner_x,
            result.bg_inner_y,
            result.bg_angle_deg,
        )
        ax.add_patch(plt.Polygon(bg_outer_box, closed=True, fill=False, edgecolor="orange", linewidth=2))
        ax.add_patch(plt.Polygon(bg_inner_box, closed=True, fill=False, edgecolor="orange", linewidth=1.5, linestyle="--"))
        ax.add_patch(
            plt.Polygon(
                selected_roi_box,
                closed=True,
                fill=False,
                edgecolor="red",
                linewidth=1.5,
                linestyle="--",
            )
        )
        ax.add_patch(plt.Polygon(measurement_roi_box, closed=True, fill=False, edgecolor="red", linewidth=2))
        ax.plot(result.center_x, result.center_y, "r+", markersize=12, markeredgewidth=2)
        ax.plot(result.bg_inner_center_x, result.bg_inner_center_y, "yx", markersize=10, markeredgewidth=2)
        ax.plot(result.bg_outer_center_x, result.bg_outer_center_y, "yx", markersize=10, markeredgewidth=2)
        ax.set_title("RGB 伪彩 + 选区/统计ROI/BG")
        ax.axis("off")

        ax = axes[1, 1]
        ax.imshow(Y, cmap="gray")
        ax.imshow(roi_mask, cmap="Reds", alpha=0.35)
        ax.imshow(bg_mask, cmap="Blues", alpha=0.25)
        ax.set_title("ROI + 背景叠加")
        ax.axis("off")

        ax = axes[1, 2]
        channels = ["R", "G1", "G2", "B"]
        corrected_values = [
            result.R_corrected,
            result.G1_corrected,
            result.G2_corrected,
            result.B_corrected,
        ]
        colors = ["red", "green", "green", "blue"]
        bars = ax.bar(channels, corrected_values, color=colors, alpha=0.7)
        ax.set_title("背景校正后的通道值")
        ax.set_ylabel("强度 (ADU)")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, corrected_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.0f}", ha="center", va="bottom", fontsize=9)

        plt.suptitle(f"ROI 提取结果: {filepath.name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{filepath.stem}_roi.png"
        plt.savefig(output_path, dpi=VISUALIZATION_DPI, bbox_inches="tight")
        plt.close()
        print(f"  可视化已保存: {output_path}")

    def batch_process(
        self,
        image_dir: Path,
        output_dir: Optional[Path] = None,
        save_visualization: bool = True,
        roi_selection: Optional[ROISelection] = None,
        bg_selection: Optional[BackgroundSelection] = None,
        interactive: bool = True,
    ) -> List[ROIResult]:
        """批量处理文件夹中的 DNG 文件。"""
        image_dir = image_dir.expanduser().resolve()
        if not image_dir.exists():
            raise FileNotFoundError(f"输入目录不存在: {image_dir}")
        if not image_dir.is_dir():
            raise NotADirectoryError(f"输入路径不是目录: {image_dir}")

        if output_dir is None:
            output_dir = image_dir.parent / "roi_results"
        else:
            output_dir = output_dir.expanduser().resolve()

        image_files = sorted(
            path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() == ".dng"
        )
        if len(image_files) == 0:
            raise ValueError(f"在 {image_dir} 中未找到 DNG 文件（支持 .dng/.DNG）")

        print(f"找到 {len(image_files)} 个 DNG 文件")

        results: List[ROIResult] = []
        failed_files: List[str] = []
        for filepath in image_files:
            try:
                result = self.process_image(
                    filepath,
                    save_visualization,
                    output_dir,
                    roi_selection=roi_selection,
                    bg_selection=bg_selection,
                    interactive=interactive,
                )
                results.append(result)
            except Exception as exc:
                print(f"  错误: {exc}")
                failed_files.append(filepath.name)

        print(f"\n处理完成: 成功 {len(results)}/{len(image_files)}")
        if failed_files:
            print(f"失败的文件: {failed_files}")

        if results:
            self._save_results(results, output_dir)

        return results

    def _save_results(self, results: List[ROIResult], output_dir: Path):
        """只保存 ROI/BG 选择记录和当前输入向量需要的参数。"""
        output_dir.mkdir(parents=True, exist_ok=True)

        records = [self._build_output_record(r) for r in results]

        json_path = output_dir / "roi_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"\nROI/BG 记录已保存: {json_path}")

        csv_path = output_dir / "roi_detailed_statistics.csv"
        fieldnames = [
            "filename",
            "concentration",
            "exposure_time",
            "iso",
            "center_x",
            "center_y",
            "width",
            "height",
            "angle_deg",
            "measurement_center_x",
            "measurement_center_y",
            "measurement_width",
            "measurement_height",
            "measurement_angle_deg",
            "bg_inner_center_x",
            "bg_inner_center_y",
            "bg_outer_center_x",
            "bg_outer_center_y",
            "bg_inner_x",
            "bg_inner_y",
            "bg_outer_x",
            "bg_outer_y",
            "bg_angle_deg",
            "roi_R_mean",
            "roi_G1_mean",
            "roi_G2_mean",
            "roi_B_mean",
            "bg_R_mean",
            "bg_G1_mean",
            "bg_G2_mean",
            "bg_B_mean",
            "R_sig",
            "G_sig",
            "B_sig",
            "ln_R_sig",
            "ln_G_sig",
            "ln_B_sig",
            "Ratio_RG",
            "Delta_RG",
            "Delta_GB",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print(f"输入向量参数已保存: {csv_path}")
        print("  输出列只包含 ROI/BG 选择记录、ROI/BG 均值和当前 6 维输入向量。")

    @staticmethod
    def _format_optional(value: Optional[float | int]) -> str:
        return "" if value is None else str(value)

    def _build_output_record(self, r: ROIResult) -> Dict[str, str]:
        g_sig = (r.G1_corrected + r.G2_corrected) / 2.0
        r_sig = max(r.R_corrected, 0.1)
        g_sig_clipped = max(g_sig, 0.1)
        b_sig = max(r.B_corrected, 0.1)
        ratio_rg = r_sig / (g_sig_clipped + 1e-6)
        delta_rg = (g_sig_clipped - r_sig) / (g_sig_clipped + r_sig + 1e-6)
        delta_gb = (b_sig - g_sig_clipped) / (b_sig + g_sig_clipped + 1e-6)
        return {
            "filename": r.filename,
            "concentration": self._format_optional(r.concentration),
            "exposure_time": self._format_optional(r.exposure_time),
            "iso": self._format_optional(r.iso),
            "center_x": f"{r.center_x:.6f}",
            "center_y": f"{r.center_y:.6f}",
            "width": f"{r.width:.6f}",
            "height": f"{r.height:.6f}",
            "angle_deg": f"{r.angle_deg:.6f}",
            "measurement_center_x": f"{r.measurement_center_x:.6f}",
            "measurement_center_y": f"{r.measurement_center_y:.6f}",
            "measurement_width": f"{r.measurement_width:.6f}",
            "measurement_height": f"{r.measurement_height:.6f}",
            "measurement_angle_deg": f"{r.measurement_angle_deg:.6f}",
            "bg_inner_center_x": f"{r.bg_inner_center_x:.6f}",
            "bg_inner_center_y": f"{r.bg_inner_center_y:.6f}",
            "bg_outer_center_x": f"{r.bg_outer_center_x:.6f}",
            "bg_outer_center_y": f"{r.bg_outer_center_y:.6f}",
            "bg_inner_x": f"{r.bg_inner_x:.6f}",
            "bg_inner_y": f"{r.bg_inner_y:.6f}",
            "bg_outer_x": f"{r.bg_outer_x:.6f}",
            "bg_outer_y": f"{r.bg_outer_y:.6f}",
            "bg_angle_deg": f"{r.bg_angle_deg:.6f}",
            "roi_R_mean": f"{r.R_mean:.6f}",
            "roi_G1_mean": f"{r.G1_mean:.6f}",
            "roi_G2_mean": f"{r.G2_mean:.6f}",
            "roi_B_mean": f"{r.B_mean:.6f}",
            "bg_R_mean": f"{r.R_bg:.6f}",
            "bg_G1_mean": f"{r.G1_bg:.6f}",
            "bg_G2_mean": f"{r.G2_bg:.6f}",
            "bg_B_mean": f"{r.B_bg:.6f}",
            "R_sig": f"{r_sig:.6f}",
            "G_sig": f"{g_sig_clipped:.6f}",
            "B_sig": f"{b_sig:.6f}",
            "ln_R_sig": f"{np.log(r_sig + 1e-6):.6f}",
            "ln_G_sig": f"{np.log(g_sig_clipped + 1e-6):.6f}",
            "ln_B_sig": f"{np.log(b_sig + 1e-6):.6f}",
            "Ratio_RG": f"{ratio_rg:.6f}",
            "Delta_RG": f"{delta_rg:.6f}",
            "Delta_GB": f"{delta_gb:.6f}",
        }


def main():
    print_config()
    processor = ROIProcessor()
    print("\n开始批量处理...")
    results = processor.batch_process(
        image_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        save_visualization=SAVE_VISUALIZATION,
    )

    if results:
        feature_matrix = np.array([r.get_feature_vector() for r in results])
        print("\n=== 统计摘要 ===")
        print(f"特征向量维度: {feature_matrix.shape}")
        print(f"R 范围: [{feature_matrix[:,0].min():.1f}, {feature_matrix[:,0].max():.1f}]")
        print(f"G1 范围: [{feature_matrix[:,1].min():.1f}, {feature_matrix[:,1].max():.1f}]")
        print(f"G2 范围: [{feature_matrix[:,2].min():.1f}, {feature_matrix[:,2].max():.1f}]")
        print(f"B 范围: [{feature_matrix[:,3].min():.1f}, {feature_matrix[:,3].max():.1f}]")


if __name__ == "__main__":
    main()
