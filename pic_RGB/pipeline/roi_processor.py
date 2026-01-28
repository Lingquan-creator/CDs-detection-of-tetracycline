from __future__ import annotations

# Allow running this file directly (python pic_RGB/pipeline/roi_processor.py)
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rawpy

from pic_RGB.config import settings as config
from pic_RGB.gui.roi_gui import BackgroundSelection, ROISelection, select_roi_with_gui
from pic_RGB.processing.roi_manual import ManualROITools

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
    center_x: float
    center_y: float
    width: float
    height: float
    angle_deg: float
    bg_center_x: float
    bg_center_y: float
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
    R_var: float = 0.0
    G1_var: float = 0.0
    G2_var: float = 0.0
    B_var: float = 0.0
    RGB_mse: float = 0.0
    exposure_time: Optional[float] = None
    iso: Optional[int] = None
    aperture: Optional[float] = None

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
        return np.array(
            [
                self.R_corrected,
                self.G1_corrected,
                self.G2_corrected,
                self.B_corrected,
            ]
        )

    def get_rgb_vector(self) -> np.ndarray:
        return np.array(
            [
                self.R_display,
                self.G_display,
                self.B_display,
            ]
        )


class ROIProcessor:
    """封装基于 GUI 的 ROI 处理流程。"""

    def __init__(
        self,
        settings: Optional[config.ROISettings] = None,
        roi_params: Optional[Dict[str, float]] = None,
        reuse_last_selection: Optional[bool] = None,
        selection_cache_path: Optional[Path] = None,
    ):
        base_settings = settings or config.get_settings()
        if roi_params:
            base_settings = base_settings.with_overrides(roi_params)
        self.settings = base_settings

        self.reuse_last_selection = (
            config.REUSE_LAST_SELECTION if reuse_last_selection is None else reuse_last_selection
        )
        self.selection_cache_path = selection_cache_path or config.SELECTION_CACHE_PATH
        self.selection_cache: Dict[str, Tuple[ROISelection, BackgroundSelection]] = (
            self._load_selection_cache()
        )

        self.manual = ManualROITools(self.settings)

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
                                float(entry.get("bg_center_x", roi_sel.center[0])),
                                float(entry.get("bg_center_y", roi_sel.center[1])),
                            ),
                            inner_x=float(entry.get("bg_inner_x", 0)),
                            inner_y=float(entry.get("bg_inner_y", 0)),
                            outer_x=float(entry.get("bg_outer_x", 0)),
                            outer_y=float(entry.get("bg_outer_y", 0)),
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
                        bg_sel = BackgroundSelection.from_dict(pair.get("bg", {}))
                        cache[fname] = (roi_sel, bg_sel)
                    except Exception:
                        continue

        # 1) load from previous roi_results.json/results.json as baseline
        load_from_path(config.OUTPUT_DIR / "roi_results.json", "roi_results.json")
        load_from_path(config.OUTPUT_DIR / "results.json", "results.json")
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
                "aperture": None,
                "black_level": None,
                "white_level": None,
            }

            try:
                if hasattr(raw, "white_level"):
                    metadata["white_level"] = raw.white_level
                elif hasattr(raw, "camera_white_level_per_channel"):
                    metadata["white_level"] = raw.camera_white_level_per_channel
            except Exception:
                pass

            # Try to read exposure / ISO / aperture when available
            try:
                md = getattr(raw, "metadata", None)
                if md:
                    if hasattr(md, "shutter"):
                        metadata["exposure_time"] = float(md.shutter)
                    if hasattr(md, "iso_speed"):
                        metadata["iso"] = int(md.iso_speed)
                    if hasattr(md, "aperture"):
                        metadata["aperture"] = float(md.aperture)
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

    def compute_roi_statistics(self, raw_channels: np.ndarray, roi_mask: np.ndarray, metadata: Optional[Dict] = None) -> Dict[str, float]:
        """计算 ROI 内的通道统计值，包括均值、方差、分位数、饱和比例等。"""
        stats: Dict[str, float] = {}
        channel_names = ["R", "G1", "G2", "B"]
        
        # 获取白电平（用于计算饱和比例）
        white_level = 16383.0  # 14-bit RAW 默认值
        if metadata and metadata.get("white_level"):
            wl = metadata.get("white_level")
            if isinstance(wl, (list, tuple, np.ndarray)):
                white_level = float(np.max(wl))
            else:
                white_level = float(wl)
        
        for i, name in enumerate(channel_names):
            channel_data = raw_channels[:, :, i]
            roi_values = channel_data[roi_mask > 0]
            if len(roi_values) > 0:
                # 基础统计量
                stats[f"{name}_mean"] = float(np.mean(roi_values))
                stats[f"{name}_var"] = float(np.var(roi_values))  # 方差
                stats[f"{name}_std"] = float(np.std(roi_values))
                stats[f"{name}_median"] = float(np.median(roi_values))
                stats[f"{name}_p10"] = float(np.percentile(roi_values, 10))
                stats[f"{name}_p50"] = float(np.percentile(roi_values, 50))  # 中位数
                stats[f"{name}_p90"] = float(np.percentile(roi_values, 90))
                
                # 饱和比例（接近白电平的像素占比）
                saturation_threshold = white_level * 0.95  # 95%白电平视为饱和
                saturated_pixels = np.sum(roi_values >= saturation_threshold)
                stats[f"{name}_saturation_ratio"] = float(saturated_pixels / len(roi_values))
            else:
                stats[f"{name}_mean"] = 0.0
                stats[f"{name}_var"] = 0.0
                stats[f"{name}_std"] = 0.0
                stats[f"{name}_median"] = 0.0
                stats[f"{name}_p10"] = 0.0
                stats[f"{name}_p50"] = 0.0
                stats[f"{name}_p90"] = 0.0
                stats[f"{name}_saturation_ratio"] = 0.0
        
        # 计算通道比值（抗整体亮度漂移）
        epsilon = 1e-6  # 防止除零
        G_mean = (stats["G1_mean"] + stats["G2_mean"]) / 2.0
        stats["R_G_ratio"] = stats["R_mean"] / (G_mean + epsilon)
        stats["B_G_ratio"] = stats["B_mean"] / (G_mean + epsilon)
        stats["R_B_ratio"] = stats["R_mean"] / (stats["B_mean"] + epsilon)
        stats["G1_G2_ratio"] = stats["G1_mean"] / (stats["G2_mean"] + epsilon)
        # Overall RGB MSE: average of per-channel variances (mean squared error to channel means)
        stats["RGB_mse"] = float(
            (stats["R_var"] + stats["G1_var"] + stats["G2_var"] + stats["B_var"]) / len(channel_names)
        )

        return stats

    def process_image(
        self,
        filepath: Path,
        save_visualization: bool = False,
        output_dir: Optional[Path] = None,
        roi_selection: Optional[ROISelection] = None,
        bg_selection: Optional[BackgroundSelection] = None,
        interactive: bool = True,
    ) -> ROIResult:
        """处理单张图像，输出 ROIResult。"""
        if output_dir is None:
            output_dir = config.OUTPUT_DIR

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
            f"  BG : center=({bg_sel.center[0]:.1f}, {bg_sel.center[1]:.1f}), inner=({bg_sel.inner_x:.1f},{bg_sel.inner_y:.1f})px, outer=({bg_sel.outer_x:.1f},{bg_sel.outer_y:.1f})px"
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
            inner_x=float(self.settings.bg_ring_inner_x),
            inner_y=float(self.settings.bg_ring_inner_y),
            outer_x=float(self.settings.bg_ring_outer_x),
            outer_y=float(self.settings.bg_ring_outer_y),
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
                center=bg_sel.center,
                inner_x=bg_sel.inner_x,
                inner_y=bg_sel.inner_y,
                outer_x=bg_sel.inner_x + 1,
                outer_y=bg_sel.outer_y,
                angle_deg=bg_sel.angle_deg,
            )
        if bg_sel.outer_y <= bg_sel.inner_y:
            bg_sel = BackgroundSelection(
                center=bg_sel.center,
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
        roi_mask = self.manual.create_fixed_roi(
            roi_sel.center, (H, W), roi_sel.width, roi_sel.height, roi_sel.angle_deg
        )
        bg_mask = self.manual.create_background_roi(
            bg_sel.center,
            (H, W),
            roi_sel.width,
            roi_sel.height,
            bg_sel.angle_deg,
            inner_offset_x=bg_sel.inner_x,
            inner_offset_y=bg_sel.inner_y,
            outer_offset_x=bg_sel.outer_x,
            outer_offset_y=bg_sel.outer_y,
            exclude_mask=roi_mask,
        )

        roi_stats = self.compute_roi_statistics(raw_channels, roi_mask, metadata)
        bg_stats = self.compute_roi_statistics(raw_channels, bg_mask, metadata)

        R_corrected = roi_stats["R_mean"] - bg_stats["R_mean"]
        G1_corrected = roi_stats["G1_mean"] - bg_stats["G1_mean"]
        G2_corrected = roi_stats["G2_mean"] - bg_stats["G2_mean"]
        B_corrected = roi_stats["B_mean"] - bg_stats["B_mean"]
        G_display = (G1_corrected + G2_corrected) / 2

        area = int(np.sum(roi_mask))
        roi_box = self.manual.get_rotated_box(
            roi_sel.center, roi_sel.width, roi_sel.height, roi_sel.angle_deg
        )
        bbox = (
            int(np.clip(np.min(roi_box[:, 0]), 0, W - 1)),
            int(np.clip(np.min(roi_box[:, 1]), 0, H - 1)),
            int(np.clip(np.max(roi_box[:, 0]), 0, W - 1)),
            int(np.clip(np.max(roi_box[:, 1]), 0, H - 1)),
        )

        result = ROIResult(
            filename=filepath.name,
            center_x=roi_sel.center[0],
            center_y=roi_sel.center[1],
            width=float(roi_sel.width),
            height=float(roi_sel.height),
            angle_deg=float(roi_sel.angle_deg),
            bg_center_x=bg_sel.center[0],
            bg_center_y=bg_sel.center[1],
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
            R_var=roi_stats["R_var"],
            G1_var=roi_stats["G1_var"],
            G2_var=roi_stats["G2_var"],
            B_var=roi_stats["B_var"],
            RGB_mse=roi_stats["RGB_mse"],
            R_bg=bg_stats["R_mean"],
            G1_bg=bg_stats["G1_mean"],
            G2_bg=bg_stats["G2_mean"],
            B_bg=bg_stats["B_mean"],
            R_corrected=max(0, R_corrected),
            G1_corrected=max(0, G1_corrected),
            G2_corrected=max(0, G2_corrected),
            B_corrected=max(0, B_corrected),
            R_display=max(0, R_corrected),
            G_display=max(0, G_display),
            B_display=max(0, B_corrected),
            exposure_time=metadata.get("exposure_time"),
            iso=metadata.get("iso"),
            aperture=metadata.get("aperture"),
        )

        print(
            f"  特征向量: R={R_corrected:.1f}, G1={G1_corrected:.1f}, G2={G2_corrected:.1f}, B={B_corrected:.1f}"
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
        fig, axes = plt.subplots(2, 3, figsize=config.VISUALIZATION_FIGSIZE)

        ax = axes[0, 0]
        im = ax.imshow(Y, cmap="gray")
        ax.set_title("强度图 Y = max(R,G1,G2,B)")
        ax.axis("off")
        plt.colorbar(im, ax=ax)

        ax = axes[0, 1]
        ax.imshow(roi_mask, cmap="Reds")
        ax.set_title("ROI mask")
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
        roi_box = self.manual.get_rotated_box(
            (result.center_x, result.center_y), result.width, result.height, result.angle_deg
        )
        bg_outer_box = self.manual.get_rotated_box(
            (result.bg_center_x, result.bg_center_y),
            result.width + 2 * result.bg_outer_x,
            result.height + 2 * result.bg_outer_y,
            result.bg_angle_deg,
        )
        bg_inner_box = self.manual.get_rotated_box(
            (result.bg_center_x, result.bg_center_y),
            result.width + 2 * result.bg_inner_x,
            result.height + 2 * result.bg_inner_y,
            result.bg_angle_deg,
        )
        ax.add_patch(plt.Polygon(bg_outer_box, closed=True, fill=False, edgecolor="orange", linewidth=2))
        ax.add_patch(plt.Polygon(bg_inner_box, closed=True, fill=False, edgecolor="orange", linewidth=1.5, linestyle="--"))
        ax.add_patch(plt.Polygon(roi_box, closed=True, fill=False, edgecolor="red", linewidth=2))
        ax.plot(result.center_x, result.center_y, "r+", markersize=12, markeredgewidth=2)
        ax.plot(result.bg_center_x, result.bg_center_y, "yx", markersize=10, markeredgewidth=2)
        ax.set_title("RGB 伪彩 + ROI/BG")
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
        plt.savefig(output_path, dpi=config.VISUALIZATION_DPI, bbox_inches="tight")
        plt.close()
        print(f"  可视化已保存: {output_path}")

    def batch_process(
        self,
        image_dir: Path,
        output_dir: Optional[Path] = None,
        save_visualization: bool = False,
        export_csv: bool = True,
        roi_selection: Optional[ROISelection] = None,
        bg_selection: Optional[BackgroundSelection] = None,
        interactive: bool = True,
    ) -> List[ROIResult]:
        """批量处理文件夹中的 DNG 文件。"""
        if output_dir is None:
            output_dir = image_dir.parent / "roi_results"

        image_files = sorted(image_dir.glob("*.dng"))
        if len(image_files) == 0:
            raise ValueError(f"在 {image_dir} 中未找到 DNG 文件")

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
            self._save_detailed_statistics(results, output_dir, export_csv=export_csv)

        return results

    def _save_results(self, results: List[ROIResult], output_dir: Path):
        """Save minimal JSON output only."""
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "roi_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        print(f"\nJSON saved: {json_path}")

    def _save_detailed_statistics(self, results: List[ROIResult], output_dir: Path, export_csv: bool = True):
        """保存详细的统计信息到单独的CSV文件，包括均值、方差、分位数、饱和比例和通道比值。"""
        if not export_csv:
            print("详细统计 CSV 未启用 (export_csv=False)")
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 重新处理图像以获取详细统计信息
        detailed_stats_list = []
        
        for result in results:
            # 构建完整路径
            filepath = config.DATA_DIR / result.filename
            if not filepath.exists():
                print(f"  警告: 找不到文件 {filepath}，跳过详细统计")
                continue
            
            try:
                # 重新加载图像并计算详细统计
                raw_channels, metadata = self.load_raw_image(filepath)
                H, W = raw_channels.shape[:2]
                roi_sel = ROISelection(
                    center=(result.center_x, result.center_y),
                    width=result.width,
                    height=result.height,
                    angle_deg=result.angle_deg,
                )
                bg_sel = BackgroundSelection(
                    center=(result.bg_center_x, result.bg_center_y),
                    inner_x=result.bg_inner_x,
                    inner_y=result.bg_inner_y,
                    outer_x=result.bg_outer_x,
                    outer_y=result.bg_outer_y,
                    angle_deg=result.bg_angle_deg,
                )
                roi_mask = self.manual.create_fixed_roi(
                    roi_sel.center, (H, W), roi_sel.width, roi_sel.height, roi_sel.angle_deg
                )
                bg_mask = self.manual.create_background_roi(
                    bg_sel.center,
                    (H, W),
                    roi_sel.width,
                    roi_sel.height,
                    bg_sel.angle_deg,
                    inner_offset_x=bg_sel.inner_x,
                    inner_offset_y=bg_sel.inner_y,
                    outer_offset_x=bg_sel.outer_x,
                    outer_offset_y=bg_sel.outer_y,
                    exclude_mask=roi_mask,
                )
                
                # 计算详细统计
                roi_stats = self.compute_roi_statistics(raw_channels, roi_mask, metadata)
                bg_stats = self.compute_roi_statistics(raw_channels, bg_mask, metadata)
                
                # 合并信息
                detailed_info = {
                    'filename': result.filename,
                    **{f'roi_{k}': v for k, v in roi_stats.items()},
                    **{f'bg_{k}': v for k, v in bg_stats.items()}
                }
                detailed_stats_list.append(detailed_info)
                
            except Exception as exc:
                print(f"  警告: 处理 {result.filename} 详细统计时出错: {exc}")
                continue
        
        if not detailed_stats_list:
            print("  警告: 没有成功生成详细统计信息")
            return
        
        # 保存到CSV
        csv_path = output_dir / "roi_detailed_statistics.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            # 写表头
            keys = detailed_stats_list[0].keys()
            f.write(','.join(keys) + '\n')
            
            # 写数据
            for stats in detailed_stats_list:
                values = []
                for key in keys:
                    val = stats[key]
                    if isinstance(val, float):
                        values.append(f"{val:.6f}")
                    else:
                        values.append(str(val))
                f.write(','.join(values) + '\n')
        
        print(f"详细统计已保存: {csv_path}")
        print(f"  包含列: 均值、方差、分位数(p10/p50/p90)、饱和比例、通道比值(R/G, B/G, R/B, G1/G2)")


# CLI entry point moved to pic_RGB.main.batch_process.batch_process
