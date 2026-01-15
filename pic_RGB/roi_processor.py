from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
)
from roi_gui import ROISelector
from roi_manual import ManualROITools

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
    bbox: Tuple[int, int, int, int]
    area: int
    bg_ring_inner_h: float  # 水平内边距
    bg_ring_outer_h: float  # 水平外边距
    bg_ring_inner_v: float  # 垂直内边距
    bg_ring_outer_v: float  # 垂直外边距
    bg_center_x: float
    bg_center_y: float
    R_mean: float
    G1_mean: float
    G2_mean: float
    B_mean: float
    R_bg: float
    G1_bg: float
    G2_bg: float
    B_bg: float
    
    # 向后兼容属性
    @property
    def bg_ring_inner(self) -> float:
        return self.bg_ring_inner_h
    
    @property
    def bg_ring_outer(self) -> float:
        return self.bg_ring_outer_h
    R_corrected: float
    G1_corrected: float
    G2_corrected: float
    B_corrected: float
    R_display: float
    G_display: float
    B_display: float
    exposure_time: Optional[float] = None
    iso: Optional[int] = None

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
    """GUI-driven ROI processing pipeline."""

    def __init__(
        self,
        settings: Optional[ROISettings] = None,
        roi_params: Optional[Dict[str, float]] = None,
    ):
        base_settings = settings or get_settings()
        if roi_params:
            base_settings = base_settings.with_overrides(roi_params)
        self.settings = base_settings

        self.manual = ManualROITools(self.settings)
        self.gui = ROISelector(self.settings)

    def load_raw_image(self, filepath: Path) -> Tuple[np.ndarray, Dict]:
        """加载 RAW 图像为 4 通道矩阵，并返回元信息。"""
        with rawpy.imread(str(filepath)) as raw:
            metadata = {
                "exposure_time": None,
                "iso": None,
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
        
        return stats

    def create_intensity_map(self, raw_channels: np.ndarray) -> np.ndarray:
        """Construct an intensity map for visualization."""
        Y = np.max(raw_channels, axis=-1)
        Y = cv2.medianBlur(Y.astype(np.uint16), 3).astype(np.float32)
        return Y


    def process_image(
        self,
        filepath: Path,
        save_visualization: bool = True,
        output_dir: Optional[Path] = None,
    ) -> ROIResult:
        """Process a single image and return ROIResult."""
        print(f"\n处理图像: {filepath.name}")

        raw_channels, metadata = self.load_raw_image(filepath)
        H, W = raw_channels.shape[:2]
        print(f"  图像尺寸: {W}x{H}")

        Y = self.create_intensity_map(raw_channels)
        selection = self.gui.select_roi(raw_channels, filepath)

        center = selection.center
        roi_width = selection.width or self.settings.fixed_roi_width
        roi_height = selection.height or self.settings.fixed_roi_height
        roi_angle = selection.angle_deg if selection.angle_deg is not None else self.settings.roi_angle_deg

        print(
            f"  ROI: center=({center[0]:.1f}, {center[1]:.1f}), size={roi_width:.1f}x{roi_height:.1f}px, angle={roi_angle:.1f} deg"
        )

        roi_mask = self.manual.create_fixed_roi(center, (H, W), roi_width, roi_height, roi_angle)
        bg_center = selection.bg_center if selection.bg_center else center
        bg_mask = self.manual.create_background_roi(
            bg_center,
            (H, W),
            roi_width,
            roi_height,
            roi_angle,
            exclude_mask=None,
            ring_inner_h=selection.bg_ring_inner_h,
            ring_outer_h=selection.bg_ring_outer_h,
            ring_inner_v=selection.bg_ring_inner_v,
            ring_outer_v=selection.bg_ring_outer_v,
        )
        selection_mask = roi_mask.copy()

        roi_stats = self.compute_roi_statistics(raw_channels, roi_mask, metadata)
        bg_stats = self.compute_roi_statistics(raw_channels, bg_mask, metadata)

        R_corrected = roi_stats["R_mean"] - bg_stats["R_mean"]
        G1_corrected = roi_stats["G1_mean"] - bg_stats["G1_mean"]
        G2_corrected = roi_stats["G2_mean"] - bg_stats["G2_mean"]
        B_corrected = roi_stats["B_mean"] - bg_stats["B_mean"]
        G_display = (G1_corrected + G2_corrected) / 2

        result = ROIResult(
            filename=filepath.name,
            center_x=center[0],
            center_y=center[1],
            width=float(roi_width),
            height=float(roi_height),
            angle_deg=float(roi_angle),
            bbox=(
                max(0, int(round(center[0] - roi_width / 2))),
                max(0, int(round(center[1] - roi_height / 2))),
                min(W, int(round(center[0] + roi_width / 2))),
                min(H, int(round(center[1] + roi_height / 2))),
            ),
            area=int(np.sum(roi_mask)),
            bg_ring_inner_h=float(selection.bg_ring_inner_h),
            bg_ring_outer_h=float(selection.bg_ring_outer_h),
            bg_ring_inner_v=float(selection.bg_ring_inner_v),
            bg_ring_outer_v=float(selection.bg_ring_outer_v),
            bg_center_x=float(bg_center[0]),
            bg_center_y=float(bg_center[1]),
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
            R_display=max(0, R_corrected),
            G_display=max(0, G_display),
            B_display=max(0, B_corrected),
            exposure_time=metadata.get("exposure_time"),
            iso=metadata.get("iso"),
        )

        print(
            f"  Features: R={R_corrected:.1f}, G1={G1_corrected:.1f}, G2={G2_corrected:.1f}, B={B_corrected:.1f}"
        )

        if save_visualization and output_dir is not None:
            self._save_visualization(
                filepath, raw_channels, Y, roi_mask, bg_mask, selection_mask, result, output_dir
            )

        return result



    def _save_visualization(
        self,
        filepath: Path,
        raw_channels: np.ndarray,
        Y: np.ndarray,
        roi_mask: np.ndarray,
        bg_mask: np.ndarray,
        selection_mask: np.ndarray,
        result: ROIResult,
        output_dir: Path,
    ):
        """Save a quick-look visualization of the selected ROI."""
        fig, axes = plt.subplots(2, 3, figsize=VISUALIZATION_FIGSIZE)

        ax = axes[0, 0]
        im = ax.imshow(Y, cmap="gray")
        ax.set_title("Intensity map (max channel)")
        ax.axis("off")
        plt.colorbar(im, ax=ax)

        ax = axes[0, 1]
        ax.imshow(selection_mask, cmap="gray")
        ax.plot(result.center_x, result.center_y, "r+", markersize=20, markeredgewidth=2)
        ax.set_title("Selected ROI mask")
        ax.axis("off")

        ax = axes[0, 2]
        ax.imshow(Y, cmap="gray")
        ax.imshow(roi_mask, cmap="Reds", alpha=0.3)
        box = self.manual.get_rotated_box(
            (result.center_x, result.center_y), result.width, result.height, result.angle_deg
        )
        poly = plt.Polygon(box, closed=True, fill=False, edgecolor="red", linewidth=2)
        ax.add_patch(poly)
        ax.plot(result.center_x, result.center_y, "r+", markersize=15, markeredgewidth=2)
        ax.plot(result.bg_center_x, result.bg_center_y, "yx", markersize=12, markeredgewidth=2)
        ax.set_title(f"ROI ({result.width:.0f}x{result.height:.0f}px, {result.angle_deg:.1f} deg)\nBG center: ({result.bg_center_x:.1f}, {result.bg_center_y:.1f})")
        ax.axis("off")

        ax = axes[1, 0]
        ax.imshow(Y, cmap="gray")
        ax.imshow(bg_mask, cmap="Blues", alpha=0.3)
        ax.set_title("Background ring ROI")
        ax.axis("off")

        ax = axes[1, 1]
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
        ax.imshow(roi_mask, cmap="Reds", alpha=0.3)
        ax.set_title("RGB preview + ROI")
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
        ax.set_title("Background-corrected channels")
        ax.set_ylabel("Intensity (ADU)")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, corrected_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.0f}", ha="center", va="bottom", fontsize=9)

        plt.suptitle(f"ROI extraction: {filepath.name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{filepath.stem}_roi.png"
        plt.savefig(output_path, dpi=VISUALIZATION_DPI, bbox_inches="tight")
        plt.close()
        print(f"  Visualization saved: {output_path}")


    def batch_process(
        self,
        image_dir: Path,
        output_dir: Optional[Path] = None,
        save_visualization: bool = True,
    ) -> List[ROIResult]:
        """Batch process DNG files (prompts GUI selection per image)."""
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
                result = self.process_image(filepath, save_visualization, output_dir)
                results.append(result)
            except Exception as exc:
                print(f"  错误: {exc}")
                failed_files.append(filepath.name)

        print(f"\n处理完成: 成功 {len(results)}/{len(image_files)}")
        if failed_files:
            print(f"失败的文件: {failed_files}")

        if results:
            self._save_results(results, output_dir)
            self._save_detailed_statistics(results, output_dir)

        return results

    def _save_results(self, results: List[ROIResult], output_dir: Path):
        """保存 JSON/CSV/NPY 结果。"""
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "roi_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        print(f"\nJSON 结果已保存: {json_path}")

        csv_path = output_dir / "roi_features.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("filename,center_x,center_y,width,height,angle_deg,area,")
            f.write("bg_ring_inner_h,bg_ring_outer_h,bg_ring_inner_v,bg_ring_outer_v,bg_center_x,bg_center_y,")
            f.write("R_mean,G1_mean,G2_mean,B_mean,")
            f.write("R_bg,G1_bg,G2_bg,B_bg,")
            f.write("R_corrected,G1_corrected,G2_corrected,B_corrected,")
            f.write("R_display,G_display,B_display,")
            f.write("exposure_time,iso\n")
            for r in results:
                f.write(
                    f"{r.filename},{r.center_x:.2f},{r.center_y:.2f},{r.width:.2f},{r.height:.2f},{r.angle_deg:.2f},{r.area},"
                )
                f.write(f"{r.bg_ring_inner_h:.2f},{r.bg_ring_outer_h:.2f},{r.bg_ring_inner_v:.2f},{r.bg_ring_outer_v:.2f},")
                f.write(f"{r.bg_center_x:.2f},{r.bg_center_y:.2f},")
                f.write(
                    f"{r.R_mean:.2f},{r.G1_mean:.2f},{r.G2_mean:.2f},{r.B_mean:.2f},"
                )
                f.write(
                    f"{r.R_bg:.2f},{r.G1_bg:.2f},{r.G2_bg:.2f},{r.B_bg:.2f},"
                )
                f.write(
                    f"{r.R_corrected:.2f},{r.G1_corrected:.2f},{r.G2_corrected:.2f},{r.B_corrected:.2f},"
                )
                f.write(f"{r.R_display:.2f},{r.G_display:.2f},{r.B_display:.2f},")
                f.write(f"{r.exposure_time},{r.iso}\n")
        print(f"CSV 特征已保存: {csv_path}")

        features_path = output_dir / "feature_matrix.npy"
        feature_matrix = np.array([r.get_feature_vector() for r in results])
        np.save(features_path, feature_matrix)
        print(f"特征矩阵已保存: {features_path} (shape: {feature_matrix.shape})")

    def _save_detailed_statistics(self, results: List[ROIResult], output_dir: Path):
        """保存详细的统计信息到单独的CSV文件，包括均值、方差、分位数、饱和比例和通道比值。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 重新处理图像以获取详细统计信息
        detailed_stats_list = []
        
        for result in results:
            # 构建完整路径
            filepath = DATA_DIR / result.filename
            if not filepath.exists():
                print(f"  警告: 找不到文件 {filepath}，跳过详细统计")
                continue
            
            try:
                # 重新加载图像并计算详细统计
                raw_channels, metadata = self.load_raw_image(filepath)

                # 重建 ROI mask
                center = (result.center_x, result.center_y)
                H, W = raw_channels.shape[:2]
                roi_mask = self.manual.create_fixed_roi(
                    center, (H, W), result.width, result.height, result.angle_deg
                )
                bg_mask = self.manual.create_background_roi(
                    (result.bg_center_x, result.bg_center_y),
                    (H, W),
                    result.width,
                    result.height,
                    result.angle_deg,
                    exclude_mask=None,
                    ring_inner_h=result.bg_ring_inner_h,
                    ring_outer_h=result.bg_ring_outer_h,
                    ring_inner_v=result.bg_ring_inner_v,
                    ring_outer_v=result.bg_ring_outer_v,
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


def main():
    print_config()
    processor = ROIProcessor()
    print("\nStarting batch processing (GUI selection per image)...")
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
