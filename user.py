
import sys
import os
import glob
import math
import traceback
from pathlib import Path
import json

# Setup paths to include workspace modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_base_path():
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return Path(base_path)

# Assuming file is in root or we need to add relative paths to modules
WORKSPACE_ROOT = CURRENT_DIR
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, 'pic_RGB'))
sys.path.append(os.path.join(CURRENT_DIR, 'concentration'))

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                               QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
                               QGroupBox, QLineEdit, QComboBox)
from PySide6.QtCore import Qt, QThread, Signal

import numpy as np
import pandas as pd
import rawpy

import matplotlib
# Use 'Agg' backend for non-interactive plotting in background threads
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Translations
TRANSLATIONS = {
    "en": {
        "window_title": "Quantum Dot Concentration Analyzer",
        "config_group": "Configuration",
        "input_label": "Input Dir:",
        "output_label": "Output Dir:",
        "select_input": "Select Input Folder",
        "select_output": "Select Output Folder",
        "roi_group": "ROI & Background Selection",
        "set_roi_btn": "Set ROI / BG ROI on First Image",
        "roi_status_not_set": "ROI Status: Not Set",
        "roi_status_set": "ROI Set (Center: {:.1f}, {:.1f})",
        "roi_status_loaded": "ROI Loaded from JSON (Center: {:.1f}, {:.1f})",
        "analysis_group": "Analysis",
        "run_btn": "Calculate Concentrations (Batch)",
        "headers": ["Filename", "Predicted Conc.", "Status"],
        "status_found_images": "Found {} images.",
        "msg_no_images": "No images found in input directory.",
        "msg_missing_config": "Please select input and output directories.",
        "msg_missing_roi": "Please set ROI and Background first.",
        "msg_model_error": "Model file not found at {}",
        "msg_load_model_fail": "Failed to load model: {}",
        "msg_save_success": "Results saved to {}",
        "msg_analysis_complete": "Analysis Complete.",
        "msg_analysis_failed": "Analysis Failed",
        "msg_error": "Error",
        "msg_success": "Success",
        "msg_roi_cancelled": "ROI selection cancelled.",
        "lang_label": "Language:",
    },
    "zh": {
        "window_title": "量子点浓度分析仪",
        "config_group": "配置",
        "input_label": "输入目录:",
        "output_label": "输出目录:",
        "select_input": "选择输入文件夹",
        "select_output": "选择输出文件夹",
        "roi_group": "ROI 与 背景选择",
        "set_roi_btn": "在第一张图上设置 ROI / 背景",
        "roi_status_not_set": "ROI 状态: 未设置",
        "roi_status_set": "ROI 已设置 (中心: {:.1f}, {:.1f})",
        "roi_status_loaded": "ROI 已从 JSON 加载 (中心: {:.1f}, {:.1f})",
        "analysis_group": "分析",
        "run_btn": "计算浓度 (批量)",
        "headers": ["文件名", "预测浓度", "状态"],
        "status_found_images": "发现 {} 张图片。",
        "msg_no_images": "输入目录中未找到图片。",
        "msg_missing_config": "请选择输入和输出目录。",
        "msg_missing_roi": "请先设置 ROI 和背景。",
        "msg_model_error": "模型文件未找到: {}",
        "msg_load_model_fail": "加载模型失败: {}",
        "msg_save_success": "结果已保存至 {}",
        "msg_analysis_complete": "分析完成。",
        "msg_analysis_failed": "分析失败",
        "msg_error": "错误",
        "msg_success": "成功",
        "msg_roi_cancelled": "ROI 选择已取消。",
        "lang_label": "语言:",
    }
}


# Import modules from workspace
try:
    from roi_gui import select_roi_with_gui, ROISelection, BackgroundSelection
    from roi_processor import ROIProcessor, ROIResult  # Added ROIResult
    from roi_config import ROISettings
    from roi_manual import ManualROITools
    import rgb_to_concentration as rtc
    import physics_features as pf
    # from config import CLIP_MIN # Avoid direct import to prevent path issues
    CLIP_MIN = rtc.CLIP_MIN
except ImportError as e:
    print(f"Error importing modules: {e}")
    # Print sys.path to help debug
    print(f"sys.path: {sys.path}")
    traceback.print_exc()

class AnalysisWorker(QThread):
    progress = Signal(str)
    result = Signal(object)
    finished = Signal()
    error = Signal(str)

    def __init__(self, file_paths, roi_sel, bg_sel, roi_cache, model_bundle, output_dir):
        super().__init__()
        self.file_paths = file_paths
        self.roi_sel = roi_sel
        self.bg_sel = bg_sel
        self.roi_cache = roi_cache or {}
        self.model_data = model_bundle
        self.output_dir = output_dir

    def run(self):
        try:
            results = []
            processor = ROIProcessor() 
            # Note: ROIProcessor init might look for roi_config. We rely on default behavior or settings.
            
            # Ensure output directory exists (for images)
            out_dir_path = Path(self.output_dir)
            out_dir_path.mkdir(parents=True, exist_ok=True)
            
            # For JSON saving, we need a list of ROIResults or dict
            roi_results_list = []

            model, g_mode, target_transform, target_eps, use_bg, calibrator = self.model_data

            for idx, file_path in enumerate(self.file_paths):
                fname = os.path.basename(file_path)
                self.progress.emit(f"Processing {idx+1}/{len(self.file_paths)}: {fname}")
                
                # Determine ROI/BG for this file
                # Check if specific ROI exists in cache
                if fname in self.roi_cache:
                     current_roi_sel, current_bg_sel = self.roi_cache[fname]
                else:
                     # Fallback to global selection
                     current_roi_sel, current_bg_sel = self.roi_sel, self.bg_sel

                # Load Image
                fpath = Path(file_path)
                raw_channels, metadata = processor.load_raw_image(fpath)
                
                # Use ROIProcessor's _process_loaded_image logic manually to allow plotting
                # Or better: call _process_loaded_image if accessible, but it returns ROIResult, not full stats dict straight away.
                # Let's recreate the logic to ensure we control the plotting and stats exactly.
                
                H, W = raw_channels.shape[:2]
                manual_tools = processor.manual # Access the ManualROITools instance inside processor

                roi_mask = manual_tools.create_fixed_roi(
                    current_roi_sel.center, (H, W), current_roi_sel.width, current_roi_sel.height, current_roi_sel.angle_deg
                )
                bg_mask = manual_tools.create_background_roi(
                    current_bg_sel.center,
                    (H, W),
                    current_roi_sel.width,
                    current_roi_sel.height,
                    current_bg_sel.angle_deg,
                    inner_offset_x=current_bg_sel.inner_x,
                    inner_offset_y=current_bg_sel.inner_y,
                    outer_offset_x=current_bg_sel.outer_x,
                    outer_offset_y=current_bg_sel.outer_y,
                    exclude_mask=roi_mask,
                )

                roi_stats = processor.compute_roi_statistics(raw_channels, roi_mask, metadata)
                bg_stats = processor.compute_roi_statistics(raw_channels, bg_mask, metadata)
                
                # ---------------------------------------------------------
                # 1. Visualization Saving (Mimic ROIProcessor._save_visualization_plot)
                # ---------------------------------------------------------
                try:
                    # Construct valid bbox for ROIResult (approximation)
                    bbox = manual_tools.get_rotated_box(
                        current_roi_sel.center, current_roi_sel.width, current_roi_sel.height, current_roi_sel.angle_deg
                    )
                    x_min, y_min = np.min(bbox, axis=0)
                    x_max, y_max = np.max(bbox, axis=0)
                    
                    # Create a temporary ROIResult object just for the plotter if needed, 
                    # or better: call the internal plotter if we can access it.
                    # _save_visualization_plot is private. We will implement simplified plotting here.
                    
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=100)
                    
                    # Row 1: Masks
                    ax = axes[0, 0]
                    # Luminance
                    Y = (0.2126 * raw_channels[:, :, 0] + 0.7152 * ((raw_channels[:, :, 1] + raw_channels[:, :, 2]) / 2) + 0.0722 * raw_channels[:, :, 3])
                    Y = np.clip(Y / np.percentile(Y, 99.5), 0, 1)
                    ax.imshow(Y, cmap="gray")
                    ax.set_title("Luminance")
                    ax.axis("off")
                    
                    ax = axes[0, 1]
                    ax.imshow(roi_mask, cmap="Reds")
                    ax.set_title("ROI mask")
                    ax.axis("off")

                    ax = axes[0, 2]
                    ax.imshow(bg_mask, cmap="Blues")
                    ax.set_title("BG mask")
                    ax.axis("off")
                    
                    # Row 2: RGB + Boxes
                    ax = axes[1, 0]
                    rgb_vis = np.stack([
                        raw_channels[:, :, 0], 
                        (raw_channels[:, :, 1] + raw_channels[:, :, 2]) / 2, 
                        raw_channels[:, :, 3]
                    ], axis=-1)
                    rgb_vis = np.clip(rgb_vis / np.percentile(rgb_vis, 99.5), 0, 1)
                    ax.imshow(rgb_vis)
                    
                    roi_box_pts = manual_tools.get_rotated_box(
                        current_roi_sel.center, current_roi_sel.width, current_roi_sel.height, current_roi_sel.angle_deg
                    )
                    bg_outer_pts = manual_tools.get_rotated_box(
                        current_bg_sel.center, 
                        current_roi_sel.width + 2*current_bg_sel.outer_x, 
                        current_roi_sel.height + 2*current_bg_sel.outer_y, 
                        current_bg_sel.angle_deg
                    )
                    bg_inner_pts = manual_tools.get_rotated_box(
                         current_bg_sel.center, 
                        current_roi_sel.width + 2*current_bg_sel.inner_x, 
                        current_roi_sel.height + 2*current_bg_sel.inner_y, 
                        current_bg_sel.angle_deg
                    )
                    
                    ax.add_patch(plt.Polygon(bg_outer_pts, closed=True, fill=False, edgecolor="orange", linewidth=2))
                    ax.add_patch(plt.Polygon(bg_inner_pts, closed=True, fill=False, edgecolor="orange", linewidth=1.5, linestyle="--"))
                    ax.add_patch(plt.Polygon(roi_box_pts, closed=True, fill=False, edgecolor="red", linewidth=2))
                    ax.set_title("RGB + Boxes")
                    ax.axis("off")
                    
                    # 1,1 Overlay
                    ax = axes[1, 1]
                    ax.imshow(Y, cmap="gray")
                    ax.imshow(roi_mask, cmap="Reds", alpha=0.35)
                    ax.imshow(bg_mask, cmap="Blues", alpha=0.25)
                    ax.set_title("Overlay")
                    ax.axis("off")
                    
                    # 1,2 Stats Bar
                    ax = axes[1, 2]
                    # Corrected values
                    R_corr = roi_stats["R_mean"] - bg_stats["R_mean"]
                    G1_corr = roi_stats["G1_mean"] - bg_stats["G1_mean"]
                    G2_corr = roi_stats["G2_mean"] - bg_stats["G2_mean"]
                    B_corr = roi_stats["B_mean"] - bg_stats["B_mean"]
                    
                    ch_names = ["R", "G1", "G2", "B"]
                    ch_vals = [R_corr, G1_corr, G2_corr, B_corr]
                    colors = ["red", "green", "green", "blue"]
                    bars = ax.bar(ch_names, ch_vals, color=colors, alpha=0.7)
                    ax.set_title("Corrected Intensity")
                    for bar, val in zip(bars, ch_vals):
                        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.0f}", ha="center", va="bottom")
                    
                    plt.suptitle(f"Analysis: {fpath.name}", fontsize=14)
                    plt.tight_layout()
                    
                    plot_path = out_dir_path / f"{fpath.stem}_roi.png"
                    plt.savefig(plot_path)
                    plt.close(fig)
                    
                except Exception as plot_err:
                    print(f"Plotting error for {fpath.name}: {plot_err}")
                
                # ---------------------------------------------------------
                # End Visualization
                # ---------------------------------------------------------

                # Prepare DataFrame row for prediction
                # Ensure keys match what physics_features.build_feature_frame expects (e.g. roi_R_mean, bg_R_mean)
                row_data = {
                    "filename": fname,
                    "roi_R_mean": roi_stats.get("R_mean", 0),
                    "roi_G1_mean": roi_stats.get("G1_mean", 0),
                    "roi_G2_mean": roi_stats.get("G2_mean", 0),
                    "roi_B_mean": roi_stats.get("B_mean", 0),
                    
                    "roi_R_var": roi_stats.get("R_var", 0),
                    "roi_G1_var": roi_stats.get("G1_var", 0),
                    "roi_G2_var": roi_stats.get("G2_var", 0),
                    "roi_B_var": roi_stats.get("B_var", 0),

                    "bg_R_mean": bg_stats.get("R_mean", 0),
                    "bg_G1_mean": bg_stats.get("G1_mean", 0),
                    "bg_G2_mean": bg_stats.get("G2_mean", 0),
                    "bg_B_mean": bg_stats.get("B_mean", 0),
                }
                
                # Calculate corrected values for potential result display (optional, but good for debugging)
                row_data["R_corrected"] = row_data["roi_R_mean"] - row_data["bg_R_mean"]
                row_data["G1_corrected"] = row_data["roi_G1_mean"] - row_data["bg_G1_mean"]
                row_data["G2_corrected"] = row_data["roi_G2_mean"] - row_data["bg_G2_mean"]
                row_data["B_corrected"] = row_data["roi_B_mean"] - row_data["bg_B_mean"]

                # Create ROIResult structure for JSON
                # Using dummy bbox since we didn't firmly calculate it for everything, or reuse the one from plotting
                roi_res_obj = ROIResult(
                    filename=fname,
                    center_x=current_roi_sel.center[0], center_y=current_roi_sel.center[1],
                    width=current_roi_sel.width, height=current_roi_sel.height, angle_deg=current_roi_sel.angle_deg,
                    bg_center_x=current_bg_sel.center[0], bg_center_y=current_bg_sel.center[1],
                    bg_inner_x=current_bg_sel.inner_x, bg_inner_y=current_bg_sel.inner_y,
                    bg_outer_x=current_bg_sel.outer_x, bg_outer_y=current_bg_sel.outer_y, bg_angle_deg=current_bg_sel.angle_deg,
                    bbox=(0,0,0,0), # Dummy
                    area= int(np.sum(roi_mask)),
                    R_mean=row_data["roi_R_mean"], G1_mean=row_data["roi_G1_mean"], G2_mean=row_data["roi_G2_mean"], B_mean=row_data["roi_B_mean"],
                    R_bg=row_data["bg_R_mean"], G1_bg=row_data["bg_G1_mean"], G2_bg=row_data["bg_G2_mean"], B_bg=row_data["bg_B_mean"],
                    R_corrected=row_data["R_corrected"], G1_corrected=row_data["G1_corrected"], G2_corrected=row_data["G2_corrected"], B_corrected=row_data["B_corrected"],
                    R_display=row_data["R_corrected"], G_display=(row_data["G1_corrected"]+row_data["G2_corrected"])/2, B_display=row_data["B_corrected"]
                )
                roi_results_list.append(roi_res_obj.to_dict())

                df_row = pd.DataFrame([row_data])

                # Build features
                # Check if model uses bg subtraction. The rgb_to_concentration logic handles this.
                # However, predict_batch expects a CSV path. We want to predict from memory.
                # We reuse build_feature_frame from physics_features.
                
                # IMPORTANT: build_feature_frame logic handles background columns if present and requested.
                # We need to ensure we pass the right flags.
                
                dataset_use_bg = use_bg
                if use_bg and not pf.has_background_columns(df_row, g_mode):
                    dataset_use_bg = False # Should not happen as we provided them
                
                features = pf.build_feature_frame(df_row, g_mode=g_mode, use_bg=dataset_use_bg, clip_min=CLIP_MIN) # Use CLIP_MIN from config to match training pipeline
                
                # Predict
                pred_target = model.predict(features.to_numpy())[0]
                pred = rtc.inverse_target(np.array([pred_target]), target_transform, target_eps)[0]
                pred = rtc.apply_calibrator(np.array([pred]), calibrator)[0]

                result_entry = {
                    "filename": os.path.basename(file_path),
                    "concentration": float(pred),
                    "stats": row_data
                }
                results.append(result_entry)

            # Save roi_results.json equivalent
            json_out_path = out_dir_path / "roi_results.json"
            with open(json_out_path, "w", encoding="utf-8") as f:
                json.dump(roi_results_list, f, indent=2)

            self.result.emit(results)
            self.finished.emit()

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))

class LabAnalyzerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_lang = "en"
        self.setWindowTitle("Quantum Dot Concentration Analyzer")
        self.resize(1000, 700)
        
        # State
        self.input_dir = ""
        self.output_dir = ""
        self.file_list = []
        self.roi_selection = None
        self.bg_selection = None
        self.roi_cache = {}  # Cache mapping filename -> (roi, bg)
        self.roi_loaded_source = None # To track where ROI came from for status text
        
        # UI Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Language Selection
        lang_layout = QHBoxLayout()
        self.lbl_lang = QLabel("Language:")
        self.combo_lang = QComboBox()
        self.combo_lang.addItems(["English", "中文"])
        self.combo_lang.currentIndexChanged.connect(self.change_language)
        lang_layout.addStretch()
        lang_layout.addWidget(self.lbl_lang)
        lang_layout.addWidget(self.combo_lang)
        main_layout.addLayout(lang_layout)

        # 1. Configuration Section
        self.group_config = QGroupBox("Configuration")
        config_layout = QGridLayout()
        
        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        self.btn_input = QPushButton("Select Input Folder")
        self.btn_input.clicked.connect(self.select_input_dir)
        
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.btn_output = QPushButton("Select Output Folder")
        self.btn_output.clicked.connect(self.select_output_dir)
        
        self.lbl_input = QLabel("Input Dir:")
        self.lbl_output = QLabel("Output Dir:")
        
        config_layout.addWidget(self.lbl_input, 0, 0)
        config_layout.addWidget(self.input_edit, 0, 1)
        config_layout.addWidget(self.btn_input, 0, 2)
        
        config_layout.addWidget(self.lbl_output, 1, 0)
        config_layout.addWidget(self.output_edit, 1, 1)
        config_layout.addWidget(self.btn_output, 1, 2)
        
        self.group_config.setLayout(config_layout)
        main_layout.addWidget(self.group_config)

        # 2. ROI/BG Setup
        self.group_roi = QGroupBox("ROI & Background Selection")
        roi_layout = QHBoxLayout()
        
        self.btn_set_roi = QPushButton("Set ROI / BG ROI on First Image")
        self.btn_set_roi.clicked.connect(self.set_roi_bg)
        self.btn_set_roi.setEnabled(False)
        
        self.lbl_roi_status = QLabel("ROI Status: Not Set")
        self.lbl_roi_status.setStyleSheet("color: red")
        
        roi_layout.addWidget(self.btn_set_roi)
        roi_layout.addWidget(self.lbl_roi_status)
        self.group_roi.setLayout(roi_layout)
        main_layout.addWidget(self.group_roi)

        # 3. Action
        self.group_action = QGroupBox("Analysis")
        action_layout = QHBoxLayout()
        
        self.btn_run = QPushButton("Calculate Concentrations (Batch)")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        
        action_layout.addWidget(self.btn_run)
        self.group_action.setLayout(action_layout)
        main_layout.addWidget(self.group_action)
        
        # 4. Results
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Filename", "Predicted Conc.", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        main_layout.addWidget(self.table)
        
        self.status_bar = self.statusBar()

        self.update_ui_text()

    def change_language(self, index):
        self.current_lang = "en" if index == 0 else "zh"
        self.update_ui_text()

    def update_ui_text(self):
        t = TRANSLATIONS[self.current_lang]
        self.setWindowTitle(t["window_title"])
        
        self.lbl_lang.setText(t["lang_label"])
        
        self.group_config.setTitle(t["config_group"])
        self.lbl_input.setText(t["input_label"])
        self.lbl_output.setText(t["output_label"])
        self.btn_input.setText(t["select_input"])
        self.btn_output.setText(t["select_output"])
        
        self.group_roi.setTitle(t["roi_group"])
        self.btn_set_roi.setText(t["set_roi_btn"])
        
        # Update ROI Status
        if self.roi_selection is None:
             self.lbl_roi_status.setText(t["roi_status_not_set"])
        else:
             cx, cy = self.roi_selection.center
             if self.roi_loaded_source: 
                 self.lbl_roi_status.setText(t["roi_status_loaded"].format(cx, cy))
             else:
                 self.lbl_roi_status.setText(t["roi_status_set"].format(cx, cy))

        self.group_action.setTitle(t["analysis_group"])
        self.btn_run.setText(t["run_btn"])
        
        # Preserve header state if needed, but here labels are fine
        self.table.setHorizontalHeaderLabels(t["headers"])

    def select_input_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if d:
            self.input_dir = d
            self.input_edit.setText(d)
            self.refresh_file_list()
            self.btn_set_roi.setEnabled(True)

    def select_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.output_dir = d
            self.output_edit.setText(d)
            self.try_load_roi_from_output()

    def try_load_roi_from_output(self):
        if not self.output_dir:
            return
            
        json_path = os.path.join(self.output_dir, "roi_results.json")
        if not os.path.exists(json_path):
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reset cache
            self.roi_cache = {}
            target_entry = None

            if isinstance(data, list):
                # Format: List of ROIResult dicts
                if len(data) > 0:
                    for entry in data:
                        fname = entry.get("filename")
                        if not fname: continue
                        try:
                            r = ROISelection(
                                center=(float(entry["center_x"]), float(entry["center_y"])),
                                width=float(entry["width"]),
                                height=float(entry["height"]),
                                angle_deg=float(entry.get("angle_deg", 0.0)),
                            )
                            b = BackgroundSelection(
                                center=(float(entry.get("bg_center_x", r.center[0])), float(entry.get("bg_center_y", r.center[1]))),
                                inner_x=float(entry.get("bg_inner_x", 0)),
                                inner_y=float(entry.get("bg_inner_y", 0)),
                                outer_x=float(entry.get("bg_outer_x", 0)),
                                outer_y=float(entry.get("bg_outer_y", 0)),
                                angle_deg=float(entry.get("bg_angle_deg", r.angle_deg)),
                            )
                            self.roi_cache[fname] = (r, b)
                        except:
                            continue
                    
                    # Use the last entry as default for UI
                    target_entry = data[-1]

            elif isinstance(data, dict):
                # Format: Cache dict {filename: {roi:..., bg:...}}
                for fname, pair in data.items():
                    try:
                        r = ROISelection.from_dict(pair.get("roi", {}))
                        b = BackgroundSelection.from_dict(pair.get("bg", {}))
                        self.roi_cache[fname] = (r, b)
                    except:
                        continue
                
                # Use first entry as default
                if len(data) > 0:
                    first_key = next(iter(data))
                    val = data[first_key]
                    # Create dummy entry dict for below logic
                    if "roi" in val:
                        r_curr = ROISelection.from_dict(val["roi"])
                        b_curr = BackgroundSelection.from_dict(val["bg"])
                        target_entry = {
                            "center_x": r_curr.center[0], "center_y": r_curr.center[1],
                            "width": r_curr.width, "height": r_curr.height,
                            "angle_deg": r_curr.angle_deg,
                            "bg_center_x": b_curr.center[0], "bg_center_y": b_curr.center[1],
                            "bg_inner_x": b_curr.inner_x, "bg_inner_y": b_curr.inner_y,
                            "bg_outer_x": b_curr.outer_x, "bg_outer_y": b_curr.outer_y,
                            "bg_angle_deg": b_curr.angle_deg
                        }

            if target_entry:
                try:
                    self.roi_selection = ROISelection(
                        center=(float(target_entry["center_x"]), float(target_entry["center_y"])),
                        width=float(target_entry["width"]),
                        height=float(target_entry["height"]),
                        angle_deg=float(target_entry.get("angle_deg", 0.0)),
                    )
                    self.bg_selection = BackgroundSelection(
                        center=(
                            float(target_entry.get("bg_center_x", self.roi_selection.center[0])),
                            float(target_entry.get("bg_center_y", self.roi_selection.center[1])),
                        ),
                        inner_x=float(target_entry.get("bg_inner_x", 0)),
                        inner_y=float(target_entry.get("bg_inner_y", 0)),
                        outer_x=float(target_entry.get("bg_outer_x", 0)),
                        outer_y=float(target_entry.get("bg_outer_y", 0)),
                        angle_deg=float(target_entry.get("bg_angle_deg", self.roi_selection.angle_deg)),
                    )
                    self._update_roi_status_ui(json_path)
                except Exception as e:
                     print(f"Error parsing ROI entry: {e}")

        except Exception as e:
            print(f"Failed to load roi_results.json: {e}")
            
    def _update_roi_status_ui(self, source_path):
        self.roi_loaded_source = source_path
        t = TRANSLATIONS[self.current_lang]
        cx, cy = self.roi_selection.center
        self.lbl_roi_status.setText(t["roi_status_loaded"].format(cx, cy))
        self.lbl_roi_status.setStyleSheet("color: blue")
        self.btn_run.setEnabled(True)
        self.status_bar.showMessage(f"Loaded ROI settings from {os.path.basename(source_path)}")

    def refresh_file_list(self):
        if not self.input_dir:
            return
        
        types = ('*.dng', '*.tiff', '*.tif', '*.raw') 
        self.file_list = []
        for t in types:
            self.file_list.extend(glob.glob(os.path.join(self.input_dir, t)))
        
        t = TRANSLATIONS[self.current_lang]
        self.status_bar.showMessage(t["status_found_images"].format(len(self.file_list)))
        
        # Update table preliminary
        self.table.setRowCount(len(self.file_list))
        for i, f in enumerate(self.file_list):
            self.table.setItem(i, 0, QTableWidgetItem(os.path.basename(f)))
            self.table.setItem(i, 1, QTableWidgetItem("-"))
            self.table.setItem(i, 2, QTableWidgetItem("Pending"))

    def set_roi_bg(self):
        t = TRANSLATIONS[self.current_lang]
        if not self.file_list:
            QMessageBox.warning(self, "No Images", t["msg_no_images"])
            return

        # Load first image
        first_img = self.file_list[0]
        try:
            processor = ROIProcessor()
            raw_channels, _ = processor.load_raw_image(Path(first_img))
            
            # Use settings from pic_RGB
            settings = ROISettings() # Defaults
            
            # Launch GUI
            # Note: select_roi_with_gui creates its own QApplication if needed, 
            # but we are already in one. Passing existing selections if available.
            
            roi, bg = select_roi_with_gui(
                raw_channels=raw_channels,
                settings=settings,
                initial_roi=self.roi_selection,
                initial_bg=self.bg_selection
            )
            
            self.roi_selection = roi
            self.bg_selection = bg
            
            self.roi_loaded_source = None
            cx, cy = roi.center
            self.lbl_roi_status.setText(t["roi_status_set"].format(cx, cy))
            self.lbl_roi_status.setStyleSheet("color: green")
            self.btn_run.setEnabled(True)
            
        except Exception as e:
            if "cancelled by user" in str(e):
                self.status_bar.showMessage(t["msg_roi_cancelled"])
                return
            QMessageBox.critical(self, t["msg_error"], f"Failed to open ROI selector: {str(e)}\n{traceback.format_exc()}")

    def run_analysis(self):
        t = TRANSLATIONS[self.current_lang]
        if not self.input_dir or not self.output_dir:
            QMessageBox.warning(self, t["config_group"], t["msg_missing_config"])
            return
        
        if not self.roi_selection:
            QMessageBox.warning(self, t["roi_group"], t["msg_missing_roi"])
            return

        # Load Model
        try:
            base_model_dir = get_base_path()
            model_path = base_model_dir / "concentration" / "rgb_fit_output" / "concentration_model_v1.pkl"
            
            # Fallback for dev mode if the structure is slightly different or if run from root
            if not model_path.exists():
                 # Try typical dev path relative to WORKSPACE_ROOT (current dir)
                 model_path = Path(WORKSPACE_ROOT) / "concentration" / "rgb_fit_output" / "concentration_model_v1.pkl"

            if not model_path.exists():
                QMessageBox.critical(self, t["msg_error"], t["msg_model_error"].format(model_path))
                return
            
            model_bundle = rtc.load_model_bundle(model_path, None)
            
        except Exception as e:
            QMessageBox.critical(self, t["msg_error"], t["msg_load_model_fail"].format(str(e)))
            return

        # Disable UI
        self.btn_run.setEnabled(False)
        self.btn_set_roi.setEnabled(False)
        
        # Start Worker
        self.worker = AnalysisWorker(self.file_list, self.roi_selection, self.bg_selection, self.roi_cache, model_bundle, self.output_dir)
        self.worker.progress.connect(self.status_bar.showMessage)
        self.worker.result.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def on_analysis_finished(self, results):
        t = TRANSLATIONS[self.current_lang]
        self.status_bar.showMessage(t["msg_analysis_complete"])
        self.btn_run.setEnabled(True)
        self.btn_set_roi.setEnabled(True)
        
        # Save CSV
        try:
            out_df = pd.DataFrame([
                {
                    "filename": r["filename"], 
                    "conc_pred": r["concentration"],
                    **r["stats"]
                }
                for r in results
            ])
            out_path = os.path.join(self.output_dir, "concentration_results.csv")
            out_df.to_csv(out_path, index=False)
            QMessageBox.information(self, t["msg_success"], t["msg_save_success"].format(out_path))
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Could not save CSV: {str(e)}")

        # Update Table
        self.table.setRowCount(len(results))
        for i, r in enumerate(results):
            self.table.setItem(i, 0, QTableWidgetItem(r["filename"]))
            self.table.setItem(i, 1, QTableWidgetItem(f"{r['concentration']:.4f}"))
            self.table.setItem(i, 2, QTableWidgetItem("Done"))

    def on_analysis_error(self, err_msg):
        t = TRANSLATIONS[self.current_lang]
        QMessageBox.critical(self, t["msg_analysis_failed"], err_msg)
        self.btn_run.setEnabled(True)
        self.btn_set_roi.setEnabled(True)
        self.status_bar.showMessage(t["msg_error"])

from PySide6.QtWidgets import QGridLayout

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabAnalyzerWindow()
    window.show()
    sys.exit(app.exec())
