from __future__ import annotations

import contextlib
import sys
import traceback
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
PIC_RGB_DIR = REPO_ROOT / "pic_RGB"
for import_path in (REPO_ROOT, PIC_RGB_DIR):
    text_path = str(import_path)
    if text_path not in sys.path:
        sys.path.insert(0, text_path)

from PySide6.QtCore import QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from roi_config import DATA_DIR, OUTPUT_DIR, SAVE_VISUALIZATION, ROISettings, get_settings
from roi_processor import ROIProcessor


APP_NAME = "pic_RGB ROI Processor"


class QtLogStream:
    def __init__(self, widget: QTextEdit):
        self.widget = widget

    def write(self, text: str) -> None:
        if not text:
            return
        cursor = self.widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.widget.setTextCursor(cursor)
        self.widget.ensureCursorVisible()
        QApplication.processEvents()

    def flush(self) -> None:
        QApplication.processEvents()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(980, 720)

        self.selected_files: List[Path] = []
        self.selected_folder: Optional[Path] = DATA_DIR

        root = QWidget()
        layout = QVBoxLayout(root)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(220)

        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_config_group())
        layout.addWidget(self._build_run_group())
        layout.addWidget(self.log, stretch=1)

        self.setCentralWidget(root)
        self._refresh_input_summary()

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("输入和输出")
        layout = QGridLayout(group)

        self.input_line = QLineEdit(str(DATA_DIR))
        self.input_line.setReadOnly(True)
        self.output_line = QLineEdit(str(OUTPUT_DIR))

        btn_folder = QPushButton("选择文件夹")
        btn_files = QPushButton("选择 DNG 文件")
        btn_output = QPushButton("选择输出目录")
        btn_folder.clicked.connect(self._choose_folder)
        btn_files.clicked.connect(self._choose_files)
        btn_output.clicked.connect(self._choose_output)

        self.input_summary = QLabel("")
        self.input_summary.setWordWrap(True)

        layout.addWidget(QLabel("输入:"), 0, 0)
        layout.addWidget(self.input_line, 0, 1)
        layout.addWidget(btn_folder, 0, 2)
        layout.addWidget(btn_files, 0, 3)
        layout.addWidget(QLabel("输出:"), 1, 0)
        layout.addWidget(self.output_line, 1, 1)
        layout.addWidget(btn_output, 1, 2, 1, 2)
        layout.addWidget(self.input_summary, 2, 1, 1, 3)
        layout.setColumnStretch(1, 1)
        return group

    def _build_config_group(self) -> QGroupBox:
        group = QGroupBox("ROI config")
        layout = QHBoxLayout(group)

        settings = get_settings()
        self.config_widgets = {}

        roi_group = QGroupBox("ROI / BG 参数")
        form = QFormLayout(roi_group)
        self._add_int_config(form, "fixed_roi_width", settings.fixed_roi_width, 1, 10000)
        self._add_int_config(form, "fixed_roi_height", settings.fixed_roi_height, 1, 10000)
        self._add_float_config(form, "roi_angle_deg", settings.roi_angle_deg, -360.0, 360.0, 0.1)
        self._add_int_config(form, "bg_ring_inner_x", settings.bg_ring_inner_x, 0, 10000)
        self._add_int_config(form, "bg_ring_inner_y", settings.bg_ring_inner_y, 0, 10000)
        self._add_int_config(form, "bg_ring_outer_x", settings.bg_ring_outer_x, 1, 10000)
        self._add_int_config(form, "bg_ring_outer_y", settings.bg_ring_outer_y, 1, 10000)

        gui_group = QGroupBox("GUI 行为")
        gui_form = QFormLayout(gui_group)
        self._add_int_config(gui_form, "move_step", settings.move_step, 1, 1000)
        self._add_int_config(gui_form, "size_step", settings.size_step, 1, 1000)
        self._add_int_config(
            gui_form,
            "spin_repeat_interval_ms",
            settings.spin_repeat_interval_ms,
            1,
            5000,
        )
        self._add_float_config(gui_form, "angle_step_deg", settings.angle_step_deg, 0.01, 90.0, 0.01)

        self.save_visualization_check = QCheckBox("保存每张图的 ROI 可视化 PNG")
        self.save_visualization_check.setChecked(bool(SAVE_VISUALIZATION))
        self.interactive_check = QCheckBox("显示 ROI/BG 选择窗口")
        self.interactive_check.setChecked(True)
        self.reuse_cache_check = QCheckBox("有缓存时直接复用 ROI/BG")
        self.reuse_cache_check.setChecked(False)
        gui_form.addRow(self.save_visualization_check)
        gui_form.addRow(self.interactive_check)
        gui_form.addRow(self.reuse_cache_check)

        layout.addWidget(roi_group, stretch=1)
        layout.addWidget(gui_group, stretch=1)
        return group

    def _build_run_group(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.run_button = QPushButton("开始处理")
        self.run_button.clicked.connect(self._run_clicked)
        clear_button = QPushButton("清空日志")
        clear_button.clicked.connect(self.log.clear)

        layout.addStretch(1)
        layout.addWidget(clear_button)
        layout.addWidget(self.run_button)
        return widget

    def _add_int_config(
        self, form: QFormLayout, name: str, value: int, minimum: int, maximum: int
    ) -> None:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(int(value))
        self.config_widgets[name] = spin
        form.addRow(name, spin)

    def _add_float_config(
        self,
        form: QFormLayout,
        name: str,
        value: float,
        minimum: float,
        maximum: float,
        step: float,
    ) -> None:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(4)
        spin.setSingleStep(step)
        spin.setValue(float(value))
        self.config_widgets[name] = spin
        form.addRow(name, spin)

    def _choose_folder(self) -> None:
        start = str(self.selected_folder or DATA_DIR)
        folder = QFileDialog.getExistingDirectory(self, "选择包含 DNG 的文件夹", start)
        if not folder:
            return
        self.selected_folder = Path(folder)
        self.selected_files = []
        self.input_line.setText(str(self.selected_folder))
        if not self.output_line.text().strip():
            self.output_line.setText(str(self.selected_folder / "roi_results"))
        self._refresh_input_summary()

    def _choose_files(self) -> None:
        start = str(self.selected_folder or DATA_DIR)
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择 DNG 文件",
            start,
            "DNG images (*.dng *.DNG);;All files (*)",
        )
        if not files:
            return
        self.selected_files = [Path(path) for path in files]
        self.selected_folder = None
        self.input_line.setText("; ".join(str(path) for path in self.selected_files))
        if not self.output_line.text().strip():
            self.output_line.setText(str(self.selected_files[0].parent / "roi_results"))
        self._refresh_input_summary()

    def _choose_output(self) -> None:
        start = self.output_line.text().strip() or str(OUTPUT_DIR)
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录", start)
        if folder:
            self.output_line.setText(folder)

    def _refresh_input_summary(self) -> None:
        try:
            files = self._resolve_input_files()
            self.input_summary.setText(f"将处理 {len(files)} 个 DNG 文件")
        except Exception as exc:
            self.input_summary.setText(str(exc))

    def _run_clicked(self) -> None:
        self.run_button.setEnabled(False)
        QTimer.singleShot(0, self._run_processing)

    def _run_processing(self) -> None:
        self.log.clear()
        stream = QtLogStream(self.log)
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                files = self._resolve_input_files()
                output_dir = self._resolve_output_dir(files)
                settings = self._build_settings()

                print(f"{APP_NAME}")
                print(f"输入 DNG 数量: {len(files)}")
                print(f"输出目录: {output_dir}")
                print("")

                processor = ROIProcessor(
                    settings=settings,
                    reuse_last_selection=self.reuse_cache_check.isChecked(),
                    selection_cache_path=output_dir / "roi_selection_cache.json",
                )

                results = []
                failed = []
                for index, filepath in enumerate(files, start=1):
                    print(f"\n[{index}/{len(files)}] {filepath}")
                    try:
                        result = processor.process_image(
                            filepath=filepath,
                            save_visualization=self.save_visualization_check.isChecked(),
                            output_dir=output_dir,
                            interactive=self.interactive_check.isChecked(),
                        )
                        results.append(result)
                    except Exception as exc:
                        print(f"  错误: {exc}")
                        failed.append(filepath.name)

                print(f"\n处理完成: 成功 {len(results)}/{len(files)}")
                if failed:
                    print(f"失败文件: {failed}")
                if results:
                    processor._save_results(results, output_dir)
                    print("\n输出文件:")
                    print(f"  {output_dir / 'roi_results.json'}")
                    print(f"  {output_dir / 'roi_detailed_statistics.csv'}")
                    print(f"  {output_dir / 'roi_selection_cache.json'}")
                else:
                    print("没有成功结果，未写出 CSV/JSON。")
        except Exception as exc:
            stream.write("\n")
            stream.write(traceback.format_exc())
            QMessageBox.critical(self, "处理失败", str(exc))
        finally:
            self.run_button.setEnabled(True)

    def _resolve_input_files(self) -> List[Path]:
        if self.selected_files:
            files = [path.expanduser().resolve() for path in self.selected_files]
        else:
            folder_text = self.input_line.text().strip()
            folder = Path(folder_text).expanduser().resolve() if folder_text else DATA_DIR
            if not folder.exists():
                raise FileNotFoundError(f"输入路径不存在: {folder}")
            if not folder.is_dir():
                raise NotADirectoryError(f"输入路径不是文件夹: {folder}")
            self.selected_folder = folder
            files = sorted(path for path in folder.iterdir() if self._is_dng(path))

        files = [path for path in files if path.is_file() and self._is_dng(path)]
        if not files:
            raise ValueError("未找到 DNG 文件")
        return files

    def _resolve_output_dir(self, files: Iterable[Path]) -> Path:
        text = self.output_line.text().strip()
        if text:
            return Path(text).expanduser().resolve()
        file_list = list(files)
        if self.selected_folder:
            return (self.selected_folder / "roi_results").resolve()
        return (file_list[0].parent / "roi_results").resolve()

    def _build_settings(self) -> ROISettings:
        values = {}
        for key, widget in self.config_widgets.items():
            values[key] = widget.value()
        return ROISettings.from_dict(values)

    @staticmethod
    def _is_dng(path: Path) -> bool:
        return path.suffix.lower() == ".dng"


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
