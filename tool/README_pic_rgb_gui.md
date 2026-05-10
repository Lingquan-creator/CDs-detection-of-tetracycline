# pic_RGB GUI 打包工具

这个目录提供 `pic_RGB` 的外层 GUI 程序，用于选择 DNG 输入文件夹或多个 DNG 文件，修改 `ROISettings` 参数，并输出原流程生成的：

- `roi_results.json`
- `roi_detailed_statistics.csv`
- `roi_selection_cache.json`

## 直接运行

运行脚本会先安装 `tool/requirements_pic_rgb_gui.txt` 中的 GUI 专用依赖。

Linux/macOS:

```bash
bash tool/run_pic_rgb_gui.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File tool\run_pic_rgb_gui.ps1
```

或者直接运行批处理包装脚本，避免当前系统的 PowerShell 脚本签名限制：

```bat
tool\run_pic_rgb_gui.bat
```

## 打包

Linux/macOS:

```bash
bash tool/build_pic_rgb_gui.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File tool\build_pic_rgb_gui.ps1
```

或者：

```bat
tool\build_pic_rgb_gui.bat
```

打包产物位于 `tool/dist/PicRGBRoiTool`。

打包脚本会安装 `tool/requirements_pic_rgb_gui.txt` 中的 GUI 专用依赖。

## 使用说明

1. 点击“选择文件夹”或“选择 DNG 文件”。
2. 选择输出目录。
3. 在 GUI 中修改 ROI/BG 和步长参数。
4. 点击“开始处理”。
5. 如果勾选“显示 ROI/BG 选择窗口”，每张 DNG 会弹出现有 ROI/BG 选择器。
6. 如果勾选“有缓存时直接复用 ROI/BG”，同名文件已有缓存时会跳过 ROI/BG 选择窗口。
