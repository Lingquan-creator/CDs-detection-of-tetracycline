# to_dng 算法说明

本目录负责把 8-bit Bayer RAW 缓冲文件转换为最小可读 DNG，也可以只对 RAW 图做 ROI 统计分析。

## 主要代码

- `convert_raw_to_dng.py`: RAW 文件收集、尺寸推断、Bayer 数据解码、DNG 写入、元数据收集、ROI 分析和输出校验。

## 算法输入

算法层面的输入是：

- 一段 8-bit Bayer RAW 字节流。
- 图像宽高，或可由文件大小推断出的宽高。
- Bayer 排列，例如 `RGGB`, `GRBG`, `BGGR`, `GBRG`。
- RAW 数据存储方式，例如直接平面或带行 padding 的格式。
- DNG 元数据：相机 make/model、曝光时间、ISO、黑电平、白电平等。
- 可选 ROI：`x, y, w, h`。

实际默认输入：

- `to_dng/` 目录下匹配 `*.raw` 的文件。

常用命令行参数：

- `--input`: RAW 文件或目录。
- `--output`: DNG 文件或输出目录，默认 `to_dng/output/`。
- `--input-glob`: 批量匹配规则，默认 `*.raw`。
- `--pattern`: Bayer pattern，默认会标准化为 DNG 支持的 pattern。
- `--width`, `--height`: 手动指定尺寸。
- `--conversion`: RAW 解码方式。
- `--black-level`, `--white-level`: 写入 DNG 的黑白电平。
- `--select-roi`, `--roi`, `--roi-only`: ROI 统计相关选项。

## 处理流程

1. 收集输入 RAW 文件。如果 `--input` 是目录，则按 `--input-glob` 批量处理。
2. 将 RAW 文件读为 `uint8` 数组。
3. 推断或读取图像尺寸：
   - 优先使用 `--width` 和 `--height`。
   - 否则根据字节数和常见宽高比推断。
4. 根据 `--conversion` 将字节流解码成二维 Bayer 平面。
5. 如果启用 ROI 模式：
   - 使用 `--roi x,y,w,h` 或交互式框选得到 ROI。
   - 裁剪 ROI 到图像范围内。
   - 计算 ROI 的 mean、std、min、max 等统计量。
   - 保存 ROI JSON。
6. 如果不是 `--roi-only`，继续收集元数据：
   - 命令行参数。
   - 同名 sidecar JSON，例如 `.json`, `.meta.json`, `.metadata.json`。
   - 文件名中可能包含的曝光/ISO 信息。
7. 使用 `tifffile` 写入未压缩 DNG：
   - 写入 RAW Bayer 平面。
   - 写入 CFA pattern、black level、white level、make/model、曝光等标签。
8. 可选用 `rawpy` 重新打开输出 DNG 做校验。

## 算法输出

算法层面的输出是：

- 解码后的 Bayer 平面。
- DNG 文件。
- 可选 ROI 统计结果。

实际默认输出：

- DNG 输出目录：`to_dng/output/`
- 单文件转换时输出 `<input_stem>.dng`。
- ROI 模式输出 `<input_stem>.roi.json`，或 `--roi-output` 指定的 JSON。

ROI JSON 通常包含：

- 输入文件路径。
- 图像宽高。
- Bayer pattern 和 conversion。
- ROI 的 `x`, `y`, `w`, `h`。
- ROI 的 `mean`, `std`, `min`, `max` 等统计量。

## 使用方式

```bash
python to_dng/convert_raw_to_dng.py
python to_dng/convert_raw_to_dng.py --input sample.raw --output sample.dng --width 4000 --height 3000 --pattern GRBG
python to_dng/convert_raw_to_dng.py --input sample.raw --width 4000 --height 3000 --roi 100,100,200,200 --roi-only
```
