# GUI ROI/BG中心位置保持修复说明

## 问题描述

在使用GUI时，通过SpinBox控件修改ROI或BG ring的尺寸后，其中心位置会发生意外的偏移。

## 问题原因

在原始实现中，`_on_controls_interaction` 方法直接将SpinBox中的所有值（包括中心坐标）应用到ROI和BG对象上。当用户只修改尺寸参数时，由于浮点数精度等原因，中心坐标可能会有微小的变化，导致中心位置发生偏移。

## 修复方案

修改 [roi_gui.py](pic_RGB/gui/roi_gui.py#L564-L603) 中的 `_on_controls_interaction` 方法：

### 修复逻辑

1. **检测用户意图**：
   - 比较新旧中心坐标（误差容限 0.01 像素）
   - 如果中心坐标基本不变，说明用户只修改了尺寸/角度参数
   - 如果中心坐标明显变化，说明用户确实要移动中心

2. **保持中心不变**：
   - 当用户只修改尺寸/角度时，使用旧的中心坐标
   - 只更新宽度、高度、角度等尺寸参数
   - 确保ROI和BG的中心位置不会因为尺寸调整而漂移

3. **允许中心移动**：
   - 当用户明确修改中心坐标SpinBox时，正常更新中心位置

### 代码变更

修改前：
```python
def _on_controls_interaction(self, data):
    if "roi" in data:
        self.roi = data["roi"]
    if "bg" in data:
        self.bg = data["bg"]
    self.canvas.set_data(self.roi, self.bg)
    self.controls.update_from_selection(self.roi, self.bg)
```

修改后：
```python
def _on_controls_interaction(self, data):
    if "roi" in data:
        new_roi = data["roi"]
        # 检查是否只是尺寸/角度变化
        if (abs(new_roi.center[0] - self.roi.center[0]) < 0.01 and
            abs(new_roi.center[1] - self.roi.center[1]) < 0.01):
            # 保持原中心，只更新尺寸/角度
            self.roi = ROISelection(
                center=self.roi.center,
                width=new_roi.width,
                height=new_roi.height,
                angle_deg=new_roi.angle_deg
            )
        else:
            # 用户修改了中心，使用新中心
            self.roi = new_roi
    
    if "bg" in data:
        new_bg = data["bg"]
        # 同样的逻辑应用于BG
        if (abs(new_bg.center[0] - self.bg.center[0]) < 0.01 and
            abs(new_bg.center[1] - self.bg.center[1]) < 0.01):
            self.bg = BackgroundSelection(
                center=self.bg.center,
                inner_x=new_bg.inner_x,
                inner_y=new_bg.inner_y,
                outer_x=new_bg.outer_x,
                outer_y=new_bg.outer_y,
                angle_deg=new_bg.angle_deg
            )
        else:
            self.bg = new_bg
    
    self.canvas.set_data(self.roi, self.bg)
    self.controls.update_from_selection(self.roi, self.bg)
```

## 测试验证

运行测试脚本 [test_roi_center_fix.py](test_roi_center_fix.py) 验证修复：

```bash
python test_roi_center_fix.py
```

测试结果：
- ✓ ROI中心保持测试通过
- ✓ BG中心保持测试通过

## 用户体验改进

### 修复前：
- 用户修改ROI宽度：50 → 60
- 结果：宽度改变 + 中心意外偏移

### 修复后：
- 用户修改ROI宽度：50 → 60  
- 结果：仅宽度改变，中心位置完全保持不变 ✓

## 相关行为保持不变

1. **键盘快捷键调整**：使用方向键调整尺寸时，保持底部/左侧边缘固定的行为不变
2. **中心移动**：通过WASD键或SpinBox修改中心坐标的功能正常工作
3. **旋转操作**：E/Q键旋转的功能不受影响

## 修改日期

2026年1月28日
