# GUI ROI/BG Ring 尺寸调整修复总结

## 问题描述
在使用GUI时，通过SpinBox控件修改ROI或BG ring的尺寸后，其中心位置会发生意外的变化。这给用户带来困扰，因为他们期望只改变尺寸而中心位置保持不变。

## 原因分析
之前的实现在`_on_controls_interaction`方法中包含了一个锚定逻辑，该逻辑试图在尺寸变化时自动调整中心位置以保持某一边缘固定。但这个行为对于通过SpinBox进行的显式修改是不合适的：

- **键盘快捷键调整**：使用箭头键调整尺寸时，自动调整中心以保持底部/左侧边缘固定是合理的
- **SpinBox调整**：用户明确输入数值时，应该保持中心位置不变（除非用户同时也修改了中心坐标）

## 解决方案
修改了 [pic_RGB/gui/roi_gui.py](pic_RGB/gui/roi_gui.py) 中的`_on_controls_interaction`方法：

### 修改前
```python
def _on_controls_interaction(self, data):
    if "roi" in data:
        new_roi = data["roi"]
        d_w = new_roi.width - self.roi.width
        d_h = new_roi.height - self.roi.height
        
        if abs(d_w) > 1e-4 or abs(d_h) > 1e-4:
            # 计算偏移以保持锚定... (这会导致中心意外移动)
            angle_rad = math.radians(self.roi.angle_deg)
            shift_x = ...
            shift_y = ...
            new_roi.center = (new_cx, new_cy)  # 修改了中心！
        
        self.roi = new_roi
    # BG ring 也有类似的问题...
```

### 修改后
```python
def _on_controls_interaction(self, data):
    # 当用户通过SpinBox修改值时，直接应用它们，不进行中心位置调整
    # SpinBox的值代表绝对期望值
    # 中心位置偏移逻辑只应用于键盘快捷键
    
    if "roi" in data:
        self.roi = data["roi"]
        
    if "bg" in data:
        self.bg = data["bg"]
        
    self.canvas.set_data(self.roi, self.bg)
    self.controls.update_from_selection(self.roi, self.bg)
```

## 行为对比

### 使用SpinBox修改尺寸
- **修改前**：中心位置会自动调整（不符合预期）
- **修改后**：中心位置保持不变（符合预期）✓

### 使用键盘箭头键调整尺寸
- **修改前**：保持底部/左侧边缘固定，中心自动调整
- **修改后**：保持底部/左侧边缘固定，中心自动调整（行为不变）✓

## 测试验证
运行 [test_gui_fix.py](test_gui_fix.py) 验证了以下场景：

1. ✓ 通过SpinBox修改ROI宽度 → 中心保持不变
2. ✓ 通过SpinBox修改ROI高度 → 中心保持不变
3. ✓ 通过SpinBox修改BG ring外圈尺寸 → 中心保持不变
4. ✓ 同时修改中心和尺寸 → 两者都更新为用户设定值

## 用户体验改进
- 用户可以独立地调整ROI/BG ring的尺寸和中心位置
- 修改尺寸时不会出现中心"跳动"的现象
- GUI行为更加直观和可预测
- 键盘快捷键的锚定行为保持不变，为高级用户提供便利

## 相关文件
- [pic_RGB/gui/roi_gui.py](pic_RGB/gui/roi_gui.py#L564-L577) - 修复的主要代码
- [test_gui_fix.py](test_gui_fix.py) - 测试脚本
