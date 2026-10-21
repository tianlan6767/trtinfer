# 灰度模板匹配（NCC）

金字塔 + 旋转搜索的归一化互相关（`TM_CCOEFF_NORMED` 等价）。看整块灰度，不看轮廓。近圆目标（螺母扁位）转一点分数几乎不变，角度不可靠，那种场景用 [形状匹配](shape_match.md) 或 [卡尺](caliper.md)。

**尺度搜索：`PATTERN` 和 `SHAPE` 都支持。** 用的是同一组 `MatcherParam.minScale` / `maxScale` / `scaleStep` / `numScales`。默认 `minScale=maxScale=1`，只搜原尺度，`MatchResult.Scale` 为 1。拉开上下界之后，`Scale` 是命中的采样尺度。形状匹配的尺度说明见 [shape_match.md](shape_match.md)。

源码：`src/cv_match/`。测试：`make test-match`（完整 `cvter.so`）。尺度搜索还可单独跑 `make test-scale`（只链 OpenCV，不需要 TensorRT）。实现变更记录：[cv_match_update.md](cv_match_update.md)。

## Python

```python
import cvter
import cv2
import numpy as np

templ = cv2.imread("template.jpg", cv2.IMREAD_GRAYSCALE)
gray = cv2.imread("search.jpg", cv2.IMREAD_GRAYSCALE)

param = cvter.MatcherParam()
param.matcherType = cvter.MatcherType.PATTERN
param.angle = 8                 # ±8°
param.scoreThreshold = 0.55
param.maxCount = 1
param.minArea = 256             # 顶层金字塔最小面积
param.iouThreshold = 0.3
param.meanBorder = True         # 旋转空白填模板均值
param.stopLayer = 0             # 0=精搜到原图；1=停在第 1 层（更快）
param.minScale = 1.0            # 与 maxScale 都为 1 时不搜尺度
param.maxScale = 1.0
param.scaleStep = 0.05          # 拉开 min/max 后使用；或改 numScales

m = cvter.MatcherWrapper(param)
m.setTemplate(templ)

# 可选 mask：uint8，非 0 有效。挖掉通孔、拉丝
# mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
# m.setTemplate(templ, mask)

for r in m.match(gray):
    print(r.Center.x, r.Center.y, r.Angle, r.Score, r.Scale)
    pts = [
        (int(r.LeftTop.x), int(r.LeftTop.y)),
        (int(r.RightTop.x), int(r.RightTop.y)),
        (int(r.RightBottom.x), int(r.RightBottom.y)),
        (int(r.LeftBottom.x), int(r.LeftBottom.y)),
    ]
    cv2.polylines(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), [np.array(pts)], True, (0, 0, 255), 2)
```

`Angle` 与 OpenCV `getRotationMatrix2D` 相同：图像 y 向下，正角为画面逆时针。

## 参数

| 字段 | 默认 | 含义 |
|---|---|---|
| `matcherType` | `PATTERN` | 灰度 NCC |
| `maxCount` | 200 | 最多保留几个目标 |
| `scoreThreshold` | 0.5 | NCC 分数阈值，0~1 |
| `iouThreshold` | 0 | 旋转矩形 IoU 去重 |
| `angle` | 0 | 搜索 ±angle 度；0 只搜 0° |
| `minArea` | 256 | 顶层模板最小面积，决定金字塔层数 |
| `meanBorder` | true | 旋转填充用模板均值 |
| `stopLayer` | 0 | 精搜停止层 |
| `minScale` | 1 | 尺度下界（相对模板） |
| `maxScale` | 1 | 尺度上界。与 `minScale` 都为 1 时只搜原尺寸 |
| `scaleStep` | 0.05 | `minScale != maxScale` 且 `numScales<=1` 时的步长；`<=0` 仍按 0.05 |
| `numScales` | 0 | `>1` 时在 `[minScale, maxScale]` 上均匀取这么多个（含端点），忽略 `scaleStep` |

`edgeMinMag` / `maxEdgePoints` / `usePolarity` / `greediness` 只对 `SHAPE` 有效。

## 尺度

每个采样尺度会把模板（以及 mask，若有）缩放到该尺寸再走原来的金字塔 + 角度 NCC。`MatchResult.Scale` 是命中的采样尺度，四角按该尺寸的模板给出。尺度是离散网格，不会在两个步长之间再插值。

多实例再加尺度时，每个尺度先多留一些峰（同一目标的几个角度、相邻尺度上的错位锁），全部搜完再按中心距和较小框的重叠去掉重复，最后才按 `maxCount` 截断。角度采样包含 `±angle` 的端点，不会因为顶层步长太大而搜出这个范围。亚像素角度若偏离离散峰，或落到 `±angle` 外面，就退回离散角。

不搜尺度时，整幅对齐的 NCC 分数不变。细环（笔画只占模板一小圈，并且边缘像素不少于非背景的一半）如果有一段边缘对不上，会丢掉，哪怕全局分到 0.8。刚过 0.5、整圈又不齐的细环也会丢掉。整圈上的一块高光只拉低灰度相关、边缘还在，会留下。实心目标（瓶盖、脸、圆点）不按这个缺口规则删：欠曝时某一块混进噪声，或者只挡一个象限，其余部分对齐就留下。两块象限被压平时，只要还有两块对得很齐也会留下，峰可能往还剩结构的一侧偏。mask 盖住的象限不参与。

模板带大块空白边、周围背景又对不上时，全局 NCC 可能刚低于 `scoreThreshold`，但非背景像素在至少三个象限都对齐。这种完整弱实例会补回来，输出分改成非背景像素的 NCC。尺度搜索的粗层门槛更低，弱的较小实例能进精搜，最后仍按阈值或这条补救取舍。边长比超过约 1.10 且中心分开的两个框，不会因为大框包住小框而并成一个。

默认 `minScale=maxScale=1`，不缩放模板，结果 `Scale` 为 1，与以前只搜角度的行为一致。mask 在每个尺度上按最近邻缩放到对应模板大小后参与 masked NCC。

尺度种类有上限（48）。模板缩得过小（短边 < 8）的尺度会跳过。某个尺度的模板比搜索图更大时，该尺度跳过，其它尺度继续。

## 螺母扁位

```python
param.angle = 8
param.minArea = 256
param.meanBorder = True
matcher.setTemplate(templ, mask)   # mask 留外沿和扁位，挖孔和拉丝
```

NCC 负责把 ROI 放到零件附近；转角用卡尺复核。

## 大图先粗定位

整幅大图跑 NCC 很慢。先用 ORB/SIFT 框出 ROI，再在这块上精匹配，见 [feature.md](feature.md) 的「粗到精定位」。

```python
param = cvter.CoarseRefineParam()
param.matcherType = cvter.MatcherType.PATTERN
param.angle = 0          # 相信粗定位朝向；要转角再加大
param.margin = 120
param.stopLayer = 1      # 不要用 2 当加速，大图上会锁错位姿
loc = cvter.match_coarse_to_fine(gray, templ, mask, param)
hit = loc.hits[0]        # Center / Angle 已是全图坐标
```

`stopLayer` 的含义没有变。ROI 视图不用再手动 `ascontiguousarray`：绑定会按 numpy 行步长拷贝。非连续切片以前会让 `match` 返回 0 个命中。

## C++

```cpp
#include "cv_match/matcher.h"

template_matching::MatcherParam param;
param.matcherType = template_matching::MatcherType::PATTERN;
param.angle = 8;
param.scoreThreshold = 0.55;
param.maxCount = 1;
auto matcher = template_matching::GetMatcher(param);
matcher->setTemplate(templ);           // 或 setTemplate(templ, mask)
std::vector<template_matching::MatchResult> results;
matcher->match(gray, results);
```

可视化（不弹窗，图片在 `tests/out_viz/pattern/`）：`make test-viz` 里的 `tests/test_pattern_match_viz.py`。同一张搜索图上会画两个尺度。掩膜对照在 `tests/test_mask_match_viz.py`，图片在 `tests/out_viz/mask/`：模板 ROI 外有文字、孔和噪声，不带 mask 会锁到干扰峰，带 mask 回到目标位姿。见 [tests/README.md](../tests/README.md)。
