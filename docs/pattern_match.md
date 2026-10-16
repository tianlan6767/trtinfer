# 灰度模板匹配（NCC）

金字塔 + 旋转搜索的归一化互相关（`TM_CCOEFF_NORMED` 等价）。看整块灰度，不看轮廓。近圆目标（螺母扁位）转一点分数几乎不变，角度不可靠，那种场景用 [形状匹配](shape_match.md) 或 [卡尺](caliper.md)。

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
