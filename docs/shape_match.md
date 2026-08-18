# 形状匹配（Shape-based）

灰度 NCC 用法见 [pattern_match.md](pattern_match.md)。全部 Python 示例见 [python.md](python.md)。

类似 Halcon `create_shape_model` / `find_shape_model`：用模板**边缘处的梯度方向**建模，在搜索图上对位置和角度打分。不看填充灰度，所以整体变亮、拉丝干扰通常比 NCC 稳。

## 和灰度 NCC 的区别

| | `PATTERN`（灰度 NCC） | `SHAPE`（形状） |
|---|---|---|
| 用什么 | 整块灰度 | 边缘点上的单位梯度 |
| 光照 | 去均值后较稳，仍吃纹理 | 几乎不吃整体亮度 |
| 近圆转角 | 差 | 更好（靠缺口/扁位等边缘） |
| 速度 | 相关 + 转图 | 顶层稀疏点打分，一般更快 |
| 遮挡 | 一块脏了分数掉 | 部分边缘对上仍可能过阈值 |

三种能力对照：

- 灰度匹配：`MatcherType.PATTERN` + `setTemplate(templ)`
- NCC+mask：`PATTERN` + `setTemplate(templ, mask)`
- 形状匹配：`MatcherType.SHAPE` + 可选 mask（只在 mask 内提边缘）

## 用法

```python
import cvter

param = cvter.MatcherParam()
param.matcherType = cvter.MatcherType.SHAPE
param.angle = 15              # ±15°
param.scoreThreshold = 0.5    # 0~1，梯度方向平均余弦
param.minArea = 256
param.edgeMinMag = 18
param.maxEdgePoints = 320
param.usePolarity = True      # 明暗方向要一致
param.greediness = 0.9
param.maxCount = 1

m = cvter.MatcherWrapper(param)
m.setTemplate(templ)                 # 或 m.setTemplate(templ, mask)
hits = m.match(gray)
# hits[0].Center / Angle / Score / 四角 与 PATTERN 相同
```

## 参数

| 字段 | 默认 | 含义 |
|---|---|---|
| `edgeMinMag` | 18 | 提边缘的最小梯度。漏边就降，噪声边就升 |
| `maxEdgePoints` | 320 | 每层最多点数，太大变慢 |
| `usePolarity` | true | true 要求亮暗方向一致；反光/曝光反了可 false |
| `greediness` | 0.9 | 越大越早放弃不可能过阈值的位置 |
| `angle` / `minArea` / `scoreThreshold` / `maxCount` / `stopLayer` | 同 NCC | 金字塔与搜索范围 |

分数是所有模型点上 `n_model · n_image` 的平均（单位法向点积），约等于方向余弦。1 为完全一致。`Angle` 与 OpenCV `getRotationMatrix2D` 相同：图像坐标 y 向下，正角为画面逆时针。

## 算法摘要

1. 学模板：Sobel → 梯度幅值局部极大 → 存相对中心的 `(dx,dy)` 和单位法向 `(nx,ny)`；可选 mask。  
2. 建搜索图金字塔并预计算单位梯度。  
3. 顶层：各角度旋转模型点，滑窗打分（OpenMP）。  
4. 下层：在候选附近 ±3 px、±2 个角度步长精搜。  
5. 按中心距/角度 NMS，输出旋转矩形。

无尺度搜索（与当前 NCC 一样）。

## 测试

```bash
make test-shape
make test-match
```
