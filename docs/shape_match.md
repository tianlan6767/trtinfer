# 形状匹配（Shape-based）

灰度 NCC 用法见 [pattern_match.md](pattern_match.md)。全部 Python 示例见 [python.md](python.md)。

**尺度搜索：`SHAPE` 和 `PATTERN` 都支持。** 参数就是 `MatcherParam.minScale` / `maxScale` / `scaleStep` / `numScales`，两边一样。默认 `minScale=maxScale=1` 不搜尺度。结果在 `MatchResult.Scale`。灰度 NCC 是把模板（和 mask）缩放到该尺度再走金字塔；形状匹配是缩放边缘点，下层再在半个步长里细化。

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
param.minScale = 1.0          # 与 maxScale 都为 1 时不搜尺度
param.maxScale = 1.0
param.scaleStep = 0.05        # 拉开 min/max 后使用；或 numScales>1

m = cvter.MatcherWrapper(param)
m.setTemplate(templ)                 # 或 m.setTemplate(templ, mask)
hits = m.match(gray)
# hits[0].Center / Angle / Score / 四角 与 PATTERN 相同
```

## 参数

| 字段 | 默认 | 含义 |
|---|---|---|
| `edgeMinMag` | 18 | 最小对比度。低于这个梯度的点不进模型 |
| `edgeContrast` | 0 | 对比度。高于它的是种子；介于它和 `edgeMinMag` 之间的点要连到种子上才留。`0` 或不超过 `edgeMinMag` 时就是原来的单阈值 |
| `maxEdgePoints` | 320 | 每层最多点数。按角度绕着轮廓轮流取，避免孔或高光把名额占满 |
| `usePolarity` | true | true 要求亮暗方向一致；反光/曝光反了可 false |
| `greediness` | 0.9 | 越大越早放弃不可能过阈值的位置 |
| `oriBins` | 0 | `0` 全程浮点余弦。`16` 时顶层把方向收成 16 档，先做成 16 张响应图（每张在 ±3 像素里取该档余弦的最大值，只做一次）。粗搜步长 4，角度步长仍是 `atan(1/R)`。四边各约 16 个顶层像素改回浮点分，从粗搜到末层都和内部高峰分开去重，两边都用整幅去重半径。贴边种子第一层是 ±4 px、±3 步；内部响应图种子第一层角度大约 ±12°，位置先 ±4 px，不够再 ±10 px。和贴边框重叠时留下贴边。进精修的名额不少于原来的 4 倍 `maxCount`。末层仍是浮点余弦，分数尺度不变 |
| `angle` / `minArea` / `scoreThreshold` / `maxCount` / `stopLayer` | 同 NCC | 金字塔与搜索范围 |
| `minScale` / `maxScale` / `scaleStep` / `numScales` | 同 NCC | 尺度网格；默认只搜 1 |

分数是画面内模型点上 `n_model · n_image` 的平均（单位法向点积），约等于方向余弦。1 为完全一致。落在画布外的点跳过，不记 0；可见点少于大约 62% 时仍用全部点数做分母，避免贴边纹理被归一化成高分。极性打开时反方向记 0。最细一层对很弱的场景梯度少算，门槛大约是 `min(32, 2.5×edgeMinMag)`，不会跟着对比度设到 100 以上再把正常边缘乘小。闭合轮廓在最细一层再按象限覆盖打折，半圈对齐的分数会掉下去。残缺如果只缺一段，点是绕着轮廓交错打分的，贪心不会因为开头连续 miss 就把 0.5 以上的实例提前丢掉。`oriBins=16` 时顶层内部改用 16 张方向响应图查表，四边一条带和末层仍是上面的浮点平均。贴边种子后面的精修窗口也和全程浮点一样。`Angle` 与 OpenCV `getRotationMatrix2D` 相同：图像坐标 y 向下，正角为画面逆时针。`Scale` 是相对模板的尺度，四角按该尺度放大。

## 算法摘要

1. 学模板：Sobel → 梯度局部极大。`edgeContrast` 高于 `edgeMinMag` 时做双阈值，弱边缘连到强种子才留。点数按角度分散后存 `(dx,dy)` 和单位法向；可选 mask。`oriBins=16` 不挪这些点，末层和全程浮点用同一套边缘。  
2. 建搜索图金字塔并预计算单位梯度。`oriBins=16` 时顶层再把方向收成 16 档，并为每一档做一张响应图：像素上的值是该档与邻域（±3）场景方向余弦的最大值。极性开着用钳位到 0 的余弦，关掉用绝对值。这张图只建一次。  
3. 顶层：各角度旋转模型点；若开启尺度搜索，再按采样尺度放大点的位置（法向不变），滑窗打分（OpenMP，按角度下标拼回，不按线程完成顺序）。搜索边距是 2 像素，不再用模型半径，贴边残件的中心也能进粗搜。`oriBins=16` 时画面内部对每个模型点查一次响应图，不再在位姿循环里做 5×5。四边各约 16 个顶层像素仍用浮点余弦。步长是 4。角度步长是 `atan(1/R)`，搜索范围再宽也不乘 2。内部响应图隔一个角度查一次，贴边每个角度都查。内部高分格点收成质心，角度只平均离种子大约 4° 以内的；贴边浮点分不收质心。两边去重半径都是整幅的大约 0.45 倍模型跨度，从粗搜到末层都分开。内部每个已留种子在 8px 以外、整幅半径以内按 8 个方向、两段距离取样，这些点用完整模型的浮点分再打一次，每个种子只留最高的一个，全局最多 24 个，第一层位置窗口是 ±4 px。这些种子在中间层不被整幅半径删掉，末层再用浮点分去重。贴边不捞这圈。普通内部种子先在 ±4 px 里精修，已经过最终阈值就不再扩到 ±10 px。学模板（create）不建响应图；图在每次 find 的顶层上现算。  
4. 下层：在候选附近精搜。`oriBins=16` 里来自内部响应图的种子，第一层角度大约 ±12°，位置先按步长 2 搜 ±4 px，再在胜者周围补一像素；这一圈已经不低于最终阈值，或整圈都低于粗搜的早停线，就不再扩到 ±10 px。内部精修先用按方向排好的前一半点找位置，隔开的几个高分再用完整模型重打。贴边浮点种子第一层是 ±4 px、±3 个角度步长，逐点打分，和 `oriBins=0` 相同。更下面的层都是 ±4 px、±3 步。角度夹在 `±angle` 里，中心可以贴到画布边缘；尺度在采样间隔的一半内再试三个值。候选按原下标并行，再按原顺序拼回。每一层去重时贴边和内部仍然分开。环带捞回的内部种子在中间层保留，末层再用浮点分按整幅半径取舍。最细一层仍用浮点余弦。输出重叠抑制也按来源分开，内部框和贴边框重叠时留下贴边。画布外的点不计入分数和象限。闭合轮廓（圆、环，至少三个象限都有边缘点）按象限覆盖打折，弧状干扰不再靠半圈对齐过阈值。L 形这类开口轮廓仍用原来的平均分。  
5. 去重不看夹角。中心距小于较小边的 0.4 倍，或尺寸接近时较小框被盖住的比例超过 `iouThreshold`（`<=0` 时按 0.5，和 Halcon `MaxOverlap` 默认一样），只留高分的那个。尺寸差超过约 10% 且中心分开的两个实例都留，避免把 0.9 倍的邻点并进 1.0 倍的框。贴在高分命中旁边（中心距小于较小边的 0.65 倍）、自己又低于 0.78 且不到对方 0.74 倍的，当成旁瓣去掉。两个分数都在 0.78 以上的不走这条，同尺度圆环的中心距门槛仍是 0.4 倍较小边。`maxCount` 在这之后才截断，避免同一圆环的多个角度把名额占满。

`minScale=maxScale=1`（默认）时不进入尺度分支，只搜尺度 1。

## 大图先粗定位

整幅搜索很慢时，用 ORB 框 ROI 再在 ROI 上跑形状匹配。角度半宽不要收到 ±3°：大图样本上 ±3° 会漏，±5° 可以。完整参数在 [feature.md](feature.md)。

```python
param = cvter.CoarseRefineParam()
param.matcherType = cvter.MatcherType.SHAPE
param.angle = 5
param.margin = 120
param.scoreThreshold = 0.4
param.edgeMinMag = 12
param.stopLayer = 1      # 不要把 stopLayer=2 当加速
loc = cvter.match_coarse_to_fine(gray, templ, mask, param)
```

## 测试

```bash
make test-shape
make test-match
make test-scale    # 只编 cv_match，验证尺度≠1，不需要 TensorRT
make test-viz     # tests/test_shape_match_viz.py，含两个尺度画在同一张图上
                  # tests/test_mask_match_viz.py：mask 内才提边缘，输出在 tests/out_viz/mask/
```
