# 灰度模板匹配技术更新（cv_match）

日期：2026-08-19  
范围：`src/cv_match/`、`src/interface.cpp`（Python 绑定）、`tests/test_pattern_match.py`  
状态：按本文「本次改动」落地，后续项不在本轮

## 1. 现状

`PatternMatcher` 是带旋转搜索的金字塔 **NCC**（`TM_CCOEFF_NORMED` 等价实现）：

1. 按 `minArea` 建模板金字塔并预计算均值/范数  
2. 顶层把**整张搜索图**转到每个候选角，做相关 + 积分图归一化  
3. 下层只裁 ROI 再转，角度在粗结果附近取 5 个  
4. 底层二次曲面亚像素（x/y/θ）  
5. 分数阈值 + 旋转矩形 IoU 去重  

分数用的是模板内**全部像素**，不是轮廓。近圆目标（螺母扁位）转一点 NCC 几乎不变，角度不可靠；那是用法问题，应用侧用卡尺测边，不在本轮改算法族。

## 2. 主要问题

| 问题 | 影响 |
|---|---|
| 顶层对每个角度 `warpAffine` 整图，角度循环串行 | `angle>0` 时最慢 |
| 精搜调用手写 SIMD 相关（`bUseSIMD=true`） | 无分块/FFT，且按 `cols` 而非 `step` 跨行，大图可能错、常比 OpenCV 慢 |
| 旋转填充色非黑即白（模板均值 &lt;128 用 255） | 贴边时污染 NCC，扁位角容易漂 |
| 无模板 mask | 孔洞、拉丝、背景与目标同权，近圆时角度更差 |
| `stopLayer` 写死为 0 | 不能按场景牺牲一层精度换速度 |
| 下层 `if (angle)` / `else` 两支都在扩 ±2 步 | 死代码，0° 时仍可能走多余角度 |

## 3. 本次改动（已实现范围）

目标：兼容现有 `setTemplate(img)` / `match` / 结果四边形约定，只加可选能力。

### 3.1 旋转填充改为模板均值

`warpAffine` 的 `BORDER_CONSTANT` 使用模板灰度均值，而不是 0/255。贴边、ROI 平移后的空白区对 NCC 的拉偏会小很多。

`MatcherParam.meanBorder`（默认 `true`）可关掉，恢复旧的黑/白填充。

### 3.2 顶层角度循环 OpenMP

每个角度独立：转图 → NCC → 取峰。线程局部收集候选再合并，结果集合与串行相同（排序后）。`angle=0` 只有 1 个角，几乎无差别。

### 3.3 精搜改回 OpenCV 相关

下层 `MatchTemplate(..., true)` 改为走 OpenCV `TM_CCORR` + 原积分图 CCOEFF。OpenCV 内部有 SIMD/FFT，并正确处理 stride。手写卷积保留代码但默认不再走。

### 3.4 可选模板 mask

```python
matcher.setTemplate(templ)                 # 旧行为
matcher.setTemplate(templ, mask)           # mask: uint8, 非 0 为有效像素
```

C++：`setTemplate(templ, mask=cv::Mat())`。

实现：有效像素上做 masked NCC（滑动窗 `sum(I·Tm)` / `sqrt(var_I_mask)·||Tm||`，`Tm=(T-meanT)·mask`）。孔、内部纹理、无关背景可涂 0，扁位外沿权重大，有利于定位和角度。

无 mask 时仍走原来的积分图 CCOEFF，分数尺度与旧版一致。

### 3.5 暴露 `stopLayer`

`MatcherParam.stopLayer`：`0` 精搜到原图层（默认，与旧版一致）；`1` 停在金字塔第 1 层（更快、略粗）。

### 3.7 `Matcher` 析构改为虚函数

`GetMatcher` 返回 `unique_ptr<Matcher>` 实际指向 `PatternMatcher`。原先析构不是虚函数，子类里的金字塔/`Mat` 不会释放，进程退出时可能堆损坏。已改为 `virtual ~Matcher()`。

`angle < 1e-7` 时每层只匹配 0°，不再生成 ±2 个无效角。

## 4. API

`MatcherParam` 新增（均有默认，旧脚本不用改）：

| 字段 | 默认 | 含义 |
|---|---|---|
| `meanBorder` | `true` | 旋转填充用模板均值 |
| `stopLayer` | `0` | 精搜停止层 |

`setTemplate` 增加可选第二参 `mask`。

结果 `MatchResult` 增加 `Scale`（相对模板的采样尺度，默认匹配为 1）。

### 4.1 尺度搜索（PATTERN 和 SHAPE 都有）

`MatcherParam.minScale` / `maxScale` / `scaleStep` / `numScales` 对灰度 NCC 和形状匹配**都生效**。默认 `minScale=maxScale=1`，只搜原尺度。

- `PATTERN`：每个尺度缩放模板和 mask，再走原来的金字塔 + 角度 NCC。
- `SHAPE`：按该尺度放大边缘点（法向不变），下层在采样间隔的一半内再试三个尺度。

`MatchResult.Scale` 是命中的采样尺度，四角按这个尺寸给出。说明见 [pattern_match.md](pattern_match.md)、[shape_match.md](shape_match.md)。`make test-scale` 不依赖 TensorRT。

## 5. 明确不做（后续）

- **顶层改转模板**（NCC）：模板一转包围盒变大，必须配合 mask，坐标和 `GetBestRotationSize` 整条链要重验，本轮不动。  
- **用 NCC 读近圆微小转角**：应用层继续定位 + 卡尺测边；转角优先用 `SHAPE` 或卡尺。

尺度搜索已经落地，不在这份清单里。

形状匹配已落地，见 [docs/shape_match.md](docs/shape_match.md)。

## 6. 风险与验收

- 无 mask、默认参数：与旧版同一条 NCC，分数应接近；允许因均值填充导致贴边结果更好/略变。  
- 合成图：亮块在已知位置，0° 中心误差 &lt; 2 px，分数 &gt; 0.9。  
- 带 `angle`：目标旋转约 8°，检出且角度误差 &lt; 2°。  
- mask：中心加噪声，有 mask（只留外框）仍应找到；测试见 `tests/test_pattern_match.py`。  
- 回归：`make test-caliper` 仍应通过。

```bash
make -j
make test-match
make test-shape
make test-caliper
make test-scale
```

## 7. 螺母扁位建议用法

```python
# 模板：扁位朝上的金样
# mask：保留外沿与扁位，挖掉通孔和中心拉丝（255=用，0=忽略）
param.angle = 8
param.minArea = 256
param.meanBorder = True
matcher.setTemplate(templ, mask)
```

角度仍建议用卡尺复核；NCC 负责把 ROI 放到螺母附近。

## 8. 多实例 + 尺度，向 Halcon Matching 例图靠拢（2026-10-21）

对照是 Halcon 官方 Matching 例图（`green_dots`、`rings_02/03/04`、`board_01`，以及已经对齐的 `face_mask` / `cap_exposure`）。这些图是 MVTec 版权，**不放进仓库**，CI 里用合成图复现同样的失败模式，见 `tests/test_scale_match.cpp` 里 `testGreenDotsMultiScale` / `testRingsNoAngleDuplicates` / `testRingClutterNotDuplicated` / `testBoardPoseUnchanged` / `testPatternBrightnessLikeExposure`。`board_02` 在 ±10° 下 Halcon 和这里都会漏，不在这次范围里。

改之前，在合成的对应场景上：

| 场景 | 改之前 | 改之后 |
|---|---|---|
| 三个不同尺度的圆点，角度 ±12°，`maxCount=8` | PATTERN 7～8（3 个真目标 + 错尺度错位影子，角度能到 ±14° 以外）；SHAPE 8（同一中心多个角度） | 两边都是 3，每个中心一个，角度留在搜索范围内 |
| SHAPE `maxCount=3` | 角度重复占掉名额，最弱的那个圆点被挤掉 | 3 个中心都在 |
| 4 个圆环，SHAPE 角度 ±15° 且带一点尺度 | 16 个高分重复 | 4 |
| 1 个圆环 + 两段弧 | SHAPE 9（真圆环的多个角度 + 弧上的误检） | 1，落在真圆环上 |
| 单个非对称板、整体加亮的 NCC | 本来就对 | 仍是 1 个命中 |

做法：

- **SHAPE** 去重不再要求夹角也接近。圆和环在一串角度上分数几乎一样，以前 `d < minDist && da < angTol` 会把它们都留下。
- 跨尺度合并（PATTERN）和 SHAPE 最终输出走同一个 `suppressMatchOverlap`：中心距 < 0.4×较小边，或较小框被盖住的比例 > `iouThreshold`（`<=0` 时按 0.5）。单尺度 PATTERN 原来的 `FilterWithRotatedRect` 没改，避免动到不做尺度搜索的 NCC。
- 尺度搜索时每个尺度先按更大的候选额度输出，合并去重后再截到 `maxCount`。
- PATTERN 的角度网格改成含端点的 `±angle`。亚像素角度甩出范围就丢掉，只用位置修正。
- SHAPE 在最细一层对「至少三个象限都有边缘点」的闭合轮廓做象限覆盖打折。半圈弧不再靠平均分过线。开口轮廓（L 形）不打折。

`make test-scale` 不需要 TensorRT。Windows 上按 [env_windows.md](env_windows.md) 用 CMake 重编 `cvter` 后，再跑原来的 Halcon 对照脚本。没有单独的图像路径要配：官方例图不在 CI 里。

## 9. PATTERN 圆环过检，以及被大框边距吞掉的邻点（2026-10-21）

第 8 节之后，Windows 上用同一套官方例图再对过：SHAPE 的 `rings_03` 5→1、`rings_04` 9→3 与 Halcon 一致，`rings_02` 10→3（Halcon 4）。PATTERN 的圆环仍是 5/4/6（Halcon 4/1/3），`green_dots` 的 PATTERN 仍是 2/3，SHAPE 是 4（Halcon 3）。

合成图上对得上的原因：

- 远处的椭圆弧，全局 NCC 大约 0.51–0.59，能过 0.5，但把命中框拧回模板后，大约一半象限几乎不相关（整圈则四个象限都在 0.8 以上）。不抬全局阈值，否则 0.6 这一档的弱目标和已经对齐的 NCC 会一起受影响。
- 模板大部分是空白边、对比只在中心时，旁边较小的实例中心已经分开，但它的框有一半以上落在大框里。旧逻辑只要覆盖比例超过 `iouThreshold` 就把低分的那个删掉，较小的真目标会消失。同尺寸、中心靠近的重叠（圆环的重复锁）仍合并。中心距门槛保持 0.4 倍较小边，避免把已经分开的同尺度圆环再并掉。
- SHAPE 只额外去掉「尺度比 > 1.08、中心仍落在 0.5 倍较小边内、分数低于更强命中的 0.72 倍」的旁瓣。同尺度不走这条。

`tests/test_scale_match.cpp` 增加 `testPatternRingArcRejected`、`testPatternKeepsNeighborInMargin`、`testPatternOneQuadrantOcclusion`，四个整圈在 ±20° 下仍是 4。`make test-scale` 仍不需要 TensorRT。Windows 上再按 [env_windows.md](env_windows.md) 用 CMake 重编后，跑原来的 Halcon 对照；官方图仍不进仓库。

## 10. 弱的第三颗点，以及还剩的一个圆环误检（2026-10-21）

第 9 节之后再对官方例图：PATTERN 的 `rings_02` 收到 4，与 Halcon 一致；`rings_03` 4→2（Halcon 1），`rings_04` 6→4（Halcon 3），各还多 1。`green_dots` 的 PATTERN 仍是 2/3，SHAPE 仍是 4（Halcon 3）。SHAPE 的 `rings_02` 仍是 3（Halcon 4），这是第 8 节把 10 个重复收到 3 个时就少的那一个，不是第 9 节退回去的。`board_01`、`face_mask`、`cap_exposure` 的 PATTERN 不变。`cap_exposure_01` 的 SHAPE 本来就是 0（那组例图是 NCC，不是形状模型）。

合成图上对得上的两件事：

- 缺一口的圆环（大约 255°–300°）全局 NCC 可以到 0.81–0.90。最差象限大约 0.39–0.60，达不到「一半象限低于 0.35」，所以第 9 节的规则留着它。整圈的四个象限差距很小。只挡掉一个象限时，另外三个仍然在 0.94 以上。
- 模板大部分是空白边时，把较小的圆点放在和模板背景不一样的噪声上，全局 NCC 大约 0.49，过不了 0.5。非背景像素的 NCC 仍然接近 1，而且四个象限都对齐。Halcon 形状分大约 0.61 指的是轮廓对上多少，不看这块空白边。

做法：

- 象限差距大的命中，不管全局分有没有到 0.85，都丢掉。只挡一个象限、整圈、整体加亮不丢。
- 全局分在阈值的 0.82 倍到阈值之间、且非背景像素在至少三个象限都对上时，把这条命中补回来，输出分用非背景像素的 NCC。调用方自己传了 mask 时不走这条，原来的 masked NCC 不变。
- 边长比超过约 1.10 的两个框，不再用覆盖比例互相吞。0.9 倍贴在 1.0 倍旁边的完整实例可以留下。同一中心仍按 0.4 倍较小边合并。
- SHAPE 去掉贴在高分命中旁边的低分锁：中心距小于较小边的 0.65 倍，且分数低于 0.78、又不到对方的 0.74 倍。两个高分圆环不进这条。`rings_02` 少的那一个不对着这条放宽，避免把已经对齐的 `rings_03/04` 又放回去。

`make test-scale` 增加 `testPatternWeakThirdOnNoise`、`testPatternAlmostRingRejected`、`testShapeDotSidelobeDropped`。Windows 上再按 [env_windows.md](env_windows.md) 用 CMake 重编后跑原来的 Halcon 对照。希望 PATTERN 的 `green_dots` 变成 3，`rings_03/04` 再各少 1；SHAPE 的 `green_dots` 变成 3。SHAPE 的 `rings_03/04` 保持 1 和 3，`rings_02` 仍可能是 3。PATTERN 的 `board_01` / `face_mask` / `cap_exposure` 个数和分数应不变。
