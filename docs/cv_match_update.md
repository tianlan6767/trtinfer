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

结果 `MatchResult` 字段不变。

## 5. 明确不做（后续）

- **顶层改转模板**（NCC）：模板一转包围盒变大，必须配合 mask，坐标和 `GetBestRotationSize` 整条链要重验，本轮不动。  
- **尺度搜索**：当前金字塔只加速，不搜尺度。  
- **用 NCC 读近圆微小转角**：应用层继续定位 + 卡尺测边；转角优先用 `SHAPE` 或卡尺。

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
