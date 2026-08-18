# 卡尺测量（Caliper）

沿指定方向做 1D 灰度剖面，按极性取亚像素边缘点，再拟合直线或圆。适合 ROI 内方向已知的轮廓边（竖边、定位孔），比 Edge Drawing 更稳。

源码：`src/cv_caliper/`，通过 pybind11 导出到 `cvter`（`workspace/cvter.so`）。

## 编译与测试

```bash
make -j
make test-caliper
```

测试入口：`tests/test_caliper.py`（合成图必跑；螺母实图存在则加跑）。

批量画线示例：

```bash
cd workspace
python3.12 ed_roi_lines.py --method caliper --contrast 18 --stride 2
python3.12 ed_roi_lines.py --method both --limit 20
```

## Python API

```python
import cvter

param = cvter.CaliperParam()
param.polarity = cvter.CaliperPolarity.LightToDark
param.select = cvter.CaliperSelect.Strongest
param.contrast = 18
param.stride = 2
param.projection = 5
param.sigma = 0.8
param.outlierRatio = 0.3
param.minPoints = 8
param.radialInward = False

cal = cvter.CaliperWrapper(param)

# 轴对齐矩形。search_horizontal=True：沿 X 搜竖边
line = cal.find_line(gray, x0, y0, x1, y1, search_horizontal=True)

# 旋转矩形。phi_deg 是长度轴相对 +X 的角度，搜索方向为长度轴顺时针 90°
# phi_deg=90 时沿 +Y 布卡尺、沿 +X 搜索
line = cal.find_line_oriented(gray, cx, cy, phi_deg=90, length1=80, length2=20)

# 环形卡尺拟合圆。radius 为预期半径，search 为径向半宽（像素）
param.polarity = cvter.CaliperPolarity.DarkToLight  # 黑洞由内向外
circle = cal.find_circle(gray, cx, cy, radius=70, search=20, start_deg=0, end_deg=360)
```

也可以不建 Wrapper，直接调用模块函数（需传入 `param`）：

```python
line = cvter.find_line(gray, x0, y0, x1, y1, True, param)
circle = cvter.find_circle(gray, cx, cy, 70, 20, 0, 360, param)
```

### LineResult

| 字段 | 含义 |
|---|---|
| `found` | 是否拟合成功 |
| `p1`, `p2` | 线段端点（`p2.y >= p1.y`） |
| `center` | 中点 |
| `angle` | 方向角（度），竖直边约 90 |
| `rms` | 点到直线 RMS（像素） |
| `length` | 端点距离 |
| `numPoints` / `numInliers` | 边缘点数 / 参与拟合的点数 |
| `points` / `inliers` | 全部边缘点 / 内点 |

### CircleResult

| 字段 | 含义 |
|---|---|
| `found` | 是否拟合成功 |
| `center` | 圆心 |
| `radius` | 半径 |
| `rms` | 径向残差 RMS |
| `numPoints` / `numInliers` | 边缘点数 / 内点数 |
| `points` / `inliers` | 点集 |

## 几何参数（调用参数，不在 CaliperParam 里）

**直线矩形**

- 框要盖住目标边，搜索方向上留 10～20 px。太窄会漏，太宽可能抓到旁边另一条边。
- `search_horizontal=True`：沿水平找竖边（搜索方向左→右）。
- `False`：沿竖直找横边（搜索方向下→上）。上暗下亮用 `LightToDark`，上亮下暗用 `DarkToLight`。
- 斜边用 `find_line_oriented`：`length1` 为沿边半长，`length2` 为搜索半宽。

**圆**

- `cx, cy, radius` 给大概值即可，偏差几个像素一般能收回来。
- `search` 是径向半宽。孔半径约 70 时用 15～25；太大容易扫到外轮廓。
- `start_deg / end_deg` 只扫一段弧，例如右半圆：`start_deg=-90, end_deg=90`。
- 拟合半径相对预期值偏离过大（超过约 `2*search + 0.25*radius`）会判失败。

## CaliperParam

| 参数 | 默认 | 含义 | 怎么调 |
|---|---|---|---|
| `polarity` | `LightToDark` | 沿搜索方向的明暗 | **必须和真实边缘一致**。零件→黑背景（左亮右暗）用 `LightToDark`。黑洞由内向外用 `DarkToLight`。极性反了会 `found=false` 或贴到另一条边。不确定且框内只有一条边时可用 `Both`。 |
| `select` | `Strongest` | 一条剖面上取哪条边 | 框里只有一条目标边：保持 `Strongest`。有内外两条边：先碰到的用 `First`，后碰到的用 `Last`。 |
| `contrast` | 18 | 灰度落差阈值 | 漏检、边很糊：降到 8～12。抓到拉丝/噪声：升到 25～40。`numPoints` 经常为 0 说明太高。 |
| `stride` | 2 | 卡尺间距（px） | 更密更稳更慢：1。更快：3～4。`numCalipers=0` 时条数 ≈ 边长/`stride`。 |
| `numCalipers` | 0（自动） | 强制条数 | 一般不用改。圆想固定 36 条就设 36。 |
| `projection` | 5 | 每条卡尺横向平均的像素 | 拉丝纹理重：提到 7～11。边很短或要贴圆角：降到 1～3。 |
| `sigma` | 0.8 | 1D 剖面平滑 | 噪声大：1.2～1.5。要更锐的亚像素：0.4～0.6。 |
| `outlierRatio` | 0.3 | 拟合时丢掉最差点的比例 | 框扫到圆角/毛刺：0.3～0.4。边很干净、点太少：降到 0.1。 |
| `minPoints` | 8 | 最少有效边缘点 | 短边可降到 5；要更严就升到 12。 |
| `radialInward` | `False` | 圆的搜索方向 | `False`：由内向外。`True`：由外向内。极性仍按搜索方向解释。 |

## 本仓库螺母图推荐值

两个 JSON 竖框（零件亮、背景暗）：

```python
param.polarity = cvter.CaliperPolarity.LightToDark
param.select = cvter.CaliperSelect.Strongest
param.contrast = 18
param.stride = 2
```

定位通孔（内部黑洞，由内向外）：

```python
param.polarity = cvter.CaliperPolarity.DarkToLight
param.radialInward = False
param.contrast = 12
cal.find_circle(gray, cx, cy, radius=70, search=20)
```

`ed_roi_lines.py` 命令行可调：`--contrast 18 --stride 2`。

## 看结果判断

- `found=False` 或 `numPoints` 很小：极性反了，或 `contrast` 太高，或框没盖住边。
- `rms` 大（>0.5 px）：框里混进圆角/第二条边，加大 `outlierRatio` 或把框收窄。
- 线偏到内部另一条边：改 `select=First/Last`，或收窄框，不要用 `Both`。
- 圆半径飞掉：减小 `search`，圆心先给准一点。

叠加图上的绿点是参与拟合的 inlier。

## 耗时（440×700 灰度，含 Python 绑定）

| 调用 | 约 |
|---|---|
| `find_line` 下框 37×180 | 0.7 ms |
| `find_line` 上框 61×184 | 0.95 ms |
| 两框合计 | 1.6 ms |
| `find_circle` r=73、径向半宽 20 | 1.3 ms |

`nut_偏移6` 600 张：纯两框卡尺约 3 ms/张；读图+卡尺+画线存 jpg 约 5 ms/张。

## C++ API

```cpp
#include "cv_caliper/caliper.h"

caliper::CaliperParam param;
param.polarity = caliper::Polarity::LightToDark;
auto line = caliper::findLineRect(gray, x0, y0, x1, y1, true, param);

param.polarity = caliper::Polarity::DarkToLight;
auto circle = caliper::findCircle(gray, {cx, cy}, 70.0, 20.0, 0.0, 360.0, param);
```

`findLineOriented` 的 `phi` 是**弧度**；Python `find_line_oriented` 的 `phi_deg` 是**角度**。
