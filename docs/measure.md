# 尺寸测量

在卡尺和已有几何辅助上补工业尺寸算子：点线圆、多边卡尺、拟合与形位公差、包围尺寸、像素到物理单位。不跑深度学习，也不做 Cp / 统计。

源码：几何仍在 `src/cv_caliper/metrology.*`，卡尺增强在 `src/cv_caliper/caliper.*`，拟合、形位、包围盒、量规和标定尺寸在 `src/cv_measure/`。pybind 导出到 `cvter`。

## 编译与测试

```bash
make test-measure          # C++ 数值，不链 CUDA / TensorRT
make test-viz              # 含 tests/test_measure_viz.py，图写到 tests/out_viz/measure/
```

`make test-viz` 编的是 `objs/pytrad/cvter.so`。Makefile 变量 `viz_srcs` 里要带上 `src/cv_measure/measure.cpp` 和 `src/cv_measure/bind.cpp`。Windows 用 CMake 时同样把这两份和 `viz_srcs` 里其余传统算法源编进名为 `cvter` 的模块。

## 1. 点 / 线 / 圆

已有的 `distance_points`、`distance_point_line`、`distance_lines`、`angle_lines`、`fit_line`、`summarize_rectangle`、`summarize_circle` 不变，说明见 [caliper.md](caliper.md)。下面是补上的。

| 函数 | 含义 |
|---|---|
| `segment_length(x0,y0,x1,y1)` | 线段长度 |
| `midpoint(...)` | 中点，返回 `Point2d` |
| `foot_of_perpendicular(px,py,x0,y0,x1,y1, clamp=False)` | 垂足。默认在无限直线上；`clamp=True` 限制在线段内。返回 `PointHit` |
| `intersect_lines(...)` | 无限直线交点，近平行时 `found=false` |
| `distance_parallel_lines(..., angle_tol_deg=5)` | 平行线法向距。夹角超过阈值返回 NaN |
| `distance_circle_centers(...)` | 圆心距 |
| `signed_distance_point_circle(...)` | 点到圆周，外部为正，内部为负 |
| `distance_point_circle(...)` | 上式的绝对值 |
| `signed_distance_circle_line(...)` | 圆心到直线的距离减半径。负值表示直线穿过圆 |
| `distance_circle_line(...)` | 直线到圆周的间隙，相交时为 0 |

```python
hit = cvter.intersect_lines(0, 0, 10, 0, 0, 0, 0, 10)
foot = cvter.foot_of_perpendicular(3, 5, 0, 0, 10, 0)
gap = cvter.distance_circle_line(0, 0, 10, 0, 12, 5, 12)  # 2
```

```cpp
auto foot = metrology::footOfPerpendicular(p, a, b, false);
double gap = metrology::distanceParallelLines(a1, a2, b1, b2, 5.0);
```

## 2. 卡尺增强

`CaliperSelect` 增加 `All` 和 `Nth`。`CaliperParam.edgeIndex` 从 0 开始，沿搜索方向数。

| 模式 | 行为 |
|---|---|
| `First` / `Last` / `Strongest` | 与原来相同，`find_line` / `find_circle` 返回一条 |
| `Nth` | 第 `edgeIndex` 个局部峰。峰比「第一个超过阈值的采样」更贴边 |
| `All` | 单线 / 单圆接口只取第 0 个峰。全部边用 `find_edges` |

`find_edges` / `find_circle_edges` **总是**按搜索方向返回每一条边，不看 `select`。极性相反的两条边（壁或槽）用 `Both`。靠得比大约 3 px 更近的峰会并成一条。

```python
param = cvter.CaliperParam()
param.polarity = cvter.CaliperPolarity.Both
param.contrast = 12
edges = cvter.find_edges(gray, x0, y0, x1, y1, True, param)
for edge in edges.edges:
    print(edge.index, edge.line.center, edge.line.found)

param.select = cvter.CaliperSelect.Nth
param.edgeIndex = 1
nth = cvter.find_line(gray, x0, y0, x1, y1, True, param)
```

定向和圆：`find_edges_oriented`、`find_circle_edges`。角度约定与 `find_line_oriented` / `find_circle` 相同。

### 双边厚度

搜索窗要同时盖住两条边。`width` 是两条拟合边的间距。

```python
wall = cvter.measure_dual_width(gray, cx, cy, phi_deg=90, length1=55, length2=45, param=param)
# 环：radius 放在内外边之间，search 盖住两边
ring = cvter.measure_radial_width(gray, cx, cy, radius=58, search=22, param=param)
```

`radial=true` 时读 `circle0` / `circle1`，`width = |r1-r0|`。直线厚度读 `edge0` / `edge1`。

### 弧段

`find_circle` 本来就能只扫一段角。`find_arc` 在同一套采样上额外给出测得张角、弧长和弦：

| 字段 | 含义 |
|---|---|
| `startDeg` / `endDeg` | 内点张角，度，y 向下 |
| `spanDeg` | 张角 |
| `arcLength` | 弧长 |
| `chord` / `chordStart` / `chordEnd` | 弦 |

请求接近整圆（≥350°）时 `spanDeg=360`，弦长为 0。

### 开口

两条边对向放置，向开口内侧搜。`phi_deg` 沿边缘，`half_gap` 是预期半宽，卡尺中心落在预期边上。亮边、暗缝、向内搜用 `LightToDark`。

```python
param.polarity = cvter.CaliperPolarity.LightToDark
gap = cvter.measure_gap(gray, cx, cy, phi_deg=90, length1=55, half_gap=30, search=16, param=param)
```

已有两条 `LineResult` 时用 `gap_between(first, second)`。夹角超过 12° 时 `parallel=false`，间距改用两端中点的平均距离。

## 3. 拟合与形位

`FitParam.method` 为 `FitMethod.LeastSquares` 或 `Ransac`。RANSAC 默认阈值 1.5 px、100 次、种子 1，并要求内点大约不少于点数的 30%。

| 函数 | 结果 |
|---|---|
| `fit_line(points, param=FitParam())` | `LineFit`。最小二乘仍是 SVD |
| `fit_circle(points, param)` | `CircleFit` |
| `fit_ellipse(points, param)` | `EllipseFit`。`a`/`b` 是半轴，`a>=b`，`angle` 是长轴角度。OpenCV 拟合失败时 `found=false` |
| `fit_parallelogram(points)` | `QuadFit`。四角接近直角且对边平行时 `rectangle=true` |
| `fit_parallelogram_contour(image)` | 先取最大外轮廓再拟合 |

点可以是 `Point2d` 列表，或 `(N,2)` 数组。轮廓点用边界，不要把填充区域里的全部像素送进去。

形位公差的 `FormResult.ok` 表示在带宽内。`value` 是测得偏差，`margin = tolerance - value`。

| 函数 | `value` |
|---|---|
| `check_parallelism(..., tol_deg)` | 夹角，0 为平行 |
| `check_perpendicularity(..., tol_deg)` | 相对 90° 的偏差 |
| `check_straightness(points, tol_px)` | 点到拟合直线的峰谷带宽度 |
| `check_roundness(points, tol_px)` | 拟合圆半径的峰谷差 |

```cpp
measure::FitParam param;
param.method = measure::FitMethod::Ransac;
auto line = measure::fitLine(points, param);
auto form = measure::checkStraightness(points, 0.2);
```

## 4. 组合尺寸

| 函数 | 含义 |
|---|---|
| `size_aabb(points)` | 轴对齐外接矩形，`angle=0` |
| `size_min_area(points)` | 最小面积旋转矩形。`width` 取较长边 |
| `size_min_area_contour(image)` | 对最大外轮廓做最小面积矩形 |
| `circle_circumscribed(points)` | 最小外接圆 |
| `circle_inscribed(points)` | 凸包的最大内切圆，光栅距离变换，直径大约有 1～2 px 偏差 |
| `circle_inscribed_mask(image)` | 按掩膜内部（保留凹陷），不是凸包 |
| `width_function(points, step_deg=1)` | 旋转卡尺。`minWidth` 是最小平行支撑距，`maxWidth` 是最大平行支撑距。矩形的最小值是短边，最大值是对角线 |
| `hole_to_edge(cx,cy,r,x0,y0,x1,y1)` | 孔边到直线的间隙：圆心距减半径。负值表示圆越过直线 |
| `hole_to_hole(x0,y0,r0,x1,y1,r1)` | 两孔边缘间隙：圆心距减两个半径。`centerDistance` 是圆心距 |
| `edge_to_edge(..., angle_tol_deg=15)` | 两边间距。不平行时 `found=false` |

`GageResult.a` / `b` 是读数两端，方便叠加。

## 5. 标定尺寸

比例尺：

| 函数 | 来源 |
|---|---|
| `scale_mm_per_px(v)` | 各向同性 mm/px |
| `scale_mm_per_px_xy(sx, sy)` | 分开的 X/Y |
| `scale_from_camera(K, z_mm)` | 正对平面，`mm/px = Z / f` |
| `scale_from_yaml(path, z_mm=0)` | 文件里有 `mmPerPx` 或 `mmPerPxX/Y` 时直接用；否则走标定 YAML 的 `cameraMatrix`，这时 `z_mm` 必须大于 0 |

`cameraMatrix` 与 [calib.md](calib.md) 的 `save_calibration` 是同一份字段，读的时候调用 `calib::loadCalibration`。单位跟着 `Z` 和 `mmPerPx` 走，字段名按毫米来写。

夹具 `FixtureFrame`：`origin` 是图像像素，`angleDeg` 是夹具 +X 相对图像 +X 的角度，y 向下。先把像素乘到平面上，再绕原点旋转进夹具。

```python
scale = cvter.scale_mm_per_px(0.05)
frame = cvter.FixtureFrame()
frame.origin = cvter.Point2d(100, 80)
frame.angleDeg = 90
dim = cvter.dimension_points(100, 80, 100, 180, scale, frame)
# dim.pixels 是像素，dim.physical 是物理长度，dim.fromFrame / toFrame 在夹具里
```

还有 `dimension_point_line`、`dimension_lines`，以及 `scale_gage(gage, scale, frame)` 把量规读数换到同一坐标系。`scaled=false` 时 `physical` 等于像素。

Cp、Cpk 和批量统计不在这里。
