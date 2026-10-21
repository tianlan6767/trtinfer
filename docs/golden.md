# 金样比对（差分缺陷）

对齐两张图，做绝对差分，再阈值、形态学、连通域，把亮斑、暗斑、缺料收成缺陷。不走 TensorRT。灰度金字塔匹配见 [pattern_match.md](pattern_match.md)，只打分、不找缺陷见 [similarity.md](similarity.md)。

源码：`src/cv_golden/`。平移复用 `locate_zncc` / `locate_ncc`，模糊、形态学和 Blob 复用 `src/cv_blob/`，单应复用 `feature::matchHomography` 的 ORB 匹配。

## 坐标系

比对在**金样坐标系**里做。`defects` 的 `center` / `left` / `top` 都是金样上的像素。`transform` 是 3×3 `float64`，把金样上的点变到原始待测图：

```python
pt = cv2.perspectiveTransform(np.float32([[[d.center.x, d.center.y]]]), hit.transform)
```

`alignedImage` 是对齐后的待测图，和金样一样大。`diff`、`signedDiff`、`mask` 也是这个尺寸。

彩色图先转成灰度，只比一个通道。这是绝对灰度差，不做光照归一。光照整体漂了，要抬高 `diffThreshold`，或者先把 ROI 收小。

## Python

```python
import cvter

p = cvter.GoldenParam()
p.align = cvter.GoldenAlign.None          # Translate / Affine / Homography
p.polarity = cvter.GoldenPolarity.Both    # Brighter：待测更亮；Darker：待测更暗
p.diffThreshold = 30
p.blurSigma = 1.0                          # 只作用于差分，不改 alignedImage
p.minArea = 40
p.roiX, p.roiY, p.roiW, p.roiH = 20, 20, 280, 200   # 宽或高 <=0 表示整图

# mask 与金样同尺寸，非 0 才参与。不传就是整图。
hit = cvter.compare_golden(test, golden, p, mask)
print(hit.ok, hit.message, hit.alignMethod, hit.alignScore)
print(hit.meanAbs, hit.meanSigned, hit.maxAbs, hit.defectCount, hit.defectArea)

for d in hit.defects:                      # 面积从大到小
    print(d.center.x, d.center.y, d.area, d.meanAbs, d.meanSigned)
```

`ok == False` 时不会编造缺陷。常见 `message`：`empty image`、`size mismatch`、`locate failed`、`not enough features`、`affine failed`、`homography failed`、`transform rejected`、`mask size mismatch`、`empty compare region`。

`meanAbs` / `meanSigned` / `maxAbs` 是有效像素上的分数（金样 ROI ∩ 掩膜 ∩ 对齐覆盖到的地方），不是只在缺陷里算。极性只决定哪些像素能变成缺陷，不改变这三个分数。`meanSigned` 是「对齐后的待测 − 金样」，正数表示待测更亮。

## 对齐

| `GoldenAlign` | 行为 |
|---|---|
| `None` | 两图必须同尺寸，否则 `size mismatch` |
| `Translate` | 金样比待测小：整张金样做 `locate_zncc`（`useNcc=True` 时用 `locate_ncc`），再裁待测。同尺寸：四边裁掉 `alignMargin` 再定位，得到整像素平移，不插值。金样比待测大：失败 |
| `Affine` | ORB 比值测试后的点做 `estimateAffine2D`。内点不够时退回 `estimateAffinePartial2D`，`alignMethod` 为 `orb-affine-partial` |
| `Homography` | 复用 `match_features` 那条 ORB 单应。`alignMethod` 为 `orb-homography` |

平移的 `alignScore` 是 ZNCC 或 NCC。特征对齐的 `alignScore` 是内点 / 匹配数，`alignInliers` 是内点数。平移时 `alignInliers` 为 0。

仿射和单应会把待测重采样回金样。采样到待测外面的像素不参与比对。高对比边缘重采样会留下细碎假缺陷，用 `blurSigma`、`minArea`、开运算，或把 ROI 收到平坦区域里。

## 差分和缺陷

1. 可选高斯模糊（两张图用同一组 `blurKsize` / `blurSigma`）。偶数核会加 1。
2. 按极性阈值。`Both` 用绝对差分，`Brighter` / `Darker` 用有符号差分。
3. `enableMorph` 为真时做形态学，默认 3×3 开运算，去掉椒盐。细划痕会被开运算咬掉，这时把 `morphKsize` 设为 1，或 `enableMorph = False`。
4. `find_blobs`：面积、外接框、质心。`minArea`、`maxCount` 照 Blob 的规则。`maxCount <= 0` 表示不按业务截断。

`GoldenDefect.area` 是像素个数。`meanAbs` / `meanSigned` 是该连通域里的差分均值。

## 参数

| 字段 | 默认 | 含义 |
|---|---|---|
| `align` | `None` | 见上表 |
| `useNcc` | false | 平移时改用 NCC |
| `alignThreshold` | 0.6 | 平移分数下限 |
| `alignMargin` | 16 | 同尺寸平移的搜索边界 |
| `featureRatio` | 0.75 | ORB 比值测试 |
| `ransacReproj` | 3 | RANSAC 重投影阈值（像素） |
| `minInliers` | 6 | 仿射 / 单应至少这么多内点 |
| `polarity` | `Both` | 亮、暗或两者 |
| `diffThreshold` | 30 | 灰度差阈值 |
| `blurSigma` / `blurKsize` | 0 / 0 | 差分前的高斯。都很小则不模糊 |
| `roiX` `roiY` `roiW` `roiH` | 0 | 金样坐标系。宽或高 ≤0 为整图 |
| `enableMorph` | true | 阈值后做形态学 |
| `morphOp` / `morphKsize` / `morphIterations` | `Open` / 3 / 1 | 与 [blob.md](blob.md) 相同 |
| `minArea` / `maxCount` | 16 / 100 | 缺陷面积和最多条数 |
| `keepDiff` / `keepMask` / `keepAligned` | true | 关掉后对应图为空，缺陷和分数仍在 |

## C++

```cpp
#include "cv_golden/golden.h"

golden::GoldenParam param;
param.align = golden::AlignMode::Translate;
param.polarity = golden::Polarity::Both;
param.diffThreshold = 30;
param.minArea = 40;

golden::GoldenResult hit = golden::compare(test, golden, param, mask);
```

## 测试

夹具是合成的，不需要螺母图，也不需要 GPU。

```bash
make test-golden   # tests/test_golden.cpp，只编传统算法
make test-viz      # 含 tests/test_golden_viz.py，图片在 tests/out_viz/golden/
```

Linux 上 `make test-viz` 生成 `objs/pytrad/cvter.so`，不链 CUDA / TensorRT。

Windows 不用这份 Makefile。按 [env_windows.md](env_windows.md) 用 CMake 编工程时，把 `src/cv_golden/golden.cpp` 和 Makefile 里 `viz_srcs` 的其它传统算法源一起编成名为 `cvter` 的 pybind 模块（Linux 上就是 `objs/pytrad/cvter.so`），再跑 `tests/test_golden_viz.py`。不弹窗。只跑数值断言时，用与 `make test-golden` 相同的源文件编 `tests/test_golden.cpp`。
