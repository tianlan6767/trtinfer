# 预处理与 Blob

灰度匹配见 [pattern_match.md](pattern_match.md)，卡尺见 [caliper.md](caliper.md)。本模块是 CPU 上的经典图像工具，不走 TensorRT。

源码：`src/cv_blob/`。不依赖 CUDA 的测试：`make test-blob`。完整 `cvter.so` 编出来之后，同样的用例在 `tests/test_blob.py`。

## Python

```python
import cvter

roi = cvter.crop_roi(gray, x, y, w, h)          # 越界会裁进图内
blur = cvter.gaussian_blur(roi, ksize=5, sigma=1.0)
blur = cvter.median_blur(blur, ksize=3)

th = cvter.ThresholdParam()
th.type = cvter.ThresholdType.Fixed             # 或 Otsu
th.value = 120
th.invert = False                               # True：暗目标变前景
binary = cvter.threshold(blur, th)

morph = cvter.MorphParam()
morph.op = cvter.MorphOp.Open                   # Erode / Dilate / Open / Close
morph.ksize = 3
morph.iterations = 1
binary = cvter.morphology(binary, morph)

filt = cvter.BlobFilter()
filt.minArea = 40
filt.maxArea = 1e7
filt.minCircularity = 0.7                      # 4π·面积 / 周长²，圆约等于 1
filt.maxAspect = 2.0                           # max(宽,高) / min(宽,高)，正方形为 1
filt.maxCount = 20
blobs = cvter.find_blobs(binary, filt)         # 也可传 roi_x, roi_y, roi_w, roi_h
for b in blobs:
    print(b.center.x, b.center.y, b.area, b.circularity, b.aspect)

stats = cvter.contour_stats(binary)            # 外轮廓，按面积从大到小
```

`find_blobs` 里 `roi_w <= 0` 或 `roi_h <= 0` 表示整图。给出 ROI 时，`center` / `left` / `top` 仍是原图坐标。

## 参数

| 对象 | 字段 | 默认 | 含义 |
|---|---|---|---|
| `ThresholdParam` | `type` | `Fixed` | `Fixed` 用 `value`；`Otsu` 用直方图 |
| | `value` | 128 | 固定阈值 |
| | `maxValue` | 255 | 前景灰度 |
| | `invert` | false | 反相，暗目标变成 255 |
| `MorphParam` | `op` | `Open` | 腐蚀 / 膨胀 / 开 / 闭 |
| | `ksize` | 3 | 方形核，偶数会加 1 |
| | `iterations` | 1 | 重复次数 |
| `BlobFilter` | `minArea` / `maxArea` | 1 / 很大 | 像素个数 |
| | `minCircularity` / `maxCircularity` | 0 / 10 | 外轮廓圆度 |
| | `minAspect` / `maxAspect` | 0 / 很大 | 外接矩形长短边比 |
| | `maxCount` | 500 | 按面积从大到小截断；`<=0` 表示不按业务截断 |

非 0 像素是前景。彩色图在阈值和 Blob 前会转成灰度。连通域是 8 连通。

`Blob.area` 是像素个数。`contour_stats` 的 `area` 是轮廓折线内部面积，实心矩形会比像素个数略小（大约 `(宽-1)×(高-1)`）。圆度都只用外轮廓，孔洞不算进周长。

## C++

```cpp
#include "cv_blob/blob.h"

cv::Mat binary = blob::threshold(gray, {});
blob::MorphParam morph;
morph.op = blob::MorphOp::Open;
binary = blob::morphology(binary, morph);

blob::BlobFilter filter;
filter.minArea = 40;
filter.minCircularity = 0.7;
std::vector<blob::Blob> blobs = blob::findBlobs(binary, filter);
```

## 测试

```bash
make test-blob
```
