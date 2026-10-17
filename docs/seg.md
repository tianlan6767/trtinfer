# 分水岭与常规分割工具

Blob 连通域见 [blob.md](blob.md)。这里补距离变换、Canny、轮廓和分水岭，用来切开粘在一起的目标。不走 TensorRT。

源码：`src/cv_seg/`。可视化：`tests/test_seg_viz.py`（`make test-viz`）。

## Python

```python
import cvter

dist = cvter.distance_transform(binary)          # CV_32F，非 0 为前景，L2
edges = cvter.canny(gray, low=50, high=150, aperture=3)
contours = cvter.extract_contours(binary, external_only=True)
for c in contours:
    print(c.area, c.center, c.points.shape)      # points：Nx2

# 距离变换找峰，再分水岭。regions 不含边界和背景。
split = cvter.split_blobs(binary, min_distance=8, dist_ratio=0.4)
print(split.labels.dtype, len(split.regions))

# 自己的种子：CV_32S，0 未知，>0 是种子编号
labels = cvter.watershed(image, markers)
```

`split_blobs` 的 `min_distance` 是两个峰之间的最小像素距离。`dist_ratio` 要求峰至少达到最大距离的这个比例，避免把平缓区域拆碎。

`WatershedResult.labels` 是 `CV_32S`，分水岭边界为 -1。`regions` 跳过标签 ≤1 和边界。`extract_contours` 的点是 `CV_64F` 的 `Nx2`。
