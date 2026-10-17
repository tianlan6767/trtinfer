# ORB / SIFT 特征匹配

模板灰度相关见 [pattern_match.md](pattern_match.md)。这里是稀疏特征：检测、描述、比值测试、RANSAC 单应，把模板四角投到场景上。

源码：`src/cv_feature/`。可视化：`tests/test_feature_viz.py`。

## Python

```python
import cvter

print(cvter.sift_available())
hit = cvter.match_features(
    scene, model,
    cvter.FeatureDetector.ORB,   # 或 FeatureDetector.SIFT
    ratio=0.75,
    ransac_reproj=3.0,
)
print(hit.method, hit.matches, hit.inliers)
print(hit.homography)   # 3x3
print(hit.corners)      # 4x2，模板四角在场景中的位置
```

SIFT 在当前 OpenCV 没有非自由模块时，`sift_available()` 为 false，`match_features(..., SIFT)` 的 `found` 为 false，`method` 为 `SIFT-unavailable`。不要用 `Algorithm::empty()` 判断 SIFT：未重写的检测器会返回 true。

`modelPoints` / `scenePoints` 是比值测试之后的点。`inlierMask` 与这些点对齐，非 0 为内点。内点少于 4 个时不算成功。
