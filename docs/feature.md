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

## 粗到精定位

大图上整幅跑 PATTERN / SHAPE 很慢。`match_coarse_to_fine` 先用 ORB 或 SIFT 把模板四角投到场景上，按四角的轴对齐框加 `margin` 裁一块 ROI，clone 成连续图后再跑 Matcher，最后把命中点加回全图。

源码：`src/cv_match/coarse_refine.cpp`。测试：`make test-coarse`（也包含在 `make test-viz` 里）。

```python
param = cvter.CoarseRefineParam()
param.detector = cvter.FeatureDetector.ORB      # 或 SIFT
param.margin = 120                               # 四角外扩，像素
param.matcherType = cvter.MatcherType.PATTERN   # 或 SHAPE
param.angle = 0                                  # 只搜 0°，相信粗定位朝向
param.scoreThreshold = 0.5
param.maxCount = 1
param.minArea = 1024                             # 大模板可加大；小图用默认 256
param.stopLayer = 1                              # 0 更准。不要用 2 当加速
# param.matcherType = cvter.MatcherType.SHAPE
# param.angle = 5                                 # SHAPE 至少 ±5°，±3° 会漏
# param.scoreThreshold = 0.4

loc = cvter.match_coarse_to_fine(scene, templ, param=param)
# 有掩膜：match_coarse_to_fine(scene, templ, mask, param)
# 掩膜只参与精匹配，粗定位仍看整块模板。
if loc.found:
    hit = loc.hits[0]
    print(hit.Score, hit.Center.x, hit.Center.y, hit.Angle, hit.Scale)
    print(loc.roi.x, loc.roi.y, loc.roi.width, loc.roi.height)
    print(loc.coarse.inliers, loc.coarse.corners)   # 四角在全图上
```

`hits` 里的点在**输入 scene** 的坐标系，不是 ROI 局部坐标。`roi` 是实际送给 Matcher 的轴对齐矩形。`coarse` 就是一次 `match_features` 的结果。

C++：`template_matching::matchCoarseToFine(scene, templ, mask, param)`，声明在 `src/cv_match/coarse_refine.h`。

### 参数

粗定位：

| 字段 | 默认 | 含义 |
|---|---|---|
| `detector` | `ORB` | `FeatureDetector.ORB` 或 `SIFT` |
| `ratio` | 0.75 | 比值测试 |
| `ransacReproj` | 3 | 单应 RANSAC 重投影阈值 |
| `minInliers` | 4 | 内点少于此数不做精匹配。低于 4 按 4。`match_features` 的 `found` 要 6 个内点，这里 4～5 个也会精匹配 |
| `margin` | 120 | 四角轴对齐框向外扩的像素 |

精匹配字段与 `MatcherParam` 同名，含义见 [pattern_match.md](pattern_match.md) 和 [shape_match.md](shape_match.md)。两处默认不同：这里 `maxCount` 是 1，`angle` 是 0。

| 场景 | 建议 |
|---|---|
| 零件基本不转，要速度 | `PATTERN`，`angle=0`，`stopLayer=0` 或 `1` |
| 要边缘、允许大约 ±5° | `SHAPE`，`angle=5`。±3° 在大图样本上会漏 |
| `stopLayer=2` | 不要当成加速。大图上会锁到大约差半个分辨率的错误位姿。本接口不改 `stopLayer` 的含义 |

大图上量过的两组：ORB + ROI 上 `PATTERN` 且 `angle=0`，或 ORB + ROI 上 `SHAPE` 且 `angle=5`。可选计时脚本 `tests/bench_aoi_coarse_refine.py` 在没设置 `CVTER_AOI_SCENE` / `CVTER_AOI_TEMPL` 时直接跳过。

numpy 的 ROI 视图（`scene[y0:y1, x0:x1]`）不是 C 连续的。绑定现在按行步长拷贝，这种视图可以直接交给 `MatcherWrapper.match`。helper 自己裁 ROI 时仍会 `clone` 成连续图。以前忽略步长时，Matcher 看到错位像素，会静默返回 0 个命中。
