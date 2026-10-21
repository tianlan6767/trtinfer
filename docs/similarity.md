# 对齐相似度与平移定位

灰度金字塔匹配（旋转 + 尺度）见 [pattern_match.md](pattern_match.md)。本模块是另一条更瘦的接口，不建金字塔，不搜角度，不搜尺度。

源码：`src/cv_sim/`。可视化：`make test-viz` 里的 `tests/test_similarity_viz.py`。

## 和 PATTERN 的差别

| | `MatcherType.PATTERN` | 本模块 |
|---|---|---|
| 搜索 | 金字塔、角度、可选尺度 | 同尺寸打分，或只平移 |
| 分数 | 一个 NCC | NCC、ZNCC、直方图、SSIM 一起给 |
| 模板学习 | `setTemplate` | 没有模型，每次传入两张图 |

## Python

```python
import cvter

# 两块已经对齐、尺寸相同的灰度（或彩色，内部转灰度）
s = cvter.compare_similarity(a, b)
print(s.found, s.ncc, s.zncc, s.histCorr, s.histBhattacharyya, s.ssim)

# 只搜平移。center = 左上角 + ((w-1)/2, (h-1)/2)
hit = cvter.locate_zncc(image, templ, threshold=0.6)
hit = cvter.locate_ncc(image, templ, threshold=0.6)
print(hit.found, hit.score, hit.topLeft, hit.center)
```

尺寸不同或空图时 `compare_similarity` 的 `found` 为 false。`locate_*` 分数低于阈值时 `found` 为 false。对齐之后做差分、找缺陷见 [golden.md](golden.md)，平移定位就是那条 `GoldenAlign.Translate` 在用的接口。

| 字段 | 含义 |
|---|---|
| `ncc` | `TM_CCORR_NORMED`，相同图约 1 |
| `zncc` | `TM_CCOEFF_NORMED`，去均值，约 `[-1, 1]` |
| `histCorr` | 灰度直方图相关 |
| `histBhattacharyya` | 0 相同，1 不同 |
| `ssim` | 结构相似，约 `[0, 1]` |
