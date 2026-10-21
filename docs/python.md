# Python 用法总览

`make` 会把 pybind 模块链到 `workspace/cvter.so`。解释器必须是编译时的 **Python 3.12**。

```bash
make -j
cd workspace
export PYTHONPATH="$(pwd)"
/usr/local/bin/python3.12
```

```python
import cvter
import cv2
```

下文覆盖当前导出的全部 Python API。专题细节见：

| 模块 | 文档 |
|---|---|
| 功能清单（已实现 / 依赖可选 / 未实现） | [feature_checklist.md](feature_checklist.md) |
| YOLO 检测 / 姿态 / 实例分割 / OBB / SAHI | [yolo.md](yolo.md) |
| DFINE 检测 / SAHI | [dfine.md](dfine.md) |
| 分类 | [classify.md](classify.md) |
| 异常分割 UVIAD | [uviad.md](uviad.md) |
| DeepLabV3+ | [deeplabv3.md](deeplabv3.md) |
| 灰度模板匹配 | [pattern_match.md](pattern_match.md) |
| 形状匹配 | [shape_match.md](shape_match.md) |
| 卡尺 | [caliper.md](caliper.md) |
| 预处理 / Blob | [blob.md](blob.md) |
| 对齐相似度 / 平移定位 | [similarity.md](similarity.md) |
| 金样比对 / 差分缺陷 | [golden.md](golden.md) |
| 分水岭 / 距离变换 / Canny | [seg.md](seg.md) |
| ORB / SIFT，粗到精定位 | [feature.md](feature.md) |
| 条码 / QR / OCR | [code.md](code.md) |
| 相机标定 | [calib.md](calib.md) |

---

## 1. TrtInfer

```python
model = cvter.TrtInfer(
    model_path,                 # engine / trtmodel
    model_type,                 # cvter.ModelType.*
    names,                      # 类别名列表；分类可传 []
    gpu_id=0,
    confidence_threshold=0.5,
    nms_threshold=0.45,
    max_batch_size=32,
    auto_slice=True,            # SAHI：按图尺寸自动切片；False 则用下面宽高
    slice_width=640,
    slice_height=640,
    slice_horizontal_ratio=0.3, # 水平重叠比例
    slice_vertical_ratio=0.3,
    seg_output=cvter.SegOutput.INSTANCES,  # 仅 DeepLab 使用
)
assert model.valid
batch = model.forwards([img])   # list[list[DetResult]]，外层=图，内层=该图目标
```

`forwards` 接受 BGR `uint8` 图（`cv2.imread`）。可一次多张，张数不要超过 `max_batch_size`。

### ModelType 与结果字段

| ModelType | 读哪个字段 |
|---|---|
| `YOLOV5` / `YOLO11` 及对应 `*SAHI` | `h.box` |
| `DFINE` / `DFINESAHI` | `h.box` |
| `YOLO11POSE` / `YOLO11POSESAHI` | `h.box` + `h.keypoints` |
| `YOLO11SEG` / `YOLO11SEGSAHI` | `h.box` + `h.seg`（相对框的 mask） |
| `YOLO11OBB` / `YOLO11OBBSAHI` | `h.obb` |
| `CLS` | `h.cls` |
| `UVIAD` | `h.box` + `h.seg` |
| `DEEPLABV3` / `DEEPLABV3SAHI` | 见 [deeplabv3.md](deeplabv3.md) |

`DetResult` 始终有 `box / keypoints / seg / obb / cls`，未用到的字段为空或默认值。

---

## 2. YOLO 检测（整图 / SAHI）

```python
names = ["白点", "划痕", "碰伤"]
model = cvter.TrtInfer(
    "model.transd-dy.trtmodel",
    cvter.ModelType.YOLO11,   # 或 YOLOV5
    names,
    0, 0.25, 0.45, 18,
    False, 0, 0, 0.0, 0.0,
)
img = cv2.imread("a.jpg")
for h in model.forwards([img])[0]:
    b = h.box
    print(b.class_name, b.score, b.left, b.top, b.right, b.bottom)
    cv2.rectangle(img, (int(b.left), int(b.top)), (int(b.right), int(b.bottom)), (0, 255, 0), 2)
```

大图切片（类型用 `*SAHI`；`max_batch_size` 必须 ≥ 切片数）：

```python
model = cvter.TrtInfer(
    "model.transd-dy.trtmodel",
    cvter.ModelType.YOLO11SAHI,
    names,
    0, 0.3, 0.45, 18,
    False, 1280, 1280, 0.2, 0.2,
)
```

YOLOv8 / YOLO11 的 ONNX 需先 `python v8trans.py xxx.onnx`，得到 `*.transd.onnx` 再转 engine。见 [yolo.md](yolo.md)。

---

## 3. YOLO 姿态

```python
model = cvter.TrtInfer(
    "yolo11l-pose.transd.engine",
    cvter.ModelType.YOLO11POSE,
    ["person"],
    0, 0.5, 0.45, 1, False, 0, 0, 0.0, 0.0,
)
for h in model.forwards([img])[0]:
    print(h.box, [(kp.x, kp.y, kp.vis) for kp in h.keypoints])
```

切片：`ModelType.YOLO11POSESAHI`，切片参数同上。

---

## 4. YOLO 实例分割

```python
model = cvter.TrtInfer(
    "yolov8s-seg.transd.trtmodel",
    cvter.ModelType.YOLO11SEG,
    names,
    0, 0.5, 0.45, 1, False, 0, 0, 0.0, 0.0,
)
for h in model.forwards([img])[0]:
    mask = h.seg          # 相对 h.box 的 uint8 mask
    print(h.box.class_name, mask.shape)
```

切片：`YOLO11SEGSAHI`。

---

## 5. YOLO 旋转框

```python
model = cvter.TrtInfer(
    "yolo11m-obb.transd.engine",
    cvter.ModelType.YOLO11OBB,
    ["plane", "ship", "small vehicle"],
    0, 0.7, 0.45, 1, False, 0, 0, 0.0, 0.0,
)
for h in model.forwards([img])[0]:
    o = h.obb
    print(o.class_name, o.cx, o.cy, o.width, o.height, o.angle, o.score)
```

切片：`YOLO11OBBSAHI`。`angle` 为弧度。

---

## 6. 分类

```python
model = cvter.TrtInfer(
    "yolo11s-cls-dy.engine",
    cvter.ModelType.CLS,
    [],
    0, 0.0, 0.0, 16, False, 0, 0, 0.0, 0.0,
)
c = model.forwards([img])[0][0].cls
print(c.class_id, c.score)
```

---

## 7. 异常分割 UVIAD

```python
model = cvter.TrtInfer(
    "net_100.trtmodel",
    cvter.ModelType.UVIAD,
    [],
    0, 0.45, 0.0, 1, False, 0, 0, 0.0, 0.0,
)
for h in model.forwards([img])[0]:
    print(h.box, h.seg.shape)
```

---

## 8. DeepLabV3+

```python
names = ["_background_", "胶路", "基准面"]
model = cvter.TrtInfer(
    "best_epoch_weights.trtmodel",
    cvter.ModelType.DEEPLABV3,
    names,
    0, 0.35, 0.0, 1, False, 0, 0, 0.0, 0.0,
    cvter.SegOutput.CLASS_MAP,
)
cls = model.forwards([img])[0][0].seg   # (H, W) uint8
```

切片用 `DEEPLABV3SAHI`。完整说明：[deeplabv3.md](deeplabv3.md)。

---

## 9. DFINE 检测（整图 / SAHI）

结果与 YOLO 检测相同，读 `h.box`。`names` 要盖住模型会输出的 `class_id`。整图可一次多张，张数不要超过 `max_batch_size`。切片一次一张图。

```python
names = ["person", "bicycle", "car"]  # COCO 引擎请传满 80 类，见 dfine.md
model = cvter.TrtInfer(
    "dfine_l_obj2coco.engine",
    cvter.ModelType.DFINE,
    names,
    0, 0.5, 0.45, 16,
    False, 0, 0, 0.0, 0.0,
)
img = cv2.imread("persons.jpg")
for h in model.forwards([img])[0]:
    b = h.box
    print(b.class_name, b.score, b.left, b.top, b.right, b.bottom)
```

大图切片（`max_batch_size` 必须 ≥ 切片数）：

```python
model = cvter.TrtInfer(
    "dfine_l_obj2coco.engine",
    cvter.ModelType.DFINESAHI,
    names,
    0, 0.5, 0.45, 16,
    False, 1280, 1280, 0.2, 0.2,
)
```

完整说明：[dfine.md](dfine.md)。

---

## 10. 灰度模板匹配

```python
param = cvter.MatcherParam()
param.matcherType = cvter.MatcherType.PATTERN
param.angle = 8
param.scoreThreshold = 0.55
param.maxCount = 1
param.minArea = 256
param.iouThreshold = 0.3
param.meanBorder = True
param.stopLayer = 0
param.minScale = 1.0             # 与 maxScale 都为 1 时不搜尺度
param.maxScale = 1.0
param.scaleStep = 0.05           # 或 numScales>1 均匀取样

m = cvter.MatcherWrapper(param)
m.setTemplate(templ)                 # 灰度图
# m.setTemplate(templ, mask)         # mask 非 0 为有效像素；尺度搜索时 mask 一并缩放
for r in m.match(gray):
    print(r.Center.x, r.Center.y, r.Angle, r.Score, r.Scale)
    print(r.LeftTop, r.RightTop, r.RightBottom, r.LeftBottom)
```

完整说明：[pattern_match.md](pattern_match.md)。

---

## 11. 形状匹配

```python
param = cvter.MatcherParam()
param.matcherType = cvter.MatcherType.SHAPE
param.angle = 15
param.scoreThreshold = 0.5
param.edgeMinMag = 18
param.maxEdgePoints = 320
param.usePolarity = True
param.greediness = 0.9
param.minScale = 1.0
param.maxScale = 1.0
param.scaleStep = 0.05
m = cvter.MatcherWrapper(param)
m.setTemplate(templ)
hits = m.match(gray)
```

完整说明：[shape_match.md](shape_match.md)。

---

## 12. 卡尺（直线 / 斜边 / 圆）

```python
param = cvter.CaliperParam()
param.polarity = cvter.CaliperPolarity.LightToDark
param.select = cvter.CaliperSelect.Strongest
param.contrast = 18
param.stride = 2

cal = cvter.CaliperWrapper(param)
line = cal.find_line(gray, x0, y0, x1, y1, search_horizontal=True)
line = cal.find_line_oriented(gray, cx, cy, phi_deg=90, length1=80, length2=20)
circle = cal.find_circle(gray, cx, cy, radius=70, search=20, start_deg=0, end_deg=360)

# 也可以不建 Wrapper
line = cvter.find_line(gray, x0, y0, x1, y1, True, param)
circle = cvter.find_circle(gray, cx, cy, 70, 20, 0, 360, param)

param.polarity = cvter.CaliperPolarity.DarkToLight
rect = cvter.find_rectangle(gray, cx, cy, 150, 90, angle_deg=7, search=16, param=param)
print(rect.center, rect.width, rect.height, rect.angle)
print(cvter.distance_points(0, 0, 3, 4), cvter.angle_lines(0, 0, 10, 0, 0, 0, 0, 10))
```

`line.found` / `circle.found` / `rect.found` 为 False 表示没拟合上。矩形搜索指向内部。完整说明：[caliper.md](caliper.md)。

---

## 13. 预处理与 Blob

不需要 TensorRT。非 0 为前景，Blob 坐标在原图上。

```python
th = cvter.ThresholdParam()
th.type = cvter.ThresholdType.Otsu
binary = cvter.threshold(cvter.gaussian_blur(gray, 5, 1.0), th)

morph = cvter.MorphParam()
morph.op = cvter.MorphOp.Close
binary = cvter.morphology(binary, morph)

filt = cvter.BlobFilter()
filt.minArea = 40
filt.minCircularity = 0.6
for b in cvter.find_blobs(binary, filt):
    print(b.center, b.area, b.circularity, b.aspect)
```

`crop_roi` / `median_blur` / `contour_stats` 以及过滤字段见 [blob.md](blob.md)。

---

## 14. 结果类型

**Box**：`left top right bottom score class_id class_name`

**KeyPoint**：`x y vis`（`vis` 对应 C++ 的 `score`）

**OBBox**：`cx cy width height angle score class_name`（`angle` 弧度）

**ClsAttribute**：`class_id score`

**MatchResult**：`LeftTop LeftBottom RightTop RightBottom Center Angle Score Scale`  
点是 `cvter.Point2d`（`.x` / `.y`）。`Angle` 与 OpenCV `getRotationMatrix2D` 相同（画面逆时针为正）。`Scale` 相对模板，默认匹配为 1。

**LineResult / CircleResult**：见 [caliper.md](caliper.md)。

**Blob**：`label area perimeter circularity aspect angle center left top width height`。`center` 是 `Point2d`。`area` 为像素个数，圆度用外轮廓。

**ContourStats**：字段与 Blob 类似，`area` 是轮廓折线面积。见 [blob.md](blob.md)。

**GoldenDefect**：`label area meanAbs meanSigned center left top width height`。坐标在金样上。见 [golden.md](golden.md)。

---

## 15. 对齐相似度（不是金字塔匹配）

`PATTERN` 已经是金字塔 + 旋转 + 尺度的 NCC。下面这组只做同尺寸打分，或只搜平移。

```python
score = cvter.compare_similarity(roi_a, roi_b)   # 必须同尺寸
print(score.ncc, score.zncc, score.ssim, score.histCorr)
hit = cvter.locate_zncc(gray, templ, threshold=0.6)  # 或 locate_ncc
print(hit.center, hit.score)
```

见 [similarity.md](similarity.md)。

---

## 16. 金样比对

平移定位只回答「模板在哪」。这里在对齐之后做绝对差分，再阈值、形态学、Blob，把多出来或少掉的区域收成缺陷。

```python
p = cvter.GoldenParam()
p.align = cvter.GoldenAlign.Translate   # None / Translate / Affine / Homography
p.polarity = cvter.GoldenPolarity.Both  # Brighter / Darker
p.diffThreshold = 30
p.minArea = 40
p.blurSigma = 1.0
hit = cvter.compare_golden(test, golden, p)          # mask 可选，与金样同尺寸
print(hit.ok, hit.defectCount, hit.meanAbs, hit.alignMethod)
for d in hit.defects:
    print(d.center, d.area, d.meanSigned)
```

`hit.transform` 是 3×3，把金样坐标变到待测图。`hit.diff` 是绝对差分，`hit.signedDiff` 是 float32 的「待测 - 金样」。见 [golden.md](golden.md)。

---

## 17. 分水岭和常规边缘

```python
dist = cvter.distance_transform(binary)
edges = cvter.canny(gray, 50, 150)
contours = cvter.extract_contours(binary, external_only=True)
split = cvter.split_blobs(binary, min_distance=8, dist_ratio=0.4)
print(split.found, len(split.regions), split.labels.shape)
```

自己准备 `CV_32S` 种子时用 `cvter.watershed(image, markers)`。见 [seg.md](seg.md)。

---

## 18. ORB / SIFT

```python
hit = cvter.match_features(scene, model, cvter.FeatureDetector.ORB, ratio=0.75, ransac_reproj=3.0)
print(hit.found, hit.inliers, hit.corners)   # 4x2，模板四角投到场景
if cvter.sift_available():
    hit = cvter.match_features(scene, model, cvter.FeatureDetector.SIFT)

param = cvter.CoarseRefineParam()
param.matcherType = cvter.MatcherType.PATTERN   # SHAPE 时把 angle 设为至少 5
param.angle = 0                                  # 0 = 只搜 0°
param.margin = 120
loc = cvter.match_coarse_to_fine(scene, model, param=param)
# loc.hits[0] 的点在全图坐标；loc.roi 是精匹配用的矩形；loc.coarse 是特征结果
```

见 [feature.md](feature.md)。不要把 `stopLayer=2` 当成加速。

---

## 19. 条码、QR、OCR

```python
for h in cvter.decode_qr(bgr):
    print(h.text, h.polygon)
for h in cvter.decode_barcode(bgr):
    print(h.format, h.text)
text = cvter.recognize_text(gray, "eng")
print(text.backend, text.text)   # 没编 Tesseract 时 backend == "none"
```

QR 用 OpenCV `QRCodeDetector`，不加载模型。一维码优先 OpenCV，解不出时用 zbar。不用微信 QR。见 [code.md](code.md)。

---

## 20. 相机标定

默认亚像素。板型：`Chessboard`、`Charuco`、`CirclesSymmetric`、`CirclesAsymmetric`、`ArucoGrid`、`CharucoDiamond`、`AprilTag`。没有单独的不对称棋盘检测器，棋盘加标记用 `Charuco`。

```python
cfg = cvter.BoardConfig()
cfg.type = cvter.BoardType.Chessboard
cfg.squaresX, cfg.squaresY = 9, 6
det = cvter.detect_board(gray, cfg)
print(det.subpixel, det.refine, det.imagePoints.shape)
calib = cvter.calibrate_images(views, cfg)
und = cvter.undistort(gray, calib.cameraMatrix, calib.distCoeffs)
cvter.save_calibration("camera.yml", calib)
```

`aruco_available()` 为 false 时，ChArUco / ArUco / 菱形 / AprilTag 检测会失败并在 `refine` 里写 `aruco-unavailable`。见 [calib.md](calib.md)。
