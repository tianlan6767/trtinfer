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
| YOLO 检测 / 姿态 / 实例分割 / OBB / SAHI | [yolo.md](yolo.md) |
| 分类 | [classify.md](classify.md) |
| 异常分割 UVIAD | [uviad.md](uviad.md) |
| DeepLabV3+ | [deeplabv3.md](deeplabv3.md) |
| 灰度模板匹配 | [pattern_match.md](pattern_match.md) |
| 形状匹配 | [shape_match.md](shape_match.md) |
| 卡尺 | [caliper.md](caliper.md) |

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
| `YOLO11POSE` / `YOLO11POSESAHI` | `h.box` + `h.keypoints` |
| `YOLO11SEG` / `YOLO11SEGSAHI` | `h.box` + `h.seg`（相对框的 mask） |
| `YOLO11OBB` / `YOLO11OBBSAHI` | `h.obb` |
| `CLS` | `h.cls` |
| `UVIAD` | `h.box` + `h.seg` |
| `DEEPLABV3` / `DEEPLABV3SAHI` | 见 [deeplabv3.md](deeplabv3.md) |

`DetResult` 始终有 `box / keypoints / seg / obb / cls`，未用到的字段为空或默认值。

`DFINE` / `DFINESAHI` 只在 C++ `load()` 里，**没有**绑到 Python。

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

## 9. 灰度模板匹配

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

m = cvter.MatcherWrapper(param)
m.setTemplate(templ)                 # 灰度图
# m.setTemplate(templ, mask)         # mask 非 0 为有效像素
for r in m.match(gray):
    print(r.Center.x, r.Center.y, r.Angle, r.Score)
    print(r.LeftTop, r.RightTop, r.RightBottom, r.LeftBottom)
```

完整说明：[pattern_match.md](pattern_match.md)。

---

## 10. 形状匹配

```python
param = cvter.MatcherParam()
param.matcherType = cvter.MatcherType.SHAPE
param.angle = 15
param.scoreThreshold = 0.5
param.edgeMinMag = 18
param.maxEdgePoints = 320
param.usePolarity = True
param.greediness = 0.9
m = cvter.MatcherWrapper(param)
m.setTemplate(templ)
hits = m.match(gray)
```

完整说明：[shape_match.md](shape_match.md)。

---

## 11. 卡尺（直线 / 斜边 / 圆）

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
```

`line.found` / `circle.found` 为 False 表示没拟合上。完整说明：[caliper.md](caliper.md)。

---

## 12. 结果类型

**Box**：`left top right bottom score class_id class_name`

**KeyPoint**：`x y vis`（`vis` 对应 C++ 的 `score`）

**OBBox**：`cx cy width height angle score class_name`（`angle` 弧度）

**ClsAttribute**：`class_id score`

**MatchResult**：`LeftTop LeftBottom RightTop RightBottom Center Angle Score`  
点是 `cvter.Point2d`（`.x` / `.y`）。`Angle` 与 OpenCV `getRotationMatrix2D` 相同（画面逆时针为正）。

**LineResult / CircleResult**：见 [caliper.md](caliper.md)。
