# YOLO 检测 / 姿态 / 实例分割 / OBB

统一入口：`cvter.TrtInfer` + `model.forwards([img])`。YOLOv8 与 YOLO11 共用同一套 decode（`YOLO11*`）。

源码：`src/trt/yolo/`、`src/trt/sahiyolo/`。C++ 示例：`src/examples/yolo11/`、`yolo11pose/`、`yolo11seg/`、`yolo11obb/`、`yolov5/`。

## 导出约定

1. 模型必须是**动态 batch**。
2. YOLOv8 / YOLO11 官方 ONNX 输出是 `1x84x8400`，需要转成 `1x8400x84`：

```bash
python src/examples/py_example/v8trans.py yolov8n.onnx
# 生成 yolov8n.transd.onnx
trtexec --onnx=yolov8n.transd.onnx --saveEngine=yolov8n.transd-dy.trtmodel --fp16 --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:18x3x640x640
```

3. TensorRT 10 的输入名一般是 `images`，输出名 `output0`（用 Netron 核对）。
4. 切片数不能超过 `max_batch_size`，否则无法推理。

## 整图检测

```python
import cvter
import cv2

names = ["person", "helmet"]
model = cvter.TrtInfer(
    "helmet.engine",
    cvter.ModelType.YOLOV5,   # 或 YOLO11
    names,
    gpu_id=0,
    confidence_threshold=0.25,
    nms_threshold=0.45,
    max_batch_size=8,
    auto_slice=False,
    slice_width=0,
    slice_height=0,
    slice_horizontal_ratio=0.0,
    slice_vertical_ratio=0.0,
)
img = cv2.imread("a.jpg")
for h in model.forwards([img])[0]:
    b = h.box
    print(b.class_name, b.score, b.left, b.top, b.right, b.bottom)
```

工业缺陷示例（与 `workspace/test.py` 相同）：

```python
names = ["白点", "边缘凹坑", "黑点", "黑块", "黑线", "划痕", "碰伤"]
model = cvter.TrtInfer(
    "td_0816.transd-dy.trtmodel",
    cvter.ModelType.YOLO11,
    names, 0, 0.25, 0.45, 18, False, 0, 0, 0.0, 0.0,
)
```

## SAHI 切片检测

大图切成重叠块，decode 时把框映射回原图，最后对整图做一次 NMS。

```python
model = cvter.TrtInfer(
    "td_0816.transd-dy.trtmodel",
    cvter.ModelType.YOLO11SAHI,   # 或 YOLOV5SAHI
    names,
    0, 0.3, 0.45, 18,
    auto_slice=False,
    slice_width=1280,
    slice_height=1280,
    slice_horizontal_ratio=0.2,
    slice_vertical_ratio=0.2,
)
```

- `auto_slice=True`：按图尺寸自动切（默认切片 640、重叠 0.3）。
- `auto_slice=False`：用给定 `slice_width/height`。
- 切片数 > `max_batch_size` 会失败。

## 姿态

```python
model = cvter.TrtInfer(
    "yolo11l-pose.transd.engine",
    cvter.ModelType.YOLO11POSE,   # 切片：YOLO11POSESAHI
    ["person"],
    0, 0.5, 0.45, 1, False, 0, 0, 0.0, 0.0,
)
for h in model.forwards([img])[0]:
    print(h.box.class_name, h.box.score)
    for kp in h.keypoints:
        print(kp.x, kp.y, kp.vis)
        if kp.vis > 0.3:
            cv2.circle(img, (int(kp.x), int(kp.y)), 3, (0, 0, 255), -1)
```

COCO 姿态一般是 17 个点。`KeyPoint.vis` 是可见性/置信度。

## 实例分割

```python
model = cvter.TrtInfer(
    "yolov8s-seg.transd.trtmodel",
    cvter.ModelType.YOLO11SEG,   # 切片：YOLO11SEGSAHI
    names,
    0, 0.5, 0.45, 1, False, 0, 0, 0.0, 0.0,
)
for h in model.forwards([img])[0]:
    x1, y1 = int(h.box.left), int(h.box.top)
    mask = h.seg                 # 相对框，uint8
    roi = img[y1:y1 + mask.shape[0], x1:x1 + mask.shape[1]]
```

语义分割（胶路等大区域）用 DeepLab，不要用 YOLO-Seg，见 [deeplabv3.md](deeplabv3.md)。

## 旋转框 OBB

```python
dot_names = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool", "container crane",
]
model = cvter.TrtInfer(
    "yolo11m-obb.transd.engine",
    cvter.ModelType.YOLO11OBB,   # 切片：YOLO11OBBSAHI
    dot_names,
    0, 0.7, 0.45, 1, False, 0, 0, 0.0, 0.0,
)
for h in model.forwards([img])[0]:
    o = h.obb
    print(o.class_name, o.cx, o.cy, o.width, o.height, o.angle, o.score)
```

`obb.angle` 是弧度。画框时绕中心旋转四个角点。

## C++

```cpp
#include "trt/infer.hpp"

auto model = load("yolo11s.engine", ModelType::YOLO11, names, 0, 0.25f, 0.45f, 8,
                  false, 0, 0, 0.0, 0.0);
auto det = model->forwards({image});
```

类型换成 `YOLO11SAHI` / `YOLO11POSE` / `YOLO11SEG` / `YOLO11OBB` 即可。可视化：`osd_detection` / `osd_pose` / `osd_segmentation` / `osd_obb`。
