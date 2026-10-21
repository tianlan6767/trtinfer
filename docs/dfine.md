# DFINE 检测

`ModelType.DFINE` 是整图检测，`ModelType.DFINESAHI` 是切片检测。两者都走 `cvter.TrtInfer`，结果与 YOLO 检测一样，读 `h.box`。

源码：`src/trt/dfine/`、`src/trt/dfinesahi/`。C++ 示例：`src/examples/dfine/dfine.cpp`。

## 引擎

动态 batch 时按本次张数（SAHI 按切片数）设置运行维度。静态 batch 时，张数或切片数不能超过引擎的 batch。

TensorRT 10 按名字绑定：

| 张量 | 方向 | 说明 |
|---|---|---|
| `images` | 输入 | NCHW |
| `orig_target_sizes` | 输入 | 每张图两个 int64：宽、高 |
| `labels` | 输出 | 类别 id |
| `boxes` | 输出 | `x1 y1 x2 y2` |
| `scores` | 输出 | 置信度 |

TensorRT 8 不按名字，绑定顺序是 `images`、`orig_target_sizes`、`labels`、`boxes`、`scores`。

预处理与现有检测模型相同：BGR 转 RGB，像素除以 255，letterbox，填充值 114。框在原图像素坐标。`class_name` 来自 `names[class_id]`，列表要盖住会出现的 id。

整图只按 `confidence_threshold` 留框，不使用 `nms_threshold`。`DFINESAHI` 把切片框加回原图偏移后，按 `nms_threshold` 做一次 NMS。

切片数不能超过 `max_batch_size`。`DFINESAHI` 一次只传一张图，与 YOLO 的 `*SAHI` 相同。整图 `DFINE` 可以一次多张，张数不要超过 `max_batch_size`。`seg_output` 对这两种类型没有作用。

仓库不带引擎。C++ 示例路径是 `models/engine/dfine_l_obj2coco.engine`，类别名是 COCO 80 类。

## Python

```python
import cvter
import cv2

# 自定义模型换成自己的类别名。COCO 引擎要传满 80 类，否则 class_id 越界。
names = ["person", "bicycle", "car"]
model = cvter.TrtInfer(
    "models/engine/dfine_l_obj2coco.engine",
    cvter.ModelType.DFINE,
    names,
    gpu_id=0,
    confidence_threshold=0.5,
    nms_threshold=0.45,
    max_batch_size=16,
    auto_slice=False,
    slice_width=0,
    slice_height=0,
    slice_horizontal_ratio=0.0,
    slice_vertical_ratio=0.0,
)
assert model.valid

img = cv2.imread("inference/persons.jpg")
for h in model.forwards([img])[0]:
    b = h.box
    print(b.class_name, b.score, b.left, b.top, b.right, b.bottom)
```

切片参数与 YOLO SAHI 相同。`auto_slice=True` 时按图尺寸自动切（默认切片 640、重叠 0.3）。

```python
sahi = cvter.TrtInfer(
    "models/engine/dfine_l_obj2coco.engine",
    cvter.ModelType.DFINESAHI,
    names,
    0, 0.5, 0.45, 16,
    auto_slice=False,
    slice_width=1280,
    slice_height=1280,
    slice_horizontal_ratio=0.2,
    slice_vertical_ratio=0.2,
)
for h in sahi.forwards([img])[0]:
    print(h.box)
```

没有引擎时 `tests/test_dfine.py` 会跳过推理。设置 `CVTER_DFINE_ENGINE`，或把文件放在 `models/engine/dfine_l_obj2coco.engine`（仓库根或 `workspace/` 下同样认）。枚举本身不需要引擎。

## C++

```cpp
auto model = load("models/engine/dfine_l_obj2coco.engine", ModelType::DFINE, names,
                  0, 0.5f, 0.45f, 16, false, 0, 0, 0.0, 0.0);
auto det = model->forwards({image});
// std::vector<object::DetectionResultArray>
```

`ModelType::DFINESAHI` 多传 `auto_slice` 和切片宽高、重叠，参数顺序与其它 SAHI 类型相同。结果同样是 `std::vector<object::DetectionResultArray>`。
