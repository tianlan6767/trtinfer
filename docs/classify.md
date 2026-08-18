# 图像分类

`ModelType.CLS`，对应 YOLO11-cls 一类的 TensorRT 分类引擎。每张图只出一个类别。

源码：`src/trt/cls/`。C++ 示例：`src/examples/classify/cls.cpp`。

## Python

```python
import cvter
import cv2

model = cvter.TrtInfer(
    "workspace/pretrain/yolo11s-cls-dy.engine",
    cvter.ModelType.CLS,
    [],                 # 分类不依赖 names，可空
    gpu_id=0,
    confidence_threshold=0.0,
    nms_threshold=0.0,
    max_batch_size=16,
    auto_slice=False,
    slice_width=0,
    slice_height=0,
    slice_horizontal_ratio=0.0,
    slice_vertical_ratio=0.0,
)
assert model.valid

img = cv2.imread("cat.jpg")
hits = model.forwards([img])[0]
c = hits[0].cls
print(c.class_id, c.score)
```

一次多张：

```python
batch = model.forwards([img0, img1, img2])
for i, hits in enumerate(batch):
    print(i, hits[0].cls.class_id, hits[0].cls.score)
```

张数不要超过 `max_batch_size`。没有 SAHI 分类类型。

## C++

```cpp
auto model = load("yolo11s-cls-dy.engine", ModelType::CLS, {}, 0, 0.0f, 0.0f, 16,
                  false, 0, 0, 0.0, 0.0);
auto out = model->forwards({image});
```

结果是 `object::ClsResultArray`，每张图一个 `ClsAttribute{score, id}`。
