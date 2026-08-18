# 异常分割（UVIAD）

无监督 / 重建类异常检测，输出前景连通域的框 + mask，走分割结果通道。

源码：`src/trt/ad/`。C++ 示例：`src/examples/anomaly/uviad.cpp`。

## Python

```python
import cvter
import cv2

model = cvter.TrtInfer(
    "workspace/ad/ad2/net_100.trtmodel",
    cvter.ModelType.UVIAD,
    [],
    gpu_id=0,
    confidence_threshold=0.45,   # 异常分数阈值
    nms_threshold=0.0,
    max_batch_size=1,
    auto_slice=False,
    slice_width=0,
    slice_height=0,
    slice_horizontal_ratio=0.0,
    slice_vertical_ratio=0.0,
)
img = cv2.imread("ng.jpg")
for h in model.forwards([img])[0]:
    b = h.box
    mask = h.seg
    print(b.score, b.left, b.top, b.right, b.bottom, mask.shape)
```

`confidence_threshold` 用来滤弱异常。没有 `UVIADSAHI`。

## C++

```cpp
auto model = load("net_100.trtmodel", ModelType::UVIAD, {}, 0, 0.45f, 0.0f, 1,
                  false, 0, 0, 0.0, 0.0);
auto det = model->forwards({image});
// std::vector<object::SegmentationResultArray>
// osd_segmentation(image, result[i]);
```
