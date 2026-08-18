# DeepLabV3+ 语义分割

把 bubbliiiing DeepLabV3+（`workspace/deepv3`）接到现有 TensorRT 推理栈。`ModelType` 只选整图还是切片；输出形态用 `SegOutput`。

## 模型

当前胶路任务权重：

| | |
|---|---|
| ONNX | `workspace/deepv3/logs_tgwy_v3/best_epoch_weights.onnx` |
| Engine | `workspace/deepv3/logs_tgwy_v3/best_epoch_weights.trtmodel` |
| 输入 | `images`，`1x3x960x960`，RGB，`/255` |
| 输出 | `output`，`1xC×960x960` logits（本任务 C=3） |
| 类别 | `_background_`、`胶路`、`基准面` |
| 预处理 | letterbox（灰 128）+ BGR→RGB |

导出约定与 `workspace/deepv3/deeplab.py` 的 `convert_to_onnx` 一致：输入名 `images`，输出名 `output`。

转 TensorRT：

```bash
trtexec \
  --onnx=workspace/deepv3/logs_tgwy_v3/best_epoch_weights.onnx \
  --saveEngine=workspace/deepv3/logs_tgwy_v3/best_epoch_weights.trtmodel \
  --fp16
```

静态 batch=1。若要动态 batch，需重新导出带 `dynamic_axes` 的 ONNX。

## ModelType 与 SegOutput

| | 含义 |
|---|---|
| `ModelType.DEEPLABV3` | 整图 letterbox 一次推理 |
| `ModelType.DEEPLABV3SAHI` | 切片推理，重叠区平均 softmax 再 argmax |
| `SegOutput.INSTANCES`（默认） | 每类连通域一个框 + ROI mask，适合缺陷等稀疏前景 |
| `SegOutput.CLASS_MAP` | 一张与原图同尺寸的 `uint8` 类别图，适合胶路/基准面等大区域 |

不要为输出形态再拆新的 ModelType。YOLO 等其它模型忽略 `seg_output`。

## Python

区域分割（胶路）：

```python
model = tinfer.TrtInfer(
    engine, tinfer.ModelType.DEEPLABV3, names,
    0, 0.35, 0.0, 1, False, 0, 0, 0.0, 0.0,
    tinfer.SegOutput.CLASS_MAP,
)
hits = model.forwards([img])[0]
cls = hits[0].seg          # (H, W) uint8，像素值=类别 id
glue = cls == 1
```

缺陷 / 要逐块量测时用默认实例输出：

```python
model = tinfer.TrtInfer(
    engine, tinfer.ModelType.DEEPLABV3, names,
    0, 0.35, 0.0, 1, False, 0, 0, 0.0, 0.0,
    tinfer.SegOutput.INSTANCES,   # 可省略，这是默认
)
hits = model.forwards([img])[0]
for h in hits:
    x1, y1, x2, y2 = h.box.left, h.box.top, h.box.right, h.box.bottom
    mask = h.seg   # 相对框的二值图
```

SAHI 切片（独立类型 `DEEPLABV3SAHI`，切片参数与 YOLO-SAHI 相同）。当前 engine 是静态 batch=1，切片会串行推理。胶路模型建议切片 960、重叠 0.1，与 `predict.py` 滑窗一致：

```python
model = tinfer.TrtInfer(
    engine, tinfer.ModelType.DEEPLABV3SAHI, names,
    0, 0.35, 0.0, 1,
    False, 960, 960, 0.1, 0.1,
    tinfer.SegOutput.CLASS_MAP,
)
```

`confidence_threshold` 只对 `INSTANCES` 生效，滤的是该类 mask 内的平均 softmax。`CLASS_MAP` 不按分数过滤。

## C++

`src/examples/deeplab/deeplabv3.cpp`：`run_deeplabv3()` / `run_deeplabv3_sahi()`。`load(..., SegOutput::CLASS_MAP)`。

## 后处理

整图：GPU argmax → 裁 letterbox 灰边 → 缩放到原图 → 按 `SegOutput` 出类别图或 `findContours`。

SAHI：CUDA 切片（`slice::SliceImage`）→ 每片 letterbox 推理 → 重叠区平均概率 → 整图 argmax → 同上。
