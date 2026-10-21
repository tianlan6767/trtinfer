# 工业视觉功能清单

状态以当前 `qa` 为准。这里只标「能不能用」，参数和示例仍看各模块文档。可视化脚本和跳过条件见 [tests/README.md](../tests/README.md)，下面不重复那张表。

| 状态 | 含义 |
|---|---|
| 已实现 | 代码已在 `qa` 上，按对应文档即可调用 |
| 依赖可选 | 接口已经接上；本机缺库或样图时，相关用例会跳过 |
| 未实现 | 还没有可用实现，或只有算法、缺采图 / 上层 |

完整模块 `make -j` 得到 `workspace/cvter.so`，深度学习路径需要 CUDA 与 TensorRT。传统算法可以单独编：`make test-viz` 生成 `objs/pytrad/cvter.so`，不链 CUDA / TensorRT。

## 已实现

| 能力 | 范围 | 文档 |
|---|---|---|
| 深度学习 TensorRT | YOLO 检测 / 姿态 / 实例分割 / OBB / SAHI、分类、UVIAD、DeepLabV3+；C++ 侧还有 `DFINE` / `DFINESAHI`。走完整 `cvter`，需要 CUDA / TensorRT | [yolo.md](yolo.md)、[classify.md](classify.md)、[uviad.md](uviad.md)、[deeplabv3.md](deeplabv3.md)、[python.md](python.md) |
| 灰度模板匹配 | `MatcherType.PATTERN`（NCC）。尺度搜索：`minScale` / `maxScale` / `scaleStep` / `numScales` | [pattern_match.md](pattern_match.md) |
| 形状匹配 | `MatcherType.SHAPE`。尺度参数与 PATTERN 同一组 | [shape_match.md](shape_match.md) |
| 预处理与 Blob | 阈值、形态学、连通域 / 轮廓 | [blob.md](blob.md) |
| 卡尺与量测 | 直线、圆、找矩形，以及距离 / 角度辅助 | [caliper.md](caliper.md) |
| ROI 相似度 | 对齐后的 NCC / ZNCC / 直方图 / SSIM，以及只做平移的定位 | [similarity.md](similarity.md) |
| 金样比对 / 差分缺陷 | 可选平移（ZNCC/NCC）或 ORB 仿射 / 单应对齐，绝对差分与有符号差分，极性、模糊、ROI / 掩膜，形态学之后取 Blob。不需要 GPU | [golden.md](golden.md) |
| 常规分割 | 分水岭、距离变换、Canny、切开粘连目标 | [seg.md](seg.md) |
| 稀疏特征 | ORB。OpenCV 带 SIFT 时同一接口也走 SIFT | [feature.md](feature.md) |
| 读码与 OCR | QR；一维码先 OpenCV、解不出再 zbar；OCR 走 Tesseract（可选） | [code.md](code.md) |
| 相机标定 | 默认亚像素。多视标定、去畸变、YAML 读写、手眼（`HandEyeMethod.Park`，另有 Tsai / Horaud / Andreff / Daniilidis） | [calib.md](calib.md) |

`BoardType`（检测默认亚像素）：

| 枚举 | 本机要求 |
|---|---|
| `Chessboard` | OpenCV 即可 |
| `CirclesSymmetric` | OpenCV 即可 |
| `CirclesAsymmetric` | OpenCV 即可 |
| `Charuco` | 需要 OpenCV aruco |
| `ArucoGrid` | 需要 OpenCV aruco |
| `CharucoDiamond` | 需要 OpenCV aruco |
| `AprilTag` | 需要 OpenCV aruco |

格子数、字典和亚像素窗口见 [calib.md](calib.md)。

## 依赖可选

这些能力已经实现。缺下面的库或样图时，可视化用例会跳过，库本身仍可编译。

| 能力 | 什么时候会跳过 |
|---|---|
| ChArUco / ArUco 网格 / ChArUco 菱形 / AprilTag | OpenCV 未编 aruco。`aruco_available()` 为 false，检测 `refine` 为 `aruco-unavailable` |
| SIFT | 当前 OpenCV 没有 SIFT。`sift_available()` 为 false，`method` 为 `SIFT-unavailable`。ORB 仍可用 |
| 一维码 | OpenCV 条码模块和 zbar 都没有时，`barcode_available()` 为 false，返回空列表。有 zbar 时作为 OpenCV 解不出之后的后备 |
| OCR | 未编 Tesseract（`-DCVTER_WITH_TESSERACT`）或加载不了 `eng`。`backend` 为 `none` |
| 螺母实图用例 | 样图不在仓库里。卡尺算法本身在，对应用例跳过 |

## 未实现

| 能力 | 现状 |
|---|---|
| DFINE 的 Python 绑定与导出打磨 | C++ `load()` 已有 `DFINE` / `DFINESAHI`，Python 侧尚未导出 |
| 相机 SDK 采图，以及产线标定 / 手眼集成层 | 标定、去畸变、手眼算法已有；相机 SDK 和接到设备上的集成层还没有 |
| HMI / 上层集成 | 还没有界面和上层对接层 |

## 可视化测试

```bash
make test-viz
```

不调用 `imshow`。图片写到 `tests/out_viz/`（已 gitignore）。各脚本覆盖预处理、匹配、卡尺、相似度、金样比对、分割、特征、读码和七种标定板，说明见 [tests/README.md](../tests/README.md)。
