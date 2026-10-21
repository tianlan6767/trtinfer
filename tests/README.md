# 传统视觉测试

能力对照（已实现 / 依赖可选 / 未实现）见 [docs/feature_checklist.md](../docs/feature_checklist.md)。

数值测试（有的要完整 `workspace/cvter.so`，含 CUDA / TensorRT）：

```bash
make test-blob
make test-caliper
make test-rect
make test-measure
make test-scale
make test-match
make test-shape
make test-golden
make test-dfine    # 需要完整 cvter.so；没有 DFINE engine 时推理用例跳过
```

## 可视化

`make test-viz` 另编一份只含传统算法的 `objs/pytrad/cvter.so`，不链 CUDA / TensorRT，不调用 `imshow`。图片写到 `tests/out_viz/`（已 gitignore）。

```bash
make test-viz
# 等价于 PYTHONPATH=objs/pytrad python3 tests/test_*_viz.py
```

| 脚本 | 内容 |
|---|---|
| `test_blob_viz.py` | 预处理、Blob、轮廓 |
| `test_pattern_match_viz.py` | NCC，含同一张图上的两个尺度 |
| `test_shape_match_viz.py` | 形状匹配，含两个尺度 |
| `test_mask_match_viz.py` | 掩膜 NCC / SHAPE：ROI 外的文字、孔、噪声。不带 mask 锁到干扰峰，带 mask 回到真目标 |
| `test_caliper_viz.py` | 直线、定向卡尺、圆 |
| `test_rect_metrology_viz.py` | 矩形和量测 |
| `test_measure_viz.py` | 几何、多边 / 双边 / 弧段 / 开口、拟合与形位、包围尺寸、标定尺寸 |
| `test_similarity_viz.py` | 对齐 NCC/ZNCC/SSIM，平移定位 |
| `test_golden_viz.py` | 金样比对：极性、ROI/掩膜、平移、模糊、ORB 仿射/单应 |
| `test_seg_viz.py` | 距离变换、Canny、分水岭切开粘连圆 |
| `test_feature_viz.py` | ORB；SIFT 可用时也跑 |
| `test_coarse_refine_viz.py` | ORB/SIFT 粗定位后，ROI 上 PATTERN/SHAPE，坐标回到全图。`make test-coarse` 只跑这一份 |
| `test_code_viz.py` | QR、EAN-13、OCR |
| `test_calib_viz.py` | 七种板、亚像素、多视标定、去畸变、YAML、手眼 |

螺母样图不在仓库里时，对应用例跳过。掩膜匹配的图在 `tests/test_mask_match_viz.py` 里现场生成，不依赖螺母样图，输出在 `tests/out_viz/mask/`。OpenCV 没编 aruco 时跳过 ChArUco / ArUco / 菱形 / AprilTag。没有 SIFT 时跳过 SIFT。没有 Tesseract 时 OCR 用例跳过。一维码在 OpenCV 解不出时走 zbar。金样比对的夹具在 `tests/synth.py` 里现场生成，不依赖螺母图。粗到精用的也是合成图。`tests/bench_aoi_coarse_refine.py` 只有在设置了 `CVTER_AOI_SCENE` 和 `CVTER_AOI_TEMPL`（或传入 `--scene` / `--templ`）时才读大图，否则直接跳过。`tests/test_dfine.py` 在 `cvter.so` 可导入时总会检查 `ModelType.DFINE` / `DFINESAHI`。推理用例在没有引擎时跳过：设置 `CVTER_DFINE_ENGINE`，或把文件放在 `models/engine/dfine_l_obj2coco.engine`（仓库根或 `workspace/` 下同样认）。可选 `CVTER_DFINE_IMAGE`，以及逗号分隔的 `CVTER_DFINE_NAMES`。

Windows 不走这份 Makefile。按 [docs/env_windows.md](../docs/env_windows.md) 用 CMake 时，把 `src/cv_golden/golden.cpp` 和 Makefile 变量 `viz_srcs` 里的其它传统算法源一起编成名为 `cvter` 的 pybind 模块（对应 Linux 的 `objs/pytrad/cvter.so`），再跑 `tests/test_golden_viz.py`。只验算法、不画图时，用与 `make test-golden` 相同的源文件编 `tests/test_golden.cpp`。
