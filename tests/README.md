# 传统视觉测试

能力对照（已实现 / 依赖可选 / 未实现）见 [docs/feature_checklist.md](../docs/feature_checklist.md)。

数值测试（有的要完整 `workspace/cvter.so`，含 CUDA / TensorRT）：

```bash
make test-blob
make test-caliper
make test-rect
make test-scale
make test-match
make test-shape
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
| `test_caliper_viz.py` | 直线、定向卡尺、圆 |
| `test_rect_metrology_viz.py` | 矩形和量测 |
| `test_similarity_viz.py` | 对齐 NCC/ZNCC/SSIM，平移定位 |
| `test_seg_viz.py` | 距离变换、Canny、分水岭切开粘连圆 |
| `test_feature_viz.py` | ORB；SIFT 可用时也跑 |
| `test_code_viz.py` | QR、EAN-13、OCR |
| `test_calib_viz.py` | 七种板、亚像素、多视标定、去畸变、YAML、手眼 |

螺母样图不在仓库里时，对应用例跳过。OpenCV 没编 aruco 时跳过 ChArUco / ArUco / 菱形 / AprilTag。没有 SIFT 时跳过 SIFT。没有 Tesseract 时 OCR 用例跳过。一维码在 OpenCV 解不出时走 zbar。
