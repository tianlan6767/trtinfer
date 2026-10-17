# 相机标定

CPU，不走 TensorRT。源码：`src/cv_calib/`。可视化：`tests/test_calib_viz.py`（`make test-viz`）。

检测默认就是亚像素。`BoardDetection.subpixel` 为 true，`refine` 字符串里带 `subpix`（圆点阵写的是 `blobSubpix`）。

## 板型

`BoardType`：

| 枚举 | 检测 | `squaresX/Y` 的含义 |
|---|---|---|
| `Chessboard` | `findChessboardCornersSB`，失败再 `findChessboardCorners`，然后 `cornerSubPix` | 内角点个数 |
| `Charuco` | 标记 `CORNER_REFINE_SUBPIX` + `cornerSubPix`，`interpolateCornersCharuco`，再 `cornerSubPix` | 棋盘格子数 |
| `CirclesSymmetric` | `findCirclesGrid` 对称网格 | 圆点行列 |
| `CirclesAsymmetric` | `findCirclesGrid` 非对称网格 | 圆点行列 |
| `ArucoGrid` | 标记 + `cornerSubPix`，按 `GridBoard` 对上 id | 标记行列 |
| `CharucoDiamond` | `detectCharucoDiamond` + `cornerSubPix` | 不参与布局，但仍须 ≥2 |
| `AprilTag` | `DICT_APRILTAG_*`，`CORNER_REFINE_APRILTAG`，再 `cornerSubPix` | 标记行列 |

没有单独的「不对称棋盘」检测器。部分遮挡和强透视用棋盘 SB；棋盘格里嵌标记的混合板用 `Charuco`。圆点不用再套 `cornerSubPix`：`findCirclesGrid` 的斑点中心已经是亚像素。

`aruco_available()` 为 false 时，后四种板画不出来，检测的 `refine` 为 `aruco-unavailable`。棋盘和两种圆点阵不依赖 aruco。

## 亚像素参数

`subpixWin` 是 OpenCV `cornerSubPix` 的**半窗**。默认 5，搜索窗是 11×11。`subpixIters` 默认 40，`subpixEps` 默认 0.001。标记类会把半窗压到不超过 5。`useSb` 默认 true，棋盘先走 SB。

## Python

```python
import cvter

cfg = cvter.BoardConfig()
cfg.type = cvter.BoardType.Charuco
cfg.squaresX = 5
cfg.squaresY = 7
cfg.squareLength = 1.0       # 物体坐标单位
cfg.markerLength = 0.62      # ChArUco 标记边长，须小于方格
cfg.markerSeparation = 0.3   # ArUco / AprilTag 标记间距
cfg.dictionary = -1          # <0：AprilTag 用 36h11，其余用 4X4_50
# cfg.dictionary = cvter.DICT_APRILTAG_36h11

board = cvter.draw_board(cfg, pixels_per_unit=40, margin=30)
det = cvter.detect_board(gray, cfg)
# det.imagePoints Nx2，det.objectPoints Nx3，都是 CV_64F

calib = cvter.calibrate_images(views, cfg)          # 至少 3 张检出
# 或 cvter.calibrate_detections(dets, width, height)
und = cvter.undistort(gray, calib.cameraMatrix, calib.distCoeffs)
cvter.save_calibration("camera.yml", calib)
loaded = cvter.load_calibration("camera.yml")
```

`calibrate_*` 使用 `CALIB_FIX_K3 | CALIB_ZERO_TANGENT_DIST`。视图需要有真实的透视差别；把一张正视图做任意四角拉伸，不能共用一个内参，重投影会偏大。

手眼输入是每帧的 gripper→base 与 target→camera，输出相机到夹爪：

```python
eye = cvter.calibrate_hand_eye(R_g2b, t_g2b, R_t2c, t_t2c, cvter.HandEyeMethod.Park)
print(eye.R, eye.t)
```

方法枚举与 OpenCV 一致：`Tsai`、`Park`、`Horaud`、`Andreff`、`Daniilidis`。

模块上还有 `DICT_4X4_50`、`DICT_5X5_100`、`DICT_APRILTAG_16h5`、`DICT_APRILTAG_25h9`、`DICT_APRILTAG_36h10`、`DICT_APRILTAG_36h11`（编进了 aruco 头文件时）。
