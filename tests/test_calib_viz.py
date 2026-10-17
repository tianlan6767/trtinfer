#!/usr/bin/env python3
"""标定板检测（默认亚像素）、标定、去畸变、YAML、手眼。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cvter
import viz_util as viz


def shift_subpix(img, dx=0.37, dy=0.21):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=255)


def perspective_views(img):
    """用同一针孔相机、不同外参把标定板投到图像上。任意四角透视不能共一个 K。"""
    h, w = img.shape[:2]
    focal = float(max(h, w) * 1.2)
    K = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], np.float64)
    shift = np.array([[1, 0, -w / 2.0], [0, 1, -h / 2.0], [0, 0, 1]], np.float64)
    angles = [
        (0.22, -0.18, 0.04),
        (-0.26, 0.20, -0.05),
        (0.16, 0.28, 0.06),
        (-0.20, -0.24, 0.03),
        (0.10, -0.12, -0.08),
    ]
    views = []
    for ax, ay, az in angles:
        R, _ = cv2.Rodrigues(np.array([ax, ay, az], np.float64))
        t = np.array([[0.0], [0.0], [focal * 0.95]], np.float64)
        H = K @ np.hstack([R[:, :2], t]) @ shift
        views.append(cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=255))
    return views


def overlay_detection(image, det):
    view = viz.to_bgr(image)
    pts = viz.as_points(det.imagePoints)
    viz.draw_points(view, pts, viz.GREEN, 3)
    if len(pts):
        viz.draw_cross(view, pts[0, 0], pts[0, 1], viz.RED, 12, 1)
    tag = "subpix" if det.subpixel else "integer"
    viz.put_text(view, f"{det.board} n={det.count} {tag}", (8, 18), viz.GREEN, 0.45)
    viz.put_text(view, det.refine[:80], (8, view.shape[0] - 8), viz.YELLOW, 0.35)
    return view


def distort_image(img, K, dist):
    h, w = img.shape[:2]
    ys, xs = np.indices((h, w), dtype=np.float32)
    pts = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    mapx = und[:, 0, 0].reshape(h, w).astype(np.float32)
    mapy = und[:, 0, 1].reshape(h, w).astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


class TestCalibViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        viz.prepare("calib")

    def _cfg(self, kind, sx, sy, **kw):
        cfg = cvter.BoardConfig()
        cfg.type = kind
        cfg.squaresX = sx
        cfg.squaresY = sy
        cfg.squareLength = kw.get("square", 1.0)
        cfg.markerLength = kw.get("marker", 0.62)
        cfg.markerSeparation = kw.get("sep", 0.28)
        cfg.subpixWin = 5
        cfg.subpixIters = 40
        cfg.subpixEps = 0.001
        if "dictionary" in kw:
            cfg.dictionary = kw["dictionary"]
        return cfg

    def _run_board(self, name, cfg, pixels, margin=36, rms_limit=1.0):
        board = cvter.draw_board(cfg, pixels, margin)
        self.assertGreater(board.size, 0, name)
        shifted = shift_subpix(board)
        det = cvter.detect_board(shifted, cfg)
        self.assertTrue(det.found, f"{name} {det.refine}")
        self.assertTrue(det.subpixel, det.refine)
        self.assertIn("subpix", det.refine.lower())
        self.assertGreaterEqual(det.count, 4)
        self.assertGreater(viz.fractional_max(det.imagePoints), 0.03, det.refine)
        views = perspective_views(board)
        calib = cvter.calibrate_images(views, cfg)
        self.assertTrue(calib.found, name)
        self.assertLess(calib.rms, rms_limit, f"{name} rms={calib.rms}")
        self.assertEqual(calib.cameraMatrix.shape, (3, 3))
        overlay = overlay_detection(shifted, det)
        zoom = viz.zoom_inset(overlay, det.imagePoints[0, 0], det.imagePoints[0, 1])
        und = cvter.undistort(views[0], calib.cameraMatrix, calib.distCoeffs)
        viz.assert_written(self, viz.save_panels("calib", name, [
            ("board", board),
            (f"PASS subpix rms={calib.rms:.3f}", overlay),
            ("corner zoom", zoom),
            ("undistort view0", und),
        ]))

    def test_chessboard(self):
        cfg = self._cfg(cvter.BoardType.Chessboard, 7, 5)
        self._run_board("01_chessboard", cfg, 36)

    def test_circles_symmetric(self):
        cfg = self._cfg(cvter.BoardType.CirclesSymmetric, 4, 4)
        self._run_board("02_circles_sym", cfg, 48)

    def test_circles_asymmetric(self):
        cfg = self._cfg(cvter.BoardType.CirclesAsymmetric, 4, 5)
        self._run_board("03_circles_asym", cfg, 36)

    def test_charuco(self):
        if not cvter.aruco_available():
            self.skipTest("无 aruco")
        cfg = self._cfg(cvter.BoardType.Charuco, 5, 7, marker=0.62)
        self._run_board("04_charuco", cfg, 46, rms_limit=1.0)

    def test_aruco_grid(self):
        if not cvter.aruco_available():
            self.skipTest("无 aruco")
        cfg = self._cfg(cvter.BoardType.ArucoGrid, 3, 3, marker=1.0, sep=0.3)
        self._run_board("05_aruco", cfg, 60, rms_limit=1.0)

    def test_charuco_diamond(self):
        if not cvter.aruco_available():
            self.skipTest("无 aruco")
        cfg = self._cfg(cvter.BoardType.CharucoDiamond, 3, 3, marker=0.62)
        self._run_board("06_diamond", cfg, 80, margin=20, rms_limit=1.0)

    def test_apriltag(self):
        if not cvter.aruco_available():
            self.skipTest("无 aruco")
        cfg = self._cfg(cvter.BoardType.AprilTag, 3, 3, marker=1.0, sep=0.35, dictionary=cvter.DICT_APRILTAG_36h11)
        self._run_board("07_apriltag", cfg, 70, rms_limit=1.0)

    def test_undistort_and_yaml(self):
        h, w = 240, 320
        img = np.full((h, w), 255, np.uint8)
        for x in range(16, w, 16):
            cv2.line(img, (x, 0), (x, h - 1), 0, 1)
        for y in range(16, h, 16):
            cv2.line(img, (0, y), (w - 1, y), 0, 1)
        K = np.array([[300.0, 0, w / 2], [0, 300.0, h / 2], [0, 0, 1]], np.float64)
        D = np.array([[-0.28, 0.08, 0, 0, 0]], np.float64)
        distorted = distort_image(img, K, D)
        und = cvter.undistort(distorted, K, D)
        center = und[50:h - 50, 50:w - 50]
        ref = img[50:h - 50, 50:w - 50]
        err = float(np.mean(np.abs(center.astype(np.int16) - ref.astype(np.int16))))
        self.assertLess(err, 18.0, err)
        result = cvter.CalibResult()
        result.found = True
        result.rms = 0.2
        result.width = w
        result.height = h
        result.cameraMatrix = K
        result.distCoeffs = D
        path = viz.OUT_DIR / "calib" / "camera.yml"
        self.assertTrue(cvter.save_calibration(str(path), result))
        loaded = cvter.load_calibration(str(path))
        self.assertTrue(loaded.found)
        np.testing.assert_allclose(np.asarray(loaded.cameraMatrix), K, atol=1e-6)
        np.testing.assert_allclose(np.asarray(loaded.distCoeffs).reshape(-1), D.reshape(-1), atol=1e-6)
        viz.assert_written(self, viz.save_panels("calib", "08_undistort", [
            ("ideal", img),
            ("distorted", distorted),
            ("PASS undistort", und),
        ]))

    def test_hand_eye(self):
        rng = np.random.RandomState(2)

        def rand_rt():
            R, _ = cv2.Rodrigues(rng.normal(size=3).astype(np.float64) * 0.5)
            t = rng.normal(size=(3, 1)).astype(np.float64) * 0.3
            return R, t

        def inv_rt(R, t):
            return R.T, -R.T @ t

        def mul_rt(R1, t1, R2, t2):
            return R1 @ R2, R1 @ t2 + t1

        Rx, tx = rand_rt()
        const_R, const_t = rand_rt()
        Rg, tg, Rt, tt = [], [], [], []
        for _ in range(6):
            G_R, G_t = rand_rt()
            # C = inv(X) * inv(G) * const
            iX_R, iX_t = inv_rt(Rx, tx)
            iG_R, iG_t = inv_rt(G_R, G_t)
            A_R, A_t = mul_rt(iX_R, iX_t, iG_R, iG_t)
            C_R, C_t = mul_rt(A_R, A_t, const_R, const_t)
            Rg.append(G_R)
            tg.append(G_t)
            Rt.append(C_R)
            tt.append(C_t)
        eye = cvter.calibrate_hand_eye(Rg, tg, Rt, tt, cvter.HandEyeMethod.Park)
        self.assertTrue(eye.found, "hand-eye")
        R = np.asarray(eye.R, np.float64).reshape(3, 3)
        t = np.asarray(eye.t, np.float64).reshape(3, 1)
        self.assertLess(np.linalg.norm(R - Rx), 0.05, R)
        self.assertLess(np.linalg.norm(t - tx), 0.05, t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
