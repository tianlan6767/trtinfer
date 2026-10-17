#!/usr/bin/env python3
"""ORB / SIFT 特征匹配 + 单应性。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cvter
import synth
import viz_util as viz


def warp_scene(model):
    h, w = model.shape[:2]
    canvas_w, canvas_h = w + 160, h + 140
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[70, 40], [canvas_w - 50, 70], [canvas_w - 80, canvas_h - 40], [40, canvas_h - 70]])
    M = cv2.getPerspectiveTransform(src, dst)
    scene = cv2.warpPerspective(model, M, (canvas_w, canvas_h), borderValue=235)
    return scene, dst


def draw_feature(model, scene, result):
    a = viz.to_bgr(model)
    b = viz.to_bgr(scene)
    height = max(a.shape[0], b.shape[0])
    canvas = np.full((height, a.shape[1] + b.shape[1], 3), 30, np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1]:a.shape[1] + b.shape[1]] = b
    off = a.shape[1]
    mp = viz.as_points(result.modelPoints)
    sp = viz.as_points(result.scenePoints)
    mask = list(result.inlierMask)
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(mp, sp)):
        inn = i < len(mask) and mask[i]
        color = viz.GREEN if inn else (80, 80, 80)
        cv2.line(canvas, (int(round(x0)), int(round(y0))),
                 (int(round(x1)) + off, int(round(y1))), color, 1, cv2.LINE_AA)
    corners = viz.as_points(result.corners)
    if len(corners) == 4:
        shifted = corners.copy()
        shifted[:, 0] += off
        viz.draw_poly(canvas, shifted, viz.ORANGE, 2)
    viz.put_text(canvas, f"{result.method} inliers={result.inliers}/{result.matches}", (8, 18), viz.GREEN)
    return canvas


class TestFeatureViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        viz.prepare("feature")

    def _check(self, detector, name, dst):
        model = synth.textured_patch()
        scene, _ = warp_scene(model)
        result = cvter.match_features(scene, model, detector, 0.8, 4.0)
        self.assertTrue(result.found, f"{name} {result.method} matches={result.matches}")
        self.assertGreaterEqual(result.inliers, 8)
        corners = viz.as_points(result.corners)
        self.assertEqual(len(corners), 4)
        # 角点顺序可能与 dst 不同，用最近邻误差
        err = []
        for p in corners:
            err.append(np.min(np.linalg.norm(dst - p, axis=1)))
        self.assertLess(max(err), 12.0, err)
        canvas = draw_feature(model, scene, result)
        viz.assert_written(self, viz.save_panels("feature", name, [(result.method, canvas)]))

    def test_orb(self):
        model = synth.textured_patch()
        _, dst = warp_scene(model)
        self._check(cvter.FeatureDetector.ORB, "01_orb", dst)

    def test_sift(self):
        if not cvter.sift_available():
            self.skipTest("SIFT 不可用")
        model = synth.textured_patch()
        _, dst = warp_scene(model)
        self._check(cvter.FeatureDetector.SIFT, "02_sift", dst)


if __name__ == "__main__":
    unittest.main(verbosity=2)
