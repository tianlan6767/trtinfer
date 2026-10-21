#!/usr/bin/env python3
"""特征粗定位 + ROI 精匹配。合成图，不依赖 AOI 大图。"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cvter
import synth
import viz_util as viz


def make_param(kind="pattern", angle=0.0, margin=120, stop_layer=0, score=0.5):
    p = cvter.CoarseRefineParam()
    p.detector = cvter.FeatureDetector.ORB
    p.margin = int(margin)
    p.angle = float(angle)
    p.stopLayer = int(stop_layer)
    p.scoreThreshold = float(score)
    p.maxCount = 1
    p.minArea = 256
    if kind == "shape":
        p.matcherType = cvter.MatcherType.SHAPE
        p.scoreThreshold = float(score)
    else:
        p.matcherType = cvter.MatcherType.PATTERN
    return p


def paste_scene(origin=(250, 180), canvas_hw=(700, 900)):
    patch = synth.textured_patch(160, 200, 3)
    canvas = np.full(canvas_hw, 128, np.uint8)
    x0, y0 = origin
    synth.paste(canvas, patch, x0, y0)
    return canvas, patch, (x0, y0)


def best(result):
    if not result.found or not result.hits:
        return None
    return max(result.hits, key=lambda r: r.Score)


def assert_pose(test, hit, x0, y0, width, height, tol):
    test.assertLess(abs(hit.LeftTop.x - x0), tol)
    test.assertLess(abs(hit.LeftTop.y - y0), tol)
    test.assertLess(abs(hit.Center.x - (x0 + width / 2.0)), tol)
    test.assertLess(abs(hit.Center.y - (y0 + height / 2.0)), tol)


class TestCoarseRefine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        viz.prepare("coarse_refine")

    def test_pattern_maps_roi_back_to_scene(self):
        scene, patch, (x0, y0) = paste_scene()
        h, w = patch.shape
        param = make_param("pattern", angle=0, margin=120, score=0.7)
        result = cvter.match_coarse_to_fine(scene, patch, param=param)
        self.assertTrue(result.found, result.message)
        self.assertGreaterEqual(result.coarse.inliers, 4)
        self.assertEqual(result.coarse.corners.shape, (4, 2))
        hit = best(result)
        assert_pose(self, hit, x0, y0, w, h, 3.0)
        self.assertLess(abs(hit.Angle), 1.5)
        self.assertAlmostEqual(hit.Scale, 1.0, places=3)
        self.assertGreater(result.roi.x, 0)
        self.assertGreater(result.roi.y, 0)
        self.assertLess(result.roi.x + result.roi.width, scene.shape[1])
        self.assertLess(result.roi.y + result.roi.height, scene.shape[0])
        self.assertLessEqual(result.roi.x, x0)
        self.assertLessEqual(result.roi.y, y0)
        self.assertGreaterEqual(result.roi.x + result.roi.width, x0 + w)
        self.assertGreaterEqual(result.roi.y + result.roi.height, y0 + h)
        corners = np.asarray(result.coarse.corners, np.float64).reshape(-1, 2)
        self.assertTrue(np.all(corners[:, 0] >= result.roi.x - 1.5))
        self.assertTrue(np.all(corners[:, 1] >= result.roi.y - 1.5))
        self.assertTrue(np.all(corners[:, 0] <= result.roi.x + result.roi.width + 1.5))
        self.assertTrue(np.all(corners[:, 1] <= result.roi.y + result.roi.height + 1.5))

        canvas = viz.to_bgr(scene)
        viz.draw_search_rect(
            canvas,
            result.roi.x,
            result.roi.y,
            result.roi.x + result.roi.width,
            result.roi.y + result.roi.height,
            viz.YELLOW,
        )
        viz.draw_match(canvas, hit)
        viz.put_text(
            canvas,
            f"ROI ({result.roi.x},{result.roi.y}) {result.roi.width}x{result.roi.height}",
            (8, canvas.shape[0] - 12),
            viz.YELLOW,
        )
        path = viz.save_panels("coarse_refine", "01_pattern", [("ORB + PATTERN", canvas)])
        viz.assert_written(self, path)

    def test_shape_angle_and_mask(self):
        scene, patch, (x0, y0) = paste_scene()
        h, w = patch.shape
        mask = np.zeros_like(patch)
        mask[6 : h - 6, 6 : w - 6] = 255
        param = make_param("shape", angle=5, margin=80, score=0.4)
        result = cvter.match_coarse_to_fine(scene, patch, mask, param)
        self.assertTrue(result.found, result.message)
        assert_pose(self, best(result), x0, y0, w, h, 8.0)
        canvas = viz.to_bgr(scene)
        viz.draw_match(canvas, best(result), viz.CYAN)
        path = viz.save_panels("coarse_refine", "02_shape_mask", [("ORB + SHAPE", canvas)])
        viz.assert_written(self, path)

    def test_rotated_pattern_uses_angle(self):
        patch = synth.textured_patch(160, 200, 3)
        h, w = patch.shape
        canvas = np.full((700, 900), 128, np.uint8)
        cx, cy = 450.0, 320.0
        x0 = int(round(cx - w / 2.0))
        y0 = int(round(cy - h / 2.0))
        synth.paste(canvas, patch, x0, y0)
        rot = 4.0
        scene = cv2.warpAffine(
            canvas,
            cv2.getRotationMatrix2D((cx, cy), rot, 1.0),
            (canvas.shape[1], canvas.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=128,
        )
        param = make_param("pattern", angle=8, margin=140, score=0.55)
        result = cvter.match_coarse_to_fine(scene, patch, param=param)
        self.assertTrue(result.found, result.message)
        hit = best(result)
        self.assertLess(math.hypot(hit.Center.x - cx, hit.Center.y - cy), 12.0)
        self.assertLess(abs(hit.Angle - rot), 3.0)

    def test_color_and_noncontiguous_view(self):
        scene, patch, (x0, y0) = paste_scene()
        h, w = patch.shape
        param = make_param("pattern", angle=0, margin=80, score=0.7)
        color_scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
        color_patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        color_hit = cvter.match_coarse_to_fine(color_scene, color_patch, param=param)
        self.assertTrue(color_hit.found, color_hit.message)
        assert_pose(self, best(color_hit), x0, y0, w, h, 3.0)

        big = np.full((scene.shape[0] + 80, scene.shape[1] + 90), 30, np.uint8)
        oy, ox = 37, 41
        big[oy : oy + scene.shape[0], ox : ox + scene.shape[1]] = scene
        view = big[oy : oy + scene.shape[0], ox : ox + scene.shape[1]]
        self.assertFalse(view.flags["C_CONTIGUOUS"])
        view_hit = cvter.match_coarse_to_fine(view, patch, param=param)
        self.assertTrue(view_hit.found, view_hit.message)
        assert_pose(self, best(view_hit), x0, y0, w, h, 3.0)

        # 单通道切片步长不是 1，走绑定里的逐元素拷贝。
        channel = color_scene[:, :, 0]
        self.assertFalse(channel.flags["C_CONTIGUOUS"])
        channel_hit = cvter.match_coarse_to_fine(channel, patch, param=param)
        self.assertTrue(channel_hit.found, channel_hit.message)
        assert_pose(self, best(channel_hit), x0, y0, w, h, 3.0)

    def test_matcher_accepts_noncontiguous_crop(self):
        scene, patch, (x0, y0) = paste_scene()
        h, w = patch.shape
        pad = 28
        crop = scene[y0 - pad : y0 + h + pad, x0 - pad : x0 + w + pad]
        self.assertFalse(crop.flags["C_CONTIGUOUS"])
        p = cvter.MatcherParam()
        p.matcherType = cvter.MatcherType.PATTERN
        p.angle = 0
        p.scoreThreshold = 0.7
        p.maxCount = 1
        p.minArea = 256
        matcher = cvter.MatcherWrapper(p)
        matcher.setTemplate(patch)
        hits = matcher.match(crop)
        self.assertTrue(hits, "非连续 ROI 应仍能匹配，而不是 0 命中")
        hit = max(hits, key=lambda r: r.Score)
        self.assertLess(abs(hit.LeftTop.x - pad), 3.0)
        self.assertLess(abs(hit.LeftTop.y - pad), 3.0)

    def test_stop_layer_one_still_finds(self):
        scene, patch, (x0, y0) = paste_scene()
        h, w = patch.shape
        param = make_param("pattern", angle=0, margin=80, stop_layer=1, score=0.45)
        result = cvter.match_coarse_to_fine(scene, patch, param=param)
        self.assertTrue(result.found, result.message)
        assert_pose(self, best(result), x0, y0, w, h, 8.0)

    def test_missing_and_bad_mask(self):
        scene, patch, _ = paste_scene()
        blank = np.full_like(scene, 40)
        param = make_param("pattern", angle=0, score=0.5)
        missed = cvter.match_coarse_to_fine(blank, patch, param=param)
        self.assertFalse(missed.found)
        self.assertEqual(len(missed.hits), 0)
        self.assertTrue(missed.message)

        bad = np.zeros((patch.shape[0] - 1, patch.shape[1]), np.uint8)
        rejected = cvter.match_coarse_to_fine(scene, patch, bad, param)
        self.assertFalse(rejected.found)
        self.assertIn("mask", rejected.message)

    def test_bench_script_skips_without_images(self):
        env = os.environ.copy()
        for key in ("CVTER_AOI_SCENE", "CVTER_AOI_TEMPL", "CVTER_AOI_MASK"):
            env.pop(key, None)
        script = Path(__file__).resolve().parent / "bench_aoi_coarse_refine.py"
        proc = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("skip", proc.stdout)

    def test_sift_detector(self):
        if not cvter.sift_available():
            self.skipTest("SIFT 不可用")
        scene, patch, (x0, y0) = paste_scene()
        h, w = patch.shape
        param = make_param("pattern", angle=0, margin=120, score=0.7)
        param.detector = cvter.FeatureDetector.SIFT
        result = cvter.match_coarse_to_fine(scene, patch, param=param)
        self.assertTrue(result.found, result.message)
        self.assertEqual(result.coarse.method, "SIFT")
        assert_pose(self, best(result), x0, y0, w, h, 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
