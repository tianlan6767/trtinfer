#!/usr/bin/env python3
"""PATTERN（NCC）可视化，含尺度搜索。尺度对 SHAPE 同样有效，见 test_shape_match_viz.py。"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cvter
import synth
import viz_util as viz

NUT_TPL = ROOT / "workspace" / "match_img" / "nut" / "nut_template.jpg"
NUT_IMG = ROOT / "workspace" / "match_img" / "nut" / (
    "D4VHW951KR80001PQH+8A31A28SH33AB000000_20260818023454_2_2_1_Color_a.jpg"
)


def make_matcher(angle=0, score=0.55, max_count=1, min_area=256, stop_layer=0,
                 min_scale=1.0, max_scale=1.0, scale_step=0.05):
    p = cvter.MatcherParam()
    p.matcherType = cvter.MatcherType.PATTERN
    p.angle = float(angle)
    p.scoreThreshold = float(score)
    p.maxCount = int(max_count)
    p.minArea = float(min_area)
    p.iouThreshold = 0.3
    p.meanBorder = True
    p.stopLayer = int(stop_layer)
    p.minScale = float(min_scale)
    p.maxScale = float(max_scale)
    p.scaleStep = float(scale_step)
    return cvter.MatcherWrapper(p)


def best(results):
    return max(results, key=lambda r: r.Score) if results else None


class TestPatternViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        viz.prepare("pattern")

    def _save(self, name, title, canvas, result, patch=None):
        view = viz.to_bgr(canvas)
        if result is not None:
            viz.draw_match(view, result)
        panels = [("input", canvas), (title, view)]
        if patch is not None:
            panels.append(("template", patch))
        viz.assert_written(self, viz.save_panels("pattern", name, panels))

    def test_unrotated(self):
        patch = synth.make_asymmetric_patch()
        canvas = np.full((220, 280), 40, np.uint8)
        x0, y0 = 70, 55
        synth.paste(canvas, patch, x0, y0)
        m = make_matcher(angle=0, score=0.7)
        m.setTemplate(patch)
        r = best(m.match(canvas))
        self.assertIsNotNone(r)
        self.assertGreater(r.Score, 0.9)
        self.assertLess(abs(r.LeftTop.x - x0), 2.0)
        self.assertLess(abs(r.LeftTop.y - y0), 2.0)
        self.assertAlmostEqual(r.Scale, 1.0, places=3)
        self._save("01_unrotated", "PASS scale=1", canvas, r, patch)

    def test_scaled(self):
        patch = synth.make_asymmetric_patch()
        scale = 1.25
        scaled = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        canvas = np.full((460, 560), 40, np.uint8)
        x0, y0 = 150, 120
        synth.paste(canvas, scaled, x0, y0)
        m = make_matcher(angle=0, score=0.55, min_scale=0.9, max_scale=1.5, scale_step=0.1)
        m.setTemplate(patch)
        r = best(m.match(canvas))
        self.assertIsNotNone(r)
        self.assertGreater(r.Score, 0.7)
        self.assertLess(abs(r.Scale - scale), 0.11)
        cx = x0 + scaled.shape[1] / 2.0
        cy = y0 + scaled.shape[0] / 2.0
        self.assertLess(math.hypot(r.Center.x - cx, r.Center.y - cy), 12.0)
        self._save("02_scale_1.25", f"PASS sc={r.Scale:.2f}", canvas, r, patch)

    def test_two_scales(self):
        """同一张图里两个不同尺度，确认 PATTERN 的 minScale/maxScale 会同时命中。"""
        patch = synth.make_asymmetric_patch()
        canvas = np.full((420, 720), 40, np.uint8)
        targets = []
        for scale, x0, y0 in ((0.9, 40, 80), (1.3, 380, 90)):
            scaled = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            synth.paste(canvas, scaled, x0, y0)
            targets.append((scale, x0 + scaled.shape[1] / 2.0, y0 + scaled.shape[0] / 2.0))
        m = make_matcher(angle=0, score=0.5, max_count=2, min_scale=0.8, max_scale=1.45, scale_step=0.1)
        m.setTemplate(patch)
        hits = m.match(canvas)
        self.assertGreaterEqual(len(hits), 2, hits)
        view = viz.to_bgr(canvas)
        for r in hits:
            viz.draw_match(view, r)
        scales = sorted(r.Scale for r in hits)
        self.assertLess(abs(scales[0] - 0.9), 0.16)
        self.assertLess(abs(scales[-1] - 1.3), 0.16)
        viz.assert_written(self, viz.save_panels("pattern", "03_two_scales", [
            ("two scales", view),
        ]))

    def test_rotated(self):
        patch = synth.make_asymmetric_patch()
        h, w = patch.shape
        canvas = np.full((320, 360), 40, np.uint8)
        cy, cx = 160, 180
        synth.paste(canvas, patch, cx - w // 2, cy - h // 2)
        rot = 8.0
        M = cv2.getRotationMatrix2D((cx, cy), rot, 1.0)
        scene = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=40)
        m = make_matcher(angle=12, score=0.55)
        m.setTemplate(patch)
        r = best(m.match(scene))
        self.assertIsNotNone(r)
        self.assertGreater(r.Score, 0.75)
        self.assertLess(abs(r.Angle - rot), 2.5)
        self._save("04_rotated", f"PASS ang={r.Angle:.1f}", scene, r, patch)

    @unittest.skipUnless(NUT_TPL.is_file() and NUT_IMG.is_file(), "缺少螺母金样")
    def test_nut(self):
        tpl = cv2.imdecode(np.fromfile(str(NUT_TPL), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.imdecode(np.fromfile(str(NUT_IMG), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        m = make_matcher(angle=8, score=0.5)
        m.setTemplate(tpl)
        r = best(m.match(img))
        self.assertIsNotNone(r)
        self.assertGreater(r.Score, 0.5)
        self._save("05_nut", "PASS nut", img, r, tpl)


if __name__ == "__main__":
    unittest.main(verbosity=2)
