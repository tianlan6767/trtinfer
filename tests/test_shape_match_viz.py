#!/usr/bin/env python3
"""SHAPE 可视化。尺度搜索与 PATTERN 使用同一组 minScale/maxScale/scaleStep/numScales。"""
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


def make_shape(angle=0, score=0.45, max_count=1, min_scale=1.0, max_scale=1.0, scale_step=0.05):
    p = cvter.MatcherParam()
    p.matcherType = cvter.MatcherType.SHAPE
    p.angle = float(angle)
    p.scoreThreshold = float(score)
    p.maxCount = int(max_count)
    p.minArea = 256
    p.edgeMinMag = 12
    p.maxEdgePoints = 280
    p.usePolarity = True
    p.greediness = 0.85
    p.minScale = float(min_scale)
    p.maxScale = float(max_scale)
    p.scaleStep = float(scale_step)
    return cvter.MatcherWrapper(p)


def best(results):
    return max(results, key=lambda r: r.Score) if results else None


class TestShapeViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        viz.prepare("shape")

    def _save(self, name, title, canvas, result, patch=None):
        view = viz.to_bgr(canvas)
        if result is not None:
            viz.draw_match(view, result)
        panels = [("input", canvas), (title, view)]
        if patch is not None:
            panels.append(("template", patch))
        viz.assert_written(self, viz.save_panels("shape", name, panels))

    def test_unrotated(self):
        patch = synth.make_l_shape()
        canvas = np.full((240, 300), 30, np.uint8)
        x0, y0 = 90, 70
        synth.paste(canvas, patch, x0, y0)
        m = make_shape(angle=0, score=0.4)
        m.setTemplate(patch)
        r = best(m.match(canvas))
        self.assertIsNotNone(r)
        self.assertGreater(r.Score, 0.55)
        self.assertAlmostEqual(r.Scale, 1.0, places=3)
        self.assertLess(abs(r.Center.x - (x0 + patch.shape[1] / 2)), 4.0)
        self._save("01_unrotated", "PASS scale=1", canvas, r, patch)

    def test_scaled(self):
        patch = synth.make_l_shape()
        scale = 1.25
        scaled = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        canvas = np.full((420, 480), 30, np.uint8)
        x0, y0 = 140, 110
        synth.paste(canvas, scaled, x0, y0)
        m = make_shape(angle=0, score=0.4, min_scale=0.9, max_scale=1.5, scale_step=0.1)
        m.setTemplate(patch)
        r = best(m.match(canvas))
        self.assertIsNotNone(r)
        self.assertGreater(r.Score, 0.45)
        self.assertLess(abs(r.Scale - scale), 0.12)
        cx = x0 + scaled.shape[1] / 2.0
        cy = y0 + scaled.shape[0] / 2.0
        self.assertLess(math.hypot(r.Center.x - cx, r.Center.y - cy), 12.0)
        self._save("02_scale_1.25", f"PASS sc={r.Scale:.2f}", canvas, r, patch)

    def test_two_scales(self):
        patch = synth.make_l_shape()
        canvas = np.full((400, 680), 30, np.uint8)
        for scale, x0, y0 in ((0.9, 30, 70), (1.3, 360, 80)):
            scaled = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            synth.paste(canvas, scaled, x0, y0)
        m = make_shape(angle=0, score=0.35, max_count=2, min_scale=0.8, max_scale=1.45, scale_step=0.1)
        m.setTemplate(patch)
        hits = m.match(canvas)
        self.assertGreaterEqual(len(hits), 2, hits)
        view = viz.to_bgr(canvas)
        for r in hits:
            viz.draw_match(view, r)
        scales = sorted(h.Scale for h in hits)
        self.assertLess(abs(scales[0] - 0.9), 0.16)
        self.assertLess(abs(scales[-1] - 1.3), 0.16)
        viz.assert_written(self, viz.save_panels("shape", "03_two_scales", [("two scales", view)]))

    def test_rotated(self):
        patch = synth.make_l_shape()
        h, w = patch.shape
        canvas = np.full((340, 380), 30, np.uint8)
        cy, cx = 170, 190
        synth.paste(canvas, patch, cx - w // 2, cy - h // 2)
        rot = 8.0
        M = cv2.getRotationMatrix2D((cx, cy), rot, 1.0)
        scene = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=30)
        m = make_shape(angle=14, score=0.35)
        m.setTemplate(patch)
        r = best(m.match(scene))
        self.assertIsNotNone(r)
        self.assertGreater(r.Score, 0.4)
        self.assertLess(abs(r.Angle - rot), 3.0)
        self._save("04_rotated", f"PASS ang={r.Angle:.1f}", scene, r, patch)

    @unittest.skipUnless(NUT_TPL.is_file() and NUT_IMG.is_file(), "缺少螺母图")
    def test_nut(self):
        tpl = cv2.imdecode(np.fromfile(str(NUT_TPL), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.imdecode(np.fromfile(str(NUT_IMG), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        m = make_shape(angle=10, score=0.35)
        m.setTemplate(tpl)
        r = best(m.match(img))
        self.assertIsNotNone(r)
        self._save("05_nut", "PASS nut", img, r, tpl)


if __name__ == "__main__":
    unittest.main(verbosity=2)
