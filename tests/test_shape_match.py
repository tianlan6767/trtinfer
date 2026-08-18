#!/usr/bin/env python3
"""形状匹配测试（MatcherType.SHAPE）。"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "workspace"
sys.path.insert(0, str(WORKSPACE))

try:
    import cvter
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"无法 import cvter: {exc}") from exc

NUT_TPL = WORKSPACE / "match_img" / "nut" / "nut_template.jpg"
NUT_IMG = WORKSPACE / "match_img" / "nut" / (
    "D4VHW951KR80001PQH+8A31A28SH33AB000000_20260818023454_2_2_1_Color_a.jpg"
)


def make_shape_matcher(angle=0, score=0.45, max_count=1, min_area=256):
    p = cvter.MatcherParam()
    p.matcherType = cvter.MatcherType.SHAPE
    p.angle = float(angle)
    p.scoreThreshold = float(score)
    p.maxCount = int(max_count)
    p.minArea = float(min_area)
    p.edgeMinMag = 12
    p.maxEdgePoints = 280
    p.usePolarity = True
    p.greediness = 0.85
    return cvter.MatcherWrapper(p)


def make_l_shape(h=64, w=80):
    img = np.full((h, w), 30, np.uint8)
    img[8:h - 8, 8:24] = 220
    img[h - 24 : h - 8, 8 : w - 8] = 220
    return img


def paste(dst, patch, x, y):
    h, w = patch.shape
    dst[y : y + h, x : x + w] = patch


def best(results):
    if not results:
        return None
    return max(results, key=lambda r: r.Score)


class TestShapeSynthetic(unittest.TestCase):
    def test_find_unrotated(self):
        patch = make_l_shape()
        canvas = np.full((240, 300), 30, np.uint8)
        x0, y0 = 90, 70
        paste(canvas, patch, x0, y0)
        m = make_shape_matcher(angle=0, score=0.4)
        m.setTemplate(patch)
        r = best(m.match(canvas))
        self.assertIsNotNone(r)
        self.assertGreater(r.Score, 0.55)
        self.assertLess(abs(r.Center.x - (x0 + patch.shape[1] / 2)), 4.0)
        self.assertLess(abs(r.Center.y - (y0 + patch.shape[0] / 2)), 4.0)
        self.assertLess(abs(r.Angle), 2.5)

    def test_find_rotated(self):
        patch = make_l_shape()
        h, w = patch.shape
        canvas = np.full((340, 380), 30, np.uint8)
        cy, cx = 170, 190
        paste(canvas, patch, cx - w // 2, cy - h // 2)
        rot = 8.0
        M = cv2.getRotationMatrix2D((cx, cy), rot, 1.0)
        scene = cv2.warpAffine(
            canvas, M, (canvas.shape[1], canvas.shape[0]),
            borderMode=cv2.BORDER_CONSTANT, borderValue=30,
        )
        m = make_shape_matcher(angle=14, score=0.35)
        m.setTemplate(patch)
        r = best(m.match(scene))
        self.assertIsNotNone(r)
        self.assertGreater(r.Score, 0.4)
        self.assertLess(abs(r.Angle - rot), 3.0)
        self.assertLess(math.hypot(r.Center.x - cx, r.Center.y - cy), 10.0)

    def test_robust_to_brightness(self):
        patch = make_l_shape()
        canvas = np.full((240, 300), 30, np.uint8)
        paste(canvas, patch, 80, 60)
        bright = cv2.add(canvas, 60)
        m = make_shape_matcher(angle=0, score=0.4)
        m.setTemplate(patch)
        r = best(m.match(bright))
        self.assertIsNotNone(r)
        self.assertGreater(r.Score, 0.5)

    def test_mask_keeps_outer_edges(self):
        patch = make_l_shape()
        h, w = patch.shape
        noisy = patch.copy()
        cv2.randu(noisy[20:h - 28, 28:w - 12], 0, 255)
        canvas = np.full((240, 300), 30, np.uint8)
        x0, y0 = 70, 50
        paste(canvas, noisy, x0, y0)
        mask = np.zeros_like(patch)
        mask[8:h - 8, 8:24] = 255
        mask[h - 24 : h - 8, 8 : w - 8] = 255
        m = make_shape_matcher(angle=0, score=0.35)
        m.setTemplate(noisy, mask)
        r = best(m.match(canvas))
        self.assertIsNotNone(r)
        self.assertLess(abs(r.Center.x - (x0 + w / 2)), 8.0)


@unittest.skipUnless(NUT_TPL.is_file() and NUT_IMG.is_file(), "缺少螺母图")
class TestShapeNut(unittest.TestCase):
    def test_nut_template(self):
        tpl = cv2.imdecode(np.fromfile(str(NUT_TPL), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.imdecode(np.fromfile(str(NUT_IMG), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        m = make_shape_matcher(angle=10, score=0.35, min_area=256)
        m.setTemplate(tpl)
        r = best(m.match(img))
        self.assertIsNotNone(r)
        self.assertGreater(r.Score, 0.35)
        self.assertTrue(80 < r.Center.x < 360)
        self.assertTrue(250 < r.Center.y < 640)


if __name__ == "__main__":
    unittest.main(verbosity=2)
