#!/usr/bin/env python3
"""cvter 灰度模板匹配测试（对应 docs/cv_match_update.md）。

用法:
  make test-match
  PYTHONPATH=workspace python3.12 tests/test_pattern_match.py -v
"""
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
    raise SystemExit(
        f"无法 import cvter（需要先 make 生成 workspace/cvter.so）: {exc}"
    ) from exc

NUT_TPL = WORKSPACE / "match_img" / "nut" / "nut_template.jpg"
NUT_IMG = WORKSPACE / "match_img" / "nut" / (
    "D4VHW951KR80001PQH+8A31A28SH33AB000000_20260818023454_2_2_1_Color_a.jpg"
)


def make_matcher(angle=0, score=0.55, max_count=1, min_area=256, stop_layer=0, mean_border=True):
    p = cvter.MatcherParam()
    p.matcherType = cvter.MatcherType.PATTERN
    p.angle = float(angle)
    p.scoreThreshold = float(score)
    p.maxCount = int(max_count)
    p.minArea = float(min_area)
    p.iouThreshold = 0.3
    p.meanBorder = bool(mean_border)
    p.stopLayer = int(stop_layer)
    return cvter.MatcherWrapper(p)


def make_asymmetric_patch(h=56, w=88):
    """非旋转对称块：亮矩形 + 左上缺口，避免 180° 歧义。"""
    img = np.full((h, w), 40, np.uint8)
    img[8:h - 8, 8:w - 8] = 210
    img[8:22, 8:28] = 40
    img[h - 20 : h - 8, w - 36 : w - 8] = 90
    return img


def paste(dst, patch, x, y):
    h, w = patch.shape
    dst[y : y + h, x : x + w] = patch


def best(results):
    if not results:
        return None
    return max(results, key=lambda r: r.Score)


class TestPatternMatchSynthetic(unittest.TestCase):
    def test_find_unrotated(self):
        patch = make_asymmetric_patch()
        canvas = np.full((220, 280), 40, np.uint8)
        x0, y0 = 70, 55
        paste(canvas, patch, x0, y0)
        m = make_matcher(angle=0, score=0.7)
        m.setTemplate(patch)
        hits = m.match(canvas)
        r = best(hits)
        self.assertIsNotNone(r, "0° 应匹配到")
        self.assertGreater(r.Score, 0.9)
        self.assertLess(abs(r.LeftTop.x - x0), 2.0)
        self.assertLess(abs(r.LeftTop.y - y0), 2.0)
        self.assertLess(abs(r.Center.x - (x0 + patch.shape[1] / 2)), 2.5)
        self.assertLess(abs(r.Angle), 1.5)

    def test_find_rotated(self):
        patch = make_asymmetric_patch()
        h, w = patch.shape
        canvas = np.full((320, 360), 40, np.uint8)
        cy, cx = 160, 180
        y0, x0 = cy - h // 2, cx - w // 2
        paste(canvas, patch, x0, y0)
        rot = 8.0
        M = cv2.getRotationMatrix2D((cx, cy), rot, 1.0)
        scene = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=40)
        m = make_matcher(angle=12, score=0.55)
        m.setTemplate(patch)
        r = best(m.match(scene))
        self.assertIsNotNone(r, "带角度应匹配到")
        self.assertGreater(r.Score, 0.75)
        self.assertLess(abs(r.Angle - rot), 2.5)
        self.assertLess(math.hypot(r.Center.x - cx, r.Center.y - cy), 8.0)

    def test_mask_ignores_center_noise(self):
        patch = make_asymmetric_patch()
        h, w = patch.shape
        noisy = patch.copy()
        cv2.randu(noisy[18:h - 18, 22:w - 22], 0, 255)
        canvas = np.full((220, 280), 40, np.uint8)
        x0, y0 = 64, 50
        paste(canvas, noisy, x0, y0)

        mask = np.zeros_like(patch)
        mask[6:h - 6, 6:12] = 255
        mask[6:h - 6, w - 12 : w - 6] = 255
        mask[6:12, 6:w - 6] = 255
        mask[h - 12 : h - 6, 6:w - 6] = 255
        mask[8:22, 8:28] = 255

        m = make_matcher(angle=0, score=0.45)
        m.setTemplate(noisy, mask)
        r = best(m.match(canvas))
        self.assertIsNotNone(r, "mask 只留外沿时应仍能定位")
        self.assertLess(abs(r.LeftTop.x - x0), 4.0)
        self.assertLess(abs(r.LeftTop.y - y0), 4.0)

    def test_stop_layer_and_mean_border_still_find(self):
        patch = make_asymmetric_patch()
        canvas = np.full((220, 280), 40, np.uint8)
        paste(canvas, patch, 80, 60)
        m = make_matcher(angle=0, score=0.6, stop_layer=0, mean_border=True)
        m.setTemplate(patch)
        self.assertIsNotNone(best(m.match(canvas)))
        m2 = make_matcher(angle=0, score=0.45, stop_layer=1, mean_border=False)
        m2.setTemplate(patch)
        r = best(m2.match(canvas))
        self.assertIsNotNone(r)
        self.assertGreater(r.Score, 0.5)

    def test_bad_mask_size_does_not_learn(self):
        patch = make_asymmetric_patch()
        canvas = np.full((180, 220), 40, np.uint8)
        paste(canvas, patch, 40, 30)
        m = make_matcher()
        m.setTemplate(patch, np.ones((10, 10), np.uint8) * 255)
        hits = m.match(canvas)
        self.assertEqual(len(hits), 0)


@unittest.skipUnless(NUT_TPL.is_file() and NUT_IMG.is_file(), "缺少螺母金样/图")
class TestPatternMatchNut(unittest.TestCase):
    def test_template_on_labeled_image(self):
        tpl = cv2.imdecode(np.fromfile(str(NUT_TPL), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.imdecode(np.fromfile(str(NUT_IMG), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(tpl)
        self.assertIsNotNone(img)
        m = make_matcher(angle=8, score=0.5, min_area=256)
        m.setTemplate(tpl)
        r = best(m.match(img))
        self.assertIsNotNone(r, "金样应在标注图上匹配到螺母")
        self.assertGreater(r.Score, 0.5)
        self.assertTrue(80 < r.Center.x < 360)
        self.assertTrue(250 < r.Center.y < 620)


if __name__ == "__main__":
    unittest.main(verbosity=2)
