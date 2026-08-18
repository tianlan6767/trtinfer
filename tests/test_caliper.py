#!/usr/bin/env python3
"""cvter 卡尺单元测试：合成图必跑，螺母实图存在则加跑。

用法:
  make test-caliper
  PYTHONPATH=workspace python3.12 tests/test_caliper.py -v
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

NUT_DIR = WORKSPACE / "match_img" / "nut"
NUT_IMG = NUT_DIR / "D4VHW951KR80001PQH+8A31A28SH33AB000000_20260818023454_2_2_1_Color_a.jpg"
NUT_JSON = NUT_DIR / "D4VHW951KR80001PQH+8A31A28SH33AB000000_20260818023454_2_2_1_Color_a.json"
NUT_BATCH = NUT_DIR / "nut_偏移6"


def _blur(img: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    return cv2.GaussianBlur(img, (5, 5), sigma)


def vertical_step(w, h, x_edge, left=200, right=40) -> np.ndarray:
    img = np.empty((h, w), np.uint8)
    img[:, :x_edge] = left
    img[:, x_edge:] = right
    return _blur(img)


def horizontal_step(w, h, y_edge, top=40, bottom=200) -> np.ndarray:
    img = np.empty((h, w), np.uint8)
    img[:y_edge, :] = top
    img[y_edge:, :] = bottom
    return _blur(img)


def dark_disk(size=280, cx=150.0, cy=140.0, r=60.0, bg=200, fg=25) -> np.ndarray:
    img = np.full((size, size), bg, np.uint8)
    yy, xx = np.ogrid[:size, :size]
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = fg
    return _blur(img, 1.0)


def two_falling_edges(w=160, h=180) -> np.ndarray:
    """左亮→中暗→更暗，两条 LightToDark 竖边：x=40（强）、x=100（弱）。"""
    img = np.empty((h, w), np.uint8)
    img[:, :40] = 220
    img[:, 40:100] = 80
    img[:, 100:] = 30
    return _blur(img)


def vert_dev_deg(angle: float) -> float:
    return abs((abs(angle) % 180.0) - 90.0)


class TestFindLineSynthetic(unittest.TestCase):
    def test_vertical_light_to_dark(self):
        x_edge = 70
        img = vertical_step(140, 200, x_edge)
        p = cvter.CaliperParam()
        p.polarity = cvter.CaliperPolarity.LightToDark
        p.contrast = 12
        r = cvter.find_line(img, 40, 20, 110, 180, True, p)
        self.assertTrue(r.found, r)
        self.assertLess(abs(r.center.x - x_edge), 1.0)
        self.assertLess(vert_dev_deg(r.angle), 1.5)
        self.assertLess(r.rms, 0.4)
        self.assertGreater(r.length, 140)
        self.assertGreaterEqual(r.numInliers, 8)

    def test_horizontal_dark_top(self):
        """横边搜索方向是从下到上；上暗下亮对应 LightToDark。"""
        y_edge = 90
        img = horizontal_step(180, 160, y_edge, top=40, bottom=200)
        p = cvter.CaliperParam()
        p.polarity = cvter.CaliperPolarity.LightToDark
        p.contrast = 12
        r = cvter.find_line(img, 20, 40, 160, 130, False, p)
        self.assertTrue(r.found, r)
        self.assertLess(abs(r.center.y - y_edge), 1.0)
        self.assertLess(min(abs(r.angle), abs(abs(r.angle) - 180.0)), 1.5)
        self.assertLess(r.rms, 0.4)

    def test_wrong_polarity_misses(self):
        img = vertical_step(140, 200, 70)
        p = cvter.CaliperParam()
        p.polarity = cvter.CaliperPolarity.DarkToLight
        p.contrast = 18
        r = cvter.find_line(img, 40, 20, 110, 180, True, p)
        self.assertFalse(r.found, r)

    def test_flat_roi_no_edge(self):
        img = np.full((120, 120), 128, np.uint8)
        r = cvter.find_line(img, 20, 20, 100, 100, True, cvter.CaliperParam())
        self.assertFalse(r.found)

    def test_select_first_last_strongest(self):
        img = two_falling_edges()
        base = cvter.CaliperParam()
        base.polarity = cvter.CaliperPolarity.LightToDark
        base.contrast = 10

        first = cvter.CaliperParam()
        first.polarity = base.polarity
        first.contrast = base.contrast
        first.select = cvter.CaliperSelect.First
        rf = cvter.find_line(img, 10, 10, 150, 170, True, first)

        last = cvter.CaliperParam()
        last.polarity = base.polarity
        last.contrast = base.contrast
        last.select = cvter.CaliperSelect.Last
        rl = cvter.find_line(img, 10, 10, 150, 170, True, last)

        strong = cvter.CaliperParam()
        strong.polarity = base.polarity
        strong.contrast = base.contrast
        strong.select = cvter.CaliperSelect.Strongest
        rs = cvter.find_line(img, 10, 10, 150, 170, True, strong)

        self.assertTrue(rf.found and rl.found and rs.found)
        self.assertLess(abs(rf.center.x - 40), 4.0)
        self.assertLess(abs(rl.center.x - 100), 4.0)
        self.assertLess(abs(rs.center.x - 40), 4.0)
        self.assertGreater(rl.center.x - rf.center.x, 40)

    def test_wrapper_matches_free_function(self):
        img = vertical_step(140, 200, 70)
        p = cvter.CaliperParam()
        p.contrast = 12
        a = cvter.find_line(img, 40, 20, 110, 180, True, p)
        b = cvter.CaliperWrapper(p).find_line(img, 40, 20, 110, 180, True)
        self.assertEqual(a.found, b.found)
        self.assertAlmostEqual(a.center.x, b.center.x, places=4)
        self.assertAlmostEqual(a.angle, b.angle, places=4)

    def test_oriented_vertical_phi_90(self):
        x_edge = 70
        img = vertical_step(140, 200, x_edge)
        p = cvter.CaliperParam()
        p.contrast = 12
        r = cvter.find_line_oriented(img, 70, 100, 90.0, 80, 30, p)
        self.assertTrue(r.found, r)
        self.assertLess(abs(r.center.x - x_edge), 1.0)
        self.assertLess(vert_dev_deg(r.angle), 1.5)

    def test_high_contrast_threshold_drops_weak_edge(self):
        img = vertical_step(140, 200, 70, left=160, right=130)
        low = cvter.CaliperParam()
        low.contrast = 8
        high = cvter.CaliperParam()
        high.contrast = 40
        r_ok = cvter.find_line(img, 40, 20, 110, 180, True, low)
        r_miss = cvter.find_line(img, 40, 20, 110, 180, True, high)
        self.assertTrue(r_ok.found, r_ok)
        self.assertFalse(r_miss.found, r_miss)


class TestFindCircleSynthetic(unittest.TestCase):
    def test_dark_hole_inner_to_outer(self):
        cx, cy, rad = 150.0, 140.0, 60.0
        img = dark_disk(280, cx, cy, rad)
        p = cvter.CaliperParam()
        p.polarity = cvter.CaliperPolarity.DarkToLight
        p.contrast = 12
        p.radialInward = False
        r = cvter.find_circle(img, cx + 3, cy - 2, rad, 18, 0, 360, p)
        self.assertTrue(r.found, r)
        self.assertLess(math.hypot(r.center.x - cx, r.center.y - cy), 1.5)
        self.assertLess(abs(r.radius - rad), 1.5)
        self.assertLess(r.rms, 0.5)
        self.assertGreaterEqual(r.numInliers, 20)

    def test_wrong_polarity_or_tiny_search_fails(self):
        img = dark_disk()
        p = cvter.CaliperParam()
        p.polarity = cvter.CaliperPolarity.LightToDark
        p.contrast = 18
        r = cvter.find_circle(img, 150, 140, 60, 18, 0, 360, p)
        self.assertFalse(r.found, r)

    def test_arc_half_circle(self):
        cx, cy, rad = 150.0, 140.0, 60.0
        img = dark_disk(280, cx, cy, rad)
        p = cvter.CaliperParam()
        p.polarity = cvter.CaliperPolarity.DarkToLight
        p.contrast = 12
        r = cvter.find_circle(img, cx, cy, rad, 16, -90, 90, p)
        self.assertTrue(r.found, r)
        self.assertLess(abs(r.radius - rad), 2.0)


@unittest.skipUnless(NUT_IMG.is_file(), "缺少螺母金样图")
class TestNutRealImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        buf = np.fromfile(str(NUT_IMG), dtype=np.uint8)
        cls.gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        assert cls.gray is not None

    def test_two_json_vertical_edges(self):
        p = cvter.CaliperParam()
        p.polarity = cvter.CaliperPolarity.LightToDark
        p.contrast = 18
        lower = cvter.find_line(self.gray, 290, 406, 327, 586, True, p)
        upper = cvter.find_line(self.gray, 363, 107, 424, 291, True, p)
        self.assertTrue(lower.found, lower)
        self.assertTrue(upper.found, upper)
        self.assertLess(vert_dev_deg(lower.angle), 2.0)
        self.assertLess(vert_dev_deg(upper.angle), 2.0)
        self.assertLess(lower.rms, 0.5)
        self.assertLess(upper.rms, 0.5)
        self.assertGreater(lower.length, 120)
        self.assertGreater(upper.length, 120)

    def test_loc_hole_circle(self):
        p = cvter.CaliperParam()
        p.polarity = cvter.CaliperPolarity.DarkToLight
        p.contrast = 12
        r = cvter.find_circle(self.gray, 255, 250, 70, 20, 0, 360, p)
        self.assertTrue(r.found, r)
        self.assertTrue(45 <= r.radius <= 95)
        self.assertTrue(180 <= r.center.x <= 330)
        self.assertTrue(180 <= r.center.y <= 320)
        self.assertLess(r.rms, 1.0)


@unittest.skipUnless(NUT_BATCH.is_dir(), "缺少 nut_偏移6 目录")
class TestNutBatchSmoke(unittest.TestCase):
    def test_first_20_both_rects_found(self):
        jpgs = sorted(p for p in NUT_BATCH.glob("*.jpg") if p.is_file())[:20]
        self.assertGreaterEqual(len(jpgs), 5)
        cal = cvter.CaliperWrapper()
        cal.param.polarity = cvter.CaliperPolarity.LightToDark
        cal.param.contrast = 18
        miss = 0
        for path in jpgs:
            gray = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            for box in ((290, 406, 327, 586), (363, 107, 424, 291)):
                r = cal.find_line(gray, box[0] - 6, box[1] - 6, box[2] + 6, box[3] + 6, True)
                if not r.found:
                    miss += 1
        self.assertEqual(miss, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
