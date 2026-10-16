#!/usr/bin/env python3
"""矩形卡尺与测量辅助。需要先 make 出 workspace/cvter.so。

不依赖 GPU 的同内容测试：make test-rect
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workspace"))

try:
    import cvter
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"无法 import cvter（需要先 make）: {exc}") from exc


def paint_rect(img, cx, cy, w, h, angle_deg, value):
    phi = math.radians(angle_deg)
    ux, uy = math.cos(phi), math.sin(phi)
    vx, vy = -math.sin(phi), math.cos(phi)
    hw, hh = 0.5 * w, 0.5 * h
    pts = np.array([
        [cx - ux * hw - vx * hh, cy - uy * hw - vy * hh],
        [cx + ux * hw - vx * hh, cy + uy * hw - vy * hh],
        [cx + ux * hw + vx * hh, cy + uy * hw + vy * hh],
        [cx - ux * hw + vx * hh, cy - uy * hw + vy * hh],
    ], np.int32)
    cv2.fillConvexPoly(img, pts, int(value))


class TestRectMetrology(unittest.TestCase):
    def test_helpers(self):
        self.assertAlmostEqual(cvter.distance_points(0, 0, 3, 4), 5.0)
        self.assertAlmostEqual(cvter.distance_point_line(0, 1, 0, 0, 10, 0), 1.0)
        self.assertAlmostEqual(cvter.angle_lines(0, 0, 10, 0, 0, 0, 0, 10), 90.0)
        summary = cvter.summarize_rectangle(0, 0, 30, 0, 30, 10, 0, 10)
        self.assertTrue(summary.found)
        self.assertAlmostEqual(summary.width, 30.0)
        self.assertAlmostEqual(summary.height, 10.0)
        circle = cvter.summarize_circle(4, 5, 7)
        self.assertAlmostEqual(circle.diameter, 14.0)

    def test_find_rotated_rectangle(self):
        img = np.full((240, 320), 30, np.uint8)
        paint_rect(img, 160, 110, 150, 90, 7.0, 210)
        img = cv2.GaussianBlur(img, (5, 5), 1.0)
        param = cvter.CaliperParam()
        param.polarity = cvter.CaliperPolarity.DarkToLight
        param.contrast = 12
        param.stride = 2
        param.projection = 3
        param.minPoints = 6
        hit = cvter.find_rectangle(img, 163, 108, 150, 90, 7.0, 16.0, param)
        self.assertTrue(hit.found)
        self.assertLess(math.hypot(hit.center.x - 160, hit.center.y - 110), 2.5)
        self.assertLess(abs(hit.width - 150), 3.0)
        self.assertLess(abs(hit.height - 90), 3.0)
        self.assertLess(abs(hit.angle - 7.0), 1.5)

        axis = np.full((200, 260), 25, np.uint8)
        cv2.rectangle(axis, (40, 30), (159, 99), 200, -1)
        axis = cv2.GaussianBlur(axis, (3, 3), 0.8)
        box = cvter.find_rectangle_box(axis, 40, 30, 160, 100, 12.0, param)
        self.assertTrue(box.found)
        self.assertLess(math.hypot(box.center.x - 100, box.center.y - 65), 2.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
