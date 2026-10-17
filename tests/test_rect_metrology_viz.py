#!/usr/bin/env python3
"""矩形卡尺与测量辅助的可视化。"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cvter
import synth
import viz_util as viz


class TestRectViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        viz.prepare("rect")

    def test_helpers_diagram(self):
        self.assertAlmostEqual(cvter.distance_points(0, 0, 3, 4), 5.0)
        self.assertAlmostEqual(cvter.distance_point_line(0, 1, 0, 0, 10, 0), 1.0)
        self.assertAlmostEqual(cvter.angle_lines(0, 0, 10, 0, 0, 0, 0, 10), 90.0)
        summary = cvter.summarize_rectangle(0, 0, 30, 0, 30, 10, 0, 10)
        self.assertTrue(summary.found)
        self.assertAlmostEqual(summary.width, 30.0)
        self.assertAlmostEqual(summary.height, 10.0)
        circle = cvter.summarize_circle(4, 5, 7)
        self.assertAlmostEqual(circle.diameter, 14.0)

        canvas = np.full((220, 360, 3), 32, np.uint8)
        pts = [(40, 40), (160, 40), (160, 90), (40, 90)]
        viz.draw_poly(canvas, pts, viz.GREEN, 2)
        viz.put_text(canvas, f"w={summary.width:.0f} h={summary.height:.0f} ang={summary.angle:.1f}", (40, 120), viz.GREEN)
        cv2.line(canvas, (220, 40), (340, 40), viz.CYAN, 2, cv2.LINE_AA)
        cv2.line(canvas, (220, 40), (220, 140), viz.ORANGE, 2, cv2.LINE_AA)
        viz.put_text(canvas, "angle=90", (240, 170), viz.WHITE)
        cv2.circle(canvas, (80, 180), 28, viz.MAGENTA, 2, cv2.LINE_AA)
        viz.put_text(canvas, f"D={circle.diameter:.0f}", (120, 186), viz.MAGENTA)
        viz.assert_written(self, viz.save_panels("rect", "01_helpers", [("distance / angle / size", canvas)]))

    def test_rotated_rectangle(self):
        img = np.full((240, 320), 30, np.uint8)
        synth.paint_rect(img, 160, 110, 150, 90, 7.0, 210)
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
        view = viz.to_bgr(img)
        viz.draw_rect_result(view, hit)
        viz.assert_written(self, viz.save_panels("rect", "02_rotated", [("PASS 7deg", view)]))

    def test_axis_box(self):
        axis = np.full((200, 260), 25, np.uint8)
        cv2.rectangle(axis, (40, 30), (159, 99), 200, -1)
        axis = cv2.GaussianBlur(axis, (3, 3), 0.8)
        param = cvter.CaliperParam()
        param.polarity = cvter.CaliperPolarity.DarkToLight
        param.contrast = 12
        param.stride = 2
        param.projection = 3
        param.minPoints = 6
        box = cvter.find_rectangle_box(axis, 40, 30, 160, 100, 12.0, param)
        self.assertTrue(box.found)
        self.assertLess(math.hypot(box.center.x - 100, box.center.y - 65), 2.5)
        view = viz.to_bgr(axis)
        viz.draw_search_rect(view, 40, 30, 160, 100, viz.YELLOW)
        viz.draw_rect_result(view, box)
        viz.assert_written(self, viz.save_panels("rect", "03_box", [("PASS axis box", view)]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
