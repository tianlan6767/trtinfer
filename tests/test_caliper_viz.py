#!/usr/bin/env python3
"""卡尺直线 / 圆可视化。"""
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

NUT_IMG = ROOT / "workspace" / "match_img" / "nut" / (
    "D4VHW951KR80001PQH+8A31A28SH33AB000000_20260818023454_2_2_1_Color_a.jpg"
)


def vert_dev(angle):
    return abs((abs(angle) % 180.0) - 90.0)


class TestCaliperViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        viz.prepare("caliper")

    def test_vertical_and_miss(self):
        img = synth.vertical_step(140, 200, 70)
        p = cvter.CaliperParam()
        p.polarity = cvter.CaliperPolarity.LightToDark
        p.contrast = 12
        hit = cvter.find_line(img, 40, 20, 110, 180, True, p)
        self.assertTrue(hit.found)
        self.assertLess(abs(hit.center.x - 70), 1.0)
        self.assertLess(vert_dev(hit.angle), 1.5)
        view = viz.to_bgr(img)
        viz.draw_search_rect(view, 40, 20, 110, 180)
        viz.draw_line_result(view, hit, viz.GREEN, "L2D")
        bad = cvter.CaliperParam()
        bad.polarity = cvter.CaliperPolarity.DarkToLight
        bad.contrast = 18
        miss = cvter.find_line(img, 40, 20, 110, 180, True, bad)
        self.assertFalse(miss.found)
        miss_view = viz.to_bgr(img)
        viz.draw_search_rect(miss_view, 40, 20, 110, 180, viz.RED)
        viz.put_text(miss_view, "FAIL wrong polarity", (8, 18), viz.RED, 0.45)
        viz.assert_written(self, viz.save_panels("caliper", "01_vertical", [
            ("PASS vertical", view),
            ("FAIL polarity", miss_view),
        ]))

    def test_horizontal(self):
        img = synth.horizontal_step(180, 160, 90, top=40, bottom=200)
        p = cvter.CaliperParam()
        p.polarity = cvter.CaliperPolarity.LightToDark
        p.contrast = 12
        hit = cvter.find_line(img, 20, 40, 160, 130, False, p)
        self.assertTrue(hit.found)
        self.assertLess(abs(hit.center.y - 90), 1.0)
        view = viz.to_bgr(img)
        viz.draw_search_rect(view, 20, 40, 160, 130)
        viz.draw_line_result(view, hit, viz.GREEN, "horiz")
        viz.assert_written(self, viz.save_panels("caliper", "02_horizontal", [("PASS", view)]))

    def test_select_edges(self):
        img = synth.two_falling_edges()
        panels = []
        expect = {"First": 40, "Last": 100, "Strongest": 40}
        for name, sel in (("First", cvter.CaliperSelect.First),
                          ("Last", cvter.CaliperSelect.Last),
                          ("Strongest", cvter.CaliperSelect.Strongest)):
            p = cvter.CaliperParam()
            p.polarity = cvter.CaliperPolarity.LightToDark
            p.contrast = 10
            p.select = sel
            hit = cvter.find_line(img, 10, 10, 150, 170, True, p)
            self.assertTrue(hit.found)
            self.assertLess(abs(hit.center.x - expect[name]), 4.0)
            view = viz.to_bgr(img)
            viz.draw_search_rect(view, 10, 10, 150, 170)
            viz.draw_line_result(view, hit, viz.GREEN, name)
            panels.append((name, view))
        viz.assert_written(self, viz.save_panels("caliper", "03_select", panels))

    def test_oriented(self):
        img = synth.vertical_step(140, 200, 70)
        p = cvter.CaliperParam()
        p.contrast = 12
        hit = cvter.find_line_oriented(img, 70, 100, 90.0, 80, 30, p)
        self.assertTrue(hit.found)
        self.assertLess(abs(hit.center.x - 70), 1.0)
        view = viz.to_bgr(img)
        viz.draw_oriented_search(view, 70, 100, 90.0, 80, 30)
        viz.draw_line_result(view, hit, viz.GREEN, "phi90")
        viz.assert_written(self, viz.save_panels("caliper", "04_oriented", [("PASS phi=90", view)]))

    def test_circle_and_arc(self):
        cx, cy, rad = 150.0, 140.0, 60.0
        img = synth.dark_disk(280, cx, cy, rad)
        p = cvter.CaliperParam()
        p.polarity = cvter.CaliperPolarity.DarkToLight
        p.contrast = 12
        p.radialInward = False
        hit = cvter.find_circle(img, cx + 3, cy - 2, rad, 18, 0, 360, p)
        self.assertTrue(hit.found)
        self.assertLess(math.hypot(hit.center.x - cx, hit.center.y - cy), 1.5)
        self.assertLess(abs(hit.radius - rad), 1.5)
        view = viz.to_bgr(img)
        viz.draw_annulus(view, cx + 3, cy - 2, rad, 18)
        viz.draw_circle_result(view, hit)
        arc = cvter.find_circle(img, cx, cy, rad, 16, -90, 90, p)
        self.assertTrue(arc.found)
        arc_view = viz.to_bgr(img)
        viz.draw_annulus(arc_view, cx, cy, rad, 16, -90, 90)
        viz.draw_circle_result(arc_view, arc)
        viz.assert_written(self, viz.save_panels("caliper", "05_circle", [
            ("PASS full", view),
            ("PASS arc", arc_view),
        ]))

    @unittest.skipUnless(NUT_IMG.is_file(), "缺少螺母图")
    def test_nut(self):
        gray = cv2.imdecode(np.fromfile(str(NUT_IMG), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        p = cvter.CaliperParam()
        p.polarity = cvter.CaliperPolarity.LightToDark
        p.contrast = 18
        lower = cvter.find_line(gray, 290, 406, 327, 586, True, p)
        self.assertTrue(lower.found)
        view = viz.to_bgr(gray)
        viz.draw_search_rect(view, 290, 406, 327, 586)
        viz.draw_line_result(view, lower, viz.GREEN, "nut")
        cp = cvter.CaliperParam()
        cp.polarity = cvter.CaliperPolarity.DarkToLight
        cp.contrast = 12
        circ = cvter.find_circle(gray, 255, 250, 70, 20, 0, 360, cp)
        self.assertTrue(circ.found)
        viz.draw_annulus(view, 255, 250, 70, 20)
        viz.draw_circle_result(view, circ)
        viz.assert_written(self, viz.save_panels("caliper", "06_nut", [("nut", view)]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
