#!/usr/bin/env python3
"""尺寸测量可视化。不 imshow，图片写到 tests/out_viz/measure/。"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cvter
import viz_util as viz

OUT = Path(__file__).resolve().parent / "out_viz" / "measure"


def ang_dist(a, b):
    d = (a - b) % 180.0
    if d > 90.0:
        d -= 180.0
    return abs(d)


def blur(img, sigma=0.8):
    return cv2.GaussianBlur(img, (5, 5), sigma)


def both_param():
    p = cvter.CaliperParam()
    p.polarity = cvter.CaliperPolarity.Both
    p.contrast = 12
    p.stride = 3
    p.projection = 3
    p.sigma = 0.6
    p.minPoints = 6
    p.outlierRatio = 0.2
    return p


def canvas(w=320, h=240, value=28):
    return np.full((h, w, 3), value, np.uint8)


def line(img, a, b, color, thickness=2):
    cv2.line(img, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))),
             color, thickness, cv2.LINE_AA)


class TestMeasureViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        viz.prepare("measure")

    def test_geometry(self):
        self.assertAlmostEqual(cvter.segment_length(0, 0, 3, 4), 5.0)
        mid = cvter.midpoint(0, 0, 4, 2)
        self.assertAlmostEqual(mid.x, 2.0)
        foot = cvter.foot_of_perpendicular(3, 5, 0, 0, 10, 0)
        self.assertTrue(foot.found)
        self.assertAlmostEqual(foot.point.y, 0.0)
        clamped = cvter.foot_of_perpendicular(-2, 5, 0, 0, 10, 0, True)
        self.assertAlmostEqual(clamped.point.x, 0.0)
        hit = cvter.intersect_lines(0, 0, 10, 0, 2, -4, 2, 6)
        self.assertTrue(hit.found)
        self.assertAlmostEqual(hit.point.x, 2.0, places=6)
        self.assertTrue(math.isnan(cvter.distance_parallel_lines(0, 0, 10, 0, 0, 0, 0, 10)))
        self.assertAlmostEqual(cvter.distance_parallel_lines(0, 0, 20, 0, 0, 30, 10, 30), 30.0)
        self.assertAlmostEqual(cvter.signed_distance_point_circle(6, 0, 0, 0, 10), -4.0)
        self.assertAlmostEqual(cvter.distance_circle_line(0, 0, 10, 0, 14, 8, 14), 4.0)
        self.assertAlmostEqual(cvter.distance_circle_line(0, 0, 10, 0, 4, 6, 4), 0.0)

        view = canvas()
        line(view, (40, 160), (260, 40), viz.YELLOW, 1)
        line(view, (40, 40), (260, 150), viz.YELLOW, 1)
        cross = cvter.intersect_lines(40, 160, 260, 40, 40, 40, 260, 150)
        self.assertTrue(cross.found)
        viz.draw_cross(view, cross.point.x, cross.point.y, viz.GREEN, 14, 2)
        cv2.circle(view, (180, 120), 36, viz.CYAN, 1, cv2.LINE_AA)
        viz.draw_cross(view, 70, 120, viz.ORANGE, 10)
        foot2 = cvter.foot_of_perpendicular(70, 120, 40, 160, 260, 40)
        line(view, (70, 120), (foot2.point.x, foot2.point.y), viz.ORANGE, 1)
        viz.put_text(view, f"cross=({cross.point.x:.0f},{cross.point.y:.0f})  circle gap={cvter.distance_circle_line(180, 120, 36, 40, 160, 260, 40):.1f}",
                     (8, 228), viz.WHITE, 0.4)
        viz.assert_written(self, viz.save_panels("measure", "01_geometry", [("geometry", view)]))

    def test_edges_width_arc(self):
        edges = np.full((180, 220), 20, np.uint8)
        edges[:, 40:100] = 200
        edges[:, 100:150] = 80
        edges[:, 150:] = 230
        edges = blur(edges)
        param = both_param()
        param.select = cvter.CaliperSelect.All
        multi = cvter.find_edges(edges, 10, 20, 190, 160, True, param)
        self.assertGreaterEqual(len(multi.edges), 3)
        xs = {e.index: e.line.center.x for e in multi.edges if e.line.found}
        self.assertLess(abs(xs[0] - 40), 1.8)
        self.assertLess(abs(xs[1] - 100), 1.8)
        self.assertLess(abs(xs[2] - 150), 1.8)
        view = viz.to_bgr(edges)
        viz.draw_search_rect(view, 10, 20, 190, 160)
        for edge in multi.edges:
            if edge.line.found:
                viz.draw_line_result(view, edge.line, viz.GREEN, f"e{edge.index}")
        param.select = cvter.CaliperSelect.Nth
        param.edgeIndex = 1
        nth = cvter.find_line(edges, 10, 20, 190, 160, True, param)
        self.assertTrue(nth.found)
        self.assertLess(abs(nth.center.x - 100), 2.0)

        bar = np.full((160, 200), 30, np.uint8)
        bar[:, 70:130] = 210
        bar = blur(bar)
        wall = cvter.measure_dual_width(bar, 100, 80, 90, 55, 45, both_param())
        self.assertTrue(wall.found)
        self.assertLess(abs(wall.width - 60), 2.0)
        wall_view = viz.to_bgr(bar)
        viz.draw_line_result(wall_view, wall.edge0, viz.GREEN, "e0")
        viz.draw_line_result(wall_view, wall.edge1, viz.ORANGE, "e1")
        viz.put_text(wall_view, f"wall={wall.width:.2f}", (8, 150), viz.WHITE, 0.5)

        slot = np.full((160, 220), 210, np.uint8)
        slot[:, 80:145] = 30
        slot = blur(slot)
        groove = cvter.measure_dual_width(slot, 112.5, 80, 90, 55, 50, both_param())
        self.assertTrue(groove.found)
        self.assertLess(abs(groove.width - 65), 2.0)
        slot_view = viz.to_bgr(slot)
        viz.draw_line_result(slot_view, groove.edge0, viz.GREEN, "s0")
        viz.draw_line_result(slot_view, groove.edge1, viz.ORANGE, "s1")
        viz.put_text(slot_view, f"slot={groove.width:.2f}", (8, 150), viz.WHITE, 0.5)

        gap = np.full((180, 250), 40, np.uint8)
        gap[:, 30:90] = 210
        gap[:, 150:220] = 210
        gap = blur(gap)
        inward = both_param()
        inward.polarity = cvter.CaliperPolarity.LightToDark
        opening = cvter.measure_gap(gap, 120, 90, 90, 55, 30, 16, inward)
        self.assertTrue(opening.found)
        self.assertLess(abs(opening.width - 60), 2.5)
        gap_view = viz.to_bgr(gap)
        viz.draw_line_result(gap_view, opening.first, viz.GREEN, "L")
        viz.draw_line_result(gap_view, opening.second, viz.ORANGE, "R")
        line(gap_view, (opening.p1.x, opening.p1.y), (opening.p2.x, opening.p2.y), viz.MAGENTA, 2)
        viz.put_text(gap_view, f"gap={opening.width:.2f}", (8, 170), viz.WHITE, 0.5)

        disk = np.full((280, 280), 200, np.uint8)
        cv2.circle(disk, (140, 140), 60, 25, -1)
        disk = blur(disk)
        arc_param = both_param()
        arc_param.polarity = cvter.CaliperPolarity.DarkToLight
        arc = cvter.find_arc(disk, 140, 140, 60, 16, -30, 110, arc_param)
        self.assertTrue(arc.found)
        self.assertLess(abs(arc.radius - 60), 2.0)
        self.assertGreater(arc.spanDeg, 90)
        self.assertLess(arc.spanDeg, 150)
        self.assertAlmostEqual(arc.arcLength, arc.radius * math.radians(arc.spanDeg), delta=1.0)
        arc_view = viz.to_bgr(disk)
        cv2.ellipse(arc_view, (int(round(arc.center.x)), int(round(arc.center.y))),
                    (int(round(arc.radius)), int(round(arc.radius))), 0,
                    arc.startDeg, arc.endDeg, viz.GREEN, 2, cv2.LINE_AA)
        line(arc_view, (arc.chordStart.x, arc.chordStart.y), (arc.chordEnd.x, arc.chordEnd.y), viz.ORANGE, 1)
        viz.draw_points(arc_view, arc.inliers, viz.CYAN, 2)
        viz.put_text(arc_view, f"span={arc.spanDeg:.1f} L={arc.arcLength:.1f} chord={arc.chord:.1f}",
                     (8, 18), viz.GREEN, 0.45)

        ring = np.full((280, 280), 25, np.uint8)
        cv2.circle(ring, (140, 140), 70, 200, -1)
        cv2.circle(ring, (140, 140), 45, 25, -1)
        ring = blur(ring, 1.0)
        radial = cvter.measure_radial_width(ring, 140, 140, 58, 22, 0, 360, both_param())
        self.assertTrue(radial.found)
        self.assertLess(abs(radial.width - 25), 2.5)
        ring_view = viz.to_bgr(ring)
        viz.draw_circle_result(ring_view, radial.circle0, viz.GREEN)
        cv2.circle(ring_view, (int(round(radial.circle1.center.x)), int(round(radial.circle1.center.y))),
                   max(1, int(round(radial.circle1.radius))), viz.ORANGE, 2, cv2.LINE_AA)
        viz.put_text(ring_view, f"t={radial.width:.2f}", (8, 40), viz.WHITE, 0.5)

        viz.assert_written(self, viz.save_panels("measure", "02_edges", [
            ("3 edges", view),
            (f"wall {wall.width:.1f}", wall_view),
            (f"slot {groove.width:.1f}", slot_view),
        ]))
        viz.assert_written(self, viz.save_panels("measure", "03_gap_arc", [
            (f"gap {opening.width:.1f}", gap_view),
            (f"arc {arc.spanDeg:.0f}deg", arc_view),
            (f"ring {radial.width:.1f}", ring_view),
        ]))

    def test_fit_form_size(self):
        line_pts = np.array([[i, 0.2 * i + 1] for i in range(30)] + [[5, 40], [15, -30], [25, 35]], np.float64)
        ls = cvter.fit_line(line_pts)
        ransac = cvter.FitParam()
        ransac.method = cvter.FitMethod.Ransac
        ransac.threshold = 1.5
        robust = cvter.fit_line(line_pts, ransac)
        self.assertGreater(ls.rms, 2.0)
        self.assertLess(robust.rms, 0.2)
        self.assertGreaterEqual(robust.numInliers, 28)
        fit_view = canvas(360, 220, 245)
        for x, y in line_pts:
            cv2.circle(fit_view, (int(round(40 + x * 8)), int(round(180 - y * 3))), 3, viz.RED, -1, cv2.LINE_AA)
        line(fit_view, (40 + robust.p1.x * 8, 180 - robust.p1.y * 3),
             (40 + robust.p2.x * 8, 180 - robust.p2.y * 3), viz.GREEN, 2)
        viz.put_text(fit_view, f"LS rms={ls.rms:.1f}  RANSAC rms={robust.rms:.2f} n={robust.numInliers}",
                     (8, 20), viz.GREEN, 0.45)

        ang = math.radians(25)
        ca, sa = math.cos(ang), math.sin(ang)
        ell = []
        for i in range(48):
            t = i * 2 * math.pi / 48
            x, y = 50 * math.cos(t), 22 * math.sin(t)
            ell.append((80 + x * ca - y * sa, 70 + x * sa + y * ca))
        ell_fit = cvter.fit_ellipse(np.asarray(ell))
        self.assertTrue(ell_fit.found)
        self.assertLess(abs(ell_fit.a - 50), 2)
        self.assertLess(abs(ell_fit.b - 22), 2)
        self.assertLess(ang_dist(ell_fit.angle, 25), 5)
        ell_view = canvas(220, 180, 245)
        for x, y in ell:
            cv2.circle(ell_view, (int(round(x)), int(round(y))), 2, viz.CYAN, -1, cv2.LINE_AA)
        cv2.ellipse(ell_view, (int(round(ell_fit.center.x)), int(round(ell_fit.center.y))),
                    (int(round(ell_fit.a)), int(round(ell_fit.b))), ell_fit.angle, 0, 360, viz.GREEN, 2, cv2.LINE_AA)

        q0, q1, q2, q3 = (30, 40), (150, 55), (130, 120), (10, 105)
        quad = [q0, q1, q2, q3]
        for i in range(1, 8):
            t = i / 8.0
            for a, b in ((q0, q1), (q1, q2), (q2, q3), (q3, q0)):
                quad.append((a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t))
        para = cvter.fit_parallelogram(np.asarray(quad))
        self.assertTrue(para.found)
        self.assertFalse(para.rectangle)
        self.assertLess(math.hypot(para.center.x - 80, para.center.y - 80), 1.0)
        quad_view = canvas(200, 160, 245)
        viz.draw_poly(quad_view, [para.c0, para.c1, para.c2, para.c3], viz.GREEN, 2)
        viz.draw_points(quad_view, quad, viz.CYAN, 2)
        viz.put_text(quad_view, f"err={para.angleError:.1f} rect={para.rectangle}", (8, 16), viz.GREEN, 0.4)

        straight = np.array([[i, 0] for i in range(21)], np.float64)
        ok_line = cvter.check_straightness(straight, 0.1)
        self.assertTrue(ok_line.ok)
        bent = np.vstack([straight, [[10, 3]]])
        bad_line = cvter.check_straightness(bent, 1.0)
        self.assertFalse(bad_line.ok)
        self.assertGreaterEqual(bad_line.value, 3.0 - 1e-6)
        parallel = cvter.check_parallelism(0, 0, 10, 0, 0, 4, 8, 4, 1.0)
        perp = cvter.check_perpendicularity(0, 0, 10, 0, 0, 0, 0, 10, 1.0)
        self.assertTrue(parallel.ok and perp.ok)
        form_view = canvas(280, 80, 245)
        viz.put_text(form_view, f"straight PASS zone={ok_line.value:.3f}", (8, 24), viz.GREEN, 0.5)
        viz.put_text(form_view, f"bump FAIL zone={bad_line.value:.2f}", (8, 48), viz.RED, 0.5)
        viz.put_text(form_view, f"parallel {parallel.value:.2f}  perp {perp.value:.2f}", (8, 70), viz.GREEN, 0.45)

        box = np.array([[0, 0], [100, 0], [100, 40], [0, 40]], np.float64)
        aabb = cvter.size_aabb(box)
        rot = cvter.size_min_area(box)
        outer = cvter.circle_circumscribed(box)
        inner = cvter.circle_inscribed(box)
        width = cvter.width_function(box, 1.0)
        self.assertAlmostEqual(aabb.width, 100)
        self.assertAlmostEqual(aabb.height, 40)
        self.assertLess(abs(max(rot.width, rot.height) - 100), 1)
        self.assertLess(abs(outer.diameter - math.hypot(100, 40)), 0.5)
        self.assertLess(abs(inner.diameter - 40), 3)
        self.assertLess(abs(width.minWidth - 40), 0.2)
        self.assertLess(abs(width.maxWidth - math.hypot(100, 40)), 0.5)
        size_view = canvas(180, 120, 245)
        viz.draw_poly(size_view, box + (20, 30), viz.YELLOW, 1)
        cv2.circle(size_view, (int(round(outer.center.x + 20)), int(round(outer.center.y + 30))),
                   int(round(outer.radius)), viz.CYAN, 1, cv2.LINE_AA)
        cv2.circle(size_view, (int(round(inner.center.x + 20)), int(round(inner.center.y + 30))),
                   max(1, int(round(inner.radius))), viz.GREEN, 1, cv2.LINE_AA)
        line(size_view, (width.minA.x + 20, width.minA.y + 30), (width.minB.x + 20, width.minB.y + 30), viz.ORANGE, 2)
        viz.put_text(size_view, f"minW={width.minWidth:.0f} maxW={width.maxWidth:.0f}", (8, 16), viz.WHITE, 0.4)

        mask = np.zeros((100, 140), np.uint8)
        cv2.rectangle(mask, (10, 10), (130, 90), 255, -1)
        cv2.rectangle(mask, (80, 35), (130, 65), 0, -1)
        notch = cvter.circle_inscribed_mask(mask)
        self.assertTrue(notch.found)
        self.assertGreater(notch.diameter, 60)
        self.assertLess(notch.diameter, 78)
        notch_view = viz.to_bgr(mask)
        cv2.circle(notch_view, (int(round(notch.center.x)), int(round(notch.center.y))),
                   max(1, int(round(notch.radius))), viz.GREEN, 1, cv2.LINE_AA)

        hole = cvter.hole_to_edge(80, 70, 18, 140, 20, 140, 120)
        holes = cvter.hole_to_hole(50, 90, 12, 130, 90, 16)
        edges = cvter.edge_to_edge(20, 30, 160, 30, 20, 70, 150, 70)
        self.assertAlmostEqual(hole.value, 140 - 80 - 18, places=4)
        self.assertAlmostEqual(holes.value, 80 - 12 - 16, places=4)
        self.assertAlmostEqual(edges.value, 40, places=4)
        gage_view = canvas(200, 140, 32)
        cv2.circle(gage_view, (80, 70), 18, viz.CYAN, 1, cv2.LINE_AA)
        line(gage_view, (140, 20), (140, 120), viz.YELLOW, 1)
        line(gage_view, (hole.a.x, hole.a.y), (hole.b.x, hole.b.y), viz.GREEN, 2)
        cv2.circle(gage_view, (50, 90), 12, viz.ORANGE, 1, cv2.LINE_AA)
        cv2.circle(gage_view, (130, 90), 16, viz.ORANGE, 1, cv2.LINE_AA)
        line(gage_view, (holes.a.x, holes.a.y), (holes.b.x, holes.b.y), viz.MAGENTA, 2)
        viz.put_text(gage_view, f"hole-edge={hole.value:.0f}  hole-hole={holes.value:.0f}  edge={edges.value:.0f}",
                     (8, 16), viz.WHITE, 0.38)

        viz.assert_written(self, viz.save_panels("measure", "04_fit", [
            ("RANSAC line", fit_view),
            ("ellipse", ell_view),
            ("parallelogram", quad_view),
        ]))
        viz.assert_written(self, viz.save_panels("measure", "05_form_size", [
            ("form", form_view),
            ("feret", size_view),
            (f"notch d={notch.diameter:.0f}", notch_view),
            ("gage", gage_view),
        ]))

    def test_scale(self):
        scale = cvter.scale_mm_per_px(0.1)
        frame = cvter.FixtureFrame()
        frame.origin = cvter.Point2d(100, 50)
        frame.angleDeg = 90
        local = cvter.to_fixture(100, 60, scale, frame)
        self.assertAlmostEqual(local.x, 1.0, places=5)
        self.assertAlmostEqual(local.y, 0.0, places=5)
        dim = cvter.dimension_points(100, 50, 100, 60, scale, frame)
        self.assertTrue(dim.scaled)
        self.assertAlmostEqual(dim.pixels, 10)
        self.assertAlmostEqual(dim.physical, 1)
        xy = cvter.scale_mm_per_px_xy(0.05, 0.0625)
        diag = cvter.dimension_points(0, 0, 100, 100, xy)
        self.assertAlmostEqual(diag.physical, math.hypot(5, 6.25), places=5)
        k = np.array([[1000, 0, 320], [0, 800, 240], [0, 0, 1]], np.float64)
        from_k = cvter.scale_from_camera(k, 50)
        self.assertAlmostEqual(from_k.mmPerPxX, 0.05)
        self.assertAlmostEqual(from_k.mmPerPxY, 0.0625)
        path = OUT / "scale.yml"
        path.parent.mkdir(parents=True, exist_ok=True)
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
        fs.write("mmPerPx", 0.05)
        fs.release()
        loaded = cvter.scale_from_yaml(str(path), 0)
        self.assertTrue(loaded.found)
        self.assertAlmostEqual(loaded.mmPerPx, 0.05)
        view = canvas(280, 120, 36)
        line(view, (40, 70), (240, 70), viz.GREEN, 2)
        span = cvter.dimension_points(40, 70, 240, 70, loaded)
        viz.put_text(view, f"200 px = {span.physical:.2f} mm   fixture +X is image +Y", (8, 24), viz.WHITE, 0.45)
        viz.draw_cross(view, 100, 50, viz.ORANGE, 12, 2)
        viz.put_text(view, f"origin -> (100,60) = ({local.x:.1f}, {local.y:.1f}) mm", (8, 100), viz.ORANGE, 0.45)
        viz.assert_written(self, viz.save_panels("measure", "06_scale", [("0.05 mm/px", view)]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
