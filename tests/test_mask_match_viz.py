#!/usr/bin/env python3
"""掩膜级模板匹配可视化。合成图，不弹窗，图片写到 tests/out_viz/mask/。

模板在目标 ROI 之外带文字、孔和噪声。搜索图上有两处：
- 真目标只保留 ROI，旁边换成另一套干扰
- 假目标贴上整幅干扰，但 ROI 被抹掉

PATTERN / SHAPE 不带 mask 会锁到假目标；带 mask 只看 ROI，回到真目标位姿。
"""
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

BG = 96
TH, TW = 150, 220
# 目标 ROI。文字、孔、噪声都画在这个矩形外面。
ROI = (12, 14, 98, 120)  # x0, y0, x1, y1
TRUE_XY = (48, 40)
DECOY_XY = (380, 230)
SCENE_H, SCENE_W = 460, 700
ROT_DEG = 7.0


def make_pattern(angle=0, score=0.55, max_count=1, min_area=1024):
    p = cvter.MatcherParam()
    p.matcherType = cvter.MatcherType.PATTERN
    p.angle = float(angle)
    p.scoreThreshold = float(score)
    p.maxCount = int(max_count)
    p.minArea = float(min_area)
    p.iouThreshold = 0.3
    p.meanBorder = True
    p.stopLayer = 0
    p.minScale = 1.0
    p.maxScale = 1.0
    p.scaleStep = 0.05
    return cvter.MatcherWrapper(p)


def make_shape(angle=0, score=0.4, max_count=1, min_area=1024):
    p = cvter.MatcherParam()
    p.matcherType = cvter.MatcherType.SHAPE
    p.angle = float(angle)
    p.scoreThreshold = float(score)
    p.maxCount = int(max_count)
    p.minArea = float(min_area)
    p.edgeMinMag = 12
    p.maxEdgePoints = 280
    p.usePolarity = True
    p.greediness = 0.8
    p.minScale = 1.0
    p.maxScale = 1.0
    p.scaleStep = 0.05
    return cvter.MatcherWrapper(p)


def best(results):
    return max(results, key=lambda r: r.Score) if results else None


def paint_object(img, x0, y0, x1, y1):
    """非对称目标：亮块、缺口、色块和圆孔，避免 180° 歧义。"""
    img[y0:y1, x0:x1] = 70
    img[y0 + 10:y1 - 10, x0 + 10:x1 - 10] = 220
    img[y0 + 10:y0 + 32, x0 + 10:x0 + 36] = 40
    img[y1 - 28:y1 - 10, x1 - 40:x1 - 10] = 120
    cv2.circle(img, (x0 + 58, y0 + 48), 10, 30, -1)


def paint_distractors(img, roi):
    """ROI 外的文字、孔和噪声。这些像素不进 mask。"""
    h, w = img.shape[:2]
    _x0, _y0, x1, y1 = roi
    img[6:h - 6, x1 + 8:w - 6] = 210
    org_x = x1 + 14
    cv2.putText(img, "LOT-19", (org_x, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 20, 2, cv2.LINE_AA)
    cv2.putText(img, "SN:8841", (org_x, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 15, 2, cv2.LINE_AA)
    cv2.putText(img, "A8K-QX", (org_x, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 25, 2, cv2.LINE_AA)
    for cx, cy, rad in ((x1 + 28, 118, 9), (x1 + 56, 122, 7), (x1 + 84, 116, 8)):
        cv2.circle(img, (cx, cy), rad + 3, 40, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), rad, 25, -1)
    for cx in (28, 52, 76):
        cv2.circle(img, (cx, y1 + 14), 8, 180, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, y1 + 14), 7, 30, -1)
    rng = np.random.RandomState(3)
    img[8:36, w - 48:w - 8] = rng.randint(0, 255, (28, 40), np.uint8)


def make_template():
    templ = np.full((TH, TW), BG, np.uint8)
    paint_object(templ, *ROI)
    paint_distractors(templ, ROI)
    mask = np.zeros((TH, TW), np.uint8)
    x0, y0, x1, y1 = ROI
    mask[y0:y1, x0:x1] = 255
    return templ, mask


def paint_other_clutter(img, origin, roi):
    """真目标旁边的另一套文字和孔，用来压低整幅 NCC。"""
    ox, oy = origin
    _x0, _y0, x1, _y1 = roi
    px0, px1 = x1 + 8, TW - 6
    img[oy + 6:oy + TH - 6, ox + px0:ox + px1] = 40
    cv2.putText(img, "NOPE", (ox + px0 + 6, oy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 230, 2, cv2.LINE_AA)
    cv2.putText(img, "ZZZ-00", (ox + px0 + 6, oy + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 200, 2, cv2.LINE_AA)
    for cx in (20, 70):
        cv2.circle(img, (ox + px0 + cx, oy + 120), 11, 220, -1)


def make_search(templ):
    scene = np.full((SCENE_H, SCENE_W), BG, np.uint8)
    x0, y0, x1, y1 = ROI
    tx, ty = TRUE_XY
    scene[ty + y0:ty + y1, tx + x0:tx + x1] = templ[y0:y1, x0:x1]
    paint_other_clutter(scene, TRUE_XY, ROI)

    dx, dy = DECOY_XY
    scene[dy:dy + TH, dx:dx + TW] = templ
    scene[dy + y0:dy + y1, dx + x0:dx + x1] = BG
    cv2.rectangle(scene, (dx + x0 + 18, dy + y0 + 30), (dx + x1 - 18, dy + y1 - 24), 180, -1)
    cv2.line(scene, (dx + x0 + 22, dy + y0 + 34), (dx + x1 - 22, dy + y1 - 28), 20, 3)
    return scene


def make_rotated(templ):
    """只把 ROI 贴上，再绕模板中心转 ROT_DEG。干扰不参与。"""
    canvas = np.full((440, 560), BG, np.uint8)
    ox, oy = 150, 120
    x0, y0, x1, y1 = ROI
    canvas[oy + y0:oy + y1, ox + x0:ox + x1] = templ[y0:y1, x0:x1]
    cx, cy = ox + TW / 2.0, oy + TH / 2.0
    matrix = cv2.getRotationMatrix2D((cx, cy), ROT_DEG, 1.0)
    scene = cv2.warpAffine(
        canvas, matrix, (canvas.shape[1], canvas.shape[0]),
        borderMode=cv2.BORDER_CONSTANT, borderValue=BG,
    )
    return scene, (cx, cy)


def center_of(origin):
    return (origin[0] + TW / 2.0, origin[1] + TH / 2.0)


def dist_to(result, xy):
    return math.hypot(result.Center.x - xy[0], result.Center.y - xy[1])


def mask_overlay(templ, mask):
    bgr = viz.to_bgr(templ)
    tint = np.zeros_like(bgr)
    tint[:] = (50, 170, 60)
    valid = mask > 0
    bgr[valid] = (0.5 * bgr[valid] + 0.5 * tint[valid]).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, contours, -1, viz.CYAN, 1, cv2.LINE_AA)
    viz.put_text(bgr, "mask = object", (8, 16), viz.CYAN, 0.4)
    return bgr


def edge_views(templ, mask):
    edges = cv2.Canny(templ, 40, 120)
    inside = edges.copy()
    inside[mask == 0] = 0
    return edges, inside


def mask_polygons(mask, result):
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    h, w = mask.shape[:2]
    src = np.float32([[0, 0], [w, 0], [0, h]])
    dst = np.float32([
        [result.LeftTop.x, result.LeftTop.y],
        [result.RightTop.x, result.RightTop.y],
        [result.LeftBottom.x, result.LeftBottom.y],
    ])
    matrix = cv2.getAffineTransform(src, dst)
    polys = []
    for contour in contours:
        warped = cv2.transform(contour.astype(np.float32), matrix)
        polys.append(warped.reshape(-1, 2))
    return polys


def contour_centroid(poly):
    moments = cv2.moments(poly.astype(np.float32))
    if abs(moments["m00"]) < 1e-6:
        mean = poly.mean(axis=0)
        return float(mean[0]), float(mean[1])
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def rotate_point(x, y, cx, cy, deg):
    rad = math.radians(deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx, dy = x - cx, y - cy
    return cx + cos_a * dx + sin_a * dy, cy - sin_a * dx + cos_a * dy


def draw_footprints(view):
    tx, ty = TRUE_XY
    dx, dy = DECOY_XY
    viz.draw_search_rect(view, tx, ty, tx + TW, ty + TH, viz.YELLOW)
    viz.draw_search_rect(view, dx, dy, dx + TW, dy + TH, viz.MAGENTA)
    viz.put_text(view, "true object", (tx + 4, max(14, ty - 6)), viz.YELLOW, 0.42)
    viz.put_text(view, "decoy text/holes/noise", (dx + 4, max(14, dy - 6)), viz.MAGENTA, 0.42)


def draw_hit(view, result, color, banner):
    viz.draw_match(view, result, color)
    viz.put_text(view, banner, (8, 18), color, 0.48)


class TestMaskMatchViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        viz.prepare("mask")
        cls.templ, cls.mask = make_template()
        cls.search = make_search(cls.templ)
        cls.true_c = center_of(TRUE_XY)
        cls.decoy_c = center_of(DECOY_XY)
        cls.rotated, cls.rot_c = make_rotated(cls.templ)
        cls.all_edges, cls.mask_edges = edge_views(cls.templ, cls.mask)

    def test_01_template_and_search(self):
        self.assertEqual(self.templ.shape, self.mask.shape)
        self.assertGreater(int((self.mask > 0).sum()), 1000)
        self.assertEqual(int(self.mask[0, 0]), 0)
        x0, y0, x1, y1 = ROI
        self.assertTrue(np.all(self.mask[y0:y1, x0:x1] > 0))
        outside = self.mask.copy()
        outside[y0:y1, x0:x1] = 0
        self.assertEqual(int(outside.sum()), 0)

        viz.assert_written(self, viz.save_panels("mask", "01_template_mask", [
            ("template", self.templ),
            ("mask", self.mask),
            ("template + mask", mask_overlay(self.templ, self.mask)),
            ("edges in mask", self.mask_edges),
        ]))
        search = viz.to_bgr(self.search)
        draw_footprints(search)
        viz.put_text(search, "yellow=true  magenta=decoy", (8, SCENE_H - 10), viz.WHITE, 0.45)
        viz.assert_written(self, viz.save_panels("mask", "02_search", [
            ("search", search),
        ]))

    def test_02_pattern_without_mask_false_peak(self):
        matcher = make_pattern(angle=0, score=0.4, max_count=1)
        matcher.setTemplate(self.templ)
        hit = best(matcher.match(self.search))
        self.assertIsNotNone(hit)
        view = viz.to_bgr(self.search)
        draw_footprints(view)
        draw_hit(view, hit, viz.RED, f"PATTERN no mask  score={hit.Score:.2f}  FALSE")
        viz.assert_written(self, viz.save_panels("mask", "03_pattern_nomask", [
            ("no mask false peak", view),
            ("template", self.templ),
        ]))
        self.assertGreater(hit.Score, 0.7)
        self.assertLess(dist_to(hit, self.decoy_c), 4.0)
        self.assertGreater(dist_to(hit, self.true_c), 80.0)
        self.assertLess(abs(hit.Angle), 1.5)

    def test_03_pattern_with_mask_pose(self):
        matcher = make_pattern(angle=0, score=0.55, max_count=1)
        matcher.setTemplate(self.templ, self.mask)
        hit = best(matcher.match(self.search))
        self.assertIsNotNone(hit)
        view = viz.to_bgr(self.search)
        draw_footprints(view)
        polys = mask_polygons(self.mask, hit)
        for poly in polys:
            viz.draw_poly(view, poly, viz.CYAN, 1, True)
        draw_hit(view, hit, viz.GREEN, f"PATTERN mask  score={hit.Score:.2f}  PASS")
        viz.assert_written(self, viz.save_panels("mask", "04_pattern_mask", [
            ("mask correct pose", view),
            ("template + mask", mask_overlay(self.templ, self.mask)),
        ]))
        self.assertGreater(hit.Score, 0.9)
        self.assertLess(dist_to(hit, self.true_c), 3.0)
        self.assertGreater(dist_to(hit, self.decoy_c), 80.0)
        self.assertLess(abs(hit.LeftTop.x - TRUE_XY[0]), 2.0)
        self.assertLess(abs(hit.LeftTop.y - TRUE_XY[1]), 2.0)
        self.assertLess(abs(hit.Angle), 1.5)
        self.assertAlmostEqual(hit.Scale, 1.0, places=3)
        self.assertTrue(polys)
        x0, y0, x1, y1 = ROI
        roi_c = (TRUE_XY[0] + (x0 + x1) / 2.0, TRUE_XY[1] + (y0 + y1) / 2.0)
        cc = contour_centroid(polys[0])
        self.assertLess(math.hypot(cc[0] - roi_c[0], cc[1] - roi_c[1]), 3.0)

    def test_04_shape_without_mask_false_peak(self):
        matcher = make_shape(angle=0, score=0.35, max_count=1)
        matcher.setTemplate(self.templ)
        hit = best(matcher.match(self.search))
        self.assertIsNotNone(hit)
        view = viz.to_bgr(self.search)
        draw_footprints(view)
        draw_hit(view, hit, viz.RED, f"SHAPE no mask  score={hit.Score:.2f}  FALSE")
        viz.assert_written(self, viz.save_panels("mask", "05_shape_nomask", [
            ("no mask distractor edges", view),
            ("all edges", self.all_edges),
        ]))
        self.assertGreater(hit.Score, 0.55)
        self.assertLess(dist_to(hit, self.decoy_c), 6.0)
        self.assertGreater(dist_to(hit, self.true_c), 80.0)

    def test_05_shape_with_mask(self):
        matcher = make_shape(angle=0, score=0.35, max_count=1)
        matcher.setTemplate(self.templ, self.mask)
        hit = best(matcher.match(self.search))
        self.assertIsNotNone(hit)
        view = viz.to_bgr(self.search)
        draw_footprints(view)
        polys = mask_polygons(self.mask, hit)
        for poly in polys:
            viz.draw_poly(view, poly, viz.CYAN, 1, True)
        draw_hit(view, hit, viz.GREEN, f"SHAPE mask  score={hit.Score:.2f}  PASS")
        viz.assert_written(self, viz.save_panels("mask", "06_shape_mask", [
            ("mask edges only", view),
            ("edges in mask", self.mask_edges),
        ]))
        self.assertGreater(hit.Score, 0.6)
        self.assertLess(dist_to(hit, self.true_c), 6.0)
        self.assertGreater(dist_to(hit, self.decoy_c), 80.0)
        self.assertLess(abs(hit.Angle), 2.5)
        self.assertAlmostEqual(hit.Scale, 1.0, places=3)
        self.assertTrue(polys)

    def test_06_rotated_pose_with_mask(self):
        pattern = make_pattern(angle=12, score=0.55, max_count=1)
        pattern.setTemplate(self.templ, self.mask)
        pattern_hit = best(pattern.match(self.rotated))
        shape = make_shape(angle=12, score=0.35, max_count=1)
        shape.setTemplate(self.templ, self.mask)
        shape_hit = best(shape.match(self.rotated))
        self.assertIsNotNone(pattern_hit)
        self.assertIsNotNone(shape_hit)

        pattern_view = viz.to_bgr(self.rotated)
        for poly in mask_polygons(self.mask, pattern_hit):
            viz.draw_poly(pattern_view, poly, viz.CYAN, 1, True)
        draw_hit(
            pattern_view, pattern_hit, viz.GREEN,
            f"PATTERN mask  score={pattern_hit.Score:.2f} ang={pattern_hit.Angle:.1f}",
        )
        shape_view = viz.to_bgr(self.rotated)
        for poly in mask_polygons(self.mask, shape_hit):
            viz.draw_poly(shape_view, poly, viz.CYAN, 1, True)
        draw_hit(
            shape_view, shape_hit, viz.GREEN,
            f"SHAPE mask  score={shape_hit.Score:.2f} ang={shape_hit.Angle:.1f}",
        )
        viz.assert_written(self, viz.save_panels("mask", "07_rotated_mask", [
            ("rotated object", self.rotated),
            ("PATTERN mask", pattern_view),
            ("SHAPE mask", shape_view),
        ]))

        self.assertGreater(pattern_hit.Score, 0.9)
        self.assertLess(abs(pattern_hit.Angle - ROT_DEG), 1.5)
        self.assertLess(dist_to(pattern_hit, self.rot_c), 3.0)
        self.assertAlmostEqual(pattern_hit.Scale, 1.0, places=3)

        self.assertGreater(shape_hit.Score, 0.7)
        self.assertLess(abs(shape_hit.Angle - ROT_DEG), 2.0)
        self.assertLess(dist_to(shape_hit, self.rot_c), 5.0)
        self.assertAlmostEqual(shape_hit.Scale, 1.0, places=3)

        x0, y0, x1, y1 = ROI
        ox = self.rot_c[0] - TW / 2.0
        oy = self.rot_c[1] - TH / 2.0
        roi_x = ox + (x0 + x1) / 2.0
        roi_y = oy + (y0 + y1) / 2.0
        expect = rotate_point(roi_x, roi_y, self.rot_c[0], self.rot_c[1], ROT_DEG)
        polys = mask_polygons(self.mask, pattern_hit)
        cc = contour_centroid(polys[0])
        self.assertLess(math.hypot(cc[0] - expect[0], cc[1] - expect[1]), 4.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
