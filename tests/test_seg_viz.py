#!/usr/bin/env python3
"""距离变换、Canny、轮廓、分水岭。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cvter
import viz_util as viz


def colorize_labels(labels):
    lab = np.asarray(labels)
    out = np.zeros(lab.shape[:2] + (3,), np.uint8)
    rng = np.random.RandomState(1)
    colors = rng.randint(40, 255, size=(max(2, int(lab.max()) + 2), 3), dtype=np.uint8)
    colors[0] = (30, 30, 30)
    for y in range(lab.shape[0]):
        for x in range(lab.shape[1]):
            v = int(lab[y, x])
            if v < 0:
                out[y, x] = (0, 0, 255)
            elif v < len(colors):
                out[y, x] = colors[v]
    return out


class TestSegViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        viz.prepare("seg")

    def test_distance_canny_contour(self):
        img = np.zeros((160, 180), np.uint8)
        cv2.circle(img, (80, 80), 40, 255, -1)
        dist = cvter.distance_transform(img)
        self.assertEqual(dist.shape, img.shape)
        peak = np.unravel_index(int(np.argmax(dist)), dist.shape)
        self.assertLess(abs(peak[1] - 80) + abs(peak[0] - 80), 4)
        dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edges = cvter.canny(img, 40, 120)
        self.assertGreater(int(edges.sum()), 0)
        contours = cvter.extract_contours(img)
        self.assertGreaterEqual(len(contours), 1)
        self.assertGreater(contours[0].area, 4000)
        view = viz.to_bgr(img)
        viz.draw_poly(view, contours[0].points, viz.GREEN, 1)
        viz.draw_cross(view, contours[0].center.x, contours[0].center.y)
        viz.assert_written(self, viz.save_panels("seg", "01_distance_canny", [
            ("disk", img),
            ("distance", dist_u8),
            ("canny", edges),
            ("contour", view),
        ]))

    def test_split_touching_blobs(self):
        img = np.zeros((200, 240), np.uint8)
        cv2.circle(img, (90, 100), 42, 255, -1)
        cv2.circle(img, (145, 100), 42, 255, -1)
        split = cvter.split_blobs(img, 10, 0.35)
        self.assertTrue(split.found, split.regions)
        self.assertGreaterEqual(len(split.regions), 2)
        colored = colorize_labels(split.labels)
        for region in split.regions[:4]:
            viz.draw_cross(colored, region.center.x, region.center.y, viz.WHITE, 10)
            viz.put_text(colored, f"{region.area:.0f}", (region.left, max(12, region.top)), viz.WHITE, 0.4)
        viz.assert_written(self, viz.save_panels("seg", "02_watershed", [
            ("touching", img),
            ("watershed", colored),
        ]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
