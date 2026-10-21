#!/usr/bin/env python3
"""金样比对 / 差分缺陷。合成图，不弹窗，图片写到 tests/out_viz/golden/。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cvter
import synth
import viz_util as viz

# 与 synth.make_golden_plate 里留出的空位一致。像素质心约 (127.5, 52.5) / (214, 168)。
BRIGHT_BOX = (112, 44, 32, 18)
DARK_CENTER = (214, 168)
DARK_RADIUS = 11


def paint_defects(img):
    x, y, w, h = BRIGHT_BOX
    cv2.rectangle(img, (x, y), (x + w, y + h), 255, -1)
    cv2.circle(img, DARK_CENTER, DARK_RADIUS, 12, -1)
    img[32:34, 32:34] = 255
    return img


def base_param():
    p = cvter.GoldenParam()
    p.diffThreshold = 30
    p.minArea = 40
    p.enableMorph = True
    p.morphKsize = 3
    p.morphIterations = 1
    return p


def map_xy(transform, x, y):
    H = np.asarray(transform, np.float64)
    pt = cv2.perspectiveTransform(np.float32([[[x, y]]]), H)
    return float(pt[0, 0, 0]), float(pt[0, 0, 1])


def overlay_defects(image, result, color=viz.RED):
    view = viz.to_bgr(image)
    mask = np.asarray(result.mask) if result.mask is not None else None
    if mask is not None and mask.ndim >= 2 and mask.shape[:2] == view.shape[:2]:
        tint = view.copy()
        tint[mask > 0] = color
        view = cv2.addWeighted(view, 0.62, tint, 0.38, 0)
    for d in result.defects:
        cv2.rectangle(view, (int(d.left), int(d.top)),
                      (int(d.left + d.width), int(d.top + d.height)), viz.ORANGE, 1, cv2.LINE_AA)
        viz.draw_cross(view, d.center.x, d.center.y, viz.ORANGE, 10)
    viz.put_text(
        view,
        f"{result.alignMethod} n={result.defectCount} meanAbs={result.meanAbs:.2f} area={result.defectArea:.0f}",
        (8, 16),
        viz.GREEN,
        0.38,
    )
    return view


def draw_on_test(test, result):
    view = viz.to_bgr(test)
    H = np.asarray(result.transform, np.float64)
    for d in result.defects:
        corners = np.float32([
            [[d.left, d.top]],
            [[d.left + d.width, d.top]],
            [[d.left + d.width, d.top + d.height]],
            [[d.left, d.top + d.height]],
        ])
        mapped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        viz.draw_poly(view, mapped, viz.ORANGE, 2)
        cx, cy = map_xy(H, d.center.x, d.center.y)
        viz.draw_cross(view, cx, cy, viz.RED, 12)
    viz.put_text(view, f"{result.alignMethod} score={result.alignScore:.3f}", (8, 16), viz.GREEN, 0.4)
    return view


class TestGoldenViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        viz.prepare("golden")

    def test_clean_and_polarity(self):
        plate = synth.make_golden_plate()
        same = cvter.compare_golden(plate, plate, base_param())
        self.assertTrue(same.ok)
        self.assertEqual(same.defectCount, 0)
        self.assertEqual(same.meanAbs, 0)
        self.assertEqual(same.alignMethod, "none")

        bgr = cv2.cvtColor(plate, cv2.COLOR_GRAY2BGR)
        color = cvter.compare_golden(bgr, bgr, base_param())
        self.assertTrue(color.ok)
        self.assertEqual(color.defectCount, 0)

        test = paint_defects(plate.copy())
        both = cvter.compare_golden(test, plate, base_param())
        self.assertTrue(both.ok)
        self.assertEqual(both.defectCount, 2)
        self.assertGreater(both.meanAbs, 1.0)
        self.assertGreater(both.maxAbs, 80.0)
        self.assertGreater(both.defectArea, 400)
        self.assertEqual(both.diff.shape, plate.shape)
        self.assertEqual(both.signedDiff.dtype, np.float32)

        bright_p = base_param()
        bright_p.polarity = cvter.GoldenPolarity.Brighter
        bright = cvter.compare_golden(test, plate, bright_p)
        dark_p = base_param()
        dark_p.polarity = cvter.GoldenPolarity.Darker
        dark = cvter.compare_golden(test, plate, dark_p)
        self.assertEqual(bright.defectCount, 1)
        self.assertEqual(dark.defectCount, 1)
        self.assertLess(abs(bright.defects[0].center.x - 127.5) + abs(bright.defects[0].center.y - 52.5), 4)
        self.assertGreater(bright.defects[0].meanSigned, 40)
        self.assertLess(abs(dark.defects[0].center.x - 214) + abs(dark.defects[0].center.y - 168), 4)
        self.assertLess(dark.defects[0].meanSigned, -40)

        viz.assert_written(self, viz.save_panels("golden", "01_polarity", [
            ("golden", plate),
            ("test", test),
            ("both", overlay_defects(both.alignedImage, both)),
            ("brighter", overlay_defects(bright.alignedImage, bright, viz.ORANGE)),
            ("darker", overlay_defects(dark.alignedImage, dark, viz.CYAN)),
            ("abs diff", both.diff),
        ]))

    def test_roi_and_mask(self):
        plate = synth.make_golden_plate()
        test = paint_defects(plate.copy())
        roi = base_param()
        roi.roiX, roi.roiY, roi.roiW, roi.roiH = 0, 0, 170, plate.shape[0]
        left = cvter.compare_golden(test, plate, roi)
        self.assertTrue(left.ok)
        self.assertEqual(left.defectCount, 1)
        self.assertLess(left.defects[0].center.x, 160)

        mask = np.full(plate.shape, 255, np.uint8)
        mask[40:66, 108:148] = 0
        masked = cvter.compare_golden(test, plate, base_param(), mask)
        self.assertTrue(masked.ok)
        self.assertEqual(masked.defectCount, 1)
        self.assertLess(abs(masked.defects[0].center.x - 214) + abs(masked.defects[0].center.y - 168), 4)

        bad = np.full((plate.shape[0], plate.shape[1] - 4), 255, np.uint8)
        mismatch = cvter.compare_golden(test, plate, base_param(), bad)
        self.assertFalse(mismatch.ok)
        self.assertEqual(mismatch.message, "mask size mismatch")

        roi_view = overlay_defects(left.alignedImage, left)
        cv2.rectangle(roi_view, (0, 0), (170, plate.shape[0] - 1), viz.YELLOW, 1)
        mask_view = overlay_defects(masked.alignedImage, masked, viz.CYAN)
        cv2.rectangle(mask_view, (108, 40), (148, 66), viz.YELLOW, 1)
        viz.assert_written(self, viz.save_panels("golden", "02_roi_mask", [
            ("roi keeps bright", roi_view),
            ("mask keeps dark", mask_view),
            ("roi diff", left.diff),
            ("mask", masked.mask),
        ]))

    def test_translate(self):
        plate = synth.make_golden_plate()
        ox, oy = 15, 9
        canvas = np.full((plate.shape[0] + 40, plate.shape[1] + 36), 35, np.uint8)
        synth.paste(canvas, plate, ox, oy)
        x, y, w, h = BRIGHT_BOX
        cv2.rectangle(canvas, (ox + x, oy + y), (ox + x + w, oy + y + h), 255, -1)

        p = base_param()
        p.align = cvter.GoldenAlign.Translate
        hit = cvter.compare_golden(canvas, plate, p)
        self.assertTrue(hit.ok, hit.message)
        self.assertEqual(hit.alignMethod, "zncc")
        self.assertGreater(hit.alignScore, 0.85)
        H = np.asarray(hit.transform, np.float64)
        self.assertLess(abs(H[0, 2] - ox), 1.0)
        self.assertLess(abs(H[1, 2] - oy), 1.0)
        self.assertEqual(hit.defectCount, 1)
        self.assertLess(abs(hit.defects[0].center.x - 127.5) + abs(hit.defects[0].center.y - 52.5), 4)
        cx, cy = map_xy(hit.transform, hit.defects[0].center.x, hit.defects[0].center.y)
        self.assertLess(abs(cx - (ox + 127.5)) + abs(cy - (oy + 52.5)), 6)

        p.useNcc = True
        ncc = cvter.compare_golden(canvas, plate, p)
        self.assertTrue(ncc.ok)
        self.assertEqual(ncc.alignMethod, "ncc")

        dx, dy = 5, -3
        shifted = synth.shift_image(plate, dx, dy, 0)
        cv2.rectangle(shifted, (x + dx, y + dy), (x + dx + w, y + dy + h), 255, -1)
        shift = base_param()
        shift.align = cvter.GoldenAlign.Translate
        shift.alignMargin = 16
        moved = cvter.compare_golden(shifted, plate, shift)
        self.assertTrue(moved.ok, moved.message)
        MH = np.asarray(moved.transform, np.float64)
        self.assertLess(abs(MH[0, 2] - dx), 1.0)
        self.assertLess(abs(MH[1, 2] - dy), 1.0)
        self.assertEqual(moved.defectCount, 1)

        raw = cvter.compare_golden(shifted, plate, base_param())
        self.assertGreater(raw.meanAbs, moved.meanAbs)

        viz.assert_written(self, viz.save_panels("golden", "03_translate", [
            ("canvas", draw_on_test(canvas, hit)),
            ("aligned", overlay_defects(hit.alignedImage, hit)),
            ("same-size shift", draw_on_test(shifted, moved)),
            ("shift diff", moved.diff),
        ]))

    def test_blur_suppresses_speckle(self):
        plate = synth.make_golden_plate()
        rng = np.random.RandomState(0)
        noisy = np.clip(plate.astype(np.int16) + rng.randint(-20, 21, size=plate.shape, dtype=np.int16), 0, 255).astype(np.uint8)
        x, y, w, h = BRIGHT_BOX
        cv2.rectangle(noisy, (x, y), (x + w, y + h), 255, -1)

        raw_p = base_param()
        raw_p.diffThreshold = 18
        raw_p.minArea = 8
        raw_p.enableMorph = False
        raw_p.blurSigma = 0
        raw_p.maxCount = 1000
        raw = cvter.compare_golden(noisy, plate, raw_p)

        den_p = base_param()
        den_p.diffThreshold = 18
        den_p.minArea = 30
        den_p.blurSigma = 1.4
        den_p.blurKsize = 5
        den = cvter.compare_golden(noisy, plate, den_p)
        self.assertGreater(raw.defectCount, den.defectCount)
        self.assertGreaterEqual(den.defectCount, 1)
        top = max(den.defects, key=lambda d: d.area)
        self.assertLess(abs(top.center.x - 127.5) + abs(top.center.y - 52.5), 8)
        self.assertGreater(top.meanSigned, 20)

        viz.assert_written(self, viz.save_panels("golden", "04_blur", [
            ("noisy", noisy),
            (f"no blur n={raw.defectCount}", overlay_defects(raw.alignedImage, raw)),
            (f"blur n={den.defectCount}", overlay_defects(den.alignedImage, den)),
            ("blur diff", den.diff),
        ]))

    def _pose(self, align, name):
        plate = synth.make_pose_plate()
        h, w = plate.shape[:2]
        # 缺陷离开旋转中心，避免正变换和逆变换在中心点上看起来一样。
        gx, gy = w / 2.0 - 18.0, h / 2.0 + 12.0
        if align == cvter.GoldenAlign.Affine:
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 3.0, 1.0)
            scene = cv2.warpAffine(plate, M, (w, h), borderValue=0)
            pt = M @ np.array([gx, gy, 1.0])
            cv2.circle(scene, (int(round(pt[0])), int(round(pt[1]))), 14, 0, -1)
        else:
            src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            dst = np.float32([[18, 10], [w + 8, 16], [w - 4, h + 12], [8, h - 6]])
            H = cv2.getPerspectiveTransform(src, dst)
            scene = cv2.warpPerspective(plate, H, (w + 30, h + 30), borderValue=0)
            pt = cv2.perspectiveTransform(np.float32([[[gx, gy]]]), H)[0, 0]
            cv2.circle(scene, (int(round(pt[0])), int(round(pt[1]))), 14, 0, -1)

        p = base_param()
        p.align = align
        p.diffThreshold = 40
        p.minArea = 40
        p.blurSigma = 0.8
        p.featureRatio = 0.8
        p.ransacReproj = 4.0
        p.minInliers = 6
        p.roiX, p.roiY, p.roiW, p.roiH = w // 2 - 40, h // 2 - 36, 80, 72
        result = cvter.compare_golden(scene, plate, p)
        aligned = np.asarray(result.alignedImage) if result.alignedImage is not None else np.zeros((0, 0), np.uint8)
        view = overlay_defects(aligned, result) if aligned.size else viz.to_bgr(plate)
        if result.ok and aligned.size:
            cv2.rectangle(view, (p.roiX, p.roiY), (p.roiX + p.roiW, p.roiY + p.roiH), viz.YELLOW, 1)
        viz.assert_written(self, viz.save_panels("golden", name, [
            ("golden", plate),
            ("scene", scene),
            (result.alignMethod or result.message, view),
            ("diff", result.diff if np.size(result.diff) else np.zeros_like(plate)),
        ]))
        self.assertTrue(result.ok, f"{result.message} method={result.alignMethod} inliers={result.alignInliers}")
        self.assertGreaterEqual(result.alignInliers, 6)
        self.assertGreaterEqual(result.defectCount, 1)
        top = max(result.defects, key=lambda d: d.area)
        self.assertLess(abs(top.center.x - gx) + abs(top.center.y - gy), 12, top)
        self.assertGreater(top.area, 150)
        self.assertLess(top.meanSigned, -20)
        return result

    def test_affine(self):
        result = self._pose(cvter.GoldenAlign.Affine, "05_affine")
        self.assertTrue(result.alignMethod.startswith("orb-affine"))

    def test_homography(self):
        result = self._pose(cvter.GoldenAlign.Homography, "06_homography")
        self.assertEqual(result.alignMethod, "orb-homography")

    def test_failures(self):
        plate = synth.make_golden_plate()
        blank = np.full_like(plate, 20)
        p = cvter.GoldenParam()
        p.align = cvter.GoldenAlign.Translate
        missed = cvter.compare_golden(blank, plate, p)
        self.assertFalse(missed.ok)
        self.assertFalse(missed.aligned)
        self.assertEqual(missed.defectCount, 0)

        none = cvter.GoldenParam()
        size = cvter.compare_golden(plate[:, :40], plate, none)
        self.assertFalse(size.ok)
        self.assertEqual(size.message, "size mismatch")

        view = viz.to_bgr(blank)
        viz.put_text(view, missed.message or "locate failed", (8, 18), viz.RED)
        viz.assert_written(self, viz.save_panels("golden", "07_fail", [
            ("blank test", view),
            ("golden", plate),
        ]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
