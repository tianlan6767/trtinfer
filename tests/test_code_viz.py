#!/usr/bin/env python3
"""QR、一维码、OCR 可视化。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cvter
import viz_util as viz

L = ["0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111", "0001011"]
G = ["0100111", "0110011", "0011011", "0100001", "0011101", "0111001", "0000101", "0010001", "0001001", "0010111"]
R = ["1110010", "1100110", "1101100", "1000010", "1011100", "1001110", "1010000", "1000100", "1001000", "1110100"]
PARITY = ["LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG", "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL"]


def ean13_bits(data12: str) -> str:
    digits = [int(c) for c in data12]
    checksum = (10 - (sum(digits[i] * (1 if i % 2 == 0 else 3) for i in range(12)) % 10)) % 10
    code = data12 + str(checksum)
    bits = "101"
    parity = PARITY[int(code[0])]
    for i in range(6):
        d = int(code[i + 1])
        bits += L[d] if parity[i] == "L" else G[d]
    bits += "01010"
    for i in range(6):
        bits += R[int(code[7 + i])]
    bits += "101"
    return bits, code


def render_ean13(data12="590123412345", module=4, height=120, quiet=20):
    bits, code = ean13_bits(data12)
    width = quiet * 2 + len(bits) * module
    img = np.full((height + 40, width), 255, np.uint8)
    x = quiet
    for bit in bits:
        if bit == "1":
            img[10:10 + height, x:x + module] = 0
        x += module
    return img, code


def draw_hits(image, hits):
    view = viz.to_bgr(image)
    for hit in hits:
        viz.draw_poly(view, hit.polygon, viz.GREEN, 2)
        viz.put_text(view, f"{hit.format}:{hit.text}", (8, 20), viz.GREEN, 0.5)
    return view


class TestCodeViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        viz.prepare("code")

    def test_qr(self):
        enc = cv2.QRCodeEncoder_create()
        qr = enc.encode("AOI-QR-17")
        qr = cv2.resize(qr, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
        canvas = np.full((qr.shape[0] + 80, qr.shape[1] + 80), 255, np.uint8)
        canvas[40:40 + qr.shape[0], 40:40 + qr.shape[1]] = qr
        hits = cvter.decode_qr(canvas)
        self.assertTrue(hits, "QR 未解出")
        self.assertEqual(hits[0].text, "AOI-QR-17")
        self.assertEqual(hits[0].format, "QR")
        self.assertGreaterEqual(viz.as_points(hits[0].polygon).shape[0], 4)
        viz.assert_written(self, viz.save_panels("code", "01_qr", [
            ("input", canvas),
            ("PASS QR", draw_hits(canvas, hits)),
        ]))

    def test_barcode(self):
        if not cvter.barcode_available():
            self.skipTest("当前 OpenCV 没有 barcode 模块")
        img, code = render_ean13()
        hits = cvter.decode_barcode(img)
        self.assertTrue(hits, "一维码未解出")
        self.assertIn(code, hits[0].text)
        viz.assert_written(self, viz.save_panels("code", "02_ean13", [
            ("input", img),
            (f"PASS {hits[0].format}", draw_hits(img, hits)),
        ]))

    def test_ocr(self):
        if not cvter.ocr_available():
            self.skipTest("未编译 Tesseract")
        img = np.full((96, 420), 255, np.uint8)
        cv2.putText(img, "HELLO", (24, 64), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 3, cv2.LINE_AA)
        result = cvter.recognize_text(img, "eng")
        self.assertTrue(result.found, result.text)
        self.assertEqual(result.backend, "tesseract")
        self.assertIn("HELLO", result.text.upper())
        view = viz.to_bgr(img)
        viz.put_text(view, f"OCR {result.text}", (8, 88), viz.GREEN, 0.5)
        viz.assert_written(self, viz.save_panels("code", "03_ocr", [("PASS OCR", view)]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
