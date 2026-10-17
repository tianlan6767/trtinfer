"""合成图，可视化测试和数值断言共用。不 import cvter。"""
from __future__ import annotations

import math

import cv2
import numpy as np


def paste(dst, patch, x, y):
    h, w = patch.shape[:2]
    dst[y:y + h, x:x + w] = patch


def make_asymmetric_patch(h=56, w=88):
    img = np.full((h, w), 40, np.uint8)
    img[8:h - 8, 8:w - 8] = 210
    img[8:22, 8:28] = 40
    img[h - 20:h - 8, w - 36:w - 8] = 90
    return img


def make_l_shape(h=64, w=80):
    img = np.full((h, w), 30, np.uint8)
    img[8:h - 8, 8:24] = 220
    img[h - 24:h - 8, 8:w - 8] = 220
    return img


def _blur(img, sigma=0.8):
    return cv2.GaussianBlur(img, (5, 5), sigma)


def vertical_step(w, h, x_edge, left=200, right=40):
    img = np.empty((h, w), np.uint8)
    img[:, :x_edge] = left
    img[:, x_edge:] = right
    return _blur(img)


def horizontal_step(w, h, y_edge, top=40, bottom=200):
    img = np.empty((h, w), np.uint8)
    img[:y_edge, :] = top
    img[y_edge:, :] = bottom
    return _blur(img)


def dark_disk(size=280, cx=150.0, cy=140.0, r=60.0, bg=200, fg=25):
    img = np.full((size, size), bg, np.uint8)
    yy, xx = np.ogrid[:size, :size]
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = fg
    return _blur(img, 1.0)


def two_falling_edges(w=160, h=180):
    img = np.empty((h, w), np.uint8)
    img[:, :40] = 220
    img[:, 40:100] = 80
    img[:, 100:] = 30
    return _blur(img)


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


def textured_patch(h=160, w=200, seed=3):
    rng = np.random.RandomState(seed)
    img = np.full((h, w), 235, np.uint8)
    for _ in range(36):
        x = int(rng.randint(4, w - 24))
        y = int(rng.randint(4, h - 24))
        ww = int(rng.randint(8, 28))
        hh = int(rng.randint(6, 22))
        img[y:y + hh, x:x + ww] = int(rng.randint(0, 60))
    cv2.rectangle(img, (12, 12), (w - 13, h - 13), 15, 2)
    cv2.circle(img, (w // 3, h // 2), 16, 20, 2)
    return img
