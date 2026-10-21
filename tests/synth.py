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


def shift_image(img, dx, dy, fill=0):
    """整像素平移。out[y+dy, x+dx] = img[y, x]，越界填 fill。"""
    out = np.full_like(img, fill)
    h, w = img.shape[:2]
    src_x0 = max(0, -int(dx))
    src_y0 = max(0, -int(dy))
    dst_x0 = max(0, int(dx))
    dst_y0 = max(0, int(dy))
    tw = w - abs(int(dx))
    th = h - abs(int(dy))
    if tw <= 0 or th <= 0:
        return out
    out[dst_y0:dst_y0 + th, dst_x0:dst_x0 + tw] = img[src_y0:src_y0 + th, src_x0:src_x0 + tw]
    return out


def make_golden_plate(h=240, w=320):
    """几何金样。亮缺陷放在 (112, 44, 32, 18)，暗缺陷圆心 (214, 168)。"""
    img = np.full((h, w), 160, np.uint8)
    cv2.rectangle(img, (12, 12), (w - 13, h - 13), 30, 3)
    cv2.circle(img, (78, 86), 24, 80, -1)
    cv2.rectangle(img, (150, 48), (240, 120), 210, -1)
    cv2.line(img, (36, 190), (280, 210), 45, 5)
    cv2.rectangle(img, (40, 150), (54, 160), 20, -1)
    cv2.rectangle(img, (260, 40), (276, 52), 240, -1)
    cv2.rectangle(img, (40, 40), (52, 50), 15, -1)
    cv2.rectangle(img, (180, 180), (198, 194), 245, -1)
    return img


def make_pose_plate(h=260, w=340, seed=5):
    """四周高对比纹理给 ORB，中间 160×140 平场放缺陷。"""
    rng = np.random.RandomState(seed)
    img = np.full((h, w), 210, np.uint8)
    for _ in range(90):
        ww = int(rng.randint(8, 26))
        hh = int(rng.randint(6, 20))
        x = int(rng.randint(6, w - ww - 6))
        y = int(rng.randint(6, h - hh - 6))
        img[y:y + hh, x:x + ww] = int(rng.randint(0, 45))
    cv2.rectangle(img, (4, 4), (w - 5, h - 5), 15, 2)
    cv2.circle(img, (28, 28), 10, 250, -1)
    cv2.rectangle(img, (w - 44, 16), (w - 22, 34), 250, -1)
    cv2.circle(img, (26, h - 26), 12, 20, -1)
    cv2.rectangle(img, (w - 50, h - 40), (w - 22, h - 26), 30, -1)
    cv2.rectangle(img, (w // 2 - 80, h // 2 - 70), (w // 2 + 80, h // 2 + 70), 180, -1)
    return img


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
