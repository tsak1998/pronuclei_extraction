#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from pathlib import Path
from PIL import Image
from pydantic import BaseModel
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import binary_closing, binary_dilation, binary_erosion
from torchvision.transforms.functional import rotate


def mask_orientation_centroid(image: np.ndarray[int]):
    label_img = label(image)
    regions = regionprops((label_img))

    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
    return props.centroid, orientation, props.axis_major_length, props.axis_minor_length


def rotate_image(image_arr: Image.Image, angle: float | None = None):
    if angle is not None:
        return np.array(rotate(image_arr, angle)), angle
    _, orientation, _, _ = mask_orientation_centroid(np.array(image_arr))
    rotation_angle = -np.rad2deg(orientation) + 90
    print("Orientation (deg):", np.rad2deg(orientation))
    return np.array(rotate(image_arr, rotation_angle)), rotation_angle


def find_signal(arr: np.ndarray) -> tuple[int, int]:
    max_len = max_start = curr_len = curr_start = 0
    for i, val in enumerate(arr):
        if val:
            if curr_len == 0:
                curr_start = i
            curr_len += 1
        else:
            if curr_len > max_len:
                max_len, max_start = curr_len, curr_start
            curr_len = 0
    if curr_len > max_len:
        max_len, max_start = curr_len, curr_start
    return max_start, max_len


def fit_circle(x, y):
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x**2 + y**2
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    x0, y0 = c[0], c[1]
    r = np.sqrt(c[2] + x0**2 + y0**2)
    return x0, y0, r


def get_circle_pts(x0, y0, r, npts=100, tmin=0, tmax=2 * np.pi):
    t = np.linspace(tmin, tmax, npts)
    x = x0 + r * np.cos(t)
    y = y0 + r * np.sin(t)
    return x, y


# --- Paths & Load ---
sample_id = 'D2016.01.11_S1183_I149_1'
base_pth = Path('/Users/tsakalis/downloads')

masks_pth = base_pth / f'{sample_id}.npy'
timelapse_pth = base_pth / f'{sample_id}_images'
all_image_paths = sorted(timelapse_pth.glob('*'),
                         key=lambda x: int(x.stem.split('_')[0]))[:200]
all_masks = np.load(masks_pth)

# --- Process specific frame ---
frame_idx = 100
whole_img = Image.open(all_image_paths[frame_idx])
mask_img = Image.fromarray(
    binary_closing(all_masks[frame_idx]).astype(np.uint8))


class PnCircle(BaseModel):
    x: float
    y: float
    r: float


class CirclesFit(BaseModel):
    pn1: PnCircle
    pn2: PnCircle

    @classmethod
    def randomize_pns(cls):
        ...


def fit_pn_circles(image: Image, mask: Image) -> CirclesFit:

    rotated_mask, angle = rotate_image(mask_img)
    centroid, _, major_len, minor_len = mask_orientation_centroid(
        rotated_mask == 1)

    rotated_whole, _ = rotate_image(whole_img, angle)

    # --- Crop around PN ---
    y0, x0 = centroid
    a1, b1 = int(x0 - major_len * 0.6), int(x0 + major_len * 0.65)
    a2, b2 = int(y0 - minor_len * 0.6), int(y0 + minor_len * 0.65)

    smoothed_img = cv2.GaussianBlur(
        binary_erosion(rotated_mask)[a2:b2, a1:b1].astype(np.uint8), (15, 15),
        0)

    # --- Contour detection ---
    contours = find_contours(smoothed_img, None)
    fig, ax = plt.subplots()

    for contour in contours:
        half1 = contour[contour[:, 1] < smoothed_img.shape[1] // 2]
        half2 = contour[contour[:, 1] > smoothed_img.shape[1] // 2]

    x = half1[:, 1]
    y = -half1[:, 0]
    x0, y0, r = fit_circle(x, y)
    print(f"Fitted circle 1: center=({x0:.3f}, {y0:.3f}), radius={r:.3f}")
    # xf, yf = get_circle_pts(x0 + a1, -y0 + a2, r)

    pn_circle1 = PnCircle(x=x0 + a1, y=-y0 + a2, r=r)

    x = half2[:, 1]
    y = -half2[:, 0]
    x0, y0, r = fit_circle(x, y)
    print(f"Fitted circle 2: center=({x0:.3f}, {y0:.3f}), radius={r:.3f}")
    # xf, yf = get_circle_pts(x0 + a1, -y0 + a2, r)

    pn_circle2 = PnCircle(x=x0 + a1, y=-y0 + a2, r=r)

    return CirclesFit(pn1=pn_circle1, pn2=pn_circle2)
