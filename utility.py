#!/usr/bin/env python3

import numpy as np
import cv2
from math import cos, sin


def get_target_pixel(obj, img_width):
    """
    Get the target pixel for distance estimation via ray-plane-intersection.

    The target pixel lies on the 2D bounding box bottom edge.
    On this edge, the pixel which is closest to the camera center is selected.
    """

    pix_y = obj["y2"]
    pix_x = 0

    x1 = obj["x1"]
    x2 = obj["x2"]

    img_width_half = img_width / 2
    if x1 < img_width_half and x2 > img_width_half:
        pix_x = img_width_half
    elif abs(img_width_half - x1) < abs(img_width_half - x2):
        pix_x = x1
    else:
        pix_x = x2

    return (pix_x, pix_y)


def get_principal_point(P2):
    """
    The principal point is the intersection point between the optical axis and the image plane.
    -> See Figure 14.3 in "Computer vision: models, learning and inference" by Simon J.D. Prince, 2012
    """

    K, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(P2)
    return (K[0, 2], K[1, 2])


def get_bbox_corners(obj):
    """
    Convert 3D bounding box (x, y, z, w, h, l) to a set of eight corners.

    Code adapted from KITTI devkit "computeBox3D.m"
    """

    l = obj["length"]
    h = obj["height"]
    w = obj["width"]
    x_values = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_values = [0, 0, 0, 0, -h, -h, -h, -h]
    z_values = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # Rotation matrix around y-axis (in camera coordinates "down")
    c = cos(obj["rot_y"])
    s = sin(obj["rot_y"])
    R = np.array([[+c, 0, s],
                  [+0, 1, 0],
                  [-s, 0, c]])

    # Rotate
    corners = np.dot(R, np.vstack([x_values, y_values, z_values]))

    # Translate
    corners[0, :] += obj["x"]
    corners[1, :] += obj["y"]
    corners[2, :] += obj["z"]

    # print(corners)

    return corners
