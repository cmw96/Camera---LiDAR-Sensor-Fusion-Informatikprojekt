#!/usr/bin/env python3

import cv2
import numpy as np


def get_train_mat(img, depth_map, normalize_pixel_values=True, normalize_depth_values=True):
    """
    Combine rgb image with depth information.
    This may be used as input to train a RGBD-YOLO network.

    If both pixel and depth values are scaled, the resulting matrix contains four channels (r, g, b, d)
    with values in the range [0.0, 1.0].
    If no normalization is requested, the r, g and b values are in the range [0, 255] and the depth channel contains
    meters in the range [0.0, 120.0].
    """

    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    if normalize_pixel_values:
        rgba = rgba / 255.0

    img_float = np.float32(rgba)

    if normalize_depth_values:
        # Maximum range of Velodyne HDL-64E sensor is 120 meters (Geiger et al. 2013)
        img_float[:, :, 3] = (depth_map / 120.0)
    else:
        img_float[:, :, 3] = depth_map

    return rgba
