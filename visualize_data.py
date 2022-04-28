#!/usr/bin/env python3
import cv2
import numpy as np
from random import randrange
from math import ceil, floor
from utility import get_bbox_corners


def draw_2d_bbox(src, objects, **kwargs):
    """
    Draw all 2D bounding boxes on an image.
    """

    img = src.copy()

    for obj in objects:
        if "color" in kwargs:
            color = kwargs["color"]
        else:
            color = (randrange(256), randrange(256), randrange(256))

        # Draw the bounding box
        top_left = (floor(obj["x1"]), floor(obj["y1"]))
        bottom_right = (ceil(obj["x2"]), ceil(obj["y2"]))
        cv2.rectangle(img, top_left, bottom_right, color)

        # Show class name
        cv2.putText(img, obj["type"], (floor(obj["x1"]),
                                       floor(obj["y1"])-2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

    return img


def draw_3d_bbox(src, objects, P2, **kwargs):
    """
    Draw all 3D bounding boxes on an image.
    """

    img = src.copy()

    for obj in objects:
        if "color" in kwargs:
            color = kwargs["color"]
        else:
            color = (randrange(256), randrange(256), randrange(256))

        corners = get_bbox_corners(obj)

        # Don't draw bounding boxes behind the camera
        if not all(z > 0.1 for z in corners[2]):
            continue

        # Expand matrix
        corners = np.vstack((corners, np.ones((1, corners.shape[1]))))
        # Project to image plane
        vertices = P2 @ corners
        vertices = (vertices[:2, :] / vertices[2, :]).round().astype("int")

        # Indices of vertices that will be connected with lines
        vertex_pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [
                                2, 6], [3, 7], [5, 6], [6, 7], [7, 4], [4, 5]])
        for vp in vertex_pairs:
            cv2.line(img, vertices.T[vp[0]],
                     vertices.T[vp[1]], color, lineType=cv2.LINE_AA)

    return img


def draw_depth_map(depth_map, **kwargs):
    """
    Draw the depth map with distance encoded in color.

    If an image is given, the depth will be drawn on this image.
    """

    height, width = depth_map.shape
    img = np.zeros((height, width, 3), np.uint8)
    # img = np.full((height, width, 3), (56, 56, 56), np.uint8)

    depth_map_bin = np.zeros((height, width), np.uint8)
    depth_map_bin[depth_map > 0] = 255

    from matplotlib import cm
    cmap = cm.get_cmap("jet", np.max(depth_map) * 10.0)

    dm = np.copy(depth_map)
    dm = dm / np.max(dm)  # Scale to [0, 1]

    # TODO: Slow. How to improve?
    for y in range(height):
        for x in range(width):
            if dm[y, x] > 0:
                rgb = np.array(cmap(dm[y, x])) * 255  # Get color value
                img[y, x] = rgb[:3]

    if "src" in kwargs:
        cv2.bitwise_or(img, kwargs["src"], dst=img,
                       mask=cv2.bitwise_not(depth_map_bin))

    return img
