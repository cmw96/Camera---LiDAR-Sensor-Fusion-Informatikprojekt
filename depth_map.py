#!/usr/bin/env python3

from math import ceil, floor
from statistics import variance
import numpy as np
import cv2


def get_depth_map(pointcloud, P2, R0_rect, Tr_velo_to_cam, height, width, **kwargs):
    """
    Project all points from the velodyne pointcloud on the image plane of camera 2 (color).

    We split the process into multiple parts instead of using P2 directly:
    0) Extract K, R and t from P2
    1) Transform all points from the Lidar coordinate system to the camera2 coordinate system using Tr_velo_to_cam, R0_rect and [R|t]
    2) Project the points to the image plane using K
    3) Conversion from homogeneous coordinates to pixel coordinates by dividing the resulting vector trough the third entry in the vector

    We do this to:
    1) Be able to filter all points behind the camera
    2) Determine the distance to a specific point as seen from the camera

    Optional: Provide a list of objects. Their 2D bounding boxes will be used as mask. Only points inside the 2D bounding boxes will added to the depth map.
    """

    depth_img = np.zeros((height, width), np.float32)

    points_xyz_velo = pointcloud[:, :3].T
    points_xyz_velo = np.vstack(
        (points_xyz_velo, np.ones((1, points_xyz_velo.shape[1]))))

    # Prepare necessary matrices.
    K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P2)
    t = t / t[3]
    R_t = np.column_stack([R, t[:3]])
    R_t = np.vstack([R_t, [0, 0, 0, 1]])
    # See https://github.com/FoamoftheSea/KITTI_visual_odometry/blob/main/KITTI_visual_odometry.ipynb for an explanation why the matrix has to be inverted
    R_t = np.linalg.inv(R_t)
    K = np.column_stack([K, [0, 0, 0]])

    # Transform Lidar points to camera2 coordinate system
    points_xyz_cam2 = R_t @ R0_rect @ Tr_velo_to_cam @ points_xyz_velo
    # Remove all points behind the camera
    points_xyz_cam2 = points_xyz_cam2[:, points_xyz_cam2[2, :] > 0]

    # Project points into image plane
    pixels_cam2 = K @ points_xyz_cam2
    # Convert from homogeneous coordinates to real pixel values
    pixels_cam2 = (pixels_cam2[:2, :] /
                   pixels_cam2[2, :]).round().astype("int")

    # Select only those pixels which are within image boundaries
    inds = np.where((pixels_cam2[0, :] < width) & (pixels_cam2[0, :] >= 0) &
                    (pixels_cam2[1, :] < height) & (pixels_cam2[1, :] >= 0))[0]

    # Numpy indexing: [height, width]
    # Use the 3D distance and assign it to the corresponding pixel
    for i in inds:
        d = np.linalg.norm(
            [points_xyz_cam2[0, i], points_xyz_cam2[1, i], points_xyz_cam2[2, i]])
        depth_img[pixels_cam2[1, i], pixels_cam2[0, i]] = d

    mask = None
    if "objects" in kwargs:
        objects = kwargs["objects"]
        mask = np.zeros((height, width), np.uint8)

        for obj in objects:
            pt1 = (floor(obj["x1"]), floor(obj["y1"]))
            pt2 = (ceil(obj["x2"]), ceil(obj["y2"]))
            cv2.rectangle(mask, pt1, pt2, 255, cv2.FILLED)

        depth_img[mask <= 0] = 0

    return depth_img


def calc_variance(obj, depth_img):
    # Define ROI
    height, width = depth_img.shape
    mask = np.zeros((height, width), np.uint8)
    pt1 = (floor(obj["x1"]), floor(obj["y1"]))
    pt2 = (ceil(obj["x2"]), ceil(obj["y2"]))
    cv2.rectangle(mask, pt1, pt2, 255, cv2.FILLED)

    # Select only distances in ROI
    distances = depth_img[mask > 0]
    # Select only distances > 0 meters
    distances = distances[distances > 0]

    if len(distances) <= 0:
        return np.nan

    return np.var(distances)
