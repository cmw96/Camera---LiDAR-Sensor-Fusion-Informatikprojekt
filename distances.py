#!/usr/bin/env python3

from utility import *

from statistics import median
import numpy as np
import cv2


def distance_planes(corners):
    """
    Calculate the distance to all six surfaces of the 3D bounding box.
    Each surface is defined by four points A, B, C, D.

    These surfaces are not infinite planes.
    Therefore we have to check if the nearest point x0 on the plane is inside this surface.
    Through the transformation of both x0 -> P and the surface to 2D (E, F, G, H) we can check if x0 is inside the rectangle defined by E and H.
    """

    plane_points = np.array([[0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7], [
                            0, 3, 4, 7], [0, 1, 3, 2], [4, 5, 7, 6]])

    valid_distances = []

    for points in plane_points:
        A = corners.T[points[0]]
        B = corners.T[points[1]]
        C = corners.T[points[2]]
        D = corners.T[points[3]]

        b = B - A
        b = b / np.linalg.norm(b)
        c = C - A
        c = c / np.linalg.norm(c)
        n = np.cross(b, c)
        n = n / np.linalg.norm(n)

        if np.dot(n, A) < 0:
            n *= -1

        # Transformation matrix T
        T = np.array(
            [
                [
                    [b[0], c[0], n[0], A[0]],
                    [b[1], c[1], n[1], A[1]],
                    [b[2], c[2], n[2], A[2]],
                    [0, 0, 0, 1],
                ]
            ]
        )

        M = np.linalg.inv(T)

        # Transform points from 3D to 2D
        E = M @ np.hstack((A, 1))
        # F = M @ np.hstack((B, 1))
        # G = M @ np.hstack((C, 1))
        H = M @ np.hstack((D, 1))

        E = E[0, :2]
        # F = F[0, :2]
        # G = G[0, :2]
        H = H[0, :2]

        # Hessian normal form E: n * x = d
        # n: Normalized normal vector
        # x: A point on the plane E (in this case point A)
        d = np.dot(n, A)

        # Get nearest point on plane
        x0 = d * n

        # Transform to 2D aswell
        P = M @ np.hstack((x0, 1))
        P = P[0, :2]
        x, y = P

        x1, y1 = E
        x2, y2 = H

        """
        2D case:

        ---------------------------H(x2, y2)
        |                                  |
        |         P(x, y)                  |
        |                                  |
        |                                  |
        E(x1, y1)---------------------------
        """

        # Check if P is inside the rectangle defined by E and H
        if x > x1 and x < x2 and y > y1 and y < y2:
            valid_distances.append(np.linalg.norm(x0))

    return valid_distances


def distance_corners(corners):
    """
    Return the euclidean distance to all corners of a 3D bounding box.
    """

    valid_distances = [np.linalg.norm(c) for c in corners.T]
    return valid_distances


def distance_edges(corners):
    """
    Calculate the distance to all 12 edges of a 3D bounding box.
    An edge is defined by two points A and B.

    Since these edges are not infinite lines, we have to check if the point x0 lies between the two points A and B.

      7 -------- 6
     /|         /|
    4 -------- 5 .
    | |        | |
    . 3 -------- 2
    |/         |/
    0 -------- 1

    ASCII art taken from https://github.com/kuixu/kitti_object_vis/blob/master/viz_util.py#L398
    """

    # Indices of vertices that will be connected with lines
    vertex_pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [
                            2, 6], [3, 7], [5, 6], [6, 7], [7, 4], [4, 5]])

    segments = []

    for vp in vertex_pairs:
        segments.append([corners.T[vp[0]], corners.T[vp[1]]])

    valid_distances = []
    for segment in segments:
        A = segment[0]
        B = segment[1]
        a = A
        b = B - A

        t0 = np.dot(-a, b) / (np.linalg.norm(b)**2)
        x0 = a + t0 * b
        d = np.linalg.norm(x0)

        d_x0_a = np.linalg.norm(x0 - a)
        d_x0_b = np.linalg.norm(x0 - B)
        d_a_b = np.linalg.norm(a - B)

        """
        Good: -----A-----x0------------B-------
        Bad: ------x0------A-----B----- or -----A--------B-----x0------
        """
        if d_x0_a < d_a_b and d_x0_b < d_a_b:
            valid_distances.append(d)

    return valid_distances


def get_distance_gt(obj):
    """
    Calculate the shortest distance to the 3D bounding box.

    This nearest point may be/lie on:
    1) One of the 8 corners
    2) One of the 12 edges
    3) One of the 6 surfaces
    """

    corners = get_bbox_corners(obj)
    d_edges = distance_edges(corners)
    d_corners = distance_corners(corners)
    d_planes = distance_planes(corners)

    return min(d_edges + d_corners + d_planes)


def get_distance_depth_map(obj, depth_map, method):
    """
    Estimate the distance to an object using the depth information inside a 2D bounding box.

    Different methods are available:
    1) Minimum
    2) Arithmetic mean
    3) Median

    Some "DontCare" objects may contain depth information.
    """

    roi = depth_map[int(obj["y1"]):int(obj["y2"]),
                    int(obj["x1"]):int(obj["x2"])]
    mask = np.zeros((roi.shape[0], roi.shape[1]), np.uint8)
    mask[roi > 0] = 255  # Ignore zero values

    if not roi[mask > 0].any():
        raise RuntimeError(
            "No Lidar data available for object with type \"{}\".".format(obj["type"]))

    if "min" == method:
        return min(roi[mask > 0])
    elif "avg" == method:
        return sum(roi[mask > 0]) / len(roi[mask > 0])
    elif "median" == method:
        return median(roi[mask > 0])
    else:
        raise NotImplementedError()


def ray_plane_intersection(Q, n, P, d, epsilon=1e-6):
    """
    Calculate the intersection between a line (defined by P and d) and a plane (defined by Q and n).

    See chapter 7.8.1 (page 168) in "Computer Graphics: Principles and Practice 3rd edition" by Hughes et al.
    """

    if (np.dot(d, n) < epsilon):
        raise RuntimeError(
            "No intersection with plane possible. Ray is parralel to plane or lies within the plane.")

    t = np.dot((Q - P), n) / np.dot(d, n)

    return P + t * d


def get_distance_ray_plane(obj, P2):
    """
    Estimate the distance to an object by constructing a ray through a pixel on the 2D bounding box.

    The location of the object is approximately the intersection between this ray and the plane that represents the floor.

    This intersection is only possible if the ray points towards the floor (the target pixel's y-position is below the principal point's y-position).
    """

    # Origin of coordinate system (z front, x right, y down)
    ray_point = np.array([0, 0, 0])
    # Representation of the road surface
    plane_normal = np.array([0, 1, 0])
    # Cameras are 1.65m above the road surface
    plane_point = np.array([0, 1.65, 0])

    # Homogeneous
    ray_target_pixel = np.array(
        [obj["ray_target_pixel_x"], obj["ray_target_pixel_y"]])
    ray_target_pixel = np.hstack((ray_target_pixel, 1))

    # Extract intrisic matrix K
    K, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(P2)

    if ray_target_pixel[1] <= obj["principal_point_y"]:
        # A valid intersection of the ray with the plane in front of the car is not possible otherwise
        raise RuntimeError(
            "The target pixel has to be below the principal point")

    # Construct a ray from the camera through the target pixel
    ray_direction = np.linalg.inv(K) @ ray_target_pixel

    p = ray_plane_intersection(
        plane_point, plane_normal, ray_point, ray_direction)

    if p[2] < 0:
        # Sanity check: Should never happen, since we make sure that the pixel lies below the principal point
        raise RuntimeError("The intersection lies behind the camera")

    return np.linalg.norm(p)
