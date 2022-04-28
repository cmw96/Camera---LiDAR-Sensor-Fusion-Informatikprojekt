#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd

from pathlib import Path


def get_image(dir, filename):
    """
    Read KITTI image from disk.
    """
    return cv2.imread(str(Path(dir, filename).with_suffix(".png")))


def get_pointcloud(dir, filename):
    """
    Read KITTI pointcloud from disk.
    """
    return np.fromfile(Path(dir, filename).with_suffix(".bin"), dtype=np.float32).reshape((-1, 4))


def get_calibration_matrices(dir, filename):
    """
    Read all available matrices from calibration file.

    Returns six matrices in total.
    """
    calib_file = pd.read_csv(Path(dir, filename).with_suffix(
        ".txt"), delimiter=' ', header=None, index_col=0)
    P0 = np.array(calib_file.loc["P0:"]).reshape((3, 4))
    P1 = np.array(calib_file.loc["P1:"]).reshape((3, 4))
    P2 = np.array(calib_file.loc["P2:"]).reshape((3, 4))
    P3 = np.array(calib_file.loc["P3:"]).reshape((3, 4))
    R0_rect = np.array(calib_file.loc["R0_rect:"][:9]).reshape((3, 3))
    Tr_velo_to_cam = np.array(
        calib_file.loc["Tr_velo_to_cam:"]).reshape((3, 4))

    return P0, P1, P2, P3, R0_rect, Tr_velo_to_cam


def get_labels(dir, filename):
    """
    Read a KITTI label file from disk.
    """
    return pd.read_csv(Path(dir, filename).with_suffix(".txt"), delimiter=' ', header=None)


def save_yolo_labels(dir, filename, yolo_labels):
    """
    Write a list of strings to a text file.

    Each label (one for each object) is placed in a new line.
    """
    with open(Path(dir, filename).with_suffix(".txt"), 'w') as yololabelfile:
        for label in yolo_labels:
            yololabelfile.write(label + '\n')
