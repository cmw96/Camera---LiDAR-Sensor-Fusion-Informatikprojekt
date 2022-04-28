#!/usr/bin/env python3

"""
This code is an adaption from https://github.com/AlexeyAB/darknet/blob/master/scripts/kitti2yolo.py

An additional distance information (the minimal distance to the 3D-Bounding Box in the GT, except for DontCare objects) is appended to each label.
"""
kitti2yolotype_dict = {'Car': '0',
                       'Van': '0',
                       'Pedestrian': '1',
                       'Person_sitting': '1',
                       'Cyclist': '2',
                       'Truck': '3',
                       'Tram': '4',
                       'Misc': '4',
                       'DontCare': '4'}


def kitti2yolo(obj, img_height, img_width, normalize_distance=True, use_alternative_gt=False):
    bb_width = obj["x2"] - obj["x1"]
    bb_height = obj["y2"] - obj["y1"]
    yolo_x = (obj["x1"] + 0.5*bb_width) / img_width
    yolo_y = (obj["y1"] + 0.5*bb_height) / img_height
    yolo_bb_width = bb_width / img_width
    yolo_bb_height = bb_height / img_height
    yolo_type = kitti2yolotype_dict[obj["type"]]

    label = (yolo_type + " "
             + str(yolo_x) + " "
             + str(yolo_y) + " "
             + str(yolo_bb_width) + " "
             + str(yolo_bb_height)) + " "

    # No ground truth information and no Lidar information available
    distance = 0.0
    if "ground_truth" in obj:
        # For all objects except "DontCare" objects a 3D-bounding box is provided and therefore the ground truth distance is available
        distance = obj["ground_truth"]
    elif use_alternative_gt and "depth_map_median" in obj:
        # If Lidar data is available for the "DontCare" region, then use this depth information
        distance = obj["depth_map_median"]

    if normalize_distance:
        distance /= 120.0

    label += str(distance)

    return label


def get_yolo_labels(objects, img_height, img_width):
    labels = []
    for obj in objects:
        labels.append(kitti2yolo(obj, img_height, img_width))

    return labels
