#!/usr/bin/env python3
import argparse
import pandas as pd
import os
import cv2
import numpy as np
from pathlib import Path
from make_yolo_labels import get_yolo_labels

from read_write_data import *
from visualize_data import *
from training_data import *
from depth_map import calc_variance, get_depth_map
from distances import get_distance_depth_map, get_distance_gt, get_distance_ray_plane, get_principal_point, get_target_pixel


def dir_path(string):
    """
    Check if a provided path is valid.
    From https://stackoverflow.com/a/51212150
    """

    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == '__main__':

    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize_2D", action="store_true",
                        help="Visualize the 2D-Bounding Boxes")
    parser.add_argument("--visualize_3D", action="store_true",
                        help="Visualize the 3D-Bounding Boxes")
    parser.add_argument("--visualize_depth",
                        type=int, choices=[1, 2, 3], help="1: Depth Map 2: Depth Map on image 3: Depth Map on image (only in 2D bounding boxes)")
    parser.add_argument("-y", "--yolo_label_output_folder", type=dir_path,
                        help="Provide a valid folder where the labels can be saved to. They will additionally include the distance in meters")
    parser.add_argument("-csv", "--save_results_as_table", type=dir_path,
                        help="Provide a valid folder where the reults can be saved as as .csv file.")
    parser.add_argument("base_path", type=dir_path,
                        help="Your KITTI data is saved there. It has to contain the following folders: data_object_image_2, data_object_calib, data_object_velodyne, data_object_label_2")
    args = parser.parse_args()

    # Construct path to images, calibration files, pointclouds and labels
    image_path = os.path.join(
        args.base_path, "data_object_image_2/training/image_2/")
    calib_path = os.path.join(
        args.base_path, "data_object_calib/training/calib/")
    pointcloud_path = os.path.join(
        args.base_path, "data_object_velodyne/training/velodyne/")
    label_path = os.path.join(
        args.base_path, "data_object_label_2/training/label_2/")

    objects_total = []

    pathlist = Path(label_path).glob('**/*.txt')
    for path in pathlist:
        id = path.stem
        # print("")
        print("ID: {}".format(id))

        pointcloud = get_pointcloud(pointcloud_path, id)
        labels = get_labels(label_path, id)
        img = get_image(image_path, id)
        height, width, channels = img.shape
        _, _, P2, _, R0_rect, Tr_velo_to_cam = get_calibration_matrices(
            calib_path, id)

        # Prepare matrices for matmul
        R0_rect = np.vstack([R0_rect, [0, 0, 0]])
        R0_rect = np.column_stack([R0_rect, [0, 0, 0, 1]])
        Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])

        depth_map = get_depth_map(
            pointcloud, P2, R0_rect, Tr_velo_to_cam, height, width)

        objects_per_image = []
        for _, row in labels.iterrows():
            # Extract information from label file. One row represents one object.
            obj = {
                "type": row[0],
                "truncated": row[1],
                "occluded": row[2],
                "alpha": row[3],
                "x1": row[4],
                "y1": row[5],
                "x2": row[6],
                "y2": row[7],
                "height": row[8],
                "width": row[9],
                "length": row[10],
                "x": row[11],
                "y": row[12],
                "z": row[13],
                "rot_y": row[14],
            }

            obj["ray_target_pixel_x"], obj["ray_target_pixel_y"] = get_target_pixel(
                obj, width)
            obj["principal_point_x"], obj["principal_point_y"] = get_principal_point(
                P2)

            try:
                obj["depth_map_minimum"] = get_distance_depth_map(
                    obj, depth_map, "min")
                obj["depth_map_average"] = get_distance_depth_map(
                    obj, depth_map, "avg")
                obj["depth_map_median"] = get_distance_depth_map(
                    obj, depth_map, "median")
            except RuntimeError as re:
                print(re)

            try:
                obj["ray_plane_intersection"] = get_distance_ray_plane(obj, P2)
            except RuntimeError as re:
                print(re)

            if obj["type"] != "DontCare":
                obj["ground_truth"] = get_distance_gt(obj)

            obj["frustrum_distance_variance"] = calc_variance(obj, depth_map)

            # Add to the list of existing objects
            objects_per_image.append(obj)

        objects_total += objects_per_image

        if args.yolo_label_output_folder:
            labels = get_yolo_labels(objects_per_image, height, width)
            save_yolo_labels(args.yolo_label_output_folder, id, labels)

        # Show image if any visualizations were requested
        if args.visualize_2D or args.visualize_3D or args.visualize_depth:

            if args.visualize_depth:
                if args.visualize_depth == 1:
                    out = draw_depth_map(depth_map)
                if args.visualize_depth == 2:
                    out = draw_depth_map(depth_map, src=img)
                if args.visualize_depth == 3:
                    depth_map_partial = get_depth_map(
                        pointcloud, P2, R0_rect, Tr_velo_to_cam, height, width, objects=objects_per_image)
                    out = draw_depth_map(
                        depth_map_partial, src=img)
                    out = draw_2d_bbox(out, objects_per_image)

                cv2.imshow(str(id) + " (Depth Map)", out)

            if args.visualize_2D:
                out = draw_2d_bbox(img, objects_per_image)
                for obj in objects_per_image:
                    pt = (round(obj["ray_target_pixel_x"]),
                          round(obj["ray_target_pixel_y"]))
                    cv2.circle(out, pt, 4, (0, 0, 255))

                cv2.imshow(str(id) + " (2D Bounding Boxes)", out)

            if args.visualize_3D:
                out = draw_3d_bbox(img, objects_per_image, P2)
                cv2.imshow(str(id) + " (3D Bounding Boxes)", out)

            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("img.png", out)
            cv2.destroyAllWindows()

    if args.save_results_as_table:
        # Save results to disk as .csv
        timestamp = pd.Timestamp.now()
        df = pd.DataFrame(objects_total)
        df.to_csv(Path(args.save_results_as_table,
                  str(timestamp)).with_suffix(".csv"))
