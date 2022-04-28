# Camera - LiDAR Sensor Fusion
This code is part of a small project on sensor fusion I did during my Bachelor's degree.
The [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) benchmark data set was used to evaluate the sensor fusion methods.


```
usage: sci_proj.py [-h] [--visualize_2D] [--visualize_3D] [--visualize_depth {1,2,3}] [-y YOLO_LABEL_OUTPUT_FOLDER] [-csv SAVE_RESULTS_AS_TABLE] base_path

positional arguments:
  base_path             Your KITTI data is saved there. It has to contain the following folders: data_object_image_2, data_object_calib, data_object_velodyne, data_object_label_2

optional arguments:
  -h, --help            show this help message and exit
  --visualize_2D        Visualize the 2D-Bounding Boxes
  --visualize_3D        Visualize the 3D-Bounding Boxes
  --visualize_depth {1,2,3}
                        1: Depth Map 2: Depth Map on image 3: Depth Map on image (only in 2D bounding boxes)
  -y YOLO_LABEL_OUTPUT_FOLDER, --yolo_label_output_folder YOLO_LABEL_OUTPUT_FOLDER
                        Provide a valid folder where the labels can be saved to. They will additionally include the distance in meters
  -csv SAVE_RESULTS_AS_TABLE, --save_results_as_table SAVE_RESULTS_AS_TABLE
                        Provide a valid folder where the reults can be saved as as .csv file.
```
