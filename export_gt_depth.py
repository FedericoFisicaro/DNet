# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os

import argparse
import numpy as np
import PIL.Image as pil

from utils import readlines
from kitti_utils import generate_depth_map


def export_gt_depths_kitti():
    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["eigen", "eigen_benchmark","nyu_Depth","umonsALL","umonsH1","umonsH2","umonsH3","umonsH1-H2","umonsH1-H3","umonsH2-H3","BigRoom-H1","BigRoom-H2","BigRoom-H3","DevRoom-H1","DevRoom-H2","DevRoom-H3","OF2-H1","OF2-H2","OF2-H3","OF1-H1","OF1-H2","OF1-H3","ref","obj"])
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []
    for line in lines:
        if opt.split == "nyu_Depth" or "umons" in opt.split or "-H" in opt.split or opt.split == "obj" or opt.split == "ref":
            #folder = (line.split()[0]).split("/")[0]
            
            #frame_name = (line.split()[0]).split('/')[-1]
            #start = frame_name.index( "_" ) + 1
            #end = frame_name.index( ".", start )
            #frame_id= int(frame_name[start:end])

            gt_depth_path = os.path.join(
                opt.data_path, line.split()[1])
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 1000

        else:
            folder, frame_id, _ = line.split()
            frame_id = int(frame_id)

            if opt.split == "eigen":
                calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
                velo_filename = os.path.join(
                    opt.data_path, folder,
                    "velodyne_points/data", "{:010d}.bin".format(frame_id))
                gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
            elif opt.split == "eigen_benchmark":
                gt_depth_path = os.path.join(
                    opt.data_path, folder, "proj_depth",
                    "groundtruth", "image_02", "{:010d}.png".format(frame_id))
                gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256
            

        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_kitti()
