from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

class NYUDataset(MonoDataset):
    """Superclass for NYU Depth dataset loader
    """
    def __init__(self, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.81, 0, 0.5, 0],
                           [0, 1.08, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (640, 480)
        # self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        velo_filename = os.path.join(
            self.data_path,
            line[1])

        return os.path.isfile(velo_filename)
        # return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class NYURAWDataset(NYUDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(NYURAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "rgb_{:05d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "sync_depth_{:05d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            f_str)
        
        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 1000

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
