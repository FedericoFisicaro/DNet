from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

class UmonsDataset(MonoDataset):
    """Superclass for Umons dataset loader
    """
    def __init__(self, *args, **kwargs):
        super(UmonsDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.41, 0, 0.51, 0],
                           [0, 0.73, 0.64, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1280, 544)
        # self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        return os.path.isfile(os.path.join(self.data_path,line[1]))
        #return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        

        return color


class UmonsRAWDataset(UmonsDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(UmonsRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "left{:06d}{}".format(frame_index, ".png")
        image_path = os.path.join(
            self.data_path, folder, f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "left_depth{:06d}.png".format(frame_index)
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
