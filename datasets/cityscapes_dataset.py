from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

class CityscapesDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(CityscapesDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[1.1, 0, 0.5, 0],
                           [0, 2.2, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (2048, 1024)
        # self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class CityscapesRAWDataset(CityscapesDataset):
    def __init__(self, *args, **kwargs):
        super(CityscapesRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        trainorval = folder.split('/')[0]
        location = folder.split('/')[1]
        sequencenbr = int((folder.split('/')[2]).split('_')[1])
        f_str = "{}_{:06d}_{:06d}_leftImg8bit{}".format(location,sequencenbr,frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, trainorval, location, f_str)
        return image_path

    def get_depth(self, folder, frame_name, do_flip):
        return False
