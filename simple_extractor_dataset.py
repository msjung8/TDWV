#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   dataset.py
@Time    :   8/30/19 9:12 PM
@Desc    :   Dataset Definition
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import cv2
import numpy as np

from torch.utils import data
from utils.transforms import get_affine_transform


class SimpleFolderDataset(data.Dataset):
    #def __init__(self, root, input_size=[512, 512], transform=None):
    def __init__(self, root, input_size_list, transform=None):
        self.root = root
        self.input_size_list = input_size_list
        self.transform = transform
        self.aspect_ratio_list=[]
        self.input_size=[]
        for ip in input_size_list:
            self.aspect_ratio_list.append(ip[1] * 1.0 / ip[0])
            self.input_size.append(np.asarray(ip))
        self.file_list = []
        for file in os.listdir(self.root):
            if file.endswith('.jpg') or file.endswith('.png'):
                self.file_list.append(file)

    def __len__(self):
        return len(self.file_list)

    def _box2cs(self, box, idx):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h, idx)

    def _xywh2cs(self, x, y, w, h, idx):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio_list[idx] * h:
            h = w * 1.0 / self.aspect_ratio_list[idx]
        elif w < self.aspect_ratio_list[idx] * h:
            w = h * self.aspect_ratio_list[idx]
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def __getitem__(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.root, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, _ = img.shape

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1], index)
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size[index])
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[index][1]), int(self.input_size[index][0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        input = self.transform(input)
        meta = {
            'name': img_name,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return input, meta
