# Copyright (c) 2018 Bhojpur Consulting Private Limited, India. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import random
import math
from PIL import Image
from skimage.transform import resize
import skimage
import torch
import matplotlib.pyplot as plt

class CustomResize(object):
    def __init__(self, network_type, trg_size):

        self.trg_size = trg_size
        self.network_type = network_type

    def __call__(self, img):
        resized_img = self.resize_image(img, self.trg_size)
        return resized_img

    def resize_image(self, img, trg_size):
        img_array = np.asarray(img.get_data())
        res = resize(img_array, trg_size, mode='reflect', anti_aliasing=False, preserve_range=True)

        # type check
        if type(res) != np.ndarray:
            raise "type error!"

        # PIL image cannot handle 3D image, only return ndarray type, which ToTensor accepts
        return res

class CustomToTensor(object):
    def __init__(self, network_type):

        self.network_type = network_type

    def __call__(self, pic):

        if isinstance(pic, np.ndarray):
            
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            
            # backward compatibility
            return img.float().div(255)
