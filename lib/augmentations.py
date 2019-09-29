import random
import math
import numpy as np
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform


class RandomErase(ImageOnlyTransform):
    def __init__(self, sl=0.02, sh=0.4, r=0.3, always_apply=False, p=0.5):
        super(RandomErase, self).__init__(always_apply, p)
        self.sl = sl
        self.sh = sh
        self.r = r

    def apply(self, img, **params):
        while True:
            area = random.uniform(self.sl, self.sh) * img.shape[0] * img.shape[1]
            ratio = random.uniform(self.r, 1/self.r)

            h = int(round(math.sqrt(area * ratio)))
            w = int(round(math.sqrt(area / ratio)))

            if h < img.shape[0] and w < img.shape[1]:
                x = random.randint(0, img.shape[0] - h)
                y = random.randint(0, img.shape[1] - w)
                img[x:x+h, y:y+w] = np.random.randint(0, 256, size=3, dtype='uint8')[None, None, :]

                return img
