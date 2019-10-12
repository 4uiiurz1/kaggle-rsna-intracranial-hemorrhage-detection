import random
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
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


class ForegroundCenterCrop(ImageOnlyTransform):
    def __init__(self, crop_size, always_apply=False, p=1.0):
        super(ForegroundCenterCrop, self).__init__(always_apply, p)
        self.crop_size = crop_size

    def apply(self, img, **params):
        yx = img.mean(axis=2)
        h, w = yx.shape
        # plt.imshow((yx > yx.mean() / 10).astype('uint8'))
        # plt.show()
        region_props = measure.regionprops((yx > yx.mean() / 10).astype('uint8'))
        if region_props:
            idx = np.argmax([p.area for p in region_props])
            yc, xc = np.round(region_props[idx].centroid).astype('int')
        else:
            yc, xc = h // 2, w // 2

        # print(xc, yc)
        xc = max(min(xc, w - self.crop_size // 2 - 1), self.crop_size // 2)
        yc = max(min(yc, h - self.crop_size // 2 - 1), self.crop_size // 2)
        # print(xc, yc)
        x1 = max(xc - self.crop_size // 2, 0)
        x2 = min(xc + self.crop_size // 2, w - 1)
        y1 = max(yc - self.crop_size // 2, 0)
        y2 = min(yc + self.crop_size // 2, h - 1)
        img = img[y1:y2, x1:x2]
        # plt.imshow(img)
        # plt.show()

        return img
