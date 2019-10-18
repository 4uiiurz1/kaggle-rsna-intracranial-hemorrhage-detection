import random
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import cv2
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform
from albumentations.augmentations import functional as F


def resized_crop(image, height, width, x_min, y_min, x_max, y_max):
    image = F.crop(image, x_min, y_min, x_max, y_max)
    image = cv2.resize(image, (width, height))
    return image


class RandomResizedCrop(ImageOnlyTransform):
    def __init__(self, height, width, scale=(0.08, 1.0), ratio=(3/4, 4/3), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.scale = scale
        self.ratio = ratio

    def apply(self, image, **params):

        height, width = image.shape[:2]
        area = height * width

        for attempt in range(15):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5 and min(self.ratio) <= (h / w) <= max(self.ratio):
                w, h = h, w

            if w <= width and h <= height:
                x_min = random.randint(0, width - w)
                y_min = random.randint(0, height - h)
                return resized_crop(image, self.height, self.width, x_min, y_min, x_min+w, y_min+h)

        min_side = min(height, width)
        x_min = random.randint(0, width - min_side)
        y_min = random.randint(0, height - min_side)
        return resized_crop(image, self.height, self.width, x_min, y_min, x_min+min_side, y_min+min_side)


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
