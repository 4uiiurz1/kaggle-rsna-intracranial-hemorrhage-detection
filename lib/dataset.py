import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import cv2
import numpy as np
import random
import scipy.ndimage as ndi
from tqdm import tqdm
import os
from PIL import Image
from skimage.io import imread
import pydicom

from albumentations.pytorch.functional import img_to_tensor


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]

        try:
            img = cv2.imread(img_path)
        except:
            img = np.zeros((512, 512, 3), 'uint8')

        if self.transform is not None:
            data = {'image': img}
            augmented = self.transform(**data)
            img = augmented['image']
            img = img_to_tensor(img)

        return img, label


    def __len__(self):
        return len(self.img_paths)
