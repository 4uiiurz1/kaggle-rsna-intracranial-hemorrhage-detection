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
import joblib


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]

        img = cv2.imread(img_path)
        if img is None:
            print('%s does not exist' %img_path)
            img = np.zeros((512, 512, 3), 'uint8')

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return img, label


    def __len__(self):
        return len(self.img_paths)


class DatasetV2(torch.utils.data.Dataset):
    def __init__(self, df, depth_size, transform=None):
        if not os.path.exists('processed/instances.pkl'):
            self.img_paths = []
            self.labels = []
            study_ids = df['StudyInstanceUID'].values
            for study_id in tqdm(np.unique(study_ids), total=len(np.unique(study_ids))):
                df_ = df[study_ids == study_id].sort_values('Axial')
                self.img_paths.append(df_['img_path'].values)
                self.labels.append(df_[['label_%d' %c for c in range(6)]].values)
            joblib.dump({'img_path': self.img_paths, 'label': self.labels}, 'processed/instances.pkl')
        else:
            data = joblib.load('processed/instances.pkl')
            self.img_paths = data['img_path']
            self.labels = data['label']
        self.depth_size = depth_size
        self.transform = transform

    def __getitem__(self, index):
        img_paths, labels = self.img_paths[index], self.labels[index]
        try:
            idx = random.randrange(len(img_paths) - self.depth_size)
        except:
            idx = 0
        img_paths = img_paths[idx:idx + self.depth_size]
        labels = labels[idx:idx + self.depth_size]

        imgs = []
        for img_path in img_paths:
            img = cv2.imread(img_path)

            if img is None:
                print('%s does not exist' %img_path)
                img = np.zeros((512, 512, 3), 'uint8')

            if self.transform is not None:
                img = self.transform(image=img)['image']

            imgs.append(img.unsqueeze(0))
        imgs = torch.cat(imgs, 0)
        labels = torch.from_numpy(labels)

        return imgs, labels


    def __len__(self):
        return len(self.img_paths)
