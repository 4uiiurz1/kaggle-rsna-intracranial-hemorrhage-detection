import os
from glob import glob
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import pydicom
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from lib.preprocess import *


def convert_dcm_to_png(dataset, centers, widths):
    def process(img_path, output_dir, centers, widths):
        dst_path = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + '.png')
        if os.path.exists(dst_path):
            return
        dcm = pydicom.dcmread(img_path)
        _, _, intercept, slope = get_windowing(dcm)
        try:
            img = dcm.pixel_array
        except:
            return
        imgs = []
        for center, width in zip(centers, widths):
            imgs.append(window_image(img, center, width, intercept, slope))
        img = np.dstack(imgs)
        img = (img * 255).astype('uint8')
        cv2.imwrite(dst_path, img)

    if dataset == 'stage_1_test':
        df = pd.read_csv('inputs/stage_1_sample_submission.csv')
    else:
        df = pd.read_csv('inputs/%s.csv' %dataset)
    img_paths = np.array([os.path.join('inputs/%s_images/' %dataset, '_'.join(s.split('_')[:-1]) + '.dcm') for s in df['ID']][::6])
    img_paths = np.unique(img_paths)

    output_dir = 'processed/%s' %dataset
    for center, width in zip(centers, widths):
        output_dir += '_c%dw%d' %(center, width)
    os.makedirs(output_dir, exist_ok=True)

    Parallel(n_jobs=-1, verbose=10)(
        [delayed(process)(img_path, output_dir, centers, widths) for img_path in img_paths])


def main():
    centers = [40, 80, 40]
    widths= [80, 200, 380]

    convert_dcm_to_png('stage_1_train', centers, widths)
    convert_dcm_to_png('stage_1_test', centers, widths)


if __name__ == '__main__':
    main()
