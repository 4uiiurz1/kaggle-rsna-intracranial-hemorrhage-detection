import os
from glob import glob
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import pydicom
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def window_image(img, window_center, window_width, intercept, slope, rescale=True):
    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img<img_min] = img_min
    img[img>img_max] = img_max

    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)

    return img


def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    dicom_fields = [
        data[('0028','1050')].value, #window center
        data[('0028','1051')].value, #window width
        data[('0028','1052')].value, #intercept
        data[('0028','1053')].value, #slope
    ]
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def resize(dataset, img_size):
    def process(img_path, output_dir, img_size):
        dst_path = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + '.png')
        if os.path.exists(dst_path):
            return
        try:
            img = cv2.imread(img_path)
        except:
            return
        try:
            img = cv2.resize(img, (img_size, img_size))
        except:
            return
        cv2.imwrite(dst_path, img)

    if dataset == 'stage_1_test':
        df = pd.read_csv('inputs/stage_1_sample_submission.csv')
    else:
        df = pd.read_csv('inputs/%s.csv' %dataset)

    img_paths = np.array(['processed/%s/' %dataset + '_'.join(s.split('_')[:-1]) + '.png' for s in df['ID']][::6])
    img_paths = np.unique(img_paths)

    output_dir = 'processed/%s_%d' %(dataset, img_size)
    if os.path.exists(output_dir):
        return output_dir
    os.makedirs(output_dir, exist_ok=True)

    Parallel(n_jobs=-1, verbose=10)(
        [delayed(process)(img_path, output_dir, img_size) for img_path in img_paths])

    return output_dir


def convert_dcm_to_png(dataset):
    def process(img_path, output_dir):
        dst_path = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + '.png')
        if os.path.exists(dst_path):
            return
        dcm = pydicom.dcmread(img_path)
        window_center , window_width, intercept, slope = get_windowing(dcm)
        try:
            img = dcm.pixel_array
        except:
            return
        img = window_image(img, window_center, window_width, intercept, slope)
        img = (img * 255).astype('uint8')
        cv2.imwrite(dst_path, img)

    if dataset == 'stage_1_test':
        df = pd.read_csv('inputs/stage_1_sample_submission.csv')
    else:
        df = pd.read_csv('inputs/%s.csv' %dataset)
    img_paths = np.array([os.path.join('inputs/%s_images/' %dataset, '_'.join(s.split('_')[:-1]) + '.dcm') for s in df['ID']][::6])
    img_paths = np.unique(img_paths)

    output_dir = 'processed/%s' %dataset
    os.makedirs(output_dir, exist_ok=True)

    Parallel(n_jobs=-1, verbose=10)(
        [delayed(process)(img_path, output_dir) for img_path in img_paths])


def main():
    # convert_dcm_to_png('stage_1_train')
    convert_dcm_to_png('stage_1_test')


if __name__ == '__main__':
    main()
