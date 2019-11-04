import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import joblib

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations.pytorch.transforms import ToTensor
from albumentations.core.transforms_interface import NoOp

from lib.dataset import Dataset
from lib.models.model_factory import get_model
from lib.utils import *
from lib.metrics import *
from lib.losses import *
from lib.preprocess import resize
from lib.augmentations import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--hflip', default=False, type=str2bool)

    args = parser.parse_args()

    return args


def apply_tta(args, input):
    inputs = []
    inputs.append(input)
    if args.hflip:
        inputs.append(torch.flip(input, dims=[3]))
    return inputs


def main():
    test_args = parse_args()

    args = joblib.load('models/%s/args.pkl' %test_args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    if args.pred_type == 'all':
        num_outputs = 6
    elif args.pred_type == 'except_any':
        num_outputs = 5
    else:
        raise NotImplementedError

    cudnn.benchmark = True

    train_transform = Compose([
        transforms.Resize(args.img_size, args.img_size),
        ForegroundCenterCrop(args.crop_size),
        transforms.Normalize(),
        ToTensor(),
    ])

    # data loading code
    if args.img_type:
        stage_1_train_dir = 'processed/stage_1_train_%s' %args.img_type
    else:
        stage_1_train_dir = 'processed/stage_1_train'

    df = pd.read_csv('inputs/stage_1_train.csv')
    img_paths = np.array([stage_1_train_dir + '/' + '_'.join(s.split('_')[:-1]) + '.png' for s in df['ID']][::6])
    labels = np.array([df.loc[c::6, 'Label'].values for c in range(6)]).T.astype('float32')

    df = df[::6]
    df['img_path'] = img_paths
    for c in range(6):
        df['label_%d' %c] = labels[:, c]
    df['ID'] = df['ID'].apply(lambda s: '_'.join(s.split('_')[:-1]))

    meta_df = pd.read_csv('processed/stage_1_train_meta.csv')
    meta_df['ID'] = meta_df['SOPInstanceUID']
    test_meta_df = pd.read_csv('processed/stage_1_test_meta.csv')
    df = pd.merge(df, meta_df, how='left')

    patient_ids = meta_df['PatientID'].unique()
    test_patiend_ids = test_meta_df['PatientID'].unique()
    patient_ids = np.array([s for s in patient_ids if not s in test_patiend_ids])

    train_img_paths = np.hstack(df[['img_path', 'PatientID']].groupby(['PatientID'])['img_path'].apply(np.array).loc[patient_ids].to_list()).astype('str')
    train_labels = []
    for c in range(6):
        train_labels.append(np.hstack(df[['label_%d' %c, 'PatientID']].groupby(['PatientID'])['label_%d' %c].apply(np.array).loc[patient_ids].to_list()))
    train_labels = np.array(train_labels).T

    train_set = Dataset(
        train_img_paths,
        train_labels,
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # pin_memory=True,
    )

    # create model
    model_path = 'models/%s/model.pth' % args.name
    model = get_model(model_name=args.arch,
                      num_outputs=num_outputs,
                      freeze_bn=args.freeze_bn,
                      dropout_p=args.dropout_p,
                      pooling=args.pooling,
                      lp_p=args.lp_p)
    model = model.cuda()
    model.load_state_dict(torch.load(model_path))

    model.eval()

    preds = []
    preds_fold = []
    with torch.no_grad():
        for i, (input, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            outputs = []
            for input in apply_tta(args, input):
                input = input.cuda()
                output = model(input)
                output = torch.sigmoid(output)
                if args.pred_type == 'except_any':
                    output = torch.cat([output, torch.max(output, 1, keepdim=True)[0]], 1)
                outputs.append(output.data.cpu().numpy())
            preds_fold.extend(np.mean(outputs, axis=0))
    preds_fold = np.vstack(preds_fold)
    preds.append(preds_fold)

    outputs = {}
    outputs['img_path'] = train_img_paths
    outputs['label'] = train_labels
    outputs['pred'] = preds

    joblib.dump(outputs, 'preds/%s.pkl' %args.name)


if __name__ == '__main__':
    main()
