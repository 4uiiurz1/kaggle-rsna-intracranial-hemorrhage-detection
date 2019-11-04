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

from sklearn.model_selection import StratifiedKFold, train_test_split
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


class HeadCNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # for p in self.model.parameters():
        #     p.requires_grad = False
        print(self.model)
        self.model._fc = nn.Sequential(*[
            nn.Linear(1408, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        ])

        self.conv = nn.Sequential(*[
            nn.Conv2d(512, 6, (5, 1), padding=(2, 0)),
        ])

    def forward_1(self, inputs):
        # Convolution layers
        x = self.model(inputs) # b, c

        return x

    def forward_2(self, inputs):
        b, c = x.size()
        x = x.view(1, -1, c, 1) # b, w, c, 1
        x = x.permute(0, 2, 1, 3) # b, c, w, 1

        x = self.conv(x)

        x = x.permute(0, 2, 1, 3) # b, w, c, 1
        x = x.view(b, 6)

        return x


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

    cudnn.benchmark = True

    test_transform = Compose([
        transforms.Resize(args.img_size, args.img_size),
        ForegroundCenterCrop(args.crop_size),
        transforms.Normalize(mean=model.mean, std=model.std),
        ToTensor(),
    ])

    # data loading code
    if args.img_type:
        stage_1_test_dir = 'processed/stage_1_test_%s' %args.img_type
    else:
        stage_1_test_dir = 'processed/stage_1_test'

    test_meta_df = pd.read_csv('processed/stage_1_test_meta.csv')
    test_meta_df['Axial'] = test_meta_df['ImagePositionPatient'].apply(lambda s: float(s.split('\'')[-2]))

    test_df = pd.read_csv('submissions/%s.csv' %test_args.name)
    ids = np.array(['_'.join(s.split('_')[:-1]) for s in test_df['ID']][:len(test_df) // 6])
    labels = np.array([test_df.loc[len(test_df) // 6 * c:len(test_df) // 6 * (c + 1) - 1, 'Label'].values for c in range(6)]).T.astype('float32')

    study_ids = test_meta_df['StudyInstanceUID'].unique()

    test_set = Dataset(
        test_img_paths,
        test_labels,
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    for study_id in tqdm(study_ids, total=len(study_ids)):
        df_ = test_meta_df[test_meta_df['StudyInstanceUID'] == study_id].sort_values('Axial')
        labels_ = np.vstack([labels[ids==s] for s in df_['SOPInstanceUID']])

        

    preds = []
    preds_fold = []
    with torch.no_grad():
        for i, (input, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
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

    preds = np.mean(preds, axis=0)

    test_df = pd.DataFrame(preds, columns=[
        'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'
    ])
    test_df['ID'] = test_img_paths

    # Unpivot table, i.e. wide (N x 6) to long format (6N x 1)
    test_df = test_df.melt(id_vars=['ID'])

    # Combine the filename column with the variable column
    test_df['ID'] = test_df.ID.apply(lambda x: os.path.basename(x).replace('.png', '')) + '_' + test_df.variable
    test_df['Label'] = test_df['value']

    if test_args.hflip:
        args.name += '_hflip'
    test_df[['ID', 'Label']].to_csv('submissions/%s.csv' %args.name, index=False)


if __name__ == '__main__':
    main()
