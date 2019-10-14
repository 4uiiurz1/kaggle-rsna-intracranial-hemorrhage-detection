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
    parser.add_argument('--tta', default=False, type=str2bool)


    args = parser.parse_args()

    return args


def apply_tta(input):
    inputs = []
    inputs.append(input)
    inputs.append(torch.flip(input, dims=[2]))
    inputs.append(torch.flip(input, dims=[3]))
    inputs.append(torch.rot90(input, k=1, dims=[2, 3]))
    inputs.append(torch.rot90(input, k=2, dims=[2, 3]))
    inputs.append(torch.rot90(input, k=3, dims=[2, 3]))
    inputs.append(torch.rot90(torch.flip(input, dims=[2]), k=1, dims=[2, 3]))
    inputs.append(torch.rot90(torch.flip(input, dims=[2]), k=3, dims=[2, 3]))
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

    test_transform = Compose([
        transforms.Resize(args.img_size, args.img_size),
        ForegroundCenterCrop(args.crop_size),
        transforms.Normalize(),
        ToTensor(),
    ])

    # data loading code
    if args.img_type:
        stage_1_test_dir = 'processed/stage_1_test_%s' %args.img_type
    else:
        stage_1_test_dir = 'processed/stage_1_test'

    test_df = pd.read_csv('inputs/stage_1_sample_submission.csv')
    test_img_paths = np.array([stage_1_test_dir + '/' + '_'.join(s.split('_')[:-1]) + '.png' for s in test_df['ID']][::6])
    test_labels = np.array([test_df.loc[c::6, 'Label'].values for c in range(6)]).T.astype('float32')

    test_set = Dataset(
        test_img_paths,
        test_labels,
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    preds = []
    for fold in range(args.n_splits):
        print('Fold [%d/%d]' %(fold+1, args.n_splits))

        # create model
        model_path = 'models/%s/model_%d.pth' % (args.name, fold+1)
        if not os.path.exists(model_path):
            print('%s is not exists.' %model_path)
            continue
        model = get_model(model_name=args.arch,
                          num_outputs=num_outputs,
                          freeze_bn=args.freeze_bn,
                          dropout_p=args.dropout_p,
                          pooling=args.pooling,
                          lp_p=args.lp_p)
        model = model.cuda()
        model.load_state_dict(torch.load(model_path))

        model.eval()

        preds_fold = []
        with torch.no_grad():
            for i, (input, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
                input = input.cuda()
                output = model(input)
                output = torch.sigmoid(output)
                if args.pred_type == 'except_any':
                    output = torch.cat([output, torch.max(output, 1, keepdim=True)[0]], 1)
                preds_fold.append(output.data.cpu().numpy())
        preds_fold = np.vstack(preds_fold)
        preds.append(preds_fold)

        if not args.cv:
            break

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

    test_df[['ID', 'Label']].to_csv('submissions/%s.csv' %args.name, index=False)


if __name__ == '__main__':
    main()
