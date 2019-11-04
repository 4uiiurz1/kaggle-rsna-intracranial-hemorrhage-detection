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
import cv2
# cv2.setNumThreads(0)
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from skimage.io import imread

from apex import amp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.backends.cudnn as cudnn
import torchvision

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations.pytorch.transforms import ToTensor
from albumentations.core.transforms_interface import NoOp

from lib.dataset import DatasetV2
from lib.models.model_factory import get_model
from lib.utils import *
from lib.metrics import *
from lib.losses import *
from lib.optimizers import *
from lib.preprocess import resize
from lib.augmentations import *


class HeadCNN(nn.Module):
    def __init__(self, model, batch_size=1):
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

        self.batch_size = 1

    def forward(self, inputs):
        # Convolution layers
        x = self.model(inputs) # b, c

        b, c = x.size()
        x = x.view(self.batch_size, -1, c, 1) # b, w, c, 1
        x = x.permute(0, 2, 1, 3) # b, c, w, 1

        x = self.conv(x)

        x = x.permute(0, 2, 1, 3) # b, w, c, 1
        x = x.view(b, 6)

        return x


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--base_name', default=None)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='efficientnet-b0',
                        help='model architecture: ' +
                        ' (default: efficientnet-b0)')
    parser.add_argument('--freeze_bn', default=False, type=str2bool)
    parser.add_argument('--dropout_p', default=0, type=float)
    parser.add_argument('--pooling', default='avg')
    parser.add_argument('--lp_p', default=2, type=int)
    parser.add_argument('--loss', default='WeightedBCEWithLogitsLoss')
    parser.add_argument('--label_smooth', default=0, type=float)
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('--accum_steps', default=1, type=int,
                        help='accumlation steps')
    parser.add_argument('--depth_size', default=8, type=int)
    parser.add_argument('--img_size', default=512, type=int,
                        help='input image size (default: 320)')
    parser.add_argument('--crop_size', default=410, type=int)
    parser.add_argument('--optimizer', default='RAdam')
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'MultiStepLR'])
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--pred_type', default='all',
                        choices=['all', 'except_any'])

    # data augmentation
    parser.add_argument('--center_crop', default=False, type=str2bool)
    parser.add_argument('--foreground_center_crop', default=False, type=str2bool)
    parser.add_argument('--random_crop', default=True, type=str2bool)
    parser.add_argument('--hflip', default=True, type=str2bool)
    parser.add_argument('--vflip', default=False, type=str2bool)
    parser.add_argument('--shift_scale_rotate', default=True, type=str2bool)
    parser.add_argument('--shift_scale_rotate_p', default=0.5, type=float)
    parser.add_argument('--shift_limit', default=0.0625, type=float)
    parser.add_argument('--scale_limit', default=0.1, type=float)
    parser.add_argument('--rotate_limit', default=0, type=int)
    parser.add_argument('--contrast', default=False, type=str2bool)
    parser.add_argument('--contrast_p', default=0.5, type=float)
    parser.add_argument('--contrast_limit', default=0.2, type=float)
    parser.add_argument('--random_erase', default=False, type=str2bool)

    # dataset
    parser.add_argument('--img_type', default='c40w80_c80w200_c40w380')
    parser.add_argument('--remove_test_patient_ids', default=False, type=str2bool)

    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--apex', action='store_true')

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    return args


def train(args, train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()

    model.train()

    pbar = tqdm(total=len(train_loader) // args.accum_steps)
    for i, (input, target) in enumerate(train_loader):
        input = input.view(32, 3, args.crop_size, args.crop_size)
        target = target.view(32, 6)

        input = input.cuda()
        target = target.cuda()

        output = model(input)
        if args.pred_type == 'all':
            loss = criterion(output, target)
        elif args.pred_type == 'except_any':
            loss = criterion(output, target[:, :-1])

        loss /= args.accum_steps

        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if (i + 1) % args.accum_steps == 0:
            # compute gradient and do optimizing step
            optimizer.step()
            optimizer.zero_grad()

            losses.update(loss.item() * args.accum_steps, input.size(0))

            pbar.set_description('loss %.4f' %losses.avg)
            pbar.update(1)

    return losses.avg


def main():
    args = parse_args()

    if args.name is None:
        args.name = '%s_%s' % (args.arch, datetime.now().strftime('%m%d%H'))

    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)

    if args.resume:
        args = joblib.load('models/%s/args.pkl' % args.name)
        args.resume = True

    print('Config -----')
    for arg in vars(args):
        print('- %s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('- %s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)

    if args.seed is not None and not args.resume:
        print('set random seed')
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.loss == 'BCEWithLogitsLoss':
        criterion = BCEWithLogitsLoss().cuda()
    elif args.loss == 'WeightedBCEWithLogitsLoss':
        criterion = BCEWithLogitsLoss(weight=torch.Tensor([1., 1., 1., 1., 1., 2.]),
                                      smooth=args.label_smooth).cuda()
    elif args.loss == 'FocalLoss':
        criterion = FocalLoss().cuda()
    elif args.loss == 'WeightedFocalLoss':
        criterion = FocalLoss(weight=torch.Tensor([1., 1., 1., 1., 1., 2.])).cuda()
    else:
        raise NotImplementedError

    if args.pred_type == 'all':
        num_outputs = 6
    elif args.pred_type == 'except_any':
        num_outputs = 5
    else:
        raise NotImplementedError

    cudnn.benchmark = True

    train_transform = Compose([
        transforms.Resize(args.img_size, args.img_size),
        transforms.HorizontalFlip() if args.hflip else NoOp(),
        transforms.VerticalFlip() if args.vflip else NoOp(),
        transforms.ShiftScaleRotate(
            shift_limit=args.shift_limit,
            scale_limit=args.scale_limit,
            rotate_limit=args.rotate_limit,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=args.shift_scale_rotate_p
        ) if args.shift_scale_rotate else NoOp(),
        transforms.RandomContrast(
            limit=args.contrast_limit,
            p=args.contrast_p
        ) if args.contrast else NoOp(),
        RandomErase() if args.random_erase else NoOp(),
        transforms.CenterCrop(args.crop_size, args.crop_size) if args.center_crop else NoOp(),
        ForegroundCenterCrop(args.crop_size) if args.foreground_center_crop else NoOp(),
        transforms.RandomCrop(args.crop_size, args.crop_size) if args.random_crop else NoOp(),
        transforms.Normalize(),
        ToTensor(),
    ])

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
    df['Axial'] = df['ImagePositionPatient'].apply(lambda s: float(s.split('\'')[-2]))

    patient_ids = meta_df['PatientID'].unique()
    test_patient_ids = test_meta_df['PatientID'].unique()
    if args.remove_test_patient_ids:
        patient_ids = np.array([s for s in patient_ids if not s in test_patient_ids])

    if args.resume:
        checkpoint = torch.load('models/%s/checkpoint.pth.tar' % args.name)

    # train
    train_set = DatasetV2(
        df,
        depth_size=args.depth_size,
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # pin_memory=True,
        drop_last=True,
    )

    # create model
    model = get_model(model_name=args.arch,
                      num_outputs=num_outputs,
                      freeze_bn=args.freeze_bn,
                      dropout_p=args.dropout_p,
                      pooling=args.pooling,
                      lp_p=args.lp_p)
    if args.base_name is not None:
        model.load_state_dict(torch.load(os.path.join('models', args.base_name, 'model.pth')))
    model = HeadCNN(model, batch_size=args.batch_size)
    model = model.cuda()
    # print(model)

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RAdam':
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    else:
        raise NotImplementedError

    if args.apex:
        amp.initialize(model, optimizer, opt_level='O1')

    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer,
            milestones=[int(e) for e in args.milestones.split(',')], gamma=args.gamma)
    else:
        raise NotImplementedError

    log = {
        'epoch': [],
        'loss': [],
    }

    start_epoch = 0

    if args.resume:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        log = pd.read_csv('models/%s/log.csv' % args.name).to_dict(orient='list')

    for epoch in range(start_epoch, args.epochs):
        print('Epoch [%d/%d]' % (epoch + 1, args.epochs))

        # train for one epoch
        train_loss = train(args, train_loader, model, criterion, optimizer, epoch)

        if args.scheduler == 'CosineAnnealingLR':
            scheduler.step()

        print('loss %.4f' % (train_loss))

        log['epoch'].append(epoch)
        log['loss'].append(train_loss)

        pd.DataFrame(log).to_csv('models/%s/log.csv' % args.name, index=False)

        torch.save(model.state_dict(), 'models/%s/model.pth' % args.name)
        print("=> saved model")

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(state, 'models/%s/checkpoint.pth.tar' % args.name)


if __name__ == '__main__':
    main()
