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
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from skimage.io import imread

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

from lib.dataset import Dataset
from lib.models.model_factory import get_model
from lib.utils import *
from lib.metrics import *
from lib.losses import *
from lib.optimizers import *
from lib.preprocess import resize
from lib.augmentations import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='efficientnet-b0',
                        help='model architecture: ' +
                        ' (default: resnet34)')
    parser.add_argument('--freeze_bn', default=False, type=str2bool)
    parser.add_argument('--dropout_p', default=0, type=float)
    parser.add_argument('--pooling', default='avg')
    parser.add_argument('--lp_p', default=2, type=int)
    parser.add_argument('--loss', default='WeightedBCEWithLogitsLoss')
    parser.add_argument('--label_smooth', default=0, type=float)
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--train_steps', default=None, type=int)
    parser.add_argument('--val_steps', default=None, type=int)
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--img_size', default=320, type=int,
                        help='input image size (default: 320)')
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--optimizer', default='RAdam')
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
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
    parser.add_argument('--img_type', default='')
    parser.add_argument('--cv', default=False, type=str2bool)
    parser.add_argument('--n_splits', default=5, type=int)

    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    return args


def train(args, train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    # scores = AverageMeter()

    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (input, target) in pbar:

        input = input.cuda()
        target = target.cuda()

        output = model(input)
        if args.pred_type == 'all':
            loss = criterion(output, target)
        elif args.pred_type == 'except_any':
            loss = criterion(output, target[:, :-1])

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if args.pred_type == 'all':
        #     output = torch.sigmoid(output)
        #     score = log_loss(output, target)
        # elif args.pred_type == 'except_any':
        #     output = torch.sigmoid(output)
        #     output = torch.cat([output, torch.max(output, 1, keepdim=True)[0]], 1)
        #     score = log_loss(output, target)

        losses.update(loss.item(), input.size(0))
        # scores.update(score, input.size(0))
        pbar.set_description('loss %.4f' %losses.avg)
        # pbar.set_description('loss %.4f - score %.4f' %(losses.avg, scores.avg))
        pbar.update(1)

    return losses.avg
    # return losses.avg, scores.avg


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    # scores = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, (input, target) in pbar:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            if args.pred_type == 'all':
                loss = criterion(output, target)
            elif args.pred_type == 'except_any':
                loss = criterion(output, target[:, :-1])

            # if args.pred_type == 'all':
            #     output = torch.sigmoid(output)
            #     score = log_loss(output, target)
            # elif args.pred_type == 'except_any':
            #     output = torch.sigmoid(output)
            #     output = torch.cat([output, torch.max(output, 1, keepdim=True)[0]], 1)
            #     score = log_loss(output, target)

            losses.update(loss.item(), input.size(0))
            # scores.update(score, input.size(0))
            pbar.set_description('val_loss %.4f' %losses.avg)
            # pbar.set_description('val_loss %.4f - val_score %.4f' %(losses.avg, scores.avg))
            pbar.update(1)

    return losses.avg
    # return losses.avg, scores.avg


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

    val_transform = Compose([
        transforms.Resize(args.img_size, args.img_size),
        ForegroundCenterCrop(args.crop_size),
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

    patient_ids = meta_df['PatientID'].unique()
    test_patiend_ids = test_meta_df['PatientID'].unique()
    patient_ids = np.array([s for s in patient_ids if not s in test_patiend_ids])

    img_path_sets = []
    label_sets = []
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=41)
    for fold, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        train_patient_ids = patient_ids[train_idx]
        val_patient_ids = patient_ids[val_idx]
        train_img_paths = np.hstack(df[['img_path', 'PatientID']].groupby(['PatientID'])['img_path'].apply(np.array).loc[train_patient_ids].to_list()).astype('str')
        val_img_paths = np.hstack(df[['img_path', 'PatientID']].groupby(['PatientID'])['img_path'].apply(np.array).loc[val_patient_ids].to_list()).astype('str')
        train_labels = []
        for c in range(6):
            train_labels.append(np.hstack(df[['label_%d' %c, 'PatientID']].groupby(['PatientID'])['label_%d' %c].apply(np.array).loc[train_patient_ids].to_list()))
        train_labels = np.array(train_labels).T
        val_labels = []
        for c in range(6):
            val_labels.append(np.hstack(df[['label_%d' %c, 'PatientID']].groupby(['PatientID'])['label_%d' %c].apply(np.array).loc[val_patient_ids].to_list()))
        val_labels = np.array(val_labels).T

        img_path_sets.append((train_img_paths, val_img_paths))
        label_sets.append((train_labels, val_labels))

        if not args.cv:
            break

    if args.resume:
        checkpoint = torch.load('models/%s/checkpoint.pth.tar' % args.name)

    folds = []
    best_losses = []
    # best_scores = []

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=41)
    for fold, ((train_img_paths, val_img_paths), (train_labels, val_labels)) in enumerate(zip(img_path_sets, label_sets)):
        print('Fold [%d/%d]' %(fold+1, args.n_splits))

        if args.resume and fold < checkpoint['fold'] - 1:
            log = pd.read_csv('models/%s/log_%d.csv' %(args.name, fold+1))
            best_loss = log.loc[log['val_loss'].values.argmin(), 'val_loss'].values
            # best_loss, best_score = log.loc[log['val_loss'].values.argmin(), ['val_loss', 'val_score']].values
            folds.append(str(fold + 1))
            best_losses.append(best_loss)
            # best_scores.append(best_score)
            continue

        # train
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

        val_set = Dataset(
            val_img_paths,
            val_labels,
            transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            # pin_memory=True,
        )

        # create model
        model = get_model(model_name=args.arch,
                          num_outputs=num_outputs,
                          freeze_bn=args.freeze_bn,
                          dropout_p=args.dropout_p,
                          pooling=args.pooling,
                          lp_p=args.lp_p)
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

        if args.scheduler == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.min_lr)
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience,
                                                       verbose=1, min_lr=args.min_lr)

        log = {
            'epoch': [],
            'loss': [],
            # 'score': [],
            'val_loss': [],
            # 'val_score': [],
        }

        best_loss = float('inf')
        # best_score = float('inf')

        start_epoch = 0

        if args.resume and fold == checkpoint['fold'] - 1:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            log = pd.read_csv('models/%s/log_%d.csv' % (args.name, fold+1)).to_dict(orient='list')
            best_loss = checkpoint['best_loss']

        for epoch in range(start_epoch, args.epochs):
            print('Epoch [%d/%d]' % (epoch + 1, args.epochs))

            # train for one epoch
            train_loss = train(args, train_loader, model, criterion, optimizer, epoch)
            # train_loss, train_score = train(
            #     args, train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            val_loss = validate(args, val_loader, model, criterion)
            # val_loss, val_score = validate(args, val_loader, model, criterion)

            if args.scheduler == 'CosineAnnealingLR':
                scheduler.step()
            elif args.scheduler == 'ReduceLROnPlateau':
                scheduler.step(val_loss)

            print('loss %.4f - val_loss %.4f' % (train_loss, val_loss))
            # print('loss %.4f - score %.4f - val_loss %.4f - val_score %.4f'
            #       % (train_loss, train_score, val_loss, val_score))

            log['epoch'].append(epoch)
            log['loss'].append(train_loss)
            # log['score'].append(train_score)
            log['val_loss'].append(val_loss)
            # log['val_score'].append(val_score)

            pd.DataFrame(log).to_csv('models/%s/log_%d.csv' % (args.name, fold+1), index=False)

            if val_loss < best_loss:
                torch.save(model.state_dict(), 'models/%s/model_%d.pth' % (args.name, fold+1))
                best_loss = val_loss
                # best_score = val_score
                print("=> saved best model")

            state = {
                'fold': fold + 1,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(state, 'models/%s/checkpoint.pth.tar' % args.name)

        print('val_loss:  %f' % best_loss)
        # print('val_score: %f' % best_score)

        folds.append(str(fold + 1))
        best_losses.append(best_loss)
        # best_scores.append(best_score)

        results = pd.DataFrame({
            'fold': folds + ['mean'],
            'best_loss': best_losses + [np.mean(best_losses)],
            # 'best_score': best_scores + [np.mean(best_scores)],
        })

        print(results)
        results.to_csv('models/%s/results.csv' % args.name, index=False)

        torch.cuda.empty_cache()

        if not args.cv:
            break


if __name__ == '__main__':
    main()
