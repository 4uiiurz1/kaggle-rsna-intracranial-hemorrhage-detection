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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None)
    parser.add_argument('--k', default='19,11,9,17,19,13')
    parser.add_argument('--sigma', default='2.90,0.60,0.55,1.10,1.50,0.70')

    args = parser.parse_args()

    return args


def main():
    test_args = parse_args()

    test_meta_df = pd.read_csv('processed/stage_1_test_meta.csv')
    # test_meta_df = pd.read_csv('processed/stage_2_test_meta.csv')
    test_meta_df['Axial'] = test_meta_df['ImagePositionPatient'].apply(lambda s: float(s.split('\'')[-2]))

    test_df = pd.read_csv('submissions/%s.csv' %test_args.name)
    ids = np.array(['_'.join(s.split('_')[:-1]) for s in test_df['ID']][:len(test_df) // 6])
    labels = np.array([test_df.loc[len(test_df) // 6 * c:len(test_df) // 6 * (c + 1) - 1, 'Label'].values for c in range(6)]).T.astype('float32')
    new_labels = labels.copy()

    study_ids = test_meta_df['StudyInstanceUID'].unique()

    ks = [int(e) for e in test_args.k.split(',')]
    sigmas = [float(e) for e in test_args.sigma.split(',')]
    ws = []
    for k, s in zip(ks, sigmas):
        w = np.exp(-(np.arange(-k // 2 + 1, k // 2 + 1))**2 / (2 * s**2))
        w /= np.sum(w)
        print(w)
        ws.append(w)

    for study_id in tqdm(study_ids, total=len(study_ids)):
        df_ = test_meta_df[test_meta_df['StudyInstanceUID'] == study_id].sort_values('Axial')
        labels_ = np.vstack([labels[ids==s] for s in df_['SOPInstanceUID']])
        # plt.imshow(labels_, vmin=0, vmax=1)
        # plt.show()
        for idx, s in enumerate(df_['SOPInstanceUID']):
            for c, k in enumerate(ks):
                x = labels_[max(0, idx - k // 2):min(len(labels_), idx + k // 2 + 1), c]
                x = np.pad(x, ((np.abs(min(0, idx - k // 2)), max(len(labels_) - 1, idx + k // 2) - len(labels_) + 1)), mode='edge')
                new_labels[ids == s, c] = np.average(x, weights=ws[c])
        # plt.imshow(np.vstack([new_labels[ids==s] for s in df_['SOPInstanceUID']]), vmin=0, vmax=1)
        # plt.show()

    test_df = pd.DataFrame(new_labels, columns=[
        'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'
    ])
    test_df['ID'] = ids

    # Unpivot table, i.e. wide (N x 6) to long format (6N x 1)
    test_df = test_df.melt(id_vars=['ID'])

    # Combine the filename column with the variable column
    test_df['ID'] = test_df.ID + '_' + test_df.variable
    test_df['Label'] = test_df['value']

    test_df[['ID', 'Label']].to_csv(
        'submissions/%s_gma_%s.csv' %(test_args.name,
        '_'.join(['k%ds%.2f' %(k, s) for k, s in zip(ks, sigmas)])), index=False)


if __name__ == '__main__':
    main()
