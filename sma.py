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
    parser.add_argument('--k', default=5, type=int)

    args = parser.parse_args()

    return args


def main():
    test_args = parse_args()

    args = joblib.load('models/%s/args.pkl' %test_args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    test_meta_df = pd.read_csv('processed/stage_1_test_meta.csv')
    test_meta_df['Axial'] = test_meta_df['ImagePositionPatient'].apply(lambda s: float(s.split('\'')[-2]))

    test_df = pd.read_csv('submissions/%s.csv' %test_args.name)
    ids = np.array(['_'.join(s.split('_')[:-1]) for s in test_df['ID']][:len(test_df) // 6])
    labels = np.array([test_df.loc[len(test_df) // 6 * c:len(test_df) // 6 * (c + 1) - 1, 'Label'].values for c in range(6)]).T.astype('float32')
    new_labels = labels.copy()

    study_ids = test_meta_df['StudyInstanceUID'].unique()

    for study_id in tqdm(study_ids, total=len(study_ids)):
        df_ = test_meta_df[test_meta_df['StudyInstanceUID'] == study_id].sort_values('Axial')
        labels_ = np.vstack([labels[ids==s] for s in df_['SOPInstanceUID']])
        # plt.imshow(labels_, vmin=0, vmax=1)
        # plt.show()
        for idx, s in enumerate(df_['SOPInstanceUID']):
            new_labels[ids == s] = labels_[max(0, idx - args.k // 2):min(len(preds_), idx + args.k // 2 + 1)].mean(axis=0)
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

    test_df[['ID', 'Label']].to_csv('submissions/%s_sma_k%d.csv' %(test_args.name, test_args.k), index=False)


if __name__ == '__main__':
    main()
