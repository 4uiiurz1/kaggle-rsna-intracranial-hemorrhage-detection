import os
import gc
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from joblib import Parallel, delayed


def concat_3n_images(dataset):
    def process(df_, study_id, indices, output_dir):
        df = df_[indices == study_id].sort_values('Axial').reset_index(drop=True)
        img1 = None
        img2 = None
        img3 = None
        for i in range(len(df)):
            dst_path = os.path.join(output_dir, os.path.basename(df.loc[i, 'img_path']))
            if os.path.exists(dst_path):
                return

            if img2 is not None:
                img1 = img2
            else:
                img1 = cv2.imread(df.loc[max(0, i - 1), 'img_path'], cv2.IMREAD_GRAYSCALE)
                if img1 is None:
                    return

            if img3 is not None:
                img2 = img3
            else:
                img2 = cv2.imread(df.loc[i, 'img_path'], cv2.IMREAD_GRAYSCALE)
                if img2 is None:
                    return

            img3 = cv2.imread(df.loc[min(len(df) - 1, i + 1), 'img_path'], cv2.IMREAD_GRAYSCALE)
            if img3 is None:
                return

            img = np.dstack([img1, img2, img3])
            try:
                cv2.imwrite(dst_path, img)
            except:
                return

    if dataset == 'stage_1_test':
        df = pd.read_csv('inputs/stage_1_sample_submission.csv')
    else:
        df = pd.read_csv('inputs/%s.csv' % dataset)

    img_paths = np.array(['processed/%s_512/' % dataset +
                          '_'.join(s.split('_')[:-1]) + '.png' for s in df['ID']][::6])
    labels = np.array(
        [df.loc[c::6, 'Label'].values for c in range(6)]).T.astype('float32')
    df = df[::6]
    df['img_path'] = img_paths
    for c in range(6):
        df['label_%d' % c] = labels[:, c]
    df['ID'] = df['ID'].apply(lambda s: '_'.join(s.split('_')[:-1]))

    meta_df = pd.read_csv('processed/%s_meta.csv' % dataset)
    meta_df['Axial'] = meta_df['ImagePositionPatient'].apply(
        lambda s: float(s.split('\'')[-2]))
    meta_df['ID'] = meta_df['SOPInstanceUID']
    df = pd.merge(df, meta_df, how='left')

    output_dir = 'processed/%s_concat_k3' % dataset
    os.makedirs(output_dir, exist_ok=True)

    indices = df['StudyInstanceUID'].values
    study_ids = df['StudyInstanceUID'].unique()
    output = Parallel(n_jobs=-1, verbose=10)([delayed(process)(df, study_id, indices, output_dir) for study_id in study_ids])


def main():
    # concat_3n_images('stage_1_train')
    concat_3n_images('stage_1_test')


if __name__ == '__main__':
    main()
