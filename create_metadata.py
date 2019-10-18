import os
import gc
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


meta_cols = ['BitsAllocated','BitsStored','Columns','HighBit',
             'Modality','PatientID','PhotometricInterpretation',
             'PixelRepresentation','RescaleIntercept','RescaleSlope',
             'Rows','SOPInstanceUID','SamplesPerPixel','SeriesInstanceUID',
             'StudyID','StudyInstanceUID','ImagePositionPatient',
             'ImageOrientationPatient','PixelSpacing']


def convert_dcm_to_df(dataset):
    def process(img_path):
        dcm = pydicom.dcmread(img_path)
        return [str(getattr(dcm, c)) for c in meta_cols]

    if dataset == 'stage_1_test':
        df = pd.read_csv('inputs/stage_1_sample_submission.csv')
    else:
        df = pd.read_csv('inputs/%s.csv' %dataset)
    img_paths = np.array([os.path.join('inputs/%s_images/' %dataset, '_'.join(s.split('_')[:-1]) + '.dcm') for s in df['ID']][::6])
    img_paths = np.unique(img_paths)

    output = Parallel(n_jobs=-1, verbose=10)([delayed(process)(img_path) for img_path in img_paths])
    output = np.array(output)

    df = pd.DataFrame({c: output[:, i] for i, c in enumerate(meta_cols)})

    df.to_csv('processed/%s_meta.csv' %dataset, index=False)


def main():
    convert_dcm_to_df('stage_1_train')
    convert_dcm_to_df('stage_1_test')


if __name__ == '__main__':
    main()
