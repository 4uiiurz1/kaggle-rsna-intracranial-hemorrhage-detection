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


def main():
    sub1 = pd.read_csv('submissions/efficientnet-b1_102407_hflip.csv')
    sub2 = pd.read_csv('submissions/efficientnet-b2_102000_hflip.csv')
    sub3 = pd.read_csv('submissions/efficientnet-b3_102112_hflip.csv')
    sub4 = pd.read_csv('submissions/efficientnet-b4_102100_hflip.csv')
    sub5 = pd.read_csv('submissions/se_resnext50_32x4d_102105_hflip.csv')

    sub1['Label'] = (sub1['Label'] + sub2['Label'] + sub3['Label'] + sub4['Label'] + sub5['Label']) / 5

    print(sub1.head())

    sub1[['ID', 'Label']].to_csv('submissions/average_102800.csv', index=False)


if __name__ == '__main__':
    main()
