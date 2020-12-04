import nibabel as nib
import numpy as np
import os
import sys
import pandas as pd
import random
import pickle

with open("train_test_split2.json", 'rb') as f:
    splits = pickle.load(f)
    train_files = splits["train_files"]
    train_labels = splits["train_labels"]
    test_files = splits["test_files"]
    test_labels = splits["test_labels"]

def load_file(filename):
    if not os.path.exists("NYU_dataset_fMRI/{}_func_preproc.nii.gz".format(filename)):
        return None
    return nib.load("NYU_dataset_fMRI/{}_func_preproc.nii.gz".format(filename)).get_fdata()

def preprocess(img):
    random_time = np.random.choice(176 - time_length + 1)
    img = img[:, :, :, random_time:random_time+time_length]
    img = (img - mean_X)/std_X
    img = img * mask
    return img

for f in train_files:
    x = load_file(f)
    if x is None:
        continue
    sum_X = 
