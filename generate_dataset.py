import nibabel as nib
import numpy as np
import os
import sys
import pandas as pd
import random
import pickle

from sklearn.model_selection import train_test_split


info = pd.read_csv("Phenotypic_NYU.csv")
filenames = info[info['FILE_ID']!='no_filename']['FILE_ID'].values
print(filenames)
group = np.array(info[info['FILE_ID']!='no_filename']['DX_GROUP'].values)
group = group - 1

num_rows = len(filenames)
print(num_rows)

train_files, test_files, train_groups, test_groups = train_test_split(filenames, group, test_size=0.25, stratify=group)
print(np.unique(train_groups, return_counts=True))
print(np.unique(test_groups, return_counts=True))

split = {}
split["train_files"] = train_files.tolist()
split["test_files"] = test_files.tolist()
split["train_labels"] = train_groups.tolist()
split["test_labels"] = test_groups.tolist()

with open("train_test_split2.json", "wb") as f:
    pickle.dump(split, f)
    print("Dump")


X_train = []
Y_train = []

for i, (f, label) in enumerate(zip(train_files, train_groups)):
    file = "NYU_dataset_fMRI/{}_func_preproc.nii.gz".format(f)
    if not os.path.exists(file):
        print("Not File")
        continue
    
    x = nib.load(file).get_fdata()
    X_train.append(x)
    Y_train.append([label])
    print(i)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

np.save("x_train.npy", X_train)
np.save("y_train.npy", Y_train)

del X_train

with open("train_test_split2.json", "rb") as f:
    split = pickle.load(f)
    print("Loaded")

X_test = []
Y_test = []

for i, (f, label) in enumerate(zip(test_files, test_groups)):
    file = "NYU_dataset_fMRI/{}_func_preproc.nii.gz".format(f)
    if not os.path.exists(file):
        print("Not File")
        continue
    
    x = nib.load(file).get_fdata()
    X_test.append(x)
    Y_test.append([label])
    print(i)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

np.save("x_test.npy", X_test)
np.save("y_test.npy", Y_test)
