import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
plt.rcParams['image.cmap'] = 'gray'

directory = "NYU_dataset_fMRI"
filenames = []
for root, dire, files in os.walk(directory):
    for f in files:
        filenames.append(os.path.join(directory, f))
common_mask = np.zeros([61, 73, 61])
for i, f in enumerate(filenames):
    if "NYU" not in f:
        continue
    data = nib.load(f).get_fdata()
    mask = np.ones_like(data)
    mask[data == 0] = 0
    mask = np.sum(mask, axis=-1)
    common_mask = common_mask + mask
    common_mask = common_mask > 0
    print(i)
    if i > 5:
        break
common_mask = common_mask.astype('int')
common_mask = np.expand_dims(common_mask, -1)
np.save("NYU_dataset_fMRI/common_mask.npy", common_mask)
