from __future__ import print_function, division
import os
import torch
import pandas as pd
import time
import sys
import io
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()  # interactive mode

######### TEST ENVIORMENT ###############

# exit()
#########################################


frame = pd.DataFrame(columns=[0,1])

with os.scandir('data-example/') as entries:
    for entry in entries:
        if str(entry.name) == '.DS_Store':
            continue
        label = str(entry.name)
        basepath = 'data-example/' + label
        print(basepath)
        for image_name in os.listdir(basepath):
            if os.path.isfile(os.path.join(basepath, image_name)):
                temp_frame = {0: image_name, 1: label}
                frame = frame.append(temp_frame, ignore_index=True)

print(frame)

n = len(frame)
img_name = frame.iloc[0, 0]
label = frame.iloc[0, 1]

print('Image name: {}'.format(img_name))
print('Image Label: {}'.format(label))


class CustomDataset(Dataset):
    """Custom dataset."""

    def __init__(self, DataFrame, root_dir, transform=None):
        """
        Args:
            DataFrame (string): Csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.frame = DataFrame
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 1])
        img_name = os.path.join(img_name, self.frame.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.frame.iloc[idx, 1]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample



def __main__(frame, root_dir):
    dataset = CustomDataset(DataFrame=frame, root_dir='data-example/')

__main__(frame, 'data-example/')






### Printing Bullshit aka. testing
"""
fig = plt.figure()

for i in range(len(frame)):
    sample = dataset[i]

    print(i, sample['image'], sample['label'])
"""
