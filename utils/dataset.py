import os
import h5py
import random
import torch
import torch.nn as nn
import numpy as np
import tifffile as tiff

from glob import glob
from tqdm import trange, tqdm

def prepare_data(data_dir, out_file):
    input_dir, label_dir = os.path.join(data_dir, 'input'), os.path.join(data_dir, 'label')
    input_data, label_data = [], []

    # retrieve all the input and label images
    all_input = glob(os.path.join(input_dir, '*.tiff'))
    all_label = glob(os.path.join(label_dir, '*.tiff'))
    assert len(all_input) == len(all_label)

    # iterate all the image
    for i in trange(len(all_input)):
        # prepare input data
        input_data.append(np.expand_dims(np.array(tiff.imread(all_input[i])).astype(np.uint8), axis=0))

        # convert raw label image to labels
        label = tiff.imread(all_label[i])
        label[np.where(label == 255)] = 1
        label_data.append(np.array(label).astype(np.uint8))

    # write input_data and label_data to h5py file
    h5 = h5py.File(out_file, 'w')
    h5.create_dataset('input', data=input_data)
    h5.create_dataset('label', data=label_data)
    h5.close()

def split_dataset(h5, train_num, dev_num):
    train_input, train_label = [], []
    dev_input, dev_label = [], []

    train_idx = random.sample(range(train_num + dev_num), train_num)
    dev_idx = list(filter(lambda x: x not in train_idx, range(train_num + dev_num)))

    for i in tqdm(train_idx):
        train_input.append(np.array(h5['input'][i]).astype(np.uint8))
        train_label.append(np.array(h5['label'][i]).astype(np.uint8))
    for i in tqdm(dev_idx):
        dev_input.append(np.array(h5['input'][i]).astype(np.uint8))
        dev_label.append(np.array(h5['label'][i]).astype(np.uint8))

    return (train_input, train_label), (dev_input, dev_label)

class LunarDataset(nn.Module):
    def __init__(self, h5_file, transform=None, augmentation=None):
        super().__init__()
        self.h5 = h5py.File(h5_file, 'r')
        self.transform = transform
        self.augmentation = augmentation

    def __getitem__(self, i):
        input = self.h5['input'][i]
        label = self.h5['label'][i]

        if self.transform:
            input = self.transform(torch.FloatTensor(input))
        if self.augmentation != None:
            input = self.augmentation(images=input.numpy())

        input, label = torch.FloatTensor(input), torch.LongTensor(label)

        return (input, label)

    def __len__(self):
        return self.h5['input'].shape[0]
