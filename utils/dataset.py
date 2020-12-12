import os
import h5py
import torch
import torch.nn as nn
import numpy as np
import tifffile as tiff

from glob import glob
from tqdm import trange

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

class LunarDataset(nn.Module):
    def __init__(self, h5_file):
        super().__init__()
        self.h5 = h5py.File(h5_file)

    def __getitem__(self, i):
        input = torch.FloatTensor(self.h5['input'][i])
        label = torch.LongTensor(self.h5['label'][i])
        return (input, label)

    def __len__(self):
        return self.h5['input'].shape[0]
