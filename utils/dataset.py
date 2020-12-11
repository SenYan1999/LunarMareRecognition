import os
import torch
import torch.nn as nn
import numpy as np
import tifffile as tiff

from glob import glob

class LunarDataset(nn.Module):
    def __init__(self, data_dir):
        self.input_data, self.label_data = self.prepare_data(data_dir)

    def prepare_data(self, data_dir):
        input_dir, label_dir = os.path.join(data_dir, 'input'), os.path.join(data_dir, 'label')
        input_data, label_data = [], []

        # retrieve all the input and label images
        all_input = glob(input_dir + '*.tiff')
        all_label = glob(input_dir + '*.tiff')
        assert len(all_input) == len(all_label)

        # iterate all the image
        for i in range(len(all_input)):
            # prepare input data
            input_data.append(torch.FloatTensor(tiff.imread(all_input[i])).unsqueeze(dim=0))

            # convert raw label image to labels
            label = tiff.imread(all_label[i])
            label[np.where(label == 255)] = 1
            label_data.append(torch.FloatTensor(label).unsqueeze(dim=0))

        # convert list to tensor
        input_data = torch.cat(input_data).unsqueeze(dim=1)
        label_data = torch.cat(input_data).unsqueeze(dim=1)

        return input_data, label_data

    def __getitem__(self, i):
        return self.input_data[i], self.label_data[i]

    def __len__(self):
        return self.input_data.shape[0]
