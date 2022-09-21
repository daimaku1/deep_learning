# 开发者：热心网友
# 开发时间：2022/8/29 21:17
# coding:utf-8


import os
import csv

import numpy as np
import torch
import torch.utils.data.dataset
import torch.utils.data.dataloader


class Origin_dataset(torch.utils.data.Dataset):
    '''
    Args:
        csv_path: train, valid, test所在的path
        data_dir: 数据所在dir
    '''
    def __init__(self, csv_path, data_dir):
        super(Origin_dataset, self).__init__()
        self.imgnames = None
        self.labels = None
        self.data_dir = data_dir

    def get_imgnames_and_labels(self, csv_path, filename):
        imgnames, labels = [], []
        path = os.path.join(csv_path, filename + '.csv')
        with open(path, 'r') as f:
            reader = csv.reader(f)
            your_list = list(reader)
        for line in your_list[1:]:
            imgnames.append(line[0])
            labels.append(0 if line[1] == 'NL' else 1)
        self.imgnames = imgnames
        self.labels = labels


    def __getitem__(self, index):
        assert self.imgnames
        imgpath = self.data_dir + self.imgnames + '.npy'
        label = self.labels[index]
        img = np.load(imgpath)
        return img, label