# 开发者：热心网友
# 开发时间：2022/8/29 22:39
# coding:utf-8

import numpy as np
from origin_dataset import Origin_dataset


class Valid_and_Test_dataset(Origin_dataset):
    '''
    Args:
        csv_dir: train, valid, test所在的path
        data_dir: 数据所在dir
        mode: test 或者 valid
    '''
    def __init__(self, csv_path, data_dir, mode):
        super(Valid_and_Test_dataset, self).__init__(csv_path, data_dir)
        self.get_imgnames_and_labels(csv_path, mode)


    def __getitem__(self, index):
        assert self.imgnames
        imgpath = self.data_dir + self.imgnames + '.npy'
        label = self.labels[index]
        img = np.load(imgpath)
        img = np.expand_dims(img, axis=0)
        return img, label