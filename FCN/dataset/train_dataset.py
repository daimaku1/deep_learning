# 开发者：热心网友
# 开发时间：2022/8/29 22:23
# coding:utf-8

import numpy as np
import torch

from origin_dataset import Origin_dataset


class Train_dataset(Origin_dataset):
    '''
    Args:
        csv_dir: train, valid, test所在的path
        data_dir: 数据所在dir
    '''
    def __init__(self, csv_path, data_dir):
        super(Train_dataset, self).__init__(csv_path, data_dir)
        self.get_imgnames_and_labels(csv_path=csv_path, filename='train')
        self.imbalanced_ratio = self.get_imbalanced_ratio()


    def __getitem__(self, index):
        assert self.imgnames
        imgpath = self.data_dir + self.imgnames + '.npy'
        label = self.labels[index]
        img = np.load(imgpath)
        # 找一个合适的起点
        random_point_x = np.random.randint(0, img.shape[0] - 47, size=(1, ), dtype='uint8')
        random_point_y = np.random.randint(0, img.shape[1] - 47, size=(1, ), dtype='uint8')
        random_point_z = np.random.randint(0, img.shape[2] - 47, size=(1, ), dtype='uint8')

        patch = img[
                random_point_x: random_point_x + 47,
                random_point_y: random_point_y + 47,
                random_point_z: random_point_z + 47
                ]
        np.expand_dims(patch, axis=0)
        patch = torch.from_numpy(patch).to(torch.float32)
        return patch, label


    def get_imbalanced_ratio(self):
        count0 = self.labels.count(0)
        count1 = self.labels.count(1)
        return count0 / count1

if __name__ == '__main__':
    t_d = Train_dataset(csv_path='../lookupcsv/exp0', data_dir="/data/datasets/ADNI_NoBack/")
    print(t_d.get_imbalanced_ratio())