# 开发者：热心网友
# 开发时间：2022/8/29 19:45
# coding:utf-8


import torch
import torchvision
from torch import nn
import numpy as np


class FCN_classical(nn.Module):
    '''
    Args:
        neurons: 第一层卷积层的神经元数，后续卷积层的神经元数为此处倍数
        useBias: 是否使用bias
    '''
    def __init__(self, neurons, useBias=False):
        super(FCN_classical, self).__init__()
        # blocks
        # input(train) = (?, 1, 47, 47, 47)
        # input(valid and test) = (?, 1, 227, 263, 227)
        self.conv1 = nn.Conv3d(1, neurons, kernel_size=3, bias=useBias)
        self.conv2 = nn.Conv3d(neurons, 2 * neurons, kernel_size=3, bias=useBias)
        self.conv3 = nn.Conv3d(2 * neurons, 4 * neurons, kernel_size=4, bias=useBias)
        self.conv4 = nn.Conv3d(4 * neurons, 6 * neurons, kernel_size=3, bias=useBias)
        self.conv5 = nn.Conv3d(6 * neurons, 2, kernel_size=3, bias=useBias)
        self.maxPool1 = nn.MaxPool3d(kernel_size=2, stride=1)
        self.maxPool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.5)

        self.fcn = nn.Sequential(
            # (47, )
            self.conv1,
            nn.BatchNorm3d(neurons),
            nn.ReLU(),      # (45, )
            self.maxPool1,
            # (44, )
            self.conv2,
            nn.BatchNorm3d(2 * neurons),
            nn.ReLU(),      # (42, )
            self.maxPool2,
            # (21, )
            self.conv3,
            nn.BatchNorm3d(4 * neurons),
            nn.ReLU(),      # (18, )
            self.maxPool2,
            # (9, )
            self.conv4,
            nn.BatchNorm3d(6 * neurons),
            nn.ReLU(),      # (6, )
            self.maxPool2,
            # (8, )
            self.conv5,
        )

    def forward(self, x):
        out = self.fcn(x)
        return out



if __name__ == '__main__':
    test_data = np.random.uniform(2, size=(1, 1, 47, 47, 47))
    print(test_data)
    test_data = torch.tensor(test_data)
    test_data = test_data.to(torch.float32)
    fcn = FCN_classical(2)
    res = fcn(test_data)
    print(res.view(-1, 2))
