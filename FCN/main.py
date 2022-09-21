# 开发者：热心网友
# 开发时间：2022/8/29 22:55
# coding:utf-8

import json
from train_and_test import train, test_and_generate_DPMs
from net import FCN_classical

'''
params:
    lr: 学习速率
    epochs: 学习期数
    
'''
lr = 0.1
epochs = 3000
exps = 0


def main():
    model = FCN_classical(20)
    train(model=model, lr=lr, epochs=epochs, exps=exps)
    test_and_generate_DPMs()


if __name__ == '__main__':
    main()