# 开发者：热心网友
# 开发时间：2022/8/29 19:44
# coding:utf-8

import sys
sys.path.append('./dataset')
import os.path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import json
from dataset.train_dataset import Train_dataset
from dataset.valid_and_test_dataset import Valid_and_Test_dataset
from net import FCN_classical
from utils import get_confusion_matrix, DPM_statistics


csv_dir = './lookupcsv/'  # 读取csv的dir
checkpoint_dir = './history_model/'     # 保存model的dir
dpms_dir = './DPMS/'


with open('cnofig.json', 'r') as f:
    config = json.loads(f.read())


def train(model: FCN_classical, lr, epochs, exps):
    # 最优模型和最优输出
    '''
    best_model = None
    best_matrix = None  # [[tn, fn], [fp, tp]]
    best_ACC = None
    best_epoch = None
    '''
    # 开始训练
    print('STARTing...')
    for exp in range(exps):
        # 加载数据...
        train_dataset = Train_dataset(csv_dir + f'exp{exp}/', data_dir=config['fcn']["Data_dir"])
        train_datalaoder = DataLoader(train_dataset, batch_size=20, shuffle=True)
        # weight  --  消除数据不平衡
        losser = nn.CrossEntropyLoss(weight=torch.Tensor([1, train_dataset.imbalanced_ratio]))
        for epoch in range(epochs):
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
            lossSum = 0
            print('-'*5, f'epoch{epoch}', '-'*5)
            for time, (imgs, labels) in enumerate(train_datalaoder):
                optimizer.zero_grad()
                dpms = model(imgs)  # (?, 2, 1, 1, 1)
                dpms = dpms.view(-1, 2)  # (?, 2)
                loss = losser(dpms, labels)
                loss.backward()
                optimizer.step()
                lossSum += loss
                # Validing...
                with torch.no_grad():
                    if (time + 1) % 20 == 0:
                        matrix, ACCU, F1, MCC = valid(model, exp)
                        if MCC > best_MCC:
                            best_MCC = MCC
                            best_model = model.state_dict()
                            best_matrix = matrix
                            best_epoch = epoch
                        print(f'VALID MCC: {MCC}, VALID ACC: {ACCU}')
            # epoch 结束
            print(f'epoch_{epoch} 训练完成, train_loss: {lossSum}')
    if os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(best_model, '{}{}_{}.pth'.format(checkpoint_dir, 'FCN', best_epoch))
    print('Train Done !!!')


def valid(model:FCN_classical, exp):
    valid_dataset = Valid_and_Test_dataset(csv_dir + f'exp{exp}/', data_dir=config['fcn']["Data_dir"], mode='valid')
    valid_dataset = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    labels = []
    dpms = []
    for img, label in valid_dataset:
        dpm = model(img).numpy().squeeze()  # (2, 46, 55, 46)
        dpms.append(dpm)
        labels.append(label)
    valid_matrix = get_confusion_matrix(dpms, labels)
    matrix, ACCU, F1, MCC = DPM_statistics(valid_matrix)
    return matrix, ACCU, F1, MCC


def test_and_generate_DPMs(model:FCN_classical, exp):
    # 加载最优模型...
    model_list = os.listdir(checkpoint_dir)
    weights = None
    for name in model_list:
        if '_FCN' in name:
            weights = torch.load(name)
            break
    assert weights
    model.load_state_dict(weights)
    print('Testing...')
    for stage in ['train', 'valid', 'test', 'AIBL', 'NACC', 'FHS']:
        # 加载数据...
        if stage in ['train', 'valid', 'test']:
            testdataset = Valid_and_Test_dataset(csv_dir + f'exp{exp}/', mode=stage)
        else:
            testdataset = Valid_and_Test_dataset(csv_dir, mode=stage)
        testloader = DataLoader(testdataset, batch_size=1, shuffle=False)
        filenames = testdataset.imgnames
        DPMs, Labels = [], []
        for idx, (img, label) in enumerate(testloader):
            DPM = model(img).numpy().squeeze()  # (2, 46, 55, 46)
            np.save(dpms_dir +f'exp{exp}' + filenames[idx] + '.npy', DPM)
            DPMs.append(DPM)
            Labels.append(label)
        matrix, ACCU, F1, MCC = DPM_statistics(DPMs, Labels)
        np.save(dpms_dir + f'exp{exp}/{stage}_MCC.npy', MCC)
        np.save(dpms_dir + f'exp{exp}/{stage}_F1.npy', F1)
        np.save(dpms_dir + f'exp{exp}/{stage}_ACC.npy', ACCU)
        print(stage + ' confusion matrix: ', matrix, ' accuracy: ', ACCU)


if __name__ == '__main__':
    neurons = 20
    exps = 1
    epochs = 20
    lr = 0.1

    model = FCN_classical(neurons=20)

    class myDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.data = np.random.uniform(0, 1, size=(100, 1, 47, 47, 47))
            self.data = torch.from_numpy(self.data).to(torch.float32)
            self.label = np.random.randint(0, 1, size = (100, ))
            self.label = torch.from_numpy(self.label).to(torch.long)
            self.imbalanced_ratio = 1.0

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            data = self.data[idx]
            label = self.label[idx]
            return data, label

    print('STARTing...')
    for exp in range(exps):
        # 加载数据...
        train_dataset = myDataset()
        train_datalaoder = DataLoader(train_dataset, batch_size=20, shuffle=True)
        # weight  --  消除数据不平衡
        losser = nn.CrossEntropyLoss(weight=torch.Tensor([1, train_dataset.imbalanced_ratio]))
        for epoch in range(epochs):
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
            lossSum = 0
            print('-'*5, f'epoch{epoch}', '-'*5)
            for time, (imgs, labels) in enumerate(train_datalaoder):
                optimizer.zero_grad()
                dpms = model(imgs)  # (?, 2, 1, 1, 1)
                dpms = dpms.view(-1, 2)  # (?, 2)
                loss = losser(dpms, labels)
                loss.backward()
                optimizer.step()
                lossSum += loss
                # Validing...
                # with torch.no_grad():
                #     if (time + 1) % 20 == 0:
                #         matrix, ACCU, F1, MCC = valid(model, exp)
                #         if MCC > best_MCC:
                #             best_MCC = MCC
                #             best_model = model.state_dict()
                #             best_matrix = matrix
                #             best_epoch = epoch
                #         print(f'VALID MCC: {MCC}, VALID ACC: {ACCU}')
            # epoch 结束
            print(f'epoch_{epoch} 训练完成, train_loss: {lossSum}')
    # if os.path.exists(checkpoint_dir):
    #     os.mkdir(checkpoint_dir)
    # torch.save(best_model, '{}{}_{}.pth'.format(checkpoint_dir, 'FCN', best_epoch))
    print('Train Done !!!')
