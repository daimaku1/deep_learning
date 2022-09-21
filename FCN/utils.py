# 开发者：热心网友
# 开发时间：2022/8/30 19:52
# coding:utf-8
import numpy as np


def get_confusion_matrix(preds, labels):
    labels = labels.numpy()
    preds = preds.numpy()
    matrix = [[0, 0], [0, 0]]
    for index, pred in enumerate(preds):
        if np.amax(pred) == pred[0]:
            if labels[index] == 0:
                matrix[0][0] += 1
            if labels[index] == 1:
                matrix[0][1] += 1
        elif np.amax(pred) == pred[1]:
            if labels[index] == 0:
                matrix[1][0] += 1
            if labels[index] == 1:
                matrix[1][1] += 1
    return matrix


def DPM_statistics(DPMs, Labels):
    shape = DPMs[0].shape[1:]
    voxel_number = shape[0] * shape[1] * shape[2]
    TP, FP, TN, FN = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for label, DPM in zip(Labels, DPMs):
        risk_map = get_AD_risk(DPM)
        if label == 0:
            TN += (risk_map < 0.5).astype(np.int)
            FP += (risk_map >= 0.5).astype(np.int)
        elif label == 1:
            TP += (risk_map >= 0.5).astype(np.int)
            FN += (risk_map < 0.5).astype(np.int)
    tn = float("{0:.2f}".format(np.sum(TN) / voxel_number))
    fn = float("{0:.2f}".format(np.sum(FN) / voxel_number))
    tp = float("{0:.2f}".format(np.sum(TP) / voxel_number))
    fp = float("{0:.2f}".format(np.sum(FP) / voxel_number))
    matrix = [[tn, fn], [fp, tp]]
    count = len(Labels)
    TP, TN, FP, FN = TP.astype(np.float)/count, TN.astype(np.float)/count, FP.astype(np.float)/count, FN.astype(np.float)/count
    ACCU = TP + TN
    F1 = 2*TP/(2*TP+FP+FN)
    MCC = (TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))+0.00000001*np.ones(shape))
    return matrix, ACCU, F1, MCC

# 进行softmax
def get_AD_risk(raw):
    x1, x2 = raw[0, :, :, :], raw[1, :, :, :]
    risk = np.exp(x2) / (np.exp(x1) + np.exp(x2))
    return risk