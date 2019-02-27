import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
# from model import locNN
import torch
import torch.nn as nn
from torch.autograd import Variable
from net import ResNet, ResidualBlock
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import pandas as pd


#
batch_size = 20
# load data
# data = sio.loadmat('../dataset/test.mat')
# test_data = data['test_data']
# test_label = data['test_label']

data = sio.loadmat('../dataset/test.mat')
test_data = data['test_data']
test_label = data['test_label']


# reshape train data size to nSample x nSubcarrier x 1 x 1
num_test_instances = len(test_data)

test_data = torch.from_numpy(test_data).type(torch.FloatTensor).view(num_test_instances, 30, 1, 1)
test_label = torch.from_numpy(test_label).type(torch.FloatTensor)
test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


resnet = torch.load('model/res18_2.pkl')
print(resnet)
resnet = resnet.cuda()


resnet.eval()

# a = resnet.conv1[0]._parameters['weight'].data
# b = a.cpu().numpy()
#
# sio.savemat('results/conv1Weights.mat', {'weight': b})


correct = 0

for i, (samples, labels) in enumerate(test_data_loader):
    samplesV = Variable(samples.cuda(), volatile=True)
    labelsV = Variable(labels.cuda(), volatile=True)

    predict_label = resnet(samplesV)

    # pred = predict_label[0][:, 0:30].data.max(1)[1]
    # correct += pred.eq(labelsV[:, 0].data.long()).sum()


    if i == 0:
        temp = predict_label.data
        sign_pred = temp

    elif i > 0:
        temp = predict_label.data
        sign_pred = np.concatenate((sign_pred, temp), axis=0)


# sio.savemat('results/train_result.mat', {'sign_pred': sign_pred})


test_label = data['train_label']

pred = sign_pred.argmax(1)

correct = 0

for i in range(len(test_label)):
    if pred[i] == test_label[i]:
        correct += 1

print('Accuracy:', 100*correct/len(test_label))
