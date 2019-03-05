import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import math
import time
import sys
import glob
import hdf5storage
from random import shuffle
import time
import os


from models.wisppn_resnet import ResNet, ResidualBlock, Bottleneck
batch_size = 32
num_epochs = 20

learning_rate = 0.001

def getMinibatch(file_names):
    file_num = len(file_names)
    csi_data = torch.zeros(file_num, 30*5, 3, 3)
    jmatrix_label = torch.zeros(file_num, 4, 18, 18)
    for i in range(file_num):
        data = hdf5storage.loadmat(file_names[i], variable_names={'csi_serial', 'jointsMatrix'})
        csi_data[i, :, :, :] = torch.from_numpy(data['csi_serial']).type(torch.FloatTensor).view(-1, 3, 3)
        jmatrix_label[i, :, :, :] = torch.from_numpy(data['jointsMatrix']).type(torch.FloatTensor)
    return csi_data, jmatrix_label


mats = glob.glob('E:/alphapose_data/train80singleperson/*.mat')
mats_num = len(mats)
batch_num = int(np.floor(mats_num/batch_size))

wisppn = ResNet(ResidualBlock, [2, 2, 2, 2])
# resnet = ResNet(ResidualBlock, [3, 4, 6, 3])
# resnet = ResNet(Bottleneck, [3, 4, 6, 3])
wisppn = wisppn.cuda()

criterion_L2 = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(wisppn.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20, 25, 30], gamma=0.5)

wisppn.train()

for epoch_index in range(num_epochs):
    scheduler.step()
    start = time.time()
    # shuffling dataset
    shuffle(mats)
    loss_x = 0
    # in each minibatch
    for batch_index in range(batch_num):
        if batch_index < batch_num:
            file_names = mats[batch_index*batch_size:(batch_index+1)*batch_size]
        else:
            file_names = mats[batch_num*batch_size:]

        csi_data, jmatrix_label = getMinibatch(file_names)

        csi_data = Variable(csi_data.cuda())
        xy = Variable(jmatrix_label[:, 0:2, :, :].cuda())
        confidence = Variable(jmatrix_label[:, 2:4, :, :].cuda())

        pred_xy = wisppn(csi_data)
        loss = criterion_L2(torch.mul(confidence, pred_xy), torch.mul(confidence, xy))

        print(loss.item())
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    endl = time.time()
    print('Costing time:', (endl-start)/60)

torch.save(wisppn, 'weights/wisppn-20190226.pkl')


