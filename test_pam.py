import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import sys
import glob
import hdf5storage
from random import shuffle
import time
import os
import cv2

limb = np.array([[0,1],[0,14],[0,15],[14,16],[15,17],[1,2],[1,5],[1,8],[1,11],[2,3],[3,4],[5,6],[6,7],[8,9],[9,10],[11,12],[12,13]])


data = hdf5storage.loadmat('exaplmes/oct17set1_fw_1814.mat', variable_names={'csi_serial', 'frame'})
csi_data = torch.zeros(1, 30*5, 3, 3)
csi_data[0,:,:,: ]= torch.from_numpy(data['csi_serial']).type(torch.FloatTensor).view(-1, 3, 3)
frame = data['frame']
# frame = frame[...,::-1]

wisppn = torch.load('weights/wisppn-20190226.pkl')
wisppn = wisppn.cuda().eval()

csi_data = Variable(csi_data.cuda())
pred_xy = wisppn(csi_data)
pred_xy = pred_xy.cpu().detach().numpy()

poseVector_x = np.zeros((1,18))
poseVector_y = np.zeros((1,18))
for index in range(18):
    poseVector_x[0,index] = pred_xy[0,0,index,index]
    poseVector_y[0,index] = pred_xy[0,1,index,index]

while 1:
    # Display the image
    plt.imshow(cv2.resize(frame, (1280, 720)))

    for i in range(len(limb)):
        plt.plot(poseVector_x[0, [limb[i, 0], limb[i, 1]]], poseVector_y[0,[limb[i, 0], limb[i, 1]]])

    plt.show()
    cv2.waitKey(15)


