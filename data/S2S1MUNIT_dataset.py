import numpy as np
import torch
import os,re
from data.base_dataset import BaseDataset
from datetime import datetime
import math
import random

class S2S1MUNITDataset(BaseDataset):
    def __init__(self, opt, transform=None):
        BaseDataset.__init__(self, opt)
        self.npyfolder = opt.dataroot
        self.files = getnpylist(self.npyfolder)
        self.transform = transform
        self.Len_samples = len(self.files)
        #self.sampleidx = rand_pairs(len(self.files),self.Len_samples)

    def __len__(self):
        return self.Len_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
             idx = idx.tolist()
        x = np.load(self.files[idx])
        #y = np.load(self.files[self.sampleidx[idx][1]])
        name = filename(self.files[idx])[:-6]
        name_split = name.split('_')
        x_nor = normalization_S2S1(x)
        #y_nor = normalization_S2S1(y)
        #x_nor = x_nor.reshape((x_nor.shape[2], x_nor.shape[0], x_nor.shape[1]))
        data = x_nor[:,:,[0,1,2,3,4,7]]
        target = x_nor[:,:,[9,9,9,9,9,7]]
        data = torch.from_numpy(data).float()
        data = torch.reshape(data,(data.shape[2], data.shape[0], data.shape[1]))
        target = torch.from_numpy(target).float()
        target = torch.reshape(target,(target.shape[2], target.shape[0], target.shape[1]))
        return {'A': data, 'B': target, 'A_paths': self.files[idx], 'B_paths': self.files[idx], 'DX': idx}

def decode(i):
    k = math.floor((1+math.sqrt(1+8*i))/2)
    return k,i-k*(k-1)//2

def rand_pair(n):
    return decode(random.randrange(n*(n-1)//2))

def rand_pairs(n,m):
    return [decode(i) for i in random.sample(range(n*(n-1)//2),m)]

def getnpylist(npyfolder):
    if type(npyfolder) is list:
        npyfiles = []
        for ifolder in npyfolder:
            inpyfiles = [os.path.join(ifolder,dI) for dI in os.listdir(ifolder)
                              if '.npy' in dI]
            npyfiles = npyfiles + inpyfiles
    else:
        npyfiles = [os.path.join(npyfolder,dI) for dI in os.listdir(npyfolder)
                          if '.npy' in dI]
    return npyfiles

def normalization_S2S1(input_array):
    array_dim = input_array.ndim
    normalindex_9bands = [17500,17500,17500,17500,17500,50,4300,30,0.5,100]
    normalindex_1band = [100]
    if array_dim > 2:
        normalindx = normalindex_9bands
    else:
        normalindx = normalindex_1band
    if array_dim > 2:
        _, _, len_third_axis = input_array.shape
        for ibands in range(len_third_axis):
            if ibands == 6:
                tband_array = input_array[:, :, ibands]
                tband_array[tband_array<-200] = -200
                input_array[:, :, ibands] = ((tband_array + 200)
                                             / normalindx[ibands]) - 1
            elif ibands == 9:
                input_array= ((input_array+100) / normalindx[9]) - 1
            else:
                input_array[:, :, ibands] = (input_array[:, :, ibands] / normalindx[ibands]) - 1
            minarray = np.min(input_array[:, :, ibands])
            maxarray = np.max(input_array[:, :, ibands])
            if minarray < -1 or maxarray > 1:
                print('bands ' + str(ibands) + ' excess boundary '
                                               ' ' + str(minarray) + ' ' + str(maxarray))
            #print('bands ' + str(ibands) + ' ' + str(minarray) + ' ' + str(maxarray))
    else:
        input_array= ((input_array+100) / normalindx[0]) - 1
        minarray = np.min(input_array)
        maxarray = np.max(input_array)
        #print('bands y ' + str(minarray) + ' ' + str(maxarray))
        if minarray < -1 or maxarray > 1:
            print('bands y ' + ' excess boundary '
                                           ' ' + str(minarray) + ' ' + str(maxarray))
    return input_array

def normalization_S1S1(input_array):
    array_dim = input_array.ndim
    normalindex = [100,100,30,100,100,30]
    if array_dim > 2:
        _, _, len_third_axis = input_array.shape
        for ibands in range(len_third_axis):
            if ibands == 2 or ibands == 5:
                input_array[:, :, ibands] = (input_array[:, :, ibands] / normalindex[ibands]) - 1
            else:
                input_array= ((input_array+100) / normalindex[ibands]) - 1
            minarray = np.min(input_array[:, :, ibands])
            maxarray = np.max(input_array[:, :, ibands])
            if minarray < -1 or maxarray > 1:
                print('bands ' + str(ibands) + ' excess boundary '
                                               ' ' + str(minarray) + ' ' + str(maxarray))
            #print('bands ' + str(ibands) + ' ' + str(minarray) + ' ' + str(maxarray))
    else:
        input_array= ((input_array+100) / normalindex[0]) - 1
        minarray = np.min(input_array)
        maxarray = np.max(input_array)
        #print('bands y ' + str(minarray) + ' ' + str(maxarray))
        if minarray < -1 or maxarray > 1:
            print('bands y ' + ' excess boundary '
                                           ' ' + str(minarray) + ' ' + str(maxarray))
    return input_array

def filename(filepath):
    tnamewithext = os.path.split(filepath)[1]
    filename = os.path.splitext(tnamewithext)[0]
    return filename

def keepnumeric(string):
    nstr = re.sub("[^0-9]", "", string)
    return nstr