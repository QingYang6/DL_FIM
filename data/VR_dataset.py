import numpy as np
import torch
import os,re
from data.base_dataset import BaseDataset
from datetime import datetime

class VRDataset(BaseDataset):
    def __init__(self, opt, transform=None):
        BaseDataset.__init__(self, opt)
        self.npyfolder = opt.dataroot
        self.files = getnpylist(self.npyfolder)
        self.transform = transform
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
             idx = idx.tolist()
        x = np.load(self.files[idx])
        name = filename(self.files[idx])
        xchange = x[:, :, 1]
        xchange[xchange >= 100] = 0 # mask out flood infor in VIIRs rescaled water
        x[:, :, 1] = xchange
        x_nor = normalization_VR(x)
        #x_nor = x_nor.reshape((x_nor.shape[2], x_nor.shape[0], x_nor.shape[1]))
        data = x_nor[:,:,5]
        condition = x_nor[:,:,2:5]
        target = x_nor[:,:,-1]
        data = torch.from_numpy(data).float()
        data = torch.reshape(data,(1, data.shape[0], data.shape[1]))
        condition = torch.from_numpy(condition).float()
        condition = torch.reshape(condition,(condition.shape[2], condition.shape[0], condition.shape[1]))
        target = torch.from_numpy(target).float()
        target = torch.reshape(target,(1, target.shape[0], target.shape[1]))
        return {'A': data, 'B': target, 'C':condition, 'A_paths': self.files[idx], 'B_paths': self.files[idx], 'DX': idx}

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

def normalization_VR(input_array):
    array_dim = input_array.ndim
    normalindex_bands = [50,100,3500,4300,50,50,50]
    normalindx = normalindex_bands
    _, _, len_third_axis = input_array.shape
    for ibands in range(len_third_axis):
        if ibands == 3:
            tband_array = input_array[:, :, ibands]
            tband_array[tband_array<-200] = -200
            againarray = ((tband_array + 200)
                        / normalindx[ibands]) - 1
            againarray [ againarray>= 1] = 1 
            input_array[:, :, ibands] = againarray
        elif ibands == 2:
            tband_array = input_array[:, :, ibands]
            tband_array[tband_array<-1] = -1
            tband_array [tband_array>=0] = tband_array [tband_array>=0] +100
            againarray = ((tband_array + 1)
                        / normalindx[ibands]) - 1
            againarray [ againarray>= 1] = 1 
            input_array[:, :, ibands] = againarray
        else:
            input_array[:, :, ibands] = (input_array[:, :, ibands] / normalindx[ibands]) - 1
        if ibands == 5:
            tband_array = input_array[:, :, ibands]
            tband_array[tband_array>1] = -1
            input_array[:, :, ibands] = tband_array
        minarray = np.min(input_array[:, :, ibands])
        maxarray = np.max(input_array[:, :, ibands])
        if minarray < -1 or maxarray > 1:
            print('bands ' + str(ibands) + ' excess boundary '
                                           ' ' + str(minarray) + ' ' + str(maxarray))
    return input_array

def filename(filepath):
    tnamewithext = os.path.split(filepath)[1]
    filename = os.path.splitext(tnamewithext)[0]
    return filename

def keepnumeric(string):
    nstr = re.sub("[^0-9]", "", string)
    return nstr