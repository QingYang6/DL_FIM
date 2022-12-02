import numpy as np
import torch
import os,re
from data.base_dataset import BaseDataset
from datetime import datetime
import math
import random

class S1S1LCCDataset(BaseDataset):
    def __init__(self, opt, transform=None):
        BaseDataset.__init__(self, opt)
        self.npyfolder = opt.dataroot
        self.files = getnpylist(self.npyfolder)
        #self.transform = transform
        #self.Len_samples = len(self.files)
        #self.sampleidx = rand_pairs(len(self.files),self.Len_samples)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
             idx = idx.tolist()
        x = np.load(self.files[idx])
        #y = np.load(self.files[self.sampleidx[idx][1]])
        name = filename(self.files[idx])[:-6]
        name_split = name.split('_')
        S1_datetime = datetime.strptime(name_split[1],"%Y%m%d%H%M%S")
        day_of_year = S1_datetime.timetuple().tm_yday
        DOY= torch.tensor(day_of_year, dtype=torch.float)
        #x_nor = normalization_S2S1(x)
        x_nor = normalization_S1S1LCC(x,self.files[idx])
        #x_nor = x_nor.reshape((x_nor.shape[2], x_nor.shape[0], x_nor.shape[1]))
        x_nor = np.moveaxis(x_nor,-1,0)
        data = x_nor[:3,:,:]
        target = x_nor[3:,:,:]
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).float()
        return {'A': data, 'B': target, 'A_paths': self.files[idx], 'B_paths': self.files[idx], 'DX': idx, 'C':DOY}

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

def normalization_S1S1LCC(input_array,file):
    normalindex = [100,100,30,100,100,30,50,50]
    for ibs in range(input_array.shape[-1]):
        tband_array = input_array[:, :, ibs]
        if ibs == 2 or ibs == 5:
            tband_array = (tband_array / normalindex[ibs]) - 1
        else:
            tband_array= ((tband_array + normalindex[ibs]) / normalindex[ibs]) - 1
        input_array[:, :, ibs] = tband_array
        minarray = np.min(input_array[:, :, ibs])
        maxarray = np.max(input_array[:, :, ibs])
        if minarray < -1 or maxarray > 1:
            print('bands ' + str(ibs) + ' excess boundary '
                                            ' ' + str(minarray) + ' ' + str(maxarray))
            print(file)
    return input_array

def filename(filepath):
    tnamewithext = os.path.split(filepath)[1]
    filename = os.path.splitext(tnamewithext)[0]
    return filename

def keepnumeric(string):
    nstr = re.sub("[^0-9]", "", string)
    return nstr