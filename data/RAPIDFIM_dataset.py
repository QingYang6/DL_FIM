import numpy as np
import torch
import os,re
from data.base_dataset import BaseDataset
from datetime import datetime
import math
import random

class RAPIDFIMDataset(BaseDataset):
    def __init__(self, opt, transform=None):
        BaseDataset.__init__(self, opt)
        self.npyfolder = opt.dataroot
        self.files = getnpylist(self.npyfolder)
        if not self.files:
            self.files = getnpylist(self.npyfolder,'.npz')
        #self.transform = transform
        #self.Len_samples = len(self.files)
        #self.sampleidx = rand_pairs(len(self.files),self.Len_samples)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
             idx = idx.tolist()
        if os.path.splitext(self.files[idx])[1] == '.npz':
            loaded = np.load(self.files[idx])
            x = loaded['allBands']
        else:
            x = np.load(self.files[idx])
        #y = np.load(self.files[self.sampleidx[idx][1]])
        name = filename(self.files[idx])[:-6]
        name_split = name.split('_')
        S1_datetime = datetime.strptime(name_split[4],"%Y%m%dT%H%M%S")
        day_of_year = S1_datetime.timetuple().tm_yday
        DOY= torch.tensor(day_of_year, dtype=torch.float)
        #x_nor = normalization_S2S1(x)
        x_nor = normalization_RAPIDonly(x,self.files[idx])
        #x_nor = x_nor.reshape((x_nor.shape[2], x_nor.shape[0], x_nor.shape[1]))
        x_nor = np.moveaxis(x_nor,-1,0)
        data = x_nor[:-1,:,:]
        target = x_nor[-1,:,:]
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).float()
        target = torch.reshape(target,(1, target.shape[0], target.shape[1]))
        return {'A': data, 'B': target, 'A_paths': self.files[idx], 'B_paths': self.files[idx], 'DX': idx, 'C':DOY}

def decode(i):
    k = math.floor((1+math.sqrt(1+8*i))/2)
    return k,i-k*(k-1)//2

def rand_pair(n):
    return decode(random.randrange(n*(n-1)//2))

def rand_pairs(n,m):
    return [decode(i) for i in random.sample(range(n*(n-1)//2),m)]

def getnpylist(npyfolder,ext='.npy'):
    if type(npyfolder) is list:
        npyfiles = []
        for ifolder in npyfolder:
            inpyfiles = [os.path.join(ifolder,dI) for dI in os.listdir(ifolder)
                              if ext in dI]
            npyfiles = npyfiles + inpyfiles
    else:
        npyfiles = [os.path.join(npyfolder,dI) for dI in os.listdir(npyfolder)
                          if ext in dI]
    return npyfiles

def normalization_S1S1LCC(input_array,file):
    normalindex = [100,100,30,100,100,30,50,50]
    for ibs in range(input_array.shape[-1]):
        tband_array = input_array[:, :, ibs]
        if ibs == 2 or ibs == 5 or ibs == 6 or ibs == 7:
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

def normalization_RAPIDonly(input_array,filename):
    array_dim = input_array.ndim
    normalindex_bands = [3500,4300,50,50,50,50]
    normalindx = normalindex_bands
    _, _, len_third_axis = input_array.shape
    for ibands in range(len_third_axis):
        if ibands == 1:
            tband_array = input_array[:, :, ibands]
            tband_array[tband_array<-200] = -200
            againarray = ((tband_array + 200)
                        / normalindx[ibands]) - 1
            againarray [ againarray>= 1] = 1 
            input_array[:, :, ibands] = againarray
        elif ibands == 0:
            tband_array = input_array[:, :, ibands]
            tband_array[tband_array<-1] = -1
            tband_array [tband_array>=0] = tband_array [tband_array>=0] +100
            againarray = ((tband_array + 1)
                        / normalindx[ibands]) - 1
            againarray [ againarray>= 1] = 1 
            input_array[:, :, ibands] = againarray
        else:
            input_array[:, :, ibands] = (input_array[:, :, ibands] / normalindx[ibands]) - 1
        minarray = np.min(input_array[:, :, ibands])
        maxarray = np.max(input_array[:, :, ibands])
        if minarray < -1 or maxarray > 1:
            print('bands ' + str(ibands) + ' excess boundary '
                                           ' ' + str(minarray) + ' ' + str(maxarray))
            print(filename)
    return input_array


def filename(filepath):
    tnamewithext = os.path.split(filepath)[1]
    filename = os.path.splitext(tnamewithext)[0]
    return filename

def keepnumeric(string):
    nstr = re.sub("[^0-9]", "", string)
    return nstr