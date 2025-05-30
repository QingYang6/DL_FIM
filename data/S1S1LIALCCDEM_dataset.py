import numpy as np
import torch
import os,re
from data.base_dataset import BaseDataset
from datetime import datetime
import math
import random

class S1S1LIALCCDEMDataset(BaseDataset):
    def __init__(self, opt, transform=None):
        BaseDataset.__init__(self, opt)
        self.npyfolder = opt.dataroot
        self.files,self.orsartypes = getnpylist(self.npyfolder)
        if not self.files:
            self.files,self.orsartypes = getnpylist(self.npyfolder,'.npz')
        self.arr_files,self.arr_orsartypes = np.asarray(self.files), np.asarray(self.orsartypes)
        #self.transform = transform
        #self.Len_samples = len(self.files)
        #self.sampleidx = rand_pairs(len(self.files),self.Len_samples)
        self.len_sample = len(self.files)
        print(len(self.files),len(self.orsartypes))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
             idx = idx.tolist()
        randidx=np.random.choice(self.len_sample, 2, replace=False)
        if os.path.splitext(self.files[idx])[1] == '.npz':
            try:
                loaded = np.load(self.files[idx])
                loaded_x2 = np.load(self.files[randidx[0]])
                x = loaded['allBands']
                x2 = loaded_x2['allBands']
            except Exception as e:
                print(e)
                print(self.files[idx]+' load sample failed')
                loaded = np.load(self.files[randidx[1]])
                loaded_x2 = np.load(self.files[randidx[0]])
                x = loaded['allBands']
                x2 = loaded_x2['allBands']
        else:
            x = np.load(self.files[idx])
            x2 = np.load(self.files[randidx[0]])
        #x = np.load(self.files[idx])
        #x_orsart = self.orsartypes[idx]
        #sameorsartypex = self.arr_files[np.where(self.arr_orsartypes ==x_orsart)[0]]
        #randidx=np.random.choice(len(sameorsartypex), 1, replace=False) 
        #x2 = np.load(sameorsartypex[randidx[0]])
        #y = np.load(self.files[self.sampleidx[idx][1]])
        name = filename(self.files[idx])[:-6]
        name_split = name.split('_')
        #S1_datetime = datetime.strptime(name_split[1],"%Y%m%d%H%M%S")
        #day_of_year = S1_datetime.timetuple().tm_yday
        #DOY= torch.tensor(day_of_year, dtype=torch.float)
        DOY=255
        x_nor = normalization_S1S1LCCDEM(x,self.files[idx])
        x_nor2 = normalization_S1S1LCCDEM(x2,self.files[randidx[0]])
        x_norRL = x_nor2
        x_nor = np.moveaxis(x_nor,-1,0)
        x_norRL = np.moveaxis(x_norRL,-1,0)
        data = x_nor 
        target = x_norRL
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


def getnpylist(npyfolder,ext='.npy'):
    if type(npyfolder) is list:
        npyfiles = []
        orbitandsartype = []
        for ifolder in npyfolder:
            inpyfiles = [os.path.join(ifolder,dI) for dI in os.listdir(ifolder)
                              if ext in dI]
            iorsartype = [dI.split('_')[0] for dI in os.listdir(ifolder)
                              if ext in dI]
            npyfiles = npyfiles + inpyfiles
            orbitandsartype = orbitandsartype + iorsartype
    else:
        npyfiles = [os.path.join(npyfolder,dI) for dI in os.listdir(npyfolder)
                          if ext in dI]
        orbitandsartype = [dI.split('_')[0] for dI in os.listdir(npyfolder)
                        if ext in dI]
    return npyfiles,orbitandsartype

def normalization_S1S1LCC(input_array,file):
    normalindex = [100,100,30,100,100,30,50,50,100,100,30]
    for ibs in range(input_array.shape[-1]):
        tband_array = input_array[:, :, ibs]
        if ibs == 2 or ibs == 5 or ibs == 6 or ibs == 7 or ibs == 10:
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

def normalization_S1S1LCCDEM(input_array,file):
    normalindex = [100,100,30,100,100,30,50,50,4300,100,100,30]
    for ibs in range(input_array.shape[-1]):
        tband_array = input_array[:, :, ibs]
        if ibs == 2 or ibs == 5 or ibs == 6 or ibs == 7 or ibs ==11:
            tband_array = (tband_array / normalindex[ibs]) - 1
        elif ibs == 8:
            tband_array[tband_array<-200] = -200
            tband_array = ((tband_array + 200) / normalindex[ibs]) - 1
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