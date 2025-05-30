import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from matplotlib import gridspec
import pandas as pd
import seaborn as sns

class plotwithlabel():
    def __init__(self, tensors=None,labels=None,nrows=None,title=None,savepath=None, stackofVminmax=None, vmm_interval = 3, hist_interval = None):
        self.tensors = tensors
        self.labels = labels
        self.nrows = nrows
        self.title = title
        self.savepath = savepath
        self.stackofVminmax = stackofVminmax
        self.vmm_interval = vmm_interval
        self.hist_interval = hist_interval
        self.gridplot()

    def gridplot(self):
        if not isinstance(self.tensors, list):
            self.tensors = [self.tensors]
        n_plots = len(self.tensors)
        if self.nrows is None:
            n_cols = int(np.sqrt(n_plots))
            self.nrows = int(np.ceil(n_plots / n_cols))
        else:
            n_cols = int(n_plots // self.nrows)
        oldncols = n_cols
        if self.hist_interval != None:
            n_cols += 1

        plt.figure(figsize = (n_cols*3,self.nrows*3))
        if self.title:
            plt.title(self.title)
        gs1 = gridspec.GridSpec(self.nrows,n_cols)
        gs1.update(wspace=0.0001, hspace=0.0001)
        for i in range(len(self.tensors)):
            trow = i//oldncols
            tcol = i - trow*oldncols
            if tcol ==0:
                hist_img_list = []
            ax1 = plt.subplot(gs1[trow,tcol])
            img = self.tensors[i]
            img = img.detach()
            image = img.numpy()
            image_dim = image.ndim
            if image_dim > 2:
                image = image[0,:,:]* 0.5 + 0.5
            else:
                image = image* 0.5 + 0.5
            if self.hist_interval is not None:
                if tcol < self.hist_interval:
                    hist_img_list.append(image.flatten())
                else:
                    if tcol == self.hist_interval:
                        ax1 = self.histcomparison(hist_img_list,ax1)
                        #ax1.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                    ax1 = plt.subplot(gs1[trow,tcol+1])
            if self.stackofVminmax != None:
                tvminmax = self.stackofVminmax[i//self.vmm_interval]
                ax1.imshow(image,interpolation='nearest',vmin=tvminmax[0], vmax=tvminmax[1], cmap=plt.cm.Greys_r)
            else:
                ax1.imshow(image,interpolation='nearest', cmap=plt.cm.Greys_r)
            if self.labels:
                ax1.text(0.82, .65, self.labels[i], horizontalalignment='center',transform=ax1.transAxes,
                    fontsize=16, color='red')
            ax1.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.savefig(self.savepath,dpi=500,format='png',bbox_inches='tight')#
        plt.close('all')
    
    def histcomparison(self, hist_img_list, ax1):
        data_hist = pd.DataFrame()
        i = 1
        for real, fake in grouped(hist_img_list, 2):
            temp = pd.DataFrame(
                {
                    'Real%d' % i: real,
                    'Fake%d' % i: fake
                }
            )
            i += 1
            data_hist = pd.concat([data_hist, temp],axis=1)
        #data_hist = pd.DataFrame(data_list)
        #data_hist = pd.DataFrame({'Real': RB_array, 'Fake': FB_array, 'Ori': RA_array})
        #anno = 'WD_ori: %.4f\nWD_fake: %.4f' %  (WD_ori,WD_fake)
        #ax1 = plt.subplot(gs1[i_ms])
        #sns.displot(data_hist, kind="kde", ax=ax1)
        sns.histplot(data_hist, element="poly", ax=ax1)
        ax1.axis('off')
        #ax1.text(5, 5, anno, fontsize = 12, color = 'k')
        return ax1


def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)