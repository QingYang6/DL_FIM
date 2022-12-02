import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from matplotlib import gridspec

class plotwithlabel():
    def __init__(self, tensors=None,labels=None,nrows=None,title=None,savepath=None, stackofVminmax=None, vmm_interval = 3):
        self.tensors = tensors
        self.labels = labels
        self.nrows = nrows
        self.title = title
        self.savepath = savepath
        self.stackofVminmax = stackofVminmax
        self.vmm_interval = vmm_interval
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

        plt.figure(figsize = (n_cols*3,self.nrows*3))
        gs1 = gridspec.GridSpec(self.nrows,n_cols)
        gs1.update(wspace=0.0001, hspace=0.0001)
        for i in range(len(self.tensors)):
            ax1 = plt.subplot(gs1[i])
            img = self.tensors[i]
            img = img.detach()
            image = img.numpy()
            image_dim = image.ndim
            if image_dim > 2:
                image = image[0,:,:]* 0.5 + 0.5
            else:
                image = image* 0.5 + 0.5
            if self.stackofVminmax != None:
                tvminmax = self.stackofVminmax[i//self.vmm_interval]
                ax1.imshow(image,interpolation='nearest',vmin=tvminmax[0], vmax=tvminmax[1], cmap=plt.cm.Greys_r)
            else:
                ax1.imshow(image,interpolation='nearest', cmap=plt.cm.Greys_r)
            if self.labels:
                ax1.text(0.82, .65, self.labels[i], horizontalalignment='center',transform=ax1.transAxes,
                    fontsize=16, color='red')
            ax1.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if self.title:
            plt.title(self.title)
        plt.savefig(self.savepath,dpi=500,format='png',bbox_inches='tight')#
        plt.close('all')