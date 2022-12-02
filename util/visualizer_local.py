from __future__ import annotations
import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
from torchvision.utils import save_image
sys.path.insert(0, '/shared/stormcenter/Qing_Y/GAN_ChangeDetection/torchimplementation')
from utils.tensorplt import *
from scipy.stats import wasserstein_distance
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec




def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape

        util.save_image(im, save_path, aspect_ratio=aspect_ratio)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


def save_images_v2(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        if label != 'fake_B':
           continue
        im = util.tensor2im(im_data)
        image_name = '%s.png' % (name)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape

        util.save_image(im, save_path, aspect_ratio=aspect_ratio)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


def save_images_vr(webpage, visuals, image_path, aspect_ratio=1.0, width=256, check_label = 'vis_A2B'):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        if label != check_label:
            continue
        im = util.tensor2im(im_data)
        image_name = '%s.png' % (name)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape

        util.save_image(im, save_path, aspect_ratio=aspect_ratio)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)

def save_images_pix2pix(visuals, image_path, Out_path, n_rowval,title):
    """Save images to the disk. Based on old school matplotlib

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    #short_path = ntpath.basename(image_path[0])
    #name = os.path.splitext(short_path)[0]
    stakckofimgs = []
    for label, im_data in visuals.items():
        stakckofimgs.append(im_data)
    stackoflistimgs = []
    stackofVminmax = []
    img_r2 = None
    if len([*stakckofimgs])==3:
        for realA, fakeB, realB in zip(*stakckofimgs):
            realA_list = util.batch2list(realA)
            realB_list = util.batch2list(realB)
            stackoflistimgs.extend(realA_list)
            stackoflistimgs.extend(realB_list)
            stackoflistimgs.extend(util.batch2list(fakeB))
            arr_RealA = tensor2numpyarray(realA_list[0])
            arr_RealB = tensor2numpyarray(realB_list[0])
            #print(np.min(arr_RealA))
            #print(np.min(arr_RealB))
            #print(np.max(arr_RealA))
            #print(np.max(arr_RealB))
            Vminmax = [(min(np.min(arr_RealA),np.min(arr_RealB)),max(np.max(arr_RealA),np.max(arr_RealB)))]
            stackofVminmax.extend(Vminmax)
    else:
        for real_B,fake_B,fake_B_random,fake_B_random2 in zip(*stakckofimgs): 
            realB_list = util.batch2list(real_B)
            stackoflistimgs.extend(realB_list)
            stackoflistimgs.extend(util.batch2list(fake_B))
            stackoflistimgs.extend(util.batch2list(fake_B_random))
            stackoflistimgs.extend(util.batch2list(fake_B_random2))
            arr_RealB = tensor2numpyarray(realB_list[0])
            Vminmax = [(min(np.min(arr_RealB),np.min(arr_RealB)),max(np.max(arr_RealB),np.max(arr_RealB)))]
            stackofVminmax.extend(Vminmax)
    plotwithlabel(stackoflistimgs,img_r2,n_rowval,title,Out_path,stackofVminmax,len([*stakckofimgs]))

def tensor2numpyarray(ten_image):
    image = ten_image.numpy()
    image_dim = image.ndim
    if image_dim > 2:
        image = image[0,:,:]* 0.5 + 0.5
    else:
        image = image* 0.5 + 0.5
    return image

def save_images_fakeandreal(visuals, image_path, Out_path, i_ms):
    realdir=os.path.join(Out_path,'real')
    fakedir=os.path.join(Out_path,'fake')
    os.makedirs(realdir, exist_ok=True)
    os.makedirs(fakedir, exist_ok=True)
    stakckofimgs = []
    for label, im_data in visuals.items():
        stakckofimgs.append(im_data)
    stackoflistimgs = []
    if len([*stakckofimgs])==3:
        for realA, fakeB, realB in zip(*stakckofimgs):
            if realA.size(0) > 1:
                realB = realB[0,:,:]* 0.5 + 0.5
                fakeB = fakeB[0,:,:]* 0.5 + 0.5
            else:
                realB = realB* 0.5 + 0.5
                fakeB = fakeB* 0.5 + 0.5
            save_image(realB, os.path.join(realdir,str(i_ms)+'.png'))
            save_image(fakeB, os.path.join(fakedir,str(i_ms)+'.png'))
            i_ms += 1
    else:
        for realB,fakeB,_,_ in zip(*stakckofimgs):
            if realB.size(0) > 1:
                realB = realB[0,:,:]* 0.5 + 0.5
                fakeB = fakeB[0,:,:]* 0.5 + 0.5
            else:
                realB = realB* 0.5 + 0.5
                fakeB = fakeB* 0.5 + 0.5
            save_image(realB, os.path.join(realdir,str(i_ms)+'.png'))
            save_image(fakeB, os.path.join(fakedir,str(i_ms)+'.png'))
            i_ms += 1
    return realdir, fakedir, i_ms

def test_compute_distance(visuals, i_ms, Histpng, Histtitle, Out_path, nrows):
    oridir=os.path.join(Out_path,'ori')
    realdir=os.path.join(Out_path,'real')
    fakedir=os.path.join(Out_path,'fake')
    os.makedirs(realdir, exist_ok=True)
    os.makedirs(fakedir, exist_ok=True)
    os.makedirs(oridir, exist_ok=True)
    histplot = True if i_ms ==0 else False
    stakckofimgs = []
    for label, im_data in visuals.items():
        batch_size = im_data.size(0)
        stakckofimgs.append(im_data)
    if histplot:
        plt,gs1 = gridplot_setup(batch_size,nrows)
        plt.title(Histtitle)
    WD_listFake = []
    WD_listOri = []
    for realA, fakeB, realB in zip(*stakckofimgs):
        if realA.size(0) > 1:
            if realA.size(0) <= 3:
                realA3b = realA* 0.5 + 0.5
                realB3b = realB* 0.5 + 0.5
                fakeB3b = fakeB* 0.5 + 0.5
            else:
                realA3b = realA[0,:,:]* 0.5 + 0.5
                realB3b = realB[0,:,:]* 0.5 + 0.5
                fakeB3b = fakeB[0,:,:]* 0.5 + 0.5
            realA = realA[0,:,:]* 0.5 + 0.5
            realB = realB[0,:,:]* 0.5 + 0.5
            fakeB = fakeB[0,:,:]* 0.5 + 0.5
        else:
            realA3b = realA* 0.5 + 0.5
            realB3b = realB* 0.5 + 0.5
            fakeB3b = fakeB* 0.5 + 0.5
            realA = realA* 0.5 + 0.5
            realB = realB* 0.5 + 0.5
            fakeB = fakeB* 0.5 + 0.5
        save_image(realB3b, os.path.join(realdir,str(i_ms)+'.png'))
        save_image(fakeB3b, os.path.join(fakedir,str(i_ms)+'.png'))
        save_image(realA3b, os.path.join(oridir,str(i_ms)+'.png'))
        FB_array = fakeB.cpu().numpy().flatten()
        RB_array = realB.cpu().numpy().flatten()
        RA_array = realA.cpu().numpy().flatten()
        WD_fake = wasserstein_distance(FB_array, RB_array)
        WD_ori  = wasserstein_distance(RA_array, RB_array)
        WD_listFake.append(WD_fake)
        WD_listOri.append(WD_ori)
        if histplot:
            data_hist = pd.DataFrame({'Real': RB_array, 'Fake': FB_array, 'Ori': RA_array})
            anno = 'WD_ori: %.4f\nWD_fake: %.4f' %  (WD_ori,WD_fake)
            ax1 = plt.subplot(gs1[i_ms])
            #sns.displot(data_hist, kind="kde", ax=ax1)
            sns.histplot(data_hist, element="poly", ax=ax1)
            ax1.axis('off')
            ax1.text(5, 5, anno, fontsize = 12, color = 'k')
        i_ms += 1
    if histplot:
        plt.savefig(Histpng, dpi=300, format='png', bbox_inches='tight')
        plt.close()
    return WD_listFake, WD_listOri, i_ms, realdir,fakedir,oridir

def gridplot_setup(batch_size,nrows):
    n_plots = batch_size
    if nrows is None:
        n_cols = int(np.sqrt(n_plots))
        nrows = int(np.ceil(n_plots / n_cols))
    else:
        n_cols = int(n_plots // nrows)

    plt.figure(figsize = (n_cols*3,nrows*3))
    gs1 = gridspec.GridSpec(nrows,n_cols)
    gs1.update(wspace=0.0001, hspace=0.0001)
    return plt,gs1

class Visualizer_local():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = True #opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        self.ncols = opt.display_ncols

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
            else:     # show each image in a separate visdom panel;
                idx = 1
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                    win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)


    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data, modelname,writer,total_iters):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.4f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
        
        # Save log
        writer.add_scalars(modelname, losses, total_iters)
