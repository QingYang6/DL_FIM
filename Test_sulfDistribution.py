import time, os, shutil, glob, re
from options.train_options import TrainOptions
from util.visualizer_local import Visualizer_local
from util.visualizer_local import save_images_pix2pix,save_images_fakeandreal,test_compute_distance
from cleanfid import fid
from data import create_dataset
from models import create_model
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torchvision.transforms.functional as F
from matplotlib import gridspec

def PTtofolder(opt,epoch,total_iters,model,visualizer,Val_dataset):
    print('getting the FID score at the end of epoch %d, iters %d' % (epoch, total_iters))
    model.eval()
    FID_log = os.path.join(opt.results_dir,'FID_log.txt')
    FID_folder = os.path.join(opt.results_dir,'FID_for_iter_%d' % total_iters)
    i_ms = 0
    os.makedirs(FID_folder, exist_ok=True)
    for ival, dataval in enumerate(Val_dataset):
        visualizer.reset()
        model.set_input(dataval)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        realdir,fakedir,i_ms=save_images_fakeandreal(visuals, img_path, FID_folder, i_ms)
    print('The number of validation (FID) images = %d' % i_ms)
    score = fid.compute_fid(realdir, fakedir,mode="clean")
    message = 'Clean-fid: %.4f ' %  score
    print(f"Clean-fid score: {score:.4f}")
    with open(FID_log, "a") as log_FID:
        log_FID.write('%s\n' % message)  # save the message
    shutil.rmtree(FID_folder)

def evalforDistance(opt,epoch,total_iters,model,visualizer,Val_dataset):
    print('getting the Distance score at the end of epoch %d, iters %d' % (epoch, total_iters))
    model.eval()
    Distance_folder = os.path.join(opt.results_dir,'Test_Distance')
    Distance_csv = os.path.join(Distance_folder,'Distance_epoch_%d.csv' % epoch)
    Distance_total_log = os.path.join(Distance_folder,'W_Distance.txt')
    i_ms = 0
    Histpng = os.path.join(Distance_folder,'Hist_%d.png' % epoch)
    Histtitle = opt.name+'_Hist_%d' % epoch
    os.makedirs(Distance_folder, exist_ok=True)
    FID_log = os.path.join(Distance_folder,'FID_log_epochs.txt')
    FID_folder = os.path.join(Distance_folder,'FID_for_iter_%d' % total_iters)
    L_fake,L_ori = [],[]
    showima = True
    for ival, dataval in enumerate(Val_dataset):
        visualizer.reset()
        model.set_input(dataval)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        WD_listFake, WD_listOri,i_ms,realdir,fakedir,oridir=test_compute_distance(visuals, i_ms, Histpng, Histtitle, FID_folder, opt.valid_nrows)
        #realdir,fakedir,i_ms=save_images_fakeandreal(visuals, img_path, FID_folder, i_ms)
        L_fake.extend(WD_listFake)
        L_ori.extend(WD_listOri)
        if showima:
            imgtitle = opt.name+'_val_%d' % total_iters
            predictout = os.path.join(Distance_folder,'trainval_%d.png' % total_iters)
            save_images_pix2pix(visuals, img_path, predictout,opt.valid_nrows,imgtitle)
            showima=False
    zipped = list(zip(L_ori, L_fake))
    df_testdistance = pd.DataFrame(zipped, columns=['WD_ori','WD_fake'])
    df_testdistance.to_csv(Distance_csv)
    print('The number of test images = %d' % i_ms)
    meanD_fake = np.mean(np.absolute(np.asarray(L_fake)))
    meanD_ori = np.mean(np.absolute(np.asarray(L_ori)))
    message = 'W-Distance Dori %.4f Dfake %.4f' %  (meanD_ori,meanD_fake)
    print(message)
    with open(Distance_total_log, "a") as log_FID:
        log_FID.write('%s\n' % message)  # save the message
    score_ori,score_fake = fid.compute_fid(realdir, oridir,mode="clean"),fid.compute_fid(realdir, fakedir,mode="clean")
    message = 'Clean-fid Ori: %.4f Fake: %.4f' %  (score_ori,score_fake)
    print(message)
    with open(FID_log, "a") as log_FID:
        log_FID.write('%s\n' % message)  # save the message
    shutil.rmtree(FID_folder)

def evalforSelfDistriDistance(opt,epoch,total_iters,model,visualizer,Val_dataset):
    print('getting the Distance score at the end of epoch %d, iters %d' % (epoch, total_iters))
    model.eval()
    Distance_folder = os.path.join(opt.results_dir,'SelfDistri_Distance')
    Distance_csv = os.path.join(Distance_folder,'Distance_epoch_%d.csv' % epoch)
    Distance_total_log = os.path.join(Distance_folder,'W_Distance.txt')
    total_dry = 0
    Histpng = os.path.join(Distance_folder,'Hist_%d.png' % epoch)
    Histtitle = opt.name+'_Hist_%d' % epoch
    os.makedirs(Distance_folder, exist_ok=True)
    FID_log = os.path.join(Distance_folder,'FID_two_randS_distribution.txt')
    L_fake,L_ori = [],[]
    showima = False
    num_irdx = int(1e3)
    for ival, dataval in enumerate(Val_dataset):
        visualizer.reset()
        model.set_input_test(dataval)
        img_path = model.get_image_paths()
        tival_dir = os.path.join(Distance_folder,'sample_'+str(ival)+'_'+os.path.splitext(os.path.split(img_path[0])[1])[0])
        os.makedirs(tival_dir, exist_ok=True)
        stacked_A = None
        stacked_B = None
        ONEdir=os.path.join(tival_dir,'Rand_ONE')
        TWOdir=os.path.join(tival_dir,'Rand_TWO')
        os.makedirs(ONEdir, exist_ok=True)
        os.makedirs(TWOdir, exist_ok=True)
        for irdx in range(num_irdx):
            model.test_selfdistri()
            visuals = model.get_current_visuals()
            stakckofimgs = getvisuals(visuals)
            for _, fakeA, _,fakeB in zip(*stakckofimgs):
                fakeA1 = fakeA[0,:,:]* 0.5 + 0.5
                fakeB1 = fakeB[0,:,:]* 0.5 + 0.5
                save_image(fakeA1, os.path.join(ONEdir,str(irdx)+'.png'))
                save_image(fakeB1, os.path.join(TWOdir,str(irdx)+'.png'))
                fakeA1 = fakeA1.cpu().numpy()
                fakeB1 = fakeB1.cpu().numpy()
                if stacked_A is None:
                    stacked_A = np.empty((num_irdx, fakeA1.shape[0], fakeA1.shape[1]))
                    stacked_B = np.empty((num_irdx, fakeB1.shape[0], fakeB1.shape[1]))
                stacked_A[irdx] = fakeA1
                stacked_B[irdx] = fakeB1
        Rand_FID = fid.compute_fid(ONEdir, TWOdir, mode="clean")
        message = 'Clean-fid: %.4f ' %  Rand_FID
        print(message)
        with open(FID_log, "a") as log_FID:
            log_FID.write('%s\n' % message)  # save the message
        Pixel_WD_arr = Dis_pixel(stacked_A,stacked_B,'WD')
        Pixel_JS_arr = Dis_pixel(stacked_A,stacked_B,'JS')
        draw_pixel_distance(Pixel_WD_arr,'WD_sample_'+str(ival)+'_'+os.path.splitext(os.path.split(img_path[0])[1])[0],\
            os.path.join(Distance_folder,'sample_'+str(ival)+'_'+'pixelWD'+'.png'))
        draw_pixel_distance(Pixel_JS_arr,'JS_sample_'+str(ival)+'_'+os.path.splitext(os.path.split(img_path[0])[1])[0],\
            os.path.join(Distance_folder,'sample_'+str(ival)+'_'+'pixelJS'+'.png'))
        shutil.rmtree(tival_dir)

def evalforSelfDistriDistance_floodanddry(opt,epoch,total_iters,model,visualizer,Val_dataset):
    print('getting the Distance score at the end of epoch %d, iters %d' % (epoch, total_iters))
    model.eval()
    Distance_folder = os.path.join(opt.results_dir,'SelfDistri_Distance_andVH_V6U40')
    os.makedirs(Distance_folder, exist_ok=True)
    FID_log = os.path.join(Distance_folder,'FID_two_randS_distribution.txt')
    L_fake,L_ori = [],[]
    showima = False
    num_irdx = int(1e3)
    for ival, dataval in enumerate(Val_dataset):
        TPNG_name = os.path.join(Distance_folder,'sample_'+str(ival)+'_'+'pixelDisandVH'+'.png')
        if os.path.exists(TPNG_name):
            continue
        visualizer.reset()
        model.set_input_test(dataval)
        img_path = model.get_image_paths()
        tival_dir = os.path.join(Distance_folder,'sample_'+str(ival)+'_'+os.path.splitext(os.path.split(img_path[0])[1])[0])
        os.makedirs(tival_dir, exist_ok=True)
        stacked_A = None
        stacked_B = None
        ONEdir=os.path.join(tival_dir,'Rand_ONE')
        TWOdir=os.path.join(tival_dir,'Rand_TWO')
        THREEdir=os.path.join(tival_dir,'Rand_THREE')
        os.makedirs(ONEdir, exist_ok=True)
        os.makedirs(TWOdir, exist_ok=True)
        os.makedirs(THREEdir, exist_ok=True)
        real_B = None
        for irdx in range(num_irdx):
            model.test_selfdistri()
            visuals = model.get_current_visuals()
            stakckofimgs = getvisuals(visuals)
            for realA, fakeA, realB,fakeB, realC, fakeC in zip(*stakckofimgs):
                fakeA1 = fakeA[0,:,:]* 0.5 + 0.5
                fakeB1 = fakeB[0,:,:]* 0.5 + 0.5
                fakeC1 = fakeC[0,:,:]* 0.5 + 0.5
                save_image(fakeA1, os.path.join(ONEdir,str(irdx)+'.png'))
                save_image(fakeB1, os.path.join(TWOdir,str(irdx)+'.png'))
                save_image(fakeC1, os.path.join(THREEdir,str(irdx)+'.png'))
                fakeA1 = fakeA1.cpu().numpy()
                fakeB1 = fakeB1.cpu().numpy()
                fakeC1 = fakeC1.cpu().numpy()
                if stacked_A is None:
                    stacked_A = np.empty((num_irdx, fakeA1.shape[0], fakeA1.shape[1]))
                    stacked_B = np.empty((num_irdx, fakeB1.shape[0], fakeB1.shape[1]))
                    stacked_C = np.empty((num_irdx, fakeC1.shape[0], fakeC1.shape[1]))
                stacked_A[irdx] = fakeA1
                stacked_B[irdx] = fakeB1
                stacked_C[irdx] = fakeC1
                if real_B is None:
                    realA1 = realA[0,:,:]* 0.5 + 0.5
                    realB1 = realB[0,:,:]* 0.5 + 0.5
                    realC1 = realC[0,:,:]* 0.5 + 0.5
                    real_A = realA1.cpu().numpy()
                    real_B = realB1.cpu().numpy()
                    real_C = realC1.cpu().numpy()
        score_DF,score_DD = fid.compute_fid(ONEdir, TWOdir, mode="clean"),fid.compute_fid(ONEdir, THREEdir, mode="clean")
        message = 'Clean-fid DF: %.4f DD: %.4f' %  (score_DF,score_DD)
        print(message)
        with open(FID_log, "a") as log_FID:
            log_FID.write('%s\n' % message)  # save the message
        Pixel_WD_arr_DF = Dis_pixel(stacked_B,stacked_A,'WD')
        Pixel_WD_arr_DD = Dis_pixel(stacked_B,stacked_C,'WD')
        #delta_DF = Dis_pixel(stacked_A,stacked_B,'MD')
        #delta_DD = Dis_pixel(stacked_A,stacked_C,'MD')
        #np.save(Pixel_WD_arr_DF, All_input_bands)
        #np.save(Pixel_WD_arr_DF, All_input_bands)
        #Pixel_JS_arr = Dis_pixel(stacked_A,stacked_B,'JS')
        #draw_pixel_distance(Pixel_WD_arr_DF,'flood_dry_pixelWD_'+str(ival)+'_'+os.path.splitext(os.path.split(img_path[0])[1])[0],\
        #    os.path.join(Distance_folder,'flood_dry_sample_'+str(ival)+'_'+'pixelWD'+'.png'))
        #draw_pixel_distance(Pixel_WD_arr_DD,'dry_dry_pixelWD_'+str(ival)+'_'+os.path.splitext(os.path.split(img_path[0])[1])[0],\
        #    os.path.join(Distance_folder,'dry_dry_sample_'+str(ival)+'_'+'pixelWD'+'.png'))
        #draw_pixel_distance(delta_DF,'flood_dry_pixeldelta_'+str(ival)+'_'+os.path.splitext(os.path.split(img_path[0])[1])[0],\
        #    os.path.join(Distance_folder,'flood_dry_sample_'+str(ival)+'_'+'pixeldelta'+'.png'))
        #draw_pixel_distance(delta_DD,'dry_dry_pixeldelta_'+str(ival)+'_'+os.path.splitext(os.path.split(img_path[0])[1])[0],\
        #   os.path.join(Distance_folder,'dry_dry_sample_'+str(ival)+'_'+'pixeldelta'+'.png'))
        draw_pixel_distance_ALL([Pixel_WD_arr_DF,Pixel_WD_arr_DD,real_B,real_C,real_A],['Potential flood WD','Potential flood WD2','Potential flood VH','Dry2 VH','Dry1 VH'],
        str(ival)+'_'+os.path.splitext(os.path.split(img_path[0])[1])[0],TPNG_name)
        shutil.rmtree(tival_dir)

def draw_pixel_distance_ALL(arrs,subtitle,title,savepath):
    list_minmax_dis = [func(l) for l in [arrs[0],arrs[1]] for func in (np.min, np.max)]
    list_minmax_VH = [func(l) for l in [arrs[2],arrs[3],arrs[4]] for func in (np.min, np.max)]
    n_plots = len(arrs)
    n_cols = 2
    nrows = int(np.ceil(n_plots / n_cols))
    plt.figure(figsize =(10,10))
    plt.title(title)
    gs1 = gridspec.GridSpec(nrows,n_cols)
    gs1.update(wspace=0.15, hspace=0.02)
    for i in range(n_plots):
        ax = plt.subplot(gs1[i])
        img = arrs[i]
        ax = plt.gca()
        if i <=1:
            im = ax.imshow(img,interpolation='nearest', cmap='jet',vmin=min(list_minmax_dis), vmax=max(list_minmax_dis))
        else:
            im = ax.imshow(img,interpolation='nearest', cmap=plt.cm.Greys_r,vmin=min(list_minmax_VH), vmax=max(list_minmax_VH))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.03)
        plt.colorbar(im, cax=cax)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        ax.title.set_text(subtitle[i])
    plt.savefig(savepath,dpi=300,format='png',bbox_inches='tight')#
    plt.close('all')

def draw_pixel_distance(Pixel_WD_arr,title,savepath):
    plt.figure(figsize =(10,10))
    ax = plt.gca()
    im = ax.imshow(Pixel_WD_arr,interpolation='nearest', cmap='jet',vmin=-1, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.title(title)
    plt.savefig(savepath,dpi=300,format='png',bbox_inches='tight')#
    plt.close('all')

def Dis_pixel(stacked_A,stacked_B,type='WD'):
    Pixel_WD_arr = np.zeros((stacked_A.shape[1],stacked_A.shape[2]))
    for ix in range(stacked_A.shape[1]):
        for iy in range(stacked_A.shape[2]):
            if type=='WD':
                Pixel_WD_arr[ix,iy] = WD(stacked_A[:,ix,iy], stacked_B[:,ix,iy])/np.mean(stacked_A[:,ix,iy])
            elif type=='JS':
                Pixel_WD_arr[ix,iy] = JS(stacked_A[:,ix,iy], stacked_B[:,ix,iy])
            else:
                Pixel_WD_arr[ix,iy] = MD(stacked_A[:,ix,iy], stacked_B[:,ix,iy])/np.mean(stacked_A[:,ix,iy])
    return Pixel_WD_arr

def WD(a,b):
    return wasserstein_distance(a,b)

def JS(a,b):
    return distance.jensenshannon(a,b)

def MD(a,b):
    return np.mean(b)-np.mean(a)

def getvisuals(visuals):
    stakckofimgs = []
    for label, im_data in visuals.items():
        stakckofimgs.append(im_data)
    return stakckofimgs

def evalforhist(opt,epoch,total_iters,model,visualizer,Val_dataset):
    print('getting the Distance score at the end of epoch %d, iters %d' % (epoch, total_iters))
    model.eval()
    Distance_folder = os.path.join(opt.results_dir,'Test_Distance')
    Histpng = os.path.join(Distance_folder,'Hist_%d.png' % epoch)
    i_ms = 0

def keepnumeric(string):
    nstr = re.sub("[^0-9]", "", string)
    return nstr

def loopmodels(list_model_iters,total_iters,opt,model,dataset_size,visualizer):
    for total_iters in list_model_iters:
        total_iters = model.setup(opt,total_iters)
        step_iter = total_iters // opt.batch_size
        opt.epoch_count = total_iters // dataset_size if total_iters // dataset_size > 0 else opt.epoch_count
        evalforDistance(opt,opt.epoch_count,total_iters,model,visualizer,Val_dataset)

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt,'Train')  # create a dataset given opt.dataset_mode and other options
    #Val_dataset = create_dataset(opt,'val') # create a test dataset
    Val_dataset = create_dataset(opt,'all')
    dataset_size = len(dataset)
    opt.continue_train = False
    opt.isTrain = False
    opt.loadbyiter = True

    list_of_savednets = glob.glob(opt.checkpoints_dir + '/*net*')
    list_model_iters = [int(keepnumeric(os.path.split(netsiter)[1])) for netsiter in list_of_savednets]
    list_model_iters = list(set(list_model_iters))
    list_model_iters.sort()

    model = create_model(opt)
    visualizer = Visualizer_local(opt)
    #for total_iters in list_model_iters:
    #    total_iters = model.setup(opt,total_iters)
    #    step_iter = total_iters // opt.batch_size
    #    opt.epoch_count = total_iters // dataset_size if total_iters // dataset_size > 0 else opt.epoch_count
    #    evalforDistance(opt,opt.epoch_count,total_iters,model,visualizer,Val_dataset)
    total_iters = model.setup(opt,opt.load_iter)
    step_iter = total_iters // opt.batch_size
    opt.epoch_count = total_iters // dataset_size if total_iters // dataset_size > 0 else opt.epoch_count
    #evalforDistance(opt,opt.epoch_count,total_iters,model,visualizer,Val_dataset)
    #evalforSelfDistriDistance(opt,opt.epoch_count,total_iters,model,visualizer,Val_dataset)
    evalforSelfDistriDistance_floodanddry(opt,opt.epoch_count,total_iters,model,visualizer,Val_dataset)