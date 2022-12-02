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
    Val_dataset = create_dataset(opt,'val') # create a test dataset
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
    evalforDistance(opt,opt.epoch_count,total_iters,model,visualizer,Val_dataset)