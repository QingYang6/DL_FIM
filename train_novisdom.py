"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time, os, shutil
from options.train_options import TrainOptions
from util.visualizer_local import Visualizer_local
from util.visualizer_local import save_images_pix2pix,save_images_fakeandreal,save_images_vr2
from cleanfid import fid
from data import create_dataset
from models import create_model
from torch.utils.tensorboard import SummaryWriter
import torch
print(torch.__version__)

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

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt,'Train')  # create a dataset given opt.dataset_mode and other options
    Val_dataset = create_dataset(opt,'val') # create a test dataset 
    dataset_size = len(dataset)    # get the number of images in the dataset.
    opt.dataset_sizeTotal = dataset_size + len(Val_dataset)
    print('The number of training images = %d' % dataset_size)
    print('The number of validating images = %d' % len(Val_dataset))

    model = create_model(opt)      # create a model given opt.model and other options
    total_iters = model.setup(opt)               # regular setup: load and print networks; create schedulers
    opt.epoch_count = total_iters // dataset_size if total_iters // dataset_size > 0 else opt.epoch_count
    visualizer = Visualizer_local(opt)
    step_iter = total_iters // opt.batch_size
    writer = SummaryWriter(opt.log_dir)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            visualizer.reset()
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            step_iter += 1
            if step_iter % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            if step_iter % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data, 
                opt.name, writer, total_iters)

            #if step_iter % (opt.save_latest_freq//opt.batch_size) == 0:   # cache our latest model every <save_latest_freq> iterations
            #   print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            #    save_suffix = '%d' % total_iters if opt.save_by_iter else 'latest_%d' % total_iters
            #    model.save_networks(save_suffix)
            if opt.save_latest_freq >0:
                if step_iter % (opt.save_latest_freq//opt.batch_size) == 0:             # cache our model every <save_epoch_freq> epochs
                    PTtofolder(opt,epoch,total_iters,model,visualizer,Val_dataset)
                    print('saving the model at stepiters %d, total iters %d' % (step_iter, total_iters))
                    model.save_networks(total_iters)

            if step_iter % (opt.predict_freq//opt.batch_size) == 0:
                imgtitle = opt.name+'_val_%d' % total_iters
                predictout = os.path.join(opt.results_dir,'trainval_%d.png' % total_iters)
                model.eval()
                valdata = next(iter(Val_dataset))
                model.set_input(valdata)  # unpack data from data loader
                model.test()           # run inference
                visuals = model.get_current_visuals()# get image results  
                img_path = model.get_image_paths()
                #save_images_pix2pix(visuals, img_path, predictout,opt.valid_nrows,imgtitle)
                save_images_vr2(visuals, img_path, predictout,opt.valid_nrows,imgtitle)

            iter_data_time = time.time()
        if opt.save_latest_freq <0:
            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                PTtofolder(opt,epoch,total_iters,model,visualizer,Val_dataset)
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks(total_iters)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
    writer.flush()
    writer.close()