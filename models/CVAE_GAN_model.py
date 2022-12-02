import torch
from .base_model import BaseModel
from . import networks
import itertools
from data import aux_dataset
import numpy as np
from torch.autograd import Variable


class CVAEGANModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        # not neccesary the same Q 07162022
        parser.set_defaults(norm='batch', netG='unet_256_concat', netCLS='AlexNet', netE='resnet18_Cin', netD='DCGAN',
        fea_r = 0.0005)
        parser.add_argument('--mask_size', type=int, default=256)
        if is_train:
            # parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--s1', type=int, default=64)
            parser.add_argument('--s2', type=int, default=32)

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names =  ['G','G_MSE','G_feaD', 'G_feaC', 'G_Dfea','G_Cfea','kl','D','CLS']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B_p', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'E', 'CLS']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_VAEG(opt.input_nc+2, opt.mask_size, opt.latent_dim, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, 
                                          self.gpu_ids, opt.mask_size, opt.s1, opt.s2)
            self.netCLS = networks.define_CLS(opt.input_nc + opt.output_nc, opt.num_class, opt.ndf, opt.netCLS,
                                opt.norm, opt.init_type, opt.init_gain, 
                                self.gpu_ids)
            self.netE = networks.define_E(opt.input_nc + opt.output_nc, opt.latent_dim, opt.num_class, opt.ndf, opt.netE,
                                opt.norm, opt.init_type, opt.init_gain, 
                                self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionVanilla = networks.GANLoss('vanilla').to(self.device)
            self.criterionLS = networks.GANLoss('lsgan').to(self.device)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionCE = torch.nn.CrossEntropyLoss().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_CLS = torch.optim.Adam(self.netCLS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)
            self.optimizers.append(self.optimizer_CLS)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.attn_A_index = input['DX']
        self.sampled_z = torch.from_numpy(np.random.normal(0, 1, (self.real_A.size(0), self.opt.latent_dim))).float().to(self.device)
    
    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2).to(self.device)
        sampled_z = torch.from_numpy(np.random.normal(0, 1, (mu.size(0), self.opt.latent_dim))).float().to(self.device)
        z = sampled_z * std + mu
        return z

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B_p = self.netG(self.real_A, self.sampled_z, self.real_C)

    def backward_CLS(self):
        self.loss_CLS = self.criterionL2(torch.reshape(self.real_C,(self.real_C.size(0), 1)), self.c_r)
        self.loss_CLS = Variable(self.loss_CLS, requires_grad = True)
        self.loss_CLS.backward()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        self.y_f, _ = self.netD(self.fake_B.detach())
        self.y_p, _ = self.netD(self.fake_B_p.detach())
        self.loss_D_r = self.criterionVanilla(self.y_r, True)
        self.loss_D_yf = self.criterionVanilla(self.y_f, False)
        self.loss_D_yp = self.criterionVanilla(self.y_p, False)
        # combine loss and calculate gradients
        self.loss_D = torch.mean(self.loss_D_r + self.loss_D_yf + self.loss_D_yp)
        self.loss_D.backward()

    def backward_GE(self):
        """G&E related losses"""
        #reconstruction
        mu, logvar = self.netE(torch.cat((self.real_B, self.real_A), 1))
        self.encoded_z = self.reparameterization(mu, logvar)
        self.fake_B = self.netG(self.real_A, self.encoded_z, self.real_C)
        #from cls
        self.c_r, self.f_C_x_r = self.netCLS(torch.cat((self.real_B, self.real_A), 1))
        _, self.f_C_x_f = self.netCLS(torch.cat((self.fake_B, self.real_A), 1))
        _, self.f_C_x_p = self.netCLS(torch.cat((self.fake_B_p, self.real_A), 1)) 
        #from dis
        self.y_r, self.f_D_x_r = self.netD(self.real_B)
        _, self.f_D_x_f = self.netD(self.fake_B)
        _, self.f_D_x_p = self.netD(self.fake_B_p)
        #Loss part1 L2
        self.loss_G_MSE = self.criterionL2(self.real_B,self.fake_B)
        self.loss_G_feaD = self.criterionL2(self.f_D_x_r,self.f_D_x_f)
        self.loss_G_feaC = self.criterionL2(self.f_C_x_r,self.f_C_x_f)
        #Loss part2 feature matching
        self.loss_G_Dfea = self.criterionL2(torch.mean(self.f_D_x_r,0),torch.mean(self.f_D_x_p,0))
        self.loss_G_Cfea = self.criterionL2(torch.mean(self.f_C_x_r,0),torch.mean(self.f_C_x_p,0))
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_MSE + self.loss_G_feaD + self.loss_G_feaC \
            + self.opt.fea_r*self.loss_G_Dfea + self.opt.fea_r*self.loss_G_Cfea
        self.loss_G.backward(retain_graph=True)
        self.optimizer_G.step()
        # loss KL
        self.loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
        self.loss_kl.backward()
        self.optimizer_E.step()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.set_requires_grad([self.netD,self.netCLS], False)
        self.set_requires_grad([self.netG,self.netE], True)
        self.optimizer_G.zero_grad()
        self.optimizer_E.zero_grad()          
        self.backward_GE()           
        # update D
        self.set_requires_grad([self.netG,self.netE,self.netCLS], False)
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update C
        self.set_requires_grad([self.netG,self.netE,self.netD], False)
        self.set_requires_grad(self.netCLS, True)  
        self.optimizer_CLS.zero_grad()        
        self.backward_CLS()                   
        self.optimizer_CLS.step()             