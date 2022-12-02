import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class MUNITUCModel(BaseModel):
    """
    This class implements the MUNIT model, for learning image-to-image translation without paired data.

    MUNIT paper: https://arxiv.org/abs/1804.04732
    Implemented by Qing, Oct 2022.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True,norm='AdaIn', netG='AdaINGen', netG_LCC='VAEGen', netD='MsImageDis')  # default MUNIT did not use dropout
        if is_train:
            parser.add_argument('--style_dim', type=float, default=8)
            parser.add_argument('--n_downsample', type=float, default=2)
            parser.add_argument('--n_res', type=float, default=4)
            parser.add_argument('--activ', type=str, default='lrelu')
            parser.add_argument('--pad_type', type=str, default='reflect')
            parser.add_argument('--mlp_dim', type=float, default=256)
            parser.add_argument('--gan_w', type=float, default=1)
            parser.add_argument('--recon_x_w', type=float, default=10) 
            parser.add_argument('--recon_uc_x_w', type=float, default=-1)
            parser.add_argument('--recon_s_w', type=float, default=5)
            parser.add_argument('--recon_c_w', type=float, default=1)
            parser.add_argument('--two_uc_w', type=float, default=0.1)
            parser.add_argument('--recon_x_cyc_w', type=float, default=10)
            parser.add_argument('--vgg_w', type=float, default=0)
            parser.add_argument('--input_nc_LCC', type=float, default=2)

        return parser

    def __init__(self, opt):
        """Initialize the MUNIT class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'D_B','D_uc_A','D_uc_B','gp_A','gp_B','gp_uc_A','gp_uc_B',\
            'gen_adv_a', 'gen_recon_x_a', 'gen_cycrecon_x_a', 'gen_recon_s_a','gen_vgg_a',\
            'gen_adv_b', 'gen_recon_x_b', 'gen_cycrecon_x_b','gen_recon_s_b','gen_vgg_b',\
            'gen_recon_uc_ba','gen_recon_uc_ab','genlcc_recon_uc','gen_adv_a_uc','gen_adv_b_uc']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_LCC', 'D_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        # define networks (both Generators and discriminators)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, 'none',
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,\
                                        opt.style_dim,opt.n_downsample, opt.n_res, opt.mlp_dim, activ='relu', pad_type='reflect')

        self.netG_LCC = networks.define_G(opt.input_nc_LCC, opt.output_nc, opt.ngf, opt.netG_LCC, 'none',
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,\
                                        opt.style_dim,opt.n_downsample, opt.n_res, opt.mlp_dim, activ='relu', pad_type='reflect')

        self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD,\
                                        opt.n_layers_D, 'none', opt.init_type, opt.init_gain, self.gpu_ids,\
                                        mask_size=128, s1=32, s2=16, activ='lrelu', num_scales=3, pad_type='reflect')

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.recon_criterion = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters()), lr=opt.lr/4, betas=(opt.beta1, 0.999))
            self.optimizer_G_LCC = torch.optim.Adam(itertools.chain(self.netG_LCC.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_G_LCC)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        raw_BC = input['B' if AtoB else 'A']
        self.real_B = raw_BC[:,[1,2],:,:].to(self.device)
        self.real_LCC = raw_BC[:,[4,5],:,:].to(self.device)
        self.real_C_a = raw_BC[:,[0,4,5],:,:].to(self.device)
        self.real_C_b = raw_BC[:,[3,4,5],:,:].to(self.device)
        #self.real_C = input['C' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def sample(self):
        '''Just sample cross domain'''
        #style_rand = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        self.realA_C = torch.cat((self.real_A, self.real_C_a), 1)
        self.realB_C = torch.cat((self.real_B, self.real_C_b), 1)
        _, style = self.netG_A.encode(self.realB_C)
        content, _ = self.netG_A.encode(self.realA_C)
        self.fake_B = self.netG_A.decode(content, style)

    def forward(self):
        self.s_a = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        self.s_b = torch.randn(self.real_B.size(0), self.opt.style_dim, 1, 1).to(self.device)
        self.realA_C = torch.cat((self.real_A, self.real_C_a), 1)
        self.realB_C = torch.cat((self.real_B, self.real_C_b), 1)
        # encode
        self.c_a, self.s_a_prime = self.netG_A.encode(self.realA_C)
        self.c_b, self.s_b_prime = self.netG_A.encode(self.realB_C)
        self.uc, _ = self.netG_LCC.encode(self.real_LCC)
        # decode (within domain)
        self.x_a_recon = self.netG_A.decode(self.c_a, self.s_a_prime)
        self.x_b_recon = self.netG_A.decode(self.c_b, self.s_b_prime)
        self.xuc_a_recon = self.netG_A.decode(self.uc, self.s_a_prime)
        self.xuc_b_recon = self.netG_A.decode(self.uc, self.s_b_prime)
        self.LCC_recon = self.netG_LCC.decode(self.uc)
        # decode (cross domain)
        self.x_ba = self.netG_A.decode(self.c_b, self.s_a)
        self.x_ab = self.netG_A.decode(self.c_a, self.s_b)
        self.x_uc_ba = self.netG_A.decode(self.uc, self.s_a)
        self.x_uc_ab = self.netG_A.decode(self.uc, self.s_b)
        # encode again
        self.x_ba_ca = torch.cat((self.x_ba, self.real_C_a), 1)
        self.x_ab_cb = torch.cat((self.x_ab, self.real_C_b), 1)
        self.c_b_recon, self.s_a_recon = self.netG_A.encode(self.x_ba_ca)
        self.c_a_recon, self.s_b_recon = self.netG_A.encode(self.x_ab_cb)
        # decode again (if needed)
        self.x_aba = self.netG_A.decode(self.c_a_recon, self.s_a_prime) if self.opt.recon_x_cyc_w > 0 else None
        self.x_bab = self.netG_A.decode(self.c_b_recon, self.s_b_prime) if self.opt.recon_x_cyc_w > 0 else None

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        #reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(self.x_a_recon, self.real_A) * self.opt.recon_x_w
        self.loss_gen_recon_x_b = self.recon_criterion(self.x_b_recon, self.real_B) * self.opt.recon_x_w
        self.loss_gen_recon_s_a = self.recon_criterion(self.s_a_recon, self.s_a) * self.opt.recon_s_w
        self.loss_gen_recon_s_b = self.recon_criterion(self.s_b_recon, self.s_b) * self.opt.recon_s_w
        self.loss_gen_recon_c_a = self.recon_criterion(self.c_a_recon, self.c_a) * self.opt.recon_c_w
        self.loss_gen_recon_c_b = self.recon_criterion(self.c_b_recon, self.c_b) * self.opt.recon_c_w
        self.loss_gen_cycrecon_x_a = self.recon_criterion(self.x_aba, self.real_A) * self.opt.recon_x_cyc_w if self.opt.recon_x_cyc_w > 0 else None
        self.loss_gen_cycrecon_x_b = self.recon_criterion(self.x_bab, self.real_B) * self.opt.recon_x_cyc_w if self.opt.recon_x_cyc_w > 0 else None
        # GAN loss
        x_ba_a = torch.cat((self.x_ba, self.real_C_b), 1)
        x_ab_b = torch.cat((self.x_ab, self.real_C_a), 1)
        self.loss_gen_adv_a = self.criterionGAN(self.netD_A(x_ba_a),True) * self.opt.gan_w
        self.loss_gen_adv_b = self.criterionGAN(self.netD_A(x_ab_b),True) * self.opt.gan_w
        #loss for LCC
        self.loss_gen_recon_uc_ba = self.recon_criterion(self.xuc_a_recon, self.real_A) * self.opt.recon_uc_x_w
        self.loss_gen_recon_uc_ab = self.recon_criterion(self.xuc_b_recon, self.real_B) * self.opt.recon_uc_x_w
        self.loss_genlcc_recon_uc = self.recon_criterion(self.LCC_recon, self.real_LCC) * self.opt.recon_x_w
        x_uc_ba_a = torch.cat((self.x_uc_ba, self.real_C_b), 1)
        x_uc_ab_b = torch.cat((self.x_uc_ab, self.real_C_a), 1)
        self.loss_gen_adv_a_uc = self.criterionGAN(self.netD_A(x_uc_ba_a),True) * self.opt.gan_w
        self.loss_gen_adv_b_uc = self.criterionGAN(self.netD_A(x_uc_ab_b),True) * self.opt.gan_w
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, self.x_ba, self.real_B) if self.opt.vgg_w > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, self.x_ab, self.real_A) if self.opt.vgg_w > 0 else 0
        # combined loss and calculate gradients
        self.loss_G = self.loss_gen_recon_x_a + self.loss_gen_recon_x_b + \
            self.loss_gen_recon_s_a + self.loss_gen_recon_s_b + self.loss_gen_recon_c_a \
                + self.loss_gen_recon_c_b+ \
            self.loss_gen_cycrecon_x_a + self.loss_gen_cycrecon_x_b +\
            self.loss_gen_adv_a + self.loss_gen_adv_b + \
                self.loss_gen_vgg_a + self.loss_gen_vgg_b +\
                    self.loss_gen_recon_uc_ba + self.loss_gen_recon_uc_ab + self.loss_genlcc_recon_uc \
                        + self.loss_gen_adv_a_uc + self.loss_gen_adv_b_uc
        self.loss_G.backward()

    def backward_D_basic(self, netD, real, fake,condi):
        real_condi = torch.cat((real, condi), 1)
        fake_condi = torch.cat((fake.detach(), condi), 1)
        #pred_real = netD(real)
        #pred_fake = netD(fake.detach())
        pred_real = netD(real_condi)
        pred_fake = netD(fake_condi)
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_gradient_penalty = 0
        # wgan-gp
        if self.opt.gan_mode == 'wgangp':
            loss_gradient_penalty, gradients = networks.cal_gradient_penalty(
                netD, real_condi, fake_condi, self.device, lambda_gp=10.0)
            #loss_gradient_penalty.backward(retain_graph=True)
            loss_D = (loss_D_fake + loss_D_real + loss_gradient_penalty) * self.opt.gan_w
        else:
            # combine loss and calculate gradients
            loss_D = (loss_D_fake + loss_D_real) * self.opt.gan_w
        #loss_D.backward()
        return loss_D, loss_gradient_penalty

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        s_a = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        s_b = torch.randn(self.real_B.size(0), self.opt.style_dim, 1, 1).to(self.device)
        # decode (cross domain)
        x_ba = self.netG_A.decode(self.c_b, s_a)
        x_ab = self.netG_A.decode(self.c_a, s_b)
        x_uc_ba = self.netG_A.decode(self.uc, self.s_a)
        x_uc_ab = self.netG_A.decode(self.uc, self.s_b)
        # D_loss
        self.loss_D_A, self.loss_gp_A = self.backward_D_basic(self.netD_A, self.real_A, x_ba, self.real_C_b)
        self.loss_D_B, self.loss_gp_B = self.backward_D_basic(self.netD_A, self.real_B, x_ab, self.real_C_a)
        self.loss_D_uc_A, self.loss_gp_uc_A = self.backward_D_basic(self.netD_A, self.real_A, x_uc_ba, self.real_C_b)
        self.loss_D_uc_B, self.loss_gp_uc_B = self.backward_D_basic(self.netD_A, self.real_B, x_uc_ab, self.real_C_a)
        self.loss_D = self.loss_D_A + self.loss_D_B + self.loss_D_uc_A + self.loss_D_uc_B
        self.loss_gp = self.loss_gp_A + self.loss_gp_B + self.loss_gp_uc_A + self.loss_gp_uc_B
        loss_Dtotal = self.loss_D + self.loss_gp
        loss_Dtotal.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A], True)  # Ds require no gradients when optimizing Gs
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A], False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights