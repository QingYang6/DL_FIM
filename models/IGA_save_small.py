import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class IGARSSSMALLModel(BaseModel):
    """
    This class implements the MUNIT model, for learning image-to-image translation without paired data.
    using Same D and G, condition with LCC and LIA, trainning with same and random locations. Oct. 30 2022
    MUNIT paper: https://arxiv.org/abs/1804.04732
    Implemented by Qing, Oct 2022.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True,norm='adain', netG='AdaINGen_IA',netD='MsImageDis',\
            use_same_D=True, use_same_G=True, CtoE = True, DEM = False)  # default MUNIT did not use dropout
        if is_train:
            parser.add_argument('--style_dim', type=float, default=8)
            parser.add_argument('--n_downsample', type=float, default=2)
            parser.add_argument('--n_res', type=float, default=1)
            parser.add_argument('--activ', type=str, default='lrelu')
            parser.add_argument('--pad_type', type=str, default='reflect')
            parser.add_argument('--mlp_dim', type=float, default=256)
            parser.add_argument('--gan_w', type=float, default=1)
            parser.add_argument('--recon_x_w', type=float, default=10)
            parser.add_argument('--recon_s_w', type=float, default=1)
            parser.add_argument('--recon_c_w', type=float, default=1)
            parser.add_argument('--recon_x_cyc_w', type=float, default=10)
            parser.add_argument('--vgg_w', type=float, default=0)

        return parser

    def __init__(self, opt):
        """Initialize the MUNIT class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_xa', 'D_xb','gp_xa' ,'gp_xb',\
            'gen_recon_ca' , 'gen_recon_cb' , 'gen_adv_rsa_xa' , 'gen_adv_rsa_xb' , \
                'gen_cycrecon_rsaxaxb']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        #visual_names_A = ['real_A', 'fake_B']
        #visual_names_B = ['real_B', 'fake_A']
        #self.visual_names = ['real_A', 'fake_B', 'real_B']  # combine visualizations for A and B
        #self.visual_names = ['real_B','fake_B','real_A', 'fake_A','real_C', 'fake_C']
        self.visual_names = ['real_B','fake_B','real_A', 'fake_A','real_C', 'fake_C','fake_C2']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            if not opt.use_same_G:
                self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
            else:
                self.model_names = ['G_A', 'D_A']
        else:  # during test time, only load Gs
            if not opt.use_same_G:
                self.model_names = ['G_A', 'G_B']
            else:
                self.model_names = ['G_A']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        if not opt.CtoE:
            input2G = 2
            input2D = opt.input_nc
        else:
            if self.opt.DEM:
                input2G = opt.input_nc
                input2D = opt.output_nc*2
            else:
                input2G = opt.input_nc-1
                input2D = opt.input_nc-2
        self.netG_A = networks.define_G(input2G, opt.output_nc, opt.ngf, opt.netG, 'none',
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,\
                                        opt.style_dim,opt.n_downsample, opt.n_res, opt.mlp_dim, activ='relu', pad_type='reflect')

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(input2D, opt.ndf, opt.netD,\
                                            opt.n_layers_D, 'none', opt.init_type, opt.init_gain, self.gpu_ids,\
                                            mask_size=128, s1=32, s2=16, activ='lrelu', num_scales=3, pad_type='reflect')

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.recon_criterion = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr/4, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input_test(self, input):
        AtoB = self.opt.direction == 'AtoB'
        raw_AC = input['A' if AtoB else 'B']
        self.real_A = raw_AC[:,[0,1],:,:].to(self.device)
        self.real_B = raw_AC[:,[3,4],:,:].to(self.device)
        self.real_C = raw_AC[:,[9,10],:,:].to(self.device)
        #self.real_LCC = raw_BC[:,[6,7],:,:].to(self.device)
        if self.opt.DEM:
            self.real_C_a = raw_AC[:,[2,6,7,8],:,:].to(self.device)
            self.real_C_b = raw_AC[:,[5,6,7,8],:,:].to(self.device)
            self.real_C_c = raw_AC[:,[11,6,7,8],:,:].to(self.device)
        else:
            self.real_C_a = raw_AC[:,[2,6,7],:,:].to(self.device)
            self.real_C_b = raw_AC[:,[5,6,7],:,:].to(self.device)
            self.real_C_c = raw_AC[:,[11,6,7],:,:].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        raw_AC = input['A' if AtoB else 'B']
        raw_BC = input['B' if AtoB else 'A']
        self.real_A = raw_AC[:,[0,1],:,:].to(self.device)
        self.real_B = raw_AC[:,[3,4],:,:].to(self.device)
        #self.real_LCC = raw_BC[:,[6,7],:,:].to(self.device)
        self.real_C = raw_BC[:,[0,1],:,:].to(self.device)
        if self.opt.DEM:
            self.real_C_a = raw_AC[:,[2,6,7,8],:,:].to(self.device)
            self.real_C_b = raw_AC[:,[5,6,7,8],:,:].to(self.device)
            self.real_C_c = raw_BC[:,[2,6,7,8],:,:].to(self.device)
        else:
            self.real_C_a = raw_AC[:,[2,6,7],:,:].to(self.device)
            self.real_C_b = raw_AC[:,[5,6,7],:,:].to(self.device)
            self.real_C_c = raw_BC[:,[2,6,7],:,:].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def sample_selfdistribution(self):
        '''Just sample cross domain'''
        #style_rand = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        self.realA_C = torch.cat((self.real_A, self.real_C_a), 1)
        self.realB_C = torch.cat((self.real_B, self.real_C_b), 1)
        self.realC_C = torch.cat((self.real_C, self.real_C_c), 1)
        self.visual_names = ['real_A', 'fake_A', 'real_B', 'fake_B', 'real_C', 'fake_C'] 
        if self.opt.CtoE: 
            content_a, _ = self.netG_A.encode(self.realA_C)
            content_b, _ = self.netG_A.encode(self.realB_C)
            content_c, _ = self.netG_A.encode(self.realC_C)
        else:
            content_a, _ = self.netG_A.encode(self.real_A)
            content_b, _ = self.netG_A.encode(self.real_B)
            content_c, _ = self.netG_A.encode(self.real_C)
        s_a = torch.randn(self.real_A.size(0), self.opt.style_dim).to(self.device)
        #s_b = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        #s_c = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        self.fake_A = self.netG_A.decode(content_a, s_a)         
        self.fake_B = self.netG_A.decode(content_b, s_a)
        self.fake_C = self.netG_A.decode(content_c, s_a)

    def sample(self):
        '''Just sample cross domain'''
        style_rand = torch.randn(self.real_A.size(0), self.opt.style_dim,1,1).to(self.device)
        if self.opt.CtoE:
            self.realA_C = torch.cat((self.real_A, self.real_C_a), 1)
            self.realB_C = torch.cat((self.real_B, self.real_C_b), 1)
            self.realC_C = torch.cat((self.real_C, self.real_C_c), 1)
        else:
            self.realA_C = self.real_A
            self.realB_C = self.real_B
            self.realC_C = self.real_C_c
        condi_c = self.netG_A.encode_condi(self.real_C_c)
        condi_b = self.netG_A.encode_condi(self.real_C_b)
        condi_a = self.netG_A.encode_condi(self.real_C_a)
        contentA, styleA = self.netG_A.encode(self.realA_C)
        contentB, styleB = self.netG_A.encode(self.realB_C) 
        contentC, styleC = self.netG_A.encode(self.realC_C)             
        self.fake_B = self.netG_A.decode(contentB, styleB, condi_b)
        #self.fake_C = self.netG_A.decode(contentC, style_rand, condi_c)
        self.fake_A = self.netG_A.decode(contentA, styleA, condi_a)
        #self.fake_C2 = self.netG_A.decode(contentC, styleB, condi_c)
        self.real_C = self.netG_A.decode(contentA, style_rand, condi_a)
        self.fake_C = self.netG_A.decode(contentB, style_rand, condi_b)
        self.fake_C2 = self.netG_A.decode(contentC, style_rand, condi_c)

    def forward(self):
        self.s_a = torch.randn(self.real_A.size(0), self.opt.style_dim,1,1).to(self.device)
        #self.s_b = torch.randn(self.real_B.size(0), self.opt.style_dim).to(self.device)
        if self.opt.CtoE:
            self.realA_C = torch.cat((self.real_A, self.real_C_a), 1)
            self.realB_C = torch.cat((self.real_B, self.real_C_b), 1)
            self.realC_C = torch.cat((self.real_C, self.real_C_c), 1)
        else:
            self.realA_C = self.real_A
            self.realB_C = self.real_B
            self.realC_C = self.real_C
        # encode
        self.c_a, self.s_a_prime = self.netG_A.encode(self.realA_C)
        self.c_b, self.s_b_prime = self.netG_A.encode(self.realB_C)
        # 
        self.ita = self.netG_A.encode_condi(self.real_C_a)
        self.itb = self.netG_A.encode_condi(self.real_C_b)
        # decode (cross domain)
        self.xa_carsaita = self.netG_A.decode(self.c_a, self.s_a, self.ita)
        self.xb_cbrsaitb = self.netG_A.decode(self.c_b, self.s_a, self.itb)
        # encode again
        if self.opt.CtoE:
            self.xa_recon_condia = torch.cat((self.xa_carsaita, self.real_C_a), 1)
            self.xb_recon_condib = torch.cat((self.xb_cbrsaitb, self.real_C_b), 1)
        else:
            self.xc_ccitcsa_condic = self.xc_ccitcsa         
        self.c_a_recon_rsa, _ = self.netG_A.encode(self.xa_recon_condia)
        self.c_b_recon_rsa, _ = self.netG_A.encode(self.xb_recon_condib)
        # decode again (if needed)
        #self.x_a_blue = self.netG_A.decode(self.c_a_recon_blue, self.s_a_prime, self.ita) if self.opt.recon_x_cyc_w > 0 else None

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        #reconstruction loss
        self.loss_gen_recon_ca = self.recon_criterion(self.c_a_recon_rsa, self.c_a) * self.opt.recon_c_w
        self.loss_gen_recon_cb = self.recon_criterion(self.c_b_recon_rsa, self.c_b) * self.opt.recon_c_w
        #
        self.loss_gen_cycrecon_rsaxaxb = self.recon_criterion(self.xa_carsaita, self.xb_cbrsaitb) * self.opt.recon_x_cyc_w if self.opt.recon_x_cyc_w > 0 else None
        # GAN loss
        rsa_xa = torch.cat((self.xa_carsaita, self.real_A), 1)
        rsa_xb = torch.cat((self.xb_cbrsaitb, self.real_B), 1)
        self.loss_gen_adv_rsa_xa = self.criterionGAN(self.netD_A(rsa_xa),True) * self.opt.gan_w
        self.loss_gen_adv_rsa_xb = self.criterionGAN(self.netD_A(rsa_xb),True) * self.opt.gan_w
        #
        #self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, self.x_ba, self.real_B) if self.opt.vgg_w > 0 else 0
        #self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, self.x_ab, self.real_A) if self.opt.vgg_w > 0 else 0
        # combined loss and calculate gradients
        self.loss_G = self.loss_gen_recon_ca + self.loss_gen_recon_cb + self.loss_gen_adv_rsa_xa + self.loss_gen_adv_rsa_xb + \
                self.loss_gen_cycrecon_rsaxaxb
        self.loss_G.backward()

    def backward_D_basic(self, netD, real, fake, condi):
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
        s_a = torch.randn(self.real_A.size(0), self.opt.style_dim,1,1).to(self.device)
        #s_b = torch.randn(self.real_B.size(0), self.opt.style_dim).to(self.device)
        # decode (cross domain)
        xa_rs = self.netG_A.decode(self.c_a, s_a, self.ita)
        xb_rs = self.netG_A.decode(self.c_b, s_a, self.ita)
        #x_ab = self.netG_A.decode(self.c_a, s_b)
        # D_loss
        self.loss_D_xa, self.loss_gp_xa = self.backward_D_basic(self.netD_A, self.real_A, xa_rs, self.real_B)
        self.loss_D_xb, self.loss_gp_xb = self.backward_D_basic(self.netD_A, self.real_B, xb_rs, self.real_A)
        self.loss_D = self.loss_D_xa + self.loss_D_xb
        self.loss_gp = self.loss_gp_xa + self.loss_gp_xb
        loss_Dtotal = self.loss_D + self.loss_gp
        loss_Dtotal.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(self.netD_A, True)
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # D_A and D_B
        self.set_requires_grad(self.netD_A, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights