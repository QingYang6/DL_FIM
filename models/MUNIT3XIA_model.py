import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class MUNIT3XIAModel(BaseModel):
    """
    This class implements the MUNIT model, for learning image-to-image translation without paired data.
    using Same D and G, condition with LCC and LIA, trainning with same and random locations. Oct. 30 2022
    MUNIT paper: https://arxiv.org/abs/1804.04732
    Implemented by Qing, Nov 2022.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True,norm='AdaIn', netG='AdaINGen_IA',netD='MsImageDis',\
            use_same_D=True, use_same_G=True, CtoE = True, DEM = True, NE_MARGIN=True)  # default MUNIT did not use dropout
        if is_train:
            parser.add_argument('--style_dim', type=float, default=8)
            parser.add_argument('--n_downsample', type=float, default=2)
            parser.add_argument('--n_res', type=float, default=4)
            parser.add_argument('--activ', type=str, default='lrelu')
            parser.add_argument('--pad_type', type=str, default='reflect')
            parser.add_argument('--mlp_dim', type=float, default=256)
            parser.add_argument('--gan_w', type=float, default=1)
            parser.add_argument('--recon_x_w', type=float, default=10)
            parser.add_argument('--recon_s_w', type=float, default=1)
            parser.add_argument('--recon_c_w', type=float, default=1)
            parser.add_argument('--recon_x_cyc_w', type=float, default=10)
            parser.add_argument('--vgg_w', type=float, default=0)
            parser.add_argument('--lambda_LDS', type=float, default=1, help='weight for diversityGAN')
            parser.add_argument('--NE_MARGIN_VALUE', type=float, default=-0.25, help='weight for diversityGAN')

        return parser

    def __init__(self, opt):
        """Initialize the MUNIT class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A' , 'D_B' , 'D_a2a' , 'D_a2b', 'D_A_realsb', 'D_B_realsa',\
            'gp_A' , 'gp_B' , 'gp_a2a' , 'gp_a2b', 'gp_A_realsb', 'gp_B_realsa', \
            'gen_adv_a', 'gen_recon_x_a', 'gen_cycrecon_x_a', 'gen_recon_s_a','gen_recon_c_a','gen_vgg_a',\
            'gen_adv_b', 'gen_recon_x_b', 'gen_cycrecon_x_b','gen_recon_s_b','gen_recon_c_b','gen_vgg_b',
            'gen_recon_x_a2' , 'gen_recon_s_a2aprime' , 'gen_recon_s_a2bprime' ,\
            'gen_recon_c_a2a' , 'gen_recon_c_a2b' , \
            'gen_cycrecon_x_a2aa2' , 'gen_cycrecon_x_a2ba2','gen_adv_a2a','gen_adv_a2b',\
             'gen_adv_arealb','gen_adv_breala',\
            'DS_xa' , 'DS_xb' , 'DS_xc_sa' , 'DS_xc_sb']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        #visual_names_A = ['real_A', 'fake_B']
        #visual_names_B = ['real_B', 'fake_A']
        self.visual_names = ['real_A', 'fake_B', 'real_B']  # combine visualizations for A and B
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
            input2D = opt.input_nc-1
        else:
            if self.opt.DEM:
                input2G = opt.input_nc-1
                input2D = opt.input_nc-1
            else:
                input2G = opt.input_nc-2
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
            self.criterionCycle = torch.nn.L1Loss().to(self.device)
            self.recon_criterion = torch.nn.L1Loss().to(self.device)
            self.criterionDS = torch.nn.L1Loss(reduce=False).to(self.device)
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
            self.real_C_a = raw_AC[:,[6,7],:,:].to(self.device)
            self.real_C_b = raw_AC[:,[6,7],:,:].to(self.device)
            self.real_C_c = raw_AC[:,[6,7],:,:].to(self.device)
            self.real_IA_a = raw_AC[:,2,:,:].to(self.device)
            self.real_IA_b = raw_AC[:,5,:,:].to(self.device)
            self.real_IA_c = raw_AC[:,11,:,:].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        raw_AC = input['A' if AtoB else 'B']
        raw_BC = input['B' if AtoB else 'A']
        self.real_A = raw_AC[:,[0,1],:,:].to(self.device)
        self.real_B = raw_AC[:,[3,4],:,:].to(self.device)
        #self.real_LCC = raw_BC[:,[6,7],:,:].to(self.device)
        self.real_A2 = raw_BC[:,[0,1],:,:].to(self.device)
        if self.opt.DEM:
            self.real_C_a = raw_AC[:,[6,7,8],:,:].to(self.device)
            self.real_C_b = raw_AC[:,[6,7,8],:,:].to(self.device)
            self.real_C_a2 = raw_BC[:,[6,7,8],:,:].to(self.device)
            self.real_IA_a = raw_AC[:,2:3,:,:].to(self.device)
            self.real_IA_b = raw_AC[:,5:6,:,:].to(self.device)
            self.real_IA_c = raw_BC[:,2:3,:,:].to(self.device)
        else:
            self.real_C_a = raw_AC[:,[6,7],:,:].to(self.device)
            self.real_C_b = raw_AC[:,[6,7],:,:].to(self.device)
            self.real_C_a2 = raw_BC[:,[6,7],:,:].to(self.device)
            self.real_IA_a = raw_AC[:,2:3,:,:].to(self.device)
            self.real_IA_b = raw_AC[:,5:6,:,:].to(self.device)
            self.real_IA_c = raw_BC[:,2:3,:,:].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def sample_selfdistribution(self):
        '''Just sample cross domain'''
        #style_rand = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        self.realA_C = torch.cat((self.real_A, self.real_C_a), 1)
        self.realB_C = torch.cat((self.real_B, self.real_C_b), 1)
        self.realC_C = torch.cat((self.real_C_a2, self.real_C_c), 1)
        self.visual_names = ['real_A', 'fake_A', 'real_B', 'fake_B', 'real_C', 'fake_C'] 
        if self.opt.CtoE: 
            content_a, _ = self.netG_A.encode(self.realA_C)
            content_b, _ = self.netG_A.encode(self.realB_C)
            content_c, _ = self.netG_A.encode(self.realC_C)
        else:
            content_a, _ = self.netG_A.encode(self.real_A)
            content_b, _ = self.netG_A.encode(self.real_B)
            content_c, _ = self.netG_A.encode(self.real_C)
        s_a = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        #s_b = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        #s_c = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        self.fake_A = self.netG_A.decode(content_a, s_a)         
        self.fake_B = self.netG_A.decode(content_b, s_a)
        self.fake_C = self.netG_A.decode(content_c, s_a)

    def sample(self):
        '''Just sample cross domain'''
        style_rand = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        if self.opt.CtoE:
            self.realA_C = torch.cat((self.real_A, self.real_C_a), 1)
            self.realB_C = torch.cat((self.real_B, self.real_C_b), 1)
        else:
            self.realA_C = self.real_A
            self.realB_C = self.real_B
        condi_a = self.netG_A.encode_condi(self.real_IA_a)
        condi_b = self.netG_A.encode_condi(self.real_IA_b)
        _, style = self.netG_A.encode(self.realB_C)
        content, _ = self.netG_A.encode(self.realA_C)            
        self.fake_B = self.netG_A.decode(content, style, condi_b)
        self.real_A = self.netG_A.decode(content, style_rand, condi_a)

    def forward(self):
        self.s_a = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        self.s_b = torch.randn(self.real_B.size(0), self.opt.style_dim, 1, 1).to(self.device)
        if self.opt.CtoE:
            self.realA_C = torch.cat((self.real_A, self.real_C_a), 1)
            self.realB_C = torch.cat((self.real_B, self.real_C_b), 1)
            self.realA2_C = torch.cat((self.real_A2, self.real_C_a2), 1)
        else:
            self.realA_C = self.real_A
            self.realB_C = self.real_B
            self.realA2_C = self.real_A2
        # encode
        self.c_a, self.s_a_prime = self.netG_A.encode(self.realA_C)
        self.c_b, self.s_b_prime = self.netG_A.encode(self.realB_C)
        self.c_a2, self.s_a_prime2 = self.netG_A.encode(self.realA2_C)
        self.condi_a = self.netG_A.encode_condi(self.real_IA_a)
        self.condi_b = self.netG_A.encode_condi(self.real_IA_b)
        self.condi_c = self.netG_A.encode_condi(self.real_IA_c)
        # decode (within domain)
        self.x_a_recon = self.netG_A.decode(self.c_a, self.s_a_prime, self.condi_a)
        self.x_b_recon = self.netG_A.decode(self.c_b, self.s_b_prime, self.condi_b)
        self.x_a2_recon = self.netG_A.decode(self.c_a2, self.s_a_prime2, self.condi_c)
        # decode (cross domain)
        self.x_cb_realsa_b = self.netG_A.decode(self.c_b, self.s_a_prime, self.condi_b)
        self.x_ca_realsb_a = self.netG_A.decode(self.c_a, self.s_b_prime, self.condi_a)
        self.x_ba = self.netG_A.decode(self.c_b, self.s_a, self.condi_a)
        self.x_ab = self.netG_A.decode(self.c_a, self.s_b, self.condi_b)
        self.x_a2a = self.netG_A.decode(self.c_a2, self.s_a_prime, self.condi_c)
        self.x_a2b = self.netG_A.decode(self.c_a2, self.s_b_prime, self.condi_c)
        # encode again
        if self.opt.CtoE:
            self.x_ba_ca = torch.cat((self.x_ba, self.real_C_a), 1)
            self.x_ab_cb = torch.cat((self.x_ab, self.real_C_b), 1)
            self.x_a2a_c = torch.cat((self.x_a2a, self.real_C_a2), 1)
            self.x_a2b_c = torch.cat((self.x_a2b, self.real_C_a2), 1)
            self.x_cb_realsa_b_cb = torch.cat((self.x_cb_realsa_b, self.real_C_b), 1)
            self.x_ca_realsb_a_ca = torch.cat((self.x_ca_realsb_a, self.real_C_a), 1)
        else:
            self.x_ba_ca = self.x_ba
            self.x_ab_cb = self.x_ab
            self.x_a2a_c = self.x_a2a
            self.x_a2b_c = self.x_a2b
            self.x_cb_realsa_b_cb = self.x_cb_realsa_b
            self.x_ca_realsb_a_ca = self.x_ca_realsb_a
        self.c_b_recon, self.s_a_recon = self.netG_A.encode(self.x_ba_ca)
        self.c_a_recon, self.s_b_recon = self.netG_A.encode(self.x_ab_cb)
        self.c_a2_recon_a, self.s_a_recon_a2a = self.netG_A.encode(self.x_a2a_c)
        self.c_a2_recon_b, self.s_b_recon_a2b = self.netG_A.encode(self.x_a2b_c)
        # decode again (if needed)
        self.x_aba = self.netG_A.decode(self.c_a_recon, self.s_a_prime, self.condi_a) if self.opt.recon_x_cyc_w > 0 else None
        self.x_bab = self.netG_A.decode(self.c_b_recon, self.s_b_prime, self.condi_b) if self.opt.recon_x_cyc_w > 0 else None
        self.x_a2aa2 = self.netG_A.decode(self.c_a2_recon_a, self.s_a_prime2, self.condi_c) if self.opt.recon_x_cyc_w > 0 else None
        self.x_a2ba2 = self.netG_A.decode(self.c_a2_recon_b, self.s_a_prime2, self.condi_c) if self.opt.recon_x_cyc_w > 0 else None

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
        x_ba_a = torch.cat((self.x_ba, self.real_C_a), 1)
        x_ab_b = torch.cat((self.x_ab, self.real_C_b), 1)
        self.loss_gen_adv_a = self.criterionGAN(self.netD_A(x_ba_a),True) * self.opt.gan_w
        self.loss_gen_adv_b = self.criterionGAN(self.netD_A(x_ab_b),True) * self.opt.gan_w
        self.loss_gen_adv_arealb = self.criterionGAN(self.netD_A(self.x_ca_realsb_a_ca),True) * self.opt.gan_w
        self.loss_gen_adv_breala = self.criterionGAN(self.netD_A(self.x_cb_realsa_b_cb),True) * self.opt.gan_w
        #a2 related loss
        self.loss_gen_recon_x_a2 = self.recon_criterion(self.x_a2_recon, self.real_A2) * self.opt.recon_x_w
        self.loss_gen_recon_s_a2aprime = self.recon_criterion(self.s_a_recon_a2a, self.s_a_prime) * self.opt.recon_s_w
        self.loss_gen_recon_s_a2bprime = self.recon_criterion(self.s_b_recon_a2b, self.s_b_prime) * self.opt.recon_s_w
        self.loss_gen_recon_c_a2a = self.recon_criterion(self.c_a2_recon_a, self.c_a2) * self.opt.recon_c_w
        self.loss_gen_recon_c_a2b = self.recon_criterion(self.c_a2_recon_b, self.c_a2) * self.opt.recon_c_w
        self.loss_gen_cycrecon_x_a2aa2 = self.recon_criterion(self.x_a2aa2, self.real_A2) * self.opt.recon_x_cyc_w if self.opt.recon_x_cyc_w > 0 else None
        self.loss_gen_cycrecon_x_a2ba2 = self.recon_criterion(self.x_a2ba2, self.real_A2) * self.opt.recon_x_cyc_w if self.opt.recon_x_cyc_w > 0 else None
        x_a2a_cford = torch.cat((self.x_a2a, self.real_C_a2), 1)
        x_a2b_cford = torch.cat((self.x_a2b, self.real_C_a2), 1)
        self.loss_gen_adv_a2a = self.criterionGAN(self.netD_A(x_a2a_cford),True) * self.opt.gan_w
        self.loss_gen_adv_a2b = self.criterionGAN(self.netD_A(x_a2b_cford),True) * self.opt.gan_w
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, self.x_ba, self.real_B) if self.opt.vgg_w > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, self.x_ab, self.real_A) if self.opt.vgg_w > 0 else 0
        # Diversity sensitivity loss
        self.loss_DS_xa = self.DS_loss(self.x_a_recon,self.x_ca_realsb_a,self.s_a_prime,self.s_b_prime)
        self.loss_DS_xb = self.DS_loss(self.x_b_recon,self.x_cb_realsa_b,self.s_b_prime,self.s_a_prime)
        self.loss_DS_xc_sa = self.DS_loss(self.x_a2_recon,self.x_a2a,self.s_a_prime2,self.s_a_prime)
        self.loss_DS_xc_sb = self.DS_loss(self.x_a2_recon,self.x_a2b,self.s_a_prime2,self.s_b_prime)
        # combined loss and calculate gradients
        self.loss_G = self.loss_gen_recon_x_a + self.loss_gen_recon_x_b + \
            self.loss_gen_recon_s_a + self.loss_gen_recon_s_b + self.loss_gen_recon_c_a \
                + self.loss_gen_recon_c_b+ \
            self.loss_gen_cycrecon_x_a + self.loss_gen_cycrecon_x_b +\
            self.loss_gen_adv_a + self.loss_gen_adv_b + \
                self.loss_gen_vgg_a + self.loss_gen_vgg_b + \
            self.loss_gen_recon_x_a2 + self.loss_gen_recon_s_a2aprime + self.loss_gen_recon_s_a2bprime +\
            self.loss_gen_recon_c_a2a + self.loss_gen_recon_c_a2b + \
            self.loss_gen_cycrecon_x_a2aa2 + self.loss_gen_cycrecon_x_a2ba2 + \
            self.loss_gen_adv_a2a + self.loss_gen_adv_a2b + \
            self.loss_gen_adv_arealb + self.loss_gen_adv_breala +\
            self.loss_DS_xa + self.loss_DS_xb + self.loss_DS_xc_sa + self.loss_DS_xc_sb
        self.loss_G.backward()
    
    def DS_loss(self,img1,img2,s1,s2):
        _eps = 1.0e-5
        batch_wise_imgs_l1 = self.criterionDS(img1.detach(), img2.detach()).sum(dim=1).sum(dim=1).sum(dim=1)
        batch_wise_imgs_l1 = batch_wise_imgs_l1 / (img1.size(1)*img1.size(2)*img1.size(3))
        batch_wise_z_l1 = self.criterionDS(s1.detach(), s2.detach()).sum(dim=1)
        batch_wise_z_l1 = batch_wise_z_l1 / s1.size(1)
        loss_errNE = - (batch_wise_imgs_l1 / (batch_wise_z_l1 + _eps)).mean()
        if self.opt.NE_MARGIN:
            loss_errNE = torch.clamp(loss_errNE, max=self.opt.NE_MARGIN_VALUE).mean()* self.opt.lambda_LDS
        else:
            loss_errNE = loss_errNE.mean()* self.opt.lambda_LDS
        return loss_errNE

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
        x_ba = self.netG_A.decode(self.c_b.detach(), s_a, self.condi_a)
        x_ab = self.netG_A.decode(self.c_a.detach(), s_b, self.condi_b)
        # D_loss
        self.loss_D_A, self.loss_gp_A = self.backward_D_basic(self.netD_A, self.real_A, x_ba, self.real_C_a)
        self.loss_D_B, self.loss_gp_B = self.backward_D_basic(self.netD_A, self.real_B, x_ab, self.real_C_b)
        self.loss_D_A_realsb, self.loss_gp_A_realsb = self.backward_D_basic(self.netD_A, self.real_A, self.x_ca_realsb_a.detach(), self.real_C_a)
        self.loss_D_B_realsa, self.loss_gp_B_realsa = self.backward_D_basic(self.netD_A, self.real_B, self.x_cb_realsa_b.detach(), self.real_C_b)
        self.loss_D_a2a, self.loss_gp_a2a = self.backward_D_basic(self.netD_A, self.real_A2, self.x_a2a.detach(), self.real_C_a2)
        self.loss_D_a2b, self.loss_gp_a2b = self.backward_D_basic(self.netD_A, self.real_A2, self.x_a2b.detach(), self.real_C_a2)
        self.loss_D = self.loss_D_A + self.loss_D_B + self.loss_D_a2a + self.loss_D_a2b + self.loss_D_A_realsb + self.loss_D_B_realsa
        self.loss_gp = self.loss_gp_A + self.loss_gp_B + self.loss_gp_a2a + self.loss_gp_a2b + self.loss_gp_A_realsb + self.loss_gp_B_realsa
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