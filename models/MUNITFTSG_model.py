import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class MUNITFTSGModel(BaseModel):
    """
    This class implements the MUNIT model, for learning image-to-image translation without paired data.
    using Same D and G, condition with LCC and LIA, trainning with same and random locations. Dec. 1 2022
    MUNIT paper: https://arxiv.org/abs/1804.04732
    Implemented by Qing, NOV 2022.
    By default Conditon not for E, just for D.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(use_same_D=False, use_same_G=True, CtoE = False, DEM = True, no_dropout=True, norm='AdaIn', netG='AdaINGen_3E',netD='MsImageDis')  # default MUNIT did not use dropout
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

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['DA_xa_cascita' , 'DA_xa_casaita' , 'DA_xa_carasitc' ,\
            'DB_xc_cbscita' , 'DB_xc_cbscaitc' , 'DB_xc_cbrscitc',\
        'gpA_xa_cascita' , 'gpA_xa_casaita' , 'gpA_xa_carasitc' , \
            'gpB_xc_cascita' , 'gpB_xc_cbscaitc' , 'gpB_xc_cbrscitc',\
        'L1_xa' , 'L1_xc' , 'L1_itc' , 'L1_prisa_xarec' , \
        'L1_prisa_xcrec' , 'L1_rsa_xarec' , 'L1_prisc_xarec' , 'L1_prisc_xcrec' , \
        'L1_rsc_xcrec' , 'L1_condixa_rec' , 'L1_ita_xarec' , 'L1_ita_xcrec' ,\
        'L1_condixc_rec' , 'L1_itc_xarec' , 'L1_itc_xcrec' , 'L1_ca_recxa1' ,\
        'L1_ca_recxa2' , 'L1_ca_recxa3' , 'L1_cb_recxa1' , 'L1_cb_recxa2' ,\
        'L1_cb_recxa3' , 'secrecon_x_a' , 'G_xa_cascita' , 'G_xa_casaita' , \
        'G_xa_carasitc' , 'G_xc_cbscita' , 'G_xc_cbscaitc' , 'G_xc_cbrscitc']
        
        self.visual_names = ['real_A', 'fake_B', 'real_B']  # combine visualizations for A and B
        if self.isTrain:
            if not opt.use_same_D:
                self.model_names = ['G_A', 'D_A', 'D_B']
            else:
                self.model_names = ['G_A', 'D_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        if not opt.CtoE:
            input2G = 2
            input2D = opt.input_nc
        else:
            if self.opt.DEM:
                input2G = opt.input_nc
                input2D = opt.input_nc
            else:
                input2G = opt.input_nc-1
                input2D = opt.input_nc-1
        self.netG_A = networks.define_G(input2G, opt.output_nc, opt.ngf, opt.netG, 'none',
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,\
                                        opt.style_dim,opt.n_downsample, opt.n_res, opt.mlp_dim, activ='relu', pad_type='reflect')

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(input2D, opt.ndf, opt.netD,\
                                            opt.n_layers_D, 'none', opt.init_type, opt.init_gain, self.gpu_ids,\
                                            mask_size=128, s1=32, s2=16, activ='lrelu', num_scales=3, pad_type='reflect')
            if not opt.use_same_D:
                self.netD_B = networks.define_D(input2D, opt.ndf, opt.netD,\
                                                opt.n_layers_D, 'none', opt.init_type, opt.init_gain, self.gpu_ids,\
                                                mask_size=128, s1=32, s2=16, activ='lrelu', num_scales=3, pad_type='reflect')              

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss().to(self.device)
            self.recon_criterion = torch.nn.L1Loss().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr/4, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if not opt.use_same_D:
                self.optimizer_D2 = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D2)

    def set_input_test(self, input):
        AtoB = self.opt.direction == 'AtoB'
        raw_AC = input['A' if AtoB else 'B']
        raw_BC = input['B' if AtoB else 'A']
        self.real_A = raw_AC[:,[0,1],:,:].to(self.device)
        self.real_B = raw_BC[:,[9,10,11,12,13],:,:].to(self.device)
        self.real_C = raw_BC[:,[0,1],:,:].to(self.device)
        #self.real_LCC = raw_BC[:,[6,7],:,:].to(self.device)
        if self.opt.DEM:
            self.real_C_a = raw_AC[:,[2,6,7,8],:,:].to(self.device)
            self.real_C_b = raw_BC[:,[2,6,7,8],:,:].to(self.device)
            self.real_C_c = raw_BC[:,[2,6,7,8],:,:].to(self.device)
        else:
            self.real_C_a = raw_AC[:,[2,6,7],:,:].to(self.device)
            self.real_C_b = raw_BC[:,[2,6,7],:,:].to(self.device)
            self.real_C_c = raw_BC[:,[2,6,7],:,:].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        raw_AC = input['A' if AtoB else 'B']
        raw_BC = input['B' if AtoB else 'A']
        self.real_A = raw_AC[:,[0,1],:,:].to(self.device)
        self.real_B = raw_BC[:,[9,10,11,12,13],:,:].to(self.device)
        #self.real_LCC = raw_BC[:,[6,7],:,:].to(self.device)
        self.real_C = raw_BC[:,[0,1],:,:].to(self.device)
        if self.opt.DEM:
            self.real_C_a = raw_AC[:,[2,8],:,:].to(self.device)
            self.real_C_b = raw_BC[:,[2,8],:,:].to(self.device)
            self.real_C_c = raw_BC[:,[2,8],:,:].to(self.device)
            self.real_A_LCC = raw_AC[:,[6,7],:,:].to(self.device)
            self.real_C_LCC = raw_BC[:,[6,7],:,:].to(self.device)
        else:
            self.real_C_a = raw_AC[:,[2,6,7],:,:].to(self.device)
            self.real_C_b = raw_BC[:,[2,6,7],:,:].to(self.device)
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
            self.realC_C = torch.cat((self.real_C, self.real_C_c), 1)
        else:
            self.realA_C = self.real_A
            self.realB_C = self.real_B
            self.realC_C = self.real_C
        content_A, _,condiA = self.netG_A.encode(self.realA_C)
        self.itc = self.netG_A.enc_condi(self.real_C_c)
        _, style,condiC = self.netG_A.encode(self.realC_C)
        content = self.netG_A.enc_sec_content(self.realB_C)
        self.fake_B = self.netG_A.decode(content, style, condiC)
        self.real_A = self.netG_A.decode(content_A, style_rand, condiA)
        self.real_B = self.real_C

    def forward(self):
        self.rsa = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        self.rsc = torch.randn(self.real_C.size(0), self.opt.style_dim, 1, 1).to(self.device)
        if self.opt.CtoE:
            self.realA_C = torch.cat((self.real_A, self.real_C_a), 1)
            self.realB_C = torch.cat((self.real_B, self.real_C_b), 1)
            self.realC_C = torch.cat((self.real_C, self.real_C_c), 1)
        else:
            self.realA_C = self.real_A
            self.realB_C = self.real_B
            self.realC_C = self.real_C
        # encode
        self.c_a, self.s_a_prime, self.condi_xa = self.netG_A.encode(self.realA_C)
        self.c_b = self.netG_A.enc_sec_content(self.realB_C)
        self.c_c, self.s_c_prime, self.condi_xc = self.netG_A.encode(self.realC_C)
        self.ita = self.netG_A.enc_condi(self.real_C_a)
        self.itc = self.netG_A.enc_condi(self.real_C_c)
        # decode (within domain)
        self.x_a_recon = self.netG_A.decode(self.c_a, self.s_a_prime, self.condi_xa)
        self.x_c_recon = self.netG_A.decode(self.c_c, self.s_c_prime, self.condi_xc)
        # decode (cross domain)
        self.xa_cascita = self.netG_A.decode(self.c_a, self.s_c_prime, self.condi_xa)
        self.xa_casaita = self.netG_A.decode(self.c_a, self.s_a_prime, self.ita)
        self.xa_carasitc = self.netG_A.decode(self.c_a, self.rsa, self.itc)
        self.xc_cbscita = self.netG_A.decode(self.c_b, self.s_c_prime, self.ita)
        self.xc_cbscaitc = self.netG_A.decode(self.c_b, self.s_a_prime, self.condi_xc)
        self.xc_cbrscitc = self.netG_A.decode(self.c_b, self.rsc, self.itc)
        # encode again
        self.c_a_recon_xa1, self.s_c_prime_xarec, self.condi_xa_xarec = self.netG_A.encode(self.xa_cascita)
        self.c_a_recon_xa2, self.s_a_prime_xarec, self.ita_xarec = self.netG_A.encode(self.xa_casaita)
        self.c_a_recon_xa3, self.rsa_xarec, self.itc_xarec = self.netG_A.encode(self.xa_carasitc)
        self.c_b_recon_xc1, self.s_c_prime_xcrec, self.ita_xcrec = self.netG_A.encode(self.xc_cbscita)
        self.c_b_recon_xc2, self.s_a_prime_xcrec, self.condi_xc_xarec = self.netG_A.encode(self.xc_cbscaitc)
        self.c_b_recon_xc3, self.rsc_prime_xcrec, self.itc_xcrec = self.netG_A.encode(self.xc_cbrscitc)
        # decode again (if needed)
        self.x_a_secrecon = self.netG_A.decode(self.c_a_recon_xa1, self.s_a_prime, self.condi_xa) if self.opt.recon_x_cyc_w > 0 else None

    def backward_G(self):
        #L1 loss
        self.loss_L1_xa = self.recon_criterion(self.x_a_recon, self.real_A) * self.opt.recon_x_w
        self.loss_L1_xc = self.recon_criterion(self.x_c_recon, self.real_C) * self.opt.recon_x_w
        self.loss_L1_ita = self.recon_criterion(self.ita, self.condi_xa) * self.opt.recon_c_w
        self.loss_L1_itc = self.recon_criterion(self.itc, self.condi_xc) * self.opt.recon_c_w
        #
        self.loss_L1_prisa_xarec = self.recon_criterion(self.s_a_prime, self.s_a_prime_xarec) * self.opt.recon_s_w
        self.loss_L1_prisa_xcrec = self.recon_criterion(self.s_a_prime, self.s_a_prime_xcrec) * self.opt.recon_s_w
        self.loss_L1_rsa_xarec = self.recon_criterion(self.rsa, self.rsa_xarec) * self.opt.recon_s_w
        self.loss_L1_prisc_xarec = self.recon_criterion(self.s_c_prime, self.s_c_prime_xarec) * self.opt.recon_s_w
        self.loss_L1_prisc_xcrec = self.recon_criterion(self.s_c_prime, self.s_c_prime_xcrec) * self.opt.recon_s_w
        self.loss_L1_rsc_xcrec = self.recon_criterion(self.rsc, self.rsc_prime_xcrec) * self.opt.recon_s_w
        #
        self.loss_L1_condixa_rec = self.recon_criterion(self.condi_xa, self.condi_xa_xarec) * self.opt.recon_c_w
        self.loss_L1_ita_xarec = self.recon_criterion(self.ita, self.ita_xarec) * self.opt.recon_c_w
        self.loss_L1_ita_xcrec = self.recon_criterion(self.ita, self.ita_xcrec) * self.opt.recon_c_w
        self.loss_L1_condixc_rec = self.recon_criterion(self.condi_xc, self.condi_xc_xarec) * self.opt.recon_c_w
        self.loss_L1_itc_xarec = self.recon_criterion(self.itc, self.itc_xarec) * self.opt.recon_c_w
        self.loss_L1_itc_xcrec = self.recon_criterion(self.itc, self.itc_xcrec) * self.opt.recon_c_w
        #
        self.loss_L1_ca_recxa1 = self.recon_criterion(self.c_a, self.c_a_recon_xa1) * self.opt.recon_c_w
        self.loss_L1_ca_recxa2 = self.recon_criterion(self.c_a, self.c_a_recon_xa2) * self.opt.recon_c_w
        self.loss_L1_ca_recxa3 = self.recon_criterion(self.c_a, self.c_a_recon_xa3) * self.opt.recon_c_w
        self.loss_L1_cb_recxa1 = self.recon_criterion(self.c_b, self.c_b_recon_xc1) * self.opt.recon_c_w
        self.loss_L1_cb_recxa2 = self.recon_criterion(self.c_b, self.c_b_recon_xc2) * self.opt.recon_c_w
        self.loss_L1_cb_recxa3 = self.recon_criterion(self.c_b, self.c_b_recon_xc3) * self.opt.recon_c_w
        #
        self.loss_secrecon_x_a = self.recon_criterion(self.x_a_secrecon, self.real_A) * self.opt.recon_x_cyc_w if self.opt.recon_x_cyc_w > 0 else None
        ####
        # GAN loss
        self.loss_G_xa_cascita = self.singleGloss(self.netD_A,self.xa_cascita,self.real_A_LCC)
        self.loss_G_xa_casaita = self.singleGloss(self.netD_A,self.xa_casaita,self.real_A_LCC)
        self.loss_G_xa_carasitc = self.singleGloss(self.netD_A,self.xa_carasitc,self.real_A_LCC)
        if not self.opt.use_same_D:
            self.loss_G_xc_cbscita = self.singleGloss(self.netD_A,self.xc_cbscita,self.real_C_LCC)
            self.loss_G_xc_cbscaitc = self.singleGloss(self.netD_A,self.xc_cbscaitc,self.real_C_LCC)
            self.loss_G_xc_cbrscitc = self.singleGloss(self.netD_A,self.xc_cbrscitc,self.real_C_LCC)
        else:
            self.loss_G_xc_cbscita = self.singleGloss(self.netD_B,self.xc_cbscita,self.real_C_LCC)
            self.loss_G_xc_cbscaitc = self.singleGloss(self.netD_B,self.xc_cbscaitc,self.real_C_LCC)
            self.loss_G_xc_cbrscitc = self.singleGloss(self.netD_B,self.xc_cbrscitc,self.real_C_LCC)
        # combined loss and calculate gradients
        self.loss_G = self.loss_L1_xa + self.loss_L1_xc + self.loss_L1_itc + self.loss_L1_prisa_xarec + \
        self.loss_L1_prisa_xcrec + self.loss_L1_rsa_xarec + self.loss_L1_prisc_xarec + self.loss_L1_prisc_xcrec + \
        self.loss_L1_rsc_xcrec + self.loss_L1_condixa_rec + self.loss_L1_ita_xarec + self.loss_L1_ita_xcrec +\
        self.loss_L1_condixc_rec + self.loss_L1_itc_xarec + self.loss_L1_itc_xcrec + self.loss_L1_ca_recxa1 +\
        self.loss_L1_ca_recxa2 + self.loss_L1_ca_recxa3 + self.loss_L1_cb_recxa1 + self.loss_L1_cb_recxa2 +\
        self.loss_L1_cb_recxa3 + self.loss_secrecon_x_a + self.loss_G_xa_cascita + self.loss_G_xa_casaita + \
        self.loss_G_xa_carasitc + self.loss_G_xc_cbscita + self.loss_G_xc_cbscaitc + self.loss_G_xc_cbrscitc
        self.loss_G.backward()
    
    def singleGloss(self,netD,img,condition,TorF=True):
        in_img = torch.cat((img, condition), 1)
        Dis_in_img = netD(in_img)
        loss = self.criterionGAN(Dis_in_img, TorF)
        return loss

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
        #s_a = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        # decode (cross domain)
        #x_ars = self.netG_A.decode(self.c_a, s_a)
        # D_loss
        self.loss_DA_xa_cascita, self.loss_gpA_xa_cascita = self.backward_D_basic(self.netD_A, self.real_A, self.xa_cascita.detach(), self.real_A_LCC)
        self.loss_DA_xa_casaita, self.loss_gpA_xa_casaita = self.backward_D_basic(self.netD_A, self.real_A, self.xa_casaita.detach(), self.real_A_LCC)
        self.loss_DA_xa_carasitc, self.loss_gpA_xa_carasitc = self.backward_D_basic(self.netD_A, self.real_A, self.xa_carasitc.detach(), self.real_A_LCC)
        if not self.opt.use_same_D:
            self.loss_DB_xc_cbscita, self.loss_gpB_xc_cascita = self.backward_D_basic(self.netD_A, self.real_C, self.xc_cbscita.detach(), self.real_C_LCC)
            self.loss_DB_xc_cbscaitc, self.loss_gpB_xc_cbscaitc = self.backward_D_basic(self.netD_A, self.real_C, self.xc_cbscaitc.detach(), self.real_C_LCC)
            self.loss_DB_xc_cbrscitc, self.loss_gpB_xc_cbrscitc = self.backward_D_basic(self.netD_A, self.real_C, self.xc_cbrscitc.detach(), self.real_C_LCC) 
        else:
            self.loss_DB_xc_cbscita, self.loss_gpB_xc_cascita = self.backward_D_basic(self.netD_B, self.real_C, self.xc_cbscita.detach(), self.real_C_LCC)
            self.loss_DB_xc_cbscaitc, self.loss_gpB_xc_cbscaitc = self.backward_D_basic(self.netD_B, self.real_C, self.xc_cbscaitc.detach(), self.real_C_LCC)
            self.loss_DB_xc_cbrscitc, self.loss_gpB_xc_cbrscitc = self.backward_D_basic(self.netD_B, self.real_C, self.xc_cbrscitc.detach(), self.real_C_LCC) 
        self.loss_D = self.loss_DA_xa_cascita + self.loss_DA_xa_casaita + self.loss_DA_xa_carasitc +\
            self.loss_DB_xc_cbscita + self.loss_DB_xc_cbscaitc + self.loss_DB_xc_cbrscitc
        self.loss_gp = self.loss_gpA_xa_cascita + self.loss_gpA_xa_casaita + self.loss_gpA_xa_carasitc + \
            self.loss_gpB_xc_cascita + self.loss_gpB_xc_cbscaitc + self.loss_gpB_xc_cbrscitc
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