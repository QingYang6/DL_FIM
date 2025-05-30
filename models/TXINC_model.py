import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class TXINCModel(BaseModel):
    """
    This class implements the MUNIT model, for learning image-to-image translation without paired data.
    using Same D and G, condition with LCC and LIA, trainning with same and random locations. Oct. 30 2022
    MUNIT paper: https://arxiv.org/abs/1804.04732
    Implemented by Qing, Dec 2022.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True,norm='AdaIn', netG='AdaINGen_3X2E',netD='MsImageDis',\
            use_same_D=True, use_same_G=True, CtoE = False, DEM = True, NE_MARGIN=True)  # default MUNIT did not use dropout
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
            parser.add_argument('--vgg_w', type=float, default=1)
            parser.add_argument('--lambda_LDS', type=float, default=1, help='weight for diversityGAN')
            parser.add_argument('--NE_MARGIN_VALUE', type=float, default=1, help='weight for diversityGAN')
        return parser

    def __init__(self, opt):
        """Initialize the MUNIT class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['D_xa_carsacondib' , 'D_xa_casbita' , 'D_xa_casccondia' , \
            'D_xb_cbrsbcondia' , 'D_xb_cbsaitb' , 'D_xb_cbsccondib' , \
            'D_xc_ccrsacondic' , 'D_xc_ccrsbcondic' , 'D_xc_ccsaic' , 'D_xc_ccsbic', \
        'gp_xa_carsacondib' , 'gp_xa_casbita' , 'gp_xa_casccondia' , \
            'gp_xb_cbrsbcondia' , 'gp_xb_cbsaitb' , 'gp_xb_cbsccondib' , \
            'gp_xc_ccrsacondic' , 'gp_xc_ccrsbcondic' , 'gp_xc_ccsaic' , 'gp_xc_ccsbic',\
                'gen_recon_x_a' , 'gen_recon_x_b'  , 'gen_recon_x_c' , 'L1_ita' , 'L1_itb' , 'L1_itc' ,\
        'L1_prisa_xbrec' , 'L1_prisa_xcrec' , 'L1_prisb_xarec' ,'L1_prisb_xcrec',\
        'L1_prisc_xarec' , 'L1_prisc_xbrec' ,'L1_rsa_xarec' ,'L1_rsa_xcrec' ,'L1_rsa_xcrec' ,\
        'L1_rsb_xcrec' , 'L1_condia_xarec' ,'L1_condia_xbrec' ,'L1_condia_xarec' ,\
        'L1_condic_xcrec1' , 'L1_condic_xcrec2' , 'L1_ita_xarec' ,'L1_itb_xbrec' ,\
        'L1_itc_xcrec1' , 'L1_itc_xcrec2' , 'secrecon_x_a' , 'secrecon_x_b' , 'secrecon_x_c1' , 'secrecon_x_c2' ,\
        'Icp_xa_carsacondib' , 'Icp_xa_casbita' , 'Icp_xa_casccondia' ,'Icp_xb_cbrsbcondia',\
        'Icp_xb_cbsaitb' , 'Icp_xb_cbsccondib' , 'Icp_xc_ccrsacondic' , 'Icp_xc_ccrsbcondic' ,\
        'Icp_xc_ccsaic' , 'Icp_xc_ccsbic' ,\
        'G_xa_carsacondib' , 'G_xa_casbita' , 'G_xa_casccondia' ,\
        'G_xb_cbrsbcondia' , 'G_xb_cbsaitb' , 'G_xb_cbsccondib' ,\
        'G_xc_ccrsacondic' , 'G_xc_ccrsbcondic' , 'G_xc_ccsaic' ,\
        'G_xc_ccsbic']

        #self.visual_names = ['real_A', 'fake_B', 'real_B']  # combine visualizations for A and B
        self.visual_names = ['real_B','fake_B','real_A', 'fake_A','real_C', 'fake_C']
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
        if not opt.CtoE:
            input2G = 2
            input2D = opt.input_nc
        else:
            if self.opt.DEM:
                input2G = 2
                input2D = opt.input_nc
            else:
                input2G = 2
                input2D = opt.input_nc-1
        self.netG_A = networks.define_G(input2G, opt.output_nc, opt.ngf, opt.netG, 'none',
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,\
                                        opt.style_dim,opt.n_downsample, opt.n_res, opt.mlp_dim, activ='relu', pad_type='reflect')
        self.netIcp = networks.feature_extractor('Custom_VGG', device = self.device)

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
            #self.real_IA_a = raw_AC[:,2:3,:,:].to(self.device)
            #self.real_IA_b = raw_AC[:,5:6,:,:].to(self.device)
            #self.real_IA_c = raw_BC[:,2:3,:,:].to(self.device)
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
        style_rand = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        self.realA_C = self.real_A
        self.realB_C = self.real_B
        self.realC_C = self.real_C
        self.visual_names = ['real_A', 'fake_A', 'real_B', 'fake_B', 'real_C', 'fake_C'] 
        #itc = self.netG_A.encode_condi(self.real_C_c)
        #itb = self.netG_A.encode_condi(self.real_C_b)
        ita = self.netG_A.encode_condi(self.real_C_a)
        contentA, _, condi_a = self.netG_A.encode(self.realA_C)
        contentB, _, condi_b = self.netG_A.encode(self.realB_C) 
        contentC, _, condi_c = self.netG_A.encode(self.realC_C)             
        self.fake_B = self.netG_A.decode(contentA, style_rand, ita)
        self.fake_C = self.netG_A.decode(contentC, style_rand, ita)
        self.fake_A = self.netG_A.decode(contentB, style_rand, ita)

    def sample(self):
        '''Just sample cross domain'''
        style_rand = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        if self.opt.CtoE:
            self.realA_C = torch.cat((self.real_A, self.real_C_a), 1)
            self.realB_C = torch.cat((self.real_B, self.real_C_b), 1)
        else:
            self.realA_C = self.real_A
            self.realB_C = self.real_B
            self.realC_C = self.real_C
        condi_c = self.netG_A.encode_condi(self.real_C_c)
        condi_b = self.netG_A.encode_condi(self.real_C_b)
        condi_a = self.netG_A.encode_condi(self.real_C_a)
        contentA, styleA, _ = self.netG_A.encode(self.realA_C)
        contentB, styleB, _ = self.netG_A.encode(self.realB_C) 
        contentC, _, _ = self.netG_A.encode(self.realC_C)             
        self.fake_B = self.netG_A.decode(contentA, styleB, condi_b)
        self.fake_C = self.netG_A.decode(contentC, style_rand, condi_c)
        self.fake_A = self.netG_A.decode(contentB, styleA, condi_a)

    def forward(self):
        self.rsa = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        self.rsb = torch.randn(self.real_B.size(0), self.opt.style_dim, 1, 1).to(self.device)
        if self.opt.CtoE:
            self.realA_C = torch.cat((self.real_A, self.real_C_a), 1)
            self.realB_C = torch.cat((self.real_B, self.real_C_b), 1)
            self.realC_C = torch.cat((self.real_C, self.real_C_c), 1)
        else:
            self.realA_C = self.real_A
            self.realB_C = self.real_B
            self.realC_C = self.real_C
        # encode
        self.c_a, self.s_a_prime, self.condi_a = self.netG_A.encode(self.realA_C)
        self.c_b, self.s_b_prime, self.condi_b = self.netG_A.encode(self.realB_C)
        self.c_c, self.s_c_prime, self.condi_c = self.netG_A.encode(self.realC_C)
        self.ita = self.netG_A.encode_condi(self.real_C_a)
        self.itb = self.netG_A.encode_condi(self.real_C_b)
        self.itc = self.netG_A.encode_condi(self.real_C_c)
        # decode (within domain)
        self.xa_recon = self.netG_A.decode(self.c_a, self.s_a_prime, self.condi_a)
        self.xb_recon = self.netG_A.decode(self.c_b, self.s_b_prime, self.condi_b)
        self.xc_recon = self.netG_A.decode(self.c_c, self.s_c_prime, self.condi_c)
        # decode (cross domain)
        self.xa_carsacondib = self.netG_A.decode(self.c_a, self.rsa, self.condi_b)
        self.xa_casbita = self.netG_A.decode(self.c_a, self.s_b_prime, self.ita)
        self.xa_casccondia = self.netG_A.decode(self.c_a, self.s_c_prime, self.condi_a)
        #
        self.xb_cbrsbcondia = self.netG_A.decode(self.c_b, self.rsb, self.condi_a)
        self.xb_cbsaitb = self.netG_A.decode(self.c_b, self.s_a_prime, self.itb)
        self.xb_cbsccondib = self.netG_A.decode(self.c_b, self.s_c_prime, self.condi_b)
        #
        self.xc_ccrsacondic = self.netG_A.decode(self.c_c, self.rsa, self.condi_c)
        self.xc_ccrsbcondic = self.netG_A.decode(self.c_c, self.rsb, self.condi_c)
        self.xc_ccsaic = self.netG_A.decode(self.c_c, self.s_a_prime, self.itc)
        self.xc_ccsbic = self.netG_A.decode(self.c_c, self.s_b_prime, self.itc)
        #
        self.xa_cbsaita = self.netG_A.decode(self.c_b, self.s_a_prime, self.ita)
        self.xb_casbitb = self.netG_A.decode(self.c_a, self.s_b_prime, self.itb)
        # encode again
        self.ca_recon_xa1, self.rsa_xarec, self.condib_xarec = self.netG_A.encode(self.xa_carsacondib)
        self.ca_recon_xa2, self.sb_xarec, self.ita_xarec = self.netG_A.encode(self.xa_casbita)
        self.ca_recon_xa3, self.sc_xarec, self.condia_xarec = self.netG_A.encode(self.xa_casccondia)
        #
        self.cb_recon_xb1, self.rsb_xbrec, self.condia_xbrec = self.netG_A.encode(self.xb_cbrsbcondia)
        self.cb_recon_xb2, self.sa_xbrec, self.itb_xbrec = self.netG_A.encode(self.xb_cbsaitb)
        self.cb_recon_xb3, self.sc_xbrec, self.condib_xbrec = self.netG_A.encode(self.xb_cbsccondib)
        #
        self.cc_recon_xc1, self.rsa_xcrec, self.condic_xcrec1 = self.netG_A.encode(self.xc_ccrsacondic)
        self.cc_recon_xc2, self.rab_xcrec, self.condic_xcrec2 = self.netG_A.encode(self.xc_ccrsbcondic)
        self.cc_recon_xc3, self.sa_xcrec, self.itc_xcrec1 = self.netG_A.encode(self.xc_ccsaic)
        self.cc_recon_xc4, self.sb_xcrec, self.itc_xcrec2 = self.netG_A.encode(self.xc_ccsbic)
        # decode again (if needed)
        self.xa_secrecon = self.netG_A.decode(self.ca_recon_xa1, self.s_a_prime, self.condi_a) if self.opt.recon_x_cyc_w > 0 else None
        self.xb_secrecon = self.netG_A.decode(self.cb_recon_xb1, self.s_b_prime, self.condi_b) if self.opt.recon_x_cyc_w > 0 else None
        self.xc_secrecon1 = self.netG_A.decode(self.cc_recon_xc1, self.s_c_prime, self.condi_c) if self.opt.recon_x_cyc_w > 0 else None
        self.xc_secrecon2 = self.netG_A.decode(self.cc_recon_xc2, self.s_c_prime, self.condi_c) if self.opt.recon_x_cyc_w > 0 else None

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        #reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(self.xa_recon, self.real_A) * self.opt.recon_x_w
        self.loss_gen_recon_x_b = self.recon_criterion(self.xb_recon, self.real_B) * self.opt.recon_x_w
        self.loss_gen_recon_x_c = self.recon_criterion(self.xc_recon, self.real_C) * self.opt.recon_x_w
        self.loss_L1_ita = self.recon_criterion(self.ita, self.condi_a) * self.opt.recon_c_w
        self.loss_L1_itb = self.recon_criterion(self.itb, self.condi_b) * self.opt.recon_c_w
        self.loss_L1_itc = self.recon_criterion(self.itc, self.condi_c) * self.opt.recon_c_w
        #
        self.loss_L1_prisa_xbrec = self.recon_criterion(self.s_a_prime, self.sa_xbrec) * self.opt.recon_s_w
        self.loss_L1_prisa_xcrec = self.recon_criterion(self.s_a_prime, self.sa_xcrec) * self.opt.recon_s_w
        self.loss_L1_prisb_xarec = self.recon_criterion(self.s_b_prime, self.sb_xarec) * self.opt.recon_s_w
        self.loss_L1_prisb_xcrec = self.recon_criterion(self.s_b_prime, self.sb_xcrec) * self.opt.recon_s_w
        self.loss_L1_prisc_xarec = self.recon_criterion(self.s_c_prime, self.sc_xarec) * self.opt.recon_s_w
        self.loss_L1_prisc_xbrec = self.recon_criterion(self.s_c_prime, self.sc_xbrec) * self.opt.recon_s_w
        self.loss_L1_rsa_xarec = self.recon_criterion(self.rsa, self.rsa_xarec) * self.opt.recon_s_w
        self.loss_L1_rsa_xcrec = self.recon_criterion(self.rsa, self.rsa_xcrec) * self.opt.recon_s_w
        self.loss_L1_rsb_xbrec = self.recon_criterion(self.rsb, self.rsb_xbrec) * self.opt.recon_s_w
        self.loss_L1_rsb_xcrec = self.recon_criterion(self.rsb, self.rab_xcrec) * self.opt.recon_s_w
        #
        self.loss_L1_condia_xarec = self.recon_criterion(self.condi_a, self.condia_xarec) * self.opt.recon_c_w
        self.loss_L1_condia_xbrec = self.recon_criterion(self.condi_a, self.condia_xbrec) * self.opt.recon_c_w
        self.loss_L1_condib_xarec = self.recon_criterion(self.condi_b, self.condib_xarec) * self.opt.recon_c_w
        self.loss_L1_condib_xbrec = self.recon_criterion(self.condi_b, self.condib_xbrec) * self.opt.recon_c_w
        self.loss_L1_condic_xcrec1 = self.recon_criterion(self.condi_c, self.condic_xcrec1) * self.opt.recon_c_w
        self.loss_L1_condic_xcrec2 = self.recon_criterion(self.condi_c, self.condic_xcrec2) * self.opt.recon_c_w
        self.loss_L1_ita_xarec = self.recon_criterion(self.ita, self.ita_xarec) * self.opt.recon_c_w
        self.loss_L1_itb_xbrec = self.recon_criterion(self.itb, self.itb_xbrec) * self.opt.recon_c_w
        self.loss_L1_itc_xcrec1 = self.recon_criterion(self.itc, self.itc_xcrec1) * self.opt.recon_c_w
        self.loss_L1_itc_xcrec2 = self.recon_criterion(self.itc, self.itc_xcrec2) * self.opt.recon_c_w
        #
        self.loss_secrecon_x_a = self.recon_criterion(self.xa_secrecon, self.real_A) * self.opt.recon_x_cyc_w if self.opt.recon_x_cyc_w > 0 else None
        self.loss_secrecon_x_b = self.recon_criterion(self.xb_secrecon, self.real_B) * self.opt.recon_x_cyc_w if self.opt.recon_x_cyc_w > 0 else None
        self.loss_secrecon_x_c1 = self.recon_criterion(self.xc_secrecon1, self.real_C) * self.opt.recon_x_cyc_w if self.opt.recon_x_cyc_w > 0 else None
        self.loss_secrecon_x_c2 = self.recon_criterion(self.xc_secrecon2, self.real_C) * self.opt.recon_x_cyc_w if self.opt.recon_x_cyc_w > 0 else None
        # Inc loss(vgg loss) supplements to GAN loss
        self.loss_Icp_xa_carsacondib = self.InceptionLoss(self.netIcp,self.xa_carsacondib,self.real_A) * self.opt.vgg_w if self.opt.vgg_w > 0 else None
        self.loss_Icp_xa_casbita = self.InceptionLoss(self.netIcp,self.xa_casbita,self.real_A) * self.opt.vgg_w if self.opt.vgg_w > 0 else None
        self.loss_Icp_xa_casccondia = self.InceptionLoss(self.netIcp,self.xa_casccondia,self.real_A) * self.opt.vgg_w if self.opt.vgg_w > 0 else None
        self.loss_Icp_xb_cbrsbcondia = self.InceptionLoss(self.netIcp,self.xb_cbrsbcondia,self.real_B) * self.opt.vgg_w if self.opt.vgg_w > 0 else None
        self.loss_Icp_xb_cbsaitb = self.InceptionLoss(self.netIcp,self.xb_cbsaitb,self.real_B) * self.opt.vgg_w if self.opt.vgg_w > 0 else None
        self.loss_Icp_xb_cbsccondib = self.InceptionLoss(self.netIcp,self.xb_cbsccondib,self.real_B) * self.opt.vgg_w if self.opt.vgg_w > 0 else None
        self.loss_Icp_xc_ccrsacondic = self.InceptionLoss(self.netIcp,self.xc_ccrsacondic,self.real_C) * self.opt.vgg_w if self.opt.vgg_w > 0 else None
        self.loss_Icp_xc_ccrsbcondic = self.InceptionLoss(self.netIcp,self.xc_ccrsbcondic,self.real_C) * self.opt.vgg_w if self.opt.vgg_w > 0 else None
        self.loss_Icp_xc_ccsaic = self.InceptionLoss(self.netIcp,self.xc_ccsaic,self.real_C) * self.opt.vgg_w if self.opt.vgg_w > 0 else None
        self.loss_Icp_xc_ccsbic = self.InceptionLoss(self.netIcp,self.xc_ccsbic,self.real_C) * self.opt.vgg_w if self.opt.vgg_w > 0 else None
        # Diversity sensitivity loss
        #self.loss_DS_xa = self.DS_loss(self.xa_casbita,self.xa_casccondia,self.s_b_prime,self.s_c_prime)
        #self.loss_DS_xb = self.DS_loss(self.xb_cbsaitb,self.xb_cbsccondib,self.s_a_prime,self.s_c_prime)
        #self.loss_DS_xc_rsab = self.DS_loss(self.xc_ccrsacondic,self.xc_ccrsbcondic,self.rsa,self.rsb)
        #self.loss_DS_xc_sab = self.DS_loss(self.xc_ccsaic,self.xc_ccsbic,self.s_a_prime,self.s_b_prime)
        # GAN loss
        self.loss_G_xa_carsacondib = self.singleGloss(self.netD_A,self.xa_carsacondib,self.real_C_a)
        self.loss_G_xa_casbita = self.singleGloss(self.netD_A,self.xa_casbita,self.real_C_a)
        self.loss_G_xa_casccondia = self.singleGloss(self.netD_A,self.xa_casccondia,self.real_C_a)
        #
        self.loss_G_xb_cbrsbcondia = self.singleGloss(self.netD_A,self.xb_cbrsbcondia,self.real_C_b)
        self.loss_G_xb_cbsaitb= self.singleGloss(self.netD_A,self.xb_cbsaitb,self.real_C_b)
        self.loss_G_xb_cbsccondib = self.singleGloss(self.netD_A,self.xb_cbsccondib,self.real_C_b)
        #
        self.loss_G_xc_ccrsacondic = self.singleGloss(self.netD_A,self.xc_ccrsacondic,self.real_C_c)
        self.loss_G_xc_ccrsbcondic= self.singleGloss(self.netD_A,self.xc_ccrsbcondic,self.real_C_c)
        self.loss_G_xc_ccsaic = self.singleGloss(self.netD_A,self.xc_ccsaic,self.real_C_c)
        self.loss_G_xc_ccsbic = self.singleGloss(self.netD_A,self.xc_ccsbic,self.real_C_c)
        # combined loss and calculate gradients
        self.loss_G = self.loss_gen_recon_x_a + self.loss_gen_recon_x_b  + self.loss_gen_recon_x_c + self.loss_L1_ita + self.loss_L1_itb + self.loss_L1_itc +\
        self.loss_L1_prisa_xbrec + self.loss_L1_prisa_xcrec + self.loss_L1_prisb_xarec +self.loss_L1_prisb_xcrec+\
        self.loss_L1_prisc_xarec + self.loss_L1_prisc_xbrec +self.loss_L1_rsa_xarec +self.loss_L1_rsa_xcrec +self.loss_L1_rsa_xcrec +\
        self.loss_L1_rsb_xcrec + self.loss_L1_condia_xarec +self.loss_L1_condia_xbrec +self.loss_L1_condia_xarec +\
        self.loss_L1_condic_xcrec1 + self.loss_L1_condic_xcrec2 + self.loss_L1_ita_xarec +self.loss_L1_itb_xbrec +\
        self.loss_L1_itc_xcrec1 + self.loss_L1_itc_xcrec2 + self.loss_secrecon_x_a + self.loss_secrecon_x_b + self.loss_secrecon_x_c1 + self.loss_secrecon_x_c2 +\
        self.loss_Icp_xa_carsacondib + self.loss_Icp_xa_casbita + self.loss_Icp_xa_casccondia +self.loss_Icp_xb_cbrsbcondia+\
        self.loss_Icp_xb_cbsaitb + self.loss_Icp_xb_cbsccondib + self.loss_Icp_xc_ccrsacondic + self.loss_Icp_xc_ccrsbcondic +\
        self.loss_Icp_xc_ccsaic + self.loss_Icp_xc_ccsbic + \
        self.loss_G_xa_carsacondib + self.loss_G_xa_casbita + self.loss_G_xa_casccondia +\
        self.loss_G_xb_cbrsbcondia + self.loss_G_xb_cbsaitb + self.loss_G_xb_cbsccondib +\
        self.loss_G_xc_ccrsacondic + self.loss_G_xc_ccrsbcondic + self.loss_G_xc_ccsaic +\
        self.loss_G_xc_ccsbic 
        self.loss_G.backward()

    def singleGloss(self,netD,img,condition,TorF=True):
        in_img = torch.cat((img, condition), 1)
        Dis_in_img = netD(in_img)
        loss = self.criterionGAN(Dis_in_img, TorF) * self.opt.gan_w
        return loss
    
    def T3C(self,img):
        return torch.cat((img, img[:,0:1,:,:]), 1)

    def InceptionLoss(self,netICP,real,fake):
        #with torch.no_grad():
        loss = self.criterionCycle(netICP(self.T3C(real)),netICP(self.T3C(fake)))
        return loss
    
    def DS_loss(self,img1,img2,s1,s2):
        _eps = 1.0e-5
        batch_wise_imgs_l1 = self.criterionDS(img1.detach(), img2.detach()).sum(dim=1).sum(dim=1).sum(dim=1)
        batch_wise_imgs_l1 = batch_wise_imgs_l1 / (img1.size(1)*img1.size(2)*img1.size(3))
        batch_wise_z_l1 = self.criterionDS(s1.detach(), s2.detach()).sum(dim=1)
        batch_wise_z_l1 = batch_wise_z_l1 / s1.size(1)
        loss_errNE = (batch_wise_imgs_l1 / (batch_wise_z_l1 + _eps)).mean()
        if self.opt.NE_MARGIN:
            loss_errNE_final = - torch.clamp(loss_errNE, max=self.opt.NE_MARGIN_VALUE).mean()* self.opt.lambda_LDS
        else:
            loss_errNE_final = - loss_errNE.mean()* self.opt.lambda_LDS
        return loss_errNE_final

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
        #s_b = torch.randn(self.real_B.size(0), self.opt.style_dim, 1, 1).to(self.device)
        # decode (cross domain)
        #x_ba = self.netG_A.decode(self.c_b.detach(), s_a, self.condi_a)
        #x_ab = self.netG_A.decode(self.c_a.detach(), s_b, self.condi_b)
        # D_loss
        self.loss_D_xa_carsacondib, self.loss_gp_xa_carsacondib = self.backward_D_basic(self.netD_A, self.real_A, self.xa_carsacondib.detach(), self.real_C_a)
        self.loss_D_xa_casbita, self.loss_gp_xa_casbita = self.backward_D_basic(self.netD_A, self.real_A, self.xa_casbita.detach(), self.real_C_a)
        self.loss_D_xa_casccondia, self.loss_gp_xa_casccondia = self.backward_D_basic(self.netD_A, self.real_A, self.xa_casccondia.detach(), self.real_C_a)
        #
        self.loss_D_xb_cbrsbcondia, self.loss_gp_xb_cbrsbcondia = self.backward_D_basic(self.netD_A, self.real_B, self.xb_cbrsbcondia.detach(), self.real_C_b)
        self.loss_D_xb_cbsaitb, self.loss_gp_xb_cbsaitb = self.backward_D_basic(self.netD_A, self.real_B, self.xb_cbsaitb.detach(), self.real_C_b)
        self.loss_D_xb_cbsccondib, self.loss_gp_xb_cbsccondib = self.backward_D_basic(self.netD_A, self.real_B, self.xb_cbsccondib.detach(), self.real_C_b)
        #
        self.loss_D_xc_ccrsacondic, self.loss_gp_xc_ccrsacondic = self.backward_D_basic(self.netD_A, self.real_C, self.xc_ccrsacondic.detach(), self.real_C_c)
        self.loss_D_xc_ccrsbcondic, self.loss_gp_xc_ccrsbcondic = self.backward_D_basic(self.netD_A, self.real_C, self.xc_ccrsbcondic.detach(), self.real_C_c)
        self.loss_D_xc_ccsaic, self.loss_gp_xc_ccsaic = self.backward_D_basic(self.netD_A, self.real_C, self.xc_ccsaic.detach(), self.real_C_c)
        self.loss_D_xc_ccsbic, self.loss_gp_xc_ccsbic = self.backward_D_basic(self.netD_A, self.real_C, self.xc_ccsaic.detach(), self.real_C_c)
        #
        self.loss_D = self.loss_D_xa_carsacondib + self.loss_D_xa_casbita + self.loss_D_xa_casccondia + \
            self.loss_D_xb_cbrsbcondia + self.loss_D_xb_cbsaitb + self.loss_D_xb_cbsccondib + \
            self.loss_D_xc_ccrsacondic + self.loss_D_xc_ccrsbcondic + self.loss_D_xc_ccsaic + self.loss_D_xc_ccsbic
        self.loss_gp = self.loss_gp_xa_carsacondib + self.loss_gp_xa_casbita + self.loss_gp_xa_casccondia + \
            self.loss_gp_xb_cbrsbcondia + self.loss_gp_xb_cbsaitb + self.loss_gp_xb_cbsccondib + \
            self.loss_gp_xc_ccrsacondic + self.loss_gp_xc_ccrsbcondic + self.loss_gp_xc_ccsaic + self.loss_gp_xc_ccsbic
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