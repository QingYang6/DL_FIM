import torch
from .base_model import BaseModel
from . import network_bicycle as networks


class DSBicycleModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(conditional_D=True, use_same_D=False, DS_on=True, NE_MARGIN=True,norm='instance', 
            netG='unet_256',netD='basic_256_multi', netD2 = 'basic_256_multi',
        netE = 'resnet_256', ngf='64', ndf='64')
        parser.add_argument('--num_Ds', type=int, default=2, help='number of Discrminators')
        parser.add_argument('--nz', type=int, default=8, help='#latent vector')
        parser.add_argument('--nef', type=int, default=64, help='# of encoder filters in the first conv layer')
        parser.add_argument('--upsample', type=str, default='basic', help='basic | bilinear')
        parser.add_argument('--nl', type=str, default='relu', help='non-linearity activation: relu | lrelu | elu')
        parser.add_argument('--no_encode', action='store_true', help='use the Encoder or not')
        parser.add_argument('--where_add', type=str, default='all', help='input|all|middle; where to add z in the network G')
        #parser.add_argument('--conditional_D', action='store_true', help='if use conditional GAN for D')
        #parser.add_argument('--use_same_D', action='store_true', help='if two Ds share the weights or not')
        parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for |B-G(A, E(B))|')
            parser.add_argument('--lambda_LDS', type=float, default=5, help='weight for diversityGAN')
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight on D loss. D(G(A, E(B)))')
            parser.add_argument('--lambda_GAN2', type=float, default=1.0, help='weight on D2 loss, D(G(A, random_z))')
            parser.add_argument('--lambda_z', type=float, default=0.5, help='weight for ||E(G(random_z)) - random_z||')
            parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for KL loss')
            parser.add_argument('--NE_MARGIN_VALUE', type=float, default=2, help='maximum diversity')
        return parser

    def __init__(self, opt):
        if opt.isTrain:
            assert opt.batch_size % 2 == 0  # load two images at one time.

        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        if opt.gan_mode == 'wgangp':
            self.loss_names = ['G_GAN', 'D', 'G_GAN2', 'D2', 'G_L1', 'z_L1', 'kl','gradient_penalty']
        else:
            self.loss_names = ['G_GAN', 'D', 'G_GAN2', 'D2', 'G_L1', 'z_L1', 'kl']
        # diversity senstivity
        if opt.DS_on:
            self.loss_names.append('errNE')
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if opt.DS_on:
            self.visual_names = ['real_B','fake_B','fake_B_random','fake_B_random2']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_D2 = opt.isTrain and opt.lambda_GAN2 > 0.0 and not opt.use_same_D
        use_E = opt.isTrain or not opt.no_encode
        use_vae = True
        self.model_names = ['G']
        print(opt.norm)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type, init_gain=opt.init_gain,
                                      gpu_ids=self.gpu_ids, where_add=opt.where_add, upsample=opt.upsample)
        D_output_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        if use_D2:
            self.model_names += ['D2']
            self.netD2 = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD2, norm=opt.norm, nl=opt.nl,
                                           init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        else:
            self.netD2 = None
        if use_E:
            self.model_names += ['E']
            self.netE = networks.define_E(opt.output_nc, opt.nz, opt.nef, netE=opt.netE, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, vaeLike=use_vae)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionZ = torch.nn.L1Loss().to(self.device)
            self.criterionDS = torch.nn.L1Loss(reduce=False).to(self.device)
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr/4, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if use_E:
                self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr/4, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_E)

            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            if use_D2:
                self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D2)

    def sample_selfdistribution(self):
        '''Just sample cross domain'''
    
    def set_input_test(self, input):
        AtoB = self.opt.direction == 'AtoB'
        raw_AC = input['A' if AtoB else 'B']
        self.real_A = raw_AC[:,[0,1],:,:].to(self.device)
        self.real_B = raw_AC[:,[3,4],:,:].to(self.device)
        self.real_C = raw_AC[:,[8,9],:,:].to(self.device)
        #self.real_LCC = raw_BC[:,[6,7],:,:].to(self.device)
        self.real_C_a = raw_AC[:,[2,6,7],:,:].to(self.device)
        self.real_C_b = raw_AC[:,[5,6,7],:,:].to(self.device)
        self.real_C_c = raw_AC[:,[10,6,7],:,:].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def is_train(self):
        """check if the current batch is good for training."""
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.detach().to(self.device)

    def encode(self, input_image):
        mu, logvar = self.netE.forward(input_image)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    def sample(self, z0=None, encode=True):
        if encode:  # use encoded z
            z0, _ = self.netE(self.real_B)
        if z0 is None:
            z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
        self.fake_B = self.netG(self.real_A, z0)
        z_random = self.get_z_random(self.real_A.size(0), self.opt.nz)
        z_random2 = self.get_z_random(self.real_A.size(0), self.opt.nz)
        self.fake_B_random = self.netG(self.real_A, z_random)
        self.fake_B_random2 = self.netG(self.real_A, z_random2)
        #return self.real_A, self.fake_B, self.real_B

    def forward(self):
        # get real images
        #half_size = self.real_A.size(0) // 2
        # A1, B1 for encoded; A2, B2 for random
        #self.real_A_encoded = self.real_A[0:half_size]
        #self.real_B_encoded = self.real_B[0:half_size]
        #self.real_A_random = self.real_A[half_size:]
        #self.real_B_random = self.real_B[half_size:]
        self.real_A_encoded = self.real_A
        self.real_B_encoded = self.real_B
        self.real_A_random = self.real_A
        self.real_B_random = self.real_B
        # get encoded z
        self.z_encoded, self.mu, self.logvar = self.encode(self.real_B_encoded)
        # get random z
        self.z_random = self.get_z_random(self.real_A_random.size(0), self.opt.nz)
        # generate fake_B_encoded
        self.fake_B_encoded = self.netG(self.real_A_encoded, self.z_encoded)
        # generate fake_B_random
        self.fake_B_random = self.netG(self.real_A_random, self.z_random)
        if self.opt.conditional_D:   # tedious conditoinal data
            self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
            self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_B_encoded], 1)
            self.fake_data_random = torch.cat([self.real_A_encoded, self.fake_B_random], 1)
            self.real_data_random = torch.cat([self.real_A_random, self.real_B_random], 1)
        else:
            self.fake_data_encoded = self.fake_B_encoded
            self.fake_data_random = self.fake_B_random
            self.real_data_encoded = self.real_B_encoded
            self.real_data_random = self.real_B_random
        # generate fake_B_random2
        self.z_random2 = self.get_z_random(self.real_A_random.size(0), self.opt.nz)
        self.fake_B_random2 = self.netG(self.real_A_random.detach(), self.z_random2)
        # compute z_predict
        if self.opt.lambda_z > 0.0:
            self.mu2, logvar2 = self.netE(self.fake_B_random)  # mu2 is a point estimate

    def backward_D(self, netD, real, fake):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD(fake.detach())
        # real
        pred_real = netD(real)
        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        if self.opt.gan_mode == 'wgangp':
            self.loss_gradient_penalty, gradients = networks.cal_gradient_penalty(
                netD, real, fake.detach(), self.device, lambda_gp=10.0)
            #self.loss_gradient_penalty.backward(retain_graph=True)
            loss_D = loss_D_fake + loss_D_real + self.loss_gradient_penalty
        else:
            loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            pred_fake = netD(fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded, self.netD, self.opt.lambda_GAN)
        if self.opt.use_same_D:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD, self.opt.lambda_GAN2)
        else:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD2, self.opt.lambda_GAN2)
        # 2. KL loss
        if self.opt.lambda_kl > 0.0:
            self.loss_kl = torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) * (-0.5 * self.opt.lambda_kl)
        else:
            self.loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1 + self.loss_kl
        self.loss_G.backward(retain_graph=True)

    def update_D(self):
        self.set_requires_grad([self.netD, self.netD2], True)
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded)
            if self.opt.use_same_D:
                self.loss_D2, self.losses_D2 = self.backward_D(self.netD, self.real_data_random, self.fake_data_random)
            self.optimizer_D.step()

        if self.opt.lambda_GAN2 > 0.0 and not self.opt.use_same_D:
            self.optimizer_D2.zero_grad()
            self.loss_D2, self.losses_D2 = self.backward_D(self.netD2, self.real_data_random, self.fake_data_random)
            self.optimizer_D2.step()

    def backward_G_alone(self):
        # 3, reconstruction |(E(G(A, z_random)))-z_random|
        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = self.criterionZ(self.mu2, self.z_random) * self.opt.lambda_z
        else:
            self.loss_z_L1 = 0.0
        Galongloss = self.loss_z_L1
        #4 diversity senstivity
        if self.opt.DS_on:
            _eps = 1.0e-5
            batch_wise_imgs_l1 = self.criterionDS(self.fake_B_random.detach(), self.fake_B_random2).sum(dim=1).sum(dim=1).sum(dim=1)
            batch_wise_imgs_l1 = batch_wise_imgs_l1 / (self.fake_B_random.size(1)*self.fake_B_random.size(2)*self.fake_B_random.size(3))
            batch_wise_z_l1 = self.criterionDS(self.z_random.detach(), self.z_random2.detach()).sum(dim=1)
            batch_wise_z_l1 = batch_wise_z_l1 / self.z_random2.size(1)
            self.loss_errNE =  (batch_wise_imgs_l1 / (batch_wise_z_l1 + _eps)).mean()
            if self.opt.NE_MARGIN:
                self.loss_errNE = -torch.clamp(self.loss_errNE, max=self.opt.NE_MARGIN_VALUE).mean()* self.opt.lambda_LDS
            else:
                self.loss_errNE = -self.loss_errNE.mean()* self.opt.lambda_LDS
            Galongloss = Galongloss + self.loss_errNE
        Galongloss.backward()

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad([self.netD, self.netD2], False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()

        # update G alone
        if self.opt.lambda_z > 0.0:
            self.set_requires_grad([self.netE], False)
            self.backward_G_alone()
            self.set_requires_grad([self.netE], True)
        else:
            self.loss_z_L1 = 0.0

        self.optimizer_E.step()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G_and_E()
        self.update_D()