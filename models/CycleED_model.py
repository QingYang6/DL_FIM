import torch
from .base_model import BaseModel
from . import network_bicycle as networks

class CycleEDModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(conditional_D=True, use_same_D=True, norm='instance',
                            netD='basic_256_multi', netE='basic_256', ngf='64', ndf='64')
        parser.add_argument('--num_Ds', type=int, default=2, help='number of Discriminators')
        parser.add_argument('--nz', type=int, default=128, help='# latent vector')
        parser.add_argument('--nef', type=int, default=64, help='# of encoder filters in the first conv layer')
        parser.add_argument('--upsample', type=str, default='basic', help='basic | bilinear')
        parser.add_argument('--nl', type=str, default='relu', help='non-linearity activation: relu | lrelu | elu')
        parser.add_argument('--no_encode', action='store_true', help='use the Encoder or not')
        parser.add_argument('--where_add', type=str, default='all', help='input|all|middle; where to add z in the network G')
        parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for |B-G(A, E(B))|')
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight on D loss. D(G(A, E(B)))')
            parser.add_argument('--lambda_GAN2', type=float, default=1.0, help='weight on D2 loss, D(G(A, random_z))')
            parser.add_argument('--lambda_z', type=float, default=1, help='weight for ||E(G(random_z)) - random_z||')
            parser.add_argument('--lambda_kl', type=float, default=0, help='weight for KL loss')
        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses to print out
        self.loss_names = ['G_GAN', 'D', 'G_L1', 'z_L1']
        # specify the images to save/display
        self.visual_names = ['real_B','fake_B','fake_B_random','fake_B_random2']

        # specify the models to save/load
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_E = True  # Always use encoder in this model
        use_vae = False
        self.model_names = ['E', 'Decoder', 'E_f']
        if use_D:
            self.model_names += ['D']

        # Initialize networks
        # Assuming 'opt' is an options object with the necessary attributes
        self.netE = networks.define_E(opt.input_nc, 128, opt.nef, netE=opt.netE, norm=opt.norm, nl=opt.nl,
                                      init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, vaeLike=use_vae)
        self.netE_f = networks.define_E(opt.output_nc, 128, opt.nef, netE=opt.netE, norm=opt.norm, nl=opt.nl,
                                       init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, vaeLike=use_vae)
        self.netDecoder = networks.define_Decoder(opt.output_nc, 128, opt.ndf, netD='basic_256', norm=opt.norm, nl=opt.nl,
                                                  init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, vaeLike=use_vae)
        if use_D:
            input_D_nc = opt.output_nc + opt.input_nc if opt.conditional_D else opt.output_nc
            self.netD = networks.define_D(input_D_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        # Define loss functions and optimizers
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionZ = torch.nn.L1Loss()  # or any other loss function for z_L1
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Decoder = torch.optim.Adam(self.netDecoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_E_f = torch.optim.Adam(self.netE_f.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_E, self.optimizer_Decoder, self.optimizer_E_f, self.optimizer_D]

    # [Implement set_input, forward, backward methods, optimize_parameters, etc., based on BiCycleGANModel's structure]

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
        self.real_A = self.real_A[:,[0,1],:,:]
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def sample(self, z0=None, encode=True):
        if encode:  # use encoded z
            #z0, _ = self.netE(self.real_B)
            z0 = self.netE_f(self.real_B)
        if z0 is None:
            z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
        #self.fake_B = self.netG(self.real_A, z0)
        self.fake_B = self.netDecoder(z0)
        z_random = self.get_z_random(self.real_A.size(0), self.opt.nz)
        z_random2 = self.get_z_random(self.real_A.size(0), self.opt.nz)
        self.fake_B_random = self.netDecoder(z_random)
        self.fake_B_random2 = self.netDecoder(z_random2)
        #return self.real_A, self.fake_B, self.real_B

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.detach().to(self.device)

    def forward(self):
        # Encode input to get latent Z
        self.latent_Z = self.netE(self.real_A)  # Assuming netE returns (mu, logvar)

        # Generate random noise Z_r
        self.z_random = torch.randn_like(self.latent_Z).to(self.device)

        # Concat latent Z with random noise Z_r
        #self.latent_Z_r = torch.cat((self.latent_Z, z_random), dim=1)

        # Decode concatenated latent vector to produce output X_z
        self.fake_B = self.netDecoder(self.latent_Z)

        # Encode observation Y to get latent Z_Y
        self.latent_Z_Y = self.netE_f(self.real_B)  # Assuming netE_f returns (mu, logvar)
        
        self.fake_Y = self.fake_B
        
        # generate fake_B_random
        self.fake_B_random = self.netDecoder(self.z_random)
        # fake_B random 2
        self.z_random2 = torch.randn_like(self.latent_Z).to(self.device)
        self.fake_B_random2 = self.netDecoder(self.z_random2)
    
    def backward_D(self, netD, real, fake, conditon):
        # Fake, stop backprop to the generator by detaching fake_B
        if self.opt.conditional_D:
            fake_concat = torch.cat((conditon.detach(), fake.detach()), 1)
            real_concat = torch.cat((conditon.detach(), real), 1)
        else:
            fake_concat = fake.detach()
            real_concat = real
        pred_fake = netD(fake_concat)
        pred_real = netD(real_concat)
        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        if self.opt.gan_mode == 'wgangp':
            self.loss_gradient_penalty, gradients = networks.cal_gradient_penalty(
                netD, real_concat, fake_concat.detach(), self.device, lambda_gp=10.0)
            #self.loss_gradient_penalty.backward(retain_graph=True)
            loss_D = loss_D_fake + loss_D_real + self.loss_gradient_penalty
        else:
            loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]
    
    def update_D(self):
        self.set_requires_grad(self.netD, True)
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_B, self.fake_Y, self.real_A)
            self.optimizer_D.step()
            
    def backward_G_GAN(self, fake, conditon, netD=None, ll=0.0):
        if self.opt.conditional_D:
            pred_fake = netD(torch.cat((conditon.detach(), fake), 1))
        else:
            pred_fake = netD(fake)
        if ll > 0.0:
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll
    
    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_Y, self.real_A, self.netD, self.opt.lambda_GAN)
        # 2. KL loss
        #if self.opt.lambda_kl > 0.0:
        #    self.loss_kl = torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) * (-0.5 * self.opt.lambda_kl)
        #else:
        #    self.loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.real_B, self.fake_Y) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward(retain_graph=True)
        
    def backward_E2_alone(self):
        # 3, reconstruction |(E(G(A, z_random)))-z_random|
        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = self.criterionZ(self.latent_Z, self.latent_Z_Y) * self.opt.lambda_z
            self.loss_z_L1.backward()
        else:
            self.loss_z_L1 = 0.0

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad(self.netD, False)
        self.optimizer_E.zero_grad()
        self.optimizer_E_f.zero_grad()
        self.optimizer_Decoder.zero_grad()
        self.backward_EG()
        self.backward_E2_alone()
        self.optimizer_E.step()
        self.optimizer_E_f.step()
        self.optimizer_Decoder.step()
        
    def update_G_and_E_ori(self):
        # update G and E
        self.set_requires_grad(self.netD, False)
        self.optimizer_E.zero_grad()
        self.optimizer_E_f.zero_grad()
        self.optimizer_Decoder.zero_grad()
        self.backward_EG()

        # update G alone
        if self.opt.lambda_z > 0.0:
            self.set_requires_grad([self.netE,self.netDecoder], False)
            self.backward_E2_alone()
            self.set_requires_grad([self.netE,self.netDecoder], True)
        else:
            self.loss_z_L1 = 0.0

        self.optimizer_E.step()
        self.optimizer_E_f.step()
        self.optimizer_Decoder.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G_and_E()
        self.update_D()