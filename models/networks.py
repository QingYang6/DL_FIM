import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import functools
from typing import Tuple, Optional, List
from torch.optim import lr_scheduler
from torch.nn import functional as func
from models.ResNet import *
from models.inception_pytorch import InceptionV3
import torchvision
import numpy as np


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            try:
                aint = len(m.weight.data)
            except Exception as e:
                print(e)
                print('using EqualizedWeight for %s' % classname)
                return
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    # it is weird to us gpu_ids, simply using torch.cuda.is_available() and get device to 'cuda'. Q
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net) 
        net.to(gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[],\
    style_dim=64,n_downsample=2, n_res=4, mlp_dim=256,activ='relu', pad_type='reflect'):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_9blocks_1':
        net = InterResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, split_mode=1)
    elif netG == 'resnet_9blocks_2':
        net = InterResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, split_mode=2)
    elif netG == 'resnet_9blocks_3':
        net = InterResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, split_mode=3)
    elif netG == 'resnet_9blocks_4':
        net = InterResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, split_mode=4)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'AdaINGen':
        net = AdaINGen(input_nc, output_nc, ngf, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim)
    elif netG == 'AdaINGen2C1S':
        net = AdaINGen2C1S(input_nc, output_nc, ngf, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim)
    elif netG == 'AdaINGen_IA':
        net = AdaINGen_IA(input_nc, output_nc, ngf, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim)
    elif netG == 'AdaINGen_3X2E':
        net = AdaINGen_3X2E(input_nc, output_nc, ngf, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim)
    elif netG == 'AdaINGen_3E':
        net = AdaINGen_3E(input_nc, output_nc, ngf, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim)
    elif netG == 'ModulateGen':
        net = ModulateGen(input_nc, output_nc, ngf, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim)
    elif netG == 'ModulateGen_IA':
        net = ModulateGen_IA(input_nc, output_nc, ngf, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim)
    elif netG == 'VAEGen':
        net = VAEGen(input_nc, ngf, style_dim, n_downsample, n_res, activ, pad_type)
    elif netG == 'lat':
        net1 = GeneratorEncoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        net2 = GeneratorDecoder(net1.decoder, norm_layer)
        return init_net(net1, init_type, init_gain, gpu_ids), init_net(net2, init_type, init_gain, gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_VAEG(input_nc, img_shape, latent_dim,  output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'unet_256_concat':
        net = UnetGenerator_concat(input_nc, img_shape, latent_dim, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'lat':
        net1 = GeneratorEncoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        net2 = GeneratorDecoder(net1.decoder, norm_layer)
        return init_net(net1, init_type, init_gain, gpu_ids), init_net(net2, init_type, init_gain, gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], \
    mask_size=128, s1=32, s2=16, activ='lrelu', num_scales=3, pad_type='reflect'):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'trainable_attn':
        net = AttnDiscriminator(input_nc, ndf, norm_layer=norm_layer, mask_shape=mask_size,
                                inner_s1=(s1, s1), inner_s2=(s2, s2))
    elif netD == 'trainable_attn_v2':
        net = AttnDiscriminator(input_nc, ndf, norm_layer=norm_layer, mask_shape=mask_size,
                                inner_s1=(s1, s1), inner_s2=(s2, s2), inner_rescale=False)
    elif netD == 'posthoc_attn':
        net = PHADiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'DCGAN':
        net = DCGANDiscriminator(input_nc)
    elif netD == 'MsImageDis':
        net = MsImageDis(input_nc, n_layers_D, ndf, norm, activ, num_scales, pad_type)
    elif netD == 'posthoc_attn_v2':
        net = PHADiscriminator(input_nc, ndf, norm_layer=norm_layer, inner_rescale=False)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_E(input_nc, latent_dim, num_class, ngf, netE, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], activ=None,pad_type=None):
    """Create a Encoder

    Returns a Encoder
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netE == 'resnet18':
        net = ResNet18Encoder(input_nc, latent_dim)
    elif netE == 'resnet18_Cin':
        net = ResNet18inC_Encoder(input_nc, latent_dim, num_class)
    elif netE == 'resnet34':
        net = ResNet34(input_nc, latent_dim)
    elif netE == 'resnet50':
        net = ResNet50(input_nc, latent_dim)
    elif netE == 'resnet66':
        net1 = GeneratorEncoder(input_nc, latent_dim, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        net2 = GeneratorDecoder(net1.decoder, norm_layer)
        return init_net(net1, init_type, init_gain, gpu_ids), init_net(net2, init_type, init_gain, gpu_ids)
    elif netE == 'MUNIT_E':
        netE = ContentEncoder(num_class, ngf, input_nc, latent_dim, norm, activ, pad_type)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % netE)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_CLS(input_nc, numberofclass, ngf, netCLS, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a Classifier

    Returns a Classifier
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netCLS == 'AlexNet':
        net = AlexNet(input_nc, numberofclass)
    elif netCLS == 'resnet34':
        net = ResNet34(input_nc, numberofclass)
    elif netCLS == 'resnet50':
        net = ResNet50(input_nc, numberofclass)
    elif netCLS == 'resnet66':
        net1 = GeneratorEncoder(input_nc, numberofclass, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        net2 = GeneratorDecoder(net1.decoder, norm_layer)
        return init_net(net1, init_type, init_gain, gpu_ids), init_net(net2, init_type, init_gain, gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netCLS)

    return init_net(net, init_type, init_gain, gpu_ids)

def define_C(norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[], mode=0):
    norm_layer = get_norm_layer(norm_type=norm)
    net = SComponent(norm_layer, mode)

    return init_net(net, init_type, init_gain, gpu_ids)

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        elif gan_mode in ['hinge']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, predictions, target_is_real,isG=False):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor list) - - tpyically the prediction output from a discriminator; supports multi Ds.
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        all_losses = []
        for prediction in predictions:
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
            elif self.gan_mode == 'hinge':
                if target_is_real:
                    if isG:
                        loss = -prediction.mean()
                    else:
                        loss = nn.ReLU()(1.0 - prediction).mean()
                else:
                    loss = nn.ReLU()(1.0 + prediction).mean()
            all_losses.append(loss)
        total_loss = sum(all_losses)
        return total_loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        grad_outputlist = []
        #if type(disc_interpolates) == list:
        for prediction in disc_interpolates:
            grad_outputlist.append(torch.ones(prediction.size()).to(device))
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=grad_outputlist,
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class InterResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', split_mode=0):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
            split_mode = 0: No split
            split_mode = 1: Split the model at the beginning of down-sampling layer
            split_mode = 2: Split the model at the beginning of residual block chain
            split_mode = 3: Split the model at the middle of residual block chain
            split_mode = 4: Split the model at the end of residual block chain

        """
        assert(n_blocks >= 0)
        if split_mode not in [0, 1, 2, 3, 4]:
            raise Exception('Unsupported split mode selected')
        super(InterResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        if split_mode == 1:
            self.model1 = nn.Sequential(*model)
            model = list()

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        if split_mode == 2:
            self.model1 = nn.Sequential(*model)
            model = list()

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            if 3 == i and 3 == split_mode:
                self.model1 = nn.Sequential(*model)
                model = list()

        if split_mode == 4:
            self.model1 = nn.Sequential(*model)
            model = list()

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model2 = nn.Sequential(*model)

        self.fe = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=2, stride=2),
            norm_layer(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, kernel_size=2, stride=2)
        )

        self.fd = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=4, stride=4),
            norm_layer(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=4, stride=4),
            norm_layer(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1)
        )

    def forward(self, x, y_t, r_t):
        """Standard forward"""
        inter = self.model1(x) # get h_t

        fe_in = torch.cat((y_t, r_t), dim=1)
        fe_out = self.fe(fe_in)

        fd_in = torch.cat((inter, fe_out), dim=1)
        fd_out = self.fd(fd_in)

        tmp = torch.split(fd_out, 256, dim=1)
        gamma, beta = tmp[0], tmp[1]

        return self.model2(gamma * inter + beta)


class GeneratorEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GeneratorEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            if 3 == i:
                self.model = nn.Sequential(*model)
                model = list()

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.decoder = nn.Sequential(*model)

    def forward(self, x):
        """Compute h_{t-1}"""
        return self.model(x)


class GeneratorDecoder(nn.Module):
    def __init__(self, net, norm_layer):
        super(GeneratorDecoder, self).__init__()
        self.model = net

        self.fe = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=2, stride=2),
            norm_layer(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, kernel_size=2, stride=2)
        )

        self.fd = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=4, stride=4),
            norm_layer(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=4, stride=4),
            norm_layer(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1)
        )

    def forward(self, x, y, r):
        fe_in = torch.cat((y, r), dim=1)
        fe_out = self.fe(fe_in)

        fd_in = torch.cat((x, fe_out), dim=1)
        fd_out = self.fd(fd_in)

        tmp = torch.split(fd_out, 256, dim=1)
        gamma, beta = tmp[0], tmp[1]

        new_h = gamma * x + beta

        return self.model(new_h), new_h


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetGenerator_concat(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, img_shape, latent_dim, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator_concat, self).__init__()
        self.h, self.w = img_shape, img_shape
        self.fc_z = nn.Linear(latent_dim, self.h * self.w)
        self.fc_c = nn.Linear(1, self.h * self.w)
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer
    def forward(self, x,z,c):
        z = self.fc_z(z).view(z.size(0), 1, self.h, self.w)
        c = torch.reshape(c,(c.size(0), 1))
        c = self.fc_c(c).view(c.size(0), 1, self.h, self.w)
        input = torch.cat((x,z,c),1)
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, kw=4):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = kw
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)
        # return func.interpolate(self.model(x), (256, 256), mode='bilinear')

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class attn_module(nn.Module):
    '''Auxiliary attention module for parallel trainable attention network'''


    def __init__(self, in_ch, out_ch, s1=(64, 64), s2=(32, 32)):
        self.s1, self.s2 = s1, s2
        super(attn_module, self).__init__()
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (128 * 128) -> (64 * 64)
        self.d1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (64 * 64) -> (32 * 32)
        self.d2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.mp3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (32 * 32) -> (16 * 16)

        self.mid = nn.Sequential(*[
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        ])

        self.u2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.u1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.last = nn.Sequential(*[
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 1, 1, bias=False),
            nn.Sigmoid()
        ])

    def forward(self, x):
        out = func.relu(self.d1(self.mp1(x)))
        out = func.relu(self.d2(self.mp2(out)))
        skip2 = func.relu(self.skip2(out))
        out = self.mid(out)
        out = func.interpolate(out, size=self.s2, mode='bilinear', align_corners=True) + skip2
        # (14 * 14 -> 28 * 28)
        out = self.last(self.u2(out))

        return out


class AttnDiscriminator(nn.Module):
    # The size of input is assumed to be 256 * 256
    def __init__(self, in_ch, int_ch, n_layers=3, mask_shape=256, norm_layer=nn.BatchNorm2d, inner_rescale=True,
                 inner_s1=None, inner_s2=None):
        super(AttnDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw, padw = 4, 1
        self.fe = nn.Sequential(*[
            nn.Conv2d(in_ch, int_ch, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(.2, True)
        ])

        trunk_model = list()
        nf_mult, nf_mult_prev = 1, 1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8)

            trunk_model += [
                nn.Conv2d(int_ch * nf_mult_prev, int_ch * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(int_ch * nf_mult),
                nn.LeakyReLU(.2, True)
            ]

        self.trunk_brunch = nn.Sequential(*trunk_model)
        if not inner_s1 and not inner_s2:
            self.mask_brunch = attn_module(int_ch, int_ch * nf_mult)
        else:
            self.mask_brunch = attn_module(int_ch, int_ch * nf_mult, inner_s1, inner_s2)

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        self.fin_phase = nn.Sequential(*[
            nn.Conv2d(int_ch * nf_mult_prev, int_ch * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(int_ch * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int_ch * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ])

        self.inner_rescale = inner_rescale
        self.mask_shape = mask_shape

    def forward(self, x):
        feature = self.fe(x)
        trunk = self.trunk_brunch(feature)
        mask = self.mask_brunch(feature)

        expand_mask = self._mask_rescale(mask, self.inner_rescale)

        return self.fin_phase((mask + 1) * trunk), expand_mask

    def _mask_rescale(self, mask_tensor, final_scale=False):
        mask_tensor = torch.mean(mask_tensor, dim=1).unsqueeze(1)

        if final_scale:
            t_max, t_min = torch.max(mask_tensor), torch.min(mask_tensor)
            mask_tensor = (mask_tensor - t_min) / (t_max - t_min)
            return func.interpolate(mask_tensor, (self.mask_shape, self.mask_shape), mode='bilinear')
        else:
            return mask_tensor


class PHADiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, attn_mode='abs_sum', inner_rescale=True):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PHADiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        self.model_p1 = nn.Sequential(*sequence)

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence = [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model_p2 = nn.Sequential(*sequence)
        self.attn_mode = attn_mode
        self.inner_rescale = inner_rescale
        self.mask_shape = 256

    def forward(self, x):
        """Standard forward."""
        inter = self.model_p1(x)
        res = self.model_p2(inter)

        expand_mask = self._mask_rescale(inter, self.inner_rescale)

        return res, expand_mask

    def _mask_rescale(self, mask_tensor, final_scale=False):
        mask_tensor = torch.mean(abs(mask_tensor), dim=1).unsqueeze(1)

        if final_scale:
            t_max, t_min = torch.max(mask_tensor), torch.min(mask_tensor)
            mask_tensor = (mask_tensor - t_min) / (t_max - t_min)
            return func.interpolate(mask_tensor, (self.mask_shape, self.mask_shape), mode='bilinear')
        else:
            return mask_tensor


class SComponent(nn.Module):
    def __init__(self, norm_layer, mode=0):
        super(SComponent, self).__init__()
        if 0 == mode:
            sequence = [nn.ConvTranspose2d(1, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                        norm_layer(64),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                        norm_layer(128),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(128, 1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                        norm_layer(64),
                        nn.Sigmoid()]
        else:
            sequence = [nn.ConvTranspose2d(1, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                        norm_layer(64),
                        nn.ReLU(True),
                        nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
                        norm_layer(128),
                        nn.ReLU(True),
                        nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=True),
                        norm_layer(64),
                        nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

class AlexNet(nn.Module):
    def __init__(self, input_nc, num_classes=None):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        # See note [TorchScript super()]
        x = self.features(inputs)
        x = self.avgpool(x)
        f = torch.flatten(x, 1)
        x = self.classifier(f)
        return x, f

class DCGANDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=128, n_layers=4, norm_layer=nn.BatchNorm2d, kw=3):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(DCGANDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = kw
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, bias=use_bias), 
        norm_layer(ndf), 
        nn.LeakyReLU(0.1, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.1, True)
            ]
        squence_feature = sequence + [nn.AdaptiveAvgPool2d((6,6))]
        self.feature = nn.Sequential(*squence_feature)
        squence_feature += [
            nn.LeakyReLU(0.3, True),
            nn.Linear(6,1),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*squence_feature)

    def forward(self, x):
        return self.model(x), self.feature(x)

##################################################################################
# Discriminator (MUNIT)
##################################################################################
class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, n_layer, dim, norm, activ, num_scales, pad_type):
        super(MsImageDis, self).__init__()
        self.n_layer = n_layer
        self.dim = dim
        self.norm = norm
        self.activ = activ
        self.num_scales = num_scales
        self.pad_type = pad_type
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

##################################################################################
# Generator (MUNIT)
##################################################################################
class AdaINGen2C1S(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, output_dim, dim, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim):
        super(AdaINGen2C1S, self).__init__()

        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_UC = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.enc_CC = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        # decoder
        self.dec = Decoder(n_downsample, n_res, self.enc_UC.output_dim + self.enc_CC.output_dim, output_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        UC, CC, style_fake = self.encode(images)
        images_recon = self.decode(UC, CC, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        UC = self.enc_UC(images)
        CC = self.enc_CC(images)
        return UC, CC, style_fake

    def decode(self, UC, CC, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(torch.cat((UC, CC), 1))
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, output_dim, dim, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim):
        super(AdaINGen, self).__init__()

        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, output_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

class AdaINGen_IA_nocondi(nn.Module):
    # AdaIN auto-encoder architecture
    # Modified as conditional encoder
    def __init__(self, input_dim, output_dim, dim, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim):
        super(AdaINGen_IA_nocondi, self).__init__()

        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        #self.dec = CondiConcatDecoder(n_downsample, n_res, self.enc_content.output_dim, output_dim, res_norm='adain', activ=activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim*2, output_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode_condi(self, images):
        # encode condition to latent
        condi = self.enc_condi(images)
        return condi

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style, condi):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        #images = self.dec(content, condi)
        images = self.dec(torch.cat((content, condi), 1))
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

class AdaINGen_IA(nn.Module):
    # AdaIN auto-encoder architecture
    # Modified as conditional encoder
    def __init__(self, input_dim, output_dim, dim, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim):
        super(AdaINGen_IA, self).__init__()

        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.enc_condi = ContentEncoder(n_downsample, n_res, 3, dim, 'in', activ, pad_type=pad_type)
        #self.dec = CondiConcatDecoder(n_downsample, n_res, self.enc_content.output_dim, output_dim, res_norm='adain', activ=activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim*2, output_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode_condi(self, images):
        # encode condition to latent
        condi = self.enc_condi(images)
        return condi

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style, condi):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        #images = self.dec(content, condi)
        images = self.dec(torch.cat((content, condi), 1))
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

class AdaINGen_3X2E(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, output_dim, dim, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim):
        super(AdaINGen_3X2E, self).__init__()

        # style encoder
        self.enc_style = StyleEnC_Mapping(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.enc_condi = ContentEncoder(n_downsample, n_res, 4, dim, 'in', activ, pad_type=pad_type)
        self.enc_condi_X = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim*2, output_dim, res_norm='adain', activ=activ, pad_type=pad_type)
        #self.dec = MoDecoder(n_downsample, style_dim, self.enc_content.output_dim*2, output_dim, res_norm='modulate', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)
        #self.mlp = MappingNetwork(style_dim,8)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode_condi(self, images):
        # encode condition to latent
        condi = self.enc_condi(images)
        return condi

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        condi = self.enc_condi_X(images)
        return content, style_fake, condi

    def decode(self, content, style, condi):
        #decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        con_cat = torch.cat((content,condi), 1)
        images = self.dec(con_cat)
        return images
        #w = self.mlp(style)
        #con_cat = torch.cat((content,condi), 1)
        #images = self.dec(con_cat,w,None)
        #if needw:
        #    return images,w
        #else:
        #    return images
        
    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

class AdaINGen_3E(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, output_dim, dim, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim):
        super(AdaINGen_3E, self).__init__()

        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.enc_sec_content = ContentEncoder(n_downsample, n_res, 5, dim, 'in', activ, pad_type=pad_type)
        self.enc_condi = ContentEncoder(n_downsample, n_res, 2, dim, 'in', activ, pad_type=pad_type)
        self.enc_condi_X = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim*2, output_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode_condi(self, images):
        # encode condition to latent
        condi = self.enc_condi(images)
        return condi

    def encode_sec_content(self, images):
        # encode condition to latent
        condi = self.enc_sec_content(images)
        return condi

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        condi = self.enc_condi_X(images)
        return content, style_fake, condi

    def decode(self, content, style, condi):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        con_cat = torch.cat((content,condi), 1)
        images = self.dec(con_cat)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

class VAEGen(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, dim, style_dim, n_downsample, n_res, activ, pad_type):
        super(VAEGen, self).__init__()

        # content encoder
        self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='in', activ=activ, pad_type=pad_type)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images

##################################################################################
# Encoder and Decoders (MUNIT)
##################################################################################

class StyleEnC_Mapping(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEnC_Mapping, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return torch.squeeze(self.model(x))

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class CondiConcatDecoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(CondiConcatDecoder, self).__init__()

        self.model = []
        self.model_norm = []
        # AdaIN residual blocks
        self.model_norm += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        self.model_norm = nn.Sequential(*self.model_norm)
        dim *=2
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x, condi):
        x = self.model_norm(x)
        return self.model(torch.cat((x,condi.detach()), 1))

##################################################################################
# Sequential Models (MUNIT)
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

##################################################################################
# Basic Blocks (MUNIT)
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

####modulateGEN###
class ModulateGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, output_dim, dim, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim):
        super(ModulateGen, self).__init__()

        # style encoder
        self.enc_style = StyleEnC_Mapping(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = MoDecoder(n_downsample, style_dim, self.enc_content.output_dim, output_dim, res_norm='modulate', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        #self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)
        self.mlp = MappingNetwork(style_dim,8)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        w = self.mlp(style)
        images = self.dec(content,w,None)
        return images

class ModulateGen_IA(nn.Module):
    # AdaIN auto-encoder architecture
    # Modified as conditional encoder
    def __init__(self, input_dim, output_dim, dim, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim):
        super(ModulateGen_IA, self).__init__()

        # style encoder
        self.enc_style = StyleEnC_Mapping(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.enc_condi = ContentEncoder(n_downsample, n_res, 3, dim, 'in', activ, pad_type=pad_type)
        self.dec = MoDecoder(n_downsample, style_dim, self.enc_content.output_dim*2, output_dim, res_norm='modulate', activ=activ, pad_type=pad_type)
        #self.dec = CondiConcatDecoder(n_downsample, n_res, self.enc_content.output_dim, output_dim, res_norm='adain', activ=activ, pad_type=pad_type)
        #self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim*2, output_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        #self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)
        self.mlp = MappingNetwork(style_dim,8)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode_condi(self, images):
        # encode condition to latent
        condi = self.enc_condi(images)
        return condi

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style, condi):
        # decode content and style codes to an image
        #adain_params = self.mlp(style)
        #self.assign_adain_params(adain_params, self.dec)
        #images = self.dec(content, condi)
        #images = self.dec(torch.cat((content, condi), 1))
        w = self.mlp(style)
        images = self.dec(torch.cat((content,condi), 1), w, None)
        return images

#modulateDecoder#
class MoDecoder(nn.Module):
    def __init__(self, n_upsample, style_dim, dim, output_dim, res_norm='modulate', activ='relu', pad_type='zero'):
        super(MoDecoder, self).__init__()

        self.model = []
        # Modulate blocks
        # Get style vector from $w$ (denoted by $A$ in the diagram) with
        # an [equalized learning-rate linear layer](#equalized_linear)
        self.to_style = EqualizedLinear(style_dim, dim, bias=1.0)
        # Weight modulated convolution layer
        self.conv = Conv2dWeightModulate(dim, dim, kernel_size=3)
        # Noise scale
        self.scale_noise = nn.Parameter(torch.zeros(1))
        # Bias
        self.bias = nn.Parameter(torch.zeros(dim))

        # Activation function
        self.activation = nn.LeakyReLU(0.2, True)

        #self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):
        # Get style vector $s$
        s = self.to_style(w)
        # Weight modulated convolution
        x = self.conv(x, s)
        # Scale and add noise
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        x = self.activation(x + self.bias[None, :, None, None])
        return self.model(x)
##################################################################################
# StyleGAN2 Convolution with Weight Modulation and Demodulation 
##################################################################################    
class Conv2dWeightModulate(nn.Module):
    """
    ### Convolution with Weight Modulation and Demodulation
    This layer scales the convolution weights by the style vector and demodulates by normalizing it.
    """
    def __init__(self, in_features: int, out_features: int, kernel_size: int,
                 demodulate: float = True, eps: float = 1e-8):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `kernel_size` is the size of the convolution kernel
        * `demodulate` is flag whether to normalize weights by its standard deviation
        * `eps` is the $\epsilon$ for normalizing
        """
        super().__init__()
        # Number of output features
        self.out_features = out_features
        # Whether to normalize weights
        self.demodulate = demodulate
        # Padding size
        self.padding = (kernel_size - 1) // 2

        # [Weights parameter with equalized learning rate](#equalized_weight)
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        # $\epsilon$
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `s` is style based scaling tensor of shape `[batch_size, in_features]`
        """

        # Get batch size, height and width
        b, _, h, w = x.shape

        # Reshape the scales
        s = s[:, None, :, None, None]
        # Get [learning rate equalized weights](#equalized_weight)
        weights = self.weight()[None, :, :, :, :]
        # $$w`_{i,j,k} = s_i * w_{i,j,k}$$
        # where $i$ is the input channel, $j$ is the output channel, and $k$ is the kernel index.
        #
        # The result has shape `[batch_size, out_features, in_features, kernel_size, kernel_size]`
        weights = weights * s

        # Demodulate
        if self.demodulate:
            # $$\sigma_j = \sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}$$
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            # $$w''_{i,j,k} = \frac{w'_{i,j,k}}{\sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}}$$
            weights = weights * sigma_inv

        # Reshape `x`
        x = x.reshape(1, -1, h, w)

        # Reshape weights
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        # Use grouped convolution to efficiently calculate the convolution with sample wise kernel.
        # i.e. we have a different kernel (weights) for each sample in the batch
        x = func.conv2d(x, weights, padding=self.padding, groups=b)

        # Reshape `x` to `[batch_size, out_features, height, width]` and return
        return x.reshape(-1, self.out_features, h, w)

class MappingNetwork(nn.Module):
    """
    <a id="mapping_network"></a>
    ## Mapping Network
    ![Mapping Network](mapping_network.svg)
    This is an MLP with 8 linear layers.
    The mapping network maps the latent vector $z \in \mathcal{W}$
    to an intermediate latent space $w \in \mathcal{W}$.
    $\mathcal{W}$ space will be disentangled from the image space
    where the factors of variation become more linear.
    """

    def __init__(self, features: int, n_layers: int):
        """
        * `features` is the number of features in $z$ and $w$
        * `n_layers` is the number of layers in the mapping network.
        """
        super().__init__()

        # Create the MLP
        layers = []
        for i in range(n_layers):
            # [Equalized learning-rate linear layers](#equalized_linear)
            layers.append(EqualizedLinear(features, features))
            # Leaky Relu
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        # Normalize $z$
        z = func.normalize(z, dim=1)
        # Map $z$ to $w$
        return self.net(z)

class EqualizedLinear(nn.Module):
    """
    <a id="equalized_linear"></a>
    ## Learning-rate Equalized Linear Layer
    This uses [learning-rate equalized weights](#equalized_weights) for a linear layer.
    """

    def __init__(self, in_features: int, out_features: int, bias: float = 0.):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `bias` is the bias initialization constant
        """

        super().__init__()
        # [Learning-rate equalized weights](#equalized_weights)
        self.weight = EqualizedWeight([out_features, in_features])
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        # Linear transformation
        return func.linear(x, self.weight(), bias=self.bias)
        
class EqualizedWeight(nn.Module):
    """
    <a id="equalized_weight"></a>
    ## Learning-rate Equalized Weights Parameter
    This is based on equalized learning rate introduced in the Progressive GAN paper.
    Instead of initializing weights at $\mathcal{N}(0,c)$ they initialize weights
    to $\mathcal{N}(0, 1)$ and then multiply them by $c$ when using it.
    $$w_i = c \hat{w}_i$$
    The gradients on stored parameters $\hat{w}$ get multiplied by $c$ but this doesn't have
    an affect since optimizers such as Adam normalize them by a running mean of the squared gradients.
    The optimizer updates on $\hat{w}$ are proportionate to the learning rate $\lambda$.
    But the effective weights $w$ get updated proportionately to $c \lambda$.
    Without equalized learning rate, the effective weights will get updated proportionately to just $\lambda$.
    So we are effectively scaling the learning rate by $c$ for these weight parameters.
    """
    def __init__(self, shape: List[int]):
        """
        * `shape` is the shape of the weight parameter
        """
        super().__init__()

        # He initialization constant
        self.c = 1 / np.sqrt(np.prod(shape[1:]))
        # Initialize the weights with $\mathcal{N}(0, 1)$
        self.weight = nn.Parameter(torch.randn(shape))
        # Weight multiplication coefficient

    def forward(self):
        # Multiply the weights by $c$ and return
        return self.weight * self.c

class PathLengthPenalty(nn.Module):
    """
    <a id="path_length_penalty"></a>
    ## Path Length Penalty
    This regularization encourages a fixed-size step in $w$ to result in a fixed-magnitude
    change in the image.
    $$\mathbb{E}_{w \sim f(z), y \sim \mathcal{N}(0, \mathbf{I})}
      \Big(\Vert \mathbf{J}^\top_{w} y \Vert_2 - a \Big)^2$$
    where $\mathbf{J}_w$ is the Jacobian
    $\mathbf{J}_w = \frac{\partial g}{\partial w}$,
    $w$ are sampled from $w \in \mathcal{W}$ from the mapping network, and
    $y$ are images with noise $\mathcal{N}(0, \mathbf{I})$.
    $a$ is the exponential moving average of $\Vert \mathbf{J}^\top_{w} y \Vert_2$
    as the training progresses.
    $\mathbf{J}^\top_{w} y$ is calculated without explicitly calculating the Jacobian using
    $$\mathbf{J}^\top_{w} y = \nabla_w \big(g(w) \cdot y \big)$$
    """

    def __init__(self, beta: float):
        """
        * `beta` is the constant $\beta$ used to calculate the exponential moving average $a$
        """
        super().__init__()

        # $\beta$
        self.beta = beta
        # Number of steps calculated $N$
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)
        # Exponential sum of $\mathbf{J}^\top_{w} y$
        # $$\sum^N_{i=1} \beta^{(N - i)}[\mathbf{J}^\top_{w} y]_i$$
        # where $[\mathbf{J}^\top_{w} y]_i$ is the value of it at $i$-th step of training
        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w: torch.Tensor, x: torch.Tensor):
        """
        * `w` is the batch of $w$ of shape `[batch_size, d_latent]`
        * `x` are the generated images of shape `[batch_size, 3, height, width]`
        """
        # Get the device
        device = x.device
        # Get number of pixels
        image_size = x.shape[2] * x.shape[3]
        # Calculate $y \in \mathcal{N}(0, \mathbf{I})$
        y = torch.randn(x.shape, device=device)
        # Calculate $\big(g(w) \cdot y \big)$ and normalize by the square root of image size.
        # This is scaling is not mentioned in the paper but was present in
        # [their implementation](https://github.com/NVlabs/stylegan2/blob/master/training/loss.py#L167).
        output = (x * y).sum() / np.sqrt(image_size)

        # Calculate gradients to get $\mathbf{J}^\top_{w} y$
        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)
        # Calculate L2-norm of $\mathbf{J}^\top_{w} y$
        #norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()
        norm = (gradients ** 2).mean(dim=1).sqrt()

        # Regularize after first step
        if self.steps > 0:
            # Calculate $a$
            # $$\frac{1}{1 - \beta^N} \sum^N_{i=1} \beta^{(N - i)}[\mathbf{J}^\top_{w} y]_i$$
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            # Calculate the penalty
            # $$\mathbb{E}_{w \sim f(z), y \sim \mathcal{N}(0, \mathbf{I})}
            # \Big(\Vert \mathbf{J}^\top_{w} y \Vert_2 - a \Big)^2$$
            loss = torch.mean((norm - a) ** 2)
        else:
            # Return a dummy loss if we can't calculate $a$
            loss = norm.new_tensor(0)

        # Calculate the mean of $\Vert \mathbf{J}^\top_{w} y \Vert_2$
        mean = norm.mean().detach()
        # Update exponential sum
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        # Increment $N$
        self.steps.add_(1.)

        # Return the penalty
        return loss

##################################################################################
# build feature network 
##################################################################################       
def Build_feature_extractor(mode, device=torch.device("cuda"), use_dataparallel=False):
    if torch.cuda.device_count() > 1:
            use_dataparallel=True
    if mode == "legacy_pytorch":
        feat_model = feature_extractor(name="pytorch_inception", resize_inside=False, device=device, use_dataparallel=use_dataparallel)
    elif mode == "legacy_tensorflow":
        feat_model = feature_extractor(name="torchscript_inception", resize_inside=True, device=device, use_dataparallel=use_dataparallel)
    elif mode == "clean":
        feat_model = feature_extractor(name="torchscript_inception", resize_inside=False, device=device, use_dataparallel=use_dataparallel)
    return feat_model

def feature_extractor(name, device=torch.device("cuda"), resize_inside=False, use_dataparallel=False, size=128):
    if name == "pytorch_inception":
        model = InceptionV3(output_blocks=[3], resize_input=False).to(device)
        model.eval()
        if use_dataparallel:
            model = torch.nn.DataParallel(model)
        def model_fn(x): return model(x)[0].squeeze(-1).squeeze(-1)
    elif name == "Custom_VGG":
        model = Custom_VGG(ipt_size=(size,size), pretrained=True).to(device)
        model.eval()
        if use_dataparallel:
            model = torch.nn.DataParallel(model)
        def model_fn(x): return model(x)[0].squeeze(-1).squeeze(-1)
    else:
        raise ValueError(f"{name} feature extractor not implemented")
    return model_fn
##################################################################################
# VGG network
##################################################################################
from torchvision.models.vgg import model_urls
VGG_TYPES = {'vgg11' : torchvision.models.vgg11, 
             'vgg11_bn' : torchvision.models.vgg11_bn, 
             'vgg13' : torchvision.models.vgg13, 
             'vgg13_bn' : torchvision.models.vgg13_bn, 
             'vgg16' : torchvision.models.vgg16, 
             'vgg16_bn' : torchvision.models.vgg16_bn,
             'vgg19_bn' : torchvision.models.vgg19_bn, 
             'vgg19' : torchvision.models.vgg19}

class Custom_VGG(nn.Module):
    def __init__(self,
                 ipt_size=(128, 128), 
                 pretrained=True, 
                 vgg_type='vgg16', 
                 num_classes=1000):
        super(Custom_VGG, self).__init__()

        # load convolutional part of vgg
        assert vgg_type in VGG_TYPES, "Unknown vgg_type '{}'".format(vgg_type)
        vgg_loader = VGG_TYPES[vgg_type]
        vgg = vgg_loader(pretrained=pretrained)
        self.features = vgg.features

    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return x

##################################################################################
# Normalization layers (MUNIT)
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias