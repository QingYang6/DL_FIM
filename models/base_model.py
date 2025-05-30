import os,glob,re
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.isTrain = opt.isTrain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # get device name: CPU or GPU
        if torch.cuda.is_available():
            print("Use GPU!")
            for i in range(torch.cuda.device_count()):
                print(torch.cuda.get_device_name(i))
        else:
            print("Use CPU!")
        self.gpu_ids = self.device
        self.save_dir = opt.checkpoints_dir  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def set_input_test(self, input):
        pass

    @abstractmethod
    def sample_selfdistribution(self):
        """Run sample pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def sample(self):
        """Run sample pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def test_selfdistri(self):
        with torch.no_grad():
            if self.opt.model == 'MUNIT' or self.opt.model == 'BiCycleGAN' \
                or self.opt.model == 'MUNIT2S1C' or self.opt.model == 'MUNITSDG' \
                    or self.opt.model == 'MUNITUC' or self.opt.model == 'MUNITSDGLCC'\
                    or self.opt.model == 'MUNITUCCLE' or self.opt.model == 'MUNITSDGCRL'\
                    or self.opt.model == 'DSBicycle' or self.opt.model == 'MUNIT3XIA' \
                    or self.opt.model == 'MUNITTSG' or self.opt.model== 'MUNITFTSG' \
                    or self.opt.model ==  'FTSGINC' or self.opt.model == 'DS3XINC' or self.opt.model=='DS3XINCL1' \
                    or self.opt.model == 'TXINCL1' or self.opt.model == 'TXINC' or self.opt.model == 'TXINCL1WEI' \
                    or self.opt.model ==  'SDGCRL1' or self.opt.model == 'SDGCRL1X2D' or self.opt.model == 'SDGT' \
                    or self.opt.model == 'IGARSS' or self.opt.model == 'IGARSSSMALL' or self.opt.model == 'IGARSSNORS' \
                    or self.opt.model == 'CycleED':
                self.sample_selfdistribution()
            else:
                self.forward()
            self.compute_visuals()

    def setup(self, opt, total_iters=0):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if opt.continue_train:
            list_of_savednets = glob.glob(opt.checkpoints_dir + '/*net*')
            if len(list_of_savednets)>0:
                latest_net= max(list_of_savednets, key=os.path.getctime)
                total_iters = int(keepnumeric(os.path.split(latest_net)[1]))
                self.load_networks(total_iters)
            else:
                total_iters = 0
        elif opt.loadbyiter:
            self.load_networks(total_iters)
        self.print_networks(opt.verbose)
        return total_iters

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            if self.opt.model == 'MUNIT' or self.opt.model == 'BiCycleGAN' \
                or self.opt.model == 'MUNIT2S1C' or self.opt.model == 'MUNITSDG' \
                    or self.opt.model == 'MUNITUC' or self.opt.model == 'MUNITSDGLCC'\
                    or self.opt.model == 'MUNITUCCLE' or self.opt.model == 'MUNITSDGCRL'\
                    or self.opt.model == 'DSBicycle' or self.opt.model == 'MUNIT3XIA' \
                    or self.opt.model == 'MUNITTSG' or self.opt.model== 'MUNITFTSG' \
                    or self.opt.model ==  'FTSGINC' or self.opt.model == 'DS3XINC' or self.opt.model=='DS3XINCL1' \
                    or self.opt.model == 'TXINCL1' or self.opt.model == 'TXINC' or self.opt.model == 'TXINCL1WEI'\
                    or self.opt.model ==  'SDGCRL1' or self.opt.model == 'SDGCRL1X2D' or self.opt.model == 'SDGT' \
                    or self.opt.model == 'IGARSS' or self.opt.model == 'IGARSSSMALL' \
                    or self.opt.model == 'IGARSSNORS' \
                    or self.opt.model == 'CycleED':
                self.sample()
            else:
                self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if isinstance(net, torch.nn.DataParallel):
                    torch.save(net.module.state_dict(), save_path)
                    #net.cuda(self.gpu_ids) ?????
                else:
                    torch.save(net.state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

def keepnumeric(string):
    nstr = re.sub("[^0-9]", "", string)
    return nstr