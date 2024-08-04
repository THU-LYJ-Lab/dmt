# python3.8
"""Define the DMT model built upon TSIT."""

import torch

import models.networks as networks
import util.util as util
from dmt_utils.script_util import create_gaussian_diffusion


class DMTModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        # Require only the generator.
        self.netG = self.initialize_networks(opt)

        # Define the diffusion process.
        self.diffuser = create_gaussian_diffusion(**opt.diffuser_kwargs)

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        noisy_input_semantics, noisy_real_image = self.preprocess_input(data)
        if mode == 'generator':
            g_loss, noisy_generated = self.compute_generator_loss(
                noisy_input_semantics, noisy_real_image)
            return g_loss, noisy_generated
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(noisy_input_semantics,
                                                         noisy_real_image)
                batch_size = fake_image.shape[0]
                t = (torch.ones((batch_size,), dtype=torch.long).cuda()
                     * self.opt.timestep_t)
                noise = torch.randn_like(fake_image)
                noisy_fake_image = self.diffuser.q_sample(x_start=fake_image,
                                                          t=t,
                                                          noise=noise)
            return noisy_fake_image
        else:
            raise ValueError("|mode| is invalid")

    # Only optimizer for generator.
    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr = opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr = opt.lr / 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))

        return optimizer_G

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)

        return netG

    # preprocess the input, such as moving the tensors to GPUs
    # and transforming the label map to one-hot encoding (for SIS)
    # |data|: dictionary of the input data
    def preprocess_input(self, data):
        # move to GPU and change data types
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['image'] = data['image'].cuda()

        input_semantics = data['label']
        noise = torch.randn_like(input_semantics)
        batch_size = input_semantics.shape[0]
        s = (torch.ones((batch_size,), dtype=torch.long).cuda()
             * self.opt.timestep_s)
        t = (torch.ones((batch_size,), dtype=torch.long).cuda()
             * self.opt.timestep_t)

        noisy_input_semantics = self.diffuser.q_sample(x_start=input_semantics,
                                                       t=s,
                                                       noise=noise)
        noisy_image = self.diffuser.q_sample(x_start=data['image'],
                                             t=t,
                                             noise=noise)
        
        return noisy_input_semantics, noisy_image

    def compute_generator_loss(self, noisy_content, noisy_style):
        G_losses = {}

        noisy_fake_image, L2_loss = self.generate_fake(noisy_content,
                                                       noisy_style)
        G_losses['l2'] = L2_loss

        return G_losses, noisy_fake_image

    def generate_fake(self, noisy_input_semantics, noisy_real_image):
        z = None

        noisy_fake_image = self.netG(noisy_input_semantics,
                                     noisy_real_image,
                                     z=z)
        L2_loss = (noisy_fake_image - noisy_real_image).square().mean()

        return noisy_fake_image, L2_loss

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
