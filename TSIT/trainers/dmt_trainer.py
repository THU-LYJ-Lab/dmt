import torch

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.dmt_model import DMTModel


class DMTTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.dmt_model = DMTModel(opt)
        if len(opt.gpu_ids) > 0:
            self.dmt_model = DataParallelWithCallback(self.dmt_model,
                                                      device_ids=opt.gpu_ids)
            self.dmt_model_on_one_gpu = self.dmt_model.module
        else:
            self.dmt_model_on_one_gpu = self.dmt_model

        self.generated = None
        if opt.isTrain:
            self.optimizer_G = self.dmt_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.dmt_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    def get_latest_losses(self):
        return {**self.g_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.dmt_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
            else:
                new_lr_G = new_lr / 2

            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
