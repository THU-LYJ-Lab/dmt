import os
from collections import OrderedDict
from functools import partial

import data
import torch
from options.test_options import TestOptions
from models.dmt_model import DMTModel
from util.visualizer import Visualizer
from util import html
from tqdm import tqdm
from dmt_utils.script_util import create_gaussian_diffusion
from dmt_utils.script_util import create_model


opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = DMTModel(opt)
model.eval()
# DMT
denoise_step = opt.denoise_step
timestep_t = opt.timestep_t
num_timesteps = opt.num_timesteps
assert timestep_t % denoise_step == 0
skip = timestep_t // denoise_step
assert num_timesteps % skip == 0
timestep_respacing = f'ddim{num_timesteps // skip}'
respaced_timestep_t = timestep_t // skip
diffuser = create_gaussian_diffusion(timestep_respacing=timestep_respacing,
                                     **opt.diffuser_kwargs)
unet = create_model(**opt.unet_kwargs)
state = torch.load(opt.diffusion_path, map_location='cpu')
unet.load_state_dict(state)
if opt.use_fp16:
    unet.convert_to_fp16()
unet.cuda().eval()
denoiser = partial(diffuser.ddim_sample_from_t_loop,
                   model=unet,
                   timestep_t=respaced_timestep_t)
visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
print('Number of images: ', len(dataloader))
num_images = 0
for i, data_i in enumerate(tqdm(dataloader)):
    if i * opt.batchSize >= opt.how_many:
        break

    noisy_generated = model(data_i, mode='inference')
    generated = denoiser(x_t=noisy_generated)

    for b in range(generated.shape[0]):
        # print(i, 'process image... %s' % img_path[b])
        if opt.show_input:
            visuals = OrderedDict([('content', data_i['label'][b]),
                                   ('style', data_i['image'][b]),
                                   ('synthesized_image', generated[b])])
        else:
            visuals = OrderedDict([('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, [f'{num_images}.png'])
        num_images += 1

webpage.save()
