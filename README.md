# DMT &mdash; Official PyTorch implementation

> **A Diffusion Model Translator for Efficient Image-to-Image Translation (TPAMI 2024)** <br>
> Mengfei Xia, Yu Zhou, Ran Yi, Yong-Jin Liu, Wenping Wang <br>

[[Paper](https://ieeexplore.ieee.org/document/10614866)]

Abstract: *Applying diffusion models to image-to-image translation (I2I) has recently received increasing attention due to its practical applications. Previous attempts inject information from the source image into each denoising step for an iterative refinement, thus resulting in a time-consuming implementation. We propose an efficient method that equips a diffusion model with a lightweight translator, dubbed a Diffusion Model Translator (DMT), to accomplish I2I. Specifically, we first offer theoretical justification that in employing the pioneering DDPM work for the I2I task, it is both feasible and sufficient to transfer the distribution from one domain to another only at some intermediate step. We further observe that the translation performance highly depends on the chosen timestep for domain transfer, and therefore propose a practical strategy to automatically select an appropriate timestep for a given task. We evaluate our approach on a range of I2I applications, including image stylization, image colorization, segmentation to image, and sketch to image, to validate its efficacy and general utility. The comparisons show that our DMT surpasses existing methods in both quality and efficiency.*

## Installation

This repository is developed based on [TSIT](https://github.com/EndlessSora/TSIT), where you can find more detailed instructions on installation. We replace the version of `torch` to support diffusion models. We summarize the necessary steps to facilitate reproduction.

1. Environment: CUDA version == 11.1.

2. Install package requirements with `conda`:

    ```shell
    conda create -n dmt python=3.8  # create virtual environment with Python 3.8
    conda activate dmt
    pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 --extra-index-url https://download.pytorch.org/whl/cu111
    pip install -r requirements.txt -f https://download.pytorch.org/whl/cu111/torch_stable.html
    pip install protobuf==3.20
    pip install absl-py einops ftfy==6.1.1 
    ```

3. Copy `dmt_utils` to the `TSIT` folder

    ```shell
    cp -r dmt_utils TSIT/
    ```

## Inference and Training

For a quick start, we have provided example [test script](TSIT/test_scripts/dmt_colorization.sh) and [train script](TSIT/train_scripts/dmt_colorization.sh) for colorization task using TSIT and ADM. One can easily modify the preset timesteps for DMT using argument `--timestep_s` and `--timestep_t`. Please check the scripts for more details.

## Customization

To customize DMT on other datasets or backbones is quite simple.

- For customization on other dataset using TSIT and ADM, it is required to implement the dataset class resembling [this](TSIT/data/colorization_dataset.py), the pre-trained ADM checkpoint, and the customized scripts with preset timesteps.
- For customization on other backbones, it is necessary to finish all steps below:
    1. Modifies the translator (*i.e.*, generator in common sense) **WITH ONLY** $L_2$ loss similar to [this](TSIT/models/dmt_model.py);
    2. Attaches diffusers to the translator, implements as below the pre- and post-processes of adding noise on paired data before training and synthesized data after inference, respectively;

        ```python
        # Pre-process to add noise on paired data before training.
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

        # Post-process to add noise on synthesized data after inference.
        def forward(self, data, mode):
        noisy_input_semantics, noisy_real_image = self.preprocess_input(data)
        if mode == 'generator':
            # Omits code block.
            pass
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
        ```

    3. Implements the function which denoises an image from a specified timestep $t$ to 0.

        ```python
        def ddim_sample_from_t_loop(
            self,
            model,
            x_t,
            timestep_t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
        ):
            # Code blocks.
            pass
        ```

## References

If you find the code useful for your research, please consider citing

```bib
@article{xia2024dmt,
  title={A Diffusion Model Translator for Efficient Image-to-Image Translation},
  author={Xia, Mengfei and Zhou, Yu and Yi, Ran and Liu, Yong-Jin and Wang, Wenping},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2024},
}
```

## LICENSE

The project is under [MIT License](./LICENSE), and is for research purpose ONLY.

## Acknowledgements

We highly appreciate [TSIT](https://github.com/EndlessSora/TSIT) and [ADM](https://github.com/openai/guided-diffusion) for their contributions to the community.
