# DMT &mdash; Official PyTorch implementation

> **A Diffusion Model Translator for Efficient Image-to-Image Translation (TPAMI 2024)** <br>
> Mengfei Xia, Yu Zhou, Ran Yi, Yong-Jin Liu, Wenping Wang <br>

<!-- [[Paper](https://arxiv.org/pdf/2311.18208)] -->

Abstract: *Applying diffusion models to image-to-image translation (I2I) has recently received increasing attention due to its practical applications. Previous attempts inject information from the source image into each denoising step for an iterative refinement, thus resulting in a time-consuming implementation. We propose an efficient method that equips a diffusion model with a lightweight translator, dubbed a Diffusion Model Translator (DMT), to accomplish I2I. Specifically, we first offer theoretical justification that in employing the pioneering DDPM work for the I2I task, it is both feasible and sufficient to transfer the distribution from one domain to another only at some intermediate step. We further observe that the translation performance highly depends on the chosen timestep for domain transfer, and therefore propose a practical strategy to automatically select an appropriate timestep for a given task. We evaluate our approach on a range of I2I applications, including image stylization, image colorization, segmentation to image, and sketch to image, to validate its efficacy and general utility. The comparisons show that our DMT surpasses existing methods in both quality and efficiency.*

## TODO List

- [ ] Release inference code.
- [ ] Release training code.

## References

If you find the code useful for your research, please consider citing

```bib
@article{xia2024dmt,
  title={A Diffusion Model Translator for Efficient Image-to-Image Translation},
  author={Xia, Mengfei and Zhou, Yu and Yi, Ran and Liu, Yong-Jin and Wang, Wenping},
  booktitle={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2024},
}
```

## LICENSE

The project is under [MIT License](./LICENSE), and is for research purpose ONLY.
