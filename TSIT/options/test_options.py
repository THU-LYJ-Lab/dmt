from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--show_input', action='store_true', help='show input images with the synthesized image')

        # for DMT
        parser.add_argument('--diffusion_path', type=str, default=None,
                            help='Path to pre-traiend DPM.')
        parser.add_argument('--denoise_step', type=int, default=40,
                            help='Timestep to diffuse.')
        parser.add_argument('--image_size', type=int, default=256,
                            help='Image size for DPM.')
        parser.add_argument('--num_channels', type=int, default=256,
                            help='Number of channels for DPM.')
        parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='Number of resnet blocks for DPM.')
        parser.add_argument('--channel_mult', type=str, default='',
                            help='Channel multiplication for DPM.')
        parser.add_argument('--class_cond', action='store_true', default=False,
                            help='Whether to use class condition for DPM.')
        parser.add_argument('--attention_resolutions', type=str,
                            default='32,16,8', help='Attn res for DPM.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of heads for DPM.')
        parser.add_argument('--num_head_channels', type=int, default=64,
                            help='Number of head channels for DPM.')
        parser.add_argument('--num_heads_upsample', type=int, default=-1,
                            help='Number of head upsample for DPM.')
        parser.add_argument('--use_scale_shift_norm', action='store_true',
                            default=False,
                            help='Whether to use scale shift norm for DPM.')
        parser.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout for DPM.')
        parser.add_argument('--resblock_updown', action='store_true',
                            default=False, help='Parameter for DPM.')
        parser.add_argument('--use_fp16', action='store_true', default=False,
                            help='Whether to use fp16 for DPM.')
        parser.add_argument('--use_new_attention_order', action='store_true',
                            default=False, help='Parameter for DPM.')

        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser
