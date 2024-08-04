set -x

TIMESTEP_S=$1
TIMESTEP_T=$2
DENOISE_STEP=$3
GPU=$4
DIFFUSION_PATH=$5
CROOT=$6
SROOT=$7

NAME='dmt_s'$TIMESTEP_S'_t'$TIMESTEP_T'_colorization'
TASK='DMT'
DATA='colorization'
CKPTROOT='./checkpoints'
WORKER=4
RESROOT='./results'
EPOCH='60'

python test_dmt.py \
    --name $NAME \
    --task $TASK \
    --num_timesteps 1000 \
    --noise_schedule linear \
    --timestep_s $TIMESTEP_S \
    --timestep_t $TIMESTEP_T \
    --learn_sigma \
    --diffusion_path $DIFFUSION_PATH \
    --denoise_step $DENOISE_STEP \
    --image_size 256 \
    --num_channels 256 \
    --num_res_block 2 \
    --attention_resolutions 32,16,8 \
    --num_heads 4 \
    --num_head_channels 64 \
    --use_scale_shift_norm \
    --dropout 0.1 \
    --resblock_updown \
    --use_fp16 \
    --gpu_ids $GPU \
    --checkpoints_dir $CKPTROOT \
    --batchSize 1 \
    --dataset_mode $DATA \
    --croot $CROOT \
    --sroot $SROOT \
    --nThreads $WORKER \
    --no_instance \
    --num_upsampling_layers more \
    --alpha 1.0 \
    --results_dir $RESROOT \
    --which_epoch $EPOCH
