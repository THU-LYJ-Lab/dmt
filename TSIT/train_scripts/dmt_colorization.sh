set -x

TIMESTEP_S=$1
TIMESTEP_T=$2
GPU=$3
CROOT=$4
SROOT=$5

NAME='dmt_s'$TIMESTEP_S'_t'$TIMESTEP_T'_colorization'
TASK='DMT'
DATA='colorization'
CKPTROOT='./checkpoints'
WORKER=4

python train_dmt.py \
    --name $NAME \
    --task $TASK \
    --num_timesteps 1000 \
    --noise_schedule linear \
    --timestep_s $TIMESTEP_S \
    --timestep_t $TIMESTEP_T \
    --learn_sigma \
    --gpu_ids $GPU \
    --checkpoints_dir $CKPTROOT \
    --batchSize 1 \
    --dataset_mode $DATA \
    --croot $CROOT \
    --sroot $SROOT \
    --nThreads $WORKER \
    --no_instance \
    --gan_mode hinge \
    --num_upsampling_layers more \
    --alpha 1.0 \
    --display_freq 200 \
    --save_epoch_freq 5 \
    --niter 60 \
    --no_ganFeat_loss \
    --no_vgg_loss
