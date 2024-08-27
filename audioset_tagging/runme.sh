#!/bin/bash

DATASET_DIR="/home/luisa/Documents/audioset_tagging_cnn/dataset"
WORKSPACE="/home/luisa/Documents/audioset_tagging_cnn/"


PRETRAINED_CHECKPOINT_PATH="/home/luisa/Documents/audioset_tagging_cnn/weights/Cnn14_mAP=0.431.pth"

CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --early_stop=10000 --workspace=$WORKSPACE --holdout_fold=1 --model_type="Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_bce --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000 --cuda

MODEL_TYPE="Cnn14"
PRETRAINED_CHECKPOINT_PATH="/vol/vssp/msos/qk/bytedance/workspaces_important/pub_audioset_tagging_cnn_transfer/checkpoints/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/660000_iterations.pth"
python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --early_stop=10000 --holdout_fold=1 --model_type=$MODEL_TYPE --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --freeze_base --loss_type=clip_bce --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --few_shots=10 --random_seed=1000 --resume_iteration=0 --stop_iteration=10000 --cuda
