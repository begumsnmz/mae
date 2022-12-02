#!/usr/bin/bash
# Inference on the finetuned model 

# Basic parameters
batch_size=(8)
accum_iter=(1)

epochs="250"
warmup_epochs="25"

# Model parameters
input_channels="6"
input_electrodes="65"
time_steps="55000"
model="vit_tiny_patchX"

patch_height="65"
patch_width="50"

# Augmentation parameters
crop_lbd="0.65"

drop_path=(0.1)
layer_decay="0.75"

# Optimizer parameters
blr=(1e-4)
weight_decay=(0.1)

# Criterion parameters
smoothing=(0.2)

# Dataset parameters
data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_DINH_701515_nf_cw_bw_fs200.pt"
labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_DINH_701515.pt"
nb_classes="2"

global_pool="True"
num_workers="32"

# Log specifications
wandb="False"

# Pretraining specifications
pre_batch_size=(4)
pre_blr=(1e-3)

folder="noExternal"
subfolder=("tiny/2d/t37000/p65x50/m0.75/wd0.1/crop0.9")

pre_data=$folder"_b"$pre_batch_size"_blr"$pre_blr

log_dir="./logs/fin/"$folder"/"$subfolder"/fin_b"$(($batch_size*$accum_iter))"_blr"$blr"_"$pre_data

# As filename: State the checkpoint for the inference of a specific model
# or state the (final) epoch for the inference of all models up to this epoch
resume="/home/oturgut/PyTorchEEG/mae_he/mae/output/fin/"$folder"/"$subfolder"/fin_b"$(($batch_size*$accum_iter))"_blr"$blr"_"$pre_data"/checkpoint-1.pth"
cmd="python3 main_finetune.py --eval --resume $resume --crop_lbd $crop_lbd --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $batch_size --epochs $epochs --accum_iter $accum_iter --drop_path $drop_path --weight_decay $weight_decay --layer_decay $layer_decay --blr $blr --warmup_epoch $warmup_epochs --smoothing $smoothing --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --log_dir $log_dir --num_workers $num_workers"

if [ "$global_pool" == "True" ]; then
    cmd=$cmd" --global_pool"
fi

if [ "$wandb" = "True" ]; then
    cmd=$cmd" --wandb"
fi

echo $cmd && $cmd