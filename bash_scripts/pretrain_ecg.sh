#!/usr/bin/bash
# Pre-training

# Basic parameters
seed="0"
batch_size="128"
accum_iter=(1)

epochs="400"
warmup_epochs="40"

# Callback parameters
patience="-1"
max_delta="0.00"

# Model parameters
input_channels="1"
input_electrodes="12"
time_steps="2500"
model_size="tiny"
model="mae_vit_"$model_size"_patchX"

patch_height="1"
patch_width=(100)

norm_pix_loss="False"

# Augmentation parameters
mask_ratio=(0.8)

jitter_sigma="0.25"
rescaling_sigma="0.5"
ft_surr_phase_noise="0.1"

# Optimizer parameters
blr_array=(1e-5)
weight_decay=(0.15)

# Data path
path="tower"
if [ "$path" = "tower" ]; then
    data_base="/home/oturgut/sprai/data/preprocessed"
    checkpoint_base="/home/oturgut"
else
    data_base="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects"
    checkpoint_base="/home/guests/oezguen_turgut"
fi

# Dataset parameters
data_path=$data_base"/ecg/ecgs_train_ecg_imaging_noBase_gn.pt"
val_data_path=$data_base"/ecg/ecgs_val_ecg_imaging_noBase_gn.pt"

num_workers="32"

# Log specifications
save_output="True"
wandb="True"
wandb_project="MAE_ECG_Pre"

# Checkpoints
resume=$checkpoint_base"/sprai/mae_he/mae/output/pre/ecg/seed0/tiny/t2500/p1x100/wd0.15/m0.8/pre_b"$(($batch_size*$accum_iter))"_blr"$blr_array"/checkpoint-260-ncc-0.95.pth"

for blr in "${blr_array[@]}"
do
    for acc_it in "${accum_iter[@]}"
    do
        for mr in "${mask_ratio[@]}"
        do

            pre_data="pre_b"$(($batch_size*$acc_it))"_blr"$blr

            folder="ecg"
            subfolder="seed$seed/$model_size/t$time_steps/p$patch_height"x"$patch_width/wd$weight_decay/m$mr"

            output_dir=$checkpoint_base"/sprai/mae_he/mae/output/pre/"$folder"/"$subfolder"/"$pre_data
            log_dir=$checkpoint_base"/sprai/mae_he/mae/logs/pre/"$folder"/"$subfolder"/"$pre_data
        
            cmd="python3 main_pretrain.py --seed $seed --patience $patience --max_delta $max_delta --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mr --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --val_data_path $val_data_path --log_dir $log_dir --num_workers $num_workers"

            if [ "$norm_pix_loss" = "True" ]; then
                cmd=$cmd" --norm_pix_loss"
            fi

            if [ "$wandb" = "True" ]; then
                cmd=$cmd" --wandb --wandb_project $wandb_project"
            fi

            if [ "$save_output" = "True" ]; then
                cmd=$cmd" --output_dir $output_dir"
            fi

            if [ ! -z "$resume" ]; then
                cmd=$cmd" --resume $resume"
            fi

            echo $cmd && $cmd

        done
    done
done