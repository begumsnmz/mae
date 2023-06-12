#!/usr/bin/bash
# Pre-training

# Basic parameters
seed="0"
batch_size="32"
accum_iter=(4)

epochs="400"
warmup_epochs="40"

# Callback parameters
patience="-1"
max_delta="0.00"

# Model parameters
input_channels="1"
input_electrodes="65"
time_steps="1000"
model_size="tiny"
model="mae_vit_"$model_size"_patchX"

patch_height="1"
patch_width=(25)

norm_pix_loss="False"

ncc_weight=0.1

# Augmentation parameters
mask_ratio=(0.75)

jitter_sigma="0.25"
rescaling_sigma="0.25"
ft_surr_phase_noise="0.1"

# Optimizer parameters
blr_array=(3e-5)
weight_decay=(0.15)

# Data path
path="tower"
if [ "$path" = "tower" ]; then
    data_base="/home/oturgut/sprai/data/preprocessed"
    checkpoint_base="/home/oturgut"
else
    data_base="/vol/aimspace/projects/ukbb/cardiac/cardiac_segmentations/projects"
    checkpoint_base="/vol/aimspace/users/tuo"
fi

# Dataset parameters
data_path=$data_base"/eeg/data_LEMONSEEDDINHHEITMANN_bw45_cw_clamped_fs100.pt"
val_data_path=$data_base"/eeg/data_LEMON_val_ec_bw45_cw_clamped_fs100.pt"

num_workers="8"

# Log specifications
save_output="True"
wandb="True"
wandb_project="MAE_EEG_Pre"
wandb_id=""

for blr in "${blr_array[@]}"
do
    for acc_it in "${accum_iter[@]}"
    do
        for mr in "${mask_ratio[@]}"
        do

            pre_data="pre_b"$(($batch_size*$acc_it))"_blr"$blr

            folder="eeg/LEMONSEEDDINHHEITMANN/lp_hp"
            subfolder="ncc_weight$ncc_weight/seed$seed/$model_size/t$time_steps/p$patch_height"x"$patch_width/wd$weight_decay/m$mr"

            output_dir=$checkpoint_base"/sprai/mae_he/mae/output/pre/"$folder"/"$subfolder"/"$pre_data
            log_dir=$checkpoint_base"/sprai/mae_he/mae/logs/pre/"$folder"/"$subfolder"/"$pre_data

            # resume=$checkpoint_base"/sprai/mae_he/mae/output/pre/"$folder"/"$subfolder"/"$pre_data"/checkpoint-8-ncc-0.37.pth"
        
            cmd="python3 main_pretrain.py --seed $seed --patience $patience --max_delta $max_delta --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --ncc_weight $ncc_weight --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mr --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --val_data_path $val_data_path --log_dir $log_dir --num_workers $num_workers"

            if [ "$norm_pix_loss" = "True" ]; then
                cmd=$cmd" --norm_pix_loss"
            fi

            if [ "$wandb" = "True" ]; then
                cmd=$cmd" --wandb --wandb_project $wandb_project"
                if [ ! -z "$wandb_id" ]; then
                    cmd=$cmd" --wandb_id $wandb_id"
                fi
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