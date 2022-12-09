#!/usr/bin/bash
# pretraining

# Basic parameters
external="off"

if [ "$external" = "on" ]; then
    batch_size="32"
    accum_iter=(4)
else
    batch_size="128"
    accum_iter=(1)
fi
epochs="400"
warmup_epochs="40"

# Model parameters
input_channels="1"
input_electrodes="12"
time_steps="2000"
model_size="tiny"
model="mae_vit_"$model_size"_patchX"

patch_height="1"
patch_width=(100)

# Augmentation parameters
mask_ratio="0.75"

jitter_sigma="0.2"
rescaling_sigma="0.5"
ft_surr_phase_noise="0.075"

# Optimizer parameters
blr_array=(1e-4)
weight_decay="0.05"

# Dataset parameters
data_path="/home/oturgut/sprai/data/preprocessed/ecg/data_train_CAD_noBase_gn.pt"
labels_path="/home/oturgut/sprai/data/preprocessed/ecg/labels_train_CAD.pt"

transfer_data_path=""
transfer_labels_path=""

num_workers="32"

# Log specifications
save_output="True"
wandb="True"

# Checkpoints
resume_from_ckpt="False"
# resume="/home/oturgut/sprai/mae_he/mae/output/pre/noExternal/tiny/2d/t37000/p65x50/m0.75/pre_noExternal_b"$(($batch_size*$accum_iter))"_blr"$blr_array"/checkpoint-450.pth"


for blr in "${blr_array[@]}"
do
    for acc_it in "${accum_iter[@]}"
    do
        for pw in "${patch_width[@]}"
        do

            if [ "$external" = "on" ]; then
                folder="ecg/-"
            else
                folder="ecg/noExternal"
            fi

            pre_data="pre_b"$(($batch_size*$acc_it))"_blr"$blr

            subfolder=$model_size"/1d/t"$time_steps"/p"$patch_height"x"$pw"/m"$mask_ratio
            output_dir="./output/pre/"$folder"/"$subfolder"/"$pre_data
            log_dir="./logs/pre/"$folder"/"$subfolder"/"$pre_data
        
            cmd="python3 main_pretrain.py --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $pw --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mask_ratio --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --labels_path $labels_path --log_dir $log_dir --num_workers $num_workers"
            
            if [ "$external" = "on" ]; then
                cmd=$cmd" --transfer_data_path $transfer_data_path --transfer_labels_path $transfer_labels_path"
            fi

            if [ "$wandb" = "True" ]; then
                cmd=$cmd" --wandb"
            fi

            if [ "$save_output" = "True" ]; then
                cmd=$cmd" --output_dir $output_dir"
            fi

            if [ "$resume_from_ckpt" = "True" ]; then
                cmd=$cmd" --resume $resume"
            fi

            echo $cmd && $cmd

        done
    done
done