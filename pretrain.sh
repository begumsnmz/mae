#!/usr/bin/bash
# pretraining

# Basic parameters
external="off"

if [ "$external" = "on" ]; then
    batch_size="32"
    accum_iter=(4)
else
    batch_size="4"
    accum_iter=(1)
fi
epochs="750"
warmup_epochs="75"

# Model parameters
input_channels="6"
input_electrodes="65"
time_steps="37000"
model="mae_vit_tiny_patchX"

patch_height="65"
patch_width=(50)

# Augmentation parameters
mask_ratio="0.75"

jitter_sigma="0.05"
rescaling_sigma="0.5"
ft_surr_phase_noise="0.1"
crop_lbd="0.9"

# Optimizer parameters
blr_array=(1e-3)
weight_decay="0.1"

# Dataset parameters
data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_DINH_701515_nf_cw_bw_fs200.pt"
labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_DINH_701515.pt"

transfer_data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_SEED_cw_bw_fs200.pt"
transfer_labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_SEED.pt"

num_workers="32"

# Log specifications
save_output="True"
wandb="True"

# Checkpoints
# resume="/home/oturgut/PyTorchEEG/mae_he/mae/output/pre/moim/0.6train/10fold/decomposed/b2048/pre_moim_b"$(($batch_size*$accum_iter))"_blr"$blr_array"/checkpoint-20.pth"


for blr in "${blr_array[@]}"
do
    for acc_it in "${accum_iter[@]}"
    do
        for pw in "${patch_width[@]}"
        do

            if [ "$external" = "on" ]; then
                folder="seed"
            else
                folder="noExternal"
            fi

            pre_data="pre_"$folder"_b"$(($batch_size*$acc_it))"_blr"$blr

            subfolder="tiny/2d/t"$time_steps"/p"$patch_height"x"$pw"/m"$mask_ratio"/wd"$weight_decay
            output_dir="./output/pre/"$folder"/"$subfolder"/"$pre_data
            log_dir="./logs/pre/"$folder"/"$subfolder"/"$pre_data
        
            cmd="python3 main_pretrain.py --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --crop_lbd $crop_lbd --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $pw --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mask_ratio --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --labels_path $labels_path --log_dir $log_dir --num_workers $num_workers"
            
            if [ "$wandb" = "True" ]; then
                cmd=$cmd" --wandb"
            fi

            if [ "$external" = "on" ]; then
                cmd=$cmd" --transfer_data_path $transfer_data_path --transfer_labels_path $transfer_labels_path"
            fi

            if [ "$save_output" = "True" ]; then
                cmd=$cmd" --output_dir $output_dir"
            fi

            echo $cmd && $cmd

        done
    done
done