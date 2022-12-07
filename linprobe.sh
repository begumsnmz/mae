#!/usr/bin/bash
# Linear probing

# Basic parameters
batch_size=(8)
accum_iter=(1)

epochs="90"
warmup_epochs="9"

# Model parameters
input_channels="6"
input_electrodes="65"
time_steps="55000"
model_size="tiny"
model="vit_"$model_size"_patchX"

patch_height="65"
patch_width="50"

# Augmentation parameters
jitter_sigma="0.05"
rescaling_sigma="0.5"
ft_surr_phase_noise="0.1"
crop_lbd="0.9"

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
save_output="True"
wandb="True"

folder="noExternal"
subfolder=($model_size"/2d/t37000/p"$patch_height"x"$patch_width"/m0.75")

# Pretraining specifications
pre_batch_size=(4)
pre_blr=(1e-3)

pre_data=$folder"_b"$pre_batch_size"_blr"$pre_blr
finetune="/home/oturgut/PyTorchEEG/mae_he/mae/output/pre/"$folder"/"$subfolder"/pre_"$pre_data"/checkpoint-50.pth"

for bs in "${batch_size[@]}"
do
    for acc_it in "${accum_iter[@]}"
    do
        for lr in "${blr[@]}"
        do

            for wd in "${weight_decay[@]}"
            do
                for smth in "${smoothing[@]}"
                do

                    output_dir="./output/lin/"$folder"/"$subfolder"/lin_b"$(($bs*$acc_it))"_blr"$lr"_"$pre_data
                    log_dir="./logs/lin/"$folder"/"$subfolder"/lin_b"$(($bs*$acc_it))"_blr"$lr"_"$pre_data

                    # resume="/home/oturgut/PyTorchEEG/mae_he/mae/output/lin/"$folder"/"$subfolder"/lin_b"$bs"_blr"$lr"_"$pre_data"/checkpoint-78.pth"

                    cmd="python3 main_linprobe.py --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --crop_lbd $crop_lbd --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --accum_iter $acc_it --weight_decay $wd --layer_decay $layer_decay --blr $lr --warmup_epoch $warmup_epochs --smoothing $smth --finetune $finetune --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --log_dir $log_dir --num_workers $num_workers"

                    if [ "$global_pool" == "True" ]; then
                        cmd=$cmd" --global_pool"
                    fi

                    if [ "$wandb" = "True" ]; then
                        cmd=$cmd" --wandb"
                    fi

                    if [ "$save_output" = "True" ]; then
                        cmd=$cmd" --output_dir $output_dir"
                    fi
                    
                    echo $cmd && $cmd

                done
            done

        done
    done
done