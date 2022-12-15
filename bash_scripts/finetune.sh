#!/usr/bin/bash
# Fine tuning 

# Basic parameters
batch_size=(1)
accum_iter=(8)

epochs="100"
warmup_epochs="10"

# Model parameters
input_channels="6"
input_electrodes="65"
time_steps="37000"
model_size="small"
model="vit_"$model_size"_patchX"

patch_height="5"
patch_width="50"

# Augmentation parameters
jitter_sigma="0.2"
rescaling_sigma="0.5"
ft_surr_phase_noise="0.075"

drop_path=(0.1)
layer_decay="0.75"

# Optimizer parameters
blr=(3e-4)
weight_decay=(0.1)

# Criterion parameters
smoothing=(0.2)

# Dataset parameters
data_path="/home/oturgut/sprai/data/preprocessed/data_DINH_701515_nf_cw_id_fs200.pt"
labels_path="/home/oturgut/sprai/data/preprocessed/labels_DINH_701515.pt"
nb_classes="2"

global_pool="False"
num_workers="32"

# Log specifications
save_output="True"
wandb="True"

folder="noExternal"
subfolder=(""$model_size"/2d/t20000/p"$patch_height"x"$patch_width"/m0.75")

# Pretraining specifications
pre_batch_size=(4)
pre_blr=(1e-3)

pre_data="b"$pre_batch_size"_blr"$pre_blr
finetune="/home/oturgut/sprai/mae_he/mae/output/pre/"$folder"/"$subfolder"/pre_"$pre_data"/checkpoint-749.pth"

# EVALUATE
eval="False"
# As filename: State the checkpoint for the inference of a specific model
# or state the (final) epoch for the inference of all models up to this epoch
resume="/home/oturgut/sprai/mae_he/mae/output/fin/"$folder"/id/"$subfolder"/fin_b"$(($batch_size*$accum_iter))"_blr"$blr"_"$pre_data"/checkpoint-89.pth"

for bs in "${batch_size[@]}"
do
    for acc_it in "${accum_iter[@]}"
    do
        for lr in "${blr[@]}"
        do

            for dp in "${drop_path[@]}"
            do 
                for wd in "${weight_decay[@]}"
                do
                    for smth in "${smoothing[@]}"
                    do

                        output_dir="/home/oturgut/sprai/mae_he/mae/output/fin/"$folder"/id/"$subfolder"/fin_b"$(($bs*$acc_it))"_blr"$lr"_"$pre_data
                        log_dir="/home/oturgut/sprai/mae_he/mae/logs/fin/"$folder"/id/"$subfolder"/fin_b"$(($bs*$acc_it))"_blr"$lr"_"$pre_data

                        # resume="/home/oturgut/sprai/mae_he/mae/output/fin/"$folder"/"$subfolder"/fin_b"$bs"_blr"$lr"_"$pre_data"/checkpoint-78.pth"

                        cmd="python3 main_finetune.py --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --accum_iter $acc_it --drop_path $dp --weight_decay $wd --layer_decay $layer_decay --blr $lr --warmup_epoch $warmup_epochs --smoothing $smth --finetune $finetune --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --log_dir $log_dir --num_workers $num_workers"

                        if [ "$global_pool" == "True" ]; then
                            cmd=$cmd" --global_pool"
                        fi

                        if [ "$wandb" = "True" ]; then
                            cmd=$cmd" --wandb"
                        fi

                        if [ "$save_output" = "True" ]; then
                            cmd=$cmd" --output_dir $output_dir"
                        fi

                        if [ "$eval" = "True" ]; then
                            cmd=$cmd" --eval --resume $resume"
                        fi
                        
                        echo $cmd && $cmd

                    done
                done
            done

        done
    done
done