#!/usr/bin/bash
# Fine tuning 

# Basic parameters seed = [0, 101, 202, 303, 404]
seed="0"
batch_size=(16)
accum_iter=(1)

epochs="50"
warmup_epochs="5"

# Model parameters
input_channels="1"
input_electrodes="12"
time_steps="2500"
model_size="tiny"
model="vit_"$model_size"_patchX"

patch_height="1"
patch_width=(100)

# Augmentation parameters
jitter_sigma="0.2"
rescaling_sigma="0.5"
ft_surr_phase_noise="0.075"

drop_path=(0.05)
layer_decay="0.75"

# Optimizer parameters
blr=(1e-6)
weight_decay=(0.1)

# Criterion parameters
smoothing=(0.2)

# Dataset parameters
data_path="/home/oturgut/sprai/data/preprocessed/ecg/data_train_CAD_noBase_gn.pt"
labels_path="/home/oturgut/sprai/data/preprocessed/ecg/labels_train_CAD.pt"
nb_classes="2"

global_pool="False"
num_workers="32"

# Log specifications
save_output="False"
wandb="True"
wandb_project="MAE_ECG_Fin"

# Pretraining specifications
pre_batch_size=(128)
pre_blr=(1e-5)

# EVALUATE
eval="False"
# As filename: State the checkpoint for the inference of a specific model
# or state the (final) epoch for the inference of all models up to this epoch
#resume="/home/oturgut/sprai/mae_he/mae/output/fin/"$folder"/id/"$subfolder"/fin_b"$(($batch_size*$accum_iter))"_blr"$blr"_"$pre_data"/checkpoint-89.pth"

for bs in "${batch_size[@]}"
do
    for acc_it in "${accum_iter[@]}"
    do
        for lr in "${blr[@]}"
        do

            for dp in "${drop_path[@]}"
            do 
                for p_width in "${patch_width[@]}"
                do
                    for pre_bs in "${pre_batch_size[@]}"
                    do

                        folder="ecg/noExternal"
                        subfolder=(""$model_size"/1d/t2500/p"$patch_height"x"$p_width"/wd0.15/m0.75")

                        pre_data="b"$pre_bs"_blr"$pre_blr
                        finetune="/home/oturgut/sprai/mae_he/mae/output/pre/"$folder"/"$subfolder"/pre_"$pre_data"/checkpoint-399.pth"
                        # finetune="/home/oturgut/ECGMultimodalContrastiveLearning/oezguen/mm_v92_mae_checkpoint.pth"
                        # finetune="/home/oturgut/ECGMultimodalContrastiveLearning/pretrained_checkpoints/tiny/checkpoint-399.pth"

                        output_dir="/home/oturgut/sprai/mae_he/mae/output/fin/"$folder"/"$subfolder"/fin_b"$(($bs*$acc_it))"_blr"$lr"_"$pre_data
                        log_dir="/home/oturgut/sprai/mae_he/mae/logs/fin/"$folder"/"$subfolder"/fin_b"$(($bs*$acc_it))"_blr"$lr"_"$pre_data

                        # resume="/home/oturgut/sprai/mae_he/mae/output/fin/"$folder"/"$subfolder"/fin_b"$bs"_blr"$lr"_"$pre_data"/checkpoint-78.pth"

                        cmd="python3 main_finetune.py --finetune $finetune --seed $seed --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $p_width --model $model --batch_size $bs --epochs $epochs --accum_iter $acc_it --drop_path $dp --weight_decay $weight_decay --layer_decay $layer_decay --blr $lr --warmup_epoch $warmup_epochs --smoothing $smoothing --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --log_dir $log_dir --num_workers $num_workers"

                        if [ "$global_pool" == "True" ]; then
                            cmd=$cmd" --global_pool"
                        fi

                        if [ "$wandb" = "True" ]; then
                            cmd=$cmd" --wandb --wandb_project $wandb_project"
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