#!/usr/bin/bash
# hyperparameter tuning for finetuning

# HYPERPARAMETERS
batch_size=(32)
accum_iter=(1)
blr=(1e-2)

# FIXED PARAMETERS
epochs="50"
warmup_epochs="5"

input_channels="1"
input_electrodes="65"
time_steps="37000"
model="vit_base_patch20"
drop_path="0.1"

patch_height=$input_electrodes
patch_width="200"

weight_decay="0.05"
layer_decay="0.75"

smoothing="0.1" # label smoothing; changes the optimizer used

data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_TARGET_10fold_fs200.pt"
labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_2classes_TARGET_10fold_fs200.pt"
nb_classes="2"

# #### THIS IS ONLY FOR SEED
# data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_SEED_snippets6s_sub5exp1_5fold_fs200.pt"
# labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_3classes_SEED_snippets6s_sub5exp1_5fold_fs200.pt"
# nb_classes="3"

global_pool="False"
num_workers="32"

pre_batch_size=(384)
pre_blr=(1e-5)

folder="seed"
subfolder=("0.6train/10fold/1d")

for pre_bs in "${pre_batch_size[@]}"
do
    for pre_lr in "${pre_blr[@]}"
    do
        for subf in "${subfolder[@]}"
        do

            pre_data=$folder"_b"$pre_bs"_blr"$pre_lr
            if [ "$folder" = "onlySeed" ]; then
                finetune="/home/oturgut/PyTorchEEG/mae_he/mae/output/pre/"$folder"/"$subf"/pre_"$pre_data"/checkpoint-249.pth"
            else
                finetune="/home/oturgut/PyTorchEEG/mae_he/mae/output/pre/"$folder"/"$subf"/pre_"$pre_data"/checkpoint-249.pth"
            fi

            for bs in "${batch_size[@]}"
            do
                for lr in "${blr[@]}"
                do
                    output_dir="./output/fin/"$folder"/"$subf"/fin_b"$bs"_blr"$lr"_"$pre_data
                    log_dir="./logs/fin/"$folder"/"$subf"/fin_b"$bs"_blr"$lr"_"$pre_data

                    # resume="/home/oturgut/PyTorchEEG/mae_he/mae/output/fin/"$folder"/"$subfolder"/fin_b"$bs"_blr"$lr"_"$pre_data"/checkpoint-18.pth"
                
                    cmd="python3 main_finetune.py --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --accum_iter $accum_iter --drop_path $drop_path --weight_decay $weight_decay --layer_decay $layer_decay --blr $lr --warmup_epoch $warmup_epochs --smoothing $smoothing --finetune $finetune --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --output_dir $output_dir --log_dir $log_dir --num_workers $num_workers"
                    # cmd="python3 main_finetune.py --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --accum_iter $accum_iter --drop_path $drop_path --weight_decay $weight_decay --layer_decay $layer_decay --blr $lr --warmup_epoch $warmup_epochs --smoothing $smoothing --finetune $finetune --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --log_dir $log_dir --num_workers $num_workers"
                    echo $cmd && $cmd
                done
            done

        done
    done
done