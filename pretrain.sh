#!/usr/bin/bash
# pretraining

# check whether external dataset is included in pretraining
external="off"

if [ "$external" = "on" ]; then
    batch_size="32"
    accum_iter=(4)
else
    batch_size="8"
    accum_iter=(4)
fi
epochs="400"
warmup_epochs="40"

input_channels="5"
input_electrodes="65"
time_steps="37000"
model="mae_vit_medium_patchX"

patch_height="13"
patch_width=(200)

mask_ratio="0.75"

weight_decay="0.05"

blr_array=(1e-1)

data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_DINH_bw_10fold_normalized_sw_decomposed_fs200.pt"
labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_2classes_DINH_bw_10fold_sw_fs200.pt"

transfer_data_path="/home/oturgut/PyTorchEEG/data/raw/data_SEED_normalized_sw_decomposed_fs200.pt"
transfer_labels_path="/home/oturgut/PyTorchEEG/data/raw/labels_2classes_SEED_fs200.pt"

num_workers="32"

# resume="/home/oturgut/PyTorchEEG/mae_he/mae/output/pre/moim/0.6train/10fold/decomposed/b2048/pre_moim_b"$(($batch_size*$accum_iter))"_blr"$blr_array"/checkpoint-20.pth"

for blr in "${blr_array[@]}"
do
    for acc_it in "${accum_iter[@]}"
    do
        for pw in "${patch_width[@]}"
        do

            if [ "$external" = "on" ]; then
                folder="seed"
                pre_data="pre_"$folder"_b"$(($batch_size*$acc_it))"_blr"$blr

                subfolder="decomposed/t37000/p"$pw"/m0.75/ncc"
                output_dir="./output/pre/"$folder"/"$subfolder"/"$pre_data
                log_dir="./logs/pre/"$folder"/"$subfolder"/"$pre_data

                cmd="python3 main_pretrain.py --wandb --transfer_learning --transfer_data_path $transfer_data_path --transfer_labels_path $transfer_labels_path --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $pw --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mask_ratio --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --labels_path $labels_path --output_dir $output_dir --log_dir $log_dir --num_workers $num_workers"
            else
                folder="noExternal"
                pre_data="pre_"$folder"_b"$(($batch_size*$acc_it))"_blr"$blr

                subfolder="decomposed/t37000/p"pw"/m0.75/ncc"
                output_dir="./output/pre/"$folder"/"$subfolder"/"$pre_data
                log_dir="./logs/pre/"$folder"/"$subfolder"/"$pre_data
            
                cmd="python3 main_pretrain.py --wandb --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $pw --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mask_ratio --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --labels_path $labels_path --output_dir $output_dir --log_dir $log_dir --num_workers $num_workers"
            fi

            echo $cmd && $cmd

        done
    done
done