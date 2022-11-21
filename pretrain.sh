#!/usr/bin/bash
# pretraining

# check whether external dataset is included in pretraining
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

input_channels="5"
input_electrodes="64"
time_steps="30000"
model="mae_vit_small_patchX"

crop_lbd="0.65"

patch_height="8"
patch_width=(50)

mask_ratio="0.75"

weight_decay="0.1"

blr_array=(1e-3)

data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_DINH_701515_nf_sw_bw_fs200.pt"
labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_DINH_701515.pt"

transfer_data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_SEED_sw_bw_fs200.pt"
transfer_labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_SEED.pt"

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

                subfolder="small/cropResize/decomposed/t"$time_steps"/p"$patch_height"x"$pw"/m"$mask_ratio"/wd"$weight_decay"/crop"$crop_lbd
                output_dir="./output/pre/"$folder"/"$subfolder"/"$pre_data
                log_dir="./logs/pre/"$folder"/"$subfolder"/"$pre_data

                cmd="python3 main_pretrain.py --wandb --crop_lbd $crop_lbd --transfer_learning --transfer_data_path $transfer_data_path --transfer_labels_path $transfer_labels_path --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $pw --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mask_ratio --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --labels_path $labels_path --output_dir $output_dir --log_dir $log_dir --num_workers $num_workers"
            else
                folder="noExternal"
                pre_data="pre_"$folder"_b"$(($batch_size*$acc_it))"_blr"$blr

                subfolder="small/cropResize/decomposed/t"$time_steps"/p"$patch_height"x"$pw"/m"$mask_ratio"/wd"$weight_decay"/crop"$crop_lbd
                output_dir="./output/pre/"$folder"/"$subfolder"/"$pre_data
                log_dir="./logs/pre/"$folder"/"$subfolder"/"$pre_data
            
                cmd="python3 main_pretrain.py --wandb --crop_lbd $crop_lbd --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $pw --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mask_ratio --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --labels_path $labels_path --output_dir $output_dir --log_dir $log_dir --num_workers $num_workers"
            fi

            echo $cmd && $cmd

        done
    done
done