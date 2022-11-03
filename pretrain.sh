#!/usr/bin/bash
# pretraining

# check whether external dataset is included in pretraining
external="off"

if [ "$external" = "on" ]; then
    batch_size="64"
    accum_iter=(1) #29
else
    batch_size="32"
    accum_iter=(1)
fi
epochs="500"
warmup_epochs="50"

input_channels="5"
input_electrodes="65"
time_steps="20000"
model="mae_vit_small_patchX"

patch_height=$input_electrodes
patch_width="10"

mask_ratio="0.40"

weight_decay="0.05"

blr_array=(1e-2)

# data_path="/home/oturgut/PyTorchEEG/data/test/data_DINH_10fold_minmax_fs200.pt"
data_path="/home/oturgut/PyTorchEEG/data/test/data_DINH_10fold_normalized_decomposed_fs200.pt"
labels_path="/home/oturgut/PyTorchEEG/data/raw/labels_2classes_DINH_10fold_fs200.pt"

transfer_data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_SEED_decomposed_ideal_fs200.pt"
transfer_labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_3classes_SEED_fs200.pt"

num_workers="32"

# resume="/home/oturgut/PyTorchEEG/mae_he/mae/output/pre/moim/0.6train/10fold/decomposed/b2048/pre_moim_b"$(($batch_size*$accum_iter))"_blr"$blr_array"/checkpoint-20.pth"

for blr in "${blr_array[@]}"
do

    for acc_it in "${accum_iter[@]}"
    do
        if [ "$external" = "on" ]; then
            folder="seed"
            pre_data="pre_"$folder"_b"$(($batch_size*$acc_it))"_blr"$blr

            subfolder="plot_signals_test"
            output_dir="./output/pre/"$folder"/"$subfolder"/"$pre_data
            log_dir="./logs/pre/"$folder"/"$subfolder"/"$pre_data

            cmd="python3 main_pretrain.py --wandb --norm_pix_loss --transfer_learning --transfer_data_path $transfer_data_path --transfer_labels_path $transfer_labels_path --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mask_ratio --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --labels_path $labels_path --output_dir $output_dir --log_dir $log_dir --num_workers $num_workers"
        else
            folder="noExternal"
            pre_data="pre_"$folder"_b"$(($batch_size*$acc_it))"_blr"$blr

            subfolder="decomposed_t20000_p10_m0.4" #/dinh/0.6train/decomposed/huge"
            output_dir="./output/pre/"$folder"/"$subfolder"/"$pre_data
            log_dir="./logs/pre/"$folder"/"$subfolder"/"$pre_data
        
            cmd="python3 main_pretrain.py --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mask_ratio --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --labels_path $labels_path --output_dir $output_dir --log_dir $log_dir --num_workers $num_workers"
        fi

        echo $cmd && $cmd
    done

done
