#!/usr/bin/bash
# pretraining

# check whether external dataset is included in pretraining
external="on"

if [ "$external" = "on" ]; then
    batch_size="64"
    accum_iter=(6) #29
else
    batch_size="4096"
    accum_iter=(1)
fi
epochs="250"
warmup_epochs="15"

input_channels="5"
input_electrodes="65"
time_steps="37000"
model="mae_vit_base_patchX"

patch_height=$input_electrodes
patch_width="200"

mask_ratio="0.75"

weight_decay="0.05"

blr_array=(1e-5)

data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_TARGET_10fold_decomposed_ideal_fs200.pt"
labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_2classes_TARGET_10fold_fs200.pt"

# transfer_data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_SEED_decomposed_2d_fs200.pt"
# transfer_labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_2classes_SEED_fs200.pt"

num_workers="32"

# resume="/home/oturgut/PyTorchEEG/mae_he/mae/output/pre/external/0.6train/10fold/test/pre_external_b"$(($batch_size*$accum_iter))"_blr"$blr_array"/checkpoint-200.pth"

for blr in "${blr_array[@]}"
do

    for acc_it in "${accum_iter[@]}"
    do
        if [ "$external" = "on" ]; then
            folder="seed"
            pre_data="pre_"$folder"_b"$(($batch_size*$acc_it))"_blr"$blr

            subfolder="0.6train/10fold/decomposed"
            output_dir="./output/pre/"$folder"/"$subfolder"/"$pre_data
            log_dir="./logs/pre/"$folder"/"$subfolder"/"$pre_data

            cmd="python3 main_pretrain.py --transfer_learning --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mask_ratio --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --labels_path $labels_path --output_dir $output_dir --log_dir $log_dir --num_workers $num_workers"
        else
            folder="noSeed"
            pre_data="pre_"$folder"_b"$(($batch_size*$acc_it))"_blr"$blr

            subfolder="snippets6s/p100/de_features"
            output_dir="./output/pre/"$folder"/"$subfolder"/"$pre_data
            log_dir="./logs/pre/"$folder"/"$subfolder"/"$pre_data
        
            cmd="python3 main_pretrain.py --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mask_ratio --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --labels_path $labels_path --output_dir $output_dir --log_dir $log_dir --num_workers $num_workers"
        fi

        echo $cmd && $cmd
    done

done
