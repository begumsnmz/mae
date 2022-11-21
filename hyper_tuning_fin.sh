#!/usr/bin/bash
# Fine tuning 

# HYPERPARAMETERS
batch_size=(2)
accum_iter=(4)
blr=(1e-4)

# FIXED PARAMETERS
epochs="250"
warmup_epochs="25"

input_channels="5"
input_electrodes="64"
time_steps="55000"
model="vit_small_patchX"

crop_lbd="0.65"

drop_path=(0.1)

patch_height="8"
patch_width="50"

weight_decay=(0.1)
layer_decay="0.75"

smoothing=(0.2) # label smoothing; changes the criterion used

data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_DINH_701515_nf_sw_bw_fs200.pt"
labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_DINH_701515.pt"
nb_classes="2"

global_pool="True"
num_workers="32"

pre_batch_size=(4)
pre_blr=(1e-3)

folder="noExternal"
subfolder=("pluto/decomposed/t15000/p8x50/m0.75/wd0.1")

output="True"

for pre_bs in "${pre_batch_size[@]}"
do
    for pre_lr in "${pre_blr[@]}"
    do
        for subf in "${subfolder[@]}"
        do

            pre_data=$folder"_b"$pre_bs"_blr"$pre_lr
            finetune="/home/oturgut/PyTorchEEG/mae_he/mae/output/pre/"$folder"/"$subf"/pre_"$pre_data"/checkpoint-749.pth"

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

                                    output_dir="./output/fin/"$folder"/"$subf"/wd"$wd"/dp"$dp"/smth"$smth"/crop_lbd"$crop_lbd"/fin_b"$(($bs*$acc_it))"_blr"$lr"_"$pre_data
                                    log_dir="./logs/fin/"$folder"/"$subf"/wd"$wd"/dp"$dp"/smth"$smth"/crop_lbd"$crop_lbd"/fin_b"$(($bs*$acc_it))"_blr"$lr"_"$pre_data

                                    # resume="/home/oturgut/PyTorchEEG/mae_he/mae/output/fin/"$folder"/"$subfolder"/fin_b"$bs"_blr"$lr"_"$pre_data"/checkpoint-78.pth"

                                    if [ "$output" = "True" ]; then
                                        cmd="python3 main_finetune.py --global_pool --crop_lbd $crop_lbd --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --accum_iter $acc_it --drop_path $dp --weight_decay $wd --layer_decay $layer_decay --blr $lr --warmup_epoch $warmup_epochs --smoothing $smth --finetune $finetune --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --output_dir $output_dir --log_dir $log_dir --num_workers $num_workers"
                                    else
                                        cmd="python3 main_finetune.py --global_pool --crop_lbd $crop_lbd --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --accum_iter $acc_it --drop_path $dp --weight_decay $wd --layer_decay $layer_decay --blr $lr --warmup_epoch $warmup_epochs --smoothing $smth --finetune $finetune --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --log_dir $log_dir --num_workers $num_workers"
                                    fi
                                    echo $cmd && $cmd

                                done
                            done
                        done

                    done
                done
            done

        done
    done
done