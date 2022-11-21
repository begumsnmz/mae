#!/usr/bin/bash
# hyperparameter tuning for linear probing

# HYPERPARAMETERS
batch_size=(4)
accum_iter=(1)
blr=(1e-3)

# FIXED PARAMETERS
epochs="150"
warmup_epochs="15"

input_channels="5"
input_electrodes="64"
time_steps="55000"
model="vit_small_patchX"

crop_lbd="0.65"

patch_height="8"
patch_width="50"

weight_decay=(0.1)

data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_DINH_701515_nf_sw_bw_fs200.pt"
labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_DINH_701515.pt"
nb_classes="2"

global_pool="True"
num_workers="32"

pre_batch_size=(4)
pre_blr=(1e-3)

folder="noExternal"
subfolder=("small/decomposed/t15000/p8x50/m0.75/wd0.1")

output="False"

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
                        for wd in "${weight_decay[@]}"
                        do

                            output_dir="./output/lin/"$folder"/"$subf"/wd"$wd"/crop_lbd"$crop_lbd"/lin_b"$(($bs*$acc_it))"_blr"$lr"_"$pre_data
                            log_dir="./logs/lin/"$folder"/"$subf"/wd"$wd"/crop_lbd"$crop_lbd"/lin_b"$(($bs*$acc_it))"_blr"$lr"_"$pre_data
                        
                            # resume="/home/oturgut/PyTorchEEG/mae_he/mae/output/lin/"$folder"/"$subfolder"/lin_b"$bs"_blr"$lr"_"$pre_data"/checkpoint-18.pth"
                        
                            if [ "$output" = "True" ]; then
                                cmd="python3 main_linprobe.py --wandb --crop_lbd $crop_lbd --global_pool --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --accum_iter $acc_it --weight_decay $wd --blr $lr --warmup_epoch $warmup_epochs --finetune $finetune --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --output_dir $output_dir --log_dir $log_dir --num_workers $num_workers"
                            else
                                cmd="python3 main_linprobe.py --wandb --crop_lbd $crop_lbd --global_pool --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --accum_iter $acc_it --weight_decay $wd --blr $lr --warmup_epoch $warmup_epochs --finetune $finetune --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --log_dir $log_dir --num_workers $num_workers"
                            fi
                            echo $cmd && $cmd

                        done
                    done
                done
            done

        done
    done
done