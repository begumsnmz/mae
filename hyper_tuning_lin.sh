#!/usr/bin/bash
# hyperparameter tuning for linear probing

# HYPERPARAMETERS
batch_size=(8)
accum_iter=(1)
blr=(1e-3)

# FIXED PARAMETERS
epochs="90"
warmup_epochs="10"

model="vit_base_patch224"

weight_decay="0"

data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_8fold_decomposed_2d_fs200_new.pt"
labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_2classes_8fold_fs200_new.pt"

nb_classes="2"
num_workers="32"

pre_batch_size=(384)
pre_blr=(1e-5)

for pre_bs in "${pre_batch_size[@]}"
do
    for pre_lr in "${pre_blr[@]}"
    do

        folder="seed"
        pre_data=$folder"_b"$pre_bs"_blr"$pre_lr
        finetune="/home/oturgut/PyTorchEEG/mae_he/mae/output/correct/pre/"$folder"/0.75train/8fold/pre_"$pre_data"/checkpoint-249.pth"
    
        for bs in "${batch_size[@]}"
        do
            for lr in "${blr[@]}"
            do
                output_dir="./output/correct/lin/"$folder"/0.75train/8fold/lin_b"$bs"_blr"$lr"_"$pre_data
                log_dir="./logs/correct/lin/"$folder"/0.75train/8fold/lin_b"$bs"_blr"$lr"_"$pre_data
            
                cmd="python3 main_linprobe.py --model $model --batch_size $bs --epochs $epochs --accum_iter $accum_iter --weight_decay $weight_decay --blr $lr --warmup_epoch $warmup_epochs --finetune $finetune --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --output_dir $output_dir --log_dir $log_dir --num_workers $num_workers"
                # cmd="python3 main_linprobe.py --model $model --batch_size $bs --epochs $epochs --accum_iter $accum_iter --weight_decay $weight_decay --blr $lr --warmup_epoch $warmup_epochs --finetune $finetune --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --log_dir $log_dir --num_workers $num_workers"
                echo $cmd && $cmd
            done
        done

    done
done