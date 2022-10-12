#!/usr/bin/bash
# linear probing

# HYPERPARAMETERS
batch_size="16"
blr="1e-3"
accum_iter="1"

# FIXED PARAMETERS
epochs="90"
warmup_epochs="10"

input_channels="5"
input_electrodes="65"
time_steps="37000"
model="vit_base_patchX"

patch_height=$input_electrodes
patch_width="200"

weight_decay="0"

data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_TARGET_10fold_decomposed_ideal_fs200.pt"
labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_2classes_TARGET_10fold_fs200.pt"
nb_classes="2"

global_pool="False"
num_workers="32"

pre_batchsize=(384)
pre_blr=(1e-5)

folder="seed"
subfolder="0.6train/10fold/decomposed"

eval="True"

for pre_bs in "${pre_batchsize[@]}"
do
    pre_data=$folder"_b"$pre_bs"_blr"$pre_blr
    output_dir="./output/lin/"$folder"/0.1train/lin_b"$batch_size"_blr"$blr"_"$pre_data
    log_dir="./logs/lin/"$folder"/0.1train/lin_b"$batch_size"_blr"$blr"_"$pre_data

    if [ "$eval" = "False" ]; then
        finetune="/home/oturgut/PyTorchEEG/mae_he/mae/output/pre/"$folder"/"$subfolder"/pre_"$pre_data"/checkpoint-249.pth"
        cmd="python3 main_linprobe.py --model $model --batch_size $batch_size --epochs $epochs --accum_iter $accum_iter --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --finetune $finetune --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --output_dir $output_dir --log_dir $log_dir --num_workers $num_workers"
    else
        resume="/home/oturgut/PyTorchEEG/mae_he/mae/output/lin/"$folder"/"$subfolder"/lin_b"$batch_size"_blr"$blr"_"$pre_data"/checkpoint-43.pth"
        cmd="python3 main_linprobe.py --eval --resume $resume --model $model --batch_size $batch_size --epochs $epochs --accum_iter $accum_iter --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --output_dir $output_dir --log_dir $log_dir --num_workers $num_workers"
    fi

    echo $cmd && $cmd
done