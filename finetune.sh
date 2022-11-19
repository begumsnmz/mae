#!/usr/bin/bash
# Inference on the fine tuned model

# HYPERPARAMETERS
batch_size="4"
accum_iter="1"
blr="1e-3"

# FIXED PARAMETERS
epochs="250"
warmup_epochs="25"

input_channels="1"
input_electrodes="65"
time_steps="37000"
model="vit_medium_patchX"
drop_path="0.2"

patch_height="65"
patch_width="50"

weight_decay="0.1"
layer_decay="0.75"

smoothing="0.1" # label smoothing; changes the optimizer used

data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_DINH_bw_10fold_normalized_sw_decomposed_shuffled_val_test_fs200.pt"
labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_2classes_DINH_bw_10fold_sw_shuffled_val_test_fs200.pt"
nb_classes="2"

global_pool="True"
num_workers="32"

pre_batch_size=(8)
pre_blr=(1e-1)

folder="noExternal"
subfolder=("decomposed/t37000/p50/m0.75/wd0.1/dp0.2/smth0.1")

for pre_bs in "${pre_batch_size[@]}"
do
    pre_data=$folder"_b"$pre_bs"_blr"$pre_blr
    log_dir="./logs/fin/"$folder"/"$subfolder"/test_real_fin_b"$batch_size"_blr"$blr"_"$pre_data
    
    # state the checkpoint for the inference of this specific model
    # state the (final) epoch for the inference of all models up to this epoch 
    resume="/home/oturgut/PyTorchEEG/mae_he/mae/output/fin/"$folder"/"$subfolder"/GAP_shuffled_bw_sw_fin_b"$batch_size"_blr"$blr"_"$pre_data"/249"
    cmd="python3 main_finetune.py --eval --resume $resume --global_pool --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $batch_size --epochs $epochs --accum_iter $accum_iter --drop_path $drop_path --weight_decay $weight_decay --layer_decay $layer_decay --blr $blr --warmup_epoch $warmup_epochs --smoothing $smoothing --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --log_dir $log_dir --num_workers $num_workers"
    
    echo $cmd && $cmd
done