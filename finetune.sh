#!/usr/bin/bash
# Inference on the fine tuned model

# HYPERPARAMETERS
batch_size="8"
accum_iter="1"
blr="1e-4"

# FIXED PARAMETERS
epochs="250"
warmup_epochs="25"

input_channels="5"
input_electrodes="64"
time_steps="55000"
model="vit_small_patchX"

crop_lbd="0.65"

drop_path="0.1"

patch_height="8"
patch_width="50"

weight_decay="0.1"
layer_decay="0.75"

smoothing="0.2" # label smoothing; changes the optimizer used

data_path="/home/oturgut/PyTorchEEG/data/preprocessed/data_DINH_701515_nf_sw_bw_fs200.pt"
labels_path="/home/oturgut/PyTorchEEG/data/preprocessed/labels_DINH_701515.pt"
nb_classes="2"

global_pool="True"
num_workers="32"

pre_batch_size=(4)
pre_blr=(1e-3)

folder="noExternal"
subfolder=("small/decomposed/t55000/p8x50/m0.75/wd0.1/dp0.1/smth0.2")

for pre_bs in "${pre_batch_size[@]}"
do
    pre_data=$folder"_b"$pre_bs"_blr"$pre_blr
    log_dir="./logs/fin/"$folder"/"$subfolder"/fin_b"$(($batch_size*$accum_iter))"_blr"$blr"_"$pre_data
    
    # state the checkpoint for the inference of this specific model
    # state the (final) epoch for the inference of all models up to this epoch 
    resume="/home/oturgut/PyTorchEEG/mae_he/mae/output/fin/"$folder"/"$subfolder"/fin_b"$(($batch_size*$accum_iter))"_blr"$blr"_"$pre_data"/checkpoint-142.pth"
    cmd="python3 main_finetune.py --eval --resume $resume --global_pool --crop_lbd $crop_lbd --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $batch_size --epochs $epochs --accum_iter $accum_iter --drop_path $drop_path --weight_decay $weight_decay --layer_decay $layer_decay --blr $blr --warmup_epoch $warmup_epochs --smoothing $smoothing --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --log_dir $log_dir --num_workers $num_workers"
    
    echo $cmd && $cmd
done