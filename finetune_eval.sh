#!/usr/bin/bash
# Inference on the finetuned model 

# Basic parameters
batch_size=(8)
accum_iter=(1)

epochs="50"
warmup_epochs="5"

# Model parameters
input_channels="6"
input_electrodes="65"
time_steps="55000"
model_size="tiny"
model="vit_"$model_size"_patchX"

patch_height="65"
patch_width="50"

# Augmentation parameters
drop_path=(0.1)
layer_decay="0.75"

# Optimizer parameters
blr=(1e-3)
weight_decay=(0.1)

# Criterion parameters
smoothing=(0.2)

# Dataset parameters
data_path="/home/oturgut/sprai/data/preprocessed/data_JOINED_701515_nf_cw_bw_fs200.pt"
labels_path="/home/oturgut/sprai/data/preprocessed/labels_JOINED_701515.pt"
nb_classes="2"

global_pool="True"
num_workers="32"

# Log specifications
wandb="False"

folder="noExternal"
subfolder=("joined/"$model_size"/2d/t37000/p"$patch_height"x"$patch_width"/m0.75")

# Pretraining specifications
pre_batch_size=(8)
pre_blr=(1e-3)

pre_data=$folder"_b"$pre_batch_size"_blr"$pre_blr

log_dir="./logs/fin/"$folder"/"$subfolder"/fin_b"$(($batch_size*$accum_iter))"_blr"$blr"_"$pre_data

# As filename: State the checkpoint for the inference of a specific model
# or state the (final) epoch for the inference of all models up to this epoch
resume="/home/oturgut/sprai/mae_he/mae/output/fin/"$folder"/"$subfolder"/fin_b"$(($batch_size*$accum_iter))"_blr"$blr"_"$pre_data"/checkpoint-42.pth"
cmd="python3 main_finetune.py --eval --resume $resume --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $batch_size --epochs $epochs --accum_iter $accum_iter --drop_path $drop_path --weight_decay $weight_decay --layer_decay $layer_decay --blr $blr --warmup_epoch $warmup_epochs --smoothing $smoothing --data_path $data_path --labels_path $labels_path --nb_classes $nb_classes --log_dir $log_dir --num_workers $num_workers"

if [ "$global_pool" == "True" ]; then
    cmd=$cmd" --global_pool"
fi

if [ "$wandb" = "True" ]; then
    cmd=$cmd" --wandb"
fi

echo $cmd && $cmd