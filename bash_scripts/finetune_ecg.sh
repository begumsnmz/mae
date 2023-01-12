#!/usr/bin/bash
# Fine tuning 
 
#SBATCH --job-name=mae_fin
#SBATCH --output=/home/guests/oezguen_turgut/slurm_output/fin/ecg/mae_fin-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=/home/guests/oezguen_turgut/slurm_output/fin/ecg/mae_fin-%A.err  # Standard error of the script
#SBATCH --time=7-23:59:59  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=24  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=126G  # Memory in GB (Don't use more than 126G per GPU)
 
# load python module
ml python/anaconda3

# activate corresponding environment
conda deactivate # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
conda activate mae

# Basic parameters seed = [0, 101, 202, 303, 404]
seed="0"
batch_size=(16)
accum_iter=(1)

epochs="400"
warmup_epochs="5"

# Callback parameters
patience="25"
max_delta="0"

# Model parameters
input_channels="1"
input_electrodes="12"
time_steps="2500"
model_size="tiny"
model="vit_"$model_size"_patchX"

patch_height="1"
patch_width=(100)

# Augmentation parameters
jitter_sigma="0.2"
rescaling_sigma="0.5"
ft_surr_phase_noise="0.075"

drop_path=(0.2)
layer_decay="0.75"

# Optimizer parameters
blr=(1e-7)
min_lr="0.0"
weight_decay=(0.1)

# Criterion parameters
smoothing=(0.2)

# Dataset parameters
# Training balanced
# data_path="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/ecgs_train_CAD_all_balanced_noBase_gn.pt"
# labels_path="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/labels_train_CAD_all_balanced.pt"
# nb_classes="2"
# data_path="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/ecgs_train_BMI_balanced_noBase_gn.pt"
# labels_path="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/labels_train_BMI_balanced.pt"
# nb_classes="2"
# val_data_path="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/ecgs_train_ecg_imaging_noBase_gn.pt"
# val_labels_path="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/labels_train_infarct_future.pt"
# num_classes="2"
data_path="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/ecgs_train_ecg_imaging_noBase_gn.pt"
labels_path="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/labelsOneHot/labels_train_LVM_regression_div300.pt"
nb_classes="1"

# Validation unbalanced
# val_data_path="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/ecgs_val_ecg_imaging_noBase_gn.pt"
# val_labels_path="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/labels_val_CAD_all.pt"
# pos_label="1"
# val_data_path="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/ecgs_val_BMI_noBase_gn.pt"
# val_labels_path="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/labels_val_BMI.pt"
# pos_label="0"
# val_data_path="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/ecgs_val_ecg_imaging_noBase_gn.pt"
# val_labels_path="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/labels_val_infarct_future.pt"
# pos_label="1"
val_data_path="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/ecgs_val_ecg_imaging_noBase_gn.pt"
val_labels_path="/home/guests/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/labelsOneHot/labels_val_LVM_regression_div300.pt"

global_pool=(True)
attention_pool=(True)
num_workers="24"

# Log specifications
save_output="False"
wandb="True"
wandb_project="MAE_ECG_Fin_Tiny_LVM"

# Pretraining specifications
pre_batch_size=(128)
pre_blr=(1e-5)

# EVALUATE
eval="False"
# As filename: State the checkpoint for the inference of a specific model
# or state the (final) epoch for the inference of all models up to this epoch
#resume="/home/guests/oezguen_turgut/sprai/mae_he/mae/output/fin/"$folder"/id/"$subfolder"/fin_b"$(($batch_size*$accum_iter))"_blr"$blr"_"$pre_data"/checkpoint-89.pth"

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

                        folder="ecg/CAD/MAE"
                        subfolder=($model_size"/1d/t2500/p"$patch_height"x"$patch_width"/wd"$weight_decay"/dp"$dp"/smth"$smth"/m0.8/atp")

                        pre_data="b"$pre_batch_size"_blr"$pre_blr
                        # finetune="/home/guests/oezguen_turgut/sprai/mae_he/mae/output/pre/"$folder"/"$subfolder"/pre_"$pre_data"/checkpoint-399.pth"
                        finetune="/home/guests/oezguen_turgut/ECGMultimodalContrastiveLearning/oezguen/checkpoints/mm_v230_mae_checkpoint.pth"
                        # finetune="/home/guests/oezguen_turgut/ECGMultimodalContrastiveLearning/pretrained_checkpoints/tiny/v1/checkpoint-399.pth"

                        output_dir="/home/guests/oezguen_turgut/sprai/mae_he/mae/output/fin/"$folder"/"$subfolder"/fin_b"$(($bs*$acc_it))"_blr"$lr"_"$pre_data
                        log_dir="/home/guests/oezguen_turgut/sprai/mae_he/mae/logs/fin/"$folder"/"$subfolder"/fin_b"$(($bs*$acc_it))"_blr"$lr"_"$pre_data

                        # resume="/home/guests/oezguen_turgut/sprai/mae_he/mae/output/fin/"$folder"/"$subfolder"/fin_b"$bs"_blr"$lr"_"$pre_data"/checkpoint-78.pth"

                        cmd="python3 main_finetune.py --finetune $finetune --seed $seed --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --patience $patience --max_delta $max_delta --accum_iter $acc_it --drop_path $dp --weight_decay $wd --layer_decay $layer_decay --min_lr $min_lr --blr $lr --warmup_epoch $warmup_epochs --smoothing $smth --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --nb_classes $nb_classes --log_dir $log_dir --num_workers $num_workers"

                        if [ ! -z "$pos_label" ]; then
                            cmd=$cmd" --pos_label $pos_label"
                        fi

                        if [ "$global_pool" = "True" ]; then
                            cmd=$cmd" --global_pool"
                        fi

                        if [ "$attention_pool" = "True" ]; then
                            cmd=$cmd" --attention_pool"
                        fi

                        if [ "$wandb" = "True" ]; then
                            cmd=$cmd" --wandb --wandb_project $wandb_project"
                        fi

                        if [ "$save_output" = "True" ]; then
                            cmd=$cmd" --output_dir $output_dir"
                        fi

                        if [ "$eval" = "True" ]; then
                            cmd=$cmd" --eval --resume $resume"
                        fi
                        
                        echo $cmd && $cmd

                    done
                done
            done

        done
    done
done