pretrain="./pretrain.sh"
linprobe="./linprobe.sh"
finetune="./hyper_tuning_fin.sh"

echo $pretrain && $pretrain
echo $finetune && $finetune