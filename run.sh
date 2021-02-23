
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model $2 \
    --seed 0 \
    --dataset cifar100