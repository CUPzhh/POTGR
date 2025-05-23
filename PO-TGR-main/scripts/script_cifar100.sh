#!/bin/bash
for ((i = 0 ; i < 9 ; i++))
do
    if [ $i -eq 0 ]; then
        python train.py \
            --root data \
            --seed 1 \
            --trainer DiffGR \
            --dataset-config-file configs/datasets/cifar100.yaml \
            --config-file configs/trainers/DiffGR/vit_b16.yaml \
            --output-dir output/DiffGR/cifar100/session0 \
            TRAINER.TASK_ID 0 
    else
        j=$(($i-1))
        python train.py \
            --root data \
            --seed 1 \
            --trainer DiffGR \
            --dataset-config-file configs/datasets/cifar100.yaml \
            --config-file configs/trainers/DiffGR/vit_b16.yaml \
            --output-dir output/DiffGR/cifar100/session${i} \
            --model-dir output/DiffGR/cifar100/session${j} \
            TRAINER.TASK_ID ${i} 
    fi
done