#!/bin/bash
for ((i = 1 ; i < 11 ; i++))
do
    if [ $i -eq 0 ]; then
        python train.py \
            --root data \
            --seed 1 \
            --trainer DiffGR \
            --dataset-config-file configs/datasets/cub200.yaml \
            --config-file configs/trainers/DiffGR/vit_b16.yaml \
            --output-dir output/DiffGR/cub200/session0 \
            TRAINER.TASK_ID 0 
    else
        j=$(($i-1))
        python train.py \
            --root data \
            --seed 1 \
            --trainer DiffGR \
            --dataset-config-file configs/datasets/cub200.yaml \
            --config-file configs/trainers/DiffGR/vit_b16.yaml \
            --output-dir output/DiffGR/cub200/session${i} \
            --model-dir output/DiffGR/cub200/session${j} \
            TRAINER.TASK_ID ${i} 
    fi
done