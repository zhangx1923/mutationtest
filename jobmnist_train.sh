#!/bin/sh
python train.py --batch-size 32 --test-batch-size 1000 --epochs 20 --lr 0.1 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset mnist| tee mnist_train.txt