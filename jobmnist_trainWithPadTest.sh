#!/bin/sh
python train.py --batch-size 32 --test-batch-size 1000 --model 1 --padtest 1 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 2 --evaluate train --dataset mnist --mutationType r --rmp 2| tee mnist_trainWithPad.txt
