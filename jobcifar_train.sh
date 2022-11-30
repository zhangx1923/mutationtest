#!/bin/sh
python train.py --batch-size 32 --test-batch-size 5000 --epochs 20 --lr 0.25 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset cifar --model 3 --mutationType r --rmp 2| tee cifar2_train.txt
python train.py --batch-size 32 --test-batch-size 5000 --epochs 20 --lr 0.25 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset cifar --model 3 --mutationType r --rmp 3| tee cifar3_train.txt
python train.py --batch-size 32 --test-batch-size 5000 --epochs 20 --lr 0.25 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset cifar --model 3 --mutationType r --rmp 4| tee cifar4_train.txt
python train.py --batch-size 32 --test-batch-size 5000 --epochs 20 --lr 0.25 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset cifar --model 3 --mutationType r --rmp 5| tee cifar5_train.txt
