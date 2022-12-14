#!/bin/sh
python train.py --batch-size 32 --test-batch-size 5000 --epochs 10 --lr 0.25 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset cifar --model 3 --mutationType r --rmp 2 --padtest 1 --aug 1| tee cifar2_trainAug1.txt
python train.py --batch-size 32 --test-batch-size 5000 --epochs 10 --lr 0.25 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset cifar --model 3 --mutationType r --rmp 3 --padtest 1 --aug 1| tee cifar3_trainAug1.txt
python train.py --batch-size 32 --test-batch-size 5000 --epochs 10 --lr 0.25 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset cifar --model 3 --mutationType r --rmp 4 --padtest 1 --aug 1| tee cifar4_trainAug1.txt
python train.py --batch-size 32 --test-batch-size 5000 --epochs 10 --lr 0.25 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset cifar --model 3 --mutationType r --rmp 5 --padtest 1 --aug 1| tee cifar5_trainAug1.txt
