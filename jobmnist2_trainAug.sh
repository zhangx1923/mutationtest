#!/bin/sh
python train.py --batch-size 32 --test-batch-size 1000 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset mnist --model 2 --mutationType r --rmp 2 --padtest 1 --aug 5| tee mnist22_trainAug5.txt
python train.py --batch-size 32 --test-batch-size 1000 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset mnist --model 2 --mutationType r --rmp 3 --padtest 1 --aug 5| tee mnist23_trainAug5.txt
python train.py --batch-size 32 --test-batch-size 1000 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset mnist --model 2 --mutationType r --rmp 4 --padtest 1 --aug 5| tee mnist24_trainAug5.txt
python train.py --batch-size 32 --test-batch-size 1000 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset mnist --model 2 --mutationType r --rmp 5 --padtest 1 --aug 5| tee mnist25_trainAug5.txt
