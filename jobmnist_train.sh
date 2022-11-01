#!/bin/sh
python train.py --batch-size 32 --test-batch-size 1000 --epochs 15 --lr 0.1 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset mnist --mutationType r --rmp 3| tee mnist_train.txt
python train.py --batch-size 32 --test-batch-size 1000 --epochs 15 --lr 0.1 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset mnist --mutationType r --rmp 2| tee mnist_train.txt

python train.py --batch-size 32 --test-batch-size 1000 --epochs 15 --lr 0.1 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset mnist --mutationType r --rmp 4| tee mnist_train.txt

python train.py --batch-size 32 --test-batch-size 1000 --epochs 15 --lr 0.1 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset mnist --mutationType r --rmp 5| tee mnist_train.txt
