#!/bin/sh
python train.py --batch-size 32 --test-batch-size 1000 --model 2 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 3 --evaluate train --dataset mnist --mutationType r --rmp 2| tee mnist2_train2.txt


python train.py --batch-size 32 --test-batch-size 1000 --model 2 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 3 --evaluate train --dataset mnist --mutationType r --rmp 3| tee mnist2_train3.txt

python train.py --batch-size 32 --test-batch-size 1000 --model 2 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 3 --evaluate train --dataset mnist --mutationType r --rmp 4| tee mnist2_train4.txt

python train.py --batch-size 32 --test-batch-size 1000 --model 2 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 3 --evaluate train --dataset mnist --mutationType r --rmp 5| tee mnist2_train5.txt
