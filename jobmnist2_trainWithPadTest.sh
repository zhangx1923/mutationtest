#!/bin/sh
python train.py --batch-size 32 --test-batch-size 1000 --model 2 --padtest 1 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 3 --evaluate train --dataset mnist --mutationType r --rmp 2| tee mnist2_trainWithPad2.txt

python train.py --batch-size 32 --test-batch-size 1000 --model 2 --padtest 1 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 3 --evaluate train --dataset mnist --mutationType r --rmp 3| tee mnist2_trainWithPad3.txt

python train.py --batch-size 32 --test-batch-size 1000 --model 2 --padtest 1 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 3 --evaluate train --dataset mnist --mutationType r --rmp 4| tee mnist2_trainWithPad4.txt

python train.py --batch-size 32 --test-batch-size 1000 --model 2 --padtest 1 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 3 --evaluate train --dataset mnist --mutationType r --rmp 5| tee mnist2_trainWithPad5.txt
