#!/bin/sh
python train.py --batch-size 32 --test-batch-size 1000 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 5 --evaluate train --dataset mnist --model 1 --mutationType r --rmp 3 --padtest 1 --aug 7| tee mnist13_trainAugNew7.txt
# python train.py --batch-size 32 --test-batch-size 1000 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 5 --evaluate train --dataset mnist --model 1 --mutationType r --rmp 3 --padtest 1 --aug 7| tee mnist13_trainAug7.txt
# python train.py --batch-size 32 --test-batch-size 1000 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 5 --evaluate train --dataset mnist --model 1 --mutationType r --rmp 4 --padtest 1 --aug 7| tee mnist14_trainAug7.txt
# python train.py --batch-size 32 --test-batch-size 1000 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 5 --evaluate train --dataset mnist --model 1 --mutationType r --rmp 5 --padtest 1 --aug 7| tee mnist15_trainAug7.txt


# python train.py --batch-size 32 --test-batch-size 1000 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset mnist --model 1 --mutationType r --rmp 2 --padtest 1 --aug 6| tee mnist12_trainAug6.txt
# python train.py --batch-size 32 --test-batch-size 1000 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset mnist --model 1 --mutationType r --rmp 3 --padtest 1 --aug 6| tee mnist13_trainAug6.txt
# python train.py --batch-size 32 --test-batch-size 1000 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset mnist --model 1 --mutationType r --rmp 4 --padtest 1 --aug 6| tee mnist14_trainAug6.txt
# python train.py --batch-size 32 --test-batch-size 1000 --epochs 10 --lr 0.1 --gamma 0.7 --device cuda --gpu 7 --evaluate train --dataset mnist --model 1 --mutationType r --rmp 5 --padtest 1 --aug 6| tee mnist15_trainAug6.txt
