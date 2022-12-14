#!/bin/sh
python train.py --batch-size 32 --test-batch-size 5000 --epochs 10 --lr 0.25 --gamma 0.7 --device cuda --gpu 6 --evaluate train --mutationType c| tee mucombine.txt
