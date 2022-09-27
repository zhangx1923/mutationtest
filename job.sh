#!/bin/sh
python train.py --batch-size 32 --test-batch-size 1000 --epochs 20 --lr 0.1 --gamma 0.7 --device cuda --gpu 5 | tee b32e20l01g07.txt
python train.py --batch-size 32 --test-batch-size 1000 --epochs 20 --lr 1 --gamma 0.7 --device cuda --gpu 5 | tee b32e20l1g07.txt
python train.py --batch-size 64 --test-batch-size 1000 --epochs 20 --lr 0.1 --gamma 0.7 --device cuda --gpu 5 | tee b64e20l01g07.txt
python train.py --batch-size 64 --test-batch-size 1000 --epochs 20 --lr 1 --gamma 0.7 --device cuda --gpu 5 | tee b64e20l1g07.txt
