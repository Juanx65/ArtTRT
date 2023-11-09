#!/bin/bash

# remeber to do a chmod +x main.sh befor runing with ./main.sh

# Ejecutar el script de Python
python3 main.py -v --batch_size=256 --dataset='datasets/dataset_val/val' --network="resnet50"