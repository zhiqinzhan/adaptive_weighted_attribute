#!/bin/bash

LOG="log/ResNet-50.`date +'%Y-%m-%d_%H-%M-%S'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python train_model.py
