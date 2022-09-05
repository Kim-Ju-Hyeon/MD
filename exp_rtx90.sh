#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
python3 run_exp.py --conf_file_path ./config/config.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=1
python3 run_exp.py --conf_file_path ./config/config.yaml &
sleep 3
