#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
python3 run_exp_grid_search.py --conf_file_path ./config/QM9/DimeNet/config.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=1
python3 run_exp_grid_search.py --conf_file_path ./config/QM9/SphereNet/config.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=2
python3 run_exp_grid_search.py --conf_file_path ./config/QM9/ComeNet/config.yaml &
sleep 3
