#!/usr/bin/env bash

echo Please run this script manually
exit 1

conda install -y python=3.8
conda install -y cudatoolkit=11.1 -c conda-forge
conda install -y pytorch==1.8.0 torchvision==0.9.0 -c pytorch
conda install -y matplotlib scipy tensorboard=2.10.0 tqdm
pip install opencv-python
