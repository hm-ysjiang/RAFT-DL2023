#!/bin/bash
mkdir -p checkpoints

cmd_scratch="python -u train-supervised.py \
                    --name raft-sintel-supervised-scratch \
                    --validation sintel \
                    --gpus 0 \
                    --num_epochs 100 \
                    --batch_size 6 \
                    --lr 0.0004 \
                    --image_size 368 768 \
                    --wdecay 0.00001"

cmd_transfer="python -u train-supervised.py \
                     --name raft-sintel-supervised-transfer \
                     --validation sintel \
                     --restore_ckpt checkpoints/raft-things.pth \
                     --gpus 0 \
                     --num_epochs 100 \
                     --batch_size 6 \
                     --lr 0.000125 \
                     --image_size 368 768 \
                     --wdecay 0.00001 \
                     --gamma=0.85"

echo ${cmd_scratch}
eval ${cmd_scratch}
