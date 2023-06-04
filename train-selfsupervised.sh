#!/bin/bash
mkdir -p checkpoints

cmd_scratch="python -u train-selfsupervised.py \
                    --name raft-sintel-selfsupervised-scratch \
                    --validation sintel \
                    --num_epochs 200 \
                    --batch_size 6 \
                    --lr 0.0004 \
                    --wdecay 0.00001"

cmd_transfer="python -u train-selfsupervised.py \
                     --name raft-sintel-selfsupervised-transfer \
                     --validation sintel \
                     --restore_ckpt checkpoints/raft-things.pth \
                     --freeze_bn \
                     --num_epochs 200 \
                     --batch_size 6 \
                     --lr 0.000125 \
                     --wdecay 0.00001 \
                     --gamma=0.85"

# echo ${cmd_scratch}
# eval ${cmd_scratch}

echo ${cmd_transfer}
eval ${cmd_transfer}
