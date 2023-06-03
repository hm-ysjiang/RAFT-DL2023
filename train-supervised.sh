#!/bin/bash
mkdir -p checkpoints

cmd_scratch="python -u train-supervised.py \
                    --name raft-sintel-supervised-scratch \
                    --validation sintel \
                    --num_epochs 100 \
                    --batch_size 6 \
                    --lr 0.0004 \
                    --wdecay 0.00001"

cmd_transfer="python -u train-supervised.py \
                     --name raft-sintel-supervised-transfer \
                     --validation sintel \
                     --restore_ckpt checkpoints/raft-things.pth \
                     --freeze_bn \
                     --num_epochs 100 \
                     --batch_size 6 \
                     --lr 0.000125 \
                     --wdecay 0.00001 \
                     --gamma=0.85"

# echo ${cmd_scratch}
# eval ${cmd_scratch}

echo ${cmd_transfer}
eval ${cmd_transfer}
