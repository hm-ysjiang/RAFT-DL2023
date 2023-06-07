# From scratch
cmd_supervised_scratch="python -u train-supervised.py \
                               --name supervised-scratch-c128 \
                               --validation sintel \
                               --num_epochs 100 \
                               --batch_size 6 \
                               --lr 0.0004 \
                               --wdecay 0.00001 \
                               --context 128"

cmd_selfsupervised_scratch="python -u train-selfsupervised.py \
                                   --name selfsupervised-scratch-c128 \
                                   --validation sintel \
                                   --num_epochs 100 \
                                   --batch_size 6 \
                                   --lr 0.0004 \
                                   --wdecay 0.00001 \
                                   --context 128"

# Transfer
cmd_supervised_transfer="python -u train-supervised.py \
                                --name supervised-transfer-c128 \
                                --validation sintel \
                                --restore_ckpt checkpoints/raft-things.pth \
                                --num_epochs 100 \
                                --batch_size 6 \
                                --lr 0.0002 \
                                --wdecay 0.00001 \
                                --gamma=0.85 \
                                --allow_nonstrict \
                                --reset_context \
                                --context 128"

cmd_selfsupervised_transfer="python -u train-selfsupervised.py \
                                    --name selfsupervised-transfer-c128 \
                                    --validation sintel \
                                    --restore_ckpt checkpoints/raft-things.pth \
                                    --num_epochs 100 \
                                    --batch_size 6 \
                                    --lr 0.0002 \
                                    --wdecay 0.00001 \
                                    --gamma=0.85 \
                                    --allow_nonstrict \
                                    --reset_context \
                                    --context 128"

cmd=$cmd_supervised_transfer    # Change this line

echo ${cmd}
eval ${cmd}