cmd_scratch="python -u train-nycu.py \
                    --name nycu-scratch \
                    --num_epochs 250 \
                    --batch_size 3 \
                    --image_size 640 360 \
                    --lr 0.0004 \
                    --wdecay 0.00001"

cmd_transfer="python -u train-nycu.py \
                    --name nycu-transfer \
                    --num_epochs 100 \
                    --batch_size 3 \
                    --image_size 640 360 \
                    --lr 0.0002 \
                    --wdecay 0.00001 \
                    --restore_ckpt checkpoints/raft-things.pth \
                    --freeze_bn \
                    --gamma=0.85"


echo ${cmd_transfer}
eval ${cmd_transfer}