supervised="python -u train-supervised.py \
                   --name supervised-transfer \
                   --num_epochs 50 \
                   --batch_size 4 \
                   --lr 0.000125 \
                   --wdecay 0.00001 \
                   --restore_ckpt checkpoints/raft-things.pth"

supervised_gm="python -u train-supervised.py \
                   --name supervised-transfer \
                   --num_epochs 50 \
                   --batch_size 4 \
                   --lr 0.000125 \
                   --wdecay 0.00001 \
                   --restore_ckpt checkpoints/raft-things.pth \
                   --global_matching"

supervised_gm="python -u train-supervised.py \
                   --name supervised-scratch \
                   --num_epochs 200 \
                   --batch_size 4 \
                   --lr 0.0004 \
                   --wdecay 0.00001 \
                   --global_matching"


cmd=$supervised_gm    # Change this line

echo ${cmd}
eval ${cmd}
