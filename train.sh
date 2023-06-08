# From scratch
supervised="python -u train-supervised.py \
                   --name supervised-scratch \
                   --num_epochs 200 \
                   --batch_size 3 \
                   --lr 0.0004 \
                   --wdecay 0.00001"

selfsupervised="python -u train-selfsupervised.py \
                       --name selfsupervised-scratch \
                       --num_epochs 200 \
                       --batch_size 3 \
                       --lr 0.0004 \
                       --wdecay 0.00001"

cmd=$supervised    # Change this line

echo ${cmd}
eval ${cmd}
