baseline="python -u train-supervised.py \
                 --name upsampling-baseline \
                 --num_epochs 100 \
                 --batch_size 3 \
                 --lr 0.0004 \
                 --wdecay 0.00001"

plus_l1="python -u train-supervised.py \
                --name upsampling-plusl1 \
                --num_epochs 100 \
                --batch_size 3 \
                --lr 0.0004 \
                --wdecay 0.00001 \
                --wloss_l1recon 1.0"

plus_ssim="python -u train-supervised.py \
                  --name upsampling-plusssim \
                  --num_epochs 100 \
                  --batch_size 3 \
                  --lr 0.0004 \
                  --wdecay 0.00001 \
                  --wloss_ssimrecon 1.0"

full="python -u train-supervised.py \
             --name upsampling-full \
             --num_epochs 100 \
             --batch_size 3 \
             --lr 0.0004 \
             --wdecay 0.00001 \
             --wloss_l1recon 1.0 \
             --wloss_ssimrecon 1.0"

cmd=$baseline    # Change this line

echo ${cmd}
eval ${cmd}
