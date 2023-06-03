exit 1
conda create -n raft-dl2023
conda activate raft-dl2023
conda install -y python=3.8
conda install -y cudatoolkit=11.1 -c conda-forge
conda install -y pytorch==1.8.0 torchvision==0.9.0 -c pytorch
conda install -y tensorboard=2.10.0 matplotlib scipy tqdm
pip install opencv-python