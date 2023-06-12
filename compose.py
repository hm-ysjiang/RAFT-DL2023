import sys

sys.path.append('core')  # nopep8

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from raft import RAFT
from tqdm import trange
from utils.flow_viz import flow_to_image
from utils.utils import InputPadder

DEVICE = 'cuda'


def to_tensor(x_np):
    x = torch.from_numpy(x_np[:, :, [2, 1, 0]]).permute(2, 0, 1).float()
    return x.to(DEVICE)[None]


def compose(args):
    vid = cv2.VideoCapture(args.input)
    if not vid.isOpened():
        print('Cannot open video file!')
        exit(1)

    VIDPROP_FRAMES = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    VIDPROP_HEIGHT = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    VIDPROP_WIDTH = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    VIDPROP_FPS = vid.get(cv2.CAP_PROP_FPS)

    frames_input = np.empty((VIDPROP_FRAMES, VIDPROP_HEIGHT,
                             VIDPROP_WIDTH, 3), np.uint8)
    frames_output = np.empty((VIDPROP_FRAMES - 1, VIDPROP_HEIGHT,
                             VIDPROP_WIDTH * 2, 3), np.uint8)
    for frame_idx in range(VIDPROP_FRAMES):
        if not vid.isOpened():
            print('Error while reading frames!')
            exit(1)

        frame_ok, frame = vid.read()
        if not frame_ok:
            print('Error while reading frames!')
            exit(1)

        frames_input[frame_idx] = frame
    vid.release()
    print('Read %d x %d, %d frames@%.2fFPS.' %
          (VIDPROP_WIDTH, VIDPROP_HEIGHT, VIDPROP_FRAMES, VIDPROP_FPS))

    model = nn.DataParallel(RAFT(args))
    checkpoint = torch.load(args.model)
    weight = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(weight)
    model = model.module
    model.to(DEVICE)
    model.eval()

    flow_init = None
    with torch.no_grad():
        for frame_idx in trange(VIDPROP_FRAMES - 1, ncols=120):
            image1 = to_tensor(frames_input[frame_idx])
            image2 = to_tensor(frames_input[frame_idx + 1])

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(
                image1, image2, iters=20, test_mode=True, flow_init=flow_init)
            if args.warmup:
                flow_init = flow_low.detach()

            flow = flow_up[0].cpu().permute(1, 2, 0).numpy()
            flow_viz = flow_to_image(flow)

            frames_output[frame_idx] = np.concatenate([frames_input[frame_idx],
                                                       flow_viz[:, :, [2, 1, 0]]], axis=1)

    out = cv2.VideoWriter('visualization/composed.avi',
                          cv2.VideoWriter_fourcc(*'XVID'),
                          VIDPROP_FPS, (VIDPROP_WIDTH * 2, VIDPROP_HEIGHT))
    for frame in frames_output:
        out.write(frame)
    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='The input video file')
    parser.add_argument('--model', type=str, required=True,
                        help='The model weight')
    parser.add_argument('--warmup', action='store_true',
                        help='use warm-up mode')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision',
                        action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true',
                        help='use efficent correlation implementation')
    parser.add_argument('--hidden', type=int, default=128,
                        help='The hidden size of the updater')
    parser.add_argument('--context', type=int, default=128,
                        help='The context size of the updater')
    args = parser.parse_args()

    os.makedirs('visualization', exist_ok=True)

    compose(args)
