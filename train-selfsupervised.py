from __future__ import division, print_function

import sys  # nopep8

sys.path.append('core')  # nopep8

import argparse
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn_backend
import torch.nn as nn
import torch.optim as optim
from logger import Logger
from raft import RAFT
from utils.utils import photometric_error

import datasets
import evaluate

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000
SSIM_WEIGHT = 0.84


def sequence_loss(flow_preds: List[torch.Tensor], flow_gt: torch.Tensor,
                  image1: torch.Tensor, image2: torch.Tensor,
                  valid: torch.Tensor, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss: torch.Tensor = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        l1_err, ssim_err = photometric_error(image1, image2,
                                             flow_preds[i],
                                             valid[:, None])
        i_loss = (1 - SSIM_WEIGHT) * l1_err + SSIM_WEIGHT * ssim_err
        flow_loss += i_weight * i_loss

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model, steps):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, steps+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def train(args):
    train_loader = datasets.fetch_dataloader(args)
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    optimizer, scheduler = fetch_optimizer(args, model,
                                           len(train_loader) * args.num_epochs)
    model.cuda()
    model.train()

    epoch_start = 0
    if args.restore_ckpt is not None:
        checkpoint = torch.load(args.restore_ckpt)
        weight: OrderedDict[str, Any] = checkpoint['model']
        if args.reset_context:
            _weight = OrderedDict()
            for key, val in checkpoint.items():
                if '.cnet.' not in key:
                    _weight[key] = val
            weight = _weight
        model.load_state_dict(weight, strict=(not args.allow_nonstrict))

        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = checkpoint['scheduler']
        epoch_start = checkpoint['epoch']

    if args.freeze_bn:
        model.module.freeze_bn()

    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(args.name, len(train_loader) * epoch_start)

    VAL_FREQ = 5000
    add_noise = True
    best_evaluation = None

    for epoch in range(epoch_start, args.num_epochs):
        logger.initPbar(len(train_loader), epoch + 1)
        for batch_idx, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*
                          image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*
                          image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)

            loss, metrics = sequence_loss(flow_predictions, flow,
                                          image1, image2, valid,
                                          args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push({'loss': loss.item()})

        logger.closePbar()
        PATH = 'checkpoints/%s/model.pth' % args.name
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler
        }, PATH)

        results = {}
        for val_dataset in args.validation:
            if val_dataset == 'chairs':
                results.update(evaluate.validate_chairs(model.module))
            elif val_dataset == 'sintel':
                results.update(evaluate.validate_sintel(model.module))
            elif val_dataset == 'kitti':
                results.update(evaluate.validate_kitti(model.module))
        logger.write_dict(results, 'epoch')

        evaluation_score = np.mean(list(results.values()))
        if best_evaluation is None or evaluation_score < best_evaluation:
            best_evaluation = evaluation_score
            PATH = 'checkpoints/%s/model-best.pth' % args.name
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler
            }, PATH)

        model.train()
        if args.freeze_bn:
            model.module.freeze_bn()

    logger.close()

    return best_evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--freeze_bn', action='store_true',
                        help="freeze the batch norm layer")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--allow_nonstrict', action='store_true',
                        help='allow non-strict loading')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int,
                        nargs='+', default=[368, 768])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--mixed_precision',
                        action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')

    parser.add_argument('--hidden', type=int, default=128,
                        help='The hidden size of the updater')
    parser.add_argument('--context', type=int, default=128,
                        help='The context size of the updater')
    parser.add_argument('--reset_context', action='store_true')

    args = parser.parse_args()
    if args.hidden != 128 or args.context != 128:
        args.reset_context = True

    torch.manual_seed(1234)
    np.random.seed(1234)

    cudnn_backend.benchmark = True

    os.makedirs(Path(__file__).parent.joinpath('checkpoints', args.name),
                exist_ok=True)

    train(args)
