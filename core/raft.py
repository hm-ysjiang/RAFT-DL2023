import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = (args.hidden // 4) * 3
            self.context_dim = cdim = args.context // 2
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = args.hidden
            self.context_dim = cdim = args.context
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(
                output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = SmallEncoder(
                output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(
                self.args, hidden_dim=hdim, input_dim=cdim)

        else:
            self.fnet = BasicEncoder(
                output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(
                output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(
                self.args, hidden_dim=hdim, input_dim=cdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, image1, image2, iters=12, flow_init=None, global_matching=False, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        batch_size, _, fmap_height, fmap_width = fmap1.shape
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(
                fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)
        softCorrMap = F.softmax(corr_fn.corrMap, dim=2) \
            * F.softmax(corr_fn.corrMap, dim=1)

        if flow_init is not None:
            coords1 = coords1 + flow_init
        elif global_matching:
            # GMFlowNet
            match12, match_idx12 = softCorrMap.max(dim=2)  # (N, fH*fW)
            match21, match_idx21 = softCorrMap.max(dim=1)

            for b_idx in range(batch_size):
                match21_b = match21[b_idx, :]
                match_idx12_b = match_idx12[b_idx, :]
                match21[b_idx, :] = match21_b[match_idx12_b]

            matched = (match12 - match21) == 0  # (N, fH*fW)
            coords_index = torch.arange(fmap_height * fmap_width) \
                .unsqueeze(0) \
                .repeat(batch_size, 1) \
                .to(softCorrMap.device)
            coords_index[matched] = match_idx12[matched]

            # matched coords
            coords_index = coords_index.reshape(batch_size,
                                                fmap_height, fmap_width)
            coords_x = coords_index % fmap_width
            coords_y = coords_index // fmap_width

            coords_xy = torch.stack([coords_x, coords_y], dim=1).float()
            coords1 = coords_xy

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(
                    net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions, softCorrMap
