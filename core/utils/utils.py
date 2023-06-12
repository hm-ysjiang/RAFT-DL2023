import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from scipy import interpolate


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2,
                         pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device),
                            torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def create_flow_grid(flow):
    B, C, H, W = flow.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if flow.is_cuda:
        grid = grid.to(flow.get_device())
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    return vgrid.permute(0, 2, 3, 1)


def warp_flow(x, flow, use_mask=False):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    Inputs:
    x: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow
    Returns:
    ouptut: [B, C, H, W]
    """
    vgrid = create_flow_grid(flow)
    return warp_vgrid(x, vgrid, use_mask)


def warp_vgrid(x: torch.Tensor, vgrid: torch.Tensor, use_mask=False):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    Inputs:
    x: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow
    Returns:
    ouptut: [B, C, H, W]
    """
    output = F.grid_sample(x, vgrid, align_corners=True)
    if use_mask:
        mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
        mask = F.grid_sample(mask, vgrid, align_corners=True)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        output = output * mask

    return output


def SSIM_error(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)

    # (input, kernel, stride, padding)
    sigma_x = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 0) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def photometric_error(img1: torch.Tensor, img2: torch.Tensor,
                      flow: torch.Tensor, valid: torch.Tensor):
    img1_warped = warp_flow(img2, flow)
    l1_err = (img1_warped * valid - img1 * valid).abs()
    ssim_err = SSIM_error(img1_warped * valid, img1 * valid)
    return l1_err.mean(), ssim_err.mean()


def photometric_error_masked(img1: torch.Tensor, img2: torch.Tensor,
                             vgrid: torch.Tensor, valid: torch.Tensor):
    maskw = valid.mean() + 1e-6
    img1_warped = warp_vgrid(img2, vgrid)
    l1_err = (img1_warped * valid - img1 * valid).abs()
    ssim_err = SSIM_error(img1_warped * valid, img1 * valid)
    return l1_err.mean() / maskw, ssim_err.mean() / maskw


# GMFlowNet
@torch.no_grad()
def compute_supervision_match(flow, occlusions, scale: int):
    N, _, H, W = flow.shape
    Hc, Wc = int(np.ceil(H / scale)), int(np.ceil(W / scale))

    occlusions_c = occlusions[:, :, ::scale, ::scale]
    flow_c = flow[:, :, ::scale, ::scale] / scale
    occlusions_c = occlusions_c.reshape(N, Hc * Wc)

    grid_c = coords_grid(N, Hc, Wc,
                         device=flow.device).permute(0, 2, 3, 1).reshape(N, Hc * Wc, 2)
    warp_c = grid_c + flow_c.permute(0, 2, 3, 1).reshape(N, Hc * Wc, 2)
    warp_c = warp_c.round().long()

    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)

    occlusions_c[out_bound_mask(warp_c, Wc, Hc)] = 1
    warp_c = warp_c[..., 0] + warp_c[..., 1] * Wc

    b_ids, i_ids = torch.split(torch.nonzero(occlusions_c == 0), 1, dim=1)
    conf_matrix_gt = torch.zeros(N, Hc * Wc, Hc * Wc, device=flow.device)
    j_ids = warp_c[b_ids, i_ids]
    conf_matrix_gt[b_ids, i_ids, j_ids] = 1

    return conf_matrix_gt


def compute_match_loss(conf, conf_gt):  # GMFlowNet
    pos_mask, neg_mask = conf_gt == 1, conf_gt == 0

    conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
    loss_pos = -torch.log(conf[pos_mask])
    loss_neg = -torch.log(1 - conf[neg_mask])

    return loss_pos.mean() + loss_neg.mean()


def magsq(x: torch.Tensor, dim):
    return torch.sum(x**2, dim, keepdim=(dim is not None))


def create_border_mask(tensor: torch.Tensor, ratio=0.1):
    B, _, H, W = tensor.shape
    sz = np.ceil(min(H, W) * ratio).astype(int).item(0)
    border_mask = torch.zeros((H, W), dtype=tensor.dtype, device=tensor.device)
    border_mask[sz:-sz, sz:-sz] = 1.0
    border_mask = border_mask.view(1, 1, H, W).expand(B, -1, -1, -1)
    return border_mask.detach()


def fwdbwd_occ_mask(flow_fwd: torch.Tensor, flow_bwd: torch.Tensor,
                    vgrid_fwd: torch.Tensor, vgrid_bwd: torch.Tensor,
                    use_border_mask=False, return_warpdiff=False):
    mag_flow = magsq(flow_fwd, 1) + magsq(flow_bwd, 1)
    flow_bwd_warped = warp_vgrid(flow_bwd, vgrid_fwd, True)
    flow_fwd_warped = warp_vgrid(flow_fwd, vgrid_bwd, True)
    flow_fwd_warpdiff = flow_fwd + flow_bwd_warped
    flow_bwd_warpdiff = flow_bwd + flow_fwd_warped
    occ_thresh = 0.01 * mag_flow + 0.5
    occ_fwd = (magsq(flow_fwd_warpdiff, 1) > occ_thresh).float()
    occ_bwd = (magsq(flow_bwd_warpdiff, 1) > occ_thresh).float()
    mask_fwd = (1 - occ_fwd)
    mask_bwd = (1 - occ_bwd)

    if use_border_mask:
        border_mask = create_border_mask(flow_fwd)
        mask_fwd = border_mask * mask_fwd
        mask_bwd = border_mask * mask_bwd

    if return_warpdiff:
        return mask_fwd, mask_bwd, flow_fwd_warpdiff, flow_bwd_warpdiff
    return mask_fwd, mask_bwd
