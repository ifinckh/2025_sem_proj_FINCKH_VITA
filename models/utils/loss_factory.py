import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.utils  import get_homograpy, gps2distance
from models.utils.Mercator import get_latlon_tensor, get_pixel_tensor

class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=10, sample = True):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.sample = sample
        self.F_LogSoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, sim_matrix, positive_indices):       
        batch_size, h, w = sim_matrix.shape      
        sim_matrix_logexp = -self.F_LogSoftmax(sim_matrix.view(batch_size,-1)/self.temperature)
        positive_indices = 2*positive_indices/(w-1)- 1
        sim_matrix_exp = F.grid_sample(sim_matrix_logexp.view(batch_size,1,h, w), positive_indices.unsqueeze(1).unsqueeze(1), align_corners=True)
            
        loss = torch.mean(sim_matrix_exp)
        
        return loss


def corr_loss_zod(corr_fn, infoLoss, y_pix, sz= [512,512]):
    """
    corr_fn: either corr_fn.corr_pyramid[0] OR the raw corr tensor from model (depends on your implementation)
            The original code reshapes corr_fn into (B,h,w,h,w).
    sz: size of the input image that corr was built from (H,W). Used only if transformed_center is used.
    """
    # corr tensor expected last two dims are (h,w)
    corr = corr_fn
    B = corr_fn.shape[0]
    h, w = corr_fn.shape[-2:]
    
    # take correlation from the center query (h//2,w//2)
    corr_map = corr.view(B, h, w, h, w)[:, h//2, w//2, :, :]  # (B,h,w)

    # map GT pixel to corr grid coords
    pos = y_pix / sz[0] * (w - 1)  # (B,2) in [0,w-1]
    return infoLoss(corr_map, pos)

def zod_homography_loss(four_pred, sat_img, rot_gt, trans_gt, resolution, orien=True, gamma=0.85, w3=10.0):
    B, _, H, W = sat_img.shape
    device = sat_img.device
    sz = [B, 1, H, W]

    if isinstance(four_pred, list):
        preds = four_pred
    else:
        preds = [four_pred]

    # --- ground-truth center location in pixels ---
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    res = resolution.view(B, 1).to(device)

    x0_m = torch.zeros(B, 1, device=device)
    y0_m = torch.zeros(B, 1, device=device)

    cos_t = rot_gt[:, 0, 0].view(B, 1)          # (B,1)
    sin_t = rot_gt[:, 1, 0].view(B, 1)          # (B,1)
    dx = trans_gt[:, 0].view(B, 1)              # (B,1)
    dy = trans_gt[:, 1].view(B, 1)              # (B,1)

    x1_m = cos_t * x0_m - sin_t * y0_m + dx   # (B,1)
    y1_m = sin_t * x0_m + cos_t * y0_m + dy   # (B,1)

    x_gt_pix = x1_m / res + cx     # (B,1)
    y_gt_pix = y1_m / res + cy     # (B,1)
    # y_gt_pix = cy - (y1_m / res)   # (B,1)  # metric +y up -> pixel y down
    y = torch.cat([x_gt_pix, y_gt_pix], dim=1)  # (B,2)
    
    # --- GT yaw (deg) from rot_gt ---
    txy_gt = trans_gt[:, :2]              # (B,2)
    yaw_gt_deg = torch.rad2deg(torch.atan2(rot_gt[:, 1, 0], rot_gt[:, 0, 0]))  # (B,)

    v_loss = 0.0
    x_last = None
    yaw_pred_last = None

    for i, four in enumerate(preds):
        weight = gamma ** (len(preds) - i - 1)
        H_mat = get_homograpy(four, sz)  # (B,3,3)

        pts = torch.tensor(
            [[[cx], [cy], [1.0]]],
            dtype=torch.float32, device=device
        ).repeat(B, 1, 1)               # (B,3,1)

        x = H_mat.bmm(pts)              # (B,3,1)
        x = x / x[:, 2:3, :]
        x = x[:, 0:2, 0]                # (B,2)
        
        dpix = x - torch.tensor(
            [cx, cy], dtype=torch.float32, device=device
        ).view(1, 2)                           # (B,2)

        txy_pred = dpix * res                  # (B,2) meters
        # txy_pred[:, 1] = -txy_pred[:, 1] # pixel y down -> metric +y up
        
        # ---- yaw from homography ----
        # apply H to center and a point a bit above center to recover orientation
        pts2 = torch.tensor(
            [[[cx], [cy - 10.0], [1.0]]],
            dtype=torch.float32,
            device=device
        ).repeat(B, 1, 1)                      # (B,3,1)

        x2 = H_mat.bmm(pts2)
        x2 = x2 / x2[:, 2:3, :]
        x2 = x2[:, 0:2, 0]                     # (B,2)

        # vector from center to "up" point in warped frame
        v = x2 - x                             # (B,2)
        yaw_pred = torch.rad2deg(torch.atan2(v[:, 0], -v[:, 1]))  # (B,)

        # ---- loss: translation + yaw ----

        # err_yaw = yaw_pred - yaw_gt_deg
        # wrap to [-180,180]
        # err_yaw = (err_yaw + 180.0) % 360.0 - 180.0
        # With this (using inputs in DEGREES):
        diff_rad = torch.deg2rad(yaw_pred - yaw_gt_deg)
        ori_loss = (1.0 - torch.cos(diff_rad)).nanmean()
        
        # print(ori_loss.item() * w3,torch.nanmean((x-y)**2).item())

        # Original vigor_gps_loss style scaling
        i_loss = torch.nanmean((x-y)**2)
        i_loss += ori_loss * w3
        v_loss += weight * i_loss
        # print("i_loss:", i_loss.item(), v_loss.item(), weight)
        
        # remember last prediction for metrics
        x_last = x
        txy_pred_last = txy_pred
        yaw_pred_last = yaw_pred

    # simple metric: pixel error and meter error from last prediction
    err_pix = (x_last - y).detach()                 # (B,2)
    err_pix_l2 = (err_pix ** 2).sum(dim=1).sqrt()

    # metrics from last prediction in m
    err_t_last = (txy_pred_last - txy_gt).detach()
    trans_l2_last = (err_t_last ** 2).sum(dim=1).sqrt()
    err_yaw_last = (yaw_pred_last - yaw_gt_deg).detach()
    err_yaw_last = (err_yaw_last + 180.0) % 360.0 - 180.0

    metrics = {
        "pix_l2": float(err_pix_l2.mean().item()),
        "pred_x_pix": float(x_last[:, 0].mean().item()),
        "pred_y_pix": float(x_last[:, 1].mean().item()),
        "trans_l2_m":       float(trans_l2_last.mean().item()),
        "trans_abs_dx_m":   float(err_t_last[:, 0].abs().mean().item()),
        "trans_abs_dy_m":   float(err_t_last[:, 1].abs().mean().item()),
        "pred_dx_m":        float(txy_pred_last[:, 0].mean().item()),
        "pred_dy_m":        float(txy_pred_last[:, 1].mean().item()),
        "gt_dx_m":          float(txy_gt[:, 0].mean().item()),
        "gt_dy_m":          float(txy_gt[:, 1].mean().item()),
        "yaw_err_deg":      float(err_yaw_last.abs().mean().item()),
        "pred_yaw_deg":     float(yaw_pred_last.mean().item()),
        "gt_yaw_deg":       float(yaw_gt_deg.mean().item()),
    }
    return v_loss, metrics, y

def predict_pose(four_pred, sat_img, resolution):
    B, _, H, W = sat_img.shape
    device = sat_img.device
    sz = [B, 1, H, W]
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    res = resolution.view(B, 1).to(device)


    if isinstance(four_pred, list):
        preds = four_pred
    else:
        preds = [four_pred]

    
    for i, four in enumerate(preds):
        H_mat = get_homograpy(four, sz)  # (B,3,3)

        pts = torch.tensor(
            [[[cx], [cy], [1.0]]],
            dtype=torch.float32, device=device
        ).repeat(B, 1, 1)               # (B,3,1)

        x = H_mat.bmm(pts)              # (B,3,1)
        x = x / x[:, 2:3, :]
        x = x[:, 0:2, 0]                # (B,2)
        
        dpix = x - torch.tensor(
            [cx, cy], dtype=torch.float32, device=device
        ).view(1, 2)                           # (B,2)

        txy_pred = dpix * res                  # (B,2) meters
        # txy_pred[:, 1] = -txy_pred[:, 1] # pixel y down -> metric +y up
        
        # ---- yaw from homography ----
        # apply H to center and a point a bit above center to recover orientation
        pts2 = torch.tensor(
            [[[cx], [cy - 10.0], [1.0]]],
            dtype=torch.float32,
            device=device
        ).repeat(B, 1, 1)                      # (B,3,1)

        x2 = H_mat.bmm(pts2)
        x2 = x2 / x2[:, 2:3, :]
        x2 = x2[:, 0:2, 0]                     # (B,2)

        # vector from center to "up" point in warped frame
        v = x2 - x                             # (B,2)
        yaw_pred = torch.rad2deg(torch.atan2(v[:, 0], -v[:, 1]))  # (B,)
        
        # remember last prediction for metrics
        txy_pred_last = txy_pred
        yaw_pred_last = yaw_pred
 
    # metrics from last prediction in m
    # x_pred = float(txy_pred_last[:, 0].mean().item())
    # y_pred = float(txy_pred_last[:, 1].mean().item())
    # yaw_pred = float(yaw_pred_last.mean().item())
    
    return txy_pred_last, yaw_pred_last   # shapes: (B,), (B,), (B,)
