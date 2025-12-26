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


# These work well, but did just 

# def zod_homo_loss(four_pred, sat_img, rot_gt, trans_gt, resolution, gamma=0.85):
#     B, _, H, W = sat_img.shape
#     device = sat_img.device
#     sz = [B, 1, H, W]

#     if isinstance(four_pred, list):
#         preds = four_pred
#     else:
#         preds = [four_pred]

#     # --- ground-truth center location in pixels ---
#     cx = (W - 1) / 2.0
#     cy = (H - 1) / 2.0
#     res = resolution.view(B, 1).to(device)

#     x0_m = torch.zeros(B, 1, device=device)
#     y0_m = torch.zeros(B, 1, device=device)

#     cos_t = rot_gt[:, 0, 0].view(B, 1)          # (B,1)
#     sin_t = rot_gt[:, 1, 0].view(B, 1)          # (B,1)
#     dx = trans_gt[:, 0].view(B, 1)              # (B,1)
#     dy = trans_gt[:, 1].view(B, 1)              # (B,1)

#     x1_m = cos_t * x0_m - sin_t * y0_m + dx   # (B,1)
#     y1_m = sin_t * x0_m + cos_t * y0_m + dy   # (B,1)

#     x_gt_pix = x1_m / res + cx     # (B,1)
#     y_gt_pix = y1_m / res + cy     # (B,1)
#     y = torch.cat([x_gt_pix, y_gt_pix], dim=1)  # (B,2)
    
#     # --- GT yaw (deg) from rot_gt ---
#     yaw_gt_deg = torch.rad2deg(torch.atan2(rot_gt[:, 1, 0], rot_gt[:, 0, 0]))  # (B,)

#     # # DEBUG
#     # print("DEBUG zod_homo_loss: B,H,W =", B, H, W)
#     # print("  x_gt_pix shape:", x_gt_pix.shape)
#     # print("  y_gt_pix shape:", y_gt_pix.shape)
#     # print("  y shape:", y.shape)
#     # print("  cos_t:", cos_t.shape)
#     # print("  cos_t:", cos_t.shape)
#     # print("  x1_m:", x1_m.shape)
#     total_loss = 0.0
#     x_last = None
#     yaw_pred_last = None

#     for i, four in enumerate(preds):
#         weight = gamma ** (len(preds) - i - 1)
#         H_mat = get_homograpy(four, sz)  # (B,3,3)

#         pts = torch.tensor(
#             [[[cx], [cy], [1.0]]],
#             dtype=torch.float32, device=device
#         ).repeat(B, 1, 1)               # (B,3,1)

#         x = H_mat.bmm(pts)              # (B,3,1)
#         x = x / x[:, 2:3, :]
#         x = x[:, 0:2, 0]                # (B,2)

#         # DEBUG
#         # print("  x shape at iter", i, ":", x.shape)
#         # assert x.shape == y.shape, f"Shape mismatch: x {x.shape}, y {y.shape}"

#         i_loss = ((x - y) ** 2).mean()
#         total_loss = total_loss + weight * i_loss
#         x_last = x

#     # simple metric: pixel error and meter error from last prediction
#     err_pix = (x_last - y).detach()                 # (B,2)
#     err_pix_l2 = (err_pix ** 2).sum(dim=1).sqrt()

#     err_m = err_pix * res                           # (B,2)
#     err_m_l2 = (err_m ** 2).sum(dim=1).sqrt()

#     metrics = {
#         "pix_l2": float(err_pix_l2.mean().item()),
#         "m_l2":   float(err_m_l2.mean().item()),
#         "pred_x_pix": float(x_last[:, 0].mean().item()),
#         "pred_y_pix": float(x_last[:, 1].mean().item()),
#     }
#     return total_loss, metrics

# def zod_se2_loss(four_pred, sat_img, rot_gt, trans_gt, resolution, gamma=0.85):
#     """
#     four_pred: list of (B,2,2,2) homography corner displacements from HCNet
#     sat_img:   (B,3,H,W) satellite input (only H,W used)
#     rot_gt:    (B,3,3) ground-truth SE(2) rotation (only yaw used)
#     trans_gt:  (B,3)   ground-truth translation [tx, ty, tz] in meters
#     resolution:(B,)    meters per pixel
#     Returns:
#         loss:    scalar tensor
#         metrics: dict with translation/yaw predictions and errors
#     """
#     B, _, H, W = sat_img.shape
#     device = sat_img.device
#     sz = [B, 1, H, W]

#     if isinstance(four_pred, list):
#         preds = four_pred
#     else:
#         preds = [four_pred]

#     txy_gt = trans_gt[:, :2]              # (B,2)
#     res = resolution.view(B, 1).to(device)

#     # ground-truth yaw in degrees from rot_gt
#     yaw_gt = torch.rad2deg(torch.atan2(rot_gt[:, 1, 0], rot_gt[:, 0, 0]))  # (B,)

#     total_loss = 0.0
#     txy_pred_last = None
#     yaw_pred_last = None

#     for i, four in enumerate(preds):
#         weight = gamma ** (len(preds) - i - 1)

#         # homography from predicted corner displacements
#         H_mat = get_homograpy(four, sz)        # (B,3,3)

#         # ---- translation from homography (center shift) ----
#         cx = (W - 1) / 2.0
#         cy = (H - 1) / 2.0
#         pts = torch.tensor(
#             [[[cx], [cy], [1.0]]],
#             dtype=torch.float32,
#             device=device
#         ).repeat(B, 1, 1)                      # (B,3,1)

#         x = H_mat.bmm(pts)                     # (B,3,1)
#         x = x / x[:, 2:3, :]                   # normalize
#         x = x[:, 0:2, 0]                       # (B,2), pixel coords

#         dpix = x - torch.tensor(
#             [cx, cy], dtype=torch.float32, device=device
#         ).view(1, 2)                           # (B,2)

#         txy_pred = dpix * res                  # (B,2) meters

#         # ---- yaw from homography ----
#         # apply H to center and a point a bit above center to recover orientation
#         pts2 = torch.tensor(
#             [[[cx], [cy - 10.0], [1.0]]],
#             dtype=torch.float32,
#             device=device
#         ).repeat(B, 1, 1)                      # (B,3,1)

#         x2 = H_mat.bmm(pts2)
#         x2 = x2 / x2[:, 2:3, :]
#         x2 = x2[:, 0:2, 0]                     # (B,2)

#         # vector from center to "up" point in warped frame
#         v = x2 - x                             # (B,2)
#         yaw_pred = torch.rad2deg(torch.atan2(v[:, 0], -v[:, 1]))  # (B,)

#         # ---- SE(2) loss: translation + yaw ----
#         err_t = txy_pred - txy_gt             # (B,2)
#         trans_l2 = (err_t ** 2).sum(dim=1).sqrt()  # (B,)

#         err_yaw = yaw_pred - yaw_gt
#         # wrap to [-180,180]
#         err_yaw = (err_yaw + 180.0) % 360.0 - 180.0

#         # simple weighted sum (tune weights as needed)
#         loss_i = trans_l2.mean() + 0.01 * err_yaw.abs().mean()
#         total_loss = total_loss + weight * loss_i

#         # remember last prediction for metrics
#         txy_pred_last = txy_pred
#         yaw_pred_last = yaw_pred

#     # metrics from last prediction
#     err_t_last = (txy_pred_last - txy_gt).detach()
#     trans_l2_last = (err_t_last ** 2).sum(dim=1).sqrt()
#     err_yaw_last = (yaw_pred_last - yaw_gt).detach()
#     err_yaw_last = (err_yaw_last + 180.0) % 360.0 - 180.0

#     metrics = {
#         "trans_l2_m":       float(trans_l2_last.mean().item()),
#         "trans_abs_dx_m":   float(err_t_last[:, 0].abs().mean().item()),
#         "trans_abs_dy_m":   float(err_t_last[:, 1].abs().mean().item()),
#         "pred_dx_m":        float(txy_pred_last[:, 0].mean().item()),
#         "pred_dy_m":        float(txy_pred_last[:, 1].mean().item()),
#         "gt_dx_m":          float(txy_gt[:, 0].mean().item()),
#         "gt_dy_m":          float(txy_gt[:, 1].mean().item()),
#         "yaw_err_deg":      float(err_yaw_last.abs().mean().item()),
#         "pred_yaw_deg":     float(yaw_pred_last.mean().item()),
#         "gt_yaw_deg":       float(yaw_gt.mean().item()),
#     }

#     return total_loss, metrics




# old losses from HCNet GPS regression (not used for ZOD)
# def corr_loss(grd_gps, sat_gps, corr_fn, infoLoss, args,  sat_delta=None, transformed_center = None, sz = [512,512]):
#     zoom = args.zoom
#     sat_size = args.sat_size
#     batch_sz =  grd_gps.shape[0]
#     h,w = corr_fn.shape[-2:]    
#     if transformed_center is not None:
#         corr_map = corr_fn.view(batch_sz,h,w,h,w).permute(0,3,4,1,2).view(batch_sz,h*w,h,w) 
#         transformed_center_ = transformed_center[:,:,0]*h/sz[0] # /4
#         transformed_center_ = 2*transformed_center_/(corr_map.shape[-1]-1) -1 
#         corr_map = F.grid_sample(corr_map, transformed_center_.unsqueeze(1).unsqueeze(1), align_corners=True) # [x,y]
#         corr_map = corr_map.view(batch_sz,h,w)
#     else: 
#         corr_map = corr_fn.view(batch_sz,h,w,h,w)[:,h//2,w//2,:,:]  

#     if len(grd_gps.shape) == 3:
#         y = get_pixel_tensor(sat_gps[:,0], sat_gps[:,1], grd_gps[:,0,0],grd_gps[:,0,1], zoom, sat_size)
#     else:
#         y = get_pixel_tensor(sat_gps[:,0], sat_gps[:,1], grd_gps[:,0],grd_gps[:,1], zoom, sat_size) # get ground truth pixel coords
#     y = torch.cat((y[0].reshape(-1,1),y[1].reshape(-1,1)),dim = 1)
#     if sat_delta is not None:
#         y = sat_delta/sz[0]*sat_size
#     loss = infoLoss(corr_map, y/sat_size*corr_map.shape[-1])

#     return loss

# def vigor_gps_loss(four_pred, grd_gps, sat_gps, args, sat_delta=None, orien = False, transformed_center = None, sz = [512,512], gamma = 0.85, ori_angle = 0, w3 = 10.0):
#     """
#     loss: pixel_wise
#     grd_gps: ground truth GPS of grd images
#     """
#     zoom = args.zoom
#     sat_size = args.sat_size

#     sz = [grd_gps.shape[0] ]+ [1] + sz
#     n_predictions = len(four_pred) if type(four_pred) == list else 1
#     v_loss = 0.0

#     # print(x[0,0].item(), x[0,1].item())
    
#     pot_num = 0
#     if len(grd_gps.shape) == 3:
#         batch_num, pot_num = grd_gps.shape[:2]
#         sat_gps = sat_gps.unsqueeze(1).repeat(1,pot_num,1).reshape(-1,2)
#         grd_gps = grd_gps.reshape(-1,2)

#     y = get_pixel_tensor(sat_gps[:,0], sat_gps[:,1], grd_gps[:,0],grd_gps[:,1], zoom, sat_size=sat_size) # get ground truth pixel coords
#     y = torch.cat((y[0].reshape(-1,1),y[1].reshape(-1,1)),dim = 1) #  [batch*pot_num, 2]
#     if sat_delta is not None:
#         y = sat_delta/sz[2]*sat_size

#     for i in range(n_predictions):
#         i_weight = gamma**(n_predictions - i - 1)
#         H = get_homograpy(four_pred[i], sz) if type(four_pred) == list else get_homograpy(four_pred, sz)
#         if transformed_center is None:
#             points = torch.cat((torch.ones((1,1))*sz[3]//2.0, torch.ones((1,1))*sz[2]//2.0, torch.ones((1,1))),
#                             dim=0).unsqueeze(0).repeat(sz[0], 1, 1).to(grd_gps.device) # [N,2,1] only one point
#         else:
#             points = torch.cat((transformed_center, torch.ones((sz[0],1,transformed_center.shape[-1])).to(grd_gps.device)), dim = 1).to(grd_gps.device)
#         x = H.bmm(points)
#         x = x / x[:, 2, :].unsqueeze(1)        
#         x[:,:2,:] = x[:,:2,:]
#         if orien:
#             dx = x[:,0, 1]- x[:,0, 0]
#             dy = x[:,1, 0]- x[:,1, 1]     
#             ori = -torch.rad2deg(torch.atan2(dx,dy))
#             ori_loss = (ori-ori_angle).abs()
#             ori_loss = ori_loss.nanmean()
#         x = x[:, 0:2, :]/sz[2]*sat_size      # [batch, 2, put_num]
#         if pot_num!= 0:
#             x=x.permute(0,2,1).reshape(-1,2)
#         else:
#             x = x[:,:,0]
#         i_loss = torch.nanmean((x-y)**2)/sat_size*sz[2]/sat_size*sz[2]
#         i_loss += ori_loss*w3 if orien else 0
#         v_loss += i_weight * i_loss    

#     est_lat, est_lon = get_latlon_tensor(sat_gps[:,0], sat_gps[:,1], x[:,0], x[:,1], zoom, sat_size)
#     epe = gps2distance(grd_gps[:,0],grd_gps[:,1], est_lat, est_lon)

#     metrics = { # for people
#         'epe': epe.nanmean().item(), 
#         '1px': (epe < 1).float().mean().item(), # R@1
#         '3px': (epe < 3).float().mean().item(),
#         '5px': (epe < 5).float().mean().item(),
#     }
#     return v_loss, metrics 

# def kitti_ori_loss(four_pred, grd_gps, args, ori_angle, sz = [512,512], gamma = 0.85):
#     sz = [grd_gps.shape[0] ]+ [1] + sz
#     n_predictions = len(four_pred) if type(four_pred) == list else 1
#     v_loss = 0.0
#     for i in range(n_predictions):
#         i_weight = gamma**(n_predictions - i - 1)
#         H = get_homograpy(four_pred[i], sz) if type(four_pred) == list else get_homograpy(four_pred, sz)
#         points = torch.cat((torch.ones((1,1))*sz[3]//2.0, torch.ones((1,1))*sz[2]//2.0, torch.ones((1,1))),
#                                     dim=0).unsqueeze(0).repeat(sz[0], 1, 1).to(grd_gps.device) # [N,2,1] only one point
#         points_ = torch.cat((torch.ones((1,1))*sz[3]//2.0, torch.ones((1,1))*sz[2]//2.0-10, torch.ones((1,1))),
#                                     dim=0).unsqueeze(0).repeat(sz[0], 1, 1).to(grd_gps.device) # [N,2,1] only one point
#         points = torch.cat((points,points_), dim = 2)
#         x = H.bmm(points)
#         x = x / x[:, 2, :].unsqueeze(1)        
#         x = x[:, 0:2, :]/sz[2]*args.sat_size
#         dx = x[:,0, 1]- x[:,0, 0]
#         dy = x[:,1, 0]- x[:,1, 1]     
#         ori = -torch.rad2deg(torch.atan2(dx,dy))
#         ori_epe = (ori-ori_angle).abs()
#         ori_loss = ori_epe # - torch.min(torch.ones_like(ori_loss), ori_epe)
#         ori_loss = ori_loss.nanmean()
#         i_loss = ori_loss
#         # torch.nanmean((x-y)**2)
#         v_loss += i_weight * i_loss    
#     return v_loss

