import torch
import torch.nn as nn
from .update import GMA
from .corr import CorrBlock
from .utils.utils import *
from .efficientnet_pytorch.model import EfficientNet
from .utils.torch_geometry import get_perspective_transform
from .LIDAR_BEV.LIDAR_BEV_backbone import LidarBEVBackbone
# import torchgeometry as tgm
import csv
import os


autocast = torch.cuda.amp.autocast

# def log_feat_csv(img_feature, lidar_feature):
#     filepath = "metrics/feature_visualization/feat.csv"
#     # Check if file already exists
#     file_exists = os.path.isfile(filepath)    
#     with open(filepath, mode="a", newline="") as f:
#         writer = csv.writer(f)

#         # Write header if new file
#         if not file_exists:
#             writer.writerow(["image_feature", "lidar_feature"])

#         # Append data row
#         writer.writerow([img_feature, lidar_feature])

import os
import torch

def append_features_pth(pth_path, img_feature, lidar_feature, correlation=None):
    """
    Appends (img_feature, lidar_feature) to a .pth file as a list of dict entries.

    img_feature / lidar_feature: torch.Tensor or numpy array
    meta: optional dict (e.g., {"epoch":..., "i_batch":..., "sample_id":...})
    flush_every: if you call this in a loop, set >1 to reduce disk I/O (you'd buffer yourself).
    """
    os.makedirs(os.path.dirname(pth_path), exist_ok=True)

    # Make tensors CPU + detached (safe for serialization)
    if not torch.is_tensor(img_feature):
        img_feature = torch.as_tensor(img_feature)
    if not torch.is_tensor(lidar_feature):
        lidar_feature = torch.as_tensor(lidar_feature)

    entry = {
        "image_feature": img_feature.detach().cpu(),
        "lidar_feature": lidar_feature.detach().cpu(),
        "correlation": [x.detach().cpu() for x in correlation] if correlation is not None else None
    }

    if os.path.isfile(pth_path):
        data = torch.load(pth_path, map_location="cpu")
        if isinstance(data, dict) and "entries" in data:
            data["entries"].append(entry)
        elif isinstance(data, list):
            data.append(entry)
            data = {"entries": data}
        else:
            # unknown format, start new container
            data = {"entries": [entry]}
    else:
        data = {"entries": [entry]}

    torch.save(data, pth_path)
    return len(data["entries"])


class HCNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = torch.device('cuda:' + str(args.gpuid[0]))
        self.args = args
        intput_dim = 3
        self.sat_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', circular=False, in_channels = intput_dim)
        # self.grd_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', circular=False, in_channels = intput_dim) if args.p_siamese else None
        bev_lidar_channels = int((args.z_range[1] - args.z_range[0]) / args.BEV_grid_resolution) + 1
        self.lidar_backbone = LidarBEVBackbone(bev_lidar_channels)
        self.corr_dim = 320 if args.CNN16 else 112
        # Project LiDAR BEV features to corr_dim
        self.lidar_proj = nn.Conv2d(96, self.corr_dim, kernel_size=1, bias=False)
        
        in_dim = 164 if args.flow else 166
        sz = 16 if args.CNN16 else 32
        self.update_block_4 = GMA(self.args, sz, in_dim)
    
    def get_flow_now_k(self, four_point, k = 4):
        N,_,h,w = self.sz
        h, w = h//k, w//k
        four_point = four_point / k # four_point is at original size， coordinate is at feature map size
        four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([w-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, h-1])
        four_point_org[:, 1, 1] = torch.Tensor([w -1, h-1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(N, 1, 1, 1)
        four_point_new = torch.autograd.Variable(four_point_org) + four_point
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
        # H = tgm.get_perspective_transform(four_point_org, four_point_new)
        H = get_perspective_transform(four_point_org, four_point_new)
        gridy, gridx = torch.meshgrid(torch.linspace(0, w-1, steps=w), torch.linspace(0,h-1, steps=h),indexing='ij')
        points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, w * h))),
                           dim=0).unsqueeze(0).repeat(N, 1, 1).to(four_point.device)
        points_new = H.bmm(points) # (N,3,3) (N,3,w*h)
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(N, w, h).unsqueeze(1),
                          points_new[:, 1, :].reshape(N, w, h).unsqueeze(1)), dim=1)
        return flow, H

    def initialize_flow_k(self, img, k= 4):
        N, C, H, W = img.shape
        coords0 = coords_grid(N,H//k, W//k).to(img.device) # [batch,2, H, W]
        coords1 = coords_grid(N, H//k, W//k).to(img.device)

        return coords0, coords1 # [x,y]

    def forward(self, bev_lidar, sat_img, sat_gps=None, iters_lev0 = 6, test_mode=False):
        # 0. Normalize input data
        # Satellite RGB to [-1, 1]; leave BEV as-is (already metric / normalized)
        sat_img = 2 * (sat_img) - 1.0
        sat_img = sat_img.contiguous()
        bev_lidar = bev_lidar.contiguous()
        self.sz = sat_img.shape # [N, 3, 128, 128]
        
        # 1. Using Backbone to Obtain Feature Maps
        # LiDAR BEV -> feature map
        lidar_feat = self.lidar_backbone(bev_lidar)  # (B, 96, H_bev', W_bev')

        # Satellite -> EfficientNet multiscale features
        _, multiscale_sat = self.sat_efficientnet.extract_features_multiscale(sat_img)
        if self.args.CNN16:
            sat_feat = multiscale_sat[15]  # (B, 320, H_sat', W_sat')
        else:
            sat_feat = multiscale_sat[10]  # (B, 112, H_sat', W_sat')
        
        # 1.2 Align spatial size
        # Choose satellite feature resolution as reference
        B, C_s, Hs, Ws = sat_feat.shape
        _, C_l, Hl, Wl = lidar_feat.shape

        # Project to common channel dimension
        lidar_feat = self.lidar_proj(lidar_feat)  # (B, corr_dim, Hl, Wl)
        # sat_feat = self.sat_proj(sat_feat)        # (B, corr_dim, Hs, Ws)

        # Resize LiDAR feat to satellite feat spatial size
        lidar_feat = torch.nn.functional.interpolate(
            lidar_feat, size=(Hs, Ws), mode='bilinear', align_corners=False
        )
        
        
        fmap1 = lidar_feat.float()  # "ground" branch
        fmap2 = sat_feat.float()    # "satellite" branch
        sz = fmap1.shape  # (B, corr_dim, Hc, Wc)

        # 2. Calculate Correlation Matrix
        
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=2, radius=4)
        # downscale factor between input image (sat_img) and feature map
        k_factor = self.sz[-1] // sz[-1]  # assumes square & equal scale in H/W
        coords0, coords1 = self.initialize_flow_k(sat_img, k=k_factor)

        four_point_disp = torch.zeros((sz[0], 2, 2, 2), device=fmap1.device)
        
        # ################################################################################
        # pth_path = "metrics/feature_visualization/features2.pth"

        # # suppose these come from your model
        # # img_feature: (C,H,W) or (C,) tensor
        # # lidar_feature: (C,H,W) or (C,) tensor
        # n = append_features_pth(
        #     pth_path,
        #     img_feature=sat_feat,
        #     lidar_feature=lidar_feat,
        #     correlation = corr_fn.corr_pyramid )
        # ################################################################################


        # 3. Recurrent Homography Estimation
        
        flow_predictions =[] ## for train 
        for itr in range(iters_lev0):
            corr = corr_fn(coords1) # batch,channel,H,W  correlation
            flow = coords1 - coords0 if self.args.flow else torch.cat((coords1,coords0),dim=1) # [batch,2, H, W] mean x, y
            with autocast(enabled=self.args.mixed_precision):
                delta_four_point = self.update_block_4(corr, flow) # input shape: [b,2+channel,H,W] , output shape [b,2,2,2]

            four_point_disp =  four_point_disp + delta_four_point

            coords1, _ = self.get_flow_now_k(four_point_disp, k=self.sz[-1]//sz[-1])
            flow_predictions.append(four_point_disp) ## for train 
        
        # 4. Output
        coords1,H = self.get_flow_now_k(four_point_disp, k=1) 
        
        self.corr_fn = corr_fn
        if test_mode:
            return four_point_disp #, offset
        else:
            return flow_predictions, corr_fn.corr_map #, offset