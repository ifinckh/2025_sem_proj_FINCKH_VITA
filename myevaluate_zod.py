import os, time, argparse, warnings
warnings.filterwarnings("ignore")

from models.utils.loss_factory import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.network import HCNet
# from models.utils.utils import get_homograpy
import dataset as datasets  # your fetch_dataloader
from collections import defaultdict

def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

@torch.no_grad()
def evaluate_hcnet_zod(model, val_loader, args):
    model.eval()
    device = next(model.parameters()).device

    meters_l2, pix_l2 = [], []
    yaw_err_deg = []
    prob_at_gt = []

    t_list = []
    _, _, w3 = args.loss_w

    for i, batch in enumerate(val_loader):
        if batch is None:
            continue
        batch = to_device(batch, device)

        # Inputs (names aligned with your dataset/collate)
        sat_img = batch.get("image")        # (B,3,H,W)
        bev     = batch["BEV_lidar"]        # (B,C,Hb,Wb)
        rot_gt  = batch["rotation"]         # (B,3,3)
        trans_gt= batch["translation"]      # (B,3)
        resolution     = batch["resolution"]       # (B,)
        # points   = batch["points"]          # (B,N,3)
        # intensity= batch["intensity"]       # (B,N)
        # heading  = batch["heading"]         # (B,)

        B, _, H, W = sat_img.shape
        sz = [B, 1, H, W]

        # Forward
        t0 = time.time()
        four_pred, corr_fn = model(bev, sat_img, sat_gps=None, iters_lev0=args.iters_lev0)
        t1 = time.time()
        t_list.append(t1 - t0)
        
        
        loss, metric, _ = zod_homography_loss(four_pred, sat_img, rot_gt, trans_gt, resolution, w3=w3, orien=True, gamma=args.gamma)

        
        pred_dx_m = metric["pred_dx_m"]
        pred_dy_m = metric["pred_dy_m"]
        gt_dx_m = metric["gt_dx_m"]
        gt_dy_m = metric["gt_dy_m"]
        pred_yaw_deg = metric["pred_yaw_deg"]
        gt_yaw_deg = metric["gt_yaw_deg"]
        
        # Compute metrics
        dx_m = pred_dx_m - gt_dx_m
        dy_m = pred_dy_m - gt_dy_m
        dyaw_deg = pred_yaw_deg - gt_yaw_deg
        # Mean location error in meters        
        l2_m = torch.sqrt(dx_m**2 + dy_m**2)
        meters_l2.extend(l2_m.cpu().numpy().tolist())
        # orientation estimation error in degrees
        yaw_err_deg.extend(torch.abs(dyaw_deg).cpu().numpy().tolist())
        
        
        
        # Optional probability at GT (if corr_fn present)
        if hasattr(model, "corr_fn") or corr_fn is not None:
            cf = corr_fn if corr_fn is not None else model.corr_fn
            corr_map = cf.corr_pyramid[0]  # (B, Hf*Wf, Hf, Wf) or similar
            h, w = corr_map.shape[-2:]
            corr_map = corr_map.view((-1, h, w, h, w))
            temp = h // 2
            sim_matrix = corr_map[:, temp, temp, :, :]       # (B, h, w)

            # Softmax over all positions with temperature
            temperature = 400.0
            sm = F.softmax(sim_matrix.view(B, -1) / temperature, dim=1)
            sm = sm.view(B, 1, h, w)

            # Normalize GT pixel to [-1,1] grid at corr resolution
            gt_grid = (gt_dx_m / W) * (w - 1)  # scale to [0,w-1]
            gt_norm = 2 * gt_grid / (w - 1) - 1
            grid = gt_norm.view(B,1,1,2)    # (B,1,1,2) as (x,y)

            prob = F.grid_sample(sm, grid, align_corners=True)  # (B,1,1,1)
            prob_at_gt.extend(prob.view(B).tolist())

        if (i+1) % args.log_every == 0:
            print(f"[{i+1}] m_l2={np.mean(meters_l2):.3f}, "
                  f"yaw_err={np.mean(yaw_err_deg):.2f}deg, "
                  f"prob@gt={np.mean(prob_at_gt):.4f}" if prob_at_gt else
                  f"[{i+1}] m_l2={np.mean(meters_l2):.3f}, "
                  f"yaw_err={np.mean(yaw_err_deg):.2f}deg")

    print("==== ZOD evaluation (HC‑Net homography) ====")
    print(f"Avg meter L2: {np.mean(meters_l2):.3f}")
    print(f"Avg yaw err: {np.mean(yaw_err_deg):.2f} deg")
    if len(prob_at_gt) > 0:
        print(f"Avg prob@GT: {np.mean(prob_at_gt):.4f}")
    print(f"Avg time per batch: {np.mean(t_list)*1000:.2f} ms")

def build_val_loader(args):
    # Reuse your dataset fetcher; expect the same zod_collate_fn as training
    # If your fetch_dataloader returns (train,val), adapt accordingly.
    split = "val" if args.validation in [None, "val", "validation"] else args.validation
    val_dataset = datasets.fetch_dataloader(args, split=split)
    # If the function returns a DataLoader already, just return it.
    if hasattr(val_dataset, "__iter__") and hasattr(val_dataset, "dataset"):
        return val_dataset
    # Otherwise wrap into a DataLoader
    from torch.utils.data import DataLoader
    try:
        from train import zod_collate_fn  # reuse your collate
    except Exception:
        zod_collate_fn = None
    return DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True,
                      collate_fn=zod_collate_fn)

def load_model(args, device):
    model = HCNet(args).to(device)
    state = None
    if args.model and os.path.isfile(args.model):
        state = torch.load(args.model, map_location=device)
        print(f"Loaded state_dict from: {args.model}")
    elif args.restore_ckpt and os.path.isfile(args.restore_ckpt):
        ckpt = torch.load(args.restore_ckpt, map_location=device)
        state = ckpt.get("model", ckpt)
        print(f"Loaded checkpoint: {args.restore_ckpt}")
    if state is not None:
        # strip 'module.' if needed
        new_state = {}
        for k,v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            new_state[nk] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        if missing:   print("Missing keys:", missing)
        if unexpected:print("Unexpected keys:", unexpected)
    model.eval()
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--restore_ckpt', type=str, default='checkpoints/best_checkpoint_zod.pth')
    p.add_argument('--model', type=str, default=None)
    p.add_argument('--gpuid', type=int, nargs='+', default=[0])
    p.add_argument('--iters_lev0', type=int, default=6)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--validation', type=str, default='val')
    p.add_argument('--log_every', type=int, default=50)
    args = p.parse_args()

    device = torch.device('cuda:'+str(args.gpuid[0]) if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device)
    val_loader = build_val_loader(args)
    evaluate_hcnet_zod(model, val_loader, args)

if __name__ == "__main__":
    main()
