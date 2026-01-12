import os, time, argparse, warnings
warnings.filterwarnings("ignore")

from models.utils.loss_factory import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import os
from datetime import datetime  
import json
from easydict import EasyDict


from models.network import HCNet
# from models.utils.utils import get_homograpy
# import dataset as datasets  # your fetch_dataloader
import dataset.ZOD as datasets

from collections import defaultdict

def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def angular_error(pred_deg, gt_deg):
    """
    Compute shortest angular distance in degrees.
    Handles wraparound: |359 - 1| = 2, not 358.
    """
    diff = torch.abs(pred_deg - gt_deg) % 360
    # If difference > 180, take the shorter path (360 - diff)
    diff = torch.min(diff, 360 - diff)
    return diff

def log_metrics_csv(filepath, scene_name, frame_name, iteration, meters_l2, yaw_err_deg, time_ms):
    """
    Append training metrics to a CSV file.

    Args:
        filepath (str): Path to the CSV file.
        scene_name (str): Name of the scene.
        batch_name (str): Name of the batch.
        iteration (int): Iteration number within the epoch.
        meters_l2 (float): Mean location error in meters.
        yaw_err_deg (float): Orientation estimation error in degrees.
        time_ms (float): Time per batch in milliseconds.
    """

    # Check if file already exists
    file_exists = os.path.isfile(filepath)    
    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header if new file
        if not file_exists:
            writer.writerow(["scene_name", "frame_name", "iteration", "meters_l2", "yaw_err_deg", "time_ms"])
            
        # make "scene_name", "frame_name" tensors into cpu lists
        if isinstance(scene_name, torch.Tensor):
            scene_name = scene_name.cpu().tolist()
        if isinstance(frame_name, torch.Tensor):
            frame_name = frame_name.cpu().tolist()

        # Append data row
        writer.writerow([scene_name, frame_name, iteration, f"{meters_l2:.4f}", f"{yaw_err_deg:.4f}", f"{time_ms:.2f}"])#meters_l2, yaw_err_deg, time_ms])

@torch.no_grad()
def evaluate_hcnet_zod(model, val_loader, args):
    torch.cuda.empty_cache()

    model.eval()
    device = next(model.parameters()).device
    
    # --- CSV Setup ---
    os.makedirs("metrics/val/", exist_ok=True)
    date_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    val_metrics_file_path = f"metrics/val/{date_str}_hcnet_zod_eval.csv"
    print(f"Logging metrics to: {val_metrics_file_path}")

    # --- Accumulators ---
    all_l2_errors = []
    all_yaw_errors = []
    all_times = []
    
    print(f"Starting evaluation on {len(val_loader)} batches...")


    for i, batch in enumerate(val_loader):
        if batch is None:
            continue
        batch = to_device(batch, device)

        # Inputs (names aligned with your dataset/collate)
        sat_img = batch["image"]                # (B,3,H,W)
        bev     = batch["bev_lidar"]            # (B,C,Hb,Wb)
        rot_gt  = batch["rotation"]             # (B,3,3)
        trans_gt = batch["translation"]         # (B,3)
        resolution = batch["resolution"]        # (B,)

        # B = sat_img.shape[0]
        # H = sat_img.shape[-2]
        # W = sat_img.shape[-1]
        # sz = [B, 1, H, W]

        # --- Inference ---
        torch.cuda.synchronize()
        t0 = time.time()
        four_pred, _ = model(bev, sat_img, sat_gps=None, iters_lev0=args.iters_lev0)
        torch.cuda.synchronize()
        t1 = time.time()
        batch_time_ms = (t1 - t0) * 1000.0 / sat_img.shape[0] # Average time per sample in batch        
        
        trans_pred, yaw_pred = predict_pose(four_pred, sat_img, resolution)

        
        # --- Ground Truth Prep ---
        gt_dx_m = trans_gt[:, 0]
        gt_dy_m = trans_gt[:, 1]
        
        # Yaw from rotation matrix: atan2(R[1,0], R[0,0])
        # Make sure your coordinate system matches! (ZOD might use different convention)
        gt_yaw_rad = torch.atan2(rot_gt[:, 1, 0], rot_gt[:, 0, 0])
        gt_yaw_deg = torch.rad2deg(gt_yaw_rad)

        # --- Compute Errors ---
        pred_dx_m = trans_pred[:, 0]
        pred_dy_m = trans_pred[:, 1]

        # 1. Position Error (L2)
        batch_l2 = torch.sqrt((pred_dx_m - gt_dx_m)**2 + (pred_dy_m - gt_dy_m)**2)
        
        # 2. Orientation Error (Angular Distance)
        batch_yaw_err = angular_error(yaw_pred, gt_yaw_deg)

        # --- Logging ---
        # Convert to list to log individually
        l2_list = batch_l2.cpu().tolist()
        yaw_list = batch_yaw_err.cpu().tolist()
        
        all_l2_errors.extend(l2_list)
        all_yaw_errors.extend(yaw_list)
        all_times.extend([batch_time_ms] * len(l2_list))

        # Log to CSV line by line
        scenes = batch["scene_name"]
        frames = batch["name"]
        # Handle if scenes/frames are strings or lists
        if isinstance(scenes, torch.Tensor): scenes = scenes.cpu().tolist()
        if isinstance(frames, torch.Tensor): frames = frames.cpu().tolist()
        
        for j in range(len(l2_list)):
            log_metrics_csv(val_metrics_file_path, scenes[j], frames[j], i, l2_list[j], yaw_list[j], batch_time_ms)

        if (i+1) % args.log_every == 0:
            print(f"[{i+1}/{len(val_loader)}] "+
                  f"Mean L2: {np.mean(all_l2_errors):.3f}m | "+
                  f"Mean Yaw: {np.mean(all_yaw_errors):.2f}°")

    # --- Final Report Calculation ---
    all_l2_errors = np.array(all_l2_errors)
    all_yaw_errors = np.array(all_yaw_errors)
    all_times = np.array(all_times)

    # 1. Basic Stats
    mean_l2 = np.mean(all_l2_errors)
    median_l2 = np.median(all_l2_errors)
    std_l2 = np.std(all_l2_errors)
    
    mean_yaw = np.mean(all_yaw_errors)
    median_yaw = np.median(all_yaw_errors)

    # 2. Success Rates / Recalls
    recall_1m = np.mean(all_l2_errors < 1.0) * 100
    recall_2m = np.mean(all_l2_errors < 2.0) * 100
    recall_5m = np.mean(all_l2_errors < 5.0) * 100
    recall_10m = np.mean(all_l2_errors < 10.0) * 100
    
    recall_1deg = np.mean(all_yaw_errors < 1.0) * 100
    recall_2deg = np.mean(all_yaw_errors < 2.0) * 100
    recall_5deg = np.mean(all_yaw_errors < 5.0) * 100
    recall_10deg = np.mean(all_yaw_errors < 10.0) * 100

    print("\n" + "="*40)
    print(f" FINAL RESULTS (N={len(all_l2_errors)})")
    print("="*40)
    print(f"Position Error (m):")
    print(f"  Mean:   {mean_l2:.4f} m")
    print(f"  Median: {median_l2:.4f} m")
    print(f"  Std:    {std_l2:.4f} m")
    print("-" * 20)
    print(f"Orientation Error (°):")
    print(f"  Mean:   {mean_yaw:.4f}°")
    print(f"  Median: {median_yaw:.4f}°")
    print("-" * 20)
    print(f"Success Rates (Recalls):")
    print(f"  @1m:    {recall_1m:.2f}%")
    print(f"  @2m:    {recall_2m:.2f}%")
    print(f"  @5m:    {recall_5m:.2f}%")
    print(f"  @10m:   {recall_10m:.2f}%")
    print("-" * 20)
    print(f"  @1°:    {recall_1deg:.2f}%")
    print(f"  @2°:    {recall_2deg:.2f}%")
    print(f"  @5°:    {recall_5deg:.2f}%")
    print(f"  @10°:   {recall_10deg:.2f}%")
    print("="*40)
    
    # Save final summary
    with open(val_metrics_file_path.replace(".csv", "_summary.txt"), "w") as f:
        f.write(f"Mean L2: {mean_l2}\nMedian L2: {median_l2}\n")
        f.write(f"Mean Yaw: {mean_yaw}\nMedian Yaw: {median_yaw}\n")
        f.write(f"Recall@1m: {recall_1m}\nRecall@2m: {recall_2m}\n")
        f.write(f"Recall@5m: {recall_5m}\nRecall@10m: {recall_10m}\n")
        f.write(f"Recall@1deg: {recall_1deg}\nRecall@2deg: {recall_2deg}\n")
        f.write(f"Recall@5deg: {recall_5deg}\nRecall@10deg: {recall_10deg}\n")

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
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 12])  # number of workers

    return DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                      num_workers=nw, pin_memory=True,
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
    p.add_argument('--config', type=str, help="path of config file")

    p.add_argument('--model', type=str, default=None)
    p.add_argument('--gpuid', type=int, nargs='+', default=[0])
    p.add_argument('--iters_lev0', type=int, default=6)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--validation', type=str, default='val')
    p.add_argument('--log_every', type=int, default=50)
    args = p.parse_args()
    
    config = json.load(open(args.config,'r'))
    config = EasyDict(config)
    config['config'] = args.config
    # config['best_dis'] = args.best_dis
    config['validation'] = args.validation
    # config['name'] = args.name
    config['restore_ckpt'] = args.restore_ckpt
    # config['start_step'] = args.start_step
    if args.batch_size: 
        config['batch_size'] = args.batch_size

    print(config)


    device = torch.device('cuda:'+str(args.gpuid[0]) if torch.cuda.is_available() else 'cpu')
    model = load_model(config, device)
    val_loader = build_val_loader(config)
    evaluate_hcnet_zod(model, val_loader, config)

if __name__ == "__main__":
    main()
