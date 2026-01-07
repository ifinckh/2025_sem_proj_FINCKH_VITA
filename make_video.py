
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

def render_pred_gt_frame(
    sat_img_chw,          
    points_xyz,           
    trans_pred_xy,        
    yaw_pred_deg,         
    trans_gt_xy,          
    rot_gt_2x2,           
    resolution_m_per_px,  
    title=None,
    max_points=30000,
    figsize=(10, 5),
    dpi=140,
):
    """Render side-by-side predicted vs GT LiDAR-satellite overlay."""
    # --- to numpy ---
    img = sat_img_chw.float().permute(1, 2, 0).numpy()  # (H,W,3)
    pts = points_xyz.float().numpy()[:, :2]             # (N,2)
    H, W, _ = img.shape
    res = float(resolution_m_per_px)

    if max_points is not None and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    # --- predicted pose transform ---
    theta = np.deg2rad(float(yaw_pred_deg))
    rot_pred = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
    trans_pred = np.asarray(trans_pred_xy, dtype=np.float32).reshape(2,)

    pts_pred = (pts - trans_pred) @ np.linalg.inv(rot_pred.T) / res
    x_pred = W // 2 + pts_pred[:, 0]
    y_pred = H // 2 - pts_pred[:, 1]
    m_pred = (x_pred >= 0) & (x_pred < W) & (y_pred >= 0) & (y_pred < H)

    # --- GT pose transform ---
    rot_gt = np.asarray(rot_gt_2x2, dtype=np.float32).reshape(2, 2)
    trans_gt = np.asarray(trans_gt_xy, dtype=np.float32).reshape(2,)

    pts_gt = (pts - trans_gt) @ np.linalg.inv(rot_gt.T) / res
    x_gt = W // 2 + pts_gt[:, 0]
    y_gt = H // 2 - pts_gt[:, 1]
    m_gt = (x_gt >= 0) & (x_gt < W) & (y_gt >= 0) & (y_gt < H)

    # --- draw ---
    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.01, hspace=0)
    

    ax[0].imshow(img, origin="upper")
    ax[0].scatter(x_pred[m_pred], y_pred[m_pred], s=0.2, c="r", alpha=0.2)
    ax[0].set_title("Predicted")
    ax[0].set_aspect("equal")
    
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].axis('off')
    
    ax[1].imshow(img, origin="upper")
    ax[1].scatter(x_gt[m_gt], y_gt[m_gt], s=0.2, c="r", alpha=0.2)
    ax[1].set_title("GT")
    ax[1].set_aspect("equal")
    ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].axis('off')
    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    # --- canvas -> RGB uint8 (FIXED FOR MATPLOTLIB 3.8+) ---
    fig.canvas.draw()
    # Get RGBA buffer and convert to numpy array
    frame = np.asarray(fig.canvas.buffer_rgba()) 
    # Drop the Alpha channel (RGBA -> RGB) to match video writer expectation
    frame = frame[:, :, :3]
    
    plt.close(fig)
    return frame



def create_drive_video(model, val_loader, args):
    """
    Stream predicted vs GT overlays to video/GIF for a single continuous drive.
    
    Args:
        model: Your localization model
        val_loader: DataLoader with shuffle=False for sequential frames
        args: Must have attributes:
            - iters_lev0: number of refinement iterations
            - video_path: output path (e.g., "drive_overlay.mp4" or "drive_overlay.gif")
            - video_fps: frames per second (default 10)
            - video_max_points: max LiDAR points to render per frame (default 30000)
    """
    torch.cuda.empty_cache()
    model.eval()
    device = next(model.parameters()).device

    # --- Video/GIF configuration ---
    video_path = getattr(args, "video_path", "drive_overlay.mp4")
    video_fps = getattr(args, "video_fps", 10)
    max_points = getattr(args, "video_max_points", 30000)
    
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    
    # Open video writer (supports .mp4, .gif, etc.)
    ext = os.path.splitext(video_path)[1].lower()

    if ext == ".mp4":
        writer = imageio.get_writer(
            video_path,
            format="FFMPEG",   # force ffmpeg backend
            mode="I",
            fps=video_fps,
            codec="libx264",
        )
    elif ext == ".gif":
        writer = imageio.get_writer(video_path, mode="I", fps=video_fps)
    else:
        raise ValueError(f"Unsupported video extension: {ext} (use .mp4 or .gif)")

    print(f"Creating drive overlay video: {video_path} @ {video_fps} fps")
    print(f"Processing {len(val_loader)} batches...")

    try:
        for i, batch in enumerate(val_loader):
            if batch is None:
                continue
            
            batch = to_device(batch, device)

            sat_img = batch["image"]           # (B,3,H,W)
            bev = batch["bev_lidar"]           # (B,C,Hb,Wb)
            rot_gt = batch["rotation"]         # (B,3,3)
            trans_gt = batch["translation"]    # (B,3)
            resolution = batch["resolution"]   # (B,)
            points = batch.get("points", None) # (B,N,3)

            # --- Inference ---
            with torch.no_grad():
                four_pred, _ = model(bev, sat_img, sat_gps=None, iters_lev0=args.iters_lev0)
            
            trans_pred, yaw_pred = predict_pose(four_pred, sat_img, resolution)

            # --- Compute errors for display ---
            gt_dx_m = trans_gt[:, 0]
            gt_dy_m = trans_gt[:, 1]
            gt_yaw_rad = torch.atan2(rot_gt[:, 1, 0], rot_gt[:, 0, 0])
            gt_yaw_deg = torch.rad2deg(gt_yaw_rad)

            pred_dx_m = trans_pred[:, 0]
            pred_dy_m = trans_pred[:, 1]

            batch_l2 = torch.sqrt((pred_dx_m - gt_dx_m)**2 + (pred_dy_m - gt_dy_m)**2)
            batch_yaw_err = yaw_pred - gt_yaw_deg

            # Get scene/frame names
            scenes = batch["scene_name"]
            frames = batch["name"]
            if isinstance(scenes, torch.Tensor): 
                scenes = scenes.cpu().tolist()
            if isinstance(frames, torch.Tensor): 
                frames = frames.cpu().tolist()

            # --- Render and append each frame in batch ---
            B = sat_img.shape[0]
            for j in range(B):
                if points is None:
                    print(f"Warning: No points in batch {i}, sample {j}. Skipping frame.")
                    continue

                # Move to CPU for rendering
                sat_cpu = sat_img[j].detach().cpu()
                pts_cpu = points[j].detach().cpu()
                res_j = float(resolution[j].detach().cpu().item())

                trans_pred_xy = trans_pred[j, :2].detach().cpu().numpy()
                yaw_pred_deg = float(yaw_pred[j].detach().cpu().item())

                trans_gt_xy = trans_gt[j, :2].detach().cpu().numpy()
                rot_gt_2x2 = rot_gt[j, :2, :2].detach().cpu().numpy()

                # Title with errors
                l2_err = float(batch_l2[j].detach().cpu().item())
                yaw_err = float(batch_yaw_err[j].detach().cpu().item())
                title = f"{scenes[j]} / {frames[j]} | L2={l2_err:.2f}m | ΔYaw={np.abs(yaw_err):.2f}°"

                frame_rgb = render_pred_gt_frame(
                    sat_img_chw=sat_cpu,
                    points_xyz=pts_cpu,
                    trans_pred_xy=trans_pred_xy,
                    yaw_pred_deg=yaw_pred_deg,
                    trans_gt_xy=trans_gt_xy,
                    rot_gt_2x2=rot_gt_2x2,
                    resolution_m_per_px=res_j,
                    title=title,
                    max_points=max_points,
                )
                writer.append_data(frame_rgb)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(val_loader)}] batches processed...")

    finally:
        writer.close()
    
    print(f"✓ Video saved to: {video_path}")

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
    p = argparse.ArgumentParser(description="Generate a video of LiDAR/Satellite overlaps")
    
    # --- Essential Model & Data Arguments ---
    p.add_argument('--restore_ckpt', type=str, default='checkpoints/best_checkpoint_zod.pth', help="Path to checkpoint")
    p.add_argument('--config', type=str, required=True, help="Path of config file (json)")
    p.add_argument('--gpuid', type=int, nargs='+', default=[0], help="GPU ID(s) to use")
    p.add_argument('--validation', type=str, default='val', help="Dataset split (e.g., 'val' or 'test')")
    p.add_argument('--batch_size', type=int, default=1, help="Batch size (1 is recommended for sequential video)")
    p.add_argument('--num_workers', type=int, default=4)
    
    # --- Video Generation Arguments ---
    p.add_argument('--iters_lev0', type=int, default=6, help="Refinement iterations for inference")
    p.add_argument('--video_path', type=str, default='metrics/val/drive_overlay.mp4', help="Output file (.mp4 or .gif)")
    p.add_argument('--video_fps', type=int, default=10, help="Frame rate for the video")
    p.add_argument('--video_max_points', type=int, default=30000, help="Max LiDAR points per frame")

    args = p.parse_args()
    
    # --- Config Setup ---
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, 'r') as f:
        config_data = json.load(f)
    
    config = EasyDict(config_data)
    
    # Update config with necessary args for the video function
    config['restore_ckpt'] = args.restore_ckpt
    config['validation'] = args.validation
    config['batch_size'] = args.batch_size
    
    # Specific args needed by create_drive_video / model
    config['iters_lev0'] = args.iters_lev0
    config['video_path'] = args.video_path
    config['video_fps'] = args.video_fps
    config['video_max_points'] = args.video_max_points

    print(f"Loaded Config: {args.config}")
    print(f"Video Target: {args.video_path} @ {args.video_fps} FPS")

    # --- Execution ---
    device = torch.device('cuda:' + str(args.gpuid[0]) if torch.cuda.is_available() else 'cpu')
    
    model = load_model(config, device)
    
    # Note: Ensure your build_val_loader uses shuffle=False when making a video!
    val_loader = build_val_loader(config)
    
    # Call the video generation function instead of evaluation
    create_drive_video(model, val_loader, config)

if __name__ == "__main__":
    main()
