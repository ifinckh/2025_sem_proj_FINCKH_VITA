import sys

sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
from models.utils.utils import get_homograpy   # already used elsewhere
from models.utils.loss_factory import zod_homo_loss
import time
import cv2

@torch.no_grad()
def validate_process(model, total_steps, val_loader, args):
    """
    Simple validation loop for ZOD:
      - expects val_loader to yield dicts from your ZOD dataset
      - computes mean zod_homo_loss over the validation set
      - returns {'val_mace': tensor} for train.py compatibility
    """
    model.eval()
    losses = []
    times = []

    for i_batch, data_blob in enumerate(val_loader):
        t0 = time.time()

        bev_lidar  = data_blob['bev_lidar'].to(model.device)      # (B, C_bev, H_bev, W_bev)
        sat_img    = data_blob['image'].to(model.device)          # (B, 3, H, W), in [0,1]
        rot_gt     = data_blob['rotation'].to(model.device)       # (B, 3, 3)
        trans_gt   = data_blob['translation'].to(model.device)    # (B, 3)
        resolution = data_blob['resolution'].to(model.device)     # (B,)

        # Forward pass: HCNet returns list of four-point preds in test_mode
        four_pred = model(bev_lidar, sat_img,
                          sat_gps=None,
                          iters_lev0=args.iters_lev0,
                          test_mode=True)

        loss, _ = zod_homo_loss(
            four_pred, sat_img,
            rot_gt=rot_gt,
            trans_gt=trans_gt,
            resolution=resolution,
            gamma=args.gamma
        )
        losses.append(loss.item())

        t1 = time.time()
        times.append(t1 - t0)

    val_loss = float(np.mean(losses))
    print(f"Validation loss (homography): {val_loss:.4f}")
    if len(times) > 2:
        print("Avg batch time: {:.2f} ms, total: {:.3f} s".format(
            np.mean(times[1:-1]) * 1000.0, np.sum(times)
        ))

    model.train()
    # train.py expects a tensor here and reads results['val_mace']
    return {'val_mace': torch.tensor(val_loss, device=model.device)}

@torch.no_grad()
def test_process(model, total_steps, args):
    """
    Simple test loop for ZOD:
      - builds a validation loader internally
      - computes mean zod_homo_loss
    """
    model.eval()
    args.batch_size = 1

    import dataset as datasets
    val_loader = DataLoader(
        datasets.fetch_dataloader(args, split='validation'),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    losses = []

    for i_batch, data_blob in enumerate(val_loader):
        bev_lidar  = data_blob['bev_lidar'].to(model.device)
        sat_img    = data_blob['image'].to(model.device)
        rot_gt     = data_blob['rotation'].to(model.device)
        trans_gt   = data_blob['translation'].to(model.device)
        resolution = data_blob['resolution'].to(model.device)

        four_pred = model(bev_lidar, sat_img,
                          sat_gps=None,
                          iters_lev0=args.iters_lev0,
                          test_mode=True)

        loss, _ = zod_homo_loss(
            four_pred, sat_img,
            rot_gt=rot_gt,
            trans_gt=trans_gt,
            resolution=resolution,
            gamma=args.gamma
        )
        losses.append(loss.item())

        # optional: visualize the first example
        if i_batch == 0:
            if not os.path.exists('watch'):
                os.makedirs('watch')
            H = get_homograpy(four_pred, sat_img.shape)
            H = H.detach().cpu().numpy()
            img1 = bev_lidar[0].permute(1, 2, 0).detach().cpu().numpy()
            img2 = sat_img[0].permute(1, 2, 0).detach().cpu().numpy()
            result = show_overlap(img1, img2, H[0])
            cv2.imwrite('./watch/result_test.png', result[:, :, ::-1])
            print("Saved test visualization at ./watch/result_test.png")

    test_loss = float(np.mean(losses))
    print(f"Test loss (homography): {test_loss:.4f}")
    model.train()
    return {'chairs_mace': test_loss}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--model_name')

    # Ablations
    parser.add_argument('--replace', default=False, action='store_true',
                        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true',
                        help='Remove residual connection. Do not add local features with the aggregated features.')

    args = parser.parse_args()