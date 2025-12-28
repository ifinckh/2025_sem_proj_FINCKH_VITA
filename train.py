from __future__ import print_function, division
import sys
# sys.path.append('models')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import json
from easydict import EasyDict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from models.network import HCNet
import evaluate
# import dataset as datasets
import dataset.ZOD as datasets
from models.utils.utils import *
from models.utils.loss_factory import *
# from myevaluate import evaluate_HCNet
from myevaluate_zod import evaluate_hcnet_zod

# from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
        
from torch.utils.data._utils.collate import default_collate

def zod_collate_fn(batch):
    # batch is a list of dataset items
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

import csv
import os

def log_metrics_csv(filepath, epoch, iteration, loss, loss_h, loss_c, trans_l2_m, yaw_err_deg):
    """
    Append training metrics to a CSV file.

    Args:
        filepath (str): Path to the CSV file.
        epoch (int): Current epoch.
        iteration (int): Iteration number within the epoch.
        loss (float): Training loss.
        loss_h (float): Hierarchical loss component.
        loss_c (float): Correspondence loss component.
        rte_trans_l2_m (float): RTE translation L2 metric.
        rte_yaw_err_deg (float): RTE yaw error in degrees.
    """

    # Check if file already exists
    file_exists = os.path.isfile(filepath)    
    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header if new file
        if not file_exists:
            writer.writerow(["epoch", "batch", "loss", "loss_h", "loss_c", "trans_l2_m", "yaw_err_deg"])

        # Append data row
        writer.writerow([epoch, iteration, loss, loss_h, loss_c, trans_l2_m, yaw_err_deg])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    if args.scheduler == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(85))
    elif args.scheduler == 'Reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, min_lr = 0.000005, patience=0,  eps=args.epsilon)
    elif args.scheduler == 'OneCycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    # elif args.scheduler == 'OneCycle': # temp
    #     scheduler = optim.lr_scheduler.OneCycleLR(
    #         optimizer, 
    #         max_lr=args.lr, 
    #         total_steps=args.num_steps + 10,  # Match total_steps strictly
    #         pct_start=0.1,                    # Spend a bit more time warming up (5% -> 10%)
    #         cycle_momentum=False,             # Keep as False for AdamW
    #         anneal_strategy='cos',            # CHANGE THIS: linear -> cos
    #         div_factor=25.0,                  # Default, but good to be explicit
    #         final_div_factor=1000.0           # Forces final LR to be very tiny (3.5e-7)
    #     )
    elif args.scheduler == 'MultiCycle':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.000005, max_lr=args.lr, # 9000  171000
            step_size_up=9000, step_size_down=171000, mode='triangular2',cycle_momentum = False)  #mode in ['triangular', 'triangular2', 'exp_range']
    return optimizer, scheduler
    

def train(args):
    
    if True:
        date_str = datetime.now().strftime("%Y_%m_%d")
        csv_folder = "metrics/train/"
        # Find available test number folder
        test_num = 0
        while True:
            file_path = os.path.join(csv_folder, f"{date_str}_{test_num}.csv")
            if not os.path.exists(file_path):
                break
            test_num += 1
        train_metrics_file_path = file_path
        csv_folder = "metrics/val/"
        # Find available test number folder
        test_num = 0
        while True:
            file_path = os.path.join(csv_folder, f"{date_str}_{test_num}.csv")
            if not os.path.exists(file_path):
                break
            test_num += 1
        val_metrics_file_path = file_path
        csv_folder = "metrics/test/"
        # Find available test number folder
        test_num = 0
        while True:
            file_path = os.path.join(csv_folder, f"{date_str}_{test_num}.csv")
            if not os.path.exists(file_path):
                break
            test_num += 1
        test_metrics_file_path = file_path
        # Find available model save path
        pth_folder = "checkpoints/"
        test_num = 0
        while True:
            file_path = os.path.join(pth_folder, f"{date_str}_best_checkpoint_zod_{test_num}.pth")
            if not os.path.exists(file_path):
                break
            test_num += 1
        checkpoint_path = file_path

    model = nn.DataParallel(HCNet(args), device_ids=args.gpuid)
    print("Parameter Count: %d" % count_parameters(model))

    train_dataset, val_dataset = datasets.fetch_dataloader(args)
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    # nw = 0  # debug
    print('Using {} dataloader workers every process'.format(nw)) # https://blog.csdn.net/ResumeProject/article/details/125449639
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=True, num_workers=nw, collate_fn=zod_collate_fn,)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,pin_memory=True, num_workers=nw, collate_fn=zod_collate_fn,)

    optimizer, scheduler = fetch_optimizer(args, model)
    best_dis = args.best_dis

    if args.restore_ckpt is not None:
        PATH = args.restore_ckpt   # 'checkpoints/best_checkpoint.pth'
        if os.path.isfile(PATH):
            checkpoint = torch.load(PATH)
            best_dis = checkpoint['best_dis']
            args.start_step = checkpoint['steps']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_schedule'])
            print("Have load state_dict from: {}".format(args.restore_ckpt))
            print('Load checkpoint at steps {}.'.format(checkpoint['steps']))      
    elif args.model is not None:
        model.load_state_dict(torch.load(args.model), strict=True)
        print("Have load state_dict from: {}".format(args.model))
    print('Best distance so far {}.'.format(best_dis))

    model.cuda()
    model.train()

    total_steps = args.start_step
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger_train(args, scheduler, optimizer, len(train_loader))
    logger.total_steps = total_steps

    epoch = args.start_step//len(train_loader)
    num_epochs = args.num_steps//len(train_loader)

    infoLoss = InfoNCELoss(temperature=args.temperature, sample = True)
    should_keep_training = True
    w1, w2, w3 = args.loss_w
    while should_keep_training:
        model.train()
        for i_batch, data_blob in enumerate(train_loader):
            if data_blob is None:
                continue
            optimizer.zero_grad()

            # ZOD batch: collate will give tensors for each dict field
            bev_lidar   = data_blob['bev_lidar'].cuda()      # (B, C_bev, H_bev, W_bev)
            sat_img     = data_blob['image'].cuda()          # (B, 3, H, W) [0,1]
            rot_gt      = data_blob['rotation'].cuda()       # (B, 3, 3) SE(2) in top-left
            trans_gt    = data_blob['translation'].cuda()    # (B, 3)
            resolution  = data_blob['resolution'].cuda()     # (B,) m/pixel

            # Forward pass: HCNet(bev, sat)
            four_pred, corr_fn = model(
                bev_lidar, sat_img,
                sat_gps=None,
                iters_lev0=args.iters_lev0
            )
            
            # ---- Original HC Net loss -----
            # Forward pass   
            # four_pred, corr_fn  = model(image1, image2, sat_gps=sat_gps.float(), iters_lev0=args.iters_lev0)       
            # loss, metrics = vigor_gps_loss(four_pred, grd_gps = grd_gps, sat_gps=sat_gps, args=args, sat_delta = sat_delta, ori_angle = ori_angle, w3 = w3,\
            #     orien = args.dataset == 'vigor' and args.orien, transformed_center = transformed_center, sz = [image1.shape[2],image1.shape[3]] ,gamma=args.gamma)
            # loss2 = corr_loss(grd_gps, sat_gps, corr_fn, infoLoss,  args=args, sat_delta = sat_delta, transformed_center = transformed_center, sz = [image1.shape[2],image1.shape[3]])
            
            # loss = loss*w1 + loss2*w2     
            
            # ---- Adapted HC Net loss -----

            loss1, metrics, y = zod_homography_loss(four_pred, sat_img, rot_gt, trans_gt, resolution, w3=w3, orien=True, gamma=args.gamma)
            loss2 = corr_loss_zod(corr_fn, infoLoss, y, sz = [sat_img.shape[2],sat_img.shape[3]])

            loss = loss1*w1 + loss2*w2


            # Backward and Optimze
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer) # https://blog.csdn.net/weixin_51723388/article/details/126260788
            
            ### Learning Rate update
            scale = scaler.get_scale()
            scaler.update()
            skip_lr_sched = (scale > scaler.get_scale())
            if not skip_lr_sched:
                if args.scheduler == 'OneCycle' or args.scheduler == 'MultiCycle'  :
                    scheduler.step()
                # scheduler.step(loss)
            else:
                print("skip the scheduler step")

            # metrics.update({'loss': loss.cpu().item()})
            # logger.push(metrics)
            
            ############################################################################################
            # print('\033[1;94m'+'Epoch: [{}/{}], Loss: {}, RTE: {}, Pred dx,dy: {}, GT {}. \033[0m'
            #       .format(epoch+1, num_epochs, loss.cpu().item(), loss_RTE.cpu().item(), 
            #               (supervision_zod_RTE["pred_dx_m"],supervision_zod_RTE["pred_dy_m"]),trans_gt[:,0:2][0].cpu().numpy()))
            
            if i_batch % 10 == 0:
                print(
                    f'Epoch: [{epoch+1}/{num_epochs}], Batch: {i_batch}/{len(train_loader)}, '
                    f'Loss: {loss.cpu().item():.3f}, '
                    f'Trans_L2_m: {metrics["trans_l2_m"]:.3f}, '
                    f'Yaw_err_deg: {metrics["yaw_err_deg"]:.3f}'
                )
            log_metrics_csv(train_metrics_file_path,epoch, i_batch, loss.cpu().item(), loss1.cpu().item(), loss2.cpu().item(), metrics["trans_l2_m"], metrics["yaw_err_deg"])
            ############################################################################################

            # if total_steps %  args.IMG_FREQ == args.IMG_FREQ-1:
            #     H = get_homograpy(four_pred[-1],  image1.shape)
            #     H = H.detach().cpu().numpy()
            #     image1 = image1[0].permute(1, 2,0).detach().cpu().numpy()
            #     image0 = image2[0].permute(1, 2,0).detach().cpu().numpy()
            #     plt.figure(figsize=(10,10))
            #     result = show_overlap(image1, image0, H[0])
            #     cv2.imwrite('./watch/' + "result_" + args.name + '.png',result[:,:,::-1])
            #     print("save at: {}".format('./watch/' + "result_" + args.name + '.png'))
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        
        if epoch % 2 == 1 and not epoch >= num_epochs:     
            if args.scheduler == 'Cosine':
                scheduler.step()
            epoch+=1
            continue


        results = evaluate.validate_process(model.module, total_steps, val_loader, args)        
        logger.write_dict(results)
        val_mdis = results['val_mace']

        if args.scheduler == 'Reduce':
             scheduler.step(val_mdis)
        elif args.scheduler == 'Cosine':
            scheduler.step()

        # print('\033[1;94m'+'Epoch: [{}/{}], Loss: {}. \033[0m'
        #           .format(epoch+1, num_epochs, val_mdis.item())        
        if val_mdis < best_dis:
            best_dis = val_mdis
            checkpoint = {
                'best_dis': best_dis,
                'steps': total_steps,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedule':scheduler.state_dict()
            }
            best_model_dict = model.state_dict()
            # PATH = 'checkpoints/best_checkpoint_{}.pth'.format(args.name)
            PATH = checkpoint_path
            torch.save(checkpoint, PATH)
            print('\033[1;94m'+"Save the best of {}, at {}\033[0m".format(val_mdis, PATH))
        else:
            print('\033[1;91m'+"Val has no improvement vs {}!\033[0m".format(best_dis)) # https://blog.csdn.net/qq_63167347/article/details/125824913 
        with open(logger.file_name, 'a') as file:
            file.write('Epoch: [{}/{}], mdis: {}, the best: {}, at {}.'
                .format(epoch+1, num_epochs, val_mdis.item(), best_dis, checkpoint['steps']) + '\n')                       
        epoch+=1 
        
        
    logger.close()
    print("The minist distance is {}m!".format(best_dis))

    model.load_state_dict(best_model_dict)
    PATH = 'checkpoints/%.3f_' % best_dis+'%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    model  = model.module
    model.eval()
    
    val_dataset = datasets.fetch_dataloader(args, split="val")
    evaluate_hcnet_zod(model, val_dataset, args=args)    

    return PATH


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="path of config file")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--name', default='HC-Net', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--best_dis', type=float, default=1e8)

    args = parser.parse_args()
    
    config = json.load(open(args.config,'r'))
    config = EasyDict(config)
    config['config'] = args.config
    config['best_dis'] = args.best_dis
    config['validation'] = args.validation
    config['name'] = args.name
    config['restore_ckpt'] = args.restore_ckpt
    config['start_step'] = args.start_step
    if args.batch_size: 
        config['batch_size'] = args.batch_size

    print(config)

    setup_seed(2023)
    print_colored(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    if config.dataset=='vigor':
        print("Dataset is VIGOR!")  

    train(config)