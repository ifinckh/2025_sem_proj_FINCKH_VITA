import sys
sys.path.append('..')

import open3d as o3d
import h5py
import matplotlib.pyplot as plt
import cv2
import torch.utils.data
import torch
import numpy as np
from typing import Dict
import random
from pathlib import Path
import os
import sys
# from imgaug import augmenters as iaa
# from scipy.spatial.transform import Rotation
import time
import json
import utm
import pandas as pd
from tqdm import tqdm

from dataset.open_visloc_essential.utils.transformations import rotation_matrix_from_angles, get_yaw_pitch_roll, filter_above_ground, compute_heading

# Load ZOD DevKit
from zod import  ZodDrives # , ZodFrames, ZodSequences
import zod.constants as constants
from zod.constants import Lidar #, Camera, Anonymization, AnnotationProject
from zod.data_classes.ego_motion import OXTS_TIMESTAMP_OFFSET, interpolate_transforms
# from zod.data_classes import LidarData
from zod.utils.geometry import transform_points
# from zod.visualization.lidar_on_image import get_3d_transform_camera_lidar
from zod.constants import Camera, Lidar
# from zod.data_classes.metadata import SequenceMetadata

sys.path.append("../")

from typing import List
# plt.rcParams["figure.figsize"] = [20, 10]

pp = [os.path.dirname((os.path.abspath(__file__))),
      os.path.dirname((os.path.abspath('.'))),
      os.path.dirname((os.path.abspath('..')))]
for p in pp:
    if p not in sys.path:
        sys.path.append(p)

# to replace cv2 and o3d in get_item, which may cause issues when trying to use several workers. 
import torch
import torch.nn.functional as F
from dataset.dataset_utils.utils_zod import approximate_meridian_convergence
from torch.utils.data import DataLoader
import csv
cv2.setNumThreads(0)

class ZOD(torch.utils.data.Dataset):

    def __init__(self, seed=None, task='test', transform=None, **config):
        super(ZOD, self).__init__()
        self.config = config
        # 
        self.lidar_dataset_root= Path(config['lidar_dataset_root']) 
        self.aerial_img_dataset_root = Path(config['aerial_img_dataset_root'])
        # self.dataset_path = self.dataset_root
        self.dataset_version = config['version']
        # self.datast_split = self.dataset_root / (task + '.txt')
        self.zod_drives = ZodDrives(dataset_root=self.lidar_dataset_root, version=self.dataset_version)
        # print(task)
        if task == 'train':
            task_data_subset = list(self.zod_drives.get_split(constants.TRAIN))
            print("Loading train split :", len(task_data_subset), "drives")
        elif task == 'val':
            task_data_subset = list(self.zod_drives.get_split(constants.VAL))
            print("Loading val split :", len(task_data_subset), "drives")
        elif task == 'test':
            task_data_subset = list(self.zod_drives.get_split(constants.VAL))
            print("Loading test split :", len(task_data_subset), "drives")
        
        
        # 'train','val','test'
        self.subset = task
        
                
        #  half_size
        self.hs = config['image_size'] // 2
        self.final_img_shape = config['image_size']
        self.image_area_dg_m = config['image_area_dg_m']
        self.image_area_zod_m = config['image_area_zod_m']
        self.max_translation = config['max_translation'] # 15.0   # ± m
        self.max_rot_deg = config['max_rot_deg'] # 20.0 # ± °
        self.z_range = config['z_range'] # (-2.5, 3.0)
        self.BEV_grid_resolution = config['BEV_grid_resolution'] # 0.1,
        # 
        self.use_augmentation = config['use_augmentation']
        # self.aug_noise = config['augmentation_noise']
        # self.aug_rotation = config['augmentation_rotation']
        # image aug
        # self.aug = iaa.Sequential([
        #     iaa.Sometimes(0.5, iaa.Add((-30, 30))),
        #     iaa.Sometimes(0.5, iaa.LinearContrast((0.7, 1.3))),  # 
        #     iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 8))),  #
        #     iaa.Sometimes(0.5, iaa.ImpulseNoise(p=(0, 0.003))),  # 
        #     iaa.Sometimes(0.5, iaa.MotionBlur(3)),  # 
        #     iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.5))),  # 
        # ])
        
        self.metadata_list = []
        self.drive_infos = {}
        
        ####### For debugging, use only subset of training drives ########
        # task_data_subset = ["000009","000007"] 
        # task_data_subset = task_data_subset[3:5]
        
        bad_samples = pd.read_csv("dataset/dataset_utils/bad_ids.csv")

        
        # count total number of frames for tqdm
        total_frames = 0
        for i in range(len(task_data_subset)):
            total_frames += len(self.zod_drives[task_data_subset[i]].info.camera_frames['front_blur']) 
        
        # Create single progress bar
        # pbar = tqdm(total=total_frames, desc="Processing ZOD Drives", unit="Frame", colour="blue")
        
        # manually add bad samples found during evaluation
        good_batches_trans = [[17,1093],[5,1579],[28,524]]
        good_batches_rot = [[17,611],[17,723],[17,2244],[7,996]]
        bad_batches_trans = [[2,266],[2,354],[2,795],[7,1626]]
        bad_batches_rot = [[5,424],[5,1399],[5,1401],[7,338]]
        interest_samples = pd.DataFrame(data=good_batches_trans+good_batches_rot+bad_batches_trans+bad_batches_rot,
                        columns=['scene_name', 'name'])
        
        # for drive_key in list(task_data_subset):
        for drive_key in task_data_subset:
            drive_idx = int(drive_key)
            if drive_idx not in interest_samples.scene_name.unique():
                continue
            drive = self.zod_drives[drive_idx]
            drive_path = os.path.join(self.lidar_dataset_root, 'drives', drive_key )
            aerial_path = os.path.join(self.aerial_img_dataset_root,'drives',drive.metadata.country_code, drive_key, "aerial")

            calibrations = drive.calibration
            T_lidar2oxts = calibrations.lidars[Lidar.VELODYNE].extrinsics.transform

            num_frames = len(drive.info.camera_frames['front_blur'])
            filename = os.path.join(drive_path, "oxts.hdf5")
            

            # Read HDF5 Data Once
            with h5py.File(filename, "r") as f:
                # def recursively_print(name, obj):
                #     print(name)
                # f.visititems(recursively_print)
                # print('GNSSAntenna/accuracyHeadingAntennas', f['GNSSAntenna/accuracyHeadingAntennas'][()])
                # print('poseSource', f['poseSource'][()])

                orientationMode = f['orientationMode'][()]
                earth_frame_bytes = f['earthFrame'][()]
                earth_frame = b''.join(earth_frame_bytes).decode('utf-8') 

                INSToHostRotation = f['INSToHostRotation'][()]

                datumEllipsoid_bytes = f['datumEllipsoid'][()]
                datumEllipsoid = b''.join(datumEllipsoid_bytes).decode('utf-8') 
                
                lat = f['posLat'][()]
                lon = f['posLon'][()]
                alt = f['posAlt'][()]
                # print(f['heading'][()] )
                yaw = 90 - f['heading'][()] 
                
                pitch = f['pitch'][()]
                roll = f['roll'][()]
                timestamp = OXTS_TIMESTAMP_OFFSET + f['timestamp'][()] + f['leapSeconds'][()][0]
            
            # compute and save info that only need to be computed once per drive
            self.drive_infos[drive_idx] = {
                'lat': lat,
                'lon': lon,
                'alt': alt,
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll,
                'timestamp': timestamp,
                'orientationMode': orientationMode,
                'earth_frame': earth_frame,
                'INSToHostRotation': INSToHostRotation,
                'datumEllipsoid': datumEllipsoid,
                'drive_path': drive_path,
                'aerial_path': aerial_path,
                'T_lidar2oxts': T_lidar2oxts,
                'num_frames': num_frames,
                'calibrations': calibrations,
            }
            
            aerial_metadata_path = os.path.join(self.aerial_img_dataset_root, "drives" ,drive.metadata.country_code, "labels.json" )

            with open(aerial_metadata_path, 'r') as f:
                aer_img_data_temp = json.load(f)
                
            aerial_img_data_df = pd.DataFrame(aer_img_data_temp)
            
            if drive_key not in aerial_img_data_df.drive_idx.unique():
                continue
            
            for frame_idx in range(num_frames):
                
                # skip bad samples
                if len(bad_samples.loc[(bad_samples.scene_name.astype(int) == drive_idx) & (bad_samples.name.astype(int) == frame_idx)]):
                    continue
                # take only the interest samples
                if len(interest_samples.loc[(interest_samples.scene_name.astype(int) == drive_idx) & (interest_samples.name.astype(int) == frame_idx)]):
                    self.metadata_list.append([drive_idx,frame_idx, aerial_img_data_df.loc[(aerial_img_data_df.drive_idx.astype(int) == drive_idx) & (aerial_img_data_df.frame_idx == frame_idx), ['aerial_latlon', 'heading', 'aerial_image', 'resolution'] ].to_dict(orient='records')[0] ])

                
                # if frame_idx > 30:
                #     break
                
                # use only subsample of frames for faster training
                # if frame_idx % 100 == 0:
                #     self.metadata_list.append([drive_idx,frame_idx, aerial_img_data_df.loc[(aerial_img_data_df.drive_idx.astype(int) == drive_idx) & (aerial_img_data_df.frame_idx == frame_idx), ['aerial_latlon', 'heading', 'aerial_image', 'resolution'] ].to_dict(orient='records')[0] ])
                # else:
                #     continue
                
                # self.metadata_list.append([drive_idx,frame_idx, aerial_img_data_df.loc[(aerial_img_data_df.drive_idx.astype(int) == drive_idx) & (aerial_img_data_df.frame_idx == frame_idx), ['aerial_latlon', 'heading', 'aerial_image', 'resolution'] ].to_dict(orient='records')[0] ])
                # pbar.update(1)
                
        # pbar.close()
            

    def __len__(self):
        return len(self.metadata_list)

    
    def __getitem__(self, index):
        data_dict = {}
            
        image_rgb, points, resolution, focal, heading, intensity, metadata, success = self.open_vilsoc_outputs(index)
        # image_rgb_orig = image_rgb.copy()
        # data_dict['image_rgb_init'] = image_rgb.astype(np.float32) # for visualisation

        
        if not success:
            drive_idx, frame_idx, _  = self.metadata_list[index]
            print(f"Focal out of image bounds, at dataset index {[drive_idx, frame_idx]}")
            
            # write in the broken batch (since we're already skipping the bad ones in the init)
            with open("dataset/dataset_utils/bad_ids.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([metadata["scene_name"],metadata["name"],4])
                
            return None
        
        
        
        ########################################## image ####################################
        H, W = image_rgb.shape[:2]
        
        # half crop in pixels to get exactly image_area_dg_m in meters
        half_crop_px = int(0.5 * self.image_area_dg_m * H / self.image_area_zod_m)

        cx, cy = int(focal[0]), int(focal[1])

        # check bounds
        if (cx - half_crop_px < 0) or (cx + half_crop_px > W) or (cy - half_crop_px < 0) or (cy + half_crop_px > H):
            drive_idx, frame_idx, _  = self.metadata_list[index]
            print(f"ZOD offset too large ! Skipping sample at index {[drive_idx, frame_idx]}. Focal = {cx, cy}, Half crop in px : {half_crop_px}")
            
            # write in the broken batch (since we're already skipping the bad ones in the init)
            with open("dataset/dataset_utils/bad_ids.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([metadata["scene_name"],metadata["name"],2])
                
            return None

        image_rgb = image_rgb[cy - half_crop_px : cy + half_crop_px,
                            cx - half_crop_px : cx + half_crop_px]

        image_rgb = cv2.resize(image_rgb, (self.final_img_shape, self.final_img_shape))
        
        # adjust resolution accordingly
        resolution = self.image_area_dg_m / (self.hs * 2) # update resolution = size of the image in m / size of the image in pixels
        
        img = image_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (3, H, W)
        
        ########################################## point cloud ####################################

        rotation = np.eye(3, dtype=np.float32)
        translation = np.zeros(3, dtype=np.float32)

        if self.use_augmentation:
            # sample synthetic offset in meters and radians
            dx = np.random.uniform(-self.max_translation, self.max_translation)
            dy = np.random.uniform(-self.max_translation, self.max_translation)
            dtheta = np.deg2rad(np.random.uniform(-self.max_rot_deg, self.max_rot_deg))

            # build 3x3 rotation around z
            cos_t, sin_t = np.cos(dtheta), np.sin(dtheta)
            Rz = np.array([[cos_t, -sin_t, 0.0],
                        [sin_t,  cos_t, 0.0],
                        [0.0,    0.0,   1.0]], dtype=np.float32)

            # apply to points (choose convention: perturb LiDAR)
            points = (Rz @ points.T).T
            points[:, 0] += dx
            points[:, 1] += dy

            # store offset between perturbed LiDAR and aerial
            rotation = Rz      # (3,3)
            translation = np.array([dx, dy, 0.0], dtype=np.float32)
        
        bev_lidar = self.zod_points_to_bev_pixor_like(points, intensity)
        if bev_lidar is None:
            drive_idx, frame_idx, _ = self.metadata_list[index]
            print(f"Empty BEV at index {[drive_idx, frame_idx]}")
            # log to bad_ids if you want, then
            return None
        
        image = img.astype(np.float32).copy()           # (3,H,W)
        bev   = bev_lidar.astype(np.float32).copy()     # (C,H,W)
        rot   = rotation.astype(np.float32).copy()      # (3,3)
        trans = translation.astype(np.float32).copy()   # (3,)
        res   = np.float32(resolution)                  # (1)        m/pix
        # subsample 30000 points
        choice = np.random.choice(points.shape[0], 30000, replace=False)
        points_subsampled = points[choice].astype(np.float32).copy()  # [30k, 3]
        intensity_subsampled = intensity[choice].astype(np.float32).copy()  # [30k,]
    

         ############## Save in dictionnary ###############

        data_dict = {
            "scene_name": metadata['scene_name'],
            "name": metadata['name'],
            "image":      torch.from_numpy(image),
            "bev_lidar":  torch.from_numpy(bev),
            "rotation":   torch.from_numpy(rot),
            "translation":torch.from_numpy(trans),
            "resolution": torch.tensor(res, dtype=torch.float32),
            "heading":   torch.tensor(np.float32(heading), dtype=torch.float32),
            "points":     torch.from_numpy(points_subsampled),
            "intensity":  torch.from_numpy(intensity_subsampled),
            }
        return data_dict


    def zod_points_to_bev_pixor_like(self, points_xyz, points_r):
        """
        points_xyzr: (N, 4) array [x, y, z, reflectance] in ego frame,
                    already rotated to match aerial heading.
        Returns:
            bev: (C, H, W) = (Z_SIZE+1, Y_SIZE, X_SIZE) float32
                occupancy for each height bin + mean intensity per (x,y).
        """
        
        x_range=(-self.image_area_dg_m/2, self.image_area_dg_m/2)
        y_range=x_range
        z_range=self.z_range
        grid_res=self.BEV_grid_resolution
        
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range

        # Filter ROI
        x = points_xyz[:, 0]
        y = points_xyz[:, 1]
        z = points_xyz[:, 2]
        r = (points_r.astype(np.float32) / 255.0) # normalize for more usable range for the model
        
        mask = (x > x_min) & (x < x_max) & \
            (y > y_min) & (y < y_max) & \
            (z > z_min) & (z < z_max)
        if not np.any(mask):
            return None  # handle upstream

        x = x[mask]; y = y[mask]; z = z[mask]; r = r[mask]

        X_SIZE = int((x_max - x_min) / grid_res)
        Y_SIZE = int((y_max - y_min) / grid_res)
        Z_SIZE = int((z_max - z_min) / grid_res)

        # Quantize
        X = ((x - x_min) / grid_res).astype(np.int32)
        Y = ((y - y_min) / grid_res).astype(np.int32)
        Z = ((z - z_min) / grid_res).astype(np.int32)

        # Safety clip
        X = np.clip(X, 0, X_SIZE - 1)
        Y = np.clip(Y, 0, Y_SIZE - 1)
        Z = np.clip(Z, 0, Z_SIZE - 1)

        # Allocate: (Y, X, Z+1)
        bev = np.zeros((Y_SIZE, X_SIZE, Z_SIZE + 1), dtype=np.float32)
        density = np.zeros((Y_SIZE, X_SIZE), dtype=np.int32)

        # Fill occupancy + accumulated intensity
        for i in range(X.shape[0]):
            yy, xx, zz = Y[i], X[i], Z[i]
            bev[yy, xx, zz] = 1.0
            bev[yy, xx, Z_SIZE] += r[i]
            density[yy, xx] += 1

        # Normalize intensity channel
        mask_den = density > 0
        bev[mask_den, Z_SIZE] /= density[mask_den]

        # (C,H,W) for PyTorch
        bev = bev.transpose(2, 0, 1)  # (Z+1, Y, X)
        return bev


    def open_vilsoc_outputs(self, index): # , method = "transformed"
        
        drive_idx, frame_idx, aer_img_metadata  = self.metadata_list[index]
        drive = self.zod_drives[drive_idx]
        lat, lon, alt = self.drive_infos[drive_idx]['lat'], self.drive_infos[drive_idx]['lon'], self.drive_infos[drive_idx]['alt']
        timestamp = self.drive_infos[drive_idx]['timestamp']
        yaw, pitch, roll = self.drive_infos[drive_idx]['yaw'], self.drive_infos[drive_idx]['pitch'], self.drive_infos[drive_idx]['roll']
        T_lidar2oxts = self.drive_infos[drive_idx]['T_lidar2oxts']
        
        # heading_difference = []
        # T_prev = np.eye(4)
        
        current_timestamp = drive.info.camera_frames['front_blur'][frame_idx].time.timestamp()
        
        # Use np.searchsorted for efficient timestamp lookup
        oxts_idx = np.searchsorted(timestamp, current_timestamp, side='right')
                
        # Extract previous and next OXTS readings
        lat_prev, lon_prev, alt_prev = lat[oxts_idx - 1], lon[oxts_idx - 1], alt[oxts_idx - 1]
        yaw_prev, pitch_prev, roll_prev = yaw[oxts_idx - 1], pitch[oxts_idx - 1], roll[oxts_idx - 1]
        easting_prev, northing_prev, zone_number_prev, zone_letter_prev = utm.from_latlon(lat_prev, lon_prev)

        lat_next, lon_next, alt_next = lat[oxts_idx], lon[oxts_idx], alt[oxts_idx]
        yaw_next, pitch_next, roll_next = yaw[oxts_idx], pitch[oxts_idx], roll[oxts_idx]
        easting_next, northing_next, zone_number_next, zone_letter_next = utm.from_latlon(lat_next, lon_next)

        gamma_prev = 0
        gamma_next = 0
        if drive.metadata.country_code == 'SE':
            gamma_prev = approximate_meridian_convergence(lat_prev, lon_prev, 15)
            gamma_next = approximate_meridian_convergence(lat_next, lon_next, 15)
        
        yaw_prev += gamma_prev # heading should subtract gamma, hence yaw should be adding 
        yaw_next += gamma_next 


        # Compute previous and next transformation matrices
        T_previous = np.eye(4)
        T_previous[:3, :3] = rotation_matrix_from_angles(
            np.radians(roll_prev), np.radians(pitch_prev), np.radians(yaw_prev), order="ZYX"
        )
        T_previous[:3, 3] = [easting_prev, northing_prev, alt_prev]

        T_next = np.eye(4)
        T_next[:3, :3] = rotation_matrix_from_angles(
            np.radians(roll_next), np.radians(pitch_next), np.radians(yaw_next), order="ZYX"
        )
        T_next[:3, 3] = [easting_next, northing_next, alt_next]
        
        # Interpolate transformation
        interp_factor = (current_timestamp - timestamp[oxts_idx - 1]) / (timestamp[oxts_idx] - timestamp[oxts_idx - 1])
        T_current = interpolate_transforms(T_previous, T_next, interp_factor)
        yaw_oxts, _, _ = get_yaw_pitch_roll(T_current)
        heading_oxts = 90 - np.degrees(yaw_oxts)
    
        # Process Lidar Data
       
        lidar = drive.get_compensated_lidar(
            drive.info.camera_frames['front_blur'][frame_idx].time
        )
        pcd = lidar.points  # (N, 3)
        pcd_utm = transform_points(pcd, T_current @ T_lidar2oxts) 

        # Intensity
        intensity = lidar.intensity  # shape (N,)
        
        aerial_lat, aerial_lon = aer_img_metadata['aerial_latlon'] # 
        aerial_easting, aerial_northing, _, _ = utm.from_latlon(aerial_lat, aerial_lon)
        aerial_heading = aer_img_metadata['heading']
        
        # Load Aerial Image and Resolution
        aerial_image_full_path = os.path.join(self.aerial_img_dataset_root, aer_img_metadata["aerial_image"])
        aerial_img = cv2.imread(aerial_image_full_path)
        aerial_img = cv2.cvtColor(aerial_img, cv2.COLOR_BGR2RGB)
        resolution = aer_img_metadata["resolution"]

        # Plot BEV (Bird’s Eye View) with LiDAR points
        bev_image = np.array(aerial_img)
        H, W = bev_image.shape[:2]

        # Rotate by aerial heading
        theta = np.radians(-(heading_oxts-aerial_heading)) 
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])


        translation = np.array([aerial_easting - T_current[0, 3] , aerial_northing - T_current[1, 3]]) 
        # points_m = pcd_utm - np.array([T_current[0, 3], T_current[1, 3], 0])
        points_m = pcd_utm - np.array(T_current[:3, 3])

        # Compute vehicle position in image coordinates
        vehicle_pixel_x = W // 2 - translation[0] / resolution
        vehicle_pixel_y = H // 2 + translation[1] / resolution  # Note: Y-axis flip

        focal = [vehicle_pixel_x, vehicle_pixel_y]
        
        points_m[:, :2] =  points_m[:, :2] @ rotation_matrix.T
        
        success = True
        if focal[0] > W or focal[1] > H or focal[0] < 0 or focal[1] < 0:
            success = False
        
        return bev_image, points_m, resolution, focal, aerial_heading, intensity, {'scene_name': drive_idx ,'name': frame_idx}, success

def fetch_dataloader(args, split='train'):
    """
    For ZOD:
      - if split == 'train': return (train_dataset, val_dataset)
      - else: return a single dataset (usually 'val' for evaluation)
    """
    # Build config dict for your ZOD class from args.data
    zod_cfg = {
        "lidar_dataset_root": args.lidar_dataset_root,
        "aerial_img_dataset_root": args.aerial_img_dataset_root,
        "version": args.version,
        "use_augmentation": args.use_augmentation, # if split == 'train' else False,
        "image_size": args.image_size,
        "image_area_dg_m": args.image_area_dg_m,
        "image_area_zod_m": args.image_area_zod_m,
        "max_translation": args.max_translation,
        "max_rot_deg": args.max_rot_deg,
        "z_range": args.z_range,
        "BEV_grid_resolution": args.BEV_grid_resolution,
    }

    if split == 'train':
        train_dataset = ZOD(task='train', **zod_cfg)
        val_dataset   = ZOD(task='val',   **zod_cfg)
        
        # from torch.utils.data import Subset
        # train_dataset = Subset(train_dataset, [0,1,2,3,4])
        # val_dataset   = Subset(val_dataset,   [0,1,2,3,4])
        print(f"Training with {len(train_dataset)} samples, validation with {len(val_dataset)} samples.")
        return train_dataset, val_dataset

    elif split in ['val', 'validation']:
        val_dataset = ZOD(task='val', **zod_cfg)
        print(f"Validation with {len(val_dataset)} samples.")
        return val_dataset

    elif split == 'test':
        test_dataset = ZOD(task='test', **zod_cfg)
        print(f"Test with {len(test_dataset)} samples.")
        return test_dataset

    else:
        raise ValueError(f"Unknown split '{split}' for dataset 'zod'.")

if __name__ == '__main__':
    cfg = {
        "lidar_dataset_root": "/work/vita/datasets/zod",
        "aerial_img_dataset_root": "/work/vita/datasets/zod_crossview_processed_100m",
        "version": "full",
        'use_augmentation': False,
        'augmentation_noise': 0.02,
        'augmentation_rotation': 1.0,
        'image_size': 492,
    }
    data_test = ZOD(task='test', **cfg)
    print(len(data_test))
    start = time.time()

    count = 0
    for i in range(len(data_test)):
        if(i >= 10):
            break
        print(i)
        a = data_test[i]
        count += a['points'].shape[0]

    end = time.time()
    print((end - start) / 1000.0, 's')
    print(count / len(data_test))
