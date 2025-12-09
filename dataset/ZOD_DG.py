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
from imgaug import augmenters as iaa
from scipy.spatial.transform import Rotation
import time
import json
import utm
import pandas as pd
from tqdm import tqdm

from open_visloc_essential.utils.transformations import rotation_matrix_from_angles, get_yaw_pitch_roll, filter_above_ground, compute_heading
from open_visloc_essential.utils.process import memory_usage

# Load ZOD DevKit
from zod import ZodFrames, ZodSequences, ZodDrives
import zod.constants as constants
from zod.constants import Camera, Lidar, Anonymization, AnnotationProject
from zod.data_classes.ego_motion import OXTS_TIMESTAMP_OFFSET, interpolate_transforms
from zod.data_classes import LidarData
from zod.utils.geometry import transform_points
from zod.visualization.lidar_on_image import get_3d_transform_camera_lidar
from zod.constants import Camera, Lidar
from zod.data_classes.metadata import SequenceMetadata

sys.path.append("../")

from typing import List
# plt.rcParams["figure.figsize"] = [20, 10]

pp = [os.path.dirname((os.path.abspath(__file__))),
      os.path.dirname((os.path.abspath('.'))),
      os.path.dirname((os.path.abspath('..')))]
for p in pp:
    if p not in sys.path:
        sys.path.append(p)

from utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    random_sample_rotation_z,
    get_transform_from_rotation_translation,
)


# to replace cv2 and o3d in get_item, which may cause issues when trying to use several workers. 
import torch
import torch.nn.functional as F
from dataset.dataset_utils.utils_zod import  approximate_meridian_convergence, voxel_downsample, create_bev_pointmap # ,rotate_image_torch, resize_bilinear,
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
        self.image_area_dg_m = config['image_area_dg_m']
        self.image_area_zod_m = config['image_area_zod_m']
        # 
        self.point_limit = config['point_limit']
        # 
        self.use_augmentation = config['use_augmentation']
        self.aug_noise = config['augmentation_noise']
        self.aug_rotation = config['augmentation_rotation']
        # image aug
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Add((-30, 30))),
            iaa.Sometimes(0.5, iaa.LinearContrast((0.7, 1.3))),  # 
            iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 8))),  #
            iaa.Sometimes(0.5, iaa.ImpulseNoise(p=(0, 0.003))),  # 
            iaa.Sometimes(0.5, iaa.MotionBlur(3)),  # 
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.5))),  # 
        ])
        
        self.metadata_list = []
        self.drive_infos = {}
        
        ####### For debugging, use only subset of training drives ########
        # task_data_subset = ["000016"] 
        task_data_subset = task_data_subset[:1]
        
        bad_samples = pd.read_csv("dataset/dataset_utils/bad_ids.csv")

        
        # count total number of frames for tqdm
        total_frames = 0
        for i in range(len(task_data_subset)):
            total_frames += len(self.zod_drives[task_data_subset[i]].info.camera_frames['front_blur']) 
        
        # Create single progress bar
        pbar = tqdm(total=total_frames, desc="Processing ZOD Drives", unit="Frame", colour="blue")
        
        # for drive_key in list(task_data_subset):
        for drive_key in task_data_subset:
            
            drive_idx = int(drive_key)
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
            
            # for frame_idx in range(num_frames):
            for frame_idx in range(num_frames):
                # skip bad samples
                
                if len(bad_samples.loc[(bad_samples.scene_name.astype(int) == drive_idx) & (bad_samples.name.astype(int) == frame_idx)]):
                    continue
                
                if frame_idx > 0:
                    break
                
                self.metadata_list.append([drive_idx,frame_idx, aerial_img_data_df.loc[(aerial_img_data_df.drive_idx.astype(int) == drive_idx) & (aerial_img_data_df.frame_idx == frame_idx), ['aerial_latlon', 'heading', 'aerial_image', 'resolution'] ].to_dict(orient='records')[0] ])

                
                # use only subsample of frames for faster training
                # if frame_idx % 4 == 0:
                #     self.metadata_list.append([drive_idx,frame_idx, aerial_img_data_df.loc[(aerial_img_data_df.drive_idx.astype(int) == drive_idx) & (aerial_img_data_df.frame_idx == frame_idx), ['aerial_latlon', 'heading', 'aerial_image', 'resolution'] ].to_dict(orient='records')[0] ])
                # else:
                #     continue
                
                # self.metadata_list.append([drive_idx,frame_idx, aerial_img_data_df.loc[(aerial_img_data_df.drive_idx.astype(int) == drive_idx) & (aerial_img_data_df.frame_idx == frame_idx), ['aerial_latlon', 'heading', 'aerial_image', 'resolution'] ].to_dict(orient='records')[0] ])
                pbar.update(1)
                
        pbar.close()
            

    def __len__(self):
        return len(self.metadata_list)

    def _augment_point_cloud(self, points, rotation, translation):
        r"""
        Augment point clouds.  点云增强
        1. Random rotation to one point cloud.
        2. Random noise.
        ref_points = src_points @ rotation.T + translation
        Args:

        """
        aug_rotation = random_sample_rotation_z(self.aug_rotation)
        aug_translation = (np.random.randn(3) - 0.5) * 1.0

        points += aug_translation
        translation -= aug_translation @ rotation.T 

        points = np.matmul(points, aug_rotation.T)
        rotation = np.matmul(rotation, aug_rotation.T)

        points += (np.random.randn(points.shape[0], 3) - 0.5) *2* self.aug_noise

        return points, rotation, translation

    
    def __getitem__(self, index):
        data_dict = {}
            
        image_rgb, points, resolution, focal, metadata, success = self.open_vilsoc_outputs(index)
        # image_rgb_orig = image_rgb.copy()
        
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
        
        img_intermediate_size = 1024
        
        image_rgb = cv2.resize(image_rgb, (img_intermediate_size, img_intermediate_size)) # make sure all the images have the same size
        # image_rgb = resize_bilinear(image_rgb, img_intermediate_size, img_intermediate_size)
        resolution = self.image_area_zod_m / img_intermediate_size        # update resolution = size of the image in m / size of the image in pixels
        focal[0] = focal[0] * (img_intermediate_size / W)
        focal[1] = focal[1] * (img_intermediate_size / H)
        W, H = image_rgb.shape[:2]
        

        # crop image to desired real world size
        # area_crop = int(self.image_area_dg_m / resolution / 2)
        
        x, y = 0, 0
        rotation_z = Rotation.from_euler('xyz', [0, 0, 0]).as_matrix()
        
        # print(rotation_z)
        if self.use_augmentation:
            roz = np.random.uniform(0, 360)
            rotation_z = Rotation.from_euler('xyz', [0, 0, roz / 360.0 * np.pi * 2]).as_matrix()
            center = (int(focal[0]), int(focal[1]))
            
            # avoid using cv2, use custom np and torch based functions
            M = cv2.getRotationMatrix2D(center, roz, 1.0)
            image_rgb = cv2.warpAffine(image_rgb, M, (image_rgb.shape[1], image_rgb.shape[0]))
            # image_rgb = rotate_image_torch(image_rgb, roz, center=center)
            
            # x = int(np.random.uniform(-100, 100))
            # y = int(np.random.uniform(-100, 100))
            
            # make sure the crop area is within the image
            x_min = max(self.hs - int(focal[0]), -100) # before, had aera_crop instead of self.hs
            x_max = min(W - int(focal[0]) -  self.hs, 100)
            y_min = max(self.hs - int(focal[1]), -100)
            y_max = min(H - int(focal[1]) -  self.hs, 100)
                        
            x = int(np.random.uniform(x_min, x_max))
            y = int(np.random.uniform(y_min, y_max))
            
        
        # center the image on the focal point and crop
        
        # Checking if for some reason, the cropping is still faulty and fixing it, even if probably bad. 
        if ((int(focal[1]) + y - self.hs) > 0) and ((int(focal[1]) + y + self.hs) < H) and ((int(focal[0]) + x - self.hs) > 0) and ((int(focal[0]) + x + self.hs) < W) :
            
            image_rgb = image_rgb[int(focal[1]) + y - self.hs:int(focal[1]) + y + self.hs, int(focal[0]) + x - self.hs:int(focal[0]) + x + self.hs]
            # adjust focal point accordingly
            focal[0], focal[1] = (self.hs - x), (self.hs - y)
        else :
            drive_idx, frame_idx, _  = self.metadata_list[index]
            print(f"Bad crop size {image_rgb.shape} at index {[drive_idx, frame_idx]}. Crop info x : [{int(focal[1]) + y - self.hs}:{int(focal[1]) + y + self.hs}, y : [{int(focal[0]) + x - self.hs}:{int(focal[0]) + x + self.hs}]")
                        
            # write in the broken batch (since we're already skipping the bad ones in the init)
            with open("dataset/dataset_utils/bad_ids.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([metadata["scene_name"],metadata["name"],1])
            
            return None
            
            
            
        # make sure the image is correct
        if image_rgb.shape[0] != 2 * self.hs or image_rgb.shape[1] != 2 * self.hs:
            drive_idx, frame_idx, _  = self.metadata_list[index]
            print(f"Bad image size {image_rgb.shape} at index {[drive_idx, frame_idx]}. Crop info x : [{int(focal[1]) + y - self.hs}:{int(focal[1]) + y + self.hs}, y : [{int(focal[0]) + x - self.hs}:{int(focal[0]) + x + self.hs}]")
            # return None
            # image_rgb = np.zeros((2*self.hs, 2*self.hs, 3))
            
            # write in the broken batch (since we're already skipping the bad ones in the init)
            with open("dataset/dataset_utils/bad_ids.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([metadata["scene_name"],metadata["name"],2])
                
            return None
            # return 2
        
        
        # adjust resolution accordingly
        resolution = self.image_area_dg_m / (self.hs * 2) # update resolution = size of the image in m / size of the image in pixels
        
        # normalize image 
        image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)[..., np.newaxis]  # 
        if self.use_augmentation:
            image = self.aug.augment_image(image)
            image = image.astype(np.float32) / 255.0
        else:
            image = image / 255.0
        
        
        ########################################## point cloud ####################################

        # downsample point cloud (replace o3d with torch and numpy version for multi worker stability)
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(points)
        # point_cloud = point_cloud.voxel_down_sample(0.3)
        # points = np.array(point_cloud.points)
        points = voxel_downsample(points, voxel_size=0.3)
        points_shape = points.shape
        
        len = np.linalg.norm(points, axis=1) # some sample have a large z weirdly, but this makes the norm explode...
        # len = np.linalg.norm(points[:,:-1], axis=1)
        idx = np.where((len < self.image_area_dg_m) & (len > 4) & (points[..., 2] > -2)) 
        points = points[idx[0]]
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[:self.point_limit]
            points = points[indices]
            
        if points.shape[0] == 0:
            drive_idx, frame_idx, _  = self.metadata_list[index]
            print(f"Empty point cloud (1) at dataset index {[drive_idx, frame_idx]}")
            
            # write in the broken batch (since we're already skipping the bad ones in the init)
            with open("dataset/dataset_utils/bad_ids.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([metadata["scene_name"],metadata["name"],3])
                
            return None
            # return 3

        # point augmentation
        rotation = rotation_z  
        translation = np.zeros((3, ))   # (3,)
        
        if self.use_augmentation:
            points, rotation, translation = self._augment_point_cloud(points, rotation, translation)
        
        
        
        ############## Save in dictionnary ###############
        # infomation
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['name'] = metadata['name']

        # input data
        data_dict['image'] = image.astype(np.float32)  # [H, W, C]
        data_dict['points'] = points.astype(np.float32)  # [N, 3]
        data_dict['points_feats'] = np.ones((points.shape[0], 1), dtype=np.float32)

        # for show
        data_dict['image_rgb'] = image_rgb.astype(np.float32)

        # trans
        data_dict['rotation'] = rotation.astype(np.float32)  # [3, 3]
        data_dict['translation'] = translation.astype(np.float32)  # [3, ]
        data_dict['scale'] = np.float32(resolution) # pair_data['scale'].astype(np.float32)  # [3, 3]
        data_dict['focal'] = np.array(focal).astype(np.float32)
        
        data_dict['BEV_pointmap'] = create_bev_pointmap(points, xy_range=self.image_area_dg_m, grid_size=2*self.hs).astype(np.float32)  # [5, 480, 480]
        # data_dict["heatmap_GT"] = make_gaussian_heatmap(data_dict, focal, translation, resolution, self.hs).astype(np.float32)  # [1, H, W]
        return data_dict


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
        pcd = drive.get_compensated_lidar(drive.info.camera_frames['front_blur'][frame_idx].time).points
        pcd_utm = transform_points(pcd, T_current @ T_lidar2oxts) 
        
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
        points_m = pcd_utm - np.array([T_current[0, 3], T_current[1, 3], 0])
        
        # Compute vehicle position in image coordinates
        vehicle_pixel_x = W // 2 - translation[0] / resolution
        vehicle_pixel_y = H // 2 + translation[1] / resolution  # Note: Y-axis flip

        focal = [vehicle_pixel_x, vehicle_pixel_y]
        
        points_m[:, :2] =  points_m[:, :2] @ rotation_matrix.T
        
        success = True
        if focal[0] > W or focal[1] > H or focal[0] < 0 or focal[1] < 0:
            success = False
        
        return bev_image, points_m, resolution, focal, {'scene_name': drive_idx ,'name': frame_idx}, success
    
        ## For plotting overlapping lidar and image
        # fig, axs = plt.subplots(1)
        # points = points_m[:, :2]
        # temp_pcd_rot_pix = ((points - translation) @ rotation_matrix.T / resolution).astype(int)
        # # temp_pcd_rot_pix = np.clip(temp_pcd_rot_pix, 0, W - 1)

        # x_ind = np.clip(W//2 + temp_pcd_rot_pix[:,0], 0, W - 1)
        # y_ind = np.clip(H//2 - temp_pcd_rot_pix[:,1], 0, H - 1)

        # axs.imshow(bev_image, origin='upper')
        # axs.scatter(x_ind, y_ind, s=0.1, c='purple', alpha=0.3, label="LiDAR Points")
        # # axs[0].quiver(W / 2, H / 2, np.sin(np.radians(heading_move)), np.cos(np.radians(heading_move)), color='cyan', scale=20, label='Movement')
        # axs.quiver(W//2 - translation[0]/resolution, H//2 + translation[1]/resolution , np.sin(np.radians(aerial_heading)), np.cos(np.radians(aerial_heading)), color='r', scale=20, label='OXTS Heading')
        # axs.legend(loc=2)
        # axs.axis('off')
        # axs.set_title('Aerial Image from Cross-View Dataset')
        
        # if method == "normal":
        #     return bev_image, points_m, translation, rotation_matrix, resolution, focal, {'scene_name': drive_idx ,'name': frame_idx}
        # elif method == "transformed":
        #     points_m[:2,:2] =  points_m[:2,:2] @ rotation_matrix.T
    
    def make_gaussian_heatmap(self, data, size=64, sigma = 2.0, device="cpu"):
        """
        center_xy: (cx, cy) in heatmap pixel coordinates (float)
        size_hw: (H, W) of heatmap
        sigma: std in pixels
        """
        H, W = size 
        
        downsize = size / (self.hs*2)
        # center = focal + translation / scale
        cy, cx = data["focal"][1] + data["translation"][1]/data["scale"], data["focal"][0] + data["translation"][0]/data["scale"]
        cy, cx = cy * downsize, cx * downsize  # downsample to heatmap size
        
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        heatmap = torch.exp(-0.5 * dist2 / (sigma ** 2))
        # normalize to sum 1 (soft target)
        heatmap = heatmap / (heatmap.sum() + 1e-6)
        return heatmap  # (H, W)
    
    

if __name__ == '__main__':
    cfg = {
        "lidar_dataset_root": "/work/vita/datasets/zod",
        "aerial_img_dataset_root": "/work/vita/datasets/zod_crossview_processed_100m",
        "version": "full",
        'point_limit': 30000,
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
