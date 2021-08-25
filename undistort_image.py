from utils.load_data import load_camera_calib
from utils import constants as _C

import torch
import numpy as np
import cv2
import json

import os, sys
import os.path as osp

from tqdm import tqdm


base_dir = _C.BASE_RAW_DATA_DIR
org_video_fldr = _C.HD_EXTRACTED_VIDEO_FLDR
undist_video_fldr = _C.HD_UNDIST_VIDEO_FLDR
calib_filename = _C.CAMERA_CALIB_FILENAME

dates = _C.EXP_DATES
exps = _C.EXP_SEQUENCES

# TODO: Implement multicore processing

for date in dates:
    for exp in exps:
        org_fldr = osp.join(base_dir, date, org_video_fldr, exp)
        trg_fldr = osp.join(base_dir, date, undist_video_fldr, exp)
        calib_file = osp.join(base_dir, date, calib_filename)

        camera_infos = load_camera_calib(calib_file)
        _, available_cameras, _ = next(os.walk(org_fldr))

        for camera_name in tqdm(available_cameras, desc='Cameras', leave=False):
            _, _, image_names = next(os.walk(osp.join(org_fldr, camera_name)))
            os.makedirs(osp.join(trg_fldr, camera_name), exist_ok=True)
            camera_info = camera_infos[camera_name]
            
            # Generate meshgrid for undistorting image
            w, h = camera_info['resolution']
            
            intrinsics = camera_info['camera_intrinsics']
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]

            grid_x = (np.arange(w, dtype=np.float32) - cx) / fx
            grid_y = (np.arange(h, dtype=np.float32) - cy) / fy
            meshgrid = np.stack(np.meshgrid(grid_x, grid_y), axis=2).reshape(-1, 2)

            # distortion information
            dist = camera_info['camera_dist']
            k = np.array([dist[0], dist[1], dist[-1]])
            p = np.array([dist[2], dist[3]])
            r2 = np.square(meshgrid).sum(axis=-1)
            r4 = r2 ** 2
            r6 = r2 ** 3
            radial = meshgrid * (1 + k[0] * r2 + k[1] * r4 + k[2] * r6).reshape(-1, 1)
            tangential_1 = p.reshape(1, 2) * np.broadcast_to(meshgrid[:, 0:1] * meshgrid[:, 1:2], (len(meshgrid), 2))
            tangential_2 = p[::-1].reshape(1, 2) * (meshgrid**2 + np.broadcast_to(r2.reshape(-1, 1), (len(meshgrid), 2)))
            
            # undistorting meshgrid
            meshgrid = radial + tangential_1 + tangential_2
            meshgrid *= np.array([fx, fy]).reshape(1, 2)
            meshgrid += np.array([cx, cy]).reshape(1, 2)
            meshgrid = meshgrid.astype(np.float32).reshape((h, w, 2))
            
            meshgrid_int16 = cv2.convertMaps(meshgrid, None, cv2.CV_16SC2)

            # # Check if this distortion is correct
            # from utils.load_data import load_op26
            # from utils.viz import draw_2d_skeleton
            # from utils.viz_utils import project2D
            
            # op26_fldr = osp.join(base_dir, date, _C.HD_KEYPOINTS_STAGE2_FLDR, exp)
            # _, _, json_names = next(os.walk(op26_fldr))
            # json_names.sort()
            # joints, ids = load_op26(op26_fldr, 100)
            
            # pose = camera_info['camera_pose']
            # transl = camera_info['camera_transl']
            
            # x2d_list, x2d_undist_list = [], []
            # for joints_, ids_ in zip(joints, ids):
            #     x2d, mask = project2D(joints_, w, h, intrinsics, pose, transl)
            #     x2d[~mask] = 0
            #     x2d_list += [x2d]

            #     x2d_undist, mask_undist = project2D(joints_, w, h, intrinsics, pose, transl, dist)
            #     x2d_undist[~mask_undist] = 0
            #     x2d_undist_list += [x2d_undist]

            for image_name in tqdm(image_names, desc='Undistorting ...', leave=False):
                dist_image = cv2.imread(osp.join(org_fldr, camera_name, image_name))
                trg_path = osp.join(trg_fldr, camera_name, image_name)
                
                undist_image = cv2.remap(dist_image.copy(), *meshgrid_int16, cv2.INTER_CUBIC)
                cv2.imwrite(trg_path, undist_image)
                
                # # Check if this distortion is correct
                # frame = image_name.split('.')[0].split('_')[-1]
                # json_name = 'body3DScene_%s.json'%frame
                # idx = json_names.index(json_name)

                # dist_image_ = dist_image.copy()
                # for subj_idx, x2d_ in enumerate(x2d_list):
                #     x = x2d_[idx, :, 0].astype('int32')
                #     y = x2d_[idx, :, 1].astype('int32')
                #     undist_image = draw_2d_skeleton(x, y, undist_image.copy(), subj_idx, "op26")
                #     dist_image = draw_2d_skeleton(x, y, dist_image.copy(), subj_idx, "op26")
                    
                #     x_undist = x2d_undist_list[subj_idx][idx, :, 0].astype('int32')
                #     y_undist = x2d_undist_list[subj_idx][idx, :, 1].astype('int32')
                #     dist_image_ = draw_2d_skeleton(x_undist, y_undist, dist_image_.copy(), subj_idx, "op26")
                
                # cv2.imwrite(osp.join(trg_fldr, camera_name, 'undist-'+image_name), undist_image)
                # cv2.imwrite(osp.join(trg_fldr, camera_name, 'dist-'+image_name), dist_image)
                # cv2.imwrite(osp.join(trg_fldr, camera_name, 'dist_-'+image_name), dist_image_)
                # import pdb; pdb.set_trace()