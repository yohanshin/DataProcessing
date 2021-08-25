from smpl import build_body_model
from smplify.smplify import SMPLify3D
from smplify.loss import align_two_joints

from utils.load_data import load_op26
from utils.geometry import rot6d_to_rotmat
from utils.viz import draw_3d_skeleton, draw_smpl_body, draw_2d_skeleton
from utils.viz_utils import project2D
from utils.sync_utils import refine_params
from utils import constants as _C

from generate_dome_video import camera_info

import torch
from torch import nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm, trange
import os
import os.path as osp

# To enabling draw smpl model in remote environment
os.environ['PYOPENGL_PLATFORM'] = 'egl'

__author__ = "Soyong Shin"


save_image = True

# Optim configuration TODO: Make it as a parser
device = 'cuda'
dtype = torch.float32
batch_size = 1
lr = 5e-2
maxiters = 100
optimizer_type = 'adam'

# Loss weight
joint_dist_weights = [1e4] * 4
body_pose_prior_weights = [20.0, 10.0, 10.0, 3.0]
shape_prior_weight = [1e2, 2 * 1e1, 1e1, 1e1]

# # Exp configuration TODO: Make it as a parser
dates = _C.EXP_DATES
exps = _C.EXP_SEQUENCES
sids = _C.EXP_SUBJECTS
base_dir = _C.BASE_RAW_DATA_DIR
op26_fldr_ = _C.HD_KEYPOINTS_STAGE2_FLDR
raw_imu_fldr = _C.RAW_IMU_DIR
result_fldr = 'smpl_result'
image_fldr = 'smpl_fit_video'
smplify_fldr = 'dataset/3D_SMPLify'

# # SMPL configuration
model_type = 'smpl'
body_model_folder = osp.join(smplify_fldr, 'models', model_type)
SMPL_regressor = osp.join(smplify_fldr, 'J_regressor_extra.npy')

# SMPL mean params
mean_params = np.load(osp.join(smplify_fldr, 'smpl_mean_params.npz'))
init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0).expand(batch_size, -1)
init_betas = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
init_pose = init_pose.to(device=device).expand(batch_size, -1)
init_betas = init_betas.to(device=device).expand(batch_size, -1)
init_pose = rot6d_to_rotmat(init_pose).view(batch_size, 24, 3, 3)
init_orient = init_pose[:, 0].unsqueeze(1)
init_pose = init_pose[:, 1:]

# Joint configuration
joint_type = 'op25'
if joint_type == 'h36m':
    ign_joint_idx = None
    joint_mapping = _C.OP26_TO_J17
    eval_idx = [i for i in range(14)]
elif joint_type == 'op25':
    ign_joint_idx = [1, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]      # Pelvis, Chest (Neck) seems not corresponding to OP25
    joint_mapping = _C.OP26_TO_OP25
    eval_idx = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
else:
    NotImplementedError, "Joint type {} is not implemented. Check again".format(joint_type)

# Build SMPLify
smplify = SMPLify3D(joint_dist_weights=joint_dist_weights,
                    body_pose_prior_weights=body_pose_prior_weights,
                    shape_prior_weight=shape_prior_weight,
                    maxiters=maxiters, optimizer_type=optimizer_type,
                    ign_joint_idx=ign_joint_idx, joint_type=joint_type, lr=lr)

for date in dates:
    for exp in exps:
        
        if save_image:
            save_image_fldr = osp.join(base_dir, date, image_fldr, exp)
            os.makedirs(save_image_fldr, exist_ok=True)
            video_fldr = osp.join(base_dir, date, image_fldr)
        
        for sidx, sid in enumerate(sids):
            output = {}
            output['SMPL_params'] = []
            table_dtype = np.dtype([
                    ('smpl_pose', np.float32, 69),
                    ('smpl_betas', np.float32, 10),
                    ('smpl_global_orient', np.float32, 3)])

            video_name = '%s_%s.mp4'%(exp, sid)
            if osp.exists(osp.join(video_fldr, video_name)):
                continue

            # Load IMU annotation ==> Sex information
            anno_fldr = osp.join(raw_imu_fldr, sid)
            annotation = pd.read_csv(osp.join(anno_fldr, '%s_annotations.csv'%sid), index_col='EventType')
            sex_info = annotation[['Value', 'Date', 'Exp']].loc['What is your sex?']
            sex_info = sex_info.loc[sex_info['Date'] == int(date)]
            sex = sex_info.loc[sex_info['Exp'] == int(exp[-2:])]['Value'][0].lower()
            
            # Build SMPL model
            body_model = build_body_model(body_model_folder, SMPL_regressor, batch_size, device, sex)

            # Load KP data
            op26_fldr = osp.join(base_dir, date, op26_fldr_, exp)
            keypoints, _ = load_op26(op26_fldr, 7000)
            
            try:
                keypoints = keypoints[sidx].copy()
            except:
                continue
            
            if sidx == 0:
                keypoints = keypoints[4000:4500]
            else:
                keypoints = keypoints[5500:6000]

            keypoints = torch.from_numpy(keypoints).to(device=device, dtype=dtype)

            J_regressor = torch.from_numpy(np.load(_C.JOINT_REGRESSOR_H36M)).float()
            J_regressor = J_regressor[None, :].expand(batch_size, -1, -1).to(device)
            
            if save_image:
                x2d, mask = project2D(keypoints.detach().cpu().numpy(), 540, 360,
                                        camera_info['K'], camera_info['R'], camera_info['t'])
                x2d[~mask] = 0

            with tqdm(total=keypoints.shape[0], leave=False) as prog_bar:
                for idx in range(keypoints.shape[0]):
                    org_keypoints = keypoints[idx].unsqueeze(0).clone()
                    keypoints_3d_gt = keypoints[idx, joint_mapping].unsqueeze(0)
                    if joint_type == 'h36m':
                        keypoints_3d_gt[:, 15] = -1  # There is no keypoints of Spine in OP26
                        
                    conf = keypoints_3d_gt[:, :, -1]
                    mask = keypoints_3d_gt[0, :, -1] < 0
                    
                    keypoints_3d_gt[:, :, :-1] *= 1e-2
                    opt_pose, opt_betas, opt_global_orient = \
                        smplify(init_pose, init_betas, init_orient, 
                                body_model, keypoints_3d_gt, device, dtype, init_frame=(idx==0))

                    opt_output = body_model(betas=opt_betas, body_pose=opt_pose, 
                                            global_orient=opt_global_orient,  pose2rot=True)
                    
                    if joint_type == 'h36m':
                        keypoints_3d_pred = torch.matmul(
                            J_regressor, opt_output.vertices)[:, _C.H36M_TO_J17, :]
                    elif joint_type == 'op25':
                        keypoints_3d_pred = opt_output.joints[:, :25]

                    gt, pred = align_two_joints(keypoints_3d_gt[:, :, :-1], keypoints_3d_pred, joint_type)
                    mpjpe = torch.sqrt((gt[0, eval_idx] - pred[0, eval_idx]).pow(2).sum(-1))[~mask[eval_idx]].mean(0) * 1e2
                    msg = 'MPJPE: %.2f'%mpjpe
                    prog_bar.set_postfix_str(msg)
                    prog_bar.update(1)
                    prog_bar.refresh()

                    table_segment = np.empty(1, dtype=table_dtype)
                    table_segment['smpl_pose'] = opt_pose.detach().cpu().numpy()
                    table_segment['smpl_betas'] = opt_betas.detach().cpu().numpy()
                    table_segment['smpl_global_orient'] = opt_global_orient.detach().cpu().numpy()
                    output['SMPL_params'].append(table_segment)

                    if save_image:
                        op26_background = np.ones((540, 540, 3)).astype(np.uint8) * 255
                        img = np.ones((540, 1080, 3)).astype(np.uint8) * 255
                        x = x2d[idx, :, 0].astype('int32')
                        y = x2d[idx, :, 1].astype('int32')
                        op26_img = draw_2d_skeleton(x, y, op26_background.copy(), 0, "op26")
                        
                        draw_smpl_body(body_model, opt_output, camera_info, 
                                       filename=osp.join(save_image_fldr, 'smpl.png'))
                        smpl_img = cv2.imread(osp.join(save_image_fldr, 'smpl.png'))
                        img_n = "dome_video_%08d.jpg"%idx
                        
                        img[:, :540] = smpl_img
                        img[:, 540:] = op26_img
                        cv2.imwrite(osp.join(save_image_fldr, img_n), img)

            output['SMPL_params'] = np.concatenate(output['SMPL_params'])
            output_fldr = osp.join(base_dir, date, result_fldr)
            os.makedirs(output_fldr, exist_ok=True)
            np.save(osp.join(output_fldr, '%s_sid%02d'%(exp, sidx)), output)

            os.system("ffmpeg -framerate 29.97 -start_number 0 -i {}/dome_video_%08d.jpg".format(save_image_fldr) + " -vcodec mpeg4 {}".format(osp.join(video_fldr, video_name)))
            os.system("echo y")