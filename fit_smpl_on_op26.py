from smpl import build_body_model
from smplify.smplify import SMPLify3D
from smplify.loss import align_two_joints
from utils.load_data import load_op26

from utils.geometry import rot6d_to_rotmat
from utils.viz import draw_3d_skeleton, draw_smpl_body
from utils.sync_utils import refine_params
from utils import constants

import torch
from torch import nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm, trange
import os
import os.path as osp
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
body_pose_prior_weights = [10.0, 4.0, 2.0, 0.5]
shape_prior_weight = [1e2, 2 * 1e1, 1e1, 4.0]

# # Exp configuration TODO: Make it as a parser
# dates = ['190503', '190510', '190517', '190607']
dates = ['190510']
exps = ['exp01', 'exp02', 'exp03', 'exp04', 'exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp11', 'exp12', 'exp13', 'exp14']
sids = [1, 2]
base_dir = 'dataset/MBL_DomeData/dome_data'
fldr_op26 = 'hdPose3d_stage1_op25'
fldr_raw_imu = 'dataset/dome_IMU'
fldr_result = 'smpl_result'
fldr_image = 'smpl_fit_video'
fldr_smplify = 'dataset/3D_SMPLify'

# # SMPL configuration
model_type = 'smpl'
body_model_folder = osp.join(fldr_smplify, 'models', model_type)
SMPL_regressor = osp.join(fldr_smplify, 'J_regressor_extra.npy')

# SMPL mean params
mean_params = np.load(osp.join(fldr_smplify, 'smpl_mean_params.npz'))
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
    joint_mapping = constants.OP26_TO_J17
    eval_idx = [i for i in range(14)]
elif joint_type == 'op25':
    ign_joint_idx = [1, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]      # Pelvis, Chest (Neck) seems not corresponding to OP25
    joint_mapping = constants.OP26_TO_OP25
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
            fldr_save_image = osp.join(base_dir, date, fldr_image, exp)
            os.makedirs(fldr_save_image, exist_ok=True)
            if osp.exists(osp.join(base_dir, date, fldr_image, '%s.mp4'%exp)):
                continue
        
        output = {}
        output['SMPL_params'] = []
        table_dtype = np.dtype([
                ('smpl_pose', np.float32, 69),
                ('smpl_betas', np.float32, 10),
                ('smpl_global_orient', np.float32, 3)])

        for sid in sids:

            # Load IMU annotation ==> Sex information
            fldr_anno = osp.join(fldr_raw_imu, 'Set%02d'%sid)
            annotation = pd.read_csv(osp.join(fldr_anno, 'Set%02d_annotations.csv'%sid), index_col='EventType')
            sex_info = annotation[['Value', 'Date', 'Exp']].loc['What is your sex?']
            sex_info = sex_info.loc[sex_info['Date'] == int(date)]
            sex = sex_info.loc[sex_info['Exp'] == int(exp[-2:])]['Value'][0].lower()
            
            # Build SMPL model
            body_model = build_body_model(body_model_folder, SMPL_regressor, batch_size, device, sex)

            # Load KP data
            fldr_kp = osp.join(base_dir, date, fldr_op26, exp)
            keypoints, _ = load_op26(fldr_kp)
            
            try:
                keypoints = keypoints[sid-1].copy()
            except:
                continue
            
            # keypoints = keypoints[1350:]
            keypoints = torch.from_numpy(keypoints).to(device=device, dtype=dtype)

            J_regressor = torch.from_numpy(np.load(constants.JOINT_REGRESSOR_H36M)).float()
            J_regressor = J_regressor[None, :].expand(batch_size, -1, -1).to(device)

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
                            J_regressor, opt_output.vertices)[:, constants.H36M_TO_J17, :]
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
                        draw_smpl_body(body_model, opt_output, filename=osp.join(fldr_save_image, 'smpl.png'))
                        img = cv2.imread(osp.join(fldr_save_image, 'smpl.png'))
                        img_n = "dome_video_%08d.jpg"%idx
                        cv2.imwrite(osp.join(fldr_save_image, img_n), img)
                    
            output['SMPL_params'] = np.concatenate(output['SMPL_params'])
            fldr_output = osp.join(base_dir, date, fldr_result)
            os.makedirs(fldr_output, exist_ok=True)
            np.save(osp.join(fldr_output, '%s_sid%02d'%(exp, sid)), output)