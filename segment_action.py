from utils.load_data import load_json, load_op26
from utils import constants

import os, sys
import os.path as osp
import numpy as np
import pandas as pd
import json

from tqdm import tqdm

__author__ = "Soyong Shin"


# dates = ['190510', '190517', '190607']
# exps = ['exp01', 'exp02', 'exp03', 'exp04', 'exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp11', 'exp12', 'exp13', 'exp14']
dates = ['190510']
exps = ['exp01']
singular_exps = ['190517_exp12', '190607_imu12']
sids = ['Set01', 'Set02']
part_list = ['chest', 'head', 'lbicep', 'lfoot', 'lforearm', 'lhand', 'lshank', 'lthigh',
             'rbicep', 'rfoot', 'rforearm', 'rhand', 'rshank', 'rthigh', 'sacrum']

base_dir = 'dataset/MBL_DomeData/dome_data'
op26_fldr_ = 'hdPose3d_stage2_op25'
imu_fldr_ = 'mc10_IMU'
target_fldr_ = 'Processed'
target_op26_fldr = 'OpenPose3D'
target_imu_fldr = 'MC10_IMU'
label_filename = 'action_label.xlsx'

label = pd.read_excel(osp.join(base_dir, label_filename), sheet_name='action labels', index_col=1)
action_list_ = label.index[4:-1].tolist()
action_list = [action[:-4] for action in action_list_ if action[-3:] == 'end']

for date in dates:
    for exp in exps:
        # Load Keypoints data
        op26_fldr = osp.join(base_dir, date, op26_fldr_, exp)
        if not osp.exists(op26_fldr):
            continue
        joints, ids = load_op26(op26_fldr)
        
        _, _, op26_files = next(os.walk(op26_fldr))
        op26_files = [op26_file for op26_file in op26_files if (op26_file[0] == 'b' and op26_file[-1] == 'n')]
        op26_files.sort()
        begin_idx = int(op26_files[0].split('.')[0].split('_')[-1])
        
        for sid in sids:
            # Extract current experiment label
            exp_name = '_'.join((date, exp, sid))
            curr_label = label[exp_name]
            
            if exp_name not in label.columns:
                continue
            
            if exp_name in singular_exps:
                curr_joints = joints[0]
            else:
                curr_joints = joints[sids.index(sid)]
            
            # Load IMU data
            imu_fldr = osp.join(base_dir, date, imu_fldr_, sid, exp)
            
            subject_id = 'S%02d'%curr_label.loc['Subject ID']
            for action in action_list:
                # Action not recorded, ignore the loop
                if pd.isna(curr_label.loc['_'.join((action, 'start'))]):
                    continue
                
                target_fldr = osp.join(base_dir, target_fldr_, subject_id, action)
                os.makedirs(osp.join(target_fldr, target_op26_fldr), exist_ok=True)
                os.makedirs(osp.join(target_fldr, target_imu_fldr), exist_ok=True)

                op26_start_idx = int(curr_label.loc['_'.join((action, 'start'))] - begin_idx)
                op26_end_idx = int(curr_label.loc['_'.join((action, 'end'))] - begin_idx)
                imu_start_idx = op26_start_idx * 5
                imu_end_idx = op26_end_idx * 5
                
                curr_action_joints = curr_joints[op26_start_idx:op26_end_idx].copy()
                curr_action_op26_files = op26_files[op26_start_idx:op26_end_idx]
                
                # Save OpenPose data
                for frame, op26_file in enumerate(tqdm(curr_action_op26_files, 
                                                  desc='Writing Keypoints...', leave=False)):
                    file_info = dict()
                    file_info["version"] = 0.7
                    file_info["univTime"] = -1.000
                    file_info["fpsType"] = "hd_29_97"
                    file_info["vgaVideoTime"] = 0.000
                    file_info["bodies"] = []
                    
                    ids = dict()
                    ids["id"] = subject_id
                    joints26 = []
                    for part in range(joints[0].shape[1]):
                        for axis in range(4):
                            joints26.append(curr_action_joints[frame, part, axis])
                    ids["joints26"] = joints26
                    file_info["bodies"].append(ids)

                    with open(osp.join(target_fldr, target_op26_fldr, op26_file), 'w') as make_file:
                        json.dump(file_info, make_file, indent=4)

                # Save MC10 data
                for part in part_list:
                    for sensor in ['accel', 'gyro']:
                        data = pd.read_csv(osp.join(imu_fldr, '%s_%s.csv'%(part, sensor)), 
                                           index_col='Timestamp (microseconds)')

                        curr_action_data = data.loc[data.index[imu_start_idx:imu_end_idx]]
                        curr_action_data.to_csv(osp.join(target_fldr, target_imu_fldr, '%s_%s.csv'%(part, sensor)),
                                                index_label='Timestamp (microseconds)')
