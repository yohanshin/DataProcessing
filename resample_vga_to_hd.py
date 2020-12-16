from utils.load_data import load_op26

import numpy as np
import pandas as pd

import os, sys
import os.path as osp
import json

from tqdm import trange, tqdm

__author__ = "Soyong Shin"

""" Resample VGA keypoints to HD keypoints frame
    Unlike the processing code by Panoptic Studio, this even handles visibility between two adjacent frames.
"""

fps_vga = 25.0
fps_hd = 29.97

def resampling(joints_bf):
    """Resample frame-per-second to before (denoted bf) to after (denoted af)
    Input: joints_bf, np.array (frame, num_joints, 4)
    Output: joints_af, np.array (frame, num_joints, 4)
    """
    conf_mask = joints_bf[:, :, -1] > 0
    joints = joints_bf

    time_bf = np.array(range(joints_bf.shape[0])) * 1 / fps_vga
    time_af = np.array(range(int(time_bf[-1] * fps_hd))) * 1 / fps_hd
    time_af_ = [i for i in time_af if not (i in time_bf)]

    joints_af = np.zeros((time_af.shape[0], joints_bf.shape[1], 4))
    for J in range(joints.shape[1]):
        df_bf = pd.DataFrame(joints[:, J], index=time_bf, columns=['X', 'Y', 'Z', 'conf'])
        df_af = pd.DataFrame(index=time_af_, columns=['X', 'Y', 'Z', 'conf'])
        df_total = df_bf.append(df_af).sort_index(axis=0)
        df_total = df_total.interpolate(limit_area='inside', method='index')
        df_af = df_total.loc[time_af]
        joints_af[:, J] = np.array(df_af)

    mask_bf = pd.DataFrame(conf_mask.astype(np.int8), index=time_bf)
    mask_af = pd.DataFrame(np.zeros((len(time_af_), 26)).astype(np.int8), index=time_af_)
    mask_total = mask_bf.append(mask_af).sort_index(axis=0)
    mask_af = mask_total.loc[time_af]
    mask_af_np = np.array(mask_af).astype(np.bool)

    # If the joint is invisible for any of subsequent frames, set the confident zero
    for F in range(conf_mask.shape[0] - 1):
        both_visible = conf_mask[F] * conf_mask[F+1]
        is_btw = (mask_af.index.values > mask_bf.index[F]) * \
                 (mask_af.index.values < mask_bf.index[F+1])
        mask_af_np[is_btw] = both_visible.copy()

    joints_af[~mask_af_np] = np.array([0, 0, 0, -1])

    return joints_af


dates = ['190503', '190510', '190517', '190607']
exps = ['exp01', 'exp02', 'exp03', 'exp04', 'exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp11', 'exp12', 'exp13', 'exp14']
base_dir = 'dataset/MBL_DomeData/dome_data'
fldr_vga = 'vgaPose3d_stage1_op25'
fldr_hd = 'hdPose3d_stage1_op25'

for date in dates:
    for exp in exps:
        op26_fldr_vga = osp.join(base_dir, date, fldr_vga, exp)
        if not osp.exists(vga_op26_fldr):
            continue
        print("Processing %s %s ..."%(date, exp))
        joints_vga, ids = load_op26(op26_fldr_vga)

        # Resampling it to HD fps
        joints_hd = []
        for joint_vga in joints_vga:
            joint_hd = resampling(joint_vga)
            joints_hd += [joint_hd]

        # Write resampled data
        op26_fldr_hd = osp.join(base_dir, date, fldr_hd, exp)
        os.makedirs(op26_fldr_hd, exist_ok=True)
        
        for frame in trange(joint_hd.shape[0], desc='Writing HD data...', leave=False):
            file_info = dict()
            file_info["version"] = 0.7
            file_info["univTime"] = -1.000
            file_info["fpsType"] = "hd_29_97"
            file_info["vgaVideoTime"] = 0.000
            file_info["bodies"] = []

            for j in range(len(joints_hd)):
                ids = dict()
                ids["id"] = j
                joints26 = []
                for k in range(joints_hd[0].shape[1]):
                    for l in range(4):
                        joints26.append(joints_hd[j][frame][k][l])
                ids["joints26"] = joints26
                file_info["bodies"].append(ids)

            with open(osp.join(op26_fldr_hd, 'body3DScene_%08d.json'%frame), 'w') as make_file:
                json.dump(file_info, make_file, indent=4)