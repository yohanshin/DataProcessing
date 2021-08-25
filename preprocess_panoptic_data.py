from utils.load_data import load_camera_calib, load_json, load_op26
from utils import constants as _C

import cv2
import glob

import numpy as np
import pandas as pd
import json

import os, sys
import os.path as osp

from tqdm import tqdm


dates = _C.EXP_DATES
exps = _C.EXP_SEQUENCES
sids = _C.EXP_SUBJECTS
part_list = _C.IMU_PARTS

base_dir = _C.BASE_RAW_DATA_DIR
processed_fldr = _C.SEGMENTED_DATA_FLDR
op26_fldr = _C.SEGMENTED_KEYPOINTS_FLDR
imu_fldr = _C.SEGMENTED_IMU_FLDR
calib_filename = _C.CAMERA_CALIB_FILENAME
label_filename = _C.ACTION_LABEL_FILENAME

out_file = osp.join(base_dir, 'label.npy')

num_cameras = 31
num_keypoints = 26
max_num_subj = 2

output = {}
output['cameras'] = np.empty(
    (len(dates), num_cameras),
    dtype=[
        ('R', np.float32, (3,3)),
        ('t', np.float32, (3,1)),
        ('K', np.float32, (3,3)),
        ('dist', np.float32, 5)
    ]
)

output['table'] = []
table_dtype = np.dtype([
    ('subject_ids', np.int8, max_num_subj),
    ('date_idx', np.int8),
    ('exp_idx', np.int8),
    ('frames', np.int16),
    ('keypoints', np.float32, (max_num_subj, num_keypoints, 4))])

# TODO: Add segmentation mask or OpenPose result

label = pd.read_excel(osp.join(base_dir, label_filename), sheet_name='action labels', index_col=1)

for date in dates:
    calib_file = osp.join(base_dir, date, calib_filename)
    camera_infos = load_camera_calib(calib_file)
    for camera_idx, (camera_name, camera_info) in enumerate(camera_infos.items()):
        camera_retval = output['cameras'][dates.index(date)][camera_idx]
        camera_retval['R'] = camera_info['camera_pose']
        camera_retval['t'] = camera_info['camera_transl']
        camera_retval['K'] = camera_info['camera_intrinsics']
        camera_retval['dist'] = camera_info['camera_dist']
    
    for exp in exps:    
        exp_name = '_'.join((date, exp))
        curr_sids = []
        for sid in sids:
            sub_exp_name = '_'.join((exp_name, sid))
            if sub_exp_name in label.columns:
                curr_sids += [sid]

        if len(curr_sids) == 0:
            continue
            
        keypoints3d, sub_ids = [], []
        for sid in curr_sids:
            sub_exp_name = '_'.join((exp_name, sid))
            sub_id = 'S%02d'%int(label[sub_exp_name]['Subject ID'])
            sub_fldr = osp.join(base_dir, processed_fldr, sub_id, 'whole_motion')
            sub_op26_fldr = osp.join(sub_fldr, op26_fldr)
            joints, _ = load_op26(sub_op26_fldr)
            keypoints3d += [joints[0][:, None]]
            sub_ids += [int(sub_id[1:])]
        
        for i in range(1, len(curr_sids)):
            assert keypoints3d[0].shape[0] == keypoints3d[i].shape[0]
        
        if len(curr_sids) == 1:
            empty_keypoints = np.zeros_like(joints[0]).astype(joints[0].dtype)
            empty_keypoints[:, :, -1] = -1
            keypoints3d += [empty_keypoints[:, None]]
        
        keypoints3d = np.concatenate(keypoints3d, axis=1)
        if len(sub_ids) == 1:
            sub_ids += [-1]
        sub_ids = np.array(sub_ids)
        date_idx = dates.index(date)
        exp_idx = exps.index(exp)

        _, _, jsons = next(os.walk(osp.join(base_dir, processed_fldr, sub_id, 'whole_motion', op26_fldr)))
        jsons.sort()
        frames = [int(json[12:-5]) for json in jsons if (json[0] == 'b' and json[-1] == 'n')]

        table_segment = np.empty(len(frames), dtype=table_dtype)
        table_segment['subject_ids'] = sub_ids
        table_segment['date_idx'] = date_idx
        table_segment['exp_idx'] = exp_idx
        table_segment['frames'] = frames
        table_segment['keypoints'] = keypoints3d
        output['table'].append(table_segment)

output['table'] = np.concatenate(output['table'])
assert output['table'].ndim == 1

print("Total frames in CMU Panoptic IMU data:", len(output['table']))
np.save(out_file, output)