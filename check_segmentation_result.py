from utils.load_data import load_op26
from utils.viz import *
from utils.viz_utils import project2D
from utils.sync_utils import *
from utils import constants as _C
from generate_dome_video import camera_info

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os, sys
import os.path as osp

import cv2
from tqdm import tqdm

__author__ = "Soyong Shin"


""" Double checking activity segmentation
"""

imw, imh = 540, 360

base_dir = _C.BASE_RAW_DATA_DIR
processed_fldr = _C.SEGMENTED_DATA_FLDR
processed_op26_fldr = _C.SEGMENTED_KEYPOINTS_FLDR
processed_imu_fldr = _C.SEGMENTED_IMU_FLDR
image_fldr = _C.IMAGE_FLDR
result_fldr = 'Processed_Result'

_, sids, _ = next(os.walk(osp.join(base_dir, processed_fldr)))
sids.sort()

for sid in tqdm(sids, desc='Subjects ...', leave=True):
    curr_result_fldr = osp.join(base_dir, result_fldr, sid)
    if osp.exists(osp.join(curr_result_fldr, 'whole_frame.mp4')):
        continue
    
    subject_fldr = osp.join(base_dir, processed_fldr, sid)
    _, actions, _ = next(os.walk(subject_fldr))

    action_videos_fldr = osp.join(curr_result_fldr, 'action_videos')
    
    whole_frame_idx = 0
    whole_img_out_fldr = osp.join(curr_result_fldr, image_fldr, 'whole_frame')
    os.makedirs(whole_img_out_fldr, exist_ok=True)

    for action in tqdm(actions, desc='Activities ...', leave=False):
        if action == 'whole_motion':
            continue
        
        curr_op26_fldr = osp.join(subject_fldr, action, processed_op26_fldr)
        curr_imu_fldr = osp.join(subject_fldr, action, processed_imu_fldr)

        img_out_fldr = osp.join(curr_result_fldr, image_fldr, action)
        if osp.exists(osp.join(action_videos_fldr, '%s.mp4'%action)):
            continue
        else:
            os.makedirs(img_out_fldr, exist_ok=True)

        # Load OP26
        joints, ids = load_op26(curr_op26_fldr)
        assert len(joints) == 1, "Check %s-%s"%(sid, action)

        _, _, op26_files = next(os.walk(curr_op26_fldr))
        op26_files.sort()
        assert len(op26_files) == len(joints[0])

        if len(op26_files) > 100:
            sampling_freq = int(len(op26_files)/100)
            joints = [joints[0][::sampling_freq]]
            op26_files = op26_files[::sampling_freq]

        # Make projected 2D joints
        x2d, mask = project2D(joints[0], imw, imh, camera_info['K'], camera_info['R'], camera_info['t'])
        x2d[~mask] = 0

        # Draw and save videos
        background = np.ones((imh, imw, 3)).astype(np.uint8) * 255
        for frame_idx in tqdm(range(x2d.shape[0]), desc='Drawing skeleton...', leave=False):
            img = background.copy()
            x = x2d[frame_idx, :, 0].astype('int32')
            y = x2d[frame_idx, :, 1].astype('int32')
            img = draw_2d_skeleton(x, y, img.copy(), 0, "op26")
            img = cv2.putText(img, text="{}".format(action), org=(200, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                fontScale=0.5, color=(30, 30, 30), thickness=2)

            img_out = "dome_video_%08d.jpg"%whole_frame_idx
            cv2.imwrite(osp.join(whole_img_out_fldr, img_out), img)

            whole_frame_idx += 1

        if action in ['front_hopping', 'end_hopping']:
            accel_kp_, _ = calculate_accel_from_keypoints(joints[0])
            accel_kp = pd.DataFrame(accel_kp_)[0]
            accel_kp.name = 'OP26'
            accel_kp = interpolate_kp_accel(accel_kp)
            accel_kp = np.array(accel_kp)

            accel_imu = pd.read_csv(osp.join(curr_imu_fldr, 'sacrum_accel.csv'), index_col=_C.INDEX_COL)
            accel_imu = accel_imu['Accel Y (g)']
            accel_imu = np.array(accel_imu)

            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(accel_imu, color='tab:red', label='IMU')
            ax.plot(accel_kp, color='tab:blue', label='KP')
            ax.legend()

            plt.savefig(osp.join(curr_result_fldr, '%s_syncing.png'%action))
            plt.close()
            import pdb; pdb.set_trace()

    os.system("ffmpeg -framerate 29.97 -start_number 0 -i {}/dome_video_%08d.jpg".format(whole_img_out_fldr) + \
                " -vcodec mpeg4 {}".format(curr_result_fldr) + "/whole_frame.mp4")

    # Delete images
    os.system("rm -rf {}".format(osp.join(curr_result_fldr, image_fldr)))