from utils.load_data import load_op26
from utils.viz import draw_2d_skeleton_for_configuration 
from utils.viz_utils import project2D
from utils import constants as _C

from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, sys
import os.path as osp

__author__ = "Soyong Shin"

"""Generate keypoints configuration figure to illustrate keypoints indexing"""

imw, imh = 960, 2080

# Sample camera view (take the view for frontal camera)
camera_info = {'R': np.array([
				[-0.9446503888,-0.03716808952,0.3259665259],
				[0.1752856194,0.7826876215,0.5972227715],
				[-0.2773275943,0.6213039677,-0.7328511344]
			    ]),
                't': np.array([
				[0.0],
				[0.0],
				[125.0]
			    ]),
                'K': np.array([
				[1396.34, 0, imw/2],
				[0, 1392.83, imh/2],
				[0, 0, 1]
			    ])}

base_dir = _C.BASE_RAW_DATA_DIR
processed_fldr = _C.SEGMENTED_DATA_FLDR
op26_fldr = _C.SEGMENTED_KEYPOINTS_FLDR
image_name = "keypoints_op26.png"
sid = 'S02'
action = 'static_motion'
frame_idx = 0 

# Load keypoints data
subject_fldr = osp.join(base_dir, processed_fldr, sid, action, op26_fldr)
keypoints, ids = load_op26(subject_fldr)
keypoints = keypoints[0][frame_idx][None]

# Centering the keypoints data
pelvis = keypoints[:, 2:3, :-1]
keypoints[:, :, :-1] -= pelvis

# Project 3D keypoints into image plane
intrinsics = camera_info['K']
translation = camera_info['t']
_r = R.from_rotvec([0, 0.8, 0])
pose = _r.as_matrix()
keypoints_2d, mask = project2D(keypoints, imw, imh, intrinsics, pose, translation)
x, y = np.split(keypoints_2d[0], 2, axis=-1)
x = x[:, 0].astype('int32'); y = y[:, 0].astype('int32')

# Hard coded joint location to set all joints can be distingushable (more clearly)
unit = 15
x[13] -= unit; x[[14, 22, 23]] -= 2*unit; x[7] += unit; x[[8, 19, 20, 21]] += 2*unit; x[16] += int(unit*2/3)
x[21] -= int(unit*3); y[21] -= int(unit/3) # Left foot is not distingushable.. hard coded

# Draw keypoints
background = np.zeros((imh, imw, 3)).astype(np.uint8)
img = draw_2d_skeleton_for_configuration(x, y, background.copy(), "op26", linewidth=10)
cv2.imwrite(osp.join(base_dir, processed_fldr, image_name), img)
