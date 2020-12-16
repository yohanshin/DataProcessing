from utils.load_data import load_op26
from utils import viz_utils as vu

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, sys
import os.path as osp

from tqdm import tqdm

imw, imh = 540, 360

# Sample camera view
camera_info = {'R': np.array([
				[-0.9446503888,-0.03716808952,0.3259665259],
				[0.1752856194,0.7826876215,0.5972227715],
				[-0.2773275943,0.6213039677,-0.7328511344]
			    ]),
                't': np.array([
				[0.0],
				[0.0],
				[2500.0]
			    ]),
                'K': np.array([
				[1396.34, 0, 270.0],
				[0, 1392.83, 180.0],
				[0, 0, 1]
			    ])}

def project2D(x3d, K, R, t, dist=None):
    """
    x3d : F * N * 4 numpy array
    K : 3 * 3 numpy array
    R : 3 * 3 numpy array
    t : 3 * 1 numpy array
    dist : 5 * 1 numpy array
    """

    conf_mask = x3d[:, :, -1] > 1e-5
    x3d = x3d[:, :, :-1]

    x2d = np.zeros_like(x3d[:, :, :2])
    R2_criterion = np.zeros_like(x3d[:, :, 0])

    for J in range(x3d.shape[1]):
        """ J is joint index """
        x = np.dot(R, x3d[:, J].T) + t
        xp = x[:2] / x[2]

        if dist is not None:
            X2 = xp[0] * xp[0]
            Y2 = xp[1] * xp[1]
            XY = X2 * Y2
            R2 = X2 + Y2
            R4 = R2 * R2
            R6 = R4 * R2
            R2_criterion[:, J] = R2

            radial = 1.0 + dist[0] * R2 + dist[1] * R4 + dist[4] * R6
            tan_x = 2.0 * dist[2] * XY + dist[3] * (R2 + 2.0 * X2)
            tan_y = 2.0 * dist[3] * XY + dist[2] * (R2 + 2.0 * Y2)

            xp[0, :] = radial * xp[0, :] + tan_x
            xp[1, :] = radial * xp[1, :] + tan_y

        pt = np.dot(K[:2, :2], xp) + K[:2, 2:]
        x2d[:, J, :] = pt.T

    x2d = x2d.astype('int32')
    x_visible = np.logical_and(x2d[:, :, 0] >= 0, x2d[:, :, 0] < imw)
    y_visible = np.logical_and(x2d[:, :, 1] >= 0, x2d[:, :, 1] < imh)
    visible = np.logical_and(x_visible, y_visible)
    vis_mask = np.logical_and(visible, R2_criterion < 1.)
    mask = np.logical_and(conf_mask, vis_mask)

    return x2d, mask


# Define experiments
dates = ['190503', '190510', '190517', '190607']
exps = ['exp01', 'exp02', 'exp03', 'exp04', 'exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp11', 'exp12', 'exp13', 'exp14']
base_dir = 'dataset/MBL_DomeData/dome_data'
op26_fldr_ = 'hdPose3d_stage1_op25'
img_fldr = 'skeleton_video'

for date in dates:
    for exp in exps:
        # Load OP26 data
        op26_fldr = osp.join(base_dir, date, op26_fldr_, exp)

        # If video is already processed, skip the loop
        target_output_fldr = osp.join(base_dir, date, img_fldr, exp)
        if osp.exists(osp.join(target_output_fldr, 'out.mp4')):
            continue
        
        os.makedirs(target_output_fldr, exist_ok=True)
        
        # If no experiments exists, skip the loop
        if not osp.exists(op26_fldr):
            continue
        print("Processing %s %s ..."%(date, exp))
        
        joints, ids = load_op26(op26_fldr)
        _, _, op26_files = next(os.walk(op26_fldr))
        op26_files.sort()

        # Make projected 2D joints
        x2d_list = []
        for joints_, ids_ in zip(joints, ids):
            x2d, mask = project2D(joints_, camera_info['K'], camera_info['R'], camera_info['t'])
            x2d[~mask] = 0
            x2d_list += [x2d]

        # Draw and save videos
        background = np.ones((imh, imw, 3)).astype(np.uint8) * 255
        for frame_idx in tqdm(range(x2d_list[0].shape[0]), desc='Drawing skeleton...', leave=False):
            img = background.copy()
            for subj_idx, x2d_ in enumerate(x2d_list):
                x = x2d_[frame_idx, :, 0].astype('int32')
                y = x2d_[frame_idx, :, 1].astype('int32')
                img = vu.draw_2d_skeleton(x, y, img.copy(), subj_idx, "op26", frame_idx)

            img = cv2.putText(img, text="Frame " + op26_files[frame_idx][-10:-5], org=(200, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.5, color=(30, 30, 30), thickness=2)
            img_n = "dome_video_%08d.jpg"%frame_idx
            cv2.imwrite(osp.join(target_output_fldr, img_n), img)

        os.system("ffmpeg -framerate 29.97 -start_number 0 -i {}/dome_video_%08d.jpg".format(target_output_fldr) + " -vcodec mpeg4 {}/out.mp4".format(target_output_fldr))
        os.system("echo y")