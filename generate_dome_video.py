from utils.load_data import load_op26
from utils.viz import draw_2d_skeleton, draw_3d_skeleton
from utils.viz_utils import project2D
from utils import constants as _C

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, sys
import os.path as osp

from tqdm import tqdm

__author__ = "Soyong Shin"


""" Generate skeletal video to see the motion
"""

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

if __name__ == '__main__':
    dates = _C.EXP_DATES
    exps = _C.EXP_SEQUENCES
    dates = ['190517']
    exps = ['exp12']
    base_dir = _C.BASE_RAW_DATA_DIR
    op26_fldr_ = _C.HD_KEYPOINTS_STAGE2_FLDR
    img_fldr = _C.IMAGE_FLDR

    for date in dates:
        for exp in exps:
            # Load OP26 data
            op26_fldr = osp.join(base_dir, date, op26_fldr_, exp)

            images_fldr = osp.join(base_dir, date, img_fldr, exp)
            video_fldr = osp.join(base_dir, date, img_fldr)
            # if osp.exists(osp.join(base_dir, date, img_fldr, '%s.mp4'%exp)):
            #     continue
            
            if not osp.exists(op26_fldr):
                continue
            
            print("Processing %s %s ..."%(date, exp))
            os.makedirs(images_fldr, exist_ok=True)
            
            joints, ids = load_op26(op26_fldr)
            _, _, op26_files = next(os.walk(op26_fldr))
            op26_files.sort()

            draw_3d_skeleton(joints[0][4464, :, :3])
            import pdb; pdb.set_trace()

            # Make projected 2D joints
            x2d_list = []
            for joints_, ids_ in zip(joints, ids):
                x2d, mask = project2D(joints_, imw, imh, camera_info['K'], camera_info['R'], camera_info['t'])
                x2d[~mask] = 0
                x2d_list += [x2d]

            # Draw and save videos
            background = np.ones((imh, imw, 3)).astype(np.uint8) * 255
            for frame_idx in tqdm(range(x2d_list[0].shape[0]), desc='Drawing skeleton...', leave=False):
                img = background.copy()
                for subj_idx, x2d_ in enumerate(x2d_list):
                    x = x2d_[frame_idx, :, 0].astype('int32')
                    y = x2d_[frame_idx, :, 1].astype('int32')
                    img = draw_2d_skeleton(x, y, img.copy(), subj_idx, "op26")

                img = cv2.putText(img, text="Frame " + op26_files[frame_idx][-10:-5], org=(200, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                fontScale=0.5, color=(30, 30, 30), thickness=2)
                img_n = "dome_video_%08d.jpg"%frame_idx
                cv2.imwrite(osp.join(images_fldr, img_n), img)

            os.system("rm -rf {}".format(video_fldr) + "/{}.mp4".format(exp))
            os.system("ffmpeg -framerate 29.97 -start_number 0 -i {}/dome_video_%08d.jpg".format(images_fldr) + " -vcodec mpeg4 {}".format(video_fldr) + "/{}.mp4".format(exp))
            os.system("rm -rf {}".format(images_fldr))
