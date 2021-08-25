from utils import constants as _C

import torch
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm

import os, sys
import os.path as osp


# dates = _C.EXP_DATES
# exps = _C.EXP_SEQUENCES
dates = ['190503']
exps = ['exp01']
sids = _C.EXP_SUBJECTS
singular_exps = _C.EXP_SINGULAR

base_dir = _C.BASE_RAW_DATA_DIR
raw_video_fldr = _C.HD_RAW_VIDEO_FLDR
extract_fldr = _C.HD_EXTRACTED_VIDEO_FLDR
label_filename = _C.ACTION_LABEL_FILENAME
processed_fldr = _C.SEGMENTED_DATA_FLDR
op26_fldr = _C.SEGMENTED_KEYPOINTS_FLDR

label = pd.read_excel(osp.join(base_dir, label_filename), sheet_name='action labels', index_col=1)

for date in dates:
    for exp in exps:
        curr_video_fldr = osp.join(base_dir, date, raw_video_fldr, exp)

        # If video folder not exists (no sequence or not downloaded yet)
        if not osp.exists(curr_video_fldr):
            continue

        curr_sids = []
        for sid in sids:
            sub_exp_name = '_'.join((date, exp, sid))
            if sub_exp_name in label.columns:
                curr_sid = 'S%02d'%int(label[sub_exp_name]['Subject ID'])
                curr_sids += [curr_sid]

        _, _, jsons = next(os.walk(osp.join(base_dir, processed_fldr, curr_sids[0], 'whole_motion', op26_fldr)))
        jsons.sort()
        jsons = [int(json[12:-5]) for json in jsons if (json[0] == 'b' and json[-1] == 'n')]

        _, _, videos = next(os.walk(curr_video_fldr))
        videos = [_v[:-4] for _v in videos if _v[:2] == 'hd' and _v[-4:] == '.mp4']

        for video in videos:
            target_extract_fldr = osp.join(base_dir, date, extract_fldr, exp, video)
            # If already extracted
            if osp.exists(target_extract_fldr):
                continue
            else:
                os.makedirs(target_extract_fldr)

            video_file = osp.join(curr_video_fldr, '%s.mp4'%video)
            vidcap = cv2.VideoCapture(video_file)
            
            for frame in tqdm(range(max(jsons) + 1), desc='Extract %s-%s-%s.mp4'%(date, exp, video), leave=True):
                success, image = vidcap.read()
                if not success:
                    break
                
                # if frame in jsons:
                if frame == 7410:
                    extract_per_n_frame = 1
                    if frame % extract_per_n_frame == 0:
                        image_name = 'hdImage_%08d.jpg'%frame
                        image_out = osp.join(target_extract_fldr, image_name)
                        cv2.imwrite(image_out, image)

                    else:
                        pass