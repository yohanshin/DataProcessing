from utils.load_data import load_json
from utils import constants

import os, sys
import os.path as osp
import numpy as np
import json

from tqdm import tqdm

__author__ = "Soyong Shin"


def refine_ids(prev_ids, curr_ids, curr_joints):
    """Refine subject ids by comparing with previously detected subjects
    input:  previously detected subject ids (prev_ids)
            currently detected subject ids (curr_ids)
            currently detected subject joints (curr_joints)
    
    output: updated accumultaed ids (prev_ids)
            updated currently detected ids (curr_ids)
    """

    if len(prev_ids) == 0:                   # Previously none id has detected
        if curr_ids == []:               # If no subject is detected at th beginning of the sequence,
            return [], []               # then discard this frame
        elif len(curr_ids) == 1:         # If only one subject detected at the beginning of the sequence,
            if curr_joints[0][0][0] > 0:     # distinguish subject by their origin position
                return [1, 0], curr_ids
            else:
                return [0, 1], curr_ids
        else:                           # If two or more are detected at the first sequence,
            curr_ids = [i for i in curr_ids if i == 1 or i ==0]   # take the value only 0 or 1
            #curr_ids = [i for i in curr_ids if i == 2 or i ==0]   # Only for 190517 exp11
        if curr_joints[0][0][0] < curr_joints[1][0][0]: # First frame, two are detected but wrong order
            return [curr_ids[1], curr_ids[0]], curr_ids
        return curr_ids, curr_ids         # First frame, but no problem occured

    if len(curr_ids) > 2:                # More than two people are detected,
        return prev_ids, []                  # No newly 

    check = [i for i in curr_ids if ids.count(i) == 1]
    if len(check) == 0:
        return prev_ids, curr_ids             # If current detected ids are totally different from original ids, just return them

    for i in prev_ids:
        if curr_ids.count(i) == 0:
            only_prev = i               # If detected id is only in previous ids, not in current ids
    for i in curr_ids:
        if prev_ids.count(i) == 0:
            only_cur = i

    try:
        prev_ids[prev_ids.index(only_prev)] = only_cur
    except:
        pass

    return prev_ids, curr_ids


dates = ['190503']
exps = ['imu01']
base_dir = 'dataset/MBL_DomeData/dome_data'
curr_fldr = 'hdPose3d_stage1_op25'
target_fldr = 'hdPose3d_stage2_op25'

singular_exps = ['190510_imu08', '190531_imu12']

for date in dates:
    for exp in exps:
        exp_name = '_'.join((date, exp))
        curr_op26_fldr = osp.join(base_dir, exp_name, curr_fldr)
        _, _, op26_files = next(os.walk(curr_op26_fldr))
        op26_files.sort()
        ids, joints, last_id, past_joints = [[], [], [-1], [0, 0]]  # Initializing variables
        for op26_file in tqdm(op26_files, desc='Loading Data'):
            # Load current frame json file
            curr_ids, curr_joints = load_json(osp.join(curr_op26_fldr, op26_file))

            # Converting joint confidence -1 to 0
            for subj_idx in range(curr_joints.shape[0]):
                conf = curr_joints[subj_idx, :, -1].copy()
                curr_joints[subj_idx, :, -1][conf == -1] = 0.0

            if exp_name not in singular_exps:
                ids, new_ids = refine_ids(ids, curr_ids, curr_joints)     # Refine ids and joints
                if (len(last_id) > len(curr_ids)):   # if currently detected subject number is smaller than previous frame ==> disappeared
                    dis_id = [di for di in last_id if curr_ids.count(di) == 0]   # Find disappeared id
                    for di in dis_id:
                        past_joints[ids.index(di)] = joints[ids.index(di)][-1]  # Store disappeared id's joints of last frame

                if (len(last_id) == 0 and len(new_ids) > 0):    # If previously no subject is detected, but now newly detected ==> disappeared subject is now appeared
                    print("No one detected. Check the file for the consistency of subject id!")
                    distance = []
                    for ni in range(len(curr_ids)):
                        for pj in range(2):
                            try:
                                both_detected = [bd for bd in range(26) if past_joints[pj][bd][0] != 0 and curr_joints[ni][bd][0] != 0]
                                bd_joints = past_joints[pj][both_detected][:2] - curr_joints[ni].reshape(-1,26,4)[0][both_detected][:2]
                                distance.append(np.sum(bd_joints**2))
                            except:
                                import pdb; pdb.set_trace()

                    if len(distance) == 2:              # Only one new id is re-appeared at this frame
                        if distance[0] > distance[1]:
                            ids[1] = curr_ids[0]         # Change conventional id to the closer new id
                        else:
                            ids[0] = curr_ids[0]

                    if len(distance) == 4:              # If both subjects are re-appeared at the same frame, match the closest subjects to the previous frame
                        min_idx = distance.index(min(distance))
                        min_new = int(min_idx/2 - 0.1)
                        min_org = min_idx%2
                        ids[min_org] = new_ids[min_new]
                        ids[abs(1-min_org)] = new_ids[abs(1-min_new)]

                last_id = [li for li in curr_ids if ids.count(li) == 1]

                # TODO: stack zeros when no one is detected from the beginning
                for id in ids:
                    if len(joints) < 2:     # Before both subjects are detected at least one time
                        if ids == new_ids and len(new_ids) != 0:
                            joints.append(curr_joints[ids.index(id)].reshape(-1,26,4))
                        elif len(ids) == len(new_ids):
                            joints.append(curr_joints[new_ids.index(id)].reshape(-1,26,4))
                        elif len(new_ids) == 0:     # No one is detected --> just skip this frame
                            pass
                        else:
                            if new_ids.count(id) == 0:
                                joints.append(np.zeros([1, 26, 4]))
                            else:
                                joints.append(curr_joints[0].reshape(-1, 26, 4))

                    else:
                        if curr_ids.count(id) != 0:          
                            joints[ids.index(id)] = np.vstack([
                                    joints[ids.index(id)], curr_joints[curr_ids.index(id)].reshape(-1, 26, 4)])
                        else:
                            joints[ids.index(id)] = np.vstack([             # Zero padding if the subject is missed this frame
                                    joints[ids.index(id)], np.zeros([1, 26, 4])])

            else:       # From here is only for the experiment with one subject
                ids = [0]
                if cur_ids != []:
                    new_ids = [0]
                else:
                    new_ids = []

                if len(new_ids) != 0:
                    try:
                        joints[0] = np.vstack([
                            joints[0], cur_joints[0].reshape(-1, 26, 4)])
                    except:
                        joints.append(cur_joints[0].reshape(1, 26, 4))
                else:
                    try:
                        joints[0] = np.vstack([joints[0], np.zeros([1, 26, 4])])
                    except:
                        import pdb; pdb.set_trace()

        target_op26_fldr = osp.join(base_dir, '_'.join((date, exp)), target_fldr)
        os.makedirs(target_op26_fldr, exist_ok=True)

        for i, op26_file in enumerate(tqdm(op26_files, desc='Writing Data')):
            file_info = dict()
            file_info["version"] = 0.7
            file_info["univTime"] = -1.000
            file_info["fpsType"] = "hd_29_97"
            file_info["vgaVideoTime"] = 0.000
            file_info["bodies"] = []

            for j in range(len(joints)):
                ids = dict()
                ids["id"] = j
                joints26 = []
                for k in range(joints[0].shape[1]):
                    for l in range(4):
                        joints26.append(joints[j][i][k][l])
                ids["joints26"] = joints26
                file_info["bodies"].append(ids)

            with open(osp.join(target_op26_fldr, op26_file), 'w') as make_file:
                json.dump(file_info, make_file, indent=4)