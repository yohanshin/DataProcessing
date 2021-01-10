import numpy as np
import pandas as pd
import os, sys
import csv, json

from tqdm import tqdm

__author__ = "Soyong Shin"

def load_imu_annotation(filename):
    """Load IMU annotation file
    Input : File name
    Output : Start and End time of each experiment data collection
    """
    
    annotation = pd.read_csv(filename, index_col='AnnotationId')
    t_start = annotation['Start Timestamp (ms)']['Activity:Data Collection']
    t_end = annotation['Stop Timestamp (ms)']['Activity:Data Collection']
    
    return t_start, t_end


def load_imu(params):
    
    path = '/home/soyongs/dataset/syncing_result/'
    path = join(path, 'Set%02d'%params['id'], params['date'])
    sensor_list = ["chest", "head", "lbicep", "lfoot", "lforearm", "lhand", "lshank", "lthigh",
                   "rbicep", "rfoot", "rforearm", "rhand", "rshank", "rthigh", "sacrum"]
    
    for sensor in sensor_list:
        tmp_accel = pd.read_csv(join(path, sensor, 'accel%02d.csv'%params['exp']))
        tmp_gyro = pd.read_csv(join(path, sensor, 'gyro%02d.csv'%params['exp']))
        tmp_accel = np.array(tmp_accel)[:, None, 1:]
        tmp_gyro = np.array(tmp_gyro)[:, None, 1:]
        
        if sensor == 'chest':
            accel, gyro = np.array(tmp_accel), np.array(tmp_gyro)
        else:
            accel = np.concatenate((accel, tmp_accel), axis=1)
            gyro = np.concatenate((gyro, tmp_gyro), axis=1)

    return accel, gyro


def load_json(file_name):
    """Load keypoints from json file
    Input : File name
    Output : index, 26 joints location
    """
    ids, joints = [[], []]
    with open(file_name) as json_file:
        data = json.load(json_file)
        for body in data["bodies"]:
            ids.append(body['id'])
            joints.append(np.array(body['joints26']).reshape(1,-1,4))
    try: 
        joints = np.vstack(joints)
    except:
        pass

    return ids, joints


def load_op26(path, num_file=-1):
    """Load 3D openpose joints after fixing
    Input : Experiment path
            num_file is not -1 only if trying to load fewer data
    Output : index, 26 joints location for entire frames
    """
    _, _, files_ = next(os.walk(path))
    files_ = [file_ for file_ in files_ if (file_[0] == 'b' and file_[-1] == 'n')]
    files_.sort()
    joints = []
    n_file = len(files_) if num_file == -1 else num_file
    with tqdm(total=n_file, desc='Loading OP26 data...', leave=False) as prog_bar:
        for file_ in files_[:n_file]:
            ids, cur_joints = load_json(os.path.join(path, file_))
            for id in ids:
                if len(joints) == 0:
                    joints.append(cur_joints[0][None])
                elif len(joints) == 1 and len(cur_joints) == 2:
                    joints.append(cur_joints[1][None])
                else:
                    joints[id] = np.vstack([joints[id], cur_joints[id][None]])
            prog_bar.update(1)
            prog_bar.refresh()
    
    ids = [i for i in range(len(joints))]
    
    # Some experiments record openpose first
    if path.split('/')[-2] == '190607' and path[-2:] == '13':
        joints[0] = joints[0][150:]
        joints[1] = joints[1][150:]
    
    return joints, ids