from utils.sync_utils import *
from utils.load_data import *
from utils.viz import plot_and_save_analysis_fig, plot_sliding_with_index

import numpy as np
import pandas as pd

import os, sys
import os.path as osp

from tqdm import trange, tqdm

__author__ = "Soyong Shin"


""" Sync IMU data with given keypoints data
"""

fps_vga = 25.0
fps_hd = 29.97
fps_imu = 125.0


def calculate_RMSE(imu, kp):
    return np.sqrt(((imu - kp)**2).mean())


def load_OP26_Y_accel(params, path_kp):
    # keypoints, ids = load_keypoints_single_exp(path_kp, num_file=2000)
    keypoints, ids = load_op26(path_kp)
    keypoints = keypoints[params['id'] - 1]

    # Calculate acceleration from the variation of sacrum Y value
    accel_, sacrum_height = calculate_accel_from_keypoints(keypoints)
    accel = pd.DataFrame(accel_)[0]
    accel.name = 'OP26'
    accel = interpolate_kp_accel(accel)

    # Get peak clue where the hopping occurs
    begin_hop_idx = np.where(sacrum_height == sacrum_height[:1200].min())[0][0]
    begin_hop_idx = int((fps_imu/fps_hd) * (begin_hop_idx - 1))
    end_hop_idx = np.where(sacrum_height == sacrum_height[-3000:].min())[0][0]
    end_hop_idx = int((fps_imu/fps_hd) * (end_hop_idx - 1))

    return accel, begin_hop_idx, end_hop_idx


def load_IMU_Y_accel(params, path_imu, path_imu_anno):
    accel = pd.read_csv(path_imu, index_col = 'Timestamp (microseconds)')
    accel = accel['Accel Y (g)']
    
    # Load annotation file and get data collection time
    anno_start, anno_end = load_imu_annotation(path_imu_anno)
    
    # Get data collection activities within the date
    i_start_date = np.where(np.array(anno_start > accel.index[0]/1000)==True)[0][0]
    i_end_date = np.where(np.array(anno_end < accel.index[-1]/1000)==True)[0][-1]
    t_start_date = np.array(anno_start)[i_start_date:i_end_date+1]
    t_end_date = np.array(anno_end)[i_start_date:i_end_date+1]

    # Get current experiment data collection activity
    t_start_, t_end_ = t_start_date[params['exp']-1], t_end_date[params['exp']-1]
    idx_start = np.where(accel.index > 1000*t_start_)[0][0]
    idx_end = np.where(accel.index < 1000*t_end_)[0][-1] + 3000     # buffer: 1000
    
    # Temporal segmentation for current experiment
    accel = accel.loc[accel.index[idx_start : idx_end]]
    
    # Drop invalid data (usually data point with accel = -4 or -8)
    invalid_data = (accel == -4) | (accel == -8)
    invalid_index = np.where(np.array(invalid_data) == True)[0]
    invalid_index_ = invalid_data.index[invalid_index]
    accel = accel.drop(invalid_index_)

    # Resample the sensor data with constant fps
    accel = resample_IMU_data(accel)
    
    accel.name = 'IMU'
    
    return accel


def get_syncing_index(params):
    """Return syncing index that matches IMU data to OP data"""
    
    print("Syncing data Info: Date (%s), Exp (%d), Subject (%d)"
          %(params['date'], params['exp'], params['id']))

    kp_params, imu_params = refine_params(params)
    
    # Determine the path of data and load acceleration
    path_kp, path_imu_anno, path_imu = get_sacrum_and_kp_path(params)
    imu_accel = load_IMU_Y_accel(imu_params, path_imu, path_imu_anno)
    kp_accel, begin_hop, end_hop = load_OP26_Y_accel(kp_params, path_kp)
    
    # Get peaks of each acceleration
    imu_peak = generate_peaks(imu_accel)
    kp_peak = generate_peaks(kp_accel)
    
    # Based on sacrum maximum height at the both ends, generate peaks
    kp_hop = (abs(begin_hop - kp_peak) < 500) | (abs(end_hop - kp_peak) < 500)
    kp_peak = kp_peak[pd.DataFrame(np.where(np.array(kp_hop)==True)[0])[0]]
    
    # Refine IMU and keypoints peaks while the refining does not impact the result
    imu_peak, kp_peak = refine_peaks(imu_accel, kp_accel, imu_peak, kp_peak)

    # Classify front and end (two hopping activities) of keypoints data
    is_front_kp_peak = kp_peak < 5000
    is_front_kp_peak = pd.DataFrame(np.where(np.array(is_front_kp_peak)==True)[0])[0]
    is_end_kp_peak = kp_peak > kp_accel.shape[0] - 5000
    is_end_kp_peak = pd.DataFrame(np.where(np.array(is_end_kp_peak)==True)[0])[0]
    front_kp_peak = kp_peak[kp_peak.index[is_front_kp_peak]]
    end_kp_peak = kp_peak[kp_peak.index[is_end_kp_peak]]
    
    buffer, check_range = 300, 500
    pad = buffer+int(check_range/2)
    
    if front_kp_peak[front_kp_peak.index[0]] < pad:
        buffer, check_range = front_kp_peak[front_kp_peak.index[0]], 200
        pad = buffer+int(check_range/2)

    # Get small window of keypoints accel
    front_kp_idx = [front_kp_peak[front_kp_peak.index[i]] for i in [0, -1]]
    front_kp_idx[-1] = max(front_kp_idx[0] + 300, front_kp_idx[-1])
    front_kp_accel = np.array(kp_accel)[front_kp_idx[0]-buffer:front_kp_idx[1]+buffer]
    
    # Get small window of IMU accel
    front_imu_accel = np.array(imu_accel)[imu_peak[imu_peak.index[0]]-pad:]
    front_imu_accel = front_imu_accel[:front_kp_accel.shape[0] + check_range]
    
    # Slide the front window of KP accel and calculate corresponding RMSEs
    rmse_list = []
    for i in range(check_range):
        start_idx, end_idx = i, i - check_range
        rmse_list += [calculate_RMSE(front_imu_accel[start_idx:end_idx], front_kp_accel)]

    # Calculate frame difference between two accel values
    rmse_result = np.array(rmse_list)
    minimum_idx = np.where(rmse_result == rmse_result.min())[0][0]
    base_diff = imu_peak[imu_peak.index[0]] - front_kp_idx[0]
    diff = base_diff - int(check_range/2) + minimum_idx

    # Check one more time whether synced IMU is correct
    if diff >= 0:
        synced_imu_accel = np.array(imu_accel)[diff:]
    else:
        synced_imu_accel = generate_imu_front_buffer(imu_accel, diff)
    
    if kp_accel.shape[0] > synced_imu_accel.shape[0]:
        sz_diff = kp_accel.shape[0] - synced_imu_accel.shape[0]
        kp_accel = kp_accel[:kp_accel.index[-sz_diff]]
        print("OpenPose data is longer than IMU data. Need to remove last %d frames of OpenPose data..."\
            %int(sz_diff/(fps_imu/fps_hd) + 1))
    else:
        synced_imu_accel = synced_imu_accel[:kp_accel.shape[0]]

    front_synced_imu_accel = synced_imu_accel[front_kp_idx[0]-buffer:front_kp_idx[1]+buffer]
    start_idx, end_idx = minimum_idx, minimum_idx - check_range
    check_ = calculate_RMSE(front_synced_imu_accel, front_kp_accel)
    assert rmse_result.min() == check_
    
    # Print the syncing result and see the result is reasonable
    if params['save_image']:
        if end_kp_peak.shape[0] == 0:
            end_kp_accel = np.array(kp_accel)[-3000:-12000]
            end_synced_imu_accel = synced_imu_accel[-3000:-1200]
        
        else:
            end_idx_ = end_kp_peak.max() + buffer if kp_accel.shape[0] - end_kp_peak.max() > buffer else -1
            start_idx_ = max(end_kp_peak.min(), end_idx_ - 1200)
            end_kp_accel = np.array(kp_accel)[start_idx_:end_idx_]
            end_synced_imu_accel = synced_imu_accel[start_idx_:end_idx_]
        
        plot_and_save_analysis_fig(params, front_synced_imu_accel, front_kp_accel,
                                   end_synced_imu_accel, end_kp_accel)

    synced_imu_index = imu_accel.index[diff:]
    synced_imu_index = synced_imu_index[:kp_accel.shape[0]]
    
    return synced_imu_index


def sync_all_sensors(params, synced_imu_index, fldr_target):
    """Given syncing index, sync IMU sensors and save files"""

    # 15 sensors list
    sensor_list = ["chest", "head", "lbicep", "lfoot", "lforearm", "lhand", "lshank", "lthigh",
                   "rbicep", "rfoot", "rforearm", "rhand", "rshank", "rthigh", "sacrum"]
    
    for part in sensor_list:
        fldr_raw, _ = get_imu_fldr(params, part)
        
        for sensor in ['accel', 'gyro']:
            # Read data
            data = pd.read_csv(os.path.join(fldr_raw, sensor+'.csv'), 
                               index_col = 'Timestamp (microseconds)')
            
            # Interpolate and get synced data
            synced_imu_index_ = delete_overlap_index(data.index, synced_imu_index)
            empty_data = pd.DataFrame(index=synced_imu_index_, columns=data.columns)
            data = data.append(empty_data).sort_index(axis=0)
            data = data.interpolate(limit_area='inside')
            data = data.loc[synced_imu_index]

            # Save files
            data.to_csv(os.path.join(fldr_target, '%s_%s.csv'%(part, sensor)))


dates = ['190503', '190510', '190517', '190607']
exps = ['exp01', 'exp02', 'exp03', 'exp04', 'exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp11', 'exp12', 'exp13', 'exp14']
ids = [1, 2]
base_dir = 'dataset/MBL_DomeData/dome_data'
fldr_op26 = 'hdPose3d_stage1_op25'
fldr_raw_imu = 'dataset/dome_IMU'
fldr_target = 'mc10_IMU'

params = dict()
params['IMU_dir'] = fldr_raw_imu
params['base_dir'] = base_dir
params['op26_dir'] = fldr_op26
params['save_video'] = False
params['save_image'] = True

for date in dates:
    for exp in exps:
        fldr_kp = osp.join(base_dir, date, fldr_op26, exp)
        fldr_target = osp.join(base_dir, date, fldr_target, exp)
        
        # If no experiments exists, skip the loop
        if not osp.exists(fldr_kp):
            continue

        for _id in ids:
            params['date'] = date
            params['exp'] = int(exp[-2:])
            params['id'] = _id

            # If only one subject exists in OP26, skip the loop
            # TODO: implement one subject syncing
            kp, _ = refine_params(params)
            if kp['id'] == 100:
                continue

            # If syncing already processed, skip the loop
            result_fldr = osp.join(params['base_dir'], 'Syncing_Result', params['date'])
            result_file = osp.join(result_fldr, 'Set%02d_exp%02d.png'%(params['id'], params['exp']))
            if osp.exists(result_file):
                continue

            synced_imu_index = get_syncing_index(params)
            # sync_all_sensors(params, synced_imu_index, fldr_target)