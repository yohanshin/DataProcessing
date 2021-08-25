import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import os.path as osp
from copy import copy


__author__ = "Soyong Shin"


fps_op26 = 29.97
fps_imu = 149.85

def refine_params(params):
    """
    Some experiments were conducted with only one subject
    Refining parameters (Experiments & Subject ID) after the experiments
    Subject ID  --> for loading keypoints
    Experiments --> for loading IMU
    """

    kp_params, imu_params = copy(params), copy(params)
    date, exp, id = params['date'], params['exp'], params['id']
    if date == '190517':
        if exp == 12:
            id = 100 if id == 1 else 1
    elif date == '190607':
        if exp == 11:
            id = 100 if id == 2 else 1
        if exp > 11 and id == 2:
            exp = exp - 1

    kp_params['id'] = id
    imu_params['exp'] = exp

    return kp_params, imu_params


def get_sacrum_and_kp_path(params):

    base_path_imu = params['IMU_dir']
    path_imu_anno = osp.join(base_path_imu, 'Set%02d'%params['id'], 'Set%02d_annotations.csv'%params['id'])
    path_imu = osp.join(base_path_imu, 'Set%02d'%params['id'], params['date'], 'sacrum', 'accel.csv')

    base_path_kp = params['base_dir']
    path_kp = osp.join(base_path_kp, params['date'], params['op26_dir'], 'exp%02d'%params['exp'])

    return path_kp, path_imu_anno, path_imu


def get_imu_fldr(params, part):

    current_fldr = params['IMU_dir']
    current_fldr = osp.join(current_fldr, 'Set%02d'%params['id'], params['date'], part)

    target_fldr = params['synced_IMU_dir']
    target_fldr = osp.join(target_fldr, 'Set%02d'%params['id'], params['date'], part)
    if not osp.exists(target_fldr):
        os.makedirs(target_fldr)

    return current_fldr, target_fldr


def interpolate_kp_accel(accel_op26):
    """ OP26 has fps of 25 (VGA) or 29.97 (HD)
    This function resamples OP26 acceleration into fps of IMU
    """

    # Generate new index by FPS
    time_op26 = np.arange(accel_op26.shape[0]) / fps_op26
    time_res = np.arange(int((accel_op26.shape[0]-1) * fps_imu / fps_op26)) / fps_imu
    time_res_ = [i for i in time_res if not (i in time_op26)]
    
    idx_op26 = pd.Index(time_op26, name='OP26')
    idx_res = pd.Index(time_res, name='OP26')
    idx_res_ = pd.Index(time_res_, name='OP26')     # IDX without overlapping

    # Interpolate accel and resample by FPS
    accel_op26.index = pd.Index(idx_op26)
    accel_res_ = pd.Series(index=idx_res_)
    accel_total = accel_op26.append(accel_res_).sort_index(axis=0)
    accel_total = accel_total.interpolate(limit_area='inside', method='index')
    accel_res = accel_total.loc[idx_res]

    return accel_res


def resample_IMU_data(accel):
    """
    MC10 IMU sensor has variational (although very small) frame-per-second (FPS).
    Assuming that camera has a constant FPS, resample MC10 IMU data to a constant FPS.
    """

    # To fine-tune indexing interpolation
    accel.index = (accel.index)

    # Generate empty data with constant FPS
    trg_period = 1e6 / fps_imu
    trg_num_data = int((accel.index[-1] - accel.index[0])/trg_period)
    res_index_ = [accel.index[0] + trg_period*(i+1) for i in range(trg_num_data)]
    
    res_index_ = pd.Index(res_index_).astype('int64')
    res_index = delete_overlap_index(accel.index, res_index_)
    empty_data = pd.Series(index=res_index)

    # INterpolate at constant data FPS points
    accel = accel.append(empty_data, verify_integrity=True).sort_index(axis=0)
    accel = accel.interpolate(limit_area='inside', method='index')

    # Extract data with resampled index
    res_accel = accel[res_index_]
    res_accel.index = (res_accel.index).astype('int64')
    
    return res_accel


def add_dummy_IMU_data(accel, num_dummy):
    """
    When the start of IMU data collection is later than OpenPose record,
    add dummy IMU data at the beginning so that syncing can be processed.
    """

    # 
    trg_period = 1e6 / fps_imu
    new_index_ = [accel.index[0] - trg_period*(i+1) for i in range(num_dummy)]
    new_index = pd.Index(new_index_).astype('int64')
    empty_data = pd.Series(index=new_index)

    accel = accel.append(empty_data, verify_integrity=True).sort_index(axis=0)
    
    return accel


def generate_peaks(data, thresh=2):
    """
    Generate the local peak points (local maximum) larger than threshold
    Data should be instance of pandas.DataFrame
    """

    is_peak = (data.shift(-1) < data) & (data.shift(1) < data) & (data > thresh)
    peak_idx = pd.DataFrame(np.where(np.array(is_peak)==True)[0])[0]
    return peak_idx


def refine_peaks_one_step(accel, index):
    """
    From detected peak points, drop some of them are not seemed to be a hopping sequence.
    1) Drop isolated peak points (Because hopping happens three to four times in few seconds)
    2) If two peaks are very close, choose the bigger peaks.
    3) Drop the mid-peaks induced by other activities
    """

    max_gap, min_gap = fps_imu, fps_imu/4

    if len(index) > 20:
        index = index.drop(index.index[10:-10])
    isolated = ((index - index.shift(1)) > max_gap) & ((index.shift(-1) - index) > max_gap)
    non_isolated = isolated == False
    index = index[non_isolated]

    bfl = index.index
    output = copy(index)
    for i in range(index.shape[0]):
        if i == 0 and index[bfl[1]] - index[bfl[0]] > max_gap:
            output = output.drop(bfl[i])
        elif i == bfl.shape[0]-1 and (index[bfl[i]] - index[bfl[i-1]]) > max_gap:
            output = output.drop(bfl[i])
        elif i > 0 and (index[bfl[i]] - index[bfl[i-1]]) < min_gap:
            if accel[accel.index[index[bfl[i]]]] > accel[accel.index[index[bfl[i-1]]]]:
                try:
                    output = output.drop(bfl[i-1])
                except:
                    continue
            else:
                output = output.drop(bfl[i])
    return output


def refine_peaks(imu_accel, kp_accel, imu_peak, kp_peak):

    imu_peak_, kp_peak_ = [], []

    while len(imu_peak_) != len(imu_peak):
        if len(imu_peak_) != 0:
            imu_peak = imu_peak_
        imu_peak_ = refine_peaks_one_step(imu_accel, imu_peak)

    while len(kp_peak_) != len(kp_peak):
        if len(kp_peak_) != 0:
            kp_peak = kp_peak_
        kp_peak_ = refine_peaks_one_step(kp_accel, kp_peak)

    return imu_peak, kp_peak


def delete_overlap_index(d_index, s_index):
    """
    Check and delete overlapped index which exists in both raw and synced data
    """

    d_df = pd.DataFrame(index=d_index)
    s_df = pd.DataFrame(index=s_index)
    i_df = d_df.append(s_df).sort_index(axis=0)
    duplicated_index = i_df.index[i_df.index.duplicated(keep='first')]
    s_df = s_df.drop(duplicated_index)
    s_index = s_df.index
    
    return s_index


def generate_imu_front_buffer(accel, diff):
    
    sampling_period = 1e6 / fps_imu
    new_index = np.arange(diff, 0) * sampling_period + accel.index[0]
    new_index = pd.Index(new_index)
    empty_data = pd.Series(index=new_index)
    accel = accel.append(empty_data).sort_index(axis=0)
    accel = np.array(accel)

    return accel


def refine_accel_with_zero_conf(accel, conf, target_value=1):
    """
    Change the acceleration value where sacrum is not detected to target value (1)
    """

    is_not_zero = conf > 0
    check = (is_not_zero[:-2] * is_not_zero[1:-1] * is_not_zero[2:]) == 0
    accel[check] = target_value

    return accel


def calculate_accel_from_keypoints(keypoints, idx=2):
    """
    Calculate acceleration from the variation of the part (sacrum) Y value
    """

    height = keypoints[:, idx, 1]
    accel = (-1) * (height[:-2] + height[2:] - 2 * height[1:-1])
    accel = accel * fps_op26 * fps_op26 / 100 / 9.81 + 1

    conf = keypoints[:, -2, -1]
    accel = refine_accel_with_zero_conf(accel, conf)

    accel_ = np.ones(accel.shape[0] + 2)
    accel_[1:-1] = accel

    return accel_, height
