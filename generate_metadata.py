from utils import constants as _C

import numpy as np
import pandas as pd
import json

import os
import os.path as osp
from tqdm import tqdm

import pdb
from pdb import set_trace as st

__author__ = "Soyong Shin"


base_dir = _C.BASE_RAW_DATA_DIR
processed_fldr = _C.SEGMENTED_DATA_FLDR
action_label_file = _C.ACTION_LABEL_FILENAME
imu_fldr = _C.RAW_IMU_DIR

action_label = pd.read_excel(osp.join(base_dir, action_label_file), sheet_name='action labels', index_col=1)
_, sids, _ = next(os.walk(osp.join(base_dir, processed_fldr)))
sids.sort()
lb_to_kg_list = ['190503_exp08_Set02', '190503_exp09_Set02', '190517_exp03_Set02', '190517_exp09_Set02', 
                 '190517_exp11_Set02', '190517_exp12_Set02', '190607_exp04_Set01', '190607_exp09_Set02']

for sid in sids:
    subject_id = float(sid[-2:])
    exp_info = action_label.loc['Subject ID'].index[action_label.loc['Subject ID'] == subject_id][0]
    date, exp, set = exp_info.split('_')
   
    annotation = pd.read_csv(osp.join(imu_fldr, set, '%s_annotations.csv'%set), index_col='EventType')
    info_type = ['age', 'sex', 'height', 'weight']
    info_dict = dict()
    json_file = dict()
    org_info_dict = dict()
    for info in info_type:
        question = 'What is your %s?'%info
        value_d_e = annotation[['Value', 'Date', 'Exp']].loc[question]
        value_d = value_d_e.loc[value_d_e['Exp'] == int(exp[-2:])]
        value = value_d.loc[value_d['Date'] == int(date)]
        value = value['Value'][0]
        org_info_dict[info] = value
        json_file[info] = dict()

        if info == 'age':
            value = int(value)
            json_file[info]['type'] = 'integer'
            json_file[info]['unit'] = 'year'
        elif info == 'sex':
            value = value[0]
            json_file[info]['type'] = 'radio'
        elif info == 'height':
            json_file[info]['type'] = 'integer'
            json_file[info]['unit'] = 'cm'
            numbers = []
            number = ''
            for v in value:
                try:
                   number += str(int(v))
                except:
                    if v == '.':
                        number += v
                    elif len(number) > 0:
                        numbers += [float(number)]
                        number = ''
                    
            if len(number) > 0:
                numbers += [float(number)]
            
            if len(numbers) == 2:
                # In ft-inch metric
                unit = 'ft'
                value = int(numbers[0] * 30.40 + numbers[1] * 2.54)
            elif len(numbers) == 1:
                if numbers[0] > 100:
                    # In cm metric
                    unit = 'cm'
                    value = int(numbers[0])
                elif numbers[0] < 2.5:
                    # In m metric
                    unit = 'm'
                    value = int(numbers[0] * 100)
                else:
                    unit = 'ft'
                    value = int(numbers[0] * 30.40)
            else:
                st()
        
        elif info == 'weight':
            if 'kg' in value:
                unit = 'kg'
            if exp_info in lb_to_kg_list:
                unit = 'ft'
            json_file[info]['type'] = 'integer'
            json_file[info]['unit'] = 'kg'
            number = ''
            for v in value:
                try: 
                    number += str(int(v))
                except:
                    if v == '.':
                        number += v
            
            number = float(number)
            if unit == 'ft':
                value = int(number * 0.453592)
            else:
                value = int(number)
        
        info_dict[info] = value
        json_file[info]['value'] = value

    filepath = osp.join(base_dir, processed_fldr, sid, 'metadata.json')
    with open(filepath, 'w') as make_file:
        json.dump(json_file, make_file, indent=4)

    print("%s %s %s"%(date, exp, set))
    print("original dict is {}".format(org_info_dict))
    print("proessed dict is {}\n".format(info_dict)) 
