"""
We create a superset of joints containing the OpenPose joints together with the ones that each dataset provides.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are the following:
"""

# Folder configuration
BASE_RAW_DATA_DIR = 'dataset/MBL_DomeData/dome_data'
VGA_KEYPOINTS_STAGE1_FLDR = 'vgaPose3d_stage1_op25'
HD_KEYPOINTS_STAGE1_FLDR = 'hdPose3d_stage1_op25'
HD_KEYPOINTS_STAGE2_FLDR = 'hdPose3d_stage2_op25'

IMAGE_FLDR = 'skeleton_video'

RAW_IMU_DIR = 'dataset/dome_IMU'
SYNCED_IMU_FLDR = 'mc10_IMU'

SEGMENTED_DATA_FLDR = 'Processed'
SEGMENTED_KEYPOINTS_FLDR = 'OpenPose3D'
SEGMENTED_IMU_FLDR = 'MC10_IMU'
ACTION_LABEL_FILENAME = 'action_label.xlsx'

# IMU sensor list
IMU_PARTS = ['chest', 'head', 'lbicep', 'lfoot', 'lforearm', 'lhand', 'lshank', 'lthigh',
             'rbicep', 'rfoot', 'rforearm', 'rhand', 'rshank', 'rthigh', 'sacrum']
INDEX_COL = 'Timestamp (microseconds)'
SENSORS = ['accel', 'gyro']

# Experiments configuration
EXP_DATES = ['190503', '190510', '190517', '190607']
EXP_SEQUENCES = ['exp01', 'exp02', 'exp03', 'exp04', 'exp05', 'exp06', 'exp07', 'exp08', 
                 'exp09', 'exp10', 'exp11', 'exp12', 'exp13', 'exp14']
EXP_SUBJECTS = ['Set01', 'Set02']
EXP_SINGULAR = ['190517_exp12', '190607_exp11']

SMPL_TO_H36 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
J32_TO_J17 = [3, 2, 1, 6, 7, 8, 27, 26, 25, 17, 18, 19, 24, 15, 11, 12, 14]
J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
OP26_TO_OP25 = [1, 0, 9, 10, 11, 3, 4, 5, 2, 12, 13, 14, 6, 7, 8, 17, 15, 18, 16, 20, 19, 21, 22, 23, 24]
OP25_TO_OP13 = [1, 5, 6, 7, 2, 3, 4, 12, 13, 14, 9, 10, 11]
OP26_TO_J17 = [14, 13, 12, 6, 7, 8, 11, 10, 9, 3, 4, 5, 0, 25, 2, 20, 1] # 20 -> mean(2, 0)

# Indices to get the 14 LSP joints from the ground truth joints
H36M_TO_J14 = H36M_TO_J17[:14]
J24_TO_J14 = J24_TO_J17[:14]

JOINT_REGRESSOR_H36M = 'dataset/3D_SMPLify/J_regressor_h36m.npy'

### SMPL Constants

JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose',
'OP Neck',
'OP RShoulder',
'OP RElbow',
'OP RWrist',
'OP LShoulder',
'OP LElbow',
'OP LWrist',
'OP MidHip',
'OP RHip',
'OP RKnee',
'OP RAnkle',
'OP LHip',
'OP LKnee',
'OP LAnkle',
'OP REye',
'OP LEye',
'OP REar',
'OP LEar',
'OP LBigToe',
'OP LSmallToe',
'OP LHeel',
'OP RBigToe',
'OP RSmallToe',
'OP RHeel',
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle',
'Right Knee',
'Right Hip',
'Left Hip',
'Left Knee',
'Left Ankle',
'Right Wrist',
'Right Elbow',
'Right Shoulder',
'Left Shoulder',
'Left Elbow',
'Left Wrist',
'Neck (LSP)',
'Top of Head (LSP)',
'Pelvis (MPII)', 
'Thorax (MPII)',
'Spine (H36M)',
'Jaw (H36M)', 
'Head (H36M)',
'Nose', 
'Left Eye',
'Right Eye',
'Left Ear',
'Right Ear'
]

SMPL_JOINT_MAP = {
'Left Hip', 'Right Hip', 'Spine 1 (Lower)', 'Left Knee', 'Right Knee', 'Spine 2 (Middle)',
'Left Ankle', 'Right Ankle', 'Spine 3 (Upper)' 'Left Foot', 'Right Foot',
'Neck', 'Left Shoulder (Inner)', 'Right Shoulder (Inner)', 'Head',
'Left Shoulder (Outer)', 'Right Shoulder (Outer)', 'Left Elbow', 'Right Elbow', 
'Left Wrist', 'Right Wrist', 'Left Hand', 'Right Hand'
}

JOINT_NAMES_H36M = [
'Right Ankle', 'Right Knee', 'Right Hip',
'Left Hip', 'Left Knee', 'Left Ankle',
'Right Wrist', 'Right Elbow', 'Right Shoulder', 
'Left Shoulder', 'Left Elbow', 'Left Wrist', 
'Neck (LSP)', 'Head (H36M)', 'Pelvis (MPII)', 'Spine (H36M)', 'Jaw (H36M)'
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
# From here H36M
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

OPENPOSE_JOINTS_NAME = [
    'nose',
    'neck',
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'pelvis',
    'right_hip',
    'right_knee',
    'right_ankle',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
]

CONNECTIVITY = {'face': [17, 15, 0, 16, 18, 'crimson'],
                'back': [0, 1, 8, 'maroon'],
                'rarm': [4, 3, 2, 1, 'forestgreen'],
                'larm': [7, 6, 5, 1, 'orange'],
                'rleg': [11, 10, 9, 8, 'darkblue'],
                'lleg': [14, 13, 12, 8, 'seagreen'],
                'rfoot': [23, 22, 11, 24, 'mediumblue'],
                'lfoot': [20, 19, 14, 21, 'mediumseagreen']}


SEGMENT_DICT = {'larm1': [7,6],
                'larm2': [6,5],
                'rarm1': [4, 3],
                'rarm2': [3, 2],
                'lleg1': [14,13],
                'lleg2': [13, 12],
                'rleg1': [11, 10],
                'rleg2': [10, 9],
                'back': [9, 12, 1]
                }

SACRUM_IDX = 8
HIP_IDXS = [9, 12]