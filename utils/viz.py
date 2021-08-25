from utils.viz_utils import CONNECTIVITY_DICT, COLOR_DICT, COLOR_DICT_KEYPOINTS

import numpy as np
import torch

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import trimesh
import pyrender

import math
import os
import os.path as osp
from copy import copy

__author__ = "Soyong Shin"


# # Draw Skeleton

def rgb_to_bgr(color_rgb):
    return np.array(color_rgb[::-1]).tolist()


def draw_2d_skeleton_for_configuration(x, y, img, joint_type='op26', linewidth=4):
    """ Draw 2D skeleton model for configuration file """
    for i in range(26):
        if (x[i]> 1e-5 and y[i] > 1e-5):
            img = cv2.circle(img, (x[i], y[i]), linewidth, rgb_to_bgr(COLOR_DICT_KEYPOINTS[joint_type][i]), thickness=-1) 
    
    # Get line segments        
    for idx, index_set in enumerate(CONNECTIVITY_DICT[joint_type]):
        xs, ys = [], []
        for index in index_set:
            if (x[index] > 1e-5 and y[index] > 1e-5):
                xs.append(x[index])
                ys.append(y[index])
        
        X = np.array(xs).astype(int)
        Y = np.array(ys).astype(int)
        
        # Draw line as elipsoid
        if len(xs) == 2:
            curr_img = img.copy()
            mX = X.mean()
            mY = Y.mean()
            length = np.sqrt((Y[0] - Y[1])**2 + (X[0] - X[1])**2)
            angle = math.degrees(math.atan2(Y[0]-Y[1], X[0]-X[1]))
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length/2), linewidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(curr_img, polygon, rgb_to_bgr(COLOR_DICT[joint_type][idx]))
            img = cv2.addWeighted(img, 0.4, curr_img, 0.6, 0)
    
    # Add keypoints index
    unit = 35
    buffer = [(unit/2, 2*unit), (0, 1.5*unit), (unit/2, -unit), # Body 
              (unit, 0), (unit, 0), (unit, 0), (unit/2, 0), (unit, 0), (unit/2, 0), # Right arm & leg 
              (-1.5*unit, 0), (-2.5*unit, 0), (-2.5*unit, 0), (-2.5*unit, 0), (-2.5*unit, 0), (-2.5*unit, 0), # Left arm and leg 
              (0, -unit/2), (unit*2/3, unit/3), (-2*unit, -unit), (-2.5*unit, 0), # Face
              (-2.2*unit, 0), (unit/3, 0), (-2*unit, 0), (unit/2, unit/3), (-2.5*unit, 0), (0, -unit/2), # Feet
              (0, -unit)]
    for i in range(26):
        if (x[i] > 1e-5 and y[i] > 1e-5):
            x_ = x[i] + int(buffer[i][0])
            y_ = y[i] + int(buffer[i][1])
            cv2.putText(img, str(i), (x_, y_), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), thickness=3, lineType=cv2.LINE_8) 

    return img
 

def draw_2d_skeleton(x, y, img, subj_idx=None, joint_type='op26', linewidth=3):
    """ Draw 2D skeleton model """

    for idx, index_set in enumerate(CONNECTIVITY_DICT[joint_type]):
        # Get line segments
        xs, ys = [], []
        for index in index_set:
            if (x[index] > 1e-5 and y[index] > 1e-5):
                xs.append(x[index])
                ys.append(y[index])
        
        if len(xs) == 2:
            # Draw line
            start = (xs[0], ys[0])
            end = (xs[1], ys[1])
            img = cv2.line(img, start, end, rgb_to_bgr(COLOR_DICT[joint_type][idx]), thickness=linewidth)

    # Write subject index
    if subj_idx is not None:
        if len(y[y>1e-5]) != 0 and y[y>1e-5].min() > 10:
            loc = (int(x[x>1e-5].mean()), int(y[y>1e-5].min() - 5))
            text_color = (209, 80, 0) if subj_idx == 0 else (80, 209, 0)
            img = cv2.putText(img, text="Set%02d"%(subj_idx + 1), org=loc, fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                              fontScale=0.5, color=text_color, thickness=2)

    return img


def draw_3d_skeleton(joints, ax=None, conf=None, joint_type='op26'):
    
    if ax is None:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.set_axis_off()
    ax.view_init(azim=-90, elev=-50)
    # ax.view_init(azim=-90, elev=-90)

    ax.set_xlim(-135, 135)
    ax.set_ylim(-270, 000)
    ax.set_zlim(-135, 135)

    # Draw ground
    X = np.arange(-1500, 1500, 100)
    Z = np.arange(-1500, 1500, 100)
    X, Z = np.meshgrid(X, Z)
    Y = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.15, linewidth=1, antialiased=False)

    x, y, z = np.split(joints, 3, axis=-1)
    
    if conf is None:
        conf = np.ones(26)
    
    for idx, index_set in enumerate(CONNECTIVITY_DICT[joint_type]):
        xs, ys, zs = [], [], []
        for index in index_set:
            if conf[index] > 1e-5:
                xs.append(x[index][0])
                ys.append(y[index][0])
                zs.append(z[index][0])

        if len(xs) == 2:
            color_ = COLOR_DICT[joint_type][idx]
            color = [c/255 for c in color_]
            ax.plot3D(xs, ys, zs, color=color)

    import pdb; pdb.set_trace()

# # Draw SMPL

def render_smpl_on_image(vertices, faces, image, intrinsics, pose, transl, 
                alpha=1.0, filename='render_sample.png'):
    
    img_size = image.shape[-2]
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))

    # Generate SMPL vertices mesh
    mesh = trimesh.Trimesh(vertices, faces)

    # Default rotation of SMPL body model
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
    scene.add(mesh, 'mesh')

    camera_pose = np.eye(4)
    camera_pose[:3, :3] = pose
    camera_pose[:3, 3] = transl
    camera = pyrender.IntrinsicsCamera(fx=intrinsics[0, 0], fy=intrinsics[1, 1],
                                       cx=intrinsics[0, 2], cy=intrinsics[1, 2])
    scene.add(camera, pose=camera_pose)

    # Light information
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
    light_pose = np.eye(4)

    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)

    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)

    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=img_size, viewport_height=img_size, point_size=1.0)

    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    valid_mask = (rend_depth > 0)[:,:,None]
    
    color = color.astype(np.float32) / 255.0
    valid_mask = (rend_depth > 0)[:,:,None]
    
    output_img = color[:, :, :3] * valid_mask * alpha + \
                 valid_mask * image / 255 * (1-alpha) + (1 - valid_mask) * image / 255
    
    cv2.imwrite(filename, (255 * output_img).astype(np.int16))


def draw_smpl_body(body_model, body_model_output, camera_info, filename):
    faces = body_model.faces
    focal_length = 5e3
    cam_pose = np.eye(3)
    cam_transl = np.array([0., 0., 30])
    intrinsics = np.array([[focal_length, 0, 256], [0, focal_length, 256], [0, 0, 1]])
    background = np.ones([540, 540, 3]) * 180

    vertices = body_model_output.vertices.detach().cpu().numpy()[0]
    render_smpl_on_image(vertices, faces, background, intrinsics, cam_pose, cam_transl, filename=filename)


# # Draw Syncing result

def plot_and_save_analysis_fig(params, front_synced_imu_accel, front_kp_accel, 
                               end_synced_imu_accel, end_kp_accel):
    
    output_fldr = osp.join(params['base_dir'], 'Syncing_Result', params['date'])
    
    if not osp.exists(output_fldr):
        os.makedirs(output_fldr)
    
    fig = plt.figure(figsize=(10,5))
    front_ax, end_ax = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    plt.suptitle("Syncing result (Left : front hopping, Right : End hopping)", 
                 y=0.95, fontsize=14)

    plot_sliding_with_index(0, front_synced_imu_accel, front_kp_accel, 
                            check_range=1, ax=front_ax)
    plot_sliding_with_index(0, end_synced_imu_accel, end_kp_accel, 
                            check_range=1, ax=end_ax)
    filename = 'Set%02d_exp%02d.png'%(params['id'], params['exp'])
    plt.savefig(osp.join(output_fldr, filename))
    plt.close()


def plot_sliding_with_index(index, imu_accel, kp_accel, check_range=200, ax=None):
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    start_idx, end_idx = index, index - check_range
    ax.plot(imu_accel[start_idx:end_idx], color='tab:red', label='IMU')
    ax.plot(kp_accel, color='tab:blue', label='KP')
    ax.legend()
    
    pass
