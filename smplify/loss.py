from utils import constants
from smplify.prior import *

import torch
from torch import nn
import numpy as np


def centering_joints(joints, joint_type):
    if joint_type == 'h36m':
        pelv = joints[:, 14]
    elif joint_type == 'op25':
        lhip, rhip = joints[:, 12].clone(), joints[:, 9].clone()
        pelv = (lhip + rhip)/2

    joints_ = joints - pelv.unsqueeze(1)
    return joints_

def align_two_joints(gt_joints, pred_joints, joint_type='h36m'):
    gt_joints = centering_joints(gt_joints, joint_type)
    pred_joints = centering_joints(pred_joints, joint_type)

    return gt_joints, pred_joints

class SMPLifyLoss(nn.Module):
    def __init__(self,
                 rho = 100,
                 body_pose_prior=None,
                 shape_prior=None,
                 angle_prior=None,
                 align_two_joints=True,
                 use_joint_conf=True,
                 dtype=torch.float32,
                 device='cuda:0',
                 joint_dist_weight=0.0,
                 body_pose_weight=0.0,
                 shape_prior_weight=0.0,
                 ign_joint_idx=None,
                 joint_type='h36m',
                 **kwargs):
        super(SMPLifyLoss, self).__init__()
        
        self.J_regressor = torch.from_numpy(np.load(constants.JOINT_REGRESSOR_H36M)).float()
        
        self.rho = rho
        self.body_pose_prior = body_pose_prior
        self.shape_prior = shape_prior
        self.angle_prior = angle_prior
        
        self.align_two_joints = align_two_joints
        self.use_joint_conf = use_joint_conf
        self.ign_joint_idx = ign_joint_idx
        self.joint_type = joint_type
        
        self.dtype = dtype
        self.device = device
        
        self.register_buffer('joint_dist_weight',
                             torch.tensor(joint_dist_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_prior_weight',
                             torch.tensor(shape_prior_weight, dtype=dtype))
        self.to(device=device)

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)


    def forward(self, 
                body_model_output, 
                gt_joints, 
                joint_conf=None,
                **kwargs):
        
        def robustifier(value):
            dist = torch.div(value**2, value**2 + self.rho ** 2)
            return self.rho ** 2 * dist

        J_regressor = self.J_regressor[None, :].expand(gt_joints.shape[0], -1, -1).to(self.device)

        joint_weight = torch.ones(gt_joints.shape[-2], dtype=self.dtype, device=self.device)
        if self.ign_joint_idx is not None:
            joint_weight[self.ign_joint_idx] = 0
        if joint_conf is not None and self.use_joint_conf:
            joint_weight = joint_weight * joint_conf
        joint_weight[joint_weight < 0] = 0
        joint_weight.unsqueeze_(-1)

        if self.joint_type == 'h36m':
            pred_joints = torch.matmul(
                J_regressor, body_model_output.vertices)[:, constants.H36M_TO_J17, :]
        else:
            pred_joints = body_model_output.joints[:, :25]

        gt_joints, pred_joints = align_two_joints(gt_joints, pred_joints, self.joint_type)
        
        # Loss 1 : Joint distance loss
        joint_dist = robustifier(gt_joints - pred_joints)
        joint_loss = torch.sum(joint_dist * joint_weight ** 2) * self.joint_dist_weight
        
        # MPJPE
        mask = (joint_weight > 0)[0, :14, 0]
        mpjpe = torch.sqrt(((gt_joints[:, :14] - pred_joints[:, :14])**2).sum(-1))[:, mask].mean(1) * 1e2

        sprior_loss = torch.sum(self.shape_prior(
            body_model_output.betas)) * self.shape_prior_weight ** 2

        # Loss 2 : Pose prior loss
        pprior_loss = self.body_pose_weight ** 2 * self.body_pose_prior(
            body_model_output.body_pose, body_model_output.betas)

        total_loss = joint_loss + pprior_loss.sum() + sprior_loss
        
        return total_loss, mpjpe


def build_loss_function(loss_type='smplify', **kwargs):
    return SMPLifyLoss(**kwargs)