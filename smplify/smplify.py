from smpl import SMPL
from smplify.loss import SMPLifyLoss
from smplify.prior import L2Prior, AnglePrior, MaxMixturePrior
from smplify import optim_factory
from utils.conversion import rotation_matrix_to_angle_axis
from utils import constants

import torch
from torch import nn
import numpy as np
import os

from tqdm import tqdm, trange

from collections import defaultdict


class FittingLoop(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-07,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smplx',
                 **kwargs):
        super(FittingLoop, self).__init__()
        
        self.summary_steps = summary_steps
        self.visualize = visualize
        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol
        self.body_color = body_color
        self.model_type = model_type

    def __enter__(self):
        self.steps = 0
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def run_optimization(self, optimizer, closure, 
                         params, body_model, stage=None,
                         **kwargs):

        def rel_loss_change(prev_loss, curr_loss):
            return (prev_loss - curr_loss) / max(np.abs(prev_loss), np.abs(curr_loss), 1)
        
        if stage is not None and stage >= 2:
            maxiters = self.maxiters * 2
        else:
            maxiters = self.maxiters

        with tqdm(total=maxiters, leave=False) as prog_bar:
            for n in range(maxiters):
                loss, mpjpe = optimizer.step(closure)

                if (torch.isnan(loss).sum() > 0 and 
                    torch.isinf(loss).sum() > 0):
                    print("Inappropriate loss value, break the loop !")
                    break
                
                # Firt convergence criterion
                if self.ftol > 0 and n > 0:
                    rel_loss_change_ = rel_loss_change(prev_loss, loss.item())
                    if rel_loss_change_ < self.ftol:
                        break
                
                # Second convergence criterion
                if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                        for var in params if var.grad is not None]):
                    break

                prev_loss = loss.item()
                msg = "Loss : %.2f"%(prev_loss) + '  |  MPJPE : %.2f'%(mpjpe)
                prog_bar.set_postfix_str(msg)
                prog_bar.update(1)
                prog_bar.refresh()

        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_model,
                               gt_joints=None, 
                               loss=None,
                               joint_conf=None,
                               create_graph=False,
                               return_verts=True, 
                               return_full_pose=True,
                               **kwargs):
        
        def loss_function(backward=True):
            if backward:
                optimizer.zero_grad()
            
            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=None,
                                           return_full_pose=return_full_pose)
            
            total_loss, mpjpe = loss(body_model_output,
                              gt_joints=gt_joints,
                              joint_conf=joint_conf,
                              **kwargs)            
            if backward:
                total_loss.backward(create_graph=create_graph)
            
            self.steps += 1

            return total_loss, mpjpe

        return loss_function


class SMPLify3D(nn.Module):
    def __init__(self, ign_joint_idx=None, joint_type='h36m',
                 joint_dist_weights=None, body_pose_prior_weights=None, shape_prior_weight=None, 
                 optimizer_type = 'adam', maxiters=100, ftol=0, gtol=0, rho=100, lr=5e-2, betas=(0.9, 0.999), weight_decay=0.0,
                 dtype=torch.float, device='cuda'):

        super(SMPLify3D, self).__init__()
        
        self.shape_prior = L2Prior(dtype=dtype)
        self.angle_prior = AnglePrior(dtype=dtype)
        self.body_pose_prior = MaxMixturePrior(dtype=dtype)

        self.joint_dist_weights = joint_dist_weights
        self.body_pose_prior_weights = body_pose_prior_weights
        self.shape_prior_weight = shape_prior_weight
        self.ign_joint_idx = ign_joint_idx
        self.joint_type = joint_type

        self.optimizer_type = optimizer_type
        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.rho = rho
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

        self.dtype = dtype
        self.device = device

    def run_smplify(self, pred_pose, pred_betas, pred_global_orient, 
                    gt_joints, body_model, batch_size, init_frame, **kwargs):
        
        loss_function = SMPLifyLoss(rho=self.rho, 
                                    ign_joint_idx=self.ign_joint_idx,
                                    joint_type=self.joint_type,
                                    body_pose_prior=self.body_pose_prior,
                                    shape_prior=self.shape_prior,
                                    angle_prior=self.angle_prior,
                                    use_joint_conf=True,
                                    dtype=self.dtype, device=self.device, **kwargs)

        if init_frame:
            maxiters = self.maxiters
            st_stage = 0
        else:
            # maxiters = int(self.maxiters/2)
            maxiters = self.maxiters
            st_stage = 2

        opt_weights_dict = {'joint_dist_weight': [w for w in self.joint_dist_weights[st_stage:]],
                            'body_pose_weight': [w for w in self.body_pose_prior_weights[st_stage:]],
                            'shape_prior_weight': [w for w in self.shape_prior_weight[st_stage:]]}
        keys = opt_weights_dict.keys()
        opt_weights = [dict(zip(keys, vals)) for vals in
                    zip(*(opt_weights_dict[k] for k in keys
                    if opt_weights_dict[k] is not None))]
        
        for weight_list in opt_weights:
            for key in weight_list:
                weight_list[key] = torch.tensor(weight_list[key],
                                                device=self.device,
                                                dtype=self.dtype)
        
        if gt_joints.shape[-1] == 4:
            joint_conf = gt_joints[:, :, -1]
            gt_joints = gt_joints[:, :, :-1]
        else:
            joint_conf = torch.ones(*gt_joints.shape[:-1]).to(device=self.device, dtype=self.dtype)
                
        with FittingLoop(maxiters=maxiters, ftol=self.ftol, gtol=self.gtol, **kwargs) as floop:
            body_model.reset_params(betas=pred_betas, body_pose=pred_pose, global_orient=pred_global_orient)

            for open_idx, curr_weights in enumerate(opt_weights):
                optim_params = [body_model.betas, body_model.body_pose, body_model.global_orient]
                final_params = list(filter(lambda x: x.requires_grad, optim_params))
                optimizer, _ = \
                    optim_factory.create_optimizer(final_params, optim_type=self.optimizer_type, 
                                                   lr=self.lr, weight_decay=self.weight_decay,
                                                   beta1=self.betas[0], beta2=self.betas[1])

                optimizer.zero_grad()
                loss_function.reset_loss_weights(curr_weights)

                closure = floop.create_fitting_closure(optimizer=optimizer,
                                                        body_model=body_model,
                                                        gt_joints=gt_joints,
                                                        loss=loss_function,
                                                        joint_conf=joint_conf,
                                                        create_graph=False)

                loss = floop.run_optimization(optimizer=optimizer,
                                                closure=closure,
                                                params=final_params,
                                                body_model=body_model,
                                                stage=open_idx)

        output = [body_model.body_pose, body_model.betas, body_model.global_orient]

        return output


    def forward(self, pred_pose, pred_betas, pred_global_orient, 
                body_model, gt_joints, device, dtype, init_frame=True, 
                **kwargs):

        batch_size = pred_pose.shape[0]

        integrated_pose = torch.cat([pred_global_orient.detach(), pred_pose.detach()], dim=1)
        pred_pose_hom = torch.cat([integrated_pose.detach().view(-1, 3, 3).detach(), torch.tensor([0,0,1], dtype=dtype,
            device=device).view(1, 3, 1).expand(batch_size * 24, -1, -1)], dim=-1)
        pred_pose_euler = rotation_matrix_to_angle_axis(pred_pose_hom).contiguous().view(batch_size, -1)
        pred_pose_euler[torch.isnan(pred_pose_euler)] = 0.0
        opt_init_pose = pred_pose_euler[:, 3:].detach()
        opt_init_global_orient = pred_pose_euler[:, :3].detach()

        opt_init_betas = pred_betas.detach()
        opt_pose, opt_betas, opt_global_orient = self.run_smplify(
            opt_init_pose, opt_init_betas, opt_init_global_orient, gt_joints, body_model,
            batch_size=batch_size, init_frame=init_frame)
        
        return opt_pose, opt_betas, opt_global_orient