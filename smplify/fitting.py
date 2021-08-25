from constants import *
import torch
from torch import nn
import numpy as np

from tqdm import tqdm

import os


class FittingLoop(object):
    def __init__(self, maxiters=100, ftol=2e-09, gtol=1e-07, **kwargs):
        super(FittingLoop, self).__init__()
        
        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

    def __enter__(self):
        self.steps = 0
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def run_optimization(self, optimizer, closure, params, body_model, 
                         pose_embedding=None, vposer=None, **kwargs):

        def rel_loss_change(prev_loss, curr_loss):
            return (prev_loss - curr_loss) / max(np.abs(prev_loss), np.abs(curr_loss), 1)

        with tqdm(total=self.maxiters, leave=False) as prog_bar:
            for n in range(self.maxiters):
                loss = optimizer.step(closure)

                if (torch.isnan(loss).sum() > 0 or 
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
                msg = "Loss : %.3f"%(prev_loss)
                prog_bar.set_postfix_str(msg); prog_bar.update(1); prog_bar.refresh()
        
        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_model,
                               length_dict=None, 
                               keypoints_3d_gt=None,
                               loss=None,
                               pose_embedding=None, 
                               vposer=None,
                               return_verts=True, 
                               return_full_pose=True,
                               **kwargs):

        def loss_function(backward=True):
            if backward:
                optimizer.zero_grad()

            body_pose = vposer.decode(pose_embedding, output_type='aa').view(1, -1)
            
            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)
            
            total_loss = loss(body_model_output,
                              keypoints_3d_gt=keypoints_3d_gt,
                              pose_embedding=pose_embedding,
                              length_dict=length_dict,
                              **kwargs)            
            
            if backward:
                total_loss.backward()
            
            self.steps += 1

            return total_loss

        return loss_function