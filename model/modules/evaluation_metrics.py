import torch
import random
import pdb

def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    If mode == "sum", we return the sum of all differences between GT and pred 
    along the prediction
    """
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)

    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode='sum'):
    """
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))

    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)