import torch
import copy
import pdb
import numpy as np
import torch.nn as nn

# Model parameters functions

def create_weights(batch, vmin, vmax, w_len=30, w_type="linear"):
    """
    """
    w = torch.ones(w_len)
    if w_type == "linear":
        w = torch.linspace(vmin, vmax, w_len)
    elif w_type == "exponential":
        w = w

    w = torch.repeat_interleave(w.unsqueeze(0), batch, dim=0)
    return w

def freeze_model(model, no_freeze_list=[]):
    """
    """
    for name, child in model.named_children():
        if name not in no_freeze_list:
            for param in child.parameters():
                param.requires_grad = False

    return model

def load_weights(model, checkpoint, layer_name="decoder."):
    """
    """
    own_state = model.state_dict()
    for name, param in checkpoint.items():
        print("name ", name)
        if (name not in own_state) or (layer_name in name):
                continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)
    return own_state

# Parameter and FLOPs

def count_parameters(model):
    """
    """

    trainable_params = 0

    modules = [module for module in model.modules()]
    params = [param.shape for param in model.parameters()]
    
    pdb.set_trace()
    
    for param in model.parameters():
        if param.requires_grad:
            
            trainable_params = trainable_params + param.numel()
            print("params: ", trainable_params)
    return trainable_params