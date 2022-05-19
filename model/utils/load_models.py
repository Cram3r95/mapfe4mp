import torch
from torch import nn
import torchvision
import torchvision.models as models

def load_VGG_model(vgg_type=19, batch_norm=False, pretrained=True, features= True):
    """
    vgg_type : {A, B, D, E}
    Paper: Very Deep Convolutional Networks For Large-Scale Image Recognition
    Trained on Imagenet
    """
    vgg_model = None
    vgg_types = [11, 13, 16, 19]
    assert (vgg_type in vgg_types), "VGG model {} is not implemented".format(vgg_type)
    if vgg_type == 11:
        if batch_norm:
            vgg_model = models.vgg11_bn(pretrained=pretrained)
        else:
            vgg_model = models.vgg11(pretrained=pretrained)
    
    elif vgg_type == 13:
        if batch_norm:
            vgg_model = models.vgg13_bn(pretrained=pretrained)
        else:   
            vgg_model = models.vgg13(pretrained=pretrained)
    elif vgg_type == 16:
        if batch_norm:
            vgg_model = models.vgg16_bn(pretrained=pretrained)
        else:
            vgg_model = models.vgg16(pretrained=pretrained)
    elif vgg_type == 19:
        if batch_norm:
            vgg_model = models.vgg19_bn(pretrained=pretrained)
        else:
            vgg_model = models.vgg19(pretrained=pretrained)
    
    return vgg_model if not features else vgg_model.features

