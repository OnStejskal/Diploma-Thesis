import time
import random
import os
import pandas as pd
import numpy as np
import torch
from torchvision import models
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init

#############################################################
### this file includes models for classification          ###
### and regressionof parameters of the carotid artery     ###
#############################################################


class Resnet18NonGeomParamsMaskMiddle(nn.Module):  
    def __init__(self, pretrained = True,input_img_dim = 3,input_seg_dim = 3,  echogenicity = True, mask_layer = 6):
        """init method for Resnet18NonGeomParamsMaskMiddle

        Args:
            pretrained (bool, optional): indicates whether to use pretrained values. Defaults to True.
            input_dim (int, optional): number of input channels. Defaults to 4.
            output_dim (int, optional): number of outputs. Defaults to 1.
        """
        super(Resnet18NonGeomParamsMaskMiddle, self).__init__()
        if mask_layer not in [5,6,7]:
            print("NOT VALID LAYER FOR CONCATING")
        num_classes = 4 if echogenicity else 2
        weights = 'DEFAULT' if pretrained else None
        self.resnet_model = models.resnet18(weights=weights)
        fc_layer_in_features=self.resnet_model.fc.in_features

        img_first_layer = nn.Conv2d(input_img_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        init.xavier_uniform_(img_first_layer.weight)
        self.img_pipe = nn.Sequential(img_first_layer, *list(self.resnet_model.children())[1:mask_layer])

        seg_first_layer = nn.Conv2d(input_seg_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        init.xavier_uniform_(seg_first_layer.weight)
        self.seg_pipe = nn.Sequential(seg_first_layer, *list(self.resnet_model.children())[1:mask_layer])
        
        self.combined_pipe = nn.Sequential(*list(self.resnet_model.children())[mask_layer:-1])
        self.output_layer = nn.Linear(fc_layer_in_features, num_classes)
        init.xavier_uniform_(self.output_layer.weight)
        if pretrained == False: # init weights using xavier initialization
            print("Not Pretrained: false, using xavier")
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        init.zeros_(m.bias)

    def forward(self, x):
        x1, x2 = x
        x1 = self.img_pipe(x1)
        x2 = self.seg_pipe(x2)
        x3 = x1*x2
        x4 = self.combined_pipe(x3)
        x4 = torch.flatten(x4, 1)
        x5 = self.output_layer(x4)
        return x5

class Resnet18NonGeomParamsCatMiddle(nn.Module):  
    def __init__(self, pretrained = True,input_img_dim = 3,input_seg_dim = 3,  echogenicity = True, concat_layer = 6):
        """init method for Resnet18NonGeomParamsCatMiddle

        Args:
            pretrained (bool, optional): indicates whether to use pretrained values. Defaults to True.
            input_dim (int, optional): number of input channels. Defaults to 4.
            output_dim (int, optional): number of outputs. Defaults to 1.
        """
        super(Resnet18NonGeomParamsCatMiddle, self).__init__()
        if concat_layer not in [5,6,7]:
            print("NOT VALID LAYER FOR CONCATING")
        num_classes = 4 if echogenicity else 2
        weights = 'DEFAULT' if pretrained else None
        self.resnet_model = models.resnet18(weights=weights)
        fc_layer_in_features=self.resnet_model.fc.in_features
        print(f'in features: {fc_layer_in_features}')
        img_first_layer = nn.Conv2d(input_img_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        init.xavier_uniform_(img_first_layer.weight)
        self.img_pipe = nn.Sequential(img_first_layer, *list(self.resnet_model.children())[1:concat_layer])

        seg_first_layer = nn.Conv2d(input_seg_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        init.xavier_uniform_(seg_first_layer.weight)
        self.seg_pipe = nn.Sequential(seg_first_layer, *list(self.resnet_model.children())[1:concat_layer])
        
        self.cat_layer = nn.Sequential(*list(self.resnet_model.children())[concat_layer][1:])
        self.combined_pipe = nn.Sequential(*list(self.resnet_model.children())[concat_layer+1:-1])
        self.output_layer = nn.Linear(fc_layer_in_features, num_classes)
        init.xavier_uniform_(self.output_layer.weight)
        
        if pretrained == False: # init weights using xavier initialization
            print("Not Pretrained: false, using xavier")
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        init.zeros_(m.bias)

    def forward(self, x):
        x1, x2 = x
        x1 = self.img_pipe(x1)
        x2 = self.seg_pipe(x2)
        x3 = torch.cat((x1, x2), dim=1)
        x4 = self.cat_layer(x3)
        x5 = self.combined_pipe(x4)
        x5 = torch.flatten(x5, 1)
        x6 = self.output_layer(x5)
        return x6


class Resnet18NonGeomParams(nn.Module):  
    def __init__(self, pretrained = True,input_dim = 3, echogenicity = True):
        """init method for Resnet18NonGeomParams

        Args:
            pretrained (bool, optional): indicates whether to use pretrained values. Defaults to True.
            input_dim (int, optional): number of input channels. Defaults to 4.
            output_dim (int, optional): number of outputs. Defaults to 1.
        """
        super(Resnet18NonGeomParams, self).__init__()
        num_classes = 4 if echogenicity else 2
        weights = 'DEFAULT' if pretrained else None
        self.resnet_model = models.resnet18(weights=weights)
        self.resnet_model.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(in_features, num_classes)
        

        init.xavier_uniform_(self.resnet_model.conv1.weight)
        init.xavier_uniform_(self.resnet_model.fc.weight)

        if pretrained == False: # init weights using xavier initialization
            print("Not Pretrained: false, using xavier")
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        init.zeros_(m.bias)

    def forward(self, x):
        return self.resnet_model(x)

class Resnet34NonGeomParams(nn.Module):  
    """Resnet neural net for regreesion

    This neural net accepts images of size 256x256 with arbitrary number of channels, the input layer is changed
    and the output classification layer is replaced by fully connected layer with 3 layers with possibility to set arbitrary number of outputs
    also the net can be modified as a feature exctractor or fullly trainable network

    """
    def __init__(self, pretrained = True,input_dim = 3, echogenicity = True):
        """init method for Resnet34NonGeomParams

        Args:
            pretrained (bool, optional): indicates whether to use pretrained values. Defaults to True.
            input_dim (int, optional): number of input channels. Defaults to 4.
            output_dim (int, optional): number of outputs. Defaults to 1.
            training_type (string, optional): fully_trainable or feature_extractor, defines the freezing of layers
        """
        super(Resnet34NonGeomParams, self).__init__()
        num_classes = 4 if echogenicity else 2
        weights = 'DEFAULT' if pretrained else None
        self.resnet_model = models.resnet34(weights=weights)
        self.resnet_model.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(in_features, num_classes)
        

        init.xavier_uniform_(self.resnet_model.conv1.weight)
        init.xavier_uniform_(self.resnet_model.fc.weight)

        if pretrained == False: # init weights using xavier initialization
            print("Not Pretrained: false, using xavier")
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        init.zeros_(m.bias)

    def forward(self, x):
        return self.resnet_model(x)
