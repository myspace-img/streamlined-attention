# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 22:59:55 2021

@author: cgnya
"""

import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
from torchsummary import summary
from torch.hub import load_state_dict_from_url
import torch.utils.model_zoo as model_zoo

from OurModelModule import SLayer

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
            'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.sl = SLayer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sl(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.sl = SLayer(planes * 4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out = self.sl(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def resnet18(num_classes=1_000):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def resnet34(num_classes=1_000, pretrained=False):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.avgpool=nn.AdaptiveAvgPool2d(1)
    num_ftrs = model.fc.in_features
    model.fc  = nn.Linear(num_ftrs, out_features = num_classes)             
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
    return model


def resnet50(num_classes=1_000, pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool=nn.AdaptiveAvgPool2d(1)
    num_ftrs = model.fc.in_features
    model.fc  = nn.Sequential(
                nn.Linear(num_ftrs, 1024),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.Dropout(0.5),
                nn.Linear(512, out_features = num_classes)
                )
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
    return model


def resnet101(num_classes=1_000, pretrained=False):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool=nn.AdaptiveAvgPool2d(1)
    num_ftrs = model.fc.in_features
    model.fc  = nn.Sequential(
                nn.Linear(num_ftrs, 1024),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.Dropout(0.5),
                nn.Linear(512, out_features = num_classes)
                )
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
    return model


def resnet152(num_classes=1_000, pretrained=False):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool=nn.AdaptiveAvgPool2d(1)
    num_ftrs = model.fc.in_features
    model.fc  = nn.Sequential(
                nn.Linear(num_ftrs, 1024),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.Dropout(0.5),
                nn.Linear(512, out_features = num_classes)
                )
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet152'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
    return model

# print(resnet50(num_classes=1000))
# summary(resnet50(num_classes=1000),(3,224,224))