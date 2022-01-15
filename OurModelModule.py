# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 23:50:03 2021

@author: cgnya
"""
import torch
import torch.nn as nn

class SLayer(nn.Module):
    def __init__(self, channel):
        super(SLayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc =nn.Linear(channel, channel,  bias=False)
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)
    
    
