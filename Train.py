# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 01:30:57 2021

@author: Lalith_B
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import time
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
from EarlyStopping import EarlyStopping
from torchvision.datasets import CIFAR100,CIFAR10
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet

from OurModel import resnet101, resnet152, resnet50, resnet34
import math
import torch.utils.model_zoo as model_zoo
CUDA_LAUNCH_BLOCKING=1

traindata_transforms = transforms.Compose([
                        transforms.Resize((192,192)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

testdata_transforms = transforms.Compose([
                        transforms.Resize((192,192)),  
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# cars_train = r'C:\Users\Lalith_B\Desktop\Gnyan ResNet\Datasets\cars\stanford-car-dataset-by-classes-folder-224\car_data\train'
# cars_test = r'C:\Users\Lalith_B\Desktop\Gnyan ResNet\Datasets\cars\stanford-car-dataset-by-classes-folder-224\car_data\test'

# aircrafts_train = r'C:\Users\Lalith_B\Desktop\Gnyan ResNet\Datasets\aircrafts\train'
# aircrafts_test = r'C:\Users\Lalith_B\Desktop\Gnyan ResNet\Datasets\aircrafts\train'

# cifar10_train = CIFAR10(download=True,root="./data",transform=traindata_transforms)
# cifar10_test = CIFAR10(root="./data",train=False,transform=testdata_transforms)

birds_train = r'C:\Users\Lalith_B\Desktop\Gnyan ResNet\Datasets\birds\Train'
birds_test = r'C:\Users\Lalith_B\Desktop\Gnyan ResNet\Datasets\birds\Test'

# cifar100_train = CIFAR100(download=True,root="./data",transform=traindata_transforms)
# cifar100_test = CIFAR100(root="./data",train=False,transform=testdata_transforms)

train_transforms = datasets.ImageFolder(os.path.join(birds_train), traindata_transforms)
test_transforms = datasets.ImageFolder(os.path.join(birds_test), testdata_transforms)
batch_size = 128

train_dl = DataLoader(train_transforms, batch_size, shuffle=True) 
test_dl = DataLoader(test_transforms, batch_size, shuffle=True) 

# class_names = os.listdir(r'C:\Users\cgnya\OneDrive\Desktop\Work\Datasets\aircrafts\train')
class_names = os.listdir(r'C:\Users\Lalith_B\Desktop\Gnyan ResNet\Datasets\birds\Train')

# class_names = ['1']*10
# class_names = ['1']*100


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()

train_dataloader = DeviceDataLoader(train_dl, device)
test_dataloader = DeviceDataLoader(test_dl, device)

def top1_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
    return acc

# Evaluating Loop 
@torch.no_grad()
def test(model, test_dl):
    model.eval()
    for batch in test_dl:
        inputs, labels = batch
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        acc = top1_accuracy(outputs, labels)
    return loss, acc

# Training Loop 
def train(epochs, train_dl, test_dl, model, optimizer, patience, name):
    history = []
    since = time.time()
    optimizer = optimizer(model.parameters(), lr=0.0001) 
    es = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        result = {}
        loop = tqdm(train_dl, total=len(train_dl))
        for batch in loop:
            inputs, labels = batch
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            acc = top1_accuracy(outputs, labels)
            train_acc.append(acc.cpu().detach().numpy())
            train_loss.append(loss.cpu().detach().numpy())
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(train_loss=np.average(train_loss),train_acc=np.average(train_acc))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        test_losses,test_accu = test(model, test_dl)
        test_loss.append(test_losses.cpu().detach().numpy())
        test_acc.append(test_accu.cpu().detach().numpy())       
        result['train_loss'] = np.average(train_loss)
        result['train_acc'] = np.average(train_acc)
        result['test_loss'] = np.average(test_loss)
        result['test_acc'] = np.average(test_acc)
        print('\nEpoch',epoch,result)
        history.append(result)
        tl_es = np.average(test_loss)
        es(tl_es, model)
        
        if es.early_stop:
            print("Early stopping")
            break
        print()
        
    time_elapsed = time.time() - since
    print('Training Completed in {:.0f} min {:.0f} sec'.format(time_elapsed//60, time_elapsed%60))
    model.load_state_dict(torch.load('checkpoint.pth'))
    torch.save(model, name + '.pth')
    torch.save(model.state_dict(), name + 'wts.pth')
    return history
            
def traintest(dataset, modelname=str, epochs=int, patience=int):
    optimizer = torch.optim.Adam
    if dataset=='cifar10' or dataset=='cifar100' or dataset=='Saircrafts' or dataset=='SCars' or dataset=='birds':
        if modelname == 'resnet50':
            model = resnet50(pretrained=True).to('cuda')
        elif modelname == 'resnet101':
            model = resnet101(pretrained=True).to('cuda')
        elif modelname == 'resnet152':
            model = resnet152(pretrained=True).to('cuda')
        else:
            print('ERROR_model_or_dataset_is_not_found')
    name = dataset + modelname    
    history = train(epochs=epochs,
              train_dl=train_dataloader,
              test_dl=test_dataloader,
              model=model,
              optimizer=optimizer,
              patience = patience,
              name= name)
    df = pd.DataFrame.from_dict(history)
    df.to_csv(name, index=False)
    return df

traintest(dataset='birds', modelname='resnet50', epochs=45, patience=12)