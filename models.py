from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import os
import utils
from tqdm import tqdm_notebook
import multiprocessing
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# This is my custom fork of torchsample which fixes some bugs.
# Install via: pip install git+https://github.com/jrieke/torchsample
import torchsample



# -------------------------- PyTorch models ---------------------------------

class ClassificationModel3D(nn.Module):
    """The model we use in the paper."""
    
    def __init__(self, dropout=0, dropout2=0):
        nn.Module.__init__(self)
        self.Conv_1 = nn.Conv3d(1, 8, 3)
        self.Conv_1_bn = nn.BatchNorm3d(8)
        self.Conv_2 = nn.Conv3d(8, 16, 3)
        self.Conv_2_bn = nn.BatchNorm3d(16)
        self.Conv_3 = nn.Conv3d(16, 32, 3)
        self.Conv_3_bn = nn.BatchNorm3d(32)
        self.Conv_4 = nn.Conv3d(32, 64, 3)
        self.Conv_4_bn = nn.BatchNorm3d(64)
        self.dense_1 = nn.Linear(5120, 128)
        self.dense_2 = nn.Linear(128, 64)
        self.dense_3 = nn.Linear(64, 2)  #1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)
        
    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = F.max_pool3d(x, 2)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = F.max_pool3d(x, 3)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = F.max_pool3d(x, 2)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = F.max_pool3d(x, 3)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.dense_1(x))
        x = self.dropout2(x)
        x = self.relu(self.dense_2(x))
        x = self.dense_3(x)
        
        # Note that no sigmoid is applied here, because the network is used in combination with BCEWithLogitsLoss,
        # which applies sigmoid and BCELoss at the same time to make it numerically stable.
            
        return x
    
    
class KorolevModel(nn.Module):
    """The model used in Korolev et al. 2017 (https://arxiv.org/abs/1701.06643)."""
    def __init__(self):
        nn.Module.__init__(self)
        
        self.relu = nn.ReLU()
        
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            self.relu,
            nn.Conv3d(8, 8, 3),
            self.relu,
            nn.BatchNorm3d(8),
            nn.MaxPool3d(2),  
            
            nn.Conv3d(8, 16, 3),
            self.relu,
            nn.Conv3d(16, 16, 3),
            self.relu,
            nn.BatchNorm3d(16),
            nn.MaxPool3d(2),
            
            nn.Conv3d(16, 32, 3),
            self.relu,
            nn.Conv3d(32, 32, 3),
            self.relu,
            nn.Conv3d(32, 32, 3),
            self.relu,
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, 3),
            self.relu,
            nn.Conv3d(64, 64, 3),
            self.relu,
            nn.Conv3d(64, 64, 3),
            self.relu,
            nn.Conv3d(64, 64, 3),
            self.relu,
            nn.BatchNorm3d(64),
            nn.MaxPool3d(3)  # 2 in original paper, increased to 3 because of larger image size
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2880, 128), 
            self.relu,
            #nn.BatchNorm1d(128), 
            nn.Dropout(0.7), 
            nn.Linear(128, 64),
            self.relu,
            nn.Linear(64, 1)  # paper uses 2 output neurons with softmax, we use 1 output neuron with sigmoid
            # TODO: Maybe try using 2 output neurons and softmax.
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
    

# ------------------------------ Wrappers ------------------------------
def build_model():
    """Build the model as used in the paper, wrap it in a torchsample trainer and move it to cuda."""
    # Option 1: Model inspired by Khvostikov et al. 2017
    net = ClassificationModel3D(dropout=0.8, dropout2=0)
    # Tested 0.001 and 0.00001 on subset of the training dataset (10 AD/10 NC), got worse results in both cases.
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    #loss_function = nn.BCEWithLogitsLoss()
    loss_function = nn.CrossEntropyLoss()

    callbacks = []
    #callbacks.append(torchsample.callbacks.ModelCheckpoint('logs/2_ClassificationModel3D_1.5T-and-3T-combined_dropout-0.8', 'epoch_{epoch}-loss_{loss}-val_loss_{val_loss}', 'val_loss', save_best_only=True, max_save=1))

    trainer = torchsample.modules.ModuleTrainer(net)
    #trainer.compile(loss=loss_function, optimizer=optimizer, metrics=[BinaryAccuracyWithLogits()], callbacks=callbacks)
    trainer.compile(loss=loss_function, optimizer=optimizer, metrics=[CategoricalAccuracyWithLogits()], callbacks=callbacks)

    if torch.cuda.is_available():
        net.cuda()
        cuda_device = torch.cuda.current_device()
        print('Moved network to GPU')
    else:
        cuda_device = -1
        print('GPU not available')

    return net, trainer, cuda_device


def train_model(trainer, train_loader, val_loader, cuda_device, num_epoch=1):
    """Train and evaluate the model via torchsample."""
    trainer.fit_loader(train_loader,
            val_loader=val_loader,
            num_epoch=num_epoch,
            verbose=1,
            cuda_device=cuda_device)
    
    
    
# ------------------------ Metrics ----------------------------

def calculate_roc_auc(trainer, val_loader, cuda_device):
    y_val_pred = F.softmax(trainer.predict_loader(val_loader, cuda_device=cuda_device)).data.cpu().numpy()
    # TODO: Both arrays have an inconsistent number of samples. 
    #       Implemented a quick fix here, but does not evaluate all samples.
    y_val_true = torch.cat([y for x, y in val_loader]).numpy()
    y_val_true = y_val_true[:len(y_val_pred)]

    return roc_auc_score(y_val_true, y_val_pred.argmax(1))


class BinaryAccuracyWithLogits(torchsample.metrics.BinaryAccuracy):
    """Same as torchsample.metrics.BinaryAccuracy, but applies a sigmoid function to the network output before calculating the accuracy. This is intended to be used in combination with BCEWightLogitsLoss."""

    def __call__(self, y_pred, y_true):
        return super(BinaryAccuracyWithLogits, self).__call__(F.sigmoid(y_pred), y_true)
    
class CategoricalAccuracyWithLogits(torchsample.metrics.CategoricalAccuracy):
    """Same as torchsample.metrics.CategoricalAccuracy, but applies a softmax function to the network output before calculating the accuracy. This is intended to be used in combination with CrossEntropyLoss."""

    def __call__(self, y_pred, y_true):
        return super(CategoricalAccuracyWithLogits, self).__call__(F.softmax(y_pred), y_true)

