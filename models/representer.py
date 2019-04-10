import os
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from IPython import embed
from config import config
from resnet import resnet18, resnet50

def save_model(model, dir_model):
    if not os.path.isdir(dir_model):
        os.mkdir(dir_model)
    path_model = os.path.join(dir_model, 'E2E.pth')
    torch.save(model.state_dict(), path_model)

def load_model(model, num_classes, dir_model):
    """Get model trained using previous data, if num_classes=0, unchange the 
    output layer's dimensionality, else change it to num_classses.
    """
    path_model = os.path.join(dir_model, 'E2E.pth')
    model_old = torch.load(path_model)
    model.load_state_dict(model_old, strict=False)
    if num_classes != 0:
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_classes)

    return model

def acc_cal(model, test_dataloader):
    """calculate the accuracy of model."""
    total = 0
    right = 0
    for i, (train_batch, label_batch) in enumerate(test_dataloader):
        train_batch = train_batch.to(config.device)
        output_batch = model(train_batch)
        total += len(train_batch)
        output_label = np.argmax(output_batch.cpu().data.numpy(), axis=1)
        right += np.sum(output_label == label_batch.cpu().data.numpy())

    acc = right/total
    return acc
