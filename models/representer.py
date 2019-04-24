import os
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from IPython import embed
from config import config

def save_model(model, dir_model):
    if not os.path.isdir(dir_model):
        os.mkdir(dir_model)
    path_model = os.path.join(dir_model, 'TS_IL.pth')
    torch.save(model.state_dict(), path_model)

def Linear(input_size, output_size, activation=None, p=0., bias=True):
    model = [nn.Linear(input_size, output_size, bias=bias)]
    if activation == 'relu':
        model += [nn.ReLU(inplace=True)]
    elif activation == 'sigmoid':
        model += [nn.Sigmoid()]
    if p > 0.:
        model += [nn.Dropout(p)]
    return nn.Sequential(*model)

def load_model(model, num_classes, dir_model):
    """Get model trained using previous data, if num_classes=0, unchange the 
    output layer's dimensionality, else change it to num_classses.
    """
    path_model = os.path.join(dir_model, 'TS_IL.pth')
    model_old = torch.load(path_model)
    model.load_state_dict(model_old, strict=False)
    if num_classes != 0:
        fc_features = model.fc[0].in_features
        model.fc = Linear(fc_features, num_classes, 'sigmoid')

    return model

def acc_cal(model, test_dataloader):
    """calculate the accuracy of model."""
    model.eval()
    total = 0
    right = 0
    with torch.no_grad():
        for i, (train_batch, label_batch) in enumerate(test_dataloader):
            train_batch = train_batch.to(config.device)
            label_batch = label_batch.to(config.device)
            output_batch = model(train_batch)
            pred = output_batch.argmax(dim=1, keepdim=True)
            right += pred.eq(label_batch.view_as(pred)).sum().item()
            # _, outputs = torch.max(output_batch, 1)
            # right += torch.sum(outputs.cpu() == label_batch.data).double()
            # embed()
            # output_batch = np.argmax(output_batch.cpu().data.numpy(), axis=1)
            # label_batch = label_batch.cpu().data.numpy()
            # for i in range(len(label_batch)):
            #     if label_batch[i] == output_batch[i]:
            #         right += 1

    total = len(test_dataloader.dataset)
    acc = right/total
    return acc
