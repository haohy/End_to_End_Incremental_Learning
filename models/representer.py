import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torchvision.models as models
from config import config
from resnet import resnet18, resnet50

def save_model(model, dir_model=config.dir_model):
    torch.save(model.state_dict(), dir_model)

def load_model(model, num_classes=0, dir_model=config.dir_model):
    """Get model trained using previous data, if num_classes=0, unchange the 
    output layer's dimensionality, else change it to num_classses.
    """
    model_old = torch.load(dir_model)
    model.load_state_dict(model_old, strict=False)
    if num_classes != 0:
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_classes)

    return model
