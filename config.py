import torch
import numpy as np

class Config():
    """Configuration of model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dir_data = "/disk1/haohy/images/mnist"
    dir_model = "/disk1/haohy/IL/models"
    dir_pool = "/disk1/haohy/IL/data_pool"
    num_epochs = 20
    cap = 10000
    num_inc = 2
    lr = 1e-3
    batch_size = 256 
    lr_schedule = False
    data_type = 'mnist'
    data_shape = (1, 28, 28)
    num_workers = 4

config = Config()
