import torch

class Config():
    """Configuration of model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dir_data = "/disk/haohy/images/CIFAR100_png"
    # dir_model = "/disk/haohy/IL/models"
    # dir_pool = "/disk/haohy/images/CIFAR100_png/data_pool"
    # dir_logs = "/disk/haohy/IL/logger"
    dir_data = "/disk1/haohy/data/UCRArchive_2018"
    num_dims = 46
    dataname = 'Crop'
    num_epochs = 100
    cap = 10000
    num_inc = 3
    lr = 1e-3
    batch_size = 16
    lr_schedule = True 
    num_workers = 4

config = Config()
