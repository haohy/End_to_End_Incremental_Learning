import torch

class Config():
    """Configuration of model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dir_data = "/disk/haohy/images/CIFAR100_png"
    # dir_model = "/disk/haohy/IL/models"
    # dir_pool = "/disk/haohy/images/CIFAR100_png/data_pool"
    # dir_logs = "/disk/haohy/IL/logger"
    dir_data = "/disk1/haohy/images/CIFAR100_png"
    dir_model = "/disk1/haohy/IL/models"
    dir_pool = "/disk1/haohy/images/CIFAR100_png/data_pool"
    dir_logs = "/disk1/haohy/IL/logger_8"
    dir_log = "/disk1/haohy/IL/logger_tmp"
    num_epochs = 100
    cap = 100000
    num_inc = 5
    lr = 1e-3
    batch_size = 256 
    lr_schedule = True 
    # data_type = 'mnist'
    data_type = 'cifar100'
    data_shape = (32, 32, 3)
    num_workers = 4

config = Config()
