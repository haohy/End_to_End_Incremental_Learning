import torch

class Config():
    """Configuration of model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dir_data = "/disk/haohy/images/mnist"
    # dir_model = "/disk/haohy/IL/models"
    # dir_pool = "/disk/haohy/IL/data_pool"
    # dir_logs = "/disk/haohy/IL/logger"
    dir_data = "/disk/haohy/images/CIFAR100_png"
    dir_model = "/disk/haohy/IL/models"
    dir_pool = "/disk/haohy/images/CIFAR100_png/data_pool"
    dir_logs = "/disk/haohy/IL/logger"
    num_epochs = 5
    cap = 1000
    num_inc = 2
    lr = 1e-3
    batch_size = 256 
    lr_schedule = False
    # data_type = 'mnist'
    data_type = 'cifar100'
    data_shape = (32, 32, 3)
    num_workers = 4

config = Config()
