import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
sys.path.append(os.getcwd())

from torch.utils.data import DataLoader
from .dataset import concat_datasets, RawDataset 
from .data_pool import DataPool

def load_data(data_dir, data_type, task, classes):
    dataset = RawDataset(data_dir, data_type, task, classes)
    return dataset

def load_dataloader(dataset, batch_size, num_workers, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
