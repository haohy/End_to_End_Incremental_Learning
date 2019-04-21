import os
import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from IPython import embed
try:
    import pickle
except:
    import CPickle as pickle

class RawDataset(data.Dataset):
    """A generic dat set where the time series are arranged in this way:

    label data
    0 -1.9630089 -1.9561449 ...
    1 -1.7745713 -1.7740359 ...
    
    Args:
        dir_data: str, the directory of data.
        task: str, 'train' or 'test'.
        classes: list of str, labels of trained dataset.
    Return:
        Dataset of the dataset.
    """
    def __init__(self, dir_data, dataname, task, classes):
        super(RawDataset, self).__init__()  
        self.dir_data = dir_data
        self.dataname = dataname
        self.task = task
        self.classes = classes
        self.label_dict = get_label_dict(dir_data, dataname)
        self.ts_list = get_ts_list(dir_data, dataname, task, classes)
       
    def __getitem__(self, index):
        """
        Return:
            data: torch.tensor.
            label: int.
        """
        ts = self.ts_list[index]
        label = self.label_dict[ts[0]]
        data = np.asarray(ts[1], dtype=np.float32)
        data = torch.from_numpy(data)

        return data, label

    def __len__(self):
        return len(self.ts_list)

def concat_datasets(datasets_list):
    """concat dataset of dataset_list."""
    return data.ConcatDataset(datasets_list)

def get_label_dict(dir_data, dataname):
    with open(os.path.join(dir_data, dataname, 'classes.txt'), 'r') as f:
        label_list = f.readlines()[0].strip().split(' ')
    label_dict = {label: i for i, label in enumerate(label_list)}

    return label_dict

def get_ts_list(dir_data, dataname, task, classes):
    """get time series list from dir_data directory.

    Args:
        dir_data: str, directory of dataset.
        task: str, 'train' or 'test'.
        classes: the classes list to be got.
    Return:
        ts_list: list, time series list.
    """
    if task == 'train':
        data_path = os.path.join(dir_data, dataname, dataname+'_TRAIN.pkl')
    elif task == 'test':
        data_path = os.path.join(dir_data, dataname, dataname+'_TEST.pkl')
    elif task == 'data_pool':
        data_path = os.path.join(dir_data, dataname, 'data_pool.pkl')
        if not os.path.exists(data_path):
            return []
    with open(data_path, 'rb') as f:
        data_dict_all = pickle.load(f)
    data_dict = {}
    for class_label in classes:
        data_dict.update({class_label: data_dict_all[class_label]})
    ts_list = [[key, value_i] for key, value in data_dict.items() for value_i in value ]
    
    return ts_list

if __name__ == '__main__':
    dir_data = "/disk1/haohy/data/UCRArchive_2018"
    dataname = 'Crop'
    task = 'train'
    classes = ['1', '2']
    total = 0
    rawdataset = RawDataset(dir_data, dataname, task, classes)
    for _ in rawdataset:
        total += 1
    print(total)

    dataset = RawDataset(dir_data, dataname, task, classes)
    embed()

