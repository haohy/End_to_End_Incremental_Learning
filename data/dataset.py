import os
import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from IPython import embed

class RawDataset(data.Dataset):
    """A generic dat set where the images are arranged in this way:
    class_1/
        1.png
        20.png
        ...
    class_2/
        3.png
        15.png
    ...
    labels.txt
    
    Args:
        data_dir: str, the directory of data.
        data_type: str, 'mnist' 'cifar10' or 'cifar100'.
        task: str, 'train', 'dev' or 'test'.
        classes: list of str, labels of trained dataset.
    Return:
        Dataset of the dataset.
    """
    def __init__(self, data_dir, data_type, task, classes):
        super(RawDataset, self).__init__()  
        self.data_dir = data_dir
        self.classes = classes
        self.task = task
        imgs_list = []
        if data_type == 'mnist':
            self.transform_train = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Lambda(lambda img: img[0].unsqueeze(0)),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        if task == 'train':
            self.dir_path = os.path.join(data_dir, 'train')
        if task == 'test':
            self.dir_path = os.path.join(data_dir, 'test')
        for class_index in classes:
            path_dir = os.path.join(self.dir_path, class_index)
            imgs_list += [os.path.join(path_dir, img) for img in os.listdir(path_dir)]
        self.imgs_list = imgs_list
       
    def __getitem__(self, index):
        img_path = self.imgs_list[index]
        label = int(img_path.strip().split('/')[-2])
        data = Image.open(img_path)
        data = self.transform_train(data)
        return data, label

    def __len__(self):
        return len(self.imgs_list)

class Dataset_Pool(data.Dataset):
    """A generic data set where the data are from data pool.

    Args:
        data_pool, DataPool(),

    Return:
        Dataset of the DataPool.
    """
    def __init__(self, data_pool):
        self.data = data_pool
        
    def __getitem__(self, index):
        data, label = self.data[index]
        return torch.from_numpy(data), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)


def concat_datasets(datasets_list):
    """concat dataset of dataset_list.
    """
    embed()
    return data.ConcatDataset(datasets_list)
