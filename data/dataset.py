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
        dir_data: str, the directory of data.
        data_type: str, 'mnist' 'cifar10' or 'cifar100'.
        task: str, 'train', 'dev' or 'test'.
        classes: list of str, labels of trained dataset.
    Return:
        Dataset of the dataset.
    """
    def __init__(self, dir_data, data_type, task, classes):
        super(RawDataset, self).__init__()  
        self.dir_data = dir_data
        self.data_type = data_type
        self.task = task
        self.classes = classes
        self.transforms = get_transforms(data_type) 
        self.dir_classes = get_dir_classes(dir_data, data_type, task, classes) 
        self.imgs_list = get_imgs_list(self.dir_classes)
        self.imgs_dict, self.imgs_name_dict = get_imgs_dict(self.dir_classes)
       
    def __getitem__(self, index):
        """
        Return:
            data: torch.Tensor.
            label: int.
        """
        img_path = self.imgs_list[index]
        label = int(img_path.strip().split('/')[-2])
        data = Image.open(img_path)
        data = self.transforms(data)
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
        return data, label

    def __len__(self):
        return len(self.data)


def concat_datasets(datasets_list):
    """concat dataset of dataset_list.
    """
    return data.ConcatDataset(datasets_list)

def get_dir_classes(dir_data, data_type, task, classes):
    """get the paths of classes."""
    dir_root = os.path.join(dir_data, task)
    dir_classes = []
    if data_type == 'mnist':
        for i in range(len(classes)):
            dir_classes.append(os.path.join(dir_root, classes[i]))
        # dir_classes = [os.path.join(dir_root, classes[i]) for i in range(len(classes))]
    elif data_type == 'cifar100' or data_type == 'cifar10':
        for i in range(len(classes)):
            dir_classes.append(os.path.join(dir_root, classes[i], classes[i]))
        # dir_classes = [os.path.join(dir_root, classes[i], classes[i]) for i in range(len(classes))]
    if task == 'data_pool':
        for dir_path in dir_classes:
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
    return dir_classes

def get_imgs_list(dir_classes):
    """get the list of entire path of all images."""
    imgs_list = []
    for dir_class in dir_classes:
        for img_name in os.listdir(dir_class):
            imgs_list.append(os.path.join(dir_class, img_name))
    return imgs_list

def get_imgs_dict(dir_classes):
    """get the image path, put them in dict like {'0':[img_path...]}."""
    imgs_dict = {}
    imgs_name_dict = {}
    for dir_class in dir_classes:
        label = dir_class.split('/')[-1]
        imgs_dict[label] = []
        imgs_name_dict[label] = []
        for img_name in os.listdir(dir_class):
            imgs_dict[label].append(os.path.join(dir_class, img_name))
            imgs_name_dict[label].append(img_name)
    return imgs_dict, imgs_name_dict

def get_transforms(data_type):
    """get image transforms to process images."""
    if data_type == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Lambda(lambda img: img[0].unsqueeze(0)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif data_type == 'cifar100' or data_type == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.view(3, 32, 32)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    return transform_train


if __name__ == '__main__':
    dir_data = '/disk/haohy/images/CIFAR100_png'
    data_type = 'cifar100'
    task = 'train'
    classes = ['0', '1']
    total = 0
    rawdataset = RawDataset(dir_data, data_type, task, classes)
    for _ in rawdataset:
        total += 1
    print(total)
