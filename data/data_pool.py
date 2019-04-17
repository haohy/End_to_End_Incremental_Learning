# -*- coding:utf-8 -*-
import os
import sys
sys.path.append("..")
import logging
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from IPython import embed

from dataset import RawDataset 

try:
    import pickle
except ImportError:
    import cPickle as pickle

logging.basicConfig( \
    level = logging.INFO, \
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')

class DataPool():
    """Data pool to store the data.
    
    Args:
        cap: int, the capacity of data pool,
        dir_pool: str, the stored directory of data pool,

    Return:
        DataPool()
        self.data_pool: 
    """
    def __init__(self, dir_data, cap, data_type):
        self.dir_data = dir_data
        self.dir_pool = os.path.join(dir_data, 'data_pool')
        self.cap = cap
        self.data_type = data_type
        self.data_pool_dict = {}
        self.feature_mean = {}
        self.classes = []
        self.num_classes = 0
        self.num_everyclass = 0
        self.data_shape = get_data_shape(data_type)
        if not os.path.isdir(self.dir_pool):
            os.mkdir(self.dir_pool)
        logging.info("initialing DataPool...")

    def add_data(self, model, new_data, num_everyclass, device):
        """add the new data to datapool and reduce the quatity of data stored.
        
        Args:
            model: representer.
            new_data: list, [[data,[label]], ...]
            label_list: list, ['0'...]
            num_everyclass: int.
        """
        # if the data pool isn't None, adjust the number of data stored
        if len(self.data_pool_dict) >= 0:
            remained_dict, msg = adjust_data_pool(self.data_pool_dict, num_everyclass)
            self.data_pool_dict = remained_dict
            logging.info(msg)
            logging.info("num_everyclass = {}".format(num_everyclass))

        data_dict_tmp = {}
        feature_mean_dict = {}
        for class_label in new_data.classes:
            dataset_tmp = RawDataset(new_data.dir_data, new_data.data_type, new_data.task, [class_label])
            dataloader_tmp = DataLoader(dataset_tmp, batch_size=64, num_workers=4)
            data_feature = get_output(model, dataloader_tmp, device)
            feature_mean = np.mean(data_feature, axis=0)
            dist_data = np.sum(data_feature-feature_mean, axis=1)
            idx_selected = np.argsort(dist_data)[:num_everyclass]
            data_path_selected = get_path_by_index(new_data.imgs_dict[class_label], idx_selected)
            data_dict_tmp[class_label] = data_path_selected
            feature_mean_dict[class_label] = feature_mean

        data_dict = from_train_to_data_pool(data_dict_tmp, self.dir_data, self.data_type, new_data.classes)

        # update the data pool
        self.data_pool_dict.update(data_dict)
        self.feature_mean.update(feature_mean_dict)
        self.classes += new_data.classes
        self.num_everyclass = num_everyclass
        
    def load_data_pool(self):
        """load data from data pool, return a Dataset."""
        logging.info("load data from data pool.")
        return RawDataset(self.dir_data, self.data_type, 'data_pool', self.classes)

    def load_feature_mean(self):
        """load the mean feature of every class.
        Return:
            feature_mean: dict, {'0':[feature_mean]}.
        """
        logging.info("load feature mean from data pool.")
        return self.feature_mean


def get_output(model, dataloader, device):

    # define hook
    inter_feature = {}
    def get_inter_feature(name):
        def hook(model, input, output):
            inter_feature['fc'] = input[0]
        return hook

    model.fc.register_forward_hook(get_inter_feature('fc'))
    output_list = []
    for data, _ in dataloader:
        output = model(data.to(device))
        output_list.append(inter_feature['fc'].detach().cpu().numpy())

    output_list = np.vstack(output_list)
    return output_list

def get_path_by_index(source_list, order_list):
    result_list = []
    for index in order_list:
        result_list.append(source_list[index])
    return result_list

def get_data_shape(data_type):
    if data_type == 'mnist':
        data_shape = (1, 28, 28)
    elif data_type == 'cifar100' or data_type == 'cifar10':
        data_shape = (32, 32, 3)
    return data_shape

def adjust_data_pool(data_pool_dict, num_everyclass):
    to_remove_dict = {}
    remained_dict = {}
    for tag, data_list in data_pool_dict.items():
        to_remove_dict[tag] = []
        if len(data_list) > num_everyclass:
            num_remove = len(data_list) - num_everyclass
            to_remove_dict[tag] = data_pool_dict[tag][-num_remove:]
            remained_dict[tag] = data_pool_dict[tag][:num_everyclass]
    if len(to_remove_dict) > 0:
        for tag, remove_list in to_remove_dict.items():
            for data_file in remove_list:
                os.remove(data_file)
    msg = "adjust data pool successfully."
    return remained_dict, msg

def from_train_to_data_pool(from_dict, dir_data, data_type, classes):
    to_dict = {}
    if data_type == 'mnist':
        for class_idx in classes:
            os.makedirs(os.path.join(dir_data, 'data_pool', class_idx))
    elif data_type == 'cifar10' or data_type == 'cifar100':
        for class_idx in classes:
            os.makedirs(os.path.join(dir_data, 'data_pool', class_idx, class_idx))
    for tag, path_list in from_dict.items():
        to_dict[tag] = []
        for path in path_list:
            pathname_list = path.split('/')
            pathname_list[pathname_list.index('train')] = 'data_pool'
            to_dict[tag].append('/'.join(pathname_list))

    # move data to data pool
    for tag, _ in from_dict.items():
        for from_path, to_path in zip(from_dict[tag], to_dict[tag]):
            shutil.copyfile(from_path, to_path)
    
    logging.info("copy training data to data pool successfully.")
    
    return to_dict


if __name__ == '__main__':
    data_pool = DataPool('/disk/haohy/images/CIFAR100_png', 1000, 'cifar100')
    embed()
