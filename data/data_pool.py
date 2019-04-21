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

    Return:
        DataPool()
        self.data_pool: 
    """
    def __init__(self, dir_data, cap, dataname):
        self.dir_data = dir_data
        self.cap = cap
        self.dataname = dataname
        self.data_pool_dict = {}
        self.feature_mean = {}
        self.classes = []
        self.num_classes = 0
        self.num_everyclass = 0
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
            dataset_tmp = RawDataset(new_data.dir_data, new_data.dataname, new_data.task, [class_label])
            dataloader_tmp = DataLoader(dataset_tmp, batch_size=16, num_workers=1)
            data_feature = get_output(model, dataloader_tmp, device)
            feature_mean = np.mean(data_feature, axis=0)
            dist_data = np.sum(data_feature-feature_mean, axis=1)
            idx_selected = np.argsort(dist_data)[:num_everyclass]
            data_selected = get_selected_idx(new_data.ts_list, idx_selected)
            data_dict_tmp[class_label] = data_selected
            feature_mean_dict[class_label] = feature_mean

        # update the data pool
        self.data_pool_dict.update(data_dict_tmp)
        self.feature_mean.update(feature_mean_dict)
        self.classes += new_data.classes
        self.num_everyclass = num_everyclass

        self.save_datapool_to_pkl()
        
    def load_data_pool(self):
        """load data from data pool, return a Dataset."""
        logging.info("load data from data pool.")
        return RawDataset(self.dir_data, self.dataname, 'data_pool', self.classes)

    def load_feature_mean(self):
        """load the mean feature of every class.
        Return:
            feature_mean: dict, {'0':[feature_mean]}.
        """
        logging.info("load feature mean from data pool.")
        return self.feature_mean

    def save_datapool_to_pkl(self):
        """save the datapool to pickle file."""
        with open(os.path.join(self.dir_data, self.dataname, 'data_pool.pkl'), 'wb') as f:
            pickle.dump(self.data_pool_dict, f)
        logging.info("save the data pool to pickle file.")


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

def get_selected_idx(source_list, idx_list):
    result_list = []
    for idx in idx_list:
        result_list.append(source_list[idx][1])

    return result_list

def adjust_data_pool(data_pool_dict, num_everyclass):
    remained_dict = {}
    for tag, data_list in data_pool_dict.items():
        remained_dict.update({tag: data_list[:num_everyclass]})
    msg = "adjust data pool successfully."
    return remained_dict, msg


if __name__ == '__main__':
    embed()
