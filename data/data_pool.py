# -*- coding:utf-8 -*-
import os
import sys
sys.path.append("..")
import logging
import numpy as np
from dataset import Dataset_Pool 
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from config import config 
from IPython import embed

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
        data pool.
    """
    def __init__(self, cap, dir_pool=config.dir_pool):
        self.data_pool = {}
        self.feature_mean = {}
        self.label_list = []
        self.cap = cap
        self.dir_pool = dir_pool
        self.num_classes = 0
        self.num_everyclass = 0
        logging.info("initialing DataPool...")

    def add_data(self, model, new_data, label_list, num_everyclass):
        """add the new data to datapool and reduce the quatity of data stored.
        
        """
        # if the data pool isn't None, adjust the number of data stored
        if len(self.data_pool) >= 0:
            for label in self.label_list:
                self.data_pool[label] = self.data_pool[label][:num_everyclass]
                
        # build dict for new-class data
        data_dict = {}
        for label in label_list:
            data_dict[label] = []
            for data in new_data:
                if str(data[1]) == label:
                    data_dict[label].append(data[0].cpu().numpy())

        # select representive data
        feature_mean_dict = {}
        for key, value in data_dict.items():
            dataset_input = []
            for value_i in value:
                dataset_input.append([value_i, key])
            dataset_tmp = Dataset_Pool(dataset_input)
            dataloader_tmp = DataLoader(dataset_tmp, batch_size=config.batch_size, num_workers=config.num_workers)
            data_feature = get_output(model, dataloader_tmp)
            feature_mean = np.mean(data_feature, axis=0)
            dist_data = np.sum(data_feature-feature_mean, axis=1)
            idx_selected = np.argsort(dist_data)
            data_selected = data_feature[idx_selected][:num_everyclass]
            data_dict[key] = data_selected
            feature_mean_dict[key] = feature_mean

        # udpate the datapool
        self.data_pool.update(data_dict)
        self.feature_mean.update(feature_mean_dict)
        self.label_list += label_list
        self.num_classes += len(label_list)
        self.num_everyclass = num_everyclass

        logging.info("add new data to data pool successfully.")

    def load_data_pool(self):
        """load data from data pool, return a Dataset."""
        logging.info("load data from data pool.")
        # loader = DataLoader(Dataset_Pool(self), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        data_list = []
        for key, value in self.data_pool.items():
            for value_i in value:
                data_list.append([value_i, [key]])
        return Dataset_Pool(data_list)

    def load_feature_mean(self):
        """load the mean feature of every class.
        Return:
            feature_mean: dict, {'0':[feature_mean]}.
        """
        logging.info("load feature mean from data pool.")
        return self.feature_mean

    def save_data_pool_file(self, path=config.dir_pool):
        path_data_pool = os.path.join(path, 'data_pool.pkl')
        with open(path_data_pool, 'wb') as f:
            data_to_stored = {'data_pool': self.data_pool, 'feature_mean': self.feature_mean}
            pickle.dump(data_to_stored, f)
        logging.info("save the data pool to file.")
        

def get_output(model, dataloader):

    # define hook
    inter_feature = {}
    def get_inter_feature(name):
        def hook(model, input, output):
            inter_feature['fc'] = input[0]
        return hook

    model.fc.register_forward_hook(get_inter_feature('fc'))
    output_list = []
    for data, _ in dataloader:
        output = model(data.to(config.device))
        output_list.append(inter_feature['fc'].detach().cpu().numpy())

    output_list = np.vstack(output_list)
    return output_list
