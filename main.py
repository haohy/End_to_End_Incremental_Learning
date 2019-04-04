import os
import torch
import torch.nn as nn
import torch.optim as optim
from IPython import embed
from data import DataPool, RawDataset, load_data, load_dataloader, concat_datasets
from models import resnet18, resnet50, save_model, load_model
from config import config

import logging
logging.basicConfig( \
    level = logging.INFO, \
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')

def train(dir_data, label_list, num_epochs, num_inc, lr, batch_size=64, device='gpu', lr_schedule=False, data_type='mnist', optimizer='adam', task='train', num_workers=4):
    logging.info("start training.")
    # seperated list of all labels, every seperation has num_inc labels
    label_sep_list = [label_list[i:i+num_inc] for i in range(0, len(label_list), num_inc)]
    num_total_classes = len(label_list)
    num_now_classes = 0

    # load model
    model = resnet18()
    save_model(model)
    logging.info("define the representer model(resnet18)")

    # define DataPool
    data_pool = DataPool(cap=config.cap)

    for label_sep in label_sep_list:
        num_now_classes += len(label_sep)

        # load stored model trained using old classes's data
        model = load_model(model, num_now_classes) 
        model = model.to(device)
        logging.info("reload the old model.")

        # dataloader of old and new datasets 
        train_dataset_old = data_pool.load_data_pool()
        train_dataset_new = load_data(dir_data, data_type, task, label_sep)
        train_dataset = concat_datasets([train_dataset_old, train_dataset_new])
        train_dataloader = load_dataloader(train_dataset, batch_size, num_workers)

        # define loss function
        criterion = nn.CrossEntropyLoss()
        if optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)

        # train the representer
        for epoch in range(num_epochs):
            
            sum_loss = 0

            for i, (train_batch, label_batch) in enumerate(train_dataloader):

                # if isinstance(label_batch, list):
                #     embed()
                train_batch = train_batch.to(device)
                label_batch = label_batch.to(device)
                output_batch = model(train_batch)

                loss = criterion(output_batch, label_batch)
                sum_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logging.info("Classes: {}/{}, Epoch: {}/{}, Loss: {:.4f}, Evel_Loss: {}".format(num_now_classes, num_total_classes, epoch+1, num_epochs, sum_loss.data, 0))

        # save samples to data pool
        num_everyclass = int(data_pool.cap/num_now_classes)
        data_pool.add_data(model, train_dataset_new, label_sep, num_everyclass) 

        # save model
        save_model(model)

    # save data pool file
    data_pool.save_data_pool_file()

if __name__ == '__main__':
    label_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    train(config.dir_data, label_list, config.num_epochs, config.num_inc, config.lr, config.batch_size, config.device, lr_schedule=config.lr_schedule, data_type=config.data_type, optimizer='adam', task='train', num_workers=config.num_workers)
