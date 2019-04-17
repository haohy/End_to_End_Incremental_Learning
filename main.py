import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from IPython import embed
from data import DataPool, RawDataset, load_data, load_dataloader, concat_datasets
from models import resnet18, resnet50, save_model, load_model, acc_cal
from config import config
from utils import Logger

import logging
logging.basicConfig( \
    level = logging.INFO, \
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')

def train(data_type='mnist', num_inc=2, cap=1000, lr=0.001, name_log='default', num_epochs=60, batch_size=64, device='gpu', lr_schedule=False, optimizer='adam', dir_data='./', dir_model='./', num_workers=4):

    logging.info("start training.")

    logging.info("delete present data pool")
    shutil.rmtree(os.path.join(dir_data, 'data_pool'))

    label_list = [str(i) for i in range(len(os.listdir(os.path.join(dir_data, 'train'))))]
    label_sep_list = [label_list[i:i+num_inc] for i in range(0, len(label_list), num_inc)]
    num_total_classes = len(label_list)
    num_now_classes = 0

    # load model
    model = resnet18(config.data_shape[2])
    save_model(model, dir_model)
    logging.info("define the representer model(resnet18)")

    # define DataPool
    data_pool = DataPool(dir_data=dir_data, cap=cap, data_type=data_type)

    for label_sep in label_sep_list:
        num_now_classes += len(label_sep)

        # load stored model trained using old classes's data
        model = load_model(model, num_now_classes, dir_model) 
        model.train()
        model = model.to(device)
        logging.info("reload the old model.")

        # define logger
        logger = SummaryWriter(config.dir_logs)

        # dataloader of old and new datasets 
        train_dataset_old = data_pool.load_data_pool()
        train_dataset_new = load_data(dir_data, data_type, 'train', label_sep)
        train_dataset = concat_datasets([train_dataset_old, train_dataset_new])
        train_dataloader = load_dataloader(train_dataset, batch_size, num_workers)
        test_dataset = load_data(dir_data, data_type, 'test', data_pool.classes+label_sep)
        test_dataloader = load_dataloader(test_dataset, 64, num_workers)

        # define loss function
        criterion = nn.CrossEntropyLoss()
        if optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)

        # lr schedule
        if lr_schedule:
            scheduler = StepLR(optimizer, step_size=60, gamma=0.1)

        # train the representer
        for epoch in range(num_epochs):

            model.train() 
            sum_loss = 0
            
            scheduler.step()

            for i, (train_batch, label_batch) in enumerate(train_dataloader):

                train_batch = train_batch.to(device)
                label_batch = label_batch.to(device)
                output_batch = model(train_batch)

                loss = criterion(output_batch, label_batch)
                sum_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = acc_cal(model, test_dataloader)

            logging.info("Classes: {}/{}, Epoch: {}/{}, Loss: {:.4f}, Acc: {:.4f}".format(num_now_classes, num_total_classes, epoch+1, num_epochs, sum_loss.data, acc))

            logger.add_scalars('data/Classes_{}'.format(num_now_classes), {'loss': sum_loss.data}, epoch)
            logger.add_scalars('data/Classes_{}'.format(num_now_classes), {'acc': acc}, epoch)

        # save samples to data pool
        num_everyclass = int(data_pool.cap/num_now_classes)
        data_pool.add_data(model, train_dataset_new, num_everyclass, device) 

        # save model
        save_model(model, dir_model)
        logger.close()

if __name__ == '__main__':
    train(data_type=config.data_type, \
            num_inc=config.num_inc, \
            cap=config.cap, \
            lr=config.lr, \
            name_log='default', \
            num_epochs=config.num_epochs, \
            batch_size=config.batch_size, \
            device=config.device, \
            lr_schedule=config.lr_schedule, \
            optimizer='adam', \
            dir_data=config.dir_data, \
            dir_model=config.dir_model, \
            num_workers=config.num_workers)
