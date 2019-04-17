# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from IPython import embed
from models import resnet18, acc_cal, save_model, load_model
from data import load_data, load_dataloader, concat_datasets
from utils import Logger
from config import config

model = resnet18(3)

trainset_1 = load_data(config.dir_data, 'cifar100', 'train', ['0', '1', '2'])
trainset_2 = load_data(config.dir_data, 'cifar100', 'train', ['3', '4'])
train_dataset = concat_datasets([trainset_1, trainset_2])
train_dataloader = load_dataloader(train_dataset, 256, 4)
test_dataset = load_data(config.dir_data, 'cifar100', 'test', ['0', '1', '2', '3', '4'])
test_dataloader = load_dataloader(test_dataset, 128, 4)

# logger = Logger(os.path.join(config.dir_logs, 'resnet.log'))
writer = SummaryWriter(config.dir_log)

save_model(model, config.dir_model)
model = load_model(model, 5, config.dir_model)
model.train()
model = model.to(config.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)
schedule = StepLR(optimizer, step_size=40, gamma=0.1)

for epoch in range(100):
    sum_loss = 0
    
    model.train()
    schedule.step()


    for i, (train_batch, label_batch) in enumerate(train_dataloader):

        train_batch = train_batch.to(config.device)
        label_batch = label_batch.to(config.device)
        output_batch = model(train_batch)

        loss = criterion(output_batch, label_batch)
        sum_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # embed()
    acc = acc_cal(model, test_dataloader)
    acc_train = acc_cal(model, train_dataloader)


    print("Epoch: {}/{}, Loss: {:.4f}, Acc: {:.4f}, Acc_train: {:.4f}".format(epoch+1, 100, sum_loss.data, acc, acc_train))

    # writer.add_scalar('loss', sum_loss.data, epoch)
    # writer.add_scalar('acc', acc, epoch)
    # writer.add_scalars('data/scalar_group', {'loss': sum_loss.data, 'acc': acc}, epoch)

    # info_log = {'Loss': sum_loss.data, 'Test Acc': acc}
    # for key, value in info_log.items():
    #     logger.scalar_summary(key, value, epoch+1)

writer.close()
