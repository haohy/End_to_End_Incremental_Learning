import os
import numpy as np
try:
    import pickle
except:
    import CPickle as pickle
from IPython import embed

def adjust_lr(optimizer, lr, epoch):
    
    if epoch >= 20 and epoch < 40:
        lr *= 0.5
    if epoch >= 40 and epoch < 60:
        lr *= (0.5*0.5)
    if epoch >= 60 and epoch < 90:
        lr *= 0.1
    if epoch >= 90:
        lr *= 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def from_raw_to_dataset(dir_data, dataname):
    dir_root = os.path.join(dir_data, dataname)
    path_train = os.path.join(dir_root, dataname+'_TRAIN.tsv')
    path_test = os.path.join(dir_root, dataname+'_TEST.tsv')
    data_dict = {}
    train_list, test_list = [], []
    with open(path_train, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.strip().split('\t')
            train_list.append([line_list[1:], int(line_list[0])])
    with open(path_test, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.strip().split('\t')
            test_list.append([line_list[1:], int(line_list[0])])
    all_list = np.array(train_list + test_list)
    classes = list(set(all_list[:, 1]))
    train_dict, test_dict = {}, {}
    num_dict = {str(class_label): np.sum(all_list[:, 1] == class_label) for class_label in classes}
    for class_label in classes:
        data_dict.update({str(class_label): all_list[all_list[:,1]==class_label][:,0]})
    for class_label in classes:
        train_dict.update({str(class_label): data_dict[str(class_label)][:int(num_dict[str(class_label)]*0.9)]})
        test_dict.update({str(class_label): data_dict[str(class_label)][int(num_dict[str(class_label)]*0.9):]})
    embed()
    with open(os.path.join(dir_root, dataname+'_TRAIN.pkl'), 'wb') as f:
        pickle.dump(train_dict, f)
    with open(os.path.join(dir_root, dataname+'_TEST.pkl'), 'wb') as f:
        pickle.dump(test_dict, f)
    print("save {} to pickle file successfully.".format(dataname))


if __name__ == '__main__':
    dir_data = "/disk1/haohy/data/UCRArchive_2018"
    dataname = 'Crop'
    from_raw_to_dataset(dir_data, dataname)
    # datasets_list = os.listdir(dir_data)
    # datasets_list = os.listdir(dir_data)
    # for dataname in datasets_list:
    #     if os.path.isdir(os.path.join(dir_data, dataname)):
    #         from_raw_to_dataset(dir_data, dataname)
    #         exit()
