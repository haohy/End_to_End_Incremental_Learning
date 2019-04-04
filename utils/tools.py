import os
import torch
import torch.utils import data
import numpy as np

def label2vec(target, label_dict, output_dim):
    """change the target vector to the matrix of (len(target), output_dim)."""
    n = target.shape[0]
    y_vec = []
    for label in target:
        y_vec.append(label_dict[label])
    y_vec = np.array(y_vec).reshape((n, output_dim))
    return y_vec
